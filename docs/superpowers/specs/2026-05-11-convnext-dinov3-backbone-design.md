# ECDet-ConvNeXt: DINOv3 ConvNeXt-Tiny Backbone for Object Detection

**Status:** Design
**Date:** 2026-05-11
**Author:** thanh.nguyen@gojitsu.com
**Scope:** Object detection only (ECDet). ECInsSeg and ECPose are out of scope but addressable by the same pattern in follow-up work.

## 1. Goal

Add a new object-detection model variant to this repo that uses `timm/convnext_tiny.dinov3_lvd1689m` as the backbone, while reusing the existing ECDet training pipeline (HybridEncoder, ECTransformer decoder, criterion, augmentations, optimizer, EMA, AMP) without modification.

## 2. Background

### 2.1 Current pipeline

The repo implements the **Stage 3 (task-specific training)** portion of the EdgeCrafter paper ([arXiv:2603.18739](https://arxiv.org/abs/2603.18739)). Stages 1 and 2 (teacher preparation and backbone distillation) are not in the repo; they were used to produce the released `ecvit*.pth` weights that `ViTAdapter` downloads.

Current backbone (`ViTAdapter` in [ecdetseg/engine/edgecrafter/ecvit.py](../../../ecdetseg/engine/edgecrafter/ecvit.py)):

1. ConvPyramidPatchEmbed (4× stride-2 convs) → stride-16 tokens.
2. 12 transformer blocks with RoPE.
3. Take last two blocks' tokens, mean-fuse them.
4. Reshape to 2D, bilinearly resize to three scales (stride 8/16/32).
5. 1×1 conv projector per scale.

The "bilinear resize from a single feature map" exists only because ViT is single-scale and the downstream HybridEncoder expects three hierarchical feature maps.

### 2.2 HybridEncoder + ECTransformer are backbone-agnostic

[hybrid_encoder.py:317](../../../ecdetseg/engine/edgecrafter/hybrid_encoder.py#L317) `HybridEncoder` takes `in_channels=[c8, c16, c32]` and `feat_strides=[8, 16, 32]`. It applies a 1×1 `input_proj` per level to unify to `hidden_dim`, then runs AIFI (self-attention on the s=32 map) and CCFF (CSP-style conv fusion). This is RT-DETR's original encoder, which was designed for ResNet/HGNet (CNN) backbones. ECViT had to manufacture hierarchical features for it. ConvNeXt produces them natively.

### 2.3 Why this design exists

- `convnext_tiny.dinov3_lvd1689m` is already a distillation product (DINOv3 ViT-7B teacher → ConvNeXt-Tiny student on LVD-1689M), placing it at the "DINOv3 distillation" tier of the paper's Figure 1b (~50.7 AP). The paper's task-specialized distillation tier (51.7 AP) costs an extra ECDet-X teacher-preparation run plus a 50-epoch backbone-distillation run — deferred.
- ConvNeXt is hierarchical with four stages at strides 4/8/16/32 — a direct structural match for HybridEncoder, which needs maps at 8/16/32. No fake pyramid construction needed.
- The whole change is a backbone swap: one new module, one new config, everything else inherits.

### 2.4 Explicit non-goals

- **Distillation stages.** Neither Stage 1 (teacher preparation) nor Stage 2 (backbone distillation) is built. The downstream-AP cost (~1 AP per paper Figure 1b) is accepted in exchange for pipeline simplicity. Adding distillation is a follow-up.
- **Backbone family.** Only `convnext_tiny.dinov3_lvd1689m` is wired up. Adding `convnext_small/base/large.dinov3_lvd1689m` is trivial via new config files; not included here to keep scope tight.
- **Other tasks.** ECInsSeg and ECPose can reuse the same `ConvNeXtAdapter` and would only need their own configs.

## 3. Architecture

### 3.1 New module: `ConvNeXtAdapter`

Lives in a new file [ecdetseg/engine/edgecrafter/convnext.py](../../../ecdetseg/engine/edgecrafter/convnext.py). Registered the same way as `ViTAdapter` so YAML configs can address it by name.

Reason for a new file (rather than extending `ecvit.py`): different forward logic (native hierarchy, no token fusion, no RoPE, no register tokens), different state-dict source (timm rather than custom EdgeCrafter checkpoint URL), different parameter conventions. Keeping them separate keeps each file focused.

#### Forward path

```
input  [B, 3, 640, 640]
  └─ timm convnext_tiny.dinov3_lvd1689m (features_only=True, out_indices=(1,2,3))
      (timm 0-indexed stages: stage0 stride 4 / stage1 stride 8 / stage2 stride 16 / stage3 stride 32)
      ├─ stage 1  →  [B, 192, 80, 80]   (stride 8)
      ├─ stage 2  →  [B, 384, 40, 40]   (stride 16)
      └─ stage 3  →  [B, 768, 20, 20]   (stride 32)
  └─ projector  (optional, 3× ConvNormLayer_fuse 1×1)
      └─ feeds HybridEncoder unchanged
```

#### Skeleton

```python
@register()
class ConvNeXtAdapter(nn.Module):
    def __init__(
        self,
        name: str = "convnext_tiny.dinov3_lvd1689m",
        pretrained: bool = True,
        out_indices: tuple = (1, 2, 3),    # stages giving strides 8, 16, 32
        proj_dim: int | list | None = None, # None = emit native channels
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True,
            out_indices=out_indices, drop_path_rate=drop_path_rate,
        )
        feat_channels = self.backbone.feature_info.channels()
        # Verify in __init__: assert feat_channels matches config's in_channels
        if proj_dim is None:
            self.projector = nn.Identity()  # let HybridEncoder.input_proj handle channels
            self.out_channels = feat_channels
        else:
            dims = proj_dim if isinstance(proj_dim, list) else [proj_dim] * len(out_indices)
            self.projector = nn.ModuleList([
                ConvNormLayer_fuse(c_in, c_out, kernel_size=1, stride=1)
                for c_in, c_out in zip(feat_channels, dims)
            ])
            self.out_channels = dims

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(self.projector, nn.Identity):
            return feats
        return [p(f) for p, f in zip(self.projector, feats)]
```

### 3.2 Channel-projection choice: project to hidden_dim in the adapter

**Correction (2026-05-11, during implementation):** an earlier draft of this spec assumed `HybridEncoder` had a per-level `input_proj`. It does not — `HybridEncoder.in_channels` is used only for module-list lengths; all internal blocks consume `hidden_dim` channels directly. Existing `ecdet_l.yml` confirms this by setting `ViTAdapter.proj_dim: 256` and `HybridEncoder.in_channels: [256, 256, 256]`.

Consequence: `ConvNeXtAdapter` must project all stage outputs to `hidden_dim` (256) before they reach the encoder. The `proj_dim` parameter on the adapter handles this with 1×1 ConvNormLayer_fuse per level. We accept the capacity squash at the deepest stage (768 → 256) as the cost of reusing the encoder unchanged.

(The `proj_dim=None` branch is still implemented in the adapter for future use cases — e.g., a modified HybridEncoder that does its own per-level projection — but is unused by the current config.)

### 3.3 Differences from `ViTAdapter`

| Aspect | `ViTAdapter` (existing) | `ConvNeXtAdapter` (new) |
|---|---|---|
| Native multi-scale | No, fused tokens upsampled bilinearly | **Yes**, stages 2/3/4 used directly |
| Patch stem | `ConvPyramidPatchEmbed` (4× stride-2) | ConvNeXt native 4×4 stride-4 stem |
| Position embedding | RoPE on tokens | None (conv inductive bias) |
| Register tokens | Yes (1) | N/A |
| Output channels | Uniform (e.g. all 192) | Hierarchical (192, 384, 768) |
| Weight source | Custom `.pth` from EdgeCrafter releases | timm `pretrained=True` |
| Loaded checkpoint | EdgeCrafter Stage-2 distillation output | DINOv3 distillation product |

## 4. Configuration

### 4.1 New config

[ecdetseg/configs/ecdet/ecdet_cnxt_t.yml](../../../ecdetseg/configs/ecdet/ecdet_cnxt_t.yml):

```yaml
__include__: [
  '../dataset/coco.yml',
  'ecdet.yml',
]

output_dir: outputs/ecdet_cnxt_t

ECDet:
  backbone: ConvNeXtAdapter

ConvNeXtAdapter:
  name: convnext_tiny.dinov3_lvd1689m
  pretrained: true
  out_indices: [1, 2, 3]
  drop_path_rate: 0.1
  proj_dim: 256            # HybridEncoder has no input_proj; adapter must emit hidden_dim channels

HybridEncoder:
  in_channels: [256, 256, 256]   # already at hidden_dim after ConvNeXtAdapter.projector
  feat_strides: [8, 16, 32]
  hidden_dim: 256
  dim_feedforward: 1024

ECTransformer:
  feat_channels: [256, 256, 256]
  hidden_dim: 256
  num_layers: 4
  eval_idx: -1

optimizer:
  type: AdamW
  params:
    - {params: '^(?=.*\.backbone)(?!.*(?:norm|bn|bias)).*$', lr: 0.000005}
    - {params: '^(?=.*\.backbone)(?=.*(?:norm|bn|bias)).*$', lr: 0.000005, weight_decay: 0.}
    - {params: '^(?!.*\.backbone)(?=.*(?:norm|bn|bias)).*$',                weight_decay: 0.}
  lr: 0.0005
  betas: [0.9, 0.999]
  weight_decay: 0.0001

epochs: 62                    # 60 with aug + 2 no-aug
warmup_iter: 2000
lr_gamma: 0.5
eval_spatial_size: [640, 640]

train_dataloader:
  dataset:
    transforms:
      mosaic_epoch: 30
      mosaic_prob: 0.75
      stop_epoch: 60
  collate_fn:
    mixup_prob: 0.75
    mixup_epoch: 30
```

### 4.2 Backbone learning rate: 5e-6

The starting checkpoint (`convnext_tiny.dinov3_lvd1689m`) is a **generic DINOv3 distillation product** with task-agnostic SSL features. This matches the paper's **Stage 1 (teacher preparation)** starting point, not Stage 3 (where the backbone is the detection-distilled ECViT student).

Paper Stage 1 backbone-LR ladder (Table 10):

| Backbone | Params | Stage 1 framework | Backbone LR |
|---|---|---|---|
| DINOv3-S | ~22M | ECDet-L | 5e-6 |
| DINOv3-B | ~87M | ECDet-X | 2.5e-6 |

ConvNeXt-Tiny is ~29M params — closest to DINOv3-S. Default **5e-6**. ConvNeXt may tolerate marginally higher LR than ViT (conv inductive bias; LayerNorm-only, no BN running-stats drift), but not enough to justify a 10× jump.

This is the **primary ablation target** if downstream AP disappoints: sweep `{2.5e-6, 5e-6, 1e-5, 2.5e-5}`.

### 4.3 Optimizer regex compatibility

Existing regex `^(?=.*\.backbone)(?!.*(?:norm|bn|bias)).*$` in [ecdet.yml:171](../../../ecdetseg/configs/ecdet/ecdet.yml#L171) matches by substring on the parameter path. With `ECDet.backbone = ConvNeXtAdapter` and `ConvNeXtAdapter.backbone = timm_model`, parameter paths look like `backbone.backbone.stages.0.blocks.0.dwconv.weight`. The regex still matches. **No regex change required.**

## 5. Code touch points

1. **New file** `ecdetseg/engine/edgecrafter/convnext.py` — `ConvNeXtAdapter` (~80 lines, structure in §3.1).
2. **Edit** `ecdetseg/engine/edgecrafter/__init__.py` — add `from .convnext import ConvNeXtAdapter` so the registry picks it up.
3. **Verify** `ecdetseg/engine/edgecrafter/modeling.py` — confirm `ECDet` constructs `backbone` from the registry by name (it should already; verify during implementation).
4. **New config** `ecdetseg/configs/ecdet/ecdet_cnxt_t.yml` (§4.1).
5. **Edit** `requirements.txt` — add/pin `timm >= 1.0.x` (DINOv3 ConvNeXt weights require a recent timm; confirm minimum version at implementation time).

## 6. Verification gates

Before launching a full training run, the implementation must pass:

1. **timm model loads.** `timm.create_model('convnext_tiny.dinov3_lvd1689m', features_only=True, out_indices=(1,2,3))` returns a model with `feature_info.channels() == [192, 384, 768]` and `reduction == [8, 16, 32]`. Asserted in `ConvNeXtAdapter.__init__`.
2. **Forward shape check.** A dummy forward at `[1, 3, 640, 640]` yields three feature maps of shape `[(1,192,80,80), (1,384,40,40), (1,768,20,20)]`. Asserted at init or via a unit test.
3. **Pretrained weights actually loaded.** Log on rank 0: model name, weight source, and an aggregate parameter checksum or norm. If `pretrained=True` silently falls back to random init, training must error out, not proceed.
4. **End-to-end smoke train.** One epoch on a small subset (e.g., COCO mini) with the new config completes without NaNs and produces a non-trivial AP > 0.
5. **Parameter-group sanity.** Print the count of parameters in each optimizer group on first step. The backbone group must contain >90% of `ConvNeXtAdapter` parameters and 0 parameters outside it.

## 7. Expected results

Reference points from the paper:

| Recipe (Figure 1b) | COCO AP |
|---|---|
| ECViT-T random init | 42.2 |
| ECViT-T IN21K supervised | 46.6 |
| ECViT-T DINOv3 distillation (generic) | 50.7 |
| ECViT-T task-specialized distillation | 51.7 (= ECDet-S in Table 2) |

Our backbone enters at the "DINOv3 distillation" tier (~50.7 equivalent). Realistic targets for `ECDet-cnxt-t`:

- **Floor:** ≥ 49 AP on COCO val2017. Below this, something is wrong (backbone LR too high, weights not loaded, channel mismatch).
- **Target:** 50–52 AP. Close to ECDet-S (51.7) with ConvNeXt's hierarchical features potentially partially offsetting the lack of task-specialized distillation.
- **Stretch:** > 52 AP. Would indicate ConvNeXt's stronger inductive bias and DINOv3 ConvNeXt distillation overcame the lack of in-repo task distillation.

These are estimates, not contractual; the experiment is the source of truth.

## 8. Risks and open questions

- **Backbone LR is the primary risk knob.** 5e-6 is the principled starting point; if features degrade rapidly (visible as loss spiking early or AP plateauing low), try 2.5e-6. If features fail to adapt (slow convergence, AP plateauing), try 1e-5.
- **Drop-path rate.** Started at 0.1. ConvNeXt-Tiny default for ImageNet training is similar; for detection fine-tuning a value in `[0.0, 0.2]` is reasonable. If overfitting, raise; if underfitting, lower.
- **Mosaic+Mixup interaction with DINOv3 features.** The paper uses these for ECViT and they help. ConvNeXt is generally robust to aggressive augmentation. If unstable, lower `mosaic_prob` and `mixup_prob` to 0.5.
- **timm version pinning.** DINOv3 ConvNeXt weights landed in timm relatively recently. CI/install reproducibility depends on a pinned `timm>=X.Y.Z`. Pin at implementation time.
- **EMA decay.** Default `0.9999` from `ecdet.yml`. No change planned, but if the backbone's BN-free/LN-only stats interact badly with EMA, this is a fallback knob.

## 9. Follow-up work (not part of this design)

- Add ConvNeXt-S/B/L variants as separate configs once Tiny is validated.
- Implement Stage 1 (DINOv3 ConvNeXt-Base/Large adapted on COCO inside ECDet) → expected +0.5–1 AP per paper Figure 1b.
- Implement Stage 2 (backbone distillation from the Stage 1 ECTeacher into Tiny) → completes the paper-faithful pipeline.
- Extend `ConvNeXtAdapter` to ECInsSeg and ECPose configs.
