# ECDet ConvNeXt-Tiny (DINOv3) Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new ECDet object-detection variant that uses `timm/convnext_tiny.dinov3_lvd1689m` as the backbone, reusing the existing HybridEncoder + ECTransformer + criterion pipeline unmodified.

**Architecture:** Single new `ConvNeXtAdapter` module wraps a timm-built ConvNeXt with `features_only=True`, emits native hierarchical features at strides 8/16/32, feeds them straight to the existing HybridEncoder. One new config file (`ecdet_cnxt_t.yml`) inherits all detection/training defaults from `ecdet.yml` and overrides only the backbone block, optimizer LR, and HybridEncoder `in_channels`. No code changes to encoder/decoder/criterion/dataloader.

**Tech Stack:** PyTorch ≥ 2.6, timm (DINOv3 ConvNeXt weights — requires recent version), existing EdgeCrafter registry / YAML config system.

**Spec:** [docs/superpowers/specs/2026-05-11-convnext-dinov3-backbone-design.md](../specs/2026-05-11-convnext-dinov3-backbone-design.md)

**Test strategy note:** This repo has no pytest infrastructure. Verification is done via standalone Python snippets (run with `python -c "..."` or saved as throwaway scripts) and a short smoke training run on a tiny subset. Each task's verification command is given verbatim.

---

## Task 1: Pin timm in requirements

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Verify DINOv3 ConvNeXt weights are available in the locally-installed timm**

Run:
```bash
python -c "import timm; print(timm.__version__); print([n for n in timm.list_pretrained('convnext_tiny*') if 'dinov3' in n])"
```
Expected output: a timm version string, and a non-empty list containing `convnext_tiny.dinov3_lvd1689m`. If the list is empty, upgrade timm with `pip install -U timm` and re-run.

Record the version printed — that's the minimum version we need to pin.

- [ ] **Step 2: Add the pin to requirements.txt**

Open `requirements.txt` and add a line for timm using the version printed in Step 1. For example, if Step 1 printed `1.0.19`:

```
timm>=1.0.19
```

The full file should now end with the new line. Keep the rest of the file unchanged.

- [ ] **Step 3: Verify the pin is satisfied**

Run:
```bash
pip install -r requirements.txt 2>&1 | tail -5
```
Expected: no errors mentioning timm; existing install satisfies the constraint.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: pin timm for DINOv3 ConvNeXt weights"
```

---

## Task 2: Create `ConvNeXtAdapter` module

**Files:**
- Create: `ecdetseg/engine/edgecrafter/convnext.py`

- [ ] **Step 1: Write the adapter file**

Create `ecdetseg/engine/edgecrafter/convnext.py` with this exact content:

```python
"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
ConvNeXt backbone adapter using DINOv3 pretrained weights from timm.
"""
from typing import List, Optional, Sequence, Union

import timm
import torch
import torch.nn as nn

from ..core import register
from .hybrid_encoder import ConvNormLayer_fuse

__all__ = ["ConvNeXtAdapter"]


@register()
class ConvNeXtAdapter(nn.Module):
    """Hierarchical ConvNeXt backbone wrapper for ECDet.

    Emits feature maps at strides 8/16/32 (timm 0-indexed stages 1/2/3)
    suitable for direct consumption by HybridEncoder.
    """

    def __init__(
        self,
        name: str = "convnext_tiny.dinov3_lvd1689m",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
        proj_dim: Optional[Union[int, List[int]]] = None,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.name = name
        self.out_indices = tuple(out_indices)

        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            drop_path_rate=drop_path_rate,
        )

        feat_channels = list(self.backbone.feature_info.channels())
        feat_strides = list(self.backbone.feature_info.reduction())
        assert feat_strides == [8, 16, 32], (
            f"Expected stage strides [8, 16, 32] for ECDet, got {feat_strides}. "
            f"Check out_indices={self.out_indices} for model {name!r}."
        )
        self.feat_channels = feat_channels
        self.feat_strides = feat_strides

        if proj_dim is None:
            self.projector = None
            self.out_channels = feat_channels
        else:
            dims = (
                list(proj_dim)
                if isinstance(proj_dim, (list, tuple))
                else [int(proj_dim)] * len(self.out_indices)
            )
            assert len(dims) == len(self.out_indices), (
                f"proj_dim length {len(dims)} must match out_indices length {len(self.out_indices)}"
            )
            self.projector = nn.ModuleList(
                [
                    ConvNormLayer_fuse(c_in, c_out, kernel_size=1, stride=1)
                    for c_in, c_out in zip(feat_channels, dims)
                ]
            )
            self.out_channels = dims

        self._log_loaded(pretrained)

    def _log_loaded(self, pretrained: bool) -> None:
        rank_ok = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_ok = torch.distributed.get_rank() == 0
        if not rank_ok:
            return
        total = sum(p.numel() for p in self.backbone.parameters())
        checksum = sum(p.detach().float().abs().sum().item() for p in self.backbone.parameters())
        status = "pretrained" if pretrained else "RANDOM INIT"
        print(
            "=" * 80 + "\n"
            f"ConvNeXtAdapter loaded: {self.name} ({status})\n"
            f"  out_indices: {self.out_indices}\n"
            f"  feat_channels: {self.feat_channels}\n"
            f"  feat_strides:  {self.feat_strides}\n"
            f"  out_channels:  {self.out_channels}\n"
            f"  backbone params: {total:,}\n"
            f"  param-abs-sum checksum: {checksum:.4f}\n"
            + "=" * 80
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        if self.projector is None:
            return list(feats)
        return [proj(f) for proj, f in zip(self.projector, feats)]
```

- [ ] **Step 2: Smoke-test the adapter standalone**

Run this verbatim:
```bash
cd /Users/thanhnn5/Workspace/Jitsu/POD/EdgeCrafter && PYTHONPATH=ecdetseg python -c "
import torch
from engine.edgecrafter.convnext import ConvNeXtAdapter
m = ConvNeXtAdapter(name='convnext_tiny.dinov3_lvd1689m', pretrained=True, out_indices=(1,2,3))
m.eval()
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    feats = m(x)
shapes = [tuple(f.shape) for f in feats]
print('Output shapes:', shapes)
assert shapes == [(1, 192, 80, 80), (1, 384, 40, 40), (1, 768, 20, 20)], f'Unexpected shapes: {shapes}'
print('OK')
"
```

Expected output: the load-banner from `_log_loaded`, then `Output shapes: [(1, 192, 80, 80), (1, 384, 40, 40), (1, 768, 20, 20)]`, then `OK`. Non-zero `param-abs-sum checksum` confirms pretrained weights loaded (random init typically gives a much smaller, near-uniform value — this is a sanity log, not a hard assertion).

- [ ] **Step 3: Smoke-test the projector branch**

Run:
```bash
cd /Users/thanhnn5/Workspace/Jitsu/POD/EdgeCrafter && PYTHONPATH=ecdetseg python -c "
import torch
from engine.edgecrafter.convnext import ConvNeXtAdapter
m = ConvNeXtAdapter(pretrained=False, proj_dim=256)
m.eval()
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    feats = m(x)
shapes = [tuple(f.shape) for f in feats]
print('Projected shapes:', shapes)
assert shapes == [(1, 256, 80, 80), (1, 256, 40, 40), (1, 256, 20, 20)], f'Unexpected: {shapes}'
print('OK')
"
```
Expected: `Projected shapes: [(1, 256, 80, 80), (1, 256, 40, 40), (1, 256, 20, 20)]` then `OK`.

- [ ] **Step 4: Commit**

```bash
git add ecdetseg/engine/edgecrafter/convnext.py
git commit -m "feat(backbone): add ConvNeXtAdapter wrapping timm DINOv3 ConvNeXt"
```

---

## Task 3: Register `ConvNeXtAdapter` in the package

**Files:**
- Modify: `ecdetseg/engine/edgecrafter/__init__.py`

- [ ] **Step 1: Add the import**

Open `ecdetseg/engine/edgecrafter/__init__.py`. Currently:

```python
from .criterion import ECCriterion
from .decoder import ECTransformer
from .ecvit import ViTAdapter
from .hybrid_encoder import HybridEncoder
from .matcher import HungarianMatcher
from .modeling import ECDet, ECSeg
from .postprocessor import PostProcessor
```

Add one new line so the file reads:

```python
from .criterion import ECCriterion
from .convnext import ConvNeXtAdapter
from .decoder import ECTransformer
from .ecvit import ViTAdapter
from .hybrid_encoder import HybridEncoder
from .matcher import HungarianMatcher
from .modeling import ECDet, ECSeg
from .postprocessor import PostProcessor
```

(Imports are kept alphabetical to match the existing style.)

- [ ] **Step 2: Verify registry resolution**

Run:
```bash
cd /Users/thanhnn5/Workspace/Jitsu/POD/EdgeCrafter && PYTHONPATH=ecdetseg python -c "
from engine.edgecrafter import ConvNeXtAdapter
from engine.core import GLOBAL_CONFIG
print('Resolved by name:', 'ConvNeXtAdapter' in GLOBAL_CONFIG)
assert 'ConvNeXtAdapter' in GLOBAL_CONFIG, 'ConvNeXtAdapter not registered in GLOBAL_CONFIG'
print('OK')
"
```
Expected: `Resolved by name: True` then `OK`. If the registry symbol isn't named `GLOBAL_CONFIG`, inspect `engine/core/__init__.py` and use whatever symbol holds the registry — the assertion's purpose is just to confirm `@register()` ran. If you can't find an introspection path, fall back to verifying the import succeeds with no traceback (the registry call has side effects at import time).

- [ ] **Step 3: Commit**

```bash
git add ecdetseg/engine/edgecrafter/__init__.py
git commit -m "feat(backbone): register ConvNeXtAdapter in package init"
```

---

## Task 4: Add `ecdet_cnxt_t.yml` config

**Files:**
- Create: `ecdetseg/configs/ecdet/ecdet_cnxt_t.yml`

- [ ] **Step 1: Write the config**

Create `ecdetseg/configs/ecdet/ecdet_cnxt_t.yml` with this exact content:

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
  proj_dim: ~                    # null: emit native channels, let HybridEncoder.input_proj unify


HybridEncoder:
  in_channels: [192, 384, 768]   # native ConvNeXt-Tiny stage channels at strides 8/16/32
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
    -
      # backbone weights (non norm/bn/bias): tiny LR to preserve DINOv3 features
      params: '^(?=.*.backbone)(?!.*(?:norm|bn|bias)).*$'
      lr: 0.000005
    -
      # backbone norm/bn/bias: same tiny LR, no weight decay
      params: '^(?=.*.backbone)(?=.*(?:norm|bn|bias)).*$'
      lr: 0.000005
      weight_decay: 0.
    -
      # head norm/bn/bias: no weight decay
      params: '^(?!.*\.backbone)(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.

  lr: 0.0005
  betas: [0.9, 0.999]
  weight_decay: 0.0001


epochs: 62
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

- [ ] **Step 2: Verify the config builds the model**

Run:
```bash
cd /Users/thanhnn5/Workspace/Jitsu/POD/EdgeCrafter/ecdetseg && python -c "
import torch
from engine.core import YAMLConfig
cfg = YAMLConfig('configs/ecdet/ecdet_cnxt_t.yml')
model = cfg.model
print('Model class:', type(model).__name__)
print('Backbone class:', type(model.backbone).__name__)
print('Encoder in_channels:', model.encoder.in_channels)
model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 640, 640))
print('Forward OK')
"
```
Expected: `Model class: ECDet`, `Backbone class: ConvNeXtAdapter`, `Encoder in_channels: [192, 384, 768]`, `Forward OK`. If the YAMLConfig import path differs in this repo, replace with the existing entrypoint used by `train.py` (open `train.py` to confirm — the import name should match what `train.py` uses).

- [ ] **Step 3: Verify optimizer parameter groups**

Run:
```bash
cd /Users/thanhnn5/Workspace/Jitsu/POD/EdgeCrafter/ecdetseg && python -c "
from engine.core import YAMLConfig
cfg = YAMLConfig('configs/ecdet/ecdet_cnxt_t.yml')
model = cfg.model
opt = cfg.optimizer
total_groups = len(opt.param_groups)
for i, g in enumerate(opt.param_groups):
    nparams = sum(p.numel() for p in g['params'])
    print(f'group {i}: lr={g[\"lr\"]:.2e} wd={g[\"weight_decay\"]:.0e} nparams={nparams:,}')
print('Total groups:', total_groups)
assert total_groups >= 3, 'expected at least 3 param groups'
# Backbone groups (lr 5e-6) should hold the bulk of params
backbone_params = sum(sum(p.numel() for p in g['params']) for g in opt.param_groups if g['lr'] == 5e-6)
head_params = sum(sum(p.numel() for p in g['params']) for g in opt.param_groups if g['lr'] != 5e-6)
print(f'backbone params: {backbone_params:,}   head params: {head_params:,}')
assert backbone_params > head_params * 0.5, 'backbone group looks too small — regex may not match'
print('OK')
"
```
Expected: at least 3 param groups, the two with `lr=5.00e-06` collectively hold the bulk of parameters, then `OK`.

- [ ] **Step 4: Commit**

```bash
git add ecdetseg/configs/ecdet/ecdet_cnxt_t.yml
git commit -m "feat(config): add ecdet_cnxt_t.yml for DINOv3 ConvNeXt-Tiny backbone"
```

---

## Task 5: End-to-end smoke training run

**Files:**
- (no code changes)

**Goal:** Validate the full training pipeline executes for a handful of iterations on a single GPU without NaN/OOM/crash, and that loss decreases over the first ~100 steps.

- [ ] **Step 1: Inspect the training entrypoint**

Run:
```bash
head -50 /Users/thanhnn5/Workspace/Jitsu/POD/EdgeCrafter/ecdetseg/train.py
```
Identify (a) the CLI flag that selects the config (likely `-c` based on README example), (b) whether `--test-only` / smoke flags exist. If there's no built-in subset/quick-run flag, the smoke run uses the full config but is cancelled after enough iterations to see loss decrease — that's still a valid pipeline-correctness check.

- [ ] **Step 2: Launch a short training run on one GPU**

Run from the `ecdetseg/` directory:
```bash
cd /Users/thanhnn5/Workspace/Jitsu/POD/EdgeCrafter/ecdetseg && CUDA_VISIBLE_DEVICES=0 python train.py -c configs/ecdet/ecdet_cnxt_t.yml 2>&1 | tee /tmp/ecdet_cnxt_t_smoke.log
```

Let it run until at least 100 training iterations have logged (the existing `print_freq` is 500 in `ecdet.yml` — you may want to temporarily lower it to 20 for this smoke run by editing the config locally; revert before committing anything else). Then Ctrl-C.

- [ ] **Step 3: Inspect the smoke log**

Run:
```bash
grep -E "loss|nan|NaN|Error|OOM|CUDA" /tmp/ecdet_cnxt_t_smoke.log | head -40
```

Pass criteria:
- No `nan`, `NaN`, `Error`, `OOM`, or `CUDA out of memory` lines.
- Loss values logged in the first ~20 iters should be finite, and the trend over ~100 iters should generally decrease (with noise). A loss starting at e.g. ~20 and reaching ~10 within 100 iters is healthy. Loss staying flat or rising indicates a real problem — most likely backbone LR too high or feature dimensions mismatched.
- The `ConvNeXtAdapter loaded:` banner from Task 2 must appear early in the log with non-trivial `param-abs-sum checksum` (pretrained weights actually loaded).

- [ ] **Step 4: If smoke run passes, document this and stop here**

The plan does not include a full COCO training run — that's a separate compute-intensive activity outside this plan's scope. The deliverable is a working configuration; running it to completion is a downstream decision.

Append a short note to the spec's §7 (or open an issue) recording the smoke-run timestamp and observed initial loss trajectory for future reference. No commit required.

- [ ] **Step 5: Final commit (only if any tracked files changed during smoke testing)**

If you temporarily edited `print_freq` for visibility, revert it now. Then:

```bash
git status
# If any tracked files changed unintentionally, revert them:
#   git checkout -- <file>
# If you intend to lower the default print_freq permanently, that's a separate decision — make it deliberately.
```

---

## Self-review checklist (run after writing this plan, before handoff)

- [x] **Spec coverage:** Each spec section is reflected in a task:
  - §3 Architecture / `ConvNeXtAdapter` → Task 2
  - §4.1 Config → Task 4
  - §4.2 Backbone LR 5e-6 → encoded literally in Task 4 config
  - §4.3 Optimizer regex compatibility → verified in Task 4 Step 3
  - §5 Code touch points → Tasks 1–4 cover them 1:1
  - §6 Verification gates → mapped to verification steps in Tasks 2, 3, 4, 5
- [x] **No placeholders:** Every command and code block is concrete. No "TBD" or "implement appropriate X".
- [x] **Type/name consistency:** `ConvNeXtAdapter`, `out_indices`, `proj_dim`, `feat_channels`, `feat_strides` used identically across module, config, and verification commands.
- [x] **File paths are absolute or clearly relative to a stated CWD.**

---

## Out of scope (explicit)

- ConvNeXt-S/B/L variants (Spec §2.4).
- Stage 1 (teacher prep) / Stage 2 (backbone distillation) (Spec §2.4, §9).
- ECInsSeg / ECPose extensions (Spec §2.4).
- Full COCO training to convergence (smoke run only).
- ONNX export, TensorRT benchmarking — separate workstreams.
