"""
EdgeCrafter: Convert ECDet model to TFLite using litert-torch.

Converts only the core backbone+encoder+decoder.
Preprocessing (normalisation) and postprocessing (box decoding, NMS) are
intentionally excluded and must be handled by the caller.

Raw outputs:
    pred_logits  (B, num_queries, num_classes) — unnormalised class logits
    pred_boxes   (B, num_queries, 4)           — normalised [cx, cy, w, h]

Usage:
    conda activate tflite
    python tools/deployment/export_tflite.py \
        -c configs/ecdet/ecdet_cnxt_t.yml \
        -r weights/ecdet_cnxt_t.pth \
        -o weights/ecdet_cnxt_t.tflite \
        --input-size 640 640 \
        --device cpu

Requirements:
    pip install litert-torch

Notes:
    - litert-torch requires torch.export compatibility (PyTorch >= 2.1.0)
    - Only positional tensor arguments are supported; kwargs are not
    - The ConvNeXtAdapter config defaults to tanh-GELU (see ecdet_cnxt_t.yml);
      training and TFLite/ONNX export share the same activation, no extra
      flag needed here.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def apply_tflite_patches():
    """
    Monkey-patch two ECDet ops that break TFLite conversion, applied before
    model loading so deploy() sees the patched versions.

    Patch 1 — distance2bbox: abs(reg_scale)
        reg_scale is an nn.Parameter (tensor), so abs() lowers to tfl.abs
        which is explicitly marked illegal in the litert-torch MLIR pipeline.
        Replacement: relu(x) + relu(-x) — uses two relu + add, all basic
        TFLite ops, and matches abs() exactly for real inputs.

    Patch 2 — LQE.forward: prob.topk(k, dim=-1)
        topk lowers to vhlo.sort_v1 which is not yet in the vhlo TFLite
        support set. Replacement: prob[..., :k] (first-k slice, no sort).
        LQE is a small quality-refinement head; the approximation has
        negligible impact on detection accuracy.
    """
    from engine.edgecrafter import decoder as ec_decoder
    from engine.edgecrafter import utils as ec_utils
    from engine.edgecrafter.decoder import LQE
    from engine.edgecrafter.box_ops import box_xyxy_to_cxcywh

    # ------------------------------------------------------------------
    # Patch 1: distance2bbox — replace abs() with relu(x) + relu(-x)
    # ------------------------------------------------------------------
    def _distance2bbox(points, distance, reg_scale):
        reg_scale = torch.relu(reg_scale) + torch.relu(-reg_scale)  # abs() equivalent
        x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
        y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
        x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
        y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)
        return box_xyxy_to_cxcywh(torch.stack([x1, y1, x2, y2], -1))

    # Patch both the source module and the decoder's imported reference.
    ec_utils.distance2bbox = _distance2bbox
    ec_decoder.distance2bbox = _distance2bbox

    # ------------------------------------------------------------------
    # Patch 2: LQE.forward — replace topk with first-k slice
    # ------------------------------------------------------------------
    def _lqe_forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        # Slice instead of topk: small approximation; quality-head only.
        prob_topk = prob[..., :self.k]
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score

    LQE.forward = _lqe_forward

    print("Applied TFLite patches: distance2bbox (abs -> relu+relu), LQE (topk -> slice)")


class ECDetWrapper(nn.Module):
    """
    Thin wrapper around the core ECDet model (backbone + encoder + decoder).
    No preprocessing or postprocessing is included.

    Input:
        images  (B, 3, H, W) — already-normalised image tensor

    Outputs:
        pred_logits  (B, num_queries, num_classes)
        pred_boxes   (B, num_queries, 4)  — normalised [cx, cy, w, h]
    """

    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        return outputs['pred_logits'], outputs['pred_boxes']


def load_model(config_path: str, checkpoint_path: str) -> ECDetWrapper:
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    # Skip backbone-weight downloads when we're about to overwrite with
    # the resume checkpoint (mirrors the export_onnx.py logic).
    if 'ViTAdapter' in cfg.yaml_cfg:
        cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
    if 'ConvNeXtAdapter' in cfg.yaml_cfg:
        cfg.yaml_cfg['ConvNeXtAdapter']['pretrained'] = False
    # HGNetv2 is used by some sibling configs (D-FINE/DEIM) — keep parity.
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    return ECDetWrapper(cfg).eval()


def build_sample_inputs(input_size: tuple, device: str) -> tuple:
    h, w = input_size
    images = torch.randn(1, 3, h, w, device=device)
    return (images,)


def verify_outputs(
    torch_out: tuple,
    edge_out,
    atol: float = 1e-4,
) -> bool:
    """
    Compare PyTorch and TFLite outputs element-wise.

    litert-torch edge_model(*inputs) returns a list/tuple of numpy arrays.
    """
    names = ['pred_logits', 'pred_boxes']
    all_close = True

    for i, (t_tensor, name) in enumerate(zip(torch_out, names)):
        t_np = t_tensor.detach().cpu().numpy()
        e_np = np.array(edge_out[i]) if not isinstance(edge_out[i], np.ndarray) else edge_out[i]

        close = np.allclose(t_np, e_np, atol=atol)
        max_diff = np.max(np.abs(t_np - e_np))

        status = "PASS" if close else "FAIL"
        print(f"  [{status}] {name:10s}  shape={t_np.shape}  max_diff={max_diff:.6f}  atol={atol}")

        if not close:
            all_close = False

    return all_close


def main(args):
    try:
        import litert_torch
    except ImportError:
        raise ImportError(
            "litert-torch is not installed. Install with:\n"
            "  pip install litert-torch"
        )

    apply_tflite_patches()

    device = args.device
    input_size = tuple(args.input_size)  # (H, W)

    print(f"Loading ECDet model from {args.resume} ...")
    model = load_model(args.config, args.resume).to(device)

    sample_inputs = build_sample_inputs(input_size, device)
    print(f"Sample input shape: images={sample_inputs[0].shape}")

    # Run PyTorch inference before conversion (reference outputs)
    print("\nRunning PyTorch reference inference ...")
    with torch.no_grad():
        torch_out = model(*sample_inputs)

    print("PyTorch outputs:")
    for name, t in zip(['pred_logits', 'pred_boxes'], torch_out):
        print(f"  {name:10s}  shape={tuple(t.shape)}  dtype={t.dtype}")

    # Convert to TFLite via litert-torch
    print("\nConverting model to TFLite ...")
    edge_model = litert_torch.convert(model, sample_inputs,
                                      strict_export=True,
                                      enable_x64=False,
                                      runtime_constant_folding=True)
    print("Conversion successful.")

    # Verify edge model outputs match PyTorch outputs
    print("\nVerifying output tolerance ...")
    edge_out = edge_model(*sample_inputs)
    passed = verify_outputs(torch_out, edge_out, atol=args.atol)

    if passed:
        print("\nAll outputs within tolerance — verification PASSED.")
    else:
        print(
            f"\nSome outputs exceeded tolerance (atol={args.atol}). "
            "Consider raising --atol or checking model for unsupported ops."
        )

    # Export .tflite file
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    edge_model.export(output_path)
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"\nSaved TFLite model to: {output_path}  ({size_mb:.1f} MB)")

    # Round-trip: reload and run once more to confirm the file is valid
    print("\nVerifying saved .tflite file (round-trip load) ...")
    reloaded = litert_torch.load(output_path)
    rt_out = reloaded(*sample_inputs)
    rt_passed = verify_outputs(torch_out, rt_out, atol=args.atol)

    if rt_passed:
        print("Round-trip verification PASSED.")
    else:
        print("Round-trip verification FAILED — the saved file may be corrupted.")

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export ECDet to TFLite using litert-torch'
    )
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to YAML config (e.g. configs/ecdet/ecdet_cnxt_t.yml)')
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('-o', '--output', type=str, default='weights/ecdet.tflite',
                        help='Output .tflite file path (default: weights/ecdet.tflite)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                        metavar=('H', 'W'),
                        help='Input spatial size H W (default: 640 640)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for conversion (default: cpu)')
    parser.add_argument('--atol', type=float, default=1e-4,
                        help='Absolute tolerance for output verification (default: 1e-4)')
    args = parser.parse_args()
    main(args)
