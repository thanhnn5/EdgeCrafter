"""
Test inference with a DEIMv2-L MNN model converted from ONNX.

The model expects:
    Input:  float32 (1, 3, H, W)  — ImageNet-normalized, NCHW (name: "images")
    Output: pred_logits (1, num_queries, num_classes)  — unnormalised logits
            pred_boxes  (1, num_queries, 4)            — normalised [cx, cy, w, h]

Usage:
    python tools/inference/mnn_inf.py \
        --model weights/deimv2_l.mnn \
        --image images/pod-138.jpg \
        --conf-thresh 0.5

    # Save annotated result
    python tools/inference/mnn_inf.py \
        --model weights/deimv2_l.mnn \
        --image images/pod-138.jpg \
        --output result.jpg

Requirements:
    pip install MNN opencv-python
"""

import argparse
import time

import cv2
import numpy as np
import MNN
import MNN.expr as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MNN forward types — values must match the MNNForwardType enum in
# include/MNN/MNNForwardType.h. Requesting an id that the linked libMNN was
# not built with (e.g. 'coreml' on Linux, 'cuda' on macOS) makes MNN silently
# fall back to its backupType (CPU) and print "Can't Find type=N backend".
BACKEND_MAP = {
    'cpu':    0,   # MNN_FORWARD_CPU
    'metal':  1,   # MNN_FORWARD_METAL    — Apple Metal GPU (macOS/iOS)
    'cuda':   2,   # MNN_FORWARD_CUDA
    'opencl': 3,   # MNN_FORWARD_OPENCL
    'auto':   4,   # MNN_FORWARD_AUTO     — MNN picks
    'coreml': 5,   # MNN_FORWARD_NN       — CoreML registers itself here
    'opengl': 6,   # MNN_FORWARD_OPENGL
    'vulkan': 7,   # MNN_FORWARD_VULKAN
    # NNAPI / user-slot backends are only valid when libMNN is built with the
    # matching option and registers itself into MNN_FORWARD_USER_{0..3}.
    # Don't add an alias unless the build registers it; otherwise it falls
    # back to CPU silently.
}

# BackendConfig::PrecisionMode — see include/MNN/Interpreter.hpp.
# Note: 'normal' is fp16-where-the-backend-allows, NOT fp32. 'high' is the
# strict fp32 path.
PRECISION_MAP = {
    'normal': 0,   # Precision_Normal  — fp16 storage/compute where supported
    'high':   1,   # Precision_High    — strict fp32
    'low':    2,   # Precision_Low     — int8/quantised fast path where supported
    'low_bf': 3,   # Precision_Low_BF16
}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class MNNDetector:
    def __init__(
        self,
        model_path: str,
        input_name: str = "images",
        output_names: tuple[str, ...] = ("pred_logits", "pred_boxes"),
        input_hw: tuple[int, int] = (640, 640),
        backend: str = "cpu",
        precision: str = "normal",
        num_thread: int = 4,
        cache_path: str | None = None,
    ):
        self.input_name   = input_name
        self.output_names = list(output_names)
        self._input_hw    = input_hw

        if backend not in BACKEND_MAP:
            raise ValueError(f"Unknown backend '{backend}'. Choices: {list(BACKEND_MAP)}")
        if precision not in PRECISION_MAP:
            raise ValueError(f"Unknown precision '{precision}'. Choices: {list(PRECISION_MAP)}")

        config = {
            'backend':   BACKEND_MAP[backend],
            'precision': PRECISION_MAP[precision],
            'numThread': num_thread,
        }
        rt = MNN.nn.create_runtime_manager((config,))
        if cache_path:
            rt.set_cache(cache_path)

        self.net = MNN.nn.load_module_from_file(
            model_path, [input_name], list(output_names),
            runtime_manager=rt,
        )

        print(f"Model loaded: {model_path}")
        print(f"Backend     : {backend} (id={config['backend']})  "
              f"precision={precision} (id={config['precision']})  "
              f"numThread={num_thread}"
              + (f"  cache={cache_path}" if cache_path else ""))
        print(f"Input name  : {input_name}  shape=(1, 3, {input_hw[0]}, {input_hw[1]})")
        for name in output_names:
            print(f"Output name : {name}")

    @property
    def input_hw(self):
        return self._input_hw

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = self.input_hw
        img = cv2.resize(image_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)           # HWC → CHW
        return img[np.newaxis].astype(np.float32)   # → (1, 3, H, W)

    def run(self, input_tensor: np.ndarray):
        h, w = self.input_hw
        # Build NCHW VARP and feed it to the module.
        input_var = F.placeholder([1, 3, h, w], F.NCHW, F.float)
        input_var.write(input_tensor)
        # Module internally uses NC4HW4 for conv-style inputs.
        input_var = F.convert(input_var, F.NC4HW4)

        outputs = self.net.forward([input_var])

        results = []
        for var in outputs:
            var = F.convert(var, F.NCHW)
            arr = np.array(var.read(), copy=True)
            results.append(arr)
        return results


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _cx_cy_wh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) normalised [cx, cy, w, h] → [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """Greedy NMS; returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, dtype=np.int32)


def postprocess(
    outputs,
    orig_hw: tuple[int, int],
    conf_thresh: float = 0.5,
    iou_thresh: float  = 0.5,
):
    """Decode raw model outputs into detection results.

    Returns list of dicts: {box_xyxy, score, class_id}
    """
    if outputs[0].shape[-1] == 4:
        pred_logits, pred_boxes = outputs[1], outputs[0]
    else:
        pred_logits, pred_boxes = outputs[0], outputs[1]

    logits = pred_logits[0]   # (Q, C)
    boxes  = pred_boxes[0]    # (Q, 4)

    probs      = 1 / (1 + np.exp(-logits))   # sigmoid
    class_ids  = probs.argmax(axis=1)
    scores     = probs.max(axis=1)

    mask = scores >= conf_thresh
    scores, class_ids, boxes = scores[mask], class_ids[mask], boxes[mask]

    if len(scores) == 0:
        return []

    oh, ow = orig_hw
    boxes_xyxy = _cx_cy_wh_to_xyxy(boxes)
    boxes_xyxy[:, [0, 2]] *= ow
    boxes_xyxy[:, [1, 3]] *= oh
    boxes_xyxy = boxes_xyxy.clip(0)

    keep = _nms(boxes_xyxy, scores, iou_thresh)

    detections = []
    for i in keep:
        detections.append({
            'box_xyxy': boxes_xyxy[i].tolist(),
            'score':    float(scores[i]),
            'class_id': int(class_ids[i]),
        })
    return detections


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

PALETTE = [
    (0,   255,   0),
    (255,   0,   0),
    (0,   0,   255),
    (255, 255,   0),
    (0,   255, 255),
]


def draw_detections(image_bgr: np.ndarray, detections: list, class_names: list | None = None):
    img = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['box_xyxy']]
        cid   = det['class_id']
        score = det['score']
        color = PALETTE[cid % len(PALETTE)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = class_names[cid] if class_names and cid < len(class_names) else f"cls{cid}"
        text  = f"{label} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - bl - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    output_names = tuple(n.strip() for n in args.output_names.split(','))
    detector = MNNDetector(
        args.model,
        input_name=args.input_name,
        output_names=output_names,
        input_hw=tuple(args.input_size),
        backend=args.backend,
        precision=args.precision,
        num_thread=args.num_thread,
        cache_path=args.cache,
    )

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    orig_hw = image_bgr.shape[:2]

    input_tensor = detector.preprocess(image_bgr)
    print(f"\nImage: {args.image}  original size: {orig_hw[1]}x{orig_hw[0]}")
    print(f"Input tensor: shape={input_tensor.shape}  dtype={input_tensor.dtype}")

    # Warm-up
    detector.run(input_tensor)

    # Timed runs
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        outputs = detector.run(input_tensor)
        times.append(time.perf_counter() - t0)

    avg_ms = 1000 * sum(times) / len(times)
    print(f"\nInference ({args.runs} run{'s' if args.runs > 1 else ''}): avg {avg_ms:.1f} ms")

    print(f"\nRaw output shapes:")
    for i, out in enumerate(outputs):
        print(f"  [{i}] shape={out.shape}  dtype={out.dtype}"
              f"  min={out.min():.4f}  max={out.max():.4f}")

    detections = postprocess(outputs, orig_hw,
                             conf_thresh=args.conf_thresh,
                             iou_thresh=args.iou_thresh)

    print(f"\nDetections (conf>{args.conf_thresh}):")
    if not detections:
        print("  (none)")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det['box_xyxy']]
        print(f"  [{i}] class={det['class_id']}  score={det['score']:.4f}"
              f"  box=[{x1},{y1},{x2},{y2}]")

    if args.output:
        class_names = args.class_names.split(',') if args.class_names else None
        annotated = draw_detections(image_bgr, detections, class_names)
        cv2.imwrite(args.output, annotated)
        print(f"\nAnnotated image saved to: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test inference with a DEIMv2-L MNN model'
    )
    parser.add_argument('--model', type=str,
                        default='weights/deimv2_l.mnn',
                        help='Path to .mnn model (default: weights/deimv2_l.mnn)')
    parser.add_argument('--image', type=str,
                        default='images/pod-138.jpg',
                        help='Input image path (default: images/pod-138.jpg)')
    parser.add_argument('--input-name', type=str, default='images',
                        help='Model input tensor name (default: images)')
    parser.add_argument('--output-names', type=str, default='pred_logits,pred_boxes',
                        help='Comma-separated model output tensor names '
                             '(default: pred_logits,pred_boxes)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                        metavar=('H', 'W'),
                        help='Model input spatial size (default: 640 640)')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='NMS IoU threshold (default: 0.5)')
    parser.add_argument('--class-names', type=str, default=None,
                        help='Comma-separated class names, e.g. "pod,background"')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save annotated output image (optional)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of inference runs for timing (default: 5)')
    parser.add_argument('--backend', type=str, default='cpu',
                        choices=list(BACKEND_MAP.keys()),
                        help='MNN forward backend (default: cpu). '
                             'On macOS try "metal" or "coreml"; '
                             'on Android/desktop GPU try "opencl" or "vulkan".')
    parser.add_argument('--precision', type=str, default='normal',
                        choices=list(PRECISION_MAP.keys()),
                        help='Compute precision: normal=fp16-where-supported, '
                             'high=strict fp32, low=int8/quant fast path, '
                             'low_bf=bf16 (default: normal)')
    parser.add_argument('--num-thread', type=int, default=4,
                        help='CPU thread count / GPU tuning level (default: 4)')
    parser.add_argument('--cache', type=str, default=None,
                        help='Path to runtime cache file (recommended for GPU/CoreML; '
                             'speeds up subsequent loads)')
    args = parser.parse_args()
    main(args)
