"""
Test inference with a DEIMv2-L TFLite model using ai_edge_litert CompiledModel.

The model was exported via litert-torch and expects:
    Input:  float32 (1, 3, H, W)  — ImageNet-normalized, NCHW
    Output: pred_logits (1, num_queries, num_classes)  — unnormalised logits
            pred_boxes  (1, num_queries, 4)            — normalised [cx, cy, w, h]

Usage:
    python tools/inference/tflite_inf.py \
        --model weights/deimv2_l_i32_v2.tflite \
        --image images/pod-138.jpg \
        --conf-thresh 0.5

    # Save annotated result
    python tools/inference/tflite_inf.py \
        --model weights/deimv2_l_i32_v2.tflite \
        --image images/pod-138.jpg \
        --output result.jpg

Requirements:
    pip install ai-edge-litert opencv-python
"""

import argparse
import time

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class TFLiteDetector:
    def __init__(self, model_path: str, device: str = 'auto'):
        from ai_edge_litert.interpreter import Interpreter

        # Use Interpreter once to read tensor shape/dtype metadata.
        _interp = Interpreter(model_path=model_path)
        _interp.allocate_tensors()
        inp = _interp.get_input_details()[0]
        out_details = _interp.get_output_details()
        del _interp

        self.input_shape  = tuple(inp['shape'])   # (1, 3, H, W)
        self.input_dtype  = inp['dtype']
        self.output_shapes = [tuple(od['shape']) for od in out_details]
        self.output_sizes  = [int(np.prod(od['shape'])) for od in out_details]
        self.output_dtypes = [od['dtype'] for od in out_details]
        self.output_names  = [od['name'] for od in out_details]

        self.device = device
        self.model, accelerator = self._load_model(model_path, device)
        self.sig_idx = 0

        # Pre-allocate buffers and reuse across calls.
        self._input_buffers  = self.model.create_input_buffers(self.sig_idx)
        self._output_buffers = self.model.create_output_buffers(self.sig_idx)

        print(f"Accelerator : device={device}  -> {accelerator.name}")
        print(f"Model input : {self.input_shape}  dtype={self.input_dtype}")
        for name, shape, dtype in zip(self.output_names, self.output_shapes, self.output_dtypes):
            print(f"Model output: {shape}  dtype={dtype}  name={name}")

    @staticmethod
    def _load_model(model_path: str, device: str):
        """Load a CompiledModel on the requested accelerator.

        'cpu'  -> CPU
        'gpu'  -> GPU (errors out if unsupported)
        'auto' -> try GPU, fall back to CPU on failure
        """
        from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator

        device = device.lower()
        if device == 'cpu':
            accel = HardwareAccelerator.CPU
        elif device == 'gpu':
            accel = HardwareAccelerator.GPU
        elif device == 'auto':
            try:
                model = CompiledModel.from_file(model_path,
                                                hardware_accel=HardwareAccelerator.GPU)
                return model, HardwareAccelerator.GPU
            except Exception as e:
                print(f"[device=auto] GPU unavailable ({e}); falling back to CPU.")
                accel = HardwareAccelerator.CPU
        else:
            raise ValueError(f"Unknown device '{device}' (expected cpu/gpu/auto)")

        return CompiledModel.from_file(model_path, hardware_accel=accel), accel

    @property
    def input_hw(self):
        return self.input_shape[2], self.input_shape[3]  # (H, W)

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = self.input_hw
        img = cv2.resize(image_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)           # HWC → CHW
        return img[np.newaxis].astype(np.float32)   # → (1, 3, H, W)

    def run(self, input_tensor: np.ndarray):
        self._input_buffers[0].write(input_tensor)
        self.model.run_by_index(self.sig_idx, self._input_buffers, self._output_buffers)
        return [
            ob.read(size, dtype).reshape(shape)
            for ob, size, dtype, shape in zip(
                self._output_buffers,
                self.output_sizes,
                self.output_dtypes,
                self.output_shapes,
            )
        ]


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
    # outputs order from litert-torch matches forward() return order:
    #   index 0 → pred_logits  (1, Q, C)
    #   index 1 → pred_boxes   (1, Q, 4)
    # Identify by shape (last dim 4 = boxes).
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

    # Convert to absolute pixel coordinates
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
    detector = TFLiteDetector(args.model, device=args.device)

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
        description='Test inference with a DEIMv2-L TFLite model'
    )
    parser.add_argument('--model', type=str,
                        default='weights/deimv2_l.tflite',
                        help='Path to .tflite model (default: weights/deimv2_l.tflite)')
    parser.add_argument('--image', type=str,
                        default='images/pod-138.jpg',
                        help='Input image path (default: images/pod-138.jpg)')
    parser.add_argument('--input-size', type=int, nargs=2, default=None,
                        metavar=('H', 'W'),
                        help='Override input spatial size (default: read from model)')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='NMS IoU threshold (default: 0.5)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (informational; derived from model output)')
    parser.add_argument('--class-names', type=str, default=None,
                        help='Comma-separated class names, e.g. "pod,background"')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save annotated output image (optional)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'gpu', 'auto'],
                        help='Accelerator: cpu, gpu, or auto=GPU|CPU fallback (default: auto)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of inference runs for timing (default: 5)')
    args = parser.parse_args()
    main(args)
