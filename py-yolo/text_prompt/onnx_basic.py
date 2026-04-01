# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "onnxruntime==1.24.4",
#    "opencv-python==4.13.0.92",
#    "numpy",
#    "torch==2.11.0",
#    "ultralytics>=8.4.31",
#    "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///
import cv2
import torch
import numpy as np
import onnxruntime as ort
import torchvision
from pathlib import Path

def letterbox_rectangular(img, new_shape=640, color=(114, 114, 114), stride=32):
    """
    Replicates the Ultralytics Letterbox 'auto' mode used during predict().
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 1. Calculate Scale Ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 2. Compute unpadded dimensions
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 3. Compute padding for stride
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw = np.mod(dw, stride)
    dh = np.mod(dh, stride)

    # Divide padding into two sides (centering)
    dw /= 2
    dh /= 2

    # 4. Resize if needed
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 5. Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, (r, r), (dw, dh)

def rescale_boxes(boxes, r, pad):
    """
    Rescale boxes from letterboxed canvas back to original image coordinates.
    boxes: [N, 4] (x1, y1, x2, y2)
    r: scale ratio
    pad: (dw, dh)
    """
    # Subtract padding
    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding

    # Divide by ratio
    boxes[:, :4] /= r[0]
    return boxes

def run_onnx_basic():
    # --- CONFIGURATION ---
    img_path = Path("img/market.jpg")
    onnx_path = "../../assets/prompt_model/yoloe-26x-text-dynamic.onnx"
    conf_threshold = 0.15
    iou_threshold = 0.7  # Official YOLO default (your sweep showed this matches reference)

    if not img_path.exists():
        print(f"Error: Could not find {img_path}")
        return

    # 1. LOAD IMAGE & PREPROCESS
    img0 = cv2.imread(str(img_path))
    original_shape = img0.shape[:2]

    # Use verified rectangular preprocessing
    img_canvas, ratio, pad = letterbox_rectangular(img0, new_shape=640, stride=32)

    # BGR to RGB, Normalize, HWC to CHW
    img_rgb = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # 2. LOAD TEXT EMBEDDING (Reference from your diag scripts)
    # In a real app, you'd use MobileCLIP here.
    # For debugging parity, we use the saved .npy from the reference run.
    try:
        text_pe = np.load("ref_text_pe.npy")
    except FileNotFoundError:
        print("Error: ref_text_pe.npy not found. Run diag_export_reference.py first!")
        return

    # 3. ONNX INFERENCE
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    outputs = session.run(None, {
        'images': img_tensor,
        'text_embeddings': text_pe
    })

    # Raw output0 shape: [1, 37, 5040] (37 = 4 box + 1 score + 32 mask coeffs)
    preds = np.squeeze(outputs[0]).T  # [5040, 37]

    # 4. DECODE AND FILTER
    # Columns 0,1,2,3: cx, cy, w, h
    # Column 4: Score for 'person'
    scores = preds[:, 4]

    # Filter candidates by confidence first
    conf_mask = scores > conf_threshold
    scores = scores[conf_mask]
    boxes_raw = preds[conf_mask, :4]

    if len(scores) == 0:
        print("No detections found above threshold.")
        return

    # Convert cx, cy, w, h -> x1, y1, x2, y2 (in the 640px space)
    x1 = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    y1 = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    x2 = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
    y2 = boxes_raw[:, 1] + boxes_raw[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 5. NMS (Using Torchvision for 1:1 Parity)
    # We pass the boxes in the 640-canvas coordinate space
    keep_indices = torchvision.ops.nms(
        torch.from_numpy(boxes_xyxy).float(),
        torch.from_numpy(scores).float(),
        iou_threshold
    )

    final_boxes = boxes_xyxy[keep_indices]
    final_scores = scores[keep_indices]

    # 6. RESCALE TO ORIGINAL IMAGE
    final_boxes_rescaled = rescale_boxes(final_boxes.copy(), ratio, pad)

    # Clip boxes to image boundaries
    final_boxes_rescaled[:, [0, 2]] = final_boxes_rescaled[:, [0, 2]].clip(0, original_shape[1])
    final_boxes_rescaled[:, [1, 3]] = final_boxes_rescaled[:, [1, 3]].clip(0, original_shape[0])

    print("\n" + "=" * 30)
    print("ONNX FINAL RESULTS")
    print("=" * 30)
    print(f"Image: {img_path.name}")
    print(f"Original Size: {original_shape[1]}x{original_shape[0]}")
    print(f"Inference Size: {img_canvas.shape[1]}x{img_canvas.shape[0]}")
    print(f"Raw Candidates: {len(scores)}")
    print(f"Detected Persons: {len(final_boxes_rescaled)}")
    print("=" * 30)

    # Optional: Print top 3 to verify box math
    for i in range(min(3, len(final_boxes_rescaled))):
        box = final_boxes_rescaled[i]
        print(f"Box {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] Score: {final_scores[i]:.4f}")

if __name__ == "__main__":
    run_onnx_basic()