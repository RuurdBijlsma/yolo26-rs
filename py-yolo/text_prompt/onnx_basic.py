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
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLOE
from pathlib import Path

def debug_onnx_market_person_count():
    # 1. Setup
    onnx_path = "assets/prompt_model/yoloe-26x-text-dynamic.onnx"
    img_path = Path("assets/img/market.jpg")
    conf_threshold = 0.15
    iou_threshold = 0.45
    img_size = 640

    # 2. Load ONNX and Helper
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    yoloe_helper = YOLOE("yoloe-26x-seg.pt")

    # 3. Preprocess Image (Standard Letterbox)
    img0 = cv2.imread(str(img_path))
    h, w = img0.shape[:2]
    r = img_size / max(h, w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w, :] = resized
    input_tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # 4. Get Text Embeddings for ["person"]
    with torch.no_grad():
        embeddings = yoloe_helper.get_text_pe(["person"]).cpu().numpy()

    # 5. Inference
    outputs = session.run(None, {
        'images': input_tensor,
        'text_embeddings': embeddings
    })

    # 6. Post-processing (Counting logic)
    # Output0 shape: [1, 4 + 1 + 32, 8400] -> Transpose to [8400, 37]
    preds = np.squeeze(outputs[0]).T

    # Slice: 0-3 = box, 4 = score for 'person'
    boxes = preds[:, :4]
    scores = preds[:, 4]

    # Filter by confidence
    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) > 0:
        # Convert xywh to xyxy for NMS
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        person_count = len(indices)
    else:
        person_count = 0

    # 7. Print Output (Matching PyTorch script format)
    print(f"\n{'='*30}")
    print(f"ONNX DEBUG RESULTS")
    print(f"{'='*30}")
    print(f"Image: {img_path.name}")
    print(f"Detected Persons: {person_count}")
    print(f"{'='*30}\n")

if __name__ == "__main__":
    debug_onnx_market_person_count()