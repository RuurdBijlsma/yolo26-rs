import torch
import numpy as np
import onnxruntime as ort
import torchvision
from pathlib import Path

def nms_sweep():
    # 1. Setup
    onnx_path = "../../assets/prompt_model/yoloe-26x-text-dynamic.onnx"
    try:
        img_tensor = np.load("ref_img_tensor.npy")
        text_pe = np.load("ref_text_pe.npy")
    except:
        print("Run diag_export_reference.py first!")
        return

    # 2. Run Inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    outputs = session.run(None, {'images': img_tensor, 'text_embeddings': text_pe})

    # 3. Prepare Boxes
    # Shape is [1, 37, 5040] -> [5040, 37]
    preds = np.squeeze(outputs[0]).T
    boxes = preds[:, :4]
    scores = preds[:, 4]

    # Convert cx,cy,w,h to x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 4. Filter by confidence FIRST (to match the 201 candidates)
    conf_mask = scores > 0.15
    candidates_boxes = torch.from_numpy(boxes_xyxy[conf_mask])
    candidates_scores = torch.from_numpy(scores[conf_mask])

    print(f"\n{'=' * 45}")
    print(f"NMS THRESHOLD SWEEP (TORCHVISION NMS)")
    print(f"{'=' * 45}")
    print(f"Candidates found: {len(candidates_scores)}")
    print(f"{'IOU Thresh':<12} | {'Detected People'}")
    print("-" * 30)

    for iou in [0.3, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Using torchvision.ops.nms is the gold standard for YOLO
        indices = torchvision.ops.nms(candidates_boxes, candidates_scores, iou)
        count = len(indices)

        marker = ""
        if iou == 0.7: marker = "<-- YOLO DEFAULT"

        print(f"{iou:<12.2f} | {count:<15} {marker}")

    print("=" * 45)

if __name__ == "__main__":
    nms_sweep()