import torch
import numpy as np
import onnxruntime as ort
from ultralytics import YOLOE
from pathlib import Path


def check_raw_candidates():
    # 1. Setup
    onnx_path = "../../assets/prompt_model/yoloe-26x-text-dynamic.onnx"
    img_path = Path("img/market.jpg")
    conf_threshold = 0.15

    # 2. Load Tensors (from previous successful diag run)
    input_tensor_np = np.load("ref_img_tensor.npy")
    text_pe_np = np.load("ref_text_pe.npy")

    # 3. Get ONNX Raw Output
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_outputs = session.run(None, {
        'images': input_tensor_np,
        'text_embeddings': text_pe_np
    })

    # Shape: [1, 37, 5040] -> Transpose to [5040, 37]
    preds = np.squeeze(onnx_outputs[0]).T
    scores = preds[:, 4]  # Column 4 is 'person'
    boxes = preds[:, :4]  # Columns 0,1,2,3 are boxes

    # 4. STATISTICAL CHECK
    high_conf_mask = scores > conf_threshold
    num_candidates = np.sum(high_conf_mask)

    print("\n" + "=" * 40)
    print("POST-PROCESS DIAGNOSIS")
    print("=" * 40)
    print(f"Total Anchors: {len(scores)}")
    print(f"Candidates with Score > {conf_threshold}: {num_candidates}")

    if num_candidates > 0:
        print(f"Max Score Found: {np.max(scores):.4f}")
        print(f"Mean Score of Candidates: {np.mean(scores[high_conf_mask]):.4f}")

        print("\nSAMPLE BOX COORDINATES (First 3 candidates):")
        sample_boxes = boxes[high_conf_mask][:3]
        for i, box in enumerate(sample_boxes):
            print(f"  Cand {i}: {box}")
    else:
        print("ALERT: No anchors passed the confidence threshold!")

    # 5. RESTORATION CHECK
    # Check if the boxes are very small or very large
    if num_candidates > 0:
        avg_w = np.mean(boxes[high_conf_mask][:, 2])
        avg_h = np.mean(boxes[high_conf_mask][:, 3])
        print(f"\nAverage Box Size: {avg_w:.2f} x {avg_h:.2f}")

        if avg_w < 1.0 or avg_h < 1.0:
            print("[!!!] ALERT: Boxes are tiny. They are likely normalized (0-1) or ltrb-encoded.")
        elif avg_w > 640 or avg_h > 640:
            print("[!!!] ALERT: Boxes are huge. Decoding is definitely wrong.")

    print("=" * 40)


if __name__ == "__main__":
    check_raw_candidates()