import torch
import numpy as np
import onnxruntime as ort
from ultralytics import YOLOE
from pathlib import Path


def to_numpy(x):
    """Recursively convert tensors/lists/tuples to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        # If it's a list/tuple, we usually want the first 'main' output
        # (the concatenated detections).
        return to_numpy(x[0])
    return x


def compare_outputs():
    # 1. Setup
    onnx_path = "../../assets/prompt_model/yoloe-26x-text-dynamic.onnx"
    pt_path = "yoloe-26x-seg.pt"

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    yolo = YOLOE(pt_path)

    # 2. Load Reference Tensors
    try:
        input_tensor_np = np.load("ref_img_tensor.npy")  # (1, 3, 384, 640)
        text_pe_np = np.load("ref_text_pe.npy")  # (1, 1, 512)
    except FileNotFoundError:
        print("Error: Run diag_export_reference.py first!")
        return

    # --- PART A: ONNX OUTPUT ---
    onnx_outputs = session.run(None, {
        'images': input_tensor_np,
        'text_embeddings': text_pe_np
    })
    onnx_raw = onnx_outputs[0]  # Usually [1, 37, 5040] or [1, 5040, 37]

    # --- PART B: PYTORCH OUTPUT ---
    img_torch = torch.from_numpy(input_tensor_np)
    txt_torch = torch.from_numpy(text_pe_np)

    yolo.model.eval()
    # CRITICAL: Match the wrapper's head setting
    yolo.model.model[-1].end2end = False

    with torch.no_grad():
        # Replicate Backbone + Neck forward pass
        y = []
        x = img_torch
        for m in yolo.model.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in yolo.model.save else None)

        # Call the head
        head = yolo.model.model[-1]
        head_input = [y[j] for j in head.f] + [txt_torch]

        # Get raw output and convert to numpy safely
        pt_raw_output = head(head_input)
        pt_raw = to_numpy(pt_raw_output)

    # --- PART C: COMPARISON ---
    print("\n" + "=" * 40)
    print("OUTPUT TENSOR COMPARISON")
    print("=" * 40)
    print(f"ONNX Raw Shape: {onnx_raw.shape}")
    print(f"PT Raw Shape:   {pt_raw.shape}")

    # Handle Axis Mismatch [Batch, Channels, Anchors] vs [Batch, Anchors, Channels]
    if onnx_raw.shape != pt_raw.shape:
        if onnx_raw.shape == np.swapaxes(pt_raw, 1, 2).shape:
            print("[!] Axis swap detected. Swapping PT axes for comparison...")
            pt_raw = np.swapaxes(pt_raw, 1, 2)
        elif onnx_raw.shape == np.swapaxes(pt_raw, 0, 1).shape:
            # For squeezed outputs
            pt_raw = pt_raw.squeeze()
            onnx_raw = onnx_raw.squeeze()
        else:
            print(f"[!] Shape Mismatch: ONNX {onnx_raw.shape} != PT {pt_raw.shape}")
            # Try to compare anyway if anchor count matches
            if onnx_raw.shape[-1] == pt_raw.shape[-1]:
                print("Attempting comparison on last dimension...")
            else:
                return

    diff = np.abs(onnx_raw - pt_raw)
    print(f"Max Diff:  {diff.max():.6e}")
    print(f"Mean Diff: {diff.mean():.6e}")

    # Analyze the Score Row (Usually index 4 for single-class 'person')
    # preds[batch, row, anchor]
    # Rows: 0,1,2,3 = Box, 4 = Score Class 0, 5... = Masks/Other classes
    onnx_scores = onnx_raw[0, 4, :]
    pt_scores = pt_raw[0, 4, :]

    print("\nSCORE ANALYSIS (First 5 anchors):")
    print(f"  ONNX Scores: {onnx_scores[:5]}")
    print(f"  PT Scores:   {pt_scores[:5]}")

    # Check for Logits vs Probabilities
    if np.max(onnx_scores) > 1.0 or np.min(onnx_scores) < -0.1:
        print("\n[!!!] ALERT: ONNX scores are likely RAW LOGITS. Sigmoid is required.")
        print(f"      Max Score: {np.max(onnx_scores):.2f}, Min Score: {np.min(onnx_scores):.2f}")

    if diff.max() > 1e-2:
        print("\n[!!!] ALERT: Large numerical difference. Check your ONNX export wrapper.")
    else:
        print("\n[SUCCESS] The math inside both models is close enough.")


if __name__ == "__main__":
    compare_outputs()