import torch
import numpy as np
from ultralytics import YOLOE
from pathlib import Path


def find_detection_tensor(x):
    """Recursively search for the tensor that looks like the detection output."""
    if isinstance(x, torch.Tensor):
        # We are looking for the tensor with 37 rows (4 box + 1 score + 32 masks)
        if x.ndim == 3 and x.shape[1] == 37:
            return x
        # If it's [Batch, Anchors, 37], we'll find it too
        if x.ndim == 3 and x.shape[2] == 37:
            return x.transpose(1, 2)
        return None

    if isinstance(x, (list, tuple)):
        for item in x:
            result = find_detection_tensor(item)
            if result is not None:
                return result

    if isinstance(x, dict):
        for val in x.values():
            result = find_detection_tensor(val)
            if result is not None:
                return result
    return None


def compare_raw_candidates():
    img_path = Path("assets/img/market.jpg")
    pt_path = "yoloe-26x-seg.pt"
    conf_thresh = 0.15

    # 1. Load Model
    model = YOLOE(pt_path)
    model.set_classes(["person"])

    # 2. Load verified input tensors
    try:
        input_tensor = torch.from_numpy(np.load("ref_img_tensor.npy"))
        text_pe = torch.from_numpy(np.load("ref_text_pe.npy"))
    except FileNotFoundError:
        print("Error: Run diag_export_reference.py first!")
        return

    # 3. Run Inference manually
    model.model.eval()
    model.model.model[-1].end2end = False  # Ensure One-to-Many head is active

    with torch.no_grad():
        # Pass through backbone/neck
        y = []
        x = input_tensor
        for m in model.model.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in model.model.save else None)

        # Pass through head
        head = model.model.model[-1]
        head_input = [y[j] for j in head.f] + [text_pe]

        # Get raw head output
        raw_output = head(head_input)

        # Find the actual [1, 37, 5040] tensor in the mess of tuples
        preds_tensor = find_detection_tensor(raw_output)

        if preds_tensor is None:
            print("Could not find detection tensor in head output!")
            print(f"Output type: {type(raw_output)}")
            return

        # Convert to [Anchors, 37]
        preds = preds_tensor.squeeze(0).transpose(0, 1)

        # 4. Extract Official Candidates
        scores = preds[:, 4]
        mask = scores > conf_thresh
        candidates = preds[mask]

        print("\n" + "=" * 40)
        print("OFFICIAL CANDIDATE ANALYSIS")
        print("=" * 40)
        print(f"Detected Output Shape: {preds_tensor.shape}")
        print(f"Total Anchors:         {len(preds)}")
        print(f"Candidates > {conf_thresh}:    {len(candidates)}")

        if len(candidates) > 0:
            print(f"Max Score:  {scores.max():.4f}")
            print(f"Mean Score: {candidates[:, 4].mean():.4f}")

            print("\nSAMPLE CANDIDATE BOXES (Raw Tensor Values):")
            for i in range(min(3, len(candidates))):
                box = candidates[i, :4].cpu().numpy()
                score = candidates[i, 4].item()
                print(f"  Cand {i}: Box={box}, Score={score:.4f}")

    print("=" * 40)


if __name__ == "__main__":
    compare_raw_candidates()