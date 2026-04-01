import cv2
import numpy as np
from pathlib import Path


def letterbox_rectangular(img, new_shape=640, color=(114, 114, 114), stride=32):
    """
    Replicates the Ultralytics Letterbox 'auto' mode used during predict().
    It resizes to the target size but only pads to the nearest stride multiple.
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 1. Calculate Scale Ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 2. Compute unpadded dimensions
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 3. Compute padding for stride
    # Note: We use np.mod here to find the minimum padding needed to reach a multiple of 'stride'
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw = np.mod(dw, stride)
    dh = np.mod(dh, stride)

    # Divide padding into two sides (centering)
    dw /= 2
    dh /= 2

    # 4. Resize if needed
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 5. Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img


def compare_preprocessing_rect():
    img_path = Path("img/market.jpg")
    img_size = 640
    stride = 32

    # 1. Load Reference (Intercepted by the Hook script)
    try:
        ref_img = np.load("ref_img_tensor.npy")
    except FileNotFoundError:
        print("Error: Run the hook-based diag_export_reference.py first!")
        return

    # 2. RUN MANUAL ONNX RECTANGULAR PREPROCESSING
    img0 = cv2.imread(str(img_path))

    # Use the rectangular logic
    onnx_canvas_bgr = letterbox_rectangular(img0, new_shape=img_size, stride=stride)

    # BGR to RGB (Required by YOLOE)
    onnx_canvas_rgb = cv2.cvtColor(onnx_canvas_bgr, cv2.COLOR_BGR2RGB)

    # Normalize and Transpose (HWC to CHW)
    onnx_img_tensor = onnx_canvas_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # 3. VISUAL EXPORT
    vis_img = (onnx_img_tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("onnx_tensor_visual_rect.png", vis_img_bgr)

    # 4. NUMERICAL COMPARISON
    print("\n" + "=" * 40)
    print("DIAGNOSTIC COMPARISON (RECTANGULAR)")
    print("=" * 40)
    print(f"Reference Shape: {ref_img.shape}")
    print(f"Your ONNX Shape:  {onnx_img_tensor.shape}")

    if ref_img.shape != onnx_img_tensor.shape:
        print("\n[!] SHAPE MISMATCH!")
        print(f"Ref: {ref_img.shape} vs ONNX: {onnx_img_tensor.shape}")
        return

    img_diff = np.abs(ref_img - onnx_img_tensor)
    print("-" * 40)
    print(f"Max Diff:  {img_diff.max():.6e}")
    print(f"Mean Diff: {img_diff.mean():.6e}")
    print("=" * 40)

    if img_diff.max() < 1e-3:
        print("SUCCESS: Rectangular Preprocessing matches reference!")
    else:
        print(f"FAILURE: Max Diff is still {img_diff.max():.4f}")


if __name__ == "__main__":
    compare_preprocessing_rect()