import torch
import numpy as np
import cv2
from ultralytics import YOLOE
from pathlib import Path

# Global variable to store the intercepted tensor
intercepted_tensor = None


def capture_hook(module, input, output):
    """Intercepts the tensor entering the first layer of the model."""
    global intercepted_tensor
    # input is a tuple of (tensor,), we take the first element
    intercepted_tensor = input[0].cpu().detach().numpy()


def verify_and_export_reference():
    model_path = "yoloe-26x-seg.pt"
    img_path = Path("img/market.jpg")
    img_size = 640

    # 1. Load Model
    yolo = YOLOE(model_path)

    # 2. Attach the Hook to the VERY FIRST layer of the model
    # For YOLOE, this is model.model.model[0]
    first_layer = yolo.model.model[0]
    handle = first_layer.register_forward_hook(capture_hook)

    print(f"--- Running Predict on {img_path.name} ---")
    # 3. Run a standard prediction.
    # This triggers the internal Ultralytics pipeline (Preprocessing -> Hook -> Forward)
    yolo.predict(
        source=str(img_path),
        imgsz=img_size,
        conf=0.15,
        save=False,
        verbose=False
    )

    # Remove the hook
    handle.remove()

    if intercepted_tensor is None:
        print("Error: Hook failed to capture tensor!")
        return

    # 4. Save the actual "Source of Truth" tensor
    # intercepted_tensor shape is (1, 3, 640, 640)
    np.save("ref_img_tensor.npy", intercepted_tensor)

    # 5. Extract Text Embeddings (These are simpler, usually no hidden preprocessing)
    with torch.no_grad():
        text_pe = yolo.get_text_pe(["person"]).cpu().numpy()
        np.save("ref_text_pe.npy", text_pe)

    # 6. VISUAL EXPORT (for human verification)
    # Reconstruct the image from the intercepted tensor
    # Model expects RGB, so intercepted_tensor is RGB.
    vis_img_rgb = (intercepted_tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    # Convert to BGR for standard image viewer compatibility
    vis_img_bgr = cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("ref_tensor_visual.png", vis_img_bgr)

    print("\n" + "=" * 40)
    print("VERIFICATION COMPLETE")
    print("=" * 40)
    print(f"Successfully intercepted tensor of shape: {intercepted_tensor.shape}")
    print(f"Mean pixel value: {intercepted_tensor.mean():.4f}")
    print(f"Exported: ref_img_tensor.npy")
    print(f"Exported: ref_tensor_visual.png")
    print("=" * 40)


if __name__ == "__main__":
    verify_and_export_reference()