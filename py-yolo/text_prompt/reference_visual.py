# /// script
# requires-python = "==3.12.*"
# dependencies = [
# "ultralytics>=8.4.31",
# "opencv-python>=4.13.0.92",
# "torch==2.11.0",
# "torchvision==0.26.0",
# "clip @ git+https://github.com/ultralytics/CLIP.git",
# "ftfy",
# "regex",
# ]
# ///

import time
import cv2
from pathlib import Path
from ultralytics import YOLOE


def run_yoloe_text_visualization():
    # 1. Setup Models and Paths
    model_name = "yoloe-26x-seg.pt"
    img_dir = Path("assets/img")

    # Create a clear output directory for your experiments
    output_dir = Path("output/text_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Loading YOLOE Model: {model_name} ---")
    model = YOLOE(model_name)

    # 2. Define Custom Vocabulary
    custom_classes = ["cat", "car", "van", "sign", "person", "lamp", "watermelon"]
    print(f"--- Vocabulary Set: {custom_classes} ---")
    model.set_classes(custom_classes)

    # 3. Get Images
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in {img_dir}")
        return

    print(f"Processing {len(images)} images...\n")

    # 4. Inference and Manual Visualization
    for img_path in images:
        print(f"Analyzing: {img_path.name}")
        start_time = time.time()

        # Run Prediction
        results = model.predict(
            source=str(img_path),
            conf=0.15,
            imgsz=640,
            save=False,  # We will handle saving manually via .plot()
            verbose=False
        )

        elapsed = (time.time() - start_time) * 1000

        for result in results:
            # result.plot() creates a BGR numpy array with:
            # - Bounding Boxes
            # - Class Tags (Labels)
            # - Segmentation Masks (alpha-blended)
            annotated_frame = result.plot(
                conf=True,  # Show confidence scores
                line_width=2,  # Thickness of the boxes
                font_size=12,  # Size of the tags
                labels=True,  # Show the tags
                boxes=True  # Show the boxes
            )

            # Define output path
            save_path = output_dir / f"pred_{img_path.name}"

            # Save using OpenCV
            cv2.imwrite(str(save_path), annotated_frame)

            # Console Output for Feedback
            found_indices = result.boxes.cls.cpu().numpy().astype(int)
            found_labels = [result.names[idx] for idx in found_indices]

            print(f"  - Speed: {elapsed:.2f}ms")
            print(f"  - Found: {found_labels if found_labels else 'Nothing detected'}")
            if result.masks is not None:
                print(f"  - Masks: {len(result.masks)} areas segmented")
            print(f"  - Saved to: {save_path}")

        print("-" * 50)

    print(f"\nDone! All visual results are in: {output_dir.absolute()}")


if __name__ == "__main__":
    run_yoloe_text_visualization()
