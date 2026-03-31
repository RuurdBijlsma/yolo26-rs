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

from ultralytics import YOLOE
from pathlib import Path


def debug_market_person_count():
    # 1. Setup
    model_path = "yoloe-26x-seg.pt"
    img_path = Path("assets/img/market.jpg")

    if not img_path.exists():
        print(f"Error: Could not find {img_path}")
        return

    # 2. Load and Configure
    model = YOLOE(model_path)
    model.set_classes(["person"])

    # 3. Predict
    # Using the same config as your previous scripts
    results = model.predict(
        source=str(img_path),
        conf=0.15,
        imgsz=640,
        verbose=False
    )

    # 4. Extract and Print Count
    # results[0].boxes contains all the detection data
    person_count = len(results[0].boxes)

    print(f"\n{'=' * 30}")
    print(f"DEBUG RESULTS")
    print(f"{'=' * 30}")
    print(f"Image: {img_path.name}")
    print(f"Detected Persons: {person_count}")
    print(f"{'=' * 30}\n")


if __name__ == "__main__":
    debug_market_person_count()
