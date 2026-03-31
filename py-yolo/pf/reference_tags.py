# /// script
# requires-python = "==3.12.*"
# dependencies = [
# "ultralytics==8.4.31",
# "opencv-python==4.13.0.92",
# "torch==2.11.0",
# "torchvision==0.26.0",
# ]
# ///

import os
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

model_name = "yoloe-26l-seg-pf.pt"

# 1. Load the Large Prompt-Free model
# This model is capable of Detection, Segmentation, and Open-Vocabulary prompting
model = YOLO(model_name)

ckpt = torch.load(model_name, map_location="cpu", weights_only=False)
# Check if there are metadata keys
print("CKPT METADATA")
print(ckpt.get("metadata", {}))

# Set the path to your images
img_dir = Path("assets/img")
image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

# Get list of images
images = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]

print(f"--- Starting inference on {len(images)} images ---\n")

print(model.overrides.get('mean'))
print(model.overrides.get('std'))

for img_path in images:
    # 2. Run Inference
    # we use 'persist=True' if we were doing video, but here it's just individual images
    # 'retina_masks=True' provides high-resolution segmentation masks
    start_time = time.time()

    # Example of feature: Running with the default 4,585 RAM++ classes
    results = model.predict(
        source=str(img_path),
        conf=0.4,
        save=False,
        imgsz=640,
        verbose=False
    )

    end_time = time.time()
    elapsed = (end_time - start_time) * 1000  # convert to ms

    for result in results:
        # Get labels
        class_names = result.names
        detected_ids = result.boxes.cls.cpu().numpy().astype(int)
        tags = [class_names[idx] for idx in detected_ids]
        unique_tags = set(tags)

        # 3. New Feature: Instance Segmentation
        # Since you are using a '-seg' model, you have access to pixel-perfect masks
        mask_count = 0
        if result.masks is not None:
            mask_count = len(result.masks)

        print(f"Image: {img_path.name}")
        print(f"  - Time: {elapsed:.2f}ms")
        print(f"  - Objects Found: {len(detected_ids)} total, {len(unique_tags)} unique")
        print(f"  - Masks Generated: {mask_count}")
        print(f"  - Tags: {', '.join(unique_tags)}")

        # 4. Feature: Object Counting
        if "car" in tags:
            print(f"  - Alert: Found {tags.count('car')} cars in this image!")

        print("-" * 30)
