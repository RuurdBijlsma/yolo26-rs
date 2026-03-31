# /// script
# requires-python = "==3.12.*"
# dependencies = [
# "ultralytics==8.4.31",
# "opencv-python==4.13.0.92",
# "torch==2.11.0",
# "torchvision==0.26.0",
# "numpy",
# ]
# ///

import os
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 1. Load the model
model = YOLO("yoloe-26n-seg-pf.pt")

# Set paths
img_dir = Path("assets/img")
output_dir = Path("output/python_reference")
output_dir.mkdir(parents=True, exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
images = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]

print(f"--- Starting inference on {len(images)} images ---\n")

for img_path in images:
    # 2. Run Inference
    # Added retina_masks=True for better quality masks
    results = model.predict(
        source=str(img_path),
        conf=0.4,
        save=False,
        imgsz=640,
        retina_masks=True,
        verbose=False
    )

    for result in results:
        img_orig = result.orig_img
        h_orig, w_orig = img_orig.shape[:2] # Get original image dimensions
        class_names = result.names

        if len(result.boxes) == 0:
            print(f"Image: {img_path.name} - No objects found.")
            continue

        print(f"Image: {img_path.name} ({len(result.boxes)} detections found)")

        for i, box in enumerate(result.boxes):
            # Create a fresh copy for this specific detection
            annotator = Annotator(img_orig.copy(), line_width=2)

            # Get box and class data
            b = box.xyxy[0]
            cls = int(box.cls[0])
            label = class_names[cls]
            color = colors(cls, True)

            # 3. Draw the mask for this specific detection
            if result.masks is not None:
                # Get the raw mask tensor
                mask_raw = result.masks.data[i].cpu().numpy()

                # --- DIMENSION FIX ---
                # Resize the low-res model mask to the original image size
                mask_resized = cv2.resize(mask_raw, (w_orig, h_orig))

                # Convert to boolean for the annotator (values > 0.5 become True)
                mask_boolean = mask_resized > 0.5

                # Pass as (1, H, W) array
                annotator.masks(mask_boolean[None], color, alpha=0.5)
                # ---------------------

            # 4. Draw the bounding box and label
            annotator.box_label(b, label, color=color)

            # 5. Save the image
            clean_label = label.replace(" ", "_").replace("/", "_")
            out_filename = f"{img_path.stem}_det_{i}_{clean_label}.jpg"
            save_path = output_dir / out_filename

            cv2.imwrite(str(save_path), annotator.result())
            print(f"  - Saved detection {i} ({label}) to {out_filename}")

        print("-" * 30)