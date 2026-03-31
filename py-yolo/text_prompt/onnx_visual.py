# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "onnxruntime==1.24.4",
#    "opencv-python==4.13.0.92",
#    "numpy",
#    "torch==2.11.0",
#    "ultralytics>=8.4.31",
#    "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLOE
from ultralytics.utils.plotting import Annotator, colors
from pathlib import Path

class YOLOE_ONNX_Predictor:
    def __init__(self, onnx_path, model_scale="x"):
        # Load ONNX session
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # Load YOLOE helper for text embeddings
        self.yoloe_helper = YOLOE(f"yoloe-26{model_scale}-seg.pt")
        self.img_size = 640

    def preprocess(self, img0):
        """Standard letterbox: resize to 640 while maintaining aspect ratio."""
        h, w = img0.shape[:2]
        r = self.img_size / max(h, w)
        new_w, new_h = int(round(w * r)), int(round(h * r))

        resized = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create 640x640 canvas (gray padding)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        # BGR to RGB, HWC to CHW, Normalize
        input_tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return input_tensor, r

    def postprocess(self, outputs, ratio, num_classes, conf_threshold=0.15, iou_threshold=0.45):
        """
        outputs[0]: Detections [1, 4 + nc + 32, 8400]
        """
        # Squeeze and Transpose to [8400, 4 + nc + 32]
        preds = np.squeeze(outputs[0]).T

        # Slice predictions: 0-3 are Box, 4 to (4+nc) are Scores
        boxes = preds[:, :4]
        scores_matrix = preds[:, 4 : 4 + num_classes]

        # Confidence filtering
        max_scores = np.max(scores_matrix, axis=1)
        class_ids = np.argmax(scores_matrix, axis=1)

        keep = max_scores > conf_threshold
        boxes = boxes[keep]
        scores = max_scores[keep]
        class_ids = class_ids[keep]

        if len(boxes) == 0: return []

        # Convert xywh (centered) to xyxy
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_threshold, iou_threshold)

        final_results = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                idx = int(i)
                final_results.append({
                    "box": boxes_xyxy[idx],
                    "score": scores[idx],
                    "class_id": class_ids[idx]
                })
        return final_results

    def visualize(self, img0, results, classes, ratio):
        """Draws bounding boxes and labels only."""
        annotator = Annotator(img0.copy(), line_width=2, font_size=12)

        for res in results:
            # Rescale box back to original image dimensions
            box = res['box'] / ratio
            cls_id = res['class_id']
            label = f"{classes[cls_id]} {res['score']:.2f}"
            color = colors(cls_id, True)

            # Draw box and label (Matches run_prompt.py style)
            annotator.box_label(box, label, color=color)

        return annotator.result()

    def run(self, image_path, classes, output_path):
        img0 = cv2.imread(str(image_path))
        if img0 is None: return

        # Preprocess and Embed text
        input_tensor, ratio = self.preprocess(img0)
        with torch.no_grad():
            # Create text embeddings for the current dynamic vocabulary
            embeddings = self.yoloe_helper.get_text_pe(classes)

        # ONNX Inference
        outputs = self.session.run(None, {
            'images': input_tensor,
            'text_embeddings': embeddings.cpu().numpy()
        })

        # Postprocess
        results = self.postprocess(outputs, ratio, len(classes), conf_threshold=0.15)

        # Visualize boxes only
        if results:
            img_res = self.visualize(img0, results, classes, ratio)
            cv2.imwrite(str(output_path), img_res)
            print(f"Processed {image_path.name} -> {len(results)} detections.")
        else:
            print(f"Processed {image_path.name} -> Nothing detected.")

if __name__ == "__main__":
    ONNX_MODEL = "assets/prompt_model/yoloe-26x-text-dynamic.onnx"
    IMG_DIR = Path("assets/img")
    OUT_FOLDER = Path("output/onnx-prompt-no-masks")

    # Vocabulary matching run_prompt.py
    MY_CLASSES = ["cat", "car", "van", "sign", "person", "lamp", "watermelon"]

    if not OUT_FOLDER.exists():
        OUT_FOLDER.mkdir(parents=True, exist_ok=True)

    predictor = YOLOE_ONNX_Predictor(ONNX_MODEL, model_scale="x")

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    images = [f for f in IMG_DIR.iterdir() if f.suffix.lower() in image_extensions]

    print(f"--- Running ONNX Detection Inference on {len(images)} images ---")
    for img_p in images:
        out_p = OUT_FOLDER / f"onnx_det_{img_p.name}"
        predictor.run(img_p, MY_CLASSES, out_p)