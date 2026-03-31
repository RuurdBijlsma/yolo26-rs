# /// script
# requires-python = "==3.12.*"
# dependencies = [
# "onnx==1.21.0",
# "opencv-python==4.13.0.92",
# "onnxruntime==1.24.4",
# ]
# ///

import cv2
import numpy as np
import onnxruntime as ort
import json
import time
from pathlib import Path

class YOLO26DynamicPredictor:
    def __init__(self, onnx_path, vocab_path):
        # 1. Load ONNX Session
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # 2. Inspect Model Outputs Dynamically
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]

        # Determine if this is a Segmentation model or Detection-only
        # Segmentation has 2 outputs (detections, protos). Detection has 1.
        self.has_masks = len(self.output_names) > 1

        print(f"Model loaded: {Path(onnx_path).name}")
        print(f"Mode: {'SEGMENTATION' if self.has_masks else 'DETECTION ONLY'}")

        # 3. Load Vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        self.imgsz = 640
        self.stride = 32

    def preprocess(self, img0):
        """Standard Ultralytics Letterbox (rect=True, auto=False)."""
        h0, w0 = img0.shape[:2]
        r = self.imgsz / max(h0, w0)
        new_unpad_w, new_unpad_h = int(round(w0 * r)), int(round(h0 * r))

        # Pad to nearest stride
        h_pad = int(np.ceil(new_unpad_h / self.stride) * self.stride)
        w_pad = int(np.ceil(new_unpad_w / self.stride) * self.stride)

        img = cv2.resize(img0, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        dw, dh = w_pad - new_unpad_w, h_pad - new_unpad_h
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, CHW
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0

        return img[None], {"ratio": r, "pad": (left, top), "orig_shape": (h0, w0), "tensor_shape": (h_pad, w_pad)}

    def process_masks(self, protos, mask_weights, meta):
        """Logic to reconstruct masks from prototypes and coefficients."""
        h_ori, w_ori = meta["orig_shape"]
        h_new, w_new = meta["tensor_shape"]
        left, top = meta["pad"]
        r = meta["ratio"]

        mh, mw = protos.shape[1:]
        # Matrix multiply [N, 32] @ [32, 160*160] -> [N, 160, 160]
        masks = (mask_weights @ protos.reshape(32, -1)).reshape(-1, mh, mw)
        masks = 1 / (1 + np.exp(-masks))  # Sigmoid

        results = []
        for mask in masks:
            # Upscale to tensor size, crop padding, then scale to original
            mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            mask = mask[top:top + int(h_ori * r), left:left + int(w_ori * r)]
            mask = cv2.resize(mask, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
            results.append(mask > 0.5)
        return results

    def predict(self, img_path, conf=0.4, iou=0.45):
        img0 = cv2.imread(str(img_path))
        if img0 is None: return []

        # 1. Inference
        img_input, meta = self.preprocess(img0)
        start = time.time()
        outputs = self.session.run(self.output_names, {'images': img_input})
        dt = (time.time() - start) * 1000

        preds = outputs[0][0]  # [300, 6] or [300, 38]

        # 2. Confidence Filter
        mask = preds[:, 4] > conf
        preds = preds[mask]
        if len(preds) == 0: return [], dt

        # 3. NMS (OpenCV helper)
        boxes_for_nms = []
        for b in preds[:, :4]:
            # Convert xyxy to xywh
            boxes_for_nms.append([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])])

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, preds[:, 4].tolist(), conf, iou)
        if len(indices) > 0:
            preds = preds[indices.flatten()]
        else:
            return [], dt

        # 4. Final Processing
        r = meta["ratio"]
        left, top = meta["pad"]

        # Rescale boxes
        boxes = preds[:, :4].copy()
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= r

        results = []

        # Optional Mask Generation
        masks = []
        if self.has_masks:
            protos = outputs[1][0] # [32, 160, 160]
            mask_weights = preds[:, 6:] # [N, 32]
            masks = self.process_masks(protos, mask_weights, meta)

        for i in range(len(preds)):
            res = {
                "box": boxes[i].tolist(),
                "conf": float(preds[i, 4]),
                "tag": self.vocab[int(preds[i, 5])],
                "mask": masks[i] if self.has_masks else None
            }
            results.append(res)

        return results, dt

def main():
    MODEL_PATH = "assets/model/yoloe-26l-det-pf.onnx"
    VOCAB_PATH = "assets/model/vocabulary_4585.json"
    IMG_DIR = Path("assets/img")

    predictor = YOLO26DynamicPredictor(MODEL_PATH, VOCAB_PATH)

    # Find images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_list = [f for f in IMG_DIR.glob("*") if f.suffix.lower() in valid_extensions]

    total_time = 0

    for img_path in image_list:
        start_tick = time.perf_counter()

        results, inference_dt = predictor.predict(img_path)

        end_tick = time.perf_counter()
        total_dt = (end_tick - start_tick) * 1000
        total_time += total_dt

        # Display Results
        print(f"Image: {img_path.name}")
        print(f"  - Inference (Model): {inference_dt:>7.2f} ms")
        print(f"  - Total (with Post): {total_dt:>7.2f} ms")
        print(f"  - Objects Found: {len(results)}")

        if results:
            unique_tags = sorted(list(set(r['tag'] for r in results)))
            print(f"  - Tags: {', '.join(unique_tags)}")

            if 'mask' in results[0] and results[0]['mask'] is not None:
                print("Has mask: TRUE")
            else:
                print("Has mask: FALSE")

        print("-" * 40)

    # Final Summary
    avg_time = total_time / len(image_list)
    fps = 1000 / avg_time
    print(f"\nBENCHMARK SUMMARY:")
    print(f"  Model: {Path(MODEL_PATH).name}")
    print(f"  Average Time per Image: {avg_time:.2f} ms")
    print(f"  Estimated Throughput:   {fps:.2f} FPS")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()