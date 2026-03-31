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


class YOLO26StandalonePredictor:
    def __init__(self, onnx_path, vocab_path):
        # Load ONNX Session
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # Load Vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        self.imgsz = 640
        self.stride = 32

    def preprocess(self, img0):
        """
        Manual implementation of Ultralytics Letterbox (rect=True, auto=False).
        """
        h0, w0 = img0.shape[:2]

        # 1. Calculate scaling ratio (long side = 640)
        r = self.imgsz / max(h0, w0)

        # 2. Calculate new unpadded dimensions
        new_unpad_w = int(round(w0 * r))
        new_unpad_h = int(round(h0 * r))

        # 3. Calculate rectangular padding (nearest multiple of stride)
        # This is exactly what the 'rect' diagnostic showed (e.g., 384x640)
        h_pad = int(np.ceil(new_unpad_h / self.stride) * self.stride)
        w_pad = int(np.ceil(new_unpad_w / self.stride) * self.stride)

        # 4. Resize
        img = cv2.resize(img0, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        # 5. Calculate padding for centering
        dw = w_pad - new_unpad_w
        dh = h_pad - new_unpad_h

        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)

        # 6. Apply Padding (Gray color: 114)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 7. BGR to RGB, HWC to CHW, Normalize to 0-1
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, CHW
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0

        # Store metadata for rescaling
        meta = {
            "ratio": r,
            "pad": (left, top),
            "orig_shape": (h0, w0),
            "tensor_shape": (h_pad, w_pad)
        }

        return img[None], meta

    def process_masks(self, protos, mask_weights, meta):
        """Standalone mask reconstruction logic."""
        if len(mask_weights) == 0: return []

        h_ori, w_ori = meta["orig_shape"]
        h_new, w_new = meta["tensor_shape"]
        left, top = meta["pad"]
        r = meta["ratio"]

        mh, mw = protos.shape[1:]  # 160x160

        # Matrix multiplication of weights and protos
        masks = (mask_weights @ protos.reshape(32, -1)).reshape(-1, mh, mw)
        masks = 1 / (1 + np.exp(-masks))  # Sigmoid

        results = []
        for mask in masks:
            # Upscale mask to the network input size (e.g. 384x640)
            mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            # Crop out the gray padding
            mask = mask[top:top + int(h_ori * r), left:left + int(w_ori * r)]
            # Scale back to original image pixels
            mask = cv2.resize(mask, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
            results.append(mask > 0.5)
        return results

    def predict(self, img_path, conf=0.4, iou=0.7):
        img0 = cv2.imread(str(img_path))
        if img0 is None: return [], 0, 0

        # 1. Preprocess
        img_input, meta = self.preprocess(img0)
        # --- DEBUG: VISUALIZE STANDALONE PREPROCESSING ---
        # 1. Take the first image in batch [1, 3, H, W] -> [3, H, W]
        # 2. Transpose to [H, W, 3]
        debug_vis = img_input[0].transpose(1, 2, 0)
        # 3. Denormalize 0-1 to 0-255
        debug_vis = (debug_vis * 255).astype(np.uint8)
        # 4. Convert RGB (internal) back to BGR (OpenCV)
        debug_vis = cv2.cvtColor(debug_vis, cv2.COLOR_RGB2BGR)
        # 5. Save the image
        cv2.imwrite(f"standalone_debug_{Path(img_path).name}", debug_vis)
        # -------------------------------------------------

        # --- EXPORT FOR RUST VERIFICATION ---
        debug_dir = Path("debug_data")
        debug_dir.mkdir(exist_ok=True)

        stem = Path(img_path).stem
        # Save exact float32 binary
        img_input.astype(np.float32).tofile(debug_dir / f"{stem}_tensor.bin")

        # Save exact metadata
        with open(debug_dir / f"{stem}_meta.json", "w") as f:
            json.dump({
                "shape": img_input.shape,  # [1, 3, H, W]
                "ratio": meta["ratio"],
                "pad": meta["pad"],
                "orig_shape": meta["orig_shape"],
                "first_5_values_red": img_input[0, 0, meta["pad"][1], meta["pad"][0]:meta["pad"][0] + 5].tolist()
            }, f, indent=2)
        # -----------------------------------------------------
        # 2. Run ONNX
        start = time.time()
        outputs = self.session.run(None, {'images': img_input})
        elapsed = (time.time() - start) * 1000

        preds = outputs[0][0]  # [300, 38]
        protos = outputs[1][0]  # [32, 160, 160]

        # 3. Confidence Filter
        mask = preds[:, 4] > conf
        preds = preds[mask]
        if len(preds) == 0: return [], 0, elapsed

        # 4. Class-Agnostic NMS (OpenCV Standalone)
        boxes_for_nms = []
        for b in preds[:, :4]:
            boxes_for_nms.append([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])])

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, preds[:, 4].tolist(), conf, iou)
        if len(indices) > 0:
            preds = preds[indices.flatten()]
        else:
            return [], 0, elapsed

        # 5. Rescale Boxes
        r = meta["ratio"]
        left, top = meta["pad"]
        boxes = preds[:, :4]
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= r

        # 6. Generate Tags and Masks
        tags = [self.vocab[int(c)] for c in preds[:, 5]]
        masks = self.process_masks(protos, preds[:, 6:], meta)

        return tags, len(masks), elapsed


def main():
    # Make sure you use the Dynamic ONNX exported previously
    MODEL_ONNX = "assets/model/yoloe-26n-seg-pf.onnx"
    VOCAB_JSON = "assets/model/vocabulary.json"
    IMG_DIR = Path("assets/img")

    predictor = YOLO26StandalonePredictor(MODEL_ONNX, VOCAB_JSON)

    print(f"--- YOLO26 Standalone Inference (No Ultralytics Dep) ---\n")
    for img_path in IMG_DIR.glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg', '.webp']: continue

        tags, mask_count, dt = predictor.predict(img_path)
        print(f"Image: {img_path.name} ({dt:.1f}ms)")
        print(f"  - Objects Found: {len(tags)}")
        if tags:
            print(f"  - Tags: {', '.join(set(tags))}")
        print("-" * 30)


if __name__ == "__main__":
    main()
