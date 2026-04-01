# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "onnxruntime==1.24.4",
#    "opencv-python==4.13.0.92",
#    "numpy",
#    "torch==2.11.0",
#    "torchvision",
#    "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
import torch.nn.functional as F
import clip
from pathlib import Path
from collections import Counter


# --- HELPERS ---

def get_color(i):
    """YOLO-style palette."""
    palette = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49), (72, 249, 10)]
    color = palette[i % len(palette)]
    return (color[2], color[1], color[0])


def letterbox_rectangular(img, new_shape=640, stride=32):
    shape = img.shape[:2]  # h, w
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = (new_shape - new_unpad[0]) % stride, (new_shape - new_unpad[1]) % stride
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (r, r), (dw, dh)


def crop_mask(masks, boxes):
    """Zeroes out mask pixels outside the bounding box to reduce noise."""
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # [N, 1, 1]
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


# --- CORE ---

class PureCLIPEncoder:
    def __init__(self, model_path="mobileclip2_b.ts"):
        print(f"Loading CLIP: {model_path}")
        self.model = torch.jit.load(model_path, map_location="cpu").eval()

    def get_embeddings(self, classes):
        tokens = clip.tokenize(classes)
        with torch.no_grad():
            output = self.model(tokens)
            embeddings = output[0] if isinstance(output, (list, tuple)) else output
            embeddings = embeddings.float()
        return embeddings.numpy()[None]


class YOLOE_Pure_Inference:
    def __init__(self, onnx_path, clip_path="mobileclip2_b.ts"):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.clip = PureCLIPEncoder(clip_path)

    def run(self, img_path, classes, conf_threshold=0.15, iou_threshold=0.7):
        img0 = cv2.imread(str(img_path))
        if img0 is None: return None, []
        ih, iw = img0.shape[:2]

        canvas, ratio, pad = letterbox_rectangular(img0)
        c_h, c_w = canvas.shape[:2]
        img_tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        text_pe = self.clip.get_embeddings(classes)
        outputs = self.session.run(None, {'images': img_tensor, 'text_embeddings': text_pe})
        preds, protos = np.squeeze(outputs[0]).T, np.squeeze(outputs[1])
        nc = len(classes)

        # Decode scores
        scores_matrix = preds[:, 4: 4 + nc]
        max_scores = np.max(scores_matrix, axis=1)
        class_ids = np.argmax(scores_matrix, axis=1)

        mask = max_scores > conf_threshold
        if not np.any(mask): return img0, []

        # Filter by confidence
        boxes_raw = preds[mask, :4]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        coeffs = preds[mask, 4 + nc:]

        # Decode Bbox: cxcywh -> xyxy
        x1, y1 = boxes_raw[:, 0] - boxes_raw[:, 2] / 2, boxes_raw[:, 1] - boxes_raw[:, 3] / 2
        x2, y2 = boxes_raw[:, 0] + boxes_raw[:, 2] / 2, boxes_raw[:, 1] + boxes_raw[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Multi-Class NMS
        offset = class_ids * 4096
        boxes_for_nms = boxes_xyxy + offset[:, None]

        keep = torchvision.ops.nms(torch.from_numpy(boxes_for_nms).float(),
                                   torch.from_numpy(max_scores).float(), iou_threshold).numpy()

        if len(keep) == 0: return img0, []

        final_boxes = boxes_xyxy[keep]
        final_scores = max_scores[keep]
        final_cids = class_ids[keep]
        final_coeffs = coeffs[keep]

        # MASK PROCESSING
        c_p, mh, mw = protos.shape
        masks = (torch.from_numpy(final_coeffs) @ torch.from_numpy(protos).view(c_p, -1)).view(-1, mh, mw)
        masks = F.interpolate(masks[None], (c_h, c_w), mode='bilinear', align_corners=False)[0]
        masks = torch.sigmoid(masks)
        masks = crop_mask(masks, torch.from_numpy(final_boxes))

        ph, pw = int(pad[1]), int(pad[0])
        unh, unw = c_h - 2 * ph, c_w - 2 * pw
        masks = masks[:, ph:ph + unh, pw:pw + unw]

        masks = F.interpolate(masks[None], (ih, iw), mode='bilinear', align_corners=False)[0].gt(0.5).numpy()

        final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - pad[0]) / ratio[0]
        final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - pad[1]) / ratio[1]

        return img0, list(zip(final_boxes, final_scores, final_cids, masks))


def visualize(img, detections, classes, out_path):
    canvas = img.copy()
    overlay = canvas.copy()

    detections = sorted(detections, key=lambda x: x[1])

    for box, score, cid, mask in detections:
        color = get_color(int(cid))
        overlay[mask] = color

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = f"{classes[cid]} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, 0, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(canvas, label, (x1, y1 - 5), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    cv2.imwrite(str(out_path), canvas)


def main():
    ONNX = "yoloe-26x-pure-clip.onnx"
    IMG_DIR = Path("img")
    OUT_DIR = Path("output/pure_onnx")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CLASSES = ["cat", "car", "van", "sign", "person", "lamp", "watermelon"]

    inf = YOLOE_Pure_Inference(ONNX)

    imgs = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))

    for img_p in imgs:
        print(f"\nProcessing {img_p.name}...")
        img, dets = inf.run(img_p, CLASSES)

        if img is not None:
            # --- START TAG COUNTING LOGIC ---
            print(f"--- Result Summary ---")
            print(f"Objects detected: {len(dets)}")

            # Extract labels from detections (index 2 is the class id)
            detected_labels = [CLASSES[d[2]] for d in dets]
            tag_counts = Counter(detected_labels)

            if tag_counts:
                print("--- Counts per Tag ---")
                # Sort by count descending, then by tag name
                for tag, count in tag_counts.most_common():
                    print(f"{tag:<12}: {count}")
            # --- END TAG COUNTING LOGIC ---

            visualize(img, dets, CLASSES, OUT_DIR / f"pure_{img_p.name}")


if __name__ == "__main__":
    main()
