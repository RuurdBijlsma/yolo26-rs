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
from pathlib import Path


class YOLOE_ONNX_Predictor:
    def __init__(self, onnx_path, model_scale="x"):
        self.session = ort.InferenceSession(onnx_path,
                                            providers=['CPUExecutionProvider'])
        self.yoloe_helper = YOLOE(f"yoloe-26{model_scale}-seg.pt")
        self.img_size = 640
        # Color palette for visualization
        self.colors = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
                       (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134)]

    def preprocess(self, img):
        h, w = img.shape[:2]
        r = self.img_size / max(h, w)
        new_w, new_h = int(w * r), int(h * r)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized
        input_tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return input_tensor, r, (new_h, new_w)

    def postprocess(self, outputs, ratio, pad_hw, num_classes, conf_threshold=0.25,
                    iou_threshold=0.45):
        preds = np.squeeze(outputs[0]).T  # [8400, C]
        protos = np.squeeze(outputs[1])  # [32, 160, 160]

        boxes = preds[:, :4]
        scores_matrix = preds[:, 4: 4 + num_classes]
        mask_coeffs = preds[:, 4 + num_classes:]

        max_scores = np.max(scores_matrix, axis=1)
        class_ids = np.argmax(scores_matrix, axis=1)

        keep = max_scores > conf_threshold
        boxes, scores, class_ids, mask_coeffs = boxes[keep], max_scores[keep], \
        class_ids[keep], mask_coeffs[keep]

        if len(boxes) == 0: return []

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_threshold,
                                   iou_threshold)

        final_results = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                idx = int(i)
                # Rescale box
                rescaled_box = boxes_xyxy[idx] / ratio

                # Mask Math
                m_coeffs = mask_coeffs[idx]
                mask_logits = (m_coeffs @ protos.reshape(32, -1)).reshape(160, 160)
                mask = 1 / (1 + np.exp(-mask_logits))  # Sigmoid

                final_results.append({
                    "box": rescaled_box,
                    "score": scores[idx],
                    "class_id": class_ids[idx],
                    "mask": mask
                })
        return final_results

    def visualize(self, img, results, classes, pad_hw, ratio):
        # Create a copy for the mask overlay

        for res in results:
            box = res['box'].astype(int)
            cls_id = res['class_id']
            score = res['score']
            color = self.colors[cls_id % len(self.colors)]

            # 2. Draw Box and Label
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{classes[cls_id]} {score:.2f}"

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (box[0], box[1] - th - 5), (box[0] + tw, box[1]), color,
                          -1)
            cv2.putText(img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)

        return img

    def run(self, image_path, classes, output_path):
        img0 = cv2.imread(str(image_path))
        input_tensor, ratio, pad_hw = self.preprocess(img0)

        # Get embeddings
        with torch.no_grad():
            embeddings = self.yoloe_helper.get_text_pe(classes)
        embeddings_np = embeddings.cpu().numpy()

        # Inference
        outputs = self.session.run(None, {
            'images': input_tensor,
            'text_embeddings': embeddings_np
        })

        # Post-process
        results = self.postprocess(outputs, ratio, pad_hw, len(classes))

        # Visualize and Save
        if results:
            img_res = self.visualize(img0, results, classes, pad_hw, ratio)
            cv2.imwrite(output_path, img_res)
            print(f"Result saved to {output_path}")
        else:
            print("No objects detected.")


if __name__ == "__main__":
    ONNX_MODEL = "assets/prompt_model/yoloe-26x-text-dynamic.onnx"
    TEST_IMAGE = "assets/img/market.jpg"
    OUT_FOLDER = Path("output/onnx-prompt")
    MY_CLASSES = ["person", "lamp", "watermelon"]

    if not OUT_FOLDER.exists():
        OUT_FOLDER.mkdir(parents=True, exist_ok=True)

    predictor = YOLOE_ONNX_Predictor(ONNX_MODEL)
    predictor.run(TEST_IMAGE, MY_CLASSES, OUT_FOLDER / 'market_res.jpg')
