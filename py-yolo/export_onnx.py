# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "torch==2.11.0",
#    "ultralytics==8.4.31",
#    "onnxscript==0.6.2",
# ]
# ///

import torch
import json
from ultralytics import YOLO

MODEL_PATH = "yoloe-26l-seg-pf.pt"
OUTPUT_ONNX = "assets/model/yoloe-26l-seg-pf.onnx"
OUTPUT_VOCAB = "assets/model/vocabulary.json"


class YOLO26SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Return only the inference tuple (detections, protos)
        inf_out = self.model(x)[0]
        return inf_out[0], inf_out[1]


def export():
    print(f"--- Loading {MODEL_PATH} ---")
    yolo_model = YOLO(MODEL_PATH)
    wrapped_model = YOLO26SegWrapper(yolo_model.model).eval()

    # Export vocabulary
    vocab = [yolo_model.names[i] for i in range(len(yolo_model.names))]
    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    # Dummy input
    fake_input = torch.randn(1, 3, 640, 640)

    print(f"--- Exporting Dynamic ONNX ---")
    torch.onnx.export(
        wrapped_model,
        fake_input,
        OUTPUT_ONNX,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['detections', 'protos'],
        # Allow height (idx 2) and width (idx 3) to change based on image aspect ratio
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'detections': {0: 'batch', 1: 'anchors'},
            'protos': {0: 'batch', 2: 'p_height', 3: 'p_width'}
        }
    )
    print(f"Success! Model: {OUTPUT_ONNX}")


if __name__ == "__main__":
    export()