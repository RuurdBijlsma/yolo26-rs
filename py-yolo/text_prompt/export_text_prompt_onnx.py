# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "torch==2.11.0",
#    "ultralytics>=8.4.31",
#    "onnx==1.21.0",
#    "numpy",
#    "onnxscript==0.6.2",
#    "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///

import torch
import json
from pathlib import Path
from ultralytics import YOLOE

class YOLOE_Dynamic_Text_Wrapper(torch.nn.Module):
    def __init__(self, yoloe_model):
        super().__init__()
        # self.model is the actual YOLOESegModel (the nn.Module)
        self.model = yoloe_model.model

        # Force One-to-Many head.
        # This is CRITICAL for dynamic classes because the One-to-One (end2end)
        # head uses fixed Top-K sorting that breaks when class counts change.
        self.model.model[-1].end2end = False

    def forward(self, x, txt):
        """
        x: Images [1, 3, 640, 640]
        txt: Pre-computed Text Embeddings [1, N, 512]
        """
        y = []  # List to store intermediate layer outputs for skip-connections

        # Iterate through the layers in the underlying Sequential model
        for m in self.model.model:
            # Handle skip connections (f is the index of the source layer)
            if m.f != -1:
                if isinstance(m.f, int):
                    # Single skip connection
                    feat = y[m.f]
                else:
                    # Multiple skip connections (concat/add)
                    feat = [x if j == -1 else y[j] for j in m.f]
            else:
                # Standard sequential connection
                feat = x

            # Check if this is the Head (the final layer)
            if m == self.model.model[-1]:
                # The YOLOE Head expects a list: [feat_p3, feat_p4, feat_p5, text_embeddings]
                # 'feat' at this stage is already the list of 3 feature maps from the neck.
                head_input = feat + [txt]
                # Returns (raw_detections, protos)
                return m(head_input)

            # Normal layer forward
            x = m(feat)

            # Save output if it's needed for a future skip connection
            y.append(x if m.i in self.model.save else None)

def export_yoloe_text_dynamic():
    # 1. Setup
    model_scale = "l"
    pt_file = f"yoloe-26{model_scale}-seg.pt"
    output_dir = Path("assets/prompt_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"yoloe-26{model_scale}-text-dynamic.onnx"

    print(f"--- Loading YOLOE-{model_scale} ---")
    yolo = YOLOE(pt_file)

    # 2. Prepare Wrapper
    wrapper = YOLOE_Dynamic_Text_Wrapper(yolo).eval()

    # 3. Create Dummy Inputs for Tracing
    # We use 5 classes for the trace, but mark it dynamic later
    dummy_img = torch.randn(1, 3, 640, 640)
    dummy_txt = torch.randn(1, 5, 512)

    print(f"--- Exporting to ONNX: {onnx_path} ---")

    # We use torch.onnx.export.
    # Note: If you are on a very new Torch version, we bypass the dynamo path
    # by ensuring we don't pass 'dynamic_shapes' and stick to 'dynamic_axes'.
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_txt),
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['images', 'text_embeddings'],
            output_names=['output0', 'protos'],
            dynamic_axes={
                'images': {0: 'batch', 2: 'height', 3: 'width'},
                'text_embeddings': {0: 'batch', 1: 'num_classes'},
                'output0': {0: 'batch', 1: 'anchors', 2: 'prediction_width'},
                'protos': {0: 'batch'}
            }
        )

    print("\nSUCCESS!")
    print(f"Output 0 (Detections): [Batch, 8400, 5 + NumClasses + 32]")
    print(f"Output 1 (Prototypes): [Batch, 32, 160, 160]")

if __name__ == "__main__":
    export_yoloe_text_dynamic()