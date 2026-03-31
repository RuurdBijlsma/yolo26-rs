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
import os
from pathlib import Path
from ultralytics import YOLO

# This is exactly your "tinkered" wrapper that you confirmed works
class YOLO26SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # We target the inference tuple (detections, protos)
        # For YOLOE-26, this is at index [0] of the raw model output
        inf_out = self.model(x)[0]
        return inf_out[0], inf_out[1]

def export_pf_family():
    # The Prompt-Free family scales
    scales = ["n", "s", "m", "l", "x"]
    output_dir = Path("assets/model")
    output_dir.mkdir(parents=True, exist_ok=True)

    for scale in scales:
        model_name = f"yoloe-26{scale}-seg-pf"
        pt_filename = f"{model_name}.pt"
        onnx_filename = f"{model_name}.onnx"
        vocab_filename = f"vocabulary_4585.json"

        print(f"\n--- Processing {model_name} ---")

        try:
            # 1. Load Model (This handles the auto-download internally)
            yolo_model = YOLO(pt_filename)

            # 2. Export Vocabulary
            # Extracting directly from yolo_model.names ensures compatibility
            vocab = [yolo_model.names[i] for i in range(len(yolo_model.names))]
            with open(output_dir / vocab_filename, "w", encoding="utf-8") as f:
                json.dump(vocab, f, indent=2)

            # 3. Wrap for ONNX
            wrapped_model = YOLO26SegWrapper(yolo_model.model).eval()

            # 4. Dummy input
            fake_input = torch.randn(1, 3, 640, 640)

            # 5. Export with your specific "tinkered" settings
            print(f"Exporting to {onnx_filename}...")
            torch.onnx.export(
                wrapped_model,
                fake_input,
                str(output_dir / onnx_filename),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['detections', 'protos'],
                # Keeping your exact dynamic axes for aspect-ratio flexibility
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'detections': {0: 'batch', 1: 'anchors'},
                    'protos': {0: 'batch', 2: 'p_height', 3: 'p_width'}
                }
            )
            print(f"Successfully exported {model_name}")

        except Exception as e:
            print(f"Failed to process {model_name}: {e}")

if __name__ == "__main__":
    export_pf_family()