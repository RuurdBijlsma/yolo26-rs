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
from pathlib import Path
from ultralytics import YOLOE

class YOLOE_Promptable_Wrapper(torch.nn.Module):
    def __init__(self, yoloe_model, export_mask=True):
        super().__init__()
        self.model = yoloe_model.model
        self.export_mask = export_mask

        # Configure the head
        self.head = self.model.model[-1]
        self.head.end2end = False
        self.head.export = True

    def forward(self, x, text_embeddings):
        # 1. Generate the Text Projection (TPE)
        projected_text = self.head.get_tpe(text_embeddings)

        # 2. Manual Forward Loop
        # We store ALL outputs in 'y' to prevent NoneType errors in Concat layers
        y = []
        feat = x
        for m in self.model.model:
            if m == self.head:
                # Prepare head input: [Visual_P3, Visual_P4, Visual_P5] + [Text_TPE]
                head_input = [y[j] for j in m.f] + [projected_text]
                out = m(head_input)

                # out[0] is detections: [Batch, 300, 4 + num_classes + 32]
                # out[1] is protos: [Batch, 32, 160, 160]
                if self.export_mask:
                    return out
                else:
                    # Strip mask coefficients (last 32 columns)
                    # Returning only detections tensor
                    return out[0][..., :-32]

            # Standard YOLO layer logic
            # Layer input 'xi' is either the previous output or skip-connections
            if m.f == -1:
                xi = feat
            elif isinstance(m.f, int):
                xi = y[m.f]
            else:
                xi = [y[j] for j in m.f]

            feat = m(xi)
            y.append(feat)

def export_all_prompt_variants():
    scales = ["n", "s", "m", "l", "x"]
    output_dir = Path("assets/model/promptable")
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = 640
    # Trace with 5 classes to establish the dynamic shape logic
    dummy_img = torch.randn(1, 3, img_size, img_size)
    dummy_txt = torch.randn(1, 5, 512)

    for scale in scales:
        pt_file = f"yoloe-26{scale}-seg.pt"

        # --- VARIANT A: SEGMENTATION ---
        print(f"\n--- Exporting {scale.upper()} (SEGMENTATION) ---")
        try:
            # Load fresh model (will download automatically if missing)
            yolo_seg = YOLOE(pt_file)
            wrapper_seg = YOLOE_Promptable_Wrapper(yolo_seg, export_mask=True).eval()

            out_path_seg = output_dir / f"yoloe-26{scale}-pure-clip-seg.onnx"
            torch.onnx.export(
                wrapper_seg,
                (dummy_img, dummy_txt),
                str(out_path_seg),
                opset_version=18,
                input_names=['images', 'text_embeddings'],
                output_names=['output0', 'protos'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'text_embeddings': {0: 'batch', 1: 'num_classes'},
                    'output0': {0: 'batch', 1: 'anchors'},
                    'protos': {0: 'batch'}
                }
            )
            print(f"Success: {out_path_seg.name}")
        except Exception as e:
            print(f"Failed to export {scale} Seg: {e}")

        # --- VARIANT B: DETECTION ---
        print(f"--- Exporting {scale.upper()} (DETECTION) ---")
        try:
            yolo_det = YOLOE(pt_file)
            wrapper_det = YOLOE_Promptable_Wrapper(yolo_det, export_mask=False).eval()

            out_path_det = output_dir / f"yoloe-26{scale}-pure-clip-det.onnx"
            torch.onnx.export(
                wrapper_det,
                (dummy_img, dummy_txt),
                str(out_path_det),
                opset_version=18,
                input_names=['images', 'text_embeddings'],
                output_names=['output0'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'text_embeddings': {0: 'batch', 1: 'num_classes'},
                    'output0': {0: 'batch', 1: 'anchors'}
                }
            )
            print(f"Success: {out_path_det.name}")
        except Exception as e:
            print(f"Failed to export {scale} Det: {e}")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    export_all_prompt_variants()