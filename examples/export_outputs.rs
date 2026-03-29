use color_eyre::Result;
use object_detector::{Detection, YOLO26Predictor};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Serialize)]
struct SerializableDetection {
    tag: String,
    score: f32,
    bbox: [f32; 4],
    mask_stats: Option<MaskStats>,
}

#[derive(Serialize)]
struct MaskStats {
    width: usize,
    height: usize,
    active_pixels: usize,
}

impl From<Detection> for SerializableDetection {
    fn from(det: Detection) -> Self {
        let mask_stats = det.mask.as_ref().map(|m| MaskStats {
            width: m.ncols(),
            height: m.nrows(),
            active_pixels: m.iter().filter(|&&p| p).count(),
        });

        Self {
            tag: det.tag,
            score: det.score,
            bbox: det.bbox,
            mask_stats,
        }
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let mut predictor = YOLO26Predictor::new(
        "assets/model/yoloe-26l-seg-pf.onnx",
        "assets/model/vocabulary.json",
    )?;

    let img_dir = Path::new("assets/img");
    let mut all_results = BTreeMap::new();

    println!("Exporting model outputs...");

    for entry in fs::read_dir(img_dir)? {
        let path = entry?.path();
        if path.extension().map_or(false, |e| e == "jpg" || e == "png") {
            let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
            println!("Processing {file_name}...");

            // Run prediction with fixed thresholds for consistency
            let results = predictor.predict(&path, 0.4, 0.7)?;

            let serializable: Vec<SerializableDetection> = results
                .into_iter()
                .map(SerializableDetection::from)
                .collect();

            all_results.insert(file_name, serializable);
        }
    }

    let output_path = "assets/expected_outputs.json";
    let json = serde_json::to_string_pretty(&all_results)?;
    fs::write(output_path, json)?;

    println!("✅ Successfully exported results to {output_path}");
    Ok(())
}
