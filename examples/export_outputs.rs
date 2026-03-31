use color_eyre::Result;
use object_detector::{ObjectBBox, ObjectDetection, YOLO26Predictor};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Serialize)]
struct SerializableDetection {
    tag: String,
    score: f32,
    bbox: ObjectBBox,
    mask_stats: Option<MaskStats>,
}

#[derive(Serialize)]
struct MaskStats {
    width: u32,
    height: u32,
    active_pixels: usize,
}

impl From<ObjectDetection> for SerializableDetection {
    fn from(det: ObjectDetection) -> Self {
        let mask_stats = det.mask.as_ref().map(|m| MaskStats {
            width: m.width,
            height: m.height,
            active_pixels: m.data.iter().map(|&b| b.count_ones() as usize).sum(),
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

    for entry in fs::read_dir(img_dir)? {
        let path = entry?.path();
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        println!("Processing {file_name}...");

        let img = image::open(&path)?;
        let results = predictor.predict(&img, 0.4, 0.7)?;

        let serializable: Vec<SerializableDetection> = results
            .into_iter()
            .map(SerializableDetection::from)
            .collect();

        all_results.insert(file_name, serializable);
    }

    let output_path = "assets/expected_outputs.json";
    fs::write(output_path, serde_json::to_string_pretty(&all_results)?)?;

    println!("✅ Successfully exported results to {output_path}");
    Ok(())
}
