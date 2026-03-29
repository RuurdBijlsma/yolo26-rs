use color_eyre::Result;
use ndarray::{Array4, s};
use ort::session::Session;
use ort::value::Value;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize, Debug)]
struct PythonMeta {
    shape: Vec<usize>,
    first_5_values_red: Vec<f32>,
}

fn main() -> Result<()> {
    let model_path = "assets/model/dynamic-onnx/yoloe-26l-seg-pf-dynamic-try-3.onnx";
    let img_stem = "cat"; // Matches the filename from Python
    let meta_path = format!("assets/model/dynamic-onnx/debug_data/{}_meta.json", img_stem);
    let tensor_path = format!("assets/model/dynamic-onnx/debug_data/{}_tensor.bin", img_stem);

    // 1. Load Meta
    let meta_str = fs::read_to_string(&meta_path)?;
    let meta: PythonMeta = serde_json::from_str(&meta_str)?;
    let h = meta.shape[2];
    let w = meta.shape[3];

    // 2. Load Binary
    let bytes = fs::read(&tensor_path)?;
    let floats: Vec<f32> = bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // 3. Construct Array
    let input_tensor = Array4::from_shape_vec((1, 3, h, w), floats)?;

    println!("--- BIT VERIFICATION ({}) ---", img_stem);

    println!("Shape: {}x{}", w, h);
    println!("Python Sample: {:?}", meta.first_5_values_red);
    // Note: Marker might differ if Python sample was taken with pad offset
    // and Rust was taken at 0,0. The 'diff' check is the real test.

    // 4. Run Model
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3).unwrap()
        .commit_from_file(model_path)?;

    let input_value = Value::from_array(input_tensor)?;
    let outputs = session.run(ort::inputs!["images" => input_value])?;

    let detections = outputs["detections"].try_extract_array::<f32>()?.to_owned();
    let preds = detections.slice(s![0, .., ..]);

    let mut candidates: Vec<(f32, usize)> = Vec::new();
    for i in 0..preds.nrows() {
        let score = preds[[i, 4]];
        if score > 0.4 {
            candidates.push((score, preds[[i, 5]] as usize));
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("\n--- SCORES FROM IDENTICAL BITS ---");
    if candidates.is_empty() {
        println!("No objects found > 0.4");
    } else {
        for (i, (score, id)) in candidates.iter().take(5).enumerate() {
            println!("{}. Score: {:.8}, ClassID: {}", i, score, id);
        }
    }

    Ok(())
}