use image::{DynamicImage, imageops::FilterType};
use ndarray::{Array4, ArrayView, IxDyn};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load Vocabulary
    let vocab_raw = fs::read_to_string("assets/model/vocabulary.json")?;
    let vocabulary: Vec<String> = serde_json::from_str(&vocab_raw)?;

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_cpus::get())?
        .commit_from_file("assets/model/yoloe-26l-seg-pf.onnx")?;

    let img_dir = Path::new("assets/img");

    println!("--- Starting inference on images in {:?} ---\n", img_dir);

    for entry in fs::read_dir(img_dir)? {
        let entry = entry?;
        let path = entry.path();
        process_image(&mut session, &path, &vocabulary)?;
    }

    Ok(())
}
fn process_image(
    session: &mut Session,
    img_path: &Path,
    vocabulary: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // 3. Pre-processing
    let img = image::open(img_path)?;
    let input_tensor = preprocess(&img, 640)?;

    // Explicitly convert the ndarray into an ort Value.
    // This is the step used in your other vision.rs file.
    let input_value = Value::from_array(input_tensor)?;

    // Pass the Value into the inputs! macro
    let outputs = session.run(ort::inputs!["images" => input_value])?;

    // 5. Post-processing
    let (shape, data) = outputs["output0"].try_extract_tensor::<f32>()?;
    let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let view = ArrayView::from_shape(IxDyn(&shape_usize), data)?;
    let detections = view.into_dimensionality::<ndarray::Ix3>()?;

    let mut tags = Vec::new();
    let mut mask_count = 0;
    let conf_threshold = 0.4;

    for i in 0..300 {
        let score = detections[[0, i, 4]];
        if score > conf_threshold {
            let class_id = detections[[0, i, 5]] as usize;
            if let Some(label) = vocabulary.get(class_id) {
                tags.push(label.clone());
            }
            mask_count += 1;
        }
    }

    let elapsed = start_time.elapsed();
    let unique_tags: HashSet<_> = tags.iter().collect();

    println!("Image: {}", img_path.file_name().unwrap().to_string_lossy());
    println!("  - Time: {:.2?}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  - Objects Found: {} total, {} unique",
        tags.len(),
        unique_tags.len()
    );
    println!("  - Masks Generated: {}", mask_count);
    println!("  - Tags: {}", tags.join(", "));

    let car_count = tags.iter().filter(|&t| t == "car").count();
    if car_count > 0 {
        println!("  - Alert: Found {} cars in this image!", car_count);
    }
    println!("{}", "-".repeat(30));

    Ok(())
}

fn preprocess(img: &DynamicImage, size: u32) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let resized = img.resize_exact(size, size, FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let mut array = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = f32::from(pixel[0]) / 255.0;
        array[[0, 1, y as usize, x as usize]] = f32::from(pixel[1]) / 255.0;
        array[[0, 2, y as usize, x as usize]] = f32::from(pixel[2]) / 255.0;
    }
    Ok(array)
}
