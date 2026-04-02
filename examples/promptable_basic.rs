use color_eyre::eyre::Context;
use object_detector::predictor::PromptableDetector;
use open_clip_inference::TextEmbedder;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    // --- CONFIGURATION ---
    let model_path = "py-yolo/text_prompt/yoloe-26x-pure-clip.onnx";
    let img_path = "assets/img/streetview2.png";
    let clip_model_id = "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX";

    // Any custom labels you want to find in the image
    let labels = vec!["cat", "car", "van", "sign", "person", "lamp", "watermelon"];

    println!("--- Initializing Promptable Pipeline ---");

    // 1. Initialize CLIP Text Embedder (Rust side)
    // This handles the text tokenization and embedding generation
    let text_embedder = TextEmbedder::from_hf(clip_model_id)
        .build()
        .await
        .wrap_err("Failed to initialize CLIP text embedder")?;

    // 2. Initialize the Detector using the crate's new PromptableDetector
    let mut detector = PromptableDetector::builder(model_path, text_embedder)
        .build()
        .wrap_err("Failed to initialize YOLOE session")?;

    // 3. Load Image
    let img = image::open(img_path).wrap_err("Failed to load image")?;

    // 4. Inference
    // All preprocessing, CLIP embedding, and post-processing (NMS)
    // are now handled internally by the crate.
    println!("Running inference for labels: {:?}...", labels);
    let detections = detector
        .predict(&img, &labels)
        .confidence_threshold(0.15)
        .intersection_over_union(0.7)
        .call()
        .wrap_err("Inference failed")?;

    // 5. Results and Stats
    println!("\n--- Result Summary ---");
    println!("Objects detected: {}", detections.len());

    let mut tag_counts: HashMap<String, usize> = HashMap::new();

    for det in &detections {
        // Track counts per tag
        *tag_counts.entry(det.tag.clone()).or_insert(0) += 1;

        // Bounding boxes are already rescaled to the original image size
        println!(
            "[{:>10}] Score: {:.4} | Box: [{:.1}, {:.1}, {:.1}, {:.1}]",
            det.tag, det.score, det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
        );

        if let Some(_mask) = &det.mask {
            // You can access mask data here if needed
            // e.g., mask.get(x, y)
        }
    }

    if tag_counts.is_empty() {
        println!("\nNo objects found matching the provided labels.");
    } else {
        println!("\n--- Counts per Tag ---");
        let mut sorted_counts: Vec<_> = tag_counts.into_iter().collect();
        sorted_counts.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency descending
        for (tag, count) in sorted_counts {
            println!("{tag:<12}: {count}");
        }
    }

    Ok(())
}