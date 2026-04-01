use color_eyre::eyre::Context;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array4, Axis, Ix2, s};
use object_detector::predictor::nms::non_maximum_suppression;
use object_detector::predictor::{ObjectBBox, YoloPreprocessMeta};
use open_clip_inference::TextEmbedder;
use ort::session::Session;
use ort::value::Value;
use rayon::prelude::*;
use std::collections::HashMap;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    // --- CONFIGURATION ---
    let model_path = "py-yolo/text_prompt/yoloe-26x-pure-clip.onnx";
    let img_path = "assets/img/parking_lot.png";
    let clip_model_id = "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX";
    let labels = vec!["cat", "car", "van", "sign", "person", "lamp", "watermelon"];
    let conf_threshold = 0.15;
    let iou_threshold = 0.7;

    println!("--- Initializing Promptable Pipeline ---");

    // 1. Initialize CLIP Text Embedder (Rust side)
    let rt = tokio::runtime::Runtime::new()?;
    let text_embedder =
        rt.block_on(async { TextEmbedder::from_hf(clip_model_id).build().await })?;

    // 2. Initialize YOLOE Session
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(num_cpus::get())
        .unwrap()
        .commit_from_file(model_path)?;

    // 3. Load and Preprocess Image (Using proven bilinear logic)
    let img = image::open(img_path).wrap_err("Failed to load image")?;
    let (img_tensor, meta) = preprocess(&img, 640, 32);

    // 4. Generate CLIP Embeddings
    println!("Generating embeddings for {} labels...", labels.len());
    let text_embs = text_embedder.embed_texts(&labels)?;
    let text_tensor = text_embs.insert_axis(Axis(0)); // [1, N, 512]

    // 5. Inference
    println!("Running YOLOE inference...");
    let outputs = session.run(ort::inputs![
        "images" => Value::from_array(img_tensor)?,
        "text_embeddings" => Value::from_array(text_tensor)?
    ])?;

    // 6. Post-processing
    let raw_output = outputs["output0"].try_extract_array::<f32>()?;
    let preds_2d = raw_output
        .slice(s![0, .., ..])
        .into_dimensionality::<Ix2>()?
        .reversed_axes(); // Result: [8400, features]

    let num_classes = labels.len();
    let mut candidate_boxes = Vec::new();
    let mut candidate_scores = Vec::new();
    let mut candidate_labels = Vec::new();

    for i in 0..preds_2d.shape()[0] {
        let row = preds_2d.row(i);
        let scores = row.slice(s![4..4 + num_classes]);

        let mut max_score = 0.0f32;
        let mut max_cls_id = 0;
        for (idx, &s) in scores.iter().enumerate() {
            if s > max_score {
                max_score = s;
                max_cls_id = idx;
            }
        }

        if max_score > conf_threshold {
            let cx = row[0];
            let cy = row[1];
            let w = row[2];
            let h = row[3];

            candidate_boxes.push(ObjectBBox {
                x1: cx - w / 2.0,
                y1: cy - h / 2.0,
                x2: cx + w / 2.0,
                y2: cy + h / 2.0,
            });
            candidate_scores.push(max_score);
            candidate_labels.push((max_cls_id, labels[max_cls_id].to_string()));
        }
    }

    // 7. Non-Maximum Suppression
    let kept_indices = non_maximum_suppression(&candidate_boxes, &candidate_scores, iou_threshold);

    println!("\n--- Result Summary ---");
    println!("Objects detected: {}", kept_indices.len());

    let mut tag_counts: HashMap<String, usize> = HashMap::new();

    for &idx in &kept_indices {
        let raw_box = &candidate_boxes[idx];
        let (_cls_id, ref label) = candidate_labels[idx];

        // Update counts
        *tag_counts.entry(label.clone()).or_insert(0) += 1;

        // 8. Rescale to original image coordinates
        let x1 = ((raw_box.x1 - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32);
        let y1 = ((raw_box.y1 - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32);
        let x2 = ((raw_box.x2 - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32);
        let y2 = ((raw_box.y2 - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32);

        println!(
            "[{:>10}] Score: {:.4} | Box: [{:.1}, {:.1}, {:.1}, {:.1}]",
            label, candidate_scores[idx], x1, y1, x2, y2
        );
    }

    if !tag_counts.is_empty() {
        println!("\n--- Counts per Tag ---");
        let mut sorted_counts: Vec<_> = tag_counts.into_iter().collect();
        sorted_counts.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency descending
        for (tag, count) in sorted_counts {
            println!("{:<12}: {}", tag, count);
        }
    }

    Ok(())
}

fn preprocess(
    img: &DynamicImage,
    image_size: u32,
    stride: u32,
) -> (Array4<f32>, YoloPreprocessMeta) {
    let (w0, h0) = img.dimensions();
    let ratio = image_size as f32 / (w0.max(h0) as f32);
    let unpad_w = (w0 as f32 * ratio).round() as u32;
    let unpad_h = (h0 as f32 * ratio).round() as u32;

    let w_pad = ((unpad_w as f32 / stride as f32).ceil() * stride as f32) as u32;
    let h_pad = ((unpad_h as f32 / stride as f32).ceil() * stride as f32) as u32;

    let left = (w_pad - unpad_w) / 2;
    let top = (h_pad - unpad_h) / 2;

    let mut input = Array4::from_elem((1, 3, h_pad as usize, w_pad as usize), 114.0 / 255.0);
    let img_rgb = img.to_rgb8();
    let (src_w, src_h) = img_rgb.dimensions();
    let src_raw = img_rgb.as_raw();

    let scale_x = src_w as f32 / unpad_w as f32;
    let scale_y = src_h as f32 / unpad_h as f32;

    let mut content_view = input.slice_mut(s![
        0,
        ..,
        top as usize..(top + unpad_h) as usize,
        left as usize..(left + unpad_w) as usize
    ]);

    content_view
        .axis_iter_mut(Axis(1))
        .enumerate()
        .par_bridge()
        .for_each(|(y, mut row_channels)| {
            let sy = (y as f32 + 0.5).mul_add(scale_y, -0.5);
            let y1 = sy.floor() as i32;
            let dy = sy - y1 as f32;
            let y1_u = y1.clamp(0, src_h as i32 - 1) as u32;
            let y2_u = (y1 + 1).clamp(0, src_h as i32 - 1) as u32;
            let inv_dy = 1.0 - dy;

            for x in 0..unpad_w {
                let sx = (x as f32 + 0.5).mul_add(scale_x, -0.5);
                let x1 = sx.floor() as i32;
                let dx = sx - x1 as f32;
                let x1_u = x1.clamp(0, src_w as i32 - 1) as u32;
                let x2_u = (x1 + 1).clamp(0, src_w as i32 - 1) as u32;
                let inv_delta_x = 1.0 - dx;

                for c in 0..3 {
                    let get_p = |px, py| f32::from(src_raw[((py * src_w + px) as usize * 3) + c]);
                    let val = (get_p(x2_u, y2_u) * dx).mul_add(
                        dy,
                        (get_p(x1_u, y2_u) * inv_delta_x).mul_add(
                            dy,
                            (get_p(x1_u, y1_u) * inv_delta_x)
                                .mul_add(inv_dy, get_p(x2_u, y1_u) * dx * inv_dy),
                        ),
                    );
                    row_channels[[c, x as usize]] = (val + 0.5).floor() / 255.0;
                }
            }
        });

    (
        input,
        YoloPreprocessMeta {
            ratio,
            pad: (left as f32, top as f32),
            orig_shape: (w0, h0),
            tensor_shape: (w_pad, h_pad),
        },
    )
}