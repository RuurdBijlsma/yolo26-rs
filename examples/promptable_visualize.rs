#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
use ab_glyph::{FontVec, PxScale};
use color_eyre::Result;
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use object_detector::predictor::{ObjectMask, PromptableDetector};
use open_clip_inference::TextEmbedder;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    // --- CONFIGURATION ---
    let model_path = "assets/model/promptable/yoloe-26x-seg-promptable.onnx";
    let clip_model_id = "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX";
    let output_dir = Path::new("output/promptable_visualize");
    fs::create_dir_all(output_dir)?;

    // Define the vocabulary we want to search for in the images
    let labels = vec!["cat", "car", "van", "sign", "person", "lamp", "watermelon"];

    println!("--- Initializing Promptable Pipeline ---");

    // 1. Setup CLIP Text Embedder
    let text_embedder = TextEmbedder::from_hf(clip_model_id).build().await?;

    // 2. Setup Promptable Detector
    let mut detector = PromptableDetector::builder(model_path, text_embedder).build()?;

    let font = FontVec::try_from_vec(fs::read("assets/Roboto-Regular.ttf")?)?;

    // 3. Process Images
    for entry in fs::read_dir("assets/img")? {
        let path = entry?.path();
        if path.is_dir() {
            continue;
        }

        let img = image::open(&path)?;
        let now = Instant::now();

        // Run inference with the dynamic labels
        let mut results = detector
            .predict(&img, &labels)
            .confidence_threshold(0.15)
            .intersection_over_union(0.7)
            .call()?;

        println!(
            "Detected {} objects in [{:?}] for image: {}",
            results.len(),
            now.elapsed(),
            path.display()
        );

        // Sort by area (largest first) so small objects' tags are drawn on top
        results.sort_by(|a, b| {
            let area_a = (a.bbox.x2 - a.bbox.x1) * (a.bbox.y2 - a.bbox.y1);
            let area_b = (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1);
            area_b
                .partial_cmp(&area_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut canvas = img.to_rgba8();
        let font_size = (canvas.height() as f32 * 0.025).max(14.0);
        let scale = PxScale::from(font_size);

        // Draw Masks first
        for det in &results {
            if let Some(mask) = &det.mask {
                apply_mask(&mut canvas, mask, get_color(det.class_id));
            }
        }

        // Draw Bboxes and Text
        for det in &results {
            let color = get_color(det.class_id);
            let b = det.bbox;

            let rect = Rect::at(b.x1 as i32, b.y1 as i32)
                .of_size((b.x2 - b.x1) as u32, (b.y2 - b.y1) as u32);
            draw_thick_rect(&mut canvas, rect, color, 3);

            let label = format!("{} {:.2}", det.tag, det.score);
            let (tw, th) = (
                (label.len() as f32 * font_size * 0.55) as u32,
                font_size as u32,
            );
            let bg_y = (b.y1 as i32 - th as i32 - 4).max(0);

            draw_filled_rect_mut(
                &mut canvas,
                Rect::at(b.x1 as i32, bg_y).of_size(tw + 6, th + 6),
                color,
            );
            draw_text_mut(
                &mut canvas,
                Rgba([255, 255, 255, 255]),
                b.x1 as i32 + 3,
                bg_y + 2,
                scale,
                &font,
                &label,
            );
        }

        let file_stem = path.file_stem().unwrap().to_string_lossy();
        canvas.save(output_dir.join(format!("prompt_{file_stem}.png")))?;
    }

    println!("Finished! Visual results saved to {}", output_dir.display());
    Ok(())
}

fn draw_thick_rect(img: &mut RgbaImage, rect: Rect, color: Rgba<u8>, thickness: i32) {
    for i in 0..thickness {
        let offset = i - (thickness / 2);
        let x = rect.left() + offset;
        let y = rect.top() + offset;
        let w = (rect.width() as i32 - (offset * 2)).max(1) as u32;
        let h = (rect.height() as i32 - (offset * 2)).max(1) as u32;
        draw_hollow_rect_mut(img, Rect::at(x, y).of_size(w, h), color);
    }
}

const fn get_color(class_id: usize) -> Rgba<u8> {
    let colors = [
        [255, 56, 56],
        [255, 112, 31],
        [255, 178, 29],
        [72, 249, 10],
        [26, 147, 238],
        [20, 54, 243],
        [146, 204, 23],
        [128, 0, 255],
    ];
    let c = colors[class_id % colors.len()];
    Rgba([c[0], c[1], c[2], 255])
}

fn apply_mask(img: &mut RgbaImage, mask: &ObjectMask, color: Rgba<u8>) {
    let (iw, ih) = img.dimensions();
    for y in 0..mask.height.min(ih) {
        for x in 0..mask.width.min(iw) {
            if mask.get(x, y) {
                let p = img.get_pixel_mut(x, y);
                // Blend 50/50
                for i in 0..3 {
                    p[i] = ((u32::from(p[i]) + u32::from(color[i])) / 2) as u8;
                }
            }
        }
    }
}
