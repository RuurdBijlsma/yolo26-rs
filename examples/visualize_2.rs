use ab_glyph::{FontVec, PxScale};
use color_eyre::Result;
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use object_detector::YOLO26Predictor;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    color_eyre::install()?;

    let mut predictor = YOLO26Predictor::new(
        "assets/model/yoloe-26l-seg-pf.onnx",
        "assets/model/vocabulary.json",
    )?;

    let img_dir = Path::new("assets/img");
    let output_dir = Path::new("output/joined_visualization");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    let font_path = "assets/Roboto-Regular.ttf";
    let font_data = fs::read(font_path)
        .expect("Failed to read font file.");
    let font = FontVec::try_from_vec(font_data).expect("Error constructing Font");

    println!("Processing images...");

    for entry in fs::read_dir(img_dir)? {
        let path = entry?.path();
        if !path.extension().map_or(false, |e| e == "jpg" || e == "png") {
            continue;
        }

        println!("  Detecting: {}", path.file_name().unwrap().to_string_lossy());

        let img = image::open(&path)?;
        let mut results = predictor.predict(&img, 0.4, 0.5)?;

        // --- SORT BY AREA DESCENDING ---
        results.sort_by(|a, b| {
            let area_a = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
            let area_b = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
            area_b.partial_cmp(&area_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut canvas = image::open(&path)?.to_rgba8();
        let (_w, img_h) = canvas.dimensions();
        let font_size = (img_h as f32 * 0.025).max(14.0);
        let scale = PxScale::from(font_size);

        // --- LAYER 1: Masks (Large to Small) ---
        for det in &results {
            if let Some(mask) = &det.mask {
                apply_mask(&mut canvas, mask, get_color(det.class_id));
            }
        }

        // --- LAYER 2: Boxes and Labels ---
        for det in &results {
            let color = get_color(det.class_id);
            let [x1, y1, x2, y2] = det.bbox;

            let width = (x2 - x1).max(1.0) as u32;
            let height = (y2 - y1).max(1.0) as u32;
            draw_thick_rect(&mut canvas, Rect::at(x1 as i32, y1 as i32).of_size(width, height), color, 3);

            let label = format!("{} {:.2}", det.tag, det.score);
            let text_w = (label.len() as f32 * font_size * 0.55) as u32;
            let text_h = font_size as u32;
            let bg_y = (y1 as i32 - text_h as i32 - 4).max(0);

            draw_filled_rect_mut(
                &mut canvas,
                Rect::at(x1 as i32, bg_y).of_size(text_w + 6, text_h + 6),
                color,
            );

            draw_text_mut(
                &mut canvas,
                Rgba([255, 255, 255, 255]),
                x1 as i32 + 3,
                bg_y + 2,
                scale,
                &font,
                &label,
            );
        }

        let out_path = output_dir.join(format!("res_{}.png", path.file_stem().unwrap().to_string_lossy()));
        canvas.save(out_path)?;
    }

    println!("Finished, output files saved to {}", output_dir.display());

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

fn get_color(class_id: usize) -> Rgba<u8> {
    let colors = [
        [255, 56, 56], [255, 112, 31], [255, 178, 29], [72, 249, 10],
        [26, 147, 238], [20, 54, 243], [146, 204, 23], [128, 0, 255]
    ];
    let c = colors[class_id % colors.len()];
    Rgba([c[0], c[1], c[2], 255])
}

fn apply_mask(img: &mut RgbaImage, mask: &object_detector::Mask, color: Rgba<u8>) {
    let (img_w, img_h) = img.dimensions();

    for y in 0..mask.height.min(img_h) {
        for x in 0..mask.width.min(img_w) {
            if mask.get(x, y) {
                let pixel = img.get_pixel_mut(x, y);
                pixel[0] = ((u32::from(pixel[0]) + u32::from(color[0])) / 2) as u8;
                pixel[1] = ((u32::from(pixel[1]) + u32::from(color[1])) / 2) as u8;
                pixel[2] = ((u32::from(pixel[2]) + u32::from(color[2])) / 2) as u8;
            }
        }
    }
}