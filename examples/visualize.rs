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
    let output_dir = Path::new("output/visualization");
    if !output_dir.exists() {
        fs::create_dir(output_dir)?;
    }

    // Load font
    let font_path = "assets/Roboto-Regular.ttf";
    let font_data = fs::read(font_path)
        .expect("Failed to read font file. Please place a .ttf file in assets/font/");
    let font = FontVec::try_from_vec(font_data).expect("Error constructing Font");

    println!("Processing images...");

    for entry in fs::read_dir(img_dir)? {
        let path = entry?.path();
        if !path.extension().map_or(false, |e| e == "jpg" || e == "png") {
            continue;
        }

        let file_stem = path.file_stem().unwrap().to_string_lossy();
        println!(
            "  Detecting: {}.{}",
            file_stem,
            path.extension().unwrap().to_string_lossy()
        );

        let results = predictor.predict(&path, 0.4, 0.5)?;
        let base_img = image::open(&path)?.to_rgba8();

        for (idx, det) in results.into_iter().enumerate() {
            let mut det_img = base_img.clone();
            let color = get_color(det.class_id);

            // 1. Draw Mask
            if let Some(mask) = det.mask {
                apply_mask(&mut det_img, &mask, color);
            }

            // 2. Draw Bounding Box
            let [x1, y1, x2, y2] = det.bbox;
            let width = (x2 - x1).max(1.0) as u32;
            let height = (y2 - y1).max(1.0) as u32;
            let rect = Rect::at(x1 as i32, y1 as i32).of_size(width, height);
            draw_hollow_rect_mut(&mut det_img, rect, color);

            // 3. Draw Label with Background
            let label = format!("{} {:.2}", det.tag, det.score);
            let scale = PxScale::from(20.0);

            let text_y = (y1 as i32 - 22).max(0);
            // Estimate text width (approx 11px per char at 20px scale)
            let box_width = (label.len() as u32 * 11).max(20);
            draw_filled_rect_mut(
                &mut det_img,
                Rect::at(x1 as i32, text_y).of_size(box_width, 22),
                color,
            );

            draw_text_mut(
                &mut det_img,
                Rgba([255, 255, 255, 255]),
                x1 as i32 + 2,
                text_y,
                scale,
                &font,
                &label,
            );

            // 4. Save individual file
            let safe_tag = det.tag.replace(' ', "_").replace('/', "-");
            let out_name = format!("{}_{}_{}.png", file_stem, idx, safe_tag);
            det_img.save(output_dir.join(out_name))?;
        }
    }

    println!("Done! Individual detection images saved in ./output/");
    Ok(())
}

fn get_color(class_id: usize) -> Rgba<u8> {
    let colors = [
        [255, 50, 50],
        [50, 255, 50],
        [50, 50, 255],
        [255, 255, 50],
        [255, 50, 255],
        [50, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
    ];
    let c = colors[class_id % colors.len()];
    Rgba([c[0], c[1], c[2], 255])
}

/// Blends the mask color onto the image using stable u32 math
fn apply_mask(img: &mut RgbaImage, mask: &ndarray::Array2<bool>, color: Rgba<u8>) {
    let (h, w) = mask.dim();
    let (img_w, img_h) = img.dimensions();

    for y in 0..h.min(img_h as usize) {
        for x in 0..w.min(img_w as usize) {
            if mask[[y, x]] {
                let pixel = img.get_pixel_mut(x as u32, y as u32);
                // Stable blending using u32 to prevent overflow during addition
                pixel[0] = ((u32::from(pixel[0]) + u32::from(color[0])) / 2) as u8;
                pixel[1] = ((u32::from(pixel[1]) + u32::from(color[1])) / 2) as u8;
                pixel[2] = ((u32::from(pixel[2]) + u32::from(color[2])) / 2) as u8;
            }
        }
    }
}
