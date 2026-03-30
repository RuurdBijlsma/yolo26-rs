use ab_glyph::{FontVec, PxScale};
use color_eyre::Result;
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use object_detector::{Mask, YOLO26Predictor};
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
        fs::create_dir_all(output_dir)?;
    }

    // Load font
    let font_path = "assets/Roboto-Regular.ttf";
    let font_data = fs::read(font_path)
        .expect("Failed to read font file. Please place a .ttf file in assets/");
    let font = FontVec::try_from_vec(font_data).expect("Error constructing Font");

    println!("Processing images...");

    for entry in fs::read_dir(img_dir)? {
        let path = entry?.path();
        if !path.extension().map_or(false, |e| e == "jpg" || e == "png") {
            continue;
        }

        let file_stem = path.file_stem().unwrap().to_string_lossy();
        println!("  Detecting: {}", path.file_name().unwrap().to_string_lossy());

        // 1. Open image and predict
        let img = image::open(&path)?;
        let results = predictor.predict(&img, 0.4, 0.5)?;
        let base_rgba = img.to_rgba8();

        for (idx, det) in results.into_iter().enumerate() {
            let mut det_img = base_rgba.clone();
            let color = get_color(det.class_id);

            // 1. Draw Mask (using the new Mask.get method)
            if let Some(mask) = det.mask {
                apply_mask(&mut det_img, &mask, color);
            }

            // 2. Draw Bounding Box (3px thickness)
            let [x1, y1, x2, y2] = det.bbox;
            let width = (x2 - x1).max(1.0) as u32;
            let height = (y2 - y1).max(1.0) as u32;

            // Draw 3 nested rectangles to simulate line thickness
            for i in 0..3 {
                let rect = Rect::at(x1 as i32 + i, y1 as i32 + i)
                    .of_size(
                        width.saturating_sub((i * 2) as u32).max(1),
                        height.saturating_sub((i * 2) as u32).max(1)
                    );
                draw_hollow_rect_mut(&mut det_img, rect, color);
            }

            // 3. Draw Label with Background
            let label = format!("{} {:.2}", det.tag, det.score);
            let font_size = 24.0; // Increased font size
            let scale = PxScale::from(font_size);

            // Estimate dimensions for the text background box
            let text_height = 28;
            let text_y = (y1 as i32 - text_height).max(0);
            let box_width = (label.len() as u32 * 13).max(40); // Approx width for 24px scale

            draw_filled_rect_mut(
                &mut det_img,
                Rect::at(x1 as i32, text_y).of_size(box_width, text_height as u32),
                color,
            );

            draw_text_mut(
                &mut det_img,
                Rgba([255, 255, 255, 255]),
                x1 as i32 + 4,
                text_y + 2,
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

    println!("Done! Individual detection images saved in {:?}", output_dir);
    Ok(())
}

fn get_color(class_id: usize) -> Rgba<u8> {
    let colors = [
        [255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50],
        [255, 50, 255], [50, 255, 255], [255, 128, 0], [128, 0, 255],
    ];
    let c = colors[class_id % colors.len()];
    Rgba([c[0], c[1], c[2], 255])
}

/// Blends the mask color onto the image using the new bit-packed Mask structure
fn apply_mask(img: &mut RgbaImage, mask: &Mask, color: Rgba<u8>) {
    let (img_w, img_h) = img.dimensions();

    // Iterate through the mask bounds, ensuring we don't go out of image bounds
    for y in 0..mask.height.min(img_h) {
        for x in 0..mask.width.min(img_w) {
            if mask.get(x, y) {
                let pixel = img.get_pixel_mut(x, y);
                // 50% Alpha blending blend
                pixel[0] = ((u32::from(pixel[0]) + u32::from(color[0])) / 2) as u8;
                pixel[1] = ((u32::from(pixel[1]) + u32::from(color[1])) / 2) as u8;
                pixel[2] = ((u32::from(pixel[2]) + u32::from(color[2])) / 2) as u8;
            }
        }
    }
}