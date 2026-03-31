#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use ab_glyph::{FontVec, PxScale};
use color_eyre::Result;
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use object_detector::{ObjectDetector, ObjectMask};
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    color_eyre::install()?;

    let mut predictor = ObjectDetector::builder(
        "assets/model/yoloe-26l-seg-pf.onnx",
        "assets/model/vocabulary.json",
    )
    .build()?;

    let output_dir = Path::new("output/visualization");
    fs::create_dir_all(output_dir)?;

    let font = FontVec::try_from_vec(fs::read("assets/Roboto-Regular.ttf")?)?;

    for entry in fs::read_dir("assets/img")? {
        let path = entry?.path();
        println!("Detecting objects: {}", path.display());

        let file_stem = path.file_stem().unwrap().to_string_lossy();
        let img = image::open(&path)?;
        let results = predictor.predict(&img).call()?;
        let base_rgba = img.to_rgba8();

        for (idx, det) in results.into_iter().enumerate() {
            let mut det_img = base_rgba.clone();
            let color = get_color(det.class_id);

            if let Some(mask) = det.mask {
                apply_mask(&mut det_img, &mask, color);
            }

            let b = det.bbox;
            let (w, h) = ((b.x2 - b.x1).max(1.0) as u32, (b.y2 - b.y1).max(1.0) as u32);

            for i in 0..3 {
                let rect = Rect::at(b.x1 as i32 + i, b.y1 as i32 + i).of_size(
                    w.saturating_sub((i * 2) as u32).max(1),
                    h.saturating_sub((i * 2) as u32).max(1),
                );
                draw_hollow_rect_mut(&mut det_img, rect, color);
            }

            let label = format!("{} {:.2}", det.tag, det.score);
            let scale = PxScale::from(24.0);
            let text_y = (b.y1 as i32 - 28).max(0);
            let box_width = (label.len() as u32 * 13).max(40);

            draw_filled_rect_mut(
                &mut det_img,
                Rect::at(b.x1 as i32, text_y).of_size(box_width, 28),
                color,
            );
            draw_text_mut(
                &mut det_img,
                Rgba([255, 255, 255, 255]),
                b.x1 as i32 + 4,
                text_y + 2,
                scale,
                &font,
                &label,
            );

            let safe_tag = det.tag.replace([' ', '/'], "_");
            det_img.save(output_dir.join(format!("{file_stem}_{idx}_{safe_tag}.png")))?;
        }
    }
    Ok(())
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
    let (img_w, img_h) = img.dimensions();
    for y in 0..mask.height.min(img_h) {
        for x in 0..mask.width.min(img_w) {
            if mask.get(x, y) {
                let p = img.get_pixel_mut(x, y);
                p[0] = u32::midpoint(u32::from(p[0]), u32::from(color[0])) as u8;
                p[1] = u32::midpoint(u32::from(p[1]), u32::from(color[1])) as u8;
                p[2] = u32::midpoint(u32::from(p[2]), u32::from(color[2])) as u8;
            }
        }
    }
}
