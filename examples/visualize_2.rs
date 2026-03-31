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
use object_detector::{ObjectDetector, ObjectMask};
use ort::ep::CUDA;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    color_eyre::install()?;

    let mut predictor = ObjectDetector::builder(
        "assets/model/yoloe-26n-seg-pf.onnx",
        "assets/model/vocabulary.json",
    )
    .with_execution_providers(&[CUDA::default().build()])
    .build()?;

    let output_dir = Path::new("output/joined_visualization");
    fs::create_dir_all(output_dir)?;

    let font = FontVec::try_from_vec(fs::read("assets/Roboto-Regular.ttf")?)?;

    for entry in fs::read_dir("assets/img")? {
        let path = entry?.path();

        let img = image::open(&path)?;
        let now = Instant::now();
        let mut results = predictor.predict(&img).call()?;
        println!("Detected objects [{:?}]: {}", now.elapsed(), path.display());

        // Sort by area using struct fields
        results.sort_by(|a, b| {
            let area_a = (a.bbox.x2 - a.bbox.x1) * (a.bbox.y2 - a.bbox.y1);
            let area_b = (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1);
            area_b.partial_cmp(&area_a).unwrap()
        });

        let mut canvas = img.to_rgba8();
        let font_size = (canvas.height() as f32 * 0.025).max(14.0);
        let scale = PxScale::from(font_size);

        for det in &results {
            if let Some(mask) = &det.mask {
                apply_mask(&mut canvas, mask, get_color(det.class_id));
            }
        }

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
        canvas.save(output_dir.join(format!("res_{file_stem}.png")))?;
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
                for i in 0..3 {
                    p[i] = u32::midpoint(u32::from(p[i]), u32::from(color[i])) as u8;
                }
            }
        }
    }
}
