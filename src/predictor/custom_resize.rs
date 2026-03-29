use image::{ImageBuffer, Rgb, RgbImage};

/// Replicates `cv2.resize(interpolation=cv2.INTER_LINEAR)` exactly.
/// Using `fast_image_resize` or `image::resize` has been tested with all resize algorithms
/// they all have an MSE of about 30 compared to opencv resize, and produce bad results with
/// this onnx model. Therefore, I had to exactly copy opencv resize implementation. The output
/// of this function has an MSE of 0.25 compared to python's opencv resize, which yolo 26 seems
/// to be trained on, and it is sensitive to changes in resize algo.
pub fn naive_bilinear_opencv(img: &RgbImage, dst_w: u32, dst_h: u32) -> RgbImage {
    let (src_w, src_h) = img.dimensions();
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    let mut out = ImageBuffer::new(dst_w, dst_h);

    for y in 0..dst_h {
        for x in 0..dst_w {
            let source_x = (x as f32 + 0.5).mul_add(scale_x, -0.5);
            let source_y = (y as f32 + 0.5).mul_add(scale_y, -0.5);

            let x1 = source_x.floor() as i32;
            let y1 = source_y.floor() as i32;
            let x2 = x1 + 1;
            let y2 = y1 + 1;

            let dx = source_x - x1 as f32;
            let dy = source_y - y1 as f32;

            let x1_u = x1.clamp(0, src_w as i32 - 1) as u32;
            let y1_u = y1.clamp(0, src_h as i32 - 1) as u32;
            let x2_u = x2.clamp(0, src_w as i32 - 1) as u32;
            let y2_u = y2.clamp(0, src_h as i32 - 1) as u32;

            let p11 = img.get_pixel(x1_u, y1_u);
            let p21 = img.get_pixel(x2_u, y1_u);
            let p12 = img.get_pixel(x1_u, y2_u);
            let p22 = img.get_pixel(x2_u, y2_u);

            let mut rgb = [0u8; 3];
            for c in 0..3 {
                let val = (p22[c] as f32 * dx).mul_add(
                    dy,
                    (p12[c] as f32 * (1.0 - dx)).mul_add(
                        dy,
                        (p11[c] as f32 * (1.0 - dx))
                            .mul_add(1.0 - dy, p21[c] as f32 * dx * (1.0 - dy)),
                    ),
                );
                rgb[c] = (val + 0.5) as u8;
            }
            out.put_pixel(x, y, Rgb(rgb));
        }
    }
    out
}
