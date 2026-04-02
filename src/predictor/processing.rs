use image::{DynamicImage, GenericImageView};
use ndarray::{Array1, Array2, Array4, Axis, s, ArrayView3};
use ort::session::Session;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub bbox: ObjectBBox,
    pub score: f32,
    pub class_id: usize,
    pub mask_weights: Array1<f32>,
}
#[derive(Debug)]
pub struct YoloEngine {
    pub session: Session,
    pub image_size: u32,
    pub stride: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectBBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectMask {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl ObjectMask {
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> bool {
        let bit_idx = (y * self.width + x) as usize;
        self.data
            .get(bit_idx >> 3)
            .is_some_and(|&byte| (byte & (1 << (bit_idx & 7))) != 0)
    }

    #[must_use]
    pub fn to_array2(&self) -> Array2<bool> {
        Array2::from_shape_fn((self.height as usize, self.width as usize), |(y, x)| {
            self.get(x as u32, y as u32)
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectDetection {
    pub bbox: ObjectBBox,
    pub score: f32,
    pub class_id: usize,
    pub tag: String,
    pub mask: Option<ObjectMask>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct YoloPreprocessMeta {
    pub ratio: f32,
    pub pad: (f32, f32),
    pub orig_shape: (u32, u32),
    pub tensor_shape: (u32, u32),
}

pub fn preprocess_image(
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

pub fn reconstruct_mask(
    protos: &ndarray::ArrayView3<f32>,
    weights: &Array1<f32>,
    meta: &YoloPreprocessMeta,
    bbox: &ObjectBBox,
) -> ObjectMask {
    let (mask_c, mask_h, mask_w) = protos.dim();
    let protos_flat = protos
        .view()
        .into_shape_with_order((mask_c, mask_h * mask_w))
        .unwrap();
    let mask_logits = weights
        .dot(&protos_flat)
        .into_shape_with_order((mask_h, mask_w))
        .unwrap();

    let (img_w, img_h) = meta.orig_shape;
    let x_map_factor = meta.ratio * (mask_w as f32 / meta.tensor_shape.0 as f32);
    let y_map_factor = meta.ratio * (mask_h as f32 / meta.tensor_shape.1 as f32);
    let x_offset = meta.pad.0 * (mask_w as f32 / meta.tensor_shape.0 as f32);
    let y_offset = meta.pad.1 * (mask_h as f32 / meta.tensor_shape.1 as f32);

    let ix1 = (bbox.x1.floor() as u32).clamp(0, img_w);
    let iy1 = (bbox.y1.floor() as u32).clamp(0, img_h);
    let ix2 = (bbox.x2.ceil() as u32).clamp(0, img_w);
    let iy2 = (bbox.y2.ceil() as u32).clamp(0, img_h);

    // Precompute X coordinates with half-pixel correction
    let x_coords: Vec<_> = (ix1..ix2)
        .map(|x| {
            let mx = (x as f32 + 0.5).mul_add(x_map_factor, x_offset) - 0.5;
            let mx_f = (mx.floor() as i32).clamp(0, mask_w as i32 - 1) as usize;
            let mx_c = (mx_f + 1).min(mask_w - 1);
            let dx = mx - mx.floor();
            (mx_f, mx_c, dx)
        })
        .collect();

    let mut data = vec![0u8; (img_w as usize * img_h as usize).div_ceil(8)];

    for y in iy1..iy2 {
        // Half-pixel correction for Y
        let my = (y as f32 + 0.5).mul_add(y_map_factor, y_offset) - 0.5;

        if my < 0.0 || my >= (mask_h as f32 - 1.0) {
            // Note: Clamping my here to avoid out of bounds if strictly necessary,
            // but the bilinear logic below handles standard cases.
        }

        let my_f = (my.floor() as i32).clamp(0, mask_h as i32 - 1) as usize;
        let my_c = (my_f + 1).min(mask_h - 1);
        let dy = my - my.floor();
        let inv_dy = 1.0 - dy;
        let row_base = (y * img_w) as usize;

        for (i, &(mx_f, mx_c, dx)) in x_coords.iter().enumerate() {
            let inv_dx = 1.0 - dx;
            let val = inv_dx.mul_add(
                inv_dy.mul_add(mask_logits[[my_f, mx_f]], dy * mask_logits[[my_c, mx_f]]),
                dx * inv_dy.mul_add(mask_logits[[my_f, mx_c]], dy * mask_logits[[my_c, mx_c]]),
            );

            if val > 0.0 {
                let bit_idx = row_base + (ix1 as usize + i);
                data[bit_idx >> 3] |= 1 << (bit_idx & 7);
            }
        }
    }

    ObjectMask {
        width: img_w,
        height: img_h,
        data,
    }
}

pub fn finalize_detections(
    candidates: Vec<Candidate>,
    protos_view: Option<&ArrayView3<f32>>, 
    meta: &YoloPreprocessMeta,
    labels: &[String],
) -> Vec<ObjectDetection> {
    candidates
        .into_par_iter()
        .map(|cand| {
            let final_bbox = ObjectBBox {
                x1: ((cand.bbox.x1 - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32),
                y1: ((cand.bbox.y1 - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32),
                x2: ((cand.bbox.x2 - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32),
                y2: ((cand.bbox.y2 - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32),
            };

            // Only attempt to reconstruct mask if we have both protos and weights
            let mask = if let Some(protos) = protos_view {
                if cand.mask_weights.is_empty() {
                    None
                } else {
                    Some(reconstruct_mask(
                        protos,
                        &cand.mask_weights,
                        meta,
                        &final_bbox,
                    ))
                }
            } else {
                None
            };

            ObjectDetection {
                bbox: final_bbox,
                score: cand.score,
                class_id: cand.class_id,
                tag: labels
                    .get(cand.class_id)
                    .cloned()
                    .unwrap_or_else(|| "unknown".into()),
                mask,
            }
        })
        .collect()
}