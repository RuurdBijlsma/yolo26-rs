use crate::model_manager::{HfModel, get_hf_model};
use crate::predictor::nms::non_maximum_suppression;
use color_eyre::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array1, Array2, Array4, Axis, s};
use ort::session::Session;
use ort::value::Value;
use rayon::prelude::*;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Mask {
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> bool {
        let bit_idx = (y * self.width + x) as usize;
        let byte_idx = bit_idx >> 3;
        let bit_offset = bit_idx & 7;
        (self.data[byte_idx] & (1 << bit_offset)) != 0
    }

    #[must_use]
    pub fn to_array2(&self) -> Array2<bool> {
        let mut arr = Array2::from_elem((self.height as usize, self.width as usize), false);
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) {
                    arr[[y as usize, x as usize]] = true;
                }
            }
        }
        arr
    }
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub class_id: usize,
    pub tag: String,
    pub mask: Option<Mask>,
}

pub struct PreprocessMeta {
    pub ratio: f32,
    pub pad: (f32, f32),
    pub orig_shape: (u32, u32),
    pub tensor_shape: (u32, u32),
}

pub struct YOLO26Predictor {
    pub session: Session,
    pub vocab: Vec<String>,
    pub imgsz: u32,
    pub stride: u32,
}

impl YOLO26Predictor {
    pub async fn from_hf() -> Result<Self> {
        let model_path = get_hf_model(HfModel::default_model()).await?;
        get_hf_model(HfModel::default_data()).await?;
        let vocabulary_path = get_hf_model(HfModel::default_vocabulary()).await?;

        Self::new(model_path, vocabulary_path)
    }

    pub fn new(model_path: impl AsRef<Path>, vocab_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(num_cpus::get())
            .unwrap()
            .commit_from_file(model_path)?;

        let vocab_file = fs::read_to_string(vocab_path)?;
        let vocab: Vec<String> = serde_json::from_str(&vocab_file)?;

        Ok(Self {
            session,
            vocab,
            imgsz: 640,
            stride: 32,
        })
    }

    #[must_use]
    pub fn preprocess(&self, img: &DynamicImage) -> (Array4<f32>, PreprocessMeta) {
        let (w0, h0) = img.dimensions();
        let r = self.imgsz as f32 / (w0.max(h0) as f32);
        let new_unpad_w = (w0 as f32 * r).round() as u32;
        let new_unpad_h = (h0 as f32 * r).round() as u32;

        let w_pad = ((new_unpad_w as f32 / self.stride as f32).ceil() * self.stride as f32) as u32;
        let h_pad = ((new_unpad_h as f32 / self.stride as f32).ceil() * self.stride as f32) as u32;

        let left = (w_pad - new_unpad_w) / 2;
        let top = (h_pad - new_unpad_h) / 2;

        // Initialize tensor with gray padding (114 / 255)
        let mut input = Array4::from_elem((1, 3, h_pad as usize, w_pad as usize), 114.0 / 255.0);

        let img_rgb = img.to_rgb8();
        let (src_w, src_h) = img_rgb.dimensions();
        let src_raw = img_rgb.as_raw();

        let scale_x = src_w as f32 / new_unpad_w as f32;
        let scale_y = src_h as f32 / new_unpad_h as f32;

        // Get a mutable view of the "content area" inside the padded tensor.
        // Shape: (1, channels, new_unpad_h, new_unpad_w)
        let mut content_view = input.slice_mut(s![
            0,
            ..,
            top as usize..(top + new_unpad_h) as usize,
            left as usize..(left + new_unpad_w) as usize
        ]);

        // Parallelize over rows (Axis 1 is Height in our (3, H, W) subview)
        content_view
            .axis_iter_mut(Axis(1))
            .enumerate()
            .par_bridge()
            .for_each(|(y, mut row_channels)| {
                let source_y = (y as f32 + 0.5).mul_add(scale_y, -0.5);
                let y1 = source_y.floor() as i32;
                let dy = source_y - y1 as f32;
                let y1_u = y1.clamp(0, src_h as i32 - 1) as u32;
                let y2_u = (y1 + 1).clamp(0, src_h as i32 - 1) as u32;
                let inv_delta_y = 1.0 - dy;

                for x in 0..new_unpad_w {
                    let source_x = (x as f32 + 0.5).mul_add(scale_x, -0.5);
                    let x1 = source_x.floor() as i32;
                    let dx = source_x - x1 as f32;
                    let x1_u = x1.clamp(0, src_w as i32 - 1) as u32;
                    let x2_u = (x1 + 1).clamp(0, src_w as i32 - 1) as u32;
                    let inv_delta_x = 1.0 - dx;

                    let get_pix = |px: u32, py: u32| {
                        let idx = (py * src_w + px) as usize * 3;
                        [src_raw[idx], src_raw[idx + 1], src_raw[idx + 2]]
                    };

                    let p11 = get_pix(x1_u, y1_u);
                    let p21 = get_pix(x2_u, y1_u);
                    let p12 = get_pix(x1_u, y2_u);
                    let p22 = get_pix(x2_u, y2_u);

                    for c in 0..3 {
                        let val = (f32::from(p22[c]) * dx).mul_add(
                            dy,
                            (f32::from(p12[c]) * inv_delta_x).mul_add(
                                dy,
                                (f32::from(p11[c]) * inv_delta_x)
                                    .mul_add(inv_delta_y, f32::from(p21[c]) * dx * inv_delta_y),
                            ),
                        );
                        // Safe assignment to the specific channel/x location in this row
                        row_channels[[c, x as usize]] = (val + 0.5).floor() / 255.0;
                    }
                }
            });

        (
            input,
            PreprocessMeta {
                ratio: r,
                pad: (left as f32, top as f32),
                orig_shape: (w0, h0),
                tensor_shape: (w_pad, h_pad),
            },
        )
    }

    #[must_use]
    pub fn process_mask(
        protos: &ndarray::ArrayView3<f32>,
        weights: &Array1<f32>,
        meta: &PreprocessMeta,
        bbox: &[f32; 4],
    ) -> Mask {
        let (mask_c, mask_h, mask_w) = protos.dim();
        let protos_flat = protos
            .view()
            .into_shape_with_order((mask_c, mask_h * mask_w))
            .unwrap();
        let mask_logits_flat = weights.dot(&protos_flat);
        let mask_logits = mask_logits_flat.to_shape((mask_h, mask_w)).unwrap();

        let [x1, y1, x2, y2] = *bbox;
        let (img_w, img_h) = meta.orig_shape;

        let x_map_factor = meta.ratio * (mask_w as f32 / meta.tensor_shape.0 as f32);
        let y_map_factor = meta.ratio * (mask_h as f32 / meta.tensor_shape.1 as f32);
        let x_offset = meta.pad.0 * (mask_w as f32 / meta.tensor_shape.0 as f32);
        let y_offset = meta.pad.1 * (mask_h as f32 / meta.tensor_shape.1 as f32);

        let ix1 = (x1.floor() as u32).clamp(0, img_w);
        let iy1 = (y1.floor() as u32).clamp(0, img_h);
        let ix2 = (x2.ceil() as u32).clamp(0, img_w);
        let iy2 = (y2.ceil() as u32).clamp(0, img_h);

        let x_coords: Vec<_> = (ix1..ix2)
            .map(|x| {
                let mx = (x as f32).mul_add(x_map_factor, x_offset);
                let mx_f = mx.floor() as usize;
                let mx_c = (mx_f + 1).min(mask_w - 1);
                let dx = mx - mx_f as f32;
                (mx_f, mx_c, dx)
            })
            .collect();

        let mut data = vec![0u8; (img_w as usize * img_h as usize).div_ceil(8)];

        for y in iy1..iy2 {
            let my = (y as f32).mul_add(y_map_factor, y_offset);
            if my < 0.0 || my >= (mask_h as f32 - 1.0) {
                continue;
            }

            let my_f = my.floor() as usize;
            let my_c = (my_f + 1).min(mask_h - 1);
            let dy = my - my_f as f32;
            let inv_dy = 1.0 - dy;

            let row_base = (y * img_w) as usize;

            for (i, &(mx_f, mx_c, dx)) in x_coords.iter().enumerate() {
                let x = ix1 as usize + i;
                let inv_dx = 1.0 - dx;

                let val = inv_dx.mul_add(
                    inv_dy.mul_add(mask_logits[[my_f, mx_f]], dy * mask_logits[[my_c, mx_f]]),
                    dx * inv_dy.mul_add(mask_logits[[my_f, mx_c]], dy * mask_logits[[my_c, mx_c]]),
                );

                if val > 0.0 {
                    let bit_idx = row_base + x;
                    data[bit_idx >> 3] |= 1 << (bit_idx & 7);
                }
            }
        }

        Mask {
            width: img_w,
            height: img_h,
            data,
        }
    }

    pub fn predict(&mut self, img: &DynamicImage, conf: f32, iou: f32) -> Result<Vec<Detection>> {
        let (input_tensor, meta) = self.preprocess(img);

        let (preds, protos) = {
            let outputs = self
                .session
                .run(ort::inputs!["images" => Value::from_array(input_tensor)?])?;
            let preds = outputs["detections"].try_extract_array::<f32>()?.to_owned();
            let protos = outputs["protos"].try_extract_array::<f32>()?.to_owned();
            (preds, protos)
        };

        let preds_view = preds.slice(s![0, .., ..]);
        let protos_view = protos.slice(s![0, .., .., ..]);

        let mut candidates = Vec::new();
        for i in 0..preds_view.shape()[0] {
            let score = preds_view[[i, 4]];
            if score > conf {
                let bbox = [
                    preds_view[[i, 0]],
                    preds_view[[i, 1]],
                    preds_view[[i, 2]],
                    preds_view[[i, 3]],
                ];
                let class_id = preds_view[[i, 5]] as usize;
                let mask_weights = preds_view.slice(s![i, 6..38]).to_owned();
                candidates.push((bbox, score, class_id, mask_weights));
            }
        }

        let kept_indices = non_maximum_suppression(&candidates, iou);

        let results: Vec<Detection> = kept_indices
            .into_par_iter()
            .map(|idx| {
                let (bbox, score, class_id, weights) = &candidates[idx];

                let x1 = ((bbox[0] - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32);
                let y1 = ((bbox[1] - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32);
                let x2 = ((bbox[2] - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32);
                let y2 = ((bbox[3] - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32);

                let final_bbox = [x1, y1, x2, y2];
                let mask = Self::process_mask(&protos_view, weights, &meta, &final_bbox);

                Detection {
                    bbox: final_bbox,
                    score: *score,
                    class_id: *class_id,
                    tag: self
                        .vocab
                        .get(*class_id)
                        .cloned()
                        .unwrap_or_else(|| "unknown".into()),
                    mask: Some(mask),
                }
            })
            .collect();

        Ok(results)
    }
}
