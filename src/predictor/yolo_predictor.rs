use crate::model_manager::{HfModel, get_hf_model};
use crate::predictor::nms::non_maximum_suppression;
use color_eyre::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array1, Array2, Array4, Axis, s};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use rayon::prelude::*;
use std::{fs, path::Path};

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

#[derive(Debug)]
pub struct YOLO26Predictor {
    pub session: Session,
    vocabulary: Vec<String>,
    image_size: u32,
    stride: u32,
}

impl YOLO26Predictor {
    /// Initialize predictor using models hosted on `HuggingFace`.
    pub async fn from_hf() -> Result<Self> {
        let model_path = get_hf_model(HfModel::default_model()).await?;
        get_hf_model(HfModel::default_data()).await?;
        let vocab_path = get_hf_model(HfModel::default_vocabulary()).await?;
        Self::new(model_path, vocab_path)
    }

    pub fn new(model_path: impl AsRef<Path>, vocab_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(num_cpus::get())
            .unwrap()
            .commit_from_file(model_path)?;

        let vocabulary: Vec<String> = serde_json::from_str(&fs::read_to_string(vocab_path)?)?;

        Ok(Self {
            session,
            vocabulary,
            image_size: 640,
            stride: 32,
        })
    }

    #[must_use]
    pub fn preprocess(&self, img: &DynamicImage) -> (Array4<f32>, YoloPreprocessMeta) {
        let (w0, h0) = img.dimensions();
        let ratio = self.image_size as f32 / (w0.max(h0) as f32);
        let unpad_w = (w0 as f32 * ratio).round() as u32;
        let unpad_h = (h0 as f32 * ratio).round() as u32;

        let w_pad = ((unpad_w as f32 / self.stride as f32).ceil() * self.stride as f32) as u32;
        let h_pad = ((unpad_h as f32 / self.stride as f32).ceil() * self.stride as f32) as u32;

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
                        let get_p =
                            |px, py| f32::from(src_raw[((py * src_w + px) as usize * 3) + c]);
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

    #[must_use]
    pub fn process_mask(
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

        // Precompute X coordinates exactly like original to ensure bit-parity
        let x_coords: Vec<_> = (ix1..ix2)
            .map(|x| {
                let mx = (x as f32).mul_add(x_map_factor, x_offset);
                let mx_f = (mx.floor() as usize).min(mask_w - 1); // Safe clamp
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

    pub fn predict(
        &mut self,
        img: &DynamicImage,
        conf: f32,
        iou: f32,
    ) -> Result<Vec<ObjectDetection>> {
        let (input_tensor, meta) = self.preprocess(img);
        let outputs = self
            .session
            .run(ort::inputs!["images" => Value::from_array(input_tensor)?])?;

        let preds = outputs["detections"].try_extract_array::<f32>()?;
        let protos = outputs["protos"].try_extract_array::<f32>()?;
        let (preds_view, protos_view) =
            (preds.slice(s![0, .., ..]), protos.slice(s![0, .., .., ..]));

        let mut candidate_boxes = Vec::new();
        let mut candidate_scores = Vec::new();
        let mut candidate_data = Vec::new();

        for i in 0..preds_view.shape()[0] {
            let score = preds_view[[i, 4]];
            if score > conf {
                candidate_boxes.push(ObjectBBox {
                    x1: preds_view[[i, 0]],
                    y1: preds_view[[i, 1]],
                    x2: preds_view[[i, 2]],
                    y2: preds_view[[i, 3]],
                });
                candidate_scores.push(score);
                candidate_data.push((
                    preds_view[[i, 5]] as usize,
                    preds_view.slice(s![i, 6..38]).to_owned(),
                ));
            }
        }

        let kept = non_maximum_suppression(&candidate_boxes, &candidate_scores, iou);

        Ok(kept
            .into_par_iter()
            .map(|idx| {
                let (class_id, weights) = &candidate_data[idx];
                let raw_box = candidate_boxes[idx];

                let final_bbox = ObjectBBox {
                    x1: ((raw_box.x1 - meta.pad.0) / meta.ratio)
                        .clamp(0.0, meta.orig_shape.0 as f32),
                    y1: ((raw_box.y1 - meta.pad.1) / meta.ratio)
                        .clamp(0.0, meta.orig_shape.1 as f32),
                    x2: ((raw_box.x2 - meta.pad.0) / meta.ratio)
                        .clamp(0.0, meta.orig_shape.0 as f32),
                    y2: ((raw_box.y2 - meta.pad.1) / meta.ratio)
                        .clamp(0.0, meta.orig_shape.1 as f32),
                };

                ObjectDetection {
                    bbox: final_bbox,
                    score: candidate_scores[idx],
                    class_id: *class_id,
                    tag: self
                        .vocabulary
                        .get(*class_id)
                        .cloned()
                        .unwrap_or_else(|| "unknown".into()),
                    mask: Some(Self::process_mask(
                        &protos_view,
                        weights,
                        &meta,
                        &final_bbox,
                    )),
                }
            })
            .collect())
    }
}
