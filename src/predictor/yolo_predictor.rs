use crate::predictor::custom_resize::naive_bilinear_opencv;
use crate::predictor::nms::non_maximum_suppression;
use color_eyre::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::{s, Array1, Array2, Array4};
use ort::session::Session;
use ort::value::Value;
use rayon::prelude::*;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub class_id: usize,
    pub tag: String,
    pub mask: Option<Array2<bool>>,
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
    pub fn new(model_path: impl AsRef<Path>, vocab_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3).unwrap()
            .with_intra_threads(num_cpus::get()).unwrap()
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

        let resized = naive_bilinear_opencv(&img.to_rgb8(), new_unpad_w, new_unpad_h);

        let mut canvas = ImageBuffer::from_pixel(w_pad, h_pad, Rgb([114, 114, 114]));
        let left = (w_pad - new_unpad_w) / 2;
        let top = (h_pad - new_unpad_h) / 2;
        image::imageops::overlay(&mut canvas, &resized, i64::from(left), i64::from(top));

        let mut input = Array4::zeros((1, 3, h_pad as usize, w_pad as usize));
        let flat_raw = canvas.as_raw();
        let channel_size = (h_pad * w_pad) as usize;

        let data = input.as_slice_mut().unwrap();
        for (i, rgb) in flat_raw.chunks_exact(3).enumerate() {
            data[i] = f32::from(rgb[0]) / 255.0;
            data[i + channel_size] = f32::from(rgb[1]) / 255.0;
            data[i + 2 * channel_size] = f32::from(rgb[2]) / 255.0;
        }

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

    pub fn process_mask(
        protos: &ndarray::ArrayView3<f32>,
        weights: &Array1<f32>,
        meta: &PreprocessMeta,
        bbox: &[f32; 4],
    ) -> Array2<bool> {
        let (mask_c, mask_h, mask_w) = protos.dim();

        // 1. Compute the raw mask (logits) at the prototype resolution (usually 160x160)
        // This is a single matrix-vector multiplication.
        let protos_flat = protos.view().into_shape_with_order((mask_c, mask_h * mask_w)).unwrap();
        let mask_logits_flat = weights.dot(&protos_flat);
        let mask_logits = mask_logits_flat.to_shape((mask_h, mask_w)).unwrap();

        let [x1, y1, x2, y2] = *bbox;
        let img_w = meta.orig_shape.0 as usize;
        let img_h = meta.orig_shape.1 as usize;

        // 2. Pre-calculate coordinate transformation constants
        // We need to map (x_img, y_img) -> (x_tensor) -> (x_mask_proto)
        let gain = meta.ratio;
        let (pad_x, pad_y) = meta.pad;

        // Scaling factor from original image pixels to mask prototype pixels
        // mask_proto is usually 1/4 of the tensor size (e.g. 160 vs 640)
        let x_map_factor = gain * (mask_w as f32 / meta.tensor_shape.0 as f32);
        let y_map_factor = gain * (mask_h as f32 / meta.tensor_shape.1 as f32);
        let x_offset = pad_x * (mask_w as f32 / meta.tensor_shape.0 as f32);
        let y_offset = pad_y * (mask_h as f32 / meta.tensor_shape.1 as f32);

        // 3. Generate the boolean mask only for the bounding box area
        // We initialize with false and only fill the BBox rectangle.
        let mut final_mask = Array2::from_elem((img_h, img_w), false);

        let ix1 = (x1.floor() as usize).clamp(0, img_w);
        let iy1 = (y1.floor() as usize).clamp(0, img_h);
        let ix2 = (x2.ceil() as usize).clamp(0, img_w);
        let iy2 = (y2.ceil() as usize).clamp(0, img_h);

        for y in iy1..iy2 {
            // Map image y to mask prototype y
            let my = (y as f32).mul_add(y_map_factor, y_offset);
            if my < 0.0 || my >= (mask_h as f32 - 1.0) { continue; }

            let my_f = my.floor() as usize;
            let my_c = (my_f + 1).min(mask_h - 1);
            let dy = my - my_f as f32;

            for x in ix1..ix2 {
                let mx = (x as f32).mul_add(x_map_factor, x_offset);
                if mx < 0.0 || mx >= (mask_w as f32 - 1.0) { continue; }

                let mx_f = mx.floor() as usize;
                let mx_c = (mx_f + 1).min(mask_w - 1);
                let dx = mx - mx_f as f32;

                // Bilinear sampling of the logit
                let val = (1.0 - dx).mul_add(
                    (1.0 - dy).mul_add(mask_logits[[my_f, mx_f]], dy * mask_logits[[my_c, mx_f]]),
                    dx * (1.0 - dy).mul_add(mask_logits[[my_f, mx_c]], dy * mask_logits[[my_c, mx_c]])
                );

                // Sigmoid(val) > 0.5  is equivalent to  val > 0.0
                if val > 0.0 {
                    final_mask[[y, x]] = true;
                }
            }
        }

        final_mask
    }


    pub fn predict(
        &mut self,
        img_path: impl AsRef<Path>,
        conf: f32,
        iou: f32,
    ) -> Result<Vec<Detection>> {
        let img = image::open(img_path)?;
        let (input_tensor, meta) = self.preprocess(&img);

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