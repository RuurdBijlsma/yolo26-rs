use crate::predictor::custom_resize::naive_bilinear_opencv;
use crate::predictor::nms::non_maximum_suppression;
use color_eyre::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, imageops::FilterType};
use ndarray::{Array1, Array2, Array4, Axis, s};
use ort::session::Session;
use ort::value::Value;
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

        let resized = naive_bilinear_opencv(&img.to_rgb8(), new_unpad_w, new_unpad_h);

        let mut canvas = ImageBuffer::from_pixel(w_pad, h_pad, Rgb([114, 114, 114]));
        let left = (w_pad - new_unpad_w) / 2;
        let top = (h_pad - new_unpad_h) / 2;
        image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);

        let mut input = Array4::zeros((1, 3, h_pad as usize, w_pad as usize));
        for (x, y, rgb) in canvas.enumerate_pixels() {
            input[[0, 0, y as usize, x as usize]] = (rgb[0] as f32) / 255.0;
            input[[0, 1, y as usize, x as usize]] = (rgb[1] as f32) / 255.0;
            input[[0, 2, y as usize, x as usize]] = (rgb[2] as f32) / 255.0;
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

    #[must_use]
    pub fn process_mask(
        protos: &ndarray::ArrayView3<f32>,
        weights: &Array1<f32>,
        meta: &PreprocessMeta,
    ) -> Array2<bool> {
        let (c, h, w) = protos.dim();
        let mut mask_flat = Array2::<f32>::zeros((h, w));
        for i in 0..c {
            mask_flat += &(protos.index_axis(Axis(0), i).to_owned() * weights[i]);
        }

        let mask_sigmoid = mask_flat.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let mask_img = ImageBuffer::from_fn(w as u32, h as u32, |x, y| {
            image::Luma([(mask_sigmoid[[y as usize, x as usize]] * 255.0) as u8])
        });

        let upscaled = DynamicImage::ImageLuma8(mask_img).resize_exact(
            meta.tensor_shape.0,
            meta.tensor_shape.1,
            FilterType::Triangle,
        );

        let crop_w = (meta.orig_shape.0 as f32 * meta.ratio).round() as u32;
        let crop_h = (meta.orig_shape.1 as f32 * meta.ratio).round() as u32;
        let final_mask = upscaled
            .crop_imm(meta.pad.0 as u32, meta.pad.1 as u32, crop_w, crop_h)
            .resize_exact(meta.orig_shape.0, meta.orig_shape.1, FilterType::Triangle)
            .to_luma8();

        Array2::from_shape_fn(
            (meta.orig_shape.1 as usize, meta.orig_shape.0 as usize),
            |(y, x)| final_mask.get_pixel(x as u32, y as u32)[0] > 127,
        )
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

        let mut results = Vec::new();
        for idx in kept_indices {
            let (bbox, score, class_id, weights) = &candidates[idx];

            let x1 = (bbox[0] - meta.pad.0) / meta.ratio;
            let y1 = (bbox[1] - meta.pad.1) / meta.ratio;
            let x2 = (bbox[2] - meta.pad.0) / meta.ratio;
            let y2 = (bbox[3] - meta.pad.1) / meta.ratio;

            let mask = Self::process_mask(&protos_view, weights, &meta);

            results.push(Detection {
                bbox: [x1, y1, x2, y2],
                score: *score,
                class_id: *class_id,
                tag: self
                    .vocab
                    .get(*class_id)
                    .cloned()
                    .unwrap_or_else(|| "unknown".into()),
                mask: Some(mask),
            });
        }

        Ok(results)
    }
}
