use color_eyre::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, imageops::FilterType};
use ndarray::{Array1, Array2, Array4, Axis, s};
use ort::session::Session;
use ort::value::Value;
use serde_json;
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

pub struct YOLO26Predictor {
    session: Session,
    vocab: Vec<String>,
    imgsz: u32,
    stride: u32,
}

impl YOLO26Predictor {
    pub fn new(model_path: impl AsRef<Path>, vocab_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .expect("with_optimization_level")
            .with_intra_threads(num_cpus::get())
            .expect("with_intra_threads")
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

    /// Replicates the Python standalone letterbox logic exactly
    fn preprocess(&self, img: &DynamicImage, debug_name: Option<&str>) -> Result<(Array4<f32>, PreprocessMeta)> {
        let (w0, h0) = img.dimensions();

        // 1. Calculate scaling ratio (long side = 640)
        let r = self.imgsz as f32 / (w0.max(h0) as f32);

        // 2. Calculate new unpadded dimensions (matches Python's int(round(w0 * r)))
        let new_unpad_w = (w0 as f32 * r).round() as u32;
        let new_unpad_h = (h0 as f32 * r).round() as u32;

        // 3. Calculate rectangular padding (nearest multiple of stride)
        let w_pad = ((new_unpad_w as f32 / self.stride as f32).ceil() * self.stride as f32) as u32;
        let h_pad = ((new_unpad_h as f32 / self.stride as f32).ceil() * self.stride as f32) as u32;

        let resized = img.resize_exact(new_unpad_w, new_unpad_h, FilterType::CatmullRom);

        // 5. Create Gray Canvas (114)
        let mut canvas = ImageBuffer::from_pixel(w_pad, h_pad, Rgb([114, 114, 114]));

        // Python: top, left = dh // 2, dw // 2
        let dw = w_pad - new_unpad_w;
        let dh = h_pad - new_unpad_h;
        let left = dw / 2;
        let top = dh / 2;

        image::imageops::overlay(&mut canvas, &resized.to_rgb8(), left as i64, top as i64);

        // 6. Normalize and Convert to CHW
        // Important: Python's BGR-to-RGB conversion is handled by Rust's DynamicImage::open (which is RGB)
        let mut input = Array4::zeros((1, 3, h_pad as usize, w_pad as usize));
        for (x, y, rgb) in canvas.enumerate_pixels() {
            input[[0, 0, y as usize, x as usize]] = (rgb[0] as f32) / 255.0;
            input[[0, 1, y as usize, x as usize]] = (rgb[1] as f32) / 255.0;
            input[[0, 2, y as usize, x as usize]] = (rgb[2] as f32) / 255.0;
        }

        // --- DEBUG VISUALIZATION ---
        if let Some(name) = debug_name {
            let mut debug_img = ImageBuffer::new(w_pad, h_pad);
            for y in 0..h_pad {
                for x in 0..w_pad {
                    let r = (input[[0, 0, y as usize, x as usize]] * 255.0) as u8;
                    let g = (input[[0, 1, y as usize, x as usize]] * 255.0) as u8;
                    let b = (input[[0, 2, y as usize, x as usize]] * 255.0) as u8;
                    debug_img.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
            debug_img.save(format!("rust_debug_{}", name))?;
        }
        // ---------------------------

        let meta = PreprocessMeta {
            ratio: r,
            pad: (left as f32, top as f32),
            orig_shape: (w0, h0),
            tensor_shape: (w_pad, h_pad),
        };

        Ok((input, meta))
    }

    pub fn predict(&mut self, img_path: impl AsRef<Path>, conf_threshold: f32, iou_threshold: f32) -> Result<Vec<Detection>> {
        let file_name = img_path.as_ref().file_name().unwrap().to_str().unwrap();
        let img = image::open(&img_path)?;

        // Pass the filename to save debug image
        let (input_tensor, meta) = self.preprocess(&img, Some(file_name))?;

        let input_value = Value::from_array(input_tensor)?;
        let outputs = self.session.run(ort::inputs!["images" => input_value])?;

        let detections_owned = outputs["detections"].try_extract_array::<f32>()?.to_owned();
        let protos_owned = outputs["protos"].try_extract_array::<f32>()?.to_owned();
        drop(outputs);

        let preds = detections_owned.slice(s![0, .., ..]);
        let protos = protos_owned.slice(s![0, .., .., ..]);

        let mut candidates = Vec::new();
        for i in 0..preds.shape()[0] {
            let score = preds[[i, 4]];
            if score > conf_threshold {
                let class_id = preds[[i, 5]] as usize;
                let bbox = [preds[[i, 0]], preds[[i, 1]], preds[[i, 2]], preds[[i, 3]]];
                let mask_weights = preds.slice(s![i, 6..38]).to_owned();
                candidates.push((bbox, score, class_id, mask_weights));
            }
        }

        let kept_indices = Self::nms(&candidates, iou_threshold);

        let mut results = Vec::new();
        for idx in kept_indices {
            let (bbox, score, class_id, weights) = &candidates[idx];

            let x1 = (bbox[0] - meta.pad.0) / meta.ratio;
            let y1 = (bbox[1] - meta.pad.1) / meta.ratio;
            let x2 = (bbox[2] - meta.pad.0) / meta.ratio;
            let y2 = (bbox[3] - meta.pad.1) / meta.ratio;

            let mask = Self::process_mask(&protos, weights, &meta);

            results.push(Detection {
                bbox: [x1, y1, x2, y2],
                score: *score,
                class_id: *class_id,
                tag: self.vocab.get(*class_id).cloned().unwrap_or_else(|| "unknown".to_string()),
                mask: Some(mask),
            });
        }

        Ok(results)
    }

    // Helper changed to associated function (no &self)
    fn nms(
        candidates: &Vec<([f32; 4], f32, usize, Array1<f32>)>,
        iou_threshold: f32,
    ) -> Vec<usize> {
        if candidates.is_empty() {
            return vec![];
        }
        let mut indices: Vec<usize> = (0..candidates.len()).collect();
        indices.sort_by(|&a, &b| candidates[b].1.partial_cmp(&candidates[a].1).unwrap());

        let mut kept = Vec::new();
        while let Some(current) = indices.first().cloned() {
            kept.push(current);
            indices.remove(0);
            indices.retain(|&idx| {
                Self::calculate_iou(&candidates[current].0, &candidates[idx].0) <= iou_threshold
            });
        }
        kept
    }

    fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
        let x1 = box1[0].max(box2[0]);
        let y1 = box1[1].max(box2[1]);
        let x2 = box1[2].min(box2[2]);
        let y2 = box1[3].min(box2[3]);
        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        intersection / (area1 + area2 - intersection)
    }

    // Helper changed to associated function (no &self)
    fn process_mask(
        protos: &ndarray::ArrayView3<f32>,
        weights: &Array1<f32>,
        meta: &PreprocessMeta,
    ) -> Array2<bool> {
        let (c_proto, h_proto, w_proto) = protos.dim();
        let mut mask_flat = Array2::<f32>::zeros((h_proto, w_proto));
        for i in 0..c_proto {
            let w = weights[i];
            let p_layer = protos.index_axis(Axis(0), i);
            mask_flat += &(p_layer.to_owned() * w);
        }

        let mask_sigmoid = mask_flat.mapv(|x| 1.0 / (1.0 + (-x).exp()));

        let mask_img = ImageBuffer::from_fn(w_proto as u32, h_proto as u32, |x, y| {
            image::Luma([(mask_sigmoid[[y as usize, x as usize]] * 255.0) as u8])
        });

        let dyn_mask = DynamicImage::ImageLuma8(mask_img);
        let upscaled = dyn_mask.resize_exact(
            meta.tensor_shape.0,
            meta.tensor_shape.1,
            FilterType::Triangle,
        );

        let (w_ori, h_ori) = meta.orig_shape;
        let crop_w = (w_ori as f32 * meta.ratio).round() as u32;
        let crop_h = (h_ori as f32 * meta.ratio).round() as u32;
        let cropped = upscaled.crop_imm(meta.pad.0 as u32, meta.pad.1 as u32, crop_w, crop_h);

        let final_mask = cropped
            .resize_exact(w_ori, h_ori, FilterType::Triangle)
            .to_luma8();

        Array2::from_shape_fn((h_ori as usize, w_ori as usize), |(y, x)| {
            final_mask.get_pixel(x as u32, y as u32)[0] > 127
        })
    }
}

struct PreprocessMeta {
    ratio: f32,
    pad: (f32, f32),
    orig_shape: (u32, u32),
    tensor_shape: (u32, u32),
}

pub fn main() -> Result<()> {
    let model_path = "assets/model/dynamic-onnx/yoloe-26l-seg-pf-dynamic-try-3.onnx";
    let vocab_path = "assets/model/dynamic-onnx/vocabulary-dynamic.json";

    let mut predictor = YOLO26Predictor::new(model_path, vocab_path)?;

    println!("--- YOLO26 Rust Standalone Inference ---");

    let img_dir = Path::new("assets/img");
    for entry in fs::read_dir(img_dir)? {
        let path = entry?.path();
        if let Some(ext) = path.extension() {
            if ext == "jpg" || ext == "jpeg" || ext == "png" {
                let start = std::time::Instant::now();
                let results = predictor.predict(&path, 0.4, 0.7)?;
                let duration = start.elapsed();

                println!("Image: {:?} ({:?})", path.file_name().unwrap(), duration);
                println!("  - Objects Found: {}", results.len());

                let tags: std::collections::HashSet<_> = results.iter().map(|r| &r.tag).collect();
                if !tags.is_empty() {
                    println!("  - Tags: {:?}", tags);
                }
                println!("------------------------------");
            }
        }
    }

    Ok(())
}
