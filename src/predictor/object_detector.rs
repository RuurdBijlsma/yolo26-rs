use crate::ObjectDetectorError;
use crate::model_manager::{HfModel, get_hf_model};
use crate::predictor::nms::non_maximum_suppression;
use crate::predictor::preprocess_image;
use crate::predictor::processing::{ObjectBBox, ObjectDetection, ObjectMask, YoloPreprocessMeta};
use bon::bon;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array1, Array4, Axis, s};
use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use rayon::prelude::*;
use std::{fs, path::Path};

#[derive(Debug)]
pub struct ObjectDetector {
    pub session: Session,
    vocabulary: Vec<String>,
    image_size: u32,
    stride: u32,
}

#[bon]
impl ObjectDetector {
    /// Initialize predictor using models hosted on Hugging Face.
    #[cfg(feature = "hf-hub")]
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(default = HfModel::default_model())] model: HfModel,
        #[builder(default = HfModel::default_data())] data_model: HfModel,
        #[builder(default = HfModel::default_vocabulary())] vocab_model: HfModel,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, ObjectDetectorError> {
        let model_path = get_hf_model(model).await?;
        // Ensure data file is downloaded (ORT expects it in the same directory)
        get_hf_model(data_model).await?;
        let vocab_path = get_hf_model(vocab_model).await?;

        Self::builder(model_path, vocab_path)
            .with_execution_providers(with_execution_providers)
            .build()
    }

    /// Initialize predictor from local file paths.
    #[builder]
    pub fn new(
        #[builder(start_fn)] model_path: impl AsRef<Path>,
        #[builder(start_fn)] vocab_path: impl AsRef<Path>,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, ObjectDetectorError> {
        let session = Session::builder()?
            .with_execution_providers(with_execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_file(model_path)?;

        let vocabulary: Vec<String> = serde_json::from_str(&fs::read_to_string(vocab_path)?)?;

        Ok(Self {
            session,
            vocabulary,
            image_size: 640,
            stride: 32,
        })
    }
    
    #[builder]
    pub fn predict(
        &mut self,
        #[builder(start_fn)] img: &DynamicImage,
        #[builder(default = 0.4)] confidence_threshold: f32,
        #[builder(default = 0.7)] intersection_over_curve: f32,
    ) -> Result<Vec<ObjectDetection>, ObjectDetectorError> {
        let (input_tensor, meta) = preprocess_image(img, self.image_size, self.stride);
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
            if score > confidence_threshold {
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

        let kept =
            non_maximum_suppression(&candidate_boxes, &candidate_scores, intersection_over_curve);

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
