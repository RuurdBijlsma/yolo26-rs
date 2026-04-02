use crate::model_manager::{get_hf_model, HfModel};
use crate::predictor::nms::non_maximum_suppression;
use crate::predictor::processing::{
    finalize_detections, preprocess_image, Candidate, ObjectBBox, ObjectDetection,
    YoloEngine,
};
use crate::ObjectDetectorError;
use bon::bon;
use image::DynamicImage;
use ndarray::s;
use ort::ep::ExecutionProviderDispatch;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::{fs, path::Path};

#[derive(Debug)]
pub struct ObjectDetector {
    engine: YoloEngine,
    vocabulary: Vec<String>,
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
            engine: YoloEngine {
                session,
                image_size: 640,
                stride: 32,
            },
            vocabulary,
        })
    }

    #[builder]
    pub fn predict(
        &mut self,
        #[builder(start_fn)] img: &DynamicImage,
        #[builder(default = 0.4)] confidence_threshold: f32,
        #[builder(default = 0.7)] intersection_over_curve: f32,
    ) -> Result<Vec<ObjectDetection>, ObjectDetectorError> {
        let (input_tensor, meta) =
            preprocess_image(img, self.engine.image_size, self.engine.stride);

        let outputs = self
            .engine
            .session
            .run(ort::inputs!["images" => Value::from_array(input_tensor)?])?;

        let preds = outputs["detections"].try_extract_array::<f32>()?;
        let protos = outputs["protos"].try_extract_array::<f32>()?;

        let preds_view = preds.slice(s![0, .., ..]);
        let protos_view = protos.slice(s![0, .., .., ..]);

        // 1. Extract candidates that pass the confidence threshold
        let mut candidates = Vec::new();
        for i in 0..preds_view.shape()[0] {
            let score = preds_view[[i, 4]];
            if score > confidence_threshold {
                candidates.push(Candidate {
                    bbox: ObjectBBox {
                        x1: preds_view[[i, 0]],
                        y1: preds_view[[i, 1]],
                        x2: preds_view[[i, 2]],
                        y2: preds_view[[i, 3]],
                    },
                    score,
                    class_id: preds_view[[i, 5]] as usize,
                    mask_weights: preds_view.slice(s![i, 6..38]).to_owned(),
                });
            }
        }

        // 2. Run Non-Maximum Suppression
        let bboxes: Vec<_> = candidates.iter().map(|c| c.bbox).collect();
        let scores: Vec<_> = candidates.iter().map(|c| c.score).collect();
        let kept_indices = non_maximum_suppression(&bboxes, &scores, intersection_over_curve);

        let kept_candidates: Vec<Candidate> = kept_indices
            .into_iter()
            .map(|idx| candidates[idx].clone())
            .collect();

        // 3. Use unified finalization logic for coordinate scaling and mask generation
        Ok(finalize_detections(
            kept_candidates,
            &protos_view,
            &meta,
            &self.vocabulary,
        ))
    }
}