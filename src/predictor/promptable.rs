use crate::ObjectDetectorError;
use crate::predictor::nms::non_maximum_suppression;
use crate::predictor::processing::{
    Candidate, ObjectBBox, ObjectDetection, YoloEngine, finalize_detections, preprocess_image,
};
use bon::bon;
use image::DynamicImage;
use ndarray::{Array1, Axis, Ix2, s};
use open_clip_inference::TextEmbedder;
use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use std::path::Path;

#[derive(Debug)]
pub struct PromptableDetector {
    engine: YoloEngine,
    pub text_embedder: TextEmbedder,
}

#[bon]
impl PromptableDetector {
    #[builder]
    pub fn new(
        #[builder(start_fn)] model_path: impl AsRef<Path>,
        #[builder(start_fn)] text_embedder: TextEmbedder,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, ObjectDetectorError> {
        let session = Session::builder()?
            .with_execution_providers(with_execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_file(model_path)?;

        Ok(Self {
            engine: YoloEngine {
                session,
                image_size: 640,
                stride: 32,
            },
            text_embedder,
        })
    }

    #[builder]
    pub fn predict(
        &mut self,
        #[builder(start_fn)] img: &DynamicImage,
        #[builder(start_fn)] labels: &[&str],
        #[builder(default = 0.15)] confidence_threshold: f32,
        #[builder(default = 0.7)] intersection_over_union: f32,
    ) -> Result<Vec<ObjectDetection>, ObjectDetectorError> {
        // 1. Generate Text Embeddings
        let text_embs = self
            .text_embedder
            .embed_texts(labels)
            .map_err(|e| ObjectDetectorError::Ort(format!("CLIP error: {e}")))?;
        let text_tensor = text_embs.insert_axis(Axis(0)); // [1, N, 512]

        // 2. Preprocess Image
        let (img_tensor, meta) = preprocess_image(img, self.engine.image_size, self.engine.stride);

        // 3. Inference
        let outputs = self.engine.session.run(ort::inputs![
            "images" => Value::from_array(img_tensor)?,
            "text_embeddings" => Value::from_array(text_tensor)?
        ])?;

        let raw_output = outputs["output0"].try_extract_array::<f32>()?;
        let protos = outputs
            .get("protos")
            .map(|p| p.try_extract_array::<f32>())
            .transpose()?;

        // Transpose output: [1, features, 8400] -> [8400, features]
        let preds_2d = raw_output
            .slice(s![0, .., ..])
            .into_dimensionality::<Ix2>()?
            .reversed_axes();

        let num_classes = labels.len();

        // Check if model has enough columns for mask weights (4 box + num_classes + 32 weights)
        let has_masks = protos.is_some() && preds_2d.shape()[1] >= 4 + num_classes + 32;

        let mut candidates = Vec::new();

        // 4. Extract candidates
        for i in 0..preds_2d.shape()[0] {
            let row = preds_2d.row(i);
            let scores = row.slice(s![4..4 + num_classes]);

            let mut max_score = 0.0f32;
            let mut max_cls_id = 0;
            for (idx, &s) in scores.iter().enumerate() {
                if s > max_score {
                    max_score = s;
                    max_cls_id = idx;
                }
            }

            if max_score > confidence_threshold {
                let mask_weights = if has_masks {
                    row.slice(s![4 + num_classes..4 + num_classes + 32])
                        .to_owned()
                } else {
                    Array1::default(0)
                };

                candidates.push(Candidate {
                    bbox: ObjectBBox {
                        x1: row[0] - row[2] / 2.0,
                        y1: row[1] - row[3] / 2.0,
                        x2: row[0] + row[2] / 2.0,
                        y2: row[1] + row[3] / 2.0,
                    },
                    score: max_score,
                    class_id: max_cls_id,
                    mask_weights,
                });
            }
        }

        // 5. NMS
        let bboxes: Vec<_> = candidates.iter().map(|c| c.bbox).collect();
        let scores: Vec<_> = candidates.iter().map(|c| c.score).collect();
        let kept_indices = non_maximum_suppression(&bboxes, &scores, intersection_over_union);

        let kept_candidates: Vec<Candidate> = kept_indices
            .into_iter()
            .map(|idx| candidates[idx].clone())
            .collect();

        // Prepare protos view if it exists
        let protos_view = protos.as_ref().map(|p| p.slice(s![0, .., .., ..]));

        // Convert slice labels to String for the shared finalizer
        let label_strings: Vec<String> = labels.iter().map(|s| s.to_string()).collect();

        // 6. Use unified finalization logic (passing protos as Option)
        Ok(finalize_detections(
            kept_candidates,
            protos_view.as_ref(),
            &meta,
            &label_strings,
        ))
    }
}
