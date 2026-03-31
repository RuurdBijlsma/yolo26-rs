use crate::ObjectDetectorError;
#[cfg(feature = "hf-hub")]
use hf_hub::api::tokio::Api;
use std::path::PathBuf;

/// Details for fetching model files from `HuggingFace` Hub.
pub struct HfModel {
    /// Repository ID (e.g., "user/repo")
    pub id: String,
    /// Filename within the repository
    pub file: String,
}

impl HfModel {
    const DEFAULT_REPO_ID: &'static str = "RuteNL/yolo26-object-detection-ONNX";

    #[must_use]
    pub fn default_model() -> Self {
        Self {
            id: Self::DEFAULT_REPO_ID.to_owned(),
            file: "yoloe-26l-seg-pf.onnx".to_owned(),
        }
    }

    #[must_use]
    pub fn default_data() -> Self {
        Self {
            id: Self::DEFAULT_REPO_ID.to_owned(),
            file: "yoloe-26l-seg-pf.onnx.data".to_owned(),
        }
    }

    #[must_use]
    pub fn default_vocabulary() -> Self {
        Self {
            id: Self::DEFAULT_REPO_ID.to_owned(),
            file: "vocabulary.json".to_owned(),
        }
    }
}

/// Downloads a file from `HuggingFace` Hub using the provided configuration.
#[cfg(feature = "hf-hub")]
pub async fn get_hf_model(model: HfModel) -> Result<PathBuf, ObjectDetectorError> {
    let api = Api::new()?;
    let repo = api.model(model.id);
    Ok(repo.get(&model.file).await?)
}
