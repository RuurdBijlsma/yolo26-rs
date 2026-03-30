use color_eyre::eyre::Result;
#[cfg(feature = "hf-hub")]
use hf_hub::api::tokio::Api;
use std::path::PathBuf;

pub struct HfModel {
    pub id: String,
    pub file: String,
}

impl HfModel {
    #[must_use]
    pub fn default_model() -> Self {
        Self {
            id: "RuteNL/yolo26-object-detection-ONNX".to_owned(),
            file: "yoloe-26l-seg-pf.onnx".to_owned(),
        }
    }

    #[must_use]
    pub fn default_data() -> Self {
        Self {
            id: "RuteNL/yolo26-object-detection-ONNX".to_owned(),
            file: "yoloe-26l-seg-pf.onnx.data".to_owned(),
        }
    }

    #[must_use]
    pub fn default_vocabulary() -> Self {
        Self {
            id: "RuteNL/yolo26-object-detection-ONNX".to_owned(),
            file: "vocabulary.json".to_owned(),
        }
    }
}

#[cfg(feature = "hf-hub")]
pub async fn get_hf_model(model: HfModel) -> Result<PathBuf> {
    let api = Api::new()?;
    let repo = api.model(model.id);

    Ok(repo.get(&model.file).await?)
}
