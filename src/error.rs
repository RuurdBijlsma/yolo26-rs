use thiserror::Error;

#[cfg(feature = "hf-hub")]
use hf_hub::api::tokio::ApiError;

#[derive(Error, Debug)]
pub enum ObjectDetectorError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image Error: {0}")]
    Image(#[from] image::ImageError),

    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("ONNX Runtime Error: {0}")]
    Ort(String),

    #[error("NdArray Error: {0}")]
    NdArray(#[from] ndarray::ShapeError),

    #[cfg(feature = "hf-hub")]
    #[error("Hugging Face Hub error: {0}")]
    HfHub(String),

    #[error("Model Consistency Error: {0}")]
    InvalidModel(String),
}

impl<T> From<ort::Error<T>> for ObjectDetectorError {
    fn from(err: ort::Error<T>) -> Self {
        Self::Ort(err.to_string())
    }
}

#[cfg(feature = "hf-hub")]
impl From<ApiError> for ObjectDetectorError {
    fn from(value: ApiError) -> Self {
        Self::HfHub(value.to_string())
    }
}