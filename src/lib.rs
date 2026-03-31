#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

pub mod model_manager;
pub mod predictor;
mod error;

pub use error::{ObjectDetectorError};
pub use predictor::{ObjectBBox, ObjectDetection, ObjectMask, YOLO26Predictor, YoloPreprocessMeta};
