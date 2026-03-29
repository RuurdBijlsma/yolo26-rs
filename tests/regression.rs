#[cfg(test)]
mod tests {
    use color_eyre::eyre::Result;
    use object_detector::YOLO26Predictor;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::fs;

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct ExpectedDetection {
        tag: String,
        score: f32,
        bbox: [f32; 4],
        mask_stats: Option<MaskStats>,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct MaskStats {
        width: usize,
        height: usize,
        active_pixels: usize,
    }

    const FLOAT_EPSILON: f32 = 1e-4;

    #[test]
    fn test_model_consistency() -> Result<()> {
        let mut predictor = YOLO26Predictor::new(
            "assets/model/yoloe-26l-seg-pf.onnx",
            "assets/model/vocabulary.json",
        )?;

        let golden_path = "assets/expected_outputs.json";
        let data = fs::read_to_string(golden_path)?;

        let expected_map: BTreeMap<String, Vec<ExpectedDetection>> =
            serde_json::from_str(&data).expect("Failed to parse golden JSON");

        for (img_name, expected_dets) in expected_map {
            let img_path = format!("assets/img/{img_name}");
            println!("Testing {img_name}...");

            let actual_dets = predictor
                .predict(&img_path, 0.4, 0.7)
                .unwrap_or_else(|_| panic!("Prediction failed for {img_name}"));

            // 1. Check count
            assert_eq!(
                actual_dets.len(),
                expected_dets.len(),
                "Detection count mismatch for {}. Expected {}, got {}",
                img_name,
                expected_dets.len(),
                actual_dets.len()
            );

            for (i, (actual, expected)) in actual_dets.iter().zip(expected_dets.iter()).enumerate()
            {
                // 2. Check Tag
                assert_eq!(
                    actual.tag, expected.tag,
                    "Tag mismatch at index {i} in {img_name}"
                );

                // 3. Check Score (Floating point)
                assert!(
                    (actual.score - expected.score).abs() < FLOAT_EPSILON,
                    "Score mismatch in {}: index {}. Expected {}, got {}",
                    img_name,
                    i,
                    expected.score,
                    actual.score
                );

                // 4. Check Bounding Box (All 4 coordinates)
                for j in 0..4 {
                    assert!(
                        (actual.bbox[j] - expected.bbox[j]).abs() < FLOAT_EPSILON,
                        "BBox coordinate [{}] mismatch in {}: index {}. Expected {}, got {}",
                        j,
                        img_name,
                        i,
                        expected.bbox[j],
                        actual.bbox[j]
                    );
                }

                // 5. Check Mask Stats
                match (&actual.mask, &expected.mask_stats) {
                    (Some(actual_mask), Some(expected_mask)) => {
                        let actual_active = actual_mask.iter().filter(|&&p| p).count();

                        assert_eq!(
                            actual_mask.ncols(),
                            expected_mask.width,
                            "Mask width mismatch in {img_name}",
                        );
                        assert_eq!(
                            actual_mask.nrows(),
                            expected_mask.height,
                            "Mask height mismatch in {img_name}",
                        );
                        assert_eq!(
                            actual_active, expected_mask.active_pixels,
                            "Mask active pixel count mismatch in {img_name}",
                        );
                    }
                    (None, None) => {} // Both missing is okay
                    _ => panic!("Mask presence mismatch in {img_name}: index {i}"),
                }
            }
            println!("✅ All good for {img_name}");
        }

        Ok(())
    }
}
