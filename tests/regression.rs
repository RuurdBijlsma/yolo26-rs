#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]

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
        width: u32,
        height: u32,
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

        let mut errors = Vec::new();
        let mut passed_count = 0;

        for (img_name, expected_dets) in &expected_map {
            let img_path = format!("assets/img/{img_name}");

            let img = match image::open(&img_path) {
                Ok(i) => i,
                Err(e) => {
                    errors.push(format!("[{img_name}] Failed to load image: {e}"));
                    continue;
                }
            };

            let actual_dets = match predictor.predict(&img, 0.4, 0.7) {
                Ok(d) => d,
                Err(e) => {
                    errors.push(format!("[{img_name}] Prediction failed: {e}"));
                    continue;
                }
            };

            // 1. Check count
            if actual_dets.len() != expected_dets.len() {
                errors.push(format!(
                    "[{img_name}] Detection count mismatch. Expected {}, got {}",
                    expected_dets.len(),
                    actual_dets.len()
                ));
                // Continue to check what we can, but zip will limit us to the shorter one
            }

            for (i, (actual, expected)) in actual_dets.iter().zip(expected_dets.iter()).enumerate()
            {
                let det_id = format!("{img_name} index {i} ({})", expected.tag);

                // 2. Check Tag
                if actual.tag != expected.tag {
                    errors.push(format!(
                        "[{det_id}] Tag mismatch. Expected '{}', got '{}'",
                        expected.tag, actual.tag
                    ));
                }

                // 3. Check Score
                if (actual.score - expected.score).abs() > FLOAT_EPSILON {
                    errors.push(format!(
                        "[{det_id}] Score mismatch. Expected {:.4}, got {:.4}",
                        expected.score, actual.score
                    ));
                }

                // 4. Check Bounding Box
                for j in 0..4 {
                    if (actual.bbox[j] - expected.bbox[j]).abs() > FLOAT_EPSILON {
                        errors.push(format!(
                            "[{det_id}] BBox coord [{j}] mismatch. Expected {:.2}, got {:.2}",
                            expected.bbox[j], actual.bbox[j]
                        ));
                    }
                }

                // 5. Check Mask Stats
                match (&actual.mask, &expected.mask_stats) {
                    (Some(actual_mask), Some(expected_mask)) => {
                        let actual_active: usize = actual_mask
                            .data
                            .iter()
                            .map(|&b| b.count_ones() as usize)
                            .sum();

                        if actual_mask.width != expected_mask.width {
                            errors.push(format!(
                                "[{det_id}] Mask width mismatch. Expected {}, got {}",
                                expected_mask.width, actual_mask.width
                            ));
                        }
                        if actual_mask.height != expected_mask.height {
                            errors.push(format!(
                                "[{det_id}] Mask height mismatch. Expected {}, got {}",
                                expected_mask.height, actual_mask.height
                            ));
                        }
                        if actual_active != expected_mask.active_pixels {
                            let difference =
                                (expected_mask.active_pixels as f64 - actual_active as f64).abs();
                            let percentage_difference = difference / actual_active as f64;
                            errors.push(format!(
                                "[{det_id}] Mask active pixels mismatch. Expected {}, got {} - {:.2}%",
                                expected_mask.active_pixels, actual_active,percentage_difference * 100.
                            ));
                        }
                    }
                    (None, None) => {}
                    (Some(_), None) => {
                        errors.push(format!("[{det_id}] Got a mask but none was expected"));
                    }
                    (None, Some(_)) => {
                        errors.push(format!("[{det_id}] Expected a mask but got none"));
                    }
                }
            }

            passed_count += 1;
        }

        // --- Final Reporting ---
        println!("\n--- Regression Test Report ---");
        println!("Images processed: {}/{}", passed_count, expected_map.len());

        if errors.is_empty() {
            println!("✅ All tests passed successfully!");
            Ok(())
        } else {
            println!(
                "❌ Found {} errors during regression testing:",
                errors.len()
            );
            for err in &errors {
                println!("  - {err}");
            }
            println!("------------------------------\n");
            panic!("Regression test failed with {} errors.", errors.len());
        }
    }
}
