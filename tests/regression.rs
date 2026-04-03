#![allow(clippy::cast_precision_loss)]

#[cfg(test)]
mod tests {
    use color_eyre::eyre::{Result, eyre};
    use object_detector::{ObjectBBox, ObjectDetector};
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::fs;

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct ExpectedDetection {
        tag: String,
        score: f32,
        bbox: ObjectBBox,
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
        let mut predictor = ObjectDetector::builder(
            "assets/model/prompt_free/yoloe-26l-seg-pf.onnx",
            "assets/model/prompt_free/vocabulary_4585.json",
        )
        .build()?;

        let data = fs::read_to_string("assets/expected_outputs.json")?;
        let expected_map: BTreeMap<String, Vec<ExpectedDetection>> = serde_json::from_str(&data)?;

        let mut all_errors = Vec::new();

        for (img_name, expected_dets) in &expected_map {
            let img_path = format!("assets/img/{img_name}");
            let img =
                image::open(&img_path).map_err(|e| eyre!("Failed to load {img_name}: {e}"))?;

            let actual_dets = predictor.predict(&img).call()?;

            if actual_dets.len() != expected_dets.len() {
                all_errors.push(format!(
                    "[{img_name}] Count mismatch: expected {}, got {}",
                    expected_dets.len(),
                    actual_dets.len()
                ));
            }

            for (i, (actual, expected)) in actual_dets.iter().zip(expected_dets.iter()).enumerate()
            {
                let det_id = format!("{img_name}#{} ({})", i, expected.tag);

                // 1. Tag
                if actual.tag != expected.tag {
                    all_errors.push(format!(
                        "[{det_id}] Tag mismatch: {} != {}",
                        expected.tag, actual.tag
                    ));
                }

                // 2. Score
                if (actual.score - expected.score).abs() > FLOAT_EPSILON {
                    all_errors.push(format!(
                        "[{det_id}] Score mismatch: {} != {}",
                        expected.score, actual.score
                    ));
                }

                // 3. Bounding Box
                let (a, e) = (actual.bbox, expected.bbox);
                if (a.x1 - e.x1).abs() > FLOAT_EPSILON {
                    all_errors.push(format!(
                        "[{det_id}] BBox x1 mismatch: {:.2} != {:.2}",
                        e.x1, a.x1
                    ));
                }
                if (a.y1 - e.y1).abs() > FLOAT_EPSILON {
                    all_errors.push(format!(
                        "[{det_id}] BBox y1 mismatch: {:.2} != {:.2}",
                        e.y1, a.y1
                    ));
                }
                if (a.x2 - e.x2).abs() > FLOAT_EPSILON {
                    all_errors.push(format!(
                        "[{det_id}] BBox x2 mismatch: {:.2} != {:.2}",
                        e.x2, a.x2
                    ));
                }
                if (a.y2 - e.y2).abs() > FLOAT_EPSILON {
                    all_errors.push(format!(
                        "[{det_id}] BBox y2 mismatch: {:.2} != {:.2}",
                        e.y2, a.y2
                    ));
                }

                // 4. Mask
                check_mask(
                    &det_id,
                    actual.mask.as_ref(),
                    expected.mask_stats.as_ref(),
                    &mut all_errors,
                );
            }
        }

        report_results(&all_errors, expected_map.len())
    }

    fn check_mask(
        id: &str,
        actual: Option<&object_detector::ObjectMask>,
        expected: Option<&MaskStats>,
        errors: &mut Vec<String>,
    ) {
        match (actual, expected) {
            (Some(a), Some(e)) => {
                let active: usize = a.data.iter().map(|b| b.count_ones() as usize).sum();
                if a.width != e.width || a.height != e.height {
                    errors.push(format!(
                        "[{id}] Mask dimensions mismatch: {}x{} != {}x{}",
                        e.width, e.height, a.width, a.height
                    ));
                }
                if active != e.active_pixels {
                    let diff =
                        (active as f64 - e.active_pixels as f64).abs() / e.active_pixels as f64;
                    errors.push(format!(
                        "[{id}] Mask pixels mismatch: {} != {} ({:.2}% diff)",
                        e.active_pixels,
                        active,
                        diff * 100.0
                    ));
                }
            }
            (Some(_), None) => errors.push(format!("[{id}] Unexpected mask found")),
            (None, Some(_)) => errors.push(format!("[{id}] Expected mask missing")),
            (None, None) => {}
        }
    }

    fn report_results(errors: &[String], total_images: usize) -> Result<()> {
        if errors.is_empty() {
            println!("✅ Regression test passed for {total_images} images.");
            Ok(())
        } else {
            for err in errors {
                eprintln!("  - {err}");
            }
            Err(eyre!("Regression failed with {} errors", errors.len()))
        }
    }
}
