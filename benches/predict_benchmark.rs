use std::hint::black_box;
use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::s;
use object_detector::predictor::nms::non_maximum_suppression;
use object_detector::predictor::yolo_predictor::YOLO26Predictor;
use ort::value::Value;

#[allow(clippy::too_many_lines)]
fn benchmark_predict_components(c: &mut Criterion) {
    let model_path = "assets/model/yoloe-26l-seg-pf.onnx";
    let vocab_path = "assets/model/vocabulary.json";
    let img_path = "assets/img/fridge.jpg";

    let mut predictor =
        YOLO26Predictor::new(model_path, vocab_path).expect("Failed to create predictor");
    let img = image::open(img_path).expect("Failed to open image");

    // 1. image::open
    c.bench_function("predict_step_1_image_open", |b| {
        b.iter(|| {
            let img = image::open(black_box(img_path)).unwrap();
            black_box(img);
        });
    });

    // 2. preprocess
    c.bench_function("predict_step_2_preprocess", |b| {
        b.iter(|| {
            black_box(predictor.preprocess(black_box(&img)));
        });
    });

    let (input_tensor, meta) = predictor.preprocess(&img);

    // 3. inference (run session + extract)
    c.bench_function("predict_step_3_inference", |b| {
        b.iter(|| {
            let outputs = predictor
                .session
                .run(ort::inputs!["images" => Value::from_array(input_tensor.clone()).unwrap()])
                .unwrap();
            let preds = outputs["detections"]
                .try_extract_array::<f32>()
                .unwrap()
                .to_owned();
            let protos = outputs["protos"]
                .try_extract_array::<f32>()
                .unwrap()
                .to_owned();
            black_box((preds, protos));
        });
    });

    let (preds, protos) = {
        let outputs = predictor
            .session
            .run(ort::inputs!["images" => Value::from_array(input_tensor.clone()).unwrap()])
            .unwrap();
        let preds = outputs["detections"]
            .try_extract_array::<f32>()
            .unwrap()
            .to_owned();
        let protos = outputs["protos"]
            .try_extract_array::<f32>()
            .unwrap()
            .to_owned();
        (preds, protos)
    };

    // 4. nms + candidate filtering
    c.bench_function("predict_step_4_nms_and_filtering", |b| {
        b.iter(|| {
            let preds_view = preds.slice(s![0, .., ..]);
            let mut candidates = Vec::new();
            for i in 0..preds_view.shape()[0] {
                let score = preds_view[[i, 4]];
                if score > 0.25 {
                    let bbox = [
                        preds_view[[i, 0]],
                        preds_view[[i, 1]],
                        preds_view[[i, 2]],
                        preds_view[[i, 3]],
                    ];
                    let class_id = preds_view[[i, 5]] as usize;
                    let mask_weights = preds_view.slice(s![i, 6..38]).to_owned();
                    candidates.push((bbox, score, class_id, mask_weights));
                }
            }
            let kept_indices = non_maximum_suppression(&candidates, 0.45);
            black_box((candidates, kept_indices));
        });
    });

    let preds_view = preds.slice(s![0, .., ..]);
    let protos_view = protos.slice(s![0, .., .., ..]);
    let mut candidates = Vec::new();
    for i in 0..preds_view.shape()[0] {
        let score = preds_view[[i, 4]];
        if score > 0.25 {
            let bbox = [
                preds_view[[i, 0]],
                preds_view[[i, 1]],
                preds_view[[i, 2]],
                preds_view[[i, 3]],
            ];
            let class_id = preds_view[[i, 5]] as usize;
            let mask_weights = preds_view.slice(s![i, 6..38]).to_owned();
            candidates.push((bbox, score, class_id, mask_weights));
        }
    }
    let kept_indices = non_maximum_suppression(&candidates, 0.45);

    // 5. process_mask (single call)
    if !kept_indices.is_empty() {
        let (bbox, _score, _class_id, weights) = &candidates[kept_indices[0]];

        // Prepare original coordinates for the benchmark
        let x1 = ((bbox[0] - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32);
        let y1 = ((bbox[1] - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32);
        let x2 = ((bbox[2] - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32);
        let y2 = ((bbox[3] - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32);
        let sample_final_bbox = [x1, y1, x2, y2];

        c.bench_function("predict_step_5_process_mask_single", |b| {
            b.iter(|| {
                black_box(YOLO26Predictor::process_mask(
                    black_box(&protos_view),
                    black_box(weights),
                    black_box(&meta),
                    black_box(&sample_final_bbox), // Added new syntax
                ));
            });
        });
    }

    // 7. Full predict function
    c.bench_function("predict_full", |b| {
        b.iter(|| {
            predictor.predict(black_box(img_path), 0.4, 0.7).unwrap();
        });
    });
}

criterion_group!(benches, benchmark_predict_components);
criterion_main!(benches);