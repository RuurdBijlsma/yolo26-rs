use color_eyre::eyre::Result;
use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::s;
use object_detector::predictor::nms::non_maximum_suppression;
use object_detector::{ObjectBBox, ObjectDetector};
use ort::value::Value;
use std::hint::black_box;

fn benchmark_predict_components(c: &mut Criterion) -> Result<()> {
    let model_path = "assets/model/yoloe-26l-seg-pf.onnx";
    let vocab_path = "assets/model/vocabulary.json";
    let img_path = "assets/img/fridge.jpg";

    let mut predictor = ObjectDetector::builder(model_path, vocab_path).build()?;
    let img = image::open(img_path).expect("Failed to open image");

    c.bench_function("preprocess", |b| {
        b.iter(|| predictor.preprocess(black_box(&img)));
    });

    let (input_tensor, meta) = predictor.preprocess(&img);

    c.bench_function("inference", |b| {
        b.iter(|| {
            let outputs = predictor
                .session
                .run(ort::inputs!["images" => Value::from_array(input_tensor.clone()).unwrap()])
                .unwrap();
            let preds = outputs["detections"].try_extract_array::<f32>().unwrap();
            let protos = outputs["protos"].try_extract_array::<f32>().unwrap();
            black_box((preds, protos));
        });
    });

    // Extract data for downstream component benchmarks
    let (preds, protos) = {
        let outputs = predictor
            .session
            .run(ort::inputs!["images" => Value::from_array(input_tensor.clone()).unwrap()])?;
        let preds = outputs["detections"].try_extract_array::<f32>()?.to_owned();
        let protos = outputs["protos"].try_extract_array::<f32>()?.to_owned();
        (preds, protos)
    };

    let preds_view = preds.slice(s![0, .., ..]);
    let protos_view = protos.slice(s![0, .., .., ..]);

    c.bench_function("nms_and_filtering", |b| {
        b.iter(|| {
            let mut boxes = Vec::new();
            let mut scores = Vec::new();
            for i in 0..preds_view.shape()[0] {
                let score = preds_view[[i, 4]];
                if score > 0.25 {
                    boxes.push(ObjectBBox {
                        x1: preds_view[[i, 0]],
                        y1: preds_view[[i, 1]],
                        x2: preds_view[[i, 2]],
                        y2: preds_view[[i, 3]],
                    });
                    scores.push(score);
                }
            }
            black_box(non_maximum_suppression(&boxes, &scores, 0.45));
        });
    });

    // Prepare data for single mask bench
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    let mut weights_vec = Vec::new();
    for i in 0..preds_view.shape()[0] {
        let score = preds_view[[i, 4]];
        if score > 0.25 {
            boxes.push(ObjectBBox {
                x1: preds_view[[i, 0]],
                y1: preds_view[[i, 1]],
                x2: preds_view[[i, 2]],
                y2: preds_view[[i, 3]],
            });
            scores.push(score);
            weights_vec.push(preds_view.slice(s![i, 6..38]).to_owned());
        }
    }
    let kept = non_maximum_suppression(&boxes, &scores, 0.45);

    if let Some(&idx) = kept.first() {
        let sample_bbox = boxes[idx];
        let weights = &weights_vec[idx];

        c.bench_function("process_mask_single", |b| {
            b.iter(|| {
                black_box(ObjectDetector::process_mask(
                    black_box(&protos_view),
                    black_box(weights),
                    black_box(&meta),
                    black_box(&sample_bbox),
                ));
            });
        });
    }

    c.bench_function("predict_full", |b| {
        b.iter(|| {
            predictor
                .predict(black_box(&img))
                .call()
                .expect("Predict failed");
        });
    });

    Ok(())
}

fn benchmark_wrapper(c: &mut Criterion) {
    benchmark_predict_components(c).unwrap();
}

criterion_group!(benches, benchmark_wrapper);
criterion_main!(benches);
