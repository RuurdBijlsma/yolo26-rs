use super::yolo_predictor::ObjectBBox;

#[must_use]
pub fn non_maximum_suppression(
    boxes: &[ObjectBBox],
    scores: &[f32],
    iou_threshold: f32,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..boxes.len()).collect();

    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut kept = Vec::new();
    let mut indices_vec = indices;

    while !indices_vec.is_empty() {
        let current = indices_vec.remove(0);
        kept.push(current);

        let current_box = &boxes[current];
        indices_vec.retain(|&idx| calculate_iou(current_box, &boxes[idx]) <= iou_threshold);
    }
    kept
}

#[must_use]
pub fn calculate_iou(b1: &ObjectBBox, b2: &ObjectBBox) -> f32 {
    let x1 = b1.x1.max(b2.x1);
    let y1 = b1.y1.max(b2.y1);
    let x2 = b1.x2.min(b2.x2);
    let y2 = b1.y2.min(b2.y2);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
    let area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);

    inter / (area1 + area2 - inter)
}
