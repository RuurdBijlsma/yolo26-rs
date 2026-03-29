use ndarray::Array1;

pub fn non_maximum_suppression(
    candidates: &[([f32; 4], f32, usize, Array1<f32>)],
    iou_thresh: f32,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|&a, &b| candidates[b].1.partial_cmp(&candidates[a].1).unwrap());

    let mut kept = Vec::new();
    while !indices.is_empty() {
        let current = indices.remove(0);
        kept.push(current);
        indices.retain(|&idx| {
            calculate_intersection_over_union(&candidates[current].0, &candidates[idx].0)
                <= iou_thresh
        });
    }
    kept
}

pub fn calculate_intersection_over_union(b1: &[f32; 4], b2: &[f32; 4]) -> f32 {
    let x1 = b1[0].max(b2[0]);
    let y1 = b1[1].max(b2[1]);
    let x2 = b1[2].min(b2[2]);
    let y2 = b1[3].min(b2[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    let area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    inter / (area1 + area2 - inter)
}
