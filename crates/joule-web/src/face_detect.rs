//! Face detection, landmark alignment, and embedding distance.
//!
//! Bounding boxes, 5-point landmarks, non-maximum suppression (IoU),
//! face alignment from eye positions, and L2 distance between face
//! embeddings.

// ── BoundingBox ─────────────────────────────────────────────────

/// Axis-aligned bounding box with confidence score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
}

impl BoundingBox {
    pub fn new(x: f64, y: f64, w: f64, h: f64, confidence: f64) -> Self {
        Self { x, y, w, h, confidence }
    }

    /// Area of the box.
    pub fn area(&self) -> f64 {
        self.w * self.h
    }

    /// Center point (cx, cy).
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    /// Intersection-over-union with another box.
    pub fn iou(&self, other: &BoundingBox) -> f64 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.h).min(other.y + other.h);

        let inter_w = (x2 - x1).max(0.0);
        let inter_h = (y2 - y1).max(0.0);
        let inter = inter_w * inter_h;

        let union = self.area() + other.area() - inter;
        if union < 1e-12 { 0.0 } else { inter / union }
    }
}

// ── Landmark ────────────────────────────────────────────────────

/// A 2D facial landmark point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmark {
    pub x: f64,
    pub y: f64,
}

impl Landmark {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another landmark.
    pub fn distance(&self, other: &Landmark) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// 5-point face landmarks.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceLandmarks {
    pub left_eye: Landmark,
    pub right_eye: Landmark,
    pub nose: Landmark,
    pub left_mouth: Landmark,
    pub right_mouth: Landmark,
}

impl FaceLandmarks {
    pub fn new(
        left_eye: Landmark,
        right_eye: Landmark,
        nose: Landmark,
        left_mouth: Landmark,
        right_mouth: Landmark,
    ) -> Self {
        Self { left_eye, right_eye, nose, left_mouth, right_mouth }
    }

    /// Inter-ocular distance.
    pub fn eye_distance(&self) -> f64 {
        self.left_eye.distance(&self.right_eye)
    }

    /// All 5 landmarks as an array.
    pub fn as_array(&self) -> [Landmark; 5] {
        [self.left_eye, self.right_eye, self.nose, self.left_mouth, self.right_mouth]
    }
}

// ── Face detection result ───────────────────────────────────────

/// A single face detection with box and optional landmarks.
#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub bbox: BoundingBox,
    pub landmarks: Option<FaceLandmarks>,
}

impl FaceDetection {
    pub fn new(bbox: BoundingBox, landmarks: Option<FaceLandmarks>) -> Self {
        Self { bbox, landmarks }
    }
}

// ── NMS ─────────────────────────────────────────────────────────

/// Non-maximum suppression. Returns indices of kept detections.
pub fn nms(detections: &[FaceDetection], iou_threshold: f64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..detections.len()).collect();
    indices.sort_by(|a, b| {
        detections[*b]
            .bbox
            .confidence
            .partial_cmp(&detections[*a].bbox.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &indices {
            if !suppressed[j] && j != i {
                if detections[i].bbox.iou(&detections[j].bbox) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }
    keep
}

/// Sort detections by confidence (descending).
pub fn sort_by_confidence(detections: &mut [FaceDetection]) {
    detections.sort_by(|a, b| {
        b.bbox
            .confidence
            .partial_cmp(&a.bbox.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ── Alignment ───────────────────────────────────────────────────

/// Compute the rotation angle (in degrees) to align a face so that
/// the line between the eyes is horizontal.
pub fn alignment_angle(landmarks: &FaceLandmarks) -> f64 {
    let dy = landmarks.right_eye.y - landmarks.left_eye.y;
    let dx = landmarks.right_eye.x - landmarks.left_eye.x;
    dy.atan2(dx).to_degrees()
}

/// Compute the center and scale for face alignment.
pub fn alignment_transform(landmarks: &FaceLandmarks) -> (f64, f64, f64, f64) {
    let center_x = (landmarks.left_eye.x + landmarks.right_eye.x) / 2.0;
    let center_y = (landmarks.left_eye.y + landmarks.right_eye.y) / 2.0;
    let angle = alignment_angle(landmarks);
    let scale = landmarks.eye_distance();
    (center_x, center_y, angle, scale)
}

// ── Crop ────────────────────────────────────────────────────────

/// Crop coordinates for a face from a bounding box, with optional padding.
/// Returns (x1, y1, x2, y2) clamped to image dimensions.
pub fn crop_coords(
    bbox: &BoundingBox,
    img_width: f64,
    img_height: f64,
    padding: f64,
) -> (f64, f64, f64, f64) {
    let pad_w = bbox.w * padding;
    let pad_h = bbox.h * padding;
    let x1 = (bbox.x - pad_w).max(0.0);
    let y1 = (bbox.y - pad_h).max(0.0);
    let x2 = (bbox.x + bbox.w + pad_w).min(img_width);
    let y2 = (bbox.y + bbox.h + pad_h).min(img_height);
    (x1, y1, x2, y2)
}

// ── Embedding distance ──────────────────────────────────────────

/// L2 (Euclidean) distance between two face embedding vectors.
pub fn embedding_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "embedding dimensions must match");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Cosine similarity between two face embedding vectors.
pub fn embedding_cosine(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "embedding dimensions must match");
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a < 1e-12 || mag_b < 1e-12 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_area_and_center() {
        let bb = BoundingBox::new(10.0, 20.0, 100.0, 50.0, 0.9);
        assert_eq!(bb.area(), 5000.0);
        assert_eq!(bb.center(), (60.0, 45.0));
    }

    #[test]
    fn test_iou_identical() {
        let bb = BoundingBox::new(0.0, 0.0, 100.0, 100.0, 0.9);
        assert!((bb.iou(&bb) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9);
        let b = BoundingBox::new(20.0, 20.0, 10.0, 10.0, 0.8);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_iou_partial() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9);
        let b = BoundingBox::new(5.0, 0.0, 10.0, 10.0, 0.8);
        // Intersection: 5*10 = 50, Union: 100+100-50 = 150
        assert!((a.iou(&b) - 50.0 / 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_landmark_distance() {
        let a = Landmark::new(0.0, 0.0);
        let b = Landmark::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_face_landmarks() {
        let lm = FaceLandmarks::new(
            Landmark::new(30.0, 50.0),
            Landmark::new(70.0, 50.0),
            Landmark::new(50.0, 70.0),
            Landmark::new(35.0, 90.0),
            Landmark::new(65.0, 90.0),
        );
        assert!((lm.eye_distance() - 40.0).abs() < 1e-9);
        assert_eq!(lm.as_array().len(), 5);
    }

    #[test]
    fn test_nms_suppression() {
        let dets = vec![
            FaceDetection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0, 0.9), None),
            FaceDetection::new(BoundingBox::new(5.0, 5.0, 100.0, 100.0, 0.8), None),
            FaceDetection::new(BoundingBox::new(300.0, 300.0, 50.0, 50.0, 0.7), None),
        ];
        let keep = nms(&dets, 0.5);
        assert_eq!(keep.len(), 2);
        assert!(keep.contains(&0));
        assert!(keep.contains(&2));
    }

    #[test]
    fn test_sort_by_confidence() {
        let mut dets = vec![
            FaceDetection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.3), None),
            FaceDetection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9), None),
            FaceDetection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.6), None),
        ];
        sort_by_confidence(&mut dets);
        assert_eq!(dets[0].bbox.confidence, 0.9);
        assert_eq!(dets[1].bbox.confidence, 0.6);
        assert_eq!(dets[2].bbox.confidence, 0.3);
    }

    #[test]
    fn test_alignment_angle_horizontal() {
        let lm = FaceLandmarks::new(
            Landmark::new(30.0, 50.0),
            Landmark::new(70.0, 50.0),
            Landmark::new(50.0, 70.0),
            Landmark::new(35.0, 90.0),
            Landmark::new(65.0, 90.0),
        );
        let angle = alignment_angle(&lm);
        assert!(angle.abs() < 1e-9, "eyes are horizontal, angle should be 0");
    }

    #[test]
    fn test_alignment_angle_tilted() {
        let lm = FaceLandmarks::new(
            Landmark::new(30.0, 50.0),
            Landmark::new(70.0, 90.0),
            Landmark::new(50.0, 70.0),
            Landmark::new(35.0, 90.0),
            Landmark::new(65.0, 90.0),
        );
        let angle = alignment_angle(&lm);
        assert!((angle - 45.0).abs() < 1e-6);
    }

    #[test]
    fn test_crop_coords() {
        let bb = BoundingBox::new(50.0, 50.0, 100.0, 100.0, 0.9);
        let (x1, y1, x2, y2) = crop_coords(&bb, 640.0, 480.0, 0.1);
        assert!(x1 < 50.0);
        assert!(y1 < 50.0);
        assert!(x2 > 150.0);
        assert!(y2 > 150.0);
    }

    #[test]
    fn test_embedding_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = embedding_distance(&a, &b);
        assert!((d - std::f64::consts::SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn test_embedding_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((embedding_cosine(&a, &b) - 1.0).abs() < 1e-9);

        let c = vec![0.0, 1.0];
        assert!(embedding_cosine(&a, &c).abs() < 1e-9);
    }
}
