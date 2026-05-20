//! Human pose detection — COCO 17-keypoint skeleton.
//!
//! Keypoint representation, skeleton connectivity, joint angle
//! computation, and non-maximum suppression on person detections.

use std::f64::consts::PI;

// ── Keypoint types ──────────────────────────────────────────────

/// Named COCO 17 keypoint indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KeypointId {
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
}

impl KeypointId {
    /// All 17 keypoint IDs in order.
    pub const ALL: [KeypointId; 17] = [
        KeypointId::Nose,
        KeypointId::LeftEye,
        KeypointId::RightEye,
        KeypointId::LeftEar,
        KeypointId::RightEar,
        KeypointId::LeftShoulder,
        KeypointId::RightShoulder,
        KeypointId::LeftElbow,
        KeypointId::RightElbow,
        KeypointId::LeftWrist,
        KeypointId::RightWrist,
        KeypointId::LeftHip,
        KeypointId::RightHip,
        KeypointId::LeftKnee,
        KeypointId::RightKnee,
        KeypointId::LeftAnkle,
        KeypointId::RightAnkle,
    ];

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            KeypointId::Nose => "nose",
            KeypointId::LeftEye => "left_eye",
            KeypointId::RightEye => "right_eye",
            KeypointId::LeftEar => "left_ear",
            KeypointId::RightEar => "right_ear",
            KeypointId::LeftShoulder => "left_shoulder",
            KeypointId::RightShoulder => "right_shoulder",
            KeypointId::LeftElbow => "left_elbow",
            KeypointId::RightElbow => "right_elbow",
            KeypointId::LeftWrist => "left_wrist",
            KeypointId::RightWrist => "right_wrist",
            KeypointId::LeftHip => "left_hip",
            KeypointId::RightHip => "right_hip",
            KeypointId::LeftKnee => "left_knee",
            KeypointId::RightKnee => "right_knee",
            KeypointId::LeftAnkle => "left_ankle",
            KeypointId::RightAnkle => "right_ankle",
        }
    }
}

/// A single keypoint with position and confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub x: f64,
    pub y: f64,
    pub confidence: f64,
}

impl Keypoint {
    pub fn new(x: f64, y: f64, confidence: f64) -> Self {
        Self { x, y, confidence }
    }
}

// ── Skeleton ────────────────────────────────────────────────────

/// COCO 17-keypoint skeleton connectivity.
pub const SKELETON_CONNECTIONS: [(KeypointId, KeypointId); 19] = [
    (KeypointId::Nose, KeypointId::LeftEye),
    (KeypointId::Nose, KeypointId::RightEye),
    (KeypointId::LeftEye, KeypointId::LeftEar),
    (KeypointId::RightEye, KeypointId::RightEar),
    (KeypointId::Nose, KeypointId::LeftShoulder),
    (KeypointId::Nose, KeypointId::RightShoulder),
    (KeypointId::LeftShoulder, KeypointId::LeftElbow),
    (KeypointId::RightShoulder, KeypointId::RightElbow),
    (KeypointId::LeftElbow, KeypointId::LeftWrist),
    (KeypointId::RightElbow, KeypointId::RightWrist),
    (KeypointId::LeftShoulder, KeypointId::LeftHip),
    (KeypointId::RightShoulder, KeypointId::RightHip),
    (KeypointId::LeftHip, KeypointId::RightHip),
    (KeypointId::LeftHip, KeypointId::LeftKnee),
    (KeypointId::RightHip, KeypointId::RightKnee),
    (KeypointId::LeftKnee, KeypointId::LeftAnkle),
    (KeypointId::RightKnee, KeypointId::RightAnkle),
    (KeypointId::LeftShoulder, KeypointId::RightShoulder),
    (KeypointId::LeftHip, KeypointId::LeftShoulder),
];

/// A detected human skeleton.
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// 17 keypoints indexed by `KeypointId as usize`.
    pub keypoints: [Keypoint; 17],
    /// Overall detection confidence.
    pub score: f64,
}

impl Skeleton {
    /// Create a skeleton with all keypoints at zero.
    pub fn empty() -> Self {
        Self {
            keypoints: [Keypoint::new(0.0, 0.0, 0.0); 17],
            score: 0.0,
        }
    }

    /// Get keypoint by ID.
    pub fn get(&self, id: KeypointId) -> &Keypoint {
        &self.keypoints[id as usize]
    }

    /// Set keypoint by ID.
    pub fn set(&mut self, id: KeypointId, kp: Keypoint) {
        self.keypoints[id as usize] = kp;
    }

    /// Compute the angle (in degrees) at the `middle` joint formed by
    /// `start` → `middle` → `end`.
    pub fn joint_angle(&self, start: KeypointId, middle: KeypointId, end: KeypointId) -> f64 {
        let a = self.get(start);
        let b = self.get(middle);
        let c = self.get(end);

        let ba = (a.x - b.x, a.y - b.y);
        let bc = (c.x - b.x, c.y - b.y);

        let dot = ba.0 * bc.0 + ba.1 * bc.1;
        let mag_ba = (ba.0 * ba.0 + ba.1 * ba.1).sqrt();
        let mag_bc = (bc.0 * bc.0 + bc.1 * bc.1).sqrt();

        if mag_ba < 1e-12 || mag_bc < 1e-12 {
            return 0.0;
        }

        let cos_angle = (dot / (mag_ba * mag_bc)).clamp(-1.0, 1.0);
        cos_angle.acos() * 180.0 / PI
    }

    /// Mean confidence across all keypoints.
    pub fn mean_confidence(&self) -> f64 {
        let sum: f64 = self.keypoints.iter().map(|kp| kp.confidence).sum();
        sum / 17.0
    }

    /// Number of keypoints above a confidence threshold.
    pub fn visible_count(&self, threshold: f64) -> usize {
        self.keypoints.iter().filter(|kp| kp.confidence >= threshold).count()
    }

    /// Bounding box [x_min, y_min, x_max, y_max] from visible keypoints.
    pub fn bounding_box(&self, conf_threshold: f64) -> Option<[f64; 4]> {
        let visible: Vec<_> = self
            .keypoints
            .iter()
            .filter(|kp| kp.confidence >= conf_threshold)
            .collect();
        if visible.is_empty() {
            return None;
        }
        let x_min = visible.iter().map(|kp| kp.x).fold(f64::INFINITY, f64::min);
        let y_min = visible.iter().map(|kp| kp.y).fold(f64::INFINITY, f64::min);
        let x_max = visible.iter().map(|kp| kp.x).fold(f64::NEG_INFINITY, f64::max);
        let y_max = visible.iter().map(|kp| kp.y).fold(f64::NEG_INFINITY, f64::max);
        Some([x_min, y_min, x_max, y_max])
    }
}

// ── Person detection + NMS ──────────────────────────────────────

/// A person detection with bounding box and skeleton.
#[derive(Debug, Clone)]
pub struct PersonDetection {
    /// Bounding box [x, y, width, height].
    pub bbox: [f64; 4],
    /// Detection confidence.
    pub confidence: f64,
    /// Associated skeleton (may have zero keypoints).
    pub skeleton: Skeleton,
}

/// Intersection-over-union for two bounding boxes [x, y, w, h].
pub fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let ax1 = a[0];
    let ay1 = a[1];
    let ax2 = a[0] + a[2];
    let ay2 = a[1] + a[3];

    let bx1 = b[0];
    let by1 = b[1];
    let bx2 = b[0] + b[2];
    let by2 = b[1] + b[3];

    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = a[2] * a[3];
    let area_b = b[2] * b[3];
    let union = area_a + area_b - inter_area;

    if union < 1e-12 { 0.0 } else { inter_area / union }
}

/// Non-maximum suppression on person detections.
pub fn nms(detections: &[PersonDetection], iou_threshold: f64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..detections.len()).collect();
    indices.sort_by(|a, b| {
        detections[*b]
            .confidence
            .partial_cmp(&detections[*a].confidence)
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
                if iou(&detections[i].bbox, &detections[j].bbox) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }
    keep
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skeleton() -> Skeleton {
        let mut skel = Skeleton::empty();
        skel.set(KeypointId::LeftShoulder, Keypoint::new(100.0, 100.0, 0.9));
        skel.set(KeypointId::LeftElbow, Keypoint::new(100.0, 200.0, 0.85));
        skel.set(KeypointId::LeftWrist, Keypoint::new(200.0, 200.0, 0.8));
        skel.score = 0.9;
        skel
    }

    #[test]
    fn test_keypoint_id_count() {
        assert_eq!(KeypointId::ALL.len(), 17);
        assert_eq!(KeypointId::Nose as usize, 0);
        assert_eq!(KeypointId::RightAnkle as usize, 16);
    }

    #[test]
    fn test_keypoint_names() {
        assert_eq!(KeypointId::Nose.name(), "nose");
        assert_eq!(KeypointId::LeftAnkle.name(), "left_ankle");
    }

    #[test]
    fn test_skeleton_get_set() {
        let skel = make_skeleton();
        let kp = skel.get(KeypointId::LeftElbow);
        assert_eq!(kp.x, 100.0);
        assert_eq!(kp.y, 200.0);
        assert_eq!(kp.confidence, 0.85);
    }

    #[test]
    fn test_joint_angle_right_angle() {
        let skel = make_skeleton();
        // Shoulder (100,100) → Elbow (100,200) → Wrist (200,200) = 90°
        let angle = skel.joint_angle(
            KeypointId::LeftShoulder,
            KeypointId::LeftElbow,
            KeypointId::LeftWrist,
        );
        assert!((angle - 90.0).abs() < 1.0);
    }

    #[test]
    fn test_joint_angle_straight() {
        let mut skel = Skeleton::empty();
        skel.set(KeypointId::LeftShoulder, Keypoint::new(0.0, 0.0, 1.0));
        skel.set(KeypointId::LeftElbow, Keypoint::new(0.0, 100.0, 1.0));
        skel.set(KeypointId::LeftWrist, Keypoint::new(0.0, 200.0, 1.0));
        let angle = skel.joint_angle(
            KeypointId::LeftShoulder,
            KeypointId::LeftElbow,
            KeypointId::LeftWrist,
        );
        assert!((angle - 180.0).abs() < 1.0);
    }

    #[test]
    fn test_visible_count() {
        let skel = make_skeleton();
        assert_eq!(skel.visible_count(0.8), 3);
        assert_eq!(skel.visible_count(0.9), 1);
    }

    #[test]
    fn test_mean_confidence() {
        let skel = make_skeleton();
        let mc = skel.mean_confidence();
        assert!(mc > 0.0 && mc < 1.0);
    }

    #[test]
    fn test_bounding_box() {
        let skel = make_skeleton();
        let bb = skel.bounding_box(0.5).unwrap();
        assert_eq!(bb[0], 100.0); // x_min
        assert_eq!(bb[1], 100.0); // y_min
        assert_eq!(bb[2], 200.0); // x_max
        assert_eq!(bb[3], 200.0); // y_max
    }

    #[test]
    fn test_iou_identical() {
        let a = [0.0, 0.0, 100.0, 100.0];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [20.0, 20.0, 10.0, 10.0];
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn test_nms() {
        let dets = vec![
            PersonDetection {
                bbox: [0.0, 0.0, 100.0, 100.0],
                confidence: 0.9,
                skeleton: Skeleton::empty(),
            },
            PersonDetection {
                bbox: [5.0, 5.0, 100.0, 100.0],
                confidence: 0.8,
                skeleton: Skeleton::empty(),
            },
            PersonDetection {
                bbox: [500.0, 500.0, 100.0, 100.0],
                confidence: 0.7,
                skeleton: Skeleton::empty(),
            },
        ];
        let keep = nms(&dets, 0.5);
        // First two overlap heavily, third is separate
        assert_eq!(keep.len(), 2);
        assert!(keep.contains(&0));
        assert!(keep.contains(&2));
    }

    #[test]
    fn test_skeleton_connections_count() {
        assert_eq!(SKELETON_CONNECTIONS.len(), 19);
    }
}
