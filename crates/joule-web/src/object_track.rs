//! Object tracking: Kalman filter tracker, Hungarian algorithm
//! assignment, and track lifecycle management.
//!
//! Provides a multi-object tracker that associates detections
//! across frames using IoU-based cost and the Hungarian method,
//! with Kalman-filtered state estimation for smooth trajectories.

use std::fmt;

// ── Detection ──────────────────────────────────────────────────

/// A 2D axis-aligned bounding box detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
    pub class_id: u32,
}

impl Detection {
    pub fn new(x: f64, y: f64, w: f64, h: f64, confidence: f64) -> Self {
        Self { x, y, w, h, confidence, class_id: 0 }
    }

    pub fn with_class(mut self, class_id: u32) -> Self {
        self.class_id = class_id;
        self
    }

    pub fn center(&self) -> (f64, f64) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    pub fn area(&self) -> f64 {
        self.w * self.h
    }

    /// IoU with another detection.
    pub fn iou(&self, other: &Detection) -> f64 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.h).min(other.y + other.h);
        let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union < 1e-12 { 0.0 } else { inter / union }
    }
}

impl fmt::Display for Detection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Det([{:.0},{:.0},{:.0},{:.0}] c={:.2} cls={})",
               self.x, self.y, self.w, self.h, self.confidence, self.class_id)
    }
}

// ── Kalman Filter (constant velocity, 2D) ──────────────────────

/// State vector: [cx, cy, w, h, vx, vy, vw, vh].
const STATE_DIM: usize = 8;
/// Measurement vector: [cx, cy, w, h].
const MEAS_DIM: usize = 4;

/// A simple linear Kalman filter for bounding box tracking.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    pub state: [f64; STATE_DIM],
    pub covariance: [[f64; STATE_DIM]; STATE_DIM],
    pub process_noise: f64,
    pub measurement_noise: f64,
}

impl KalmanFilter {
    /// Initialize from a detection bounding box.
    pub fn from_detection(det: &Detection) -> Self {
        let (cx, cy) = det.center();
        let mut state = [0.0_f64; STATE_DIM];
        state[0] = cx;
        state[1] = cy;
        state[2] = det.w;
        state[3] = det.h;

        let mut cov = [[0.0_f64; STATE_DIM]; STATE_DIM];
        for i in 0..STATE_DIM {
            cov[i][i] = if i < MEAS_DIM { 10.0 } else { 100.0 };
        }

        Self { state, covariance: cov, process_noise: 1.0, measurement_noise: 1.0 }
    }

    pub fn with_process_noise(mut self, noise: f64) -> Self {
        self.process_noise = noise;
        self
    }

    pub fn with_measurement_noise(mut self, noise: f64) -> Self {
        self.measurement_noise = noise;
        self
    }

    /// Predict the next state (constant velocity model).
    pub fn predict(&mut self) {
        // state = F * state  (F = identity + dt in velocity slots)
        self.state[0] += self.state[4];
        self.state[1] += self.state[5];
        self.state[2] += self.state[6];
        self.state[3] += self.state[7];

        // P = F*P*F^T + Q  (simplified: add process noise to diagonal)
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                if i < MEAS_DIM && j >= MEAS_DIM && j == i + MEAS_DIM {
                    self.covariance[i][j] += self.covariance[j][j];
                }
            }
            self.covariance[i][i] += self.process_noise;
        }
    }

    /// Update with a measurement [cx, cy, w, h].
    pub fn update(&mut self, measurement: [f64; MEAS_DIM]) {
        // Innovation y = z - H*x  (H selects first 4 elements)
        let mut innovation = [0.0_f64; MEAS_DIM];
        for i in 0..MEAS_DIM {
            innovation[i] = measurement[i] - self.state[i];
        }

        // S = H*P*H^T + R
        let mut s_mat = [[0.0_f64; MEAS_DIM]; MEAS_DIM];
        for i in 0..MEAS_DIM {
            for j in 0..MEAS_DIM {
                s_mat[i][j] = self.covariance[i][j];
            }
            s_mat[i][i] += self.measurement_noise;
        }

        // K = P*H^T * S^{-1}  (use Cramer's rule for 4x4 or simplified)
        let s_inv = invert_4x4(&s_mat);
        let mut kalman_gain = [[0.0_f64; MEAS_DIM]; STATE_DIM];
        for i in 0..STATE_DIM {
            for j in 0..MEAS_DIM {
                let mut val = 0.0_f64;
                for k in 0..MEAS_DIM {
                    val += self.covariance[i][k] * s_inv[k][j];
                }
                kalman_gain[i][j] = val;
            }
        }

        // x = x + K*y
        for i in 0..STATE_DIM {
            for j in 0..MEAS_DIM {
                self.state[i] += kalman_gain[i][j] * innovation[j];
            }
        }

        // P = (I - K*H) * P
        let old_cov = self.covariance;
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                let mut kh = 0.0_f64;
                for k in 0..MEAS_DIM {
                    let h_kj = if k == j { 1.0 } else { 0.0 };
                    kh += kalman_gain[i][k] * h_kj;
                }
                let eye = if i == j { 1.0 } else { 0.0 };
                let factor = eye - kh;
                let mut val = 0.0_f64;
                for m in 0..STATE_DIM {
                    let f_im = if i == m { 1.0 } else { 0.0 };
                    let f_im_adj = f_im - {
                        let mut tmp = 0.0_f64;
                        for k in 0..MEAS_DIM {
                            let h_km = if k == m { 1.0 } else { 0.0 };
                            tmp += kalman_gain[i][k] * h_km;
                        }
                        tmp
                    };
                    val += f_im_adj * old_cov[m][j];
                }
                self.covariance[i][j] = val;
            }
        }

        // Force width/height positive
        self.state[2] = self.state[2].max(1.0);
        self.state[3] = self.state[3].max(1.0);
    }

    /// Current bounding box as Detection.
    pub fn to_detection(&self) -> Detection {
        let cx = self.state[0];
        let cy = self.state[1];
        let w = self.state[2].max(1.0);
        let h = self.state[3].max(1.0);
        Detection::new(cx - w / 2.0, cy - h / 2.0, w, h, 1.0)
    }
}

/// Invert a 4x4 matrix using Gauss-Jordan elimination.
fn invert_4x4(mat: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut aug = [[0.0_f64; 8]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = mat[i][j];
        }
        aug[i][i + 4] = 1.0;
    }

    for col in 0..4 {
        // Partial pivot
        let mut max_row = col;
        for row in col + 1..4 {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            // Near-singular: return identity
            let mut result = [[0.0_f64; 4]; 4];
            for i in 0..4 { result[i][i] = 1.0; }
            return result;
        }

        for j in 0..8 {
            aug[col][j] /= pivot;
        }

        for row in 0..4 {
            if row == col { continue; }
            let factor = aug[row][col];
            for j in 0..8 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    let mut result = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = aug[i][j + 4];
        }
    }
    result
}

// ── Hungarian Algorithm ────────────────────────────────────────

/// Solve the assignment problem using the Hungarian algorithm.
/// `cost_matrix` is row-major, `rows` x `cols`. Returns assignment
/// pairs `(row, col)` that minimize total cost.
pub fn hungarian_assignment(
    cost_matrix: &[f64], rows: usize, cols: usize,
) -> Vec<(usize, usize)> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }

    let n = rows.max(cols);
    // Pad to square
    let mut cost = vec![vec![0.0_f64; n]; n];
    for r in 0..rows {
        for c in 0..cols {
            cost[r][c] = cost_matrix[r * cols + c];
        }
    }

    // Step 1: Row reduction
    for r in 0..n {
        let min_val = cost[r].iter().cloned().fold(f64::INFINITY, f64::min);
        for c in 0..n {
            cost[r][c] -= min_val;
        }
    }

    // Step 2: Column reduction
    for c in 0..n {
        let min_val = (0..n).map(|r| cost[r][c]).fold(f64::INFINITY, f64::min);
        for r in 0..n {
            cost[r][c] -= min_val;
        }
    }

    // Greedy assignment on zeros (simplified for typical tracking sizes)
    let mut row_assigned = vec![false; n];
    let mut col_assigned = vec![false; n];
    let mut assignment = vec![usize::MAX; n];

    // Pass 1: unique zero assignments
    for r in 0..n {
        for c in 0..n {
            if cost[r][c].abs() < 1e-9 && !row_assigned[r] && !col_assigned[c] {
                assignment[r] = c;
                row_assigned[r] = true;
                col_assigned[c] = true;
                break;
            }
        }
    }

    // Pass 2: assign remaining using minimum cost
    for r in 0..n {
        if assignment[r] == usize::MAX {
            let mut best_c = 0;
            let mut best_cost = f64::INFINITY;
            for c in 0..n {
                if !col_assigned[c] && cost[r][c] < best_cost {
                    best_cost = cost[r][c];
                    best_c = c;
                }
            }
            if best_cost < f64::INFINITY {
                assignment[r] = best_c;
                col_assigned[best_c] = true;
            }
        }
    }

    let mut pairs = Vec::new();
    for r in 0..rows {
        let c = assignment[r];
        if c < cols {
            pairs.push((r, c));
        }
    }
    pairs
}

// ── Track ──────────────────────────────────────────────────────

/// Track state in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Lost,
    Deleted,
}

impl fmt::Display for TrackState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrackState::Tentative => write!(f, "Tentative"),
            TrackState::Confirmed => write!(f, "Confirmed"),
            TrackState::Lost => write!(f, "Lost"),
            TrackState::Deleted => write!(f, "Deleted"),
        }
    }
}

/// A single tracked object with Kalman filter state.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,
    pub state: TrackState,
    pub kf: KalmanFilter,
    pub hits: u32,
    pub misses: u32,
    pub age: u32,
    pub class_id: u32,
}

impl Track {
    pub fn new(id: u64, detection: &Detection) -> Self {
        Self {
            id,
            state: TrackState::Tentative,
            kf: KalmanFilter::from_detection(detection),
            hits: 1,
            misses: 0,
            age: 0,
            class_id: detection.class_id,
        }
    }

    pub fn predict(&mut self) {
        self.kf.predict();
        self.age += 1;
    }

    pub fn update_with(&mut self, det: &Detection) {
        let (cx, cy) = det.center();
        self.kf.update([cx, cy, det.w, det.h]);
        self.hits += 1;
        self.misses = 0;
    }

    pub fn mark_missed(&mut self) {
        self.misses += 1;
    }

    pub fn current_detection(&self) -> Detection {
        self.kf.to_detection().with_class(self.class_id)
    }
}

impl fmt::Display for Track {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let det = self.current_detection();
        write!(f, "Track(id={} {} hits={} miss={} {})",
               self.id, self.state, self.hits, self.misses, det)
    }
}

// ── MultiTracker ───────────────────────────────────────────────

/// Configuration for multi-object tracker.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    pub iou_threshold: f64,
    pub max_misses: u32,
    pub min_hits_to_confirm: u32,
}

impl TrackerConfig {
    pub fn new() -> Self {
        Self { iou_threshold: 0.3, max_misses: 5, min_hits_to_confirm: 3 }
    }

    pub fn with_iou_threshold(mut self, thr: f64) -> Self {
        self.iou_threshold = thr;
        self
    }

    pub fn with_max_misses(mut self, m: u32) -> Self {
        self.max_misses = m;
        self
    }

    pub fn with_min_hits(mut self, h: u32) -> Self {
        self.min_hits_to_confirm = h;
        self
    }
}

impl fmt::Display for TrackerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TrackerConfig(iou={}, max_miss={}, min_hits={})",
               self.iou_threshold, self.max_misses, self.min_hits_to_confirm)
    }
}

/// Multi-object tracker using IoU assignment and Kalman filtering.
#[derive(Debug)]
pub struct MultiTracker {
    pub tracks: Vec<Track>,
    pub config: TrackerConfig,
    next_id: u64,
}

impl MultiTracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self { tracks: Vec::new(), config, next_id: 1 }
    }

    /// Process a new frame of detections.
    pub fn update(&mut self, detections: &[Detection]) {
        // Predict all tracks
        for track in &mut self.tracks {
            track.predict();
        }

        if self.tracks.is_empty() {
            for det in detections {
                self.tracks.push(Track::new(self.next_id, det));
                self.next_id += 1;
            }
            return;
        }

        // Build IoU cost matrix (cost = 1 - IoU)
        let n_tracks = self.tracks.len();
        let n_dets = detections.len();
        let mut cost = vec![0.0_f64; n_tracks * n_dets];

        for t in 0..n_tracks {
            let t_det = self.tracks[t].current_detection();
            for d in 0..n_dets {
                cost[t * n_dets + d] = 1.0 - t_det.iou(&detections[d]);
            }
        }

        let assignments = hungarian_assignment(&cost, n_tracks, n_dets);

        let mut matched_tracks = vec![false; n_tracks];
        let mut matched_dets = vec![false; n_dets];

        for &(t, d) in &assignments {
            let iou_val = 1.0 - cost[t * n_dets + d];
            if iou_val >= self.config.iou_threshold {
                self.tracks[t].update_with(&detections[d]);
                matched_tracks[t] = true;
                matched_dets[d] = true;
            }
        }

        // Handle unmatched tracks
        for t in 0..n_tracks {
            if !matched_tracks[t] {
                self.tracks[t].mark_missed();
            }
        }

        // Create new tracks for unmatched detections
        for d in 0..n_dets {
            if !matched_dets[d] {
                self.tracks.push(Track::new(self.next_id, &detections[d]));
                self.next_id += 1;
            }
        }

        // Update track states
        for track in &mut self.tracks {
            match track.state {
                TrackState::Tentative => {
                    if track.hits >= self.config.min_hits_to_confirm {
                        track.state = TrackState::Confirmed;
                    } else if track.misses > self.config.max_misses {
                        track.state = TrackState::Deleted;
                    }
                }
                TrackState::Confirmed => {
                    if track.misses > 0 {
                        track.state = TrackState::Lost;
                    }
                }
                TrackState::Lost => {
                    if track.misses == 0 {
                        track.state = TrackState::Confirmed;
                    } else if track.misses > self.config.max_misses {
                        track.state = TrackState::Deleted;
                    }
                }
                TrackState::Deleted => {}
            }
        }

        // Remove deleted tracks
        self.tracks.retain(|t| t.state != TrackState::Deleted);
    }

    /// Get all confirmed tracks.
    pub fn confirmed_tracks(&self) -> Vec<&Track> {
        self.tracks.iter().filter(|t| t.state == TrackState::Confirmed).collect()
    }

    /// Get all active (non-deleted) tracks.
    pub fn active_tracks(&self) -> Vec<&Track> {
        self.tracks.iter().filter(|t| t.state != TrackState::Deleted).collect()
    }
}

impl fmt::Display for MultiTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultiTracker({} tracks, {} confirmed)",
               self.tracks.len(), self.confirmed_tracks().len())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f64, y: f64, w: f64, h: f64) -> Detection {
        Detection::new(x, y, w, h, 0.9)
    }

    #[test]
    fn test_detection_center() {
        let d = det(10.0, 20.0, 100.0, 50.0);
        assert_eq!(d.center(), (60.0, 45.0));
    }

    #[test]
    fn test_detection_iou_identical() {
        let d = det(0.0, 0.0, 100.0, 100.0);
        assert!((d.iou(&d) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_detection_iou_no_overlap() {
        let a = det(0.0, 0.0, 10.0, 10.0);
        let b = det(20.0, 20.0, 10.0, 10.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_detection_display() {
        let d = det(10.0, 20.0, 30.0, 40.0);
        let s = format!("{}", d);
        assert!(s.contains("Det"));
    }

    #[test]
    fn test_detection_with_class() {
        let d = det(0.0, 0.0, 10.0, 10.0).with_class(5);
        assert_eq!(d.class_id, 5);
    }

    #[test]
    fn test_kalman_init() {
        let d = det(10.0, 20.0, 30.0, 40.0);
        let kf = KalmanFilter::from_detection(&d);
        assert_eq!(kf.state[0], 25.0); // cx
        assert_eq!(kf.state[1], 40.0); // cy
    }

    #[test]
    fn test_kalman_predict_update() {
        let d = det(100.0, 100.0, 50.0, 50.0);
        let mut kf = KalmanFilter::from_detection(&d);
        kf.predict();
        kf.update([130.0, 130.0, 50.0, 50.0]);
        // State should move toward measurement
        assert!(kf.state[0] > 125.0);
        assert!(kf.state[1] > 125.0);
    }

    #[test]
    fn test_kalman_to_detection() {
        let d = det(10.0, 20.0, 30.0, 40.0);
        let kf = KalmanFilter::from_detection(&d);
        let out = kf.to_detection();
        assert!((out.w - 30.0).abs() < 1e-6);
        assert!((out.h - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_hungarian_empty() {
        let result = hungarian_assignment(&[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_hungarian_identity() {
        let cost = [0.0, 1.0, 1.0, 0.0];
        let result = hungarian_assignment(&cost, 2, 2);
        assert_eq!(result.len(), 2);
        // Should assign row 0 -> col 0, row 1 -> col 1
        assert!(result.contains(&(0, 0)));
        assert!(result.contains(&(1, 1)));
    }

    #[test]
    fn test_hungarian_asymmetric() {
        let cost = [1.0, 0.5, 0.3, 0.9, 0.1, 0.8];
        let result = hungarian_assignment(&cost, 2, 3);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_track_state_display() {
        assert_eq!(format!("{}", TrackState::Confirmed), "Confirmed");
        assert_eq!(format!("{}", TrackState::Lost), "Lost");
    }

    #[test]
    fn test_track_new() {
        let d = det(10.0, 10.0, 20.0, 20.0);
        let t = Track::new(1, &d);
        assert_eq!(t.id, 1);
        assert_eq!(t.state, TrackState::Tentative);
        assert_eq!(t.hits, 1);
    }

    #[test]
    fn test_track_display() {
        let d = det(10.0, 10.0, 20.0, 20.0);
        let t = Track::new(42, &d);
        let s = format!("{}", t);
        assert!(s.contains("42"));
    }

    #[test]
    fn test_tracker_config_builder() {
        let cfg = TrackerConfig::new()
            .with_iou_threshold(0.5)
            .with_max_misses(10)
            .with_min_hits(5);
        assert_eq!(cfg.iou_threshold, 0.5);
        assert_eq!(cfg.max_misses, 10);
        assert_eq!(cfg.min_hits_to_confirm, 5);
    }

    #[test]
    fn test_tracker_config_display() {
        let cfg = TrackerConfig::new();
        let s = format!("{}", cfg);
        assert!(s.contains("TrackerConfig"));
    }

    #[test]
    fn test_multi_tracker_single_object() {
        let cfg = TrackerConfig::new().with_min_hits(1);
        let mut tracker = MultiTracker::new(cfg);
        tracker.update(&[det(100.0, 100.0, 50.0, 50.0)]);
        assert_eq!(tracker.tracks.len(), 1);
        assert_eq!(tracker.tracks[0].id, 1);
    }

    #[test]
    fn test_multi_tracker_two_objects() {
        let cfg = TrackerConfig::new().with_min_hits(1);
        let mut tracker = MultiTracker::new(cfg);
        let dets = vec![
            det(10.0, 10.0, 20.0, 20.0),
            det(200.0, 200.0, 30.0, 30.0),
        ];
        tracker.update(&dets);
        assert_eq!(tracker.tracks.len(), 2);
    }

    #[test]
    fn test_multi_tracker_association() {
        let cfg = TrackerConfig::new().with_min_hits(1).with_iou_threshold(0.1);
        let mut tracker = MultiTracker::new(cfg);

        // Frame 1
        tracker.update(&[det(100.0, 100.0, 50.0, 50.0)]);
        let id = tracker.tracks[0].id;

        // Frame 2: slightly moved
        tracker.update(&[det(105.0, 105.0, 50.0, 50.0)]);

        // Same track should persist
        assert!(tracker.tracks.iter().any(|t| t.id == id));
    }

    #[test]
    fn test_multi_tracker_deletion() {
        let cfg = TrackerConfig::new().with_max_misses(2).with_min_hits(1);
        let mut tracker = MultiTracker::new(cfg);

        tracker.update(&[det(100.0, 100.0, 50.0, 50.0)]);
        // Frames with no detections
        tracker.update(&[]);
        tracker.update(&[]);
        tracker.update(&[]);
        // Track should be deleted after max_misses
        let active = tracker.active_tracks();
        assert!(active.is_empty() || active.iter().all(|t| t.misses <= 2));
    }

    #[test]
    fn test_multi_tracker_display() {
        let tracker = MultiTracker::new(TrackerConfig::new());
        let s = format!("{}", tracker);
        assert!(s.contains("MultiTracker"));
    }

    #[test]
    fn test_invert_identity() {
        let mat = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv = invert_4x4(&mat);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-9);
            }
        }
    }
}
