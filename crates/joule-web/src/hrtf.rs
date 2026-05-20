//! Head-Related Transfer Function for binaural audio rendering.
//!
//! HRTF dataset with per-direction impulse response pairs, interpolation
//! between measured directions, convolution, azimuth/elevation computation,
//! head shadow, and interaural time delay.

use std::collections::HashMap;

// ── Types ──────────────────────────────────────────────────────

/// A direction in spherical coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Direction {
    /// Azimuth in radians (-PI to PI, 0 = front, positive = right).
    pub azimuth: f64,
    /// Elevation in radians (-PI/2 to PI/2, 0 = horizon, positive = up).
    pub elevation: f64,
}

impl Direction {
    pub fn new(azimuth: f64, elevation: f64) -> Self {
        Self { azimuth, elevation }
    }

    /// Angular distance between two directions on the unit sphere.
    pub fn angular_distance(&self, other: &Direction) -> f64 {
        let cos_d = self.elevation.sin() * other.elevation.sin()
            + self.elevation.cos() * other.elevation.cos()
            * (self.azimuth - other.azimuth).cos();
        cos_d.clamp(-1.0, 1.0).acos()
    }
}

/// A single HRTF measurement: impulse response for left and right ears.
#[derive(Debug, Clone, PartialEq)]
pub struct HrtfMeasurement {
    pub direction: Direction,
    pub left_ir: Vec<f64>,
    pub right_ir: Vec<f64>,
}

/// HRTF dataset containing measurements at multiple directions.
#[derive(Debug, Clone)]
pub struct HrtfDataset {
    measurements: Vec<HrtfMeasurement>,
    sample_rate: u32,
}

impl HrtfDataset {
    /// Create a new dataset.
    pub fn new(sample_rate: u32) -> Self {
        Self { measurements: Vec::new(), sample_rate }
    }

    /// Add a measurement.
    pub fn add_measurement(&mut self, measurement: HrtfMeasurement) {
        self.measurements.push(measurement);
    }

    /// Number of measurements.
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Find the nearest measurement to a direction.
    pub fn nearest(&self, dir: &Direction) -> Option<&HrtfMeasurement> {
        self.measurements.iter().min_by(|a, b| {
            let da = a.direction.angular_distance(dir);
            let db = b.direction.angular_distance(dir);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the K nearest measurements to a direction.
    pub fn k_nearest(&self, dir: &Direction, k: usize) -> Vec<(&HrtfMeasurement, f64)> {
        let mut indexed: Vec<(&HrtfMeasurement, f64)> = self.measurements.iter()
            .map(|m| (m, m.direction.angular_distance(dir)))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Interpolate HRTF for a direction using inverse-distance weighting
    /// of the K nearest measurements.
    pub fn interpolate(&self, dir: &Direction, k: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        if self.measurements.is_empty() { return None; }
        let nearest = self.k_nearest(dir, k);
        if nearest.is_empty() { return None; }

        // If any measurement is at exactly this direction, use it
        if nearest[0].1 < 1e-12 {
            return Some((nearest[0].0.left_ir.clone(), nearest[0].0.right_ir.clone()));
        }

        let ir_len = nearest[0].0.left_ir.len();
        let mut left = vec![0.0f64; ir_len];
        let mut right = vec![0.0f64; ir_len];
        let mut total_weight = 0.0;

        for (measurement, dist) in &nearest {
            let w = 1.0 / dist;
            total_weight += w;
            for i in 0..ir_len.min(measurement.left_ir.len()) {
                left[i] += measurement.left_ir[i] * w;
            }
            for i in 0..ir_len.min(measurement.right_ir.len()) {
                right[i] += measurement.right_ir[i] * w;
            }
        }

        if total_weight > 1e-12 {
            for s in &mut left { *s /= total_weight; }
            for s in &mut right { *s /= total_weight; }
        }

        Some((left, right))
    }
}

// ── Convolution ────────────────────────────────────────────────

/// Time-domain convolution of signal with impulse response.
pub fn convolve(signal: &[f64], ir: &[f64]) -> Vec<f64> {
    if signal.is_empty() || ir.is_empty() {
        return Vec::new();
    }
    let out_len = signal.len() + ir.len() - 1;
    let mut output = vec![0.0f64; out_len];
    for i in 0..signal.len() {
        for j in 0..ir.len() {
            output[i + j] += signal[i] * ir[j];
        }
    }
    output
}

/// Overlap-add convolution (block-based, simulates FFT approach).
pub fn convolve_overlap_add(signal: &[f64], ir: &[f64], block_size: usize) -> Vec<f64> {
    if signal.is_empty() || ir.is_empty() || block_size == 0 {
        return Vec::new();
    }
    let out_len = signal.len() + ir.len() - 1;
    let mut output = vec![0.0f64; out_len];
    let num_blocks = (signal.len() + block_size - 1) / block_size;

    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(signal.len());
        let block = &signal[start..end];
        let conv = convolve(block, ir);
        for (i, &val) in conv.iter().enumerate() {
            let idx = start + i;
            if idx < output.len() {
                output[idx] += val;
            }
        }
    }
    output
}

// ── HRTF Processor ─────────────────────────────────────────────

/// Head radius in meters (average adult).
const HEAD_RADIUS: f64 = 0.0875;

/// Compute interaural time delay (ITD) in seconds.
///
/// Uses Woodworth's formula: ITD = (r/c)(azimuth + sin(azimuth))
/// where r is head radius, c is speed of sound.
pub fn compute_itd(azimuth: f64, speed_of_sound: f64) -> f64 {
    let az = azimuth.clamp(-std::f64::consts::PI, std::f64::consts::PI);
    (HEAD_RADIUS / speed_of_sound) * (az.abs() + az.abs().sin())
}

/// Compute ITD in samples for a given sample rate.
pub fn itd_samples(azimuth: f64, speed_of_sound: f64, sample_rate: u32) -> usize {
    let itd = compute_itd(azimuth, speed_of_sound);
    (itd * sample_rate as f64).round().max(0.0) as usize
}

/// Head shadow attenuation factor for the far ear.
///
/// High frequencies are more attenuated; this gives a simplified
/// broadband factor (0.0-1.0) based on angle.
pub fn head_shadow_factor(azimuth: f64) -> f64 {
    // At 0 degrees (front), no shadow. At 90 degrees, maximum shadow.
    let angle = azimuth.abs().clamp(0.0, std::f64::consts::PI);
    let shadow = 1.0 - 0.6 * (angle / std::f64::consts::PI);
    shadow.clamp(0.2, 1.0)
}

/// Frequency-dependent head shadow (simplified model).
/// Returns attenuation factor for a given frequency and azimuth.
pub fn frequency_dependent_shadow(azimuth: f64, frequency: f64) -> f64 {
    let angle = azimuth.abs().clamp(0.0, std::f64::consts::PI);
    // Higher frequencies are shadowed more
    let freq_factor = (frequency / 2000.0).clamp(0.0, 1.0);
    let base_shadow = 1.0 - 0.5 * (angle / std::f64::consts::PI);
    let adjusted = base_shadow * (1.0 - 0.4 * freq_factor);
    adjusted.clamp(0.1, 1.0)
}

/// HRTF binaural audio processor.
#[derive(Debug, Clone)]
pub struct HrtfProcessor {
    dataset: HrtfDataset,
    speed_of_sound: f64,
    interpolation_k: usize,
    enable_itd: bool,
    enable_head_shadow: bool,
}

impl HrtfProcessor {
    /// Create a new processor with an HRTF dataset.
    pub fn new(dataset: HrtfDataset) -> Self {
        Self {
            dataset,
            speed_of_sound: 343.0,
            interpolation_k: 3,
            enable_itd: true,
            enable_head_shadow: true,
        }
    }

    /// Set the number of nearest directions used for interpolation.
    pub fn set_interpolation_k(&mut self, k: usize) {
        self.interpolation_k = k.max(1);
    }

    /// Enable or disable ITD.
    pub fn set_itd_enabled(&mut self, enabled: bool) {
        self.enable_itd = enabled;
    }

    /// Enable or disable head shadow.
    pub fn set_head_shadow_enabled(&mut self, enabled: bool) {
        self.enable_head_shadow = enabled;
    }

    /// Process a mono signal for a given direction, producing binaural output.
    /// Returns (left_channel, right_channel).
    pub fn process(&self, signal: &[f64], direction: &Direction) -> (Vec<f64>, Vec<f64>) {
        // Get interpolated HRTF
        let (left_ir, right_ir) = match self.dataset.interpolate(direction, self.interpolation_k) {
            Some(pair) => pair,
            None => {
                // No dataset, pass through
                return (signal.to_vec(), signal.to_vec());
            }
        };

        // Convolve signal with left and right IRs
        let mut left = convolve(signal, &left_ir);
        let mut right = convolve(signal, &right_ir);

        // Apply head shadow to the far ear
        if self.enable_head_shadow {
            let shadow = head_shadow_factor(direction.azimuth);
            if direction.azimuth > 0.0 {
                // Source is to the right, left ear is shadowed
                for s in &mut left { *s *= shadow; }
            } else if direction.azimuth < 0.0 {
                // Source is to the left, right ear is shadowed
                for s in &mut right { *s *= shadow; }
            }
        }

        // Apply ITD: delay the far ear
        if self.enable_itd {
            let delay = itd_samples(direction.azimuth, self.speed_of_sound, self.dataset.sample_rate);
            if delay > 0 {
                if direction.azimuth > 0.0 {
                    // Source right → delay left ear
                    let mut delayed = vec![0.0; delay];
                    delayed.extend_from_slice(&left);
                    left = delayed;
                } else {
                    // Source left → delay right ear
                    let mut delayed = vec![0.0; delay];
                    delayed.extend_from_slice(&right);
                    right = delayed;
                }
            }
        }

        // Match lengths
        let max_len = left.len().max(right.len());
        left.resize(max_len, 0.0);
        right.resize(max_len, 0.0);

        (left, right)
    }

    /// Compute direction from 3D positions (listener at origin, facing -Z).
    pub fn direction_from_positions(
        listener_pos: &[f64; 3],
        listener_fwd: &[f64; 3],
        listener_up: &[f64; 3],
        source_pos: &[f64; 3],
    ) -> Direction {
        let dx = source_pos[0] - listener_pos[0];
        let dy = source_pos[1] - listener_pos[1];
        let dz = source_pos[2] - listener_pos[2];

        let fwd_len = (listener_fwd[0].powi(2) + listener_fwd[1].powi(2) + listener_fwd[2].powi(2)).sqrt();
        let up_len = (listener_up[0].powi(2) + listener_up[1].powi(2) + listener_up[2].powi(2)).sqrt();

        if fwd_len < 1e-12 || up_len < 1e-12 {
            return Direction::new(0.0, 0.0);
        }

        let fwd = [listener_fwd[0] / fwd_len, listener_fwd[1] / fwd_len, listener_fwd[2] / fwd_len];
        let up = [listener_up[0] / up_len, listener_up[1] / up_len, listener_up[2] / up_len];
        let right = [
            fwd[1] * up[2] - fwd[2] * up[1],
            fwd[2] * up[0] - fwd[0] * up[2],
            fwd[0] * up[1] - fwd[1] * up[0],
        ];

        let proj_fwd = dx * fwd[0] + dy * fwd[1] + dz * fwd[2];
        let proj_right = dx * right[0] + dy * right[1] + dz * right[2];
        let proj_up = dx * up[0] + dy * up[1] + dz * up[2];

        let azimuth = proj_right.atan2(proj_fwd);
        let horiz_dist = (proj_fwd * proj_fwd + proj_right * proj_right).sqrt();
        let elevation = proj_up.atan2(horiz_dist);

        Direction::new(azimuth, elevation)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_dataset() -> HrtfDataset {
        let mut ds = HrtfDataset::new(44100);
        // Front (azimuth=0, elevation=0)
        ds.add_measurement(HrtfMeasurement {
            direction: Direction::new(0.0, 0.0),
            left_ir: vec![1.0, 0.5, 0.25],
            right_ir: vec![1.0, 0.5, 0.25],
        });
        // Right (azimuth=PI/2)
        ds.add_measurement(HrtfMeasurement {
            direction: Direction::new(PI / 2.0, 0.0),
            left_ir: vec![0.5, 0.25, 0.1],
            right_ir: vec![1.2, 0.6, 0.3],
        });
        // Left (azimuth=-PI/2)
        ds.add_measurement(HrtfMeasurement {
            direction: Direction::new(-PI / 2.0, 0.0),
            left_ir: vec![1.2, 0.6, 0.3],
            right_ir: vec![0.5, 0.25, 0.1],
        });
        // Behind (azimuth=PI)
        ds.add_measurement(HrtfMeasurement {
            direction: Direction::new(PI, 0.0),
            left_ir: vec![0.8, 0.4, 0.2],
            right_ir: vec![0.8, 0.4, 0.2],
        });
        ds
    }

    #[test]
    fn test_direction_angular_distance_same() {
        let d = Direction::new(0.0, 0.0);
        assert!(d.angular_distance(&d) < 1e-10);
    }

    #[test]
    fn test_direction_angular_distance_opposite() {
        let a = Direction::new(0.0, 0.0);
        let b = Direction::new(PI, 0.0);
        assert!((a.angular_distance(&b) - PI).abs() < 1e-6);
    }

    #[test]
    fn test_dataset_nearest() {
        let ds = make_dataset();
        let nearest = ds.nearest(&Direction::new(0.05, 0.0)).unwrap();
        assert!(nearest.direction.azimuth.abs() < 0.1);
    }

    #[test]
    fn test_dataset_k_nearest() {
        let ds = make_dataset();
        let result = ds.k_nearest(&Direction::new(0.0, 0.0), 2);
        assert_eq!(result.len(), 2);
        assert!(result[0].1 <= result[1].1);
    }

    #[test]
    fn test_convolve_impulse() {
        let signal = vec![1.0, 2.0, 3.0];
        let ir = vec![1.0];
        let out = convolve(&signal, &ir);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolve_delay() {
        let signal = vec![1.0, 0.0, 0.0];
        let ir = vec![0.0, 0.0, 1.0];
        let out = convolve(&signal, &ir);
        assert_eq!(out.len(), 5);
        assert!((out[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolve_overlap_add() {
        let signal: Vec<f64> = (0..16).map(|i| i as f64 / 16.0).collect();
        let ir = vec![1.0, 0.5];
        let direct = convolve(&signal, &ir);
        let ola = convolve_overlap_add(&signal, &ir, 4);
        assert_eq!(direct.len(), ola.len());
        for i in 0..direct.len() {
            assert!((direct[i] - ola[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_itd_zero_azimuth() {
        let itd = compute_itd(0.0, 343.0);
        assert!(itd.abs() < 1e-10);
    }

    #[test]
    fn test_itd_positive_azimuth() {
        let itd = compute_itd(PI / 2.0, 343.0);
        assert!(itd > 0.0);
        // Should be less than 1ms typically
        assert!(itd < 0.001);
    }

    #[test]
    fn test_itd_samples() {
        let samples = itd_samples(PI / 2.0, 343.0, 44100);
        assert!(samples > 0);
        assert!(samples < 50); // ~0.7ms max → ~31 samples at 44.1kHz
    }

    #[test]
    fn test_head_shadow_front() {
        let factor = head_shadow_factor(0.0);
        assert!((factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_head_shadow_side() {
        let factor = head_shadow_factor(PI / 2.0);
        assert!(factor < 1.0);
        assert!(factor > 0.2);
    }

    #[test]
    fn test_frequency_shadow() {
        let low = frequency_dependent_shadow(PI / 2.0, 200.0);
        let high = frequency_dependent_shadow(PI / 2.0, 8000.0);
        assert!(low > high);
    }

    #[test]
    fn test_processor_front_symmetric() {
        let ds = make_dataset();
        let proc = HrtfProcessor::new(ds);
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let dir = Direction::new(0.0, 0.0);
        let (left, right) = proc.process(&signal, &dir);
        // Front source → similar left/right (no ITD, no head shadow)
        assert_eq!(left.len(), right.len());
        // Since azimuth is 0, no shadow/ITD applied, so left ≈ right
        for i in 0..left.len().min(right.len()) {
            assert!((left[i] - right[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_processor_right_asymmetric() {
        let ds = make_dataset();
        let mut proc = HrtfProcessor::new(ds);
        proc.set_head_shadow_enabled(false);
        proc.set_itd_enabled(false);
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let dir = Direction::new(PI / 2.0, 0.0);
        let (left, right) = proc.process(&signal, &dir);
        // Right source → right ear louder
        let left_energy: f64 = left.iter().map(|s| s * s).sum();
        let right_energy: f64 = right.iter().map(|s| s * s).sum();
        assert!(right_energy > left_energy);
    }

    #[test]
    fn test_interpolation() {
        let ds = make_dataset();
        let result = ds.interpolate(&Direction::new(PI / 4.0, 0.0), 3);
        assert!(result.is_some());
        let (left, right) = result.unwrap();
        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 3);
    }

    #[test]
    fn test_direction_from_positions_front() {
        let dir = HrtfProcessor::direction_from_positions(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, -1.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, -5.0],
        );
        assert!(dir.azimuth.abs() < 0.01);
        assert!(dir.elevation.abs() < 0.01);
    }

    #[test]
    fn test_direction_from_positions_right() {
        let dir = HrtfProcessor::direction_from_positions(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, -1.0],
            &[0.0, 1.0, 0.0],
            &[5.0, 0.0, 0.0],
        );
        assert!((dir.azimuth - PI / 2.0).abs() < 0.1);
    }

    #[test]
    fn test_empty_dataset() {
        let ds = HrtfDataset::new(44100);
        assert_eq!(ds.measurement_count(), 0);
        assert!(ds.nearest(&Direction::new(0.0, 0.0)).is_none());
    }

    #[test]
    fn test_convolve_empty() {
        assert!(convolve(&[], &[1.0]).is_empty());
        assert!(convolve(&[1.0], &[]).is_empty());
    }

    #[test]
    fn test_processor_with_itd() {
        let ds = make_dataset();
        let mut proc = HrtfProcessor::new(ds);
        proc.set_itd_enabled(true);
        proc.set_head_shadow_enabled(false);
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let dir = Direction::new(PI / 2.0, 0.0);
        let (left, right) = proc.process(&signal, &dir);
        // ITD should make left longer than convolution alone
        // because left ear gets delayed for right source
        assert!(left.len() >= right.len());
    }
}
