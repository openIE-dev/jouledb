//! Sonar ranging — time-of-flight distance measurement, ultrasonic beam
//! pattern modeling, multi-sensor ring fusion, and occupancy grid mapping
//! for obstacle avoidance in robotic systems.
//!
//! Pure-Rust sonar processing for ultrasonic and acoustic sensors,
//! suitable for embedded navigation without external dependencies.

use std::f64::consts::PI;
use std::fmt;

// ── Sonar Reading ───────────────────────────────────────────────

/// A single sonar measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SonarReading {
    /// Measured range in meters.
    pub range_m: f64,
    /// Beam center azimuth in radians.
    pub azimuth_rad: f64,
    /// Measurement timestamp in seconds.
    pub timestamp_s: f64,
    /// Signal amplitude (0.0 to 1.0).
    pub amplitude: f64,
    /// Sensor identifier.
    pub sensor_id: usize,
}

impl SonarReading {
    pub fn new(range_m: f64, azimuth_rad: f64, timestamp_s: f64) -> Self {
        Self {
            range_m,
            azimuth_rad,
            timestamp_s,
            amplitude: 1.0,
            sensor_id: 0,
        }
    }

    pub fn with_amplitude(mut self, amp: f64) -> Self {
        self.amplitude = amp.clamp(0.0, 1.0);
        self
    }

    pub fn with_sensor_id(mut self, id: usize) -> Self {
        self.sensor_id = id;
        self
    }

    /// Point in 2D Cartesian coordinates (x, y) from polar measurement.
    pub fn to_cartesian(&self) -> (f64, f64) {
        (
            self.range_m * self.azimuth_rad.cos(),
            self.range_m * self.azimuth_rad.sin(),
        )
    }
}

impl fmt::Display for SonarReading {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sonar(r={:.2}m, az={:.1}deg, amp={:.2})",
            self.range_m,
            self.azimuth_rad.to_degrees(),
            self.amplitude,
        )
    }
}

// ── Time-of-Flight Measurement ──────────────────────────────────

/// Configuration for time-of-flight sonar ranging.
#[derive(Debug, Clone)]
pub struct TofConfig {
    /// Speed of sound in m/s.
    pub speed_of_sound: f64,
    /// Minimum detectable range in meters.
    pub min_range: f64,
    /// Maximum detectable range in meters.
    pub max_range: f64,
    /// Blanking time in seconds (dead zone after transmit).
    pub blanking_time_s: f64,
    /// Temperature in Celsius for speed-of-sound correction.
    pub temperature_c: f64,
}

impl TofConfig {
    pub fn new() -> Self {
        Self {
            speed_of_sound: 343.0,
            min_range: 0.02,
            max_range: 4.0,
            blanking_time_s: 0.0001,
            temperature_c: 20.0,
        }
    }

    pub fn with_speed_of_sound(mut self, v: f64) -> Self {
        self.speed_of_sound = v;
        self
    }

    pub fn with_range_limits(mut self, min: f64, max: f64) -> Self {
        self.min_range = min;
        self.max_range = max;
        self
    }

    pub fn with_temperature(mut self, temp_c: f64) -> Self {
        self.temperature_c = temp_c;
        // Update speed of sound: v = 331.3 + 0.606 * T
        self.speed_of_sound = 331.3 + 0.606 * temp_c;
        self
    }

    pub fn with_blanking_time(mut self, time_s: f64) -> Self {
        self.blanking_time_s = time_s;
        self
    }

    /// Convert a round-trip echo time to distance.
    pub fn echo_time_to_range(&self, echo_time_s: f64) -> Option<f64> {
        if echo_time_s < self.blanking_time_s {
            return None;
        }
        let range = self.speed_of_sound * echo_time_s / 2.0;
        if range < self.min_range || range > self.max_range {
            None
        } else {
            Some(range)
        }
    }

    /// Convert range to expected echo time.
    pub fn range_to_echo_time(&self, range_m: f64) -> f64 {
        2.0 * range_m / self.speed_of_sound
    }

    /// Minimum dead zone range based on blanking time.
    pub fn dead_zone(&self) -> f64 {
        self.speed_of_sound * self.blanking_time_s / 2.0
    }
}

impl fmt::Display for TofConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ToF(v={:.1}m/s, range=[{:.2}..{:.2}]m, T={:.1}C)",
            self.speed_of_sound, self.min_range, self.max_range, self.temperature_c,
        )
    }
}

// ── Beam Pattern Model ──────────────────────────────────────────

/// Ultrasonic beam pattern model using a simple cone approximation.
#[derive(Debug, Clone)]
pub struct BeamPattern {
    /// Half-angle of the main lobe in radians.
    pub half_angle_rad: f64,
    /// Transducer diameter in meters.
    pub diameter_m: f64,
    /// Operating frequency in Hz.
    pub frequency_hz: f64,
    /// Speed of sound in medium.
    pub speed_of_sound: f64,
}

impl BeamPattern {
    pub fn new(frequency_hz: f64, diameter_m: f64) -> Self {
        let speed = 343.0;
        let wavelength = speed / frequency_hz;
        // First null of circular piston: sin(theta) = 1.22 * lambda / D
        let sin_theta = (1.22 * wavelength / diameter_m).min(1.0);
        let half_angle = sin_theta.asin();
        Self {
            half_angle_rad: half_angle,
            diameter_m,
            frequency_hz,
            speed_of_sound: speed,
        }
    }

    pub fn with_speed_of_sound(mut self, v: f64) -> Self {
        self.speed_of_sound = v;
        let wavelength = v / self.frequency_hz;
        let sin_theta = (1.22 * wavelength / self.diameter_m).min(1.0);
        self.half_angle_rad = sin_theta.asin();
        self
    }

    /// Beam directivity pattern at angle theta from boresight.
    /// Returns gain relative to on-axis (0 to 1).
    pub fn gain_at_angle(&self, theta_rad: f64) -> f64 {
        let theta_abs = theta_rad.abs();
        if theta_abs < 1e-12 {
            return 1.0;
        }
        if theta_abs >= PI / 2.0 {
            return 0.0;
        }
        let wavelength = self.speed_of_sound / self.frequency_hz;
        let ka_sin = PI * self.diameter_m * theta_abs.sin() / wavelength;
        if ka_sin.abs() < 1e-12 {
            return 1.0;
        }
        // Jinc approximation: 2*J1(x)/x using sinc-like approx
        // For a circular piston: D(theta) = [2*J1(ka*sin(theta)) / (ka*sin(theta))]^2
        // Approximation using sinc: good for narrow beams
        let sinc_val = ka_sin.sin() / ka_sin;
        (sinc_val * sinc_val).max(0.0)
    }

    /// Beam width at -3dB in radians.
    pub fn beam_width_3db(&self) -> f64 {
        self.half_angle_rad * 0.7 // Approximate -3dB point
    }

    /// Footprint radius at a given range.
    pub fn footprint_radius(&self, range_m: f64) -> f64 {
        range_m * self.half_angle_rad.tan()
    }

    /// Check if an angle is within the main lobe.
    pub fn in_main_lobe(&self, theta_rad: f64) -> bool {
        theta_rad.abs() < self.half_angle_rad
    }
}

impl fmt::Display for BeamPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Beam(freq={:.0}Hz, diam={:.3}m, half_angle={:.1}deg)",
            self.frequency_hz, self.diameter_m, self.half_angle_rad.to_degrees(),
        )
    }
}

// ── Multi-Sensor Ring Fusion ────────────────────────────────────

/// Configuration for a ring of sonar sensors.
#[derive(Debug, Clone)]
pub struct SonarRing {
    pub sensor_azimuths: Vec<f64>,
    pub beam_half_angle: f64,
    pub max_range: f64,
}

impl SonarRing {
    /// Create a uniformly-spaced ring of sensors.
    pub fn uniform(num_sensors: usize, max_range: f64, beam_half_angle: f64) -> Self {
        let mut azimuths = Vec::with_capacity(num_sensors);
        for i in 0..num_sensors {
            azimuths.push(2.0 * PI * i as f64 / num_sensors as f64);
        }
        Self { sensor_azimuths: azimuths, beam_half_angle, max_range }
    }

    pub fn with_custom_azimuths(mut self, azimuths: Vec<f64>) -> Self {
        self.sensor_azimuths = azimuths;
        self
    }

    pub fn num_sensors(&self) -> usize {
        self.sensor_azimuths.len()
    }

    /// Fuse a set of readings from the ring into a set of obstacle points.
    /// Returns (x, y) obstacle positions in the robot frame.
    pub fn fuse_readings(&self, readings: &[SonarReading]) -> Vec<(f64, f64)> {
        let mut points = Vec::new();
        for reading in readings {
            if reading.range_m <= 0.0 || reading.range_m > self.max_range {
                continue;
            }
            // Generate arc of points within the beam width
            let num_arc_points = 5;
            let start_angle = reading.azimuth_rad - self.beam_half_angle;
            let end_angle = reading.azimuth_rad + self.beam_half_angle;
            let step = (end_angle - start_angle) / (num_arc_points - 1).max(1) as f64;
            for j in 0..num_arc_points {
                let angle = start_angle + j as f64 * step;
                let x = reading.range_m * angle.cos();
                let y = reading.range_m * angle.sin();
                points.push((x, y));
            }
        }
        points
    }

    /// Find the nearest obstacle distance and direction from a set of readings.
    pub fn nearest_obstacle(&self, readings: &[SonarReading]) -> Option<(f64, f64)> {
        let mut nearest_range = f64::MAX;
        let mut nearest_azimuth = 0.0;
        for reading in readings {
            if reading.range_m > 0.0 && reading.range_m < nearest_range
                && reading.range_m <= self.max_range
            {
                nearest_range = reading.range_m;
                nearest_azimuth = reading.azimuth_rad;
            }
        }
        if nearest_range < f64::MAX {
            Some((nearest_range, nearest_azimuth))
        } else {
            None
        }
    }
}

impl fmt::Display for SonarRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SonarRing(sensors={}, max_range={:.1}m, beam={:.1}deg)",
            self.sensor_azimuths.len(),
            self.max_range,
            self.beam_half_angle.to_degrees(),
        )
    }
}

// ── Occupancy Grid ──────────────────────────────────────────────

/// 2D occupancy grid for sonar-based environment mapping.
#[derive(Debug, Clone)]
pub struct OccupancyGrid {
    pub cells: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub origin_x: f64,
    pub origin_y: f64,
}

impl OccupancyGrid {
    /// Create a grid centered at (origin_x, origin_y) with given resolution (m/cell).
    pub fn new(width: usize, height: usize, resolution: f64) -> Self {
        Self {
            cells: vec![0.5; width * height], // 0.5 = unknown
            width,
            height,
            resolution,
            origin_x: -(width as f64 * resolution / 2.0),
            origin_y: -(height as f64 * resolution / 2.0),
        }
    }

    pub fn with_origin(mut self, x: f64, y: f64) -> Self {
        self.origin_x = x;
        self.origin_y = y;
        self
    }

    /// World coordinates to cell indices.
    pub fn world_to_cell(&self, wx: f64, wy: f64) -> Option<(usize, usize)> {
        let col = ((wx - self.origin_x) / self.resolution).floor();
        let row = ((wy - self.origin_y) / self.resolution).floor();
        if col >= 0.0 && row >= 0.0 {
            let c = col as usize;
            let r = row as usize;
            if c < self.width && r < self.height {
                return Some((r, c));
            }
        }
        None
    }

    /// Cell indices to world center coordinates.
    pub fn cell_to_world(&self, row: usize, col: usize) -> (f64, f64) {
        let wx = self.origin_x + (col as f64 + 0.5) * self.resolution;
        let wy = self.origin_y + (row as f64 + 0.5) * self.resolution;
        (wx, wy)
    }

    /// Get occupancy probability at a cell.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.height && col < self.width {
            self.cells[row * self.width + col]
        } else {
            0.5
        }
    }

    /// Set occupancy probability at a cell.
    pub fn set(&mut self, row: usize, col: usize, prob: f64) {
        if row < self.height && col < self.width {
            self.cells[row * self.width + col] = prob.clamp(0.001, 0.999);
        }
    }

    /// Update grid with a sonar reading using an inverse sensor model.
    /// `robot_x`, `robot_y` are the robot's world coordinates.
    pub fn update_with_reading(
        &mut self,
        robot_x: f64,
        robot_y: f64,
        reading: &SonarReading,
        beam_half_angle: f64,
    ) {
        let range = reading.range_m;
        let azimuth = reading.azimuth_rad;
        let hit_x = robot_x + range * azimuth.cos();
        let hit_y = robot_y + range * azimuth.sin();

        // Bresenham-like ray traversal
        let step_size = self.resolution * 0.5;
        let num_steps = (range / step_size).ceil() as usize;

        for s in 0..=num_steps {
            let frac = s as f64 / num_steps.max(1) as f64;
            let px = robot_x + frac * (hit_x - robot_x);
            let py = robot_y + frac * (hit_y - robot_y);

            // Check if this point is within the beam cone
            let dx = px - robot_x;
            let dy = py - robot_y;
            let point_range = (dx * dx + dy * dy).sqrt();
            let point_angle = dy.atan2(dx);
            let mut angle_diff = (point_angle - azimuth).abs();
            if angle_diff > PI {
                angle_diff = 2.0 * PI - angle_diff;
            }
            if angle_diff > beam_half_angle {
                continue;
            }

            if let Some((r, c)) = self.world_to_cell(px, py) {
                let current = self.get(r, c);
                // Log-odds update
                let log_prior = (current / (1.0 - current)).ln();
                let measurement_log_odds = if frac < 0.95 {
                    // Free space
                    -0.4
                } else {
                    // Occupied
                    0.85
                };
                let log_posterior = log_prior + measurement_log_odds;
                let new_prob = 1.0 / (1.0 + (-log_posterior).exp());
                self.set(r, c, new_prob);
            }
        }
    }

    /// Count cells classified as occupied (probability > threshold).
    pub fn occupied_count(&self, threshold: f64) -> usize {
        self.cells.iter().filter(|&&p| p > threshold).count()
    }

    /// Count cells classified as free (probability < threshold).
    pub fn free_count(&self, threshold: f64) -> usize {
        self.cells.iter().filter(|&&p| p < threshold).count()
    }
}

impl fmt::Display for OccupancyGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OccGrid({}x{}, res={:.3}m, occ={}, free={})",
            self.width,
            self.height,
            self.resolution,
            self.occupied_count(0.7),
            self.free_count(0.3),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_sonar_reading_cartesian() {
        let r = SonarReading::new(1.0, 0.0, 0.0);
        let (x, y) = r.to_cartesian();
        assert!((x - 1.0).abs() < 1e-12);
        assert!(y.abs() < 1e-12);
    }

    #[test]
    fn test_sonar_reading_45deg() {
        let r = SonarReading::new(2.0_f64.sqrt(), PI / 4.0, 0.0);
        let (x, y) = r.to_cartesian();
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sonar_reading_display() {
        let r = SonarReading::new(1.5, 0.0, 0.0);
        let s = format!("{r}");
        assert!(s.contains("Sonar"));
    }

    #[test]
    fn test_tof_echo_to_range() {
        let config = TofConfig::new();
        // 343 m/s, echo at 0.01s => range = 343*0.01/2 = 1.715m
        let range = config.echo_time_to_range(0.01).unwrap();
        assert!((range - 1.715).abs() < 0.01);
    }

    #[test]
    fn test_tof_range_to_echo() {
        let config = TofConfig::new();
        let echo = config.range_to_echo_time(1.715);
        assert!((echo - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_tof_blanking() {
        let config = TofConfig::new().with_blanking_time(0.001);
        assert!(config.echo_time_to_range(0.0005).is_none());
    }

    #[test]
    fn test_tof_temperature() {
        let config = TofConfig::new().with_temperature(0.0);
        assert!((config.speed_of_sound - 331.3).abs() < 0.1);
    }

    #[test]
    fn test_tof_display() {
        let config = TofConfig::new();
        let s = format!("{config}");
        assert!(s.contains("ToF"));
    }

    #[test]
    fn test_beam_on_axis() {
        let beam = BeamPattern::new(40000.0, 0.016);
        let g = beam.gain_at_angle(0.0);
        assert!((g - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_beam_off_axis() {
        let beam = BeamPattern::new(40000.0, 0.016);
        let g90 = beam.gain_at_angle(PI / 2.0);
        assert!(g90 < 0.01);
    }

    #[test]
    fn test_beam_footprint() {
        let beam = BeamPattern::new(40000.0, 0.016);
        let r = beam.footprint_radius(1.0);
        assert!(r > 0.0);
        assert!(r < 1.0);
    }

    #[test]
    fn test_beam_display() {
        let beam = BeamPattern::new(40000.0, 0.016);
        let s = format!("{beam}");
        assert!(s.contains("Beam"));
    }

    #[test]
    fn test_sonar_ring_uniform() {
        let ring = SonarRing::uniform(8, 4.0, 15.0_f64.to_radians());
        assert_eq!(ring.num_sensors(), 8);
    }

    #[test]
    fn test_sonar_ring_fuse() {
        let ring = SonarRing::uniform(4, 4.0, 0.2);
        let readings = vec![
            SonarReading::new(1.0, 0.0, 0.0).with_sensor_id(0),
            SonarReading::new(2.0, PI / 2.0, 0.0).with_sensor_id(1),
        ];
        let points = ring.fuse_readings(&readings);
        assert!(!points.is_empty());
    }

    #[test]
    fn test_sonar_ring_nearest() {
        let ring = SonarRing::uniform(4, 4.0, 0.2);
        let readings = vec![
            SonarReading::new(3.0, 0.0, 0.0),
            SonarReading::new(1.5, PI, 0.0),
        ];
        let (range, _) = ring.nearest_obstacle(&readings).unwrap();
        assert!((range - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_sonar_ring_display() {
        let ring = SonarRing::uniform(8, 4.0, 0.26);
        let s = format!("{ring}");
        assert!(s.contains("SonarRing"));
    }

    #[test]
    fn test_occupancy_grid_init() {
        let grid = OccupancyGrid::new(100, 100, 0.05);
        // All unknown
        assert!((grid.get(50, 50) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_occupancy_world_cell_roundtrip() {
        let grid = OccupancyGrid::new(100, 100, 0.1);
        let (wx, wy) = grid.cell_to_world(50, 50);
        let (r, c) = grid.world_to_cell(wx, wy).unwrap();
        assert_eq!(r, 50);
        assert_eq!(c, 50);
    }

    #[test]
    fn test_occupancy_update() {
        let mut grid = OccupancyGrid::new(100, 100, 0.1);
        let reading = SonarReading::new(2.0, 0.0, 0.0);
        grid.update_with_reading(0.0, 0.0, &reading, 0.2);
        // Some cells should no longer be 0.5
        let changed = grid.cells.iter().filter(|&&p| (p - 0.5).abs() > 0.01).count();
        assert!(changed > 0);
    }

    #[test]
    fn test_occupancy_display() {
        let grid = OccupancyGrid::new(50, 50, 0.1);
        let s = format!("{grid}");
        assert!(s.contains("OccGrid"));
    }
}
