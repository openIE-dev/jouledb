//! Rotary encoder processing — quadrature decoding with edge detection,
//! velocity estimation via differentiation and low-pass filtering, absolute
//! position tracking with wraparound, and index pulse synchronization.
//!
//! Pure-Rust encoder signal processing for motor control and odometry,
//! suitable for embedded real-time systems without external dependencies.

use std::fmt;

// ── Quadrature State ────────────────────────────────────────────

/// Quadrature channel levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadState {
    /// A=0, B=0
    S00,
    /// A=0, B=1
    S01,
    /// A=1, B=0
    S10,
    /// A=1, B=1
    S11,
}

impl QuadState {
    pub fn from_channels(a: bool, b: bool) -> Self {
        match (a, b) {
            (false, false) => QuadState::S00,
            (false, true) => QuadState::S01,
            (true, false) => QuadState::S10,
            (true, true) => QuadState::S11,
        }
    }

    pub fn channel_a(&self) -> bool {
        matches!(self, QuadState::S10 | QuadState::S11)
    }

    pub fn channel_b(&self) -> bool {
        matches!(self, QuadState::S01 | QuadState::S11)
    }

    /// Gray code value for this state (used for transition validation).
    fn gray_code(&self) -> u8 {
        match self {
            QuadState::S00 => 0,
            QuadState::S01 => 1,
            QuadState::S11 => 2,
            QuadState::S10 => 3,
        }
    }
}

impl fmt::Display for QuadState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuadState::S00 => write!(f, "00"),
            QuadState::S01 => write!(f, "01"),
            QuadState::S10 => write!(f, "10"),
            QuadState::S11 => write!(f, "11"),
        }
    }
}

// ── Quadrature Decoder ──────────────────────────────────────────

/// Counting mode for the quadrature decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountMode {
    /// Count on all edges (4x resolution).
    X4,
    /// Count on A edges only (2x resolution).
    X2,
    /// Count on A rising edge only (1x resolution).
    X1,
}

/// Quadrature decoder with direction detection and error checking.
#[derive(Debug, Clone)]
pub struct QuadDecoder {
    pub count: i64,
    pub mode: CountMode,
    pub errors: usize,
    state: QuadState,
    initialized: bool,
}

impl QuadDecoder {
    pub fn new(mode: CountMode) -> Self {
        Self {
            count: 0,
            mode,
            errors: 0,
            state: QuadState::S00,
            initialized: false,
        }
    }

    pub fn with_mode(mut self, mode: CountMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_initial_count(mut self, count: i64) -> Self {
        self.count = count;
        self
    }

    /// Process a new sample of channel A and B.
    /// Returns the direction: +1 forward, -1 backward, 0 no change.
    pub fn update(&mut self, a: bool, b: bool) -> i32 {
        let new_state = QuadState::from_channels(a, b);
        if !self.initialized {
            self.state = new_state;
            self.initialized = true;
            return 0;
        }
        if new_state == self.state {
            return 0;
        }

        // Validate transition (only adjacent Gray code states are valid)
        let old_gray = self.state.gray_code();
        let new_gray = new_state.gray_code();
        let diff = (new_gray as i8 - old_gray as i8 + 4) % 4;

        let direction = match diff {
            1 => 1i32,   // Forward
            3 => -1i32,  // Backward (equivalent to -1 in mod 4)
            _ => {
                // Invalid transition (skipped state = noise/error)
                self.errors += 1;
                self.state = new_state;
                return 0;
            }
        };

        let should_count = match self.mode {
            CountMode::X4 => true,
            CountMode::X2 => {
                // Count on A channel edges
                self.state.channel_a() != new_state.channel_a()
            }
            CountMode::X1 => {
                // Count on A rising edge only
                !self.state.channel_a() && new_state.channel_a()
            }
        };

        if should_count {
            self.count += direction as i64;
        }
        self.state = new_state;
        direction
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.errors = 0;
        self.initialized = false;
    }
}

impl fmt::Display for QuadDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuadDec(count={}, mode={:?}, errors={})",
            self.count, self.mode, self.errors
        )
    }
}

// ── Encoder Configuration ───────────────────────────────────────

/// Encoder physical parameters.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Counts per revolution (before quadrature multiplication).
    pub cpr: u32,
    /// Wheel or shaft diameter in meters (for linear distance).
    pub wheel_diameter_m: f64,
    /// Gear ratio (output turns / encoder turns).
    pub gear_ratio: f64,
    /// Has index (Z) channel.
    pub has_index: bool,
}

impl EncoderConfig {
    pub fn new(cpr: u32) -> Self {
        Self {
            cpr,
            wheel_diameter_m: 0.0,
            gear_ratio: 1.0,
            has_index: false,
        }
    }

    pub fn with_wheel_diameter(mut self, diameter_m: f64) -> Self {
        self.wheel_diameter_m = diameter_m;
        self
    }

    pub fn with_gear_ratio(mut self, ratio: f64) -> Self {
        self.gear_ratio = ratio;
        self
    }

    pub fn with_index(mut self, has_index: bool) -> Self {
        self.has_index = has_index;
        self
    }

    /// Counts per revolution accounting for quadrature mode.
    pub fn effective_cpr(&self, mode: CountMode) -> u32 {
        match mode {
            CountMode::X4 => self.cpr * 4,
            CountMode::X2 => self.cpr * 2,
            CountMode::X1 => self.cpr,
        }
    }

    /// Convert encoder counts to radians.
    pub fn counts_to_radians(&self, counts: i64, mode: CountMode) -> f64 {
        let ecpr = self.effective_cpr(mode) as f64;
        (counts as f64 / ecpr) * 2.0 * std::f64::consts::PI / self.gear_ratio
    }

    /// Convert encoder counts to linear distance (requires wheel diameter).
    pub fn counts_to_distance(&self, counts: i64, mode: CountMode) -> f64 {
        let revolutions = counts as f64 / self.effective_cpr(mode) as f64 / self.gear_ratio;
        revolutions * std::f64::consts::PI * self.wheel_diameter_m
    }
}

impl fmt::Display for EncoderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Encoder(cpr={}, gear={:.2}, wheel_d={:.3}m)",
            self.cpr, self.gear_ratio, self.wheel_diameter_m
        )
    }
}

// ── Velocity Estimator ──────────────────────────────────────────

/// Velocity estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityMethod {
    /// Finite difference: delta_count / delta_time.
    FiniteDifference,
    /// Period measurement: time between edges.
    PeriodMeasurement,
}

/// Encoder velocity estimator with low-pass filtering.
#[derive(Debug, Clone)]
pub struct VelocityEstimator {
    pub method: VelocityMethod,
    pub filter_alpha: f64,
    pub velocity_cps: f64,
    last_count: i64,
    last_time: f64,
    last_edge_time: f64,
    initialized: bool,
}

impl VelocityEstimator {
    pub fn new(method: VelocityMethod) -> Self {
        Self {
            method,
            filter_alpha: 0.2,
            velocity_cps: 0.0,
            last_count: 0,
            last_time: 0.0,
            last_edge_time: 0.0,
            initialized: false,
        }
    }

    pub fn with_filter_alpha(mut self, alpha: f64) -> Self {
        self.filter_alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Update with current count and timestamp. Returns velocity in counts/second.
    pub fn update(&mut self, count: i64, timestamp_s: f64) -> f64 {
        if !self.initialized {
            self.last_count = count;
            self.last_time = timestamp_s;
            self.last_edge_time = timestamp_s;
            self.initialized = true;
            return 0.0;
        }

        let raw_velocity = match self.method {
            VelocityMethod::FiniteDifference => {
                let dt = timestamp_s - self.last_time;
                if dt > 0.0 {
                    let dc = count - self.last_count;
                    dc as f64 / dt
                } else {
                    self.velocity_cps
                }
            }
            VelocityMethod::PeriodMeasurement => {
                if count != self.last_count {
                    let dt = timestamp_s - self.last_edge_time;
                    self.last_edge_time = timestamp_s;
                    if dt > 0.0 {
                        let direction = if count > self.last_count { 1.0 } else { -1.0 };
                        direction / dt
                    } else {
                        self.velocity_cps
                    }
                } else {
                    // No edge — decay toward zero for timeout detection
                    let dt = timestamp_s - self.last_edge_time;
                    if dt > 0.1 {
                        0.0
                    } else {
                        self.velocity_cps
                    }
                }
            }
        };

        // Low-pass filter
        self.velocity_cps = self.filter_alpha * raw_velocity
            + (1.0 - self.filter_alpha) * self.velocity_cps;
        self.last_count = count;
        self.last_time = timestamp_s;
        self.velocity_cps
    }

    /// Convert velocity from counts/s to rad/s using config.
    pub fn velocity_rad_s(&self, config: &EncoderConfig, mode: CountMode) -> f64 {
        let ecpr = config.effective_cpr(mode) as f64;
        (self.velocity_cps / ecpr) * 2.0 * std::f64::consts::PI / config.gear_ratio
    }

    /// Convert velocity from counts/s to linear m/s using config.
    pub fn velocity_m_s(&self, config: &EncoderConfig, mode: CountMode) -> f64 {
        let ecpr = config.effective_cpr(mode) as f64;
        let rps = self.velocity_cps / ecpr / config.gear_ratio;
        rps * std::f64::consts::PI * config.wheel_diameter_m
    }

    pub fn reset(&mut self) {
        self.velocity_cps = 0.0;
        self.initialized = false;
    }
}

impl fmt::Display for VelocityEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VelEst(method={:?}, vel={:.1} cps, alpha={:.2})",
            self.method, self.velocity_cps, self.filter_alpha
        )
    }
}

// ── Position Tracker ────────────────────────────────────────────

/// Absolute position tracker with index pulse synchronization.
#[derive(Debug, Clone)]
pub struct PositionTracker {
    pub position_rad: f64,
    pub total_revolutions: i64,
    pub index_count: usize,
    pub index_offset: f64,
    config: EncoderConfig,
    mode: CountMode,
    last_count: i64,
    initialized: bool,
}

impl PositionTracker {
    pub fn new(config: EncoderConfig, mode: CountMode) -> Self {
        Self {
            position_rad: 0.0,
            total_revolutions: 0,
            index_count: 0,
            index_offset: 0.0,
            config,
            mode,
            last_count: 0,
            initialized: false,
        }
    }

    pub fn with_initial_position(mut self, rad: f64) -> Self {
        self.position_rad = rad;
        self
    }

    /// Update position from encoder count.
    pub fn update(&mut self, count: i64) {
        let rad = self.config.counts_to_radians(count, self.mode);
        let two_pi = 2.0 * std::f64::consts::PI;

        if self.initialized {
            let delta = count - self.last_count;
            let delta_rad = self.config.counts_to_radians(delta, self.mode);
            self.position_rad += delta_rad;

            // Track full revolutions
            while self.position_rad >= two_pi {
                self.position_rad -= two_pi;
                self.total_revolutions += 1;
            }
            while self.position_rad < 0.0 {
                self.position_rad += two_pi;
                self.total_revolutions -= 1;
            }
        } else {
            self.position_rad = rad % two_pi;
            if self.position_rad < 0.0 {
                self.position_rad += two_pi;
            }
            self.initialized = true;
        }
        self.last_count = count;
    }

    /// Process an index pulse event. Corrects position to the known index angle.
    pub fn on_index_pulse(&mut self) {
        self.index_count += 1;
        // Compute position error and apply correction
        let expected = self.index_offset;
        let error = expected - self.position_rad;
        // Only correct if error is small (not a full revolution off)
        if error.abs() < std::f64::consts::PI / 4.0 {
            self.position_rad = expected;
        }
    }

    /// Set the known angle of the index pulse.
    pub fn set_index_offset(&mut self, offset_rad: f64) {
        self.index_offset = offset_rad;
    }

    /// Total absolute position including full revolutions.
    pub fn absolute_position_rad(&self) -> f64 {
        self.total_revolutions as f64 * 2.0 * std::f64::consts::PI + self.position_rad
    }

    /// Position in degrees (0-360).
    pub fn position_deg(&self) -> f64 {
        self.position_rad.to_degrees()
    }

    /// Linear distance traveled (requires wheel diameter in config).
    pub fn linear_distance(&self) -> f64 {
        let total_rad = self.absolute_position_rad();
        let revs = total_rad / (2.0 * std::f64::consts::PI);
        revs * std::f64::consts::PI * self.config.wheel_diameter_m
    }

    pub fn reset(&mut self) {
        self.position_rad = 0.0;
        self.total_revolutions = 0;
        self.index_count = 0;
        self.initialized = false;
        self.last_count = 0;
    }
}

impl fmt::Display for PositionTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PosTrack(pos={:.2}deg, revs={}, idx={})",
            self.position_deg(),
            self.total_revolutions,
            self.index_count,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_quad_state_from_channels() {
        assert_eq!(QuadState::from_channels(false, false), QuadState::S00);
        assert_eq!(QuadState::from_channels(true, true), QuadState::S11);
    }

    #[test]
    fn test_quad_state_channels() {
        assert!(!QuadState::S00.channel_a());
        assert!(QuadState::S10.channel_a());
        assert!(!QuadState::S10.channel_b());
        assert!(QuadState::S01.channel_b());
    }

    #[test]
    fn test_quad_state_display() {
        assert_eq!(format!("{}", QuadState::S00), "00");
        assert_eq!(format!("{}", QuadState::S11), "11");
    }

    #[test]
    fn test_decoder_x4_forward() {
        let mut dec = QuadDecoder::new(CountMode::X4);
        // Quadrature sequence: 00 -> 10 -> 11 -> 01 -> 00
        // Gray codes: 0 -> 3 -> 2 -> 1 -> 0, diffs are 3 (=-1 mod 4) each
        let sequence = [(false, false), (true, false), (true, true), (false, true), (false, false)];
        for &(a, b) in &sequence {
            dec.update(a, b);
        }
        assert_eq!(dec.count, -4);
    }

    #[test]
    fn test_decoder_x4_backward() {
        let mut dec = QuadDecoder::new(CountMode::X4);
        // Reverse sequence: 00 -> 01 -> 11 -> 10 -> 00
        // Gray codes: 0 -> 1 -> 2 -> 3 -> 0, diffs are 1 (=+1 mod 4) each
        let sequence = [(false, false), (false, true), (true, true), (true, false), (false, false)];
        for &(a, b) in &sequence {
            dec.update(a, b);
        }
        assert_eq!(dec.count, 4);
    }

    #[test]
    fn test_decoder_x1() {
        let mut dec = QuadDecoder::new(CountMode::X1);
        // Sequence: 00 -> 10 -> 11 -> 01 -> 00
        // Gray code direction is -1, X1 counts on A rising edge (00->10)
        let sequence = [(false, false), (true, false), (true, true), (false, true), (false, false)];
        for &(a, b) in &sequence {
            dec.update(a, b);
        }
        assert_eq!(dec.count, -1);
    }

    #[test]
    fn test_decoder_error_detection() {
        let mut dec = QuadDecoder::new(CountMode::X4);
        dec.update(false, false);
        // Skip a state: 00 -> 11 (invalid)
        dec.update(true, true);
        assert_eq!(dec.errors, 1);
    }

    #[test]
    fn test_decoder_display() {
        let dec = QuadDecoder::new(CountMode::X4);
        let s = format!("{dec}");
        assert!(s.contains("QuadDec"));
    }

    #[test]
    fn test_encoder_config_cpr() {
        let config = EncoderConfig::new(1000);
        assert_eq!(config.effective_cpr(CountMode::X4), 4000);
        assert_eq!(config.effective_cpr(CountMode::X2), 2000);
        assert_eq!(config.effective_cpr(CountMode::X1), 1000);
    }

    #[test]
    fn test_counts_to_radians() {
        let config = EncoderConfig::new(1000);
        let rad = config.counts_to_radians(4000, CountMode::X4);
        assert!((rad - 2.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn test_counts_to_distance() {
        let config = EncoderConfig::new(1000).with_wheel_diameter(0.1);
        let dist = config.counts_to_distance(4000, CountMode::X4);
        // 1 revolution * pi * 0.1 = 0.31416
        assert!((dist - PI * 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_encoder_config_display() {
        let config = EncoderConfig::new(2048).with_gear_ratio(3.0);
        let s = format!("{config}");
        assert!(s.contains("Encoder"));
        assert!(s.contains("2048"));
    }

    #[test]
    fn test_velocity_finite_diff() {
        let mut est = VelocityEstimator::new(VelocityMethod::FiniteDifference)
            .with_filter_alpha(1.0);
        est.update(0, 0.0);
        let vel = est.update(100, 0.1);
        assert!((vel - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_velocity_period() {
        let mut est = VelocityEstimator::new(VelocityMethod::PeriodMeasurement)
            .with_filter_alpha(1.0);
        est.update(0, 0.0);
        let vel = est.update(1, 0.01);
        assert!((vel - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_velocity_filtering() {
        let mut est = VelocityEstimator::new(VelocityMethod::FiniteDifference)
            .with_filter_alpha(0.5);
        est.update(0, 0.0);
        est.update(100, 0.1); // 1000 cps raw
        // Filtered = 0.5 * 1000 + 0.5 * 0 = 500
        assert!((est.velocity_cps - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_velocity_rad_s() {
        let config = EncoderConfig::new(1000);
        let mut est = VelocityEstimator::new(VelocityMethod::FiniteDifference)
            .with_filter_alpha(1.0);
        est.update(0, 0.0);
        est.update(4000, 1.0); // 4000 counts/s in X4 mode
        let rad_s = est.velocity_rad_s(&config, CountMode::X4);
        assert!((rad_s - 2.0 * PI).abs() < 1e-6);
    }

    #[test]
    fn test_velocity_display() {
        let est = VelocityEstimator::new(VelocityMethod::FiniteDifference);
        let s = format!("{est}");
        assert!(s.contains("VelEst"));
    }

    #[test]
    fn test_position_tracker_basic() {
        let config = EncoderConfig::new(1000);
        let mut tracker = PositionTracker::new(config, CountMode::X4);
        tracker.update(0);
        tracker.update(2000);
        // 2000 / 4000 = 0.5 revolutions = PI radians
        assert!((tracker.position_rad - PI).abs() < 1e-6);
    }

    #[test]
    fn test_position_tracker_revolution() {
        let config = EncoderConfig::new(100);
        let mut tracker = PositionTracker::new(config, CountMode::X4);
        tracker.update(0);
        tracker.update(400); // Full revolution
        assert_eq!(tracker.total_revolutions, 1);
        assert!(tracker.position_rad < 0.01);
    }

    #[test]
    fn test_position_tracker_index_pulse() {
        let config = EncoderConfig::new(1000).with_index(true);
        let mut tracker = PositionTracker::new(config, CountMode::X4);
        tracker.set_index_offset(0.0);
        tracker.update(0);
        tracker.update(10); // Small position
        tracker.on_index_pulse();
        assert_eq!(tracker.index_count, 1);
        // Position corrected to index offset (0.0)
        assert!((tracker.position_rad - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_tracker_display() {
        let config = EncoderConfig::new(1000);
        let tracker = PositionTracker::new(config, CountMode::X4);
        let s = format!("{tracker}");
        assert!(s.contains("PosTrack"));
    }
}
