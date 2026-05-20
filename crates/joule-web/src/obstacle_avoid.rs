//! Obstacle Avoidance — Vector Field Histogram (VFH), nearness diagram (ND),
//! Bug1/Bug2 algorithms, and configurable safety zones.
//!
//! Reactive obstacle avoidance layers that consume range sensor data (LIDAR or
//! sonar sectors) and emit steering commands. These algorithms complement the
//! local planner by adding a fast reflexive safety layer.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Obstacle avoidance errors.
#[derive(Debug, Clone, PartialEq)]
pub enum AvoidError {
    /// No safe direction available.
    NoSafeDirection,
    /// Invalid configuration.
    InvalidConfig(String),
    /// Sensor data inconsistency.
    SensorError(String),
}

impl fmt::Display for AvoidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSafeDirection => write!(f, "no safe direction available"),
            Self::InvalidConfig(m) => write!(f, "invalid config: {m}"),
            Self::SensorError(m) => write!(f, "sensor error: {m}"),
        }
    }
}

impl std::error::Error for AvoidError {}

// ── Steering Command ────────────────────────────────────────────

/// Steering output from the avoidance layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteerCmd {
    /// Recommended heading (radians, robot-relative).
    pub heading: f64,
    /// Recommended speed scale [0, 1].
    pub speed_scale: f64,
    /// True if the robot should halt immediately.
    pub emergency_stop: bool,
}

impl SteerCmd {
    pub fn new(heading: f64, speed_scale: f64) -> Self {
        Self { heading, speed_scale: speed_scale.clamp(0.0, 1.0), emergency_stop: false }
    }

    pub fn stop() -> Self {
        Self { heading: 0.0, speed_scale: 0.0, emergency_stop: true }
    }
}

impl fmt::Display for SteerCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.emergency_stop {
            write!(f, "SteerCmd(STOP)")
        } else {
            write!(f, "SteerCmd(h={:.3}, spd={:.2})", self.heading, self.speed_scale)
        }
    }
}

// ── Safety Zone ─────────────────────────────────────────────────

/// Concentric safety zones around the robot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafetyZone {
    /// Critical radius — emergency stop if any reading falls within.
    pub critical: f64,
    /// Warning radius — reduce speed proportionally.
    pub warning: f64,
    /// Free radius — normal operation beyond this distance.
    pub free: f64,
}

impl SafetyZone {
    pub fn new(critical: f64, warning: f64, free: f64) -> Result<Self, AvoidError> {
        if critical <= 0.0 || warning <= critical || free <= warning {
            return Err(AvoidError::InvalidConfig(
                "zones must satisfy 0 < critical < warning < free".into(),
            ));
        }
        Ok(Self { critical, warning, free })
    }

    /// Compute speed scale based on minimum obstacle distance.
    pub fn speed_factor(&self, min_range: f64) -> f64 {
        if min_range <= self.critical {
            0.0
        } else if min_range >= self.free {
            1.0
        } else {
            (min_range - self.critical) / (self.free - self.critical)
        }
    }
}

impl fmt::Display for SafetyZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SafetyZone(crit={:.2}, warn={:.2}, free={:.2})",
            self.critical, self.warning, self.free
        )
    }
}

// ── VFH (Vector Field Histogram) ────────────────────────────────

/// Polar histogram bin for VFH.
#[derive(Debug, Clone, Copy)]
struct VfhBin {
    angle: f64,
    density: f64,
}

/// Vector Field Histogram obstacle avoidance.
///
/// Builds a polar obstacle density histogram from range readings, applies
/// threshold to identify candidate valleys (free directions), then selects
/// the valley closest to the desired heading.
#[derive(Debug, Clone)]
pub struct Vfh {
    num_sectors: usize,
    threshold: f64,
    safety: SafetyZone,
    max_range: f64,
    smoothing_window: usize,
}

impl Vfh {
    pub fn new(num_sectors: usize, safety: SafetyZone) -> Result<Self, AvoidError> {
        if num_sectors < 8 {
            return Err(AvoidError::InvalidConfig("need at least 8 sectors".into()));
        }
        Ok(Self {
            num_sectors,
            threshold: 0.5,
            safety,
            max_range: 10.0,
            smoothing_window: 3,
        })
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t.clamp(0.0, 1.0);
        self
    }

    pub fn with_max_range(mut self, r: f64) -> Self {
        self.max_range = r.max(0.1);
        self
    }

    pub fn with_smoothing(mut self, window: usize) -> Self {
        self.smoothing_window = window.max(1);
        self
    }

    /// Compute a steering command from range readings.
    ///
    /// `ranges` is a slice of `(angle_rad, distance)` pairs in robot frame.
    /// `desired_heading` is the target direction in robot frame.
    pub fn compute(
        &self,
        ranges: &[(f64, f64)],
        desired_heading: f64,
    ) -> Result<SteerCmd, AvoidError> {
        if ranges.is_empty() {
            return Err(AvoidError::SensorError("empty range data".into()));
        }

        // Check emergency stop.
        let min_range = ranges.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        if min_range <= self.safety.critical {
            return Ok(SteerCmd::stop());
        }

        // Build polar histogram.
        let sector_width = 2.0 * std::f64::consts::PI / self.num_sectors as f64;
        let mut histogram = vec![0.0f64; self.num_sectors];

        for &(angle, dist) in ranges {
            if dist > self.max_range {
                continue;
            }
            let norm_angle = Self::normalize_angle(angle);
            let idx = ((norm_angle + std::f64::consts::PI) / sector_width) as usize;
            let idx = idx.min(self.num_sectors - 1);
            // Density inversely proportional to distance squared.
            let density = ((self.max_range - dist) / self.max_range).powi(2);
            if density > histogram[idx] {
                histogram[idx] = density;
            }
        }

        // Smooth the histogram.
        let smoothed = self.smooth_histogram(&histogram);

        // Find candidate valleys (sectors below threshold).
        let valleys = self.find_valleys(&smoothed);
        if valleys.is_empty() {
            return Err(AvoidError::NoSafeDirection);
        }

        // Select the valley closest to desired heading.
        let best_sector = self.select_valley(&valleys, desired_heading, sector_width);
        let heading = -std::f64::consts::PI + (best_sector as f64 + 0.5) * sector_width;
        let speed = self.safety.speed_factor(min_range);

        Ok(SteerCmd::new(heading, speed))
    }

    fn smooth_histogram(&self, hist: &[f64]) -> Vec<f64> {
        let n = hist.len();
        let half = self.smoothing_window / 2;
        let mut smoothed = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            let mut count = 0;
            for j in 0..self.smoothing_window {
                let offset = j as isize - half as isize;
                let idx = ((i as isize + offset).rem_euclid(n as isize)) as usize;
                let weight = 1.0 - (offset.unsigned_abs() as f64 / (half as f64 + 1.0));
                sum += hist[idx] * weight;
                count += 1;
            }
            smoothed[i] = sum / count as f64;
        }
        smoothed
    }

    fn find_valleys(&self, smoothed: &[f64]) -> Vec<usize> {
        smoothed
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d < self.threshold)
            .map(|(i, _)| i)
            .collect()
    }

    fn select_valley(&self, valleys: &[usize], desired: f64, sector_width: f64) -> usize {
        let mut best = valleys[0];
        let mut best_diff = f64::INFINITY;
        for &v in valleys {
            let angle = -std::f64::consts::PI + (v as f64 + 0.5) * sector_width;
            let diff = Self::normalize_angle(angle - desired).abs();
            if diff < best_diff {
                best_diff = diff;
                best = v;
            }
        }
        best
    }

    fn normalize_angle(a: f64) -> f64 {
        let mut a = a % (2.0 * std::f64::consts::PI);
        if a > std::f64::consts::PI {
            a -= 2.0 * std::f64::consts::PI;
        } else if a < -std::f64::consts::PI {
            a += 2.0 * std::f64::consts::PI;
        }
        a
    }
}

impl fmt::Display for Vfh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VFH(sectors={}, thresh={:.2})", self.num_sectors, self.threshold)
    }
}

// ── Nearness Diagram (ND) ───────────────────────────────────────

/// Situation classification for the Nearness Diagram method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NdSituation {
    /// Free space on both sides — go towards goal.
    HighSafety,
    /// Obstacle on one side — follow the gap.
    LowSafety1,
    /// Obstacles on both sides — navigate a narrow passage.
    LowSafety2,
    /// Very close obstacle — emergency manoeuvre.
    HighDanger,
}

impl fmt::Display for NdSituation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HighSafety => write!(f, "HighSafety"),
            Self::LowSafety1 => write!(f, "LowSafety1"),
            Self::LowSafety2 => write!(f, "LowSafety2"),
            Self::HighDanger => write!(f, "HighDanger"),
        }
    }
}

/// Nearness Diagram obstacle avoidance.
#[derive(Debug, Clone)]
pub struct NearnessDiagram {
    safety: SafetyZone,
    max_range: f64,
    robot_width: f64,
}

impl NearnessDiagram {
    pub fn new(safety: SafetyZone, robot_width: f64) -> Self {
        Self { safety, max_range: 10.0, robot_width }
    }

    pub fn with_max_range(mut self, r: f64) -> Self {
        self.max_range = r.max(0.1);
        self
    }

    /// Classify the situation and produce a steering command.
    pub fn compute(
        &self,
        ranges: &[(f64, f64)],
        desired_heading: f64,
    ) -> Result<(NdSituation, SteerCmd), AvoidError> {
        if ranges.is_empty() {
            return Err(AvoidError::SensorError("empty range data".into()));
        }

        let min_range = ranges.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        if min_range <= self.safety.critical {
            return Ok((NdSituation::HighDanger, SteerCmd::stop()));
        }

        // Split into left and right hemispheres.
        let (left_min, right_min) = self.hemisphere_min(ranges);

        let situation = if left_min > self.safety.free && right_min > self.safety.free {
            NdSituation::HighSafety
        } else if left_min > self.safety.warning && right_min <= self.safety.warning {
            NdSituation::LowSafety1
        } else if left_min <= self.safety.warning && right_min > self.safety.warning {
            NdSituation::LowSafety1
        } else {
            NdSituation::LowSafety2
        };

        let heading = match situation {
            NdSituation::HighSafety => desired_heading,
            NdSituation::LowSafety1 => {
                if left_min > right_min {
                    desired_heading + 0.3
                } else {
                    desired_heading - 0.3
                }
            }
            NdSituation::LowSafety2 => {
                // Navigate narrow passage: steer toward the wider side.
                if left_min > right_min {
                    desired_heading + 0.15
                } else {
                    desired_heading - 0.15
                }
            }
            NdSituation::HighDanger => 0.0,
        };

        let speed = self.safety.speed_factor(min_range);
        Ok((situation, SteerCmd::new(heading, speed)))
    }

    fn hemisphere_min(&self, ranges: &[(f64, f64)]) -> (f64, f64) {
        let mut left_min = self.max_range;
        let mut right_min = self.max_range;
        for &(angle, dist) in ranges {
            let d = dist.min(self.max_range);
            if angle >= 0.0 {
                left_min = left_min.min(d);
            } else {
                right_min = right_min.min(d);
            }
        }
        (left_min, right_min)
    }
}

impl fmt::Display for NearnessDiagram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NearnessDiagram(width={:.2})", self.robot_width)
    }
}

// ── Bug Algorithms ──────────────────────────────────────────────

/// State of the Bug algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BugState {
    /// Moving directly toward the goal.
    MoveToGoal,
    /// Following an obstacle boundary.
    FollowBoundary,
    /// Goal has been reached.
    Reached,
}

impl fmt::Display for BugState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MoveToGoal => write!(f, "MoveToGoal"),
            Self::FollowBoundary => write!(f, "FollowBoundary"),
            Self::Reached => write!(f, "Reached"),
        }
    }
}

/// Bug2 obstacle avoidance algorithm.
///
/// Alternates between moving directly toward the goal and following obstacle
/// boundaries. The robot leaves the boundary when it crosses the start-goal
/// line and is closer to the goal.
#[derive(Debug, Clone)]
pub struct Bug2 {
    goal: (f64, f64),
    start: (f64, f64),
    state: BugState,
    hit_point: Option<(f64, f64)>,
    closest_on_boundary: f64,
    reach_threshold: f64,
    obstacle_threshold: f64,
}

impl Bug2 {
    pub fn new(start: (f64, f64), goal: (f64, f64)) -> Self {
        Self {
            goal,
            start,
            state: BugState::MoveToGoal,
            hit_point: None,
            closest_on_boundary: f64::INFINITY,
            reach_threshold: 0.5,
            obstacle_threshold: 1.0,
        }
    }

    pub fn with_reach_threshold(mut self, t: f64) -> Self {
        self.reach_threshold = t.max(0.01);
        self
    }

    pub fn with_obstacle_threshold(mut self, t: f64) -> Self {
        self.obstacle_threshold = t.max(0.01);
        self
    }

    pub fn state(&self) -> BugState {
        self.state
    }

    /// Step the Bug2 algorithm given the current position and front range reading.
    pub fn step(&mut self, pos: (f64, f64), front_range: f64) -> SteerCmd {
        let dist_to_goal = Self::dist(pos, self.goal);
        if dist_to_goal < self.reach_threshold {
            self.state = BugState::Reached;
            return SteerCmd::new(0.0, 0.0);
        }

        match self.state {
            BugState::MoveToGoal => {
                if front_range < self.obstacle_threshold {
                    self.state = BugState::FollowBoundary;
                    self.hit_point = Some(pos);
                    self.closest_on_boundary = dist_to_goal;
                    SteerCmd::new(std::f64::consts::FRAC_PI_2, 0.6)
                } else {
                    let heading = (self.goal.1 - pos.1).atan2(self.goal.0 - pos.0);
                    SteerCmd::new(heading, 1.0)
                }
            }
            BugState::FollowBoundary => {
                let d = dist_to_goal;
                if d < self.closest_on_boundary {
                    self.closest_on_boundary = d;
                }
                // Check if we crossed the M-line and are closer to goal.
                if self.on_m_line(pos) && d < Self::dist(self.hit_point.unwrap_or(pos), self.goal) {
                    self.state = BugState::MoveToGoal;
                    let heading = (self.goal.1 - pos.1).atan2(self.goal.0 - pos.0);
                    SteerCmd::new(heading, 1.0)
                } else {
                    // Follow boundary: turn right and keep obstacle on left.
                    SteerCmd::new(std::f64::consts::FRAC_PI_2, 0.6)
                }
            }
            BugState::Reached => SteerCmd::new(0.0, 0.0),
        }
    }

    /// Check if position is on the start-goal line (M-line).
    fn on_m_line(&self, pos: (f64, f64)) -> bool {
        let dx = self.goal.0 - self.start.0;
        let dy = self.goal.1 - self.start.1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            return true;
        }
        // Perpendicular distance to the M-line.
        let cross = ((pos.0 - self.start.0) * dy - (pos.1 - self.start.1) * dx).abs() / len;
        cross < 0.3
    }

    fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        (dx * dx + dy * dy).sqrt()
    }
}

impl fmt::Display for Bug2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bug2(state={}, goal=({:.1},{:.1}))",
            self.state, self.goal.0, self.goal.1
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_safety() -> SafetyZone {
        SafetyZone::new(0.3, 1.0, 3.0).unwrap()
    }

    #[test]
    fn test_safety_zone_creation() {
        let sz = SafetyZone::new(0.5, 1.0, 2.0);
        assert!(sz.is_ok());
    }

    #[test]
    fn test_safety_zone_invalid() {
        assert!(SafetyZone::new(2.0, 1.0, 3.0).is_err());
        assert!(SafetyZone::new(0.5, 0.5, 0.5).is_err());
    }

    #[test]
    fn test_speed_factor_critical() {
        let sz = default_safety();
        assert_eq!(sz.speed_factor(0.1), 0.0);
    }

    #[test]
    fn test_speed_factor_free() {
        let sz = default_safety();
        assert_eq!(sz.speed_factor(5.0), 1.0);
    }

    #[test]
    fn test_speed_factor_middle() {
        let sz = default_safety();
        let f = sz.speed_factor(1.65);
        assert!(f > 0.0 && f < 1.0);
    }

    #[test]
    fn test_vfh_empty_data() {
        let vfh = Vfh::new(36, default_safety()).unwrap();
        assert!(vfh.compute(&[], 0.0).is_err());
    }

    #[test]
    fn test_vfh_clear_field() {
        let vfh = Vfh::new(36, default_safety()).unwrap();
        let ranges: Vec<(f64, f64)> = (0..36)
            .map(|i| {
                let a = -std::f64::consts::PI + (i as f64 / 36.0) * 2.0 * std::f64::consts::PI;
                (a, 8.0)
            })
            .collect();
        let cmd = vfh.compute(&ranges, 0.0).unwrap();
        assert!(!cmd.emergency_stop);
        assert!(cmd.speed_scale > 0.5);
    }

    #[test]
    fn test_vfh_emergency_stop() {
        let vfh = Vfh::new(36, default_safety()).unwrap();
        let ranges = vec![(0.0, 0.1)]; // inside critical zone
        let cmd = vfh.compute(&ranges, 0.0).unwrap();
        assert!(cmd.emergency_stop);
    }

    #[test]
    fn test_vfh_blocked_front() {
        let vfh = Vfh::new(36, default_safety()).unwrap().with_threshold(0.3);
        let mut ranges: Vec<(f64, f64)> = Vec::new();
        for i in 0..36 {
            let a = -std::f64::consts::PI + (i as f64 / 36.0) * 2.0 * std::f64::consts::PI;
            let d = if a.abs() < 0.5 { 0.8 } else { 8.0 };
            ranges.push((a, d));
        }
        let cmd = vfh.compute(&ranges, 0.0).unwrap();
        // Should steer away from front
        assert!(cmd.heading.abs() > 0.01 || cmd.speed_scale < 1.0);
    }

    #[test]
    fn test_vfh_too_few_sectors() {
        assert!(Vfh::new(4, default_safety()).is_err());
    }

    #[test]
    fn test_nd_high_safety() {
        let nd = NearnessDiagram::new(default_safety(), 0.5);
        let ranges: Vec<(f64, f64)> = vec![(0.5, 8.0), (-0.5, 8.0), (1.0, 8.0), (-1.0, 8.0)];
        let (sit, cmd) = nd.compute(&ranges, 0.0).unwrap();
        assert_eq!(sit, NdSituation::HighSafety);
        assert!(!cmd.emergency_stop);
    }

    #[test]
    fn test_nd_empty_data() {
        let nd = NearnessDiagram::new(default_safety(), 0.5);
        assert!(nd.compute(&[], 0.0).is_err());
    }

    #[test]
    fn test_nd_danger() {
        let nd = NearnessDiagram::new(default_safety(), 0.5);
        let ranges = vec![(0.0, 0.1)];
        let (sit, cmd) = nd.compute(&ranges, 0.0).unwrap();
        assert_eq!(sit, NdSituation::HighDanger);
        assert!(cmd.emergency_stop);
    }

    #[test]
    fn test_bug2_direct_path() {
        let mut bug = Bug2::new((0.0, 0.0), (5.0, 0.0));
        let cmd = bug.step((0.0, 0.0), 10.0);
        assert_eq!(bug.state(), BugState::MoveToGoal);
        assert!(cmd.speed_scale > 0.0);
    }

    #[test]
    fn test_bug2_hit_obstacle() {
        let mut bug = Bug2::new((0.0, 0.0), (5.0, 0.0));
        let cmd = bug.step((1.0, 0.0), 0.5);
        assert_eq!(bug.state(), BugState::FollowBoundary);
        assert!(cmd.heading.abs() > 0.1);
    }

    #[test]
    fn test_bug2_reached() {
        let mut bug = Bug2::new((0.0, 0.0), (1.0, 0.0)).with_reach_threshold(0.5);
        let cmd = bug.step((0.9, 0.0), 10.0);
        assert_eq!(bug.state(), BugState::Reached);
        assert_eq!(cmd.speed_scale, 0.0);
    }

    #[test]
    fn test_steer_cmd_display() {
        let c = SteerCmd::new(0.5, 0.8);
        assert!(format!("{c}").contains("0.500"));
    }

    #[test]
    fn test_steer_cmd_stop_display() {
        let c = SteerCmd::stop();
        assert!(format!("{c}").contains("STOP"));
    }

    #[test]
    fn test_bug2_display() {
        let bug = Bug2::new((0.0, 0.0), (5.0, 3.0));
        let s = format!("{bug}");
        assert!(s.contains("MoveToGoal"));
    }
}
