//! Fixed-timestep game loop with variable rendering.
//!
//! Accumulator-based update with a fixed dt (e.g., 1/60s), interpolation
//! alpha for smooth rendering, frame timing, FPS tracking, frame count,
//! pause / slow-motion (time scale), max frame skip, and energy tracking
//! per frame via joule counters.

use std::collections::VecDeque;

// ── Constants ───────────────────────────────────────────────────

const DEFAULT_FIXED_DT: f64 = 1.0 / 60.0;
const DEFAULT_MAX_FRAME_SKIP: u32 = 5;
const DEFAULT_TIME_SCALE: f64 = 1.0;
const FPS_SAMPLE_WINDOW: usize = 60;

// ── Energy tracking ─────────────────────────────────────────────

/// Tracks per-frame energy consumption in microjoules.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyCounter {
    pub total_uj: u64,
    pub frame_uj: u64,
    pub peak_frame_uj: u64,
    pub frame_count: u64,
}

impl EnergyCounter {
    pub fn new() -> Self {
        Self {
            total_uj: 0,
            frame_uj: 0,
            peak_frame_uj: 0,
            frame_count: 0,
        }
    }

    /// Record energy spent this frame and advance the counter.
    pub fn record_frame(&mut self, uj: u64) {
        self.frame_uj = uj;
        self.total_uj = self.total_uj.saturating_add(uj);
        if uj > self.peak_frame_uj {
            self.peak_frame_uj = uj;
        }
        self.frame_count += 1;
    }

    /// Average energy per frame (returns 0 when no frames recorded).
    pub fn avg_frame_uj(&self) -> u64 {
        if self.frame_count == 0 {
            return 0;
        }
        self.total_uj / self.frame_count
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ── Frame statistics ────────────────────────────────────────────

/// Rolling FPS and frame-time statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStats {
    frame_times: VecDeque<f64>,
    pub frame_count: u64,
    pub total_time: f64,
    pub last_frame_time: f64,
}

impl FrameStats {
    pub fn new() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(FPS_SAMPLE_WINDOW),
            frame_count: 0,
            total_time: 0.0,
            last_frame_time: 0.0,
        }
    }

    pub fn record(&mut self, dt: f64) {
        self.last_frame_time = dt;
        self.total_time += dt;
        self.frame_count += 1;
        self.frame_times.push_back(dt);
        if self.frame_times.len() > FPS_SAMPLE_WINDOW {
            self.frame_times.pop_front();
        }
    }

    /// Current smoothed FPS over the sample window.
    pub fn fps(&self) -> f64 {
        if self.frame_times.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.frame_times.iter().sum();
        if sum < 1e-12 {
            return 0.0;
        }
        self.frame_times.len() as f64 / sum
    }

    /// Min frame time in the sample window.
    pub fn min_frame_time(&self) -> f64 {
        self.frame_times
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Max frame time in the sample window.
    pub fn max_frame_time(&self) -> f64 {
        self.frame_times
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ── Tick result ─────────────────────────────────────────────────

/// The result of advancing the game loop by one real-time frame.
#[derive(Debug, Clone, PartialEq)]
pub struct TickResult {
    /// How many fixed-step updates were run this frame.
    pub update_count: u32,
    /// Interpolation alpha in [0, 1) for rendering between the last two
    /// fixed-step states.
    pub alpha: f64,
    /// Whether the frame was capped by `max_frame_skip`.
    pub frame_skip_capped: bool,
}

// ── Game loop ───────────────────────────────────────────────────

/// Fixed-timestep game loop with variable-rate rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct GameTickLoop {
    /// Seconds per fixed-step update.
    pub fixed_dt: f64,
    /// Multiplier applied to incoming real-time dt. 1.0 = normal,
    /// 0.5 = half speed, 2.0 = double speed.
    pub time_scale: f64,
    /// Maximum number of fixed-step updates allowed in a single frame.
    pub max_frame_skip: u32,
    /// When true, time accumulator freezes (no updates run).
    pub paused: bool,

    accumulator: f64,
    stats: FrameStats,
    energy: EnergyCounter,
}

impl GameTickLoop {
    pub fn new() -> Self {
        Self {
            fixed_dt: DEFAULT_FIXED_DT,
            time_scale: DEFAULT_TIME_SCALE,
            max_frame_skip: DEFAULT_MAX_FRAME_SKIP,
            paused: false,
            accumulator: 0.0,
            stats: FrameStats::new(),
            energy: EnergyCounter::new(),
        }
    }

    /// Create with a custom fixed timestep (in seconds).
    pub fn with_fixed_dt(mut self, dt: f64) -> Self {
        assert!(dt > 0.0, "fixed_dt must be positive");
        self.fixed_dt = dt;
        self
    }

    pub fn with_max_frame_skip(mut self, n: u32) -> Self {
        self.max_frame_skip = n;
        self
    }

    pub fn with_time_scale(mut self, scale: f64) -> Self {
        self.time_scale = scale.max(0.0);
        self
    }

    // ── Accessors ───────────────────────────────────────────────

    pub fn stats(&self) -> &FrameStats {
        &self.stats
    }

    pub fn energy(&self) -> &EnergyCounter {
        &self.energy
    }

    pub fn accumulator(&self) -> f64 {
        self.accumulator
    }

    // ── Mutators ────────────────────────────────────────────────

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn set_time_scale(&mut self, scale: f64) {
        self.time_scale = scale.max(0.0);
    }

    /// Record energy consumed during the current frame.
    pub fn record_energy(&mut self, uj: u64) {
        self.energy.record_frame(uj);
    }

    pub fn reset(&mut self) {
        self.accumulator = 0.0;
        self.stats.reset();
        self.energy.reset();
        self.paused = false;
        self.time_scale = DEFAULT_TIME_SCALE;
    }

    // ── Core loop ───────────────────────────────────────────────

    /// Advance the loop by `real_dt` seconds of wall-clock time.
    ///
    /// Returns a [`TickResult`] describing how many fixed updates should
    /// be run and the interpolation alpha for rendering.
    pub fn tick(&mut self, real_dt: f64) -> TickResult {
        // Clamp negative/zero and very large dt to avoid spiral of death.
        let real_dt = real_dt.clamp(0.0, 0.25);
        self.stats.record(real_dt);

        if self.paused {
            return TickResult {
                update_count: 0,
                alpha: 0.0,
                frame_skip_capped: false,
            };
        }

        let scaled_dt = real_dt * self.time_scale;
        self.accumulator += scaled_dt;

        let mut update_count: u32 = 0;
        let mut capped = false;

        while self.accumulator >= self.fixed_dt {
            self.accumulator -= self.fixed_dt;
            update_count += 1;
            if update_count >= self.max_frame_skip {
                // Drain remaining accumulator to prevent spiral of death.
                self.accumulator = 0.0;
                capped = true;
                break;
            }
        }

        let alpha = if self.fixed_dt > 1e-12 {
            self.accumulator / self.fixed_dt
        } else {
            0.0
        };

        TickResult {
            update_count,
            alpha,
            frame_skip_capped: capped,
        }
    }

    /// Convenience: run a full "frame" cycle, calling `update_fn` for each
    /// fixed step, then returning the alpha for the render phase.
    pub fn run_frame<F>(&mut self, real_dt: f64, mut update_fn: F) -> f64
    where
        F: FnMut(f64),
    {
        let result = self.tick(real_dt);
        for _ in 0..result.update_count {
            update_fn(self.fixed_dt);
        }
        result.alpha
    }
}

// ── Simulation helper ───────────────────────────────────────────

/// Runs a simple simulation for the given number of frames, returning
/// total update steps performed, final FPS, and total energy.
pub fn simulate_loop(
    fixed_dt: f64,
    frame_dts: &[f64],
    energy_per_frame: u64,
) -> (u64, f64, u64) {
    let mut game_loop = GameTickLoop::new().with_fixed_dt(fixed_dt);
    let mut total_updates: u64 = 0;

    for &dt in frame_dts {
        let result = game_loop.tick(dt);
        total_updates += result.update_count as u64;
        game_loop.record_energy(energy_per_frame);
    }

    (total_updates, game_loop.stats().fps(), game_loop.energy().total_uj)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn test_default_construction() {
        let gl = GameTickLoop::new();
        assert!((gl.fixed_dt - DEFAULT_FIXED_DT).abs() < EPS);
        assert!((gl.time_scale - 1.0).abs() < EPS);
        assert_eq!(gl.max_frame_skip, DEFAULT_MAX_FRAME_SKIP);
        assert!(!gl.paused);
        assert!(gl.accumulator().abs() < EPS);
    }

    #[test]
    fn test_builder_methods() {
        let gl = GameTickLoop::new()
            .with_fixed_dt(1.0 / 30.0)
            .with_max_frame_skip(10)
            .with_time_scale(0.5);
        assert!((gl.fixed_dt - 1.0 / 30.0).abs() < EPS);
        assert_eq!(gl.max_frame_skip, 10);
        assert!((gl.time_scale - 0.5).abs() < EPS);
    }

    #[test]
    fn test_single_update_per_frame() {
        let mut gl = GameTickLoop::new().with_fixed_dt(1.0 / 60.0);
        let result = gl.tick(1.0 / 60.0);
        assert_eq!(result.update_count, 1);
        assert!(!result.frame_skip_capped);
        assert!(result.alpha < 1e-6);
    }

    #[test]
    fn test_multiple_updates_per_frame() {
        let mut gl = GameTickLoop::new().with_fixed_dt(1.0 / 60.0);
        // 3 updates worth of time
        let result = gl.tick(3.0 / 60.0);
        assert_eq!(result.update_count, 3);
        assert!(!result.frame_skip_capped);
    }

    #[test]
    fn test_accumulator_carries_remainder() {
        let mut gl = GameTickLoop::new().with_fixed_dt(1.0 / 60.0);
        let dt = 1.5 / 60.0; // 1.5 steps
        let result = gl.tick(dt);
        assert_eq!(result.update_count, 1);
        // alpha ~0.5
        assert!((result.alpha - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_alpha_interpolation() {
        let mut gl = GameTickLoop::new().with_fixed_dt(0.1);
        let result = gl.tick(0.15);
        assert_eq!(result.update_count, 1);
        assert!((result.alpha - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_frame_skip_cap() {
        let mut gl = GameTickLoop::new()
            .with_fixed_dt(1.0 / 60.0)
            .with_max_frame_skip(3);
        // Feed huge dt that would require many updates
        let result = gl.tick(0.2); // ~12 updates at 60fps
        assert_eq!(result.update_count, 3);
        assert!(result.frame_skip_capped);
        // Accumulator should be drained
        assert!(gl.accumulator().abs() < EPS);
    }

    #[test]
    fn test_pause_prevents_updates() {
        let mut gl = GameTickLoop::new().with_fixed_dt(1.0 / 60.0);
        gl.pause();
        let result = gl.tick(1.0 / 60.0);
        assert_eq!(result.update_count, 0);
        assert!(result.alpha.abs() < EPS);
    }

    #[test]
    fn test_resume_after_pause() {
        let mut gl = GameTickLoop::new().with_fixed_dt(1.0 / 60.0);
        gl.pause();
        gl.tick(1.0 / 60.0);
        gl.resume();
        let result = gl.tick(1.0 / 60.0);
        assert_eq!(result.update_count, 1);
    }

    #[test]
    fn test_toggle_pause() {
        let mut gl = GameTickLoop::new();
        assert!(!gl.paused);
        gl.toggle_pause();
        assert!(gl.paused);
        gl.toggle_pause();
        assert!(!gl.paused);
    }

    #[test]
    fn test_slow_motion() {
        let mut gl = GameTickLoop::new()
            .with_fixed_dt(1.0 / 60.0)
            .with_time_scale(0.5);
        // At half speed, one frame at 1/60s contributes only half a step.
        let result = gl.tick(1.0 / 60.0);
        assert_eq!(result.update_count, 0);
        // Second frame pushes it past one step.
        let result2 = gl.tick(1.0 / 60.0);
        assert_eq!(result2.update_count, 1);
    }

    #[test]
    fn test_fast_forward() {
        let mut gl = GameTickLoop::new()
            .with_fixed_dt(1.0 / 60.0)
            .with_time_scale(2.0);
        let result = gl.tick(1.0 / 60.0);
        assert_eq!(result.update_count, 2);
    }

    #[test]
    fn test_zero_dt_no_update() {
        let mut gl = GameTickLoop::new().with_fixed_dt(1.0 / 60.0);
        let result = gl.tick(0.0);
        assert_eq!(result.update_count, 0);
    }

    #[test]
    fn test_negative_dt_clamped() {
        let mut gl = GameTickLoop::new().with_fixed_dt(1.0 / 60.0);
        let result = gl.tick(-1.0);
        assert_eq!(result.update_count, 0);
    }

    #[test]
    fn test_large_dt_clamped() {
        let mut gl = GameTickLoop::new().with_fixed_dt(1.0 / 60.0);
        // dt > 0.25 gets clamped to 0.25
        let result = gl.tick(10.0);
        // At 60fps, 0.25s = 15 updates, capped by max_frame_skip=5
        assert_eq!(result.update_count, 5);
        assert!(result.frame_skip_capped);
    }

    #[test]
    fn test_frame_stats_fps() {
        let mut gl = GameTickLoop::new();
        for _ in 0..60 {
            gl.tick(1.0 / 60.0);
        }
        let fps = gl.stats().fps();
        assert!((fps - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_frame_stats_count() {
        let mut gl = GameTickLoop::new();
        for _ in 0..10 {
            gl.tick(0.016);
        }
        assert_eq!(gl.stats().frame_count, 10);
    }

    #[test]
    fn test_frame_stats_min_max() {
        let mut gl = GameTickLoop::new();
        gl.tick(0.010);
        gl.tick(0.020);
        gl.tick(0.015);
        assert!((gl.stats().min_frame_time() - 0.010).abs() < EPS);
        assert!((gl.stats().max_frame_time() - 0.020).abs() < EPS);
    }

    #[test]
    fn test_energy_counter() {
        let mut ec = EnergyCounter::new();
        ec.record_frame(100);
        ec.record_frame(200);
        ec.record_frame(50);
        assert_eq!(ec.total_uj, 350);
        assert_eq!(ec.peak_frame_uj, 200);
        assert_eq!(ec.frame_uj, 50);
        assert_eq!(ec.avg_frame_uj(), 116); // 350/3
        assert_eq!(ec.frame_count, 3);
    }

    #[test]
    fn test_energy_counter_reset() {
        let mut ec = EnergyCounter::new();
        ec.record_frame(500);
        ec.reset();
        assert_eq!(ec.total_uj, 0);
        assert_eq!(ec.peak_frame_uj, 0);
        assert_eq!(ec.frame_count, 0);
    }

    #[test]
    fn test_energy_avg_zero_frames() {
        let ec = EnergyCounter::new();
        assert_eq!(ec.avg_frame_uj(), 0);
    }

    #[test]
    fn test_record_energy_through_loop() {
        let mut gl = GameTickLoop::new();
        gl.tick(0.016);
        gl.record_energy(42);
        gl.tick(0.016);
        gl.record_energy(58);
        assert_eq!(gl.energy().total_uj, 100);
        assert_eq!(gl.energy().frame_count, 2);
    }

    #[test]
    fn test_run_frame_callback() {
        let mut gl = GameTickLoop::new().with_fixed_dt(0.1);
        let mut count = 0u32;
        let alpha = gl.run_frame(0.25, |_dt| {
            count += 1;
        });
        assert_eq!(count, 2);
        assert!(alpha >= 0.0 && alpha < 1.0);
    }

    #[test]
    fn test_simulate_loop() {
        let frames: Vec<f64> = (0..100).map(|_| 1.0 / 60.0).collect();
        let (updates, fps, energy) = simulate_loop(1.0 / 60.0, &frames, 10);
        assert_eq!(updates, 100);
        assert!((fps - 60.0).abs() < 2.0);
        assert_eq!(energy, 1000);
    }

    #[test]
    fn test_reset_loop() {
        let mut gl = GameTickLoop::new();
        gl.tick(0.1);
        gl.record_energy(100);
        gl.pause();
        gl.set_time_scale(0.5);
        gl.reset();
        assert!(!gl.paused);
        assert!((gl.time_scale - 1.0).abs() < EPS);
        assert!(gl.accumulator().abs() < EPS);
        assert_eq!(gl.stats().frame_count, 0);
        assert_eq!(gl.energy().total_uj, 0);
    }

    #[test]
    fn test_set_time_scale_clamps_negative() {
        let mut gl = GameTickLoop::new();
        gl.set_time_scale(-5.0);
        assert!(gl.time_scale.abs() < EPS);
    }

    #[test]
    fn test_frame_stats_reset() {
        let mut stats = FrameStats::new();
        stats.record(0.016);
        stats.record(0.017);
        stats.reset();
        assert_eq!(stats.frame_count, 0);
        assert!(stats.total_time.abs() < EPS);
        assert!(stats.fps().abs() < EPS);
    }
}
