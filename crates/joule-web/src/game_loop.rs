//! Game loop — fixed timestep with variable rendering, accumulator pattern,
//! fps tracking, delta time smoothing, frame rate limiting, pause/resume,
//! slow-motion support, performance statistics.
//!
//! Replaces JavaScript requestAnimationFrame loops and game loop libraries
//! (mainloop.js, game-loop) with a pure-Rust tick engine.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────

/// Game loop configuration.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Fixed update timestep in seconds.
    pub fixed_dt: f64,
    /// Maximum updates per frame to prevent spiral of death.
    pub max_updates_per_frame: u32,
    /// Target frame rate (0 = unlimited).
    pub target_fps: f64,
    /// Number of frames for delta time smoothing.
    pub smoothing_window: usize,
    /// Slow-motion factor (1.0 = normal, 0.5 = half speed).
    pub time_scale: f64,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            fixed_dt: 1.0 / 60.0,
            max_updates_per_frame: 5,
            target_fps: 0.0,
            smoothing_window: 10,
            time_scale: 1.0,
        }
    }
}

// ── Performance stats ───────────────────────────────────────────

/// Frame time histogram bucket.
#[derive(Debug, Clone, Copy)]
pub struct HistogramBucket {
    /// Upper bound in milliseconds.
    pub upper_ms: f64,
    /// Count of frames in this bucket.
    pub count: u64,
}

/// Performance statistics collected over time.
#[derive(Debug, Clone)]
pub struct PerfStats {
    /// Total frames rendered.
    pub total_frames: u64,
    /// Total fixed updates.
    pub total_updates: u64,
    /// Current FPS (smoothed).
    pub fps: f64,
    /// Average frame time in ms.
    pub avg_frame_time_ms: f64,
    /// Minimum frame time in ms.
    pub min_frame_time_ms: f64,
    /// Maximum frame time in ms.
    pub max_frame_time_ms: f64,
    /// Frame time histogram.
    pub histogram: Vec<HistogramBucket>,
    /// Recent frame times for percentile calculations.
    frame_times: VecDeque<f64>,
    /// Window size for stats.
    window: usize,
}

impl PerfStats {
    fn new(window: usize) -> Self {
        Self {
            total_frames: 0,
            total_updates: 0,
            fps: 0.0,
            avg_frame_time_ms: 0.0,
            min_frame_time_ms: f64::INFINITY,
            max_frame_time_ms: 0.0,
            histogram: vec![
                HistogramBucket { upper_ms: 4.0, count: 0 },    // ≤4ms (250+ fps)
                HistogramBucket { upper_ms: 8.0, count: 0 },    // ≤8ms (125+ fps)
                HistogramBucket { upper_ms: 16.67, count: 0 },  // ≤16.67ms (60+ fps)
                HistogramBucket { upper_ms: 33.33, count: 0 },  // ≤33.33ms (30+ fps)
                HistogramBucket { upper_ms: 66.67, count: 0 },  // ≤66.67ms (15+ fps)
                HistogramBucket { upper_ms: f64::INFINITY, count: 0 }, // >66.67ms
            ],
            frame_times: VecDeque::new(),
            window,
        }
    }

    fn record_frame(&mut self, dt_ms: f64) {
        self.total_frames += 1;
        self.frame_times.push_back(dt_ms);
        if self.frame_times.len() > self.window {
            self.frame_times.pop_front();
        }

        // Update running stats.
        if dt_ms < self.min_frame_time_ms { self.min_frame_time_ms = dt_ms; }
        if dt_ms > self.max_frame_time_ms { self.max_frame_time_ms = dt_ms; }

        let sum: f64 = self.frame_times.iter().sum();
        let count = self.frame_times.len() as f64;
        self.avg_frame_time_ms = sum / count;
        self.fps = if self.avg_frame_time_ms > 0.0 { 1000.0 / self.avg_frame_time_ms } else { 0.0 };

        // Histogram.
        for bucket in &mut self.histogram {
            if dt_ms <= bucket.upper_ms {
                bucket.count += 1;
                break;
            }
        }
    }

    fn record_update(&mut self) {
        self.total_updates += 1;
    }

    /// Get the 95th percentile frame time in ms.
    pub fn p95_frame_time_ms(&self) -> f64 {
        if self.frame_times.is_empty() { return 0.0; }
        let mut sorted: Vec<f64> = self.frame_times.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64) * 0.95) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

// ── Delta time smoother ─────────────────────────────────────────

/// Smooths delta time to reduce jitter.
#[derive(Debug, Clone)]
struct DtSmoother {
    history: VecDeque<f64>,
    window: usize,
}

impl DtSmoother {
    fn new(window: usize) -> Self {
        Self { history: VecDeque::new(), window: window.max(1) }
    }

    fn smooth(&mut self, dt: f64) -> f64 {
        self.history.push_back(dt);
        if self.history.len() > self.window {
            self.history.pop_front();
        }
        let sum: f64 = self.history.iter().sum();
        sum / self.history.len() as f64
    }
}

// ── Game Loop ───────────────────────────────────────────────────

/// The game loop state machine.
pub struct GameLoop {
    pub config: LoopConfig,
    pub stats: PerfStats,
    accumulator: f64,
    smoother: DtSmoother,
    paused: bool,
    last_time: Option<Instant>,
    min_frame_duration: Duration,
}

impl GameLoop {
    /// Create a new game loop with the given configuration.
    pub fn new(config: LoopConfig) -> Self {
        let min_frame_duration = if config.target_fps > 0.0 {
            Duration::from_secs_f64(1.0 / config.target_fps)
        } else {
            Duration::ZERO
        };
        let window = config.smoothing_window;
        Self {
            stats: PerfStats::new(window),
            smoother: DtSmoother::new(config.smoothing_window),
            config,
            accumulator: 0.0,
            paused: false,
            last_time: None,
            min_frame_duration,
        }
    }

    /// Process a single frame. Calls `update_fn` for each fixed step and
    /// `render_fn` once with the interpolation alpha.
    ///
    /// Returns the number of fixed updates performed this frame.
    pub fn frame<F, R>(&mut self, now: Instant, mut update_fn: F, mut render_fn: R) -> u32
    where
        F: FnMut(f64),  // dt
        R: FnMut(f64),  // alpha (interpolation factor 0..1)
    {
        let raw_dt = match self.last_time {
            Some(prev) => now.duration_since(prev).as_secs_f64(),
            None => self.config.fixed_dt,
        };
        self.last_time = Some(now);

        let dt_ms = raw_dt * 1000.0;
        self.stats.record_frame(dt_ms);

        if self.paused {
            render_fn(0.0);
            return 0;
        }

        let smoothed_dt = self.smoother.smooth(raw_dt) * self.config.time_scale;
        self.accumulator += smoothed_dt;

        let mut updates = 0u32;
        while self.accumulator >= self.config.fixed_dt && updates < self.config.max_updates_per_frame {
            update_fn(self.config.fixed_dt);
            self.accumulator -= self.config.fixed_dt;
            updates += 1;
            self.stats.record_update();
        }

        // Clamp accumulator to prevent spiral.
        if updates >= self.config.max_updates_per_frame {
            self.accumulator = 0.0;
        }

        let alpha = self.accumulator / self.config.fixed_dt;
        render_fn(alpha);

        updates
    }

    /// Convenience: process a frame using raw delta time in seconds.
    pub fn frame_with_dt<F, R>(&mut self, dt_seconds: f64, mut update_fn: F, mut render_fn: R) -> u32
    where
        F: FnMut(f64),
        R: FnMut(f64),
    {
        let dt_ms = dt_seconds * 1000.0;
        self.stats.record_frame(dt_ms);

        if self.paused {
            render_fn(0.0);
            return 0;
        }

        let smoothed_dt = self.smoother.smooth(dt_seconds) * self.config.time_scale;
        self.accumulator += smoothed_dt;

        let mut updates = 0u32;
        while self.accumulator >= self.config.fixed_dt && updates < self.config.max_updates_per_frame {
            update_fn(self.config.fixed_dt);
            self.accumulator -= self.config.fixed_dt;
            updates += 1;
            self.stats.record_update();
        }

        if updates >= self.config.max_updates_per_frame {
            self.accumulator = 0.0;
        }

        let alpha = self.accumulator / self.config.fixed_dt;
        render_fn(alpha);

        updates
    }

    /// Pause the game loop.
    pub fn pause(&mut self) { self.paused = true; }

    /// Resume the game loop.
    pub fn resume(&mut self) {
        self.paused = false;
        self.last_time = None; // reset to avoid huge dt spike
    }

    /// Check if paused.
    pub fn is_paused(&self) -> bool { self.paused }

    /// Set time scale (slow motion). 1.0 = normal, 0.5 = half speed, 2.0 = double.
    pub fn set_time_scale(&mut self, scale: f64) {
        self.config.time_scale = scale.max(0.0);
    }

    /// Get current time scale.
    pub fn time_scale(&self) -> f64 { self.config.time_scale }

    /// Set target FPS (frame rate limiting).
    pub fn set_target_fps(&mut self, fps: f64) {
        self.config.target_fps = fps;
        self.min_frame_duration = if fps > 0.0 {
            Duration::from_secs_f64(1.0 / fps)
        } else {
            Duration::ZERO
        };
    }

    /// Get the minimum frame duration for frame rate limiting.
    pub fn min_frame_duration(&self) -> Duration {
        self.min_frame_duration
    }

    /// Should we sleep before the next frame? Returns remaining time.
    pub fn time_until_next_frame(&self, frame_start: Instant) -> Duration {
        if self.min_frame_duration == Duration::ZERO {
            return Duration::ZERO;
        }
        let elapsed = frame_start.elapsed();
        if elapsed < self.min_frame_duration {
            self.min_frame_duration - elapsed
        } else {
            Duration::ZERO
        }
    }

    /// Reset all statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PerfStats::new(self.config.smoothing_window);
    }
}

impl Default for GameLoop {
    fn default() -> Self {
        Self::new(LoopConfig::default())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_timestep_updates() {
        let config = LoopConfig { fixed_dt: 1.0 / 60.0, ..Default::default() };
        let mut game_loop = GameLoop::new(config);
        let mut update_count = 0;
        let mut render_count = 0;
        // Simulate 2 frames worth of time.
        let updates = game_loop.frame_with_dt(
            2.0 / 60.0,
            |_dt| { update_count += 1; },
            |_alpha| { render_count += 1; },
        );
        assert_eq!(updates, 2);
        assert_eq!(update_count, 2);
        assert_eq!(render_count, 1); // always 1 render per frame
    }

    #[test]
    fn spiral_of_death_prevention() {
        let config = LoopConfig {
            fixed_dt: 1.0 / 60.0,
            max_updates_per_frame: 3,
            ..Default::default()
        };
        let mut game_loop = GameLoop::new(config);
        let mut updates_done = 0;
        // Simulate huge dt (10 frames worth).
        let updates = game_loop.frame_with_dt(
            10.0 / 60.0,
            |_| { updates_done += 1; },
            |_| {},
        );
        assert!(updates <= 3);
        assert_eq!(updates_done, updates as i32 as usize);
    }

    #[test]
    fn pause_and_resume() {
        let mut game_loop = GameLoop::default();
        game_loop.pause();
        assert!(game_loop.is_paused());

        let mut update_count = 0;
        game_loop.frame_with_dt(1.0 / 60.0, |_| { update_count += 1; }, |_| {});
        assert_eq!(update_count, 0); // no updates while paused

        game_loop.resume();
        assert!(!game_loop.is_paused());
    }

    #[test]
    fn slow_motion() {
        let config = LoopConfig { fixed_dt: 1.0 / 60.0, time_scale: 0.5, ..Default::default() };
        let mut game_loop = GameLoop::new(config);
        // With time_scale=0.5, 2 frames of dt should produce ~1 update.
        let updates = game_loop.frame_with_dt(2.0 / 60.0, |_| {}, |_| {});
        assert_eq!(updates, 1);
    }

    #[test]
    fn double_speed() {
        let config = LoopConfig { fixed_dt: 1.0 / 60.0, time_scale: 2.0, ..Default::default() };
        let mut game_loop = GameLoop::new(config);
        let updates = game_loop.frame_with_dt(1.0 / 60.0, |_| {}, |_| {});
        assert_eq!(updates, 2);
    }

    #[test]
    fn render_alpha_interpolation() {
        let config = LoopConfig { fixed_dt: 1.0 / 60.0, ..Default::default() };
        let mut game_loop = GameLoop::new(config);
        let mut alpha_val = -1.0;
        // Half a frame worth of time — no update, but render with alpha.
        game_loop.frame_with_dt(0.5 / 60.0, |_| {}, |alpha| { alpha_val = alpha; });
        assert!(alpha_val >= 0.0 && alpha_val <= 1.0);
    }

    #[test]
    fn fps_tracking() {
        let mut game_loop = GameLoop::default();
        for _ in 0..60 {
            game_loop.frame_with_dt(1.0 / 60.0, |_| {}, |_| {});
        }
        // FPS should be approximately 60.
        assert!((game_loop.stats.fps - 60.0).abs() < 5.0);
        assert_eq!(game_loop.stats.total_frames, 60);
    }

    #[test]
    fn performance_histogram() {
        let mut game_loop = GameLoop::default();
        // 16.67ms frames → should land in the ≤16.67ms bucket.
        for _ in 0..10 {
            game_loop.frame_with_dt(1.0 / 60.0, |_| {}, |_| {});
        }
        let bucket = &game_loop.stats.histogram[2]; // ≤16.67ms
        assert!(bucket.count > 0);
    }

    #[test]
    fn p95_frame_time() {
        let mut game_loop = GameLoop::default();
        for _ in 0..100 {
            game_loop.frame_with_dt(1.0 / 60.0, |_| {}, |_| {});
        }
        let p95 = game_loop.stats.p95_frame_time_ms();
        assert!(p95 > 0.0);
        assert!(p95 < 100.0);
    }

    #[test]
    fn frame_rate_limiting() {
        let mut game_loop = GameLoop::default();
        game_loop.set_target_fps(30.0);
        assert!((game_loop.min_frame_duration().as_secs_f64() - 1.0 / 30.0).abs() < 1e-6);
    }

    #[test]
    fn delta_time_smoothing() {
        let mut smoother = DtSmoother::new(3);
        assert!((smoother.smooth(1.0) - 1.0).abs() < 1e-9);
        assert!((smoother.smooth(2.0) - 1.5).abs() < 1e-9);
        assert!((smoother.smooth(3.0) - 2.0).abs() < 1e-9);
        // Window slides: drops 1.0.
        assert!((smoother.smooth(3.0) - 8.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn reset_stats() {
        let mut game_loop = GameLoop::default();
        for _ in 0..10 {
            game_loop.frame_with_dt(1.0 / 60.0, |_| {}, |_| {});
        }
        assert_eq!(game_loop.stats.total_frames, 10);
        game_loop.reset_stats();
        assert_eq!(game_loop.stats.total_frames, 0);
    }

    #[test]
    fn frame_with_instant() {
        let config = LoopConfig { fixed_dt: 1.0 / 60.0, ..Default::default() };
        let mut game_loop = GameLoop::new(config);
        let now = Instant::now();
        let mut update_count = 0;
        // First frame initializes.
        game_loop.frame(now, |_| { update_count += 1; }, |_| {});
        // Second frame with realistic delta.
        let later = now + Duration::from_millis(17);
        game_loop.frame(later, |_| { update_count += 1; }, |_| {});
        assert!(update_count >= 1);
    }

    #[test]
    fn time_scale_setter() {
        let mut game_loop = GameLoop::default();
        game_loop.set_time_scale(0.25);
        assert!((game_loop.time_scale() - 0.25).abs() < 1e-9);
        game_loop.set_time_scale(-1.0);
        assert!((game_loop.time_scale()).abs() < 1e-9); // clamped to 0
    }
}
