//! SVG animation engine: SMIL-style animations (animate, animateTransform,
//! animateMotion), timing (begin, dur, repeatCount), calcMode (linear, discrete,
//! spline, paced), additive/accumulate modes, keyTimes/keySplines.
//!
//! Pure math — no browser dependency.

use std::fmt;

// ── Timing ─────────────────────────────────────────────────────

/// Duration of an animation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Duration {
    /// Fixed duration in seconds.
    Seconds(f64),
    /// Indefinite — runs forever until explicitly stopped.
    Indefinite,
}

impl Duration {
    pub fn as_secs(&self) -> Option<f64> {
        match self {
            Duration::Seconds(s) => Some(*s),
            Duration::Indefinite => None,
        }
    }
}

/// Repeat behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RepeatCount {
    /// Repeat a fixed number of times.
    Count(f64),
    /// Repeat indefinitely.
    Indefinite,
}

impl RepeatCount {
    pub fn is_indefinite(&self) -> bool {
        matches!(self, RepeatCount::Indefinite)
    }
}

/// Animation timing specification.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationTiming {
    pub begin: f64,
    pub dur: Duration,
    pub repeat_count: RepeatCount,
    pub fill: FillMode,
}

impl Default for AnimationTiming {
    fn default() -> Self {
        Self {
            begin: 0.0,
            dur: Duration::Seconds(1.0),
            repeat_count: RepeatCount::Count(1.0),
            fill: FillMode::Remove,
        }
    }
}

impl AnimationTiming {
    /// Total active duration in seconds (None if indefinite).
    pub fn active_duration(&self) -> Option<f64> {
        let dur = self.dur.as_secs()?;
        match self.repeat_count {
            RepeatCount::Count(n) => Some(dur * n),
            RepeatCount::Indefinite => None,
        }
    }

    /// Whether the animation is active at a given absolute time.
    pub fn is_active(&self, t: f64) -> bool {
        if t < self.begin {
            return false;
        }
        match self.active_duration() {
            Some(ad) => t < self.begin + ad,
            None => true,
        }
    }

    /// Compute the simple time (0..dur) and iteration count at absolute time t.
    pub fn simple_time(&self, t: f64) -> Option<(f64, u32)> {
        let dur = self.dur.as_secs()?;
        if dur <= 0.0 {
            return None;
        }
        let local = t - self.begin;
        if local < 0.0 {
            return None;
        }
        let iteration = (local / dur).floor() as u32;
        let simple = local - (iteration as f64) * dur;
        Some((simple, iteration))
    }

    /// Normalized progress (0.0–1.0) at time t, accounting for repeats.
    pub fn progress(&self, t: f64) -> Option<f64> {
        let dur = self.dur.as_secs()?;
        if dur <= 0.0 {
            return None;
        }
        let local = t - self.begin;
        if local < 0.0 {
            if self.fill == FillMode::Freeze {
                return Some(0.0);
            }
            return None;
        }
        match self.active_duration() {
            Some(ad) if local >= ad => {
                if self.fill == FillMode::Freeze {
                    Some(1.0)
                } else {
                    None
                }
            }
            _ => {
                let within_iter = (local % dur) / dur;
                Some(within_iter)
            }
        }
    }
}

/// What happens when the animation ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    /// Attribute reverts to base value.
    Remove,
    /// Attribute stays at final animated value.
    Freeze,
}

// ── Calc Mode ──────────────────────────────────────────────────

/// Interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalcMode {
    Linear,
    Discrete,
    Paced,
    Spline,
}

/// Cubic bezier control points for spline interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeySpline {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl KeySpline {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Evaluate the cubic bezier at parameter t using De Casteljau.
    pub fn evaluate(&self, t: f64) -> f64 {
        // Solve for parameter u where bezier_x(u) = t, then return bezier_y(u).
        // Use Newton's method.
        let mut u = t;
        for _ in 0..8 {
            let bx = cubic_bezier(u, 0.0, self.x1, self.x2, 1.0);
            let dx = cubic_bezier_derivative(u, 0.0, self.x1, self.x2, 1.0);
            if dx.abs() < 1e-12 {
                break;
            }
            u -= (bx - t) / dx;
            u = u.clamp(0.0, 1.0);
        }
        cubic_bezier(u, 0.0, self.y1, self.y2, 1.0)
    }
}

fn cubic_bezier(t: f64, p0: f64, p1: f64, p2: f64, p3: f64) -> f64 {
    let mt = 1.0 - t;
    mt * mt * mt * p0 + 3.0 * mt * mt * t * p1 + 3.0 * mt * t * t * p2 + t * t * t * p3
}

fn cubic_bezier_derivative(t: f64, p0: f64, p1: f64, p2: f64, p3: f64) -> f64 {
    let mt = 1.0 - t;
    3.0 * mt * mt * (p1 - p0) + 6.0 * mt * t * (p2 - p1) + 3.0 * t * t * (p3 - p2)
}

// ── Additive / Accumulate ──────────────────────────────────────

/// How animated values combine with the base value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdditiveMode {
    /// Animated value replaces base.
    Replace,
    /// Animated value is added to base.
    Sum,
}

/// How repeated iterations accumulate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulateMode {
    /// Each iteration starts from scratch.
    None,
    /// Each iteration adds to the result of the previous.
    Sum,
}

// ── Animation Value ────────────────────────────────────────────

/// A single animated value (scalar or multi-component).
#[derive(Debug, Clone, PartialEq)]
pub enum AnimValue {
    Scalar(f64),
    Vec2(f64, f64),
    Vec3(f64, f64, f64),
    Color(f64, f64, f64, f64),
}

impl AnimValue {
    pub fn lerp(&self, other: &AnimValue, t: f64) -> AnimValue {
        match (self, other) {
            (AnimValue::Scalar(a), AnimValue::Scalar(b)) => {
                AnimValue::Scalar(a + (b - a) * t)
            }
            (AnimValue::Vec2(ax, ay), AnimValue::Vec2(bx, by)) => {
                AnimValue::Vec2(ax + (bx - ax) * t, ay + (by - ay) * t)
            }
            (AnimValue::Vec3(ax, ay, az), AnimValue::Vec3(bx, by, bz)) => {
                AnimValue::Vec3(
                    ax + (bx - ax) * t,
                    ay + (by - ay) * t,
                    az + (bz - az) * t,
                )
            }
            (AnimValue::Color(ar, ag, ab, aa), AnimValue::Color(br, bg, bb, ba)) => {
                AnimValue::Color(
                    ar + (br - ar) * t,
                    ag + (bg - ag) * t,
                    ab + (bb - ab) * t,
                    aa + (ba - aa) * t,
                )
            }
            _ => self.clone(),
        }
    }

    pub fn add(&self, other: &AnimValue) -> AnimValue {
        match (self, other) {
            (AnimValue::Scalar(a), AnimValue::Scalar(b)) => AnimValue::Scalar(a + b),
            (AnimValue::Vec2(ax, ay), AnimValue::Vec2(bx, by)) => AnimValue::Vec2(ax + bx, ay + by),
            (AnimValue::Vec3(ax, ay, az), AnimValue::Vec3(bx, by, bz)) => {
                AnimValue::Vec3(ax + bx, ay + by, az + bz)
            }
            (AnimValue::Color(ar, ag, ab, aa), AnimValue::Color(br, bg, bb, ba)) => {
                AnimValue::Color(ar + br, ag + bg, ab + bb, aa + ba)
            }
            _ => self.clone(),
        }
    }

    pub fn scale(&self, s: f64) -> AnimValue {
        match self {
            AnimValue::Scalar(v) => AnimValue::Scalar(v * s),
            AnimValue::Vec2(x, y) => AnimValue::Vec2(x * s, y * s),
            AnimValue::Vec3(x, y, z) => AnimValue::Vec3(x * s, y * s, z * s),
            AnimValue::Color(r, g, b, a) => AnimValue::Color(r * s, g * s, b * s, a * s),
        }
    }

    /// Distance between two values (for paced interpolation).
    pub fn distance(&self, other: &AnimValue) -> f64 {
        match (self, other) {
            (AnimValue::Scalar(a), AnimValue::Scalar(b)) => (b - a).abs(),
            (AnimValue::Vec2(ax, ay), AnimValue::Vec2(bx, by)) => {
                ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt()
            }
            (AnimValue::Vec3(ax, ay, az), AnimValue::Vec3(bx, by, bz)) => {
                ((bx - ax).powi(2) + (by - ay).powi(2) + (bz - az).powi(2)).sqrt()
            }
            _ => 0.0,
        }
    }
}

// ── Transform Type ─────────────────────────────────────────────

/// Transform type for animateTransform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    Translate,
    Scale,
    Rotate,
    SkewX,
    SkewY,
}

// ── Animation Types ────────────────────────────────────────────

/// An SVG animation element.
#[derive(Debug, Clone, PartialEq)]
pub enum Animation {
    /// `<animate>` — animate a single attribute.
    Animate {
        attribute: String,
        values: Vec<AnimValue>,
        key_times: Option<Vec<f64>>,
        key_splines: Option<Vec<KeySpline>>,
        calc_mode: CalcMode,
        timing: AnimationTiming,
        additive: AdditiveMode,
        accumulate: AccumulateMode,
    },
    /// `<animateTransform>` — animate a transform attribute.
    AnimateTransform {
        transform_type: TransformType,
        values: Vec<AnimValue>,
        key_times: Option<Vec<f64>>,
        key_splines: Option<Vec<KeySpline>>,
        calc_mode: CalcMode,
        timing: AnimationTiming,
        additive: AdditiveMode,
        accumulate: AccumulateMode,
    },
    /// `<animateMotion>` — animate position along a path.
    AnimateMotion {
        /// Path as (x, y) waypoints.
        path: Vec<(f64, f64)>,
        key_times: Option<Vec<f64>>,
        calc_mode: CalcMode,
        timing: AnimationTiming,
        rotate: MotionRotate,
    },
}

/// Rotation behavior for animateMotion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MotionRotate {
    /// Fixed angle in degrees.
    Fixed(f64),
    /// Auto — follow path tangent.
    Auto,
    /// Auto-reverse — follow path tangent + 180°.
    AutoReverse,
}

// ── Interpolation Engine ───────────────────────────────────────

/// Interpolate between keyframe values.
pub fn interpolate_values(
    values: &[AnimValue],
    progress: f64,
    calc_mode: CalcMode,
    key_times: Option<&[f64]>,
    key_splines: Option<&[KeySpline]>,
) -> AnimValue {
    if values.is_empty() {
        return AnimValue::Scalar(0.0);
    }
    if values.len() == 1 {
        return values[0].clone();
    }

    let n = values.len();
    let p = progress.clamp(0.0, 1.0);

    match calc_mode {
        CalcMode::Discrete => {
            let idx = if let Some(kt) = key_times {
                discrete_index(kt, p)
            } else {
                ((p * n as f64).floor() as usize).min(n - 1)
            };
            values[idx].clone()
        }
        CalcMode::Linear => {
            let (seg, local_t) = segment_and_t(n, p, key_times);
            values[seg].lerp(&values[(seg + 1).min(n - 1)], local_t)
        }
        CalcMode::Spline => {
            let (seg, local_t) = segment_and_t(n, p, key_times);
            let splined_t = if let Some(splines) = key_splines {
                if seg < splines.len() {
                    splines[seg].evaluate(local_t)
                } else {
                    local_t
                }
            } else {
                local_t
            };
            values[seg].lerp(&values[(seg + 1).min(n - 1)], splined_t)
        }
        CalcMode::Paced => {
            // Distribute evenly by cumulative distance.
            let distances: Vec<f64> = (0..n - 1)
                .map(|i| values[i].distance(&values[i + 1]))
                .collect();
            let total: f64 = distances.iter().sum();
            if total < 1e-12 {
                return values[0].clone();
            }
            let target = p * total;
            let mut accum = 0.0;
            for (i, d) in distances.iter().enumerate() {
                if accum + d >= target || i == distances.len() - 1 {
                    let local_t = if *d < 1e-12 { 0.0 } else { (target - accum) / d };
                    return values[i].lerp(&values[i + 1], local_t);
                }
                accum += d;
            }
            values[n - 1].clone()
        }
    }
}

fn discrete_index(key_times: &[f64], p: f64) -> usize {
    for i in (0..key_times.len()).rev() {
        if p >= key_times[i] {
            return i;
        }
    }
    0
}

fn segment_and_t(n: usize, p: f64, key_times: Option<&[f64]>) -> (usize, f64) {
    let segs = n - 1;
    if segs == 0 {
        return (0, 0.0);
    }

    if let Some(kt) = key_times {
        for i in 0..segs {
            let t0 = kt[i];
            let t1 = kt.get(i + 1).copied().unwrap_or(1.0);
            if p <= t1 || i == segs - 1 {
                let span = t1 - t0;
                let local = if span < 1e-12 { 0.0 } else { (p - t0) / span };
                return (i, local.clamp(0.0, 1.0));
            }
        }
        (segs - 1, 1.0)
    } else {
        let seg_f = p * segs as f64;
        let seg = (seg_f.floor() as usize).min(segs - 1);
        let local = seg_f - seg as f64;
        (seg, local.clamp(0.0, 1.0))
    }
}

/// Evaluate an Animation at a given time.
pub fn evaluate_animation(anim: &Animation, t: f64, base_value: &AnimValue) -> Option<AnimValue> {
    match anim {
        Animation::Animate {
            values,
            key_times,
            key_splines,
            calc_mode,
            timing,
            additive,
            accumulate,
            ..
        }
        | Animation::AnimateTransform {
            values,
            key_times,
            key_splines,
            calc_mode,
            timing,
            additive,
            accumulate,
            ..
        } => {
            let progress = timing.progress(t)?;
            let (_, iteration) = timing.simple_time(t)?;
            let raw = interpolate_values(
                values,
                progress,
                *calc_mode,
                key_times.as_deref(),
                key_splines.as_deref(),
            );

            let accumulated = if *accumulate == AccumulateMode::Sum && iteration > 0 {
                let last_val = values.last()?;
                let accum_delta = last_val.scale(iteration as f64);
                raw.add(&accum_delta)
            } else {
                raw
            };

            let result = match additive {
                AdditiveMode::Replace => accumulated,
                AdditiveMode::Sum => base_value.add(&accumulated),
            };

            Some(result)
        }
        Animation::AnimateMotion { path, key_times, calc_mode, timing, rotate } => {
            let progress = timing.progress(t)?;
            let pos = interpolate_path(path, progress, *calc_mode, key_times.as_deref());
            Some(AnimValue::Vec2(pos.0, pos.1))
        }
    }
}

/// Interpolate along a path of waypoints.
pub fn interpolate_path(
    path: &[(f64, f64)],
    progress: f64,
    calc_mode: CalcMode,
    key_times: Option<&[f64]>,
) -> (f64, f64) {
    if path.is_empty() {
        return (0.0, 0.0);
    }
    if path.len() == 1 {
        return path[0];
    }

    let values: Vec<AnimValue> = path.iter().map(|(x, y)| AnimValue::Vec2(*x, *y)).collect();
    let result = interpolate_values(&values, progress, calc_mode, key_times, None);
    match result {
        AnimValue::Vec2(x, y) => (x, y),
        _ => (0.0, 0.0),
    }
}

/// Compute path tangent angle at a given progress.
pub fn path_tangent_angle(path: &[(f64, f64)], progress: f64) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }
    let n = path.len();
    let seg_f = progress * (n - 1) as f64;
    let seg = (seg_f.floor() as usize).min(n - 2);
    let (x0, y0) = path[seg];
    let (x1, y1) = path[seg + 1];
    (y1 - y0).atan2(x1 - x0) * 180.0 / std::f64::consts::PI
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_active() {
        let t = AnimationTiming {
            begin: 1.0,
            dur: Duration::Seconds(2.0),
            repeat_count: RepeatCount::Count(1.0),
            fill: FillMode::Remove,
        };
        assert!(!t.is_active(0.5));
        assert!(t.is_active(1.0));
        assert!(t.is_active(2.5));
        assert!(!t.is_active(3.0));
    }

    #[test]
    fn test_timing_active_duration() {
        let t = AnimationTiming {
            begin: 0.0,
            dur: Duration::Seconds(2.0),
            repeat_count: RepeatCount::Count(3.0),
            fill: FillMode::Remove,
        };
        assert_eq!(t.active_duration(), Some(6.0));
    }

    #[test]
    fn test_timing_indefinite() {
        let t = AnimationTiming {
            begin: 0.0,
            dur: Duration::Seconds(1.0),
            repeat_count: RepeatCount::Indefinite,
            fill: FillMode::Remove,
        };
        assert_eq!(t.active_duration(), None);
        assert!(t.is_active(1000.0));
    }

    #[test]
    fn test_progress_linear() {
        let t = AnimationTiming {
            begin: 0.0,
            dur: Duration::Seconds(2.0),
            repeat_count: RepeatCount::Count(1.0),
            fill: FillMode::Remove,
        };
        assert!((t.progress(0.0).unwrap() - 0.0).abs() < 1e-10);
        assert!((t.progress(1.0).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_progress_freeze() {
        let t = AnimationTiming {
            begin: 0.0,
            dur: Duration::Seconds(1.0),
            repeat_count: RepeatCount::Count(1.0),
            fill: FillMode::Freeze,
        };
        assert_eq!(t.progress(2.0), Some(1.0));
    }

    #[test]
    fn test_linear_interpolation() {
        let values = vec![AnimValue::Scalar(0.0), AnimValue::Scalar(10.0)];
        let r = interpolate_values(&values, 0.5, CalcMode::Linear, None, None);
        match r {
            AnimValue::Scalar(v) => assert!((v - 5.0).abs() < 1e-10),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_discrete_interpolation() {
        let values = vec![
            AnimValue::Scalar(0.0),
            AnimValue::Scalar(5.0),
            AnimValue::Scalar(10.0),
        ];
        let r = interpolate_values(&values, 0.3, CalcMode::Discrete, None, None);
        match r {
            AnimValue::Scalar(v) => assert!((v - 0.0).abs() < 1e-10),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_paced_interpolation() {
        let values = vec![
            AnimValue::Scalar(0.0),
            AnimValue::Scalar(2.0),
            AnimValue::Scalar(10.0),
        ];
        // Total distance = 2 + 8 = 10. At p=0.2 → target=2.0 → exactly at value[1].
        let r = interpolate_values(&values, 0.2, CalcMode::Paced, None, None);
        match r {
            AnimValue::Scalar(v) => assert!((v - 2.0).abs() < 1e-10),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_spline_interpolation() {
        let values = vec![AnimValue::Scalar(0.0), AnimValue::Scalar(100.0)];
        let spline = KeySpline::new(0.0, 0.0, 1.0, 1.0); // linear bezier
        let r = interpolate_values(
            &values,
            0.5,
            CalcMode::Spline,
            None,
            Some(&[spline]),
        );
        match r {
            AnimValue::Scalar(v) => assert!((v - 50.0).abs() < 1.0),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_key_times() {
        let values = vec![
            AnimValue::Scalar(0.0),
            AnimValue::Scalar(100.0),
            AnimValue::Scalar(100.0),
        ];
        let key_times = vec![0.0, 0.2, 1.0];
        // At p=0.1 → segment 0, local_t = 0.1/0.2 = 0.5 → value = 50
        let r = interpolate_values(&values, 0.1, CalcMode::Linear, Some(&key_times), None);
        match r {
            AnimValue::Scalar(v) => assert!((v - 50.0).abs() < 1e-10),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_additive_sum() {
        let anim = Animation::Animate {
            attribute: "x".into(),
            values: vec![AnimValue::Scalar(0.0), AnimValue::Scalar(10.0)],
            key_times: None,
            key_splines: None,
            calc_mode: CalcMode::Linear,
            timing: AnimationTiming {
                begin: 0.0,
                dur: Duration::Seconds(1.0),
                repeat_count: RepeatCount::Count(1.0),
                fill: FillMode::Remove,
            },
            additive: AdditiveMode::Sum,
            accumulate: AccumulateMode::None,
        };
        let base = AnimValue::Scalar(100.0);
        let r = evaluate_animation(&anim, 0.5, &base).unwrap();
        match r {
            AnimValue::Scalar(v) => assert!((v - 105.0).abs() < 1e-10),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_animate_motion() {
        let path = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)];
        let pos = interpolate_path(&path, 0.5, CalcMode::Linear, None);
        assert!((pos.0 - 100.0).abs() < 1e-10);
        assert!(pos.1.abs() < 1e-10);
    }

    #[test]
    fn test_path_tangent() {
        let path = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let angle = path_tangent_angle(&path, 0.0);
        assert!(angle.abs() < 1e-10); // horizontal → 0°
        let angle2 = path_tangent_angle(&path, 0.9);
        assert!((angle2 - 90.0).abs() < 1e-10); // vertical → 90°
    }

    #[test]
    fn test_key_spline_ease() {
        let ease_in = KeySpline::new(0.42, 0.0, 1.0, 1.0);
        let v = ease_in.evaluate(0.5);
        // Ease-in: should be less than 0.5 at midpoint of x.
        assert!(v < 0.55);
    }

    #[test]
    fn test_vec2_lerp() {
        let a = AnimValue::Vec2(0.0, 0.0);
        let b = AnimValue::Vec2(10.0, 20.0);
        let r = a.lerp(&b, 0.5);
        match r {
            AnimValue::Vec2(x, y) => {
                assert!((x - 5.0).abs() < 1e-10);
                assert!((y - 10.0).abs() < 1e-10);
            }
            _ => panic!("wrong type"),
        }
    }
}
