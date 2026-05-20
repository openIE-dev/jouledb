//! CSS transition engine: transition configs, active transitions,
//! interpolation with timing functions, and a transition manager.

use std::collections::HashMap;

// ── Timing Functions ────────────────────────────────────────────

/// CSS timing function for transition interpolation.
#[derive(Debug, Clone, PartialEq)]
pub enum TimingFunction {
    Linear,
    Ease,
    EaseIn,
    EaseOut,
    EaseInOut,
    CubicBezier(f64, f64, f64, f64),
    Steps(u32, StepPosition),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepPosition {
    Start,
    End,
}

impl TimingFunction {
    /// Map normalized progress `t` in [0,1] to eased output.
    pub fn interpolate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            TimingFunction::Linear => t,
            TimingFunction::Ease => cubic_bezier(0.25, 0.1, 0.25, 1.0, t),
            TimingFunction::EaseIn => cubic_bezier(0.42, 0.0, 1.0, 1.0, t),
            TimingFunction::EaseOut => cubic_bezier(0.0, 0.0, 0.58, 1.0, t),
            TimingFunction::EaseInOut => cubic_bezier(0.42, 0.0, 0.58, 1.0, t),
            TimingFunction::CubicBezier(x1, y1, x2, y2) => cubic_bezier(*x1, *y1, *x2, *y2, t),
            TimingFunction::Steps(n, pos) => {
                if *n == 0 {
                    return t;
                }
                let step = match pos {
                    StepPosition::Start => (t * *n as f64).ceil() / *n as f64,
                    StepPosition::End => (t * *n as f64).floor() / *n as f64,
                };
                step.clamp(0.0, 1.0)
            }
        }
    }
}

fn cubic_bezier(x1: f64, y1: f64, x2: f64, y2: f64, t: f64) -> f64 {
    // Newton-Raphson to find parameter for x, then evaluate y
    let mut guess = t;
    for _ in 0..8 {
        let x = bezier_component(x1, x2, guess) - t;
        let dx = bezier_derivative(x1, x2, guess);
        if dx.abs() < 1e-10 {
            break;
        }
        guess -= x / dx;
        guess = guess.clamp(0.0, 1.0);
    }
    bezier_component(y1, y2, guess)
}

fn bezier_component(p1: f64, p2: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    3.0 * (1.0 - t) * (1.0 - t) * t * p1 + 3.0 * (1.0 - t) * t2 * p2 + t3
}

fn bezier_derivative(p1: f64, p2: f64, t: f64) -> f64 {
    let t2 = t * t;
    3.0 * (1.0 - t) * (1.0 - t) * p1 + 6.0 * (1.0 - t) * t * (p2 - p1) + 3.0 * t2 * (1.0 - p2)
}

// ── Transition Config ───────────────────────────────────────────

/// Configuration for a single property transition.
#[derive(Debug, Clone)]
pub struct TransitionConfig {
    pub property: String,
    pub duration_ms: f64,
    pub delay_ms: f64,
    pub timing_function: TimingFunction,
}

impl TransitionConfig {
    pub fn new(property: &str, duration_ms: f64) -> Self {
        Self {
            property: property.to_string(),
            duration_ms,
            delay_ms: 0.0,
            timing_function: TimingFunction::Ease,
        }
    }

    pub fn with_delay(mut self, delay_ms: f64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    pub fn with_timing(mut self, tf: TimingFunction) -> Self {
        self.timing_function = tf;
        self
    }
}

// ── Active Transition ───────────────────────────────────────────

/// A currently running transition.
#[derive(Debug, Clone)]
pub struct ActiveTransition {
    pub property: String,
    pub from_value: f64,
    pub to_value: f64,
    pub start_time: f64,
    pub config: TransitionConfig,
}

impl ActiveTransition {
    /// Get the interpolated value at a given time (ms).
    pub fn value_at(&self, current_time: f64) -> f64 {
        let elapsed = current_time - self.start_time - self.config.delay_ms;
        if elapsed < 0.0 {
            return self.from_value;
        }
        if elapsed >= self.config.duration_ms {
            return self.to_value;
        }
        let progress = elapsed / self.config.duration_ms;
        let eased = self.config.timing_function.interpolate(progress);
        self.from_value + (self.to_value - self.from_value) * eased
    }

    /// Whether this transition is complete at the given time.
    pub fn is_complete(&self, current_time: f64) -> bool {
        let elapsed = current_time - self.start_time - self.config.delay_ms;
        elapsed >= self.config.duration_ms
    }
}

// ── Completion Event ────────────────────────────────────────────

/// Emitted when a transition completes.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionEvent {
    pub property: String,
    pub elapsed_ms: f64,
}

// ── Transition Manager ──────────────────────────────────────────

/// Manages active transitions, interpolation, and completion events.
#[derive(Debug)]
pub struct TransitionManager {
    configs: HashMap<String, TransitionConfig>,
    active: HashMap<String, ActiveTransition>,
    current_values: HashMap<String, f64>,
}

impl TransitionManager {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            active: HashMap::new(),
            current_values: HashMap::new(),
        }
    }

    /// Register a transition configuration for a property.
    pub fn configure(&mut self, config: TransitionConfig) {
        self.configs.insert(config.property.clone(), config);
    }

    /// Set a property value. If a transition config exists and the value changed,
    /// start a transition.
    pub fn set_property(&mut self, property: &str, value: f64, current_time: f64) {
        let old_value = self.current_values.get(property).copied();
        self.current_values.insert(property.to_string(), value);

        if let Some(old) = old_value {
            if (old - value).abs() > f64::EPSILON {
                if let Some(config) = self.configs.get(property).cloned() {
                    // If there's an active transition, start from the current interpolated value
                    let from = if let Some(active) = self.active.get(property) {
                        active.value_at(current_time)
                    } else {
                        old
                    };
                    self.active.insert(property.to_string(), ActiveTransition {
                        property: property.to_string(),
                        from_value: from,
                        to_value: value,
                        start_time: current_time,
                        config,
                    });
                }
            }
        }
    }

    /// Advance time: compute interpolated values and collect completion events.
    pub fn tick(&mut self, current_time: f64) -> (HashMap<String, f64>, Vec<TransitionEvent>) {
        let mut values = HashMap::new();
        let mut events = Vec::new();
        let mut completed = Vec::new();

        for (prop, transition) in &self.active {
            let value = transition.value_at(current_time);
            values.insert(prop.clone(), value);

            if transition.is_complete(current_time) {
                let elapsed = current_time - transition.start_time;
                events.push(TransitionEvent {
                    property: prop.clone(),
                    elapsed_ms: elapsed,
                });
                completed.push(prop.clone());
            }
        }

        for prop in completed {
            self.active.remove(&prop);
        }

        (values, events)
    }

    /// Cancel a specific property transition.
    pub fn cancel(&mut self, property: &str) {
        self.active.remove(property);
    }

    /// Cancel all active transitions.
    pub fn cancel_all(&mut self) {
        self.active.clear();
    }

    /// Get the number of active transitions.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Check if a specific property is currently transitioning.
    pub fn is_transitioning(&self, property: &str) -> bool {
        self.active.contains_key(property)
    }
}

impl Default for TransitionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Shorthand Parsing ───────────────────────────────────────────

/// Parse a CSS transition shorthand string.
/// Format: `property duration [delay] [timing-function]`
/// Example: `"opacity 300ms 100ms ease-in"`
pub fn parse_shorthand(input: &str) -> Option<TransitionConfig> {
    let parts: Vec<&str> = input.trim().split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }

    let property = parts[0].to_string();
    let duration_ms = parse_time(parts[1])?;

    let mut delay_ms = 0.0;
    let mut timing = TimingFunction::Ease;
    let mut idx = 2;

    // Next token could be delay or timing function
    if idx < parts.len() {
        if let Some(d) = parse_time(parts[idx]) {
            delay_ms = d;
            idx += 1;
        }
    }

    if idx < parts.len() {
        if let Some(tf) = parse_timing_function(parts[idx]) {
            timing = tf;
        }
    }

    Some(TransitionConfig {
        property,
        duration_ms,
        delay_ms,
        timing_function: timing,
    })
}

fn parse_time(s: &str) -> Option<f64> {
    if let Some(ms) = s.strip_suffix("ms") {
        ms.parse::<f64>().ok()
    } else if let Some(sec) = s.strip_suffix('s') {
        sec.parse::<f64>().ok().map(|v| v * 1000.0)
    } else {
        s.parse::<f64>().ok()
    }
}

fn parse_timing_function(s: &str) -> Option<TimingFunction> {
    match s {
        "linear" => Some(TimingFunction::Linear),
        "ease" => Some(TimingFunction::Ease),
        "ease-in" => Some(TimingFunction::EaseIn),
        "ease-out" => Some(TimingFunction::EaseOut),
        "ease-in-out" => Some(TimingFunction::EaseInOut),
        _ => None,
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn linear_timing() {
        let tf = TimingFunction::Linear;
        assert!(approx(tf.interpolate(0.0), 0.0));
        assert!(approx(tf.interpolate(0.5), 0.5));
        assert!(approx(tf.interpolate(1.0), 1.0));
    }

    #[test]
    fn steps_timing() {
        let tf = TimingFunction::Steps(4, StepPosition::End);
        assert!(approx(tf.interpolate(0.0), 0.0));
        assert!(approx(tf.interpolate(0.3), 0.25));
        assert!(approx(tf.interpolate(0.6), 0.5));
        assert!(approx(tf.interpolate(1.0), 1.0));
    }

    #[test]
    fn active_transition_interpolates() {
        let config = TransitionConfig::new("opacity", 1000.0)
            .with_timing(TimingFunction::Linear);
        let t = ActiveTransition {
            property: "opacity".to_string(),
            from_value: 0.0,
            to_value: 1.0,
            start_time: 0.0,
            config,
        };
        assert!(approx(t.value_at(0.0), 0.0));
        assert!(approx(t.value_at(500.0), 0.5));
        assert!(approx(t.value_at(1000.0), 1.0));
    }

    #[test]
    fn transition_with_delay() {
        let config = TransitionConfig::new("opacity", 1000.0)
            .with_delay(200.0)
            .with_timing(TimingFunction::Linear);
        let t = ActiveTransition {
            property: "opacity".to_string(),
            from_value: 0.0,
            to_value: 1.0,
            start_time: 0.0,
            config,
        };
        // During delay, stays at from_value
        assert!(approx(t.value_at(100.0), 0.0));
        // After delay, interpolates
        assert!(approx(t.value_at(700.0), 0.5));
    }

    #[test]
    fn manager_starts_transition() {
        let mut mgr = TransitionManager::new();
        mgr.configure(TransitionConfig::new("opacity", 1000.0)
            .with_timing(TimingFunction::Linear));

        // Set initial value (no transition — first set)
        mgr.set_property("opacity", 0.0, 0.0);
        assert_eq!(mgr.active_count(), 0);

        // Change value — starts transition
        mgr.set_property("opacity", 1.0, 100.0);
        assert_eq!(mgr.active_count(), 1);
        assert!(mgr.is_transitioning("opacity"));
    }

    #[test]
    fn manager_tick_interpolates() {
        let mut mgr = TransitionManager::new();
        mgr.configure(TransitionConfig::new("opacity", 1000.0)
            .with_timing(TimingFunction::Linear));
        mgr.set_property("opacity", 0.0, 0.0);
        mgr.set_property("opacity", 1.0, 0.0);

        let (values, events) = mgr.tick(500.0);
        assert!(approx(*values.get("opacity").unwrap(), 0.5));
        assert!(events.is_empty());
    }

    #[test]
    fn manager_emits_completion() {
        let mut mgr = TransitionManager::new();
        mgr.configure(TransitionConfig::new("opacity", 100.0)
            .with_timing(TimingFunction::Linear));
        mgr.set_property("opacity", 0.0, 0.0);
        mgr.set_property("opacity", 1.0, 0.0);

        let (values, events) = mgr.tick(200.0);
        assert!(approx(*values.get("opacity").unwrap(), 1.0));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].property, "opacity");
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn manager_cancel() {
        let mut mgr = TransitionManager::new();
        mgr.configure(TransitionConfig::new("opacity", 1000.0));
        mgr.set_property("opacity", 0.0, 0.0);
        mgr.set_property("opacity", 1.0, 0.0);
        assert_eq!(mgr.active_count(), 1);

        mgr.cancel("opacity");
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn manager_override_transition() {
        let mut mgr = TransitionManager::new();
        mgr.configure(TransitionConfig::new("left", 1000.0)
            .with_timing(TimingFunction::Linear));
        mgr.set_property("left", 0.0, 0.0);
        mgr.set_property("left", 100.0, 0.0);

        // Mid-transition, change target
        let (values, _) = mgr.tick(500.0);
        let mid_value = *values.get("left").unwrap();
        assert!(approx(mid_value, 50.0));

        mgr.set_property("left", 200.0, 500.0);
        // New transition starts from mid_value
        let (values2, _) = mgr.tick(500.0);
        let start_value = *values2.get("left").unwrap();
        assert!(approx(start_value, mid_value)); // at t=500 the new transition just started
    }

    #[test]
    fn parse_shorthand_basic() {
        let config = parse_shorthand("opacity 300ms").unwrap();
        assert_eq!(config.property, "opacity");
        assert!(approx(config.duration_ms, 300.0));
    }

    #[test]
    fn parse_shorthand_with_delay_and_timing() {
        let config = parse_shorthand("opacity 300ms 100ms ease-in").unwrap();
        assert_eq!(config.property, "opacity");
        assert!(approx(config.duration_ms, 300.0));
        assert!(approx(config.delay_ms, 100.0));
        assert_eq!(config.timing_function, TimingFunction::EaseIn);
    }

    #[test]
    fn parse_shorthand_seconds() {
        let config = parse_shorthand("transform 0.5s").unwrap();
        assert!(approx(config.duration_ms, 500.0));
    }

    #[test]
    fn parse_shorthand_invalid() {
        assert!(parse_shorthand("opacity").is_none());
    }
}
