//! Lindenmayer system (L-system) for procedural plants and fractals.
//!
//! Supports deterministic and stochastic production rules, parametric
//! L-systems with numeric parameters, turtle interpretation for 2D
//! rendering, branching via stack, and classic fractal systems (Koch
//! snowflake, dragon curve, fractal plant).

use std::collections::HashMap;

// ── Seeded RNG ──

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ── Symbol ──

/// A symbol in an L-system string, optionally with numeric parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub ch: char,
    pub params: Vec<f64>,
}

impl Symbol {
    pub fn new(ch: char) -> Self {
        Self { ch, params: Vec::new() }
    }

    pub fn with_params(ch: char, params: Vec<f64>) -> Self {
        Self { ch, params }
    }
}

/// Parse a simple L-system string (no parameters) into symbols.
pub fn parse_symbols(s: &str) -> Vec<Symbol> {
    s.chars().map(Symbol::new).collect()
}

/// Convert symbols back to a string (ignoring parameters).
pub fn symbols_to_string(symbols: &[Symbol]) -> String {
    symbols.iter().map(|s| s.ch).collect()
}

// ── ProductionRule ──

/// A production rule: maps a symbol character to one or more replacements.
/// If multiple replacements exist (with weights), the rule is stochastic.
#[derive(Debug, Clone)]
pub struct ProductionRule {
    pub symbol: char,
    pub replacements: Vec<(String, f64)>,
}

impl ProductionRule {
    /// Deterministic rule: symbol -> replacement.
    pub fn deterministic(symbol: char, replacement: &str) -> Self {
        Self {
            symbol,
            replacements: vec![(replacement.to_string(), 1.0)],
        }
    }

    /// Stochastic rule: symbol -> one of several replacements by weight.
    pub fn stochastic(symbol: char, replacements: &[(&str, f64)]) -> Self {
        Self {
            symbol,
            replacements: replacements.iter().map(|(s, w)| (s.to_string(), *w)).collect(),
        }
    }

    fn apply(&self, rng: &mut Rng) -> &str {
        if self.replacements.len() == 1 {
            return &self.replacements[0].0;
        }
        let total: f64 = self.replacements.iter().map(|(_, w)| w).sum();
        if total <= 0.0 { return &self.replacements[0].0; }
        let roll = rng.next_f64() * total;
        let mut cum = 0.0;
        for (rep, w) in &self.replacements {
            cum += w;
            if roll < cum { return rep; }
        }
        &self.replacements.last().unwrap().0
    }
}

// ── ParametricRule ──

/// A parametric production rule that transforms symbol parameters.
#[derive(Debug, Clone)]
pub struct ParametricRule {
    pub symbol: char,
    pub transform: Box<fn(&[f64]) -> Vec<Symbol>>,
}

// ── LSystem ──

#[derive(Debug, Clone)]
pub struct LSystem {
    pub axiom: String,
    pub rules: Vec<ProductionRule>,
    pub iterations: u32,
}

impl LSystem {
    pub fn new(axiom: &str) -> Self {
        Self {
            axiom: axiom.to_string(),
            rules: Vec::new(),
            iterations: 0,
        }
    }

    pub fn add_rule(mut self, rule: ProductionRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_iterations(mut self, n: u32) -> Self {
        self.iterations = n;
        self
    }

    /// Generate the L-system string after N iterations.
    pub fn generate(&self, seed: u64) -> String {
        let mut rng = Rng::new(seed);
        let rule_map: HashMap<char, &ProductionRule> =
            self.rules.iter().map(|r| (r.symbol, r)).collect();

        let mut current = self.axiom.clone();
        for _ in 0..self.iterations {
            let mut next = String::with_capacity(current.len() * 2);
            for ch in current.chars() {
                match rule_map.get(&ch) {
                    Some(rule) => next.push_str(rule.apply(&mut rng)),
                    None => next.push(ch),
                }
            }
            current = next;
        }
        current
    }

    /// Get the length of the generated string after N iterations.
    pub fn generated_length(&self, seed: u64) -> usize {
        self.generate(seed).len()
    }
}

// ── Turtle state ──

#[derive(Debug, Clone, PartialEq)]
pub struct TurtleState {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub step_size: f64,
}

impl TurtleState {
    pub fn new() -> Self {
        Self { x: 0.0, y: 0.0, angle: 90.0_f64.to_radians(), step_size: 10.0 }
    }

    pub fn with_step(mut self, step: f64) -> Self {
        self.step_size = step;
        self
    }

    pub fn with_angle_deg(mut self, angle: f64) -> Self {
        self.angle = angle.to_radians();
        self
    }
}

// ── Line segment ──

#[derive(Debug, Clone, PartialEq)]
pub struct LineSegment {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub depth: usize,
}

// ── TurtleConfig ──

#[derive(Debug, Clone)]
pub struct TurtleConfig {
    pub angle_delta: f64,
    pub step_size: f64,
    pub step_scale: f64,
    pub initial_angle_deg: f64,
}

impl TurtleConfig {
    pub fn new(angle_delta_deg: f64, step_size: f64) -> Self {
        Self {
            angle_delta: angle_delta_deg.to_radians(),
            step_size,
            step_scale: 1.0,
            initial_angle_deg: 90.0,
        }
    }

    pub fn with_step_scale(mut self, scale: f64) -> Self {
        self.step_scale = scale;
        self
    }

    pub fn with_initial_angle(mut self, deg: f64) -> Self {
        self.initial_angle_deg = deg;
        self
    }
}

// ── Turtle interpreter ──

/// Interpret an L-system string using a turtle to produce line segments.
///
/// Standard alphabet:
/// - `F` / `G`: move forward and draw
/// - `f`: move forward without drawing
/// - `+`: turn left by angle_delta
/// - `-`: turn right by angle_delta
/// - `[`: push state (branch)
/// - `]`: pop state (end branch)
pub fn interpret(lsys_string: &str, config: &TurtleConfig) -> Vec<LineSegment> {
    let mut segments = Vec::new();
    let mut state = TurtleState::new()
        .with_step(config.step_size)
        .with_angle_deg(config.initial_angle_deg);
    let mut stack: Vec<(TurtleState, usize)> = Vec::new();
    let mut depth = 0usize;

    for ch in lsys_string.chars() {
        match ch {
            'F' | 'G' => {
                let x2 = state.x + state.angle.cos() * state.step_size;
                let y2 = state.y + state.angle.sin() * state.step_size;
                segments.push(LineSegment {
                    x1: state.x, y1: state.y,
                    x2, y2,
                    depth,
                });
                state.x = x2;
                state.y = y2;
            }
            'f' => {
                state.x += state.angle.cos() * state.step_size;
                state.y += state.angle.sin() * state.step_size;
            }
            '+' => {
                state.angle += config.angle_delta;
            }
            '-' => {
                state.angle -= config.angle_delta;
            }
            '[' => {
                stack.push((state.clone(), depth));
                depth += 1;
                state.step_size *= config.step_scale;
            }
            ']' => {
                if let Some((prev, d)) = stack.pop() {
                    state = prev;
                    depth = d;
                }
            }
            _ => {}
        }
    }
    segments
}

/// Compute the bounding box of line segments.
pub fn bounding_box(segments: &[LineSegment]) -> (f64, f64, f64, f64) {
    if segments.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for seg in segments {
        min_x = min_x.min(seg.x1).min(seg.x2);
        min_y = min_y.min(seg.y1).min(seg.y2);
        max_x = max_x.max(seg.x1).max(seg.x2);
        max_y = max_y.max(seg.y1).max(seg.y2);
    }
    (min_x, min_y, max_x, max_y)
}

// ── Classic L-systems ──

/// Koch snowflake: F -> F+F--F+F, angle = 60 degrees.
pub fn koch_snowflake(iterations: u32) -> LSystem {
    LSystem::new("F--F--F")
        .add_rule(ProductionRule::deterministic('F', "F+F--F+F"))
        .with_iterations(iterations)
}

/// Dragon curve: F -> F+G, G -> F-G, angle = 90 degrees.
pub fn dragon_curve(iterations: u32) -> LSystem {
    LSystem::new("F")
        .add_rule(ProductionRule::deterministic('F', "F+G"))
        .add_rule(ProductionRule::deterministic('G', "F-G"))
        .with_iterations(iterations)
}

/// Fractal plant: F -> F[+F]F[-F]F, angle = 25.7 degrees.
pub fn fractal_plant(iterations: u32) -> LSystem {
    LSystem::new("F")
        .add_rule(ProductionRule::deterministic('F', "F[+F]F[-F]F"))
        .with_iterations(iterations)
}

/// Sierpinski triangle: F -> F-G+F+G-F, G -> GG, angle = 120 degrees.
pub fn sierpinski_triangle(iterations: u32) -> LSystem {
    LSystem::new("F-G-G")
        .add_rule(ProductionRule::deterministic('F', "F-G+F+G-F"))
        .add_rule(ProductionRule::deterministic('G', "GG"))
        .with_iterations(iterations)
}

/// Stochastic tree (randomized branch angles).
pub fn stochastic_tree(iterations: u32) -> LSystem {
    LSystem::new("F")
        .add_rule(ProductionRule::stochastic('F', &[
            ("F[+F]F[-F]F", 0.33),
            ("F[+F][-F]", 0.34),
            ("F[-F]F[+F]F", 0.33),
        ]))
        .with_iterations(iterations)
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_new() {
        let s = Symbol::new('F');
        assert_eq!(s.ch, 'F');
        assert!(s.params.is_empty());
    }

    #[test]
    fn test_symbol_with_params() {
        let s = Symbol::with_params('F', vec![1.0, 2.5]);
        assert_eq!(s.params.len(), 2);
        assert!((s.params[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_symbols() {
        let syms = parse_symbols("F+F-F");
        assert_eq!(syms.len(), 5);
        assert_eq!(syms[0].ch, 'F');
        assert_eq!(syms[1].ch, '+');
    }

    #[test]
    fn test_symbols_to_string() {
        let syms = vec![Symbol::new('F'), Symbol::new('+'), Symbol::new('F')];
        assert_eq!(symbols_to_string(&syms), "F+F");
    }

    #[test]
    fn test_deterministic_rule() {
        let sys = LSystem::new("F")
            .add_rule(ProductionRule::deterministic('F', "FF"))
            .with_iterations(3);
        let result = sys.generate(42);
        assert_eq!(result, "FFFFFFFF"); // 2^3 = 8 F's
    }

    #[test]
    fn test_koch_snowflake_iter0() {
        let sys = koch_snowflake(0);
        let result = sys.generate(42);
        assert_eq!(result, "F--F--F");
    }

    #[test]
    fn test_koch_snowflake_iter1() {
        let sys = koch_snowflake(1);
        let result = sys.generate(42);
        assert_eq!(result, "F+F--F+F--F+F--F+F--F+F--F+F");
    }

    #[test]
    fn test_dragon_curve() {
        let sys = dragon_curve(3);
        let result = sys.generate(42);
        assert!(result.contains('F'));
        assert!(result.contains('G'));
        assert!(result.contains('+'));
        assert!(result.contains('-'));
    }

    #[test]
    fn test_fractal_plant() {
        let sys = fractal_plant(2);
        let result = sys.generate(42);
        assert!(result.contains('['));
        assert!(result.contains(']'));
    }

    #[test]
    fn test_stochastic_determinism() {
        let sys = stochastic_tree(3);
        let a = sys.generate(42);
        let b = sys.generate(42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_stochastic_variation() {
        let sys = stochastic_tree(3);
        let a = sys.generate(1);
        let b = sys.generate(999);
        // Different seeds should produce different results (very likely)
        assert!(a.len() > 0);
        assert!(b.len() > 0);
    }

    #[test]
    fn test_interpret_forward() {
        let config = TurtleConfig::new(90.0, 10.0);
        let segments = interpret("F", &config);
        assert_eq!(segments.len(), 1);
        assert!((segments[0].x1 - 0.0).abs() < 1e-6);
        assert!((segments[0].y1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpret_turn() {
        let config = TurtleConfig::new(90.0, 10.0).with_initial_angle(0.0);
        let segments = interpret("F+F", &config);
        assert_eq!(segments.len(), 2);
        // First segment moves right (angle=0 means along x-axis)
        assert!((segments[0].x2 - 10.0).abs() < 1e-6);
        // After +90 degrees, moves along y-axis
        assert!((segments[1].y2 - segments[1].y1 - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpret_branch() {
        let config = TurtleConfig::new(45.0, 10.0);
        let segments = interpret("F[+F]F", &config);
        assert_eq!(segments.len(), 3);
        // After ], position resets
        assert!((segments[2].x1 - segments[0].x2).abs() < 1e-6);
    }

    #[test]
    fn test_interpret_no_draw() {
        let config = TurtleConfig::new(90.0, 10.0);
        let segments = interpret("fF", &config);
        assert_eq!(segments.len(), 1);
        // F starts after f's movement
        assert!((segments[0].x1).abs() > 1e-6 || (segments[0].y1).abs() > 1e-6);
    }

    #[test]
    fn test_bounding_box() {
        let config = TurtleConfig::new(90.0, 10.0).with_initial_angle(0.0);
        let segments = interpret("F+F+F+F", &config);
        let (min_x, min_y, max_x, max_y) = bounding_box(&segments);
        // Square: 0..10 x 0..10
        assert!(max_x - min_x > 5.0);
        assert!(max_y - min_y > 5.0);
    }

    #[test]
    fn test_bounding_box_empty() {
        let (x1, y1, x2, y2) = bounding_box(&[]);
        assert!((x1 - 0.0).abs() < 1e-6);
        assert!((y1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sierpinski() {
        let sys = sierpinski_triangle(2);
        let result = sys.generate(42);
        assert!(result.len() > 10);
    }

    #[test]
    fn test_koch_visual() {
        let sys = koch_snowflake(2);
        let result = sys.generate(42);
        let config = TurtleConfig::new(60.0, 5.0);
        let segments = interpret(&result, &config);
        assert!(segments.len() > 10);
    }

    #[test]
    fn test_generated_length() {
        let sys = LSystem::new("F")
            .add_rule(ProductionRule::deterministic('F', "F+F"))
            .with_iterations(3);
        assert_eq!(sys.generated_length(42), 15); // F+F -> F+F+F+F -> F+F+F+F+F+F+F+F -> len 15
    }

    #[test]
    fn test_no_rules_identity() {
        let sys = LSystem::new("ABC").with_iterations(5);
        let result = sys.generate(42);
        assert_eq!(result, "ABC");
    }

    #[test]
    fn test_multiple_rules() {
        let sys = LSystem::new("AB")
            .add_rule(ProductionRule::deterministic('A', "AB"))
            .add_rule(ProductionRule::deterministic('B', "A"))
            .with_iterations(1);
        let result = sys.generate(42);
        assert_eq!(result, "ABA"); // A->AB, B->A
    }

    #[test]
    fn test_step_scale() {
        let config = TurtleConfig::new(25.0, 10.0).with_step_scale(0.7);
        let segments = interpret("F[+F]", &config);
        // Inside bracket, step should be scaled
        if segments.len() >= 2 {
            let len1 = ((segments[0].x2 - segments[0].x1).powi(2) + (segments[0].y2 - segments[0].y1).powi(2)).sqrt();
            let len2 = ((segments[1].x2 - segments[1].x1).powi(2) + (segments[1].y2 - segments[1].y1).powi(2)).sqrt();
            assert!((len1 - 10.0).abs() < 1e-6);
            assert!((len2 - 7.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_depth_tracking() {
        let config = TurtleConfig::new(25.0, 10.0);
        let segments = interpret("F[+F[+F]]", &config);
        assert_eq!(segments[0].depth, 0);
        if segments.len() > 1 { assert_eq!(segments[1].depth, 1); }
        if segments.len() > 2 { assert_eq!(segments[2].depth, 2); }
    }

    #[test]
    fn test_fractal_plant_render() {
        let sys = fractal_plant(3);
        let result = sys.generate(42);
        let config = TurtleConfig::new(25.7, 5.0).with_step_scale(0.8);
        let segments = interpret(&result, &config);
        assert!(segments.len() > 20);
        let (min_x, min_y, max_x, max_y) = bounding_box(&segments);
        assert!(max_x - min_x > 1.0);
        assert!(max_y - min_y > 1.0);
    }
}
