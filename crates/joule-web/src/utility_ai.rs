//! Utility-based AI decision making — actions scored by response curves
//! (linear, quadratic, logistic, step, bell) applied to world state inputs.
//! Input normalization (0.0-1.0), action scoring (multiply normalized outputs),
//! dual utility selection, inertia/momentum, contextual scoring (filter then score).
//!
//! Replaces JavaScript utility AI libraries with a pure-Rust decision system
//! for game characters and NPCs.

// ── Response curves ─────────────────────────────────────────────

/// A response curve that maps a normalized input (0.0-1.0) to a score (0.0-1.0).
#[derive(Debug, Clone, PartialEq)]
pub enum Curve {
    /// Linear: y = slope * x + intercept, clamped to [0,1].
    Linear { slope: f64, intercept: f64 },

    /// Quadratic: y = a * x^2 + b * x + c, clamped to [0,1].
    Quadratic { a: f64, b: f64, c: f64 },

    /// Logistic sigmoid: y = 1 / (1 + exp(-k * (x - midpoint))), clamped to [0,1].
    Logistic { k: f64, midpoint: f64 },

    /// Step function: y = 0 if x < threshold, 1 if x >= threshold.
    Step { threshold: f64 },

    /// Bell curve (Gaussian): y = exp(-((x - center)^2) / (2 * width^2)), clamped to [0,1].
    Bell { center: f64, width: f64 },
}

impl Curve {
    /// Evaluate the curve at a given input value (0.0-1.0).
    pub fn evaluate(&self, x: f64) -> f64 {
        let raw = match self {
            Curve::Linear { slope, intercept } => slope * x + intercept,
            Curve::Quadratic { a, b, c } => a * x * x + b * x + c,
            Curve::Logistic { k, midpoint } => 1.0 / (1.0 + (-k * (x - midpoint)).exp()),
            Curve::Step { threshold } => if x >= *threshold { 1.0 } else { 0.0 },
            Curve::Bell { center, width } => {
                if width.abs() < 1e-12 {
                    if (x - center).abs() < 1e-12 { 1.0 } else { 0.0 }
                } else {
                    (-((x - center).powi(2)) / (2.0 * width * width)).exp()
                }
            }
        };
        raw.clamp(0.0, 1.0)
    }
}

// ── Input normalization ─────────────────────────────────────────

/// An input axis that normalizes a raw world value to 0.0-1.0.
#[derive(Debug, Clone, PartialEq)]
pub struct InputAxis {
    pub name: String,
    pub min_value: f64,
    pub max_value: f64,
}

impl InputAxis {
    pub fn new(name: &str, min_value: f64, max_value: f64) -> Self {
        Self {
            name: name.to_string(),
            min_value,
            max_value,
        }
    }

    /// Normalize a raw value to 0.0-1.0.
    pub fn normalize(&self, raw: f64) -> f64 {
        let range = self.max_value - self.min_value;
        if range.abs() < 1e-12 {
            return 0.5;
        }
        ((raw - self.min_value) / range).clamp(0.0, 1.0)
    }
}

// ── Consideration (input + curve) ───────────────────────────────

/// A consideration: an input axis evaluated through a response curve.
#[derive(Debug, Clone, PartialEq)]
pub struct Consideration {
    pub input_name: String,
    pub curve: Curve,
}

impl Consideration {
    pub fn new(input_name: &str, curve: Curve) -> Self {
        Self { input_name: input_name.to_string(), curve }
    }

    /// Evaluate this consideration given normalized inputs.
    pub fn evaluate(&self, inputs: &WorldState) -> f64 {
        let input_val = inputs.get(&self.input_name).unwrap_or(0.0);
        self.curve.evaluate(input_val)
    }
}

// ── Action ──────────────────────────────────────────────────────

/// An AI action with its considerations.
#[derive(Debug, Clone, PartialEq)]
pub struct UtilityAction {
    pub name: String,
    pub considerations: Vec<Consideration>,
    /// Base weight multiplier.
    pub weight: f64,
    /// Minimum score threshold (action is discarded below this).
    pub min_threshold: f64,
}

impl UtilityAction {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            considerations: Vec::new(),
            weight: 1.0,
            min_threshold: 0.0,
        }
    }

    pub fn with_consideration(mut self, input_name: &str, curve: Curve) -> Self {
        self.considerations.push(Consideration::new(input_name, curve));
        self
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.min_threshold = threshold;
        self
    }

    /// Score this action against normalized world inputs.
    /// Uses multiplicative scoring with compensation factor.
    pub fn score(&self, inputs: &WorldState) -> f64 {
        if self.considerations.is_empty() {
            return self.weight;
        }

        let n = self.considerations.len() as f64;
        let mut product = 1.0;

        for c in &self.considerations {
            let val = c.evaluate(inputs);
            if val < 1e-10 {
                return 0.0; // short-circuit: any zero kills the score
            }
            product *= val;
        }

        // Compensation factor (Dave Mark's formula):
        // Adjust for number of considerations so more considerations
        // don't unfairly lower the score.
        let modification = 1.0 - (1.0 / n);
        let make_up = (1.0 - product) * modification;
        let final_score = product + (make_up * product);

        final_score * self.weight
    }
}

// ── World state (normalized inputs) ─────────────────────────────

/// World state: collection of normalized input values.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldState {
    values: std::collections::HashMap<String, f64>,
}

impl WorldState {
    pub fn new() -> Self {
        Self { values: std::collections::HashMap::new() }
    }

    /// Set a normalized input value.
    pub fn set(&mut self, name: &str, value: f64) {
        self.values.insert(name.to_string(), value.clamp(0.0, 1.0));
    }

    /// Set a raw value through an input axis.
    pub fn set_raw(&mut self, axis: &InputAxis, raw_value: f64) {
        let normalized = axis.normalize(raw_value);
        self.values.insert(axis.name.clone(), normalized);
    }

    /// Get a normalized input value.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.values.get(name).copied()
    }
}

// ── Decision engine ─────────────────────────────────────────────

/// Result of action selection.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub action_name: String,
    pub score: f64,
}

/// Score all actions and return them sorted by score (highest first).
pub fn score_actions(actions: &[UtilityAction], inputs: &WorldState) -> Vec<Decision> {
    let mut scored: Vec<Decision> = actions.iter()
        .map(|a| Decision {
            action_name: a.name.clone(),
            score: a.score(inputs),
        })
        .filter(|d| {
            let action = actions.iter().find(|a| a.name == d.action_name).unwrap();
            d.score >= action.min_threshold
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

/// Select the highest-scoring action (dual utility: best of the rest).
pub fn select_best(actions: &[UtilityAction], inputs: &WorldState) -> Option<Decision> {
    score_actions(actions, inputs).into_iter().next()
}

// ── Inertia / momentum ─────────────────────────────────────────

/// Apply inertia bonus to prevent oscillation between actions.
/// `current_action`: name of currently executing action
/// `inertia_bonus`: bonus score multiplier for current action (e.g. 1.2 = 20% bonus)
pub fn select_with_inertia(
    actions: &[UtilityAction],
    inputs: &WorldState,
    current_action: Option<&str>,
    inertia_bonus: f64,
) -> Option<Decision> {
    let mut scored = score_actions(actions, inputs);

    if let Some(current) = current_action {
        for d in &mut scored {
            if d.action_name == current {
                d.score *= inertia_bonus;
            }
        }
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }

    scored.into_iter().next()
}

// ── Contextual scoring ──────────────────────────────────────────

/// Context filter: reduce the action set before scoring.
pub fn filter_actions<'a>(
    actions: &'a [UtilityAction],
    inputs: &WorldState,
    context_fn: impl Fn(&UtilityAction, &WorldState) -> bool,
) -> Vec<&'a UtilityAction> {
    actions.iter()
        .filter(|a| context_fn(a, inputs))
        .collect()
}

/// Contextual select: filter actions then pick the best.
pub fn contextual_select(
    actions: &[UtilityAction],
    inputs: &WorldState,
    context_fn: impl Fn(&UtilityAction, &WorldState) -> bool,
) -> Option<Decision> {
    let filtered: Vec<UtilityAction> = actions.iter()
        .filter(|a| context_fn(a, inputs))
        .cloned()
        .collect();
    select_best(&filtered, inputs)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_curve() {
        let c = Curve::Linear { slope: 1.0, intercept: 0.0 };
        assert!((c.evaluate(0.0) - 0.0).abs() < 1e-6);
        assert!((c.evaluate(0.5) - 0.5).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_curve_clamped() {
        let c = Curve::Linear { slope: 2.0, intercept: 0.0 };
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6); // 2*1 clamped to 1
    }

    #[test]
    fn test_quadratic_curve() {
        let c = Curve::Quadratic { a: 1.0, b: 0.0, c: 0.0 };
        assert!((c.evaluate(0.5) - 0.25).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_logistic_curve() {
        let c = Curve::Logistic { k: 10.0, midpoint: 0.5 };
        let at_mid = c.evaluate(0.5);
        assert!((at_mid - 0.5).abs() < 1e-6);
        assert!(c.evaluate(1.0) > 0.9);
        assert!(c.evaluate(0.0) < 0.1);
    }

    #[test]
    fn test_step_curve() {
        let c = Curve::Step { threshold: 0.5 };
        assert!((c.evaluate(0.3) - 0.0).abs() < 1e-6);
        assert!((c.evaluate(0.5) - 1.0).abs() < 1e-6);
        assert!((c.evaluate(0.8) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bell_curve() {
        let c = Curve::Bell { center: 0.5, width: 0.2 };
        let peak = c.evaluate(0.5);
        assert!((peak - 1.0).abs() < 1e-6);
        let off = c.evaluate(0.0);
        assert!(off < 0.1);
    }

    #[test]
    fn test_bell_zero_width() {
        let c = Curve::Bell { center: 0.5, width: 0.0 };
        assert!((c.evaluate(0.5) - 1.0).abs() < 1e-6);
        assert!((c.evaluate(0.3) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_input_axis_normalize() {
        let axis = InputAxis::new("hp", 0.0, 100.0);
        assert!((axis.normalize(50.0) - 0.5).abs() < 1e-6);
        assert!((axis.normalize(0.0) - 0.0).abs() < 1e-6);
        assert!((axis.normalize(100.0) - 1.0).abs() < 1e-6);
        assert!((axis.normalize(150.0) - 1.0).abs() < 1e-6); // clamped
    }

    #[test]
    fn test_input_axis_zero_range() {
        let axis = InputAxis::new("x", 5.0, 5.0);
        assert!((axis.normalize(5.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_world_state() {
        let mut ws = WorldState::new();
        ws.set("hp", 0.8);
        assert!((ws.get("hp").unwrap() - 0.8).abs() < 1e-6);
        assert!(ws.get("missing").is_none());
    }

    #[test]
    fn test_world_state_raw() {
        let axis = InputAxis::new("hp", 0.0, 100.0);
        let mut ws = WorldState::new();
        ws.set_raw(&axis, 75.0);
        assert!((ws.get("hp").unwrap() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_action_single_consideration() {
        let action = UtilityAction::new("attack")
            .with_consideration("enemy_dist", Curve::Linear { slope: -1.0, intercept: 1.0 });
        let mut ws = WorldState::new();
        ws.set("enemy_dist", 0.2); // close enemy = high score
        let score = action.score(&ws);
        assert!(score > 0.5);
    }

    #[test]
    fn test_action_zero_consideration_kills_score() {
        let action = UtilityAction::new("attack")
            .with_consideration("ammo", Curve::Linear { slope: 1.0, intercept: 0.0 })
            .with_consideration("hp", Curve::Linear { slope: 1.0, intercept: 0.0 });
        let mut ws = WorldState::new();
        ws.set("ammo", 0.0); // no ammo
        ws.set("hp", 1.0);
        assert!((action.score(&ws) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_action_weight() {
        let action = UtilityAction::new("heal")
            .with_consideration("hp", Curve::Linear { slope: -1.0, intercept: 1.0 })
            .with_weight(2.0);
        let mut ws = WorldState::new();
        ws.set("hp", 0.0); // low hp
        let score = action.score(&ws);
        assert!(score > 1.0); // weighted above 1
    }

    #[test]
    fn test_action_threshold() {
        let action = UtilityAction::new("flee")
            .with_consideration("hp", Curve::Linear { slope: -1.0, intercept: 1.0 })
            .with_threshold(0.5);
        let mut ws = WorldState::new();
        ws.set("hp", 0.8); // high hp -> low flee score
        let scored = score_actions(&[action], &ws);
        // Score is ~0.2, below threshold of 0.5 -> filtered out
        assert!(scored.is_empty());
    }

    #[test]
    fn test_select_best() {
        let actions = vec![
            UtilityAction::new("attack")
                .with_consideration("aggro", Curve::Linear { slope: 1.0, intercept: 0.0 }),
            UtilityAction::new("hide")
                .with_consideration("aggro", Curve::Linear { slope: -1.0, intercept: 1.0 }),
        ];
        let mut ws = WorldState::new();
        ws.set("aggro", 0.9); // high aggression
        let best = select_best(&actions, &ws);
        assert_eq!(best.unwrap().action_name, "attack");
    }

    #[test]
    fn test_inertia_keeps_current() {
        let actions = vec![
            UtilityAction::new("patrol")
                .with_consideration("threat", Curve::Linear { slope: -1.0, intercept: 1.0 }),
            UtilityAction::new("attack")
                .with_consideration("threat", Curve::Linear { slope: 1.0, intercept: 0.0 }),
        ];
        let mut ws = WorldState::new();
        ws.set("threat", 0.48); // slightly below 0.5 — attack marginally loses

        // Without inertia, patrol wins
        let normal = select_best(&actions, &ws).unwrap();
        assert_eq!(normal.action_name, "patrol");

        // With inertia for attack, attack should win
        let inertia = select_with_inertia(&actions, &ws, Some("attack"), 1.3).unwrap();
        assert_eq!(inertia.action_name, "attack");
    }

    #[test]
    fn test_contextual_filter() {
        let actions = vec![
            UtilityAction::new("melee")
                .with_consideration("dist", Curve::Linear { slope: -1.0, intercept: 1.0 }),
            UtilityAction::new("ranged")
                .with_consideration("dist", Curve::Linear { slope: 1.0, intercept: 0.0 }),
        ];
        let ws = WorldState::new();

        // Filter to only "ranged" actions
        let filtered = filter_actions(&actions, &ws, |a, _| a.name.contains("ranged"));
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "ranged");
    }

    #[test]
    fn test_contextual_select() {
        let actions = vec![
            UtilityAction::new("swim")
                .with_consideration("water", Curve::Step { threshold: 0.5 }),
            UtilityAction::new("walk")
                .with_consideration("water", Curve::Linear { slope: -1.0, intercept: 1.0 }),
        ];
        let mut ws = WorldState::new();
        ws.set("water", 0.8);

        // Only allow actions for water context
        let decision = contextual_select(&actions, &ws, |a, _ws| a.name == "swim");
        assert_eq!(decision.unwrap().action_name, "swim");
    }

    #[test]
    fn test_no_considerations() {
        let action = UtilityAction::new("idle").with_weight(0.5);
        let ws = WorldState::new();
        assert!((action.score(&ws) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_score_actions_ordering() {
        let actions = vec![
            UtilityAction::new("low")
                .with_consideration("x", Curve::Linear { slope: 0.2, intercept: 0.0 }),
            UtilityAction::new("high")
                .with_consideration("x", Curve::Linear { slope: 1.0, intercept: 0.0 }),
        ];
        let mut ws = WorldState::new();
        ws.set("x", 0.8);
        let scored = score_actions(&actions, &ws);
        assert_eq!(scored[0].action_name, "high");
    }

    #[test]
    fn test_consideration_evaluate() {
        let c = Consideration::new("hp", Curve::Linear { slope: 1.0, intercept: 0.0 });
        let mut ws = WorldState::new();
        ws.set("hp", 0.7);
        let score = c.evaluate(&ws);
        assert!((score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_consideration_missing_input() {
        let c = Consideration::new("missing", Curve::Linear { slope: 1.0, intercept: 0.0 });
        let ws = WorldState::new();
        let score = c.evaluate(&ws);
        assert!((score - 0.0).abs() < 1e-6); // defaults to 0
    }
}
