//! Motion preference responder: prefers-reduced-motion detection, animation
//! toggle, alternative static representations, transition duration override,
//! scroll behavior override, and motion budget tracking.
//!
//! Pure data — no browser dependency. Tracks motion preferences and provides
//! a policy engine for animation decisions.

use std::collections::HashMap;

// ── Motion Preference ─────────────────────────────────────────

/// User motion preference (mirrors `prefers-reduced-motion`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionPreference {
    /// No preference — full animations.
    NoPreference,
    /// Reduce motion — minimize animations.
    Reduce,
}

// ── Animation State ───────────────────────────────────────────

/// Whether an animation is allowed, replaced, or suppressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationDecision {
    /// Play the full animation.
    Allow,
    /// Use a reduced/static alternative.
    Reduce,
    /// Suppress entirely (no visual change).
    Suppress,
}

// ── Scroll Behavior ───────────────────────────────────────────

/// Scroll behavior override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollBehavior {
    /// Smooth scrolling.
    Smooth,
    /// Instant jump (reduced motion).
    Instant,
    /// System default.
    Auto,
}

// ── Animation Entry ───────────────────────────────────────────

/// A registered animation with its alternatives.
#[derive(Debug, Clone)]
pub struct AnimationEntry {
    pub id: String,
    /// Base duration in milliseconds.
    pub duration_ms: u64,
    /// Whether this animation is essential (always plays regardless of pref).
    pub essential: bool,
    /// Alternative static representation (CSS class, description, etc.).
    pub alternative: Option<String>,
}

impl AnimationEntry {
    pub fn new(id: &str, duration_ms: u64) -> Self {
        Self {
            id: id.into(),
            duration_ms,
            essential: false,
            alternative: None,
        }
    }

    pub fn with_essential(mut self, essential: bool) -> Self {
        self.essential = essential;
        self
    }

    pub fn with_alternative(mut self, alt: &str) -> Self {
        self.alternative = Some(alt.into());
        self
    }
}

// ── Motion Budget ─────────────────────────────────────────────

/// Tracks cumulative animation "cost" to enforce a motion budget.
#[derive(Debug, Clone)]
pub struct MotionBudget {
    /// Maximum total animation milliseconds allowed per interaction.
    pub max_ms: u64,
    /// Currently consumed milliseconds.
    pub consumed_ms: u64,
}

impl MotionBudget {
    pub fn new(max_ms: u64) -> Self {
        Self { max_ms, consumed_ms: 0 }
    }

    /// Try to consume duration. Returns true if within budget.
    pub fn try_consume(&mut self, duration_ms: u64) -> bool {
        if self.consumed_ms + duration_ms <= self.max_ms {
            self.consumed_ms += duration_ms;
            true
        } else {
            false
        }
    }

    /// Remaining budget.
    pub fn remaining_ms(&self) -> u64 {
        self.max_ms.saturating_sub(self.consumed_ms)
    }

    /// Reset the budget (e.g., after an interaction completes).
    pub fn reset(&mut self) {
        self.consumed_ms = 0;
    }

    /// Whether the budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.consumed_ms >= self.max_ms
    }
}

// ── Motion Policy ─────────────────────────────────────────────

/// Central motion policy engine.
#[derive(Debug)]
pub struct MotionPolicy {
    preference: MotionPreference,
    animations: HashMap<String, AnimationEntry>,
    /// Duration multiplier (0.0 = instant, 1.0 = full speed).
    duration_scale: f64,
    scroll_behavior: ScrollBehavior,
    budget: Option<MotionBudget>,
    /// Global enable/disable toggle.
    enabled: bool,
}

impl MotionPolicy {
    pub fn new() -> Self {
        Self {
            preference: MotionPreference::NoPreference,
            animations: HashMap::new(),
            duration_scale: 1.0,
            scroll_behavior: ScrollBehavior::Auto,
            budget: None,
            enabled: true,
        }
    }

    /// Set the user's motion preference.
    pub fn set_preference(&mut self, pref: MotionPreference) {
        self.preference = pref;
        match pref {
            MotionPreference::Reduce => {
                self.duration_scale = 0.0;
                self.scroll_behavior = ScrollBehavior::Instant;
            }
            MotionPreference::NoPreference => {
                self.duration_scale = 1.0;
                self.scroll_behavior = ScrollBehavior::Auto;
            }
        }
    }

    pub fn preference(&self) -> MotionPreference {
        self.preference
    }

    /// Override the duration scale (0.0–1.0).
    pub fn set_duration_scale(&mut self, scale: f64) {
        self.duration_scale = scale.clamp(0.0, 1.0);
    }

    pub fn duration_scale(&self) -> f64 {
        self.duration_scale
    }

    /// Override scroll behavior.
    pub fn set_scroll_behavior(&mut self, behavior: ScrollBehavior) {
        self.scroll_behavior = behavior;
    }

    pub fn scroll_behavior(&self) -> ScrollBehavior {
        self.scroll_behavior
    }

    /// Enable/disable all animations.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set a motion budget.
    pub fn set_budget(&mut self, budget: MotionBudget) {
        self.budget = Some(budget);
    }

    /// Reset the motion budget.
    pub fn reset_budget(&mut self) {
        if let Some(b) = &mut self.budget {
            b.reset();
        }
    }

    /// Register an animation.
    pub fn register(&mut self, entry: AnimationEntry) {
        self.animations.insert(entry.id.clone(), entry);
    }

    /// Decide whether/how to play a specific animation.
    pub fn decide(&mut self, animation_id: &str) -> AnimationDecision {
        if !self.enabled {
            return AnimationDecision::Suppress;
        }
        let entry = match self.animations.get(animation_id) {
            Some(e) => e.clone(),
            None => return AnimationDecision::Allow, // unknown animation — allow by default
        };

        // Essential animations always play.
        if entry.essential {
            if let Some(b) = &mut self.budget {
                b.try_consume(entry.duration_ms);
            }
            return AnimationDecision::Allow;
        }

        // Reduced motion preference.
        if self.preference == MotionPreference::Reduce {
            return if entry.alternative.is_some() {
                AnimationDecision::Reduce
            } else {
                AnimationDecision::Suppress
            };
        }

        // Budget check.
        if let Some(b) = &mut self.budget {
            let scaled = (entry.duration_ms as f64 * self.duration_scale) as u64;
            if !b.try_consume(scaled) {
                return AnimationDecision::Suppress;
            }
        }

        AnimationDecision::Allow
    }

    /// Get the effective duration for an animation (applying scale).
    pub fn effective_duration_ms(&self, animation_id: &str) -> u64 {
        if !self.enabled || self.preference == MotionPreference::Reduce {
            return 0;
        }
        match self.animations.get(animation_id) {
            Some(entry) => (entry.duration_ms as f64 * self.duration_scale) as u64,
            None => 0,
        }
    }

    /// Get the alternative representation for a reduced animation.
    pub fn alternative(&self, animation_id: &str) -> Option<&str> {
        self.animations
            .get(animation_id)
            .and_then(|e| e.alternative.as_deref())
    }

    /// Generate a CSS `@media` block for reduced motion.
    pub fn reduced_motion_css(&self) -> String {
        "@media (prefers-reduced-motion: reduce) {\n  \
         *, *::before, *::after {\n    \
           animation-duration: 0.01ms !important;\n    \
           animation-iteration-count: 1 !important;\n    \
           transition-duration: 0.01ms !important;\n    \
           scroll-behavior: auto !important;\n  \
         }\n\
         }".into()
    }
}

impl Default for MotionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_allows_all() {
        let mut policy = MotionPolicy::new();
        policy.register(AnimationEntry::new("fade", 300));
        assert_eq!(policy.decide("fade"), AnimationDecision::Allow);
    }

    #[test]
    fn reduced_motion_suppresses() {
        let mut policy = MotionPolicy::new();
        policy.register(AnimationEntry::new("fade", 300));
        policy.set_preference(MotionPreference::Reduce);
        assert_eq!(policy.decide("fade"), AnimationDecision::Suppress);
    }

    #[test]
    fn reduced_with_alternative() {
        let mut policy = MotionPolicy::new();
        policy.register(AnimationEntry::new("slide", 500).with_alternative("fade-static"));
        policy.set_preference(MotionPreference::Reduce);
        assert_eq!(policy.decide("slide"), AnimationDecision::Reduce);
    }

    #[test]
    fn essential_always_plays() {
        let mut policy = MotionPolicy::new();
        policy.register(AnimationEntry::new("loading", 200).with_essential(true));
        policy.set_preference(MotionPreference::Reduce);
        assert_eq!(policy.decide("loading"), AnimationDecision::Allow);
    }

    #[test]
    fn disabled_suppresses_all() {
        let mut policy = MotionPolicy::new();
        policy.register(AnimationEntry::new("fade", 300).with_essential(true));
        policy.set_enabled(false);
        assert_eq!(policy.decide("fade"), AnimationDecision::Suppress);
    }

    #[test]
    fn duration_scale() {
        let mut policy = MotionPolicy::new();
        policy.register(AnimationEntry::new("fade", 300));
        policy.set_duration_scale(0.5);
        assert_eq!(policy.effective_duration_ms("fade"), 150);
    }

    #[test]
    fn reduced_duration_zero() {
        let mut policy = MotionPolicy::new();
        policy.register(AnimationEntry::new("fade", 300));
        policy.set_preference(MotionPreference::Reduce);
        assert_eq!(policy.effective_duration_ms("fade"), 0);
    }

    #[test]
    fn scroll_behavior_reduced() {
        let mut policy = MotionPolicy::new();
        policy.set_preference(MotionPreference::Reduce);
        assert_eq!(policy.scroll_behavior(), ScrollBehavior::Instant);
    }

    #[test]
    fn motion_budget_basic() {
        let mut budget = MotionBudget::new(500);
        assert!(budget.try_consume(200));
        assert_eq!(budget.remaining_ms(), 300);
        assert!(budget.try_consume(300));
        assert!(budget.is_exhausted());
        assert!(!budget.try_consume(1));
    }

    #[test]
    fn motion_budget_reset() {
        let mut budget = MotionBudget::new(100);
        budget.try_consume(100);
        budget.reset();
        assert_eq!(budget.remaining_ms(), 100);
    }

    #[test]
    fn budget_enforcement() {
        let mut policy = MotionPolicy::new();
        policy.set_budget(MotionBudget::new(300));
        policy.register(AnimationEntry::new("a", 200));
        policy.register(AnimationEntry::new("b", 200));
        assert_eq!(policy.decide("a"), AnimationDecision::Allow);
        assert_eq!(policy.decide("b"), AnimationDecision::Suppress);
    }

    #[test]
    fn reduced_motion_css_output() {
        let policy = MotionPolicy::new();
        let css = policy.reduced_motion_css();
        assert!(css.contains("prefers-reduced-motion: reduce"));
        assert!(css.contains("animation-duration: 0.01ms"));
    }

    #[test]
    fn alternative_lookup() {
        let mut policy = MotionPolicy::new();
        policy.register(AnimationEntry::new("slide", 300).with_alternative("slide-static"));
        assert_eq!(policy.alternative("slide"), Some("slide-static"));
        assert_eq!(policy.alternative("nonexistent"), None);
    }

    #[test]
    fn preference_toggle() {
        let mut policy = MotionPolicy::new();
        policy.set_preference(MotionPreference::Reduce);
        assert_eq!(policy.duration_scale(), 0.0);
        policy.set_preference(MotionPreference::NoPreference);
        assert_eq!(policy.duration_scale(), 1.0);
    }
}
