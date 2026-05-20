//! Tier Auto-Selector — picks the cheapest tier that satisfies constraints.

use super::tier::{InferenceTier, TierConstraints};
use std::time::Duration;

/// Query complexity classification (heuristic, no ML).
#[derive(Debug, Clone, Copy)]
pub struct ComplexityScore(pub f32); // 0.0 = trivial, 1.0 = complex

/// Hardware profile for tier availability checking.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub has_gpu: bool,
    pub has_npu: bool,
    pub has_api_keys: bool,
    pub is_wasm: bool,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            has_gpu: false,
            has_npu: false,
            has_api_keys: false,
            is_wasm: cfg!(target_arch = "wasm32"),
        }
    }
}

/// Which tiers are available at runtime.
#[derive(Debug, Clone)]
pub struct TierAvailability {
    pub holographic: bool,   // Always true
    pub embedded: bool,      // Feature flag + model loaded
    pub local: bool,         // Feature flag + GPU available
    pub frontier: bool,      // Feature flag + API keys configured
}

impl TierAvailability {
    pub fn holographic_only() -> Self {
        Self {
            holographic: true,
            embedded: false,
            local: false,
            frontier: false,
        }
    }

    pub fn all() -> Self {
        Self {
            holographic: true,
            embedded: true,
            local: true,
            frontier: true,
        }
    }

    pub fn is_available(&self, tier: InferenceTier) -> bool {
        match tier {
            InferenceTier::Holographic => self.holographic,
            InferenceTier::Embedded => self.embedded,
            InferenceTier::Local => self.local,
            InferenceTier::Frontier => self.frontier,
        }
    }
}

/// Select the optimal inference tier for a query.
pub fn select_tier(
    complexity: ComplexityScore,
    constraints: &TierConstraints,
    availability: &TierAvailability,
    energy_remaining: Option<f64>,
) -> InferenceTier {
    // All tiers in cost order
    let all_tiers = [
        InferenceTier::Holographic,
        InferenceTier::Embedded,
        InferenceTier::Local,
        InferenceTier::Frontier,
    ];

    // Filter to available + allowed tiers
    let candidates: Vec<InferenceTier> = all_tiers
        .iter()
        .filter(|&&t| availability.is_available(t))
        .filter(|&&t| {
            constraints
                .allowed_tiers
                .as_ref()
                .map(|allowed| allowed.contains(&t))
                .unwrap_or(true)
        })
        .copied()
        .collect();

    if candidates.is_empty() {
        return InferenceTier::Holographic; // Always available as fallback
    }

    // Determine minimum tier from complexity
    let min_tier = complexity_to_min_tier(complexity);

    // Filter by latency constraint
    let candidates: Vec<InferenceTier> = candidates
        .into_iter()
        .filter(|t| {
            constraints
                .max_latency
                .map(|max| estimated_latency(*t) <= max)
                .unwrap_or(true)
        })
        .collect();

    if candidates.is_empty() {
        return InferenceTier::Holographic;
    }

    // Filter by energy constraint
    let candidates: Vec<InferenceTier> = candidates
        .into_iter()
        .filter(|t| {
            constraints
                .max_energy_joules
                .map(|max| estimated_energy(*t) <= max)
                .unwrap_or(true)
        })
        .collect();

    if candidates.is_empty() {
        return InferenceTier::Holographic;
    }

    // If energy budget is low, force cheapest
    if let Some(remaining) = energy_remaining {
        if remaining < 0.01 {
            // < 10mJ remaining — holographic only
            return InferenceTier::Holographic;
        }
    }

    // Select cheapest tier >= min_tier
    if constraints.prefer_cheapest {
        candidates
            .into_iter()
            .find(|&t| t >= min_tier)
            .unwrap_or(InferenceTier::Holographic)
    } else {
        // Prefer strongest available
        candidates
            .into_iter()
            .rev()
            .find(|&t| t >= min_tier)
            .unwrap_or(InferenceTier::Holographic)
    }
}

/// Map complexity score to minimum capable tier.
fn complexity_to_min_tier(c: ComplexityScore) -> InferenceTier {
    match c.0 {
        x if x <= 0.3 => InferenceTier::Holographic,
        x if x <= 0.5 => InferenceTier::Embedded,
        x if x <= 0.8 => InferenceTier::Local,
        _ => InferenceTier::Frontier,
    }
}

/// Estimate latency for a tier (p50, rough).
fn estimated_latency(tier: InferenceTier) -> Duration {
    match tier {
        InferenceTier::Holographic => Duration::from_micros(1),
        InferenceTier::Embedded => Duration::from_millis(10),
        InferenceTier::Local => Duration::from_secs(1),
        InferenceTier::Frontier => Duration::from_secs(2),
    }
}

/// Estimate energy for a tier (single operation, rough).
fn estimated_energy(tier: InferenceTier) -> f64 {
    match tier {
        InferenceTier::Holographic => 0.000_000_2, // 0.2 µJ
        InferenceTier::Embedded => 0.002,           // 2 mJ
        InferenceTier::Local => 1.0,                // 1 J
        InferenceTier::Frontier => 0.5,             // 0.5 J (modeled)
    }
}

/// Classify query complexity using lightweight heuristics.
pub fn classify_complexity(query: &str) -> ComplexityScore {
    let lower = query.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    let mut score = 0.0f32;

    // Length-based baseline
    score += (words.len() as f32 / 50.0).min(0.3);

    // Question indicators
    if lower.contains('?') || lower.starts_with("what") || lower.starts_with("why")
        || lower.starts_with("how") || lower.starts_with("explain")
    {
        score += 0.3;
    }

    // Generation indicators
    if lower.contains("generate") || lower.contains("create") || lower.contains("write")
        || lower.contains("compose") || lower.contains("draft")
    {
        score += 0.4;
    }

    // Summarization/analysis indicators
    if lower.contains("summarize") || lower.contains("analyze") || lower.contains("compare")
        || lower.contains("evaluate")
    {
        score += 0.3;
    }

    // Reasoning indicators
    if lower.contains("reason") || lower.contains("because") || lower.contains("therefore")
        || lower.contains("step by step")
    {
        score += 0.4;
    }

    // Simple operations (lower complexity)
    if lower.starts_with("find") || lower.starts_with("search") || lower.starts_with("show")
        || lower.starts_with("list") || lower.starts_with("get")
    {
        score -= 0.1;
    }

    // Similarity keywords (pure holographic)
    if lower.contains("similar") || lower.contains("like") || lower.contains("match")
        || lower.contains("nearest")
    {
        score -= 0.2;
    }

    ComplexityScore(score.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_classification() {
        // Simple queries → low complexity
        assert!(classify_complexity("find movies similar to Inception").0 < 0.3);
        assert!(classify_complexity("show me trending content").0 < 0.4);

        // Complex queries → higher complexity than simple ones
        assert!(classify_complexity("explain why user engagement dropped last month").0 > 0.3);
        assert!(classify_complexity("generate a marketing summary for Q1").0 > 0.5);
    }

    #[test]
    fn test_tier_selection_basic() {
        let avail = TierAvailability::all();
        let constraints = TierConstraints::default();

        // Low complexity → Tier 1
        let tier = select_tier(ComplexityScore(0.1), &constraints, &avail, None);
        assert_eq!(tier, InferenceTier::Holographic);

        // High complexity → Tier 3 or 4
        let tier = select_tier(ComplexityScore(0.9), &constraints, &avail, None);
        assert!(tier >= InferenceTier::Local);
    }

    #[test]
    fn test_tier_selection_constrained() {
        let avail = TierAvailability::all();

        // Latency constraint forces Tier 1
        let constraints = TierConstraints::default()
            .with_max_latency(Duration::from_millis(1));
        let tier = select_tier(ComplexityScore(0.9), &constraints, &avail, None);
        assert_eq!(tier, InferenceTier::Holographic);
    }

    #[test]
    fn test_tier_selection_limited_availability() {
        let avail = TierAvailability::holographic_only();
        let constraints = TierConstraints::default();

        // Even complex queries fall back to Tier 1 if nothing else available
        let tier = select_tier(ComplexityScore(0.9), &constraints, &avail, None);
        assert_eq!(tier, InferenceTier::Holographic);
    }

    #[test]
    fn test_tier_selection_energy_budget() {
        let avail = TierAvailability::all();
        let constraints = TierConstraints::default();

        // Nearly exhausted energy budget → force Tier 1
        let tier = select_tier(ComplexityScore(0.9), &constraints, &avail, Some(0.001));
        assert_eq!(tier, InferenceTier::Holographic);
    }

    #[test]
    fn test_holographic_only_constraint() {
        let avail = TierAvailability::all();
        let constraints = TierConstraints::holographic_only();

        let tier = select_tier(ComplexityScore(0.9), &constraints, &avail, None);
        assert_eq!(tier, InferenceTier::Holographic);
    }
}
