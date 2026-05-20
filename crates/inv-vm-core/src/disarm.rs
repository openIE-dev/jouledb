use std::fmt;

use serde::{Deserialize, Serialize};

/// Hint for which mesh tier can serve a given DISARM level.
/// Kept here (in inv-core) to avoid circular dependency on inv-mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshTierHint {
    Leaf,
    Gateway,
    Backbone,
}

impl fmt::Display for MeshTierHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leaf => write!(f, "leaf"),
            Self::Gateway => write!(f, "gateway"),
            Self::Backbone => write!(f, "backbone"),
        }
    }
}

/// DISARM complexity level for AI inference queries.
///
/// L0-L3 classify queries by computational cost so the scheduler can route
/// them to the cheapest capable tier — ARM leaf nodes for lookups, GPUs only
/// for genuine reasoning.
///
/// Reference: <https://askdavidc.ai/disarm.html>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisarmLevel {
    /// L0 — Lookup: dictionary, conversion, simple facts.
    L0,
    /// L1 — Extraction: pull specific data from small context.
    L1,
    /// L2 — Aggregation: compare, summarise, multi-step over medium context.
    L2,
    /// L3 — Reasoning: open-ended generation, creative, large context.
    L3,
}

impl DisarmLevel {
    /// All levels in ascending order.
    pub fn all() -> &'static [DisarmLevel] {
        &[Self::L0, Self::L1, Self::L2, Self::L3]
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::L0 => "Lookup",
            Self::L1 => "Extraction",
            Self::L2 => "Aggregation",
            Self::L3 => "Reasoning",
        }
    }

    /// Reference energy cost in joules for a single query at this level.
    pub fn reference_energy_j(&self) -> f64 {
        match self {
            Self::L0 => 0.01,
            Self::L1 => 0.05,
            Self::L2 => 0.3,
            Self::L3 => 6.0,
        }
    }

    /// Reference energy cost in micro-joules.
    pub fn reference_energy_uj(&self) -> u64 {
        (self.reference_energy_j() * 1_000_000.0) as u64
    }

    /// Whether this level requires GPU hardware.
    pub fn requires_gpu(&self) -> bool {
        matches!(self, Self::L3)
    }

    /// Mesh tiers eligible to serve this level (cheapest first).
    pub fn eligible_tiers(&self) -> &'static [MeshTierHint] {
        match self {
            Self::L0 => &[
                MeshTierHint::Leaf,
                MeshTierHint::Gateway,
                MeshTierHint::Backbone,
            ],
            Self::L1 => &[
                MeshTierHint::Leaf,
                MeshTierHint::Gateway,
                MeshTierHint::Backbone,
            ],
            Self::L2 => &[MeshTierHint::Gateway, MeshTierHint::Backbone],
            Self::L3 => &[MeshTierHint::Backbone],
        }
    }

    /// Preferred (cheapest capable) tier.
    pub fn preferred_tier(&self) -> MeshTierHint {
        self.eligible_tiers()[0]
    }

    /// Parse a DISARM level from a string like "L0", "l1", "L2", "L3".
    /// Returns `L1` for unrecognized strings (the most common inference tier).
    pub fn parse(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "L0" => Self::L0,
            "L1" => Self::L1,
            "L2" => Self::L2,
            "L3" => Self::L3,
            _ => Self::L1,
        }
    }

    /// Numeric index (0-3) for scoring and array indexing.
    pub fn index(&self) -> usize {
        match self {
            Self::L0 => 0,
            Self::L1 => 1,
            Self::L2 => 2,
            Self::L3 => 3,
        }
    }
}

impl fmt::Display for DisarmLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{} ({})", self.index(), self.name())
    }
}

/// Classify a query into a DISARM level using lightweight heuristics.
///
/// This is intentionally simple — a production classifier would use
/// the full cognitive cascade from ask-davidc. Here we use keyword
/// patterns and context size to give the scheduler a fast first pass.
pub fn classify_query(query: &str, context_token_count: usize) -> DisarmLevel {
    let q = query.to_lowercase();

    // L0 patterns: simple lookups, definitions, conversions
    const L0_PATTERNS: &[&str] = &[
        "what is",
        "define",
        "convert",
        "calculate",
        "how many",
        "what does",
        "translate",
        "spell",
        "meaning of",
        "definition of",
        "who is",
        "when was",
        "where is",
        "what year",
        "how old",
        "what time",
        "capital of",
        "population of",
        "symbol for",
        "formula for",
    ];

    // L1 patterns: extraction from context
    const L1_PATTERNS: &[&str] = &[
        "extract",
        "find in",
        "list the",
        "list all",
        "get the",
        "pull out",
        "identify the",
        "what are the",
        "show me the",
        "parse",
        "look up",
        "search for",
        "filter",
        "select",
        "which of",
    ];

    // L2 patterns: aggregation and multi-step
    const L2_PATTERNS: &[&str] = &[
        "compare",
        "summarize",
        "summarise",
        "contrast",
        "analyze",
        "analyse",
        "evaluate",
        "rank",
        "categorize",
        "categorise",
        "group",
        "classify",
        "correlate",
        "aggregate",
        "pros and cons",
        "advantages and disadvantages",
        "differences between",
        "similarities between",
    ];

    // Check L0 first (cheapest)
    for pat in L0_PATTERNS {
        if q.contains(pat) && context_token_count < 200 {
            return DisarmLevel::L0;
        }
    }

    // Check L1 (extraction)
    for pat in L1_PATTERNS {
        if q.contains(pat) && context_token_count < 500 {
            return DisarmLevel::L1;
        }
    }

    // Check L2 (aggregation)
    for pat in L2_PATTERNS {
        if q.contains(pat) {
            return DisarmLevel::L2;
        }
    }

    // Medium context with L1 patterns → bump to L2
    for pat in L1_PATTERNS {
        if q.contains(pat) && context_token_count >= 500 {
            return DisarmLevel::L2;
        }
    }

    // L0 patterns with larger context → L1
    for pat in L0_PATTERNS {
        if q.contains(pat) && (200..1000).contains(&context_token_count) {
            return DisarmLevel::L1;
        }
    }

    // Large context even with simple patterns → L2
    for pat in L0_PATTERNS {
        if q.contains(pat) && context_token_count >= 1000 {
            return DisarmLevel::L2;
        }
    }

    // Short queries with no recognized patterns — still might be simple
    if q.split_whitespace().count() <= 5 && context_token_count == 0 {
        return DisarmLevel::L1;
    }

    // Everything else → L3 (reasoning)
    DisarmLevel::L3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disarm_levels_ordered() {
        assert!(DisarmLevel::L0 < DisarmLevel::L1);
        assert!(DisarmLevel::L1 < DisarmLevel::L2);
        assert!(DisarmLevel::L2 < DisarmLevel::L3);
    }

    #[test]
    fn test_reference_energy() {
        assert!((DisarmLevel::L0.reference_energy_j() - 0.01).abs() < f64::EPSILON);
        assert!((DisarmLevel::L3.reference_energy_j() - 6.0).abs() < f64::EPSILON);
        assert_eq!(DisarmLevel::L0.reference_energy_uj(), 10_000);
        assert_eq!(DisarmLevel::L3.reference_energy_uj(), 6_000_000);
    }

    #[test]
    fn test_requires_gpu() {
        assert!(!DisarmLevel::L0.requires_gpu());
        assert!(!DisarmLevel::L1.requires_gpu());
        assert!(!DisarmLevel::L2.requires_gpu());
        assert!(DisarmLevel::L3.requires_gpu());
    }

    #[test]
    fn test_eligible_tiers() {
        assert_eq!(DisarmLevel::L0.eligible_tiers().len(), 3);
        assert_eq!(DisarmLevel::L3.eligible_tiers(), &[MeshTierHint::Backbone]);
        assert_eq!(DisarmLevel::L0.preferred_tier(), MeshTierHint::Leaf);
        assert_eq!(DisarmLevel::L3.preferred_tier(), MeshTierHint::Backbone);
    }

    #[test]
    fn test_classify_lookup() {
        assert_eq!(
            classify_query("what is the capital of France", 0),
            DisarmLevel::L0
        );
        assert_eq!(classify_query("define photosynthesis", 0), DisarmLevel::L0);
        assert_eq!(classify_query("convert 5 miles to km", 0), DisarmLevel::L0);
        assert_eq!(classify_query("calculate 2+2", 0), DisarmLevel::L0);
    }

    #[test]
    fn test_classify_extraction() {
        assert_eq!(
            classify_query("extract the email addresses", 100),
            DisarmLevel::L1
        );
        assert_eq!(
            classify_query("list all the names mentioned", 200),
            DisarmLevel::L1
        );
        assert_eq!(
            classify_query("find in the document the date", 50),
            DisarmLevel::L1
        );
    }

    #[test]
    fn test_classify_aggregation() {
        assert_eq!(
            classify_query("compare Python and Rust", 0),
            DisarmLevel::L2
        );
        assert_eq!(
            classify_query("summarize this article", 500),
            DisarmLevel::L2
        );
        assert_eq!(
            classify_query("pros and cons of microservices", 0),
            DisarmLevel::L2
        );
    }

    #[test]
    fn test_classify_reasoning() {
        assert_eq!(
            classify_query("write a complete implementation of a B-tree in Rust", 0),
            DisarmLevel::L3
        );
        assert_eq!(
            classify_query(
                "design a distributed consensus algorithm for my use case",
                2000
            ),
            DisarmLevel::L3
        );
    }

    #[test]
    fn test_context_size_escalation() {
        // Same L0 pattern with increasing context bumps level
        assert_eq!(classify_query("what is X", 0), DisarmLevel::L0);
        assert_eq!(classify_query("what is X", 300), DisarmLevel::L1);
        assert_eq!(classify_query("what is X", 1500), DisarmLevel::L2);
    }

    #[test]
    fn test_extraction_with_large_context() {
        // L1 pattern + large context → L2
        assert_eq!(
            classify_query("extract the key points", 100),
            DisarmLevel::L1
        );
        assert_eq!(
            classify_query("extract the key points", 800),
            DisarmLevel::L2
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        for level in DisarmLevel::all() {
            let json = serde_json::to_string(level).unwrap();
            let back: DisarmLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(*level, back);
        }
    }

    #[test]
    fn test_display() {
        assert_eq!(DisarmLevel::L0.to_string(), "L0 (Lookup)");
        assert_eq!(DisarmLevel::L3.to_string(), "L3 (Reasoning)");
    }

    #[test]
    fn test_all_levels() {
        let all = DisarmLevel::all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], DisarmLevel::L0);
        assert_eq!(all[3], DisarmLevel::L3);
    }

    #[test]
    fn test_mesh_tier_hint_display() {
        assert_eq!(MeshTierHint::Leaf.to_string(), "leaf");
        assert_eq!(MeshTierHint::Gateway.to_string(), "gateway");
        assert_eq!(MeshTierHint::Backbone.to_string(), "backbone");
    }

    #[test]
    fn test_parse_levels() {
        assert_eq!(DisarmLevel::parse("L0"), DisarmLevel::L0);
        assert_eq!(DisarmLevel::parse("L1"), DisarmLevel::L1);
        assert_eq!(DisarmLevel::parse("L2"), DisarmLevel::L2);
        assert_eq!(DisarmLevel::parse("L3"), DisarmLevel::L3);
        // Case-insensitive
        assert_eq!(DisarmLevel::parse("l0"), DisarmLevel::L0);
        assert_eq!(DisarmLevel::parse("l3"), DisarmLevel::L3);
        // Unknown → L1
        assert_eq!(DisarmLevel::parse("L5"), DisarmLevel::L1);
        assert_eq!(DisarmLevel::parse(""), DisarmLevel::L1);
        assert_eq!(DisarmLevel::parse("foo"), DisarmLevel::L1);
    }

    #[test]
    fn test_mesh_tier_hint_serde() {
        let json = serde_json::to_string(&MeshTierHint::Leaf).unwrap();
        assert_eq!(json, "\"leaf\"");
        let back: MeshTierHint = serde_json::from_str(&json).unwrap();
        assert_eq!(back, MeshTierHint::Leaf);
    }
}
