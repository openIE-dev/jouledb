//! Pattern-Lang Resolver: wires the real pattern-core Bridge into JouleDB AI.
//!
//! This is Tier 0c: deterministic algorithmic resolution at zero LLM cost.
//! "Sort this array" → QuickSort. "Find shortest path" → Dijkstra.
//! Microseconds. Microjoules. 419 canonical patterns.
//!
//! Requires the `pattern-lang` feature flag.

#[cfg(feature = "pattern-lang")]
mod inner {
    use crate::ai::pattern_bridge::{MatchKind, OodaPosition, PatternMatch, PatternResolver};

    /// Adapter: wraps pattern-core's Bridge to implement JouleDB's PatternResolver.
    pub struct PatternLangResolver {
        bridge: pattern_core::bridge::Bridge,
    }

    impl PatternLangResolver {
        /// Create from pattern-core's combined bridge (vocabulary + primitives).
        pub fn from_primitives() -> Self {
            Self {
                bridge: pattern_core::bridge::build_from_primitives(),
            }
        }

        /// Create from a vocabulary (if available).
        pub fn from_vocabulary(vocab: &pattern_core::vocabulary::Vocabulary) -> Self {
            Self {
                bridge: pattern_core::bridge::build_combined(vocab),
            }
        }

        /// Number of patterns in the bridge.
        pub fn pattern_count(&self) -> usize {
            self.bridge.pattern_count()
        }

        /// Summary of the bridge state.
        pub fn summary(&self) -> String {
            self.bridge.summary()
        }
    }

    impl PatternResolver for PatternLangResolver {
        fn resolve(&self, keywords: &[String], arity: Option<usize>) -> Vec<PatternMatch> {
            let bridge_matches = self.bridge.resolve(keywords, arity);

            bridge_matches
                .into_iter()
                .map(|bm| {
                    let match_kind = match bm.match_kind {
                        pattern_core::bridge::MatchKind::ExactName => MatchKind::ExactName,
                        pattern_core::bridge::MatchKind::FullKeyword => MatchKind::FullKeyword,
                        pattern_core::bridge::MatchKind::VerifiedMapping => {
                            MatchKind::VerifiedMapping
                        }
                        pattern_core::bridge::MatchKind::Partial => MatchKind::Partial,
                    };

                    PatternMatch {
                        pattern_name: bm.pattern_name,
                        score: bm.score,
                        match_kind,
                        keywords_matched: keywords.to_vec(),
                        arity: None,
                    }
                })
                .collect()
        }

        fn lookup(&self, name: &str) -> Option<PatternMatch> {
            if self.bridge.is_pattern_name(name) {
                Some(PatternMatch {
                    pattern_name: name.to_string(),
                    score: 1.0,
                    match_kind: MatchKind::ExactName,
                    keywords_matched: vec![name.to_string()],
                    arity: None,
                })
            } else {
                None
            }
        }

        fn position(&self) -> OodaPosition {
            OodaPosition::Mount // Bridge is always in cache-hit mode
        }
    }
}

#[cfg(feature = "pattern-lang")]
pub use inner::PatternLangResolver;

/// Stub when pattern-lang feature is not enabled.
#[cfg(not(feature = "pattern-lang"))]
pub struct PatternLangResolver;

#[cfg(not(feature = "pattern-lang"))]
impl PatternLangResolver {
    pub fn from_primitives() -> Self {
        Self
    }
    pub fn pattern_count(&self) -> usize {
        0
    }
    pub fn summary(&self) -> String {
        "pattern-lang feature not enabled".to_string()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_resolver_exists() {
        let resolver = super::PatternLangResolver::from_primitives();
        eprintln!("Pattern-Lang resolver: {} patterns", resolver.pattern_count());
        eprintln!("{}", resolver.summary());

        #[cfg(feature = "pattern-lang")]
        {
            use crate::ai::pattern_bridge::PatternResolver;

            // Test resolution
            let results =
                resolver.resolve(&["sort".to_string(), "array".to_string()], None);
            eprintln!("'sort array' → {} matches", results.len());
            for m in &results {
                eprintln!(
                    "  {} (score={:.2}, kind={:?})",
                    m.pattern_name, m.score, m.match_kind
                );
            }
            assert!(!results.is_empty(), "should find sorting patterns");

            // Test resolution with different keywords
            let qs = resolver.resolve(&["quick".to_string(), "sort".to_string()], None);
            eprintln!("'quick sort' → {} matches", qs.len());
            for m in qs.iter().take(3) {
                eprintln!("  {} (score={:.2})", m.pattern_name, m.score);
            }

            let bfs = resolver.resolve(&["breadth".to_string(), "first".to_string(), "search".to_string()], None);
            eprintln!("'breadth first search' → {} matches", bfs.len());
            for m in bfs.iter().take(3) {
                eprintln!("  {} (score={:.2})", m.pattern_name, m.score);
            }

            // Should find something for common algorithmic queries
            assert!(results.len() > 0, "should find patterns for 'sort array'");
        }
    }
}
