//! Pattern-Lang as Deterministic Resolution Layer
//!
//! Wires Pattern-Lang's semantic bridge as the L0-L2 resolution path before flowR.
//! If the bridge finds a deterministic match (ExactName, FullKeyword, VerifiedMapping),
//! the query resolves at zero LLM cost. If no match → escalate to flowR.
//!
//! Energy cascade: 0 joules (cache) → µJ (pattern match) → mJ (flowR trivial)
//!                → J (flowR tractable) → $ (API)
//!
//! Pattern-Lang's OODA position maps to JouleDB's metabolic state:
//! - Mount = Resting, SideControl = Alert, Guard = Active, Scramble/Tap = Surge

use crate::ai::metabolic::MetabolicState;
use crate::ai::receipt::AiReceipt;
use crate::ai::traits::AiOutput;

/// How the pattern was matched — determines confidence and energy cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchKind {
    /// Pattern name appears directly in query. Highest confidence.
    ExactName,
    /// All keywords matched against pattern index. High confidence.
    FullKeyword,
    /// Benchmark-verified mapping (ground truth). Perfect confidence.
    VerifiedMapping,
    /// Partial keyword overlap + arity match. Medium confidence.
    Partial,
    /// No match found — escalate to flowR.
    NoMatch,
}

impl MatchKind {
    pub fn is_deterministic(&self) -> bool {
        matches!(
            self,
            MatchKind::ExactName | MatchKind::FullKeyword | MatchKind::VerifiedMapping
        )
    }

    pub fn confidence(&self) -> f32 {
        match self {
            MatchKind::VerifiedMapping => 1.0,
            MatchKind::ExactName => 0.98,
            MatchKind::FullKeyword => 0.90,
            MatchKind::Partial => 0.60,
            MatchKind::NoMatch => 0.0,
        }
    }

    pub fn energy_joules(&self) -> f64 {
        match self {
            MatchKind::VerifiedMapping => 0.0,       // cache hit
            MatchKind::ExactName => 0.000_001,        // 1 µJ
            MatchKind::FullKeyword => 0.000_010,      // 10 µJ
            MatchKind::Partial => 0.000_100,          // 100 µJ
            MatchKind::NoMatch => 0.0,                // didn't resolve
        }
    }
}

/// A pattern match result from the semantic bridge.
#[derive(Clone, Debug)]
pub struct PatternMatch {
    pub pattern_name: String,
    pub score: f64,
    pub match_kind: MatchKind,
    pub keywords_matched: Vec<String>,
    pub arity: Option<(usize, usize)>, // (inputs, outputs)
}

/// OODA engagement position — maps to metabolic state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OodaPosition {
    /// Direct cache hit — highest confidence, lowest energy.
    Mount,
    /// Single pattern via semantic search.
    SideControl,
    /// Domain-specific recipe (NCO doctrine).
    Doctrine,
    /// 2-pattern composition.
    Guard,
    /// Decomposition into sub-task DAG.
    HalfGuard,
    /// Adaptive retry with different candidate.
    Scramble,
    /// Honest escalation — nothing left to try.
    Tap,
}

impl OodaPosition {
    /// Map OODA position to JouleDB metabolic state.
    pub fn to_metabolic_state(&self) -> MetabolicState {
        match self {
            OodaPosition::Mount | OodaPosition::SideControl => MetabolicState::Resting,
            OodaPosition::Doctrine | OodaPosition::Guard => MetabolicState::Alert,
            OodaPosition::HalfGuard | OodaPosition::Scramble => MetabolicState::Active,
            OodaPosition::Tap => MetabolicState::Surge,
        }
    }

    /// Estimated energy for this position.
    pub fn energy_joules(&self) -> f64 {
        match self {
            OodaPosition::Mount => 0.0,
            OodaPosition::SideControl => 0.000_001,
            OodaPosition::Doctrine => 0.000_010,
            OodaPosition::Guard => 0.000_100,
            OodaPosition::HalfGuard => 0.001,
            OodaPosition::Scramble => 0.010,
            OodaPosition::Tap => 1.0,
        }
    }
}

/// Trait for the Pattern-Lang semantic bridge.
/// Implemented by the actual pattern-lang crate; this trait lives in amorphic
/// so there's no direct dependency.
pub trait PatternResolver: Send + Sync {
    /// Resolve a query against the pattern index.
    /// Returns matches sorted by score (highest first).
    fn resolve(&self, keywords: &[String], arity: Option<usize>) -> Vec<PatternMatch>;

    /// Quick lookup by exact pattern name.
    fn lookup(&self, name: &str) -> Option<PatternMatch>;

    /// Get the current OODA engagement position for this resolution attempt.
    fn position(&self) -> OodaPosition;
}

/// Result of attempting pattern-based resolution before flowR.
#[derive(Clone, Debug)]
pub enum PatternResolution {
    /// Deterministic resolution — no LLM needed.
    Resolved {
        output: AiOutput,
        receipt: AiReceipt,
        pattern: PatternMatch,
        position: OodaPosition,
    },
    /// No deterministic match — escalate to flowR.
    Escalate {
        attempted_keywords: Vec<String>,
        best_partial: Option<PatternMatch>,
        position: OodaPosition,
    },
}

/// The pattern bridge: attempts deterministic resolution before flowR.
pub struct PatternBridge {
    /// Cached pattern names for fast exact-name lookup.
    known_patterns: Vec<String>,
    /// Keyword extraction: simple word tokenization + stop word removal.
    stop_words: Vec<&'static str>,
}

impl PatternBridge {
    pub fn new() -> Self {
        Self {
            known_patterns: Vec::new(),
            stop_words: vec![
                "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "shall", "can", "need", "must", "of", "in",
                "to", "for", "with", "on", "at", "by", "from", "as", "into", "through",
                "about", "between", "after", "before", "above", "below", "and", "or",
                "but", "not", "no", "nor", "so", "if", "then", "than", "that", "this",
                "what", "which", "who", "whom", "how", "when", "where", "why", "all",
                "each", "every", "both", "few", "more", "most", "some", "any", "it",
                "its", "me", "my", "we", "our", "you", "your", "he", "she", "they",
            ],
        }
    }

    /// Register known pattern names for exact-match lookup.
    pub fn register_patterns(&mut self, names: &[String]) {
        self.known_patterns = names.to_vec();
    }

    /// Extract keywords from a natural language query.
    pub fn extract_keywords(&self, query: &str) -> Vec<String> {
        query
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .filter(|w| !self.stop_words.contains(w))
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_')
                    .collect::<String>()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }

    /// Attempt deterministic resolution using the pattern resolver.
    /// If successful, returns a resolved output with zero LLM cost.
    /// If not, returns an escalation with the keywords attempted.
    pub fn try_resolve(
        &self,
        query: &str,
        resolver: &dyn PatternResolver,
    ) -> PatternResolution {
        let keywords = self.extract_keywords(query);

        // First: check for exact pattern name in query
        for name in &self.known_patterns {
            if query.to_lowercase().contains(&name.to_lowercase()) {
                if let Some(m) = resolver.lookup(name) {
                    if m.match_kind.is_deterministic() {
                        return PatternResolution::Resolved {
                            output: AiOutput::Classification {
                                label: m.pattern_name.clone(),
                                confidence: m.match_kind.confidence(),
                            },
                            receipt: AiReceipt::holographic(
                                "pattern-lang",
                                m.match_kind.energy_joules(),
                                0,
                            ),
                            pattern: m,
                            position: OodaPosition::Mount,
                        };
                    }
                }
            }
        }

        // Second: keyword-based semantic search
        let matches = resolver.resolve(&keywords, None);
        if let Some(best) = matches.first() {
            if best.match_kind.is_deterministic() {
                return PatternResolution::Resolved {
                    output: AiOutput::Classification {
                        label: best.pattern_name.clone(),
                        confidence: best.match_kind.confidence(),
                    },
                    receipt: AiReceipt::holographic(
                        "pattern-lang",
                        best.match_kind.energy_joules(),
                        0,
                    ),
                    pattern: best.clone(),
                    position: OodaPosition::SideControl,
                };
            }
        }

        // No deterministic resolution — escalate
        let best_partial = matches.first().cloned();
        let position = if best_partial.is_some() {
            OodaPosition::Guard
        } else {
            OodaPosition::Tap
        };

        PatternResolution::Escalate {
            attempted_keywords: keywords,
            best_partial,
            position,
        }
    }
}

impl Default for PatternBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_extraction() {
        let bridge = PatternBridge::new();
        let kw = bridge.extract_keywords("find the quicksort algorithm for sorting numbers");
        assert!(kw.contains(&"quicksort".to_string()));
        assert!(kw.contains(&"algorithm".to_string()));
        assert!(kw.contains(&"sorting".to_string()));
        assert!(kw.contains(&"numbers".to_string()));
        assert!(!kw.contains(&"the".to_string()));
        assert!(!kw.contains(&"for".to_string()));
    }

    #[test]
    fn test_match_kind_ordering() {
        assert!(MatchKind::ExactName < MatchKind::FullKeyword);
        assert!(MatchKind::FullKeyword < MatchKind::VerifiedMapping);
    }

    #[test]
    fn test_match_kind_deterministic() {
        assert!(MatchKind::ExactName.is_deterministic());
        assert!(MatchKind::FullKeyword.is_deterministic());
        assert!(MatchKind::VerifiedMapping.is_deterministic());
        assert!(!MatchKind::Partial.is_deterministic());
        assert!(!MatchKind::NoMatch.is_deterministic());
    }

    #[test]
    fn test_ooda_to_metabolic() {
        assert_eq!(
            OodaPosition::Mount.to_metabolic_state(),
            MetabolicState::Resting
        );
        assert_eq!(
            OodaPosition::Guard.to_metabolic_state(),
            MetabolicState::Alert
        );
        assert_eq!(
            OodaPosition::Scramble.to_metabolic_state(),
            MetabolicState::Active
        );
        assert_eq!(
            OodaPosition::Tap.to_metabolic_state(),
            MetabolicState::Surge
        );
    }

    /// Mock resolver for testing the bridge without the actual pattern-lang crate.
    struct MockResolver {
        patterns: Vec<PatternMatch>,
    }

    impl PatternResolver for MockResolver {
        fn resolve(&self, keywords: &[String], _arity: Option<usize>) -> Vec<PatternMatch> {
            self.patterns
                .iter()
                .filter(|p| {
                    keywords
                        .iter()
                        .any(|k| p.pattern_name.to_lowercase().contains(&k.to_lowercase()))
                })
                .cloned()
                .collect()
        }

        fn lookup(&self, name: &str) -> Option<PatternMatch> {
            self.patterns
                .iter()
                .find(|p| p.pattern_name.eq_ignore_ascii_case(name))
                .cloned()
        }

        fn position(&self) -> OodaPosition {
            OodaPosition::Mount
        }
    }

    #[test]
    fn test_resolve_exact_name() {
        let resolver = MockResolver {
            patterns: vec![PatternMatch {
                pattern_name: "QuickSort".into(),
                score: 1.0,
                match_kind: MatchKind::ExactName,
                keywords_matched: vec!["sort".into(), "quick".into()],
                arity: Some((1, 1)),
            }],
        };

        let mut bridge = PatternBridge::new();
        bridge.register_patterns(&["QuickSort".to_string()]);

        let result = bridge.try_resolve("use QuickSort to sort this array", &resolver);
        match result {
            PatternResolution::Resolved { pattern, .. } => {
                assert_eq!(pattern.pattern_name, "QuickSort");
            }
            _ => panic!("expected resolved"),
        }
    }

    #[test]
    fn test_resolve_no_match_escalates() {
        let resolver = MockResolver {
            patterns: vec![],
        };
        let bridge = PatternBridge::new();
        let result = bridge.try_resolve("explain quantum entanglement", &resolver);
        match result {
            PatternResolution::Escalate { position, .. } => {
                assert_eq!(position, OodaPosition::Tap);
            }
            _ => panic!("expected escalation"),
        }
    }

    #[test]
    fn test_resolve_keyword_match() {
        let resolver = MockResolver {
            patterns: vec![PatternMatch {
                pattern_name: "BinarySearch".into(),
                score: 0.95,
                match_kind: MatchKind::FullKeyword,
                keywords_matched: vec!["binary".into(), "search".into()],
                arity: Some((2, 1)),
            }],
        };
        let bridge = PatternBridge::new();
        let result = bridge.try_resolve("find element using binary search algorithm", &resolver);
        match result {
            PatternResolution::Resolved {
                pattern, position, ..
            } => {
                assert_eq!(pattern.pattern_name, "BinarySearch");
                assert_eq!(position, OodaPosition::SideControl);
            }
            _ => panic!("expected resolved via keyword"),
        }
    }
}
