//! The Materializer — ported from UCG engine/materializer.py
//!
//! Not artificial intelligence. A synthetic functional utility
//! that materializes human intent into usable output.
//!
//! Resolution cascade:
//!   1. Cache (was this answered before?) → LUT cost
//!   2. Route by entropy (ZERO/LOW/MEDIUM/HIGH)
//!   3. Skills (pre-resolved task→output) → LUT cost
//!   4. Pattern-Lang (deterministic algorithmic resolution) → LUT cost
//!   5. Eigenbasis (structural pattern matching) → compute cost
//!   6. Holographic (HDC unbinding/traversal) → compute cost
//!   7. flowR (reasoning DAG) → compute cost
//!   8. Neural fallback (only for genuinely novel composition) → neural cost
//!   9. Verify output
//!  10. Cache in state store + promote to skill if verified
//!
//! Target: 90%+ of queries never reach neural layer.
//! The system gets cheaper over time via skill promotion.

use std::collections::HashMap;
use std::time::Instant;

use super::energy_receipt::EnergyReceipt;

/// Entropy level of a query — determines which tier handles it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntropyLevel {
    /// Exact answer exists. Arithmetic, unit conversion, known facts.
    Zero,
    /// Deterministic tools sufficient. Logic, grammar, lookup.
    Low,
    /// Structural reasoning needed. Analogy, similarity, contrast.
    Medium,
    /// Novel composition required. Generation, synthesis, creative.
    High,
}

/// Where the answer came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Cache,
    Skill,
    PatternLang,
    /// Intelligence attractor primitive dispatch. One of the 37 convergent
    /// cells from the Periodic Stack of Intelligence resolves the query
    /// directly — cheaper than Eigenbasis because the operation is typed
    /// and the compiler knows its exact thermodynamic floor (per-attractor
    /// cost from `flowg::energy::x86_64_attractor_baseline`).
    Attractor,
    Eigenbasis,
    /// 3D spatial-index probe (kd-tree / R-tree / octree). Picojoule cost,
    /// resolved before Holographic/flowR/Neural fallback.
    Spatial3d,
    Holographic,
    FlowR,
    Neural,
}

impl Source {
    pub fn cost_class(&self) -> &'static str {
        match self {
            Source::Cache | Source::Skill | Source::PatternLang | Source::Attractor => "lut",
            Source::Eigenbasis | Source::Spatial3d | Source::Holographic | Source::FlowR => {
                "compute"
            }
            Source::Neural => "neural",
        }
    }
}

/// Result of materializing an idea.
#[derive(Clone, Debug)]
pub struct MaterializeResult {
    pub output: String,
    pub source: Source,
    pub entropy: EntropyLevel,
    pub verified: bool,
    /// Actual energy spent on this query, in joules. Kept for backward
    /// compatibility with existing metric aggregation. Equivalent to
    /// `receipt.actual_joules()`.
    pub energy_joules: f64,
    pub elapsed_us: u64,
    /// Per-query energy receipt. The load-bearing artifact of the joule
    /// cascade — records actual cost, native silicon floor, GPU baseline,
    /// and LLM baseline so the caller can display or aggregate savings.
    pub receipt: EnergyReceipt,
}

/// Metrics tracking.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub total: u64,
    pub by_source: HashMap<String, u64>,
    pub by_entropy: HashMap<String, u64>,
    pub verification_failures: u64,
    pub promotions: u64,
    pub total_energy: f64,
}

impl Metrics {
    pub fn pct_avoided_neural(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }
        let neural = self.by_source.get("neural").copied().unwrap_or(0);
        (1.0 - neural as f64 / self.total as f64) * 100.0
    }
}

/// A registered skill: deterministic resolver for a specific query pattern.
#[derive(Clone)]
pub struct Skill {
    pub name: String,
    pub match_fn: fn(&str) -> Option<f64>, // Returns confidence or None
    pub execute_fn: fn(&str) -> Option<String>, // Returns answer or None
}

/// The Materializer: routes human intent to cheapest resolver.
pub struct Materializer {
    /// Answer cache.
    pub(crate) cache: HashMap<String, String>,
    /// Registered skills (deterministic resolvers).
    skills: Vec<Skill>,
    /// Pattern-Lang resolver (if available).
    #[cfg(feature = "pattern-lang")]
    pattern_resolver: Option<super::pattern_resolver::PatternLangResolver>,
    /// Tier 0 (eigenbasis + facts).
    tier0: Option<super::tier0::Tier0>,
    /// Attractor tier — dispatches queries to intelligence attractor primitives
    /// from the Periodic Stack of Intelligence. Resolved before eigenbasis
    /// because attractor dispatch is typed and has known picojoule cost.
    /// Typed entry: `materialize_attractor`. See `attractor_tier.rs`.
    pub(crate) attractor_tier: Option<super::attractor_tier::AttractorTier>,
    /// Spatial 3D scene (kd-tree / R-tree / octree). Resolved by the
    /// `materialize_spatial` typed entry point — see `spatial_tier.rs`.
    pub(crate) spatial_scene: Option<super::spatial_tier::SpatialScene>,
    /// Metrics.
    pub metrics: Metrics,
}

impl Materializer {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            skills: Vec::new(),
            #[cfg(feature = "pattern-lang")]
            pattern_resolver: None,
            tier0: None,
            attractor_tier: None,
            spatial_scene: None,
            metrics: Metrics::default(),
        }
    }

    /// Set the Tier 0 engine.
    pub fn set_tier0(&mut self, tier0: super::tier0::Tier0) {
        self.tier0 = Some(tier0);
    }

    /// Set the Pattern-Lang resolver.
    #[cfg(feature = "pattern-lang")]
    pub fn set_pattern_resolver(&mut self, resolver: super::pattern_resolver::PatternLangResolver) {
        self.pattern_resolver = Some(resolver);
    }

    /// Register a skill.
    pub fn register_skill(&mut self, skill: Skill) {
        self.skills.push(skill);
    }

    /// Materialize: route to cheapest resolver.
    pub fn materialize(&mut self, input: &str) -> MaterializeResult {
        let start = Instant::now();
        self.metrics.total += 1;

        let entropy = self.classify_entropy(input);
        *self.metrics.by_entropy.entry(format!("{:?}", entropy)).or_insert(0) += 1;

        // 1. CACHE
        let cache_key = input.to_lowercase();
        if let Some(cached) = self.cache.get(&cache_key) {
            *self.metrics.by_source.entry("cache".into()).or_insert(0) += 1;
            // 100 pJ actual cost — HashMap lookup + tag compare on commodity arch.
            let actual_pj = 100u64;
            return MaterializeResult {
                output: cached.clone(),
                source: Source::Cache,
                entropy,
                verified: true,
                energy_joules: EnergyReceipt::pj_to_joules(actual_pj),
                elapsed_us: start.elapsed().as_micros() as u64,
                receipt: EnergyReceipt::for_tier(Source::Cache, actual_pj),
            };
        }

        // 2. SKILLS (pre-resolved)
        if entropy == EntropyLevel::Zero || entropy == EntropyLevel::Low {
            for skill in &self.skills {
                if let Some(confidence) = (skill.match_fn)(input) {
                    if confidence > 0.5 {
                        if let Some(answer) = (skill.execute_fn)(input) {
                            self.cache.insert(cache_key.clone(), answer.clone());
                            *self.metrics.by_source.entry("skill".into()).or_insert(0) += 1;
                            // 1 nJ actual cost — hash lookup + function pointer call.
                            let actual_pj = 1_000u64;
                            return MaterializeResult {
                                output: answer,
                                source: Source::Skill,
                                entropy,
                                verified: true,
                                energy_joules: EnergyReceipt::pj_to_joules(actual_pj),
                                elapsed_us: start.elapsed().as_micros() as u64,
                                receipt: EnergyReceipt::for_tier(Source::Skill, actual_pj),
                            };
                        }
                    }
                }
            }
        }

        // 3. PATTERN-LANG (deterministic algorithmic resolution)
        #[cfg(feature = "pattern-lang")]
        if let Some(ref resolver) = self.pattern_resolver {
            use crate::ai::pattern_bridge::PatternResolver;
            let keywords: Vec<String> = input
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 2)
                .map(|w| w.to_string())
                .collect();
            let matches = resolver.resolve(&keywords, None);
            if let Some(best) = matches.first() {
                if best.score > 0.5 {
                    let answer = format!("Pattern: {} (score: {:.2})", best.pattern_name, best.score);
                    self.cache.insert(cache_key.clone(), answer.clone());
                    *self.metrics.by_source.entry("pattern_lang".into()).or_insert(0) += 1;
                    // 10 nJ actual cost — trie walk + score calc on commodity arch.
                    let actual_pj = 10_000u64;
                    return MaterializeResult {
                        output: answer,
                        source: Source::PatternLang,
                        entropy,
                        verified: true,
                        energy_joules: EnergyReceipt::pj_to_joules(actual_pj),
                        elapsed_us: start.elapsed().as_micros() as u64,
                        receipt: EnergyReceipt::for_tier(Source::PatternLang, actual_pj),
                    };
                }
            }
        }

        // 3.5 ATTRACTOR TIER (v10 — intelligence attractor dispatch)
        // Dispatches queries that match an attractor pattern to the
        // corresponding intelligence primitive. Cheaper than eigenbasis
        // because attractor dispatch is typed and has known picojoule cost.
        // Uses a structural cache (kind + content hash) so surface-form-
        // different queries that resolve to the same cell share entries.
        let attractor_resolution = self
            .attractor_tier
            .as_mut()
            .and_then(|tier| tier.dispatch_string(input));
        if let Some(resolution) = attractor_resolution {
            self.cache.insert(cache_key.clone(), resolution.output.clone());
            self.metrics.total_energy += resolution.energy_joules;
            *self
                .metrics
                .by_source
                .entry("attractor".into())
                .or_insert(0) += 1;
            // Attractor dispatch returns picojoules directly from the
            // per-attractor native floor (from `native_floor_picojoules`).
            // Use it as both actual and native floor — the attractor tier
            // already computes the silicon-optimal cost.
            let actual_pj = (resolution.energy_joules * 1e12) as u64;
            let native_floor_pj = resolution
                .native_floor_pj
                .unwrap_or(actual_pj);
            return MaterializeResult {
                output: resolution.output,
                source: Source::Attractor,
                entropy,
                verified: resolution.confidence > 0.5,
                energy_joules: resolution.energy_joules,
                elapsed_us: start.elapsed().as_micros() as u64,
                receipt: EnergyReceipt::for_tier_with_floor(
                    Source::Attractor,
                    actual_pj,
                    native_floor_pj,
                ),
            };
        }

        // 4. TIER 0 — EIGENBASIS (structural pattern matching)
        if let Some(ref tier0) = self.tier0 {
            let result = tier0.query(input);
            if let Some(answer) = result.answer {
                if result.confidence > 0.5 {
                    self.cache.insert(cache_key.clone(), answer.clone());
                    let source = match result.source {
                        super::tier0::Tier0Source::DatabaseExact => Source::Skill,
                        super::tier0::Tier0Source::EigenbasisStructural => Source::Eigenbasis,
                        super::tier0::Tier0Source::None => Source::Eigenbasis,
                    };
                    let source_name = format!("{:?}", source);
                    *self.metrics.by_source.entry(source_name).or_insert(0) += 1;
                    let actual_pj = (result.energy * 1e12).max(0.0) as u64;
                    return MaterializeResult {
                        output: answer,
                        source: source.clone(),
                        entropy,
                        verified: true,
                        energy_joules: result.energy,
                        elapsed_us: start.elapsed().as_micros() as u64,
                        receipt: EnergyReceipt::for_tier(source, actual_pj),
                    };
                }
            }
        }

        // 5-7. HOLOGRAPHIC / FLOWR / NEURAL — escalation
        // These would wire to the AI facade's tier system.
        // For now: return that we need higher-tier processing.
        *self.metrics.by_source.entry("neural".into()).or_insert(0) += 1;
        // 10 mJ actual cost — full LLM stack is our neural stub today.
        let actual_pj = 10_000_000_000u64;
        MaterializeResult {
            output: format!("Requires higher-tier processing for: {}", input),
            source: Source::Neural,
            entropy,
            verified: false,
            energy_joules: EnergyReceipt::pj_to_joules(actual_pj),
            elapsed_us: start.elapsed().as_micros() as u64,
            receipt: EnergyReceipt::for_tier(Source::Neural, actual_pj),
        }
    }

    /// Promote a neural result to a skill (deterministic LUT entry).
    pub fn promote_to_skill(&mut self, input_pattern: &str, answer: &str) {
        let pattern = input_pattern.to_lowercase();
        let cached_answer = answer.to_string();
        self.cache.insert(pattern, cached_answer);
        self.metrics.promotions += 1;
    }

    /// Classify query entropy.
    fn classify_entropy(&self, input: &str) -> EntropyLevel {
        let lower = input.to_lowercase();

        // ZERO: exact answers (arithmetic, facts, definitions)
        if lower.starts_with("what is ")
            || lower.starts_with("define ")
            || lower.contains("capital of")
            || lower.chars().any(|c| c.is_ascii_digit())
                && (lower.contains('+') || lower.contains('-') || lower.contains('*') || lower.contains('/'))
        {
            return EntropyLevel::Zero;
        }

        // LOW: deterministic tools (sorting, searching, algorithm lookup)
        if lower.contains("sort") || lower.contains("search") || lower.contains("find")
            || lower.contains("calculate") || lower.contains("convert")
        {
            return EntropyLevel::Low;
        }

        // MEDIUM: structural reasoning (comparison, analogy, contrast)
        if lower.contains("how") && (lower.contains("related") || lower.contains("similar"))
            || lower.contains("compare") || lower.contains("vs")
            || lower.contains("contrast") || lower.contains("analogy")
        {
            return EntropyLevel::Medium;
        }

        // HIGH: novel composition (generation, explanation, creative)
        EntropyLevel::High
    }

    /// Get metrics snapshot.
    pub fn status(&self) -> &Metrics {
        &self.metrics
    }

    /// Cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

impl Default for Materializer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arithmetic_match(input: &str) -> Option<f64> {
        let lower = input.to_lowercase();
        if lower.contains('+') || lower.contains('-') || lower.contains('*') {
            Some(0.9)
        } else {
            None
        }
    }

    fn arithmetic_execute(input: &str) -> Option<String> {
        // Very basic: "2 + 3" → "5"
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() == 3 {
            let a: f64 = parts[0].parse().ok()?;
            let b: f64 = parts[2].parse().ok()?;
            let result = match parts[1] {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" if b != 0.0 => a / b,
                _ => return None,
            };
            Some(format!("{}", result))
        } else {
            None
        }
    }

    #[test]
    fn test_cache_hit() {
        let mut m = Materializer::new();
        m.cache.insert("test query".into(), "cached answer".into());
        let result = m.materialize("test query");
        assert_eq!(result.source, Source::Cache);
        assert_eq!(result.output, "cached answer");
        assert!(result.energy_joules < 0.000_001);
    }

    #[test]
    fn test_skill_resolution() {
        let mut m = Materializer::new();
        m.register_skill(Skill {
            name: "arithmetic".into(),
            match_fn: arithmetic_match,
            execute_fn: arithmetic_execute,
        });

        let result = m.materialize("2 + 3");
        assert_eq!(result.source, Source::Skill);
        assert_eq!(result.output, "5");
        assert_eq!(result.entropy, EntropyLevel::Zero);
    }

    #[test]
    fn test_entropy_classification() {
        let m = Materializer::new();
        assert_eq!(m.classify_entropy("What is the capital of France?"), EntropyLevel::Zero);
        assert_eq!(m.classify_entropy("sort this array"), EntropyLevel::Low);
        assert_eq!(m.classify_entropy("How are cancer and war related?"), EntropyLevel::Medium);
        assert_eq!(m.classify_entropy("Write me a poem about the ocean"), EntropyLevel::High);
    }

    #[test]
    fn test_promotion() {
        let mut m = Materializer::new();
        m.promote_to_skill("what is gravity", "Gravity is the force of attraction between masses");
        assert_eq!(m.metrics.promotions, 1);

        // Now it should be cached
        let result = m.materialize("what is gravity");
        assert_eq!(result.source, Source::Cache);
    }

    #[test]
    fn test_cascade_to_neural() {
        let m = Materializer::new();
        // No skills, no tier0, no pattern-lang → falls through to neural
        let mut m = Materializer::new();
        let result = m.materialize("Write me a sonnet about entropy");
        assert_eq!(result.source, Source::Neural);
        assert_eq!(result.entropy, EntropyLevel::High);
    }

    #[test]
    fn test_metrics_neural_avoidance() {
        let mut m = Materializer::new();
        m.register_skill(Skill {
            name: "arithmetic".into(),
            match_fn: arithmetic_match,
            execute_fn: arithmetic_execute,
        });

        // 3 skill hits, 1 neural fallback
        m.materialize("2 + 3");
        m.materialize("5 * 7");
        m.materialize("10 - 4");
        m.materialize("explain quantum entanglement");

        assert_eq!(m.metrics.total, 4);
        assert!(m.metrics.pct_avoided_neural() >= 75.0);
    }

    #[test]
    fn test_full_cascade_with_tier0() {
        let mut m = Materializer::new();

        // Set up Tier 0 with eigenbasis
        let mut tier0 = super::super::tier0::Tier0::new();
        tier0.register_fact("capital of france", "Paris");

        // Register concept with pattern scores
        let cancer = super::super::eigenbasis::PatternScores::from_sparse(&[
            ("replication", 0.9),
            ("feedback", 0.8),
            ("emergence", 0.7),
        ]);
        let basis = super::super::eigenbasis::Eigenbasis::from_scores(&[cancer.clone()], 0.80);
        tier0.set_eigenbasis(basis);
        tier0.register_concept("cancer", cancer);

        m.set_tier0(tier0);

        // Arithmetic skill
        m.register_skill(Skill {
            name: "arithmetic".into(),
            match_fn: arithmetic_match,
            execute_fn: arithmetic_execute,
        });

        // Test cascade
        let r1 = m.materialize("2 + 3");
        assert_eq!(r1.source, Source::Skill);
        assert_eq!(r1.output, "5");

        let r2 = m.materialize("What is capital of france?");
        assert_eq!(r2.output, "Paris");

        let r3 = m.materialize("What is cancer?");
        assert!(r3.output.contains("cancer"), "should find cancer: {}", r3.output);

        let r4 = m.materialize("Write poetry about stars");
        assert_eq!(r4.source, Source::Neural); // Falls through

        assert!(m.metrics.pct_avoided_neural() >= 75.0,
            "should avoid neural 75%+: {:.1}%", m.metrics.pct_avoided_neural());
    }

    // ── Energy receipt integration tests ─────────────────────────────

    #[test]
    fn receipt_attached_to_cache_hit() {
        let mut m = Materializer::new();
        m.cache.insert("already known".into(), "answer".into());
        let r = m.materialize("already known");
        assert_eq!(r.source, Source::Cache);
        assert_eq!(r.receipt.tier, Source::Cache);
        // Cache hit should be cheap (< 1 nJ) on commodity arch.
        assert!(r.receipt.actual_pj < 1_000, "cache receipt too expensive: {}", r.receipt.actual_pj);
        // Savings vs LLM should be astronomical for a cache hit.
        assert!(r.receipt.savings_vs_llm() > 1e6);
    }

    #[test]
    fn receipt_attached_to_skill_hit() {
        let mut m = Materializer::new();
        m.register_skill(Skill {
            name: "arithmetic".into(),
            match_fn: arithmetic_match,
            execute_fn: arithmetic_execute,
        });
        let r = m.materialize("2 + 3");
        assert_eq!(r.source, Source::Skill);
        assert_eq!(r.receipt.tier, Source::Skill);
        assert_eq!(r.receipt.actual_pj, 1_000);
        // Savings vs LLM: 10 mJ / 1 nJ = 10M×.
        assert!(r.receipt.savings_vs_llm() > 1e6);
    }

    #[test]
    fn receipt_attached_to_neural_fallback() {
        let mut m = Materializer::new();
        let r = m.materialize("write an original sonnet about photons");
        assert_eq!(r.source, Source::Neural);
        assert_eq!(r.receipt.tier, Source::Neural);
        // Neural tier: actual == LLM baseline (10 mJ), so savings ≈ 1×.
        assert!(
            (r.receipt.savings_vs_llm() - 1.0).abs() < 0.01,
            "neural savings should be ~1×, got {}",
            r.receipt.savings_vs_llm()
        );
    }

    #[test]
    fn receipt_energy_joules_matches_legacy_field() {
        // energy_joules (legacy) and receipt.actual_joules() must agree so
        // metric aggregators that still use the old field stay consistent
        // with code that reads from the receipt.
        let mut m = Materializer::new();
        let r = m.materialize("anything");
        assert!(
            (r.energy_joules - r.receipt.actual_joules()).abs() < 1e-15,
            "legacy {} vs receipt {}",
            r.energy_joules,
            r.receipt.actual_joules()
        );
    }

    #[test]
    fn receipt_format_compact_includes_tier_and_pj() {
        let mut m = Materializer::new();
        m.cache.insert("test".into(), "yes".into());
        let r = m.materialize("test");
        let s = r.receipt.format_compact();
        assert!(s.contains("cache"), "compact should name tier: {}", s);
        assert!(s.contains("pJ") || s.contains("nJ"), "compact should show energy: {}", s);
        assert!(s.contains("vs LLM"), "compact should show savings: {}", s);
    }

    /// Visual showcase — prints a full cascade receipt report for a tiny
    /// mixed workload. Not a strict assertion, just makes the "meter AI
    /// in joules" story demoable via `cargo test -- --nocapture`.
    #[test]
    fn receipt_showcase_prints_cascade_receipts() {
        let mut m = Materializer::new();
        m.register_skill(Skill {
            name: "arithmetic".into(),
            match_fn: arithmetic_match,
            execute_fn: arithmetic_execute,
        });
        m.cache.insert("known answer".into(), "42".into());

        let queries = [
            ("known answer", "cache hit"),
            ("2 + 3", "skill dispatch"),
            ("write a poem about the stars", "neural fallback"),
        ];

        println!("\n═══════════════════════════════════════════════════");
        println!("  CASCADE ENERGY RECEIPTS");
        println!("═══════════════════════════════════════════════════");
        let mut total_actual_pj: u64 = 0;
        let mut total_llm_baseline_pj: u64 = 0;
        for (q, label) in queries {
            let r = m.materialize(q);
            println!("\n[{}] \"{}\"", label, q);
            println!("{}", r.receipt.format_verbose());
            total_actual_pj += r.receipt.actual_pj;
            total_llm_baseline_pj += r.receipt.llm_baseline_pj;
        }
        let overall_savings = total_llm_baseline_pj as f64 / total_actual_pj.max(1) as f64;
        println!("\n───────────────────────────────────────────────────");
        println!(
            "  TOTAL actual: {}   baseline: {}",
            super::super::energy_receipt::format_picojoules(total_actual_pj),
            super::super::energy_receipt::format_picojoules(total_llm_baseline_pj)
        );
        println!(
            "  Overall cascade savings vs LLM-only: {}×",
            super::super::energy_receipt::format_ratio(overall_savings)
        );
        println!("═══════════════════════════════════════════════════\n");

        // Smoke: cache and skill combined should still massively outweigh
        // the single neural fallback in savings.
        assert!(overall_savings > 1.0);
    }

    #[test]
    fn receipt_impedance_is_nonzero_for_all_tiers() {
        // Every tier should produce a receipt where impedance_ratio is defined
        // and finite. No panics, no divide-by-zero.
        let mut m = Materializer::new();
        m.cache.insert("cached".into(), "x".into());
        m.register_skill(Skill {
            name: "arithmetic".into(),
            match_fn: arithmetic_match,
            execute_fn: arithmetic_execute,
        });

        let r_cache = m.materialize("cached");
        let r_skill = m.materialize("5 * 7");
        let r_neural = m.materialize("novel composition query");

        for r in [&r_cache, &r_skill, &r_neural] {
            let imp = r.receipt.impedance_ratio();
            assert!(imp.is_finite(), "non-finite impedance for {:?}", r.source);
            assert!(imp > 0.0, "zero impedance for {:?}", r.source);
        }
    }
}
