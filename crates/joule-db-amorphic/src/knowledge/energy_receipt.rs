//! Energy Receipts — meter every cascade query in picojoules.
//!
//! Every [`Materializer::materialize`] call returns a [`MaterializeResult`]
//! with an attached [`EnergyReceipt`] that records four numbers:
//!
//! 1. **actual_pj** — picojoules consumed on the current execution path
//!    (x86_64 / ARM64 / whatever this process is running on).
//! 2. **native_floor_pj** — what the same computation would cost on
//!    purpose-built flowG silicon. This is the thermodynamic floor.
//! 3. **gpu_baseline_pj** — what the same *query* would cost if routed
//!    through a general-purpose GPU inference pipeline (H100-class).
//! 4. **llm_baseline_pj** — what the same query would cost if routed
//!    through a full LLM stack (API call → transformer inference →
//!    token decode).
//!
//! From these four numbers fall out three ratios:
//!
//! - **impedance**: actual / native_floor — how far from silicon-optimal.
//! - **savings vs GPU**: gpu_baseline / actual — how much cheaper we are
//!   than routing through a GPU pipeline right now.
//! - **savings vs LLM**: llm_baseline / actual — how much cheaper we are
//!   than asking an LLM. This is the "meter AI in joules, not tokens"
//!   differentiator made concrete.
//!
//! ## Why this exists
//!
//! Every other AI system meters in tokens or dollars. Tokens are a billing
//! abstraction; dollars are a business abstraction. Both hide the physics.
//! Joules are the physics. A receipt in joules lets you compare a cascade
//! hit (picojoules) against an LLM round-trip (tens of millijoules) and
//! see the ~10⁸× gap directly. That gap is the product.
//!
//! ## Non-goals
//!
//! This is not a benchmark. The baselines are coarse constants derived
//! from publicly reported figures (H100 TDP / query throughput, GPT-4
//! tier token energy). The numbers are honest, not precise. Refine them
//! as better measurements become available.

use super::materializer::Source;

// ============================================================================
// Baseline constants
// ============================================================================
//
// Sources / reasoning:
//
// - LLM baseline: GPT-4-class short query (~100 tokens out). Public figures
//   put H100 token generation energy at ~50 µJ/token for dense FP16 inference;
//   add network + batch overhead → ~10 mJ per query. Rounded to 10 mJ.
//
// - GPU baseline: a small dense inference (BERT-base or similar vision model)
//   on an H100 runs at ~1 mJ per query in batched steady-state. This is the
//   "cheaper than LLM, still GPU" tier.
//
// - Native floors are per-tier: flowG silicon cells are ~1-5 pJ for LUT ops,
//   ~100 pJ for structured compute, ~100 nJ for even neural-native cells.

/// Picojoules for a typical short LLM query end-to-end.
/// 10 mJ = 10_000_000_000 pJ.
pub const LLM_BASELINE_PJ: u64 = 10_000_000_000;

/// Picojoules for a typical small dense GPU inference.
/// 1 mJ = 1_000_000_000 pJ.
pub const GPU_BASELINE_PJ: u64 = 1_000_000_000;

// ============================================================================
// EnergyReceipt
// ============================================================================

/// Per-query energy receipt. The load-bearing artifact of the joule cascade.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnergyReceipt {
    /// Which tier resolved the query.
    pub tier: Source,
    /// Actual picojoules consumed on the current execution path.
    pub actual_pj: u64,
    /// Thermodynamic floor — flowG native silicon cost for this same work.
    pub native_floor_pj: u64,
    /// GPU pipeline baseline for the same query (H100-class dense inference).
    pub gpu_baseline_pj: u64,
    /// LLM stack baseline for the same query (GPT-4-class short query).
    pub llm_baseline_pj: u64,
}

impl EnergyReceipt {
    /// Build a receipt with per-tier default baselines. The caller supplies
    /// the actual cost; native floor, GPU baseline, and LLM baseline are
    /// derived from [`tier_native_floor_pj`] and the global baseline constants.
    pub fn for_tier(tier: Source, actual_pj: u64) -> Self {
        let native_floor_pj = tier_native_floor_pj(&tier);
        Self {
            tier,
            actual_pj,
            native_floor_pj,
            gpu_baseline_pj: GPU_BASELINE_PJ,
            llm_baseline_pj: LLM_BASELINE_PJ,
        }
    }

    /// Build a receipt with a caller-supplied native floor. Used by the
    /// attractor tier because attractor floors are per-attractor, not per-tier
    /// class.
    pub fn for_tier_with_floor(tier: Source, actual_pj: u64, native_floor_pj: u64) -> Self {
        Self {
            tier,
            actual_pj,
            native_floor_pj: native_floor_pj.max(1),
            gpu_baseline_pj: GPU_BASELINE_PJ,
            llm_baseline_pj: LLM_BASELINE_PJ,
        }
    }

    /// Impedance ratio: actual / native_floor. 1.0 = silicon-optimal.
    pub fn impedance_ratio(&self) -> f64 {
        self.actual_pj as f64 / self.native_floor_pj.max(1) as f64
    }

    /// Savings vs a GPU inference pipeline.
    pub fn savings_vs_gpu(&self) -> f64 {
        self.gpu_baseline_pj as f64 / self.actual_pj.max(1) as f64
    }

    /// Savings vs a full LLM stack.
    pub fn savings_vs_llm(&self) -> f64 {
        self.llm_baseline_pj as f64 / self.actual_pj.max(1) as f64
    }

    /// Actual energy in joules (for aggregation with other metric fields).
    pub fn actual_joules(&self) -> f64 {
        Self::pj_to_joules(self.actual_pj)
    }

    /// Convert picojoules to joules. Factored out so `MaterializeResult`
    /// can populate `energy_joules` without repeating the `* 1e-12` literal.
    #[inline]
    pub fn pj_to_joules(pj: u64) -> f64 {
        pj as f64 * 1e-12
    }

    /// One-line compact receipt suitable for CLI output.
    ///
    /// Example: `attractor · 12 pJ · 833M× vs LLM`
    pub fn format_compact(&self) -> String {
        format!(
            "{} · {} · {}× vs LLM",
            tier_label(&self.tier),
            format_picojoules(self.actual_pj),
            format_ratio(self.savings_vs_llm()),
        )
    }

    /// Multi-line verbose receipt for `--verbose` CLI output.
    pub fn format_verbose(&self) -> String {
        format!(
            "tier          {}\n\
             actual        {}\n\
             native floor  {}  (impedance {:.2}×)\n\
             gpu baseline  {}  (savings {}×)\n\
             llm baseline  {}  (savings {}×)",
            tier_label(&self.tier),
            format_picojoules(self.actual_pj),
            format_picojoules(self.native_floor_pj),
            self.impedance_ratio(),
            format_picojoules(self.gpu_baseline_pj),
            format_ratio(self.savings_vs_gpu()),
            format_picojoules(self.llm_baseline_pj),
            format_ratio(self.savings_vs_llm()),
        )
    }
}

// ============================================================================
// Per-tier defaults
// ============================================================================

/// Native silicon floor (picojoules) for a generic query handled by the given
/// tier. Attractor queries override this with per-attractor floors; all other
/// tiers use these per-class defaults.
///
/// These numbers correspond to what the dispatched op would cost on a flowG
/// native silicon cell:
///
/// - Cache hit: single SRAM read + tag compare (~50 pJ)
/// - Skill: hash lookup + function pointer call (~200 pJ)
/// - PatternLang: few-ns trie/regex walk (~1 nJ)
/// - Attractor: generic placeholder — real value comes from per-attractor floor
/// - Eigenbasis: small dense dot product (~5 nJ)
/// - Spatial3d: kd-tree descent (~10 nJ)
/// - Holographic: HDC unbind (~100 nJ)
/// - FlowR: reasoning DAG walk (~500 nJ)
/// - Neural: even native neural cells are µJ-scale for a small forward pass
pub fn tier_native_floor_pj(tier: &Source) -> u64 {
    match tier {
        Source::Cache => 50,
        Source::Skill => 200,
        Source::PatternLang => 1_000,
        Source::Attractor => 500,
        Source::Eigenbasis => 5_000,
        Source::Spatial3d => 10_000,
        Source::Holographic => 100_000,
        Source::FlowR => 500_000,
        Source::Neural => 100_000_000,
    }
}

// ============================================================================
// Formatting helpers
// ============================================================================

fn tier_label(tier: &Source) -> &'static str {
    match tier {
        Source::Cache => "cache",
        Source::Skill => "skill",
        Source::PatternLang => "pattern-lang",
        Source::Attractor => "attractor",
        Source::Eigenbasis => "eigenbasis",
        Source::Spatial3d => "spatial3d",
        Source::Holographic => "holographic",
        Source::FlowR => "flowR",
        Source::Neural => "neural",
    }
}

/// Human-readable picojoule formatter.
///
/// - `< 1_000` → `{} pJ`
/// - `< 1_000_000` → `{:.1} nJ`
/// - `< 1_000_000_000` → `{:.1} µJ`
/// - `< 1_000_000_000_000` → `{:.1} mJ`
/// - `>=` → `{:.1} J`
pub fn format_picojoules(pj: u64) -> String {
    let f = pj as f64;
    if pj < 1_000 {
        format!("{} pJ", pj)
    } else if pj < 1_000_000 {
        format!("{:.1} nJ", f / 1_000.0)
    } else if pj < 1_000_000_000 {
        format!("{:.1} µJ", f / 1_000_000.0)
    } else if pj < 1_000_000_000_000 {
        format!("{:.1} mJ", f / 1_000_000_000.0)
    } else {
        format!("{:.1} J", f / 1_000_000_000_000.0)
    }
}

/// Human-readable ratio formatter (for savings multipliers).
pub fn format_ratio(r: f64) -> String {
    if r >= 1_000_000_000.0 {
        format!("{:.1}B", r / 1_000_000_000.0)
    } else if r >= 1_000_000.0 {
        format!("{:.1}M", r / 1_000_000.0)
    } else if r >= 1_000.0 {
        format!("{:.1}k", r / 1_000.0)
    } else if r >= 10.0 {
        format!("{:.0}", r)
    } else {
        format!("{:.1}", r)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_for_cache_tier_has_sram_floor() {
        let r = EnergyReceipt::for_tier(Source::Cache, 100);
        assert_eq!(r.native_floor_pj, 50);
        assert_eq!(r.actual_pj, 100);
        assert_eq!(r.gpu_baseline_pj, GPU_BASELINE_PJ);
        assert_eq!(r.llm_baseline_pj, LLM_BASELINE_PJ);
    }

    #[test]
    fn receipt_for_neural_tier_has_neural_floor() {
        let r = EnergyReceipt::for_tier(Source::Neural, 10_000_000);
        assert_eq!(r.native_floor_pj, 100_000_000);
        // Neural actual is *below* the native floor because we stubbed it —
        // impedance < 1 means we're not respecting the floor model, but the
        // math is still well-defined.
        assert!(r.impedance_ratio() < 1.0);
    }

    #[test]
    fn impedance_ratio_is_actual_over_floor() {
        let r = EnergyReceipt::for_tier_with_floor(Source::Attractor, 120, 10);
        assert_eq!(r.impedance_ratio(), 12.0);
    }

    #[test]
    fn savings_vs_llm_huge_for_attractor() {
        // Attractor dispatch at 12 pJ vs 10 mJ LLM = 833M× savings.
        let r = EnergyReceipt::for_tier_with_floor(Source::Attractor, 12, 5);
        let s = r.savings_vs_llm();
        assert!(
            s > 8e8 && s < 9e8,
            "expected ~833M× savings, got {}",
            s
        );
    }

    #[test]
    fn savings_vs_gpu_smaller_but_still_large() {
        let r = EnergyReceipt::for_tier_with_floor(Source::Attractor, 12, 5);
        let s = r.savings_vs_gpu();
        // 1 mJ / 12 pJ = ~83M×.
        assert!(s > 8e7 && s < 9e7, "expected ~83M× gpu savings, got {}", s);
    }

    #[test]
    fn actual_joules_matches_picojoules_times_1e_minus_12() {
        let r = EnergyReceipt::for_tier(Source::Cache, 5_000);
        let expected = 5e-9;
        assert!((r.actual_joules() - expected).abs() < 1e-15);
    }

    #[test]
    fn format_picojoules_unit_scales() {
        assert_eq!(format_picojoules(500), "500 pJ");
        assert_eq!(format_picojoules(5_000), "5.0 nJ");
        assert_eq!(format_picojoules(5_000_000), "5.0 µJ");
        assert_eq!(format_picojoules(5_000_000_000), "5.0 mJ");
        assert_eq!(format_picojoules(5_000_000_000_000), "5.0 J");
    }

    #[test]
    fn format_ratio_unit_scales() {
        assert_eq!(format_ratio(0.5), "0.5");
        assert_eq!(format_ratio(5.0), "5.0");
        assert_eq!(format_ratio(42.0), "42");
        assert_eq!(format_ratio(1_500.0), "1.5k");
        assert_eq!(format_ratio(1_500_000.0), "1.5M");
        assert_eq!(format_ratio(1_500_000_000.0), "1.5B");
    }

    #[test]
    fn format_compact_is_one_line() {
        let r = EnergyReceipt::for_tier_with_floor(Source::Attractor, 12, 5);
        let s = r.format_compact();
        assert!(!s.contains('\n'));
        assert!(s.contains("attractor"));
        assert!(s.contains("12 pJ"));
        assert!(s.contains("vs LLM"));
    }

    #[test]
    fn format_verbose_has_all_fields() {
        let r = EnergyReceipt::for_tier_with_floor(Source::Attractor, 12, 5);
        let s = r.format_verbose();
        for field in ["tier", "actual", "native floor", "gpu baseline", "llm baseline"] {
            assert!(s.contains(field), "missing {} in {}", field, s);
        }
        assert!(s.contains("impedance"));
        assert!(s.contains("savings"));
    }

    #[test]
    fn tier_native_floor_monotonic_lut_to_neural() {
        // LUT tiers should have lower floors than compute tiers,
        // which should have lower floors than neural.
        assert!(tier_native_floor_pj(&Source::Cache) < tier_native_floor_pj(&Source::Eigenbasis));
        assert!(tier_native_floor_pj(&Source::Eigenbasis) < tier_native_floor_pj(&Source::Neural));
        assert!(tier_native_floor_pj(&Source::Skill) < tier_native_floor_pj(&Source::Holographic));
    }

    #[test]
    fn for_tier_with_floor_clamps_zero_to_one() {
        // Avoid division by zero in impedance_ratio.
        let r = EnergyReceipt::for_tier_with_floor(Source::Cache, 100, 0);
        assert_eq!(r.native_floor_pj, 1);
        assert_eq!(r.impedance_ratio(), 100.0);
    }

    #[test]
    fn attractor_tier_default_floor_is_placeholder() {
        // Attractor tier has a generic 500 pJ default floor because the real
        // floor is per-attractor. Callers that know the attractor should use
        // for_tier_with_floor.
        assert_eq!(tier_native_floor_pj(&Source::Attractor), 500);
    }

    #[test]
    fn receipt_is_clonable_and_comparable() {
        let r1 = EnergyReceipt::for_tier(Source::Cache, 100);
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }
}
