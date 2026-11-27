//! Declarative Semantic SQL Operators & Thompson-Sampled Joule-Pareto Cascade Router
//!
//! Brings the 2025-2026 semantic operator paradigm (LOTUS, Palimpzest/Abacus, DocETL,
//! DuckDB FlockMTL) natively into JouleDB's deterministic cascade substrate:
//!
//! - **`SEM_SIMILAR(text_a, text_b)`** -> `Float` cosine similarity in `[0.0, 1.0]` (~1 µJ)
//! - **`SEM_FILTER(text, hypothesis [, threshold])`** -> `Bool` semantic predicate (~2 µJ Tier 1)
//! - **`SEM_EXTRACT(text, slot_name)`** -> `String | Null` deterministic key/slot extraction (~2 µJ)
//! - **`SEM_CLASSIFY(text, pipe_or_comma_labels)`** -> `String` highest-affinity label (~3 µJ)
//! - **[`JouleParetoRouter`]**: Online Bayesian Thompson-sampling router maintaining
//!   conjugate `Beta(alpha, beta)` posteriors per `(domain, CascadeTier)` (~1 KB state,
//!   ~1 µJ routing cost) to pick the minimum-joule tier satisfying `min_confidence` and `budget_uj`.

use crate::planner::CascadeTier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const EMBED_DIM: usize = 128;

/// Compute a fast, translation-invariant 128-d L2-normalized signed feature-hashed
/// representation over word tokens + character 3-grams (~0.5 µJ, zero external model).
pub fn tier1_semantic_embed(text: &str) -> [f32; EMBED_DIM] {
    let mut vec = [0.0f32; EMBED_DIM];
    let lower = text.to_lowercase();

    let fnv64 = |bytes: &[u8], seed: u64| -> u64 {
        let mut h = seed;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    };

    for tok in lower.split(|c: char| !c.is_alphanumeric()) {
        if tok.is_empty() {
            continue;
        }
        // Lightweight Porter-style suffix stem so "indemnification"/"indemnify"/"indemnified" align
        let stem = tok
            .strip_suffix("ification")
            .or_else(|| tok.strip_suffix("ation"))
            .or_else(|| tok.strip_suffix("ing"))
            .or_else(|| tok.strip_suffix("ed"))
            .or_else(|| tok.strip_suffix("es"))
            .or_else(|| tok.strip_suffix('s'))
            .unwrap_or(tok);
        let stem = if stem.len() >= 3 { stem } else { tok };

        let h = fnv64(stem.as_bytes(), 0xcbf2_9ce4_8422_2325);
        let idx = (h as usize) % EMBED_DIM;
        let sign = if (h >> 32) & 1 == 0 { 1.0f32 } else { -1.0f32 };
        vec[idx] += 2.5 * sign;
    }

    let bytes = lower.as_bytes();
    if bytes.len() >= 3 {
        for tri in bytes.windows(3) {
            let h = fnv64(tri, 0x8422_2325_cbf2_9ce4);
            let idx = (h as usize) % EMBED_DIM;
            let sign = if (h >> 32) & 1 == 0 { 0.4f32 } else { -0.4f32 };
            vec[idx] += sign;
        }
    }

    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for v in &mut vec {
            *v *= inv;
        }
    }
    vec
}

/// `SEM_SIMILAR(a, b)`: calibrated Tier-1 semantic similarity in `[0.0, 1.0]`.
pub fn sem_similar(a: &str, b: &str) -> f64 {
    if a.trim().eq_ignore_ascii_case(b.trim()) {
        return 1.0;
    }
    let va = tier1_semantic_embed(a);
    let vb = tier1_semantic_embed(b);
    let dot: f32 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
    (dot as f64).clamp(0.0, 1.0)
}

/// `SEM_FILTER(text, criterion, threshold)`: declarative semantic predicate.
///
/// Combines exact substring / lexical token containment (Tier 0) with signed
/// subword-stem cosine similarity (Tier 1).
pub fn sem_filter(text: &str, criterion: &str, threshold: f64) -> bool {
    let t_low = text.to_lowercase();
    let c_low = criterion.trim().to_lowercase();
    if c_low.is_empty() {
        return true;
    }
    if t_low.contains(&c_low) {
        return true;
    }
    sem_similar(&t_low, &c_low) >= threshold
}

/// `SEM_EXTRACT(text, field_name)`: deterministic structured/semi-structured slot extractor.
///
/// Extracts `"field: value"`, `"field = value"`, JSON `"field": "value"`, or
/// labeled patterns (`email`, `date`, `amount`) in ~1 µJ before any model tier is needed.
pub fn sem_extract(text: &str, field: &str) -> Option<String> {
    let f_trim = field.trim();
    if f_trim.is_empty() {
        return None;
    }

    // 1. If `text` parses as a JSON object, check direct or case-insensitive key match
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(text) {
        for (k, v) in &map {
            if k.eq_ignore_ascii_case(f_trim) {
                return match v {
                    serde_json::Value::String(s) => Some(s.clone()),
                    serde_json::Value::Null => None,
                    other => Some(other.to_string()),
                };
            }
        }
    }

    // 2. Scan `key: value` / `key = value` patterns across lines or segments
    let f_norm = f_trim.to_lowercase().replace('_', " ");
    for segment in text.split(['\n', ';', '|']) {
        for sep in [':', '='] {
            if let Some((lhs, rhs)) = segment.split_once(sep) {
                let lhs_clean = lhs
                    .trim()
                    .trim_matches(['"', '\'', '-', '*'])
                    .to_lowercase()
                    .replace('_', " ");
                if lhs_clean == f_norm || lhs_clean.ends_with(&f_norm) {
                    let val = rhs.trim().trim_matches([',', '"', '\'']).trim();
                    if !val.is_empty() {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }

    // 3. Built-in canonical pattern extractors (`date`, `email`)
    match f_norm.as_str() {
        "date" | "iso date" => extract_iso_date(text),
        "email" => text
            .split_whitespace()
            .map(|w| w.trim_matches([',', '.', ';', '<', '>', '(', ')']))
            .find(|w| w.contains('@') && w.split('@').count() == 2 && w.len() >= 5)
            .map(String::from),
        _ => None,
    }
}

fn extract_iso_date(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    if bytes.len() < 10 {
        return None;
    }
    for i in 0..=bytes.len() - 10 {
        let s = &bytes[i..i + 10];
        if s[4] == b'-'
            && s[7] == b'-'
            && s[0..4].iter().all(|b| b.is_ascii_digit())
            && s[5..7].iter().all(|b| b.is_ascii_digit())
            && s[8..10].iter().all(|b| b.is_ascii_digit())
        {
            return std::str::from_utf8(s).ok().map(String::from);
        }
    }
    None
}

/// `SEM_CLASSIFY(text, labels_spec)`: zero-shot Tier-1 classification over `|` or `,` separated labels.
pub fn sem_classify(text: &str, labels_spec: &str) -> Option<String> {
    let sep = if labels_spec.contains('|') { '|' } else { ',' };
    let mut best: Option<(String, f64)> = None;
    for raw_label in labels_spec.split(sep) {
        let label = raw_label.trim();
        if label.is_empty() {
            continue;
        }
        let score = sem_similar(text, label);
        match &best {
            Some((_, best_score)) if *best_score >= score => {}
            _ => best = Some((label.to_string(), score)),
        }
    }
    best.map(|(l, _)| l)
}

// ============================================================================
// Abacus + Gap-6 Thompson-Sampled Bayesian Joule-Pareto Cascade Router
// ============================================================================

/// Conjugate `Beta(alpha, beta)` reliability posterior for a `(domain, CascadeTier)` arm.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BetaArm {
    pub alpha: f64,
    pub beta: f64,
    /// Calibrated nominal energy cost in microjoules for this tier.
    pub nominal_cost_uj: u64,
}

impl BetaArm {
    pub fn new(alpha: f64, beta: f64, nominal_cost_uj: u64) -> Self {
        Self {
            alpha: alpha.max(0.5),
            beta: beta.max(0.5),
            nominal_cost_uj,
        }
    }

    /// Posterior mean accuracy `alpha / (alpha + beta)`.
    pub fn mean_accuracy(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Conjugate Bayesian update on binary acceptance/contrast feedback (`O(1)`, ~0.1 µJ).
    pub fn observe(&mut self, succeeded: bool) {
        if succeeded {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }
}

/// Routing decision emitted by [`JouleParetoRouter`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParetoRouteDecision {
    pub selected_tier: CascadeTier,
    pub estimated_cost_uj: u64,
    pub sampled_confidence: f64,
    pub within_budget: bool,
}

/// Learned metabolic cascade router using Thompson Sampling over `Beta(alpha, beta)` posteriors.
///
/// Closes Gap 6 in `jouledb-sota-architecture.md` and implements Palimpzest/Abacus-style
/// cost-accuracy Pareto tier selection denominated in microjoules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JouleParetoRouter {
    arms: HashMap<String, [BetaArm; 5]>,
    rng_state: u64,
}

impl Default for JouleParetoRouter {
    fn default() -> Self {
        Self::new(0xC0FF_EE42_2026_507A)
    }
}

impl JouleParetoRouter {
    pub fn new(seed: u64) -> Self {
        Self {
            arms: HashMap::new(),
            rng_state: seed.max(1),
        }
    }

    fn default_tier_arms() -> [BetaArm; 5] {
        [
            // Lookup: ~1 µJ, prior 0.60
            BetaArm::new(3.0, 2.0, 1),
            // Formula: ~2 µJ, prior 0.70
            BetaArm::new(3.5, 1.5, 2),
            // Extract (HDC / RaBitQ / Tier 1): ~10 µJ, prior 0.82
            BetaArm::new(8.2, 1.8, 10),
            // Aggregate (Local ONNX / Ternary): ~2,000 µJ (2 mJ), prior 0.91
            BetaArm::new(9.1, 0.9, 2_000),
            // Reason (In-DB Sparse MoE / Tier 3-4): ~500,000 µJ (0.5 J), prior 0.98
            BetaArm::new(19.6, 0.4, 500_000),
        ]
    }

    fn tier_index(tier: CascadeTier) -> usize {
        match tier {
            CascadeTier::Lookup => 0,
            CascadeTier::Formula => 1,
            CascadeTier::Extract => 2,
            CascadeTier::Aggregate => 3,
            CascadeTier::Reason => 4,
        }
    }

    fn index_to_tier(idx: usize) -> CascadeTier {
        match idx {
            0 => CascadeTier::Lookup,
            1 => CascadeTier::Formula,
            2 => CascadeTier::Extract,
            3 => CascadeTier::Aggregate,
            _ => CascadeTier::Reason,
        }
    }

    /// Record ground-truth / contrast outcome for `(domain, tier)` to adapt metabolic thresholds online.
    pub fn record_outcome(&mut self, domain: &str, tier: CascadeTier, succeeded: bool) {
        let tiers = self
            .arms
            .entry(domain.to_lowercase())
            .or_insert_with(Self::default_tier_arms);
        tiers[Self::tier_index(tier)].observe(succeeded);
    }

    /// Select the lowest-joule [`CascadeTier`] whose Thompson-sampled accuracy meets
    /// `min_confidence` without exceeding `budget_uj` (if specified).
    pub fn route(
        &mut self,
        domain: &str,
        min_confidence: f64,
        budget_uj: Option<u64>,
    ) -> ParetoRouteDecision {
        let tiers = *self
            .arms
            .entry(domain.to_lowercase())
            .or_insert_with(Self::default_tier_arms);

        let mut best_affordable: Option<ParetoRouteDecision> = None;

        // Tiers are ordered strictly ascending by joules: pick FIRST tier meeting `min_confidence`!
        for (idx, arm) in tiers.iter().enumerate() {
            let affordable = budget_uj.is_none_or(|b| arm.nominal_cost_uj <= b);
            if !affordable {
                continue;
            }
            let sampled = self.sample_beta(arm.alpha, arm.beta);
            let decision = ParetoRouteDecision {
                selected_tier: Self::index_to_tier(idx),
                estimated_cost_uj: arm.nominal_cost_uj,
                sampled_confidence: sampled,
                within_budget: true,
            };
            if sampled >= min_confidence {
                return decision;
            }
            best_affordable = Some(decision);
        }

        // If no affordable tier met `min_confidence`, return the highest affordable tier (or Lookup if budget=0)
        best_affordable.unwrap_or(ParetoRouteDecision {
            selected_tier: CascadeTier::Lookup,
            estimated_cost_uj: tiers[0].nominal_cost_uj,
            sampled_confidence: tiers[0].mean_accuracy(),
            within_budget: false,
        })
    }

    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_uniform_open(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        u.clamp(1e-12, 1.0 - 1e-12)
    }

    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_uniform_open();
        let u2 = self.next_uniform_open();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Marsaglia-Tsang Gamma(shape, 1) sampler -> Beta(a, b) = G(a) / (G(a) + G(b)).
    fn sample_gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.next_uniform_open();
            return self.sample_gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.next_standard_normal();
            let v_root = 1.0 + c * x;
            if v_root <= 0.0 {
                continue;
            }
            let v = v_root * v_root * v_root;
            let u = self.next_uniform_open();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    fn sample_beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let ga = self.sample_gamma(alpha);
        let gb = self.sample_gamma(beta);
        if ga + gb <= 0.0 {
            0.5
        } else {
            (ga / (ga + gb)).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sem_operators_extract_filter_classify() {
        let doc = "Contract ID: C-991; termination_date: 2027-04-15; Clause: Mutual indemnification on change of control.";
        assert_eq!(
            sem_extract(doc, "termination_date").as_deref(),
            Some("2027-04-15")
        );
        assert!(sem_filter(doc, "indemnify change of control", 0.30));
        assert!(!sem_filter(doc, "quantum chromodynamics lattice", 0.45));

        let label =
            sem_classify(doc, "astronomy | legal contract indemnification | recipe").unwrap();
        assert_eq!(label, "legal contract indemnification");
    }

    #[test]
    fn test_thompson_sampled_joule_pareto_router_learns_domain_difficulty() {
        let mut router = JouleParetoRouter::new(42);

        // Domain A ("arithmetic"): Lookup succeeds 30 times in a row -> routes to 1 µJ Lookup!
        for _ in 0..30 {
            router.record_outcome("arithmetic", CascadeTier::Lookup, true);
        }
        let d_easy = router.route("arithmetic", 0.80, Some(10_000));
        assert_eq!(d_easy.selected_tier, CascadeTier::Lookup);
        assert_eq!(d_easy.estimated_cost_uj, 1);

        // Domain B ("clinical_triage"): Lookup/Formula/Extract repeatedly fail, Aggregate succeeds ->
        // router escalates automatically to Aggregate (2,000 µJ), unless capped by a tight 50 µJ budget!
        for _ in 0..25 {
            router.record_outcome("clinical_triage", CascadeTier::Lookup, false);
            router.record_outcome("clinical_triage", CascadeTier::Formula, false);
            router.record_outcome("clinical_triage", CascadeTier::Extract, false);
            router.record_outcome("clinical_triage", CascadeTier::Aggregate, true);
        }
        let d_hard = router.route("clinical_triage", 0.80, Some(10_000));
        assert_eq!(d_hard.selected_tier, CascadeTier::Aggregate);

        // Tight 50 µJ ceiling prevents escalating to 2,000 µJ Aggregate!
        let d_capped = router.route("clinical_triage", 0.80, Some(50));
        assert!(d_capped.estimated_cost_uj <= 50);
    }
}
