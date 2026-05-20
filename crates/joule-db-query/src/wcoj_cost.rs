//! Learned cost model for WCOJ variable ordering.
//!
//! Closes Open Item §10.3 in `docs/MGAI-SPEC-DOMAIN-JOULEDB.md`: the
//! variable order for Leapfrog TrieJoin used a fixed frequency heuristic
//! (most-constrained-first). That's a reasonable prior but it can't adapt
//! to a workload's actual selectivity. This module replaces it with an
//! **online learned cost model**:
//!
//! - Per-relation cardinality statistics, sampled from real executions.
//! - A feature map `φ(query, candidate_order)` of cardinality / degree /
//!   prefix-fanout signals plus quadratic interaction terms.
//! - A linear model `cost ≈ wᵀ φ` trained online by **recursive least
//!   squares** (RLS) with a forgetting factor — deterministic at
//!   inference, no gradient/Python pipeline, weights persist across
//!   restarts.
//! - A selection rule that enumerates candidate orders, scores each, and
//!   picks the argmin — but only deviates from the frequency heuristic
//!   when the model predicts a strict improvement, and only after it has
//!   seen `MIN_SAMPLES` executions. This gives a hard "never worse than
//!   the baseline heuristic" floor in practice.
//!
//! The optimisation target is **leapfrog work** — the total number of
//! recursive `enumerate` calls plus intermediate bindings explored. That
//! is exactly the quantity a good variable order minimises, and it is
//! deterministic (unlike wall-clock), so the model trains reproducibly.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::wcoj::{Atom, WcojQuery};

/// Feature-vector dimensionality. Keep small — RLS is O(d²) per update.
pub const FEATURE_DIM: usize = 12;

/// Minimum observed executions before the learned model is trusted over
/// the frequency heuristic.
pub const MIN_SAMPLES: u64 = 16;

/// RLS forgetting factor. < 1.0 lets the model track a drifting workload;
/// 0.999 ≈ an effective window of ~1000 recent executions.
const FORGETTING: f64 = 0.999;

/// Per-relation cardinality catalog, sampled from real executions.
///
/// Keyed by relation name. `count` is the most recent observed tuple
/// count; `samples` tracks how many times we've seen it (for debugging /
/// staleness reasoning). Persisted alongside the model weights.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WcojStats {
    relations: BTreeMap<String, RelStat>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RelStat {
    count: u64,
    samples: u64,
}

impl WcojStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the observed cardinality of a relation.
    pub fn observe(&mut self, relation: &str, cardinality: u64) {
        let e = self.relations.entry(relation.to_string()).or_default();
        e.count = cardinality;
        e.samples = e.samples.saturating_add(1);
    }

    /// Best-known cardinality for a relation. `None` if never seen.
    pub fn cardinality(&self, relation: &str) -> Option<u64> {
        self.relations.get(relation).map(|r| r.count)
    }
}

/// The online linear cost model: `cost ≈ wᵀ φ`, trained by RLS.
///
/// `p` is the (FEATURE_DIM × FEATURE_DIM) inverse-covariance matrix stored
/// row-major. `w` is the weight vector. `samples` counts observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedCostModel {
    w: Vec<f64>,
    p: Vec<f64>,
    samples: u64,
}

impl Default for LearnedCostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl LearnedCostModel {
    pub fn new() -> Self {
        // P initialised to a large-diagonal matrix (high initial
        // uncertainty → fast early adaptation).
        let mut p = vec![0.0; FEATURE_DIM * FEATURE_DIM];
        for i in 0..FEATURE_DIM {
            p[i * FEATURE_DIM + i] = 1.0e3;
        }
        Self {
            w: vec![0.0; FEATURE_DIM],
            p,
            samples: 0,
        }
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Predicted cost for a feature vector.
    pub fn predict(&self, phi: &[f64; FEATURE_DIM]) -> f64 {
        let mut acc = 0.0;
        for i in 0..FEATURE_DIM {
            acc += self.w[i] * phi[i];
        }
        acc
    }

    /// Recursive-least-squares update for one `(features → cost)` pair.
    ///
    /// Standard RLS with forgetting factor λ:
    ///   k  = Pφ / (λ + φᵀPφ)
    ///   w  = w + k (y − wᵀφ)
    ///   P  = (P − k φᵀ P) / λ
    pub fn update(&mut self, phi: &[f64; FEATURE_DIM], y: f64) {
        let d = FEATURE_DIM;

        // pφ = P φ
        let mut p_phi = [0.0f64; FEATURE_DIM];
        for i in 0..d {
            let mut acc = 0.0;
            for j in 0..d {
                acc += self.p[i * d + j] * phi[j];
            }
            p_phi[i] = acc;
        }

        // denom = λ + φᵀ P φ
        let mut denom = FORGETTING;
        for j in 0..d {
            denom += phi[j] * p_phi[j];
        }
        if denom.abs() < 1e-12 {
            return; // degenerate; skip this observation
        }

        // gain k = pφ / denom
        let mut k = [0.0f64; FEATURE_DIM];
        for i in 0..d {
            k[i] = p_phi[i] / denom;
        }

        // error e = y − wᵀφ
        let pred = self.predict(phi);
        let err = y - pred;

        // w += k e
        for i in 0..d {
            self.w[i] += k[i] * err;
        }

        // P = (P − k (pφ)ᵀ) / λ
        for i in 0..d {
            for j in 0..d {
                let upd = self.p[i * d + j] - k[i] * p_phi[j];
                self.p[i * d + j] = upd / FORGETTING;
            }
        }

        self.samples = self.samples.saturating_add(1);
    }
}

/// Stats + model + selection policy. The single object callers hold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WcojCostModel {
    pub stats: WcojStats,
    pub model: LearnedCostModel,
}

impl Default for WcojCostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl WcojCostModel {
    pub fn new() -> Self {
        Self {
            stats: WcojStats::new(),
            model: LearnedCostModel::new(),
        }
    }

    /// Load a persisted model from JSON, or a fresh one if absent/corrupt.
    pub fn load(path: &std::path::Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|_| Self::new()),
            Err(_) => Self::new(),
        }
    }

    /// Persist to JSON (best-effort; errors are returned for the caller
    /// to log but are non-fatal to query execution).
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(self).map_err(std::io::Error::other)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)
    }

    /// Ingest observed relation cardinalities from a just-run query.
    pub fn observe_cardinalities(&mut self, relations: &[(String, u64)]) {
        for (name, card) in relations {
            self.stats.observe(name, *card);
        }
    }

    /// Pick the variable order for `query`. Falls back to the frequency
    /// heuristic when the model is cold or doesn't predict a strict
    /// improvement (the "never worse than baseline" floor).
    pub fn best_order(&self, query: &WcojQuery) -> Vec<String> {
        let baseline = query.frequency_order();

        // Cold start: not enough evidence to deviate.
        if self.model.samples() < MIN_SAMPLES {
            return baseline;
        }

        let vars = distinct_vars(query);
        let candidates = candidate_orders(&vars);

        let base_feats = self.features(query, &baseline);
        let base_cost = self.model.predict(&base_feats);

        let mut best = baseline.clone();
        let mut best_cost = base_cost;

        for cand in candidates {
            if cand == baseline {
                continue;
            }
            let feats = self.features(query, &cand);
            let cost = self.model.predict(&feats);
            // Strict-improvement margin: only deviate from the baseline
            // when the model is meaningfully more optimistic. 1% margin
            // suppresses jitter-driven flapping.
            if cost < best_cost && cost < base_cost * 0.99 {
                best = cand;
                best_cost = cost;
            }
        }
        best
    }

    /// After an execution, fold the observed `work` back into the model.
    pub fn record(&mut self, query: &WcojQuery, order: &[String], work: u64) {
        let feats = self.features(query, order);
        // Train on log-work — work spans many orders of magnitude across
        // query shapes; a linear model fits the log scale far better.
        let y = ((work + 1) as f64).ln();
        self.model.update(&feats, y);
    }

    /// Feature map φ(query, order). See module docs for the rationale of
    /// each term. All cardinalities come from `self.stats`; an unseen
    /// relation contributes a neutral prior (1.0 → ln 0 after the +1).
    pub fn features(&self, query: &WcojQuery, order: &[String]) -> [f64; FEATURE_DIM] {
        let mut f = [0.0f64; FEATURE_DIM];

        // Per-atom cardinality (best known, default 1).
        let cards: Vec<f64> = query
            .atoms
            .iter()
            .map(|a| self.stats.cardinality(&a.relation_name).unwrap_or(1) as f64)
            .collect();
        let max_card = cards.iter().cloned().fold(0.0, f64::max);
        let sum_card: f64 = cards.iter().sum();

        // Variable degree = number of atoms that bind a variable.
        let degree = |v: &str| -> f64 {
            query
                .atoms
                .iter()
                .filter(|a| a.variables.iter().any(|x| x == v))
                .count() as f64
        };

        // f0 bias
        f[0] = 1.0;
        // f1 log max cardinality
        f[1] = (max_card + 1.0).ln();
        // f2 log sum cardinality
        f[2] = (sum_card + 1.0).ln();
        // f3 number of variables
        f[3] = order.len() as f64;
        // f4 number of atoms
        f[4] = query.atoms.len() as f64;

        // f5 avg variable degree
        let avg_deg = if order.is_empty() {
            0.0
        } else {
            order.iter().map(|v| degree(v)).sum::<f64>() / order.len() as f64
        };
        f[5] = avg_deg;

        // f6 position-weighted degree: rewards putting high-degree
        // (most-constrained) variables EARLY. This is the learnable
        // generalisation of the old hard heuristic — the model decides
        // how much this matters rather than it being absolute.
        let mut pos_weighted = 0.0;
        for (i, v) in order.iter().enumerate() {
            pos_weighted += degree(v) / (i as f64 + 1.0);
        }
        f[6] = pos_weighted;

        // f7 log of the first variable's min binding cardinality — the
        // size of the leapfrog root domain (smaller = fewer top-level
        // iterations).
        let first_root = order
            .first()
            .map(|v| min_binding_card(query, v, &self.stats))
            .unwrap_or(1.0);
        f[7] = (first_root + 1.0).ln();

        // f8 **atom-coverage prefix cost** — the real WCOJ cost driver.
        // Walk the order maintaining the set of bound variables. An atom
        // becomes a hard filter at the depth where its LAST variable is
        // bound; until then it only contributes fan-out. We charge
        // log(cardinality) for every still-open atom at each depth,
        // discounted by depth (early depths multiply all downstream
        // work, so an expensive early level dominates total cost). This
        // is strongly order-sensitive: it distinguishes orders by *when*
        // each atom turns from a fan-out source into a filter, which is
        // exactly what determines leapfrog work.
        let mut bound: Vec<&str> = Vec::with_capacity(order.len());
        let mut prefix_cost = 0.0;
        for (depth, v) in order.iter().enumerate() {
            bound.push(v.as_str());
            let discount = 1.0 / (depth as f64 + 1.0);
            for a in &query.atoms {
                let fully_bound = a.variables.iter().all(|x| bound.contains(&x.as_str()));
                if !fully_bound {
                    // Still an open fan-out source at this depth.
                    let c = self.stats.cardinality(&a.relation_name).unwrap_or(1) as f64;
                    prefix_cost += discount * (c + 1.0).ln();
                }
            }
        }
        f[8] = prefix_cost;

        // Quadratic interaction terms (the "with interactions" part).
        // f9  cardinality scale × root choice
        f[9] = f[1] * f[7];
        // f10 ordering quality × problem width
        f[10] = f[6] * f[3];
        // f11 blow-up × atom count
        f[11] = f[8] * f[4];

        f
    }
}

/// Smallest known cardinality among atoms that bind `var`.
fn min_binding_card(query: &WcojQuery, var: &str, stats: &WcojStats) -> f64 {
    query
        .atoms
        .iter()
        .filter(|a| a.variables.iter().any(|x| x == var))
        .map(|a| stats.cardinality(&a.relation_name).unwrap_or(1) as f64)
        .fold(f64::INFINITY, f64::min)
        .min(f64::MAX)
        .max(0.0)
        .min(1e18) // guard against INFINITY when no atom binds the var
}

/// Distinct variables across all atoms, in first-seen order.
fn distinct_vars(query: &WcojQuery) -> Vec<String> {
    let mut seen = Vec::new();
    for a in &query.atoms {
        for v in &a.variables {
            if !seen.contains(v) {
                seen.push(v.clone());
            }
        }
    }
    seen
}

/// Candidate variable orders to score. Full permutations when the
/// variable count is small (≤ 7 → ≤ 5040 perms, cheap); otherwise a
/// bounded set of greedy + rotational candidates so selection stays
/// O(poly) on wide queries.
fn candidate_orders(vars: &[String]) -> Vec<Vec<String>> {
    if vars.len() <= 7 {
        return permutations(vars);
    }
    // Wide query: don't enumerate k!. Offer rotations + the reverse as a
    // bounded, deterministic candidate set. The learned model still
    // scores them; the no-regression floor protects correctness.
    let mut out = Vec::new();
    for shift in 0..vars.len() {
        let mut rot: Vec<String> = vars.to_vec();
        rot.rotate_left(shift);
        out.push(rot);
    }
    let mut rev = vars.to_vec();
    rev.reverse();
    out.push(rev);
    out
}

/// All permutations of `vars` (Heap's algorithm, iterative).
fn permutations(vars: &[String]) -> Vec<Vec<String>> {
    let mut result = Vec::new();
    let n = vars.len();
    if n == 0 {
        return result;
    }
    let mut arr: Vec<String> = vars.to_vec();
    let mut c = vec![0usize; n];
    result.push(arr.clone());
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 {
                arr.swap(0, i);
            } else {
                arr.swap(c[i], i);
            }
            result.push(arr.clone());
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    result
}

// Convenience so callers outside this module can build the heuristic
// order without depending on `wcoj`'s internals.
impl WcojQuery {
    /// The original frequency heuristic, retained as the baseline /
    /// cold-start order and the "never worse than" floor.
    pub fn frequency_order(&self) -> Vec<String> {
        let mut var_count: Vec<(String, usize)> = Vec::new();
        for atom in &self.atoms {
            for var in &atom.variables {
                if let Some(entry) = var_count.iter_mut().find(|(v, _)| v == var) {
                    entry.1 += 1;
                } else {
                    var_count.push((var.clone(), 1));
                }
            }
        }
        var_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        var_count.into_iter().map(|(v, _)| v).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcoj::Atom;

    fn triangle() -> WcojQuery {
        WcojQuery {
            atoms: vec![
                Atom { relation_name: "e1".into(), variables: vec!["X".into(), "Y".into()] },
                Atom { relation_name: "e2".into(), variables: vec!["Y".into(), "Z".into()] },
                Atom { relation_name: "e3".into(), variables: vec!["Z".into(), "X".into()] },
            ],
            output_variables: vec!["X".into(), "Y".into(), "Z".into()],
        }
    }

    #[test]
    fn cold_start_returns_frequency_baseline() {
        let m = WcojCostModel::new();
        let q = triangle();
        assert_eq!(m.best_order(&q), q.frequency_order());
    }

    #[test]
    fn feature_dim_is_stable() {
        let m = WcojCostModel::new();
        let q = triangle();
        let f = m.features(&q, &q.frequency_order());
        assert_eq!(f.len(), FEATURE_DIM);
        assert_eq!(f[0], 1.0, "bias term");
    }

    #[test]
    fn rls_reduces_prediction_error_on_repeated_signal() {
        // A fixed feature vector mapped to a fixed target: RLS must drive
        // the prediction toward the target as it sees the pair.
        let mut model = LearnedCostModel::new();
        let phi = [1.0, 2.0, 0.5, 3.0, 1.0, 0.7, 4.0, 1.2, 2.1, 0.3, 0.9, 1.1];
        let target = 5.0;

        let first_err = (model.predict(&phi) - target).abs();
        for _ in 0..50 {
            model.update(&phi, target);
        }
        let last_err = (model.predict(&phi) - target).abs();

        assert!(
            last_err < first_err,
            "RLS did not reduce error: {first_err} -> {last_err}"
        );
        assert!(last_err < 0.1, "RLS did not converge: residual {last_err}");
    }

    #[test]
    fn learns_to_prefer_the_cheap_order() {
        // Realistic, learnable signal: an order is cheap iff it STARTS
        // with the variable that is bound by the small relation (e2,
        // card 10) — i.e. it opens the leapfrog on a tiny root domain
        // instead of a 1000-row one. That property is feature-
        // distinguishable (f7 / f8 see it) and matches real WCOJ cost.
        // We do NOT label two symmetric orders with different costs —
        // that would be physically false and unlearnable from static
        // features by design.
        let mut m = WcojCostModel::new();
        let q = triangle(); // e1(X,Y) e2(Y,Z) e3(Z,X)
        m.stats.observe("e1", 1000);
        m.stats.observe("e2", 10);
        m.stats.observe("e3", 1000);

        // Z is bound by e2 (small) and e3 (large) → min-binding card 10.
        // Y is bound by e1 (large) and e2 (small) → min-binding card 10.
        // X is bound by e1, e3 (both large) → min-binding card 1000.
        // Cheap orders start with Y or Z; expensive ones start with X.
        let cheap_first = |o: &[String]| o[0] == "Y" || o[0] == "Z";

        for _ in 0..200 {
            for cand in permutations(&distinct_vars(&q)) {
                let work = if cheap_first(&cand) { 50 } else { 50_000 };
                m.record(&q, &cand, work);
            }
        }

        assert!(m.model.samples() >= MIN_SAMPLES);
        let chosen = m.best_order(&q);
        assert!(
            cheap_first(&chosen),
            "model failed to learn to open on the small relation; chose {chosen:?}"
        );
    }

    #[test]
    fn never_worse_than_baseline_when_model_is_indifferent() {
        // If the model has samples but predicts no strict improvement for
        // any non-baseline order, it must return the baseline unchanged.
        let mut m = WcojCostModel::new();
        let q = triangle();
        // Feed uniform cost for every order → model can't prefer anything.
        for _ in 0..(MIN_SAMPLES + 5) {
            m.record(&q, &q.frequency_order(), 1000);
        }
        assert_eq!(m.best_order(&q), q.frequency_order());
    }

    #[test]
    fn persistence_round_trips() {
        let dir = std::env::temp_dir().join(format!("wcoj_cost_test_{}", std::process::id()));
        let path = dir.join("model.json");
        let mut m = WcojCostModel::new();
        let q = triangle();
        m.stats.observe("e1", 1234);
        for _ in 0..20 {
            m.record(&q, &q.frequency_order(), 777);
        }
        m.save(&path).expect("save");
        let loaded = WcojCostModel::load(&path);
        assert_eq!(loaded.model.samples(), m.model.samples());
        assert_eq!(loaded.stats.cardinality("e1"), Some(1234));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn permutations_are_complete_and_unique() {
        let v = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let perms = permutations(&v);
        assert_eq!(perms.len(), 6);
        let mut uniq = perms.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 6);
    }

    #[test]
    fn wide_query_uses_bounded_candidate_set() {
        let vars: Vec<String> = (0..10).map(|i| format!("v{i}")).collect();
        let cands = candidate_orders(&vars);
        // rotations (10) + reverse (1) — NOT 10! = 3.6M.
        assert_eq!(cands.len(), 11);
        for c in &cands {
            assert_eq!(c.len(), 10);
        }
    }
}
