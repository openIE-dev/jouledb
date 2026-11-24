//! Goldman Sachs `SecDB` Reactive Dependency-Graph Engine & Google `Napa` Freshness-SLA Controller.
//!
//! Absorbs two iconic private internal database architectures from Family 2:
//! 1. **Goldman Sachs `SecDB` / JPMorgan `Athena` (`SecDbReactiveGraph`):** Stores market/sensor
//!    source cells and derived pricing/risk/control formulas in one persisted topological DAG.
//!    When a source cell ticks, ONLY its reachable dirty cone is re-evaluated (with input-value
//!    change short-circuiting so unchanged intermediate thresholds halt downstream propagation!).
//! 2. **Google `Napa` (`NapaFreshnessController`):** Governs the **Freshness vs Maintenance Cost vs
//!    Query Latency** trade-off via explicit `(max_staleness_ms, max_eager_joules_nj)` SLAs —
//!    dynamically switching between `EagerPush`, `BoundedStalenessSnapshot` (`12 nJ` cache hit),
//!    and `LazyOnReadFlush` under bursty write firehoses.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Formula operator evaluated inside a `SecDB` derived node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecDbFormula {
    /// Weighted linear combination $\text{bias} + \sum_i w_i x_i$.
    LinearComb { weights: Vec<f64>, bias: f64 },
    /// Pairwise spread $x_0 - x_1$.
    Spread,
    /// Product $x_0 \times x_1$ (e.g. notional $\times$ FX spot or torque $=$ force $\times$ arm).
    Product,
    /// Clamped risk breach indicator (`1.0` if $|x_0| > \text{limit}$ else `0.0`) — demonstrates
    /// **early dirty-propagation cutoff** when $x_0$ fluctuates without crossing `limit`!
    BreachGate { limit: f64 },
    /// Fast analytical Black-Scholes-style call option intrinsic + time-value approximation
    /// over `[spot, strike, vol_annual, years_to_expiry]`.
    OptionMarkApprox,
}

impl SecDbFormula {
    fn eval(&self, inputs: &[f64]) -> f64 {
        match self {
            Self::LinearComb { weights, bias } => {
                let mut acc = *bias;
                for (&w, &x) in weights.iter().zip(inputs.iter()) {
                    acc += w * x;
                }
                acc
            }
            Self::Spread => {
                inputs.first().copied().unwrap_or(0.0) - inputs.get(1).copied().unwrap_or(0.0)
            }
            Self::Product => {
                inputs.first().copied().unwrap_or(0.0) * inputs.get(1).copied().unwrap_or(0.0)
            }
            Self::BreachGate { limit } => {
                if inputs.first().copied().unwrap_or(0.0).abs() > *limit {
                    1.0
                } else {
                    0.0
                }
            }
            Self::OptionMarkApprox => {
                let spot = inputs.first().copied().unwrap_or(100.0).max(1e-6);
                let strike = inputs.get(1).copied().unwrap_or(100.0).max(1e-6);
                let vol = inputs.get(2).copied().unwrap_or(0.20).clamp(0.001, 5.0);
                let tau = inputs.get(3).copied().unwrap_or(0.25).max(0.0);
                let intrinsic = (spot - strike).max(0.0);
                // Brenner-Subrahmanyam ATM time-value decayed by moneyness distance
                let atm_tv = 0.39894228 * spot * vol * tau.sqrt();
                let moneyness_dist = ((spot / strike).ln() / (vol * tau.sqrt().max(1e-4))).powi(2);
                intrinsic + atm_tv * (-0.5 * moneyness_dist).exp()
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum NodeKind {
    Source,
    Derived {
        inputs: Vec<String>,
        formula: SecDbFormula,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DagNode {
    name: String,
    kind: NodeKind,
    value: f64,
    dirty: bool,
    eval_count: u64,
    last_updated_ms: u64,
}

/// Receipt from a tick/recompute pass over a `SecDB` reactive graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecDbTickReceipt {
    pub total_nodes: usize,
    pub dirty_candidates_visited: usize,
    pub nodes_recomputed: usize,
    pub nodes_skipped_clean: usize,
    pub compute_saved_ratio: f64,
    pub energy_nj: u64,
}

/// Goldman Sachs `SecDB`-inspired persisted reactive dependency DAG.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecDbReactiveGraph {
    nodes: BTreeMap<String, DagNode>,
    dependents: BTreeMap<String, Vec<String>>,
    topo_order: Vec<String>,
}

impl SecDbReactiveGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert_source(&mut self, name: impl Into<String>, initial_value: f64, now_ms: u64) {
        let name = name.into();
        self.nodes.insert(
            name.clone(),
            DagNode {
                name: name.clone(),
                kind: NodeKind::Source,
                value: initial_value,
                dirty: false,
                eval_count: 1,
                last_updated_ms: now_ms,
            },
        );
        self.rebuild_topo_order();
    }

    pub fn upsert_derived(
        &mut self,
        name: impl Into<String>,
        inputs: Vec<String>,
        formula: SecDbFormula,
        now_ms: u64,
    ) -> Result<f64, String> {
        let name = name.into();
        for inp in &inputs {
            if !self.nodes.contains_key(inp) {
                return Err(format!("missing upstream dependency '{inp}' for '{name}'"));
            }
            if inp == &name {
                return Err(format!("self-cycle detected on '{name}'"));
            }
        }
        let vals: Vec<f64> = inputs
            .iter()
            .map(|k| self.nodes.get(k).map_or(0.0, |n| n.value))
            .collect();
        let init_val = formula.eval(&vals);

        for inp in &inputs {
            let deps = self.dependents.entry(inp.clone()).or_default();
            if !deps.contains(&name) {
                deps.push(name.clone());
            }
        }
        self.nodes.insert(
            name.clone(),
            DagNode {
                name,
                kind: NodeKind::Derived { inputs, formula },
                value: init_val,
                dirty: false,
                eval_count: 1,
                last_updated_ms: now_ms,
            },
        );
        self.rebuild_topo_order();
        Ok(init_val)
    }

    fn rebuild_topo_order(&mut self) {
        let mut indeg: BTreeMap<String, usize> = BTreeMap::new();
        for (name, node) in &self.nodes {
            let d = match &node.kind {
                NodeKind::Source => 0,
                NodeKind::Derived { inputs, .. } => inputs.len(),
            };
            indeg.insert(name.clone(), d);
        }
        let mut q: VecDeque<String> = indeg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(k, _)| k.clone())
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(u) = q.pop_front() {
            order.push(u.clone());
            if let Some(children) = self.dependents.get(&u) {
                for ch in children {
                    if let Some(d) = indeg.get_mut(ch) {
                        *d = d.saturating_sub(1);
                        if *d == 0 {
                            q.push_back(ch.clone());
                        }
                    }
                }
            }
        }
        self.topo_order = order;
    }

    /// Stage a source mutation (marks direct dependents dirty without immediate recompute so
    /// `NapaFreshnessController` can coalesce high-frequency bursts!).
    pub fn stage_source_tick(
        &mut self,
        source: &str,
        new_value: f64,
        now_ms: u64,
    ) -> Result<bool, String> {
        let Some(node) = self.nodes.get_mut(source) else {
            return Err(format!("source node '{source}' not found"));
        };
        if (node.value - new_value).abs() <= 1e-12 {
            return Ok(false);
        }
        node.value = new_value;
        node.eval_count += 1;
        node.last_updated_ms = now_ms;
        if let Some(children) = self.dependents.get(source).cloned() {
            for ch in children {
                if let Some(cn) = self.nodes.get_mut(&ch) {
                    cn.dirty = true;
                }
            }
        }
        Ok(true)
    }

    /// Recompute ONLY dirty nodes in strict topological order, halting propagation downstream of
    /// any node whose newly evaluated output equals its prior value (`early cutoff`).
    pub fn propagate_dirty(&mut self, now_ms: u64) -> SecDbTickReceipt {
        let total_nodes = self.nodes.len();
        let mut visited = 0usize;
        let mut recomputed = 0usize;

        let order = self.topo_order.clone();
        for name in &order {
            let is_dirty = self.nodes.get(name).is_some_and(|n| n.dirty);
            if !is_dirty {
                continue;
            }
            visited += 1;
            let eval_opt = if let Some(node) = self.nodes.get(name)
                && let NodeKind::Derived { inputs, formula } = &node.kind
            {
                let in_vals: Vec<f64> = inputs
                    .iter()
                    .map(|k| self.nodes.get(k).map_or(0.0, |un| un.value))
                    .collect();
                Some((node.value, formula.eval(&in_vals)))
            } else {
                None
            };

            if let Some((old_val, new_val)) = eval_opt {
                recomputed += 1;
                let changed = (old_val - new_val).abs() > 1e-12;
                if let Some(node) = self.nodes.get_mut(name) {
                    node.value = new_val;
                    node.dirty = false;
                    node.eval_count += 1;
                    node.last_updated_ms = now_ms;
                }
                if changed && let Some(children) = self.dependents.get(name).cloned() {
                    for ch in children {
                        if let Some(cn) = self.nodes.get_mut(&ch) {
                            cn.dirty = true;
                        }
                    }
                }
            }
        }

        let skipped = total_nodes.saturating_sub(recomputed);
        let saved_ratio = if total_nodes == 0 {
            0.0
        } else {
            (skipped as f64) / (total_nodes as f64)
        };
        SecDbTickReceipt {
            total_nodes,
            dirty_candidates_visited: visited,
            nodes_recomputed: recomputed,
            nodes_skipped_clean: skipped,
            compute_saved_ratio: saved_ratio,
            energy_nj: (recomputed as u64) * 35 + 8,
        }
    }

    pub fn value_of(&self, name: &str) -> Option<f64> {
        self.nodes.get(name).map(|n| n.value)
    }

    pub fn has_dirty_nodes(&self) -> bool {
        self.nodes.values().any(|n| n.dirty)
    }

    /// Return the minimal upstream ancestor cone of `target` so lazy on-read queries only
    /// recompute dirty ancestors of the requested view!
    pub fn read_with_ancestor_refresh(
        &mut self,
        target: &str,
        now_ms: u64,
    ) -> Option<(f64, usize)> {
        if !self.nodes.contains_key(target) {
            return None;
        }
        let mut needed: BTreeSet<String> = BTreeSet::new();
        let mut stack = vec![target.to_string()];
        while let Some(cur) = stack.pop() {
            if needed.insert(cur.clone())
                && let Some(n) = self.nodes.get(&cur)
                && let NodeKind::Derived { inputs, .. } = &n.kind
            {
                for inp in inputs {
                    stack.push(inp.clone());
                }
            }
        }
        let mut recomputed = 0usize;
        let order = self.topo_order.clone();
        for name in &order {
            if !needed.contains(name) {
                continue;
            }
            let is_dirty = self.nodes.get(name).is_some_and(|n| n.dirty);
            if !is_dirty {
                continue;
            }
            let eval_opt = if let Some(node) = self.nodes.get(name)
                && let NodeKind::Derived { inputs, formula } = &node.kind
            {
                let in_vals: Vec<f64> = inputs
                    .iter()
                    .map(|k| self.nodes.get(k).map_or(0.0, |un| un.value))
                    .collect();
                Some((node.value, formula.eval(&in_vals)))
            } else {
                None
            };
            if let Some((old_val, new_val)) = eval_opt {
                recomputed += 1;
                let changed = (old_val - new_val).abs() > 1e-12;
                if let Some(node) = self.nodes.get_mut(name) {
                    node.value = new_val;
                    node.dirty = false;
                    node.eval_count += 1;
                    node.last_updated_ms = now_ms;
                }
                if changed && let Some(children) = self.dependents.get(name).cloned() {
                    for ch in children {
                        if let Some(cn) = self.nodes.get_mut(&ch) {
                            cn.dirty = true;
                        }
                    }
                }
            }
        }
        self.nodes.get(target).map(|n| (n.value, recomputed))
    }
}

/// Google `Napa`-inspired Freshness-vs-Joule SLA Controller over a `SecDbReactiveGraph`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NapaFreshnessController {
    pub max_staleness_ms: u64,
    pub eager_budget_nj_per_window: u64,
    spent_eager_nj: u64,
    last_materialized_ms: u64,
    pub coalesced_writes: u64,
    pub cache_hits_within_sla: u64,
    pub lazy_refreshes: u64,
}

impl NapaFreshnessController {
    pub fn new(max_staleness_ms: u64, eager_budget_nj_per_window: u64) -> Self {
        Self {
            max_staleness_ms,
            eager_budget_nj_per_window,
            spent_eager_nj: 0,
            last_materialized_ms: 0,
            coalesced_writes: 0,
            cache_hits_within_sla: 0,
            lazy_refreshes: 0,
        }
    }

    /// Ingest a source tick: eagerly propagates ONLY if `spent_eager_nj < eager_budget_nj_per_window`,
    /// otherwise stages and coalesces writes at `0` derived recompute cost!
    pub fn on_write(
        &mut self,
        dag: &mut SecDbReactiveGraph,
        source: &str,
        value: f64,
        now_ms: u64,
    ) -> Result<u64, String> {
        let changed = dag.stage_source_tick(source, value, now_ms)?;
        if !changed {
            return Ok(0);
        }
        if self.spent_eager_nj < self.eager_budget_nj_per_window {
            let rec = dag.propagate_dirty(now_ms);
            self.spent_eager_nj += rec.energy_nj;
            self.last_materialized_ms = now_ms;
            Ok(rec.energy_nj)
        } else {
            self.coalesced_writes += 1;
            Ok(4) // 4 nJ source staging only!
        }
    }

    /// Query `target` subject to `max_staleness_ms` SLA:
    /// - If clean OR `now_ms - last_materialized_ms <= max_staleness_ms`, serves O(1) snapshot at `12 nJ`.
    /// - If stale beyond SLA, refreshes ONLY `target`'s dirty ancestor cone!
    pub fn query_with_sla(
        &mut self,
        dag: &mut SecDbReactiveGraph,
        target: &str,
        now_ms: u64,
    ) -> Option<(f64, u64)> {
        if !dag.has_dirty_nodes()
            || now_ms.saturating_sub(self.last_materialized_ms) <= self.max_staleness_ms
        {
            self.cache_hits_within_sla += 1;
            return dag.value_of(target).map(|v| (v, 12));
        }
        self.lazy_refreshes += 1;
        let (val, recomputed) = dag.read_with_ancestor_refresh(target, now_ms)?;
        if !dag.has_dirty_nodes() {
            self.last_materialized_ms = now_ms;
        }
        Some((val, 12 + (recomputed as u64) * 35))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secdb_dirty_cone_cutoff_and_napa_freshness_sla() {
        let mut dag = SecDbReactiveGraph::new();
        dag.upsert_source("fx_eur_usd", 1.08, 0);
        for desk in 0..40 {
            let s = format!("desk_{desk}_spot");
            dag.upsert_source(&s, 100.0 + desk as f64, 0);
            let _ = dag.upsert_derived(
                format!("desk_{desk}_opt"),
                vec![s],
                SecDbFormula::OptionMarkApprox,
                0,
            );
        }
        let _ = dag.upsert_derived(
            "eur_notional",
            vec!["fx_eur_usd".to_string()],
            SecDbFormula::LinearComb {
                weights: vec![1_000_000.0],
                bias: 0.0,
            },
            0,
        );
        // Tick ONLY `fx_eur_usd`: the 40 unrelated `desk_*_opt` option nodes MUST stay untouched!
        let _ = dag.stage_source_tick("fx_eur_usd", 1.10, 10);
        let rec = dag.propagate_dirty(10);
        assert_eq!(rec.nodes_recomputed, 1);
        assert!(rec.compute_saved_ratio > 0.95);

        // Napa burst coalescing: 100 rapid ticks within 50ms SLA window coalesce into 1 lazy refresh!
        let mut napa = NapaFreshnessController::new(50, 0);
        for t in 1..=100u64 {
            let _ = napa.on_write(&mut dag, "fx_eur_usd", 1.10 + (t as f64) * 0.0001, t);
        }
        assert_eq!(napa.coalesced_writes, 100);
        let (v_fresh, _) = napa
            .query_with_sla(&mut dag, "eur_notional", 120)
            .expect("query");
        assert!((v_fresh - 1_110_000.0).abs() < 1.0);
        assert_eq!(napa.lazy_refreshes, 1);
    }
}
