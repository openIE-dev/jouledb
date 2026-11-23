//! Feldera-Class `DBSP` Incremental View Maintenance (`ZSet` $\mathbb{Z}$-Set Delta Algebra)
//!
//! Replaces legacy polling and $O(|T|)$ full-table view recomputation with mathematical
//! **Database Stream Processing (`DBSP`, Budiu et al. 2023-2026)**:
//! - Every relation or stream change batch is a **[`ZSet<T>`]** mapping tuples `T` to non-zero
//!   integer multiplicity weights (`+1` insert, `-1` delete/retract; an update `old -> new`
//!   is `{(old, -1), (new, +1)}`, and opposing weights annihilate to zero!).
//! - **Linear Operators (`filter`, `map`)** commute with differentiation: $\Delta(\sigma(R)) = \sigma(\Delta R)$!
//! - **Bilinear Incremental Equi-Join ([`IncrementalEquiJoin`])**:
//!   $$\Delta(A \bowtie B) = (\Delta A \bowtie B_{\text{prev}}) + (A_{\text{prev}} \bowtie \Delta B) + (\Delta A \bowtie \Delta B)$$
//!   computing exact view diffs in $O(|\Delta|)$ microjoules for sub-millisecond HMI & agent sync!

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Mathematical $\mathbb{Z}$-set: finite collection of tuples `T` with non-zero integer weights.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZSet<T: Ord> {
    weights: BTreeMap<T, i64>,
}

impl<T: Ord> Default for ZSet<T> {
    fn default() -> Self {
        Self {
            weights: BTreeMap::new(),
        }
    }
}

impl<T: Ord + Clone> ZSet<T> {
    /// Construct an empty $\mathbb{Z}$-set (`0` additive identity).
    pub fn new() -> Self {
        Self {
            weights: BTreeMap::new(),
        }
    }

    /// Construct a $\mathbb{Z}$-set from `(tuple, weight)` pairs, summing duplicates and pruning zeros.
    pub fn from_weighted<I: IntoIterator<Item = (T, i64)>>(iter: I) -> Self {
        let mut z = Self::new();
        for (item, w) in iter {
            z.add_weight(item, w);
        }
        z
    }

    /// Add `delta_weight` to `item`, automatically removing `item` if its net weight reaches `0`.
    pub fn add_weight(&mut self, item: T, delta_weight: i64) {
        if delta_weight == 0 {
            return;
        }
        let next = self.weights.get(&item).copied().unwrap_or(0) + delta_weight;
        if next == 0 {
            self.weights.remove(&item);
        } else {
            self.weights.insert(item, next);
        }
    }

    /// Convenience helper for an atomic row update `old -> new` (`-1` on `old`, `+1` on `new`).
    pub fn record_update(&mut self, old: T, new: T) {
        self.add_weight(old, -1);
        self.add_weight(new, 1);
    }

    /// Abelian group addition `self + other`.
    pub fn add(&self, other: &Self) -> Self {
        let mut out = self.clone();
        for (k, &w) in &other.weights {
            out.add_weight(k.clone(), w);
        }
        out
    }

    /// Additive inverse `-self`.
    pub fn negate(&self) -> Self {
        let weights = self.weights.iter().map(|(k, &w)| (k.clone(), -w)).collect();
        Self { weights }
    }

    /// Linear `DBSP` filter operator ($\Delta \sigma_P(R) = \sigma_P(\Delta R)$).
    pub fn filter<F: Fn(&T) -> bool>(&self, pred: F) -> Self {
        let mut out = Self::new();
        for (k, &w) in &self.weights {
            if pred(k) {
                out.add_weight(k.clone(), w);
            }
        }
        out
    }

    /// Linear `DBSP` map/projection operator ($\Delta \pi_f(R) = \pi_f(\Delta R)$, coalescing weights).
    pub fn map<U: Ord + Clone, F: Fn(&T) -> U>(&self, f: F) -> ZSet<U> {
        let mut out = ZSet::new();
        for (k, &w) in &self.weights {
            out.add_weight(f(k), w);
        }
        out
    }

    /// Number of distinct non-zero weighted tuples in this $\mathbb{Z}$-set.
    pub fn support_len(&self) -> usize {
        self.weights.len()
    }

    /// True if all weights canceled to zero.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Iterate over `(tuple, weight)` entries in deterministic sorted order.
    pub fn iter(&self) -> impl Iterator<Item = (&T, i64)> {
        self.weights.iter().map(|(k, &w)| (k, w))
    }
}

/// Stateful `DBSP` bilinear incremental equi-join operator (`A ⋈_K B`) with $O(|\Delta|)$ step cost.
pub struct IncrementalEquiJoin<K: Ord + Clone, V1: Ord + Clone, V2: Ord + Clone> {
    /// Integrated left state $z^{-1}(I(\Delta A))$ indexed by join key `K`.
    left_indexed: BTreeMap<K, ZSet<V1>>,
    /// Integrated right state $z^{-1}(I(\Delta B))$ indexed by join key `K`.
    right_indexed: BTreeMap<K, ZSet<V2>>,
    total_delta_tuples_emitted: u64,
}

impl<K: Ord + Clone, V1: Ord + Clone, V2: Ord + Clone> Default
    for IncrementalEquiJoin<K, V1, V2>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone, V1: Ord + Clone, V2: Ord + Clone> IncrementalEquiJoin<K, V1, V2> {
    /// Create a new empty incremental equi-join circuit.
    pub fn new() -> Self {
        Self {
            left_indexed: BTreeMap::new(),
            right_indexed: BTreeMap::new(),
            total_delta_tuples_emitted: 0,
        }
    }

    /// Advance one logical clock tick with input delta batches `(delta_left, delta_right)`,
    /// returning the exact output delta $\Delta(A \bowtie B)$ in $O(|\Delta|)$ work!
    pub fn step(
        &mut self,
        delta_left: &ZSet<(K, V1)>,
        delta_right: &ZSet<(K, V2)>,
    ) -> ZSet<(K, V1, V2)> {
        let mut out = ZSet::new();

        // Group `delta_left` and `delta_right` by key `K`
        let mut dl_by_k: BTreeMap<K, Vec<(V1, i64)>> = BTreeMap::new();
        for ((k, v1), w) in delta_left.iter() {
            dl_by_k.entry(k.clone()).or_default().push((v1.clone(), w));
        }
        let mut dr_by_k: BTreeMap<K, Vec<(V2, i64)>> = BTreeMap::new();
        for ((k, v2), w) in delta_right.iter() {
            dr_by_k.entry(k.clone()).or_default().push((v2.clone(), w));
        }

        // 1. `ΔA ⋈ B_prev`
        for (k, dl_entries) in &dl_by_k {
            if let Some(b_prev) = self.right_indexed.get(k) {
                for (v1, wa) in dl_entries {
                    for (v2, wb) in b_prev.iter() {
                        out.add_weight((k.clone(), v1.clone(), v2.clone()), wa * wb);
                    }
                }
            }
        }

        // 2. `A_prev ⋈ ΔB`
        for (k, dr_entries) in &dr_by_k {
            if let Some(a_prev) = self.left_indexed.get(k) {
                for (v1, wa) in a_prev.iter() {
                    for (v2, wb) in dr_entries {
                        out.add_weight((k.clone(), v1.clone(), v2.clone()), wa * wb);
                    }
                }
            }
        }

        // 3. `ΔA ⋈ ΔB`
        for (k, dl_entries) in &dl_by_k {
            if let Some(dr_entries) = dr_by_k.get(k) {
                for (v1, wa) in dl_entries {
                    for (v2, wb) in dr_entries {
                        out.add_weight((k.clone(), v1.clone(), v2.clone()), wa * wb);
                    }
                }
            }
        }

        // 4. Integrate `A_prev += ΔA` and `B_prev += ΔB`
        for (k, dl_entries) in dl_by_k {
            let z = self.left_indexed.entry(k.clone()).or_default();
            for (v1, w) in dl_entries {
                z.add_weight(v1, w);
            }
            if z.is_empty() {
                self.left_indexed.remove(&k);
            }
        }
        for (k, dr_entries) in dr_by_k {
            let z = self.right_indexed.entry(k.clone()).or_default();
            for (v2, w) in dr_entries {
                z.add_weight(v2, w);
            }
            if z.is_empty() {
                self.right_indexed.remove(&k);
            }
        }

        self.total_delta_tuples_emitted += out.support_len() as u64;
        out
    }
}

/// Incremental `GROUP BY` `(COUNT, SUM)` maintainer over weighted `(group_key, numeric_val)` $\mathbb{Z}$-set deltas.
#[derive(Default)]
pub struct IncrementalGroupSum<G: Ord + Clone> {
    /// `group -> (net_row_count, weighted_sum)`.
    groups: BTreeMap<G, (i64, i64)>,
}

impl<G: Ord + Clone> IncrementalGroupSum<G> {
    /// Construct an empty incremental group aggregator.
    pub fn new() -> Self {
        Self {
            groups: BTreeMap::new(),
        }
    }

    /// Apply a delta batch of `(group, value)` with $\mathbb{Z}$-set weights and return changed groups!
    pub fn step(&mut self, delta: &ZSet<(G, i64)>) -> BTreeMap<G, Option<(i64, i64)>> {
        let mut changed = BTreeMap::new();
        for ((g, val), w) in delta.iter() {
            let entry = self.groups.entry(g.clone()).or_insert((0, 0));
            entry.0 += w;
            entry.1 += (*val) * w;
            if entry.0 == 0 && entry.1 == 0 {
                self.groups.remove(g);
                changed.insert(g.clone(), None);
            } else {
                changed.insert(g.clone(), Some(*entry));
            }
        }
        changed
    }

    /// Lookup current `(count, sum)` for `group`.
    pub fn get(&self, group: &G) -> Option<(i64, i64)> {
        self.groups.get(group).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbsp_bilinear_incremental_join_matches_full_materialized_join_under_updates_and_deletes()
    {
        let mut circuit = IncrementalEquiJoin::<String, String, String>::new();
        let mut materialized_view = ZSet::new();

        // Tick 1: Insert robots + zones
        let d_robots_1 = ZSet::from_weighted([
            (("zone_a".to_string(), "bot_1".to_string()), 1),
            (("zone_b".to_string(), "bot_2".to_string()), 1),
        ]);
        let d_hazards_1 = ZSet::from_weighted([(
            ("zone_a".to_string(), "wet_floor".to_string()),
            1,
        )]);

        let out_1 = circuit.step(&d_robots_1, &d_hazards_1);
        materialized_view = materialized_view.add(&out_1);
        assert_eq!(materialized_view.support_len(), 1);

        // Tick 2: `bot_1` moves from `zone_a -> zone_b` (`-1` old, `+1` new) while `zone_b` gets a hazard!
        let d_robots_2 = ZSet::from_weighted([
            (("zone_a".to_string(), "bot_1".to_string()), -1),
            (("zone_b".to_string(), "bot_1".to_string()), 1),
        ]);
        let d_hazards_2 = ZSet::from_weighted([(
            ("zone_b".to_string(), "sparks".to_string()),
            1,
        )]);

        let out_2 = circuit.step(&d_robots_2, &d_hazards_2);
        materialized_view = materialized_view.add(&out_2);

        // Now `zone_a` join should have annihilated to 0, and BOTH `bot_1` & `bot_2` match `zone_b` `sparks`!
        let items: Vec<_> = materialized_view.iter().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0],
            (&( "zone_b".into(), "bot_1".into(), "sparks".into() ), 1)
        );
        assert_eq!(
            items[1],
            (&( "zone_b".into(), "bot_2".into(), "sparks".into() ), 1)
        );
    }

    #[test]
    fn test_dbsp_incremental_group_sum_retractions() {
        let mut agg = IncrementalGroupSum::<String>::new();
        let d1 = ZSet::from_weighted([
            (("arm_left".to_string(), 12), 1),
            (("arm_left".to_string(), 18), 1),
        ]);
        agg.step(&d1);
        assert_eq!(agg.get(&"arm_left".to_string()), Some((2, 30)));

        // Retract 12 and insert 20 (`+8` net sum, same count 2!)
        let d2 = ZSet::from_weighted([
            (("arm_left".to_string(), 12), -1),
            (("arm_left".to_string(), 20), 1),
        ]);
        agg.step(&d2);
        assert_eq!(agg.get(&"arm_left".to_string()), Some((2, 38)));
    }
}
