//! `ClickBench` / `Vortex` Granule `ZoneMap` + Late-Materialization Selection-Vector Columnar Scanner.
//!
//! Absorbs Family 5 & 6 (`ClickHouse MergeTree` granule skipping + `DuckDB`/`Vortex` late
//! materialization):
//! 1. **Granule `ZoneMap` (`min`, `max`, `null_count`, `bloom_64`):** Summarizes every `1,024`-row
//!    horizontal granule in 32 bytes so selective `WHERE` filters skip `95%–99.9%` of cold blocks
//!    in $O(1)$ without decoding payload columns!
//! 2. **Compact `SelectionVector` (`Vec<u16>`) Late Materialization:** Evaluates multi-column
//!    predicates strictly on surviving row indices, reading non-filter projection/aggregation
//!    columns ONLY for matching tuples.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const GRANULE_ROWS: usize = 1024;

/// Per-granule min/max + 64-bit SplitMix Bloom summary (`32` bytes).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Int64ZoneMap {
    pub min: i64,
    pub max: i64,
    pub null_count: u16,
    pub row_count: u16,
    pub bloom_64: u64,
}

impl Int64ZoneMap {
    pub fn build(slice: &[Option<i64>]) -> Self {
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        let mut nulls = 0u16;
        let mut bloom = 0u64;
        for v in slice {
            match v {
                Some(x) => {
                    min = min.min(*x);
                    max = max.max(*x);
                    bloom |= bloom_mask_i64(*x);
                }
                None => nulls = nulls.saturating_add(1),
            }
        }
        Self {
            min,
            max,
            null_count: nulls,
            row_count: slice.len() as u16,
            bloom_64: bloom,
        }
    }

    #[inline]
    pub fn may_contain_eq(&self, target: i64) -> bool {
        if self.null_count as usize == self.row_count as usize {
            return false;
        }
        if target < self.min || target > self.max {
            return false;
        }
        let m = bloom_mask_i64(target);
        (self.bloom_64 & m) == m
    }

    #[inline]
    pub fn may_overlap_range(&self, lo: i64, hi: i64) -> bool {
        if self.null_count as usize == self.row_count as usize {
            return false;
        }
        !(self.max < lo || self.min > hi)
    }
}

#[inline]
fn bloom_mask_i64(x: i64) -> u64 {
    let mut z = (x as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let b1 = (z & 63) as u32;
    let b2 = ((z >> 16) & 63) as u32;
    (1u64 << b1) | (1u64 << b2)
}

/// Predicate pushdown filter evaluated against `Int64ZoneMap`s before touching granules.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ZonePredicate {
    Eq(i64),
    Between { lo: i64, hi: i64 },
}

impl ZonePredicate {
    #[inline]
    fn zone_may_match(&self, zm: &Int64ZoneMap) -> bool {
        match *self {
            Self::Eq(v) => zm.may_contain_eq(v),
            Self::Between { lo, hi } => zm.may_overlap_range(lo, hi),
        }
    }

    #[inline]
    fn val_matches(&self, v: Option<i64>) -> bool {
        let Some(x) = v else { return false };
        match *self {
            Self::Eq(t) => x == t,
            Self::Between { lo, hi } => x >= lo && x <= hi,
        }
    }
}

/// Receipt from a `ClickBench`-style ZoneMap + Late-Materialization scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneScanReceipt {
    pub total_rows: usize,
    pub total_granules: usize,
    pub granules_skipped_by_zonemap: usize,
    pub granules_scanned: usize,
    pub granule_skip_ratio: f64,
    pub matched_rows: usize,
    pub late_materialized_sum: f64,
    pub distinct_payload_keys: usize,
    pub energy_nj: u64,
}

/// Granule-indexed columnar table supporting `ZoneMap` pruning + `SelectionVector` late materialization.
#[derive(Debug, Clone)]
pub struct ZoneMappedTable {
    filter_granules: Vec<Vec<Option<i64>>>,
    filter_zonemaps: Vec<Int64ZoneMap>,
    metric_granules: Vec<Vec<f64>>,
    user_granules: Vec<Vec<u64>>,
    total_rows: usize,
}

impl ZoneMappedTable {
    pub fn from_columns(
        filter_col: &[Option<i64>],
        metric_col: &[f64],
        user_col: &[u64],
    ) -> Result<Self, String> {
        let n = filter_col.len();
        if metric_col.len() != n || user_col.len() != n {
            return Err("column length mismatch in ZoneMappedTable".to_string());
        }
        let mut fg = Vec::new();
        let mut zm = Vec::new();
        let mut mg = Vec::new();
        let mut ug = Vec::new();
        let mut start = 0usize;
        while start < n {
            let end = (start + GRANULE_ROWS).min(n);
            let fs = filter_col[start..end].to_vec();
            zm.push(Int64ZoneMap::build(&fs));
            fg.push(fs);
            mg.push(metric_col[start..end].to_vec());
            ug.push(user_col[start..end].to_vec());
            start = end;
        }
        Ok(Self {
            filter_granules: fg,
            filter_zonemaps: zm,
            metric_granules: mg,
            user_granules: ug,
            total_rows: n,
        })
    }

    /// Execute `SELECT COUNT(*), SUM(metric), COUNT(DISTINCT user_id) WHERE pred` using:
    /// 1. $O(1)$ `Int64ZoneMap` min/max + 64-bit Bloom granule pruning,
    /// 2. Tight `SelectionVector` (`Vec<u16>`) construction on surviving granules,
    /// 3. Late materialization of `metric_granules` & `user_granules` ONLY at `sel` indices!
    pub fn scan_late_materialized(&self, pred: ZonePredicate) -> ZoneScanReceipt {
        let total_granules = self.filter_zonemaps.len();
        let mut skipped = 0usize;
        let mut scanned = 0usize;
        let mut matched_rows = 0usize;
        let mut sum_metric = 0.0f64;
        let mut distinct_users: BTreeSet<u64> = BTreeSet::new();
        let mut sel_vec: Vec<u16> = Vec::with_capacity(GRANULE_ROWS);

        for (gi, zm) in self.filter_zonemaps.iter().enumerate() {
            if !pred.zone_may_match(zm) {
                skipped += 1;
                continue;
            }
            scanned += 1;
            sel_vec.clear();
            for (ri, &fv) in self.filter_granules[gi].iter().enumerate() {
                if pred.val_matches(fv) {
                    sel_vec.push(ri as u16);
                }
            }
            if sel_vec.is_empty() {
                continue;
            }
            // Late-materialize `metric` and `user_id` ONLY for surviving `sel_vec` offsets!
            let m_slice = &self.metric_granules[gi];
            let u_slice = &self.user_granules[gi];
            matched_rows += sel_vec.len();
            for &idx16 in &sel_vec {
                let r = idx16 as usize;
                sum_metric += m_slice[r];
                distinct_users.insert(u_slice[r]);
            }
        }

        let skip_ratio = if total_granules == 0 {
            0.0
        } else {
            (skipped as f64) / (total_granules as f64)
        };
        // ZoneMap check = 2 nJ/granule; full granule scan = 180 nJ + 6 nJ per late-materialized tuple
        let energy_nj =
            (total_granules as u64) * 2 + (scanned as u64) * 180 + (matched_rows as u64) * 6;
        ZoneScanReceipt {
            total_rows: self.total_rows,
            total_granules,
            granules_skipped_by_zonemap: skipped,
            granules_scanned: scanned,
            granule_skip_ratio: skip_ratio,
            matched_rows,
            late_materialized_sum: sum_metric,
            distinct_payload_keys: distinct_users.len(),
            energy_nj,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zonemap_skips_98_percent_of_cold_granules_with_exact_late_materialization() {
        let n = 100_000usize;
        let ts: Vec<Option<i64>> = (0..n).map(|i| Some(1_700_000_000 + i as i64)).collect();
        let rev: Vec<f64> = (0..n).map(|i| ((i % 100) as f64) * 0.5).collect();
        let uid: Vec<u64> = (0..n).map(|i| (i % 250) as u64).collect();
        let tbl = ZoneMappedTable::from_columns(&ts, &rev, &uid).expect("ZoneMappedTable");

        let rec = tbl.scan_late_materialized(ZonePredicate::Between {
            lo: 1_700_050_000,
            hi: 1_700_050_499,
        });
        assert_eq!(rec.matched_rows, 500);
        assert!(rec.granule_skip_ratio > 0.96);
        assert_eq!(rec.distinct_payload_keys, 250);
    }
}
