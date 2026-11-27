//! Vectorized Out-of-Core Multi-Table Columnar Hash Join & Subquery Engine.
//!
//! Executes star/snowflake multi-table equi-joins, derived subqueries (`FROM (SELECT ...)`),
//! CTEs (`WITH ...`), `LEFT OUTER JOIN` grouped aggregations, correlated `EXISTS` /
//! `NOT EXISTS` semi/anti-joins (including self-inequality `inner.c <> outer.c`),
//! uncorrelated `IN` / `NOT IN` sets, and correlated scalar `MIN`/`MAX`/`SUM`/`AVG`
//! subqueries directly over `.jcol` disk segments (`DiskColumnTable`) and in-memory
//! `TableColumns` snapshots — without materializing `Vec<serde_json::Value>` table scans.

#![deny(clippy::unwrap_used)]

use crate::amorphic_adapter::AmorphicTableStorage;
use crate::columnar_exec::{Col, Ints, TableColumns};
use crate::disk_col_segments::{DiskColKind, DiskColumnTable};
use crate::query::QueryResponse;
use joule_db_query::ast::{
    Expression, JoinType, Operator, OrderBy, Query as AstQuery, UnaryOperator, Value as AstValue,
};
use joule_db_query::like::LikePattern;
use joule_db_query::sql::{FromSource, SqlColumn, SqlFrom, SqlJoin, SqlQuery};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;
use std::time::Instant;

/// Fast non-cryptographic FxHasher for integer join keys & group keys.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ (b as u64)).wrapping_mul(0x517cc1b727220a95);
        }
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.0 = (self.0 ^ (i as u64)).wrapping_mul(0x517cc1b727220a95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0 ^ i).wrapping_mul(0x517cc1b727220a95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0 ^ (i as u64)).wrapping_mul(0x517cc1b727220a95);
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
type FxSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>;

const NULL_TEXT_CODE: u32 = u32::MAX;

#[derive(Clone)]
pub enum ColRel {
    Disk(Arc<DiskColumnTable>),
    Mem(Arc<TableColumns>),
}

enum SegData<'a> {
    Owned(TableColumns),
    Borrowed(&'a TableColumns),
}

impl<'a> SegData<'a> {
    #[inline]
    fn tc(&self) -> &TableColumns {
        match self {
            SegData::Owned(t) => t,
            SegData::Borrowed(t) => t,
        }
    }
}

impl ColRel {
    fn total_rows(&self) -> usize {
        match self {
            ColRel::Disk(dt) => dt.total_rows,
            ColRel::Mem(tc) => tc.len,
        }
    }

    fn num_segments(&self) -> usize {
        match self {
            ColRel::Disk(dt) => dt.segments.len(),
            ColRel::Mem(tc) => usize::from(tc.len > 0),
        }
    }

    fn col_order(&self) -> &[String] {
        match self {
            ColRel::Disk(dt) => &dt.col_order,
            ColRel::Mem(tc) => &tc.col_order,
        }
    }

    fn resolve_col_name(&self, name: &str) -> Option<String> {
        self.col_order()
            .iter()
            .find(|c| c.eq_ignore_ascii_case(name))
            .cloned()
    }

    fn col_is_int(&self, col_name: &str) -> bool {
        match self {
            ColRel::Disk(dt) => dt.col_index.get(col_name).is_some_and(|&ci| {
                matches!(
                    dt.inferred_col_kind(ci),
                    DiskColKind::I8 | DiskColKind::I16 | DiskColKind::I32 | DiskColKind::I64
                )
            }),
            ColRel::Mem(tc) => matches!(tc.cols.get(col_name), Some(Col::Int(..))),
        }
    }

    /// Prove that `col_name` is a non-null, globally unique integer key across the entire relation.
    /// Uses `O(num_segments)` zero-I/O ZoneMap interval verification first (covering dense PKs like
    /// `c_custkey`, `s_suppkey`, `p_partkey`, `n_nationkey`), falling back to a parallel key scan
    /// when `total_rows <= 2_000_000`.
    fn col_is_unique_nonnull_int(&self, col_name: &str) -> bool {
        let Some(rc) = self.resolve_col_name(col_name) else {
            return false;
        };
        if !self.col_is_int(&rc) {
            return false;
        }
        if let ColRel::Disk(dt) = self
            && let Some(&ci) = dt.col_index.get(&rc)
        {
            let mut ranges: Vec<(i64, i64)> = Vec::with_capacity(dt.segments.len());
            let mut zm_dense_ok = true;
            for seg in &dt.segments {
                if seg.len == 0 {
                    continue;
                }
                let cm = &seg.cols[ci];
                if cm.null_count != 0 {
                    return false;
                }
                if let (Some(lo), Some(hi)) = (cm.min_int(seg.len), cm.max_int(seg.len))
                    && (hi as i128) - (lo as i128) + 1 == (seg.len as i128)
                {
                    ranges.push((lo, hi));
                } else {
                    zm_dense_ok = false;
                    break;
                }
            }
            if zm_dense_ok {
                ranges.sort_unstable_by_key(|r| r.0);
                if ranges.windows(2).all(|w| w[0].1 < w[1].0) {
                    return true;
                }
            }
        }
        let total = self.total_rows();
        if total == 0 {
            return true;
        }
        if total > 2_000_000 {
            return false;
        }
        let col_refs = [rc.as_str()];
        let n_segs = self.num_segments();
        let Some(parts): Option<Vec<Vec<i64>>> = (0..n_segs)
            .into_par_iter()
            .map(|si| {
                let seg = self.load_segment(si, &col_refs)?;
                let Col::Int(ref ints, ref nulls) = *seg.tc().cols.get(&rc)? else {
                    return None;
                };
                if (0..ints.len()).any(|i| nl(nulls, i)) {
                    return None;
                }
                Some((0..ints.len()).map(|i| ints.at(i)).collect())
            })
            .collect()
        else {
            return false;
        };
        let mut all = Vec::with_capacity(total);
        for p in parts {
            all.extend(p);
        }
        IntKeySet::from_vec(all).estimated_len() == total
    }

    fn load_segment<'a>(&'a self, seg_idx: usize, cols: &[&str]) -> Option<SegData<'a>> {
        match self {
            ColRel::Disk(dt) => dt
                .load_segment_columns(seg_idx, cols)
                .ok()
                .map(SegData::Owned),
            ColRel::Mem(tc) => {
                if seg_idx == 0 {
                    // Verify none of the requested columns are Col::Other
                    for &cn in cols {
                        if matches!(tc.cols.get(cn), Some(Col::Other)) {
                            return None;
                        }
                    }
                    Some(SegData::Borrowed(tc))
                } else {
                    None
                }
            }
        }
    }

    /// ZoneMap-pruned + two-stage late-materializing segment loader:
    /// - Returns `Some(None)` when ZoneMaps or segment-local `filters` eliminate all rows
    ///   (skipping payload column `pread`s and string dictionary allocations completely!).
    /// - Returns `Some(Some((seg, sel)))` with non-empty `sel` and sparsely decoded dictionaries.
    /// - Returns `None` only on schema/expression errors so the caller can fall back cleanly.
    fn load_filtered_segment<'a>(
        &'a self,
        seg_idx: usize,
        payload_cols: &[&str],
        filters: &[LocalFilter],
    ) -> Option<Option<(SegData<'a>, Vec<u32>)>> {
        if filters.is_empty() {
            let seg = self.load_segment(seg_idx, payload_cols)?;
            let n = seg.tc().len;
            if n == 0 {
                return Some(None);
            }
            return Some(Some((seg, (0..n as u32).collect())));
        }
        match self {
            ColRel::Disk(dt) => {
                let seg_meta = dt.segments.get(seg_idx)?;
                if seg_meta.len == 0 {
                    return Some(None);
                }
                // 1. Zero-I/O ZoneMap prune against positive `KeyIn` bitsets/sets!
                for lf in filters {
                    if let LocalFilter::KeyIn {
                        col,
                        set,
                        negated: false,
                    } = lf
                    {
                        let ci_opt = dt.col_index.get(col).copied().or_else(|| {
                            dt.col_order
                                .iter()
                                .position(|x| x.eq_ignore_ascii_case(col))
                        });
                        if let Some(ci) = ci_opt {
                            let cm = &seg_meta.cols[ci];
                            if cm.null_count >= seg_meta.len {
                                return Some(None);
                            }
                            if let (Some(lo), Some(hi)) =
                                (cm.min_int(seg_meta.len), cm.max_int(seg_meta.len))
                                && !set.overlaps_range(lo, hi)
                            {
                                return Some(None);
                            }
                        }
                    }
                }
                // 2. Stage 1: load ONLY filter columns first!
                let mut fcols_set = HashSet::new();
                for lf in filters {
                    lf.collect_cols(self, &mut fcols_set);
                }
                let fcols_vec: Vec<String> = fcols_set.into_iter().collect();
                let fcols_refs: Vec<&str> = fcols_vec.iter().map(String::as_str).collect();
                let mut tc = dt.load_segment_columns(seg_idx, &fcols_refs).ok()?;
                let sel = filter_segment_rows(&tc, filters)?;
                if sel.is_empty() {
                    return Some(None);
                }
                // 3. Stage 2: sparsely load remaining payload columns only for surviving `sel` rows!
                dt.load_columns_into_selected(seg_idx, payload_cols, Some(&sel), &mut tc)
                    .ok()?;
                Some(Some((SegData::Owned(tc), sel)))
            }
            ColRel::Mem(_) => {
                let seg = self.load_segment(seg_idx, payload_cols)?;
                let sel = filter_segment_rows(seg.tc(), filters)?;
                if sel.is_empty() {
                    Some(None)
                } else {
                    Some(Some((seg, sel)))
                }
            }
        }
    }
}

/// Owned scalar value for compact build rows, group keys, and output assembly.
#[derive(Debug, Clone)]
enum OwnedVal {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
}

impl OwnedVal {
    fn to_json(&self) -> serde_json::Value {
        match self {
            OwnedVal::Null => serde_json::Value::Null,
            OwnedVal::Bool(b) => serde_json::Value::Bool(*b),
            OwnedVal::Int(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            OwnedVal::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            OwnedVal::Str(s) => serde_json::Value::String(s.to_string()),
        }
    }
}

/// Borrowed scalar value during per-row segment evaluation (zero heap allocations!).
#[derive(Debug, Clone, Copy)]
enum BVal<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'a str),
}

impl<'a> BVal<'a> {
    #[inline]
    fn as_f64(self) -> Option<f64> {
        match self {
            BVal::Int(i) => Some(i as f64),
            BVal::Float(f) => Some(f),
            _ => None,
        }
    }

    #[inline]
    fn as_i64(self) -> Option<i64> {
        match self {
            BVal::Int(i) => Some(i),
            _ => None,
        }
    }

    fn to_owned_val(self) -> OwnedVal {
        match self {
            BVal::Null => OwnedVal::Null,
            BVal::Bool(b) => OwnedVal::Bool(b),
            BVal::Int(i) => OwnedVal::Int(i),
            BVal::Float(f) => OwnedVal::Float(f),
            BVal::Str(s) => OwnedVal::Str(Arc::from(s)),
        }
    }
}

#[inline]
fn nl(nulls: &[bool], row: usize) -> bool {
    !nulls.is_empty() && nulls[row]
}

#[inline]
fn read_col_bval(col: &Col, row: usize) -> BVal<'_> {
    match col {
        Col::Int(ints, nulls) => {
            if nl(nulls, row) {
                BVal::Null
            } else {
                BVal::Int(ints.at(row))
            }
        }
        Col::Float(vals, nulls) => {
            if nl(nulls, row) {
                BVal::Null
            } else {
                BVal::Float(vals[row])
            }
        }
        Col::Text(codes, dict) => {
            let c = codes[row];
            if c == NULL_TEXT_CODE {
                BVal::Null
            } else {
                dict.get(c as usize)
                    .map_or(BVal::Null, |s| BVal::Str(s.as_ref()))
            }
        }
        Col::Other => BVal::Null,
    }
}

/// Summary for correlated `EXISTS (... WHERE inner.k = outer.k AND inner.c <> outer.c)`.
#[derive(Debug, Clone, Copy)]
enum NeqSummary {
    One(i64),
    Many,
}

/// Dense integer bitset for `[min_key ..= max_key]` semi/anti-joins (`<= 8 MB`).
#[derive(Debug, Clone)]
struct DenseKeyBits {
    min_key: i64,
    max_key: i64,
    words: Vec<u64>,
}

impl DenseKeyBits {
    fn from_keys(keys: &[i64]) -> Option<Self> {
        if keys.is_empty() {
            return Some(Self {
                min_key: 1,
                max_key: 0,
                words: Vec::new(),
            });
        }
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        for &k in keys {
            lo = lo.min(k);
            hi = hi.max(k);
        }
        let span = (hi as i128) - (lo as i128);
        if !(0..=64_000_000).contains(&span) {
            return None;
        }
        let nwords = ((span as usize) + 64) / 64;
        let mut words = vec![0u64; nwords];
        for &k in keys {
            let bit = (k - lo) as usize;
            words[bit >> 6] |= 1u64 << (bit & 63);
        }
        Some(Self {
            min_key: lo,
            max_key: hi,
            words,
        })
    }

    #[inline]
    fn contains(&self, k: i64) -> bool {
        if k < self.min_key || k > self.max_key {
            return false;
        }
        let bit = (k - self.min_key) as usize;
        (self.words[bit >> 6] >> (bit & 63)) & 1 != 0
    }

    #[inline]
    fn overlaps_range(&self, lo: i64, hi: i64) -> bool {
        if self.words.is_empty() || hi < self.min_key || lo > self.max_key {
            return false;
        }
        let c_lo = lo.max(self.min_key);
        let c_hi = hi.min(self.max_key);
        let b0 = (c_lo - self.min_key) as usize;
        let b1 = (c_hi - self.min_key) as usize;
        let w0 = b0 >> 6;
        let w1 = b1 >> 6;
        let start_mask = !0u64 << (b0 & 63);
        let end_mask = if (b1 & 63) == 63 {
            !0u64
        } else {
            (1u64 << ((b1 & 63) + 1)) - 1
        };
        if w0 == w1 {
            return (self.words[w0] & start_mask & end_mask) != 0;
        }
        if (self.words[w0] & start_mask) != 0 || (self.words[w1] & end_mask) != 0 {
            return true;
        }
        self.words[(w0 + 1)..w1].iter().any(|&w| w != 0)
    }
}

#[derive(Clone)]
enum IntKeySet {
    Dense(Arc<DenseKeyBits>),
    Hash(Arc<FxSet<i64>>),
}

impl IntKeySet {
    fn from_vec(keys: Vec<i64>) -> Self {
        if let Some(d) = DenseKeyBits::from_keys(&keys) {
            IntKeySet::Dense(Arc::new(d))
        } else {
            let mut s = FxSet::default();
            s.reserve(keys.len());
            for k in keys {
                s.insert(k);
            }
            IntKeySet::Hash(Arc::new(s))
        }
    }

    #[inline]
    fn contains(&self, k: i64) -> bool {
        match self {
            IntKeySet::Dense(d) => d.contains(k),
            IntKeySet::Hash(s) => s.contains(&k),
        }
    }

    #[inline]
    fn overlaps_range(&self, lo: i64, hi: i64) -> bool {
        if lo > hi {
            return false;
        }
        match self {
            IntKeySet::Dense(d) => d.overlaps_range(lo, hi),
            IntKeySet::Hash(s) => {
                if s.is_empty() {
                    false
                } else if (hi as i128) - (lo as i128) <= (s.len() as i128).min(256) {
                    (lo..=hi).any(|k| s.contains(&k))
                } else {
                    s.iter().any(|&k| k >= lo && k <= hi)
                }
            }
        }
    }

    fn estimated_len(&self) -> usize {
        match self {
            IntKeySet::Dense(d) => d.words.iter().map(|w| w.count_ones() as usize).sum(),
            IntKeySet::Hash(s) => s.len(),
        }
    }
}

/// Segment-local predicate over a single relation (evaluated before probing joins).
#[derive(Clone)]
enum LocalFilter {
    Expr(Expression),
    KeyIn {
        col: String,
        set: IntKeySet,
        negated: bool,
    },
    NeqExists {
        key_col: String,
        neq_col: String,
        shards: Arc<Vec<FxMap<i64, NeqSummary>>>,
        negated: bool,
    },
    CorrScalar1 {
        key_col: String,
        lhs_expr: Expression,
        op: Operator,
        map: Arc<FxMap<i64, f64>>,
    },
    CorrScalar2 {
        key_col1: String,
        key_col2: String,
        lhs_expr: Expression,
        op: Operator,
        map: Arc<FxMap<(i64, i64), f64>>,
    },
}

impl LocalFilter {
    fn collect_cols(&self, rel: &ColRel, out: &mut HashSet<String>) {
        match self {
            LocalFilter::Expr(e) => collect_single_rel_cols(e, rel, out),
            LocalFilter::KeyIn { col, .. } => {
                if let Some(c) = rel.resolve_col_name(col) {
                    out.insert(c);
                }
            }
            LocalFilter::NeqExists {
                key_col, neq_col, ..
            } => {
                if let Some(c) = rel.resolve_col_name(key_col) {
                    out.insert(c);
                }
                if let Some(c) = rel.resolve_col_name(neq_col) {
                    out.insert(c);
                }
            }
            LocalFilter::CorrScalar1 {
                key_col, lhs_expr, ..
            } => {
                if let Some(c) = rel.resolve_col_name(key_col) {
                    out.insert(c);
                }
                collect_single_rel_cols(lhs_expr, rel, out);
            }
            LocalFilter::CorrScalar2 {
                key_col1,
                key_col2,
                lhs_expr,
                ..
            } => {
                if let Some(c) = rel.resolve_col_name(key_col1) {
                    out.insert(c);
                }
                if let Some(c) = rel.resolve_col_name(key_col2) {
                    out.insert(c);
                }
                collect_single_rel_cols(lhs_expr, rel, out);
            }
        }
    }
}

/// Public entrypoint called by `SimpleQueryExecutor::execute_statement`.
pub fn try_execute_columnar_join(
    query: &SqlQuery,
    amorphic: &AmorphicTableStorage,
    request_limit: Option<usize>,
    start: Instant,
) -> Option<QueryResponse> {
    if query.set_op.is_some() || query.as_of_timestamp.is_some() || query.distinct {
        return None;
    }
    if query.from.as_ref().is_some_and(|f| f.lateral)
        || query
            .joins
            .iter()
            .any(|j| j.lateral || j.source_subquery.is_some())
    {
        return None;
    }
    if !query_benefits_from_columnar_join(query, amorphic) {
        return None;
    }

    let mut cte_rels: HashMap<String, ColRel> = HashMap::new();
    for cte in &query.ctes {
        if cte.recursive || cte.query.set_op.is_some() || cte.query.distinct {
            return None;
        }
        let tc = exec_query_to_table_columns(&cte.query, Some(&cte.columns), amorphic, &cte_rels)?;
        cte_rels.insert(cte.name.to_ascii_lowercase(), ColRel::Mem(Arc::new(tc)));
    }

    let mut q_no_ctes = query.clone();
    q_no_ctes.ctes.clear();
    let res = exec_core_query(&q_no_ctes, amorphic, &cte_rels);
    drop(cte_rels);
    if start.elapsed().as_millis() >= 25 {
        crate::disk_col_segments::trim_unused_heap();
    }
    let (columns, mut rows) = res?;

    let mut truncated = false;
    if query.limit.is_none()
        && let Some(rlim) = request_limit
        && rows.len() > rlim
    {
        rows.truncate(rlim);
        truncated = true;
    }

    Some(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    })
}

fn table_is_columnar_scale(name: &str, amorphic: &AmorphicTableStorage) -> bool {
    if amorphic.disk_column_table(name).is_some() {
        return true;
    }
    amorphic.table_columns(name).is_some_and(|tc| tc.len >= 256)
}

fn expr_has_columnar_scale_subquery(expr: &Expression, amorphic: &AmorphicTableStorage) -> bool {
    match expr {
        Expression::Exists(subquery) | Expression::Subquery(subquery) => {
            ast_query_has_columnar_scale(subquery, amorphic)
        }
        Expression::In {
            expr: inner, list, ..
        } => {
            expr_has_columnar_scale_subquery(inner, amorphic)
                || list
                    .iter()
                    .any(|e| expr_has_columnar_scale_subquery(e, amorphic))
        }
        Expression::Binary { left, right, .. } => {
            expr_has_columnar_scale_subquery(left, amorphic)
                || expr_has_columnar_scale_subquery(right, amorphic)
        }
        Expression::Unary { expr: inner, .. }
        | Expression::IsNull { expr: inner, .. }
        | Expression::Like { expr: inner, .. }
        | Expression::Cast { expr: inner, .. } => expr_has_columnar_scale_subquery(inner, amorphic),
        Expression::Between {
            expr: e, low, high, ..
        } => {
            expr_has_columnar_scale_subquery(e, amorphic)
                || expr_has_columnar_scale_subquery(low, amorphic)
                || expr_has_columnar_scale_subquery(high, amorphic)
        }
        _ => false,
    }
}

fn ast_query_has_columnar_scale(q: &AstQuery, amorphic: &AmorphicTableStorage) -> bool {
    if let Some(ref src) = q.source
        && table_is_columnar_scale(src, amorphic)
    {
        return true;
    }
    if q.joins
        .iter()
        .any(|j| table_is_columnar_scale(&j.table, amorphic))
    {
        return true;
    }
    q.filter
        .as_ref()
        .is_some_and(|w| expr_has_columnar_scale_subquery(w, amorphic))
}

fn query_benefits_from_columnar_join(query: &SqlQuery, amorphic: &AmorphicTableStorage) -> bool {
    for cte in &query.ctes {
        if query_benefits_from_columnar_join(&cte.query, amorphic) {
            return true;
        }
    }
    if let Some(ref f) = query.from {
        match &f.source {
            FromSource::Table(name) if table_is_columnar_scale(name, amorphic) => return true,
            FromSource::Subquery(sub) if query_benefits_from_columnar_join(sub, amorphic) => {
                return true;
            }
            _ => {}
        }
    }
    for j in &query.joins {
        if table_is_columnar_scale(&j.table, amorphic) {
            return true;
        }
    }
    query
        .where_clause
        .as_ref()
        .is_some_and(|w| expr_has_columnar_scale_subquery(w, amorphic))
}

fn load_rel(
    name: &str,
    amorphic: &AmorphicTableStorage,
    cte_rels: &HashMap<String, ColRel>,
) -> Option<ColRel> {
    if let Some(r) = cte_rels.get(&name.to_ascii_lowercase()) {
        return Some(r.clone());
    }
    if amorphic.get_view(name).is_some() || amorphic.has_computed_columns(name) {
        return None;
    }
    if let Ok(defs) = amorphic.get_column_defs(name)
        && defs.iter().any(|d| {
            let u = d.data_type.to_ascii_uppercase();
            u.starts_with("NUMERIC") || u.starts_with("DECIMAL")
        })
    {
        return None;
    }
    if let Some(dt) = amorphic.disk_column_table(name) {
        return Some(ColRel::Disk(dt));
    }
    amorphic.table_columns(name).map(ColRel::Mem)
}

fn rows_to_table_columns(
    col_names: &[String],
    rows: &[Vec<serde_json::Value>],
) -> Option<TableColumns> {
    let n = rows.len();
    let ncols = col_names.len();
    let mut decoded: Vec<(String, Col)> = Vec::with_capacity(ncols);
    for (ci, cname) in col_names.iter().enumerate() {
        let mut saw_int = false;
        let mut saw_float = false;
        let mut saw_str = false;
        for r in rows {
            match r.get(ci) {
                None | Some(serde_json::Value::Null) => {}
                Some(serde_json::Value::Number(num)) => {
                    if num.is_i64() {
                        saw_int = true;
                    } else {
                        saw_float = true;
                    }
                }
                Some(serde_json::Value::String(_)) => saw_str = true,
                _ => return None,
            }
        }
        if saw_str && (saw_int || saw_float) {
            return None;
        }
        let col = if saw_str {
            let mut dict: Vec<Arc<str>> = Vec::new();
            let mut map: HashMap<String, u32> = HashMap::new();
            let mut codes = Vec::with_capacity(n);
            for r in rows {
                match r.get(ci) {
                    Some(serde_json::Value::String(s)) => {
                        if let Some(&c) = map.get(s) {
                            codes.push(c);
                        } else {
                            let c = dict.len() as u32;
                            dict.push(Arc::from(s.as_str()));
                            map.insert(s.clone(), c);
                            codes.push(c);
                        }
                    }
                    _ => codes.push(NULL_TEXT_CODE),
                }
            }
            Col::Text(codes, dict)
        } else if saw_float {
            let mut vals = Vec::with_capacity(n);
            let mut nulls = Vec::with_capacity(n);
            let mut any_null = false;
            for r in rows {
                match r.get(ci).and_then(|v| v.as_f64()) {
                    Some(f) => {
                        vals.push(f);
                        nulls.push(false);
                    }
                    None => {
                        vals.push(0.0);
                        nulls.push(true);
                        any_null = true;
                    }
                }
            }
            if !any_null {
                nulls.clear();
            }
            Col::Float(vals, nulls)
        } else {
            let mut vals = Vec::with_capacity(n);
            let mut nulls = Vec::with_capacity(n);
            let mut any_null = false;
            for r in rows {
                match r.get(ci).and_then(|v| v.as_i64()) {
                    Some(v) => {
                        vals.push(v);
                        nulls.push(false);
                    }
                    None => {
                        vals.push(0);
                        nulls.push(true);
                        any_null = true;
                    }
                }
            }
            if !any_null {
                nulls.clear();
            }
            Col::Int(Ints::I64(vals), nulls)
        };
        decoded.push((cname.clone(), col));
    }
    let mut tc = TableColumns::empty_with_len(n, col_names.to_vec());
    tc.insert_decoded_columns(n, decoded);
    Some(tc)
}

fn exec_query_to_table_columns(
    query: &SqlQuery,
    rename_cols: Option<&[String]>,
    amorphic: &AmorphicTableStorage,
    cte_rels: &HashMap<String, ColRel>,
) -> Option<TableColumns> {
    // Fast single-table path via `columnar_exec` if applicable
    if query.ctes.is_empty()
        && query.joins.is_empty()
        && query.set_op.is_some() == false
        && !query.distinct
        && query.as_of_timestamp.is_none()
        && let Some(tbl) = query.from.as_ref().and_then(|f| f.table_name())
        && !cte_rels.contains_key(&tbl.to_ascii_lowercase())
        && !expr_opt_has_subquery(query.where_clause.as_ref())
        && !expr_opt_has_subquery(query.having.as_ref())
    {
        if let Some(rel) = load_rel(tbl, amorphic, cte_rels) {
            let res_opt = match &rel {
                ColRel::Disk(dt) => crate::columnar_exec::execute_segmented(query, dt),
                ColRel::Mem(tc) => crate::columnar_exec::execute(query, tc),
            };
            if let Some(res) = res_opt {
                let cols = match rename_cols {
                    Some(rc) if rc.len() == res.columns.len() => rc.to_vec(),
                    _ => res.columns,
                };
                return rows_to_table_columns(&cols, &res.rows);
            }
        }
    }

    let (cols, rows) = exec_core_query(query, amorphic, cte_rels)?;
    let final_cols = match rename_cols {
        Some(rc) if rc.len() == cols.len() => rc.to_vec(),
        _ => cols,
    };
    rows_to_table_columns(&final_cols, &rows)
}

fn expr_opt_has_subquery(e: Option<&Expression>) -> bool {
    e.is_some_and(expr_has_subquery)
}

fn expr_has_subquery(e: &Expression) -> bool {
    match e {
        Expression::Subquery(_) | Expression::Exists(_) => true,
        Expression::Binary { left, right, .. } => {
            expr_has_subquery(left) || expr_has_subquery(right)
        }
        Expression::Unary { expr: x, .. }
        | Expression::IsNull { expr: x, .. }
        | Expression::Like { expr: x, .. }
        | Expression::Cast { expr: x, .. } => expr_has_subquery(x),
        Expression::Function { args, .. } => args.iter().any(expr_has_subquery),
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_deref().is_some_and(expr_has_subquery)
                || when_clauses
                    .iter()
                    .any(|(w, t)| expr_has_subquery(w) || expr_has_subquery(t))
                || else_clause.as_deref().is_some_and(expr_has_subquery)
        }
        Expression::In { expr: x, list, .. } => {
            expr_has_subquery(x) || list.iter().any(expr_has_subquery)
        }
        Expression::Between {
            expr: x, low, high, ..
        } => expr_has_subquery(x) || expr_has_subquery(low) || expr_has_subquery(high),
        _ => false,
    }
}

fn ast_query_to_sql_query(q: &AstQuery) -> Option<SqlQuery> {
    let src = q.source.as_ref()?;
    let columns: Vec<SqlColumn> = q
        .columns
        .iter()
        .map(|cn| {
            if cn == "*" {
                SqlColumn {
                    expr: Expression::Wildcard,
                    alias: None,
                }
            } else if let Some(dexpr) = q.derived_columns.get(cn) {
                SqlColumn {
                    expr: dexpr.clone(),
                    alias: Some(cn.clone()),
                }
            } else {
                SqlColumn {
                    expr: Expression::Column(cn.clone()),
                    alias: None,
                }
            }
        })
        .collect();
    let mut ctes = Vec::with_capacity(q.ctes.len());
    for c in &q.ctes {
        ctes.push(joule_db_query::sql::SqlCTE {
            name: c.name.clone(),
            columns: c.columns.clone(),
            query: Box::new(ast_query_to_sql_query(&c.query)?),
            recursive: c.recursive,
        });
    }
    let joins: Vec<SqlJoin> = q
        .joins
        .iter()
        .map(|j| SqlJoin {
            join_type: j.join_type,
            table: j.table.clone(),
            alias: j.alias.clone(),
            condition: j.condition.clone(),
            using_columns: j.using_columns.clone(),
            source_subquery: None,
            lateral: false,
        })
        .collect();
    Some(SqlQuery {
        ctes,
        distinct: q.distinct,
        columns,
        from: Some(SqlFrom {
            source: FromSource::Table(src.clone()),
            alias: q.source_alias.clone(),
            lateral: false,
        }),
        joins,
        where_clause: q.filter.clone(),
        group_by: q.group_by.clone(),
        having: q.having.clone(),
        order_by: q.order_by.clone(),
        limit: q.limit,
        offset: q.offset,
        set_op: None,
        as_of_timestamp: None,
    })
}

fn expr_has_any_agg(e: &Expression) -> bool {
    match e {
        Expression::Function { name, args } => {
            let up = name.to_ascii_uppercase();
            if matches!(
                up.as_str(),
                "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | "COUNT_DISTINCT"
            ) {
                return true;
            }
            args.iter().any(expr_has_any_agg)
        }
        Expression::Binary { left, right, .. } => expr_has_any_agg(left) || expr_has_any_agg(right),
        Expression::Unary { expr: x, .. } | Expression::Cast { expr: x, .. } => expr_has_any_agg(x),
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_deref().is_some_and(expr_has_any_agg)
                || when_clauses
                    .iter()
                    .any(|(w, t)| expr_has_any_agg(w) || expr_has_any_agg(t))
                || else_clause.as_deref().is_some_and(expr_has_any_agg)
        }
        _ => false,
    }
}

fn subst_derived_expr(
    e: &Expression,
    sub_alias: Option<&str>,
    map: &HashMap<String, Expression>,
) -> Expression {
    match e {
        Expression::Column(c) => {
            if let Some((t, bare)) = c.rsplit_once('.') {
                if sub_alias.is_some_and(|a| a.eq_ignore_ascii_case(t))
                    && let Some(rep) = map.get(&bare.to_ascii_lowercase())
                {
                    return rep.clone();
                }
            }
            map.get(&c.to_ascii_lowercase())
                .cloned()
                .unwrap_or_else(|| e.clone())
        }
        Expression::QualifiedColumn { table, column } => {
            if sub_alias.is_some_and(|a| a.eq_ignore_ascii_case(table))
                && let Some(rep) = map.get(&column.to_ascii_lowercase())
            {
                return rep.clone();
            }
            e.clone()
        }
        Expression::Binary { left, op, right } => Expression::Binary {
            left: Box::new(subst_derived_expr(left, sub_alias, map)),
            op: *op,
            right: Box::new(subst_derived_expr(right, sub_alias, map)),
        },
        Expression::Unary { op, expr: inner } => Expression::Unary {
            op: *op,
            expr: Box::new(subst_derived_expr(inner, sub_alias, map)),
        },
        Expression::Function { name, args } => Expression::Function {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| subst_derived_expr(a, sub_alias, map))
                .collect(),
        },
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => Expression::Case {
            operand: operand
                .as_ref()
                .map(|o| Box::new(subst_derived_expr(o, sub_alias, map))),
            when_clauses: when_clauses
                .iter()
                .map(|(w, t)| {
                    (
                        subst_derived_expr(w, sub_alias, map),
                        subst_derived_expr(t, sub_alias, map),
                    )
                })
                .collect(),
            else_clause: else_clause
                .as_ref()
                .map(|el| Box::new(subst_derived_expr(el, sub_alias, map))),
        },
        Expression::In {
            expr: inner,
            list,
            negated,
        } => Expression::In {
            expr: Box::new(subst_derived_expr(inner, sub_alias, map)),
            list: list
                .iter()
                .map(|x| subst_derived_expr(x, sub_alias, map))
                .collect(),
            negated: *negated,
        },
        Expression::Between {
            expr: inner,
            low,
            high,
            negated,
        } => Expression::Between {
            expr: Box::new(subst_derived_expr(inner, sub_alias, map)),
            low: Box::new(subst_derived_expr(low, sub_alias, map)),
            high: Box::new(subst_derived_expr(high, sub_alias, map)),
            negated: *negated,
        },
        Expression::IsNull {
            expr: inner,
            negated,
        } => Expression::IsNull {
            expr: Box::new(subst_derived_expr(inner, sub_alias, map)),
            negated: *negated,
        },
        Expression::Like {
            expr: inner,
            pattern,
            negated,
            case_insensitive,
        } => Expression::Like {
            expr: Box::new(subst_derived_expr(inner, sub_alias, map)),
            pattern: pattern.clone(),
            negated: *negated,
            case_insensitive: *case_insensitive,
        },
        Expression::Cast {
            expr: inner,
            target_type,
        } => Expression::Cast {
            expr: Box::new(subst_derived_expr(inner, sub_alias, map)),
            target_type: target_type.clone(),
        },
        other => other.clone(),
    }
}

fn flatten_derived_subquery(query: &SqlQuery) -> Option<SqlQuery> {
    let from = query.from.as_ref()?;
    let FromSource::Subquery(sub) = &from.source else {
        return None;
    };
    if !query.joins.is_empty()
        || !sub.ctes.is_empty()
        || !sub.group_by.is_empty()
        || sub.having.is_some()
        || !sub.order_by.is_empty()
        || sub.limit.is_some()
        || sub.offset.is_some()
        || sub.distinct
        || sub.set_op.is_some()
        || sub.columns.iter().any(|c| {
            expr_has_any_agg(&c.expr)
                || matches!(
                    c.expr,
                    Expression::Wildcard | Expression::QualifiedWildcard(_)
                )
        })
    {
        return None;
    }

    let mut map: HashMap<String, Expression> = HashMap::with_capacity(sub.columns.len());
    for sc in &sub.columns {
        let out_name = sc.alias.clone().or_else(|| match &sc.expr {
            Expression::Column(c) => Some(c.clone()),
            Expression::QualifiedColumn { column, .. } => Some(column.clone()),
            _ => None,
        })?;
        map.insert(out_name.to_ascii_lowercase(), sc.expr.clone());
    }
    let sub_alias = from.alias.as_deref();

    let new_cols: Vec<SqlColumn> = query
        .columns
        .iter()
        .map(|sc| {
            let default_alias = sc.alias.clone().or_else(|| match &sc.expr {
                Expression::Column(c) => Some(c.clone()),
                Expression::QualifiedColumn { column, .. } => Some(column.clone()),
                _ => None,
            });
            SqlColumn {
                expr: subst_derived_expr(&sc.expr, sub_alias, &map),
                alias: default_alias,
            }
        })
        .collect();

    let outer_w = query
        .where_clause
        .as_ref()
        .map(|w| subst_derived_expr(w, sub_alias, &map));
    let merged_where = match (sub.where_clause.clone(), outer_w) {
        (Some(a), Some(b)) => Some(Expression::and(a, b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    let new_group_by: Vec<Expression> = query
        .group_by
        .iter()
        .map(|g| subst_derived_expr(g, sub_alias, &map))
        .collect();
    let new_having = query
        .having
        .as_ref()
        .map(|h| subst_derived_expr(h, sub_alias, &map));
    let new_order_by: Vec<OrderBy> = query
        .order_by
        .iter()
        .map(|ob| OrderBy {
            expr: subst_derived_expr(&ob.expr, sub_alias, &map),
            descending: ob.descending,
            nulls_first: ob.nulls_first,
        })
        .collect();

    Some(SqlQuery {
        ctes: Vec::new(),
        distinct: query.distinct,
        columns: new_cols,
        from: sub.from.clone(),
        joins: sub.joins.clone(),
        where_clause: merged_where,
        group_by: new_group_by,
        having: new_having,
        order_by: new_order_by,
        limit: query.limit,
        offset: query.offset,
        set_op: None,
        as_of_timestamp: None,
    })
}

#[derive(Clone)]
struct RelEntry {
    alias: String,
    table: String,
    rel: ColRel,
}

impl RelEntry {
    fn matches_qualifier(&self, q: &str) -> bool {
        self.alias.eq_ignore_ascii_case(q)
            || (self.alias.eq_ignore_ascii_case(&self.table) && self.table.eq_ignore_ascii_case(q))
    }
}

fn resolve_rel_col(rels: &[RelEntry], expr: &Expression) -> Option<(usize, String)> {
    match expr {
        Expression::QualifiedColumn { table, column } => {
            let mut found = None;
            for (ri, r) in rels.iter().enumerate() {
                if r.matches_qualifier(table)
                    && let Some(real_c) = r.rel.resolve_col_name(column)
                {
                    if found.is_some() {
                        return None;
                    }
                    found = Some((ri, real_c));
                }
            }
            found
        }
        Expression::Column(name) => {
            if let Some((t, c)) = name.rsplit_once('.') {
                return resolve_rel_col(
                    rels,
                    &Expression::QualifiedColumn {
                        table: t.to_string(),
                        column: c.to_string(),
                    },
                );
            }
            let mut found = None;
            for (ri, r) in rels.iter().enumerate() {
                if let Some(real_c) = r.rel.resolve_col_name(name) {
                    if found.is_some() {
                        return None;
                    }
                    found = Some((ri, real_c));
                }
            }
            found
        }
        _ => None,
    }
}

fn expr_referenced_rels(expr: &Expression, rels: &[RelEntry], out: &mut HashSet<usize>) -> bool {
    match expr {
        Expression::Literal(_) => true,
        Expression::Column(_) | Expression::QualifiedColumn { .. } => {
            if let Some((ri, _)) = resolve_rel_col(rels, expr) {
                out.insert(ri);
                true
            } else {
                false
            }
        }
        Expression::Binary { left, right, .. } => {
            expr_referenced_rels(left, rels, out) && expr_referenced_rels(right, rels, out)
        }
        Expression::Unary { expr: x, .. }
        | Expression::IsNull { expr: x, .. }
        | Expression::Like { expr: x, .. }
        | Expression::Cast { expr: x, .. } => expr_referenced_rels(x, rels, out),
        Expression::Function { args, .. } => {
            args.iter().all(|a| expr_referenced_rels(a, rels, out))
        }
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand
                .as_deref()
                .is_none_or(|o| expr_referenced_rels(o, rels, out))
                && when_clauses.iter().all(|(w, t)| {
                    expr_referenced_rels(w, rels, out) && expr_referenced_rels(t, rels, out)
                })
                && else_clause
                    .as_deref()
                    .is_none_or(|el| expr_referenced_rels(el, rels, out))
        }
        Expression::In { expr: x, list, .. } => {
            expr_referenced_rels(x, rels, out)
                && list.iter().all(|i| expr_referenced_rels(i, rels, out))
        }
        Expression::Between {
            expr: x, low, high, ..
        } => {
            expr_referenced_rels(x, rels, out)
                && expr_referenced_rels(low, rels, out)
                && expr_referenced_rels(high, rels, out)
        }
        _ => false,
    }
}

fn split_and_owned(e: &Expression, out: &mut Vec<Expression>) {
    if let Expression::Binary {
        left,
        op: Operator::And,
        right,
    } = e
    {
        split_and_owned(left, out);
        split_and_owned(right, out);
    } else {
        out.push(e.clone());
    }
}

fn split_or_ref<'a>(e: &'a Expression, out: &mut Vec<&'a Expression>) {
    if let Expression::Binary {
        left,
        op: Operator::Or,
        right,
    } = e
    {
        split_or_ref(left, out);
        split_or_ref(right, out);
    } else {
        out.push(e);
    }
}

fn combine_and(preds: Vec<Expression>) -> Option<Expression> {
    preds.into_iter().reduce(Expression::and)
}

fn combine_or(preds: Vec<Expression>) -> Option<Expression> {
    preds.into_iter().reduce(Expression::or)
}

/// Factor common conjuncts out of top-level `OR` branches AND extract per-relation
/// bounding union filters (`(A1 AND B1) OR (A2 AND B2)` => `A1 OR A2` on rel A, `B1 OR B2` on rel B).
fn extract_or_bounding_filters(
    or_expr: &Expression,
    rels: &[RelEntry],
) -> Vec<(usize, Expression)> {
    let mut branches = Vec::new();
    split_or_ref(or_expr, &mut branches);
    if branches.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for ri in 0..rels.len() {
        let mut rel_branch_exprs = Vec::with_capacity(branches.len());
        let mut all_branches_constrain_ri = true;
        for b in &branches {
            let mut conjs = Vec::new();
            split_and_owned(b, &mut conjs);
            let mut ri_conjs = Vec::new();
            for c in conjs {
                let mut touched = HashSet::new();
                if expr_referenced_rels(&c, rels, &mut touched)
                    && touched.len() == 1
                    && touched.contains(&ri)
                {
                    ri_conjs.push(c);
                }
            }
            if let Some(combined) = combine_and(ri_conjs) {
                rel_branch_exprs.push(combined);
            } else {
                all_branches_constrain_ri = false;
                break;
            }
        }
        if all_branches_constrain_ri && let Some(or_bound) = combine_or(rel_branch_exprs) {
            out.push((ri, or_bound));
        }
    }
    out
}

fn collect_single_rel_cols(e: &Expression, rel: &ColRel, out: &mut HashSet<String>) {
    match e {
        Expression::Column(c) => {
            let bare = c.rsplit_once('.').map_or(c.as_str(), |(_, b)| b);
            if let Some(rc) = rel.resolve_col_name(bare) {
                out.insert(rc);
            }
        }
        Expression::QualifiedColumn { column, .. } => {
            if let Some(rc) = rel.resolve_col_name(column) {
                out.insert(rc);
            }
        }
        Expression::Binary { left, right, .. } => {
            collect_single_rel_cols(left, rel, out);
            collect_single_rel_cols(right, rel, out);
        }
        Expression::Unary { expr: x, .. }
        | Expression::IsNull { expr: x, .. }
        | Expression::Like { expr: x, .. }
        | Expression::Cast { expr: x, .. } => collect_single_rel_cols(x, rel, out),
        Expression::Function { args, .. } => {
            for a in args {
                collect_single_rel_cols(a, rel, out);
            }
        }
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(o) = operand {
                collect_single_rel_cols(o, rel, out);
            }
            for (w, t) in when_clauses {
                collect_single_rel_cols(w, rel, out);
                collect_single_rel_cols(t, rel, out);
            }
            if let Some(el) = else_clause {
                collect_single_rel_cols(el, rel, out);
            }
        }
        Expression::In { expr: x, list, .. } => {
            collect_single_rel_cols(x, rel, out);
            for i in list {
                collect_single_rel_cols(i, rel, out);
            }
        }
        Expression::Between {
            expr: x, low, high, ..
        } => {
            collect_single_rel_cols(x, rel, out);
            collect_single_rel_cols(low, rel, out);
            collect_single_rel_cols(high, rel, out);
        }
        _ => {}
    }
}

/// Precompiled per-segment expression over a single `TableColumns` segment (fast dictionary bitsets!).
enum SegExpr {
    LitNull,
    LitBool(bool),
    LitInt(i64),
    LitFloat(f64),
    LitStr(String),
    Col(usize),
    Neg(Box<SegExpr>),
    Not(Box<SegExpr>),
    IsNull(Box<SegExpr>, bool),
    Arith(Box<SegExpr>, Operator, Box<SegExpr>),
    Cmp(Box<SegExpr>, Operator, Box<SegExpr>),
    And(Box<SegExpr>, Box<SegExpr>),
    Or(Box<SegExpr>, Box<SegExpr>),
    Between {
        expr: Box<SegExpr>,
        low: Box<SegExpr>,
        high: Box<SegExpr>,
        negated: bool,
    },
    InList {
        expr: Box<SegExpr>,
        list: Vec<SegExpr>,
        negated: bool,
    },
    Like {
        expr: Box<SegExpr>,
        pat: LikePattern,
        negated: bool,
    },
    /// O(1) per-row dictionary bitset for any single-column `Text` predicate (`=`, `<`, `IN`, `LIKE`, `SUBSTRING IN`)!
    DictBitset {
        col_idx: usize,
        mask: Vec<bool>,
    },
    Substr {
        expr: Box<SegExpr>,
        start_1based: usize,
        len: Option<usize>,
    },
    ExtractYear(Box<SegExpr>),
    ExtractMonth(Box<SegExpr>),
    ExtractDay(Box<SegExpr>),
}

fn parse_iso_ymd_fast(s: &str) -> Option<(i32, u32, u32)> {
    let b = s.as_bytes();
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let d = |c: u8| -> Option<i32> {
        if c.is_ascii_digit() {
            Some((c - b'0') as i32)
        } else {
            None
        }
    };
    let y = d(b[0])? * 1000 + d(b[1])? * 100 + d(b[2])? * 10 + d(b[3])?;
    let m = (d(b[5])? * 10 + d(b[6])?) as u32;
    let day = (d(b[8])? * 10 + d(b[9])?) as u32;
    Some((y, m, day))
}

fn epoch_secs_to_ymd(secs: i64) -> (i32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u32, d as u32)
}

fn epoch_secs_to_iso10(secs: i64) -> String {
    let (y, m, d) = epoch_secs_to_ymd(secs);
    format!("{y:04}-{m:02}-{d:02}")
}

#[inline]
fn cmp_bvals(a: BVal<'_>, b: BVal<'_>) -> Option<Ordering> {
    match (a, b) {
        (BVal::Null, _) | (_, BVal::Null) => None,
        (BVal::Int(x), BVal::Int(y)) => Some(x.cmp(&y)),
        (BVal::Float(x), BVal::Float(y)) => x.partial_cmp(&y),
        (BVal::Int(x), BVal::Float(y)) => (x as f64).partial_cmp(&y),
        (BVal::Float(x), BVal::Int(y)) => x.partial_cmp(&(y as f64)),
        (BVal::Int(x), BVal::Str(y)) => {
            y.trim()
                .parse::<i64>()
                .ok()
                .map(|yi| x.cmp(&yi))
                .or_else(|| {
                    y.trim()
                        .parse::<f64>()
                        .ok()
                        .and_then(|yf| (x as f64).partial_cmp(&yf))
                })
        }
        (BVal::Str(x), BVal::Int(y)) => {
            x.trim()
                .parse::<i64>()
                .ok()
                .map(|xi| xi.cmp(&y))
                .or_else(|| {
                    x.trim()
                        .parse::<f64>()
                        .ok()
                        .and_then(|xf| xf.partial_cmp(&(y as f64)))
                })
        }
        (BVal::Float(x), BVal::Str(y)) => y
            .trim()
            .parse::<f64>()
            .ok()
            .and_then(|yf| x.partial_cmp(&yf)),
        (BVal::Str(x), BVal::Float(y)) => x
            .trim()
            .parse::<f64>()
            .ok()
            .and_then(|xf| xf.partial_cmp(&y)),
        (BVal::Str(x), BVal::Str(y)) => {
            if x.len() >= 10
                && y.len() == 10
                && x.as_bytes().get(4) == Some(&b'-')
                && y.as_bytes().get(4) == Some(&b'-')
            {
                Some(x[..10].cmp(y))
            } else if y.len() >= 10
                && x.len() == 10
                && x.as_bytes().get(4) == Some(&b'-')
                && y.as_bytes().get(4) == Some(&b'-')
            {
                Some(x.cmp(&y[..10]))
            } else {
                Some(x.cmp(y))
            }
        }
        (BVal::Bool(x), BVal::Bool(y)) => Some(x.cmp(&y)),
        _ => None,
    }
}

#[inline]
fn ord_matches(ord: Ordering, op: Operator) -> bool {
    match op {
        Operator::Eq => ord == Ordering::Equal,
        Operator::Ne => ord != Ordering::Equal,
        Operator::Lt => ord == Ordering::Less,
        Operator::Le => ord != Ordering::Greater,
        Operator::Gt => ord == Ordering::Greater,
        Operator::Ge => ord != Ordering::Less,
        _ => false,
    }
}

#[inline]
fn eval_float_cmp(lhs: f64, op: Operator, rhs: f64) -> bool {
    if op == Operator::Eq {
        (lhs - rhs).abs() <= 1e-9 * lhs.abs().max(rhs.abs()).max(1.0)
    } else if op == Operator::Ne {
        (lhs - rhs).abs() > 1e-9 * lhs.abs().max(rhs.abs()).max(1.0)
    } else {
        lhs.partial_cmp(&rhs).is_some_and(|o| ord_matches(o, op))
    }
}

fn substr_ascii<'a>(s: &'a str, start_1based: usize, len: Option<usize>) -> &'a str {
    let idx0 = start_1based.saturating_sub(1);
    if s.is_ascii() {
        if idx0 >= s.len() {
            return "";
        }
        let end = match len {
            Some(l) => idx0.saturating_add(l).min(s.len()),
            None => s.len(),
        };
        &s[idx0..end]
    } else {
        let mut chars = s.char_indices();
        let Some((b_start, _)) = chars.nth(idx0) else {
            return "";
        };
        match len {
            None => &s[b_start..],
            Some(0) => "",
            Some(l) => {
                let b_end = chars.nth(l - 1).map_or(s.len(), |(bi, _)| bi);
                &s[b_start..b_end]
            }
        }
    }
}

fn compile_seg_expr(expr: &Expression, cols: &[(&str, &Col)]) -> Option<SegExpr> {
    let find_col = |name: &str| -> Option<usize> {
        let bare = name.rsplit_once('.').map_or(name, |(_, b)| b);
        cols.iter()
            .position(|(cn, _)| cn.eq_ignore_ascii_case(bare))
    };

    // Try compiling single-column `Col::Text` predicates into an `O(1)` dictionary bitset!
    let try_dict_pred = |col_idx: usize, test_fn: &dyn Fn(&str) -> bool| -> Option<SegExpr> {
        if let Col::Text(_, dict) = cols[col_idx].1 {
            let mask: Vec<bool> = dict.iter().map(|s| test_fn(s.as_ref())).collect();
            return Some(SegExpr::DictBitset { col_idx, mask });
        }
        None
    };

    match expr {
        Expression::Literal(v) => match v {
            AstValue::Null => Some(SegExpr::LitNull),
            AstValue::Bool(b) => Some(SegExpr::LitBool(*b)),
            AstValue::Int(i) => Some(SegExpr::LitInt(*i)),
            AstValue::Float(f) => Some(SegExpr::LitFloat(*f)),
            AstValue::String(s) => Some(SegExpr::LitStr(s.clone())),
            AstValue::Timestamp(ts) => Some(SegExpr::LitStr(epoch_secs_to_iso10(*ts))),
            _ => None,
        },
        Expression::Column(c) => Some(SegExpr::Col(find_col(c)?)),
        Expression::QualifiedColumn { column, .. } => Some(SegExpr::Col(find_col(column)?)),
        Expression::Cast {
            expr: inner,
            target_type,
        } => {
            let tt = target_type.to_ascii_uppercase();
            let sub = compile_seg_expr(inner, cols)?;
            if tt == "DATE" {
                match sub {
                    SegExpr::LitStr(s) => {
                        let iso = if s.len() >= 10 {
                            s[..10].to_string()
                        } else {
                            s
                        };
                        Some(SegExpr::LitStr(iso))
                    }
                    _ => None,
                }
            } else if tt.contains("INT") {
                match sub {
                    SegExpr::LitInt(i) => Some(SegExpr::LitInt(i)),
                    SegExpr::LitFloat(f) => Some(SegExpr::LitInt(f as i64)),
                    SegExpr::LitStr(s) => s.trim().parse::<i64>().ok().map(SegExpr::LitInt),
                    _ => None,
                }
            } else if tt.contains("FLOAT")
                || tt.contains("DOUBLE")
                || tt.contains("DECIMAL")
                || tt.contains("NUMERIC")
                || tt.contains("REAL")
            {
                match sub {
                    SegExpr::LitFloat(f) => Some(SegExpr::LitFloat(f)),
                    SegExpr::LitInt(i) => Some(SegExpr::LitFloat(i as f64)),
                    SegExpr::LitStr(s) => s.trim().parse::<f64>().ok().map(SegExpr::LitFloat),
                    _ => None,
                }
            } else {
                None
            }
        }
        Expression::Unary { op, expr: inner } => {
            let sub = compile_seg_expr(inner, cols)?;
            match op {
                UnaryOperator::Neg => match sub {
                    SegExpr::LitInt(i) => Some(SegExpr::LitInt(-i)),
                    SegExpr::LitFloat(f) => Some(SegExpr::LitFloat(-f)),
                    other => Some(SegExpr::Neg(Box::new(other))),
                },
                UnaryOperator::Not => Some(SegExpr::Not(Box::new(sub))),
                _ => None,
            }
        }
        Expression::IsNull {
            expr: inner,
            negated,
        } => Some(SegExpr::IsNull(
            Box::new(compile_seg_expr(inner, cols)?),
            *negated,
        )),
        Expression::Like {
            expr: inner,
            pattern,
            negated,
            case_insensitive,
        } => {
            let pat = LikePattern::new(pattern, *case_insensitive);
            let sub = compile_seg_expr(inner, cols)?;
            if let SegExpr::Col(ci) = &sub
                && let Some(db) = try_dict_pred(*ci, &|s| pat.matches(s) != *negated)
            {
                return Some(db);
            }
            Some(SegExpr::Like {
                expr: Box::new(sub),
                pat,
                negated: *negated,
            })
        }
        Expression::Between {
            expr: inner,
            low,
            high,
            negated,
        } => {
            let sub = compile_seg_expr(inner, cols)?;
            let lo = compile_seg_expr(low, cols)?;
            let hi = compile_seg_expr(high, cols)?;
            if let (SegExpr::Col(ci), SegExpr::LitStr(ls), SegExpr::LitStr(hs)) = (&sub, &lo, &hi)
                && let Some(db) = try_dict_pred(*ci, &|s| {
                    let sc = if s.len() >= 10 && ls.len() == 10 {
                        &s[..10]
                    } else {
                        s
                    };
                    let inside = sc >= ls.as_str() && sc <= hs.as_str();
                    inside != *negated
                })
            {
                return Some(db);
            }
            Some(SegExpr::Between {
                expr: Box::new(sub),
                low: Box::new(lo),
                high: Box::new(hi),
                negated: *negated,
            })
        }
        Expression::In {
            expr: inner,
            list,
            negated,
        } => {
            let sub = compile_seg_expr(inner, cols)?;
            let mut clist = Vec::with_capacity(list.len());
            let mut str_set: Option<HashSet<String>> = Some(HashSet::new());
            for item in list {
                let ci = compile_seg_expr(item, cols)?;
                if let SegExpr::LitStr(s) = &ci {
                    if let Some(ss) = &mut str_set {
                        ss.insert(s.clone());
                    }
                } else {
                    str_set = None;
                }
                clist.push(ci);
            }
            if let Some(ss) = str_set {
                match &sub {
                    SegExpr::Col(ci) => {
                        if let Some(db) = try_dict_pred(*ci, &|s| ss.contains(s) != *negated) {
                            return Some(db);
                        }
                    }
                    SegExpr::Substr {
                        expr: inner_col,
                        start_1based,
                        len,
                    } => {
                        if let SegExpr::Col(ci) = inner_col.as_ref() {
                            let st = *start_1based;
                            let ln = *len;
                            if let Some(db) = try_dict_pred(*ci, &|s| {
                                ss.contains(substr_ascii(s, st, ln)) != *negated
                            }) {
                                return Some(db);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Some(SegExpr::InList {
                expr: Box::new(sub),
                list: clist,
                negated: *negated,
            })
        }
        Expression::Function { name, args } => {
            let up = name.to_ascii_uppercase();
            match up.as_str() {
                "SUBSTRING" | "SUBSTR" if args.len() == 2 || args.len() == 3 => {
                    let inner = compile_seg_expr(&args[0], cols)?;
                    let SegExpr::LitInt(st) = compile_seg_expr(&args[1], cols)? else {
                        return None;
                    };
                    let len = if args.len() == 3 {
                        let SegExpr::LitInt(l) = compile_seg_expr(&args[2], cols)? else {
                            return None;
                        };
                        Some(l.max(0) as usize)
                    } else {
                        None
                    };
                    Some(SegExpr::Substr {
                        expr: Box::new(inner),
                        start_1based: st.max(1) as usize,
                        len,
                    })
                }
                "EXTRACT" | "DATE_PART" if args.len() == 2 => {
                    let SegExpr::LitStr(part) = compile_seg_expr(&args[0], cols)? else {
                        return None;
                    };
                    let inner = compile_seg_expr(&args[1], cols)?;
                    match part.to_ascii_uppercase().as_str() {
                        "YEAR" => Some(SegExpr::ExtractYear(Box::new(inner))),
                        "MONTH" => Some(SegExpr::ExtractMonth(Box::new(inner))),
                        "DAY" => Some(SegExpr::ExtractDay(Box::new(inner))),
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        Expression::Binary { left, op, right } => {
            let l = compile_seg_expr(left, cols)?;
            let r = compile_seg_expr(right, cols)?;
            match op {
                Operator::And => Some(SegExpr::And(Box::new(l), Box::new(r))),
                Operator::Or => Some(SegExpr::Or(Box::new(l), Box::new(r))),
                Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
                    // Constant folding
                    match (&l, op, &r) {
                        (SegExpr::LitInt(a), Operator::Add, SegExpr::LitInt(b)) => {
                            Some(SegExpr::LitInt(a.saturating_add(*b)))
                        }
                        (SegExpr::LitInt(a), Operator::Sub, SegExpr::LitInt(b)) => {
                            Some(SegExpr::LitInt(a.saturating_sub(*b)))
                        }
                        (SegExpr::LitInt(a), Operator::Mul, SegExpr::LitInt(b)) => {
                            Some(SegExpr::LitInt(a.saturating_mul(*b)))
                        }
                        (SegExpr::LitFloat(a), Operator::Add, SegExpr::LitFloat(b)) => {
                            Some(SegExpr::LitFloat(a + b))
                        }
                        (SegExpr::LitFloat(a), Operator::Sub, SegExpr::LitFloat(b)) => {
                            Some(SegExpr::LitFloat(a - b))
                        }
                        (SegExpr::LitFloat(a), Operator::Mul, SegExpr::LitFloat(b)) => {
                            Some(SegExpr::LitFloat(a * b))
                        }
                        _ => Some(SegExpr::Arith(Box::new(l), *op, Box::new(r))),
                    }
                }
                Operator::Eq
                | Operator::Ne
                | Operator::Lt
                | Operator::Le
                | Operator::Gt
                | Operator::Ge => {
                    if let (SegExpr::Col(ci), SegExpr::LitStr(rs)) = (&l, &r)
                        && let Some(db) = try_dict_pred(*ci, &|s| {
                            cmp_bvals(BVal::Str(s), BVal::Str(rs.as_str()))
                                .is_some_and(|o| ord_matches(o, *op))
                        })
                    {
                        return Some(db);
                    }
                    if let (SegExpr::LitStr(ls), SegExpr::Col(ci)) = (&l, &r)
                        && let Some(db) = try_dict_pred(*ci, &|s| {
                            cmp_bvals(BVal::Str(ls.as_str()), BVal::Str(s))
                                .is_some_and(|o| ord_matches(o, *op))
                        })
                    {
                        return Some(db);
                    }
                    Some(SegExpr::Cmp(Box::new(l), *op, Box::new(r)))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

impl SegExpr {
    #[inline]
    fn eval<'a>(&'a self, cols: &[(&str, &'a Col)], row: usize) -> BVal<'a> {
        match self {
            SegExpr::LitNull => BVal::Null,
            SegExpr::LitBool(b) => BVal::Bool(*b),
            SegExpr::LitInt(i) => BVal::Int(*i),
            SegExpr::LitFloat(f) => BVal::Float(*f),
            SegExpr::LitStr(s) => BVal::Str(s.as_str()),
            SegExpr::Col(ci) => read_col_bval(cols[*ci].1, row),
            SegExpr::DictBitset { col_idx, mask } => {
                if let Col::Text(codes, _) = cols[*col_idx].1 {
                    let c = codes[row];
                    if c == NULL_TEXT_CODE {
                        BVal::Bool(false)
                    } else {
                        BVal::Bool(*mask.get(c as usize).unwrap_or(&false))
                    }
                } else {
                    BVal::Bool(false)
                }
            }
            SegExpr::Neg(x) => match x.eval(cols, row) {
                BVal::Int(i) => BVal::Int(i.wrapping_neg()),
                BVal::Float(f) => BVal::Float(-f),
                _ => BVal::Null,
            },
            SegExpr::Not(x) => match x.eval(cols, row) {
                BVal::Bool(b) => BVal::Bool(!b),
                _ => BVal::Null,
            },
            SegExpr::IsNull(x, neg) => {
                let is_n = matches!(x.eval(cols, row), BVal::Null);
                BVal::Bool(if *neg { !is_n } else { is_n })
            }
            SegExpr::And(a, b) => {
                let av = a.eval(cols, row);
                if matches!(av, BVal::Bool(false)) {
                    return BVal::Bool(false);
                }
                let bv = b.eval(cols, row);
                match (av, bv) {
                    (BVal::Bool(true), BVal::Bool(true)) => BVal::Bool(true),
                    (_, BVal::Bool(false)) => BVal::Bool(false),
                    _ => BVal::Null,
                }
            }
            SegExpr::Or(a, b) => {
                let av = a.eval(cols, row);
                if matches!(av, BVal::Bool(true)) {
                    return BVal::Bool(true);
                }
                let bv = b.eval(cols, row);
                match (av, bv) {
                    (_, BVal::Bool(true)) => BVal::Bool(true),
                    (BVal::Bool(false), BVal::Bool(false)) => BVal::Bool(false),
                    _ => BVal::Null,
                }
            }
            SegExpr::Arith(l, op, r) => {
                let lv = l.eval(cols, row);
                let rv = r.eval(cols, row);
                match (lv, rv) {
                    (BVal::Null, _) | (_, BVal::Null) => BVal::Null,
                    (BVal::Int(a), BVal::Int(b)) => match op {
                        Operator::Add => BVal::Int(a.wrapping_add(b)),
                        Operator::Sub => BVal::Int(a.wrapping_sub(b)),
                        Operator::Mul => BVal::Int(a.wrapping_mul(b)),
                        Operator::Div => {
                            if b == 0 {
                                BVal::Null
                            } else {
                                BVal::Float((a as f64) / (b as f64))
                            }
                        }
                        _ => BVal::Null,
                    },
                    (a, b) => {
                        let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) else {
                            return BVal::Null;
                        };
                        match op {
                            Operator::Add => BVal::Float(fa + fb),
                            Operator::Sub => BVal::Float(fa - fb),
                            Operator::Mul => BVal::Float(fa * fb),
                            Operator::Div => {
                                if fb == 0.0 {
                                    BVal::Null
                                } else {
                                    BVal::Float(fa / fb)
                                }
                            }
                            _ => BVal::Null,
                        }
                    }
                }
            }
            SegExpr::Cmp(l, op, r) => {
                let lv = l.eval(cols, row);
                let rv = r.eval(cols, row);
                match (lv, rv) {
                    (BVal::Float(fa), BVal::Float(fb)) => BVal::Bool(eval_float_cmp(fa, *op, fb)),
                    (BVal::Int(ia), BVal::Float(fb)) => {
                        BVal::Bool(eval_float_cmp(ia as f64, *op, fb))
                    }
                    (BVal::Float(fa), BVal::Int(ib)) => {
                        BVal::Bool(eval_float_cmp(fa, *op, ib as f64))
                    }
                    _ => match cmp_bvals(lv, rv) {
                        Some(ord) => BVal::Bool(ord_matches(ord, *op)),
                        None => BVal::Null,
                    },
                }
            }
            SegExpr::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let v = expr.eval(cols, row);
                let lv = low.eval(cols, row);
                let hv = high.eval(cols, row);
                let (Some(c1), Some(c2)) = (cmp_bvals(v, lv), cmp_bvals(v, hv)) else {
                    return BVal::Null;
                };
                let inside = c1 != Ordering::Less && c2 != Ordering::Greater;
                BVal::Bool(if *negated { !inside } else { inside })
            }
            SegExpr::InList {
                expr,
                list,
                negated,
            } => {
                let v = expr.eval(cols, row);
                if matches!(v, BVal::Null) {
                    return BVal::Null;
                }
                let mut found = false;
                for item in list {
                    if cmp_bvals(v, item.eval(cols, row)) == Some(Ordering::Equal) {
                        found = true;
                        break;
                    }
                }
                BVal::Bool(if *negated { !found } else { found })
            }
            SegExpr::Like { expr, pat, negated } => match expr.eval(cols, row) {
                BVal::Str(s) => BVal::Bool(pat.matches(s) != *negated),
                _ => BVal::Null,
            },
            SegExpr::Substr {
                expr,
                start_1based,
                len,
            } => match expr.eval(cols, row) {
                BVal::Str(s) => BVal::Str(substr_ascii(s, *start_1based, *len)),
                _ => BVal::Null,
            },
            SegExpr::ExtractYear(x) => match x.eval(cols, row) {
                BVal::Str(s) => {
                    parse_iso_ymd_fast(s).map_or(BVal::Null, |(y, _, _)| BVal::Int(y as i64))
                }
                BVal::Int(ts) => BVal::Int(epoch_secs_to_ymd(ts).0 as i64),
                _ => BVal::Null,
            },
            SegExpr::ExtractMonth(x) => match x.eval(cols, row) {
                BVal::Str(s) => {
                    parse_iso_ymd_fast(s).map_or(BVal::Null, |(_, m, _)| BVal::Int(m as i64))
                }
                BVal::Int(ts) => BVal::Int(epoch_secs_to_ymd(ts).1 as i64),
                _ => BVal::Null,
            },
            SegExpr::ExtractDay(x) => match x.eval(cols, row) {
                BVal::Str(s) => {
                    parse_iso_ymd_fast(s).map_or(BVal::Null, |(_, _, d)| BVal::Int(d as i64))
                }
                BVal::Int(ts) => BVal::Int(epoch_secs_to_ymd(ts).2 as i64),
                _ => BVal::Null,
            },
        }
    }
}

/// Evaluate a list of `LocalFilter`s over a single loaded `TableColumns` segment, returning
/// matching row indices `0..tc.len` (evaluating cheap `KeyIn` / `DictBitset` filters first!).
fn filter_segment_rows(tc: &TableColumns, filters: &[LocalFilter]) -> Option<Vec<u32>> {
    let n = tc.len;
    if filters.is_empty() {
        return Some((0..n as u32).collect());
    }
    let col_slice: Vec<(&str, &Col)> = tc.cols.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let find_ci = |cn: &str| -> Option<usize> {
        col_slice
            .iter()
            .position(|(k, _)| k.eq_ignore_ascii_case(cn))
    };

    enum CompiledLF<'a> {
        KeyIn {
            ci: usize,
            set: &'a IntKeySet,
            negated: bool,
        },
        NeqExists {
            k_ci: usize,
            n_ci: usize,
            shards: &'a [FxMap<i64, NeqSummary>],
            negated: bool,
        },
        Corr1 {
            k_ci: usize,
            lhs: SegExpr,
            op: Operator,
            map: &'a FxMap<i64, f64>,
        },
        Corr2 {
            k1_ci: usize,
            k2_ci: usize,
            lhs: SegExpr,
            op: Operator,
            map: &'a FxMap<(i64, i64), f64>,
        },
        Expr(SegExpr),
    }

    let mut compiled = Vec::with_capacity(filters.len());
    // Put `KeyIn` semi-joins FIRST so 99% of rows skip string/arithmetic filters!
    for pass in 0..2 {
        for f in filters {
            let is_key_in = matches!(f, LocalFilter::KeyIn { .. });
            if (pass == 0) != is_key_in {
                continue;
            }
            match f {
                LocalFilter::KeyIn { col, set, negated } => {
                    compiled.push(CompiledLF::KeyIn {
                        ci: find_ci(col)?,
                        set,
                        negated: *negated,
                    });
                }
                LocalFilter::NeqExists {
                    key_col,
                    neq_col,
                    shards,
                    negated,
                } => {
                    compiled.push(CompiledLF::NeqExists {
                        k_ci: find_ci(key_col)?,
                        n_ci: find_ci(neq_col)?,
                        shards: shards.as_slice(),
                        negated: *negated,
                    });
                }
                LocalFilter::CorrScalar1 {
                    key_col,
                    lhs_expr,
                    op,
                    map,
                } => {
                    compiled.push(CompiledLF::Corr1 {
                        k_ci: find_ci(key_col)?,
                        lhs: compile_seg_expr(lhs_expr, &col_slice)?,
                        op: *op,
                        map,
                    });
                }
                LocalFilter::CorrScalar2 {
                    key_col1,
                    key_col2,
                    lhs_expr,
                    op,
                    map,
                } => {
                    compiled.push(CompiledLF::Corr2 {
                        k1_ci: find_ci(key_col1)?,
                        k2_ci: find_ci(key_col2)?,
                        lhs: compile_seg_expr(lhs_expr, &col_slice)?,
                        op: *op,
                        map,
                    });
                }
                LocalFilter::Expr(e) => {
                    compiled.push(CompiledLF::Expr(compile_seg_expr(e, &col_slice)?));
                }
            }
        }
    }

    let mut out = Vec::new();
    'row_loop: for i in 0..n {
        for cf in &compiled {
            let ok = match cf {
                CompiledLF::KeyIn { ci, set, negated } => {
                    match read_col_bval(col_slice[*ci].1, i) {
                        BVal::Int(k) => set.contains(k) != *negated,
                        _ => false,
                    }
                }
                CompiledLF::NeqExists {
                    k_ci,
                    n_ci,
                    shards,
                    negated,
                } => match (
                    read_col_bval(col_slice[*k_ci].1, i),
                    read_col_bval(col_slice[*n_ci].1, i),
                ) {
                    (BVal::Int(k), BVal::Int(nv)) => {
                        let sh = (((k as u64).wrapping_mul(0x517cc1b727220a95)) >> 58) as usize;
                        let exists = match shards[sh].get(&k) {
                            None => false,
                            Some(NeqSummary::Many) => true,
                            Some(NeqSummary::One(only)) => *only != nv,
                        };
                        exists != *negated
                    }
                    _ => *negated,
                },
                CompiledLF::Corr1 { k_ci, lhs, op, map } => {
                    let Some(lv) = lhs.eval(&col_slice, i).as_f64() else {
                        continue 'row_loop;
                    };
                    let BVal::Int(k) = read_col_bval(col_slice[*k_ci].1, i) else {
                        continue 'row_loop;
                    };
                    let Some(&rv) = map.get(&k) else {
                        continue 'row_loop;
                    };
                    eval_float_cmp(lv, *op, rv)
                }
                CompiledLF::Corr2 {
                    k1_ci,
                    k2_ci,
                    lhs,
                    op,
                    map,
                } => {
                    let Some(lv) = lhs.eval(&col_slice, i).as_f64() else {
                        continue 'row_loop;
                    };
                    let (BVal::Int(k1), BVal::Int(k2)) = (
                        read_col_bval(col_slice[*k1_ci].1, i),
                        read_col_bval(col_slice[*k2_ci].1, i),
                    ) else {
                        continue 'row_loop;
                    };
                    let Some(&rv) = map.get(&(k1, k2)) else {
                        continue 'row_loop;
                    };
                    eval_float_cmp(lv, *op, rv)
                }
                CompiledLF::Expr(se) => matches!(se.eval(&col_slice, i), BVal::Bool(true)),
            };
            if !ok {
                continue 'row_loop;
            }
        }
        out.push(i as u32);
    }
    Some(out)
}

/// Compact columnar storage for a materialized/folded dimension branch (`0` `serde_json::Value`!).
struct CompactCol {
    /// `(rel_idx, canonical_col_name)`
    id: (usize, String),
    data: Col,
}

enum BranchLookup {
    /// Dense array `slots[key - min_key] = row_idx` (`u32::MAX` = absent) for unique PKs!
    DenseUnique1 {
        min_key: i64,
        max_key: i64,
        slots: Vec<u32>,
    },
    /// Hash map for unique `1:1` integer keys (`row_idx`).
    HashUnique1(FxMap<i64, u32>),
    /// Hash map for `1:N` integer keys (`(start_idx, len)` into `multi_rows`).
    HashMulti1 {
        map: FxMap<i64, (u32, u32)>,
        row_ids: Vec<u32>,
    },
    /// Composite 2-key unique hash map (`(k1, k2) -> row_idx`, e.g. `partsupp`).
    HashUnique2(FxMap<(i64, i64), u32>),
    /// Composite 2-key `1:N` hash map.
    HashMulti2 {
        map: FxMap<(i64, i64), (u32, u32)>,
        row_ids: Vec<u32>,
    },
}

struct BuiltBranch {
    /// Probe-table join key column name(s) (`1` or `2` integer columns on `probe_rel`).
    probe_key_cols: Vec<String>,
    num_rows: usize,
    cols: Vec<CompactCol>,
    lookup: BranchLookup,
}

impl BuiltBranch {
    fn from_rows_1key(probe_key_col: String, keys: &[i64], cols: Vec<CompactCol>) -> Self {
        let n = keys.len();
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        for &k in keys {
            lo = lo.min(k);
            hi = hi.max(k);
        }
        let span = if n > 0 {
            (hi as i128) - (lo as i128)
        } else {
            -1
        };
        if n > 0
            && (0..=32_000_000).contains(&span)
            && (span as usize) <= n.saturating_mul(12).max(4096)
        {
            let mut slots = vec![u32::MAX; (span as usize) + 1];
            let mut all_unique = true;
            for (ri, &k) in keys.iter().enumerate() {
                let slot = &mut slots[(k - lo) as usize];
                if *slot != u32::MAX {
                    all_unique = false;
                    break;
                }
                *slot = ri as u32;
            }
            if all_unique {
                return Self {
                    probe_key_cols: vec![probe_key_col],
                    num_rows: n,
                    cols,
                    lookup: BranchLookup::DenseUnique1 {
                        min_key: lo,
                        max_key: hi,
                        slots,
                    },
                };
            }
        }

        let mut umap: FxMap<i64, u32> = FxMap::default();
        umap.reserve(n);
        let mut all_unique = true;
        for (ri, &k) in keys.iter().enumerate() {
            if umap.insert(k, ri as u32).is_some() {
                all_unique = false;
                break;
            }
        }
        if all_unique {
            return Self {
                probe_key_cols: vec![probe_key_col],
                num_rows: n,
                cols,
                lookup: BranchLookup::HashUnique1(umap),
            };
        }

        let mut groups: FxMap<i64, Vec<u32>> = FxMap::default();
        for (ri, &k) in keys.iter().enumerate() {
            groups.entry(k).or_default().push(ri as u32);
        }
        let mut map: FxMap<i64, (u32, u32)> = FxMap::default();
        map.reserve(groups.len());
        let mut row_ids = Vec::with_capacity(n);
        for (k, list) in groups {
            let start = row_ids.len() as u32;
            let len = list.len() as u32;
            row_ids.extend_from_slice(&list);
            map.insert(k, (start, len));
        }
        Self {
            probe_key_cols: vec![probe_key_col],
            num_rows: n,
            cols,
            lookup: BranchLookup::HashMulti1 { map, row_ids },
        }
    }

    fn from_rows_2key(
        probe_k1: String,
        probe_k2: String,
        keys: &[(i64, i64)],
        cols: Vec<CompactCol>,
    ) -> Self {
        let n = keys.len();
        let mut umap: FxMap<(i64, i64), u32> = FxMap::default();
        umap.reserve(n);
        let mut all_unique = true;
        for (ri, &k) in keys.iter().enumerate() {
            if umap.insert(k, ri as u32).is_some() {
                all_unique = false;
                break;
            }
        }
        if all_unique {
            return Self {
                probe_key_cols: vec![probe_k1, probe_k2],
                num_rows: n,
                cols,
                lookup: BranchLookup::HashUnique2(umap),
            };
        }
        let mut groups: FxMap<(i64, i64), Vec<u32>> = FxMap::default();
        for (ri, &k) in keys.iter().enumerate() {
            groups.entry(k).or_default().push(ri as u32);
        }
        let mut map: FxMap<(i64, i64), (u32, u32)> = FxMap::default();
        map.reserve(groups.len());
        let mut row_ids = Vec::with_capacity(n);
        for (k, list) in groups {
            let start = row_ids.len() as u32;
            let len = list.len() as u32;
            row_ids.extend_from_slice(&list);
            map.insert(k, (start, len));
        }
        Self {
            probe_key_cols: vec![probe_k1, probe_k2],
            num_rows: n,
            cols,
            lookup: BranchLookup::HashMulti2 { map, row_ids },
        }
    }

    #[inline]
    fn probe_single_1(&self, k: i64) -> Option<u32> {
        match &self.lookup {
            BranchLookup::DenseUnique1 {
                min_key,
                max_key,
                slots,
            } => {
                if k < *min_key || k > *max_key {
                    None
                } else {
                    let s = slots[(k - *min_key) as usize];
                    (s != u32::MAX).then_some(s)
                }
            }
            BranchLookup::HashUnique1(m) => m.get(&k).copied(),
            _ => None,
        }
    }

    fn key_set_1(&self) -> Option<IntKeySet> {
        match &self.lookup {
            BranchLookup::DenseUnique1 {
                min_key,
                max_key,
                slots,
            } => {
                let span = (*max_key - *min_key) as usize;
                let mut words = vec![0u64; (span + 64) / 64];
                for (i, &s) in slots.iter().enumerate() {
                    if s != u32::MAX {
                        words[i >> 6] |= 1u64 << (i & 63);
                    }
                }
                Some(IntKeySet::Dense(Arc::new(DenseKeyBits {
                    min_key: *min_key,
                    max_key: *max_key,
                    words,
                })))
            }
            BranchLookup::HashUnique1(m) => Some(IntKeySet::from_vec(m.keys().copied().collect())),
            BranchLookup::HashMulti1 { map, .. } => {
                Some(IntKeySet::from_vec(map.keys().copied().collect()))
            }
            _ => None,
        }
    }
}

/// Column builder accumulating selected rows from segments + joined child branches.
enum ColBuilder {
    Int(Vec<i64>, Vec<bool>, bool),
    Float(Vec<f64>, Vec<bool>, bool),
    Text {
        codes: Vec<u32>,
        dict: Vec<Arc<str>>,
        map: HashMap<String, u32>,
    },
}

impl ColBuilder {
    fn new_for_sample(col: &Col) -> Option<Self> {
        match col {
            Col::Int(..) => Some(ColBuilder::Int(Vec::new(), Vec::new(), false)),
            Col::Float(..) => Some(ColBuilder::Float(Vec::new(), Vec::new(), false)),
            Col::Text(..) => Some(ColBuilder::Text {
                codes: Vec::new(),
                dict: Vec::new(),
                map: HashMap::new(),
            }),
            Col::Other => None,
        }
    }

    fn push_from_col(&mut self, src: &Col, row: usize) {
        let bv = read_col_bval(src, row);
        match self {
            ColBuilder::Int(vals, nulls, any_null) => match bv {
                BVal::Int(v) => {
                    vals.push(v);
                    nulls.push(false);
                }
                _ => {
                    vals.push(0);
                    nulls.push(true);
                    *any_null = true;
                }
            },
            ColBuilder::Float(vals, nulls, any_null) => match bv.as_f64() {
                Some(f) => {
                    vals.push(f);
                    nulls.push(false);
                }
                None => {
                    vals.push(0.0);
                    nulls.push(true);
                    *any_null = true;
                }
            },
            ColBuilder::Text { codes, dict, map } => match bv {
                BVal::Str(s) => {
                    if let Some(&c) = map.get(s) {
                        codes.push(c);
                    } else {
                        let c = dict.len() as u32;
                        dict.push(Arc::from(s));
                        map.insert(s.to_string(), c);
                        codes.push(c);
                    }
                }
                _ => codes.push(NULL_TEXT_CODE),
            },
        }
    }

    fn finish(self) -> Col {
        match self {
            ColBuilder::Int(vals, mut nulls, any_null) => {
                if !any_null {
                    nulls.clear();
                }
                Col::Int(Ints::I64(vals), nulls)
            }
            ColBuilder::Float(vals, mut nulls, any_null) => {
                if !any_null {
                    nulls.clear();
                }
                Col::Float(vals, nulls)
            }
            ColBuilder::Text { codes, dict, .. } => Col::Text(codes, dict),
        }
    }
}

/// Core multi-table / subquery execution engine.
fn exec_core_query(
    raw_query: &SqlQuery,
    amorphic: &AmorphicTableStorage,
    cte_rels: &HashMap<String, ColRel>,
) -> Option<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
    exec_core_query_seeded(raw_query, amorphic, cte_rels, None)
}

fn exec_core_query_seeded(
    raw_query: &SqlQuery,
    amorphic: &AmorphicTableStorage,
    cte_rels: &HashMap<String, ColRel>,
    seed_filter: Option<(usize, LocalFilter)>,
) -> Option<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
    // 1. Flatten non-grouped `FROM (SELECT ...)` derived tables (`Q07, Q08, Q09, Q22`)
    let mut query = flatten_derived_subquery(raw_query).unwrap_or_else(|| raw_query.clone());

    // 2. Materialize grouped `FROM (SELECT ... GROUP BY ...)` derived tables (`Q13`) into local `cte_rels`
    let mut local_ctes;
    let active_ctes = if let Some(ref from) = query.from
        && let FromSource::Subquery(ref sub) = from.source
        && query.joins.is_empty()
    {
        let sub_alias = from.effective_name().to_string();
        let sub_tc = exec_query_to_table_columns(sub, None, amorphic, cte_rels)?;
        local_ctes = cte_rels.clone();
        local_ctes.insert(
            sub_alias.to_ascii_lowercase(),
            ColRel::Mem(Arc::new(sub_tc)),
        );
        query.from = Some(SqlFrom {
            source: FromSource::Table(sub_alias),
            alias: None,
            lateral: false,
        });
        &local_ctes
    } else {
        cte_rels
    };

    // 3. Fast `LEFT OUTER JOIN` grouped count/sum (`Q13`: `customer LEFT OUTER JOIN orders ON c_custkey = o_custkey AND ... GROUP BY c_custkey`)
    if query.joins.len() == 1 && query.joins[0].join_type == JoinType::Left {
        return exec_left_outer_grouped(&query, amorphic, active_ctes);
    }
    if query
        .joins
        .iter()
        .any(|j| !matches!(j.join_type, JoinType::Inner | JoinType::Cross))
    {
        return None;
    }

    // 4. Load all relations in `FROM` + `JOIN`
    let from = query.from.as_ref()?;
    let base_tbl = from.table_name()?;
    let mut rels: Vec<RelEntry> = Vec::with_capacity(1 + query.joins.len());
    rels.push(RelEntry {
        alias: from.alias.clone().unwrap_or_else(|| base_tbl.to_string()),
        table: base_tbl.to_string(),
        rel: load_rel(base_tbl, amorphic, active_ctes)?,
    });
    for j in &query.joins {
        if !j.using_columns.is_empty() {
            return None;
        }
        rels.push(RelEntry {
            alias: j.alias.clone().unwrap_or_else(|| j.table.clone()),
            table: j.table.clone(),
            rel: load_rel(&j.table, amorphic, active_ctes)?,
        });
    }

    // 5. Merge `WHERE` + `INNER JOIN ON` conditions, factor common `OR` conjuncts
    let mut raw_conjs: Vec<Expression> = Vec::new();
    if let Some(ref w) = query.where_clause {
        let factored = crate::query::SimpleQueryExecutor::factor_common_or_conjuncts_pub(w);
        split_and_owned(&factored, &mut raw_conjs);
    }
    for j in &query.joins {
        if let Some(ref cond) = j.condition {
            let factored = crate::query::SimpleQueryExecutor::factor_common_or_conjuncts_pub(cond);
            split_and_owned(&factored, &mut raw_conjs);
        }
    }

    // 6. Resolve uncorrelated scalar subqueries in `HAVING` (`Q11`)
    let resolved_having = match &query.having {
        Some(h) => Some(resolve_uncorrelated_subqueries_in_expr(
            h,
            &rels,
            amorphic,
            active_ctes,
        )?),
        None => None,
    };

    // 7. Partition conjuncts: pure single-table filters FIRST (for domain pushdown into correlated subqueries!),
    //    equi-join edges `(ri, col_i) == (rj, col_j)`, subqueries, and cross-table residuals.
    let mut local_filters: Vec<Vec<LocalFilter>> = vec![Vec::new(); rels.len()];
    if let Some((sri, slf)) = seed_filter
        && sri < local_filters.len()
    {
        local_filters[sri].push(slf);
    }
    let mut equi_edges: Vec<((usize, String), (usize, String))> = Vec::new();
    let mut deferred_subq_conjs: Vec<Expression> = Vec::new();
    let mut residual_conjs: Vec<Expression> = Vec::new();

    for conj in raw_conjs {
        if expr_has_subquery(&conj) {
            deferred_subq_conjs.push(conj);
            continue;
        }
        // Check pure `r1.c1 = r2.c2` integer equi-join
        if let Expression::Binary {
            ref left,
            op: Operator::Eq,
            ref right,
        } = conj
            && let (Some((ra, ca)), Some((rb, cb))) =
                (resolve_rel_col(&rels, left), resolve_rel_col(&rels, right))
            && ra != rb
            && rels[ra].rel.col_is_int(&ca)
            && rels[rb].rel.col_is_int(&cb)
        {
            equi_edges.push(((ra, ca), (rb, cb)));
            continue;
        }

        let mut touched = HashSet::new();
        if expr_referenced_rels(&conj, &rels, &mut touched) {
            if touched.len() == 1 {
                let ri = *touched.iter().next()?;
                local_filters[ri].push(LocalFilter::Expr(conj));
                continue;
            }
            if matches!(
                conj,
                Expression::Binary {
                    op: Operator::Or,
                    ..
                }
            ) {
                for (ri, bound) in extract_or_bounding_filters(&conj, &rels) {
                    local_filters[ri].push(LocalFilter::Expr(bound));
                }
            }
        }
        residual_conjs.push(conj);
    }

    // 8a. Propagate selective dimension key sets BEFORE decorrelating subqueries (`Q02`, `Q21`!)
    if !deferred_subq_conjs.is_empty() {
        propagate_initial_key_sets(&rels, &equi_edges, &mut local_filters)?;
    }

    // 8b. Decorrelate `IN`, `EXISTS` / `NOT EXISTS`, correlated scalar `< / > / =`, and uncorrelated scalar subqueries!
    for sconj in deferred_subq_conjs {
        decorrelate_conjunct(
            &sconj,
            &rels,
            &equi_edges,
            &mut local_filters,
            &mut residual_conjs,
            amorphic,
            active_ctes,
        )?;
    }

    // 9. Propagate newly produced `KeyIn` / selective dimension key sets across equivalence classes!
    propagate_initial_key_sets(&rels, &equi_edges, &mut local_filters)?;

    // 10. Execute single-relation or bushy star-join query!
    exec_bushy_plan(
        &query,
        resolved_having.as_ref(),
        &rels,
        &equi_edges,
        &mut local_filters,
        &residual_conjs,
    )
}

fn exec_left_outer_grouped(
    query: &SqlQuery,
    amorphic: &AmorphicTableStorage,
    cte_rels: &HashMap<String, ColRel>,
) -> Option<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
    // Supports `SELECT a.k, COUNT(b.col) FROM A LEFT OUTER JOIN B ON a.k = b.k [AND P(B)] [WHERE P(A)] GROUP BY a.k` (`Q13`)
    if query.having.is_some() || !query.order_by.is_empty() || query.limit.is_some() {
        return None;
    }
    let from = query.from.as_ref()?;
    let left_tbl = from.table_name()?;
    let join = &query.joins[0];
    let left_entry = RelEntry {
        alias: from.alias.clone().unwrap_or_else(|| left_tbl.to_string()),
        table: left_tbl.to_string(),
        rel: load_rel(left_tbl, amorphic, cte_rels)?,
    };
    let right_entry = RelEntry {
        alias: join.alias.clone().unwrap_or_else(|| join.table.clone()),
        table: join.table.clone(),
        rel: load_rel(&join.table, amorphic, cte_rels)?,
    };
    let rels = [left_entry, right_entry];

    let cond = join.condition.as_ref()?;
    let mut on_conjs = Vec::new();
    split_and_owned(cond, &mut on_conjs);
    let mut join_pair: Option<(String, String)> = None;
    let mut right_filters: Vec<LocalFilter> = Vec::new();

    for c in on_conjs {
        if let Expression::Binary {
            ref left,
            op: Operator::Eq,
            ref right,
        } = c
            && let (Some((ra, ca)), Some((rb, cb))) =
                (resolve_rel_col(&rels, left), resolve_rel_col(&rels, right))
            && ra != rb
        {
            let (lc, rc) = if ra == 0 { (ca, cb) } else { (cb, ca) };
            if rels[0].rel.col_is_int(&lc) && rels[1].rel.col_is_int(&rc) && join_pair.is_none() {
                join_pair = Some((lc, rc));
                continue;
            }
        }
        let mut touched = HashSet::new();
        if expr_referenced_rels(&c, &rels, &mut touched)
            && touched.len() == 1
            && touched.contains(&1)
        {
            right_filters.push(LocalFilter::Expr(c));
        } else {
            return None;
        }
    }
    let (left_key, right_key) = join_pair?;

    let mut left_filters: Vec<LocalFilter> = Vec::new();
    if let Some(ref w) = query.where_clause {
        let mut wconjs = Vec::new();
        split_and_owned(w, &mut wconjs);
        for wc in wconjs {
            let mut touched = HashSet::new();
            if expr_referenced_rels(&wc, &rels, &mut touched)
                && touched.len() == 1
                && touched.contains(&0)
            {
                left_filters.push(LocalFilter::Expr(wc));
            } else {
                return None;
            }
        }
    }

    if query.group_by.len() != 1 || query.columns.len() != 2 {
        return None;
    }
    let (g_ri, g_col) = resolve_rel_col(&rels, &query.group_by[0])?;
    if g_ri != 0 || !g_col.eq_ignore_ascii_case(&left_key) {
        return None;
    }
    let (c0_ri, c0_col) = resolve_rel_col(&rels, &query.columns[0].expr)?;
    if c0_ri != 0 || !c0_col.eq_ignore_ascii_case(&left_key) {
        return None;
    }
    let Expression::Function {
        name: fn_name,
        args: fn_args,
    } = &query.columns[1].expr
    else {
        return None;
    };
    if !fn_name.eq_ignore_ascii_case("COUNT") || fn_args.len() != 1 {
        return None;
    }
    let (cnt_ri, cnt_col) = resolve_rel_col(&rels, &fn_args[0])?;
    if cnt_ri != 1 {
        return None;
    }

    // Stream right relation (`orders`) in parallel, counting non-null `cnt_col` per `right_key`!
    let mut r_needed = HashSet::new();
    r_needed.insert(right_key.clone());
    r_needed.insert(cnt_col.clone());
    for rf in &right_filters {
        rf.collect_cols(&rels[1].rel, &mut r_needed);
    }
    let r_col_vec: Vec<String> = r_needed.into_iter().collect();
    let r_col_refs: Vec<&str> = r_col_vec.iter().map(String::as_str).collect();

    let n_rsegs = rels[1].rel.num_segments();
    let part_maps: Option<Vec<FxMap<i64, i64>>> = (0..n_rsegs)
        .into_par_iter()
        .map(|si| {
            let seg = rels[1].rel.load_segment(si, &r_col_refs)?;
            let tc = seg.tc();
            let sel = filter_segment_rows(tc, &right_filters)?;
            let rk_col = tc.cols.get(&right_key)?;
            let rc_col = tc.cols.get(&cnt_col)?;
            let mut m: FxMap<i64, i64> = FxMap::default();
            for &row_u32 in &sel {
                let row = row_u32 as usize;
                if let BVal::Int(k) = read_col_bval(rk_col, row)
                    && !matches!(read_col_bval(rc_col, row), BVal::Null)
                {
                    *m.entry(k).or_insert(0) += 1;
                }
            }
            Some(m)
        })
        .collect();

    let mut right_counts: FxMap<i64, i64> = FxMap::default();
    for pm in part_maps? {
        for (k, c) in pm {
            *right_counts.entry(k).or_insert(0) += c;
        }
    }

    // Stream left relation (`customer`) and accumulate per `left_key`
    let mut l_needed = HashSet::new();
    l_needed.insert(left_key.clone());
    for lf in &left_filters {
        lf.collect_cols(&rels[0].rel, &mut l_needed);
    }
    let l_col_vec: Vec<String> = l_needed.into_iter().collect();
    let l_col_refs: Vec<&str> = l_col_vec.iter().map(String::as_str).collect();

    let mut left_groups: FxMap<i64, i64> = FxMap::default();
    for si in 0..rels[0].rel.num_segments() {
        let seg = rels[0].rel.load_segment(si, &l_col_refs)?;
        let tc = seg.tc();
        let sel = filter_segment_rows(tc, &left_filters)?;
        let lk_col = tc.cols.get(&left_key)?;
        for &row_u32 in &sel {
            if let BVal::Int(k) = read_col_bval(lk_col, row_u32 as usize) {
                let add_c = right_counts.get(&k).copied().unwrap_or(0);
                *left_groups.entry(k).or_insert(0) += add_c;
            }
        }
    }

    let out_cols = vec![
        query.columns[0]
            .alias
            .clone()
            .unwrap_or_else(|| left_key.clone()),
        query.columns[1]
            .alias
            .clone()
            .unwrap_or_else(|| "count".to_string()),
    ];
    let rows: Vec<Vec<serde_json::Value>> = left_groups
        .into_iter()
        .map(|(k, c)| {
            vec![
                serde_json::Value::Number(serde_json::Number::from(k)),
                serde_json::Value::Number(serde_json::Number::from(c)),
            ]
        })
        .collect();
    Some((out_cols, rows))
}

fn expr_refs_outer(expr: &Expression, inner_rels: &[RelEntry], outer_rels: &[RelEntry]) -> bool {
    match expr {
        Expression::Column(_) | Expression::QualifiedColumn { .. } => {
            resolve_rel_col(inner_rels, expr).is_none()
                && resolve_rel_col(outer_rels, expr).is_some()
        }
        Expression::Binary { left, right, .. } => {
            expr_refs_outer(left, inner_rels, outer_rels)
                || expr_refs_outer(right, inner_rels, outer_rels)
        }
        Expression::Unary { expr: x, .. }
        | Expression::IsNull { expr: x, .. }
        | Expression::Like { expr: x, .. }
        | Expression::Cast { expr: x, .. } => expr_refs_outer(x, inner_rels, outer_rels),
        Expression::Function { args, .. } => args
            .iter()
            .any(|a| expr_refs_outer(a, inner_rels, outer_rels)),
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand
                .as_deref()
                .is_some_and(|o| expr_refs_outer(o, inner_rels, outer_rels))
                || when_clauses.iter().any(|(w, t)| {
                    expr_refs_outer(w, inner_rels, outer_rels)
                        || expr_refs_outer(t, inner_rels, outer_rels)
                })
                || else_clause
                    .as_deref()
                    .is_some_and(|el| expr_refs_outer(el, inner_rels, outer_rels))
        }
        Expression::In { expr: x, list, .. } => {
            expr_refs_outer(x, inner_rels, outer_rels)
                || list
                    .iter()
                    .any(|i| expr_refs_outer(i, inner_rels, outer_rels))
        }
        Expression::Between {
            expr: x, low, high, ..
        } => {
            expr_refs_outer(x, inner_rels, outer_rels)
                || expr_refs_outer(low, inner_rels, outer_rels)
                || expr_refs_outer(high, inner_rels, outer_rels)
        }
        _ => false,
    }
}

fn subquery_rels(
    sub: &SqlQuery,
    amorphic: &AmorphicTableStorage,
    cte_rels: &HashMap<String, ColRel>,
) -> Option<Vec<RelEntry>> {
    let from = sub.from.as_ref()?;
    let base_tbl = from.table_name()?;
    let mut rels = Vec::with_capacity(1 + sub.joins.len());
    rels.push(RelEntry {
        alias: from.alias.clone().unwrap_or_else(|| base_tbl.to_string()),
        table: base_tbl.to_string(),
        rel: load_rel(base_tbl, amorphic, cte_rels)?,
    });
    for j in &sub.joins {
        rels.push(RelEntry {
            alias: j.alias.clone().unwrap_or_else(|| j.table.clone()),
            table: j.table.clone(),
            rel: load_rel(&j.table, amorphic, cte_rels)?,
        });
    }
    Some(rels)
}

fn eval_uncorrelated_scalar_subquery(
    sub_ast: &AstQuery,
    outer_rels: &[RelEntry],
    amorphic: &AmorphicTableStorage,
    cte_rels: &HashMap<String, ColRel>,
) -> Option<Expression> {
    let sub_sql = ast_query_to_sql_query(sub_ast)?;
    let inner_rels = subquery_rels(&sub_sql, amorphic, cte_rels)?;
    if sub_sql
        .where_clause
        .as_ref()
        .is_some_and(|w| expr_refs_outer(w, &inner_rels, outer_rels))
    {
        return None;
    }
    let (_, rows) = exec_core_query(&sub_sql, amorphic, cte_rels)?;
    let cell = rows.first().and_then(|r| r.first())?;
    match cell {
        serde_json::Value::Null => Some(Expression::Literal(AstValue::Null)),
        serde_json::Value::Bool(b) => Some(Expression::Literal(AstValue::Bool(*b))),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Expression::Literal(AstValue::Int(i)))
            } else {
                Some(Expression::Literal(AstValue::Float(n.as_f64()?)))
            }
        }
        serde_json::Value::String(s) => Some(Expression::Literal(AstValue::String(s.clone()))),
        _ => None,
    }
}

fn resolve_uncorrelated_subqueries_in_expr(
    expr: &Expression,
    outer_rels: &[RelEntry],
    amorphic: &AmorphicTableStorage,
    cte_rels: &HashMap<String, ColRel>,
) -> Option<Expression> {
    match expr {
        Expression::Subquery(sq) => {
            eval_uncorrelated_scalar_subquery(sq, outer_rels, amorphic, cte_rels)
        }
        Expression::Binary { left, op, right } => Some(Expression::Binary {
            left: Box::new(resolve_uncorrelated_subqueries_in_expr(
                left, outer_rels, amorphic, cte_rels,
            )?),
            op: *op,
            right: Box::new(resolve_uncorrelated_subqueries_in_expr(
                right, outer_rels, amorphic, cte_rels,
            )?),
        }),
        Expression::Unary { op, expr: inner } => Some(Expression::Unary {
            op: *op,
            expr: Box::new(resolve_uncorrelated_subqueries_in_expr(
                inner, outer_rels, amorphic, cte_rels,
            )?),
        }),
        Expression::Exists(_) => None,
        other => Some(other.clone()),
    }
}

/// Compute selective surviving integer key set for `(outer_ri, outer_col)` if `outer_ri`
/// (or an equi-joined neighbor) has local filters (`Q02`, `Q17`, `Q20`, `Q21`)!
fn try_compute_outer_domain_set(
    outer_ri: usize,
    outer_col: &str,
    outer_rels: &[RelEntry],
    equi_edges: &[((usize, String), (usize, String))],
    local_filters: &[Vec<LocalFilter>],
) -> Option<IntKeySet> {
    const MAX_DOMAIN_SCAN_ROWS: usize = 65_000_000;
    let est_effective_rows = |ri: usize| -> usize {
        let total = outer_rels[ri].rel.total_rows();
        let mut est = total;
        for lf in &local_filters[ri] {
            match lf {
                LocalFilter::KeyIn {
                    set,
                    negated: false,
                    ..
                } => {
                    // Selective dimension semi-join (`Q21`: `l_suppkey IN {4,000}` on `60M` `lineitem` -> ~4%!)
                    let klen = set.estimated_len();
                    if klen <= 50_000 && total >= 1_000_000 {
                        est = est.min(total / 16);
                    } else {
                        est = est.min(klen.saturating_mul(4).max(1));
                    }
                }
                LocalFilter::Expr(_) | LocalFilter::NeqExists { .. } => {
                    est = (est / 2).max(1);
                }
                _ => {}
            }
        }
        est
    };
    let mut best: Option<(usize, String, usize)> = None;
    let mut consider = |ri: usize, col: &str| {
        if !local_filters[ri].is_empty() {
            let nrows = outer_rels[ri].rel.total_rows();
            let eff = est_effective_rows(ri);
            if nrows <= MAX_DOMAIN_SCAN_ROWS && best.as_ref().is_none_or(|b| eff < b.2) {
                best = Some((ri, col.to_string(), eff));
            }
        }
    };
    consider(outer_ri, outer_col);
    for ((ra, ca), (rb, cb)) in equi_edges {
        if *ra == outer_ri && ca.eq_ignore_ascii_case(outer_col) {
            consider(*rb, cb);
        } else if *rb == outer_ri && cb.eq_ignore_ascii_case(outer_col) {
            consider(*ra, ca);
        }
    }
    let (ri, kcol, _) = best?;
    scan_distinct_int_keys(&outer_rels[ri].rel, &kcol, &local_filters[ri])
}

fn scan_distinct_int_keys(
    rel: &ColRel,
    key_col: &str,
    filters: &[LocalFilter],
) -> Option<IntKeySet> {
    let resolved_k = rel.resolve_col_name(key_col)?;
    let col_refs: [&str; 1] = [resolved_k.as_str()];
    let n_segs = rel.num_segments();
    let parts: Option<Vec<Vec<i64>>> = (0..n_segs)
        .into_par_iter()
        .map(|si| {
            let Some((seg, sel)) = rel.load_filtered_segment(si, &col_refs, filters)? else {
                return Some(Vec::new());
            };
            let tc = seg.tc();
            let kc = tc.cols.get(&resolved_k)?;
            let mut v = Vec::with_capacity(sel.len());
            for &r in &sel {
                if let BVal::Int(k) = read_col_bval(kc, r as usize) {
                    v.push(k);
                }
            }
            Some(v)
        })
        .collect();
    let mut all = Vec::new();
    for p in parts? {
        all.extend(p);
    }
    Some(IntKeySet::from_vec(all))
}

fn decorrelate_conjunct(
    conj: &Expression,
    outer_rels: &[RelEntry],
    equi_edges: &[((usize, String), (usize, String))],
    local_filters: &mut [Vec<LocalFilter>],
    residual_conjs: &mut Vec<Expression>,
    amorphic: &AmorphicTableStorage,
    cte_rels: &HashMap<String, ColRel>,
) -> Option<()> {
    // Case 1: `col [NOT] IN (SELECT inner_col FROM ...)` (`Q16, Q18, Q20`)
    if let Expression::In {
        expr: lhs,
        list,
        negated,
    } = conj
        && list.len() == 1
        && let Expression::Subquery(sq) = &list[0]
    {
        let (outer_ri, outer_col) = resolve_rel_col(outer_rels, lhs)?;
        if !outer_rels[outer_ri].rel.col_is_int(&outer_col) {
            return None;
        }
        let sub_sql = ast_query_to_sql_query(sq)?;
        let (_, rows) = exec_core_query(&sub_sql, amorphic, cte_rels)?;
        let mut keys = Vec::with_capacity(rows.len());
        for r in &rows {
            match r.first() {
                Some(serde_json::Value::Number(n)) => {
                    if let Some(i) = n.as_i64() {
                        keys.push(i);
                    }
                }
                Some(serde_json::Value::Null) => {
                    if *negated {
                        // SQL `x NOT IN (..., NULL)` is never TRUE!
                        return None;
                    }
                }
                _ => return None,
            }
        }
        local_filters[outer_ri].push(LocalFilter::KeyIn {
            col: outer_col,
            set: IntKeySet::from_vec(keys),
            negated: *negated,
        });
        return Some(());
    }

    // Case 2: `EXISTS (SELECT ...)` or `NOT EXISTS (SELECT ...)` (`Q04, Q21, Q22`)
    let exists_info = match conj {
        Expression::Exists(sq) => Some((sq.as_ref(), false)),
        Expression::Unary {
            op: UnaryOperator::Not,
            expr: inner,
        } => match inner.as_ref() {
            Expression::Exists(sq) => Some((sq.as_ref(), true)),
            _ => None,
        },
        _ => None,
    };
    if let Some((sq, is_not)) = exists_info {
        let sub_sql = ast_query_to_sql_query(sq)?;
        if !sub_sql.joins.is_empty() || !sub_sql.group_by.is_empty() {
            return None;
        }
        let inner_rels = subquery_rels(&sub_sql, amorphic, cte_rels)?;
        let inner_rel = &inner_rels[0].rel;
        let mut sub_conjs = Vec::new();
        if let Some(ref w) = sub_sql.where_clause {
            split_and_owned(w, &mut sub_conjs);
        }
        let mut eq_corr: Option<(String, usize, String)> = None;
        let mut ne_corr: Option<(String, usize, String)> = None;
        let mut inner_filters: Vec<LocalFilter> = Vec::new();

        for sc in sub_conjs {
            if !expr_refs_outer(&sc, &inner_rels, outer_rels) {
                inner_filters.push(LocalFilter::Expr(sc));
                continue;
            }
            if let Expression::Binary {
                ref left,
                op: op @ (Operator::Eq | Operator::Ne),
                ref right,
            } = sc
            {
                let pair = if let (Some((_, ic)), Some((ori, oc))) = (
                    resolve_rel_col(&inner_rels, left),
                    resolve_rel_col(outer_rels, right),
                ) {
                    Some((ic, ori, oc))
                } else if let (Some((_, ic)), Some((ori, oc))) = (
                    resolve_rel_col(&inner_rels, right),
                    resolve_rel_col(outer_rels, left),
                ) {
                    Some((ic, ori, oc))
                } else {
                    None
                };
                let (ic, ori, oc) = pair?;
                if !inner_rel.col_is_int(&ic) || !outer_rels[ori].rel.col_is_int(&oc) {
                    return None;
                }
                if op == Operator::Eq && eq_corr.is_none() {
                    eq_corr = Some((ic, ori, oc));
                } else if op == Operator::Ne && ne_corr.is_none() {
                    ne_corr = Some((ic, ori, oc));
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }

        let (in_eq_col, out_ri, out_eq_col) = eq_corr?;
        // Push selective outer correlation key domain into BOTH `NeqExists` (`Q21`) and `EXISTS` (`Q04`)!
        if let Some(ds) =
            try_compute_outer_domain_set(out_ri, &out_eq_col, outer_rels, equi_edges, local_filters)
            && ds.estimated_len().saturating_mul(2) <= inner_rel.total_rows().max(1)
        {
            inner_filters.push(LocalFilter::KeyIn {
                col: in_eq_col.clone(),
                set: ds,
                negated: false,
            });
        }
        if let Some((in_ne_col, ne_out_ri, out_ne_col)) = ne_corr {
            if ne_out_ri != out_ri {
                return None;
            }
            // Stream `inner_rel` in parallel building 64 radix-partitioned `FxMap<i64, NeqSummary>` shards (`Q21`!)
            const NEQ_SHARDS: usize = 64;
            let rk = inner_rel.resolve_col_name(&in_eq_col)?;
            let rn = inner_rel.resolve_col_name(&in_ne_col)?;
            let col_refs: [&str; 2] = [rk.as_str(), rn.as_str()];
            let any_neq_err = std::sync::atomic::AtomicBool::new(false);
            let mut worker_shards: Vec<Vec<FxMap<i64, NeqSummary>>> = (0..inner_rel.num_segments())
                .into_par_iter()
                .fold(
                    || {
                        (0..NEQ_SHARDS)
                            .map(|_| FxMap::<i64, NeqSummary>::default())
                            .collect::<Vec<_>>()
                    },
                    |mut shards, si| {
                        if any_neq_err.load(std::sync::atomic::Ordering::Relaxed) {
                            return shards;
                        }
                        let Some(loaded) =
                            inner_rel.load_filtered_segment(si, &col_refs, &inner_filters)
                        else {
                            any_neq_err.store(true, std::sync::atomic::Ordering::Relaxed);
                            return shards;
                        };
                        let Some((seg, sel)) = loaded else {
                            return shards;
                        };
                        let tc = seg.tc();
                        let (Some(kc), Some(nc)) = (tc.cols.get(&rk), tc.cols.get(&rn)) else {
                            any_neq_err.store(true, std::sync::atomic::Ordering::Relaxed);
                            return shards;
                        };
                        for &r in &sel {
                            let row = r as usize;
                            if let (BVal::Int(k), BVal::Int(nv)) =
                                (read_col_bval(kc, row), read_col_bval(nc, row))
                            {
                                let sh =
                                    (((k as u64).wrapping_mul(0x517cc1b727220a95)) >> 58) as usize;
                                shards[sh]
                                    .entry(k)
                                    .and_modify(|s| {
                                        if let NeqSummary::One(prev) = *s
                                            && prev != nv
                                        {
                                            *s = NeqSummary::Many;
                                        }
                                    })
                                    .or_insert(NeqSummary::One(nv));
                            }
                        }
                        shards
                    },
                )
                .collect();
            if any_neq_err.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            let mut by_shard: Vec<Vec<FxMap<i64, NeqSummary>>> = (0..NEQ_SHARDS)
                .map(|_| Vec::with_capacity(worker_shards.len()))
                .collect();
            for wv in &mut worker_shards {
                for (sh, m) in wv.iter_mut().enumerate() {
                    by_shard[sh].push(std::mem::take(m));
                }
            }
            drop(worker_shards);
            let merged_shards: Vec<FxMap<i64, NeqSummary>> = by_shard
                .into_par_iter()
                .map(|mut parts| {
                    let mut merged = parts.pop().unwrap_or_default();
                    for pm in parts {
                        for (k, v) in pm {
                            merged
                                .entry(k)
                                .and_modify(|cur| match (*cur, v) {
                                    (NeqSummary::Many, _) => {}
                                    (_, NeqSummary::Many) => *cur = NeqSummary::Many,
                                    (NeqSummary::One(a), NeqSummary::One(b)) if a != b => {
                                        *cur = NeqSummary::Many;
                                    }
                                    _ => {}
                                })
                                .or_insert(v);
                        }
                    }
                    merged
                })
                .collect();
            local_filters[out_ri].push(LocalFilter::NeqExists {
                key_col: out_eq_col,
                neq_col: out_ne_col,
                shards: Arc::new(merged_shards),
                negated: is_not,
            });
            return Some(());
        } else {
            let key_set = scan_distinct_int_keys(inner_rel, &in_eq_col, &inner_filters)?;
            local_filters[out_ri].push(LocalFilter::KeyIn {
                col: out_eq_col,
                set: key_set,
                negated: is_not,
            });
            return Some(());
        }
    }

    // Case 3: Correlated or Uncorrelated Scalar Comparison `lhs <op> (SELECT ...)` (`Q02, Q15, Q17, Q20, Q22`)
    let cmp_sub = if let Expression::Binary { left, op, right } = conj
        && matches!(
            op,
            Operator::Eq | Operator::Ne | Operator::Lt | Operator::Le | Operator::Gt | Operator::Ge
        ) {
        if let Expression::Subquery(sq) = right.as_ref() {
            Some((left.as_ref().clone(), *op, sq.as_ref()))
        } else if let Expression::Subquery(sq) = left.as_ref() {
            let flipped = match op {
                Operator::Lt => Operator::Gt,
                Operator::Le => Operator::Ge,
                Operator::Gt => Operator::Lt,
                Operator::Ge => Operator::Le,
                same => *same,
            };
            Some((right.as_ref().clone(), flipped, sq.as_ref()))
        } else {
            None
        }
    } else {
        None
    };

    if let Some((outer_lhs, cmp_op, sq)) = cmp_sub {
        let sub_sql = ast_query_to_sql_query(sq)?;
        let inner_rels = subquery_rels(&sub_sql, amorphic, cte_rels)?;
        let has_corr = sub_sql
            .where_clause
            .as_ref()
            .is_some_and(|w| expr_refs_outer(w, &inner_rels, outer_rels));

        if !has_corr {
            let lit_rhs = eval_uncorrelated_scalar_subquery(sq, outer_rels, amorphic, cte_rels)?;
            let new_cmp = Expression::Binary {
                left: Box::new(outer_lhs),
                op: cmp_op,
                right: Box::new(lit_rhs),
            };
            let mut touched = HashSet::new();
            if expr_referenced_rels(&new_cmp, outer_rels, &mut touched) && touched.len() == 1 {
                let ri = *touched.iter().next()?;
                local_filters[ri].push(LocalFilter::Expr(new_cmp));
            } else {
                residual_conjs.push(new_cmp);
            }
            return Some(());
        }

        // Correlated scalar aggregate subquery (`Q02`, `Q17`, `Q20`)!
        if sub_sql.columns.len() != 1 || !sub_sql.group_by.is_empty() {
            return None;
        }
        let mut sub_conjs = Vec::new();
        if let Some(ref w) = sub_sql.where_clause {
            split_and_owned(w, &mut sub_conjs);
        }
        let mut corr_pairs: Vec<((usize, String), (usize, String))> = Vec::new();
        let mut inner_only: Vec<Expression> = Vec::new();
        for sc in sub_conjs {
            if !expr_refs_outer(&sc, &inner_rels, outer_rels) {
                inner_only.push(sc);
                continue;
            }
            if let Expression::Binary {
                ref left,
                op: Operator::Eq,
                ref right,
            } = sc
            {
                if let (Some(ic), Some(oc)) = (
                    resolve_rel_col(&inner_rels, left),
                    resolve_rel_col(outer_rels, right),
                ) {
                    corr_pairs.push((ic, oc));
                    continue;
                }
                if let (Some(ic), Some(oc)) = (
                    resolve_rel_col(&inner_rels, right),
                    resolve_rel_col(outer_rels, left),
                ) {
                    corr_pairs.push((ic, oc));
                    continue;
                }
            }
            return None;
        }
        if corr_pairs.is_empty() || corr_pairs.len() > 2 {
            return None;
        }

        // Find which single outer relation owns `outer_lhs` AND can probe the correlation key(s)!
        // Note: in `Q02` (`p_partkey = ps_partkey`), `outer_lhs` is `ps_supplycost` (on `partsupp`) while
        // `p_partkey` is on `part` — and `(part, p_partkey) == (partsupp, ps_partkey)` is in `equi_edges`!
        let mut lhs_touched = HashSet::new();
        if !expr_referenced_rels(&outer_lhs, outer_rels, &mut lhs_touched) || lhs_touched.len() != 1
        {
            return None;
        }
        let target_outer_ri = *lhs_touched.iter().next()?;

        let map_to_target_col = |ori: usize, ocol: &str| -> Option<String> {
            if ori == target_outer_ri {
                return Some(ocol.to_string());
            }
            for ((ra, ca), (rb, cb)) in equi_edges {
                if *ra == ori && ca.eq_ignore_ascii_case(ocol) && *rb == target_outer_ri {
                    return Some(cb.clone());
                }
                if *rb == ori && cb.eq_ignore_ascii_case(ocol) && *ra == target_outer_ri {
                    return Some(ca.clone());
                }
            }
            None
        };

        // Push selective outer domain (`Q02` 600 brass parts, `Q17` 200 Brand#23 MED BOX parts,
        // `Q20` 2,100 forest parts) directly into `inner_only` as an `IN` filter if small, OR run grouped query!
        let mut grouped_cols: Vec<SqlColumn> = corr_pairs
            .iter()
            .enumerate()
            .map(|(ki, ((iri, ic), _))| SqlColumn {
                expr: Expression::QualifiedColumn {
                    table: inner_rels[*iri].alias.clone(),
                    column: ic.clone(),
                },
                alias: Some(format!("__ck{ki}")),
            })
            .collect();
        grouped_cols.push(SqlColumn {
            expr: sub_sql.columns[0].expr.clone(),
            alias: Some("__cagg".to_string()),
        });
        let grouped_by: Vec<Expression> = corr_pairs
            .iter()
            .map(|((iri, ic), _)| Expression::QualifiedColumn {
                table: inner_rels[*iri].alias.clone(),
                column: ic.clone(),
            })
            .collect();

        // Semi-join domain pushdown for correlation key 0!
        let ((in_ri0, ref in_c0), (out_ri0, ref out_c0)) = corr_pairs[0];
        let domain_set0 =
            try_compute_outer_domain_set(out_ri0, out_c0, outer_rels, equi_edges, local_filters);

        // Fast single-table inner correlated aggregate (`Q17`, `Q20`) with domain pushdown!
        if inner_rels.len() == 1
            && let Some(fast_res) = try_fast_single_table_corr_agg(
                &inner_rels[0].rel,
                &corr_pairs,
                &sub_sql.columns[0].expr,
                &inner_only,
                domain_set0.as_ref(),
            )
        {
            match fast_res {
                FastCorrMap::M1(m) => {
                    let tcol = map_to_target_col(out_ri0, out_c0)?;
                    local_filters[target_outer_ri].push(LocalFilter::CorrScalar1 {
                        key_col: tcol,
                        lhs_expr: outer_lhs,
                        op: cmp_op,
                        map: Arc::new(m),
                    });
                }
                FastCorrMap::M2(m) => {
                    let tcol1 = map_to_target_col(out_ri0, out_c0)?;
                    let (_, (out_ri1, ref out_c1)) = corr_pairs[1];
                    let tcol2 = map_to_target_col(out_ri1, out_c1)?;
                    local_filters[target_outer_ri].push(LocalFilter::CorrScalar2 {
                        key_col1: tcol1,
                        key_col2: tcol2,
                        lhs_expr: outer_lhs,
                        op: cmp_op,
                        map: Arc::new(m),
                    });
                }
            }
            return Some(());
        }

        // General multi-table correlated aggregate (`Q02`!): inject `KeyIn` bitset/hash directly as a seed filter!
        let seed_f = domain_set0.map(|ds| {
            (
                in_ri0,
                LocalFilter::KeyIn {
                    col: in_c0.clone(),
                    set: ds,
                    negated: false,
                },
            )
        });

        let grouped_q = SqlQuery {
            ctes: Vec::new(),
            distinct: false,
            columns: grouped_cols,
            from: sub_sql.from.clone(),
            joins: sub_sql.joins.clone(),
            where_clause: combine_and(inner_only),
            group_by: grouped_by,
            having: None,
            order_by: Vec::new(),
            limit: None,
            offset: None,
            set_op: None,
            as_of_timestamp: None,
        };
        let (_, grows) = exec_core_query_seeded(&grouped_q, amorphic, cte_rels, seed_f)?;
        if corr_pairs.len() == 1 {
            let tcol = map_to_target_col(out_ri0, out_c0)?;
            let mut m: FxMap<i64, f64> = FxMap::default();
            m.reserve(grows.len());
            for gr in &grows {
                if let (Some(k), Some(v)) = (
                    gr.first().and_then(|x| x.as_i64()),
                    gr.get(1).and_then(|x| x.as_f64()),
                ) {
                    m.insert(k, v);
                }
            }
            local_filters[target_outer_ri].push(LocalFilter::CorrScalar1 {
                key_col: tcol,
                lhs_expr: outer_lhs,
                op: cmp_op,
                map: Arc::new(m),
            });
            return Some(());
        } else {
            let tcol1 = map_to_target_col(out_ri0, out_c0)?;
            let (_, (out_ri1, ref out_c1)) = corr_pairs[1];
            let tcol2 = map_to_target_col(out_ri1, out_c1)?;
            let mut m: FxMap<(i64, i64), f64> = FxMap::default();
            m.reserve(grows.len());
            for gr in &grows {
                if let (Some(k1), Some(k2), Some(v)) = (
                    gr.first().and_then(|x| x.as_i64()),
                    gr.get(1).and_then(|x| x.as_i64()),
                    gr.get(2).and_then(|x| x.as_f64()),
                ) {
                    m.insert((k1, k2), v);
                }
            }
            local_filters[target_outer_ri].push(LocalFilter::CorrScalar2 {
                key_col1: tcol1,
                key_col2: tcol2,
                lhs_expr: outer_lhs,
                op: cmp_op,
                map: Arc::new(m),
            });
            return Some(());
        }
    }

    None
}

enum FastCorrMap {
    M1(FxMap<i64, f64>),
    M2(FxMap<(i64, i64), f64>),
}

fn try_fast_single_table_corr_agg(
    rel: &ColRel,
    corr_pairs: &[((usize, String), (usize, String))],
    select_expr: &Expression,
    inner_preds: &[Expression],
    domain_k0: Option<&IntKeySet>,
) -> Option<FastCorrMap> {
    // Unwrap optional `scale * AGG(col)` or `AGG(col) * scale`
    let (scale, agg_fn, agg_arg) = match select_expr {
        Expression::Function { name, args } if args.len() == 1 => (1.0, name.as_str(), &args[0]),
        Expression::Binary {
            left,
            op: Operator::Mul,
            right,
        } => {
            if let (Expression::Literal(v), Expression::Function { name, args }) =
                (left.as_ref(), right.as_ref())
                && args.len() == 1
            {
                let sc = match v {
                    AstValue::Float(f) => *f,
                    AstValue::Int(i) => *i as f64,
                    _ => return None,
                };
                (sc, name.as_str(), &args[0])
            } else if let (Expression::Function { name, args }, Expression::Literal(v)) =
                (left.as_ref(), right.as_ref())
                && args.len() == 1
            {
                let sc = match v {
                    AstValue::Float(f) => *f,
                    AstValue::Int(i) => *i as f64,
                    _ => return None,
                };
                (sc, name.as_str(), &args[0])
            } else {
                return None;
            }
        }
        _ => return None,
    };

    #[derive(Clone, Copy)]
    enum SimpleAgg {
        Sum,
        Avg,
        Min,
        Max,
    }
    let kind = match agg_fn.to_ascii_uppercase().as_str() {
        "SUM" => SimpleAgg::Sum,
        "AVG" => SimpleAgg::Avg,
        "MIN" => SimpleAgg::Min,
        "MAX" => SimpleAgg::Max,
        _ => return None,
    };

    let val_col = match agg_arg {
        Expression::Column(c) => rel.resolve_col_name(c)?,
        Expression::QualifiedColumn { column, .. } => rel.resolve_col_name(column)?,
        _ => return None,
    };
    let k0_col = &corr_pairs[0].0.1;

    let mut filters: Vec<LocalFilter> = Vec::new();
    if let Some(ds) = domain_k0 {
        filters.push(LocalFilter::KeyIn {
            col: k0_col.clone(),
            set: ds.clone(),
            negated: false,
        });
    }
    for p in inner_preds {
        filters.push(LocalFilter::Expr(p.clone()));
    }

    let rk0 = rel.resolve_col_name(k0_col)?;
    if corr_pairs.len() == 1 {
        let col_refs: [&str; 2] = [rk0.as_str(), val_col.as_str()];
        let parts: Option<Vec<FxMap<i64, (f64, u64)>>> = (0..rel.num_segments())
            .into_par_iter()
            .map(|si| {
                let Some((seg, sel)) = rel.load_filtered_segment(si, &col_refs, &filters)? else {
                    return Some(FxMap::default());
                };
                let tc = seg.tc();
                let kc = tc.cols.get(&rk0)?;
                let vc = tc.cols.get(&val_col)?;
                let mut m: FxMap<i64, (f64, u64)> = FxMap::default();
                for &r in &sel {
                    let row = r as usize;
                    if let (BVal::Int(k), Some(v)) =
                        (read_col_bval(kc, row), read_col_bval(vc, row).as_f64())
                    {
                        m.entry(k)
                            .and_modify(|(acc, cnt)| {
                                match kind {
                                    SimpleAgg::Sum | SimpleAgg::Avg => *acc += v,
                                    SimpleAgg::Min => *acc = acc.min(v),
                                    SimpleAgg::Max => *acc = acc.max(v),
                                }
                                *cnt += 1;
                            })
                            .or_insert((v, 1));
                    }
                }
                Some(m)
            })
            .collect();
        let mut merged: FxMap<i64, (f64, u64)> = FxMap::default();
        for pm in parts? {
            for (k, (v, c)) in pm {
                merged
                    .entry(k)
                    .and_modify(|(acc, cnt)| {
                        match kind {
                            SimpleAgg::Sum | SimpleAgg::Avg => *acc += v,
                            SimpleAgg::Min => *acc = acc.min(v),
                            SimpleAgg::Max => *acc = acc.max(v),
                        }
                        *cnt += c;
                    })
                    .or_insert((v, c));
            }
        }
        let mut out: FxMap<i64, f64> = FxMap::default();
        out.reserve(merged.len());
        for (k, (acc, cnt)) in merged {
            let final_v = match kind {
                SimpleAgg::Avg => (acc / (cnt as f64)) * scale,
                _ => acc * scale,
            };
            out.insert(k, final_v);
        }
        Some(FastCorrMap::M1(out))
    } else {
        let rk1 = rel.resolve_col_name(&corr_pairs[1].0.1)?;
        let col_refs: [&str; 3] = [rk0.as_str(), rk1.as_str(), val_col.as_str()];
        let parts: Option<Vec<FxMap<(i64, i64), (f64, u64)>>> = (0..rel.num_segments())
            .into_par_iter()
            .map(|si| {
                let Some((seg, sel)) = rel.load_filtered_segment(si, &col_refs, &filters)? else {
                    return Some(FxMap::default());
                };
                let tc = seg.tc();
                let kc0 = tc.cols.get(&rk0)?;
                let kc1 = tc.cols.get(&rk1)?;
                let vc = tc.cols.get(&val_col)?;
                let mut m: FxMap<(i64, i64), (f64, u64)> = FxMap::default();
                for &r in &sel {
                    let row = r as usize;
                    if let (BVal::Int(k0), BVal::Int(k1), Some(v)) = (
                        read_col_bval(kc0, row),
                        read_col_bval(kc1, row),
                        read_col_bval(vc, row).as_f64(),
                    ) {
                        m.entry((k0, k1))
                            .and_modify(|(acc, cnt)| {
                                match kind {
                                    SimpleAgg::Sum | SimpleAgg::Avg => *acc += v,
                                    SimpleAgg::Min => *acc = acc.min(v),
                                    SimpleAgg::Max => *acc = acc.max(v),
                                }
                                *cnt += 1;
                            })
                            .or_insert((v, 1));
                    }
                }
                Some(m)
            })
            .collect();
        let mut merged: FxMap<(i64, i64), (f64, u64)> = FxMap::default();
        for pm in parts? {
            for (k, (v, c)) in pm {
                merged
                    .entry(k)
                    .and_modify(|(acc, cnt)| {
                        match kind {
                            SimpleAgg::Sum | SimpleAgg::Avg => *acc += v,
                            SimpleAgg::Min => *acc = acc.min(v),
                            SimpleAgg::Max => *acc = acc.max(v),
                        }
                        *cnt += c;
                    })
                    .or_insert((v, c));
            }
        }
        let mut out: FxMap<(i64, i64), f64> = FxMap::default();
        out.reserve(merged.len());
        for (k, (acc, cnt)) in merged {
            let final_v = match kind {
                SimpleAgg::Avg => (acc / (cnt as f64)) * scale,
                _ => acc * scale,
            };
            out.insert(k, final_v);
        }
        Some(FastCorrMap::M2(out))
    }
}

fn propagate_initial_key_sets(
    rels: &[RelEntry],
    equi_edges: &[((usize, String), (usize, String))],
    local_filters: &mut [Vec<LocalFilter>],
) -> Option<()> {
    // Multi-hop transitive semi-join pushdown (`Q21`: `nation ('SAUDI ARABIA') -> supplier (4,000) -> l1`!)
    let mut scanned_cols: HashSet<(usize, String)> = HashSet::new();
    let mut pushed_edges: HashSet<(usize, String, usize, String)> = HashSet::new();

    for round in 0..3 {
        let mut seeds: Vec<(usize, String, IntKeySet)> = Vec::new();
        if round == 0 {
            for (ri, flist) in local_filters.iter().enumerate() {
                for f in flist {
                    if let LocalFilter::KeyIn {
                        col,
                        set,
                        negated: false,
                    } = f
                    {
                        seeds.push((ri, col.clone(), set.clone()));
                    }
                }
            }
        }

        for (ri, r) in rels.iter().enumerate() {
            let max_rows = if round == 0 { 20_000_000 } else { 2_000_000 };
            if !local_filters[ri].is_empty() && r.rel.total_rows() <= max_rows {
                let mut key_cols = HashSet::new();
                for ((ra, ca), (rb, cb)) in equi_edges {
                    if *ra == ri {
                        key_cols.insert(ca.clone());
                    } else if *rb == ri {
                        key_cols.insert(cb.clone());
                    }
                }
                for kc in key_cols {
                    let key_id = (ri, kc.to_ascii_lowercase());
                    if scanned_cols.contains(&key_id) {
                        continue;
                    }
                    scanned_cols.insert(key_id);
                    if let Some(ks) = scan_distinct_int_keys(&r.rel, &kc, &local_filters[ri])
                        && ks.estimated_len().saturating_mul(2) <= r.rel.total_rows().max(1)
                    {
                        seeds.push((ri, kc, ks));
                    }
                }
            }
        }

        if seeds.is_empty() {
            break;
        }
        let mut any_new = false;
        for (src_ri, src_col, set) in seeds {
            for ((ra, ca), (rb, cb)) in equi_edges {
                if *ra == src_ri && ca.eq_ignore_ascii_case(&src_col) {
                    let eid = (
                        src_ri,
                        ca.to_ascii_lowercase(),
                        *rb,
                        cb.to_ascii_lowercase(),
                    );
                    if pushed_edges.insert(eid) {
                        local_filters[*rb].push(LocalFilter::KeyIn {
                            col: cb.clone(),
                            set: set.clone(),
                            negated: false,
                        });
                        any_new = true;
                    }
                } else if *rb == src_ri && cb.eq_ignore_ascii_case(&src_col) {
                    let eid = (
                        src_ri,
                        cb.to_ascii_lowercase(),
                        *ra,
                        ca.to_ascii_lowercase(),
                    );
                    if pushed_edges.insert(eid) {
                        local_filters[*ra].push(LocalFilter::KeyIn {
                            col: ca.clone(),
                            set: set.clone(),
                            negated: false,
                        });
                        any_new = true;
                    }
                }
            }
        }
        if !any_new {
            break;
        }
    }
    Some(())
}

/// Multi-relation expression evaluated during tuple probing (`JoinExpr`).
#[derive(Clone)]
enum JoinExpr {
    LitNull,
    LitBool(bool),
    LitInt(i64),
    LitFloat(f64),
    LitStr(String),
    Slot(usize),
    Neg(Box<JoinExpr>),
    Not(Box<JoinExpr>),
    IsNull(Box<JoinExpr>, bool),
    Arith(Box<JoinExpr>, Operator, Box<JoinExpr>),
    Cmp(Box<JoinExpr>, Operator, Box<JoinExpr>),
    And(Box<JoinExpr>, Box<JoinExpr>),
    Or(Box<JoinExpr>, Box<JoinExpr>),
    Between {
        expr: Box<JoinExpr>,
        low: Box<JoinExpr>,
        high: Box<JoinExpr>,
        negated: bool,
    },
    InList {
        expr: Box<JoinExpr>,
        list: Vec<JoinExpr>,
        negated: bool,
    },
    Like {
        expr: Box<JoinExpr>,
        pat: Arc<LikePattern>,
        negated: bool,
    },
    Substr {
        expr: Box<JoinExpr>,
        start_1based: usize,
        len: Option<usize>,
    },
    ExtractYear(Box<JoinExpr>),
    ExtractMonth(Box<JoinExpr>),
    ExtractDay(Box<JoinExpr>),
    Case {
        operand: Option<Box<JoinExpr>>,
        whens: Vec<(JoinExpr, JoinExpr)>,
        else_expr: Option<Box<JoinExpr>>,
    },
}

impl JoinExpr {
    fn collect_slots(&self, out: &mut HashSet<usize>) {
        match self {
            JoinExpr::LitNull
            | JoinExpr::LitBool(_)
            | JoinExpr::LitInt(_)
            | JoinExpr::LitFloat(_)
            | JoinExpr::LitStr(_) => {}
            JoinExpr::Slot(s) => {
                out.insert(*s);
            }
            JoinExpr::Neg(e)
            | JoinExpr::Not(e)
            | JoinExpr::IsNull(e, _)
            | JoinExpr::Like { expr: e, .. }
            | JoinExpr::Substr { expr: e, .. }
            | JoinExpr::ExtractYear(e)
            | JoinExpr::ExtractMonth(e)
            | JoinExpr::ExtractDay(e) => e.collect_slots(out),
            JoinExpr::Arith(a, _, b)
            | JoinExpr::Cmp(a, _, b)
            | JoinExpr::And(a, b)
            | JoinExpr::Or(a, b) => {
                a.collect_slots(out);
                b.collect_slots(out);
            }
            JoinExpr::Between {
                expr, low, high, ..
            } => {
                expr.collect_slots(out);
                low.collect_slots(out);
                high.collect_slots(out);
            }
            JoinExpr::InList { expr, list, .. } => {
                expr.collect_slots(out);
                for x in list {
                    x.collect_slots(out);
                }
            }
            JoinExpr::Case {
                operand,
                whens,
                else_expr,
            } => {
                if let Some(op) = operand {
                    op.collect_slots(out);
                }
                for (w, t) in whens {
                    w.collect_slots(out);
                    t.collect_slots(out);
                }
                if let Some(el) = else_expr {
                    el.collect_slots(out);
                }
            }
        }
    }
}

fn compile_join_expr(
    expr: &Expression,
    rels: &[RelEntry],
    slot_map: &mut Vec<(usize, String)>,
) -> Option<JoinExpr> {
    let mut get_slot = |ri: usize, col: String| -> usize {
        if let Some(pos) = slot_map
            .iter()
            .position(|(r, c)| *r == ri && c.eq_ignore_ascii_case(&col))
        {
            pos
        } else {
            let pos = slot_map.len();
            slot_map.push((ri, col));
            pos
        }
    };

    match expr {
        Expression::Literal(v) => match v {
            AstValue::Null => Some(JoinExpr::LitNull),
            AstValue::Bool(b) => Some(JoinExpr::LitBool(*b)),
            AstValue::Int(i) => Some(JoinExpr::LitInt(*i)),
            AstValue::Float(f) => Some(JoinExpr::LitFloat(*f)),
            AstValue::String(s) => Some(JoinExpr::LitStr(s.clone())),
            AstValue::Timestamp(ts) => Some(JoinExpr::LitStr(epoch_secs_to_iso10(*ts))),
            _ => None,
        },
        Expression::Column(_) | Expression::QualifiedColumn { .. } => {
            let (ri, col) = resolve_rel_col(rels, expr)?;
            Some(JoinExpr::Slot(get_slot(ri, col)))
        }
        Expression::Cast {
            expr: inner,
            target_type,
        } => {
            let sub = compile_join_expr(inner, rels, slot_map)?;
            let tt = target_type.to_ascii_uppercase();
            if tt == "DATE" {
                if let JoinExpr::LitStr(s) = sub {
                    return Some(JoinExpr::LitStr(if s.len() >= 10 {
                        s[..10].to_string()
                    } else {
                        s
                    }));
                }
                return None;
            }
            if tt.contains("FLOAT")
                || tt.contains("DOUBLE")
                || tt.contains("DECIMAL")
                || tt.contains("NUMERIC")
                || tt.contains("REAL")
            {
                return match sub {
                    JoinExpr::LitFloat(f) => Some(JoinExpr::LitFloat(f)),
                    JoinExpr::LitInt(i) => Some(JoinExpr::LitFloat(i as f64)),
                    JoinExpr::LitStr(s) => s.trim().parse::<f64>().ok().map(JoinExpr::LitFloat),
                    _ => None,
                };
            }
            if tt.contains("INT") {
                return match sub {
                    JoinExpr::LitInt(i) => Some(JoinExpr::LitInt(i)),
                    JoinExpr::LitFloat(f) => Some(JoinExpr::LitInt(f as i64)),
                    JoinExpr::LitStr(s) => s.trim().parse::<i64>().ok().map(JoinExpr::LitInt),
                    _ => None,
                };
            }
            None
        }
        Expression::Unary { op, expr: inner } => {
            let sub = compile_join_expr(inner, rels, slot_map)?;
            match op {
                UnaryOperator::Neg => Some(JoinExpr::Neg(Box::new(sub))),
                UnaryOperator::Not => Some(JoinExpr::Not(Box::new(sub))),
                _ => None,
            }
        }
        Expression::IsNull {
            expr: inner,
            negated,
        } => Some(JoinExpr::IsNull(
            Box::new(compile_join_expr(inner, rels, slot_map)?),
            *negated,
        )),
        Expression::Like {
            expr: inner,
            pattern,
            negated,
            case_insensitive,
        } => Some(JoinExpr::Like {
            expr: Box::new(compile_join_expr(inner, rels, slot_map)?),
            pat: Arc::new(LikePattern::new(pattern, *case_insensitive)),
            negated: *negated,
        }),
        Expression::Between {
            expr: inner,
            low,
            high,
            negated,
        } => Some(JoinExpr::Between {
            expr: Box::new(compile_join_expr(inner, rels, slot_map)?),
            low: Box::new(compile_join_expr(low, rels, slot_map)?),
            high: Box::new(compile_join_expr(high, rels, slot_map)?),
            negated: *negated,
        }),
        Expression::In {
            expr: inner,
            list,
            negated,
        } => {
            let sub = compile_join_expr(inner, rels, slot_map)?;
            let mut cl = Vec::with_capacity(list.len());
            for item in list {
                cl.push(compile_join_expr(item, rels, slot_map)?);
            }
            Some(JoinExpr::InList {
                expr: Box::new(sub),
                list: cl,
                negated: *negated,
            })
        }
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            let op_c = match operand {
                Some(o) => Some(Box::new(compile_join_expr(o, rels, slot_map)?)),
                None => None,
            };
            let mut wc = Vec::with_capacity(when_clauses.len());
            for (w, t) in when_clauses {
                wc.push((
                    compile_join_expr(w, rels, slot_map)?,
                    compile_join_expr(t, rels, slot_map)?,
                ));
            }
            let el_c = match else_clause {
                Some(el) => Some(Box::new(compile_join_expr(el, rels, slot_map)?)),
                None => None,
            };
            Some(JoinExpr::Case {
                operand: op_c,
                whens: wc,
                else_expr: el_c,
            })
        }
        Expression::Function { name, args } => {
            let up = name.to_ascii_uppercase();
            match up.as_str() {
                "SUBSTRING" | "SUBSTR" if args.len() == 2 || args.len() == 3 => {
                    let inner = compile_join_expr(&args[0], rels, slot_map)?;
                    let JoinExpr::LitInt(st) = compile_join_expr(&args[1], rels, slot_map)? else {
                        return None;
                    };
                    let len = if args.len() == 3 {
                        let JoinExpr::LitInt(l) = compile_join_expr(&args[2], rels, slot_map)?
                        else {
                            return None;
                        };
                        Some(l.max(0) as usize)
                    } else {
                        None
                    };
                    Some(JoinExpr::Substr {
                        expr: Box::new(inner),
                        start_1based: st.max(1) as usize,
                        len,
                    })
                }
                "EXTRACT" | "DATE_PART" if args.len() == 2 => {
                    let JoinExpr::LitStr(part) = compile_join_expr(&args[0], rels, slot_map)?
                    else {
                        return None;
                    };
                    let inner = compile_join_expr(&args[1], rels, slot_map)?;
                    match part.to_ascii_uppercase().as_str() {
                        "YEAR" => Some(JoinExpr::ExtractYear(Box::new(inner))),
                        "MONTH" => Some(JoinExpr::ExtractMonth(Box::new(inner))),
                        "DAY" => Some(JoinExpr::ExtractDay(Box::new(inner))),
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        Expression::Binary { left, op, right } => {
            let l = compile_join_expr(left, rels, slot_map)?;
            let r = compile_join_expr(right, rels, slot_map)?;
            match op {
                Operator::And => Some(JoinExpr::And(Box::new(l), Box::new(r))),
                Operator::Or => Some(JoinExpr::Or(Box::new(l), Box::new(r))),
                Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
                    Some(JoinExpr::Arith(Box::new(l), *op, Box::new(r)))
                }
                Operator::Eq
                | Operator::Ne
                | Operator::Lt
                | Operator::Le
                | Operator::Gt
                | Operator::Ge => Some(JoinExpr::Cmp(Box::new(l), *op, Box::new(r))),
                _ => None,
            }
        }
        _ => None,
    }
}

impl JoinExpr {
    #[inline]
    fn eval<'a>(&'a self, tuple: &[BVal<'a>]) -> BVal<'a> {
        match self {
            JoinExpr::LitNull => BVal::Null,
            JoinExpr::LitBool(b) => BVal::Bool(*b),
            JoinExpr::LitInt(i) => BVal::Int(*i),
            JoinExpr::LitFloat(f) => BVal::Float(*f),
            JoinExpr::LitStr(s) => BVal::Str(s.as_str()),
            JoinExpr::Slot(si) => tuple[*si],
            JoinExpr::Neg(x) => match x.eval(tuple) {
                BVal::Int(i) => BVal::Int(i.wrapping_neg()),
                BVal::Float(f) => BVal::Float(-f),
                _ => BVal::Null,
            },
            JoinExpr::Not(x) => match x.eval(tuple) {
                BVal::Bool(b) => BVal::Bool(!b),
                _ => BVal::Null,
            },
            JoinExpr::IsNull(x, neg) => {
                let is_n = matches!(x.eval(tuple), BVal::Null);
                BVal::Bool(if *neg { !is_n } else { is_n })
            }
            JoinExpr::And(a, b) => {
                let av = a.eval(tuple);
                if matches!(av, BVal::Bool(false)) {
                    return BVal::Bool(false);
                }
                let bv = b.eval(tuple);
                match (av, bv) {
                    (BVal::Bool(true), BVal::Bool(true)) => BVal::Bool(true),
                    (_, BVal::Bool(false)) => BVal::Bool(false),
                    _ => BVal::Null,
                }
            }
            JoinExpr::Or(a, b) => {
                let av = a.eval(tuple);
                if matches!(av, BVal::Bool(true)) {
                    return BVal::Bool(true);
                }
                let bv = b.eval(tuple);
                match (av, bv) {
                    (_, BVal::Bool(true)) => BVal::Bool(true),
                    (BVal::Bool(false), BVal::Bool(false)) => BVal::Bool(false),
                    _ => BVal::Null,
                }
            }
            JoinExpr::Arith(l, op, r) => {
                let lv = l.eval(tuple);
                let rv = r.eval(tuple);
                match (lv, rv) {
                    (BVal::Null, _) | (_, BVal::Null) => BVal::Null,
                    (BVal::Int(a), BVal::Int(b)) => match op {
                        Operator::Add => BVal::Int(a.wrapping_add(b)),
                        Operator::Sub => BVal::Int(a.wrapping_sub(b)),
                        Operator::Mul => BVal::Int(a.wrapping_mul(b)),
                        Operator::Div => {
                            if b == 0 {
                                BVal::Null
                            } else {
                                BVal::Float((a as f64) / (b as f64))
                            }
                        }
                        _ => BVal::Null,
                    },
                    (a, b) => {
                        let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) else {
                            return BVal::Null;
                        };
                        match op {
                            Operator::Add => BVal::Float(fa + fb),
                            Operator::Sub => BVal::Float(fa - fb),
                            Operator::Mul => BVal::Float(fa * fb),
                            Operator::Div => {
                                if fb == 0.0 {
                                    BVal::Null
                                } else {
                                    BVal::Float(fa / fb)
                                }
                            }
                            _ => BVal::Null,
                        }
                    }
                }
            }
            JoinExpr::Cmp(l, op, r) => {
                let lv = l.eval(tuple);
                let rv = r.eval(tuple);
                match (lv, rv) {
                    (BVal::Float(fa), BVal::Float(fb)) => BVal::Bool(eval_float_cmp(fa, *op, fb)),
                    (BVal::Int(ia), BVal::Float(fb)) => {
                        BVal::Bool(eval_float_cmp(ia as f64, *op, fb))
                    }
                    (BVal::Float(fa), BVal::Int(ib)) => {
                        BVal::Bool(eval_float_cmp(fa, *op, ib as f64))
                    }
                    _ => match cmp_bvals(lv, rv) {
                        Some(ord) => BVal::Bool(ord_matches(ord, *op)),
                        None => BVal::Null,
                    },
                }
            }
            JoinExpr::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let v = expr.eval(tuple);
                let lv = low.eval(tuple);
                let hv = high.eval(tuple);
                let (Some(c1), Some(c2)) = (cmp_bvals(v, lv), cmp_bvals(v, hv)) else {
                    return BVal::Null;
                };
                let inside = c1 != Ordering::Less && c2 != Ordering::Greater;
                BVal::Bool(if *negated { !inside } else { inside })
            }
            JoinExpr::InList {
                expr,
                list,
                negated,
            } => {
                let v = expr.eval(tuple);
                if matches!(v, BVal::Null) {
                    return BVal::Null;
                }
                let mut found = false;
                for item in list {
                    if cmp_bvals(v, item.eval(tuple)) == Some(Ordering::Equal) {
                        found = true;
                        break;
                    }
                }
                BVal::Bool(if *negated { !found } else { found })
            }
            JoinExpr::Like { expr, pat, negated } => match expr.eval(tuple) {
                BVal::Str(s) => BVal::Bool(pat.matches(s) != *negated),
                _ => BVal::Null,
            },
            JoinExpr::Substr {
                expr,
                start_1based,
                len,
            } => match expr.eval(tuple) {
                BVal::Str(s) => BVal::Str(substr_ascii(s, *start_1based, *len)),
                _ => BVal::Null,
            },
            JoinExpr::ExtractYear(x) => match x.eval(tuple) {
                BVal::Str(s) => {
                    parse_iso_ymd_fast(s).map_or(BVal::Null, |(y, _, _)| BVal::Int(y as i64))
                }
                BVal::Int(ts) => BVal::Int(epoch_secs_to_ymd(ts).0 as i64),
                _ => BVal::Null,
            },
            JoinExpr::ExtractMonth(x) => match x.eval(tuple) {
                BVal::Str(s) => {
                    parse_iso_ymd_fast(s).map_or(BVal::Null, |(_, m, _)| BVal::Int(m as i64))
                }
                BVal::Int(ts) => BVal::Int(epoch_secs_to_ymd(ts).1 as i64),
                _ => BVal::Null,
            },
            JoinExpr::ExtractDay(x) => match x.eval(tuple) {
                BVal::Str(s) => {
                    parse_iso_ymd_fast(s).map_or(BVal::Null, |(_, _, d)| BVal::Int(d as i64))
                }
                BVal::Int(ts) => BVal::Int(epoch_secs_to_ymd(ts).2 as i64),
                _ => BVal::Null,
            },
            JoinExpr::Case {
                operand,
                whens,
                else_expr,
            } => {
                if let Some(op_e) = operand {
                    let target = op_e.eval(tuple);
                    for (w, t) in whens {
                        if cmp_bvals(target, w.eval(tuple)) == Some(Ordering::Equal) {
                            return t.eval(tuple);
                        }
                    }
                } else {
                    for (w, t) in whens {
                        if matches!(w.eval(tuple), BVal::Bool(true)) {
                            return t.eval(tuple);
                        }
                    }
                }
                else_expr.as_ref().map_or(BVal::Null, |el| el.eval(tuple))
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AggOp {
    CountStar,
    Count,
    CountDistinctInt,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Clone)]
struct AggSpec {
    op: AggOp,
    arg: Option<JoinExpr>,
    raw_expr: Expression,
}

#[derive(Clone)]
enum PostAggExpr {
    Group(usize),
    Agg(usize),
    Lit(OwnedVal),
    Neg(Box<PostAggExpr>),
    Binary(Box<PostAggExpr>, Operator, Box<PostAggExpr>),
}

fn compile_post_agg(
    expr: &Expression,
    group_raw: &[Expression],
    rels: &[RelEntry],
    slot_map: &mut Vec<(usize, String)>,
    agg_specs: &mut Vec<AggSpec>,
) -> Option<PostAggExpr> {
    if let Some(gpos) = group_raw.iter().position(|ge| ge == expr) {
        return Some(PostAggExpr::Group(gpos));
    }
    // Also match qualified vs bare column equivalence against `group_raw`
    if let Some(rc) = resolve_rel_col(rels, expr) {
        for (gi, ge) in group_raw.iter().enumerate() {
            if resolve_rel_col(rels, ge).as_ref() == Some(&rc) {
                return Some(PostAggExpr::Group(gi));
            }
        }
    }

    match expr {
        Expression::Literal(v) => match v {
            AstValue::Null => Some(PostAggExpr::Lit(OwnedVal::Null)),
            AstValue::Bool(b) => Some(PostAggExpr::Lit(OwnedVal::Bool(*b))),
            AstValue::Int(i) => Some(PostAggExpr::Lit(OwnedVal::Int(*i))),
            AstValue::Float(f) => Some(PostAggExpr::Lit(OwnedVal::Float(*f))),
            AstValue::String(s) => Some(PostAggExpr::Lit(OwnedVal::Str(Arc::from(s.as_str())))),
            _ => None,
        },
        Expression::Unary {
            op: UnaryOperator::Neg,
            expr: inner,
        } => Some(PostAggExpr::Neg(Box::new(compile_post_agg(
            inner, group_raw, rels, slot_map, agg_specs,
        )?))),
        Expression::Binary { left, op, right } => Some(PostAggExpr::Binary(
            Box::new(compile_post_agg(
                left, group_raw, rels, slot_map, agg_specs,
            )?),
            *op,
            Box::new(compile_post_agg(
                right, group_raw, rels, slot_map, agg_specs,
            )?),
        )),
        Expression::Function { name, args } => {
            let up = name.to_ascii_uppercase();
            let (op, arg_je) = match up.as_str() {
                "COUNT"
                    if args.is_empty()
                        || (args.len() == 1 && matches!(args[0], Expression::Wildcard)) =>
                {
                    (AggOp::CountStar, None)
                }
                "COUNT" if args.len() == 1 => (
                    AggOp::Count,
                    Some(compile_join_expr(&args[0], rels, slot_map)?),
                ),
                "COUNT_DISTINCT" if args.len() == 1 => {
                    let (ri, col) = resolve_rel_col(rels, &args[0])?;
                    if !rels[ri].rel.col_is_int(&col) {
                        return None;
                    }
                    (
                        AggOp::CountDistinctInt,
                        Some(compile_join_expr(&args[0], rels, slot_map)?),
                    )
                }
                "SUM" if args.len() == 1 => (
                    AggOp::Sum,
                    Some(compile_join_expr(&args[0], rels, slot_map)?),
                ),
                "AVG" if args.len() == 1 => (
                    AggOp::Avg,
                    Some(compile_join_expr(&args[0], rels, slot_map)?),
                ),
                "MIN" if args.len() == 1 => (
                    AggOp::Min,
                    Some(compile_join_expr(&args[0], rels, slot_map)?),
                ),
                "MAX" if args.len() == 1 => (
                    AggOp::Max,
                    Some(compile_join_expr(&args[0], rels, slot_map)?),
                ),
                _ => return None,
            };
            if let Some(pos) = agg_specs
                .iter()
                .position(|s| s.op == op && &s.raw_expr == expr)
            {
                return Some(PostAggExpr::Agg(pos));
            }
            let pos = agg_specs.len();
            agg_specs.push(AggSpec {
                op,
                arg: arg_je,
                raw_expr: expr.clone(),
            });
            Some(PostAggExpr::Agg(pos))
        }
        _ => None,
    }
}

#[derive(Clone)]
struct AggState {
    cnt: u64,
    isum: i128,
    fsum: f64,
    fcomp: f64,
    saw_float: bool,
    min: Option<OwnedVal>,
    max: Option<OwnedVal>,
    distinct_ints: Option<FxSet<i64>>,
}

impl AggState {
    fn new(op: AggOp) -> Self {
        Self {
            cnt: 0,
            isum: 0,
            fsum: 0.0,
            fcomp: 0.0,
            saw_float: false,
            min: None,
            max: None,
            distinct_ints: (op == AggOp::CountDistinctInt).then(FxSet::default),
        }
    }

    #[inline]
    fn add_f64(&mut self, v: f64) {
        let t = self.fsum + v;
        if self.fsum.abs() >= v.abs() {
            self.fcomp += (self.fsum - t) + v;
        } else {
            self.fcomp += (v - t) + self.fsum;
        }
        self.fsum = t;
    }

    #[inline]
    fn update(&mut self, op: AggOp, val: BVal<'_>) {
        match op {
            AggOp::CountStar => self.cnt += 1,
            AggOp::Count => {
                if !matches!(val, BVal::Null) {
                    self.cnt += 1;
                }
            }
            AggOp::CountDistinctInt => {
                if let (BVal::Int(k), Some(set)) = (val, &mut self.distinct_ints) {
                    set.insert(k);
                }
            }
            AggOp::Sum | AggOp::Avg => match val {
                BVal::Int(i) => {
                    self.cnt += 1;
                    self.isum += i as i128;
                }
                BVal::Float(f) => {
                    self.cnt += 1;
                    self.saw_float = true;
                    self.add_f64(f);
                }
                _ => {}
            },
            AggOp::Min => {
                if !matches!(val, BVal::Null) {
                    let replace = match &self.min {
                        None => true,
                        Some(cur) => cmp_owned_vs_bval(cur, val) == Some(Ordering::Greater),
                    };
                    if replace {
                        self.min = Some(val.to_owned_val());
                    }
                }
            }
            AggOp::Max => {
                if !matches!(val, BVal::Null) {
                    let replace = match &self.max {
                        None => true,
                        Some(cur) => cmp_owned_vs_bval(cur, val) == Some(Ordering::Less),
                    };
                    if replace {
                        self.max = Some(val.to_owned_val());
                    }
                }
            }
        }
    }

    fn merge(&mut self, op: AggOp, other: Self) {
        self.cnt += other.cnt;
        self.isum += other.isum;
        self.add_f64(other.fsum);
        self.add_f64(other.fcomp);
        self.saw_float |= other.saw_float;
        if let (Some(dst), Some(src)) = (&mut self.distinct_ints, other.distinct_ints) {
            for k in src {
                dst.insert(k);
            }
        }
        if let Some(omin) = other.min {
            let replace = match &self.min {
                None => true,
                Some(cur) => cmp_owned_vals(cur, &omin) == Ordering::Greater,
            };
            if replace {
                self.min = Some(omin);
            }
        }
        if let Some(omax) = other.max {
            let replace = match &self.max {
                None => true,
                Some(cur) => cmp_owned_vals(cur, &omax) == Ordering::Less,
            };
            if replace {
                self.max = Some(omax);
            }
        }
        let _ = op;
    }

    fn finalize(&self, op: AggOp) -> OwnedVal {
        let total_f = (self.fsum + self.fcomp) + (self.isum as f64);
        match op {
            AggOp::CountStar | AggOp::Count => OwnedVal::Int(self.cnt as i64),
            AggOp::CountDistinctInt => {
                OwnedVal::Int(self.distinct_ints.as_ref().map_or(0, |s| s.len() as i64))
            }
            AggOp::Sum => {
                if self.cnt == 0 {
                    OwnedVal::Null
                } else if self.saw_float {
                    OwnedVal::Float(total_f)
                } else if let Ok(i) = i64::try_from(self.isum) {
                    OwnedVal::Int(i)
                } else {
                    OwnedVal::Float(self.isum as f64)
                }
            }
            AggOp::Avg => {
                if self.cnt == 0 {
                    OwnedVal::Null
                } else {
                    OwnedVal::Float(total_f / (self.cnt as f64))
                }
            }
            AggOp::Min => self.min.clone().unwrap_or(OwnedVal::Null),
            AggOp::Max => self.max.clone().unwrap_or(OwnedVal::Null),
        }
    }
}

fn cmp_owned_vs_bval(a: &OwnedVal, b: BVal<'_>) -> Option<Ordering> {
    let ab = match a {
        OwnedVal::Null => BVal::Null,
        OwnedVal::Bool(x) => BVal::Bool(*x),
        OwnedVal::Int(x) => BVal::Int(*x),
        OwnedVal::Float(x) => BVal::Float(*x),
        OwnedVal::Str(x) => BVal::Str(x.as_ref()),
    };
    cmp_bvals(ab, b)
}

fn cmp_owned_vals(a: &OwnedVal, b: &OwnedVal) -> Ordering {
    match (a, b) {
        (OwnedVal::Null, OwnedVal::Null) => Ordering::Equal,
        (OwnedVal::Null, _) => Ordering::Greater,
        (_, OwnedVal::Null) => Ordering::Less,
        (OwnedVal::Int(x), OwnedVal::Int(y)) => x.cmp(y),
        (OwnedVal::Float(x), OwnedVal::Float(y)) => x.total_cmp(y),
        (OwnedVal::Int(x), OwnedVal::Float(y)) => (*x as f64).total_cmp(y),
        (OwnedVal::Float(x), OwnedVal::Int(y)) => x.total_cmp(&(*y as f64)),
        (OwnedVal::Str(x), OwnedVal::Str(y)) => x.cmp(y),
        (OwnedVal::Bool(x), OwnedVal::Bool(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

impl PostAggExpr {
    fn eval(&self, gvals: &[OwnedVal], avals: &[OwnedVal]) -> OwnedVal {
        match self {
            PostAggExpr::Group(gi) => gvals[*gi].clone(),
            PostAggExpr::Agg(ai) => avals[*ai].clone(),
            PostAggExpr::Lit(v) => v.clone(),
            PostAggExpr::Neg(x) => match x.eval(gvals, avals) {
                OwnedVal::Int(i) => OwnedVal::Int(-i),
                OwnedVal::Float(f) => OwnedVal::Float(-f),
                _ => OwnedVal::Null,
            },
            PostAggExpr::Binary(l, op, r) => {
                let lv = l.eval(gvals, avals);
                let rv = r.eval(gvals, avals);
                match op {
                    Operator::And => match (lv, rv) {
                        (OwnedVal::Bool(a), OwnedVal::Bool(b)) => OwnedVal::Bool(a && b),
                        _ => OwnedVal::Bool(false),
                    },
                    Operator::Or => match (lv, rv) {
                        (OwnedVal::Bool(a), OwnedVal::Bool(b)) => OwnedVal::Bool(a || b),
                        _ => OwnedVal::Bool(false),
                    },
                    Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
                        match (&lv, &rv) {
                            (OwnedVal::Null, _) | (_, OwnedVal::Null) => OwnedVal::Null,
                            (OwnedVal::Int(a), OwnedVal::Int(b)) if *op != Operator::Div => {
                                match op {
                                    Operator::Add => OwnedVal::Int(a.wrapping_add(*b)),
                                    Operator::Sub => OwnedVal::Int(a.wrapping_sub(*b)),
                                    Operator::Mul => OwnedVal::Int(a.wrapping_mul(*b)),
                                    _ => OwnedVal::Null,
                                }
                            }
                            _ => {
                                let fa = match lv {
                                    OwnedVal::Int(i) => i as f64,
                                    OwnedVal::Float(f) => f,
                                    _ => return OwnedVal::Null,
                                };
                                let fb = match rv {
                                    OwnedVal::Int(i) => i as f64,
                                    OwnedVal::Float(f) => f,
                                    _ => return OwnedVal::Null,
                                };
                                match op {
                                    Operator::Add => OwnedVal::Float(fa + fb),
                                    Operator::Sub => OwnedVal::Float(fa - fb),
                                    Operator::Mul => OwnedVal::Float(fa * fb),
                                    Operator::Div => {
                                        if fb == 0.0 {
                                            OwnedVal::Null
                                        } else {
                                            OwnedVal::Float(fa / fb)
                                        }
                                    }
                                    _ => OwnedVal::Null,
                                }
                            }
                        }
                    }
                    Operator::Eq
                    | Operator::Ne
                    | Operator::Lt
                    | Operator::Le
                    | Operator::Gt
                    | Operator::Ge => {
                        if matches!(lv, OwnedVal::Null) || matches!(rv, OwnedVal::Null) {
                            return OwnedVal::Bool(false);
                        }
                        let ord = cmp_owned_vals(&lv, &rv);
                        OwnedVal::Bool(ord_matches(ord, *op))
                    }
                    _ => OwnedVal::Null,
                }
            }
        }
    }
}

/// Hashable group key for multi-column `GROUP BY`.
#[derive(Clone, PartialEq, Eq, Hash)]
enum GKeyElem {
    Null,
    Bool(bool),
    Int(i64),
    FloatBits(u64),
    Str(Arc<str>),
}

impl GKeyElem {
    #[inline]
    fn from_bval(v: BVal<'_>) -> Self {
        match v {
            BVal::Null => GKeyElem::Null,
            BVal::Bool(b) => GKeyElem::Bool(b),
            BVal::Int(i) => GKeyElem::Int(i),
            BVal::Float(f) => GKeyElem::FloatBits(if f == 0.0 {
                0.0f64.to_bits()
            } else {
                f.to_bits()
            }),
            BVal::Str(s) => GKeyElem::Str(Arc::from(s)),
        }
    }

    fn to_owned_val(&self) -> OwnedVal {
        match self {
            GKeyElem::Null => OwnedVal::Null,
            GKeyElem::Bool(b) => OwnedVal::Bool(*b),
            GKeyElem::Int(i) => OwnedVal::Int(*i),
            GKeyElem::FloatBits(bits) => OwnedVal::Float(f64::from_bits(*bits)),
            GKeyElem::Str(s) => OwnedVal::Str(s.clone()),
        }
    }
}

/// Zero-allocation borrowed group key element for `Fast Path B` (`Q10`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum BorrowKeyElem<'a> {
    Null,
    Bool(bool),
    Int(i64),
    FloatBits(u64),
    Str(&'a str),
}

impl<'a> BorrowKeyElem<'a> {
    #[inline]
    fn from_bval(v: BVal<'a>) -> Self {
        match v {
            BVal::Null => BorrowKeyElem::Null,
            BVal::Bool(b) => BorrowKeyElem::Bool(b),
            BVal::Int(i) => BorrowKeyElem::Int(i),
            BVal::Float(f) => BorrowKeyElem::FloatBits(if f == 0.0 {
                0.0f64.to_bits()
            } else {
                f.to_bits()
            }),
            BVal::Str(s) => BorrowKeyElem::Str(s),
        }
    }

    #[inline]
    fn to_owned_val(self) -> OwnedVal {
        match self {
            BorrowKeyElem::Null => OwnedVal::Null,
            BorrowKeyElem::Bool(b) => OwnedVal::Bool(b),
            BorrowKeyElem::Int(i) => OwnedVal::Int(i),
            BorrowKeyElem::FloatBits(bits) => OwnedVal::Float(f64::from_bits(bits)),
            BorrowKeyElem::Str(s) => OwnedVal::Str(Arc::from(s)),
        }
    }
}

/// Compact 48-byte `Copy` numeric accumulator (`0` `Drop` glue, `4.3x` smaller than `[AggState; 2]`)
/// for high-cardinality single-integer `GROUP BY` (`Q18` `15M` groups at `SF=10`).
#[derive(Clone, Copy, Default)]
struct CompactNumAgg {
    isum: i128,
    fsum: f64,
    fcomp: f64,
    cnt: u64,
    saw_float: bool,
}

impl CompactNumAgg {
    #[inline]
    fn add_f64(&mut self, v: f64) {
        let t = self.fsum + v;
        if self.fsum.abs() >= v.abs() {
            self.fcomp += (self.fsum - t) + v;
        } else {
            self.fcomp += (v - t) + self.fsum;
        }
        self.fsum = t;
    }

    #[inline]
    fn update(&mut self, op: AggOp, val: BVal<'_>) {
        match op {
            AggOp::CountStar => self.cnt += 1,
            AggOp::Count => {
                if !matches!(val, BVal::Null) {
                    self.cnt += 1;
                }
            }
            AggOp::Sum | AggOp::Avg => match val {
                BVal::Int(i) => {
                    self.cnt += 1;
                    self.isum += i as i128;
                }
                BVal::Float(f) => {
                    self.cnt += 1;
                    self.saw_float = true;
                    self.add_f64(f);
                }
                _ => {}
            },
            _ => {}
        }
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        self.cnt += other.cnt;
        self.isum += other.isum;
        self.add_f64(other.fsum);
        self.add_f64(other.fcomp);
        self.saw_float |= other.saw_float;
    }

    #[inline]
    fn finalize(&self, op: AggOp) -> OwnedVal {
        let total_f = (self.fsum + self.fcomp) + (self.isum as f64);
        match op {
            AggOp::CountStar | AggOp::Count => OwnedVal::Int(self.cnt as i64),
            AggOp::Sum => {
                if self.cnt == 0 {
                    OwnedVal::Null
                } else if self.saw_float {
                    OwnedVal::Float(total_f)
                } else if let Ok(i) = i64::try_from(self.isum) {
                    OwnedVal::Int(i)
                } else {
                    OwnedVal::Float(self.isum as f64)
                }
            }
            AggOp::Avg => {
                if self.cnt == 0 {
                    OwnedVal::Null
                } else {
                    OwnedVal::Float(total_f / (self.cnt as f64))
                }
            }
            _ => OwnedVal::Null,
        }
    }
}

fn post_agg_uses_any_group(pe: &PostAggExpr) -> bool {
    match pe {
        PostAggExpr::Group(_) => true,
        PostAggExpr::Agg(_) | PostAggExpr::Lit(_) => false,
        PostAggExpr::Neg(inner) => post_agg_uses_any_group(inner),
        PostAggExpr::Binary(l, _, r) => post_agg_uses_any_group(l) || post_agg_uses_any_group(r),
    }
}

fn post_agg_uses_other_group(pe: &PostAggExpr, allowed_gi: usize) -> bool {
    match pe {
        PostAggExpr::Group(gi) => *gi != allowed_gi,
        PostAggExpr::Agg(_) | PostAggExpr::Lit(_) => false,
        PostAggExpr::Neg(inner) => post_agg_uses_other_group(inner, allowed_gi),
        PostAggExpr::Binary(l, _, r) => {
            post_agg_uses_other_group(l, allowed_gi) || post_agg_uses_other_group(r, allowed_gi)
        }
    }
}

/// Build a Snowflake branch tree rooted at `root_ri` (`root_ri != probe_ri`) post-order,
/// with top-down FK semi-join pushdown when `node` is filtered and `child` is large & unfiltered!
#[allow(clippy::too_many_arguments)]
fn build_snowflake_branch(
    root_ri: usize,
    probe_attach: &[(String, String)], // `(probe_col, root_col)` (1 or 2 keys)
    branch_nodes: &[usize],
    tree_children: &HashMap<usize, Vec<(usize, String, String)>>, // `parent -> [(child, parent_col, child_col)]`
    rels: &[RelEntry],
    local_filters: &mut [Vec<LocalFilter>],
    needed_cols_by_rel: &[HashSet<String>],
) -> Option<BuiltBranch> {
    fn collect_surviving_fk_set(
        node: usize,
        fk_col_name: &str,
        rels: &[RelEntry],
        filters: &[LocalFilter],
        child_total_rows: usize,
    ) -> Option<IntKeySet> {
        let resolved_fk = rels[node].rel.resolve_col_name(fk_col_name)?;
        if !rels[node].rel.col_is_int(&resolved_fk) {
            return None;
        }
        let n_segs = rels[node].rel.num_segments();
        let fk_refs = [resolved_fk.as_str()];
        let seg_keys: Vec<Vec<i64>> = (0..n_segs)
            .into_par_iter()
            .map(|si| {
                let Some((seg, sel)) = rels[node]
                    .rel
                    .load_filtered_segment(si, &fk_refs, filters)?
                else {
                    return Some(Vec::new());
                };
                let Col::Int(ref ints, ref nulls) = *seg.tc().cols.get(&resolved_fk)? else {
                    return None;
                };
                let mut out = Vec::with_capacity(sel.len());
                for &r_u32 in &sel {
                    let ri = r_u32 as usize;
                    if !nl(nulls, ri) {
                        out.push(ints.at(ri));
                    }
                }
                Some(out)
            })
            .collect::<Option<Vec<_>>>()?;
        let total_fk: usize = seg_keys.iter().map(Vec::len).sum();
        if total_fk > 4_000_000 || total_fk.saturating_mul(2) > child_total_rows {
            return None;
        }
        let mut all_fk = Vec::with_capacity(total_fk);
        for sk in seg_keys {
            all_fk.extend(sk);
        }
        Some(IntKeySet::from_vec(all_fk))
    }

    // Post-order build: build every child of `node` first so its surviving PK set pushes into `node`!
    fn materialize_node(
        node: usize,
        parent_join_cols: &[String],
        tree_children: &HashMap<usize, Vec<(usize, String, String)>>,
        rels: &[RelEntry],
        local_filters: &mut [Vec<LocalFilter>],
        needed_cols_by_rel: &[HashSet<String>],
    ) -> Option<Vec<CompactCol>> {
        let children = tree_children.get(&node).cloned().unwrap_or_default();
        let mut built_children: Vec<(String, BuiltBranch)> = Vec::with_capacity(children.len());
        for (child_ri, parent_col, child_col) in &children {
            // Top-down semi-join pushdown (`Q10`: filtered `orders` -> unfiltered `1.5M` `customer`)
            let child_rows = rels[*child_ri].rel.total_rows();
            if !local_filters[node].is_empty()
                && local_filters[*child_ri].is_empty()
                && child_rows >= 10_000
                && let Some(fk_set) = collect_surviving_fk_set(
                    node,
                    parent_col,
                    rels,
                    &local_filters[node],
                    child_rows,
                )
            {
                local_filters[*child_ri].push(LocalFilter::KeyIn {
                    col: child_col.clone(),
                    set: fk_set,
                    negated: false,
                });
            }

            let child_cols = materialize_node(
                *child_ri,
                std::slice::from_ref(child_col),
                tree_children,
                rels,
                local_filters,
                needed_cols_by_rel,
            )?;
            // Extract key vector for `(*child_ri, child_col)`
            let kc_pos = child_cols
                .iter()
                .position(|c| c.id.0 == *child_ri && c.id.1.eq_ignore_ascii_case(child_col))?;
            let Col::Int(ref k_ints, ref k_nulls) = child_cols[kc_pos].data else {
                return None;
            };
            let mut ckeys = Vec::with_capacity(k_ints.len());
            for i in 0..k_ints.len() {
                if !nl(k_nulls, i) {
                    ckeys.push(k_ints.at(i));
                } else {
                    return None;
                }
            }
            let cb = BuiltBranch::from_rows_1key(parent_col.clone(), &ckeys, child_cols);
            if let Some(ks) = cb.key_set_1() {
                local_filters[node].push(LocalFilter::KeyIn {
                    col: parent_col.clone(),
                    set: ks,
                    negated: false,
                });
            }
            built_children.push((parent_col.clone(), cb));
        }

        // Determine columns to read from `node`
        let mut node_cols_set = needed_cols_by_rel[node].clone();
        for pc in parent_join_cols {
            if let Some(rc) = rels[node].rel.resolve_col_name(pc) {
                node_cols_set.insert(rc);
            }
        }
        let mut resolved_child_fks: Vec<String> = Vec::with_capacity(built_children.len());
        for (pcol, _) in &built_children {
            let rc = rels[node].rel.resolve_col_name(pcol)?;
            node_cols_set.insert(rc.clone());
            resolved_child_fks.push(rc);
        }
        if node_cols_set.is_empty()
            && let Some(first_c) = rels[node].rel.col_order().first()
        {
            node_cols_set.insert(first_c.clone());
        }
        let keep_node_cols: Vec<String> = node_cols_set.into_iter().collect();
        let keep_refs: Vec<&str> = keep_node_cols.iter().map(String::as_str).collect();

        // Prepare output builders for `keep_node_cols` + all `built_children` columns
        let mut out_builders: Option<(Vec<(usize, String)>, Vec<ColBuilder>)> = None;

        let n_segs = rels[node].rel.num_segments();
        const PAR_BATCH_SEGS: usize = 16;
        for seg_base in (0..n_segs).step_by(PAR_BATCH_SEGS) {
            let seg_end = (seg_base + PAR_BATCH_SEGS).min(n_segs);
            let loaded_batch: Vec<Option<(SegData<'_>, Vec<u32>)>> = (seg_base..seg_end)
                .into_par_iter()
                .map(|si| {
                    rels[node]
                        .rel
                        .load_filtered_segment(si, &keep_refs, &local_filters[node])
                })
                .collect::<Option<Vec<_>>>()?;

            for entry in loaded_batch {
                let Some((seg, sel)) = entry else {
                    continue;
                };
                let tc = seg.tc();
                if out_builders.is_none() {
                    let mut ids = Vec::new();
                    let mut blds = Vec::new();
                    for kc in &keep_node_cols {
                        let c = tc.cols.get(kc)?;
                        ids.push((node, kc.clone()));
                        blds.push(ColBuilder::new_for_sample(c)?);
                    }
                    for (_, cb) in &built_children {
                        for cc in &cb.cols {
                            ids.push(cc.id.clone());
                            blds.push(ColBuilder::new_for_sample(&cc.data)?);
                        }
                    }
                    out_builders = Some((ids, blds));
                }
                if sel.is_empty() {
                    continue;
                }
                let (_, blds) = out_builders.as_mut()?;

                let node_col_refs: Vec<&Col> = keep_node_cols
                    .iter()
                    .filter_map(|kc| tc.cols.get(kc))
                    .collect();
                if node_col_refs.len() != keep_node_cols.len() {
                    return None;
                }
                let child_fk_cols: Vec<&Col> = resolved_child_fks
                    .iter()
                    .filter_map(|fk| tc.cols.get(fk))
                    .collect();
                if child_fk_cols.len() != built_children.len() {
                    return None;
                }

                // Recursive emission across `built_children` (handles both 1:1 and 1:N child matches)
                fn emit_with_children(
                    ci_idx: usize,
                    node_row: usize,
                    node_col_refs: &[&Col],
                    child_fk_cols: &[&Col],
                    built_children: &[(String, BuiltBranch)],
                    chosen_child_rows: &mut Vec<usize>,
                    blds: &mut [ColBuilder],
                ) {
                    if ci_idx == built_children.len() {
                        let mut bi = 0usize;
                        for &nc in node_col_refs {
                            blds[bi].push_from_col(nc, node_row);
                            bi += 1;
                        }
                        for (chi, (_, cb)) in built_children.iter().enumerate() {
                            let crow = chosen_child_rows[chi];
                            for cc in &cb.cols {
                                blds[bi].push_from_col(&cc.data, crow);
                                bi += 1;
                            }
                        }
                        return;
                    }
                    let BVal::Int(fk) = read_col_bval(child_fk_cols[ci_idx], node_row) else {
                        return;
                    };
                    let cb = &built_children[ci_idx].1;
                    if let Some(r_idx) = cb.probe_single_1(fk) {
                        chosen_child_rows.push(r_idx as usize);
                        emit_with_children(
                            ci_idx + 1,
                            node_row,
                            node_col_refs,
                            child_fk_cols,
                            built_children,
                            chosen_child_rows,
                            blds,
                        );
                        chosen_child_rows.pop();
                    } else if let BranchLookup::HashMulti1 { map, row_ids } = &cb.lookup
                        && let Some(&(st, ln)) = map.get(&fk)
                    {
                        for &ri_u32 in &row_ids[st as usize..(st + ln) as usize] {
                            chosen_child_rows.push(ri_u32 as usize);
                            emit_with_children(
                                ci_idx + 1,
                                node_row,
                                node_col_refs,
                                child_fk_cols,
                                built_children,
                                chosen_child_rows,
                                blds,
                            );
                            chosen_child_rows.pop();
                        }
                    }
                }

                let mut chosen = Vec::with_capacity(built_children.len());
                for &r_u32 in &sel {
                    emit_with_children(
                        0,
                        r_u32 as usize,
                        &node_col_refs,
                        &child_fk_cols,
                        &built_children,
                        &mut chosen,
                        blds,
                    );
                }
            }
        }

        if out_builders.is_none() && n_segs > 0 {
            // Zero surviving rows across all segments: initialize empty typed columns from segment 0
            let seg0 = rels[node].rel.load_segment(0, &keep_refs)?;
            let tc0 = seg0.tc();
            let mut ids = Vec::new();
            let mut blds = Vec::new();
            for kc in &keep_node_cols {
                let c = tc0.cols.get(kc)?;
                ids.push((node, kc.clone()));
                blds.push(ColBuilder::new_for_sample(c)?);
            }
            for (_, cb) in &built_children {
                for cc in &cb.cols {
                    ids.push(cc.id.clone());
                    blds.push(ColBuilder::new_for_sample(&cc.data)?);
                }
            }
            out_builders = Some((ids, blds));
        }

        let (ids, blds) = match out_builders {
            Some(pair) => pair,
            None => {
                // Empty 0-segment relation
                return Some(Vec::new());
            }
        };
        Some(
            ids.into_iter()
                .zip(blds)
                .map(|(id, b)| CompactCol {
                    id,
                    data: b.finish(),
                })
                .collect(),
        )
    }

    let _ = branch_nodes;
    let root_join_cols: Vec<String> = probe_attach.iter().map(|(_, rc)| rc.clone()).collect();
    let cols = materialize_node(
        root_ri,
        &root_join_cols,
        tree_children,
        rels,
        local_filters,
        needed_cols_by_rel,
    )?;

    if probe_attach.len() == 1 {
        let (ref pcol, ref rcol) = probe_attach[0];
        if cols.is_empty() {
            return Some(BuiltBranch::from_rows_1key(pcol.clone(), &[], Vec::new()));
        }
        let kpos = cols
            .iter()
            .position(|c| c.id.0 == root_ri && c.id.1.eq_ignore_ascii_case(rcol))?;
        let Col::Int(ref ints, ref nulls) = cols[kpos].data else {
            return None;
        };
        let mut keys = Vec::with_capacity(ints.len());
        let mut non_null_rows = Vec::new();
        let has_any_null = (0..ints.len()).any(|i| nl(nulls, i));
        if !has_any_null {
            for i in 0..ints.len() {
                keys.push(ints.at(i));
            }
            Some(BuiltBranch::from_rows_1key(pcol.clone(), &keys, cols))
        } else {
            for i in 0..ints.len() {
                if !nl(nulls, i) {
                    keys.push(ints.at(i));
                    non_null_rows.push(i);
                }
            }
            let filtered_cols = filter_compact_cols(cols, &non_null_rows)?;
            Some(BuiltBranch::from_rows_1key(
                pcol.clone(),
                &keys,
                filtered_cols,
            ))
        }
    } else if probe_attach.len() == 2 {
        let (ref pk1, ref rk1) = probe_attach[0];
        let (ref pk2, ref rk2) = probe_attach[1];
        if cols.is_empty() {
            return Some(BuiltBranch::from_rows_2key(
                pk1.clone(),
                pk2.clone(),
                &[],
                Vec::new(),
            ));
        }
        let kp1 = cols
            .iter()
            .position(|c| c.id.0 == root_ri && c.id.1.eq_ignore_ascii_case(rk1))?;
        let kp2 = cols
            .iter()
            .position(|c| c.id.0 == root_ri && c.id.1.eq_ignore_ascii_case(rk2))?;
        let (Col::Int(i1, n1), Col::Int(i2, n2)) = (&cols[kp1].data, &cols[kp2].data) else {
            return None;
        };
        let mut keys = Vec::with_capacity(i1.len());
        for i in 0..i1.len() {
            if nl(n1, i) || nl(n2, i) {
                return None;
            }
            keys.push((i1.at(i), i2.at(i)));
        }
        Some(BuiltBranch::from_rows_2key(
            pk1.clone(),
            pk2.clone(),
            &keys,
            cols,
        ))
    } else {
        None
    }
}

fn filter_compact_cols(cols: Vec<CompactCol>, rows: &[usize]) -> Option<Vec<CompactCol>> {
    let mut out = Vec::with_capacity(cols.len());
    for cc in cols {
        let mut b = ColBuilder::new_for_sample(&cc.data)?;
        for &r in rows {
            b.push_from_col(&cc.data, r);
        }
        out.push(CompactCol {
            id: cc.id,
            data: b.finish(),
        });
    }
    Some(out)
}

fn exec_bushy_plan(
    query: &SqlQuery,
    having_opt: Option<&Expression>,
    rels: &[RelEntry],
    equi_edges: &[((usize, String), (usize, String))],
    local_filters: &mut [Vec<LocalFilter>],
    residual_conjs: &[Expression],
) -> Option<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
    let k = rels.len();

    // 1. Resolve `GROUP BY` expressions (ordinals `1..N` and `SELECT` aliases)
    let mut group_raw: Vec<Expression> = Vec::with_capacity(query.group_by.len());
    for ge in &query.group_by {
        let resolved = match ge {
            Expression::Literal(AstValue::Int(n))
                if *n >= 1 && (*n as usize) <= query.columns.len() =>
            {
                query.columns[*n as usize - 1].expr.clone()
            }
            Expression::Column(alias_c) if resolve_rel_col(rels, ge).is_none() => query
                .columns
                .iter()
                .find(|sc| {
                    sc.alias
                        .as_ref()
                        .is_some_and(|a| a.eq_ignore_ascii_case(alias_c))
                })
                .map(|sc| sc.expr.clone())?,
            other => other.clone(),
        };
        group_raw.push(resolved);
    }

    let is_agg_query = !group_raw.is_empty()
        || having_opt.is_some()
        || query.columns.iter().any(|c| expr_has_any_agg(&c.expr));

    // 2. Choose `probe_ri`: the largest effective relation!
    let probe_ri = (0..k).max_by_key(|&ri| {
        let base = rels[ri].rel.total_rows();
        if local_filters[ri].is_empty() {
            base.saturating_mul(2)
        } else {
            base
        }
    })?;

    // 3. Build spanning Snowflake forest over `{0..k} \ {probe_ri}`!
    let mut visited = vec![false; k];
    visited[probe_ri] = true;

    // Direct attachments `(root_ri -> Vec<(probe_col, root_col)>)`
    let mut root_attachments: Vec<(usize, Vec<(String, String)>)> = Vec::new();
    let mut used_edge = vec![false; equi_edges.len()];

    for (ei, ((ra, ca), (rb, cb))) in equi_edges.iter().enumerate() {
        if *ra == probe_ri && *rb != probe_ri {
            if let Some((_, list)) = root_attachments.iter_mut().find(|(r, _)| *r == *rb) {
                if list.len() < 2 {
                    list.push((ca.clone(), cb.clone()));
                    used_edge[ei] = true;
                }
            } else {
                visited[*rb] = true;
                root_attachments.push((*rb, vec![(ca.clone(), cb.clone())]));
                used_edge[ei] = true;
            }
        } else if *rb == probe_ri && *ra != probe_ri {
            if let Some((_, list)) = root_attachments.iter_mut().find(|(r, _)| *r == *ra) {
                if list.len() < 2 {
                    list.push((cb.clone(), ca.clone()));
                    used_edge[ei] = true;
                }
            } else {
                visited[*ra] = true;
                root_attachments.push((*ra, vec![(cb.clone(), ca.clone())]));
                used_edge[ei] = true;
            }
        }
    }

    // BFS down from `root_attachments` to attach dimension chains (`orders -> customer -> nation -> region`)
    let mut tree_children: HashMap<usize, Vec<(usize, String, String)>> = HashMap::new();
    let mut branch_owner: Vec<Option<usize>> = vec![None; k];
    let mut bfs_queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for &(r_ri, _) in &root_attachments {
        branch_owner[r_ri] = Some(r_ri);
        bfs_queue.push_back(r_ri);
    }
    while let Some(curr) = bfs_queue.pop_front() {
        for (ei, ((ra, ca), (rb, cb))) in equi_edges.iter().enumerate() {
            if used_edge[ei] {
                continue;
            }
            if *ra == curr && !visited[*rb] {
                visited[*rb] = true;
                used_edge[ei] = true;
                branch_owner[*rb] = branch_owner[curr];
                tree_children
                    .entry(curr)
                    .or_default()
                    .push((*rb, ca.clone(), cb.clone()));
                bfs_queue.push_back(*rb);
            } else if *rb == curr && !visited[*ra] {
                visited[*ra] = true;
                used_edge[ei] = true;
                branch_owner[*ra] = branch_owner[curr];
                tree_children
                    .entry(curr)
                    .or_default()
                    .push((*ra, cb.clone(), ca.clone()));
                bfs_queue.push_back(*ra);
            }
        }
    }

    // Reject disconnected cross joins (every build relation must reach `probe_ri`)
    if visited.iter().any(|&v| !v) {
        return None;
    }

    // Any unused equi_edge (e.g. cycle `c_nationkey = s_nationkey` in `Q05` or `p_partkey = ps_partkey` in `Q02/Q09`)
    // becomes an exact residual equality predicate so zero semantics are ever lost!
    let mut all_residuals: Vec<Expression> = residual_conjs.to_vec();
    for (ei, ((ra, ca), (rb, cb))) in equi_edges.iter().enumerate() {
        if !used_edge[ei] {
            all_residuals.push(Expression::Binary {
                left: Box::new(Expression::QualifiedColumn {
                    table: rels[*ra].alias.clone(),
                    column: ca.clone(),
                }),
                op: Operator::Eq,
                right: Box::new(Expression::QualifiedColumn {
                    table: rels[*rb].alias.clone(),
                    column: cb.clone(),
                }),
            });
        }
    }

    // 4. Compile `SELECT`, `GROUP BY`, `HAVING`, `ORDER BY`, and `all_residuals` into `slot_map`!
    let mut slot_map: Vec<(usize, String)> = Vec::new();
    let mut compiled_residuals: Vec<JoinExpr> = Vec::with_capacity(all_residuals.len());
    for rc in &all_residuals {
        compiled_residuals.push(compile_join_expr(rc, rels, &mut slot_map)?);
    }

    let mut out_names: Vec<String> = Vec::with_capacity(query.columns.len());
    for sc in &query.columns {
        let default_n = match &sc.expr {
            Expression::Column(c) => c.clone(),
            Expression::QualifiedColumn { column, .. } => column.clone(),
            Expression::Function { name, .. } => name.clone(),
            _ => "expr".to_string(),
        };
        out_names.push(sc.alias.clone().unwrap_or(default_n));
    }
    let visible_cols = out_names.len();

    let mut compiled_groups: Vec<JoinExpr> = Vec::with_capacity(group_raw.len());
    for ge in &group_raw {
        compiled_groups.push(compile_join_expr(ge, rels, &mut slot_map)?);
    }

    let mut agg_specs: Vec<AggSpec> = Vec::new();
    let mut post_selects: Vec<PostAggExpr> = Vec::new();
    let mut proj_selects: Vec<JoinExpr> = Vec::new();
    let mut having_compiled: Option<PostAggExpr> = None;

    if is_agg_query {
        for sc in &query.columns {
            post_selects.push(compile_post_agg(
                &sc.expr,
                &group_raw,
                rels,
                &mut slot_map,
                &mut agg_specs,
            )?);
        }
        if let Some(h) = having_opt {
            having_compiled = Some(compile_post_agg(
                h,
                &group_raw,
                rels,
                &mut slot_map,
                &mut agg_specs,
            )?);
        }
    } else {
        for sc in &query.columns {
            if matches!(
                sc.expr,
                Expression::Wildcard | Expression::QualifiedWildcard(_)
            ) {
                return None;
            }
            proj_selects.push(compile_join_expr(&sc.expr, rels, &mut slot_map)?);
        }
    }

    // Compile `ORDER BY` (resolving ordinals, SELECT aliases, or extra hidden columns)
    let mut order_specs: Vec<(usize, bool, Option<bool>)> =
        Vec::with_capacity(query.order_by.len());
    for ob in &query.order_by {
        let idx = match &ob.expr {
            Expression::Literal(AstValue::Int(i)) if *i >= 1 && (*i as usize) <= visible_cols => {
                *i as usize - 1
            }
            Expression::Column(c) if out_names.iter().any(|n| n.eq_ignore_ascii_case(c)) => {
                out_names.iter().position(|n| n.eq_ignore_ascii_case(c))?
            }
            e => {
                if let Some(pos) = query.columns.iter().position(|sc| &sc.expr == e) {
                    pos
                } else if is_agg_query {
                    let hidden_idx = post_selects.len();
                    post_selects.push(compile_post_agg(
                        e,
                        &group_raw,
                        rels,
                        &mut slot_map,
                        &mut agg_specs,
                    )?);
                    hidden_idx
                } else {
                    let hidden_idx = proj_selects.len();
                    proj_selects.push(compile_join_expr(e, rels, &mut slot_map)?);
                    hidden_idx
                }
            }
        };
        order_specs.push((idx, ob.descending, ob.nulls_first));
    }

    // 4b. Detect Primary-Key Functional-Dependency (`PK-FD`) Group-By Key Compression + Top-K Late Materialization (`Q10`!):
    // When `GROUP BY` includes a unique non-null integer key `(det_ri, det_col)` (`customer.c_custkey`),
    // every other `GROUP BY` expression comes solely from `det_ri`'s functionally-dependent 1:1 subtree
    // (`customer -> nation`), `HAVING` is absent, and `ORDER BY ... LIMIT k` (`k <= 4096`) only sorts on
    // aggregates and/or `pk_gi`, strip all dependent payload columns during the join + aggregation scan
    // and late-materialize them ONLY for the winning `<= k` rows!
    let pk_fd_plan: Option<(usize, usize, String, usize)> = if is_agg_query
        && having_compiled.is_none()
        && compiled_groups.len() >= 2
        && !order_specs.is_empty()
        && let Some(keep_k) = query
            .limit
            .map(|lim| query.offset.unwrap_or(0).saturating_add(lim))
            .filter(|&k_lim| k_lim > 0 && k_lim <= 4096)
    {
        (0..group_raw.len()).find_map(|cand_gi| {
            let JoinExpr::Slot(_) = compiled_groups[cand_gi] else {
                return None;
            };
            let (det_ri, det_col) = resolve_rel_col(rels, &group_raw[cand_gi])?;
            if det_ri == probe_ri || !rels[det_ri].rel.col_is_unique_nonnull_int(&det_col) {
                return None;
            }
            if order_specs
                .iter()
                .any(|&(ci, _, _)| post_agg_uses_other_group(&post_selects[ci], cand_gi))
            {
                return None;
            }
            // Collect 1:1 functionally-dependent subtree rooted at `det_ri`
            let mut fd_subtree: HashSet<usize> = HashSet::new();
            let mut dq = std::collections::VecDeque::new();
            fd_subtree.insert(det_ri);
            dq.push_back(det_ri);
            while let Some(u) = dq.pop_front() {
                if let Some(ch_list) = tree_children.get(&u) {
                    for (v, _pcol, ccol) in ch_list {
                        if rels[*v].rel.col_is_unique_nonnull_int(ccol) && fd_subtree.insert(*v) {
                            dq.push_back(*v);
                        }
                    }
                }
            }
            for (gi, ge) in group_raw.iter().enumerate() {
                if gi == cand_gi {
                    continue;
                }
                let mut touched = HashSet::new();
                if !expr_referenced_rels(ge, rels, &mut touched)
                    || touched.is_empty()
                    || !touched.is_subset(&fd_subtree)
                {
                    return None;
                }
            }
            Some((cand_gi, det_ri, det_col, keep_k))
        })
    } else {
        None
    };

    // 5. Collect needed columns per relation from active slots (excluding deferred `PK-FD` columns!)
    let mut active_slots: HashSet<usize> = HashSet::new();
    if let Some((pk_gi, _, _, _)) = &pk_fd_plan {
        for rj in &compiled_residuals {
            rj.collect_slots(&mut active_slots);
        }
        compiled_groups[*pk_gi].collect_slots(&mut active_slots);
        for sp in &agg_specs {
            if let Some(ref je) = sp.arg {
                je.collect_slots(&mut active_slots);
            }
        }
    } else {
        for si in 0..slot_map.len() {
            active_slots.insert(si);
        }
    }
    let mut needed_cols_by_rel: Vec<HashSet<String>> = vec![HashSet::new(); k];
    for (si, (ri, cn)) in slot_map.iter().enumerate() {
        if active_slots.contains(&si) {
            needed_cols_by_rel[*ri].insert(cn.clone());
        }
    }

    // 6. Build all Snowflake branches (`root_attachments`) — build selective 1-key branches FIRST
    // so they push `KeyIn` semi-join filters into sibling multi-key branches (`Q09`: `part -> partsupp`)!
    root_attachments.sort_by_key(|(r_ri, attach)| {
        let priority = if !local_filters[*r_ri].is_empty() && attach.len() == 1 {
            0usize
        } else if attach.len() == 1 {
            1usize
        } else {
            2usize
        };
        (priority, rels[*r_ri].rel.total_rows())
    });

    let mut branches: Vec<BuiltBranch> = Vec::with_capacity(root_attachments.len());
    for idx_ra in 0..root_attachments.len() {
        let (root_ri, ref attach) = root_attachments[idx_ra];
        let b_nodes: Vec<usize> = (0..k)
            .filter(|&ri| branch_owner[ri] == Some(root_ri))
            .collect();
        let built = build_snowflake_branch(
            root_ri,
            attach,
            &b_nodes,
            &tree_children,
            rels,
            local_filters,
            &needed_cols_by_rel,
        )?;
        // Short-circuit if any inner-joined branch has 0 rows!
        if built.num_rows == 0 {
            if is_agg_query && group_raw.is_empty() && having_compiled.is_none() {
                let gvals: Vec<OwnedVal> = Vec::new();
                let avals: Vec<OwnedVal> = agg_specs
                    .iter()
                    .map(|s| AggState::new(s.op).finalize(s.op))
                    .collect();
                let row: Vec<serde_json::Value> = post_selects[..visible_cols]
                    .iter()
                    .map(|pe| pe.eval(&gvals, &avals).to_json())
                    .collect();
                return Some((out_names, vec![row]));
            }
            return Some((out_names, Vec::new()));
        }
        // Push selective 1-key branch keyset into any not-yet-built sibling branch sharing `probe_col` (`Q09`)
        if built.probe_key_cols.len() == 1
            && built.num_rows <= 2_000_000
            && let Some(ks) = built.key_set_1()
        {
            let pcol = &built.probe_key_cols[0];
            for &(other_root, ref other_attach) in &root_attachments[idx_ra + 1..] {
                if built.num_rows.saturating_mul(3) <= rels[other_root].rel.total_rows() {
                    for (other_pcol, other_rcol) in other_attach {
                        if other_pcol.eq_ignore_ascii_case(pcol) {
                            local_filters[other_root].push(LocalFilter::KeyIn {
                                col: other_rcol.clone(),
                                set: ks.clone(),
                                negated: false,
                            });
                        }
                    }
                }
            }
        }
        branches.push(built);
    }

    branches.sort_by_key(|b| b.num_rows);

    // Push single-key selective branches (`<= 4,000,000` rows and `<= probe_rows / 3`) into `probe_ri` local_filters!
    let probe_total = rels[probe_ri].rel.total_rows();
    for b in &branches {
        if b.probe_key_cols.len() == 1
            && b.num_rows <= 4_000_000
            && b.num_rows.saturating_mul(3) <= probe_total
            && let Some(ks) = b.key_set_1()
        {
            local_filters[probe_ri].push(LocalFilter::KeyIn {
                col: b.probe_key_cols[0].clone(),
                set: ks,
                negated: false,
            });
        }
    }

    // 7. Prepare `probe_ri` Stage-2 payload column list & fast slot lookup table!
    // Note: filter-only columns stay in `local_filters[probe_ri]` so `load_filtered_segment`
    // reads them in Stage 1 and never materializes unused payload columns on pruned segments.
    let mut probe_cols_set = needed_cols_by_rel[probe_ri].clone();
    for b in &branches {
        for pk in &b.probe_key_cols {
            if let Some(rc) = rels[probe_ri].rel.resolve_col_name(pk) {
                probe_cols_set.insert(rc);
            }
        }
    }
    if probe_cols_set.is_empty()
        && let Some(first_c) = rels[probe_ri].rel.col_order().first()
    {
        probe_cols_set.insert(first_c.clone());
    }
    let probe_col_vec: Vec<String> = probe_cols_set.into_iter().collect();
    let probe_col_refs: Vec<&str> = probe_col_vec.iter().map(String::as_str).collect();

    // Map each slot in `slot_map` to either `SlotSource::Probe(pci)` or `SlotSource::Branch(bi, ci)`!
    // Slots deferred by `pk_fd_plan` map to a dummy `SlotSource::Probe(0)` since they are never read during Stage 1.
    let mut slot_srcs: Vec<SlotSource> = Vec::with_capacity(slot_map.len());
    for (si, (s_ri, s_col)) in slot_map.iter().enumerate() {
        if !active_slots.contains(&si) {
            slot_srcs.push(SlotSource::Probe(0));
            continue;
        }
        if *s_ri == probe_ri {
            let pci = probe_col_vec
                .iter()
                .position(|c| c.eq_ignore_ascii_case(s_col))?;
            slot_srcs.push(SlotSource::Probe(pci));
        } else {
            let mut found = None;
            for (bi, b) in branches.iter().enumerate() {
                if let Some(ci) = b
                    .cols
                    .iter()
                    .position(|cc| cc.id.0 == *s_ri && cc.id.1.eq_ignore_ascii_case(s_col))
                {
                    found = Some(SlotSource::Branch(bi, ci));
                    break;
                }
            }
            slot_srcs.push(found?);
        }
    }

    // Map each branch's `probe_key_cols` to `probe_col_vec` indices
    let mut branch_probe_key_idxs: Vec<Vec<usize>> = Vec::with_capacity(branches.len());
    for b in &branches {
        let mut idxs = Vec::with_capacity(b.probe_key_cols.len());
        for pk in &b.probe_key_cols {
            idxs.push(
                probe_col_vec
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case(pk))?,
            );
        }
        branch_probe_key_idxs.push(idxs);
    }

    let n_probe_segs = rels[probe_ri].rel.num_segments();
    let n_slots = slot_map.len();

    // 8. Stream `probe_ri` segments in parallel via Rayon (with ZoneMap + 2-stage late materialization)!
    if let Some((pk_gi, det_ri, ref det_col, keep_k)) = pk_fd_plan {
        // Fast Path PK-FD (`Q10`): Aggregate solely by `pk_val: i64` across 64 radix shards,
        // select global Top `keep_k` (`<= 20`) winners, and late-materialize dependent `GROUP BY`
        // attributes (`c_name, c_acctbal, n_name, c_address, c_phone, c_comment`) ONLY for those `<= 20` keys!
        const FD_SHARDS: usize = 64;
        let JoinExpr::Slot(pk_slot) = compiled_groups[pk_gi] else {
            return None;
        };
        let mut stage1_slots_vec: Vec<usize> = active_slots.iter().copied().collect();
        stage1_slots_vec.sort_unstable();

        let any_fd_err = std::sync::atomic::AtomicBool::new(false);
        let mut worker_fd: Vec<Vec<FxMap<i64, Vec<AggState>>>> = (0..n_probe_segs)
            .into_par_iter()
            .fold(
                || {
                    (0..FD_SHARDS)
                        .map(|_| FxMap::<i64, Vec<AggState>>::default())
                        .collect::<Vec<_>>()
                },
                |mut fshards: Vec<FxMap<i64, Vec<AggState>>>, si| {
                    if any_fd_err.load(std::sync::atomic::Ordering::Relaxed) {
                        return fshards;
                    }
                    let Some(loaded) = rels[probe_ri].rel.load_filtered_segment(
                        si,
                        &probe_col_refs,
                        &local_filters[probe_ri],
                    ) else {
                        any_fd_err.store(true, std::sync::atomic::Ordering::Relaxed);
                        return fshards;
                    };
                    let Some((seg, sel)) = loaded else {
                        return fshards;
                    };
                    let tc = seg.tc();
                    let pcols: Vec<&Col> = probe_col_vec
                        .iter()
                        .filter_map(|c| tc.cols.get(c))
                        .collect();
                    if pcols.len() != probe_col_vec.len() {
                        any_fd_err.store(true, std::sync::atomic::Ordering::Relaxed);
                        return fshards;
                    }
                    let mut chosen_b = vec![0usize; branches.len()];
                    let mut tuple = vec![BVal::Null; n_slots];

                    for &r_u32 in &sel {
                        probe_row_recursive_sparse(
                            0,
                            r_u32 as usize,
                            &pcols,
                            &branches,
                            &branch_probe_key_idxs,
                            &slot_srcs,
                            &stage1_slots_vec,
                            &mut chosen_b,
                            &mut tuple,
                            &mut |_cb, tup| {
                                for res in &compiled_residuals {
                                    if !matches!(res.eval(tup), BVal::Bool(true)) {
                                        return;
                                    }
                                }
                                let BVal::Int(pk_val) = tup[pk_slot] else {
                                    return;
                                };
                                let sh = (((pk_val as u64).wrapping_mul(0x517cc1b727220a95)) >> 58)
                                    as usize;
                                if let Some(accs) = fshards[sh].get_mut(&pk_val) {
                                    for (ai, spec) in agg_specs.iter().enumerate() {
                                        let v = spec
                                            .arg
                                            .as_ref()
                                            .map_or(BVal::Int(1), |je| je.eval(tup));
                                        accs[ai].update(spec.op, v);
                                    }
                                } else {
                                    let mut accs: Vec<AggState> =
                                        agg_specs.iter().map(|s| AggState::new(s.op)).collect();
                                    for (ai, spec) in agg_specs.iter().enumerate() {
                                        let v = spec
                                            .arg
                                            .as_ref()
                                            .map_or(BVal::Int(1), |je| je.eval(tup));
                                        accs[ai].update(spec.op, v);
                                    }
                                    fshards[sh].insert(pk_val, accs);
                                }
                            },
                        );
                    }
                    fshards
                },
            )
            .collect();
        if any_fd_err.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        drop(branches);

        let mut fd_by_shard: Vec<Vec<FxMap<i64, Vec<AggState>>>> = (0..FD_SHARDS)
            .map(|_| Vec::with_capacity(worker_fd.len()))
            .collect();
        for wv in &mut worker_fd {
            for (sh, m) in wv.iter_mut().enumerate() {
                fd_by_shard[sh].push(std::mem::take(m));
            }
        }
        drop(worker_fd);

        let cmp_fd = |a: &(i64, Vec<OwnedVal>, Vec<OwnedVal>),
                      b: &(i64, Vec<OwnedVal>, Vec<OwnedVal>)|
         -> Ordering {
            for (oi, &(_, desc, nulls_first)) in order_specs.iter().enumerate() {
                let av = &a.2[oi];
                let bv = &b.2[oi];
                let a_null = matches!(av, OwnedVal::Null);
                let b_null = matches!(bv, OwnedVal::Null);
                if a_null || b_null {
                    if a_null && b_null {
                        continue;
                    }
                    let nf = nulls_first.unwrap_or(desc);
                    return if a_null {
                        if nf {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    } else if nf {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };
                }
                let ord = cmp_owned_vals(av, bv);
                if ord != Ordering::Equal {
                    return if desc { ord.reverse() } else { ord };
                }
            }
            a.0.cmp(&b.0)
        };

        let shard_top: Vec<Vec<(i64, Vec<OwnedVal>, Vec<OwnedVal>)>> = fd_by_shard
            .into_par_iter()
            .map(|mut parts| {
                let mut merged = parts.pop().unwrap_or_default();
                for pm in parts {
                    for (pk_val, saccs) in pm {
                        if let Some(dst) = merged.get_mut(&pk_val) {
                            for (ai, part) in saccs.into_iter().enumerate() {
                                dst[ai].merge(agg_specs[ai].op, part);
                            }
                        } else {
                            merged.insert(pk_val, saccs);
                        }
                    }
                }
                let mut tmp_gvals = vec![OwnedVal::Null; compiled_groups.len()];
                let mut cands: Vec<(i64, Vec<OwnedVal>, Vec<OwnedVal>)> =
                    Vec::with_capacity(merged.len());
                for (pk_val, st_vec) in merged {
                    tmp_gvals[pk_gi] = OwnedVal::Int(pk_val);
                    let avals: Vec<OwnedVal> = st_vec
                        .iter()
                        .enumerate()
                        .map(|(ai, st)| st.finalize(agg_specs[ai].op))
                        .collect();
                    let sort_keys: Vec<OwnedVal> = order_specs
                        .iter()
                        .map(|&(ci, _, _)| post_selects[ci].eval(&tmp_gvals, &avals))
                        .collect();
                    cands.push((pk_val, avals, sort_keys));
                }
                if cands.len() > keep_k {
                    cands.select_nth_unstable_by(keep_k - 1, cmp_fd);
                    cands.truncate(keep_k);
                }
                cands
            })
            .collect();

        let mut global_cands: Vec<(i64, Vec<OwnedVal>, Vec<OwnedVal>)> =
            shard_top.into_iter().flatten().collect();
        if global_cands.len() > keep_k {
            global_cands.select_nth_unstable_by(keep_k - 1, cmp_fd);
            global_cands.truncate(keep_k);
        }
        global_cands.sort_by(cmp_fd);

        let off = query.offset.unwrap_or(0);
        if off >= global_cands.len() {
            return Some((out_names, Vec::new()));
        }
        let end = query.limit.map_or(global_cands.len(), |l| {
            off.saturating_add(l).min(global_cands.len())
        });
        let winners = &global_cands[off..end];
        if winners.is_empty() {
            return Some((out_names, Vec::new()));
        }

        // Late-materialize `det_ri` subtree ONLY for the `<= keep_k` winning primary keys!
        let winner_keys: Vec<i64> = winners.iter().map(|(k_val, _, _)| *k_val).collect();
        let mut fd_filters: Vec<Vec<LocalFilter>> = vec![Vec::new(); k];
        fd_filters[det_ri].push(LocalFilter::KeyIn {
            col: det_col.clone(),
            set: IntKeySet::from_vec(winner_keys),
            negated: false,
        });
        let mut fd_slot_map: Vec<(usize, String)> = Vec::new();
        let mut fd_compiled_groups: Vec<JoinExpr> = Vec::with_capacity(group_raw.len());
        for ge in &group_raw {
            fd_compiled_groups.push(compile_join_expr(ge, rels, &mut fd_slot_map)?);
        }
        let mut fd_needed: Vec<HashSet<String>> = vec![HashSet::new(); k];
        for (ri, cn) in &fd_slot_map {
            fd_needed[*ri].insert(cn.clone());
        }
        let det_branch = build_snowflake_branch(
            det_ri,
            &[(det_col.clone(), det_col.clone())],
            &[],
            &tree_children,
            rels,
            &mut fd_filters,
            &fd_needed,
        )?;
        let mut fd_col_idxs: Vec<usize> = Vec::with_capacity(fd_slot_map.len());
        for (s_ri, s_col) in &fd_slot_map {
            let ci = det_branch
                .cols
                .iter()
                .position(|cc| cc.id.0 == *s_ri && cc.id.1.eq_ignore_ascii_case(s_col))?;
            fd_col_idxs.push(ci);
        }

        let mut fd_tuple = vec![BVal::Null; fd_slot_map.len()];
        let mut final_rows: Vec<Vec<serde_json::Value>> = Vec::with_capacity(winners.len());
        for (pk_val, avals, _) in winners {
            let d_row = det_branch.probe_single_1(*pk_val)? as usize;
            for (si, &ci) in fd_col_idxs.iter().enumerate() {
                fd_tuple[si] = read_col_bval(&det_branch.cols[ci].data, d_row);
            }
            let gvals: Vec<OwnedVal> = fd_compiled_groups
                .iter()
                .map(|ge| ge.eval(&fd_tuple).to_owned_val())
                .collect();
            let row: Vec<serde_json::Value> = post_selects[..visible_cols]
                .iter()
                .map(|pe| pe.eval(&gvals, avals).to_json())
                .collect();
            final_rows.push(row);
        }
        return Some((out_names, final_rows));
    }

    if is_agg_query {
        if compiled_groups.is_empty() {
            // Ungrouped aggregate (`Q06`, `Q14`, `Q17`, `Q19`)
            let seg_accs: Option<Vec<Vec<AggState>>> = (0..n_probe_segs)
                .into_par_iter()
                .map(|si| {
                    let Some((seg, sel)) = rels[probe_ri].rel.load_filtered_segment(
                        si,
                        &probe_col_refs,
                        &local_filters[probe_ri],
                    )?
                    else {
                        return Some(agg_specs.iter().map(|s| AggState::new(s.op)).collect());
                    };
                    let tc = seg.tc();
                    let pcols: Vec<&Col> = probe_col_vec
                        .iter()
                        .filter_map(|c| tc.cols.get(c))
                        .collect();
                    if pcols.len() != probe_col_vec.len() {
                        return None;
                    }
                    let mut accs: Vec<AggState> =
                        agg_specs.iter().map(|s| AggState::new(s.op)).collect();
                    let mut chosen_b = vec![0usize; branches.len()];
                    let mut tuple = vec![BVal::Null; n_slots];

                    for &r_u32 in &sel {
                        probe_row_recursive(
                            0,
                            r_u32 as usize,
                            &pcols,
                            &branches,
                            &branch_probe_key_idxs,
                            &slot_srcs,
                            &compiled_residuals,
                            &mut chosen_b,
                            &mut tuple,
                            &mut |tup| {
                                for (ai, spec) in agg_specs.iter().enumerate() {
                                    let v =
                                        spec.arg.as_ref().map_or(BVal::Int(1), |je| je.eval(tup));
                                    accs[ai].update(spec.op, v);
                                }
                            },
                        );
                    }
                    Some(accs)
                })
                .collect();

            let mut final_accs: Vec<AggState> =
                agg_specs.iter().map(|s| AggState::new(s.op)).collect();
            for sa in seg_accs? {
                for (ai, part) in sa.into_iter().enumerate() {
                    final_accs[ai].merge(agg_specs[ai].op, part);
                }
            }
            let gvals: Vec<OwnedVal> = Vec::new();
            let avals: Vec<OwnedVal> = final_accs
                .iter()
                .enumerate()
                .map(|(ai, st)| st.finalize(agg_specs[ai].op))
                .collect();
            if let Some(ref hc) = having_compiled
                && !matches!(hc.eval(&gvals, &avals), OwnedVal::Bool(true))
            {
                return Some((out_names, Vec::new()));
            }
            let row: Vec<serde_json::Value> = post_selects[..visible_cols]
                .iter()
                .map(|pe| pe.eval(&gvals, &avals).to_json())
                .collect();
            return Some((out_names, vec![row]));
        } else {
            // Fast Path A: 64-Shard Radix-Partitioned Single-`i64` `GROUP BY` (`Q18` `lineitem` 15M groups!)
            let single_int_group = branches.is_empty()
                && compiled_residuals.is_empty()
                && compiled_groups.len() == 1
                && agg_specs.len() <= 2
                && !agg_specs
                    .iter()
                    .any(|s| matches!(s.op, AggOp::CountDistinctInt | AggOp::Min | AggOp::Max))
                && matches!(compiled_groups[0], JoinExpr::Slot(gs) if matches!(slot_srcs[gs], SlotSource::Probe(pci) if rels[probe_ri].rel.col_is_int(&probe_col_vec[pci])));

            if single_int_group {
                const RADIX_SHARDS: usize = 64;
                let op0 = agg_specs.first().map_or(AggOp::CountStar, |s| s.op);
                let op1 = agg_specs.get(1).map_or(AggOp::CountStar, |s| s.op);

                // Ultra-lean single-agg worker-folded path (`Q18`: 60M rows / 15M groups -> ~3.5 GB instead of 15.6 GB!)
                if agg_specs.len() <= 1 {
                    let any_err = std::sync::atomic::AtomicBool::new(false);
                    let mut worker_shards: Vec<Vec<FxMap<Option<i64>, CompactNumAgg>>> = (0
                        ..n_probe_segs)
                        .into_par_iter()
                        .fold(
                            || {
                                (0..RADIX_SHARDS)
                                    .map(|_| FxMap::<Option<i64>, CompactNumAgg>::default())
                                    .collect::<Vec<_>>()
                            },
                            |mut shards: Vec<FxMap<Option<i64>, CompactNumAgg>>, si| {
                                if any_err.load(std::sync::atomic::Ordering::Relaxed) {
                                    return shards;
                                }
                                let Some(loaded) = rels[probe_ri].rel.load_filtered_segment(
                                    si,
                                    &probe_col_refs,
                                    &local_filters[probe_ri],
                                ) else {
                                    any_err.store(true, std::sync::atomic::Ordering::Relaxed);
                                    return shards;
                                };
                                let Some((seg, sel)) = loaded else {
                                    return shards;
                                };
                                let tc = seg.tc();
                                let pcols: Vec<&Col> = probe_col_vec
                                    .iter()
                                    .filter_map(|c| tc.cols.get(c))
                                    .collect();
                                if pcols.len() != probe_col_vec.len() {
                                    any_err.store(true, std::sync::atomic::Ordering::Relaxed);
                                    return shards;
                                }
                                let mut tuple = vec![BVal::Null; n_slots];
                                for &r_u32 in &sel {
                                    let r_idx = r_u32 as usize;
                                    for (si_slot, src) in slot_srcs.iter().enumerate() {
                                        if let SlotSource::Probe(pci) = *src {
                                            tuple[si_slot] = read_col_bval(pcols[pci], r_idx);
                                        }
                                    }
                                    let gk = match compiled_groups[0].eval(&tuple) {
                                        BVal::Int(k) => Some(k),
                                        BVal::Null => None,
                                        _ => {
                                            any_err
                                                .store(true, std::sync::atomic::Ordering::Relaxed);
                                            return shards;
                                        }
                                    };
                                    let sh = match gk {
                                        Some(k) => {
                                            (((k as u64).wrapping_mul(0x517cc1b727220a95)) >> 58)
                                                as usize
                                        }
                                        None => 0,
                                    };
                                    let v0 = agg_specs.first().map_or(BVal::Null, |sp| {
                                        sp.arg.as_ref().map_or(BVal::Int(1), |je| je.eval(&tuple))
                                    });
                                    let acc = shards[sh].entry(gk).or_default();
                                    if !agg_specs.is_empty() {
                                        acc.update(op0, v0);
                                    }
                                }
                                shards
                            },
                        )
                        .collect();
                    if any_err.load(std::sync::atomic::Ordering::Relaxed) {
                        return None;
                    }
                    // Transpose `[worker][shard]` -> `[shard][worker]` via `mem::take` so each shard drops worker maps immediately after merge!
                    let mut by_shard: Vec<Vec<FxMap<Option<i64>, CompactNumAgg>>> = (0
                        ..RADIX_SHARDS)
                        .map(|_| Vec::with_capacity(worker_shards.len()))
                        .collect();
                    for wvec in &mut worker_shards {
                        for (sh, m) in wvec.iter_mut().enumerate() {
                            by_shard[sh].push(std::mem::take(m));
                        }
                    }
                    drop(worker_shards);

                    let shard_rows: Vec<Vec<Vec<OwnedVal>>> = by_shard
                        .into_par_iter()
                        .map(|mut part_maps| {
                            let mut merged: FxMap<Option<i64>, CompactNumAgg> =
                                part_maps.pop().unwrap_or_default();
                            for pm in part_maps {
                                for (gk, part) in pm {
                                    merged.entry(gk).or_default().merge(part);
                                }
                            }
                            let mut out_sh: Vec<Vec<OwnedVal>> = Vec::new();
                            let mut gvals = vec![OwnedVal::Null; 1];
                            let mut avals = vec![OwnedVal::Null; agg_specs.len()];
                            for (gk, st0) in merged {
                                gvals[0] = gk.map_or(OwnedVal::Null, OwnedVal::Int);
                                if !agg_specs.is_empty() {
                                    avals[0] = st0.finalize(op0);
                                }
                                if let Some(ref hc) = having_compiled
                                    && !matches!(hc.eval(&gvals, &avals), OwnedVal::Bool(true))
                                {
                                    continue;
                                }
                                let row: Vec<OwnedVal> = post_selects
                                    .iter()
                                    .map(|pe| pe.eval(&gvals, &avals))
                                    .collect();
                                out_sh.push(row);
                            }
                            out_sh
                        })
                        .collect();

                    let total_out: usize = shard_rows.iter().map(Vec::len).sum();
                    let mut out_owned: Vec<Vec<OwnedVal>> = Vec::with_capacity(total_out);
                    for sr in shard_rows {
                        out_owned.extend(sr);
                    }
                    sort_and_slice_owned(&mut out_owned, &order_specs, query.offset, query.limit);
                    let final_rows: Vec<Vec<serde_json::Value>> = out_owned
                        .into_iter()
                        .map(|r| {
                            r.into_iter()
                                .take(visible_cols)
                                .map(|v| v.to_json())
                                .collect()
                        })
                        .collect();
                    return Some((out_names, final_rows));
                }

                let seg_shards: Option<Vec<Vec<FxMap<Option<i64>, [AggState; 2]>>>> = (0
                    ..n_probe_segs)
                    .into_par_iter()
                    .map(|si| {
                        let mut shards: Vec<FxMap<Option<i64>, [AggState; 2]>> =
                            (0..RADIX_SHARDS).map(|_| FxMap::default()).collect();
                        let Some((seg, sel)) = rels[probe_ri].rel.load_filtered_segment(
                            si,
                            &probe_col_refs,
                            &local_filters[probe_ri],
                        )?
                        else {
                            return Some(shards);
                        };
                        let tc = seg.tc();
                        let pcols: Vec<&Col> = probe_col_vec
                            .iter()
                            .filter_map(|c| tc.cols.get(c))
                            .collect();
                        if pcols.len() != probe_col_vec.len() {
                            return None;
                        }
                        let mut tuple = vec![BVal::Null; n_slots];
                        for &r_u32 in &sel {
                            let r_idx = r_u32 as usize;
                            for (si_slot, src) in slot_srcs.iter().enumerate() {
                                if let SlotSource::Probe(pci) = *src {
                                    tuple[si_slot] = read_col_bval(pcols[pci], r_idx);
                                }
                            }
                            let gk = match compiled_groups[0].eval(&tuple) {
                                BVal::Int(k) => Some(k),
                                BVal::Null => None,
                                _ => return None,
                            };
                            let sh = match gk {
                                Some(k) => {
                                    (((k as u64).wrapping_mul(0x517cc1b727220a95)) >> 58) as usize
                                }
                                None => 0,
                            };
                            let v0 = agg_specs.first().map_or(BVal::Null, |sp| {
                                sp.arg.as_ref().map_or(BVal::Int(1), |je| je.eval(&tuple))
                            });
                            let v1 = agg_specs.get(1).map_or(BVal::Null, |sp| {
                                sp.arg.as_ref().map_or(BVal::Int(1), |je| je.eval(&tuple))
                            });
                            let accs = shards[sh]
                                .entry(gk)
                                .or_insert_with(|| [AggState::new(op0), AggState::new(op1)]);
                            if !agg_specs.is_empty() {
                                accs[0].update(op0, v0);
                            }
                            if agg_specs.len() == 2 {
                                accs[1].update(op1, v1);
                            }
                        }
                        Some(shards)
                    })
                    .collect();

                let seg_shards = seg_shards?;
                let shard_rows: Vec<Vec<Vec<OwnedVal>>> = (0..RADIX_SHARDS)
                    .into_par_iter()
                    .map(|sh| {
                        let mut merged: FxMap<Option<i64>, [AggState; 2]> = FxMap::default();
                        for seg_vec in &seg_shards {
                            for (&gk, part) in &seg_vec[sh] {
                                if let Some(dst) = merged.get_mut(&gk) {
                                    if !agg_specs.is_empty() {
                                        dst[0].merge(op0, part[0].clone());
                                    }
                                    if agg_specs.len() == 2 {
                                        dst[1].merge(op1, part[1].clone());
                                    }
                                } else {
                                    merged.insert(gk, part.clone());
                                }
                            }
                        }
                        let mut out_sh: Vec<Vec<OwnedVal>> = Vec::new();
                        let mut gvals = vec![OwnedVal::Null; 1];
                        let mut avals = vec![OwnedVal::Null; agg_specs.len()];
                        for (gk, st_arr) in merged {
                            gvals[0] = gk.map_or(OwnedVal::Null, OwnedVal::Int);
                            for (ai, spec) in agg_specs.iter().enumerate() {
                                avals[ai] = st_arr[ai].finalize(spec.op);
                            }
                            if let Some(ref hc) = having_compiled
                                && !matches!(hc.eval(&gvals, &avals), OwnedVal::Bool(true))
                            {
                                continue;
                            }
                            let row: Vec<OwnedVal> = post_selects
                                .iter()
                                .map(|pe| pe.eval(&gvals, &avals))
                                .collect();
                            out_sh.push(row);
                        }
                        out_sh
                    })
                    .collect();

                let total_out: usize = shard_rows.iter().map(Vec::len).sum();
                let mut out_owned: Vec<Vec<OwnedVal>> = Vec::with_capacity(total_out);
                for sr in shard_rows {
                    out_owned.extend(sr);
                }
                sort_and_slice_owned(&mut out_owned, &order_specs, query.offset, query.limit);
                let final_rows: Vec<Vec<serde_json::Value>> = out_owned
                    .into_iter()
                    .map(|r| {
                        r.into_iter()
                            .take(visible_cols)
                            .map(|v| v.to_json())
                            .collect()
                    })
                    .collect();
                return Some((out_names, final_rows));
            }

            // Fast Path B: Branch-Row-Indexed `u32` Pre-Aggregation (`Q10`: 7 wide dimension columns!)
            // When `branches.len() == 1`, `compiled_residuals.is_empty()`, and all `GROUP BY` slots
            // originate in `Branch(0, _)`, pre-aggregate by `b_row: u32` during the `probe_ri` scan
            // so zero strings/dimension columns are touched per fact row!
            let mut grp_slots_set = HashSet::new();
            for ge in &compiled_groups {
                ge.collect_slots(&mut grp_slots_set);
            }
            let can_branch_preagg = branches.len() == 1
                && compiled_residuals.is_empty()
                && !grp_slots_set.is_empty()
                && grp_slots_set
                    .iter()
                    .all(|&si| matches!(slot_srcs[si], SlotSource::Branch(0, _)));

            if can_branch_preagg {
                const B_SHARDS: usize = 64;
                let mut agg_slots_set = HashSet::new();
                for sp in &agg_specs {
                    if let Some(ref je) = sp.arg {
                        je.collect_slots(&mut agg_slots_set);
                    }
                }
                let agg_slots: Vec<usize> = agg_slots_set.into_iter().collect();
                let grp_slots: Vec<usize> = grp_slots_set.into_iter().collect();

                // 1. Worker-folded 64-shard `b_row` pre-aggregation during `probe_ri` scan
                let any_b_err = std::sync::atomic::AtomicBool::new(false);
                let mut worker_bmaps: Vec<Vec<FxMap<u32, Vec<AggState>>>> = (0..n_probe_segs)
                    .into_par_iter()
                    .fold(
                        || {
                            (0..B_SHARDS)
                                .map(|_| FxMap::<u32, Vec<AggState>>::default())
                                .collect::<Vec<_>>()
                        },
                        |mut bshards: Vec<FxMap<u32, Vec<AggState>>>, si| {
                            if any_b_err.load(std::sync::atomic::Ordering::Relaxed) {
                                return bshards;
                            }
                            let Some(loaded) = rels[probe_ri].rel.load_filtered_segment(
                                si,
                                &probe_col_refs,
                                &local_filters[probe_ri],
                            ) else {
                                any_b_err.store(true, std::sync::atomic::Ordering::Relaxed);
                                return bshards;
                            };
                            let Some((seg, sel)) = loaded else {
                                return bshards;
                            };
                            let tc = seg.tc();
                            let pcols: Vec<&Col> = probe_col_vec
                                .iter()
                                .filter_map(|c| tc.cols.get(c))
                                .collect();
                            if pcols.len() != probe_col_vec.len() {
                                any_b_err.store(true, std::sync::atomic::Ordering::Relaxed);
                                return bshards;
                            }
                            let mut chosen_b = [0usize; 1];
                            let mut tuple = vec![BVal::Null; n_slots];

                            for &r_u32 in &sel {
                                let r_idx = r_u32 as usize;
                                probe_row_recursive_sparse(
                                    0,
                                    r_idx,
                                    &pcols,
                                    &branches,
                                    &branch_probe_key_idxs,
                                    &slot_srcs,
                                    &agg_slots,
                                    &mut chosen_b,
                                    &mut tuple,
                                    &mut |cb, tup| {
                                        let b_row = cb[0] as u32;
                                        let sh = (((b_row as u64)
                                            .wrapping_mul(0x9E37_79B9_7F4A_7C15))
                                            >> 58)
                                            as usize;
                                        if let Some(accs) = bshards[sh].get_mut(&b_row) {
                                            for (ai, spec) in agg_specs.iter().enumerate() {
                                                let v = spec
                                                    .arg
                                                    .as_ref()
                                                    .map_or(BVal::Int(1), |je| je.eval(tup));
                                                accs[ai].update(spec.op, v);
                                            }
                                        } else {
                                            let mut accs: Vec<AggState> = agg_specs
                                                .iter()
                                                .map(|s| AggState::new(s.op))
                                                .collect();
                                            for (ai, spec) in agg_specs.iter().enumerate() {
                                                let v = spec
                                                    .arg
                                                    .as_ref()
                                                    .map_or(BVal::Int(1), |je| je.eval(tup));
                                                accs[ai].update(spec.op, v);
                                            }
                                            bshards[sh].insert(b_row, accs);
                                        }
                                    },
                                );
                            }
                            bshards
                        },
                    )
                    .collect();
                if any_b_err.load(std::sync::atomic::Ordering::Relaxed) {
                    return None;
                }

                // 2. Parallel 64-shard `b_row` merge + re-partition by `compiled_groups[0]` hash!
                let mut b_by_shard: Vec<Vec<FxMap<u32, Vec<AggState>>>> = (0..B_SHARDS)
                    .map(|_| Vec::with_capacity(worker_bmaps.len()))
                    .collect();
                for wv in &mut worker_bmaps {
                    for (sh, m) in wv.iter_mut().enumerate() {
                        b_by_shard[sh].push(std::mem::take(m));
                    }
                }
                drop(worker_bmaps);

                let g_repart: Vec<Vec<Vec<(u32, Vec<AggState>)>>> = b_by_shard
                    .into_par_iter()
                    .map(|mut parts| {
                        use std::hash::{Hash, Hasher};
                        let mut merged_b: FxMap<u32, Vec<AggState>> =
                            parts.pop().unwrap_or_default();
                        for pm in parts {
                            for (b_row, saccs) in pm {
                                if let Some(dst) = merged_b.get_mut(&b_row) {
                                    for (ai, part) in saccs.into_iter().enumerate() {
                                        dst[ai].merge(agg_specs[ai].op, part);
                                    }
                                } else {
                                    merged_b.insert(b_row, saccs);
                                }
                            }
                        }
                        let mut buckets: Vec<Vec<(u32, Vec<AggState>)>> =
                            (0..B_SHARDS).map(|_| Vec::new()).collect();
                        let mut tuple = vec![BVal::Null; n_slots];
                        for (b_row, baccs) in merged_b {
                            let br = b_row as usize;
                            for &si_slot in &grp_slots {
                                if let SlotSource::Branch(0, ci) = slot_srcs[si_slot] {
                                    tuple[si_slot] = read_col_bval(&branches[0].cols[ci].data, br);
                                }
                            }
                            let g0 = BorrowKeyElem::from_bval(compiled_groups[0].eval(&tuple));
                            let mut hasher = rustc_hash::FxHasher::default();
                            g0.hash(&mut hasher);
                            let gsh = ((hasher.finish().wrapping_mul(0x517c_c1b7_2722_0a95)) >> 58)
                                as usize;
                            buckets[gsh].push((b_row, baccs));
                        }
                        buckets
                    })
                    .collect();

                let mut g_by_shard: Vec<Vec<(u32, Vec<AggState>)>> =
                    (0..B_SHARDS).map(|_| Vec::new()).collect();
                for mut bvecs in g_repart {
                    for (gsh, chunk) in bvecs.iter_mut().enumerate() {
                        g_by_shard[gsh].append(chunk);
                    }
                }

                // 3. Parallel 64-shard zero-String-allocation grouping (`BorrowKeyElem<'b>`)
                //    + per-shard Top-K selection BEFORE `Arc<str>` allocation when `ORDER BY` is on aggregates (`Q10`)!
                let order_is_agg_only = !order_specs.is_empty()
                    && order_specs
                        .iter()
                        .all(|&(col_idx, _, _)| !post_agg_uses_any_group(&post_selects[col_idx]));
                let shard_top_k: Option<usize> = query
                    .limit
                    .map(|lim| query.offset.unwrap_or(0).saturating_add(lim))
                    .filter(|&k_lim| k_lim > 0 && k_lim <= 4096 && !order_specs.is_empty());

                let shard_outputs: Vec<Vec<Vec<OwnedVal>>> = g_by_shard
                    .into_par_iter()
                    .map(|items| {
                        let mut gmap: FxMap<Vec<BorrowKeyElem<'_>>, (u32, Vec<AggState>)> =
                            FxMap::with_capacity_and_hasher(items.len(), Default::default());
                        let mut tuple = vec![BVal::Null; n_slots];
                        let mut key_buf: Vec<BorrowKeyElem<'_>> =
                            Vec::with_capacity(compiled_groups.len());

                        for (b_row, baccs) in items {
                            let br = b_row as usize;
                            for &si_slot in &grp_slots {
                                if let SlotSource::Branch(0, ci) = slot_srcs[si_slot] {
                                    tuple[si_slot] = read_col_bval(&branches[0].cols[ci].data, br);
                                }
                            }
                            key_buf.clear();
                            for ge in &compiled_groups {
                                key_buf.push(BorrowKeyElem::from_bval(ge.eval(&tuple)));
                            }
                            if let Some((_, dst)) = gmap.get_mut(&key_buf) {
                                for (ai, part) in baccs.into_iter().enumerate() {
                                    dst[ai].merge(agg_specs[ai].op, part);
                                }
                            } else {
                                gmap.insert(key_buf.clone(), (b_row, baccs));
                            }
                        }

                        if order_is_agg_only
                            && having_compiled.is_none()
                            && let Some(keep_k) = shard_top_k
                            && gmap.len() > keep_k
                        {
                            // Deferred String Materialization (`Q10`): evaluate ONLY aggregate sort columns,
                            // select top `keep_k` per shard, and allocate `Arc<str>` ONLY for those `keep_k` winners!
                            let dummy_gvals = vec![OwnedVal::Null; compiled_groups.len()];
                            let mut cands: Vec<(
                                Vec<BorrowKeyElem<'_>>,
                                Vec<OwnedVal>,
                                Vec<OwnedVal>,
                            )> = Vec::with_capacity(gmap.len());
                            for (bk, (_, st_vec)) in gmap {
                                let avals: Vec<OwnedVal> = st_vec
                                    .iter()
                                    .enumerate()
                                    .map(|(ai, st)| st.finalize(agg_specs[ai].op))
                                    .collect();
                                let sort_keys: Vec<OwnedVal> = order_specs
                                    .iter()
                                    .map(|&(ci, _, _)| post_selects[ci].eval(&dummy_gvals, &avals))
                                    .collect();
                                cands.push((bk, avals, sort_keys));
                            }
                            let cmp_cand =
                                |a: &(Vec<BorrowKeyElem<'_>>, Vec<OwnedVal>, Vec<OwnedVal>),
                                 b: &(Vec<BorrowKeyElem<'_>>, Vec<OwnedVal>, Vec<OwnedVal>)|
                                 -> Ordering {
                                    for (oi, &(_, desc, nulls_first)) in
                                        order_specs.iter().enumerate()
                                    {
                                        let av = &a.2[oi];
                                        let bv = &b.2[oi];
                                        let a_null = matches!(av, OwnedVal::Null);
                                        let b_null = matches!(bv, OwnedVal::Null);
                                        if a_null || b_null {
                                            if a_null && b_null {
                                                continue;
                                            }
                                            let nf = nulls_first.unwrap_or(desc);
                                            return if a_null {
                                                if nf {
                                                    Ordering::Less
                                                } else {
                                                    Ordering::Greater
                                                }
                                            } else if nf {
                                                Ordering::Greater
                                            } else {
                                                Ordering::Less
                                            };
                                        }
                                        let ord = cmp_owned_vals(av, bv);
                                        if ord != Ordering::Equal {
                                            return if desc { ord.reverse() } else { ord };
                                        }
                                    }
                                    Ordering::Equal
                                };
                            cands.select_nth_unstable_by(keep_k - 1, cmp_cand);
                            cands.truncate(keep_k);
                            let mut out_sh = Vec::with_capacity(cands.len());
                            for (bk, avals, _) in cands {
                                let gvals: Vec<OwnedVal> =
                                    bk.into_iter().map(BorrowKeyElem::to_owned_val).collect();
                                let row: Vec<OwnedVal> = post_selects
                                    .iter()
                                    .map(|pe| pe.eval(&gvals, &avals))
                                    .collect();
                                out_sh.push(row);
                            }
                            return out_sh;
                        }

                        let mut out_sh = Vec::with_capacity(gmap.len());
                        for (bk, (_, st_vec)) in gmap {
                            let gvals: Vec<OwnedVal> =
                                bk.into_iter().map(BorrowKeyElem::to_owned_val).collect();
                            let avals: Vec<OwnedVal> = st_vec
                                .iter()
                                .enumerate()
                                .map(|(ai, st)| st.finalize(agg_specs[ai].op))
                                .collect();
                            if let Some(ref hc) = having_compiled
                                && !matches!(hc.eval(&gvals, &avals), OwnedVal::Bool(true))
                            {
                                continue;
                            }
                            let row: Vec<OwnedVal> = post_selects
                                .iter()
                                .map(|pe| pe.eval(&gvals, &avals))
                                .collect();
                            out_sh.push(row);
                        }
                        if let Some(keep_k) = shard_top_k
                            && out_sh.len() > keep_k
                        {
                            sort_and_slice_owned(&mut out_sh, &order_specs, None, Some(keep_k));
                        }
                        out_sh
                    })
                    .collect();

                let total_out: usize = shard_outputs.iter().map(Vec::len).sum();
                let mut out_owned: Vec<Vec<OwnedVal>> = Vec::with_capacity(total_out);
                for sh_vec in shard_outputs {
                    out_owned.extend(sh_vec);
                }
                sort_and_slice_owned(&mut out_owned, &order_specs, query.offset, query.limit);
                let final_rows: Vec<Vec<serde_json::Value>> = out_owned
                    .into_iter()
                    .map(|r| {
                        r.into_iter()
                            .take(visible_cols)
                            .map(|v| v.to_json())
                            .collect()
                    })
                    .collect();
                return Some((out_names, final_rows));
            }

            let merged: FxMap<Vec<GKeyElem>, Vec<AggState>> = {
                // General grouped aggregate (`Q03, Q04, Q05, Q07, Q08, Q09, Q11, Q12, Q16, Q21, Q22`)
                let seg_maps: Option<Vec<FxMap<Vec<GKeyElem>, Vec<AggState>>>> = (0..n_probe_segs)
                    .into_par_iter()
                    .map(|si| {
                        let Some((seg, sel)) = rels[probe_ri].rel.load_filtered_segment(
                            si,
                            &probe_col_refs,
                            &local_filters[probe_ri],
                        )?
                        else {
                            return Some(FxMap::default());
                        };
                        let tc = seg.tc();
                        let pcols: Vec<&Col> = probe_col_vec
                            .iter()
                            .filter_map(|c| tc.cols.get(c))
                            .collect();
                        if pcols.len() != probe_col_vec.len() {
                            return None;
                        }
                        let mut gmap: FxMap<Vec<GKeyElem>, Vec<AggState>> = FxMap::default();
                        let mut chosen_b = vec![0usize; branches.len()];
                        let mut tuple = vec![BVal::Null; n_slots];
                        let mut key_buf: Vec<GKeyElem> = Vec::with_capacity(compiled_groups.len());

                        for &r_u32 in &sel {
                            probe_row_recursive(
                                0,
                                r_u32 as usize,
                                &pcols,
                                &branches,
                                &branch_probe_key_idxs,
                                &slot_srcs,
                                &compiled_residuals,
                                &mut chosen_b,
                                &mut tuple,
                                &mut |tup| {
                                    key_buf.clear();
                                    for ge in &compiled_groups {
                                        key_buf.push(GKeyElem::from_bval(ge.eval(tup)));
                                    }
                                    if let Some(accs) = gmap.get_mut(&key_buf) {
                                        for (ai, spec) in agg_specs.iter().enumerate() {
                                            let v = spec
                                                .arg
                                                .as_ref()
                                                .map_or(BVal::Int(1), |je| je.eval(tup));
                                            accs[ai].update(spec.op, v);
                                        }
                                    } else {
                                        let mut accs: Vec<AggState> =
                                            agg_specs.iter().map(|s| AggState::new(s.op)).collect();
                                        for (ai, spec) in agg_specs.iter().enumerate() {
                                            let v = spec
                                                .arg
                                                .as_ref()
                                                .map_or(BVal::Int(1), |je| je.eval(tup));
                                            accs[ai].update(spec.op, v);
                                        }
                                        gmap.insert(key_buf.clone(), accs);
                                    }
                                },
                            );
                        }
                        Some(gmap)
                    })
                    .collect();

                let mut gm: FxMap<Vec<GKeyElem>, Vec<AggState>> = FxMap::default();
                for sm in seg_maps? {
                    for (gk, saccs) in sm {
                        if let Some(dst) = gm.get_mut(&gk) {
                            for (ai, part) in saccs.into_iter().enumerate() {
                                dst[ai].merge(agg_specs[ai].op, part);
                            }
                        } else {
                            gm.insert(gk, saccs);
                        }
                    }
                }
                gm
            };

            let mut out_owned: Vec<Vec<OwnedVal>> = Vec::with_capacity(merged.len());
            for (gk, st_vec) in merged {
                let gvals: Vec<OwnedVal> = gk.iter().map(GKeyElem::to_owned_val).collect();
                let avals: Vec<OwnedVal> = st_vec
                    .iter()
                    .enumerate()
                    .map(|(ai, st)| st.finalize(agg_specs[ai].op))
                    .collect();
                if let Some(ref hc) = having_compiled
                    && !matches!(hc.eval(&gvals, &avals), OwnedVal::Bool(true))
                {
                    continue;
                }
                let row: Vec<OwnedVal> = post_selects
                    .iter()
                    .map(|pe| pe.eval(&gvals, &avals))
                    .collect();
                out_owned.push(row);
            }

            sort_and_slice_owned(&mut out_owned, &order_specs, query.offset, query.limit);
            let final_rows: Vec<Vec<serde_json::Value>> = out_owned
                .into_iter()
                .map(|r| {
                    r.into_iter()
                        .take(visible_cols)
                        .map(|v| v.to_json())
                        .collect()
                })
                .collect();
            return Some((out_names, final_rows));
        }
    } else {
        // Non-aggregated projection + `ORDER BY` + `LIMIT` (`Q02`, `Q15`, `Q20`)
        let seg_rows: Option<Vec<Vec<Vec<OwnedVal>>>> = (0..n_probe_segs)
            .into_par_iter()
            .map(|si| {
                let Some((seg, sel)) = rels[probe_ri].rel.load_filtered_segment(
                    si,
                    &probe_col_refs,
                    &local_filters[probe_ri],
                )?
                else {
                    return Some(Vec::new());
                };
                let tc = seg.tc();
                let pcols: Vec<&Col> = probe_col_vec
                    .iter()
                    .filter_map(|c| tc.cols.get(c))
                    .collect();
                if pcols.len() != probe_col_vec.len() {
                    return None;
                }
                let mut rows_out: Vec<Vec<OwnedVal>> = Vec::new();
                let mut chosen_b = vec![0usize; branches.len()];
                let mut tuple = vec![BVal::Null; n_slots];

                for &r_u32 in &sel {
                    probe_row_recursive(
                        0,
                        r_u32 as usize,
                        &pcols,
                        &branches,
                        &branch_probe_key_idxs,
                        &slot_srcs,
                        &compiled_residuals,
                        &mut chosen_b,
                        &mut tuple,
                        &mut |tup| {
                            let prow: Vec<OwnedVal> = proj_selects
                                .iter()
                                .map(|pe| pe.eval(tup).to_owned_val())
                                .collect();
                            rows_out.push(prow);
                        },
                    );
                }
                Some(rows_out)
            })
            .collect();

        let mut all_owned: Vec<Vec<OwnedVal>> = Vec::new();
        for sr in seg_rows? {
            all_owned.extend(sr);
        }
        sort_and_slice_owned(&mut all_owned, &order_specs, query.offset, query.limit);
        let final_rows: Vec<Vec<serde_json::Value>> = all_owned
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .take(visible_cols)
                    .map(|v| v.to_json())
                    .collect()
            })
            .collect();
        return Some((out_names, final_rows));
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn probe_row_recursive_sparse<'a, F>(
    bi: usize,
    probe_row: usize,
    pcols: &[&'a Col],
    branches: &'a [BuiltBranch],
    branch_probe_key_idxs: &[Vec<usize>],
    slot_srcs: &[impl ProbeSlotSrc],
    needed_slots: &[usize],
    chosen_b: &mut [usize],
    tuple: &mut [BVal<'a>],
    emit: &mut F,
) where
    F: FnMut(&[usize], &[BVal<'a>]),
{
    if bi == branches.len() {
        for &si in needed_slots {
            tuple[si] = slot_srcs[si].read_val(probe_row, pcols, branches, chosen_b);
        }
        emit(chosen_b, tuple);
        return;
    }

    let b = &branches[bi];
    let kidxs = &branch_probe_key_idxs[bi];
    if kidxs.len() == 1 {
        let BVal::Int(k) = read_col_bval(pcols[kidxs[0]], probe_row) else {
            return;
        };
        if let Some(b_row) = b.probe_single_1(k) {
            chosen_b[bi] = b_row as usize;
            probe_row_recursive_sparse(
                bi + 1,
                probe_row,
                pcols,
                branches,
                branch_probe_key_idxs,
                slot_srcs,
                needed_slots,
                chosen_b,
                tuple,
                emit,
            );
        } else if let BranchLookup::HashMulti1 { map, row_ids } = &b.lookup
            && let Some(&(st, ln)) = map.get(&k)
        {
            for &ri_u32 in &row_ids[st as usize..(st + ln) as usize] {
                chosen_b[bi] = ri_u32 as usize;
                probe_row_recursive_sparse(
                    bi + 1,
                    probe_row,
                    pcols,
                    branches,
                    branch_probe_key_idxs,
                    slot_srcs,
                    needed_slots,
                    chosen_b,
                    tuple,
                    emit,
                );
            }
        }
    } else if kidxs.len() == 2 {
        let (BVal::Int(k1), BVal::Int(k2)) = (
            read_col_bval(pcols[kidxs[0]], probe_row),
            read_col_bval(pcols[kidxs[1]], probe_row),
        ) else {
            return;
        };
        match &b.lookup {
            BranchLookup::HashUnique2(m) => {
                if let Some(&b_row) = m.get(&(k1, k2)) {
                    chosen_b[bi] = b_row as usize;
                    probe_row_recursive_sparse(
                        bi + 1,
                        probe_row,
                        pcols,
                        branches,
                        branch_probe_key_idxs,
                        slot_srcs,
                        needed_slots,
                        chosen_b,
                        tuple,
                        emit,
                    );
                }
            }
            BranchLookup::HashMulti2 { map, row_ids } => {
                if let Some(&(st, ln)) = map.get(&(k1, k2)) {
                    for &ri_u32 in &row_ids[st as usize..(st + ln) as usize] {
                        chosen_b[bi] = ri_u32 as usize;
                        probe_row_recursive_sparse(
                            bi + 1,
                            probe_row,
                            pcols,
                            branches,
                            branch_probe_key_idxs,
                            slot_srcs,
                            needed_slots,
                            chosen_b,
                            tuple,
                            emit,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn probe_row_recursive<'a, F>(
    bi: usize,
    probe_row: usize,
    pcols: &[&'a Col],
    branches: &'a [BuiltBranch],
    branch_probe_key_idxs: &[Vec<usize>],
    slot_srcs: &[impl ProbeSlotSrc],
    residuals: &[JoinExpr],
    chosen_b: &mut [usize],
    tuple: &mut [BVal<'a>],
    emit: &mut F,
) where
    F: FnMut(&[BVal<'a>]),
{
    if bi == branches.len() {
        for (si, src) in slot_srcs.iter().enumerate() {
            tuple[si] = src.read_val(probe_row, pcols, branches, chosen_b);
        }
        for res in residuals {
            if !matches!(res.eval(tuple), BVal::Bool(true)) {
                return;
            }
        }
        emit(tuple);
        return;
    }

    let b = &branches[bi];
    let kidxs = &branch_probe_key_idxs[bi];
    if kidxs.len() == 1 {
        let BVal::Int(k) = read_col_bval(pcols[kidxs[0]], probe_row) else {
            return;
        };
        if let Some(b_row) = b.probe_single_1(k) {
            chosen_b[bi] = b_row as usize;
            probe_row_recursive(
                bi + 1,
                probe_row,
                pcols,
                branches,
                branch_probe_key_idxs,
                slot_srcs,
                residuals,
                chosen_b,
                tuple,
                emit,
            );
        } else if let BranchLookup::HashMulti1 { map, row_ids } = &b.lookup
            && let Some(&(st, ln)) = map.get(&k)
        {
            for &ri_u32 in &row_ids[st as usize..(st + ln) as usize] {
                chosen_b[bi] = ri_u32 as usize;
                probe_row_recursive(
                    bi + 1,
                    probe_row,
                    pcols,
                    branches,
                    branch_probe_key_idxs,
                    slot_srcs,
                    residuals,
                    chosen_b,
                    tuple,
                    emit,
                );
            }
        }
    } else if kidxs.len() == 2 {
        let (BVal::Int(k1), BVal::Int(k2)) = (
            read_col_bval(pcols[kidxs[0]], probe_row),
            read_col_bval(pcols[kidxs[1]], probe_row),
        ) else {
            return;
        };
        match &b.lookup {
            BranchLookup::HashUnique2(m) => {
                if let Some(&b_row) = m.get(&(k1, k2)) {
                    chosen_b[bi] = b_row as usize;
                    probe_row_recursive(
                        bi + 1,
                        probe_row,
                        pcols,
                        branches,
                        branch_probe_key_idxs,
                        slot_srcs,
                        residuals,
                        chosen_b,
                        tuple,
                        emit,
                    );
                }
            }
            BranchLookup::HashMulti2 { map, row_ids } => {
                if let Some(&(st, ln)) = map.get(&(k1, k2)) {
                    for &ri_u32 in &row_ids[st as usize..(st + ln) as usize] {
                        chosen_b[bi] = ri_u32 as usize;
                        probe_row_recursive(
                            bi + 1,
                            probe_row,
                            pcols,
                            branches,
                            branch_probe_key_idxs,
                            slot_srcs,
                            residuals,
                            chosen_b,
                            tuple,
                            emit,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

trait ProbeSlotSrc {
    fn read_val<'a>(
        &self,
        probe_row: usize,
        pcols: &[&'a Col],
        branches: &'a [BuiltBranch],
        chosen_b: &[usize],
    ) -> BVal<'a>;
}

// Implement `ProbeSlotSrc` on tuple/enum inside `exec_bushy_plan` via a module-level enum
#[derive(Clone, Copy)]
enum SlotSource {
    Probe(usize),
    Branch(usize, usize),
}

impl ProbeSlotSrc for SlotSource {
    #[inline]
    fn read_val<'a>(
        &self,
        probe_row: usize,
        pcols: &[&'a Col],
        branches: &'a [BuiltBranch],
        chosen_b: &[usize],
    ) -> BVal<'a> {
        match *self {
            SlotSource::Probe(pci) => read_col_bval(pcols[pci], probe_row),
            SlotSource::Branch(bi, ci) => read_col_bval(&branches[bi].cols[ci].data, chosen_b[bi]),
        }
    }
}

fn sort_and_slice_owned(
    rows: &mut Vec<Vec<OwnedVal>>,
    order_specs: &[(usize, bool, Option<bool>)],
    offset: Option<usize>,
    limit: Option<usize>,
) {
    if !order_specs.is_empty() {
        rows.sort_by(|a, b| {
            for &(col_idx, desc, nulls_first) in order_specs {
                let av = &a[col_idx];
                let bv = &b[col_idx];
                let a_null = matches!(av, OwnedVal::Null);
                let b_null = matches!(bv, OwnedVal::Null);
                if a_null || b_null {
                    if a_null && b_null {
                        continue;
                    }
                    let nf = nulls_first.unwrap_or(desc);
                    return if a_null {
                        if nf {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    } else if nf {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };
                }
                let ord = cmp_owned_vals(av, bv);
                if ord != Ordering::Equal {
                    return if desc { ord.reverse() } else { ord };
                }
            }
            Ordering::Equal
        });
    }
    let off = offset.unwrap_or(0);
    if off > 0 {
        if off >= rows.len() {
            rows.clear();
            return;
        }
        rows.drain(0..off);
    }
    if let Some(lim) = limit
        && rows.len() > lim
    {
        rows.truncate(lim);
    }
}
