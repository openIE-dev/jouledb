//! Vectorized single-table analytics over a cached columnar snapshot.
//!
//! The general executor turns every amorphic record into a `RowData`, then a
//! JSON row and a per-row context map before evaluating WHERE/GROUP BY, which
//! costs a few microseconds per row (perf: malloc, String clones, HashMap
//! clones). For the common analytic shape:
//!
//! `SELECT <group cols | aggregates> FROM t [WHERE conjunction of col <op> literal
//! / col [NOT] [I]LIKE pat] [GROUP BY cols] [ORDER BY outputs] [LIMIT/OFFSET]`
//!
//! this module reads typed column vectors (built once per table, then reused
//! until the store's generation changes), filters to a selection vector, and
//! aggregates in tight loops. Any other shape returns `None` and runs on the
//! general executor (a real fallback, same results).

use joule_db_query::ast::{Expression, Operator, UnaryOperator, Value as AstValue};
use joule_db_query::sql::SqlQuery;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Col {
    /// Values and per-row NULL flags; an empty flag vector means no NULLs.
    Int(Vec<i64>, Vec<bool>),
    Float(Vec<f64>, Vec<bool>),
    /// Dictionary-encoded text: per-row code (u32::MAX = NULL) and the dictionary
    Text(Vec<u32>, Vec<Arc<str>>),
    /// Mixed or non-scalar values: queries touching it fall back
    Other,
}

/// Typed columns of one table, in record id order.
#[derive(Debug, Clone)]
pub struct TableColumns {
    pub len: usize,
    pub cols: HashMap<String, Col>,
    /// Reverse dictionaries for text columns, built on first append
    text_index: HashMap<String, FxMap<Arc<str>, u32>>,
    /// Zone maps for numeric columns: min/max/NULL count per ZONE rows.
    /// Kept current by build() and append(); used to answer MIN/MAX/COUNT
    /// without a scan and to skip or short-circuit morsels on filters.
    zones: HashMap<String, Vec<Zone>>,
}

/// Rows per zone-map entry (divides MORSEL, so morsels align to zones).
pub const ZONE: usize = 8_192;

#[derive(Debug, Clone, Copy)]
struct Zone {
    /// None when every row in the zone is NULL
    lo: Option<Num>,
    hi: Option<Num>,
    nulls: u32,
}

fn zone_of(c: &Col, s: usize, e: usize) -> Option<Zone> {
    let mut z = Zone { lo: None, hi: None, nulls: 0 };
    match c {
        Col::Int(v, n) => {
            let (mut lo, mut hi, mut any) = (i64::MAX, i64::MIN, false);
            for i in s..e {
                if nl(n, i) {
                    z.nulls += 1;
                } else {
                    let x = v[i];
                    lo = lo.min(x);
                    hi = hi.max(x);
                    any = true;
                }
            }
            if any {
                z.lo = Some(Num::I(lo));
                z.hi = Some(Num::I(hi));
            }
        }
        Col::Float(v, n) => {
            let (mut lo, mut hi, mut any) = (f64::INFINITY, f64::NEG_INFINITY, false);
            for i in s..e {
                if nl(n, i) {
                    z.nulls += 1;
                } else {
                    let x = v[i];
                    if x.is_nan() {
                        return None; // NaN ordering: no zone, scan instead
                    }
                    lo = lo.min(x);
                    hi = hi.max(x);
                    any = true;
                }
            }
            if any {
                z.lo = Some(Num::F(lo));
                z.hi = Some(Num::F(hi));
            }
        }
        _ => return None,
    }
    Some(z)
}

impl TableColumns {
    /// Recompute zone maps from the zone containing row `from` onward.
    fn refresh_zones(&mut self, from: usize) {
        use rayon::prelude::*;
        let n = self.len;
        let first = from / ZONE;
        let cols = &self.cols;
        let mut zones = std::mem::take(&mut self.zones);
        let updated: Vec<(String, Option<Vec<Zone>>)> = cols
            .par_iter()
            .filter(|(_, c)| matches!(c, Col::Int(..) | Col::Float(..)))
            .map(|(name, c)| {
                let mut z: Vec<Zone> = zones.get(name).cloned().unwrap_or_default();
                if z.len() < first {
                    // missing history: rebuild this column from the start
                    z.clear();
                }
                let start = z.len().min(first);
                z.truncate(start);
                for zi in start..n.div_ceil(ZONE) {
                    match zone_of(c, zi * ZONE, ((zi + 1) * ZONE).min(n)) {
                        Some(x) => z.push(x),
                        None => return (name.clone(), None),
                    }
                }
                (name.clone(), Some(z))
            })
            .collect();
        zones.clear();
        for (name, z) in updated {
            if let Some(z) = z {
                zones.insert(name, z);
            }
        }
        self.zones = zones;
    }
}

impl TableColumns {
    pub fn build(columns: &[String], records: &[&joule_db_amorphic::AmorphicRecord]) -> Self {
        use joule_db_amorphic::Value as V;
        let n = records.len();
        let mut cols = HashMap::new();
        for c in columns {
            let (mut ints, mut floats, mut texts, mut other) = (0, 0, 0, false);
            for r in records {
                match r.get(c) {
                    None | Some(V::Null) => {}
                    Some(V::Int(_)) => ints += 1,
                    Some(V::Float(_)) => floats += 1,
                    Some(V::String(_)) => texts += 1,
                    Some(_) => other = true,
                }
            }
            let col = if other || (texts > 0 && ints + floats > 0) {
                Col::Other
            } else if texts > 0 {
                let mut codes_of: HashMap<&str, u32> = HashMap::new();
                let mut dict: Vec<Arc<str>> = Vec::new();
                let codes = records
                    .iter()
                    .map(|r| match r.get(c) {
                        Some(V::String(s)) => *codes_of.entry(s.as_str()).or_insert_with(|| {
                            dict.push(Arc::from(s.as_str()));
                            (dict.len() - 1) as u32
                        }),
                        _ => NULL_CODE,
                    })
                    .collect();
                Col::Text(codes, dict)
            } else if floats > 0 {
                let mut v = Vec::with_capacity(n);
                let mut nulls = Vec::with_capacity(n);
                for r in records {
                    match r.get(c) {
                        Some(V::Float(f)) => (v.push(*f), nulls.push(false)),
                        Some(V::Int(i)) => (v.push(*i as f64), nulls.push(false)),
                        _ => (v.push(0.0), nulls.push(true)),
                    };
                }
                if !nulls.iter().any(|&b| b) {
                    nulls = Vec::new();
                }
                Col::Float(v, nulls)
            } else {
                // all-int or all-null
                let mut v = Vec::with_capacity(n);
                let mut nulls = Vec::with_capacity(n);
                for r in records {
                    match r.get(c) {
                        Some(V::Int(i)) => (v.push(*i), nulls.push(false)),
                        _ => (v.push(0), nulls.push(true)),
                    };
                }
                if !nulls.iter().any(|&b| b) {
                    nulls = Vec::new();
                }
                Col::Int(v, nulls)
            };
            cols.insert(c.clone(), col);
        }
        let mut t = Self {
            len: n,
            cols,
            text_index: HashMap::new(),
            zones: HashMap::new(),
        };
        t.refresh_zones(0);
        t
    }
}

impl TableColumns {
    /// Append newly committed records (in record id order, after every
    /// record already here). Returns false when a value does not fit the
    /// column's current type; the caller then drops this snapshot and the
    /// next query rebuilds it from the store.
    pub fn append(&mut self, records: &[&joule_db_amorphic::AmorphicRecord]) -> bool {
        use rayon::prelude::*;
        let old = self.len;
        let mut idxs = std::mem::take(&mut self.text_index);
        // one task per column; each text column carries its own reverse dictionary
        let mut work: Vec<(&String, &mut Col, Option<FxMap<Arc<str>, u32>>)> = self
            .cols
            .iter_mut()
            .map(|(name, col)| {
                let idx = match col {
                    Col::Text(_, dict) => Some(idxs.remove(name).unwrap_or_else(|| {
                        dict.iter()
                            .enumerate()
                            .map(|(i, d)| (d.clone(), i as u32))
                            .collect()
                    })),
                    _ => None,
                };
                (name, col, idx)
            })
            .collect();
        let ok = work
            .par_iter_mut()
            .map(|(name, col, idx)| append_col(name, col, idx.as_mut(), old, records))
            .collect::<Vec<bool>>()
            .into_iter()
            .all(|b| b);
        for (name, _, idx) in work {
            if let Some(i) = idx {
                idxs.insert(name.clone(), i);
            }
        }
        self.text_index = idxs;
        self.len = old + records.len();
        if ok {
            self.refresh_zones(old);
        }
        ok
    }
}

fn append_col(
    name: &str,
    col: &mut Col,
    idx: Option<&mut FxMap<Arc<str>, u32>>,
    old: usize,
    records: &[&joule_db_amorphic::AmorphicRecord],
) -> bool {
    use joule_db_amorphic::Value as V;
    match col {
        Col::Int(v, nulls) => {
            v.reserve(records.len());
            for (k, r) in records.iter().enumerate() {
                match r.get(name) {
                    Some(V::Int(i)) => v.push(*i),
                    None | Some(V::Null) => {
                        if nulls.is_empty() {
                            *nulls = vec![false; old + k];
                        }
                        v.push(0);
                        nulls.push(true);
                        continue;
                    }
                    _ => return false,
                }
                if !nulls.is_empty() {
                    nulls.push(false);
                }
            }
        }
        Col::Float(v, nulls) => {
            v.reserve(records.len());
            for (k, r) in records.iter().enumerate() {
                match r.get(name) {
                    Some(V::Float(f)) => v.push(*f),
                    Some(V::Int(i)) => v.push(*i as f64),
                    None | Some(V::Null) => {
                        if nulls.is_empty() {
                            *nulls = vec![false; old + k];
                        }
                        v.push(0.0);
                        nulls.push(true);
                        continue;
                    }
                    _ => return false,
                }
                if !nulls.is_empty() {
                    nulls.push(false);
                }
            }
        }
        Col::Text(codes, dict) => {
            let Some(idx) = idx else { return false };
            codes.reserve(records.len());
            for r in records {
                match r.get(name) {
                    Some(V::String(s)) => {
                        let code = match idx.get(s.as_str()) {
                            Some(&c) => c,
                            None => {
                                let a: Arc<str> = Arc::from(s.as_str());
                                dict.push(a.clone());
                                let c = (dict.len() - 1) as u32;
                                idx.insert(a, c);
                                c
                            }
                        };
                        codes.push(code);
                    }
                    None | Some(V::Null) => codes.push(NULL_CODE),
                    _ => return false,
                }
            }
        }
        Col::Other => {}
    }
    true
}

const NULL_CODE: u32 = u32::MAX;

/// NULL flag of row `i` (empty flags = no NULLs).
#[inline(always)]
fn nl(n: &[bool], i: usize) -> bool {
    !n.is_empty() && n[i]
}

type FxMap<K, V> = rustc_hash::FxHashMap<K, V>;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Num {
    I(i64),
    F(f64),
}

enum Pred<'a> {
    Cmp {
        col: &'a str,
        op: Operator,
        lit: Lit,
    },
    Like {
        col: &'a str,
        pattern: &'a str,
        negated: bool,
        ci: bool,
    },
}

#[derive(Clone)]
enum Lit {
    N(Num),
    S(String),
}

fn literal(e: &Expression) -> Option<Lit> {
    match e {
        Expression::Literal(AstValue::Int(i)) => Some(Lit::N(Num::I(*i))),
        Expression::Literal(AstValue::Float(f)) => Some(Lit::N(Num::F(*f))),
        Expression::Literal(AstValue::String(s)) => Some(Lit::S(s.clone())),
        Expression::Unary {
            op: UnaryOperator::Neg,
            expr,
        } => match literal(expr)? {
            Lit::N(Num::I(i)) => Some(Lit::N(Num::I(-i))),
            Lit::N(Num::F(f)) => Some(Lit::N(Num::F(-f))),
            Lit::S(_) => None,
        },
        _ => None,
    }
}

fn column(e: &Expression) -> Option<&str> {
    match e {
        Expression::Column(c) => Some(c.as_str()),
        Expression::QualifiedColumn { column, .. } => Some(column.as_str()),
        _ => None,
    }
}

fn flip(op: &Operator) -> Option<Operator> {
    Some(match op {
        Operator::Eq => Operator::Eq,
        Operator::Ne => Operator::Ne,
        Operator::Lt => Operator::Gt,
        Operator::Le => Operator::Ge,
        Operator::Gt => Operator::Lt,
        Operator::Ge => Operator::Le,
        _ => return None,
    })
}

fn collect_preds<'a>(e: &'a Expression, out: &mut Vec<Pred<'a>>) -> Option<()> {
    match e {
        Expression::Binary {
            left,
            op: Operator::And,
            right,
        } => {
            collect_preds(left, out)?;
            collect_preds(right, out)
        }
        Expression::Binary { left, op, right } => {
            if let (Some(c), Some(l)) = (column(left), literal(right)) {
                flip(op)?;
                out.push(Pred::Cmp {
                    col: c,
                    op: op.clone(),
                    lit: l,
                });
            } else if let (Some(l), Some(c)) = (literal(left), column(right)) {
                out.push(Pred::Cmp {
                    col: c,
                    op: flip(op)?,
                    lit: l,
                });
            } else {
                return None;
            }
            Some(())
        }
        Expression::Like {
            expr,
            pattern,
            negated,
            case_insensitive,
        } => {
            out.push(Pred::Like {
                col: column(expr)?,
                pattern,
                negated: *negated,
                ci: *case_insensitive,
            });
            Some(())
        }
        _ => None,
    }
}

fn cmp_ok(o: Ordering, op: &Operator) -> bool {
    match op {
        Operator::Eq => o == Ordering::Equal,
        Operator::Ne => o != Ordering::Equal,
        Operator::Lt => o == Ordering::Less,
        Operator::Le => o != Ordering::Greater,
        Operator::Gt => o == Ordering::Greater,
        Operator::Ge => o != Ordering::Less,
        _ => false,
    }
}

/// Apply predicates; returns the selected row indices.
fn filter(t: &TableColumns, preds: &[Pred]) -> Option<Vec<u32>> {
    let mut sel: Vec<u32> = (0..t.len as u32).collect();
    for p in preds {
        match p {
            Pred::Cmp { col, op, lit } => match (t.cols.get(*col)?, lit) {
                (Col::Int(v, nulls), Lit::N(Num::I(x))) => {
                    sel.retain(|&i| !nl(nulls, i as usize) && cmp_ok(v[i as usize].cmp(x), op))
                }
                (Col::Int(v, nulls), Lit::N(Num::F(x))) => sel.retain(|&i| {
                    !nl(nulls, i as usize)
                        && (v[i as usize] as f64)
                            .partial_cmp(x)
                            .is_some_and(|o| cmp_ok(o, op))
                }),
                (Col::Float(v, nulls), Lit::N(n)) => {
                    let x = match n {
                        Num::I(i) => *i as f64,
                        Num::F(f) => *f,
                    };
                    sel.retain(|&i| {
                        !nl(nulls, i as usize)
                            && v[i as usize].partial_cmp(&x).is_some_and(|o| cmp_ok(o, op))
                    })
                }
                (Col::Text(codes, dict), Lit::S(x)) => {
                    let ok: Vec<bool> = dict
                        .iter()
                        .map(|d| cmp_ok((**d).cmp(x.as_str()), op))
                        .collect();
                    sel.retain(|&i| {
                        let c = codes[i as usize];
                        c != NULL_CODE && ok[c as usize]
                    })
                }
                _ => return None,
            },
            Pred::Like {
                col,
                pattern,
                negated,
                ci,
            } => match t.cols.get(*col)? {
                Col::Text(codes, dict) => {
                    // evaluate once per dictionary entry
                    let ok: Vec<bool> = dict
                        .iter()
                        .map(|d| joule_db_query::like::like(d, pattern, *ci) != *negated)
                        .collect();
                    sel.retain(|&i| {
                        let c = codes[i as usize];
                        c != NULL_CODE && ok[c as usize]
                    })
                }
                _ => return None,
            },
        }
    }
    Some(sel)
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Null,
    I(i64),
    F(u64),
    S(Arc<str>),
}

fn key_at(c: &Col, i: usize) -> Key {
    match c {
        Col::Int(v, n) => {
            if nl(n, i) {
                Key::Null
            } else {
                Key::I(v[i])
            }
        }
        Col::Float(v, n) => {
            if nl(n, i) {
                Key::Null
            } else {
                Key::F(v[i].to_bits())
            }
        }
        Col::Text(codes, dict) => match codes[i] {
            NULL_CODE => Key::Null,
            c => Key::S(dict[c as usize].clone()),
        },
        Col::Other => Key::Null,
    }
}

fn key_json(k: &Key) -> serde_json::Value {
    match k {
        Key::Null => serde_json::Value::Null,
        Key::I(i) => serde_json::json!(i),
        Key::F(b) => serde_json::json!(f64::from_bits(*b)),
        Key::S(s) => serde_json::Value::String(s.to_string()),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum AggKind {
    CountStar,
    Count,
    CountDistinct,
    Sum,
    Avg,
    Min,
    Max,
}

enum OutCol<'a> {
    Group(usize),
    Agg(AggKind, Option<&'a str>),
}

#[derive(Clone, Default)]
struct Acc {
    count: u64,
    isum: i128,
    fsum: f64,
    fcomp: f64,
    min: Option<Num>,
    max: Option<Num>,
    distinct: Option<Box<HashSet<u64>>>,
    /// Distinct bitmap (small-range ints, one group)
    bits: Vec<u64>,
}

fn num_at(c: &Col, i: usize) -> Option<Num> {
    match c {
        Col::Int(v, n) => (!nl(n, i)).then(|| Num::I(v[i])),
        Col::Float(v, n) => (!nl(n, i)).then(|| Num::F(v[i])),
        _ => None,
    }
}

fn num_lt(a: Num, b: Num) -> bool {
    match (a, b) {
        (Num::I(x), Num::I(y)) => x < y,
        (Num::F(x), Num::F(y)) => x < y,
        (Num::I(x), Num::F(y)) => (x as f64) < y,
        (Num::F(x), Num::I(y)) => x < y as f64,
    }
}

fn num_json(n: Option<Num>) -> serde_json::Value {
    match n {
        None => serde_json::Value::Null,
        Some(Num::I(i)) => serde_json::json!(i),
        Some(Num::F(f)) => serde_json::json!(f),
    }
}

fn cmp_json(a: &serde_json::Value, b: &serde_json::Value) -> Ordering {
    use serde_json::Value as J;
    match (a, b) {
        (J::Null, J::Null) => Ordering::Equal,
        // NULLs sort last ascending (Postgres default)
        (J::Null, _) => Ordering::Greater,
        (_, J::Null) => Ordering::Less,
        (J::Number(x), J::Number(y)) => match (x.as_i64(), y.as_i64()) {
            (Some(p), Some(q)) => p.cmp(&q),
            _ => x
                .as_f64()
                .unwrap_or(0.0)
                .partial_cmp(&y.as_f64().unwrap_or(0.0))
                .unwrap_or(Ordering::Equal),
        },
        (J::String(x), J::String(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

fn is_null(c: &Col, i: usize) -> bool {
    match c {
        Col::Int(_, n) | Col::Float(_, n) => nl(n, i),
        Col::Text(codes, _) => codes[i] == NULL_CODE,
        Col::Other => true,
    }
}

/// Hashable distinct key (text by dictionary code, which is unique per string)
fn distinct_key(c: &Col, i: usize) -> Option<u64> {
    match c {
        Col::Int(v, n) => (!nl(n, i)).then(|| v[i] as u64),
        Col::Float(v, n) => (!nl(n, i)).then(|| v[i].to_bits()),
        Col::Text(codes, _) => (codes[i] != NULL_CODE).then(|| codes[i] as u64),
        Col::Other => None,
    }
}

/// Code of row `i` within one column (u64; NULL gets its own code).
fn code_of(c: &Col, i: usize) -> u64 {
    match c {
        Col::Int(v, n) => {
            if nl(n, i) {
                u64::MAX
            } else {
                v[i] as u64
            }
        }
        Col::Float(v, n) => {
            if nl(n, i) {
                u64::MAX
            } else {
                v[i].to_bits()
            }
        }
        Col::Text(codes, _) => codes[i] as u64,
        Col::Other => u64::MAX,
    }
}

fn is_null_code(c: &Col, code: u64, i: usize) -> bool {
    match c {
        Col::Text(..) => code == NULL_CODE as u64,
        _ => code == u64::MAX && is_null(c, i),
    }
}

/// Dense group id per selected row (first-appearance order) and each group's key.
fn group_ids(gcols: &[&Col], sel: &[u32]) -> (Vec<u32>, Vec<Vec<Key>>) {
    if gcols.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut ids = Vec::with_capacity(sel.len());
    let mut keys: Vec<Vec<Key>> = Vec::new();
    if let [c] = gcols {
        // single column: dense array for dictionary text, hash map otherwise
        if let Col::Text(codes, dict) = c {
            let mut map = vec![u32::MAX; dict.len() + 1];
            for &ri in sel {
                let code = codes[ri as usize];
                let slot = if code == NULL_CODE {
                    dict.len()
                } else {
                    code as usize
                };
                if map[slot] == u32::MAX {
                    map[slot] = keys.len() as u32;
                    keys.push(vec![key_at(c, ri as usize)]);
                }
                ids.push(map[slot]);
            }
            return (ids, keys);
        }
        let mut map: FxMap<(u64, bool), u32> = FxMap::default();
        for &ri in sel {
            let ri = ri as usize;
            let code = code_of(c, ri);
            let k = (code, is_null_code(c, code, ri));
            let next = keys.len() as u32;
            let id = *map.entry(k).or_insert_with(|| {
                keys.push(vec![key_at(c, ri)]);
                next
            });
            ids.push(id);
        }
        return (ids, keys);
    }
    let mut map: FxMap<Vec<(u64, bool)>, u32> = FxMap::default();
    let mut buf = Vec::with_capacity(gcols.len());
    for &ri in sel {
        let ri = ri as usize;
        buf.clear();
        for c in gcols {
            let code = code_of(c, ri);
            buf.push((code, is_null_code(c, code, ri)));
        }
        let id = match map.get(&buf) {
            Some(&id) => id,
            None => {
                let id = keys.len() as u32;
                keys.push(gcols.iter().map(|c| key_at(c, ri)).collect());
                map.insert(buf.clone(), id);
                id
            }
        };
        ids.push(id);
    }
    (ids, keys)
}

/// Rows per morsel: work unit for parallel scans (DuckDB uses 122,880).
const MORSEL: usize = 122_880;

/// How rows map to groups.
enum Grouping<'a> {
    /// No GROUP BY: one group
    One,
    /// Single dictionary-text column: slot = code (NULL = dict.len())
    Text(&'a [u32], usize),
    /// Single int column with a small value range: slot = v - base (NULL = range)
    IntRange(&'a [i64], &'a [bool], i64, usize),
    /// Anything else: hash of per-column codes
    Hash,
}

impl Grouping<'_> {
    fn slots(&self) -> usize {
        match self {
            Grouping::One => 1,
            Grouping::Text(_, n) => n + 1,
            Grouping::IntRange(_, _, _, n) => n + 1,
            Grouping::Hash => 0,
        }
    }
}

/// Per-morsel (then merged) aggregation state.
struct Part {
    /// first row index per group (u32::MAX = group unseen); dense or hash-indexed
    first: Vec<u32>,
    accs: Vec<Acc>,
    /// Hash grouping only: key -> group index
    map: FxMap<Vec<(u64, bool)>, u32>,
}

fn neumaier(sum: &mut f64, comp: &mut f64, x: f64) {
    let t = *sum + x;
    if sum.abs() >= x.abs() {
        *comp += (*sum - t) + x;
    } else {
        *comp += (x - t) + *sum;
    }
    *sum = t;
}

impl Acc {
    fn merge(&mut self, o: &Acc) {
        self.count += o.count;
        self.isum += o.isum;
        neumaier(&mut self.fsum, &mut self.fcomp, o.fsum);
        neumaier(&mut self.fsum, &mut self.fcomp, o.fcomp);
        if let Some(m) = o.min {
            if self.min.is_none_or(|x| num_lt(m, x)) {
                self.min = Some(m);
            }
        }
        if let Some(m) = o.max {
            if self.max.is_none_or(|x| num_lt(x, m)) {
                self.max = Some(m);
            }
        }
        if let (Some(d), Some(od)) = (self.distinct.as_mut(), o.distinct.as_ref()) {
            d.extend(od.iter().copied());
        }
        if !o.bits.is_empty() {
            if self.bits.is_empty() {
                self.bits = o.bits.clone();
            } else {
                for (a, b) in self.bits.iter_mut().zip(&o.bits) {
                    *a |= *b;
                }
            }
        }
    }
}

fn num_cmp(a: Num, b: Num) -> Option<Ordering> {
    match (a, b) {
        (Num::I(x), Num::I(y)) => Some(x.cmp(&y)),
        (Num::I(x), Num::F(y)) => (x as f64).partial_cmp(&y),
        (Num::F(x), Num::I(y)) => x.partial_cmp(&(y as f64)),
        (Num::F(x), Num::F(y)) => x.partial_cmp(&y),
    }
}

/// Zone verdict for `col op lit` over rows `[s, e)`: Some(true) every row
/// passes, Some(false) no row passes, None mixed or unknown.
fn zone_verdict(t: &TableColumns, col: &str, op: &Operator, x: Num, s: usize, e: usize) -> Option<bool> {
    let zs = t.zones.get(col)?;
    if s % ZONE != 0 || (e % ZONE != 0 && e != t.len) {
        return None; // morsel not zone-aligned
    }
    let (mut lo, mut hi, mut nulls): (Option<Num>, Option<Num>, usize) = (None, None, 0);
    for zi in s / ZONE..e.div_ceil(ZONE) {
        let z = zs.get(zi)?;
        nulls += z.nulls as usize;
        if let Some(l) = z.lo {
            lo = Some(lo.map_or(l, |c| if num_lt(l, c) { l } else { c }));
        }
        if let Some(h) = z.hi {
            hi = Some(hi.map_or(h, |c| if num_lt(c, h) { h } else { c }));
        }
    }
    let (Some(lo), Some(hi)) = (lo, hi) else {
        return Some(false); // all NULL: comparisons never pass
    };
    let (cl, ch) = (num_cmp(lo, x)?, num_cmp(hi, x)?);
    use Ordering::*;
    let (all, none) = match op {
        Operator::Lt => (ch == Less, cl != Less),
        Operator::Le => (ch != Greater, cl == Greater),
        Operator::Gt => (cl == Greater, ch != Greater),
        Operator::Ge => (cl != Less, ch == Less),
        Operator::Eq => (cl == Equal && ch == Equal, cl == Greater || ch == Less),
        Operator::Ne => (cl == Greater || ch == Less, cl == Equal && ch == Equal),
        _ => return None,
    };
    if none {
        Some(false)
    } else if all && nulls == 0 {
        Some(true)
    } else {
        None
    }
}

/// AND one predicate into `mask` for rows `[s, s + mask.len())`.
fn mask_pred(t: &TableColumns, p: &Pred, s: usize, mask: &mut [bool]) -> Option<()> {
    let e = s + mask.len();
    match p {
        Pred::Cmp { col, op, lit } => match (t.cols.get(*col)?, lit) {
            (Col::Int(v, nulls), Lit::N(Num::I(x))) => {
                let x = *x;
                let vs = &v[s..e];
                // tight per-operator loops (vectorizable)
                macro_rules! go {
                    ($f:expr) => {
                        for (m, &y) in mask.iter_mut().zip(vs) {
                            *m &= $f(y);
                        }
                    };
                }
                match op {
                    Operator::Eq => go!(|y: i64| y == x),
                    Operator::Ne => go!(|y: i64| y != x),
                    Operator::Lt => go!(|y: i64| y < x),
                    Operator::Le => go!(|y: i64| y <= x),
                    Operator::Gt => go!(|y: i64| y > x),
                    Operator::Ge => go!(|y: i64| y >= x),
                    _ => return None,
                }
                if !nulls.is_empty() {
                    for (m, &n) in mask.iter_mut().zip(&nulls[s..e]) {
                        *m &= !n;
                    }
                }
            }
            (Col::Int(v, nulls), Lit::N(Num::F(x))) => {
                for (k, m) in mask.iter_mut().enumerate() {
                    let i = s + k;
                    *m &= !nl(nulls, i) && (v[i] as f64).partial_cmp(x).is_some_and(|o| cmp_ok(o, op));
                }
            }
            (Col::Float(v, nulls), Lit::N(n)) => {
                let x = match n {
                    Num::I(i) => *i as f64,
                    Num::F(f) => *f,
                };
                for (k, m) in mask.iter_mut().enumerate() {
                    let i = s + k;
                    *m &= !nl(nulls, i) && v[i].partial_cmp(&x).is_some_and(|o| cmp_ok(o, op));
                }
            }
            (Col::Text(codes, dict), Lit::S(x)) => {
                if dict.len() > 4096 {
                    return None; // per-zone dictionary pass too costly; scan path
                }
                let ok: Vec<bool> = dict.iter().map(|d| cmp_ok((**d).cmp(x.as_str()), op)).collect();
                for (m, &c) in mask.iter_mut().zip(&codes[s..e]) {
                    *m &= c != NULL_CODE && ok[c as usize];
                }
            }
            _ => return None,
        },
        Pred::Like { col, pattern, negated, ci } => match t.cols.get(*col)? {
            Col::Text(codes, dict) => {
                let ok = like_table(dict, pattern, *negated, *ci);
                for (m, &c) in mask.iter_mut().zip(&codes[s..e]) {
                    *m &= c != NULL_CODE && ok[c as usize];
                }
            }
            _ => return None,
        },
    }
    Some(())
}

/// COUNT(*) under a conjunction of predicates: zone-pruned, parallel over
/// zones, with a reusable stack mask per zone (no selection vectors).
fn count_where(t: &TableColumns, preds: &[Pred]) -> Option<u64> {
    use rayon::prelude::*;
    let nz = t.len.div_ceil(ZONE);
    (0..nz)
        .into_par_iter()
        .map(|zi| -> Option<u64> {
            let (s, e) = (zi * ZONE, ((zi + 1) * ZONE).min(t.len));
            let mut live: Vec<&Pred> = Vec::with_capacity(preds.len());
            for p in preds {
                if let Pred::Cmp { col, op, lit: Lit::N(x) } = p {
                    match zone_verdict(t, col, op, *x, s, e) {
                        Some(false) => return Some(0),
                        Some(true) => continue,
                        None => {}
                    }
                }
                live.push(p);
            }
            if live.is_empty() {
                return Some((e - s) as u64);
            }
            let mut buf = [true; ZONE];
            let mask = &mut buf[..e - s];
            for p in live {
                mask_pred(t, p, s, mask)?;
            }
            Some(mask.iter().map(|&b| b as u64).sum())
        })
        .sum::<Option<u64>>()
}

/// COUNT(*) / COUNT(col) / MIN / MAX over a whole table from the row count
/// and zone maps, no scan. The column cache is a committed snapshot (the
/// vectorized path never runs inside a transaction), so this is the same
/// answer a scan of that snapshot gives.
fn metadata_aggregate(t: &TableColumns, outs: &[OutCol]) -> Option<Vec<Acc>> {
    let mut accs = Vec::with_capacity(outs.len());
    for o in outs {
        let OutCol::Agg(kind, arg) = o else { return None };
        let mut a = Acc::default();
        match (kind, arg) {
            (AggKind::CountStar, _) => a.count = t.len as u64,
            (AggKind::Count | AggKind::Min | AggKind::Max, Some(c)) => {
                let zs = t.zones.get(*c)?;
                let nulls: usize = zs.iter().map(|z| z.nulls as usize).sum();
                a.count = (t.len - nulls) as u64;
                if matches!(kind, AggKind::Min | AggKind::Max) {
                    for z in zs {
                        if let Some(l) = z.lo {
                            if a.min.is_none_or(|m| num_lt(l, m)) {
                                a.min = Some(l);
                            }
                        }
                        if let Some(h) = z.hi {
                            if a.max.is_none_or(|m| num_lt(m, h)) {
                                a.max = Some(h);
                            }
                        }
                    }
                }
            }
            _ => return None,
        }
        accs.push(a);
    }
    Some(accs)
}

/// Selected absolute row indices of `[s, e)` (None = all rows).
fn morsel_sel(t: &TableColumns, preds: &[Pred], s: usize, e: usize) -> Option<Option<Vec<u32>>> {
    if preds.is_empty() {
        return Some(None);
    }
    // zone maps: drop predicates every row passes, skip the morsel if any
    // predicate passes no row
    let mut live: Vec<&Pred> = Vec::with_capacity(preds.len());
    for p in preds {
        if let Pred::Cmp { col, op, lit: Lit::N(x) } = p {
            match zone_verdict(t, col, op, *x, s, e) {
                Some(false) => return Some(Some(Vec::new())),
                Some(true) => continue,
                None => {}
            }
        }
        live.push(p);
    }
    if live.is_empty() {
        return Some(None);
    }
    let mut sel: Vec<u32> = (s as u32..e as u32).collect();
    for p in live {
        match p {
            Pred::Cmp { col, op, lit } => match (t.cols.get(*col)?, lit) {
                (Col::Int(v, nulls), Lit::N(Num::I(x))) => {
                    sel.retain(|&i| !nl(nulls, i as usize) && cmp_ok(v[i as usize].cmp(x), op))
                }
                (Col::Int(v, nulls), Lit::N(Num::F(x))) => sel.retain(|&i| {
                    !nl(nulls, i as usize)
                        && (v[i as usize] as f64)
                            .partial_cmp(x)
                            .is_some_and(|o| cmp_ok(o, op))
                }),
                (Col::Float(v, nulls), Lit::N(n)) => {
                    let x = match n {
                        Num::I(i) => *i as f64,
                        Num::F(f) => *f,
                    };
                    sel.retain(|&i| {
                        !nl(nulls, i as usize)
                            && v[i as usize].partial_cmp(&x).is_some_and(|o| cmp_ok(o, op))
                    })
                }
                (Col::Text(codes, dict), Lit::S(x)) => {
                    let ok: Vec<bool> = dict
                        .iter()
                        .map(|d| cmp_ok((**d).cmp(x.as_str()), op))
                        .collect();
                    sel.retain(|&i| {
                        let c = codes[i as usize];
                        c != NULL_CODE && ok[c as usize]
                    })
                }
                _ => return None,
            },
            Pred::Like {
                col,
                pattern,
                negated,
                ci,
            } => match t.cols.get(*col)? {
                Col::Text(codes, dict) => {
                    let ok = like_table(dict, pattern, *negated, *ci);
                    sel.retain(|&i| {
                        let c = codes[i as usize];
                        c != NULL_CODE && ok[c as usize]
                    })
                }
                _ => return None,
            },
        }
    }
    Some(Some(sel))
}

/// LIKE result per dictionary entry, cached per (dictionary, pattern) for
/// the duration of a call through a thread-local (dictionaries are shared
/// across morsels).
fn like_table(dict: &[Arc<str>], pattern: &str, negated: bool, ci: bool) -> Arc<Vec<bool>> {
    thread_local! {
        static CACHE: std::cell::RefCell<Option<(usize, usize, String, bool, bool, Arc<Vec<bool>>)>> =
            const { std::cell::RefCell::new(None) };
    }
    let key = (dict.as_ptr() as usize, dict.len());
    if let Some(hit) = CACHE.with(|c| {
        c.borrow().as_ref().and_then(|(p, n, pat, ng, cc, v)| {
            (*p == key.0 && *n == key.1 && pat == pattern && *ng == negated && *cc == ci)
                .then(|| v.clone())
        })
    }) {
        return hit;
    }
    let v: Arc<Vec<bool>> = Arc::new(
        dict.iter()
            .map(|d| joule_db_query::like::like(d, pattern, ci) != negated)
            .collect(),
    );
    CACHE.with(|c| {
        *c.borrow_mut() = Some((key.0, key.1, pattern.to_string(), negated, ci, v.clone()))
    });
    v
}

fn int_range(v: &[i64], nulls: &[bool]) -> Option<(i64, i64)> {
    use rayon::prelude::*;
    v.par_chunks(MORSEL)
        .enumerate()
        .filter_map(|(ci, c)| {
            let mut lo = i64::MAX;
            let mut hi = i64::MIN;
            for (j, &x) in c.iter().enumerate() {
                if !nl(nulls, ci * MORSEL + j) {
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
            }
            (lo <= hi).then_some((lo, hi))
        })
        .reduce_with(|a, b| (a.0.min(b.0), a.1.max(b.1)))
}

/// Dense slot per selected row, recording each slot's first row.
fn dense_gid(grouping: &Grouping, rows: &[u32], first: &mut [u32]) -> Vec<u32> {
    let mut put = |slot: usize, r: u32| {
        if first[slot] == u32::MAX {
            first[slot] = r;
        }
        slot as u32
    };
    match grouping {
        Grouping::Text(codes, nd) => rows
            .iter()
            .map(|&r| {
                let c = codes[r as usize];
                put(if c == NULL_CODE { *nd } else { c as usize }, r)
            })
            .collect(),
        Grouping::IntRange(v, nulls, base, range) => rows
            .iter()
            .map(|&r| {
                let ri = r as usize;
                put(
                    if nl(nulls, ri) {
                        *range
                    } else {
                        (v[ri] - base) as usize
                    },
                    r,
                )
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Struct-of-arrays kernel for dense GROUP BY with only COUNT/SUM/AVG:
/// per-slot counters and sums in flat arrays (no per-group structs on the hot
/// path), merged by adding arrays. Same results as the general kernel.
fn aggregate_dense_soa(
    t: &TableColumns,
    preds: &[Pred],
    grouping: &Grouping,
    kinds: &[Option<AggKind>],
    agg_cols: &[Option<&Col>],
    proto: &[Acc],
) -> Option<(Vec<u32>, Vec<Acc>)> {
    use rayon::prelude::*;
    let n = t.len;
    let dense = grouping.slots();
    let nout = kinds.len();
    struct P {
        first: Vec<u32>,
        cnt: Vec<Vec<u64>>,
        isum: Vec<Vec<i128>>,
        fs: Vec<Vec<f64>>,
        fc: Vec<Vec<f64>>,
    }
    let msize = MORSEL.max(dense.saturating_mul(2));
    let morsels: Vec<(usize, usize)> = (0..n)
        .step_by(msize)
        .map(|s| (s, (s + msize).min(n)))
        .collect();
    let run = |&(s, e): &(usize, usize)| -> Option<P> {
        let sel = morsel_sel(t, preds, s, e)?;
        let owned: Vec<u32>;
        let rows: &[u32] = match &sel {
            Some(v) => v,
            None => {
                owned = (s as u32..e as u32).collect();
                &owned
            }
        };
        let mut p = P {
            first: vec![u32::MAX; dense],
            cnt: vec![],
            isum: vec![],
            fs: vec![],
            fc: vec![],
        };
        let gid = dense_gid(grouping, rows, &mut p.first);
        for (oi, k) in kinds.iter().enumerate() {
            let mut cnt = Vec::new();
            let mut isum = Vec::new();
            let (mut fs, mut fc) = (Vec::new(), Vec::new());
            match (k, agg_cols[oi]) {
                (Some(AggKind::CountStar), _) => {
                    cnt = vec![0u64; dense];
                    for &g in &gid {
                        cnt[g as usize] += 1;
                    }
                }
                (Some(AggKind::Count), Some(c)) => {
                    cnt = vec![0u64; dense];
                    for (&g, &r) in gid.iter().zip(rows) {
                        cnt[g as usize] += !is_null(c, r as usize) as u64;
                    }
                }
                (Some(AggKind::Sum | AggKind::Avg), Some(Col::Int(v, nulls))) => {
                    cnt = vec![0u64; dense];
                    isum = vec![0i128; dense];
                    for (&g, &r) in gid.iter().zip(rows) {
                        let ri = r as usize;
                        if !nl(nulls, ri) {
                            cnt[g as usize] += 1;
                            isum[g as usize] += v[ri] as i128;
                        }
                    }
                }
                (Some(AggKind::Sum | AggKind::Avg), Some(Col::Float(v, nulls))) => {
                    cnt = vec![0u64; dense];
                    fs = vec![0f64; dense];
                    fc = vec![0f64; dense];
                    for (&g, &r) in gid.iter().zip(rows) {
                        let ri = r as usize;
                        if !nl(nulls, ri) {
                            let g = g as usize;
                            cnt[g] += 1;
                            neumaier(&mut fs[g], &mut fc[g], v[ri]);
                        }
                    }
                }
                (None, _) => {}
                _ => return None,
            }
            p.cnt.push(cnt);
            p.isum.push(isum);
            p.fs.push(fs);
            p.fc.push(fc);
        }
        Some(p)
    };
    let mut parts: Vec<P> = morsels.par_iter().map(run).collect::<Option<_>>()?;
    if parts.is_empty() {
        parts.push(P {
            first: vec![u32::MAX; dense],
            cnt: vec![vec![]; nout],
            isum: vec![vec![]; nout],
            fs: vec![vec![]; nout],
            fc: vec![vec![]; nout],
        });
    }
    let mut it = parts.into_iter();
    let mut m = it.next()?;
    for p in it {
        for (a, b) in m.first.iter_mut().zip(&p.first) {
            if *a == u32::MAX {
                *a = *b;
            }
        }
        for oi in 0..nout {
            m.cnt[oi]
                .iter_mut()
                .zip(&p.cnt[oi])
                .for_each(|(a, b)| *a += b);
            m.isum[oi]
                .iter_mut()
                .zip(&p.isum[oi])
                .for_each(|(a, b)| *a += b);
            for g in 0..m.fs[oi].len() {
                let (s, c) = (&mut m.fs[oi][g], &mut m.fc[oi][g]);
                neumaier(s, c, p.fs[oi][g]);
                neumaier(s, c, p.fc[oi][g]);
            }
        }
    }
    let mut idx: Vec<usize> = (0..dense).filter(|&i| m.first[i] != u32::MAX).collect();
    idx.sort_unstable_by_key(|&i| m.first[i]);
    let mut out = Vec::with_capacity(idx.len() * nout);
    for &g in &idx {
        for oi in 0..nout {
            let mut a = proto[oi].clone();
            a.count = m.cnt[oi].get(g).copied().unwrap_or(0);
            a.isum = m.isum[oi].get(g).copied().unwrap_or(0);
            a.fsum = m.fs[oi].get(g).copied().unwrap_or(0.0);
            a.fcomp = m.fc[oi].get(g).copied().unwrap_or(0.0);
            out.push(a);
        }
    }
    Some((idx.iter().map(|&i| m.first[i]).collect(), out))
}

/// Filter and aggregate morsel-parallel. Returns each group's first row index
/// (groups in first-appearance order, like a sequential scan) and its
/// accumulators.
fn aggregate(
    t: &TableColumns,
    preds: &[Pred],
    gcols: &[&Col],
    kinds: &[Option<AggKind>],
    agg_cols: &[Option<&Col>],
) -> Option<(Vec<u32>, Vec<Acc>)> {
    use rayon::prelude::*;
    let n = t.len;
    let grouping = match gcols {
        [] => Grouping::One,
        [Col::Text(codes, dict)] => Grouping::Text(codes, dict.len()),
        [Col::Int(v, nulls)] => match int_range(v, nulls) {
            Some((lo, hi))
                if (hi as i128 - lo as i128) < (1 << 20)
                    && ((hi - lo) as usize) <= 4 * n + 1024 =>
            {
                Grouping::IntRange(v, nulls, lo, (hi - lo) as usize + 1)
            }
            None => Grouping::IntRange(v, nulls, 0, 0),
            _ => Grouping::Hash,
        },
        _ => Grouping::Hash,
    };
    // validate column shapes once
    for (k, c) in kinds.iter().zip(agg_cols) {
        match (k, c) {
            (None, _) | (Some(AggKind::CountStar), _) => {}
            (
                Some(AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max),
                Some(Col::Int(..) | Col::Float(..)),
            ) => {}
            (Some(AggKind::Count | AggKind::CountDistinct), Some(c))
                if !matches!(c, Col::Other) => {}
            _ => return None,
        }
    }
    // Distinct over a single group of a small-range int column: a bitmap.
    let bit_base: Vec<Option<(i64, usize)>> = kinds
        .iter()
        .zip(agg_cols)
        .map(|(k, c)| match (k, c, &grouping) {
            (Some(AggKind::CountDistinct), Some(Col::Int(v, nulls)), Grouping::One) => {
                match int_range(v, nulls) {
                    Some((lo, hi)) if (hi as i128 - lo as i128) < (1 << 28) => {
                        Some((lo, (hi - lo) as usize + 1))
                    }
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();
    let proto: Vec<Acc> = kinds
        .iter()
        .zip(&bit_base)
        .map(|(k, bb)| {
            let mut a = Acc::default();
            if matches!(k, Some(AggKind::CountDistinct)) && bb.is_none() {
                a.distinct = Some(Box::default());
            }
            a
        })
        .collect();
    let dense = grouping.slots();
    let nout = kinds.len();
    if dense > 0
        && !matches!(grouping, Grouping::One)
        && kinds.iter().zip(agg_cols).all(|(k, c)| match k {
            None | Some(AggKind::CountStar) => true,
            Some(AggKind::Count) => true,
            Some(AggKind::Sum | AggKind::Avg) => matches!(c, Some(Col::Int(..) | Col::Float(..))),
            _ => false,
        })
    {
        return aggregate_dense_soa(t, preds, &grouping, kinds, agg_cols, &proto);
    }
    // Each partial holds one accumulator row per dense slot, so with many
    // groups use bigger morsels (fewer partials to allocate and merge).
    let msize = MORSEL.max(dense.saturating_mul(8));
    let morsels: Vec<(usize, usize)> = (0..n)
        .step_by(msize)
        .map(|s| (s, (s + msize).min(n)))
        .collect();

    let run = |&(s, e): &(usize, usize)| -> Option<Part> {
        let sel = morsel_sel(t, preds, s, e)?;
        let owned: Vec<u32>;
        let selv: &[u32] = match &sel {
            Some(v) => v,
            None => {
                owned = (s as u32..e as u32).collect();
                &owned
            }
        };
        let rows = || selv.iter().copied();
        let nsel = sel.as_ref().map_or(e - s, |v| v.len());
        let mut part = Part {
            first: vec![u32::MAX; dense],
            accs: Vec::new(),
            map: FxMap::default(),
        };
        // group id per selected row
        let gid: Vec<u32> = match &grouping {
            Grouping::One => {
                if nsel > 0 {
                    part.first[0] = rows().next().unwrap_or(u32::MAX);
                }
                Vec::new()
            }
            Grouping::Text(codes, nd) => rows()
                .map(|r| {
                    let c = codes[r as usize];
                    let slot = if c == NULL_CODE { *nd } else { c as usize };
                    if part.first[slot] == u32::MAX {
                        part.first[slot] = r;
                    }
                    slot as u32
                })
                .collect(),
            Grouping::IntRange(v, nulls, base, range) => rows()
                .map(|r| {
                    let ri = r as usize;
                    let slot = if nl(nulls, ri) {
                        *range
                    } else {
                        (v[ri] - base) as usize
                    };
                    if part.first[slot] == u32::MAX {
                        part.first[slot] = r;
                    }
                    slot as u32
                })
                .collect(),
            Grouping::Hash => {
                let mut buf = Vec::with_capacity(gcols.len());
                rows()
                    .map(|r| {
                        let ri = r as usize;
                        buf.clear();
                        for c in gcols {
                            let code = code_of(c, ri);
                            buf.push((code, is_null_code(c, code, ri)));
                        }
                        match part.map.get(&buf) {
                            Some(&g) => g,
                            None => {
                                let g = part.first.len() as u32;
                                part.map.insert(buf.clone(), g);
                                part.first.push(r);
                                g
                            }
                        }
                    })
                    .collect()
            }
        };
        let ng = part.first.len();
        part.accs = (0..ng).flat_map(|_| proto.iter().cloned()).collect();
        for (oi, kind) in kinds.iter().enumerate() {
            let Some(kind) = kind else { continue };
            if let Some((_, nbits)) = bit_base[oi] {
                part.accs[oi].bits = vec![0u64; nbits.div_ceil(64)];
            }
            let accs = &mut part.accs;
            let g = |k: usize| if gid.is_empty() { 0 } else { gid[k] as usize };
            let col = agg_cols[oi].unwrap_or(&Col::Other);
            match (kind, col) {
                (AggKind::CountStar, _) => {
                    if gid.is_empty() {
                        accs[oi].count += nsel as u64;
                    } else {
                        for &x in &gid {
                            accs[x as usize * nout + oi].count += 1;
                        }
                    }
                }
                (AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max, Col::Int(v, nulls)) => {
                    if gid.is_empty() && sel.is_none() && nulls.is_empty() {
                        // contiguous, no NULLs: tight loop the compiler vectorizes
                        let sl = &v[s..e];
                        let a = &mut accs[oi];
                        a.count += sl.len() as u64;
                        a.isum += sl.iter().map(|&x| x as i128).sum::<i128>();
                        if let (Some(&lo), Some(&hi)) = (sl.iter().min(), sl.iter().max()) {
                            a.min = Some(a.min.map_or(Num::I(lo), |m| {
                                if num_lt(Num::I(lo), m) { Num::I(lo) } else { m }
                            }));
                            a.max = Some(a.max.map_or(Num::I(hi), |m| {
                                if num_lt(m, Num::I(hi)) { Num::I(hi) } else { m }
                            }));
                        }
                        continue;
                    }
                    for (k, r) in rows().enumerate() {
                        let ri = r as usize;
                        if nl(nulls, ri) {
                            continue;
                        }
                        let x = v[ri];
                        let a = &mut accs[g(k) * nout + oi];
                        a.count += 1;
                        a.isum += x as i128;
                        if a.min.is_none_or(|m| num_lt(Num::I(x), m)) {
                            a.min = Some(Num::I(x));
                        }
                        if a.max.is_none_or(|m| num_lt(m, Num::I(x))) {
                            a.max = Some(Num::I(x));
                        }
                    }
                }
                (
                    AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max,
                    Col::Float(v, nulls),
                ) => {
                    let want_mm = matches!(kind, AggKind::Min | AggKind::Max);
                    if gid.is_empty() && sel.is_none() && nulls.is_empty() {
                        let sl = &v[s..e];
                        let a = &mut accs[oi];
                        a.count += sl.len() as u64;
                        for &x in sl {
                            neumaier(&mut a.fsum, &mut a.fcomp, x);
                        }
                        if want_mm {
                            for &x in sl {
                                if a.min.is_none_or(|m| num_lt(Num::F(x), m)) {
                                    a.min = Some(Num::F(x));
                                }
                                if a.max.is_none_or(|m| num_lt(m, Num::F(x))) {
                                    a.max = Some(Num::F(x));
                                }
                            }
                        }
                        continue;
                    }
                    for (k, r) in rows().enumerate() {
                        let ri = r as usize;
                        if nl(nulls, ri) {
                            continue;
                        }
                        let x = v[ri];
                        let a = &mut accs[g(k) * nout + oi];
                        a.count += 1;
                        neumaier(&mut a.fsum, &mut a.fcomp, x);
                        if want_mm {
                            if a.min.is_none_or(|m| num_lt(Num::F(x), m)) {
                                a.min = Some(Num::F(x));
                            }
                            if a.max.is_none_or(|m| num_lt(m, Num::F(x))) {
                                a.max = Some(Num::F(x));
                            }
                        }
                    }
                }
                (AggKind::Count, c) => {
                    for (k, r) in rows().enumerate() {
                        if !is_null(c, r as usize) {
                            accs[g(k) * nout + oi].count += 1;
                        }
                    }
                }
                (AggKind::CountDistinct, c) => {
                    if let (Some((base, _)), Col::Int(v, nulls)) = (bit_base[oi], c) {
                        let bits = &mut accs[oi].bits;
                        for r in rows() {
                            let ri = r as usize;
                            if !nl(nulls, ri) {
                                let b = (v[ri] - base) as usize;
                                bits[b >> 6] |= 1u64 << (b & 63);
                            }
                        }
                        continue;
                    }
                    for (k, r) in rows().enumerate() {
                        if let Some(key) = distinct_key(c, r as usize) {
                            if let Some(d) = accs[g(k) * nout + oi].distinct.as_mut() {
                                d.insert(key);
                            }
                        }
                    }
                }
                _ => return None,
            }
        }
        Some(part)
    };
    let tr = std::env::var_os("JOULEDB_VEC_TRACE").is_some();
    let t0 = std::time::Instant::now();
    let parts: Vec<Part> = morsels.par_iter().map(run).collect::<Option<_>>()?;
    let t1 = std::time::Instant::now();

    let parts_len = parts.len();
    // Merge partials in morsel order (keeps first-appearance order exact).
    let mut first: Vec<u32>;
    let mut accs: Vec<Acc>;
    if dense > 0 {
        first = vec![u32::MAX; dense];
        accs = (0..dense).flat_map(|_| proto.iter().cloned()).collect();
        for p in &parts {
            for (slot, &f) in p.first.iter().enumerate() {
                if f == u32::MAX {
                    continue;
                }
                if first[slot] == u32::MAX {
                    first[slot] = f;
                }
                for (a, b) in accs[slot * nout..(slot + 1) * nout]
                    .iter_mut()
                    .zip(&p.accs[slot * nout..(slot + 1) * nout])
                {
                    a.merge(b);
                }
            }
        }
    } else {
        let mut map: FxMap<Vec<(u64, bool)>, usize> = FxMap::default();
        first = Vec::new();
        accs = Vec::new();
        for p in parts {
            let mut entries: Vec<(&Vec<(u64, bool)>, &u32)> = p.map.iter().collect();
            entries.sort_by_key(|(_, g)| **g);
            for (k, &g) in entries {
                let gi = *map.entry(k.clone()).or_insert_with(|| {
                    first.push(p.first[g as usize]);
                    accs.extend(proto.iter().cloned());
                    first.len() - 1
                });
                for (a, b) in accs[gi * nout..(gi + 1) * nout]
                    .iter_mut()
                    .zip(&p.accs[g as usize * nout..(g as usize + 1) * nout])
                {
                    a.merge(b);
                }
            }
        }
    }
    // keep only seen groups, ordered by first appearance
    let mut idx: Vec<usize> = (0..first.len()).filter(|&i| first[i] != u32::MAX).collect();
    if !matches!(grouping, Grouping::One) {
        idx.sort_unstable_by_key(|&i| first[i]);
    } else if idx.is_empty() {
        idx.push(0); // aggregates over zero rows still produce one row
    }
    if tr {
        eprintln!(
            "agg parts={} run={:?} merge={:?} dense={dense}",
            parts_len,
            t1 - t0,
            t1.elapsed()
        );
    }
    let firsts = idx.iter().map(|&i| first[i]).collect();
    // compact to the kept groups, in order (moves, no clones)
    let mut slots: Vec<Option<Acc>> = accs.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(idx.len() * nout);
    for &i in &idx {
        for oi in 0..nout {
            out.push(
                slots
                    .get_mut(i * nout + oi)
                    .and_then(Option::take)
                    .unwrap_or_else(|| proto[oi].clone()),
            );
        }
    }
    Some((firsts, out))
}

/// Output value before JSON rendering.
enum Sv {
    Null,
    U(u64),
    I(i64),
    F(f64),
    S(Arc<str>),
}

impl Sv {
    fn from_key(k: &Key) -> Sv {
        match k {
            Key::Null => Sv::Null,
            Key::I(i) => Sv::I(*i),
            Key::F(b) => Sv::F(f64::from_bits(*b)),
            Key::S(s) => Sv::S(s.clone()),
        }
    }
    fn from_num(n: Option<Num>) -> Sv {
        match n {
            None => Sv::Null,
            Some(Num::I(i)) => Sv::I(i),
            Some(Num::F(f)) => Sv::F(f),
        }
    }
    fn is_null(&self) -> bool {
        matches!(self, Sv::Null)
    }
    fn json(&self) -> serde_json::Value {
        match self {
            Sv::Null => serde_json::Value::Null,
            Sv::U(u) => serde_json::json!(u),
            Sv::I(i) => serde_json::json!(i),
            Sv::F(f) => serde_json::json!(f),
            Sv::S(s) => serde_json::Value::String(s.to_string()),
        }
    }
    /// Same order as comparing the rendered JSON values (`cmp_json`).
    fn cmp_val(&self, o: &Sv) -> Ordering {
        let int = |v: &Sv| match v {
            Sv::I(i) => Some(*i as i128),
            Sv::U(u) => Some(*u as i128),
            _ => None,
        };
        let fl = |v: &Sv| match v {
            Sv::I(i) => Some(*i as f64),
            Sv::U(u) => Some(*u as f64),
            Sv::F(f) => Some(*f),
            _ => None,
        };
        match (self, o) {
            (Sv::S(a), Sv::S(b)) => a.cmp(b),
            _ => match (int(self), int(o)) {
                (Some(a), Some(b)) => a.cmp(&b),
                _ => match (fl(self), fl(o)) {
                    (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
                    _ => Ordering::Equal,
                },
            },
        }
    }
}

/// Result of a vectorized SELECT: column names and JSON rows.
pub struct VecResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// Run `query` against `t` if it has a supported shape, else `None`.
pub fn execute(query: &SqlQuery, t: &TableColumns) -> Option<VecResult> {
    if !query.joins.is_empty()
        || !query.ctes.is_empty()
        || query.set_op.is_some()
        || query.having.is_some()
        || query.distinct
        || query.as_of_timestamp.is_some()
    {
        return None;
    }
    let group_cols: Vec<&str> = query.group_by.iter().map(column).collect::<Option<_>>()?;
    for g in &group_cols {
        if matches!(t.cols.get(*g)?, Col::Other) {
            return None;
        }
    }

    // Output columns
    let mut outs = Vec::new();
    let mut names = Vec::new();
    for c in &query.columns {
        let (out, default_name) = match &c.expr {
            e if column(e).is_some() => {
                let name = column(e)?;
                (
                    OutCol::Group(group_cols.iter().position(|g| *g == name)?),
                    name.to_string(),
                )
            }
            Expression::Function { name, args } => {
                let upper = name.to_ascii_uppercase();
                let arg = match args.as_slice() {
                    [] => None,
                    [Expression::Wildcard] => None,
                    [e] => Some(column(e)?),
                    _ => return None,
                };
                let kind = match (upper.as_str(), arg) {
                    ("COUNT", None) => AggKind::CountStar,
                    ("COUNT", Some(_)) => AggKind::Count,
                    ("COUNT_DISTINCT", Some(_)) => AggKind::CountDistinct,
                    ("SUM", Some(_)) => AggKind::Sum,
                    ("AVG", Some(_)) => AggKind::Avg,
                    ("MIN", Some(_)) => AggKind::Min,
                    ("MAX", Some(_)) => AggKind::Max,
                    _ => return None,
                };
                if let Some(a) = arg {
                    let col = t.cols.get(a)?;
                    match kind {
                        AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max => {
                            if !matches!(col, Col::Int(..) | Col::Float(..)) {
                                return None;
                            }
                        }
                        _ => {
                            if matches!(col, Col::Other) {
                                return None;
                            }
                        }
                    }
                }
                let n = if upper == "COUNT_DISTINCT" {
                    "count".to_string()
                } else {
                    name.to_ascii_lowercase()
                };
                (OutCol::Agg(kind, arg), n)
            }
            _ => return None,
        };
        outs.push(out);
        names.push(c.alias.clone().unwrap_or(default_name));
    }
    let has_agg = outs.iter().any(|o| matches!(o, OutCol::Agg(..)));
    if !has_agg && group_cols.is_empty() {
        return None; // plain projection: general executor
    }

    let mut preds = Vec::new();
    if let Some(w) = &query.where_clause {
        collect_preds(w, &mut preds)?;
    }
    // ORDER BY must name an output column (alias, name, or same expression)
    let mut order = Vec::new();
    for ob in &query.order_by {
        let idx = match &ob.expr {
            Expression::Literal(AstValue::Int(i)) if *i >= 1 && (*i as usize) <= names.len() => {
                *i as usize - 1
            }
            e => {
                if let Some(n) = column(e) {
                    names.iter().position(|x| x == n)?
                } else {
                    query.columns.iter().position(|c| &c.expr == e)?
                }
            }
        };
        order.push((idx, ob.descending, ob.nulls_first));
    }

    let agg_cols: Vec<Option<&Col>> = outs
        .iter()
        .map(|o| match o {
            OutCol::Agg(_, Some(a)) => t.cols.get(*a),
            _ => None,
        })
        .collect();
    let gcols: Vec<&Col> = group_cols
        .iter()
        .map(|g| t.cols.get(*g))
        .collect::<Option<_>>()?;
    let kinds: Vec<Option<AggKind>> = outs
        .iter()
        .map(|o| match o {
            OutCol::Agg(k, _) => Some(*k),
            _ => None,
        })
        .collect();
    let ta = std::time::Instant::now();
    let meta = if preds.is_empty() && gcols.is_empty() {
        metadata_aggregate(t, &outs)
    } else if gcols.is_empty()
        && outs.iter().all(|o| matches!(o, OutCol::Agg(AggKind::CountStar, _)))
    {
        count_where(t, &preds).map(|c| {
            outs.iter()
                .map(|_| Acc { count: c, ..Acc::default() })
                .collect()
        })
    } else {
        None
    };
    let (firsts, accs) = match meta {
        Some(accs) => (vec![0], accs),
        None => aggregate(t, &preds, &gcols, &kinds, &agg_cols)?,
    };
    let ta = ta.elapsed();
    let order_of: Vec<Vec<Key>> = if gcols.is_empty() {
        vec![Vec::new()]
    } else {
        firsts
            .iter()
            .map(|&r| gcols.iter().map(|c| key_at(c, r as usize)).collect())
            .collect()
    };

    // Cheap typed output values; JSON only for the rows returned.
    let no = outs.len();
    let vals: Vec<Sv> = order_of
        .iter()
        .enumerate()
        .flat_map(|(gi, key)| {
            let accs = &accs[gi * outs.len()..(gi + 1) * outs.len()];
            outs.iter()
                .enumerate()
                .map(|(oi, o)| match o {
                    OutCol::Group(g) => Sv::from_key(&key[*g]),
                    OutCol::Agg(kind, _) => {
                        let a = &accs[oi];
                        let is_int = matches!(agg_cols[oi], Some(Col::Int(..)));
                        match kind {
                            AggKind::CountStar | AggKind::Count => Sv::U(a.count),
                            AggKind::CountDistinct if !a.bits.is_empty() => {
                                Sv::U(a.bits.iter().map(|w| w.count_ones() as u64).sum())
                            }
                            AggKind::CountDistinct => {
                                Sv::U(a.distinct.as_ref().map_or(0, |d| d.len() as u64))
                            }
                            AggKind::Sum if a.count == 0 => Sv::Null,
                            AggKind::Sum if is_int => match i64::try_from(a.isum) {
                                Ok(v) => Sv::I(v),
                                Err(_) => Sv::F(a.isum as f64),
                            },
                            AggKind::Sum => Sv::F(a.fsum + a.fcomp),
                            AggKind::Avg if a.count == 0 => Sv::Null,
                            AggKind::Avg => {
                                let s = if is_int {
                                    a.isum as f64
                                } else {
                                    a.fsum + a.fcomp
                                };
                                Sv::F(s / a.count as f64)
                            }
                            AggKind::Min => Sv::from_num(a.min),
                            AggKind::Max => Sv::from_num(a.max),
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let ngroups = order_of.len();
    let mut idx: Vec<usize> = (0..ngroups).collect();
    let off = query.offset.unwrap_or(0);
    let take = query.limit.map(|l| off.saturating_add(l));
    if !order.is_empty() {
        let cmp = |&ia: &usize, &ib: &usize| -> Ordering {
            let (a, b) = (&vals[ia * no..(ia + 1) * no], &vals[ib * no..(ib + 1) * no]);
            for (i, desc, nulls_first) in &order {
                let (x, y) = (&a[*i], &b[*i]);
                let nf = nulls_first.unwrap_or(*desc);
                let o = match (x.is_null(), y.is_null()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => {
                        if nf {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    }
                    (false, true) => {
                        if nf {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }
                    _ => {
                        let o = x.cmp_val(y);
                        if *desc { o.reverse() } else { o }
                    }
                };
                if o != Ordering::Equal {
                    return o;
                }
            }
            ia.cmp(&ib) // stable: earlier groups first on ties
        };
        // top-N: partition the first `take` then sort only those
        if let Some(k) = take.filter(|&k| k > 0 && k < idx.len()) {
            idx.select_nth_unstable_by(k - 1, cmp);
            idx.truncate(k);
        }
        idx.sort_unstable_by(cmp);
    }
    if std::env::var_os("JOULEDB_VEC_TRACE").is_some() {
        eprintln!("render agg={ta:?} groups={ngroups}");
    }
    let end = take.unwrap_or(usize::MAX).min(idx.len());
    let rows: Vec<Vec<serde_json::Value>> = idx[off.min(end)..end]
        .iter()
        .map(|&i| vals[i * no..(i + 1) * no].iter().map(Sv::json).collect())
        .collect();
    Some(VecResult {
        columns: names,
        rows,
    })
}
