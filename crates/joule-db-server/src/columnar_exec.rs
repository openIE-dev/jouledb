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

#[derive(Debug)]
pub enum Col {
    Int(Vec<i64>, Vec<bool>),
    Float(Vec<f64>, Vec<bool>),
    /// Dictionary-encoded text: per-row code (u32::MAX = NULL) and the dictionary
    Text(Vec<u32>, Vec<Arc<str>>),
    /// Mixed or non-scalar values: queries touching it fall back
    Other,
}

/// Typed columns of one table, in record id order.
#[derive(Debug)]
pub struct TableColumns {
    pub len: usize,
    pub cols: HashMap<String, Col>,
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
                Col::Int(v, nulls)
            };
            cols.insert(c.clone(), col);
        }
        Self { len: n, cols }
    }
}

const NULL_CODE: u32 = u32::MAX;

type FxMap<K, V> = rustc_hash::FxHashMap<K, V>;

#[derive(Clone, Copy, PartialEq)]
enum Num {
    I(i64),
    F(f64),
}

enum Pred<'a> {
    Cmp { col: &'a str, op: Operator, lit: Lit },
    Like { col: &'a str, pattern: &'a str, negated: bool, ci: bool },
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
        Expression::Unary { op: UnaryOperator::Neg, expr } => match literal(expr)? {
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
        Expression::Binary { left, op: Operator::And, right } => {
            collect_preds(left, out)?;
            collect_preds(right, out)
        }
        Expression::Binary { left, op, right } => {
            if let (Some(c), Some(l)) = (column(left), literal(right)) {
                flip(op)?;
                out.push(Pred::Cmp { col: c, op: op.clone(), lit: l });
            } else if let (Some(l), Some(c)) = (literal(left), column(right)) {
                out.push(Pred::Cmp { col: c, op: flip(op)?, lit: l });
            } else {
                return None;
            }
            Some(())
        }
        Expression::Like { expr, pattern, negated, case_insensitive } => {
            out.push(Pred::Like { col: column(expr)?, pattern, negated: *negated, ci: *case_insensitive });
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
                    sel.retain(|&i| !nulls[i as usize] && cmp_ok(v[i as usize].cmp(x), op))
                }
                (Col::Int(v, nulls), Lit::N(Num::F(x))) => sel.retain(|&i| {
                    !nulls[i as usize]
                        && (v[i as usize] as f64).partial_cmp(x).is_some_and(|o| cmp_ok(o, op))
                }),
                (Col::Float(v, nulls), Lit::N(n)) => {
                    let x = match n {
                        Num::I(i) => *i as f64,
                        Num::F(f) => *f,
                    };
                    sel.retain(|&i| {
                        !nulls[i as usize] && v[i as usize].partial_cmp(&x).is_some_and(|o| cmp_ok(o, op))
                    })
                }
                (Col::Text(codes, dict), Lit::S(x)) => {
                    let ok: Vec<bool> = dict.iter().map(|d| cmp_ok((**d).cmp(x.as_str()), op)).collect();
                    sel.retain(|&i| {
                        let c = codes[i as usize];
                        c != NULL_CODE && ok[c as usize]
                    })
                }
                _ => return None,
            },
            Pred::Like { col, pattern, negated, ci } => match t.cols.get(*col)? {
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
        Col::Int(v, n) => if n[i] { Key::Null } else { Key::I(v[i]) },
        Col::Float(v, n) => if n[i] { Key::Null } else { Key::F(v[i].to_bits()) },
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
    distinct: Option<HashSet<u64>>,
}

fn num_at(c: &Col, i: usize) -> Option<Num> {
    match c {
        Col::Int(v, n) => (!n[i]).then(|| Num::I(v[i])),
        Col::Float(v, n) => (!n[i]).then(|| Num::F(v[i])),
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
        (J::Number(x), J::Number(y)) => {
            match (x.as_i64(), y.as_i64()) {
                (Some(p), Some(q)) => p.cmp(&q),
                _ => x.as_f64().unwrap_or(0.0).partial_cmp(&y.as_f64().unwrap_or(0.0)).unwrap_or(Ordering::Equal),
            }
        }
        (J::String(x), J::String(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

fn is_null(c: &Col, i: usize) -> bool {
    match c {
        Col::Int(_, n) | Col::Float(_, n) => n[i],
        Col::Text(codes, _) => codes[i] == NULL_CODE,
        Col::Other => true,
    }
}

/// Hashable distinct key (text by dictionary code, which is unique per string)
fn distinct_key(c: &Col, i: usize) -> Option<u64> {
    match c {
        Col::Int(v, n) => (!n[i]).then(|| v[i] as u64),
        Col::Float(v, n) => (!n[i]).then(|| v[i].to_bits()),
        Col::Text(codes, _) => (codes[i] != NULL_CODE).then(|| codes[i] as u64),
        Col::Other => None,
    }
}

/// Code of row `i` within one column (u64; NULL gets its own code).
fn code_of(c: &Col, i: usize) -> u64 {
    match c {
        Col::Int(v, n) => if n[i] { u64::MAX } else { v[i] as u64 },
        Col::Float(v, n) => if n[i] { u64::MAX } else { v[i].to_bits() },
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
                let slot = if code == NULL_CODE { dict.len() } else { code as usize };
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
                (OutCol::Group(group_cols.iter().position(|g| *g == name)?), name.to_string())
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
                let n = if upper == "COUNT_DISTINCT" { "count".to_string() } else { name.to_ascii_lowercase() };
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
            Expression::Literal(AstValue::Int(i)) if *i >= 1 && (*i as usize) <= names.len() => *i as usize - 1,
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

    let sel = filter(t, &preds)?;
    let agg_cols: Vec<Option<&Col>> = outs
        .iter()
        .map(|o| match o {
            OutCol::Agg(_, Some(a)) => t.cols.get(*a),
            _ => None,
        })
        .collect();
    let gcols: Vec<&Col> = group_cols.iter().map(|g| t.cols.get(*g)).collect::<Option<_>>()?;

    let mut proto = vec![Acc::default(); outs.len()];
    for (i, o) in outs.iter().enumerate() {
        if matches!(o, OutCol::Agg(AggKind::CountDistinct, _)) {
            proto[i].distinct = Some(HashSet::new());
        }
    }
    // Dense group id per selected row, in first-appearance order.
    let (gid, order_of) = group_ids(&gcols, &sel);
    let ngroups = order_of.len().max(if gcols.is_empty() { 1 } else { 0 });
    let mut accs: Vec<Vec<Acc>> = vec![proto; ngroups];
    for (oi, o) in outs.iter().enumerate() {
        let OutCol::Agg(kind, _) = o else { continue };
        let col = agg_cols[oi].unwrap_or(&Col::Other);
        let g = |k: usize| if gid.is_empty() { 0 } else { gid[k] as usize };
        match (kind, col) {
            (AggKind::CountStar, _) => {
                for k in 0..sel.len() {
                    accs[g(k)][oi].count += 1;
                }
            }
            (AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max, Col::Int(v, nulls)) => {
                for (k, &ri) in sel.iter().enumerate() {
                    let ri = ri as usize;
                    if nulls[ri] {
                        continue;
                    }
                    let x = v[ri];
                    let a = &mut accs[g(k)][oi];
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
            (AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max, Col::Float(v, nulls)) => {
                for (k, &ri) in sel.iter().enumerate() {
                    let ri = ri as usize;
                    if nulls[ri] {
                        continue;
                    }
                    let x = v[ri];
                    let a = &mut accs[g(k)][oi];
                    a.count += 1;
                    // Neumaier compensated sum: matches exact decimal sums
                    let t = a.fsum + x;
                    if a.fsum.abs() >= x.abs() {
                        a.fcomp += (a.fsum - t) + x;
                    } else {
                        a.fcomp += (x - t) + a.fsum;
                    }
                    a.fsum = t;
                    if a.min.is_none_or(|m| num_lt(Num::F(x), m)) {
                        a.min = Some(Num::F(x));
                    }
                    if a.max.is_none_or(|m| num_lt(m, Num::F(x))) {
                        a.max = Some(Num::F(x));
                    }
                }
            }
            (AggKind::Count, c) => {
                for (k, &ri) in sel.iter().enumerate() {
                    if !is_null(c, ri as usize) {
                        accs[g(k)][oi].count += 1;
                    }
                }
            }
            (AggKind::CountDistinct, c) => {
                for (k, &ri) in sel.iter().enumerate() {
                    let key = distinct_key(c, ri as usize);
                    if let Some(key) = key {
                        if let Some(d) = accs[g(k)][oi].distinct.as_mut() {
                            d.insert(key);
                        }
                    }
                }
            }
            _ => return None,
        }
    }
    let order_of: Vec<Vec<Key>> = if gcols.is_empty() { vec![Vec::new()] } else { order_of };

    let mut rows: Vec<Vec<serde_json::Value>> = order_of
        .iter()
        .enumerate()
        .map(|(gi, key)| {
            let accs = &accs[gi];
            outs.iter()
                .enumerate()
                .map(|(oi, o)| match o {
                    OutCol::Group(g) => key_json(&key[*g]),
                    OutCol::Agg(kind, _) => {
                        let a = &accs[oi];
                        let is_int = matches!(agg_cols[oi], Some(Col::Int(..)));
                        match kind {
                            AggKind::CountStar | AggKind::Count => serde_json::json!(a.count),
                            AggKind::CountDistinct => {
                                serde_json::json!(a.distinct.as_ref().map_or(0, |d| d.len()))
                            }
                            AggKind::Sum if a.count == 0 => serde_json::Value::Null,
                            AggKind::Sum if is_int => match i64::try_from(a.isum) {
                                Ok(v) => serde_json::json!(v),
                                Err(_) => serde_json::json!(a.isum as f64),
                            },
                            AggKind::Sum => serde_json::json!(a.fsum + a.fcomp),
                            AggKind::Avg if a.count == 0 => serde_json::Value::Null,
                            AggKind::Avg => {
                                let s = if is_int { a.isum as f64 } else { a.fsum + a.fcomp };
                                serde_json::json!(s / a.count as f64)
                            }
                            AggKind::Min => num_json(a.min),
                            AggKind::Max => num_json(a.max),
                        }
                    }
                })
                .collect()
        })
        .collect();

    if !order.is_empty() {
        rows.sort_by(|a, b| {
            for (idx, desc, nulls_first) in &order {
                let (x, y) = (&a[*idx], &b[*idx]);
                let nf = nulls_first.unwrap_or(*desc);
                let o = match (x.is_null(), y.is_null()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => if nf { Ordering::Less } else { Ordering::Greater },
                    (false, true) => if nf { Ordering::Greater } else { Ordering::Less },
                    _ => {
                        let o = cmp_json(x, y);
                        if *desc { o.reverse() } else { o }
                    }
                };
                if o != Ordering::Equal {
                    return o;
                }
            }
            Ordering::Equal
        });
    }
    let off = query.offset.unwrap_or(0).min(rows.len());
    rows.drain(..off);
    if let Some(l) = query.limit {
        rows.truncate(l);
    }
    Some(VecResult { columns: names, rows })
}
