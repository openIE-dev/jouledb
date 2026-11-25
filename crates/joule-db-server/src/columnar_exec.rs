//! Vectorized single-table analytics over a cached columnar snapshot.
//!
//! Executes single-table SQL scans, projections, Top-K `ORDER BY ... LIMIT`,
//! scalar expressions (`EXTRACT`, `DATE_TRUNC`, `STRLEN`/`LENGTH`, `REGEXP_REPLACE`,
//! `CASE WHEN`, integer arithmetic, constants), dictionary-pushed `LIKE`/equality,
//! `IN (...)` / `BETWEEN` / ISO-date predicates, morsel-parallel `GROUP BY`,
//! `MIN`/`MAX` over numeric and dictionary-text columns, and `HAVING` filters
//! directly over narrow typed columns (`i8`/`i16`/`i32`/`i64`, `f64`, dictionary `u32`).
//! Any unsupported shape returns `None` and falls back to the general executor.

use joule_db_query::ast::{Expression, Operator, UnaryOperator, Value as AstValue};
use joule_db_query::sql::SqlQuery;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Width-adaptive integer column backing: cuts integer RAM 2x–8x on narrow
/// Parquet/Arrow columns while widening transparently on out-of-range appends.
#[derive(Debug, Clone)]
pub enum Ints {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

macro_rules! with_ints {
    ($ints:expr, |$s:ident| $body:expr) => {
        match $ints {
            Ints::I8($s) => $body,
            Ints::I16($s) => $body,
            Ints::I32($s) => $body,
            Ints::I64($s) => $body,
        }
    };
}

impl Ints {
    #[inline(always)]
    pub fn len(&self) -> usize {
        with_ints!(self, |s| s.len())
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn at(&self, i: usize) -> i64 {
        match self {
            Ints::I8(v) => v[i] as i64,
            Ints::I16(v) => v[i] as i64,
            Ints::I32(v) => v[i] as i64,
            Ints::I64(v) => v[i],
        }
    }

    #[inline]
    fn raw_ptr_id(&self) -> usize {
        match self {
            Ints::I8(v) => v.as_ptr() as usize,
            Ints::I16(v) => v.as_ptr() as usize,
            Ints::I32(v) => v.as_ptr() as usize,
            Ints::I64(v) => v.as_ptr() as usize,
        }
    }

    fn reserve(&mut self, additional: usize) {
        match self {
            Ints::I8(v) => v.reserve(additional),
            Ints::I16(v) => v.reserve(additional),
            Ints::I32(v) => v.reserve(additional),
            Ints::I64(v) => v.reserve(additional),
        }
    }

    fn push_i64(&mut self, x: i64) {
        match self {
            Ints::I8(v) => {
                if let Ok(n) = i8::try_from(x) {
                    v.push(n);
                } else if let Ok(n) = i16::try_from(x) {
                    let mut w: Vec<i16> = v.iter().map(|&y| y as i16).collect();
                    w.push(n);
                    *self = Ints::I16(w);
                } else if let Ok(n) = i32::try_from(x) {
                    let mut w: Vec<i32> = v.iter().map(|&y| y as i32).collect();
                    w.push(n);
                    *self = Ints::I32(w);
                } else {
                    let mut w: Vec<i64> = v.iter().map(|&y| y as i64).collect();
                    w.push(x);
                    *self = Ints::I64(w);
                }
            }
            Ints::I16(v) => {
                if let Ok(n) = i16::try_from(x) {
                    v.push(n);
                } else if let Ok(n) = i32::try_from(x) {
                    let mut w: Vec<i32> = v.iter().map(|&y| y as i32).collect();
                    w.push(n);
                    *self = Ints::I32(w);
                } else {
                    let mut w: Vec<i64> = v.iter().map(|&y| y as i64).collect();
                    w.push(x);
                    *self = Ints::I64(w);
                }
            }
            Ints::I32(v) => {
                if let Ok(n) = i32::try_from(x) {
                    v.push(n);
                } else {
                    let mut w: Vec<i64> = v.iter().map(|&y| y as i64).collect();
                    w.push(x);
                    *self = Ints::I64(w);
                }
            }
            Ints::I64(v) => v.push(x),
        }
    }

    fn from_i64_vec(v: Vec<i64>) -> Self {
        if v.is_empty() {
            return Ints::I8(Vec::new());
        }
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for &x in &v {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        if lo >= i8::MIN as i64 && hi <= i8::MAX as i64 {
            Ints::I8(v.into_iter().map(|x| x as i8).collect())
        } else if lo >= i16::MIN as i64 && hi <= i16::MAX as i64 {
            Ints::I16(v.into_iter().map(|x| x as i16).collect())
        } else if lo >= i32::MIN as i64 && hi <= i32::MAX as i64 {
            Ints::I32(v.into_iter().map(|x| x as i32).collect())
        } else {
            Ints::I64(v)
        }
    }
}

#[derive(Debug, Clone)]
pub enum Col {
    /// Values and per-row NULL flags; an empty flag vector means no NULLs.
    Int(Ints, Vec<bool>),
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
    pub col_order: Vec<String>,
    pub cols: HashMap<String, Col>,
    /// Reverse dictionaries for text columns, built on first append
    text_index: HashMap<String, FxMap<Arc<str>, u32>>,
    /// Zone maps for numeric columns: min/max/NULL count per ZONE rows.
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
    let mut z = Zone {
        lo: None,
        hi: None,
        nulls: 0,
    };
    match c {
        Col::Int(ints, n) => {
            let (mut lo, mut hi, mut any) = (i64::MAX, i64::MIN, false);
            with_ints!(ints, |v| {
                if n.is_empty() {
                    for &raw in &v[s..e] {
                        let x = raw as i64;
                        lo = lo.min(x);
                        hi = hi.max(x);
                    }
                    any = s < e;
                } else {
                    for i in s..e {
                        if n[i] {
                            z.nulls += 1;
                        } else {
                            let x = v[i] as i64;
                            lo = lo.min(x);
                            hi = hi.max(x);
                            any = true;
                        }
                    }
                }
            });
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
                        return None;
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
                Col::Int(Ints::from_i64_vec(v), nulls)
            };
            cols.insert(c.clone(), col);
        }
        let mut t = Self {
            len: n,
            col_order: columns.to_vec(),
            cols,
            text_index: HashMap::new(),
            zones: HashMap::new(),
        };
        t.refresh_zones(0);
        t
    }

    pub fn append<R: FieldGet + Sync>(&mut self, records: &[R]) -> bool {
        use rayon::prelude::*;
        let old = self.len;
        let mut idxs = std::mem::take(&mut self.text_index);
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

    pub fn from_batch(
        columns: &[String],
        batch: &joule_db_amorphic::column_batch::ColumnBatch,
    ) -> Self {
        use joule_db_amorphic::column_batch::ColData;
        use rayon::prelude::*;
        let n = batch.len;
        let built: Vec<(String, Col, Option<FxMap<Arc<str>, u32>>)> = columns
            .par_iter()
            .map(|c| {
                let bc = batch.cols.iter().find(|b| &b.name == c);
                let any = bc.is_some_and(|b| (0..n).any(|i| !b.is_null(i)));
                let nulls_of = |b: &joule_db_amorphic::column_batch::BatchColumn| match &b.valid {
                    Some(v) if v.iter().any(|x| !x) => v.iter().map(|x| !x).collect(),
                    _ => Vec::new(),
                };
                let mut rev: Option<FxMap<Arc<str>, u32>> = None;
                let col = match bc {
                    Some(b) if any => match &b.data {
                        ColData::Int8(v) => Col::Int(
                            Ints::I8((0..n).map(|i| if b.is_null(i) { 0 } else { v[i] }).collect()),
                            nulls_of(b),
                        ),
                        ColData::Int16(v) => Col::Int(
                            Ints::I16(
                                (0..n).map(|i| if b.is_null(i) { 0 } else { v[i] }).collect(),
                            ),
                            nulls_of(b),
                        ),
                        ColData::Int32(v) => Col::Int(
                            Ints::I32(
                                (0..n).map(|i| if b.is_null(i) { 0 } else { v[i] }).collect(),
                            ),
                            nulls_of(b),
                        ),
                        ColData::Int(v) => Col::Int(
                            Ints::I64(
                                (0..n).map(|i| if b.is_null(i) { 0 } else { v[i] }).collect(),
                            ),
                            nulls_of(b),
                        ),
                        ColData::Float(v) => Col::Float(
                            (0..n)
                                .map(|i| if b.is_null(i) { 0.0 } else { v[i] })
                                .collect(),
                            nulls_of(b),
                        ),
                        ColData::Bool(_) => Col::Other,
                        ColData::Str { .. } => {
                            let mut idx: FxMap<Arc<str>, u32> = FxMap::default();
                            let mut dict: Vec<Arc<str>> = Vec::new();
                            let codes = (0..n)
                                .map(|i| {
                                    if b.is_null(i) {
                                        return NULL_CODE;
                                    }
                                    let s = b.str_at(i);
                                    if let Some(&code) = idx.get(s) {
                                        code
                                    } else {
                                        let a: Arc<str> = Arc::from(s);
                                        dict.push(a.clone());
                                        let code = (dict.len() - 1) as u32;
                                        idx.insert(a, code);
                                        code
                                    }
                                })
                                .collect();
                            rev = Some(idx);
                            Col::Text(codes, dict)
                        }
                    },
                    _ => Col::Int(Ints::I8(vec![0; n]), vec![true; n]),
                };
                (c.clone(), col, rev)
            })
            .collect();
        let mut cols = HashMap::with_capacity(built.len());
        let mut text_index = HashMap::new();
        for (name, col, rev) in built {
            if let Some(r) = rev {
                text_index.insert(name.clone(), r);
            }
            cols.insert(name, col);
        }
        let mut t = Self {
            len: n,
            col_order: columns.to_vec(),
            cols,
            text_index,
            zones: HashMap::new(),
        };
        t.refresh_zones(0);
        t
    }

    pub fn append_batch(&mut self, batch: &joule_db_amorphic::column_batch::ColumnBatch) -> bool {
        use rayon::prelude::*;
        let old = self.len;
        let n = batch.len;
        let mut idxs = std::mem::take(&mut self.text_index);
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
            .map(|(name, col, idx)| {
                let bc = batch.cols.iter().find(|b| &b.name == *name);
                match col {
                    Col::Int(v, nulls) => {
                        v.reserve(n);
                        for i in 0..n {
                            match cell_from_batch(bc, i) {
                                Cell::I(x) => {
                                    v.push_i64(x);
                                    if !nulls.is_empty() {
                                        nulls.push(false);
                                    }
                                }
                                Cell::Null => {
                                    if nulls.is_empty() {
                                        *nulls = vec![false; old + i];
                                    }
                                    v.push_i64(0);
                                    nulls.push(true);
                                }
                                _ => return false,
                            }
                        }
                    }
                    Col::Float(v, nulls) => {
                        v.reserve(n);
                        for i in 0..n {
                            let x = match cell_from_batch(bc, i) {
                                Cell::F(f) => f,
                                Cell::I(x) => x as f64,
                                Cell::Null => {
                                    if nulls.is_empty() {
                                        *nulls = vec![false; old + i];
                                    }
                                    v.push(0.0);
                                    nulls.push(true);
                                    continue;
                                }
                                _ => return false,
                            };
                            v.push(x);
                            if !nulls.is_empty() {
                                nulls.push(false);
                            }
                        }
                    }
                    Col::Text(codes, dict) => {
                        let Some(idx) = idx.as_mut() else {
                            return false;
                        };
                        codes.reserve(n);
                        for i in 0..n {
                            match cell_from_batch(bc, i) {
                                Cell::S(s) => {
                                    let code = match idx.get(s) {
                                        Some(&c) => c,
                                        None => {
                                            let a: Arc<str> = Arc::from(s);
                                            dict.push(a.clone());
                                            let c = (dict.len() - 1) as u32;
                                            idx.insert(a, c);
                                            c
                                        }
                                    };
                                    codes.push(code);
                                }
                                Cell::Null => codes.push(NULL_CODE),
                                _ => return false,
                            }
                        }
                    }
                    Col::Other => {}
                }
                true
            })
            .collect::<Vec<bool>>()
            .into_iter()
            .all(|b| b);
        for (name, _, idx) in work {
            if let Some(i) = idx {
                idxs.insert(name.clone(), i);
            }
        }
        self.text_index = idxs;
        self.len = old + n;
        if ok {
            self.refresh_zones(old);
        }
        ok
    }
}

#[derive(Clone, Copy)]
enum Cell<'a> {
    Null,
    I(i64),
    F(f64),
    S(&'a str),
    B,
}

fn cell_from_batch<'a>(
    c: Option<&'a joule_db_amorphic::column_batch::BatchColumn>,
    i: usize,
) -> Cell<'a> {
    use joule_db_amorphic::column_batch::ColData;
    let Some(c) = c else { return Cell::Null };
    if c.is_null(i) {
        return Cell::Null;
    }
    match &c.data {
        ColData::Int8(v) => Cell::I(v[i] as i64),
        ColData::Int16(v) => Cell::I(v[i] as i64),
        ColData::Int32(v) => Cell::I(v[i] as i64),
        ColData::Int(v) => Cell::I(v[i]),
        ColData::Float(v) => Cell::F(v[i]),
        ColData::Bool(_) => Cell::B,
        ColData::Str { .. } => Cell::S(c.str_at(i)),
    }
}

pub trait FieldGet {
    fn field(&self, name: &str) -> Option<&joule_db_amorphic::Value>;
}

impl FieldGet for &joule_db_amorphic::AmorphicRecord {
    fn field(&self, name: &str) -> Option<&joule_db_amorphic::Value> {
        self.get(name)
    }
}

impl FieldGet for (u64, std::collections::HashMap<String, joule_db_amorphic::Value>) {
    fn field(&self, name: &str) -> Option<&joule_db_amorphic::Value> {
        self.1.get(name)
    }
}

fn append_col<R: FieldGet>(
    name: &str,
    col: &mut Col,
    idx: Option<&mut FxMap<Arc<str>, u32>>,
    old: usize,
    records: &[R],
) -> bool {
    use joule_db_amorphic::Value as V;
    match col {
        Col::Int(v, nulls) => {
            v.reserve(records.len());
            for (k, r) in records.iter().enumerate() {
                match r.field(name) {
                    Some(V::Int(i)) => v.push_i64(*i),
                    None | Some(V::Null) => {
                        if nulls.is_empty() {
                            *nulls = vec![false; old + k];
                        }
                        v.push_i64(0);
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
                match r.field(name) {
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
                match r.field(name) {
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

// ============================================================================
// Pure-integer civil calendar helpers (Howard Hinnant's algorithm, UTC)
// ============================================================================

/// Days since 1970-01-01 (`1970-01-01` -> `0`, `2013-07-15` -> `15901`).
#[inline]
fn days_from_civil(mut y: i32, m: u32, d: u32) -> i64 {
    if m <= 2 {
        y -= 1;
    }
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as u32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146_097 + doe as i64 - 719_468
}

/// `(year, month 1..=12, day 1..=31, doy 1..=366)` from days since 1970-01-01.
#[inline]
fn civil_from_days(z_in: i64) -> (i32, u32, u32, u32) {
    let z = z_in + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy_mar = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy_mar + 2) / 153;
    let d = doy_mar - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = (y + if m <= 2 { 1 } else { 0 }) as i32;
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let ordinal = if m >= 3 {
        doy_mar + 59 + is_leap as u32 + 1
    } else if m == 1 {
        d
    } else {
        31 + d
    };
    (year, m, d, ordinal)
}

/// Parse strict `YYYY-MM-DD` ISO date literal into days since 1970-01-01.
fn parse_iso_date_days(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let y: i32 = s[0..4].parse().ok()?;
    let m: u32 = s[5..7].parse().ok()?;
    let d: u32 = s[8..10].parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

/// Parse `YYYY-MM-DD[T ]HH:MM:SS` into Unix epoch seconds.
fn parse_iso_timestamp_secs(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 || (b[10] != b' ' && b[10] != b'T') || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let days = parse_iso_date_days(&s[0..10])?;
    let hh: u32 = s[11..13].parse().ok()?;
    let mm: u32 = s[14..16].parse().ok()?;
    let ss: u32 = s[17..19].parse().ok()?;
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    Some(days * 86_400 + hh as i64 * 3_600 + mm as i64 * 60 + ss as i64)
}

// ============================================================================
// Predicates (parsed AST -> precompiled dictionary/zone-aware `PrepPred`)
// ============================================================================

#[derive(Clone)]
enum Lit {
    N(Num),
    S(String),
}

#[derive(Clone)]
enum Pred<'a> {
    Cmp {
        col: &'a str,
        op: Operator,
        lit: Lit,
    },
    In {
        col: &'a str,
        lits: Vec<Lit>,
        negated: bool,
    },
    Like {
        col: &'a str,
        pattern: &'a str,
        negated: bool,
        ci: bool,
    },
    IsNull {
        col: &'a str,
        negated: bool,
    },
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
            Lit::N(Num::I(i)) => Some(Lit::N(Num::I(i.checked_neg()?))),
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
                    op: *op,
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
        Expression::In {
            expr,
            list,
            negated,
        } => {
            let col = column(expr)?;
            let lits: Vec<Lit> = list.iter().map(literal).collect::<Option<_>>()?;
            out.push(Pred::In {
                col,
                lits,
                negated: *negated,
            });
            Some(())
        }
        Expression::Between {
            expr,
            low,
            high,
            negated: false,
        } => {
            let col = column(expr)?;
            out.push(Pred::Cmp {
                col,
                op: Operator::Ge,
                lit: literal(low)?,
            });
            out.push(Pred::Cmp {
                col,
                op: Operator::Le,
                lit: literal(high)?,
            });
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
        Expression::IsNull { expr, negated } => {
            out.push(Pred::IsNull {
                col: column(expr)?,
                negated: *negated,
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

/// Precompiled predicate: resolves column references and dictionary bitmaps ONCE
/// before parallel morsel/zone execution.
enum PrepPred<'a> {
    IntCmp {
        col_name: &'a str,
        ints: &'a Ints,
        nulls: &'a [bool],
        op: Operator,
        val: i64,
    },
    IntCmpFloat {
        col_name: &'a str,
        ints: &'a Ints,
        nulls: &'a [bool],
        op: Operator,
        val: f64,
    },
    FloatCmp {
        col_name: &'a str,
        vals: &'a [f64],
        nulls: &'a [bool],
        op: Operator,
        val: f64,
    },
    IntIn {
        ints: &'a Ints,
        nulls: &'a [bool],
        set: Vec<i64>,
        negated: bool,
    },
    FloatIn {
        vals: &'a [f64],
        nulls: &'a [bool],
        set: Vec<f64>,
        negated: bool,
    },
    /// `col = 'lit'` on dictionary text: single `u32` code check (None => no non-null row matches)
    TextEqCode {
        codes: &'a [u32],
        code: Option<u32>,
    },
    /// `col <> 'lit'` on dictionary text: `c != NULL_CODE && Some(c) != code`
    TextNeCode {
        codes: &'a [u32],
        code: Option<u32>,
    },
    /// General dictionary-level boolean table (`LIKE`, `<`, `<=`, `>`, `>=`, `IN`)
    TextMask {
        codes: &'a [u32],
        ok: Vec<bool>,
    },
    IsNull {
        col: &'a Col,
        negated: bool,
    },
}

fn coerce_iso_temporal_for_int(t: &TableColumns, col_name: &str, s: &str) -> Option<i64> {
    if let Some(days) = parse_iso_date_days(s) {
        // Epoch-day integer column (e.g. ClickBench `EventDate` = 15901 for 2013-07-15)
        // or epoch-second column if values exceed 100,000.
        if let Some(zs) = t.zones.get(col_name) {
            let max_v = zs.iter().filter_map(|z| z.hi).fold(None, |acc, h| match h {
                Num::I(v) => Some(acc.map_or(v, |a: i64| a.max(v))),
                _ => acc,
            });
            if let Some(mv) = max_v
                && mv > 200_000
            {
                return Some(days * 86_400);
            }
        }
        return Some(days);
    }
    parse_iso_timestamp_secs(s)
}

fn prepare_preds<'a>(t: &'a TableColumns, preds: &[Pred<'a>]) -> Option<Vec<PrepPred<'a>>> {
    use rayon::prelude::*;
    let mut out = Vec::with_capacity(preds.len());
    for p in preds {
        match p {
            Pred::Cmp { col, op, lit } => {
                let c = t.cols.get(*col)?;
                match (c, lit) {
                    (Col::Int(ints, nulls), Lit::N(Num::I(x))) => out.push(PrepPred::IntCmp {
                        col_name: col,
                        ints,
                        nulls,
                        op: *op,
                        val: *x,
                    }),
                    (Col::Int(ints, nulls), Lit::N(Num::F(x))) => out.push(PrepPred::IntCmpFloat {
                        col_name: col,
                        ints,
                        nulls,
                        op: *op,
                        val: *x,
                    }),
                    (Col::Int(ints, nulls), Lit::S(s)) => {
                        let epoch_i = coerce_iso_temporal_for_int(t, col, s)?;
                        out.push(PrepPred::IntCmp {
                            col_name: col,
                            ints,
                            nulls,
                            op: *op,
                            val: epoch_i,
                        });
                    }
                    (Col::Float(vals, nulls), Lit::N(n)) => {
                        let val = match n {
                            Num::I(i) => *i as f64,
                            Num::F(f) => *f,
                        };
                        out.push(PrepPred::FloatCmp {
                            col_name: col,
                            vals,
                            nulls,
                            op: *op,
                            val,
                        });
                    }
                    (Col::Text(codes, dict), Lit::S(x)) => match op {
                        Operator::Eq => {
                            let code = if let Some(rev) = t.text_index.get(*col) {
                                rev.get(x.as_str()).copied()
                            } else {
                                dict.iter().position(|d| &**d == x.as_str()).map(|i| i as u32)
                            };
                            out.push(PrepPred::TextEqCode { codes, code });
                        }
                        Operator::Ne => {
                            let code = if let Some(rev) = t.text_index.get(*col) {
                                rev.get(x.as_str()).copied()
                            } else {
                                dict.iter().position(|d| &**d == x.as_str()).map(|i| i as u32)
                            };
                            out.push(PrepPred::TextNeCode { codes, code });
                        }
                        _ => {
                            let x_s = x.as_str();
                            let ok: Vec<bool> = if dict.len() >= 4096 {
                                dict.par_iter()
                                    .map(|d| cmp_ok((**d).cmp(x_s), op))
                                    .collect()
                            } else {
                                dict.iter().map(|d| cmp_ok((**d).cmp(x_s), op)).collect()
                            };
                            out.push(PrepPred::TextMask { codes, ok });
                        }
                    },
                    _ => return None,
                }
            }
            Pred::In { col, lits, negated } => {
                let c = t.cols.get(*col)?;
                match c {
                    Col::Int(ints, nulls) => {
                        let mut set = Vec::with_capacity(lits.len());
                        for l in lits {
                            match l {
                                Lit::N(Num::I(x)) => set.push(*x),
                                Lit::S(s) => set.push(coerce_iso_temporal_for_int(t, col, s)?),
                                _ => return None,
                            }
                        }
                        out.push(PrepPred::IntIn {
                            ints,
                            nulls,
                            set,
                            negated: *negated,
                        });
                    }
                    Col::Float(vals, nulls) => {
                        let mut set = Vec::with_capacity(lits.len());
                        for l in lits {
                            match l {
                                Lit::N(Num::I(x)) => set.push(*x as f64),
                                Lit::N(Num::F(f)) => set.push(*f),
                                _ => return None,
                            }
                        }
                        out.push(PrepPred::FloatIn {
                            vals,
                            nulls,
                            set,
                            negated: *negated,
                        });
                    }
                    Col::Text(codes, dict) => {
                        let want: HashSet<&str> = lits
                            .iter()
                            .map(|l| match l {
                                Lit::S(s) => Some(s.as_str()),
                                _ => None,
                            })
                            .collect::<Option<_>>()?;
                        let neg = *negated;
                        let ok: Vec<bool> = dict
                            .iter()
                            .map(|d| want.contains(&**d) != neg)
                            .collect();
                        out.push(PrepPred::TextMask { codes, ok });
                    }
                    Col::Other => return None,
                }
            }
            Pred::Like {
                col,
                pattern,
                negated,
                ci,
            } => match t.cols.get(*col)? {
                Col::Text(codes, dict) => {
                    let (pat, neg, ci) = (*pattern, *negated, *ci);
                    let ok: Vec<bool> = if dict.len() >= 2048 {
                        dict.par_iter()
                            .map(|d| joule_db_query::like::like(d, pat, ci) != neg)
                            .collect()
                    } else {
                        dict.iter()
                            .map(|d| joule_db_query::like::like(d, pat, ci) != neg)
                            .collect()
                    };
                    out.push(PrepPred::TextMask { codes, ok });
                }
                _ => return None,
            },
            Pred::IsNull { col, negated } => {
                let c = t.cols.get(*col)?;
                if matches!(c, Col::Other) {
                    return None;
                }
                out.push(PrepPred::IsNull {
                    col: c,
                    negated: *negated,
                });
            }
        }
    }
    Some(out)
}

#[inline]
fn eval_prep_pred_row(p: &PrepPred, i: usize) -> bool {
    match p {
        PrepPred::IntCmp {
            ints,
            nulls,
            op,
            val,
            ..
        } => !nl(nulls, i) && cmp_ok(ints.at(i).cmp(val), op),
        PrepPred::IntCmpFloat {
            ints,
            nulls,
            op,
            val,
            ..
        } => !nl(nulls, i) && (ints.at(i) as f64).partial_cmp(val).is_some_and(|o| cmp_ok(o, op)),
        PrepPred::FloatCmp {
            vals,
            nulls,
            op,
            val,
            ..
        } => !nl(nulls, i) && vals[i].partial_cmp(val).is_some_and(|o| cmp_ok(o, op)),
        PrepPred::IntIn {
            ints,
            nulls,
            set,
            negated,
        } => !nl(nulls, i) && (set.contains(&ints.at(i)) != *negated),
        PrepPred::FloatIn {
            vals,
            nulls,
            set,
            negated,
        } => !nl(nulls, i) && (set.iter().any(|x| *x == vals[i]) != *negated),
        PrepPred::TextEqCode { codes, code } => {
            let c = codes[i];
            c != NULL_CODE && Some(c) == *code
        }
        PrepPred::TextNeCode { codes, code } => {
            let c = codes[i];
            c != NULL_CODE && Some(c) != *code
        }
        PrepPred::TextMask { codes, ok } => {
            let c = codes[i];
            c != NULL_CODE && ok[c as usize]
        }
        PrepPred::IsNull { col, negated } => is_null(col, i) != *negated,
    }
}

fn prep_zone_verdict(t: &TableColumns, p: &PrepPred, s: usize, e: usize) -> Option<bool> {
    match p {
        PrepPred::IntCmp {
            col_name, op, val, ..
        } => zone_verdict(t, col_name, op, Num::I(*val), s, e),
        PrepPred::IntCmpFloat {
            col_name, op, val, ..
        } => zone_verdict(t, col_name, op, Num::F(*val), s, e),
        PrepPred::FloatCmp {
            col_name, op, val, ..
        } => zone_verdict(t, col_name, op, Num::F(*val), s, e),
        PrepPred::TextEqCode { code: None, .. } => Some(false),
        _ => None,
    }
}

fn mask_prep_pred(p: &PrepPred, s: usize, mask: &mut [bool]) {
    let e = s + mask.len();
    match p {
        PrepPred::IntCmp {
            ints,
            nulls,
            op,
            val,
            ..
        } => {
            let x = *val;
            with_ints!(ints, |v| {
                let vs = &v[s..e];
                macro_rules! go {
                    ($f:expr) => {
                        for (m, &raw) in mask.iter_mut().zip(vs) {
                            *m &= $f(raw as i64);
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
                    _ => {
                        for m in mask.iter_mut() {
                            *m = false;
                        }
                    }
                }
            });
            if !nulls.is_empty() {
                for (m, &n) in mask.iter_mut().zip(&nulls[s..e]) {
                    *m &= !n;
                }
            }
        }
        PrepPred::TextEqCode { codes, code } => match code {
            None => mask.fill(false),
            Some(target) => {
                let t = *target;
                for (m, &c) in mask.iter_mut().zip(&codes[s..e]) {
                    *m &= c == t;
                }
            }
        },
        PrepPred::TextNeCode { codes, code } => match code {
            None => {
                for (m, &c) in mask.iter_mut().zip(&codes[s..e]) {
                    *m &= c != NULL_CODE;
                }
            }
            Some(target) => {
                let t = *target;
                for (m, &c) in mask.iter_mut().zip(&codes[s..e]) {
                    *m &= c != NULL_CODE && c != t;
                }
            }
        },
        PrepPred::TextMask { codes, ok } => {
            for (m, &c) in mask.iter_mut().zip(&codes[s..e]) {
                *m &= c != NULL_CODE && ok[c as usize];
            }
        }
        _ => {
            for (k, m) in mask.iter_mut().enumerate() {
                if *m {
                    *m = eval_prep_pred_row(p, s + k);
                }
            }
        }
    }
}

fn morsel_sel_prep(t: &TableColumns, preds: &[PrepPred], s: usize, e: usize) -> Option<Vec<u32>> {
    if preds.is_empty() {
        return None;
    }
    let mut live: Vec<&PrepPred> = Vec::with_capacity(preds.len());
    for p in preds {
        match prep_zone_verdict(t, p, s, e) {
            Some(false) => return Some(Vec::new()),
            Some(true) => continue,
            None => live.push(p),
        }
    }
    if live.is_empty() {
        return None;
    }
    let mut sel: Vec<u32> = (s as u32..e as u32).collect();
    for p in live {
        match p {
            PrepPred::IntCmp {
                ints,
                nulls,
                op,
                val,
                ..
            } => {
                let x = *val;
                with_ints!(ints, |v| {
                    sel.retain(|&i| !nl(nulls, i as usize) && cmp_ok((v[i as usize] as i64).cmp(&x), op))
                });
            }
            PrepPred::TextEqCode { codes, code } => match code {
                None => {
                    sel.clear();
                    return Some(sel);
                }
                Some(tcode) => {
                    let tc = *tcode;
                    sel.retain(|&i| codes[i as usize] == tc);
                }
            },
            PrepPred::TextNeCode { codes, code } => match code {
                None => sel.retain(|&i| codes[i as usize] != NULL_CODE),
                Some(tcode) => {
                    let tc = *tcode;
                    sel.retain(|&i| {
                        let c = codes[i as usize];
                        c != NULL_CODE && c != tc
                    });
                }
            },
            PrepPred::TextMask { codes, ok } => {
                sel.retain(|&i| {
                    let c = codes[i as usize];
                    c != NULL_CODE && ok[c as usize]
                });
            }
            _ => {
                sel.retain(|&i| eval_prep_pred_row(p, i as usize));
            }
        }
        if sel.is_empty() {
            break;
        }
    }
    Some(sel)
}

// ============================================================================
// Virtual / Computed Scalar Columns (`VCol`)
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArithOp {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExtractPart {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Dow,
    Doy,
    Epoch,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TruncPart {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

enum TextLeaf<'a> {
    Col(&'a [u32], &'a [Arc<str>]),
    Lit(Arc<str>, u32),
    Null,
}

/// Zero-materialization scalar expression column over `TableColumns`.
enum VCol<'a> {
    Phys(&'a Col),
    ConstInt(i64),
    ConstFloat(f64),
    ConstText(Arc<str>),
    IntArith {
        ints: &'a Ints,
        nulls: &'a [bool],
        op: ArithOp,
        rhs: i64,
    },
    ExtractInt {
        ints: &'a Ints,
        nulls: &'a [bool],
        part: ExtractPart,
    },
    DateTruncInt {
        ints: &'a Ints,
        nulls: &'a [bool],
        part: TruncPart,
    },
    /// `STRLEN(col)` / `LENGTH(col)` precomputed per dictionary entry (zero per-row alloc).
    DictStrLen {
        codes: &'a [u32],
        lens: Vec<i32>,
    },
    /// `REGEXP_REPLACE` / `LOWER` / `UPPER` / `TRIM` precomputed once per dictionary entry.
    DictRemap {
        codes: &'a [u32],
        remap: Vec<u32>,
        dict: Vec<Arc<str>>,
    },
    /// `CASE WHEN <preds> THEN <text> ELSE <text> END` evaluated lazily on surviving rows.
    CaseText {
        branches: Vec<(Vec<PrepPred<'a>>, TextLeaf<'a>)>,
        else_leaf: TextLeaf<'a>,
        lit_dict: Vec<Arc<str>>,
    },
}

#[inline]
fn extract_epoch(ts: i64, part: ExtractPart) -> i64 {
    match part {
        ExtractPart::Second => ts.rem_euclid(60),
        ExtractPart::Minute => ts.div_euclid(60).rem_euclid(60),
        ExtractPart::Hour => ts.div_euclid(3_600).rem_euclid(24),
        ExtractPart::Epoch => ts,
        ExtractPart::Dow => (ts.div_euclid(86_400) + 4).rem_euclid(7),
        ExtractPart::Year | ExtractPart::Month | ExtractPart::Day | ExtractPart::Doy => {
            let days = ts.div_euclid(86_400);
            let (y, m, d, doy) = civil_from_days(days);
            match part {
                ExtractPart::Year => y as i64,
                ExtractPart::Month => m as i64,
                ExtractPart::Day => d as i64,
                ExtractPart::Doy => doy as i64,
                _ => 0,
            }
        }
    }
}

#[inline]
fn trunc_epoch(ts: i64, part: TruncPart) -> i64 {
    match part {
        TruncPart::Second => ts,
        TruncPart::Minute => ts - ts.rem_euclid(60),
        TruncPart::Hour => ts - ts.rem_euclid(3_600),
        TruncPart::Day => ts - ts.rem_euclid(86_400),
        TruncPart::Month => {
            let (y, m, _, _) = civil_from_days(ts.div_euclid(86_400));
            days_from_civil(y, m, 1) * 86_400
        }
        TruncPart::Year => {
            let (y, _, _, _) = civil_from_days(ts.div_euclid(86_400));
            days_from_civil(y, 1, 1) * 86_400
        }
    }
}

impl<'a> VCol<'a> {
    fn is_int_like(&self) -> bool {
        matches!(
            self,
            VCol::Phys(Col::Int(..))
                | VCol::ConstInt(_)
                | VCol::IntArith { .. }
                | VCol::ExtractInt { .. }
                | VCol::DateTruncInt { .. }
                | VCol::DictStrLen { .. }
        )
    }

    fn is_numeric(&self) -> bool {
        self.is_int_like() || matches!(self, VCol::Phys(Col::Float(..)) | VCol::ConstFloat(_))
    }

    fn is_text_like(&self) -> bool {
        matches!(
            self,
            VCol::Phys(Col::Text(..))
                | VCol::ConstText(_)
                | VCol::DictRemap { .. }
                | VCol::CaseText { .. }
        )
    }

    #[inline]
    fn is_null_at(&self, i: usize) -> bool {
        match self {
            VCol::Phys(c) => is_null(c, i),
            VCol::ConstInt(_) | VCol::ConstFloat(_) | VCol::ConstText(_) => false,
            VCol::IntArith { nulls, .. }
            | VCol::ExtractInt { nulls, .. }
            | VCol::DateTruncInt { nulls, .. } => nl(nulls, i),
            VCol::DictStrLen { codes, .. } => codes[i] == NULL_CODE,
            VCol::DictRemap { codes, remap, .. } => {
                let c = codes[i];
                c == NULL_CODE || remap[c as usize] == NULL_CODE
            }
            VCol::CaseText { .. } => matches!(self.sv_at(i), Sv::Null),
        }
    }

    #[inline]
    fn int_at(&self, i: usize) -> Option<i64> {
        match self {
            VCol::Phys(Col::Int(v, nulls)) => (!nl(nulls, i)).then(|| v.at(i)),
            VCol::ConstInt(c) => Some(*c),
            VCol::IntArith {
                ints,
                nulls,
                op,
                rhs,
            } => {
                if nl(nulls, i) {
                    return None;
                }
                let x = ints.at(i);
                Some(match op {
                    ArithOp::Add => x.wrapping_add(*rhs),
                    ArithOp::Sub => x.wrapping_sub(*rhs),
                    ArithOp::Mul => x.wrapping_mul(*rhs),
                })
            }
            VCol::ExtractInt { ints, nulls, part } => {
                (!nl(nulls, i)).then(|| extract_epoch(ints.at(i), *part))
            }
            VCol::DateTruncInt { ints, nulls, part } => {
                (!nl(nulls, i)).then(|| trunc_epoch(ints.at(i), *part))
            }
            VCol::DictStrLen { codes, lens } => {
                let c = codes[i];
                (c != NULL_CODE).then(|| lens[c as usize] as i64)
            }
            _ => None,
        }
    }

    #[inline]
    fn float_at(&self, i: usize) -> Option<f64> {
        match self {
            VCol::Phys(Col::Float(v, nulls)) => (!nl(nulls, i)).then(|| v[i]),
            VCol::ConstFloat(f) => Some(*f),
            _ => self.int_at(i).map(|x| x as f64),
        }
    }

    #[inline]
    fn str_ref_at(&'a self, i: usize) -> Option<&'a str> {
        match self {
            VCol::Phys(Col::Text(codes, dict)) => {
                let c = codes[i];
                (c != NULL_CODE).then(|| dict[c as usize].as_ref())
            }
            VCol::ConstText(s) => Some(s.as_ref()),
            VCol::DictRemap { codes, remap, dict } => {
                let c = codes[i];
                if c == NULL_CODE {
                    return None;
                }
                let rc = remap[c as usize];
                (rc != NULL_CODE).then(|| dict[rc as usize].as_ref())
            }
            VCol::CaseText {
                branches,
                else_leaf,
                ..
            } => {
                for (preds, leaf) in branches {
                    if preds.iter().all(|p| eval_prep_pred_row(p, i)) {
                        return match leaf {
                            TextLeaf::Col(codes, dict) => {
                                let c = codes[i];
                                (c != NULL_CODE).then(|| dict[c as usize].as_ref())
                            }
                            TextLeaf::Lit(s, _) => Some(s.as_ref()),
                            TextLeaf::Null => None,
                        };
                    }
                }
                match else_leaf {
                    TextLeaf::Col(codes, dict) => {
                        let c = codes[i];
                        (c != NULL_CODE).then(|| dict[c as usize].as_ref())
                    }
                    TextLeaf::Lit(s, _) => Some(s.as_ref()),
                    TextLeaf::Null => None,
                }
            }
            _ => None,
        }
    }

    #[inline]
    fn arc_str_at(&self, i: usize) -> Option<Arc<str>> {
        match self {
            VCol::Phys(Col::Text(codes, dict)) => {
                let c = codes[i];
                (c != NULL_CODE).then(|| dict[c as usize].clone())
            }
            VCol::ConstText(s) => Some(s.clone()),
            VCol::DictRemap { codes, remap, dict } => {
                let c = codes[i];
                if c == NULL_CODE {
                    return None;
                }
                let rc = remap[c as usize];
                (rc != NULL_CODE).then(|| dict[rc as usize].clone())
            }
            VCol::CaseText {
                branches,
                else_leaf,
                ..
            } => {
                let eval_leaf = |leaf: &TextLeaf<'a>| match leaf {
                    TextLeaf::Col(codes, dict) => {
                        let c = codes[i];
                        (c != NULL_CODE).then(|| dict[c as usize].clone())
                    }
                    TextLeaf::Lit(s, _) => Some(s.clone()),
                    TextLeaf::Null => None,
                };
                for (preds, leaf) in branches {
                    if preds.iter().all(|p| eval_prep_pred_row(p, i)) {
                        return eval_leaf(leaf);
                    }
                }
                eval_leaf(else_leaf)
            }
            _ => None,
        }
    }

    /// Unique `(code, is_null)` per distinct value of this column for `GROUP BY` / `DISTINCT`.
    #[inline]
    fn group_code_at(&self, i: usize) -> (u64, bool) {
        match self {
            VCol::Phys(c) => {
                let code = code_of(c, i);
                (code, is_null_code(c, code, i))
            }
            VCol::ConstInt(x) => (*x as u64, false),
            VCol::ConstFloat(f) => (f.to_bits(), false),
            VCol::ConstText(_) => (0, false),
            VCol::IntArith { .. }
            | VCol::ExtractInt { .. }
            | VCol::DateTruncInt { .. }
            | VCol::DictStrLen { .. } => match self.int_at(i) {
                Some(v) => (v as u64, false),
                None => (u64::MAX, true),
            },
            VCol::DictRemap { codes, remap, .. } => {
                let c = codes[i];
                if c == NULL_CODE {
                    (NULL_CODE as u64, true)
                } else {
                    let rc = remap[c as usize];
                    (rc as u64, rc == NULL_CODE)
                }
            }
            VCol::CaseText {
                branches,
                else_leaf,
                lit_dict,
            } => {
                let code_leaf = |leaf: &TextLeaf<'a>| -> (u64, bool) {
                    match leaf {
                        TextLeaf::Col(codes, _) => {
                            let c = codes[i];
                            if c == NULL_CODE {
                                (u64::MAX, true)
                            } else {
                                ((c as u64) + lit_dict.len() as u64, false)
                            }
                        }
                        TextLeaf::Lit(_, tag) => (*tag as u64, false),
                        TextLeaf::Null => (u64::MAX, true),
                    }
                };
                for (preds, leaf) in branches {
                    if preds.iter().all(|p| eval_prep_pred_row(p, i)) {
                        return code_leaf(leaf);
                    }
                }
                code_leaf(else_leaf)
            }
        }
    }

    #[inline]
    fn sv_at(&self, i: usize) -> Sv {
        if let Some(x) = self.int_at(i) {
            return Sv::I(x);
        }
        if matches!(self, VCol::Phys(Col::Float(..)) | VCol::ConstFloat(_))
            && let Some(f) = self.float_at(i)
        {
            return Sv::F(f);
        }
        if let Some(s) = self.arc_str_at(i) {
            return Sv::S(s);
        }
        Sv::Null
    }
}

fn build_vcol<'a>(t: &'a TableColumns, e: &'a Expression) -> Option<VCol<'a>> {
    use rayon::prelude::*;
    if let Some(c) = column(e) {
        let col = t.cols.get(c)?;
        if matches!(col, Col::Other) {
            return None;
        }
        return Some(VCol::Phys(col));
    }
    match e {
        Expression::Literal(AstValue::Int(i)) => Some(VCol::ConstInt(*i)),
        Expression::Literal(AstValue::Float(f)) => Some(VCol::ConstFloat(*f)),
        Expression::Literal(AstValue::String(s)) => Some(VCol::ConstText(Arc::from(s.as_str()))),
        Expression::Unary {
            op: UnaryOperator::Neg,
            expr,
        } => match build_vcol(t, expr)? {
            VCol::ConstInt(i) => Some(VCol::ConstInt(i.checked_neg()?)),
            VCol::ConstFloat(f) => Some(VCol::ConstFloat(-f)),
            _ => None,
        },
        Expression::Binary { left, op, right } => {
            let arith = match op {
                Operator::Add => ArithOp::Add,
                Operator::Sub => ArithOp::Sub,
                Operator::Mul => ArithOp::Mul,
                _ => return None,
            };
            if let (Some(c), Some(Lit::N(Num::I(rhs)))) = (column(left), literal(right)) {
                let Col::Int(ints, nulls) = t.cols.get(c)? else {
                    return None;
                };
                return Some(VCol::IntArith {
                    ints,
                    nulls,
                    op: arith,
                    rhs,
                });
            }
            if matches!(arith, ArithOp::Add | ArithOp::Mul)
                && let (Some(Lit::N(Num::I(lhs))), Some(c)) = (literal(left), column(right))
            {
                let Col::Int(ints, nulls) = t.cols.get(c)? else {
                    return None;
                };
                return Some(VCol::IntArith {
                    ints,
                    nulls,
                    op: arith,
                    rhs: lhs,
                });
            }
            None
        }
        Expression::Function { name, args } => {
            let upper = name.to_ascii_uppercase();
            match upper.as_str() {
                "EXTRACT" | "DATE_PART" if args.len() == 2 => {
                    let Expression::Literal(AstValue::String(p)) = &args[0] else {
                        return None;
                    };
                    let part = match p.to_ascii_uppercase().as_str() {
                        "YEAR" => ExtractPart::Year,
                        "MONTH" => ExtractPart::Month,
                        "DAY" => ExtractPart::Day,
                        "HOUR" => ExtractPart::Hour,
                        "MINUTE" => ExtractPart::Minute,
                        "SECOND" => ExtractPart::Second,
                        "DOW" | "DAYOFWEEK" => ExtractPart::Dow,
                        "DOY" | "DAYOFYEAR" => ExtractPart::Doy,
                        "EPOCH" => ExtractPart::Epoch,
                        _ => return None,
                    };
                    let Col::Int(ints, nulls) = t.cols.get(column(&args[1])?)? else {
                        return None;
                    };
                    Some(VCol::ExtractInt { ints, nulls, part })
                }
                "DATE_TRUNC" if args.len() == 2 => {
                    let Expression::Literal(AstValue::String(p)) = &args[0] else {
                        return None;
                    };
                    let part = match p.to_ascii_uppercase().as_str() {
                        "YEAR" => TruncPart::Year,
                        "MONTH" => TruncPart::Month,
                        "DAY" => TruncPart::Day,
                        "HOUR" => TruncPart::Hour,
                        "MINUTE" => TruncPart::Minute,
                        "SECOND" => TruncPart::Second,
                        _ => return None,
                    };
                    let Col::Int(ints, nulls) = t.cols.get(column(&args[1])?)? else {
                        return None;
                    };
                    Some(VCol::DateTruncInt { ints, nulls, part })
                }
                "STRLEN" | "OCTET_LENGTH" | "LENGTH" | "LEN" | "CHAR_LENGTH" | "CHARACTER_LENGTH"
                    if args.len() == 1 =>
                {
                    let Col::Text(codes, dict) = t.cols.get(column(&args[0])?)? else {
                        return None;
                    };
                    let byte_len = matches!(upper.as_str(), "STRLEN" | "OCTET_LENGTH");
                    let lens: Vec<i32> = if dict.len() >= 2048 {
                        dict.par_iter()
                            .map(|d| {
                                if byte_len {
                                    d.len() as i32
                                } else {
                                    d.chars().count() as i32
                                }
                            })
                            .collect()
                    } else {
                        dict.iter()
                            .map(|d| {
                                if byte_len {
                                    d.len() as i32
                                } else {
                                    d.chars().count() as i32
                                }
                            })
                            .collect()
                    };
                    Some(VCol::DictStrLen { codes, lens })
                }
                "REGEXP_REPLACE" if (3..=4).contains(&args.len()) => {
                    let Col::Text(codes, dict) = t.cols.get(column(&args[0])?)? else {
                        return None;
                    };
                    let Expression::Literal(AstValue::String(pattern)) = &args[1] else {
                        return None;
                    };
                    let Expression::Literal(AstValue::String(replacement)) = &args[2] else {
                        return None;
                    };
                    let global = match args.get(3) {
                        None => false,
                        Some(Expression::Literal(AstValue::String(flags))) => flags.contains('g'),
                        _ => return None,
                    };
                    let re = regex::Regex::new(pattern).ok()?;
                    let rust_repl =
                        joule_db_query::functions::normalize_posix_regex_replacement(replacement);
                    let replaced: Vec<String> = if dict.len() >= 1024 {
                        dict.par_iter()
                            .map(|d| {
                                if global {
                                    re.replace_all(d, rust_repl.as_str()).into_owned()
                                } else {
                                    re.replace(d, rust_repl.as_str()).into_owned()
                                }
                            })
                            .collect()
                    } else {
                        dict.iter()
                            .map(|d| {
                                if global {
                                    re.replace_all(d, rust_repl.as_str()).into_owned()
                                } else {
                                    re.replace(d, rust_repl.as_str()).into_owned()
                                }
                            })
                            .collect()
                    };
                    let mut new_idx: FxMap<Arc<str>, u32> = FxMap::default();
                    let mut new_dict: Vec<Arc<str>> = Vec::new();
                    let mut remap = Vec::with_capacity(replaced.len());
                    for s in replaced {
                        if let Some(&c) = new_idx.get(s.as_str()) {
                            remap.push(c);
                        } else {
                            let a: Arc<str> = Arc::from(s.as_str());
                            new_dict.push(a.clone());
                            let c = (new_dict.len() - 1) as u32;
                            new_idx.insert(a, c);
                            remap.push(c);
                        }
                    }
                    Some(VCol::DictRemap {
                        codes,
                        remap,
                        dict: new_dict,
                    })
                }
                _ => None,
            }
        }
        Expression::Case {
            operand: None,
            when_clauses,
            else_clause,
        } => {
            // Support searched CASE WHEN returning text columns or string/null literals
            // (covers ClickBench Q39 cleanly with zero full-table allocation!).
            let mut single_phys_col: Option<&str> = None;
            let mut lit_dict: Vec<Arc<str>> = Vec::new();
            let mut build_leaf = |ex: &'a Expression| -> Option<TextLeaf<'a>> {
                if let Some(cn) = column(ex) {
                    if let Some(prev) = single_phys_col
                        && prev != cn
                    {
                        // group_code_at assumes at most one physical text column mixed with literals
                        return None;
                    }
                    single_phys_col = Some(cn);
                    let Col::Text(codes, dict) = t.cols.get(cn)? else {
                        return None;
                    };
                    if let Some(rev) = t.text_index.get(cn) {
                        // check if literals overlap dict codes? Better: if a literal exists in `dict`,
                        // give it `dict_code + lit_dict_offset`! To guarantee unique group codes even
                        // when a literal string equals a `Referer` entry (e.g. `''`), look up in `dict`!
                        let _ = rev;
                    }
                    return Some(TextLeaf::Col(codes, dict));
                }
                match ex {
                    Expression::Literal(AstValue::String(s)) => {
                        let a: Arc<str> = Arc::from(s.as_str());
                        let pos = lit_dict
                            .iter()
                            .position(|x| **x == *s)
                            .unwrap_or_else(|| {
                                lit_dict.push(a.clone());
                                lit_dict.len() - 1
                            });
                        Some(TextLeaf::Lit(a, pos as u32))
                    }
                    Expression::Literal(AstValue::Null) => Some(TextLeaf::Null),
                    _ => None,
                }
            };
            let mut branches = Vec::with_capacity(when_clauses.len());
            for (cond, then_ex) in when_clauses {
                let mut raw_p = Vec::new();
                collect_preds(cond, &mut raw_p)?;
                let prep = prepare_preds(t, &raw_p)?;
                let leaf = build_leaf(then_ex)?;
                branches.push((prep, leaf));
            }
            let else_leaf = match else_clause {
                Some(el) => build_leaf(el)?,
                None => TextLeaf::Null,
            };
            // Ensure literal strings that also appear in `single_phys_col`'s dictionary map to
            // the SAME group code! If `single_phys_col` dictionary contains `lit`, rewrite that
            // `TextLeaf::Lit` tag if needed—or simpler: remap literal tags that match `dict`!
            if let Some(cn) = single_phys_col
                && let Some(Col::Text(_, dict)) = t.cols.get(cn)
            {
                let fix_leaf = |leaf: &mut TextLeaf<'a>| {
                    if let TextLeaf::Lit(s, tag) = leaf {
                        if let Some(di) = dict.iter().position(|d| **d == **s) {
                            *tag = (di as u32) + (lit_dict.len() as u32);
                        }
                    }
                };
                for (_, leaf) in &mut branches {
                    fix_leaf(leaf);
                }
                let mut el = else_leaf;
                fix_leaf(&mut el);
                return Some(VCol::CaseText {
                    branches,
                    else_leaf: el,
                    lit_dict,
                });
            }
            Some(VCol::CaseText {
                branches,
                else_leaf,
                lit_dict,
            })
        }
        _ => None,
    }
}

// ============================================================================
// Aggregators, Group Keys, Zone Metadata & Morsel Engine
// ============================================================================

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

struct AggSpec<'a> {
    kind: AggKind,
    vcol: Option<VCol<'a>>,
    /// Algebraic offset for `SUM(col ± k)` / `AVG(col ± k)` / `MIN(col ± k)` / `MAX(col ± k)`
    /// so `Q29`'s 90 `SUM(ResolutionWidth + k)` projections share ONE column scan!
    const_add: i64,
}

enum OutCol {
    Group(usize),
    Agg(usize),
}

#[derive(Clone, Default)]
struct Acc {
    count: u64,
    isum: i128,
    fsum: f64,
    fcomp: f64,
    min: Option<Num>,
    max: Option<Num>,
    smin: Option<Arc<str>>,
    smax: Option<Arc<str>>,
    distinct: Option<Box<HashSet<u64>>>,
    bits: Vec<u64>,
}

fn num_lt(a: Num, b: Num) -> bool {
    match (a, b) {
        (Num::I(x), Num::I(y)) => x < y,
        (Num::F(x), Num::F(y)) => x < y,
        (Num::I(x), Num::F(y)) => (x as f64) < y,
        (Num::F(x), Num::I(y)) => x < y as f64,
    }
}

fn is_null(c: &Col, i: usize) -> bool {
    match c {
        Col::Int(_, n) | Col::Float(_, n) => nl(n, i),
        Col::Text(codes, _) => codes[i] == NULL_CODE,
        Col::Other => true,
    }
}

fn code_of(c: &Col, i: usize) -> u64 {
    match c {
        Col::Int(v, n) => {
            if nl(n, i) {
                u64::MAX
            } else {
                v.at(i) as u64
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

const MORSEL: usize = 122_880;

enum Grouping<'a> {
    One,
    Text(&'a [u32], usize),
    IntRange(&'a Ints, &'a [bool], i64, usize),
    ExtractMinute(&'a Ints, &'a [bool]),
    Hash,
}

impl Grouping<'_> {
    fn slots(&self) -> usize {
        match self {
            Grouping::One => 1,
            Grouping::Text(_, n) => n + 1,
            Grouping::IntRange(_, _, _, n) => n + 1,
            Grouping::ExtractMinute(..) => 61,
            Grouping::Hash => 0,
        }
    }
}

/// Zero-heap-allocation hash key for 1-, 2-, and 3-column `GROUP BY`s.
#[derive(Clone, PartialEq, Eq, Hash)]
enum HKey {
    K1(u64, u8),
    K2(u64, u64, u8),
    K3(u64, u64, u64, u8),
    Many(Box<[(u64, bool)]>),
}

#[inline]
fn make_hkey(gcols: &[VCol], ri: usize) -> HKey {
    match gcols {
        [a] => {
            let (ca, na) = a.group_code_at(ri);
            HKey::K1(ca, na as u8)
        }
        [a, b] => {
            let (ca, na) = a.group_code_at(ri);
            let (cb, nb) = b.group_code_at(ri);
            HKey::K2(ca, cb, (na as u8) | ((nb as u8) << 1))
        }
        [a, b, c] => {
            let (ca, na) = a.group_code_at(ri);
            let (cb, nb) = b.group_code_at(ri);
            let (cc, nc) = c.group_code_at(ri);
            HKey::K3(ca, cb, cc, (na as u8) | ((nb as u8) << 1) | ((nc as u8) << 2))
        }
        _ => HKey::Many(gcols.iter().map(|g| g.group_code_at(ri)).collect()),
    }
}

struct Part {
    first: Vec<u32>,
    accs: Vec<Acc>,
    map: FxMap<HKey, u32>,
}

fn neumaier_lanes(sum: &mut f64, comp: &mut f64, xs: &[f64]) {
    const L: usize = 8;
    let mut s = [0f64; L];
    let mut c = [0f64; L];
    let mut it = xs.chunks_exact(L);
    for ch in &mut it {
        for l in 0..L {
            let x = ch[l];
            let t = s[l] + x;
            let bp = t - s[l];
            c[l] += (s[l] - (t - bp)) + (x - bp);
            s[l] = t;
        }
    }
    for l in 0..L {
        neumaier(sum, comp, s[l]);
        neumaier(sum, comp, c[l]);
    }
    for &x in it.remainder() {
        neumaier(sum, comp, x);
    }
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
        if let Some(m) = o.min
            && self.min.is_none_or(|x| num_lt(m, x))
        {
            self.min = Some(m);
        }
        if let Some(m) = o.max
            && self.max.is_none_or(|x| num_lt(x, m))
        {
            self.max = Some(m);
        }
        if let Some(ref s) = o.smin
            && self.smin.as_ref().is_none_or(|x| s < x)
        {
            self.smin = Some(s.clone());
        }
        if let Some(ref s) = o.smax
            && self.smax.as_ref().is_none_or(|x| x < s)
        {
            self.smax = Some(s.clone());
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

fn zone_verdict(
    t: &TableColumns,
    col: &str,
    op: &Operator,
    x: Num,
    s: usize,
    e: usize,
) -> Option<bool> {
    let zs = t.zones.get(col)?;
    if s % ZONE != 0 || (e % ZONE != 0 && e != t.len) {
        return None;
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
        return Some(false);
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

fn count_where(t: &TableColumns, preds: &[PrepPred]) -> u64 {
    use rayon::prelude::*;
    let nz = t.len.div_ceil(ZONE);
    (0..nz)
        .into_par_iter()
        .map(|zi| -> u64 {
            let (s, e) = (zi * ZONE, ((zi + 1) * ZONE).min(t.len));
            let mut live: Vec<&PrepPred> = Vec::with_capacity(preds.len());
            for p in preds {
                match prep_zone_verdict(t, p, s, e) {
                    Some(false) => return 0,
                    Some(true) => continue,
                    None => live.push(p),
                }
            }
            if live.is_empty() {
                return (e - s) as u64;
            }
            if let [
                PrepPred::IntCmp {
                    ints,
                    nulls,
                    op,
                    val,
                    ..
                },
            ] = live.as_slice()
                && nulls.is_empty()
            {
                let x = *val;
                let fast: Option<u64> = with_ints!(*ints, |v| {
                    let vs = &v[s..e];
                    match op {
                        Operator::Eq => Some(vs.iter().map(|&y| ((y as i64) == x) as u64).sum()),
                        Operator::Ne => Some(vs.iter().map(|&y| ((y as i64) != x) as u64).sum()),
                        Operator::Lt => Some(vs.iter().map(|&y| ((y as i64) < x) as u64).sum()),
                        Operator::Le => Some(vs.iter().map(|&y| ((y as i64) <= x) as u64).sum()),
                        Operator::Gt => Some(vs.iter().map(|&y| ((y as i64) > x) as u64).sum()),
                        Operator::Ge => Some(vs.iter().map(|&y| ((y as i64) >= x) as u64).sum()),
                        _ => None,
                    }
                });
                if let Some(c) = fast {
                    return c;
                }
            }
            let mut buf = [true; ZONE];
            let mask = &mut buf[..e - s];
            for p in live {
                mask_prep_pred(p, s, mask);
            }
            mask.iter().map(|&b| b as u64).sum()
        })
        .sum()
}

fn metadata_aggregate(t: &TableColumns, aggs: &[AggSpec]) -> Option<Vec<Acc>> {
    let mut accs = Vec::with_capacity(aggs.len());
    for ag in aggs {
        let mut a = Acc::default();
        match (ag.kind, &ag.vcol) {
            (AggKind::CountStar, _) => a.count = t.len as u64,
            (AggKind::Count | AggKind::Min | AggKind::Max, Some(VCol::Phys(col))) => {
                let name = t
                    .cols
                    .iter()
                    .find_map(|(k, v)| std::ptr::eq(v, *col).then_some(k.as_str()))?;
                let zs = t.zones.get(name)?;
                let nulls: usize = zs.iter().map(|z| z.nulls as usize).sum();
                a.count = (t.len - nulls) as u64;
                if matches!(ag.kind, AggKind::Min | AggKind::Max) {
                    for z in zs {
                        if let Some(l) = z.lo
                            && a.min.is_none_or(|m| num_lt(l, m))
                        {
                            a.min = Some(l);
                        }
                        if let Some(h) = z.hi
                            && a.max.is_none_or(|m| num_lt(m, h))
                        {
                            a.max = Some(h);
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

fn int_range_t(t: &TableColumns, ints: &Ints, nulls: &[bool]) -> Option<(i64, i64)> {
    let target_id = ints.raw_ptr_id();
    for (name, c) in &t.cols {
        if let Col::Int(cv, _) = c
            && cv.raw_ptr_id() == target_id
            && cv.len() == ints.len()
            && let Some(zs) = t.zones.get(name)
        {
            let (mut lo, mut hi) = (None::<i64>, None::<i64>);
            for z in zs {
                if let (Some(Num::I(l)), Some(Num::I(h))) = (z.lo, z.hi) {
                    lo = Some(lo.map_or(l, |x| x.min(l)));
                    hi = Some(hi.map_or(h, |x| x.max(h)));
                }
            }
            return lo.zip(hi);
        }
    }
    use rayon::prelude::*;
    let n = ints.len();
    (0..n.div_ceil(MORSEL))
        .into_par_iter()
        .filter_map(|ci| {
            let (s, e) = (ci * MORSEL, ((ci + 1) * MORSEL).min(n));
            let mut lo = i64::MAX;
            let mut hi = i64::MIN;
            with_ints!(ints, |v| {
                for i in s..e {
                    if !nl(nulls, i) {
                        let x = v[i] as i64;
                        lo = lo.min(x);
                        hi = hi.max(x);
                    }
                }
            });
            (lo <= hi).then_some((lo, hi))
        })
        .reduce_with(|a, b| (a.0.min(b.0), a.1.max(b.1)))
}

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
        Grouping::IntRange(ints, nulls, base, range) => with_ints!(*ints, |v| {
            rows.iter()
                .map(|&r| {
                    let ri = r as usize;
                    put(
                        if nl(nulls, ri) {
                            *range
                        } else {
                            ((v[ri] as i64) - base) as usize
                        },
                        r,
                    )
                })
                .collect()
        }),
        Grouping::ExtractMinute(ints, nulls) => with_ints!(*ints, |v| {
            rows.iter()
                .map(|&r| {
                    let ri = r as usize;
                    put(
                        if nl(nulls, ri) {
                            60
                        } else {
                            (v[ri] as i64).div_euclid(60).rem_euclid(60) as usize
                        },
                        r,
                    )
                })
                .collect()
        }),
        _ => Vec::new(),
    }
}

fn aggregate_dense_soa(
    t: &TableColumns,
    preds: &[PrepPred],
    grouping: &Grouping,
    aggs: &[AggSpec],
    proto: &[Acc],
) -> Option<(Vec<u32>, Vec<Acc>)> {
    use rayon::prelude::*;
    let n = t.len;
    let dense = grouping.slots();
    let nagg = aggs.len();
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
        let sel = morsel_sel_prep(t, preds, s, e);
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
        for ag in aggs {
            let mut cnt: Vec<u64>;
            let mut isum = Vec::new();
            let (mut fs, mut fc) = (Vec::new(), Vec::new());
            match (ag.kind, &ag.vcol) {
                (AggKind::CountStar, _) => {
                    cnt = vec![0u64; dense];
                    for &g in &gid {
                        cnt[g as usize] += 1;
                    }
                }
                (AggKind::Count, Some(vc)) => {
                    cnt = vec![0u64; dense];
                    for (&g, &r) in gid.iter().zip(rows) {
                        cnt[g as usize] += !vc.is_null_at(r as usize) as u64;
                    }
                }
                (AggKind::Sum | AggKind::Avg, Some(VCol::Phys(Col::Int(ints, nulls)))) => {
                    cnt = vec![0u64; dense];
                    isum = vec![0i128; dense];
                    with_ints!(ints, |v| {
                        for (&g, &r) in gid.iter().zip(rows) {
                            let ri = r as usize;
                            if !nl(nulls, ri) {
                                cnt[g as usize] += 1;
                                isum[g as usize] += v[ri] as i128;
                            }
                        }
                    });
                }
                (AggKind::Sum | AggKind::Avg, Some(vc)) if vc.is_int_like() => {
                    cnt = vec![0u64; dense];
                    isum = vec![0i128; dense];
                    for (&g, &r) in gid.iter().zip(rows) {
                        if let Some(x) = vc.int_at(r as usize) {
                            cnt[g as usize] += 1;
                            isum[g as usize] += x as i128;
                        }
                    }
                }
                (AggKind::Sum | AggKind::Avg, Some(VCol::Phys(Col::Float(v, nulls)))) => {
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
            cnt: vec![vec![]; nagg],
            isum: vec![vec![]; nagg],
            fs: vec![vec![]; nagg],
            fc: vec![vec![]; nagg],
        });
    }
    const SLOTS: usize = 4096;
    let mut m = P {
        first: vec![u32::MAX; dense],
        cnt: parts[0].cnt.iter().map(|v| vec![0u64; v.len()]).collect(),
        isum: parts[0].isum.iter().map(|v| vec![0i128; v.len()]).collect(),
        fs: parts[0].fs.iter().map(|v| vec![0f64; v.len()]).collect(),
        fc: parts[0].fc.iter().map(|v| vec![0f64; v.len()]).collect(),
    };
    {
        let parts = &parts;
        m.first
            .par_chunks_mut(SLOTS)
            .enumerate()
            .for_each(|(ci, out)| {
                let base = ci * SLOTS;
                for p in parts {
                    for (k, a) in out.iter_mut().enumerate() {
                        if *a == u32::MAX {
                            *a = p.first[base + k];
                        }
                    }
                }
            });
        for oi in 0..nagg {
            m.cnt[oi]
                .par_chunks_mut(SLOTS)
                .enumerate()
                .for_each(|(ci, out)| {
                    let base = ci * SLOTS;
                    for p in parts {
                        for (a, b) in out.iter_mut().zip(&p.cnt[oi][base..]) {
                            *a += b;
                        }
                    }
                });
            m.isum[oi]
                .par_chunks_mut(SLOTS)
                .enumerate()
                .for_each(|(ci, out)| {
                    let base = ci * SLOTS;
                    for p in parts {
                        for (a, b) in out.iter_mut().zip(&p.isum[oi][base..]) {
                            *a += b;
                        }
                    }
                });
            m.fs[oi]
                .par_chunks_mut(SLOTS)
                .zip(m.fc[oi].par_chunks_mut(SLOTS))
                .enumerate()
                .for_each(|(ci, (os, oc))| {
                    let base = ci * SLOTS;
                    for p in parts {
                        for k in 0..os.len() {
                            neumaier(&mut os[k], &mut oc[k], p.fs[oi][base + k]);
                            neumaier(&mut os[k], &mut oc[k], p.fc[oi][base + k]);
                        }
                    }
                });
        }
    }
    let mut idx: Vec<usize> = (0..dense).filter(|&i| m.first[i] != u32::MAX).collect();
    idx.sort_unstable_by_key(|&i| m.first[i]);
    let mut out = Vec::with_capacity(idx.len() * nagg.max(1));
    for &g in &idx {
        for oi in 0..nagg {
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

fn aggregate<'a>(
    t: &'a TableColumns,
    preds: &[PrepPred<'a>],
    gcols: &[VCol<'a>],
    aggs: &[AggSpec<'a>],
) -> Option<(Vec<u32>, Vec<Acc>)> {
    use rayon::prelude::*;
    let n = t.len;
    let grouping = match gcols {
        [] => Grouping::One,
        [VCol::Phys(Col::Text(codes, dict))] => Grouping::Text(codes, dict.len()),
        [VCol::DictRemap { codes, dict, remap }] if dict.len() == remap.len() => {
            let _ = codes;
            Grouping::Hash
        }
        [VCol::DictRemap { dict, .. }] if dict.is_empty() => Grouping::Hash,
        [VCol::Phys(Col::Int(ints, nulls))] => match int_range_t(t, ints, nulls) {
            Some((lo, hi))
                if (hi as i128 - lo as i128) < (1 << 20)
                    && ((hi - lo) as usize) <= 4 * n + 1024 =>
            {
                Grouping::IntRange(ints, nulls, lo, (hi - lo) as usize + 1)
            }
            None => Grouping::IntRange(ints, nulls, 0, 0),
            _ => Grouping::Hash,
        },
        [
            VCol::ExtractInt {
                ints,
                nulls,
                part: ExtractPart::Minute,
            },
        ] => Grouping::ExtractMinute(ints, nulls),
        _ => Grouping::Hash,
    };

    for ag in aggs {
        match (ag.kind, &ag.vcol) {
            (AggKind::CountStar, _) => {}
            (AggKind::Sum | AggKind::Avg, Some(vc)) if vc.is_numeric() => {}
            (AggKind::Min | AggKind::Max, Some(vc)) if vc.is_numeric() || vc.is_text_like() => {}
            (AggKind::Count | AggKind::CountDistinct, Some(_)) => {}
            _ => return None,
        }
    }

    // Bitmap fast path for single-group `COUNT(DISTINCT col)` on small-range ints OR dictionary text!
    let bit_base: Vec<Option<(i64, usize)>> = aggs
        .iter()
        .map(|ag| match (ag.kind, &ag.vcol, &grouping) {
            (
                AggKind::CountDistinct,
                Some(VCol::Phys(Col::Int(ints, nulls))),
                Grouping::One,
            ) => match int_range_t(t, ints, nulls) {
                Some((lo, hi)) if (hi as i128 - lo as i128) < (1 << 28) => {
                    Some((lo, (hi - lo) as usize + 1))
                }
                _ => None,
            },
            (
                AggKind::CountDistinct,
                Some(VCol::Phys(Col::Text(_, dict))),
                Grouping::One,
            ) if dict.len() < (1 << 28) => Some((0, dict.len())),
            _ => None,
        })
        .collect();

    let proto: Vec<Acc> = aggs
        .iter()
        .zip(&bit_base)
        .map(|(ag, bb)| {
            let mut a = Acc::default();
            if matches!(ag.kind, AggKind::CountDistinct) && bb.is_none() {
                a.distinct = Some(Box::default());
            }
            a
        })
        .collect();

    let dense = grouping.slots();
    let nagg = aggs.len();
    if dense > 0
        && !matches!(grouping, Grouping::One)
        && nagg > 0
        && aggs.iter().all(|ag| match (ag.kind, &ag.vcol) {
            (AggKind::CountStar | AggKind::Count, _) => true,
            (AggKind::Sum | AggKind::Avg, Some(vc)) => {
                vc.is_int_like() || matches!(vc, VCol::Phys(Col::Float(..)))
            }
            _ => false,
        })
    {
        return aggregate_dense_soa(t, preds, &grouping, aggs, &proto);
    }

    let msize = MORSEL.max(dense.saturating_mul(8));
    let morsels: Vec<(usize, usize)> = (0..n)
        .step_by(msize)
        .map(|s| (s, (s + msize).min(n)))
        .collect();

    let run = |&(s, e): &(usize, usize)| -> Option<Part> {
        let sel = morsel_sel_prep(t, preds, s, e);
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
            Grouping::IntRange(ints, nulls, base, range) => with_ints!(*ints, |v| {
                rows()
                    .map(|r| {
                        let ri = r as usize;
                        let slot = if nl(nulls, ri) {
                            *range
                        } else {
                            ((v[ri] as i64) - base) as usize
                        };
                        if part.first[slot] == u32::MAX {
                            part.first[slot] = r;
                        }
                        slot as u32
                    })
                    .collect()
            }),
            Grouping::ExtractMinute(ints, nulls) => with_ints!(*ints, |v| {
                rows()
                    .map(|r| {
                        let ri = r as usize;
                        let slot = if nl(nulls, ri) {
                            60
                        } else {
                            (v[ri] as i64).div_euclid(60).rem_euclid(60) as usize
                        };
                        if part.first[slot] == u32::MAX {
                            part.first[slot] = r;
                        }
                        slot as u32
                    })
                    .collect()
            }),
            Grouping::Hash => rows()
                .map(|r| {
                    let ri = r as usize;
                    let key = make_hkey(gcols, ri);
                    match part.map.get(&key) {
                        Some(&g) => g,
                        None => {
                            let g = part.first.len() as u32;
                            part.map.insert(key, g);
                            part.first.push(r);
                            g
                        }
                    }
                })
                .collect(),
        };
        let ng = part.first.len();
        if ng == 0 || nagg == 0 {
            return Some(part);
        }
        part.accs = (0..ng).flat_map(|_| proto.iter().cloned()).collect();
        for (oi, ag) in aggs.iter().enumerate() {
            if let Some((_, nbits)) = bit_base[oi] {
                part.accs[oi].bits = vec![0u64; nbits.div_ceil(64)];
            }
            let accs = &mut part.accs;
            let g = |k: usize| if gid.is_empty() { 0 } else { gid[k] as usize };

            // Reuse identical physical `SUM`/`AVG` scan (powers Q29's 90 `SUM(ResolutionWidth + k)`!)
            if matches!(ag.kind, AggKind::Sum | AggKind::Avg)
                && let Some(VCol::Phys(col)) = &ag.vcol
            {
                let twin = (0..oi).find(|&oj| {
                    matches!(aggs[oj].kind, AggKind::Sum | AggKind::Avg)
                        && bit_base[oj].is_none()
                        && matches!(&aggs[oj].vcol, Some(VCol::Phys(c)) if std::ptr::eq(*c, *col))
                });
                if let Some(oj) = twin {
                    for gi in 0..ng {
                        accs[gi * nagg + oi] = accs[gi * nagg + oj].clone();
                    }
                    continue;
                }
            }

            match (ag.kind, &ag.vcol) {
                (AggKind::CountStar, _) => {
                    if gid.is_empty() {
                        accs[oi].count += nsel as u64;
                    } else {
                        for &x in &gid {
                            accs[x as usize * nagg + oi].count += 1;
                        }
                    }
                }
                (
                    AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max,
                    Some(VCol::Phys(Col::Int(ints, nulls))),
                ) => {
                    if gid.is_empty() && sel.is_none() && nulls.is_empty() {
                        with_ints!(ints, |v| {
                            let sl = &v[s..e];
                            let a = &mut accs[oi];
                            a.count += sl.len() as u64;
                            a.isum += sl.iter().map(|&x| x as i128).sum::<i128>();
                            if matches!(ag.kind, AggKind::Min | AggKind::Max)
                                && let (Some(&lo_r), Some(&hi_r)) = (sl.iter().min(), sl.iter().max())
                            {
                                let (lo, hi) = (lo_r as i64, hi_r as i64);
                                a.min = Some(a.min.map_or(Num::I(lo), |m| {
                                    if num_lt(Num::I(lo), m) { Num::I(lo) } else { m }
                                }));
                                a.max = Some(a.max.map_or(Num::I(hi), |m| {
                                    if num_lt(m, Num::I(hi)) { Num::I(hi) } else { m }
                                }));
                            }
                        });
                        continue;
                    }
                    let want_mm = matches!(ag.kind, AggKind::Min | AggKind::Max);
                    with_ints!(ints, |v| {
                        for (k, r) in rows().enumerate() {
                            let ri = r as usize;
                            if nl(nulls, ri) {
                                continue;
                            }
                            let x = v[ri] as i64;
                            let a = &mut accs[g(k) * nagg + oi];
                            a.count += 1;
                            a.isum += x as i128;
                            if want_mm {
                                if a.min.is_none_or(|m| num_lt(Num::I(x), m)) {
                                    a.min = Some(Num::I(x));
                                }
                                if a.max.is_none_or(|m| num_lt(m, Num::I(x))) {
                                    a.max = Some(Num::I(x));
                                }
                            }
                        }
                    });
                }
                (
                    AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max,
                    Some(vc),
                ) if vc.is_int_like() => {
                    let want_mm = matches!(ag.kind, AggKind::Min | AggKind::Max);
                    for (k, r) in rows().enumerate() {
                        if let Some(x) = vc.int_at(r as usize) {
                            let a = &mut accs[g(k) * nagg + oi];
                            a.count += 1;
                            a.isum += x as i128;
                            if want_mm {
                                if a.min.is_none_or(|m| num_lt(Num::I(x), m)) {
                                    a.min = Some(Num::I(x));
                                }
                                if a.max.is_none_or(|m| num_lt(m, Num::I(x))) {
                                    a.max = Some(Num::I(x));
                                }
                            }
                        }
                    }
                }
                (
                    AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max,
                    Some(VCol::Phys(Col::Float(v, nulls))),
                ) => {
                    let want_mm = matches!(ag.kind, AggKind::Min | AggKind::Max);
                    if gid.is_empty() && sel.is_none() && nulls.is_empty() {
                        let sl = &v[s..e];
                        let a = &mut accs[oi];
                        a.count += sl.len() as u64;
                        neumaier_lanes(&mut a.fsum, &mut a.fcomp, sl);
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
                        let a = &mut accs[g(k) * nagg + oi];
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
                (AggKind::Min | AggKind::Max, Some(vc)) if vc.is_text_like() => {
                    for (k, r) in rows().enumerate() {
                        let ri = r as usize;
                        if let Some(sr) = vc.str_ref_at(ri) {
                            let a = &mut accs[g(k) * nagg + oi];
                            a.count += 1;
                            if matches!(ag.kind, AggKind::Min)
                                && a.smin.as_deref().is_none_or(|cur| sr < cur)
                            {
                                a.smin = vc.arc_str_at(ri);
                            }
                            if matches!(ag.kind, AggKind::Max)
                                && a.smax.as_deref().is_none_or(|cur| sr > cur)
                            {
                                a.smax = vc.arc_str_at(ri);
                            }
                        }
                    }
                }
                (AggKind::Count, Some(vc)) => {
                    for (k, r) in rows().enumerate() {
                        if !vc.is_null_at(r as usize) {
                            accs[g(k) * nagg + oi].count += 1;
                        }
                    }
                }
                (AggKind::CountDistinct, Some(vc)) => {
                    if let Some((base, _)) = bit_base[oi] {
                        let bits = &mut accs[oi].bits;
                        match vc {
                            VCol::Phys(Col::Int(ints, nulls)) => {
                                with_ints!(ints, |v| {
                                    for r in rows() {
                                        let ri = r as usize;
                                        if !nl(nulls, ri) {
                                            let b = ((v[ri] as i64) - base) as usize;
                                            bits[b >> 6] |= 1u64 << (b & 63);
                                        }
                                    }
                                });
                            }
                            VCol::Phys(Col::Text(codes, _)) => {
                                for r in rows() {
                                    let c = codes[r as usize];
                                    if c != NULL_CODE {
                                        let b = c as usize;
                                        bits[b >> 6] |= 1u64 << (b & 63);
                                    }
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }
                    for (k, r) in rows().enumerate() {
                        let (code, is_n) = vc.group_code_at(r as usize);
                        if !is_n
                            && let Some(d) = accs[g(k) * nagg + oi].distinct.as_mut()
                        {
                            d.insert(code);
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
                if nagg > 0 {
                    for (a, b) in accs[slot * nagg..(slot + 1) * nagg]
                        .iter_mut()
                        .zip(&p.accs[slot * nagg..(slot + 1) * nagg])
                    {
                        a.merge(b);
                    }
                }
            }
        }
    } else {
        let mut map: FxMap<HKey, usize> = FxMap::default();
        first = Vec::new();
        accs = Vec::new();
        for p in parts {
            let Part {
                first: p_first,
                accs: p_accs,
                map: p_map,
            } = p;
            let mut entries: Vec<(HKey, u32)> = p_map.into_iter().collect();
            entries.sort_unstable_by_key(|(_, g)| *g);
            for (k, g) in entries {
                let gi = *map.entry(k).or_insert_with(|| {
                    first.push(p_first[g as usize]);
                    accs.extend(proto.iter().cloned());
                    first.len() - 1
                });
                if nagg > 0 {
                    for (a, b) in accs[gi * nagg..(gi + 1) * nagg]
                        .iter_mut()
                        .zip(&p_accs[g as usize * nagg..(g as usize + 1) * nagg])
                    {
                        a.merge(b);
                    }
                }
            }
        }
    }

    let mut idx: Vec<usize> = (0..first.len()).filter(|&i| first[i] != u32::MAX).collect();
    if !matches!(grouping, Grouping::One) {
        idx.sort_unstable_by_key(|&i| first[i]);
    } else if idx.is_empty() {
        idx.push(0);
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
    if nagg == 0 {
        return Some((firsts, Vec::new()));
    }
    let mut slots: Vec<Option<Acc>> = accs.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(idx.len() * nagg);
    for &i in &idx {
        for oi in 0..nagg {
            out.push(
                slots
                    .get_mut(i * nagg + oi)
                    .and_then(Option::take)
                    .unwrap_or_else(|| proto[oi].clone()),
            );
        }
    }
    Some((firsts, out))
}

// ============================================================================
// Output Cell Rendering (`Sv`), `HAVING`, and Morsel-Parallel Top-K `SELECT`
// ============================================================================

#[derive(Clone)]
enum Sv {
    Null,
    U(u64),
    I(i64),
    F(f64),
    S(Arc<str>),
}

impl Sv {
    fn from_num_offset(n: Option<Num>, add: i64) -> Sv {
        match n {
            None => Sv::Null,
            Some(Num::I(i)) => Sv::I(i.saturating_add(add)),
            Some(Num::F(f)) => Sv::F(f + add as f64),
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

enum HavingPred {
    And(Box<HavingPred>, Box<HavingPred>),
    Or(Box<HavingPred>, Box<HavingPred>),
    Not(Box<HavingPred>),
    Cmp {
        out_idx: usize,
        op: Operator,
        lit: Sv,
    },
}

fn lit_to_sv(l: Lit) -> Sv {
    match l {
        Lit::N(Num::I(i)) => Sv::I(i),
        Lit::N(Num::F(f)) => Sv::F(f),
        Lit::S(s) => Sv::S(Arc::from(s.as_str())),
    }
}

fn eval_having(hp: &HavingPred, cell_fn: &dyn Fn(usize) -> Sv) -> bool {
    match hp {
        HavingPred::And(a, b) => eval_having(a, cell_fn) && eval_having(b, cell_fn),
        HavingPred::Or(a, b) => eval_having(a, cell_fn) || eval_having(b, cell_fn),
        HavingPred::Not(x) => !eval_having(x, cell_fn),
        HavingPred::Cmp { out_idx, op, lit } => {
            let v = cell_fn(*out_idx);
            if v.is_null() || lit.is_null() {
                return false;
            }
            cmp_ok(v.cmp_val(lit), op)
        }
    }
}

/// Plain projection / Top-K `SELECT ... WHERE ... [ORDER BY ...] [LIMIT/OFFSET]`
/// without `GROUP BY` or aggregates (`Q19`, `Q23`, `Q24`, `Q25`, `Q26`).
fn execute_scan<'a>(
    query: &'a SqlQuery,
    t: &'a TableColumns,
    preds: &[PrepPred<'a>],
) -> Option<VecResult> {
    use rayon::prelude::*;
    let mut proj: Vec<VCol<'a>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for sc in &query.columns {
        match &sc.expr {
            Expression::Wildcard | Expression::QualifiedWildcard(_) => {
                if sc.alias.is_some() {
                    return None;
                }
                for cn in &t.col_order {
                    let col = t.cols.get(cn)?;
                    if matches!(col, Col::Other) {
                        return None;
                    }
                    proj.push(VCol::Phys(col));
                    names.push(cn.clone());
                }
            }
            e => {
                let vc = build_vcol(t, e)?;
                let default_name = match e {
                    Expression::Column(n) => n.clone(),
                    Expression::QualifiedColumn { column, .. } => column.clone(),
                    Expression::Function { name, .. } => name.clone(),
                    _ => "expr".to_string(),
                };
                proj.push(vc);
                names.push(sc.alias.clone().unwrap_or(default_name));
            }
        }
    }
    if proj.is_empty() {
        return None;
    }

    // Build ORDER BY virtual columns (may referenceprojected aliases OR unprojected table columns!)
    let mut order_cols: Vec<(VCol<'a>, bool, Option<bool>)> = Vec::with_capacity(query.order_by.len());
    for ob in &query.order_by {
        let vc = match &ob.expr {
            Expression::Literal(AstValue::Int(i)) if *i >= 1 && (*i as usize) <= query.columns.len() => {
                let target_sc = &query.columns[*i as usize - 1];
                build_vcol(t, &target_sc.expr)?
            }
            e => {
                if let Some(n) = column(e)
                    && !t.cols.contains_key(n)
                    && let Some(pos) = names.iter().position(|x| x == n)
                    && pos < query.columns.len()
                {
                    build_vcol(t, &query.columns[pos].expr)?
                } else {
                    build_vcol(t, e)?
                }
            }
        };
        order_cols.push((vc, ob.descending, ob.nulls_first));
    }

    let off = query.offset.unwrap_or(0);
    let take = query.limit.map(|l| off.saturating_add(l));
    // Guard unbounded full-table row materialization on giant tables without WHERE/LIMIT
    if preds.is_empty() && take.is_none() && t.len > 250_000 {
        return None;
    }

    let cmp_rows = |&ra: &u32, &rb: &u32| -> Ordering {
        let (ia, ib) = (ra as usize, rb as usize);
        for (vc, desc, nulls_first) in &order_cols {
            let na = vc.is_null_at(ia);
            let nb = vc.is_null_at(ib);
            let nf = nulls_first.unwrap_or(*desc);
            let o = match (na, nb) {
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
                (false, false) => {
                    let ord = if let (Some(xa), Some(xb)) = (vc.int_at(ia), vc.int_at(ib)) {
                        xa.cmp(&xb)
                    } else if let (Some(sa), Some(sb)) = (vc.str_ref_at(ia), vc.str_ref_at(ib)) {
                        sa.cmp(sb)
                    } else if let (Some(fa), Some(fb)) = (vc.float_at(ia), vc.float_at(ib)) {
                        fa.partial_cmp(&fb).unwrap_or(Ordering::Equal)
                    } else {
                        Ordering::Equal
                    };
                    if *desc { ord.reverse() } else { ord }
                }
            };
            if o != Ordering::Equal {
                return o;
            }
        }
        ra.cmp(&rb)
    };

    let n = t.len;
    let morsels: Vec<(usize, usize)> = (0..n)
        .step_by(MORSEL)
        .map(|s| (s, (s + MORSEL).min(n)))
        .collect();

    let mut winners: Vec<u32> = if order_cols.is_empty() {
        // No ORDER BY: preserve row order and stop collecting once `take` is satisfied
        let chunks: Vec<Vec<u32>> = morsels
            .par_iter()
            .map(|&(s, e)| match morsel_sel_prep(t, preds, s, e) {
                Some(v) => v,
                None => {
                    let cap = take.map_or(e - s, |k| (e - s).min(k));
                    if s == 0 {
                        (0..cap as u32).collect()
                    } else {
                        (s as u32..e as u32).collect()
                    }
                }
            })
            .collect();
        let mut flat = Vec::new();
        for ch in chunks {
            flat.extend(ch);
            if let Some(k) = take
                && flat.len() >= k
            {
                flat.truncate(k);
                break;
            }
        }
        flat
    } else {
        // Morsel-parallel filter + per-morsel Top-K selection (`select_nth_unstable_by`)!
        let mut cand: Vec<u32> = morsels
            .par_iter()
            .flat_map_iter(|&(s, e)| {
                let mut v: Vec<u32> = match morsel_sel_prep(t, preds, s, e) {
                    Some(v) => v,
                    None => (s as u32..e as u32).collect(),
                };
                if let Some(k) = take
                    && k > 0
                    && k < v.len()
                {
                    v.select_nth_unstable_by(k - 1, &cmp_rows);
                    v.truncate(k);
                }
                v
            })
            .collect();
        if let Some(k) = take {
            if k == 0 {
                cand.clear();
            } else if k < cand.len() {
                cand.select_nth_unstable_by(k - 1, &cmp_rows);
                cand.truncate(k);
            }
        }
        cand.sort_unstable_by(&cmp_rows);
        cand
    };

    if take == Some(0) {
        winners.clear();
    }
    let end = take.unwrap_or(usize::MAX).min(winners.len());
    let slice = &winners[off.min(end)..end];
    let rows: Vec<Vec<serde_json::Value>> = slice
        .iter()
        .map(|&ri| proj.iter().map(|vc| vc.sv_at(ri as usize).json()).collect())
        .collect();
    Some(VecResult {
        columns: names,
        rows,
    })
}

pub struct VecResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

const SMALL_TABLE_ROWS: usize = 65_536;

thread_local! {
    static SMALL_POOL: Option<rayon::ThreadPool> = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .thread_name(|_| "jouledb-small-q".into())
        .build()
        .ok();
}

pub fn execute(query: &SqlQuery, t: &TableColumns) -> Option<VecResult> {
    if t.len <= SMALL_TABLE_ROWS
        && !query.group_by.is_empty()
        && rayon::current_thread_index().is_none()
        && let Some(r) =
            SMALL_POOL.with(|p| p.as_ref().map(|p| p.install(|| execute_inner(query, t))))
    {
        return r;
    }
    execute_inner(query, t)
}

fn parse_agg_expr<'a>(t: &'a TableColumns, e: &'a Expression) -> Option<(AggKind, Option<VCol<'a>>, i64, String)> {
    let Expression::Function { name, args } = e else {
        return None;
    };
    let upper = name.to_ascii_uppercase();
    let default_name = if upper == "COUNT_DISTINCT" {
        "count".to_string()
    } else {
        name.to_ascii_lowercase()
    };
    match (upper.as_str(), args.as_slice()) {
        ("COUNT", [] | [Expression::Wildcard]) => {
            Some((AggKind::CountStar, None, 0, default_name))
        }
        ("COUNT", [arg]) => Some((AggKind::Count, Some(build_vcol(t, arg)?), 0, default_name)),
        ("COUNT_DISTINCT", [arg]) => Some((
            AggKind::CountDistinct,
            Some(build_vcol(t, arg)?),
            0,
            default_name,
        )),
        ("SUM" | "AVG" | "MIN" | "MAX", [arg]) => {
            let kind = match upper.as_str() {
                "SUM" => AggKind::Sum,
                "AVG" => AggKind::Avg,
                "MIN" => AggKind::Min,
                _ => AggKind::Max,
            };
            // Algebraic fast path: `SUM(col + k)` / `AVG(col + k)` / `MIN(col + k)` / `MAX(col + k)`
            // reuses physical `col` scan directly (critical for ClickBench Q29's 90 projections!).
            if let Expression::Binary { left, op, right } = arg
                && matches!(op, Operator::Add | Operator::Sub)
                && let (Some(cn), Some(Lit::N(Num::I(rhs)))) = (column(left), literal(right))
                && let Some(col @ Col::Int(..)) = t.cols.get(cn)
            {
                let delta = if *op == Operator::Add { rhs } else { rhs.checked_neg()? };
                return Some((kind, Some(VCol::Phys(col)), delta, default_name));
            }
            let vc = build_vcol(t, arg)?;
            match kind {
                AggKind::Sum | AggKind::Avg if !vc.is_numeric() => return None,
                AggKind::Min | AggKind::Max if !(vc.is_numeric() || vc.is_text_like()) => {
                    return None
                }
                _ => {}
            }
            Some((kind, Some(vc), 0, default_name))
        }
        _ => None,
    }
}

fn execute_inner(query: &SqlQuery, t: &TableColumns) -> Option<VecResult> {
    if !query.joins.is_empty()
        || !query.ctes.is_empty()
        || query.set_op.is_some()
        || query.distinct
        || query.as_of_timestamp.is_some()
    {
        return None;
    }

    let mut raw_preds = Vec::new();
    if let Some(w) = &query.where_clause {
        collect_preds(w, &mut raw_preds)?;
    }
    let preds = prepare_preds(t, &raw_preds)?;

    // Resolve GROUP BY expressions (ordinals `1..=N` and SELECT aliases like `AS m`, `AS k`, `AS Src`)
    let mut group_exprs: Vec<&Expression> = Vec::with_capacity(query.group_by.len());
    let mut gcols: Vec<VCol> = Vec::with_capacity(query.group_by.len());
    for ge in &query.group_by {
        let resolved: &Expression = match ge {
            Expression::Literal(AstValue::Int(n))
                if *n >= 1 && (*n as usize) <= query.columns.len() =>
            {
                &query.columns[*n as usize - 1].expr
            }
            Expression::Column(alias_candidate) if !t.cols.contains_key(alias_candidate.as_str()) => {
                if let Some(sc) = query
                    .columns
                    .iter()
                    .find(|sc| sc.alias.as_deref() == Some(alias_candidate.as_str()))
                {
                    &sc.expr
                } else {
                    return None;
                }
            }
            other => other,
        };
        gcols.push(build_vcol(t, resolved)?);
        group_exprs.push(resolved);
    }

    // Check if query has aggregates
    let query_has_agg = query.columns.iter().any(|c| parse_agg_expr(t, &c.expr).is_some())
        || query.having.is_some();

    if !query_has_agg && group_exprs.is_empty() {
        return execute_scan(query, t, &preds);
    }

    let mut outs: Vec<OutCol> = Vec::with_capacity(query.columns.len());
    let mut out_exprs: Vec<Option<&Expression>> = Vec::with_capacity(query.columns.len());
    let mut names: Vec<String> = Vec::with_capacity(query.columns.len());
    let mut aggs: Vec<AggSpec> = Vec::new();

    for c in &query.columns {
        if let Some(gpos) = group_exprs.iter().position(|&ge| ge == &c.expr) {
            let default_name = match &c.expr {
                Expression::Column(n) => n.clone(),
                Expression::QualifiedColumn { column, .. } => column.clone(),
                Expression::Function { name, .. } => name.clone(),
                _ => "expr".to_string(),
            };
            outs.push(OutCol::Group(gpos));
            out_exprs.push(Some(&c.expr));
            names.push(c.alias.clone().unwrap_or(default_name));
        } else if let Some((kind, vcol, const_add, default_name)) = parse_agg_expr(t, &c.expr) {
            let ai = aggs.len();
            aggs.push(AggSpec {
                kind,
                vcol,
                const_add,
            });
            outs.push(OutCol::Agg(ai));
            out_exprs.push(Some(&c.expr));
            names.push(c.alias.clone().unwrap_or(default_name));
        } else {
            return None;
        }
    }

    let visible_no = outs.len();

    // Compile optional HAVING clause (may reference projected aliases/exprs OR unprojected aggregates!)
    fn compile_having<'b>(
        e: &'b Expression,
        t: &'b TableColumns,
        names: &[String],
        group_exprs: &[&'b Expression],
        out_exprs: &mut Vec<Option<&'b Expression>>,
        outs: &mut Vec<OutCol>,
        aggs: &mut Vec<AggSpec<'b>>,
    ) -> Option<HavingPred> {
        match e {
            Expression::Binary {
                left,
                op: Operator::And,
                right,
            } => Some(HavingPred::And(
                Box::new(compile_having(
                    left,
                    t,
                    names,
                    group_exprs,
                    out_exprs,
                    outs,
                    aggs,
                )?),
                Box::new(compile_having(
                    right,
                    t,
                    names,
                    group_exprs,
                    out_exprs,
                    outs,
                    aggs,
                )?),
            )),
            Expression::Binary {
                left,
                op: Operator::Or,
                right,
            } => Some(HavingPred::Or(
                Box::new(compile_having(
                    left,
                    t,
                    names,
                    group_exprs,
                    out_exprs,
                    outs,
                    aggs,
                )?),
                Box::new(compile_having(
                    right,
                    t,
                    names,
                    group_exprs,
                    out_exprs,
                    outs,
                    aggs,
                )?),
            )),
            Expression::Unary {
                op: UnaryOperator::Not,
                expr,
            } => Some(HavingPred::Not(Box::new(compile_having(
                expr,
                t,
                names,
                group_exprs,
                out_exprs,
                outs,
                aggs,
            )?))),
            Expression::Binary { left, op, right } => {
                let mut resolve_side = |side: &'b Expression| -> Option<usize> {
                    if let Some(pos) = out_exprs.iter().position(|oe| *oe == Some(side)) {
                        return Some(pos);
                    }
                    if let Some(cn) = column(side)
                        && let Some(pos) = names.iter().position(|n| n == cn)
                    {
                        return Some(pos);
                    }
                    if let Some(gpos) = group_exprs.iter().position(|&ge| ge == side) {
                        let idx = outs.len();
                        outs.push(OutCol::Group(gpos));
                        out_exprs.push(Some(side));
                        return Some(idx);
                    }
                    if let Some((kind, vcol, const_add, _)) = parse_agg_expr(t, side) {
                        let ai = aggs.len();
                        aggs.push(AggSpec {
                            kind,
                            vcol,
                            const_add,
                        });
                        let idx = outs.len();
                        outs.push(OutCol::Agg(ai));
                        out_exprs.push(Some(side));
                        return Some(idx);
                    }
                    None
                };
                if let Some(lit) = literal(right)
                    && let Some(out_idx) = resolve_side(left)
                {
                    flip(op)?;
                    return Some(HavingPred::Cmp {
                        out_idx,
                        op: *op,
                        lit: lit_to_sv(lit),
                    });
                }
                if let Some(lit) = literal(left)
                    && let Some(out_idx) = resolve_side(right)
                {
                    return Some(HavingPred::Cmp {
                        out_idx,
                        op: flip(op)?,
                        lit: lit_to_sv(lit),
                    });
                }
                None
            }
            _ => None,
        }
    }

    let having_pred = match &query.having {
        Some(h) => Some(compile_having(
            h,
            t,
            &names,
            &group_exprs,
            &mut out_exprs,
            &mut outs,
            &mut aggs,
        )?),
        None => None,
    };

    // Compile ORDER BY (may reference output ordinal, alias, output expr, group expr, or hidden aggregate!)
    let mut order = Vec::with_capacity(query.order_by.len());
    for ob in &query.order_by {
        let idx = match &ob.expr {
            Expression::Literal(AstValue::Int(i)) if *i >= 1 && (*i as usize) <= visible_no => {
                *i as usize - 1
            }
            e => {
                if let Some(n) = column(e)
                    && let Some(pos) = names.iter().position(|x| x == n)
                {
                    pos
                } else if let Some(pos) = out_exprs.iter().position(|oe| *oe == Some(e)) {
                    pos
                } else if let Some(gpos) = group_exprs.iter().position(|&ge| ge == e) {
                    let pos = outs.len();
                    outs.push(OutCol::Group(gpos));
                    out_exprs.push(Some(e));
                    pos
                } else if let Some((kind, vcol, const_add, _)) = parse_agg_expr(t, e) {
                    let ai = aggs.len();
                    aggs.push(AggSpec {
                        kind,
                        vcol,
                        const_add,
                    });
                    let pos = outs.len();
                    outs.push(OutCol::Agg(ai));
                    out_exprs.push(Some(e));
                    pos
                } else {
                    return None;
                }
            }
        };
        order.push((idx, ob.descending, ob.nulls_first));
    }

    let ta = std::time::Instant::now();
    let meta = if preds.is_empty()
        && gcols.is_empty()
        && aggs.iter().all(|a| a.const_add == 0)
    {
        metadata_aggregate(t, &aggs)
    } else if gcols.is_empty()
        && aggs.iter().all(|a| matches!(a.kind, AggKind::CountStar))
    {
        let c = count_where(t, &preds);
        Some(
            aggs.iter()
                .map(|_| Acc {
                    count: c,
                    ..Acc::default()
                })
                .collect(),
        )
    } else {
        None
    };

    let (firsts, accs) = match meta {
        Some(accs) => (vec![0], accs),
        None => aggregate(t, &preds, &gcols, &aggs)?,
    };
    let ta = ta.elapsed();
    let ngroups = if gcols.is_empty() { 1 } else { firsts.len() };
    let nagg = aggs.len();
    let total_no = outs.len();

    let cell = |gi: usize, oi: usize| -> Sv {
        let first = firsts.get(gi).copied().unwrap_or(0) as usize;
        match &outs[oi] {
            OutCol::Group(g) => gcols[*g].sv_at(first),
            OutCol::Agg(ai) => {
                let a = &accs[gi * nagg + *ai];
                let spec = &aggs[*ai];
                let is_int = spec.vcol.as_ref().is_some_and(VCol::is_int_like);
                let is_txt = spec.vcol.as_ref().is_some_and(VCol::is_text_like);
                match spec.kind {
                    AggKind::CountStar | AggKind::Count => Sv::U(a.count),
                    AggKind::CountDistinct if !a.bits.is_empty() => {
                        Sv::U(a.bits.iter().map(|w| w.count_ones() as u64).sum())
                    }
                    AggKind::CountDistinct => {
                        Sv::U(a.distinct.as_ref().map_or(0, |d| d.len() as u64))
                    }
                    AggKind::Sum if a.count == 0 => Sv::Null,
                    AggKind::Sum if is_int => {
                        let shifted = a.isum + (spec.const_add as i128) * (a.count as i128);
                        match i64::try_from(shifted) {
                            Ok(v) => Sv::I(v),
                            Err(_) => Sv::F(shifted as f64),
                        }
                    }
                    AggKind::Sum => {
                        Sv::F((a.fsum + a.fcomp) + (spec.const_add as f64) * (a.count as f64))
                    }
                    AggKind::Avg if a.count == 0 => Sv::Null,
                    AggKind::Avg => {
                        let s = if is_int {
                            a.isum as f64
                        } else {
                            a.fsum + a.fcomp
                        };
                        Sv::F(s / a.count as f64 + spec.const_add as f64)
                    }
                    AggKind::Min if is_txt => match &a.smin {
                        Some(s) => Sv::S(s.clone()),
                        None => Sv::Null,
                    },
                    AggKind::Max if is_txt => match &a.smax {
                        Some(s) => Sv::S(s.clone()),
                        None => Sv::Null,
                    },
                    AggKind::Min => Sv::from_num_offset(a.min, spec.const_add),
                    AggKind::Max => Sv::from_num_offset(a.max, spec.const_add),
                }
            }
        }
    };

    use rayon::prelude::*;
    // Filter surviving group indices via HAVING (if present) BEFORE Top-N sorting!
    let kept_groups: Vec<usize> = if let Some(ref hp) = having_pred {
        (0..ngroups)
            .into_par_iter()
            .filter(|&gi| eval_having(hp, &|oi| cell(gi, oi)))
            .collect()
    } else {
        (0..ngroups).collect()
    };
    let nkept = kept_groups.len();

    let off = query.offset.unwrap_or(0);
    let take = query.limit.map(|l| off.saturating_add(l));
    let small_top = !order.is_empty()
        && take.is_some_and(|k| k > 0 && k <= 4096 && k.saturating_mul(8) < nkept);

    let (vals, idx, row_stride): (Vec<Sv>, Vec<usize>, usize) = if small_top {
        let k = take.unwrap_or(0);
        let ocols: Vec<usize> = order.iter().map(|(i, _, _)| *i).collect();
        let nv = ocols.len();
        let ovals: Vec<Sv> = kept_groups
            .par_iter()
            .flat_map_iter(|&gi| ocols.iter().map(move |&oi| cell(gi, oi)).collect::<Vec<_>>())
            .collect();
        let order_v: Vec<(usize, bool, Option<bool>)> = order
            .iter()
            .enumerate()
            .map(|(j, (_, d, n))| (j, *d, *n))
            .collect();
        let cmp = mk_cmp(&ovals, nv, &order_v);
        let mut cand: Vec<usize> = (0..nkept)
            .collect::<Vec<_>>()
            .par_chunks(16_384)
            .flat_map_iter(|ch| {
                let mut v = ch.to_vec();
                if k < v.len() {
                    v.select_nth_unstable_by(k - 1, &cmp);
                    v.truncate(k);
                }
                v
            })
            .collect();
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, &cmp);
            cand.truncate(k);
        }
        cand.sort_unstable_by(&cmp);
        let vals: Vec<Sv> = cand
            .iter()
            .flat_map(|&ki| {
                let gi = kept_groups[ki];
                (0..visible_no).map(move |oi| cell(gi, oi))
            })
            .collect();
        let n = cand.len();
        (vals, (0..n).collect(), visible_no)
    } else {
        let vals: Vec<Sv> = kept_groups
            .par_iter()
            .flat_map_iter(|&gi| (0..total_no).map(move |oi| cell(gi, oi)).collect::<Vec<_>>())
            .collect();
        let mut idx: Vec<usize> = (0..nkept).collect();
        if !order.is_empty() {
            let cmp = mk_cmp(&vals, total_no, &order);
            if let Some(k) = take.filter(|&k| k > 0 && k < idx.len()) {
                idx.select_nth_unstable_by(k - 1, &cmp);
                idx.truncate(k);
            }
            idx.sort_unstable_by(&cmp);
        }
        (vals, idx, total_no)
    };

    if std::env::var_os("JOULEDB_VEC_TRACE").is_some() {
        eprintln!("render agg={ta:?} groups={ngroups} kept={nkept}");
    }
    let end = take.unwrap_or(usize::MAX).min(idx.len());
    let rows: Vec<Vec<serde_json::Value>> = idx[off.min(end)..end]
        .iter()
        .map(|&i| {
            vals[i * row_stride..i * row_stride + visible_no]
                .iter()
                .map(Sv::json)
                .collect()
        })
        .collect();
    Some(VecResult {
        columns: names,
        rows,
    })
}

fn mk_cmp<'a>(
    vals: &'a [Sv],
    no: usize,
    order: &'a [(usize, bool, Option<bool>)],
) -> impl Fn(&usize, &usize) -> Ordering + Sync + 'a {
    move |&ia: &usize, &ib: &usize| -> Ordering {
        let (a, b) = (&vals[ia * no..(ia + 1) * no], &vals[ib * no..(ib + 1) * no]);
        for (i, desc, nulls_first) in order.iter() {
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
        ia.cmp(&ib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_db_amorphic::column_batch::{BatchColumn, ColData, ColumnBatch};
    use joule_db_query::sql::{SqlParser, SqlStatement};

    fn sample_hits_columns() -> TableColumns {
        let urls = [
            "https://www.google.com/search?q=rust",
            "https://example.org/docs/page",
            "",
            "http://blog.rust-lang.org/2026/01",
        ];
        let titles = ["Google Search", "Example Docs", "", "Rust Blog Google"];
        let phrases = ["rust db", "", "", "vectorized sql"];
        let mut u_off = vec![0u32];
        let mut u_bytes = Vec::new();
        let mut t_off = vec![0u32];
        let mut t_bytes = Vec::new();
        let mut p_off = vec![0u32];
        let mut p_bytes = Vec::new();
        let n = 8usize;
        for i in 0..n {
            u_bytes.extend_from_slice(urls[i % 4].as_bytes());
            u_off.push(u_bytes.len() as u32);
            t_bytes.extend_from_slice(titles[i % 4].as_bytes());
            t_off.push(t_bytes.len() as u32);
            p_bytes.extend_from_slice(phrases[i % 4].as_bytes());
            p_off.push(p_bytes.len() as u32);
        }
        let batch = ColumnBatch {
            table_field: "_table".into(),
            table: "hits".into(),
            first_id: 1,
            len: n,
            cols: vec![
                BatchColumn {
                    name: "CounterID".into(),
                    data: ColData::Int32(vec![62, 62, 10, 62, 62, 62, 10, 62]),
                    valid: None,
                },
                BatchColumn {
                    name: "EventDate".into(),
                    // 2013-07-15 == 15901 days since 1970-01-01
                    data: ColData::Int16(vec![15901; n]),
                    valid: None,
                },
                BatchColumn {
                    name: "EventTime".into(),
                    // 2013-07-15 12:34:56 UTC = 15901 * 86400 + 12*3600 + 34*60 + 56 = 1373891696
                    data: ColData::Int((0..n).map(|i| 1_373_891_696 + (i as i64) * 30).collect()),
                    valid: None,
                },
                BatchColumn {
                    name: "ResolutionWidth".into(),
                    data: ColData::Int16(vec![1920, 1440, 1280, 1920, 1920, 1440, 1280, 1920]),
                    valid: None,
                },
                BatchColumn {
                    name: "TraficSourceID".into(),
                    data: ColData::Int8(vec![-1, 6, 0, -1, 6, 2, 0, -1]),
                    valid: None,
                },
                BatchColumn {
                    name: "SearchEngineID".into(),
                    data: ColData::Int16(vec![0, 2, 0, 0, 2, 0, 0, 0]),
                    valid: None,
                },
                BatchColumn {
                    name: "AdvEngineID".into(),
                    data: ColData::Int8(vec![0; n]),
                    valid: None,
                },
                BatchColumn {
                    name: "URL".into(),
                    data: ColData::Str {
                        offsets: u_off,
                        bytes: u_bytes,
                    },
                    valid: None,
                },
                BatchColumn {
                    name: "Title".into(),
                    data: ColData::Str {
                        offsets: t_off,
                        bytes: t_bytes,
                    },
                    valid: None,
                },
                BatchColumn {
                    name: "SearchPhrase".into(),
                    data: ColData::Str {
                        offsets: p_off,
                        bytes: p_bytes,
                    },
                    valid: None,
                },
            ],
        };
        let names: Vec<String> = batch.cols.iter().map(|c| c.name.clone()).collect();
        TableColumns::from_batch(&names, &batch)
    }

    fn run_vec(t: &TableColumns, sql: &str) -> VecResult {
        let stmt = SqlParser::new().parse(sql).expect("valid SQL");
        let SqlStatement::Select(q) = stmt else {
            panic!("expected SELECT");
        };
        execute(&q, t).unwrap_or_else(|| panic!("expected vectorized support for: {sql}"))
    }

    #[test]
    fn test_clickbench_shapes_all_vectorized() {
        let t = sample_hits_columns();

        // Plain Top-K scan with unprojected ORDER BY (Q24/Q26) and SELECT * (Q23)
        let r23 = run_vec(
            &t,
            "SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 5",
        );
        assert_eq!(r23.columns.len(), 10);
        assert_eq!(r23.rows.len(), 2);

        let r24 = run_vec(
            &t,
            "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 3",
        );
        assert_eq!(r24.rows.len(), 3);
        assert_eq!(r24.rows[0][0], serde_json::json!("rust db"));

        // Text MIN/MAX (Q21/Q22)
        let r21 = run_vec(
            &t,
            "SELECT SearchPhrase, MIN(URL), MAX(Title), COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC",
        );
        assert_eq!(r21.rows.len(), 2);

        // HAVING + STRLEN + REGEXP_REPLACE backreference (Q27/Q28)
        let r28 = run_vec(
            &t,
            "SELECT REGEXP_REPLACE(URL, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k, AVG(STRLEN(URL)) AS l, COUNT(*) AS c, MIN(URL) FROM hits WHERE URL <> '' GROUP BY k HAVING COUNT(*) > 1 ORDER BY l DESC LIMIT 10",
        );
        assert_eq!(r28.rows.len(), 3);
        let domains: Vec<String> = r28
            .rows
            .iter()
            .map(|r| r[0].as_str().unwrap_or_default().to_string())
            .collect();
        assert!(domains.contains(&"google.com".to_string()));
        assert!(domains.contains(&"example.org".to_string()));
        assert!(domains.contains(&"blog.rust-lang.org".to_string()));

        // Algebraic SUM(col + k) sharing (Q29)
        let r29 = run_vec(
            &t,
            "SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), SUM(ResolutionWidth + 10) FROM hits",
        );
        let s0 = r29.rows[0][0].as_i64().unwrap();
        let s1 = r29.rows[0][1].as_i64().unwrap();
        let s10 = r29.rows[0][2].as_i64().unwrap();
        assert_eq!(s1 - s0, 8);
        assert_eq!(s10 - s0, 80);

        // EXTRACT(minute FROM EventTime) + alias in GROUP BY (Q18)
        let r18 = run_vec(
            &t,
            "SELECT extract(minute FROM EventTime) AS m, COUNT(*) FROM hits GROUP BY m ORDER BY m",
        );
        assert!(!r18.rows.is_empty());

        // ISO date literal against epoch-day Int16 + IN (...) + CASE WHEN + DATE_TRUNC (Q36..Q42)
        let r39 = run_vec(
            &t,
            "SELECT TraficSourceID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN URL ELSE '' END AS Src, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND TraficSourceID IN (-1, 6) GROUP BY TraficSourceID, Src ORDER BY PageViews DESC",
        );
        assert!(!r39.rows.is_empty());

        let r42 = run_vec(
            &t,
            "SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10",
        );
        assert!(!r42.rows.is_empty());
    }
}
