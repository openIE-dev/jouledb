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

    pub fn empty_with_len(len: usize, col_order: Vec<String>) -> Self {
        Self {
            len,
            col_order,
            cols: HashMap::new(),
            text_index: HashMap::new(),
            zones: HashMap::new(),
        }
    }

    pub fn insert_decoded_columns(&mut self, len: usize, decoded: Vec<(String, Col)>) {
        self.len = len;
        for (name, col) in decoded {
            if matches!(&col, Col::Int(..) | Col::Float(..)) {
                let mut z = Vec::with_capacity(len.div_ceil(ZONE));
                let mut ok = true;
                for zi in 0..len.div_ceil(ZONE) {
                    match zone_of(&col, zi * ZONE, ((zi + 1) * ZONE).min(len)) {
                        Some(x) => z.push(x),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    self.zones.insert(name.clone(), z);
                }
            }
            self.cols.insert(name, col);
        }
    }

    pub fn from_proto_cols(
        col_order: Vec<String>,
        cols: HashMap<String, Col>,
        int_mm: HashMap<String, (i64, i64)>,
        float_mm: HashMap<String, (f64, f64)>,
    ) -> Self {
        let mut zones = HashMap::new();
        for (k, (lo, hi)) in int_mm {
            zones.insert(
                k,
                vec![Zone {
                    lo: Some(Num::I(lo)),
                    hi: Some(Num::I(hi)),
                    nulls: 0,
                }],
            );
        }
        for (k, (lo, hi)) in float_mm {
            zones.insert(
                k,
                vec![Zone {
                    lo: Some(Num::F(lo)),
                    hi: Some(Num::F(hi)),
                    nulls: 0,
                }],
            );
        }
        Self {
            len: 0,
            col_order,
            cols,
            text_index: HashMap::new(),
            zones,
        }
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
        Expression::Literal(AstValue::Timestamp(ts)) => {
            let (y, m, d, _) = civil_from_days(ts.div_euclid(86_400));
            Some(Lit::S(format!("{y:04}-{m:02}-{d:02}")))
        }
        Expression::Cast { expr, target_type } => {
            let inner = literal(expr)?;
            let tt = target_type.to_ascii_uppercase();
            if matches!(
                tt.as_str(),
                "DATE" | "TIMESTAMP" | "TEXT" | "VARCHAR" | "CHAR" | "STRING"
            ) {
                Some(inner)
            } else if tt.starts_with("DECIMAL")
                || tt.starts_with("NUMERIC")
                || matches!(tt.as_str(), "FLOAT" | "DOUBLE" | "REAL")
            {
                match inner {
                    Lit::N(n) => Some(Lit::N(n)),
                    Lit::S(s) => s.parse::<f64>().ok().map(|f| Lit::N(Num::F(f))),
                }
            } else if matches!(tt.as_str(), "INT" | "INTEGER" | "BIGINT" | "SMALLINT") {
                match inner {
                    Lit::N(Num::I(i)) => Some(Lit::N(Num::I(i))),
                    Lit::S(s) => s.parse::<i64>().ok().map(|i| Lit::N(Num::I(i))),
                    _ => None,
                }
            } else {
                None
            }
        }
        Expression::Unary {
            op: UnaryOperator::Neg,
            expr,
        } => match literal(expr)? {
            Lit::N(Num::I(i)) => Some(Lit::N(Num::I(i.checked_neg()?))),
            Lit::N(Num::F(f)) => Some(Lit::N(Num::F(-f))),
            Lit::S(_) => None,
        },
        Expression::Binary { left, op, right } => {
            if let (
                Expression::Literal(AstValue::Timestamp(ts)),
                Expression::Literal(AstValue::Int(sec)),
            ) = (left.as_ref(), right.as_ref())
            {
                let ts2 = match op {
                    Operator::Add => ts.checked_add(*sec)?,
                    Operator::Sub => ts.checked_sub(*sec)?,
                    _ => return None,
                };
                let (y, m, d, _) = civil_from_days(ts2.div_euclid(86_400));
                return Some(Lit::S(format!("{y:04}-{m:02}-{d:02}")));
            }
            let (Lit::N(nl), Lit::N(nr)) = (literal(left)?, literal(right)?) else {
                return None;
            };
            if let (Num::I(a), Num::I(b)) = (nl, nr) {
                match op {
                    Operator::Add => return Some(Lit::N(Num::I(a.checked_add(b)?))),
                    Operator::Sub => return Some(Lit::N(Num::I(a.checked_sub(b)?))),
                    Operator::Mul => return Some(Lit::N(Num::I(a.checked_mul(b)?))),
                    _ => {}
                }
            }
            let af = match nl {
                Num::I(i) => i as f64,
                Num::F(f) => f,
            };
            let bf = match nr {
                Num::I(i) => i as f64,
                Num::F(f) => f,
            };
            match op {
                Operator::Add => Some(Lit::N(Num::F(af + bf))),
                Operator::Sub => Some(Lit::N(Num::F(af - bf))),
                Operator::Mul => Some(Lit::N(Num::F(af * bf))),
                Operator::Div if bf != 0.0 => Some(Lit::N(Num::F(af / bf))),
                _ => None,
            }
        }
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
/// (no `_` or inner `%`), avoiding general wildcard backtracking per dictionary entry.
#[inline]
fn fast_like_literal_needle(pattern: &str, ci: bool) -> Option<&[u8]> {
    let b = pattern.as_bytes();
    if b.len() < 2 || b[0] != b'%' || b[b.len() - 1] != b'%' {
        return None;
    }
    let inner = &b[1..b.len() - 1];
    if inner.iter().any(|&c| c == b'%' || c == b'_') {
        return None;
    }
    if ci && !inner.is_ascii() {
        return None;
    }
    Some(inner)
}

#[inline]
fn fast_bytes_contains(hay: &[u8], needle: &[u8], ci: bool) -> bool {
    let n = needle.len();
    if n == 0 {
        return true;
    }
    if hay.len() < n {
        return false;
    }
    if !ci {
        let first = needle[0];
        let last = needle[n - 1];
        let end = hay.len() - n;
        let mut i = 0usize;
        while i <= end {
            if hay[i] == first && hay[i + n - 1] == last && &hay[i..i + n] == needle {
                return true;
            }
            i += 1;
        }
        false
    } else {
        let f_lo = needle[0].to_ascii_lowercase();
        let l_lo = needle[n - 1].to_ascii_lowercase();
        let end = hay.len() - n;
        let mut i = 0usize;
        while i <= end {
            if hay[i].to_ascii_lowercase() == f_lo
                && hay[i + n - 1].to_ascii_lowercase() == l_lo
                && hay[i..i + n].eq_ignore_ascii_case(needle)
            {
                return true;
            }
            i += 1;
        }
        false
    }
}
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
                        let epoch_i = s
                            .trim()
                            .parse::<i64>()
                            .ok()
                            .or_else(|| coerce_iso_temporal_for_int(t, col, s))?;
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
                    let ok: Vec<bool> = if let Some(needle) = fast_like_literal_needle(pat, ci) {
                        if dict.len() >= 2048 {
                            dict.par_iter()
                                .map(|d| fast_bytes_contains(d.as_bytes(), needle, ci) != neg)
                                .collect()
                        } else {
                            dict.iter()
                                .map(|d| fast_bytes_contains(d.as_bytes(), needle, ci) != neg)
                                .collect()
                        }
                    } else if dict.len() >= 2048 {
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
enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
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
    /// Composable vectorized `f64` arithmetic (`l_extendedprice * (1 - l_discount) * (1 + l_tax)`).
    FloatArith {
        lhs: Box<VCol<'a>>,
        op: FloatOp,
        rhs: Box<VCol<'a>>,
    },
    /// `EXTRACT(YEAR|MONTH|DAY FROM iso_date_text_col)` precomputed per dictionary entry.
    DictExtractInt {
        codes: &'a [u32],
        vals: Vec<Option<i64>>,
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
                | VCol::DictExtractInt { .. }
        )
    }

    fn is_numeric(&self) -> bool {
        self.is_int_like()
            || matches!(
                self,
                VCol::Phys(Col::Float(..)) | VCol::ConstFloat(_) | VCol::FloatArith { .. }
            )
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
            VCol::DictExtractInt { codes, vals } => {
                let c = codes[i];
                c == NULL_CODE || vals[c as usize].is_none()
            }
            VCol::DictRemap { codes, remap, .. } => {
                let c = codes[i];
                c == NULL_CODE || remap[c as usize] == NULL_CODE
            }
            VCol::CaseText { .. } => matches!(self.sv_at(i), Sv::Null),
            VCol::FloatArith { .. } => self.float_at(i).is_none(),
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
            VCol::DictExtractInt { codes, vals } => {
                let c = codes[i];
                if c == NULL_CODE {
                    None
                } else {
                    vals[c as usize]
                }
            }
            _ => None,
        }
    }

    #[inline]
    fn float_at(&self, i: usize) -> Option<f64> {
        match self {
            VCol::Phys(Col::Float(v, nulls)) => (!nl(nulls, i)).then(|| v[i]),
            VCol::ConstFloat(f) => Some(*f),
            VCol::FloatArith { lhs, op, rhs } => {
                let a = lhs.float_at(i)?;
                let b = rhs.float_at(i)?;
                match op {
                    FloatOp::Add => Some(a + b),
                    FloatOp::Sub => Some(a - b),
                    FloatOp::Mul => Some(a * b),
                    FloatOp::Div => (b != 0.0).then(|| a / b),
                }
            }
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
            | VCol::DictStrLen { .. }
            | VCol::DictExtractInt { .. } => match self.int_at(i) {
                Some(v) => (v as u64, false),
                None => (u64::MAX, true),
            },
            VCol::FloatArith { .. } => match self.float_at(i) {
                Some(f) => (if f == 0.0 { 0u64 } else { f.to_bits() }, false),
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
        if matches!(
            self,
            VCol::Phys(Col::Float(..)) | VCol::ConstFloat(_) | VCol::FloatArith { .. }
        ) && let Some(f) = self.float_at(i)
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
        Expression::Cast { expr, target_type } => {
            let tt = target_type.to_ascii_uppercase();
            if matches!(
                tt.as_str(),
                "DATE" | "TIMESTAMP" | "TEXT" | "VARCHAR" | "CHAR" | "STRING"
            ) {
                let inner = build_vcol(t, expr)?;
                if inner.is_text_like() {
                    Some(inner)
                } else {
                    None
                }
            } else if tt.starts_with("DECIMAL")
                || tt.starts_with("NUMERIC")
                || matches!(tt.as_str(), "FLOAT" | "DOUBLE" | "REAL")
            {
                let inner = build_vcol(t, expr)?;
                if inner.is_numeric() {
                    Some(inner)
                } else {
                    None
                }
            } else {
                None
            }
        }
        Expression::Binary { left, op, right } => {
            let arith = match op {
                Operator::Add => Some(ArithOp::Add),
                Operator::Sub => Some(ArithOp::Sub),
                Operator::Mul => Some(ArithOp::Mul),
                _ => None,
            };
            if let Some(arith) = arith {
                if let (Some(c), Some(Lit::N(Num::I(rhs)))) = (column(left), literal(right))
                    && let Some(Col::Int(ints, nulls)) = t.cols.get(c)
                {
                    return Some(VCol::IntArith {
                        ints,
                        nulls,
                        op: arith,
                        rhs,
                    });
                }
                if matches!(arith, ArithOp::Add | ArithOp::Mul)
                    && let (Some(Lit::N(Num::I(lhs))), Some(c)) = (literal(left), column(right))
                    && let Some(Col::Int(ints, nulls)) = t.cols.get(c)
                {
                    return Some(VCol::IntArith {
                        ints,
                        nulls,
                        op: arith,
                        rhs: lhs,
                    });
                }
            }
            let fop = match op {
                Operator::Add => FloatOp::Add,
                Operator::Sub => FloatOp::Sub,
                Operator::Mul => FloatOp::Mul,
                Operator::Div => FloatOp::Div,
                _ => return None,
            };
            let lv = build_vcol(t, left)?;
            let rv = build_vcol(t, right)?;
            if lv.is_numeric() && rv.is_numeric() && (!lv.is_int_like() || !rv.is_int_like()) {
                Some(VCol::FloatArith {
                    lhs: Box::new(lv),
                    op: fop,
                    rhs: Box::new(rv),
                })
            } else {
                None
            }
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
                    match t.cols.get(column(&args[1])?)? {
                        Col::Int(ints, nulls) => Some(VCol::ExtractInt { ints, nulls, part }),
                        Col::Text(codes, dict) => {
                            let vals: Vec<Option<i64>> = dict
                                .iter()
                                .map(|d| {
                                    let secs = parse_iso_timestamp_secs(d)
                                        .or_else(|| parse_iso_date_days(d).map(|days| days * 86_400))?;
                                    Some(extract_epoch(secs, part))
                                })
                                .collect();
                            Some(VCol::DictExtractInt { codes, vals })
                        }
                        _ => None,
                    }
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
enum DistinctSet {
    #[default]
    Empty,
    Inline4([u64; 4], u8),
    Heap(Box<rustc_hash::FxHashSet<u64>>),
    HeapStr(Box<rustc_hash::FxHashSet<Arc<str>>>),
}

impl DistinctSet {
    #[inline]
    fn insert_u64(&mut self, v: u64) {
        match self {
            DistinctSet::Empty => {
                *self = DistinctSet::Inline4([v, 0, 0, 0], 1);
            }
            DistinctSet::Inline4(arr, n) => {
                let len = *n as usize;
                for &x in &arr[..len] {
                    if x == v {
                        return;
                    }
                }
                if len < 4 {
                    arr[len] = v;
                    *n += 1;
                } else {
                    let mut set: rustc_hash::FxHashSet<u64> =
                        rustc_hash::FxHashSet::with_capacity_and_hasher(16, Default::default());
                    for &x in arr.iter() {
                        set.insert(x);
                    }
                    set.insert(v);
                    *self = DistinctSet::Heap(Box::new(set));
                }
            }
            DistinctSet::Heap(set) => {
                set.insert(v);
            }
            DistinctSet::HeapStr(_) => {}
        }
    }

    #[inline]
    fn insert_str(&mut self, s: Arc<str>) {
        match self {
            DistinctSet::HeapStr(set) => {
                set.insert(s);
            }
            _ => {
                let mut set: rustc_hash::FxHashSet<Arc<str>> =
                    rustc_hash::FxHashSet::with_capacity_and_hasher(8, Default::default());
                set.insert(s);
                *self = DistinctSet::HeapStr(Box::new(set));
            }
        }
    }

    fn merge_from(&mut self, other: &DistinctSet) {
        match other {
            DistinctSet::Empty => {}
            DistinctSet::Inline4(arr, n) => {
                for &x in &arr[..(*n as usize)] {
                    self.insert_u64(x);
                }
            }
            DistinctSet::Heap(oset) => match self {
                DistinctSet::Empty => {
                    *self = DistinctSet::Heap(oset.clone());
                }
                DistinctSet::Heap(set) => {
                    set.extend(oset.iter().copied());
                }
                DistinctSet::Inline4(arr, n) => {
                    let mut set = oset.clone();
                    for &x in &arr[..(*n as usize)] {
                        set.insert(x);
                    }
                    *self = DistinctSet::Heap(set);
                }
                DistinctSet::HeapStr(_) => {}
            },
            DistinctSet::HeapStr(oset) => match self {
                DistinctSet::HeapStr(set) => {
                    for s in oset.iter() {
                        set.insert(s.clone());
                    }
                }
                _ => {
                    *self = DistinctSet::HeapStr(oset.clone());
                }
            },
        }
    }

    #[inline]
    fn len(&self) -> u64 {
        match self {
            DistinctSet::Empty => 0,
            DistinctSet::Inline4(_, n) => *n as u64,
            DistinctSet::Heap(s) => s.len() as u64,
            DistinctSet::HeapStr(s) => s.len() as u64,
        }
    }
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
    distinct: DistinctSet,
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
        self.distinct.merge_from(&o.distinct);
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

    let proto: Vec<Acc> = (0..aggs.len()).map(|_| Acc::default()).collect();

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
                (
                    AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max,
                    Some(vc),
                ) if vc.is_numeric() => {
                    let want_mm = matches!(ag.kind, AggKind::Min | AggKind::Max);
                    for (k, r) in rows().enumerate() {
                        if let Some(x) = vc.float_at(r as usize) {
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
                        if !is_n {
                            accs[g(k) * nagg + oi].distinct.insert_u64(code);
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
                    AggKind::CountDistinct => Sv::U(a.distinct.len()),
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

// ============================================================================
// Out-of-Core Segmented Execution (`execute_segmented` over `DiskColumnTable`)
// ============================================================================

fn empty_arc_str() -> Arc<str> {
    static EMPTY: std::sync::OnceLock<Arc<str>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(|| Arc::from("")).clone()
}

fn collect_expr_cols<'a>(e: &'a Expression, valid: &HashMap<String, usize>, out: &mut HashSet<String>) {
    match e {
        Expression::Column(c) => {
            if valid.contains_key(c) {
                out.insert(c.clone());
            }
        }
        Expression::QualifiedColumn { column, .. } => {
            if valid.contains_key(column) {
                out.insert(column.clone());
            }
        }
        Expression::Binary { left, right, .. } => {
            collect_expr_cols(left, valid, out);
            collect_expr_cols(right, valid, out);
        }
        Expression::Unary { expr, .. }
        | Expression::IsNull { expr, .. }
        | Expression::Like { expr, .. }
        | Expression::RegexMatch { expr, .. }
        | Expression::SimilarTo { expr, .. }
        | Expression::LikeMeaning { expr, .. }
        | Expression::Cast { expr, .. } => {
            collect_expr_cols(expr, valid, out);
        }
        Expression::Function { args, .. } | Expression::WindowFunction { args, .. } => {
            for a in args {
                collect_expr_cols(a, valid, out);
            }
        }
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                collect_expr_cols(op, valid, out);
            }
            for (w, t) in when_clauses {
                collect_expr_cols(w, valid, out);
                collect_expr_cols(t, valid, out);
            }
            if let Some(el) = else_clause {
                collect_expr_cols(el, valid, out);
            }
        }
        Expression::In { expr, list, .. } => {
            collect_expr_cols(expr, valid, out);
            for x in list {
                collect_expr_cols(x, valid, out);
            }
        }
        Expression::Between {
            expr, low, high, ..
        } => {
            collect_expr_cols(expr, valid, out);
            collect_expr_cols(low, valid, out);
            collect_expr_cols(high, valid, out);
        }
        _ => {}
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SegVerdict {
    Skip,
    AllMatch,
    Maybe,
}

fn seg_pred_one(
    dt: &crate::disk_col_segments::DiskColumnTable,
    proto_tc: &TableColumns,
    seg_idx: usize,
    p: &Pred,
) -> Option<bool> {
    use crate::disk_col_segments::DiskColKind;
    let seg = &dt.segments[seg_idx];
    if seg.len == 0 {
        return Some(false);
    }
    match p {
        Pred::IsNull { col, negated } => {
            let &ci = dt.col_index.get(*col)?;
            let m = &seg.cols[ci];
            if *negated {
                if m.null_count >= seg.len {
                    Some(false)
                } else if m.null_count == 0 {
                    Some(true)
                } else {
                    None
                }
            } else if m.null_count == 0 {
                Some(false)
            } else if m.null_count >= seg.len {
                Some(true)
            } else {
                None
            }
        }
        Pred::Cmp { col, op, lit } => {
            let &ci = dt.col_index.get(*col)?;
            let m = &seg.cols[ci];
            if m.null_count >= seg.len {
                return Some(false);
            }
            let global_kind = dt.inferred_col_kind(ci);
            match global_kind {
                DiskColKind::I8 | DiskColKind::I16 | DiskColKind::I32 | DiskColKind::I64 => {
                    let (lo, hi) = (m.min_int(seg.len)?, m.max_int(seg.len)?);
                    let x_num = match lit {
                        Lit::N(n) => *n,
                        Lit::S(s) => Num::I(coerce_iso_temporal_for_int(proto_tc, col, s)?),
                    };
                    let (cl, ch) = (num_cmp(Num::I(lo), x_num)?, num_cmp(Num::I(hi), x_num)?);
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
                    } else if all && m.null_count == 0 {
                        Some(true)
                    } else {
                        None
                    }
                }
                DiskColKind::F64 => {
                    let (lo, hi) = match (m.min_float(seg.len), m.max_float(seg.len)) {
                        (Some(a), Some(b)) => (a, b),
                        _ => match (m.min_int(seg.len), m.max_int(seg.len)) {
                            (Some(a), Some(b)) => (a as f64, b as f64),
                            _ => return None,
                        },
                    };
                    let Lit::N(x_num) = lit else { return None };
                    let (cl, ch) = (num_cmp(Num::F(lo), *x_num)?, num_cmp(Num::F(hi), *x_num)?);
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
                    } else if all && m.null_count == 0 {
                        Some(true)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        Pred::In {
            col,
            lits,
            negated: false,
        } => {
            let &ci = dt.col_index.get(*col)?;
            let m = &seg.cols[ci];
            if m.null_count >= seg.len {
                return Some(false);
            }
            if matches!(
                dt.inferred_col_kind(ci),
                DiskColKind::I8 | DiskColKind::I16 | DiskColKind::I32 | DiskColKind::I64
            ) && let (Some(lo), Some(hi)) = (m.min_int(seg.len), m.max_int(seg.len))
            {
                let mut any_in_range = false;
                for l in lits {
                    let v = match l {
                        Lit::N(Num::I(x)) => *x,
                        Lit::S(s) => coerce_iso_temporal_for_int(proto_tc, col, s)?,
                        _ => return None,
                    };
                    if v >= lo && v <= hi {
                        any_in_range = true;
                        break;
                    }
                }
                if !any_in_range {
                    return Some(false);
                }
            }
            None
        }
        _ => None,
    }
}

fn seg_verdict(
    dt: &crate::disk_col_segments::DiskColumnTable,
    proto_tc: &TableColumns,
    seg_idx: usize,
    preds: &[Pred],
) -> SegVerdict {
    if dt.segments[seg_idx].len == 0 {
        return SegVerdict::Skip;
    }
    let mut all = true;
    for p in preds {
        match seg_pred_one(dt, proto_tc, seg_idx, p) {
            Some(false) => return SegVerdict::Skip,
            Some(true) => {}
            None => all = false,
        }
    }
    if all {
        SegVerdict::AllMatch
    } else {
        SegVerdict::Maybe
    }
}

fn eval_segment_where(seg_tc: &TableColumns, raw_preds: &[Pred]) -> Option<Option<Vec<u32>>> {
    if raw_preds.is_empty() {
        return Some(None);
    }
    let preds = prepare_preds(seg_tc, raw_preds)?;
    let n = seg_tc.len;
    let mut sel = Vec::new();
    let mut all_matched = true;
    for s in (0..n).step_by(ZONE) {
        let e = (s + ZONE).min(n);
        match morsel_sel_prep(seg_tc, &preds, s, e) {
            None => {
                if !all_matched {
                    sel.extend(s as u32..e as u32);
                }
            }
            Some(v) => {
                if all_matched && v.len() < e - s {
                    all_matched = false;
                    sel.reserve(s + v.len());
                    sel.extend(0..s as u32);
                }
                if !all_matched {
                    sel.extend(v);
                }
            }
        }
    }
    if all_matched {
        Some(None)
    } else {
        Some(Some(sel))
    }
}

#[inline]
fn mix64(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    h
}

#[inline]
fn hash_bytes64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ (bytes.len() as u64);
    let mut chunks = bytes.chunks_exact(8);
    for ch in &mut chunks {
        let w = u64::from_le_bytes([ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7]]);
        h = (h ^ w).wrapping_mul(0x100000001b3);
    }
    for &b in chunks.remainder() {
        h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
    }
    mix64(h)
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum GVal {
    Null,
    I(i64),
    F(u64),
    S(Arc<str>),
}

impl GVal {
    fn to_sv(&self) -> Sv {
        match self {
            GVal::Null => Sv::Null,
            GVal::I(i) => Sv::I(*i),
            GVal::F(bits) => Sv::F(f64::from_bits(*bits)),
            GVal::S(s) => Sv::S(s.clone()),
        }
    }

    #[inline]
    fn slot_hash(&self) -> u64 {
        match self {
            GVal::Null => 0x9e3779b97f4a7c15,
            GVal::I(i) => mix64((*i as u64) ^ 0x517cc1b727220a95),
            GVal::F(bits) => mix64(*bits ^ 0x2545f4914f6cdd1d),
            GVal::S(s) => hash_bytes64(s.as_bytes()),
        }
    }
}

/// Cross-segment canonical group key with inline zero-Box layouts for 1-, 2-, and 3-key ClickBench shapes.
#[derive(Clone, PartialEq, Eq, Hash)]
enum GKey {
    I1(i64, bool),
    I2(i64, i64, u8),
    S1(Option<Arc<str>>),
    IS(i64, bool, Option<Arc<str>>),
    IIS(i64, i64, u8, Option<Arc<str>>),
    Many(Box<[GVal]>),
}

#[derive(Clone, Copy)]
enum GKeyShape {
    I1,
    I2,
    S1,
    IS,
    IIS,
    Many,
}

impl GKey {
    fn slot_sv(&self, idx: usize) -> Sv {
        match (self, idx) {
            (GKey::I1(a, na), 0) => {
                if *na {
                    Sv::Null
                } else {
                    Sv::I(*a)
                }
            }
            (GKey::I2(a, _, m), 0) => {
                if m & 1 != 0 {
                    Sv::Null
                } else {
                    Sv::I(*a)
                }
            }
            (GKey::I2(_, b, m), 1) => {
                if m & 2 != 0 {
                    Sv::Null
                } else {
                    Sv::I(*b)
                }
            }
            (GKey::S1(s), 0) => match s {
                Some(v) => Sv::S(v.clone()),
                None => Sv::Null,
            },
            (GKey::IS(a, na, _), 0) => {
                if *na {
                    Sv::Null
                } else {
                    Sv::I(*a)
                }
            }
            (GKey::IS(_, _, s), 1) => match s {
                Some(v) => Sv::S(v.clone()),
                None => Sv::Null,
            },
            (GKey::IIS(a, _, m, _), 0) => {
                if m & 1 != 0 {
                    Sv::Null
                } else {
                    Sv::I(*a)
                }
            }
            (GKey::IIS(_, b, m, _), 1) => {
                if m & 2 != 0 {
                    Sv::Null
                } else {
                    Sv::I(*b)
                }
            }
            (GKey::IIS(_, _, _, s), 2) => match s {
                Some(v) => Sv::S(v.clone()),
                None => Sv::Null,
            },
            (GKey::Many(vs), i) => vs.get(i).map_or(Sv::Null, GVal::to_sv),
            _ => Sv::Null,
        }
    }

    fn slot_int(&self, idx: usize) -> Option<i64> {
        match self.slot_sv(idx) {
            Sv::I(x) => Some(x),
            _ => None,
        }
    }

    #[inline]
    fn canonical_hash(&self) -> u64 {
        let step = |acc: u64, sh: u64| mix64(acc.wrapping_mul(0x9e3779b97f4a7c15) ^ sh);
        let h_int = |v: i64, is_n: bool| {
            if is_n {
                0x9e3779b97f4a7c15
            } else {
                mix64((v as u64) ^ 0x517cc1b727220a95)
            }
        };
        let h_str = |s: &Option<Arc<str>>| match s {
            None => 0x9e3779b97f4a7c15,
            Some(v) => hash_bytes64(v.as_bytes()),
        };
        match self {
            GKey::I1(a, na) => step(0, h_int(*a, *na)),
            GKey::I2(a, b, m) => step(step(0, h_int(*a, m & 1 != 0)), h_int(*b, m & 2 != 0)),
            GKey::S1(s) => step(0, h_str(s)),
            GKey::IS(a, na, s) => step(step(0, h_int(*a, *na)), h_str(s)),
            GKey::IIS(a, b, m, s) => step(
                step(step(0, h_int(*a, m & 1 != 0)), h_int(*b, m & 2 != 0)),
                h_str(s),
            ),
            GKey::Many(vs) => {
                let mut acc = 0u64;
                for v in vs.iter() {
                    acc = step(acc, v.slot_hash());
                }
                acc
            }
        }
    }
}

#[inline]
fn row_canonical_hash(vcols: &[VCol], ri: usize) -> u64 {
    let mut acc = 0u64;
    for vc in vcols {
        let sh = if vc.is_int_like() {
            match vc.int_at(ri) {
                Some(v) => mix64((v as u64) ^ 0x517cc1b727220a95),
                None => 0x9e3779b97f4a7c15,
            }
        } else if matches!(
            vc,
            VCol::Phys(Col::Float(..)) | VCol::ConstFloat(_) | VCol::FloatArith { .. }
        ) {
            match vc.float_at(ri) {
                Some(f) => {
                    let bits = if f == 0.0 { 0u64 } else { f.to_bits() };
                    mix64(bits ^ 0x2545f4914f6cdd1d)
                }
                None => 0x9e3779b97f4a7c15,
            }
        } else {
            match vc.str_ref_at(ri) {
                Some(s) => hash_bytes64(s.as_bytes()),
                None => 0x9e3779b97f4a7c15,
            }
        };
        acc = mix64(acc.wrapping_mul(0x9e3779b97f4a7c15) ^ sh);
    }
    acc
}

#[inline]
fn opt_arc_at(vc: &VCol, ri: usize) -> Option<Arc<str>> {
    let sr = vc.str_ref_at(ri)?;
    if sr.is_empty() {
        Some(empty_arc_str())
    } else {
        vc.arc_str_at(ri)
    }
}

#[inline]
fn build_gkey(shape: GKeyShape, vcols: &[VCol], ri: usize) -> GKey {
    match shape {
        GKeyShape::I1 => match vcols[0].int_at(ri) {
            Some(a) => GKey::I1(a, false),
            None => GKey::I1(0, true),
        },
        GKeyShape::I2 => {
            let (a, na) = match vcols[0].int_at(ri) {
                Some(x) => (x, 0u8),
                None => (0, 1u8),
            };
            let (b, nb) = match vcols[1].int_at(ri) {
                Some(x) => (x, 0u8),
                None => (0, 2u8),
            };
            GKey::I2(a, b, na | nb)
        }
        GKeyShape::S1 => GKey::S1(opt_arc_at(&vcols[0], ri)),
        GKeyShape::IS => {
            let (a, na) = match vcols[0].int_at(ri) {
                Some(x) => (x, false),
                None => (0, true),
            };
            GKey::IS(a, na, opt_arc_at(&vcols[1], ri))
        }
        GKeyShape::IIS => {
            let (a, na) = match vcols[0].int_at(ri) {
                Some(x) => (x, 0u8),
                None => (0, 1u8),
            };
            let (b, nb) = match vcols[1].int_at(ri) {
                Some(x) => (x, 0u8),
                None => (0, 2u8),
            };
            GKey::IIS(a, b, na | nb, opt_arc_at(&vcols[2], ri))
        }
        GKeyShape::Many => {
            let vs: Box<[GVal]> = vcols
                .iter()
                .map(|vc| {
                    if vc.is_int_like() {
                        vc.int_at(ri).map_or(GVal::Null, GVal::I)
                    } else if matches!(
                        vc,
                        VCol::Phys(Col::Float(..)) | VCol::ConstFloat(_) | VCol::FloatArith { .. }
                    ) {
                        vc.float_at(ri).map_or(GVal::Null, |f| {
                            GVal::F(if f == 0.0 { 0u64 } else { f.to_bits() })
                        })
                    } else {
                        opt_arc_at(vc, ri).map_or(GVal::Null, GVal::S)
                    }
                })
                .collect();
            GKey::Many(vs)
        }
    }
}

#[derive(Clone)]
enum GroupKeyMap {
    Slot(usize),
    Const(Sv),
    DerivedInt {
        base_slot: usize,
        op: ArithOp,
        rhs: i64,
    },
}

fn compile_having_seg<'b>(
    e: &'b Expression,
    proto_tc: &'b TableColumns,
    names: &[String],
    group_exprs: &[&'b Expression],
    out_exprs: &mut Vec<Option<&'b Expression>>,
    outs: &mut Vec<OutCol>,
    agg_exprs: &mut Vec<(&'b Expression, AggKind, i64, bool, bool)>,
) -> Option<HavingPred> {
    match e {
        Expression::Binary {
            left,
            op: Operator::And,
            right,
        } => Some(HavingPred::And(
            Box::new(compile_having_seg(
                left,
                proto_tc,
                names,
                group_exprs,
                out_exprs,
                outs,
                agg_exprs,
            )?),
            Box::new(compile_having_seg(
                right,
                proto_tc,
                names,
                group_exprs,
                out_exprs,
                outs,
                agg_exprs,
            )?),
        )),
        Expression::Binary {
            left,
            op: Operator::Or,
            right,
        } => Some(HavingPred::Or(
            Box::new(compile_having_seg(
                left,
                proto_tc,
                names,
                group_exprs,
                out_exprs,
                outs,
                agg_exprs,
            )?),
            Box::new(compile_having_seg(
                right,
                proto_tc,
                names,
                group_exprs,
                out_exprs,
                outs,
                agg_exprs,
            )?),
        )),
        Expression::Unary {
            op: UnaryOperator::Not,
            expr,
        } => Some(HavingPred::Not(Box::new(compile_having_seg(
            expr,
            proto_tc,
            names,
            group_exprs,
            out_exprs,
            outs,
            agg_exprs,
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
                if let Some((kind, vc_proto, const_add, _)) = parse_agg_expr(proto_tc, side) {
                    let is_int = vc_proto.as_ref().is_some_and(VCol::is_int_like);
                    let is_txt = vc_proto.as_ref().is_some_and(VCol::is_text_like);
                    let ai = agg_exprs.len();
                    agg_exprs.push((side, kind, const_add, is_int, is_txt));
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

fn render_acc_cell(a: &Acc, kind: AggKind, const_add: i64, is_int: bool, is_txt: bool) -> Sv {
    match kind {
        AggKind::CountStar | AggKind::Count => Sv::U(a.count),
        AggKind::CountDistinct if !a.bits.is_empty() => {
            Sv::U(a.bits.iter().map(|w| w.count_ones() as u64).sum())
        }
        AggKind::CountDistinct => Sv::U(a.distinct.len()),
        AggKind::Sum if a.count == 0 => Sv::Null,
        AggKind::Sum if is_int => {
            let shifted = a.isum + (const_add as i128) * (a.count as i128);
            match i64::try_from(shifted) {
                Ok(v) => Sv::I(v),
                Err(_) => Sv::F(shifted as f64),
            }
        }
        AggKind::Sum => Sv::F((a.fsum + a.fcomp) + (const_add as f64) * (a.count as f64)),
        AggKind::Avg if a.count == 0 => Sv::Null,
        AggKind::Avg => {
            let s = if is_int {
                a.isum as f64
            } else {
                a.fsum + a.fcomp
            };
            Sv::F(s / a.count as f64 + const_add as f64)
        }
        AggKind::Min if is_txt => match &a.smin {
            Some(s) => Sv::S(s.clone()),
            None => Sv::Null,
        },
        AggKind::Max if is_txt => match &a.smax {
            Some(s) => Sv::S(s.clone()),
            None => Sv::Null,
        },
        AggKind::Min => Sv::from_num_offset(a.min, const_add),
        AggKind::Max => Sv::from_num_offset(a.max, const_add),
    }
}

fn update_acc_row(a: &mut Acc, kind: AggKind, vc: Option<&VCol>, ri: usize) {
    match (kind, vc) {
        (AggKind::CountStar, _) => a.count += 1,
        (AggKind::Count, Some(c)) => {
            if !c.is_null_at(ri) {
                a.count += 1;
            }
        }
        (AggKind::CountDistinct, Some(c)) => {
            if c.is_text_like() {
                if let Some(s) = opt_arc_at(c, ri) {
                    a.distinct.insert_str(s);
                }
            } else {
                let (code, is_n) = c.group_code_at(ri);
                if !is_n {
                    a.distinct.insert_u64(code);
                }
            }
        }
        (AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max, Some(c)) if c.is_int_like() => {
            if let Some(x) = c.int_at(ri) {
                a.count += 1;
                a.isum += x as i128;
                if matches!(kind, AggKind::Min | AggKind::Max) {
                    if a.min.is_none_or(|m| num_lt(Num::I(x), m)) {
                        a.min = Some(Num::I(x));
                    }
                    if a.max.is_none_or(|m| num_lt(m, Num::I(x))) {
                        a.max = Some(Num::I(x));
                    }
                }
            }
        }
        (AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max, Some(c)) if c.is_numeric() => {
            if let Some(x) = c.float_at(ri) {
                a.count += 1;
                neumaier(&mut a.fsum, &mut a.fcomp, x);
                if matches!(kind, AggKind::Min | AggKind::Max) {
                    if a.min.is_none_or(|m| num_lt(Num::F(x), m)) {
                        a.min = Some(Num::F(x));
                    }
                    if a.max.is_none_or(|m| num_lt(m, Num::F(x))) {
                        a.max = Some(Num::F(x));
                    }
                }
            }
        }
        (AggKind::Min | AggKind::Max, Some(c)) if c.is_text_like() => {
            if let Some(sr) = c.str_ref_at(ri) {
                a.count += 1;
                if matches!(kind, AggKind::Min) && a.smin.as_deref().is_none_or(|cur| sr < cur) {
                    a.smin = opt_arc_at(c, ri);
                }
                if matches!(kind, AggKind::Max) && a.smax.as_deref().is_none_or(|cur| sr > cur) {
                    a.smax = opt_arc_at(c, ri);
                }
            }
        }
        _ => {}
    }
}

/// Two-phase Top-K / plain scan across `.jcol` segments (`Q19, Q23, Q24, Q25, Q26`).
/// Reads ONLY `WHERE` + `ORDER BY` columns across segments, then loads `SELECT`
/// projection columns (`SELECT *` = 105 cols!) ONLY for segments containing winning rows!
fn execute_scan_segmented(
    query: &SqlQuery,
    dt: &crate::disk_col_segments::DiskColumnTable,
    proto_tc: &TableColumns,
    raw_preds: &[Pred],
) -> Option<VecResult> {
    use rayon::prelude::*;
    let mut names = Vec::new();
    let mut has_wildcard = false;
    let mut proj_col_set: HashSet<String> = HashSet::new();

    for sc in &query.columns {
        match &sc.expr {
            Expression::Wildcard | Expression::QualifiedWildcard(_) => {
                if sc.alias.is_some() {
                    return None;
                }
                for cn in &dt.col_order {
                    if matches!(proto_tc.cols.get(cn)?, Col::Other) {
                        return None;
                    }
                    names.push(cn.clone());
                }
                has_wildcard = true;
            }
            e => {
                let _ = build_vcol(proto_tc, e)?;
                collect_expr_cols(e, &dt.col_index, &mut proj_col_set);
                let default_name = match e {
                    Expression::Column(n) => n.clone(),
                    Expression::QualifiedColumn { column, .. } => column.clone(),
                    Expression::Function { name, .. } => name.clone(),
                    _ => "expr".to_string(),
                };
                names.push(sc.alias.clone().unwrap_or(default_name));
            }
        }
    }
    if names.is_empty() {
        return None;
    }

    let mut order_exprs: Vec<(&Expression, bool, Option<bool>)> =
        Vec::with_capacity(query.order_by.len());
    let mut order_col_set: HashSet<String> = HashSet::new();
    for ob in &query.order_by {
        let resolved: &Expression = match &ob.expr {
            Expression::Literal(AstValue::Int(i))
                if *i >= 1 && (*i as usize) <= query.columns.len() =>
            {
                &query.columns[*i as usize - 1].expr
            }
            e => {
                if let Some(n) = column(e)
                    && !proto_tc.cols.contains_key(n)
                    && !has_wildcard
                    && let Some(pos) = names.iter().position(|x| x == n)
                    && pos < query.columns.len()
                {
                    &query.columns[pos].expr
                } else {
                    e
                }
            }
        };
        let _ = build_vcol(proto_tc, resolved)?;
        collect_expr_cols(resolved, &dt.col_index, &mut order_col_set);
        order_exprs.push((resolved, ob.descending, ob.nulls_first));
    }

    let off = query.offset.unwrap_or(0);
    let take = query.limit.map(|l| off.saturating_add(l));
    if raw_preds.is_empty() && take.is_none() && dt.total_rows > 250_000 {
        return None;
    }
    if take == Some(0) {
        return Some(VecResult {
            columns: names,
            rows: Vec::new(),
        });
    }

    let mut where_col_set: HashSet<String> = HashSet::new();
    if let Some(w) = &query.where_clause {
        collect_expr_cols(w, &dt.col_index, &mut where_col_set);
    }
    let where_cols: Vec<&str> = where_col_set.iter().map(String::as_str).collect();
    let order_only_cols: Vec<&str> = order_col_set
        .iter()
        .filter(|c| !where_col_set.contains(*c))
        .map(String::as_str)
        .collect();

    let active_segs: Vec<usize> = (0..dt.segments.len())
        .filter(|&si| seg_verdict(dt, proto_tc, si, raw_preds) != SegVerdict::Skip)
        .collect();

    let winners: Vec<(usize, u32)> = if order_exprs.is_empty() {
        if (take.is_none() || take.is_some_and(|tk| tk >= dt.total_rows)) && active_segs.len() > 1 {
            let seg_matches: Vec<Vec<(usize, u32)>> = active_segs
                .par_iter()
                .map(|&si| -> Option<Vec<(usize, u32)>> {
                    let seg_tc = dt.load_segment_columns(si, &where_cols).ok()?;
                    let sel = eval_segment_where(&seg_tc, raw_preds)?;
                    let out = match sel {
                        None => (0..seg_tc.len as u32).map(|r| (si, r)).collect(),
                        Some(v) => v.into_iter().map(|r| (si, r)).collect(),
                    };
                    Some(out)
                })
                .collect::<Option<_>>()?;
            seg_matches.into_iter().flatten().collect()
        } else {
            let mut collected: Vec<(usize, u32)> = Vec::new();
            for &si in &active_segs {
                let mut seg_tc = dt.load_segment_columns(si, &where_cols).ok()?;
                let sel = eval_segment_where(&seg_tc, raw_preds)?;
                match sel {
                    None => {
                        let need = take.map_or(seg_tc.len, |k| k.saturating_sub(collected.len()));
                        let lim = seg_tc.len.min(need);
                        for r in 0..lim {
                            collected.push((si, r as u32));
                        }
                    }
                    Some(v) => {
                        let need = take.map_or(v.len(), |k| k.saturating_sub(collected.len()));
                        for &r in v.iter().take(need) {
                            collected.push((si, r));
                        }
                    }
                }
                seg_tc.cols.clear();
                if let Some(k) = take
                    && collected.len() >= k
                {
                    collected.truncate(k);
                    break;
                }
            }
            collected
        }
    } else {
        let nv = order_exprs.len();
        let order_spec: Vec<(usize, bool, Option<bool>)> = order_exprs
            .iter()
            .enumerate()
            .map(|(j, (_, d, nf))| (j, *d, *nf))
            .collect();

        // Dynamic Top-K ZoneMap bound pruning on leading physical integer ORDER BY column (`Q23, Q24, Q26`):
        // When `ORDER BY int_col ASC [NULLS LAST] LIMIT k`, seed the k-th threshold from the
        // minimum-ZoneMap segments first so segments with `seg_min_int > kth_bound` skip reading
        // heavyweight `WHERE` text slices (`URL`, `SearchPhrase`) entirely!
        let asc_int_ord_ci: Option<usize> = if let Some(k) = take
            && k > 0
            && !order_exprs[0].1
            && !order_exprs[0].2.unwrap_or(false)
            && let Some(cn) = column(order_exprs[0].0)
            && let Some(&ci) = dt.col_index.get(cn)
            && matches!(
                dt.inferred_col_kind(ci),
                crate::disk_col_segments::DiskColKind::I8
                    | crate::disk_col_segments::DiskColKind::I16
                    | crate::disk_col_segments::DiskColKind::I32
                    | crate::disk_col_segments::DiskColKind::I64
            )
        {
            Some(ci)
        } else {
            None
        };

        let mut ordered_segs = active_segs.clone();
        if let Some(ci) = asc_int_ord_ci {
            ordered_segs.sort_by_key(|&si| {
                let seg = &dt.segments[si];
                seg.cols[ci].min_int(seg.len).unwrap_or(i64::MAX)
            });
        }

        let kth_bound = std::sync::atomic::AtomicI64::new(i64::MAX);
        let ord_col_name: Option<&str> = asc_int_ord_ci.map(|ci| dt.col_order[ci].as_str());
        let eval_one_top_seg = |si: usize| -> Option<Vec<(Vec<Sv>, usize, u32)>> {
            let mut pre_ord_cands: Option<Vec<u32>> = None;
            if let Some(ci) = asc_int_ord_ci {
                let cur_b = kth_bound.load(std::sync::atomic::Ordering::Relaxed);
                if cur_b < i64::MAX {
                    let seg = &dt.segments[si];
                    if seg.cols[ci].null_count == 0
                        && let Some(seg_lo) = seg.cols[ci].min_int(seg.len)
                        && seg_lo > cur_b
                    {
                        return Some(Vec::new());
                    }
                    // Order-column-first row pruning (`Q23, Q24, Q26`): load ONLY the 1-8 MB
                    // integer `order_col` (`EventTime`) BEFORE reading 25+ MB text `WHERE` columns
                    // (`URL`, `SearchPhrase`). If 0 rows have `order_col <= cur_b`, skip the
                    // heavyweight text columns completely; otherwise sparsely decode ONLY the
                    // candidate rows!
                    if let Some(ocn) = ord_col_name
                        && where_cols.iter().any(|&wc| wc != ocn)
                    {
                        let ord_tc = dt.load_segment_columns(si, &[ocn]).ok()?;
                        if let Some(Col::Int(ints, nulls)) = ord_tc.cols.get(ocn) {
                            let mut cands = Vec::new();
                            if nulls.is_empty() {
                                for r in 0..ord_tc.len {
                                    if ints.at(r) <= cur_b {
                                        cands.push(r as u32);
                                    }
                                }
                            } else {
                                for r in 0..ord_tc.len {
                                    if nulls[r] || ints.at(r) <= cur_b {
                                        cands.push(r as u32);
                                    }
                                }
                            }
                            if cands.is_empty() {
                                return Some(Vec::new());
                            }
                            pre_ord_cands = Some(cands);
                        }
                    }
                }
            }
            let (mut seg_tc, sel) = if let Some(ord_cands) = pre_ord_cands {
                let mut tc = dt
                    .load_segment_columns_selected(si, &where_cols, Some(&ord_cands))
                    .ok()?;
                let preds = prepare_preds(&tc, raw_preds)?;
                let mut filtered = Vec::with_capacity(ord_cands.len().min(256));
                for &r in &ord_cands {
                    let ri = r as usize;
                    if preds.iter().all(|p| eval_prep_pred_row(p, ri)) {
                        filtered.push(r);
                    }
                }
                if filtered.is_empty() {
                    return Some(Vec::new());
                }
                if !order_only_cols.is_empty() {
                    dt.load_columns_into_selected(si, &order_only_cols, Some(&filtered), &mut tc)
                        .ok()?;
                }
                (tc, Some(filtered))
            } else {
                let mut tc = dt.load_segment_columns(si, &where_cols).ok()?;
                let sel = eval_segment_where(&tc, raw_preds)?;
                if let Some(ref v) = sel
                    && v.is_empty()
                {
                    return Some(Vec::new());
                }
                if !order_only_cols.is_empty() {
                    dt.load_columns_into_selected(si, &order_only_cols, sel.as_deref(), &mut tc)
                        .ok()?;
                }
                (tc, sel)
            };
            let _ = &mut seg_tc;
            let o_vcols: Vec<(VCol, bool, Option<bool>)> = order_exprs
                .iter()
                .map(|(e, d, nf)| Some((build_vcol(&seg_tc, e)?, *d, *nf)))
                .collect::<Option<_>>()?;

            let cmp_seg = |&ra: &u32, &rb: &u32| -> Ordering {
                let (ia, ib) = (ra as usize, rb as usize);
                for (vc, desc, nulls_first) in &o_vcols {
                    let (na, nb) = (vc.is_null_at(ia), vc.is_null_at(ib));
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
                            let ord = if let (Some(xa), Some(xb)) = (vc.int_at(ia), vc.int_at(ib))
                            {
                                xa.cmp(&xb)
                            } else if let (Some(sa), Some(sb)) =
                                (vc.str_ref_at(ia), vc.str_ref_at(ib))
                            {
                                sa.cmp(sb)
                            } else if let (Some(fa), Some(fb)) =
                                (vc.float_at(ia), vc.float_at(ib))
                            {
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

            let mut v: Vec<u32> = match sel {
                Some(v) => v,
                None => (0..seg_tc.len as u32).collect(),
            };
            if let Some(k) = take
                && k > 0
                && k < v.len()
            {
                v.select_nth_unstable_by(k - 1, &cmp_seg);
                v.truncate(k);
            }
            if asc_int_ord_ci.is_some()
                && let Some(k) = take
                && v.len() >= k
                && k > 0
            {
                let mut max_in_k = i64::MIN;
                let mut all_non_null = true;
                for &ri in &v[..k] {
                    match o_vcols[0].0.int_at(ri as usize) {
                        Some(val) => max_in_k = max_in_k.max(val),
                        None => {
                            all_non_null = false;
                            break;
                        }
                    }
                }
                if all_non_null {
                    kth_bound.fetch_min(max_in_k, std::sync::atomic::Ordering::Relaxed);
                }
            }
            let out = v
                .into_iter()
                .map(|ri| {
                    let keys: Vec<Sv> =
                        o_vcols.iter().map(|(vc, _, _)| vc.sv_at(ri as usize)).collect();
                    (keys, si, ri)
                })
                .collect();
            Some(out)
        };

        let seed_n = if asc_int_ord_ci.is_some() && ordered_segs.len() > 2 {
            4usize.min(ordered_segs.len())
        } else {
            0usize
        };
        let mut seg_cands: Vec<Vec<(Vec<Sv>, usize, u32)>> = Vec::with_capacity(ordered_segs.len());
        if seed_n > 0 {
            let mut seeded_rows = 0usize;
            let mut idx = 0usize;
            while idx < ordered_segs.len() && (idx < seed_n || seeded_rows < take.unwrap_or(0)) {
                let c = eval_one_top_seg(ordered_segs[idx])?;
                seeded_rows += c.len();
                seg_cands.push(c);
                idx += 1;
                if seeded_rows >= take.unwrap_or(0) && idx >= seed_n {
                    break;
                }
            }
            // Tighten bound across combined seed candidates before parallel fan-out
            if let Some(k) = take
                && seeded_rows >= k
                && k > 0
            {
                let mut vals: Vec<i64> = seg_cands
                    .iter()
                    .flatten()
                    .filter_map(|(ks, _, _)| match ks.first()? {
                        Sv::I(x) => Some(*x),
                        _ => None,
                    })
                    .collect();
                if vals.len() >= k {
                    vals.select_nth_unstable(k - 1);
                    kth_bound.fetch_min(vals[k - 1], std::sync::atomic::Ordering::Relaxed);
                }
            }
            let rest_cands: Vec<Vec<(Vec<Sv>, usize, u32)>> = ordered_segs[idx..]
                .par_iter()
                .map(|&si| eval_one_top_seg(si))
                .collect::<Option<_>>()?;
            seg_cands.extend(rest_cands);
        } else {
            seg_cands = ordered_segs
                .par_iter()
                .map(|&si| eval_one_top_seg(si))
                .collect::<Option<_>>()?;
        }

        let mut flat: Vec<(Vec<Sv>, usize, u32)> = seg_cands.into_iter().flatten().collect();
        let cmp_global = |a: &(Vec<Sv>, usize, u32), b: &(Vec<Sv>, usize, u32)| -> Ordering {
            for &(j, desc, nulls_first) in &order_spec {
                let (x, y) = (&a.0[j], &b.0[j]);
                let nf = nulls_first.unwrap_or(desc);
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
                    (false, false) => {
                        let ord = x.cmp_val(y);
                        if desc { ord.reverse() } else { ord }
                    }
                };
                if o != Ordering::Equal {
                    return o;
                }
            }
            let _ = nv;
            (a.1, a.2).cmp(&(b.1, b.2))
        };
        if let Some(k) = take
            && k > 0
            && k < flat.len()
        {
            flat.select_nth_unstable_by(k - 1, &cmp_global);
            flat.truncate(k);
        }
        flat.sort_unstable_by(&cmp_global);
        flat.into_iter().map(|(_, si, ri)| (si, ri)).collect()
    };

    let end = take.unwrap_or(usize::MAX).min(winners.len());
    let final_slice = &winners[off.min(end)..end];
    if final_slice.is_empty() {
        return Some(VecResult {
            columns: names,
            rows: Vec::new(),
        });
    }

    let proj_cols: Vec<&str> = if has_wildcard {
        dt.col_order.iter().map(String::as_str).collect()
    } else {
        proj_col_set.iter().map(String::as_str).collect()
    };

    // Group winning rows by segment and sparsely decode only winning row codes (`SELECT *` in `Q23`!)
    let mut seg_row_map: HashMap<usize, Vec<u32>> = HashMap::new();
    for &(si, ri) in final_slice {
        seg_row_map.entry(si).or_default().push(ri);
    }
    let mut loaded_segs: HashMap<usize, TableColumns> = HashMap::with_capacity(seg_row_map.len());
    for (si, mut ris) in seg_row_map {
        ris.sort_unstable();
        ris.dedup();
        loaded_segs.insert(
            si,
            dt.load_segment_columns_selected(si, &proj_cols, Some(&ris))
                .ok()?,
        );
    }

    let mut rows = Vec::with_capacity(final_slice.len());
    for &(si, ri) in final_slice {
        let seg_tc = loaded_segs.get(&si)?;
        let mut row = Vec::with_capacity(names.len());
        for sc in &query.columns {
            match &sc.expr {
                Expression::Wildcard | Expression::QualifiedWildcard(_) => {
                    for cn in &dt.col_order {
                        let col = seg_tc.cols.get(cn)?;
                        row.push(VCol::Phys(col).sv_at(ri as usize).json());
                    }
                }
                e => {
                    let vc = build_vcol(seg_tc, e)?;
                    row.push(vc.sv_at(ri as usize).json());
                }
            }
        }
        rows.push(row);
    }

    Some(VecResult {
        columns: names,
        rows,
    })
}

/// Vectorized out-of-core execution over a `DiskColumnTable` (`.jcol` segments).
pub fn execute_segmented(
    query: &SqlQuery,
    dt: &crate::disk_col_segments::DiskColumnTable,
) -> Option<VecResult> {
    use rayon::prelude::*;
    if !query.joins.is_empty()
        || !query.ctes.is_empty()
        || query.set_op.is_some()
        || query.distinct
        || query.as_of_timestamp.is_some()
    {
        return None;
    }

    let proto_tc = dt.proto_table_columns();
    let mut raw_preds = Vec::new();
    if let Some(w) = &query.where_clause {
        collect_preds(w, &mut raw_preds)?;
    }
    let _ = prepare_preds(&proto_tc, &raw_preds)?;

    let mut group_exprs: Vec<&Expression> = Vec::with_capacity(query.group_by.len());
    for ge in &query.group_by {
        let resolved: &Expression = match ge {
            Expression::Literal(AstValue::Int(n))
                if *n >= 1 && (*n as usize) <= query.columns.len() =>
            {
                &query.columns[*n as usize - 1].expr
            }
            Expression::Column(alias_candidate)
                if !proto_tc.cols.contains_key(alias_candidate.as_str()) =>
            {
                query
                    .columns
                    .iter()
                    .find(|sc| sc.alias.as_deref() == Some(alias_candidate.as_str()))
                    .map(|sc| &sc.expr)?
            }
            other => other,
        };
        let _ = build_vcol(&proto_tc, resolved)?;
        group_exprs.push(resolved);
    }

    let query_has_agg = query
        .columns
        .iter()
        .any(|c| parse_agg_expr(&proto_tc, &c.expr).is_some())
        || query.having.is_some();

    if !query_has_agg && group_exprs.is_empty() {
        return execute_scan_segmented(query, dt, &proto_tc, &raw_preds);
    }

    let mut outs: Vec<OutCol> = Vec::with_capacity(query.columns.len());
    let mut out_exprs: Vec<Option<&Expression>> = Vec::with_capacity(query.columns.len());
    let mut names: Vec<String> = Vec::with_capacity(query.columns.len());
    // (full_agg_expr, kind, const_add, is_int, is_txt)
    let mut agg_defs: Vec<(&Expression, AggKind, i64, bool, bool)> = Vec::new();

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
        } else if let Some((kind, vc_proto, const_add, default_name)) =
            parse_agg_expr(&proto_tc, &c.expr)
        {
            let is_int = vc_proto.as_ref().is_some_and(VCol::is_int_like);
            let is_txt = vc_proto.as_ref().is_some_and(VCol::is_text_like);
            let ai = agg_defs.len();
            agg_defs.push((&c.expr, kind, const_add, is_int, is_txt));
            outs.push(OutCol::Agg(ai));
            out_exprs.push(Some(&c.expr));
            names.push(c.alias.clone().unwrap_or(default_name));
        } else {
            return None;
        }
    }

    let visible_no = outs.len();
    let having_pred = match &query.having {
        Some(h) => Some(compile_having_seg(
            h,
            &proto_tc,
            &names,
            &group_exprs,
            &mut out_exprs,
            &mut outs,
            &mut agg_defs,
        )?),
        None => None,
    };

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
                } else if let Some((kind, vc_proto, const_add, _)) = parse_agg_expr(&proto_tc, e) {
                    let is_int = vc_proto.as_ref().is_some_and(VCol::is_int_like);
                    let is_txt = vc_proto.as_ref().is_some_and(VCol::is_text_like);
                    let ai = agg_defs.len();
                    agg_defs.push((e, kind, const_add, is_int, is_txt));
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

    let total_no = outs.len();
    let nagg = agg_defs.len();
    let off = query.offset.unwrap_or(0);
    let take = query.limit.map(|l| off.saturating_add(l));

    // ------------------------------------------------------------------------
    // Fast Path 1: Zero-I/O Metadata Aggregates (`Q0: COUNT(*)`, `Q2/Q3/Q29: SUM/AVG`, `Q6: MIN/MAX`)
    // ------------------------------------------------------------------------
    if raw_preds.is_empty() && group_exprs.is_empty() {
        fn agg_base_col(ae: &Expression) -> Option<&str> {
            let Expression::Function { args, .. } = ae else {
                return None;
            };
            let arg = args.first()?;
            if let Some(cn) = column(arg) {
                return Some(cn);
            }
            if let Expression::Binary { left, op, right } = arg
                && matches!(op, Operator::Add | Operator::Sub)
                && literal(right).is_some()
            {
                return column(left);
            }
            None
        }
        let mut meta_accs = Vec::with_capacity(nagg);
        let mut all_meta_ok = true;
        for &(ae, kind, ca, _, _) in &agg_defs {
            let mut a = Acc::default();
            match kind {
                AggKind::CountStar if ca == 0 => a.count = dt.total_rows as u64,
                AggKind::Count | AggKind::Min | AggKind::Max => {
                    let Some(cn) = agg_base_col(ae) else {
                        all_meta_ok = false;
                        break;
                    };
                    let Some((dk, nulls, min_i, max_i, min_f, max_f)) = dt.global_col_summary(cn)
                    else {
                        all_meta_ok = false;
                        break;
                    };
                    if matches!(kind, AggKind::Min | AggKind::Max)
                        && matches!(
                            dk,
                            crate::disk_col_segments::DiskColKind::Text
                                | crate::disk_col_segments::DiskColKind::Other
                        )
                    {
                        all_meta_ok = false;
                        break;
                    }
                    a.count = (dt.total_rows as u64).saturating_sub(nulls);
                    if let (Some(lo), Some(hi)) = (min_i, max_i) {
                        a.min = Some(Num::I(lo));
                        a.max = Some(Num::I(hi));
                    } else if let (Some(lo), Some(hi)) = (min_f, max_f) {
                        a.min = Some(Num::F(lo));
                        a.max = Some(Num::F(hi));
                    } else if a.count > 0 && matches!(kind, AggKind::Min | AggKind::Max) {
                        all_meta_ok = false;
                        break;
                    }
                }
                AggKind::Sum | AggKind::Avg => {
                    let Some(cn) = agg_base_col(ae) else {
                        all_meta_ok = false;
                        break;
                    };
                    let Some((non_null, sum_i128)) = dt.global_int_sum(cn) else {
                        all_meta_ok = false;
                        break;
                    };
                    a.count = non_null;
                    a.isum = sum_i128;
                }
                _ => {
                    all_meta_ok = false;
                    break;
                }
            }
            meta_accs.push(a);
        }
        if all_meta_ok {
            let cell = |oi: usize| match &outs[oi] {
                OutCol::Group(_) => Sv::Null,
                OutCol::Agg(ai) => {
                    let (_, kind, ca, is_int, is_txt) = agg_defs[*ai];
                    render_acc_cell(&meta_accs[*ai], kind, ca, is_int, is_txt)
                }
            };
            if having_pred.as_ref().is_some_and(|hp| !eval_having(hp, &cell)) {
                return Some(VecResult {
                    columns: names,
                    rows: Vec::new(),
                });
            }
            let row: Vec<serde_json::Value> =
                (0..visible_no).map(|oi| cell(oi).json()).collect();
            let all_rows = vec![row];
            let end = take.unwrap_or(usize::MAX).min(all_rows.len());
            return Some(VecResult {
                columns: names,
                rows: all_rows[off.min(end)..end].to_vec(),
            });
        }
    }

    let mut where_col_set: HashSet<String> = HashSet::new();
    if let Some(w) = &query.where_clause {
        collect_expr_cols(w, &dt.col_index, &mut where_col_set);
    }
    let mut rest_col_set: HashSet<String> = HashSet::new();
    for &ge in &group_exprs {
        collect_expr_cols(ge, &dt.col_index, &mut rest_col_set);
    }
    for &(ae, _, _, _, _) in &agg_defs {
        collect_expr_cols(ae, &dt.col_index, &mut rest_col_set);
    }
    rest_col_set.retain(|c| !where_col_set.contains(c));
    let where_cols: Vec<&str> = where_col_set.iter().map(String::as_str).collect();
    let rest_cols: Vec<&str> = rest_col_set.iter().map(String::as_str).collect();

    // ------------------------------------------------------------------------
    // Fast Path 2: `SELECT COUNT(*) FROM t WHERE ...` (`Q1`, `Q20`)
    // ------------------------------------------------------------------------
    if group_exprs.is_empty() && agg_defs.iter().all(|(_, k, _, _, _)| *k == AggKind::CountStar) {
        // Zero-`Arc<str>` RawTextSlice `COUNT(*)` for single-column text `LIKE` / `=` / `<>` (`Q20`)
        let single_raw_text_pred = if raw_preds.len() == 1 {
            match &raw_preds[0] {
                Pred::Like {
                    col,
                    pattern,
                    negated,
                    ci,
                } if matches!(proto_tc.cols.get(*col), Some(Col::Text(..))) => {
                    Some((*col, None, Some((*pattern, *negated, *ci))))
                }
                Pred::Cmp {
                    col,
                    op: op @ (Operator::Eq | Operator::Ne),
                    lit: Lit::S(s),
                } if matches!(proto_tc.cols.get(*col), Some(Col::Text(..))) => {
                    Some((*col, Some((*op, s.as_bytes())), None))
                }
                _ => None,
            }
        } else {
            None
        };

        let total: Option<u64> = (0..dt.segments.len())
            .into_par_iter()
            .map(|si| -> Option<u64> {
                match seg_verdict(dt, &proto_tc, si, &raw_preds) {
                    SegVerdict::Skip => Some(0),
                    SegVerdict::AllMatch => Some(dt.segments[si].len as u64),
                    SegVerdict::Maybe => {
                        if let Some((tcol, eq_spec, like_spec)) = single_raw_text_pred
                            && let Ok(Some(ts)) = dt.read_raw_text_slice(si, tcol)
                        {
                            if ts.dict_len == 0 {
                                return Some(0);
                            }
                            let ok: Vec<bool> = if let Some((op, lit_b)) = eq_spec {
                                let is_eq = op == Operator::Eq;
                                (0..ts.dict_len)
                                    .map(|di| (ts.dict_bytes(di) == lit_b) == is_eq)
                                    .collect()
                            } else if let Some((pat, neg, ci)) = like_spec {
                                if let Some(needle) = fast_like_literal_needle(pat, ci) {
                                    (0..ts.dict_len)
                                        .map(|di| {
                                            fast_bytes_contains(ts.dict_bytes(di), needle, ci)
                                                != neg
                                        })
                                        .collect()
                                } else {
                                    (0..ts.dict_len)
                                        .map(|di| {
                                            let s =
                                                std::str::from_utf8(ts.dict_bytes(di)).unwrap_or("");
                                            joule_db_query::like::like(s, pat, ci) != neg
                                        })
                                        .collect()
                                }
                            } else {
                                Vec::new()
                            };
                            if !ok.iter().any(|&b| b) {
                                return Some(0);
                            }
                            let codes_raw = &ts.raw[ts.codes_off..];
                            let mut cnt = 0u64;
                            match ts.code_width {
                                1 => {
                                    for &b in &codes_raw[..ts.seg_len] {
                                        if b != 0xFF && ok[b as usize] {
                                            cnt += 1;
                                        }
                                    }
                                }
                                2 => {
                                    for ch in codes_raw[..ts.seg_len * 2].chunks_exact(2) {
                                        let u = u16::from_le_bytes([ch[0], ch[1]]);
                                        if u != 0xFFFF && ok[u as usize] {
                                            cnt += 1;
                                        }
                                    }
                                }
                                _ => {
                                    for ch in codes_raw[..ts.seg_len * 4].chunks_exact(4) {
                                        let u = u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
                                        if u != NULL_CODE && ok[u as usize] {
                                            cnt += 1;
                                        }
                                    }
                                }
                            }
                            return Some(cnt);
                        }
                        let seg_tc = dt.load_segment_columns(si, &where_cols).ok()?;
                        let preds = prepare_preds(&seg_tc, &raw_preds)?;
                        Some(count_where(&seg_tc, &preds))
                    }
                }
            })
            .try_fold(|| 0u64, |a, b| b.map(|v| a + v))
            .try_reduce(|| 0u64, |a, b| Some(a + b));
        let c = total?;
        let cell = |oi: usize| match &outs[oi] {
            OutCol::Group(_) => Sv::Null,
            OutCol::Agg(_) => Sv::U(c),
        };
        if having_pred.as_ref().is_some_and(|hp| !eval_having(hp, &cell)) {
            return Some(VecResult {
                columns: names,
                rows: Vec::new(),
            });
        }
        let row: Vec<serde_json::Value> = (0..visible_no).map(|oi| cell(oi).json()).collect();
        let all_rows = vec![row];
        let end = take.unwrap_or(usize::MAX).min(all_rows.len());
        return Some(VecResult {
            columns: names,
            rows: all_rows[off.min(end)..end].to_vec(),
        });
    }

    let active_segs: Vec<usize> = (0..dt.segments.len())
        .filter(|&si| seg_verdict(dt, &proto_tc, si, &raw_preds) != SegVerdict::Skip)
        .collect();

    // ------------------------------------------------------------------------
    // Case A: Ungrouped Aggregations (`Q2, Q3, Q4, Q5, Q29`)
    // ------------------------------------------------------------------------
    if group_exprs.is_empty() {
        // Zero-`Arc<str>` Raw `.jcol` Text `COUNT(DISTINCT col)` Fast Path (`Q05`)
        if raw_preds.is_empty()
            && having_pred.is_none()
            && nagg == 1
            && agg_defs[0].1 == AggKind::CountDistinct
            && agg_defs[0].4
            && let Expression::Function { args, .. } = agg_defs[0].0
            && let Some(text_cn) = args.first().and_then(column)
            && matches!(proto_tc.cols.get(text_cn), Some(Col::Text(..)))
        {
            const RAW_D_SHARDS: usize = 256;
            let shards: Vec<std::sync::Mutex<rustc_hash::FxHashSet<Box<[u8]>>>> = (0..RAW_D_SHARDS)
                .map(|_| std::sync::Mutex::new(rustc_hash::FxHashSet::default()))
                .collect();
            let raw_ok = active_segs
                .par_iter()
                .map(|&si| -> Option<()> {
                    let ts = dt.read_raw_text_slice(si, text_cn).ok()??;
                    if ts.dict_len == 0 {
                        return Some(());
                    }
                    let mut buckets: Vec<Vec<usize>> =
                        (0..RAW_D_SHARDS).map(|_| Vec::new()).collect();
                    for ci in 0..ts.dict_len {
                        let b = (hash_bytes64(ts.dict_bytes(ci)) as usize) & (RAW_D_SHARDS - 1);
                        buckets[b].push(ci);
                    }
                    for (b, cis) in buckets.into_iter().enumerate() {
                        if !cis.is_empty()
                            && let Ok(mut g) = shards[b].lock()
                        {
                            for ci in cis {
                                let s_bytes = ts.dict_bytes(ci);
                                if !g.contains(s_bytes) {
                                    g.insert(Box::from(s_bytes));
                                }
                            }
                        }
                    }
                    Some(())
                })
                .collect::<Option<Vec<()>>>();
            if raw_ok.is_some() {
                let cnt: u64 = shards
                    .into_iter()
                    .map(|m| m.into_inner().map_or(0, |s| s.len() as u64))
                    .sum();
                let row: Vec<serde_json::Value> = (0..visible_no)
                    .map(|oi| match &outs[oi] {
                        OutCol::Group(_) => serde_json::Value::Null,
                        OutCol::Agg(_) => serde_json::json!(cnt),
                    })
                    .collect();
                let all_rows = vec![row];
                let end = take.unwrap_or(usize::MAX).min(all_rows.len());
                return Some(VecResult {
                    columns: names,
                    rows: all_rows[off.min(end)..end].to_vec(),
                });
            }
        }

        const D_SHARDS: usize = 64;
        let has_distinct = agg_defs
            .iter()
            .any(|(_, k, _, _, _)| *k == AggKind::CountDistinct);
        let int_d_shards: Vec<Vec<std::sync::Mutex<rustc_hash::FxHashSet<u64>>>> = (0..nagg)
            .map(|ai| {
                if agg_defs[ai].1 == AggKind::CountDistinct && !agg_defs[ai].4 {
                    (0..D_SHARDS)
                        .map(|_| std::sync::Mutex::new(rustc_hash::FxHashSet::default()))
                        .collect()
                } else {
                    Vec::new()
                }
            })
            .collect();
        let str_d_shards: Vec<Vec<std::sync::Mutex<rustc_hash::FxHashSet<Arc<str>>>>> = (0..nagg)
            .map(|ai| {
                if agg_defs[ai].1 == AggKind::CountDistinct && agg_defs[ai].4 {
                    (0..D_SHARDS)
                        .map(|_| std::sync::Mutex::new(rustc_hash::FxHashSet::default()))
                        .collect()
                } else {
                    Vec::new()
                }
            })
            .collect();

        let seg_accs: Vec<Vec<Acc>> = active_segs
            .par_iter()
            .map(|&si| -> Option<Vec<Acc>> {
                let mut seg_tc = dt.load_segment_columns(si, &where_cols).ok()?;
                let sel = eval_segment_where(&seg_tc, &raw_preds)?;
                if let Some(ref v) = sel
                    && v.is_empty()
                {
                    return Some((0..nagg).map(|_| Acc::default()).collect());
                }
                if !rest_cols.is_empty() {
                    dt.load_columns_into_selected(si, &rest_cols, sel.as_deref(), &mut seg_tc)
                        .ok()?;
                }
                let specs: Vec<(AggKind, Option<VCol>, i64)> = agg_defs
                    .iter()
                    .map(|(ae, _, _, _, _)| {
                        let (k, vc, ca, _) = parse_agg_expr(&seg_tc, ae)?;
                        Some((k, vc, ca))
                    })
                    .collect::<Option<_>>()?;

                let mut accs: Vec<Acc> = (0..nagg).map(|_| Acc::default()).collect();
                for (oi, (kind, vc_opt, _)) in specs.iter().enumerate() {
                    // Share identical physical `SUM`/`AVG` scans (`Q29` 90 projections!)
                    if matches!(kind, AggKind::Sum | AggKind::Avg)
                        && let Some(VCol::Phys(col)) = vc_opt
                        && let Some(oj) = (0..oi).find(|&oj| {
                            matches!(specs[oj].0, AggKind::Sum | AggKind::Avg)
                                && matches!(&specs[oj].1, Some(VCol::Phys(c)) if std::ptr::eq(*c, *col))
                        })
                    {
                        accs[oi] = accs[oj].clone();
                        continue;
                    }

                    if *kind == AggKind::CountDistinct && has_distinct {
                        let Some(vc) = vc_opt.as_ref() else { continue };
                        if agg_defs[oi].4 {
                            // Text COUNT(DISTINCT): deduplicate segment dictionary first!
                            if let VCol::Phys(Col::Text(codes, dict)) = vc {
                                let mut seen_code = vec![false; dict.len()];
                                match &sel {
                                    None => {
                                        for &c in codes {
                                            if c != NULL_CODE {
                                                seen_code[c as usize] = true;
                                            }
                                        }
                                    }
                                    Some(rows) => {
                                        for &r in rows {
                                            let c = codes[r as usize];
                                            if c != NULL_CODE {
                                                seen_code[c as usize] = true;
                                            }
                                        }
                                    }
                                }
                                let mut buckets: Vec<Vec<Arc<str>>> =
                                    (0..D_SHARDS).map(|_| Vec::new()).collect();
                                for (ci, &ok) in seen_code.iter().enumerate() {
                                    if ok {
                                        let s = if dict[ci].is_empty() {
                                            empty_arc_str()
                                        } else {
                                            dict[ci].clone()
                                        };
                                        let b = (hash_bytes64(s.as_bytes()) as usize)
                                            & (D_SHARDS - 1);
                                        buckets[b].push(s);
                                    }
                                }
                                for (b, items) in buckets.into_iter().enumerate() {
                                    if !items.is_empty()
                                        && let Ok(mut g) = str_d_shards[oi][b].lock()
                                    {
                                        for s in items {
                                            g.insert(s);
                                        }
                                    }
                                }
                            } else {
                                let mut local: rustc_hash::FxHashSet<Arc<str>> =
                                    rustc_hash::FxHashSet::default();
                                match &sel {
                                    None => {
                                        for ri in 0..seg_tc.len {
                                            if let Some(s) = opt_arc_at(vc, ri) {
                                                local.insert(s);
                                            }
                                        }
                                    }
                                    Some(rows) => {
                                        for &r in rows {
                                            if let Some(s) = opt_arc_at(vc, r as usize) {
                                                local.insert(s);
                                            }
                                        }
                                    }
                                }
                                for s in local {
                                    let b =
                                        (hash_bytes64(s.as_bytes()) as usize) & (D_SHARDS - 1);
                                    if let Ok(mut g) = str_d_shards[oi][b].lock() {
                                        g.insert(s);
                                    }
                                }
                            }
                        } else {
                            let mut local: rustc_hash::FxHashSet<u64> =
                                rustc_hash::FxHashSet::default();
                            match &sel {
                                None => {
                                    for ri in 0..seg_tc.len {
                                        let (code, is_n) = vc.group_code_at(ri);
                                        if !is_n {
                                            local.insert(code);
                                        }
                                    }
                                }
                                Some(rows) => {
                                    for &r in rows {
                                        let (code, is_n) = vc.group_code_at(r as usize);
                                        if !is_n {
                                            local.insert(code);
                                        }
                                    }
                                }
                            }
                            let mut buckets: Vec<Vec<u64>> =
                                (0..D_SHARDS).map(|_| Vec::new()).collect();
                            for code in local {
                                let b = (mix64(code) as usize) & (D_SHARDS - 1);
                                buckets[b].push(code);
                            }
                            for (b, items) in buckets.into_iter().enumerate() {
                                if !items.is_empty()
                                    && let Ok(mut g) = int_d_shards[oi][b].lock()
                                {
                                    g.extend(items);
                                }
                            }
                        }
                        continue;
                    }

                    // Fast ungrouped slice scan
                    match (kind, vc_opt.as_ref()) {
                        (AggKind::CountStar, _) => {
                            accs[oi].count = sel.as_ref().map_or(seg_tc.len, Vec::len) as u64;
                        }
                        (
                            AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max,
                            Some(VCol::Phys(Col::Int(ints, nulls))),
                        ) if sel.is_none() && nulls.is_empty() => {
                            with_ints!(ints, |v| {
                                accs[oi].count = v.len() as u64;
                                accs[oi].isum = v.iter().map(|&x| x as i128).sum();
                                if matches!(kind, AggKind::Min | AggKind::Max)
                                    && let (Some(&lo), Some(&hi)) = (v.iter().min(), v.iter().max())
                                {
                                    accs[oi].min = Some(Num::I(lo as i64));
                                    accs[oi].max = Some(Num::I(hi as i64));
                                }
                            });
                        }
                        _ => match &sel {
                            None => {
                                for ri in 0..seg_tc.len {
                                    update_acc_row(&mut accs[oi], *kind, vc_opt.as_ref(), ri);
                                }
                            }
                            Some(rows) => {
                                for &r in rows {
                                    update_acc_row(
                                        &mut accs[oi],
                                        *kind,
                                        vc_opt.as_ref(),
                                        r as usize,
                                    );
                                }
                            }
                        },
                    }
                }
                Some(accs)
            })
            .collect::<Option<_>>()?;

        let mut merged: Vec<Acc> = (0..nagg).map(|_| Acc::default()).collect();
        for sa in &seg_accs {
            for oi in 0..nagg {
                merged[oi].merge(&sa[oi]);
            }
        }
        for oi in 0..nagg {
            if agg_defs[oi].1 == AggKind::CountDistinct {
                let cnt: u64 = if agg_defs[oi].4 {
                    str_d_shards[oi]
                        .iter()
                        .map(|m| m.lock().map_or(0, |s| s.len() as u64))
                        .sum()
                } else {
                    int_d_shards[oi]
                        .iter()
                        .map(|m| m.lock().map_or(0, |s| s.len() as u64))
                        .sum()
                };
                // Store distinct total into `bits`-free synthetic count via `render_acc_cell`
                merged[oi].count = cnt;
            }
        }

        let cell = |oi: usize| match &outs[oi] {
            OutCol::Group(_) => Sv::Null,
            OutCol::Agg(ai) => {
                let (_, kind, ca, is_int, is_txt) = agg_defs[*ai];
                if kind == AggKind::CountDistinct {
                    Sv::U(merged[*ai].count)
                } else {
                    render_acc_cell(&merged[*ai], kind, ca, is_int, is_txt)
                }
            }
        };
        if having_pred.as_ref().is_some_and(|hp| !eval_having(hp, &cell)) {
            return Some(VecResult {
                columns: names,
                rows: Vec::new(),
            });
        }
        let row: Vec<serde_json::Value> = (0..visible_no).map(|oi| cell(oi).json()).collect();
        let all_rows = vec![row];
        let end = take.unwrap_or(usize::MAX).min(all_rows.len());
        return Some(VecResult {
            columns: names,
            rows: all_rows[off.min(end)..end].to_vec(),
        });
    }

    // ------------------------------------------------------------------------
    // Case B: Grouped Aggregations (`GROUP BY` — `Q7..Q18`, `Q21..Q22`, `Q27..Q42`)
    // ------------------------------------------------------------------------
    if active_segs.is_empty() {
        return Some(VecResult {
            columns: names,
            rows: Vec::new(),
        });
    }

    // Eliminate constant & functionally-determined `col ± k` keys (`Q34`, `Q35`!)
    let mut key_exprs: Vec<&Expression> = Vec::new();
    let mut key_phys_cols: Vec<Option<&str>> = Vec::new();
    let mut group_map: Vec<GroupKeyMap> = Vec::with_capacity(group_exprs.len());

    for &ge in &group_exprs {
        let proto_vc = build_vcol(&proto_tc, ge)?;
        match proto_vc {
            VCol::ConstInt(i) => group_map.push(GroupKeyMap::Const(Sv::I(i))),
            VCol::ConstFloat(f) => group_map.push(GroupKeyMap::Const(Sv::F(f))),
            VCol::ConstText(s) => group_map.push(GroupKeyMap::Const(Sv::S(s))),
            VCol::IntArith {
                op: op @ (ArithOp::Add | ArithOp::Sub),
                rhs,
                ..
            } => {
                let base_cn = match ge {
                    Expression::Binary { left, right, .. } => column(left).or_else(|| column(right)),
                    _ => None,
                };
                if let Some(cn) = base_cn
                    && let Some(base_slot) = key_phys_cols.iter().position(|&c| c == Some(cn))
                {
                    group_map.push(GroupKeyMap::DerivedInt {
                        base_slot,
                        op,
                        rhs,
                    });
                } else {
                    let slot = key_exprs.len();
                    key_exprs.push(ge);
                    key_phys_cols.push(None);
                    group_map.push(GroupKeyMap::Slot(slot));
                }
            }
            _ => {
                let phys_cn = column(ge).filter(|&cn| matches!(proto_tc.cols.get(cn), Some(Col::Int(..))));
                if let Some(existing_slot) = key_exprs.iter().position(|&ke| ke == ge) {
                    group_map.push(GroupKeyMap::Slot(existing_slot));
                } else {
                    let slot = key_exprs.len();
                    key_exprs.push(ge);
                    key_phys_cols.push(phys_cn);
                    group_map.push(GroupKeyMap::Slot(slot));
                }
            }
        }
    }
    // Backfill `DerivedInt` even if `ClientIP - 1` appeared before `ClientIP` (or keep ≥1 key if all const)
    if key_exprs.is_empty() {
        key_exprs.push(group_exprs[0]);
        group_map[0] = GroupKeyMap::Slot(0);
    }

    let key_proto_vcols: Vec<VCol> = key_exprs
        .iter()
        .map(|&ke| build_vcol(&proto_tc, ke))
        .collect::<Option<_>>()?;
    let gkey_shape = match key_proto_vcols.as_slice() {
        [a] if a.is_int_like() => GKeyShape::I1,
        [a] if a.is_text_like() => GKeyShape::S1,
        [a, b] if a.is_int_like() && b.is_int_like() => GKeyShape::I2,
        [a, b] if a.is_int_like() && b.is_text_like() => GKeyShape::IS,
        [a, b, c] if a.is_int_like() && b.is_int_like() && c.is_text_like() => GKeyShape::IIS,
        _ => GKeyShape::Many,
    };

    let count_star_only = nagg == 1 && agg_defs[0].1 == AggKind::CountStar;
    const SHARDS: usize = 64;

    // Helper that scans one segment and emits `(hash, GKey, first_ord, count_or_accs)`
    // Either using segment-local `HKey` pre-aggregation or direct row scatter (`bypass_local`).
    let scan_one_seg = |si: usize,
                        bypass_local: bool,
                        hash_mask: u64,
                        hash_target: u64|
     -> Option<(usize, Vec<(u64, GKey, u64, u64, Vec<Acc>)>)> {
        let mut seg_tc = dt.load_segment_columns(si, &where_cols).ok()?;
        let sel = eval_segment_where(&seg_tc, &raw_preds)?;
        if let Some(ref v) = sel
            && v.is_empty()
        {
            return Some((0, Vec::new()));
        }
        if !rest_cols.is_empty() {
            dt.load_columns_into_selected(si, &rest_cols, sel.as_deref(), &mut seg_tc)
                .ok()?;
        }
        let key_vcols: Vec<VCol> = key_exprs
            .iter()
            .map(|&ke| build_vcol(&seg_tc, ke))
            .collect::<Option<_>>()?;
        let seg_aggs: Vec<(AggKind, Option<VCol>)> = agg_defs
            .iter()
            .map(|(ae, _, _, _, _)| {
                let (k, vc, _, _) = parse_agg_expr(&seg_tc, ae)?;
                Some((k, vc))
            })
            .collect::<Option<_>>()?;

        let seg_base_ord = (si as u64) << 32;

        if bypass_local {
            let cap = sel.as_ref().map_or(seg_tc.len, Vec::len)
                / (if hash_mask > 0 { (hash_mask as usize) + 1 } else { 1 }).max(1);
            let mut out = Vec::with_capacity(cap.min(262_144));
            let mut push_row = |ri: usize| {
                let h = row_canonical_hash(&key_vcols, ri);
                if (h & hash_mask) != hash_target {
                    return;
                }
                let gk = build_gkey(gkey_shape, &key_vcols, ri);
                let ord = seg_base_ord | (ri as u64);
                if count_star_only {
                    out.push((h, gk, ord, 1u64, Vec::new()));
                } else {
                    let mut accs = Vec::with_capacity(nagg);
                    for &(kind, ref vc_opt) in &seg_aggs {
                        let mut a = Acc::default();
                        update_acc_row(&mut a, kind, vc_opt.as_ref(), ri);
                        accs.push(a);
                    }
                    out.push((h, gk, ord, 1u64, accs));
                }
            };
            match &sel {
                None => {
                    for ri in 0..seg_tc.len {
                        push_row(ri);
                    }
                }
                Some(rows) => {
                    for &r in rows {
                        push_row(r as usize);
                    }
                }
            }
            let n_emitted = out.len();
            return Some((n_emitted, out));
        }

        // Segment-local pre-aggregation using zero-alloc `HKey`!
        let mut local_map: FxMap<HKey, u32> = FxMap::default();
        let mut local_first: Vec<u32> = Vec::new();
        let mut local_cnt: Vec<u64> = Vec::new();
        let mut local_accs: Vec<Acc> = Vec::new();

        let mut add_row = |ri: usize| {
            let hk = make_hkey(&key_vcols, ri);
            let gi = match local_map.get(&hk) {
                Some(&g) => g as usize,
                None => {
                    let g = local_first.len();
                    local_map.insert(hk, g as u32);
                    local_first.push(ri as u32);
                    if count_star_only {
                        local_cnt.push(0);
                    } else {
                        local_accs.extend((0..nagg).map(|_| Acc::default()));
                    }
                    g
                }
            };
            if count_star_only {
                local_cnt[gi] += 1;
            } else {
                for (oi, &(kind, ref vc_opt)) in seg_aggs.iter().enumerate() {
                    update_acc_row(&mut local_accs[gi * nagg + oi], kind, vc_opt.as_ref(), ri);
                }
            }
        };

        match &sel {
            None => {
                for ri in 0..seg_tc.len {
                    add_row(ri);
                }
            }
            Some(rows) => {
                for &r in rows {
                    add_row(r as usize);
                }
            }
        }

        let total_local_groups = local_first.len();
        drop(local_map);
        let mut out = Vec::with_capacity(total_local_groups);
        let mut acc_drain = local_accs.into_iter();
        for (gi, &first_ri) in local_first.iter().enumerate() {
            let group_accs: Vec<Acc> = if count_star_only {
                Vec::new()
            } else {
                acc_drain.by_ref().take(nagg).collect()
            };
            let gk = build_gkey(gkey_shape, &key_vcols, first_ri as usize);
            let h = gk.canonical_hash();
            if (h & hash_mask) != hash_target {
                continue;
            }
            let ord = seg_base_ord | (first_ri as u64);
            let c = if count_star_only { local_cnt[gi] } else { 0 };
            out.push((h, gk, ord, c, group_accs));
        }
        Some((total_local_groups, out))
    };

    let render_group_cell = |gk: &GKey, gpos: usize| -> Sv {
        match &group_map[gpos] {
            GroupKeyMap::Slot(s) => gk.slot_sv(*s),
            GroupKeyMap::Const(sv) => sv.clone(),
            GroupKeyMap::DerivedInt {
                base_slot,
                op,
                rhs,
            } => match gk.slot_int(*base_slot) {
                Some(x) => Sv::I(match op {
                    ArithOp::Add => x.wrapping_add(*rhs),
                    ArithOp::Sub => x.wrapping_sub(*rhs),
                    ArithOp::Mul => x.wrapping_mul(*rhs),
                }),
                None => Sv::Null,
            },
        }
    };

    let eval_and_render_group = |gk: &GKey, cnt: u64, accs: &[Acc]| -> Option<Vec<Sv>> {
        let cell = |oi: usize| -> Sv {
            match &outs[oi] {
                OutCol::Group(gpos) => render_group_cell(gk, *gpos),
                OutCol::Agg(ai) => {
                    if count_star_only {
                        Sv::U(cnt)
                    } else {
                        let (_, kind, ca, is_int, is_txt) = agg_defs[*ai];
                        render_acc_cell(&accs[*ai], kind, ca, is_int, is_txt)
                    }
                }
            }
        };
        if let Some(ref hp) = having_pred
            && !eval_having(hp, &cell)
        {
            return None;
        }
        Some((0..total_no).map(cell).collect())
    };

    let cmp_rendered = |a: &(Vec<Sv>, u64), b: &(Vec<Sv>, u64)| -> Ordering {
        for &(i, desc, nulls_first) in &order {
            let (x, y) = (&a.0[i], &b.0[i]);
            let nf = nulls_first.unwrap_or(desc);
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
                (false, false) => {
                    let ord = x.cmp_val(y);
                    if desc { ord.reverse() } else { ord }
                }
            };
            if o != Ordering::Equal {
                return o;
            }
        }
        a.1.cmp(&b.1)
    };

    // ------------------------------------------------------------------------
    // Special Fast Path S1-Raw: 1-Pass Zero-`Arc<str>` `.jcol` Text `GROUP BY` (`Q12, Q33, Q34`)
    // Eliminates 4-pass re-reading and ~200M `Arc<str>` heap allocations on `GROUP BY URL`!
    // ------------------------------------------------------------------------
    let non_empty_pred_only = raw_preds.len() == 1
        && matches!(
            &raw_preds[0],
            Pred::Cmp {
                col,
                op: Operator::Ne,
                lit: Lit::S(s),
            } if column(key_exprs[0]) == Some(*col) && s.is_empty()
        );
    if matches!(gkey_shape, GKeyShape::S1)
        && count_star_only
        && having_pred.is_none()
        && key_exprs.len() == 1
        && (raw_preds.is_empty() || non_empty_pred_only)
        && let Some(text_cn) = column(key_exprs[0])
        && matches!(proto_tc.cols.get(text_cn), Some(Col::Text(..)))
    {
        const TXT_SHARDS: usize = 256;
        let shards: Vec<std::sync::Mutex<rustc_hash::FxHashMap<Box<[u8]>, (u64, u64)>>> = (0
            ..TXT_SHARDS)
            .map(|_| std::sync::Mutex::new(rustc_hash::FxHashMap::default()))
            .collect();
        let null_grp = std::sync::Mutex::new((u64::MAX, 0u64));

        let raw_s1_ok = active_segs
            .par_iter()
            .map(|&si| -> Option<()> {
                let ts = dt.read_raw_text_slice(si, text_cn).ok()??;
                let seg_base_ord = (si as u64) << 32;
                let empty_code: Option<u32> = if non_empty_pred_only {
                    (0..ts.dict_len)
                        .find(|&ci| ts.dict_bytes(ci).is_empty())
                        .map(|ci| ci as u32)
                } else {
                    None
                };
                let mut freq = vec![0u32; ts.dict_len];
                let mut first_ri = vec![u32::MAX; ts.dict_len];
                let mut local_null_cnt = 0u64;
                let mut local_null_ord = u64::MAX;

                let codes_bytes = &ts.raw[ts.codes_off..];
                match ts.code_width {
                    0 => {
                        if !non_empty_pred_only && ts.seg_len > 0 {
                            local_null_cnt = ts.seg_len as u64;
                            local_null_ord = seg_base_ord;
                        }
                    }
                    1 => {
                        for (ri, &b) in codes_bytes[..ts.seg_len].iter().enumerate() {
                            if b == 0xFF {
                                if !non_empty_pred_only {
                                    local_null_cnt += 1;
                                    local_null_ord = local_null_ord.min(seg_base_ord | ri as u64);
                                }
                            } else {
                                let c = b as usize;
                                freq[c] += 1;
                                if first_ri[c] == u32::MAX {
                                    first_ri[c] = ri as u32;
                                }
                            }
                        }
                    }
                    2 => {
                        for (ri, ch) in codes_bytes[..ts.seg_len * 2].chunks_exact(2).enumerate() {
                            let u = u16::from_le_bytes([ch[0], ch[1]]);
                            if u == 0xFFFF {
                                if !non_empty_pred_only {
                                    local_null_cnt += 1;
                                    local_null_ord = local_null_ord.min(seg_base_ord | ri as u64);
                                }
                            } else {
                                let c = u as usize;
                                freq[c] += 1;
                                if first_ri[c] == u32::MAX {
                                    first_ri[c] = ri as u32;
                                }
                            }
                        }
                    }
                    _ => {
                        for (ri, ch) in codes_bytes[..ts.seg_len * 4].chunks_exact(4).enumerate() {
                            let u = u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
                            if u == u32::MAX {
                                if !non_empty_pred_only {
                                    local_null_cnt += 1;
                                    local_null_ord = local_null_ord.min(seg_base_ord | ri as u64);
                                }
                            } else {
                                let c = u as usize;
                                freq[c] += 1;
                                if first_ri[c] == u32::MAX {
                                    first_ri[c] = ri as u32;
                                }
                            }
                        }
                    }
                }

                if local_null_cnt > 0
                    && let Ok(mut ng) = null_grp.lock()
                {
                    ng.0 = ng.0.min(local_null_ord);
                    ng.1 += local_null_cnt;
                }

                let mut buckets: Vec<Vec<(u32, u64, u64)>> =
                    (0..TXT_SHARDS).map(|_| Vec::new()).collect();
                for ci in 0..ts.dict_len {
                    let cnt = freq[ci];
                    if cnt == 0 || empty_code == Some(ci as u32) {
                        continue;
                    }
                    let ord = seg_base_ord | (first_ri[ci] as u64);
                    let b = (hash_bytes64(ts.dict_bytes(ci)) as usize) & (TXT_SHARDS - 1);
                    buckets[b].push((ci as u32, ord, cnt as u64));
                }
                for (b, b_items) in buckets.into_iter().enumerate() {
                    if !b_items.is_empty()
                        && let Ok(mut g) = shards[b].lock()
                    {
                        for (ci, ord, cnt) in b_items {
                            let s_bytes = ts.dict_bytes(ci as usize);
                            if let Some((e_ord, e_cnt)) = g.get_mut(s_bytes) {
                                *e_ord = (*e_ord).min(ord);
                                *e_cnt += cnt;
                            } else {
                                g.insert(Box::from(s_bytes), (ord, cnt));
                            }
                        }
                    }
                }
                Some(())
            })
            .collect::<Option<Vec<()>>>();

        if raw_s1_ok.is_some() {
            let agg_only_order: Option<bool> = if order.len() == 1
                && matches!(outs[order[0].0], OutCol::Agg(0))
            {
                Some(order[0].1)
            } else {
                None
            };

            let shard_cands: Vec<Vec<(Vec<Sv>, u64)>> = shards
                .into_par_iter()
                .map(|mtx| {
                    let Ok(map) = mtx.into_inner() else {
                        return Vec::new();
                    };
                    if let (Some(desc), Some(k)) = (agg_only_order, take)
                        && k > 0
                        && map.len() > k
                    {
                        let mut raw_v: Vec<(Box<[u8]>, u64, u64)> =
                            map.into_iter().map(|(b, (o, c))| (b, o, c)).collect();
                        let cmp_cnt = |a: &(Box<[u8]>, u64, u64), b: &(Box<[u8]>, u64, u64)| {
                            let o = if desc { b.2.cmp(&a.2) } else { a.2.cmp(&b.2) };
                            if o != Ordering::Equal { o } else { a.1.cmp(&b.1) }
                        };
                        raw_v.select_nth_unstable_by(k - 1, cmp_cnt);
                        raw_v.truncate(k);
                        return raw_v
                            .into_iter()
                            .filter_map(|(bytes, ord, cnt)| {
                                let s = std::str::from_utf8(&bytes).ok()?;
                                let arc: Arc<str> = if s.is_empty() {
                                    empty_arc_str()
                                } else {
                                    Arc::from(s)
                                };
                                let gk = GKey::S1(Some(arc));
                                Some((eval_and_render_group(&gk, cnt, &[])?, ord))
                            })
                            .collect();
                    }

                    let mut v: Vec<(Vec<Sv>, u64)> = map
                        .into_iter()
                        .filter_map(|(bytes, (ord, cnt))| {
                            let s = std::str::from_utf8(&bytes).ok()?;
                            let arc: Arc<str> = if s.is_empty() {
                                empty_arc_str()
                            } else {
                                Arc::from(s)
                            };
                            let gk = GKey::S1(Some(arc));
                            Some((eval_and_render_group(&gk, cnt, &[])?, ord))
                        })
                        .collect();
                    if !order.is_empty()
                        && let Some(k) = take
                        && k > 0
                        && k < v.len()
                    {
                        v.select_nth_unstable_by(k - 1, &cmp_rendered);
                        v.truncate(k);
                    }
                    v
                })
                .collect();

            let mut global_cands: Vec<(Vec<Sv>, u64)> = shard_cands.into_iter().flatten().collect();
            if let Ok((null_ord, null_cnt)) = null_grp.into_inner()
                && null_cnt > 0
                && let Some(svs) = eval_and_render_group(&GKey::S1(None), null_cnt, &[])
            {
                global_cands.push((svs, null_ord));
            }
            if !order.is_empty() {
                if let Some(k) = take {
                    if k == 0 {
                        global_cands.clear();
                    } else if k < global_cands.len() {
                        global_cands.select_nth_unstable_by(k - 1, &cmp_rendered);
                        global_cands.truncate(k);
                    }
                }
                global_cands.sort_unstable_by(&cmp_rendered);
            } else {
                global_cands.sort_unstable_by_key(|(_, ord)| *ord);
            }
            let end = take.unwrap_or(usize::MAX).min(global_cands.len());
            let rows: Vec<Vec<serde_json::Value>> = global_cands[off.min(end)..end]
                .iter()
                .map(|(svs, _)| svs[..visible_no].iter().map(Sv::json).collect())
                .collect();
            return Some(VecResult {
                columns: names,
                rows,
            });
        }
    }

    // Ultra-Fast Small-Domain Integer `GROUP BY` Array (`Q7: AdvEngineID` `0..255`):
    // Direct flat-array aggregation per segment with zero hashing!
    if matches!(gkey_shape, GKeyShape::I1)
        && !agg_defs
            .iter()
            .any(|(_, k, _, _, _)| *k == AggKind::CountDistinct)
        && let Some(kcol_name) = column(key_exprs[0])
        && let Some((_, _, Some(lo_i), Some(hi_i), _, _)) = dt.global_col_summary(kcol_name)
        && hi_i >= lo_i
        && (hi_i as i128 - lo_i as i128) <= 4_096
    {
        let span = (hi_i - lo_i + 1) as usize;
        let nslots = span + 1; // last slot = NULL
        let seg_tables: Option<Vec<Vec<(u64, u64, Vec<Acc>)>>> = active_segs
            .par_iter()
            .map(|&si| -> Option<Vec<(u64, u64, Vec<Acc>)>> {
                let mut seg_tc = dt.load_segment_columns(si, &where_cols).ok()?;
                let sel = eval_segment_where(&seg_tc, &raw_preds)?;
                if let Some(ref v) = sel
                    && v.is_empty()
                {
                    return Some(Vec::new());
                }
                if !rest_cols.is_empty() {
                    dt.load_columns_into_selected(si, &rest_cols, sel.as_deref(), &mut seg_tc)
                        .ok()?;
                }
                let kvc = build_vcol(&seg_tc, key_exprs[0])?;
                let seg_aggs: Vec<(AggKind, Option<VCol>)> = agg_defs
                    .iter()
                    .map(|(ae, _, _, _, _)| {
                        let (k, vc, _, _) = parse_agg_expr(&seg_tc, ae)?;
                        Some((k, vc))
                    })
                    .collect::<Option<_>>()?;
                let seg_base_ord = (si as u64) << 32;
                let mut slots: Vec<(u64, u64, Vec<Acc>)> = (0..nslots)
                    .map(|_| {
                        (
                            u64::MAX,
                            0u64,
                            if count_star_only {
                                Vec::new()
                            } else {
                                (0..nagg).map(|_| Acc::default()).collect()
                            },
                        )
                    })
                    .collect();
                let mut visit = |ri: usize| {
                    let slot = match kvc.int_at(ri) {
                        Some(v) => (v - lo_i) as usize,
                        None => span,
                    };
                    let entry = &mut slots[slot];
                    let ord = seg_base_ord | (ri as u64);
                    if ord < entry.0 {
                        entry.0 = ord;
                    }
                    if count_star_only {
                        entry.1 += 1;
                    } else {
                        for (j, &(kind, ref vc_opt)) in seg_aggs.iter().enumerate() {
                            update_acc_row(&mut entry.2[j], kind, vc_opt.as_ref(), ri);
                        }
                        entry.1 += 1;
                    }
                };
                match &sel {
                    None => {
                        for ri in 0..seg_tc.len {
                            visit(ri);
                        }
                    }
                    Some(rows) => {
                        for &r in rows {
                            visit(r as usize);
                        }
                    }
                }
                Some(slots)
            })
            .collect();

        if let Some(seg_tables) = seg_tables {
            let mut merged: Vec<(u64, u64, Vec<Acc>)> = (0..nslots)
                .map(|_| {
                    (
                        u64::MAX,
                        0u64,
                        if count_star_only {
                            Vec::new()
                        } else {
                            (0..nagg).map(|_| Acc::default()).collect()
                        },
                    )
                })
                .collect();
            for st in seg_tables {
                if st.is_empty() {
                    continue;
                }
                for (slot, (ord, cnt, accs)) in st.into_iter().enumerate() {
                    if ord == u64::MAX {
                        continue;
                    }
                    let dst = &mut merged[slot];
                    dst.0 = dst.0.min(ord);
                    dst.1 += cnt;
                    if !count_star_only {
                        for (a, b) in dst.2.iter_mut().zip(&accs) {
                            a.merge(b);
                        }
                    }
                }
            }
            let mut cands: Vec<(Vec<Sv>, u64)> = Vec::new();
            for (slot, (ord, cnt, accs)) in merged.into_iter().enumerate() {
                if ord == u64::MAX {
                    continue;
                }
                let gk = if slot == span {
                    GKey::I1(0, true)
                } else {
                    GKey::I1(lo_i + slot as i64, false)
                };
                if let Some(svs) = eval_and_render_group(&gk, cnt, &accs) {
                    cands.push((svs, ord));
                }
            }
            let cmp_cand = |a: &(Vec<Sv>, u64), b: &(Vec<Sv>, u64)| -> Ordering {
                for &(i, desc, nulls_first) in &order {
                    let (x, y) = (&a.0[i], &b.0[i]);
                    let nf = nulls_first.unwrap_or(desc);
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
                        (false, false) => {
                            let ord = x.cmp_val(y);
                            if desc { ord.reverse() } else { ord }
                        }
                    };
                    if o != Ordering::Equal {
                        return o;
                    }
                }
                a.1.cmp(&b.1)
            };
            cands.sort_unstable_by(cmp_cand);
            let end = take.unwrap_or(usize::MAX).min(cands.len());
            let rows = cands[off.min(end)..end]
                .iter()
                .map(|(svs, _)| svs[..visible_no].iter().map(Sv::json).collect())
                .collect();
            return Some(VecResult {
                columns: names,
                rows,
            });
        }
    }

    // Probe first non-empty segment to gauge cardinality
    let mut probe_seg_idx = 0usize;
    let mut seg0_groups = 0usize;
    let mut seg0_items = Vec::new();
    while probe_seg_idx < active_segs.len() {
        let (ng, items) = scan_one_seg(active_segs[probe_seg_idx], false, 0, 0)?;
        if ng > 0 {
            seg0_groups = ng;
            seg0_items = items;
            break;
        }
        probe_seg_idx += 1;
    }
    if seg0_groups == 0 {
        return Some(VecResult {
            columns: names,
            rows: Vec::new(),
        });
    }

    let rem_segs = &active_segs[probe_seg_idx + 1..];
    let active_rows: usize = active_segs[probe_seg_idx..]
        .iter()
        .map(|&si| dt.segments[si].len as usize)
        .sum();
    let bypass_local = seg0_groups > 65_536 && rem_segs.len() >= 2;

    // Special Fast Path: Unordered `GROUP BY ... LIMIT k` without `HAVING` (`Q17`)!
    if order.is_empty()
        && having_pred.is_none()
        && let Some(k) = take
        && seg0_groups > 4_096
        && !rem_segs.is_empty()
    {
        const Q17_MASK: u64 = 63;
        let mut shard0: FxMap<GKey, (u64, u64, Vec<Acc>)> = FxMap::default();
        for (h, gk, ord, cnt, accs) in &seg0_items {
            if (*h & Q17_MASK) == 0 {
                shard0.insert(gk.clone(), (*ord, *cnt, accs.clone()));
            }
        }
        let shard0_mtx = std::sync::Mutex::new(shard0);
        let ok_q17 = rem_segs
            .par_iter()
            .map(|&si| -> Option<()> {
                let (_, items) = scan_one_seg(si, bypass_local, Q17_MASK, 0)?;
                if !items.is_empty()
                    && let Ok(mut g) = shard0_mtx.lock()
                {
                    for (_, gk, ord, cnt, accs) in items {
                        match g.get_mut(&gk) {
                            Some((e_ord, e_cnt, e_accs)) => {
                                *e_ord = (*e_ord).min(ord);
                                if count_star_only {
                                    *e_cnt += cnt;
                                } else {
                                    for (a, b) in e_accs.iter_mut().zip(&accs) {
                                        a.merge(b);
                                    }
                                }
                            }
                            None => {
                                g.insert(gk, (ord, cnt, accs));
                            }
                        }
                    }
                }
                Some(())
            })
            .collect::<Option<Vec<()>>>();

        if ok_q17.is_some()
            && let Ok(map) = shard0_mtx.into_inner()
            && map.len() >= k
        {
            let mut cands: Vec<(Vec<Sv>, u64)> = map
                .into_iter()
                .filter_map(|(gk, (ord, cnt, accs))| {
                    Some((eval_and_render_group(&gk, cnt, &accs)?, ord))
                })
                .collect();
            cands.sort_unstable_by_key(|(_, ord)| *ord);
            let end = k.min(cands.len());
            let rows = cands[off.min(end)..end]
                .iter()
                .map(|(svs, _)| svs[..visible_no].iter().map(Sv::json).collect())
                .collect();
            return Some(VecResult {
                columns: names,
                rows,
            });
        }
    }

    // Single-Pass 256-Partition `Copy` Tuple Scatter-Sort-Reduce for High-Cardinality `I1`/`I2`
    // Integer `GROUP BY` (`Q15, Q31, Q32, Q35`): scans `.jcol` segments ONCE instead of 4 passes,
    // eliminates 100M heap `Vec<Acc>` allocations, and reduces each partition via in-place sort +
    // zero-allocation Top-K pruning!
    let all_compact_int_aggs = nagg <= 3
        && agg_defs.iter().all(|(_, k, _, is_int, is_txt)| {
            !*is_txt
                && match k {
                    AggKind::CountStar => true,
                    AggKind::Count | AggKind::Sum | AggKind::Avg | AggKind::Min | AggKind::Max => {
                        *is_int
                    }
                    AggKind::CountDistinct => false,
                }
        });
    let use_compact_i2 = matches!(gkey_shape, GKeyShape::I1 | GKeyShape::I2)
        && all_compact_int_aggs
        && ((active_rows > 5_000_000 && seg0_groups > 512)
            || (cfg!(test) && dt.segments.len() >= 2));

    if use_compact_i2 {
        #[derive(Clone, Copy)]
        struct CompactI2Row {
            k0: i64,
            k1: i64,
            ord: u64,
            v: [i64; 3],
            cnt_star: u32,
            val_null_mask: u8,
            key_null_mask: u8,
        }

        const P_SHARDS: usize = 256;
        let parts: Vec<std::sync::Mutex<Vec<CompactI2Row>>> = (0..P_SHARDS)
            .map(|_| std::sync::Mutex::new(Vec::new()))
            .collect();
        drop(seg0_items);

        let is_i2 = matches!(gkey_shape, GKeyShape::I2);
        let scatter_ok = active_segs[probe_seg_idx..]
            .par_iter()
            .map(|&si| -> Option<()> {
                let mut seg_tc = dt.load_segment_columns(si, &where_cols).ok()?;
                let sel = eval_segment_where(&seg_tc, &raw_preds)?;
                if let Some(ref v) = sel
                    && v.is_empty()
                {
                    return Some(());
                }
                if !rest_cols.is_empty() {
                    dt.load_columns_into_selected(si, &rest_cols, sel.as_deref(), &mut seg_tc)
                        .ok()?;
                }
                let key_vcols: Vec<VCol> = key_exprs
                    .iter()
                    .map(|&ke| build_vcol(&seg_tc, ke))
                    .collect::<Option<_>>()?;
                let seg_aggs: Vec<(AggKind, Option<VCol>)> = agg_defs
                    .iter()
                    .map(|(ae, _, _, _, _)| {
                        let (k, vc, _, _) = parse_agg_expr(&seg_tc, ae)?;
                        Some((k, vc))
                    })
                    .collect::<Option<_>>()?;

                let seg_base_ord = (si as u64) << 32;
                let mut local_bins: Vec<Vec<CompactI2Row>> =
                    (0..P_SHARDS).map(|_| Vec::with_capacity(1024)).collect();

                let mut emit = |ri: usize| {
                    let (k0, na) = match key_vcols[0].int_at(ri) {
                        Some(x) => (x, 0u8),
                        None => (0, 1u8),
                    };
                    let (k1, nb) = if is_i2 {
                        match key_vcols[1].int_at(ri) {
                            Some(x) => (x, 0u8),
                            None => (0, 2u8),
                        }
                    } else {
                        (0, 0u8)
                    };
                    let key_null_mask = na | nb;
                    let mut v = [0i64; 3];
                    let mut val_null_mask = 0u8;
                    for (j, (kind, vc_opt)) in seg_aggs.iter().enumerate() {
                        if *kind == AggKind::CountStar {
                            v[j] = 1;
                        } else {
                            match vc_opt.as_ref().and_then(|vc| vc.int_at(ri)) {
                                Some(x) => v[j] = x,
                                None => val_null_mask |= 1u8 << j,
                            }
                        }
                    }
                    let h = mix64(
                        (k0 as u64)
                            .wrapping_mul(0x9e3779b97f4a7c15)
                            .wrapping_add((k1 as u64).wrapping_mul(0x517cc1b727220a95))
                            ^ (key_null_mask as u64),
                    );
                    let b = (h as usize) & (P_SHARDS - 1);
                    local_bins[b].push(CompactI2Row {
                        k0,
                        k1,
                        ord: seg_base_ord | (ri as u64),
                        v,
                        cnt_star: 1,
                        val_null_mask,
                        key_null_mask,
                    });
                };

                match &sel {
                    None => {
                        for ri in 0..seg_tc.len {
                            emit(ri);
                        }
                    }
                    Some(rows) => {
                        for &r in rows {
                            emit(r as usize);
                        }
                    }
                }

                for (b, mut bin) in local_bins.into_iter().enumerate() {
                    if bin.is_empty() {
                        continue;
                    }
                    // Pre-reduce repeated keys inside the segment worker when `COUNT(*)`-only (`Q15`)
                    // or medium cardinality (`Q30`) so cross-thread `parts[b]` locks and final sort
                    // move significantly fewer tuples!
                    if count_star_only && bin.len() > 64 {
                        bin.sort_unstable_by(|a, b| {
                            (a.key_null_mask, a.k0, a.k1, a.ord)
                                .cmp(&(b.key_null_mask, b.k0, b.k1, b.ord))
                        });
                        let mut w = 0usize;
                        let mut i = 0usize;
                        while i < bin.len() {
                            let mut cur = bin[i];
                            i += 1;
                            while i < bin.len()
                                && bin[i].key_null_mask == cur.key_null_mask
                                && bin[i].k0 == cur.k0
                                && bin[i].k1 == cur.k1
                            {
                                cur.cnt_star += bin[i].cnt_star;
                                i += 1;
                            }
                            bin[w] = cur;
                            w += 1;
                        }
                        bin.truncate(w);
                    }
                    if let Ok(mut g) = parts[b].lock() {
                        g.append(&mut bin);
                    }
                }
                Some(())
            })
            .collect::<Option<Vec<()>>>();
        scatter_ok?;

        let cmp_sv_ord = |a_sv: &[Sv], a_ord: u64, b_sv: &[Sv], b_ord: u64| -> Ordering {
            for &(i, desc, nulls_first) in &order {
                let (x, y) = (&a_sv[i], &b_sv[i]);
                let nf = nulls_first.unwrap_or(desc);
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
                    (false, false) => {
                        let ord = x.cmp_val(y);
                        if desc { ord.reverse() } else { ord }
                    }
                };
                if o != Ordering::Equal {
                    return o;
                }
            }
            a_ord.cmp(&b_ord)
        };

        let part_cands: Vec<Vec<(Vec<Sv>, u64)>> = parts
            .into_par_iter()
            .map(|mtx| {
                let Ok(mut rows) = mtx.into_inner() else {
                    return Vec::new();
                };
                if rows.is_empty() {
                    return Vec::new();
                }
                rows.sort_unstable_by(|a, b| {
                    (a.key_null_mask, a.k0, a.k1, a.ord)
                        .cmp(&(b.key_null_mask, b.k0, b.k1, b.ord))
                });

                let ordered_topk = if !order.is_empty() { take } else { None };
                let mut v: Vec<(Vec<Sv>, u64)> = Vec::new();
                let mut scratch_sv: Vec<Sv> = vec![Sv::Null; total_no];

                let n = rows.len();
                let mut idx = 0usize;
                while idx < n {
                    let r0 = rows[idx];
                    let run_ord = r0.ord;
                    let mut accs = [Acc::default(), Acc::default(), Acc::default()];
                    let mut end = idx;
                    while end < n
                        && rows[end].key_null_mask == r0.key_null_mask
                        && rows[end].k0 == r0.k0
                        && rows[end].k1 == r0.k1
                    {
                        let r = rows[end];
                        for j in 0..nagg {
                            let kind = agg_defs[j].1;
                            if kind == AggKind::CountStar {
                                accs[j].count += r.cnt_star as u64;
                            } else if (r.val_null_mask & (1u8 << j)) == 0 {
                                let x = r.v[j];
                                accs[j].count += 1;
                                accs[j].isum += x as i128;
                                if matches!(kind, AggKind::Min) {
                                    accs[j].min = Some(match accs[j].min {
                                        Some(Num::I(m)) if m <= x => Num::I(m),
                                        _ => Num::I(x),
                                    });
                                }
                                if matches!(kind, AggKind::Max) {
                                    accs[j].max = Some(match accs[j].max {
                                        Some(Num::I(m)) if m >= x => Num::I(m),
                                        _ => Num::I(x),
                                    });
                                }
                            }
                        }
                        end += 1;
                    }
                    idx = end;

                    let gk = if is_i2 {
                        GKey::I2(r0.k0, r0.k1, r0.key_null_mask)
                    } else {
                        GKey::I1(r0.k0, r0.key_null_mask != 0)
                    };
                    for oi in 0..total_no {
                        scratch_sv[oi] = match &outs[oi] {
                            OutCol::Group(gpos) => render_group_cell(&gk, *gpos),
                            OutCol::Agg(ai) => {
                                let (_, kind, ca, is_int, is_txt) = agg_defs[*ai];
                                render_acc_cell(&accs[*ai], kind, ca, is_int, is_txt)
                            }
                        };
                    }
                    if let Some(ref hp) = having_pred
                        && !eval_having(hp, &|oi| scratch_sv[oi].clone())
                    {
                        continue;
                    }

                    if let Some(k) = ordered_topk {
                        if k == 0 {
                            continue;
                        }
                        if v.len() < k {
                            v.push((scratch_sv.clone(), run_ord));
                            if v.len() == k {
                                v.sort_unstable_by(&cmp_rendered);
                            }
                        } else if cmp_sv_ord(&scratch_sv, run_ord, &v[k - 1].0, v[k - 1].1)
                            == Ordering::Less
                        {
                            v[k - 1] = (scratch_sv.clone(), run_ord);
                            // Bubble new candidate into sorted position in `v[..k]` (`k` typically 10..100)
                            if k <= 64 {
                                let mut p = k - 1;
                                while p > 0
                                    && cmp_rendered(&v[p], &v[p - 1]) == Ordering::Less
                                {
                                    v.swap(p, p - 1);
                                    p -= 1;
                                }
                            } else {
                                v.sort_unstable_by(&cmp_rendered);
                            }
                        }
                    } else {
                        v.push((scratch_sv.clone(), run_ord));
                    }
                }
                drop(rows);
                v
            })
            .collect();

        let mut global_cands: Vec<(Vec<Sv>, u64)> = part_cands.into_iter().flatten().collect();
        if !order.is_empty() {
            if let Some(k) = take {
                if k == 0 {
                    global_cands.clear();
                } else if k < global_cands.len() {
                    global_cands.select_nth_unstable_by(k - 1, &cmp_rendered);
                    global_cands.truncate(k);
                }
            }
            global_cands.sort_unstable_by(&cmp_rendered);
        } else {
            global_cands.sort_unstable_by_key(|(_, ord)| *ord);
        }
        let end = take.unwrap_or(usize::MAX).min(global_cands.len());
        let rows: Vec<Vec<serde_json::Value>> = global_cands[off.min(end)..end]
            .iter()
            .map(|(svs, _)| svs[..visible_no].iter().map(Sv::json).collect())
            .collect();
        return Some(VecResult {
            columns: names,
            rows,
        });
    }

    // Multi-pass Radix-64 sharding when `active_rows > 20M` and high cardinality (`Q16, Q18`)
    let n_passes: usize = if active_rows > 20_000_000 && seg0_groups > 32_768 {
        4
    } else {
        1
    };
    let pass_mask = (n_passes as u64) - 1;
    let pass_bits = n_passes.trailing_zeros();

    let mut global_cands: Vec<(Vec<Sv>, u64)> = Vec::new();

    for pass in 0..n_passes {
        let shards: Vec<std::sync::Mutex<FxMap<GKey, (u64, u64, Vec<Acc>)>>> = (0..SHARDS)
            .map(|_| std::sync::Mutex::new(FxMap::default()))
            .collect();

        // Seed from already-decoded `seg0_items`!
        {
            let mut buckets: Vec<Vec<(GKey, u64, u64, Vec<Acc>)>> =
                (0..SHARDS).map(|_| Vec::new()).collect();
            for (h, gk, ord, cnt, accs) in &seg0_items {
                if (*h & pass_mask) == (pass as u64) {
                    let b = ((*h >> pass_bits) as usize) & (SHARDS - 1);
                    buckets[b].push((gk.clone(), *ord, *cnt, accs.clone()));
                }
            }
            for (b, items) in buckets.into_iter().enumerate() {
                if !items.is_empty()
                    && let Ok(mut g) = shards[b].lock()
                {
                    for (gk, ord, cnt, accs) in items {
                        g.insert(gk, (ord, cnt, accs));
                    }
                }
            }
        }

        if n_passes == 1 {
            seg0_items = Vec::new();
        }

        let scan_ok = rem_segs
            .par_iter()
            .map(|&si| -> Option<()> {
                let (_, items) = scan_one_seg(si, bypass_local, pass_mask, pass as u64)?;
                let mut buckets: Vec<Vec<(GKey, u64, u64, Vec<Acc>)>> =
                    (0..SHARDS).map(|_| Vec::new()).collect();
                for (h, gk, ord, cnt, accs) in items {
                    let b = ((h >> pass_bits) as usize) & (SHARDS - 1);
                    buckets[b].push((gk, ord, cnt, accs));
                }
                for (b, b_items) in buckets.into_iter().enumerate() {
                    if b_items.is_empty() {
                        continue;
                    }
                    if let Ok(mut g) = shards[b].lock() {
                        for (gk, ord, cnt, accs) in b_items {
                            match g.get_mut(&gk) {
                                Some((e_ord, e_cnt, e_accs)) => {
                                    *e_ord = (*e_ord).min(ord);
                                    if count_star_only {
                                        *e_cnt += cnt;
                                    } else {
                                        for (a, x) in e_accs.iter_mut().zip(&accs) {
                                            a.merge(x);
                                        }
                                    }
                                }
                                None => {
                                    g.insert(gk, (ord, cnt, accs));
                                }
                            }
                        }
                    }
                }
                Some(())
            })
            .collect::<Option<Vec<()>>>();
        scan_ok?;

        // Parallel per-shard HAVING + Top-K reduction + immediate hash-table drop!
        let pass_cands: Vec<Vec<(Vec<Sv>, u64)>> = shards
            .into_par_iter()
            .map(|mtx| {
                let Ok(map) = mtx.into_inner() else {
                    return Vec::new();
                };
                let mut v: Vec<(Vec<Sv>, u64)> = map
                    .into_iter()
                    .filter_map(|(gk, (ord, cnt, accs))| {
                        Some((eval_and_render_group(&gk, cnt, &accs)?, ord))
                    })
                    .collect();
                if !order.is_empty()
                    && let Some(k) = take
                    && k > 0
                    && k < v.len()
                {
                    v.select_nth_unstable_by(k - 1, &cmp_rendered);
                    v.truncate(k);
                }
                v
            })
            .collect();

        for ch in pass_cands {
            global_cands.extend(ch);
        }
        if !order.is_empty()
            && let Some(k) = take
            && k > 0
            && global_cands.len() > k.saturating_mul(64).max(4096)
        {
            global_cands.select_nth_unstable_by(k - 1, &cmp_rendered);
            global_cands.truncate(k);
        }
    }

    if !order.is_empty() {
        if let Some(k) = take {
            if k == 0 {
                global_cands.clear();
            } else if k < global_cands.len() {
                global_cands.select_nth_unstable_by(k - 1, &cmp_rendered);
                global_cands.truncate(k);
            }
        }
        global_cands.sort_unstable_by(&cmp_rendered);
    } else {
        global_cands.sort_unstable_by_key(|(_, ord)| *ord);
    }

    let end = take.unwrap_or(usize::MAX).min(global_cands.len());
    let rows: Vec<Vec<serde_json::Value>> = global_cands[off.min(end)..end]
        .iter()
        .map(|(svs, _)| svs[..visible_no].iter().map(Sv::json).collect())
        .collect();

    Some(VecResult {
        columns: names,
        rows,
    })
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

    #[test]
    fn test_disk_segmented_matches_in_memory_and_recovers() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let in_mem = sample_hits_columns();

        // Build two separate batches with different string ordering so segment dictionaries
        // assign DIFFERENT local u32 codes to the same strings across segments!
        let make_batch = |first_id: u64, rev_order: bool, event_day: i16| -> ColumnBatch {
            let base_urls = [
                "https://www.google.com/search?q=rust",
                "https://example.org/docs/page",
                "",
                "http://blog.rust-lang.org/2026/01",
            ];
            let base_titles = ["Google Search", "Example Docs", "", "Rust Blog Google"];
            let base_phrases = ["rust db", "", "", "vectorized sql"];
            let mut u_off = vec![0u32];
            let mut u_bytes = Vec::new();
            let mut t_off = vec![0u32];
            let mut t_bytes = Vec::new();
            let mut p_off = vec![0u32];
            let mut p_bytes = Vec::new();
            let n = 4usize;
            for k in 0..n {
                let idx = if rev_order { 3 - k } else { k };
                u_bytes.extend_from_slice(base_urls[idx].as_bytes());
                u_off.push(u_bytes.len() as u32);
                t_bytes.extend_from_slice(base_titles[idx].as_bytes());
                t_off.push(t_bytes.len() as u32);
                p_bytes.extend_from_slice(base_phrases[idx].as_bytes());
                p_off.push(p_bytes.len() as u32);
            }
            ColumnBatch {
                table_field: "__table__".into(),
                table: "hits".into(),
                first_id,
                len: n,
                cols: vec![
                    BatchColumn {
                        name: "CounterID".into(),
                        data: ColData::Int32(if rev_order {
                            vec![62, 10, 62, 62]
                        } else {
                            vec![62, 62, 10, 62]
                        }),
                        valid: None,
                    },
                    BatchColumn {
                        name: "EventDate".into(),
                        data: ColData::Int16(vec![event_day; n]),
                        valid: None,
                    },
                    BatchColumn {
                        name: "EventTime".into(),
                        data: ColData::Int(
                            (0..n)
                                .map(|i| {
                                    let idx = if rev_order { 3 - i + 4 } else { i };
                                    1_373_891_696 + (idx as i64) * 30
                                })
                                .collect(),
                        ),
                        valid: None,
                    },
                    BatchColumn {
                        name: "ResolutionWidth".into(),
                        data: ColData::Int16(if rev_order {
                            vec![1920, 1280, 1440, 1920]
                        } else {
                            vec![1920, 1440, 1280, 1920]
                        }),
                        valid: None,
                    },
                    BatchColumn {
                        name: "TraficSourceID".into(),
                        data: ColData::Int8(if rev_order {
                            vec![-1, 0, 2, 6]
                        } else {
                            vec![-1, 6, 0, -1]
                        }),
                        valid: None,
                    },
                    BatchColumn {
                        name: "SearchEngineID".into(),
                        data: ColData::Int16(if rev_order {
                            vec![0, 0, 0, 2]
                        } else {
                            vec![0, 2, 0, 0]
                        }),
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
            }
        };

        let b0 = make_batch(1, false, 15901);
        let b1 = make_batch(5, true, 15901);
        let col_names: Vec<String> = b0.cols.iter().map(|c| c.name.clone()).collect();
        let mut dt =
            crate::disk_col_segments::DiskColumnTable::create_new(tmp.path(), "hits", &col_names)
                .expect("create DiskColumnTable");
        dt.append_column_batch_from(&b0, 0, 1, true).expect("seg 0");
        dt.append_column_batch_from(&b1, 0, 5, true).expect("seg 1");
        assert_eq!(dt.segments.len(), 2);
        assert_eq!(dt.total_rows, 8);

        // Reopen from disk to verify self-describing `.jcol` footer + manifest recovery
        let reopened = crate::disk_col_segments::DiskColumnTable::open_all(tmp.path());
        let dt = reopened.get("hits").expect("reopened hits");
        assert_eq!(dt.total_rows, 8);

        let queries = [
            "SELECT COUNT(*) FROM hits",
            "SELECT MIN(EventDate), MAX(EventDate), MIN(ResolutionWidth), MAX(ResolutionWidth) FROM hits",
            "SELECT COUNT(*) FROM hits WHERE CounterID = 10",
            "SELECT COUNT(*) FROM hits WHERE CounterID = 99999",
            "SELECT COUNT(DISTINCT CounterID), COUNT(DISTINCT SearchPhrase), COUNT(DISTINCT URL) FROM hits",
            "SELECT COUNT(DISTINCT SearchPhrase) FROM hits",
            "SELECT SearchPhrase, MIN(URL), MAX(Title), COUNT(*) AS c, COUNT(DISTINCT CounterID) FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC, SearchPhrase ASC",
            "SELECT REGEXP_REPLACE(URL, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k, AVG(STRLEN(URL)) AS l, COUNT(*) AS c, MIN(URL) FROM hits WHERE URL <> '' GROUP BY k HAVING COUNT(*) > 1 ORDER BY l DESC, k ASC LIMIT 10",
            "SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), SUM(ResolutionWidth + 10) FROM hits",
            "SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC, URL ASC LIMIT 10",
            "SELECT CounterID, CounterID - 1, CounterID - 2, COUNT(*) AS c FROM hits GROUP BY CounterID, CounterID - 1, CounterID - 2 ORDER BY c DESC",
            "SELECT CounterID, TraficSourceID, COUNT(*) AS c, SUM(SearchEngineID), AVG(ResolutionWidth) FROM hits GROUP BY CounterID, TraficSourceID ORDER BY c DESC, CounterID ASC, TraficSourceID ASC LIMIT 5",
            "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC, SearchPhrase ASC LIMIT 5",
            "SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 5",
            "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 3",
            "SELECT TraficSourceID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN URL ELSE '' END AS Src, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND TraficSourceID IN (-1, 6) GROUP BY TraficSourceID, Src ORDER BY PageViews DESC, TraficSourceID ASC, Src ASC",
        ];

        for sql in queries {
            let stmt = SqlParser::new().parse(sql).expect("valid SQL");
            let SqlStatement::Select(q) = stmt else {
                panic!("expected SELECT");
            };
            let mem_res = execute(&q, &in_mem).expect("in-mem exec");
            let seg_res = execute_segmented(&q, dt).expect("seg exec");
            assert_eq!(mem_res.columns, seg_res.columns, "cols mismatch on {sql}");
            assert_eq!(mem_res.rows, seg_res.rows, "rows mismatch on {sql}");
        }
    }
}
