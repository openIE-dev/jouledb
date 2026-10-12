//! Columnar-first bulk ingest batches.
//!
//! A bulk load (COPY / Parquet / Arrow) arrives as typed columns. The
//! durable store logs them straight to the binary WAL (`JWB1`, the same
//! records the row path writes, so recovery is unchanged) and keeps the
//! batch as columns until it is materialized into row records and indexes,
//! which happens in the background or before any reader that needs rows.
//! Materialized records are identical to the row path's: every column of
//! the batch present (NULL cells as `Value::Null`) plus the table tag field.

use crate::{RecordId, Value};
use std::collections::HashMap;

/// One column's values for every row of a batch.
#[derive(Debug, Clone)]
pub enum ColData {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
    /// UTF-8 text: `offsets` has len + 1 entries into `bytes`
    Str { offsets: Vec<u32>, bytes: Vec<u8> },
}

#[derive(Debug, Clone)]
pub struct BatchColumn {
    pub name: String,
    pub data: ColData,
    /// Per-row validity; `None` = no NULLs
    pub valid: Option<Vec<bool>>,
}

impl BatchColumn {
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        self.valid.as_ref().is_some_and(|v| !v[i])
    }

    #[inline]
    pub fn str_at(&self, i: usize) -> &str {
        match &self.data {
            ColData::Str { offsets, bytes } => {
                // validated UTF-8 at construction
                std::str::from_utf8(&bytes[offsets[i] as usize..offsets[i + 1] as usize])
                    .unwrap_or_default()
            }
            _ => "",
        }
    }

    /// Row value as the row path stores it.
    pub fn value(&self, i: usize) -> Value {
        if self.is_null(i) {
            return Value::Null;
        }
        match &self.data {
            ColData::Int(v) => Value::Int(v[i]),
            ColData::Float(v) => Value::Float(v[i]),
            ColData::Bool(v) => Value::Bool(v[i]),
            ColData::Str { .. } => Value::String(self.str_at(i).to_string()),
        }
    }
}

/// A bulk batch of `len` rows for one table, ids `first_id..first_id+len`
/// (assigned by the store on ingest).
#[derive(Debug, Clone)]
pub struct ColumnBatch {
    /// Name of the table tag field and the table it marks
    pub table_field: String,
    pub table: String,
    pub first_id: RecordId,
    pub len: usize,
    pub cols: Vec<BatchColumn>,
}

impl ColumnBatch {
    /// Checks shapes (every column `len` rows, valid UTF-8, finite floats:
    /// the row path stores a non-finite float as NULL, which the typed
    /// WAL block cannot express, so callers take the row path for those).
    pub fn validate(&self) -> Result<(), String> {
        for c in &self.cols {
            let n = match &c.data {
                ColData::Int(v) => v.len(),
                ColData::Float(v) => {
                    if v.iter().enumerate().any(|(i, x)| !x.is_finite() && !c.is_null(i)) {
                        return Err(format!("column {}: non-finite float", c.name));
                    }
                    v.len()
                }
                ColData::Bool(v) => v.len(),
                ColData::Str { offsets, bytes } => {
                    if offsets.len() != self.len + 1
                        || offsets.windows(2).any(|w| w[0] > w[1])
                        || *offsets.last().unwrap_or(&0) as usize > bytes.len()
                    {
                        return Err(format!("column {}: bad offsets", c.name));
                    }
                    for i in 0..self.len {
                        if std::str::from_utf8(&bytes[offsets[i] as usize..offsets[i + 1] as usize]).is_err() {
                            return Err(format!("column {}: invalid UTF-8", c.name));
                        }
                    }
                    self.len
                }
            };
            if n != self.len || c.valid.as_ref().is_some_and(|v| v.len() != self.len) {
                return Err(format!("column {}: {} rows, batch has {}", c.name, n, self.len));
            }
            if c.name == self.table_field {
                return Err(format!("column {} collides with the table tag", c.name));
            }
        }
        Ok(())
    }

    /// Rows `from..to` as (id, fields), exactly as the row path builds them.
    pub fn rows(&self, from: usize, to: usize) -> Vec<(RecordId, HashMap<String, Value>)> {
        use rayon::prelude::*;
        (from..to)
            .into_par_iter()
            .map(|i| {
                let mut f = HashMap::with_capacity(self.cols.len() + 1);
                f.insert(self.table_field.clone(), Value::String(self.table.clone()));
                for c in &self.cols {
                    f.insert(c.name.clone(), c.value(i));
                }
                (self.first_id + i as u64, f)
            })
            .collect()
    }

    /// `JWB1` WAL records for rows `from..to`, byte-compatible with
    /// `wal_codec::encode_chunk` input of the same rows (decode yields the
    /// same (id, fields)). A column with NULLs uses the mixed (JSON) block,
    /// as the row encoder does for a column holding `Value::Null`.
    pub fn encode_chunk(&self, from: usize, to: usize) -> Vec<u8> {
        let n = to - from;
        let ncols = self.cols.len() + 1;
        let mut out = Vec::with_capacity(16 + n * (8 + ncols * 9));
        out.extend_from_slice(crate::wal_codec::MAGIC);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&(ncols as u32).to_le_bytes());
        for i in from..to {
            out.extend_from_slice(&(self.first_id + i as u64).to_le_bytes());
        }
        let bm_len = n.div_ceil(8);
        let all_present = |out: &mut Vec<u8>| {
            let at = out.len();
            out.resize(at + bm_len, 0xff);
            if n % 8 != 0 {
                out[at + bm_len - 1] = (1u8 << (n % 8)) - 1;
            }
        };
        // table tag column (string)
        out.extend_from_slice(&(self.table_field.len() as u16).to_le_bytes());
        out.extend_from_slice(self.table_field.as_bytes());
        out.push(crate::wal_codec::T_STR);
        all_present(&mut out);
        let tl = self.table.len() as u32;
        for k in 0..=n as u32 {
            out.extend_from_slice(&(k * tl).to_le_bytes());
        }
        for _ in 0..n {
            out.extend_from_slice(self.table.as_bytes());
        }
        for c in &self.cols {
            out.extend_from_slice(&(c.name.len() as u16).to_le_bytes());
            out.extend_from_slice(c.name.as_bytes());
            let has_null = (from..to).any(|i| c.is_null(i));
            if has_null {
                out.push(crate::wal_codec::T_MIXED);
                all_present(&mut out);
                for i in from..to {
                    let j = serde_json::to_vec(&c.value(i)).expect("Value serializes");
                    out.extend_from_slice(&(j.len() as u32).to_le_bytes());
                    out.extend_from_slice(&j);
                }
                continue;
            }
            match &c.data {
                ColData::Int(v) => {
                    out.push(crate::wal_codec::T_INT);
                    all_present(&mut out);
                    for x in &v[from..to] {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
                ColData::Float(v) => {
                    out.push(crate::wal_codec::T_FLOAT);
                    all_present(&mut out);
                    for x in &v[from..to] {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
                ColData::Bool(v) => {
                    out.push(crate::wal_codec::T_BOOL);
                    all_present(&mut out);
                    for x in &v[from..to] {
                        out.push(*x as u8);
                    }
                }
                ColData::Str { offsets, bytes } => {
                    out.push(crate::wal_codec::T_STR);
                    all_present(&mut out);
                    let base = offsets[from];
                    for o in &offsets[from..=to] {
                        out.extend_from_slice(&(o - base).to_le_bytes());
                    }
                    out.extend_from_slice(&bytes[base as usize..offsets[to] as usize]);
                }
            }
        }
        let crc = crc32fast::hash(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn sample(n: usize) -> ColumnBatch {
        let mut offsets = vec![0u32];
        let mut bytes = Vec::new();
        for i in 0..n {
            bytes.extend_from_slice(format!("u{}ü", i % 7).as_bytes());
            offsets.push(bytes.len() as u32);
        }
        ColumnBatch {
            table_field: "_table".into(),
            table: "hits".into(),
            first_id: 10,
            len: n,
            cols: vec![
                BatchColumn { name: "id".into(), data: ColData::Int((0..n as i64).collect()), valid: None },
                BatchColumn {
                    name: "d".into(),
                    data: ColData::Float((0..n).map(|i| i as f64 / 3.0).collect()),
                    valid: Some((0..n).map(|i| i % 5 != 0).collect()),
                },
                BatchColumn { name: "s".into(), data: ColData::Str { offsets, bytes }, valid: None },
                BatchColumn { name: "b".into(), data: ColData::Bool((0..n).map(|i| i % 2 == 0).collect()), valid: None },
            ],
        }
    }

    #[test]
    fn wal_chunk_decodes_to_row_path_rows() {
        let b = sample(1001);
        b.validate().unwrap();
        for (from, to) in [(0, 1001), (3, 517), (1000, 1001)] {
            let enc = b.encode_chunk(from, to);
            let dec = crate::wal_codec::decode_chunk(&enc).unwrap();
            let want = b.rows(from, to);
            assert_eq!(dec.len(), want.len());
            for (x, y) in dec.iter().zip(&want) {
                assert_eq!(x, y, "range {from}..{to}");
            }
            // and the row encoder's output decodes to the same rows
            let row_enc = crate::wal_codec::encode_chunk(&b.rows(from, to));
            assert_eq!(crate::wal_codec::decode_chunk(&row_enc).unwrap(), dec);
        }
    }

    #[test]
    fn validate_rejects_bad_shapes() {
        let mut b = sample(10);
        if let ColData::Float(v) = &mut b.cols[1].data {
            v[1] = f64::NAN;
        }
        assert!(b.validate().is_err());
        let mut b = sample(10);
        b.cols[0].data = ColData::Int(vec![1, 2]);
        assert!(b.validate().is_err());
    }
}
