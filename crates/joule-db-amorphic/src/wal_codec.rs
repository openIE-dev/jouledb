//! Binary columnar WAL record for batch ingest (`JWB1`).
//!
//! Layout (little endian):
//! ```text
//! "JWB1" | u32 nrows | u32 ncols | nrows x u64 ids
//! per column: u16 name_len | name | u8 tag | presence bitmap (ceil(n/8))
//!             | data block (by tag)
//! u32 crc32 of everything before it
//! ```
//! Tags: 1 Int (n x i64), 2 Float (n x f64), 3 Bool (n x u8),
//! 4 String (u32 offsets x (n+1) + bytes), 0 Mixed (per present value:
//! u32 len + JSON). Absent slots hold zeros. JSON WAL entries (old
//! databases) start with `{` and are still read by recovery.

use crate::{RecordId, Value};
use std::collections::HashMap;

pub const MAGIC: &[u8; 4] = b"JWB1";

const T_MIXED: u8 = 0;
const T_INT: u8 = 1;
const T_FLOAT: u8 = 2;
const T_BOOL: u8 = 3;
const T_STR: u8 = 4;

pub fn is_binary(data: &[u8]) -> bool {
    data.len() >= 4 && &data[..4] == MAGIC
}

/// Encode a chunk of rows (column order = first appearance).
pub fn encode_chunk(rows: &[(RecordId, HashMap<String, Value>)]) -> Vec<u8> {
    let n = rows.len();
    let mut columns: Vec<&str> = Vec::new();
    for (_, f) in rows {
        for k in f.keys() {
            if !columns.contains(&k.as_str()) {
                columns.push(k.as_str());
            }
        }
    }
    let mut out = Vec::with_capacity(16 + n * (8 + columns.len() * 9));
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(columns.len() as u32).to_le_bytes());
    for (id, _) in rows {
        out.extend_from_slice(&id.to_le_bytes());
    }
    let bm_len = n.div_ceil(8);
    for name in &columns {
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        let vals: Vec<Option<&Value>> = rows.iter().map(|(_, f)| f.get(*name)).collect();
        let mut tag = None;
        for v in vals.iter().flatten() {
            let t = match v {
                Value::Int(_) => T_INT,
                Value::Float(_) => T_FLOAT,
                Value::Bool(_) => T_BOOL,
                Value::String(_) => T_STR,
                _ => T_MIXED,
            };
            match tag {
                None => tag = Some(t),
                Some(p) if p != t => {
                    tag = Some(T_MIXED);
                    break;
                }
                _ => {}
            }
            if t == T_MIXED {
                tag = Some(T_MIXED);
                break;
            }
        }
        let tag = tag.unwrap_or(T_MIXED);
        out.push(tag);
        let bm_at = out.len();
        out.resize(bm_at + bm_len, 0);
        for (i, v) in vals.iter().enumerate() {
            if v.is_some() {
                out[bm_at + i / 8] |= 1 << (i % 8);
            }
        }
        match tag {
            T_INT => {
                out.reserve(n * 8);
                for v in &vals {
                    let x = if let Some(Value::Int(x)) = v { *x } else { 0 };
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            T_FLOAT => {
                out.reserve(n * 8);
                for v in &vals {
                    let x = if let Some(Value::Float(x)) = v { *x } else { 0.0 };
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            T_BOOL => {
                for v in &vals {
                    out.push(matches!(v, Some(Value::Bool(true))) as u8);
                }
            }
            T_STR => {
                let mut off = 0u32;
                out.extend_from_slice(&off.to_le_bytes());
                for v in &vals {
                    if let Some(Value::String(s)) = v {
                        off += s.len() as u32;
                    }
                    out.extend_from_slice(&off.to_le_bytes());
                }
                for v in &vals {
                    if let Some(Value::String(s)) = v {
                        out.extend_from_slice(s.as_bytes());
                    }
                }
            }
            _ => {
                for v in vals.iter().flatten() {
                    let j = serde_json::to_vec(v).expect("Value serializes");
                    out.extend_from_slice(&(j.len() as u32).to_le_bytes());
                    out.extend_from_slice(&j);
                }
            }
        }
    }
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.p..self.p + n).ok_or("truncated JWB1 record")?;
        self.p += n;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// Decode a `JWB1` record into (id, fields) rows. Errors on a bad CRC or
/// truncation (recovery then skips the record like any torn entry).
pub fn decode_chunk(data: &[u8]) -> Result<Vec<(RecordId, HashMap<String, Value>)>, String> {
    if !is_binary(data) || data.len() < 16 {
        return Err("not a JWB1 record".into());
    }
    let (body, tail) = data.split_at(data.len() - 4);
    if crc32fast::hash(body) != u32::from_le_bytes(tail.try_into().unwrap()) {
        return Err("JWB1 crc mismatch".into());
    }
    let mut r = Rd { b: body, p: 4 };
    let n = r.u32()? as usize;
    let nc = r.u32()? as usize;
    let mut rows: Vec<(RecordId, HashMap<String, Value>)> = Vec::with_capacity(n);
    for _ in 0..n {
        rows.push((r.u64()?, HashMap::with_capacity(nc)));
    }
    let bm_len = n.div_ceil(8);
    for _ in 0..nc {
        let nl = r.u16()? as usize;
        let name = std::str::from_utf8(r.take(nl)?).map_err(|e| e.to_string())?.to_string();
        let tag = r.take(1)?[0];
        let bm = r.take(bm_len)?;
        let has = |i: usize| bm[i / 8] & (1 << (i % 8)) != 0;
        match tag {
            T_INT | T_FLOAT => {
                let d = r.take(n * 8)?;
                for (i, row) in rows.iter_mut().enumerate() {
                    if has(i) {
                        let w: [u8; 8] = d[i * 8..i * 8 + 8].try_into().unwrap();
                        let v = if tag == T_INT {
                            Value::Int(i64::from_le_bytes(w))
                        } else {
                            Value::Float(f64::from_le_bytes(w))
                        };
                        row.1.insert(name.clone(), v);
                    }
                }
            }
            T_BOOL => {
                let d = r.take(n)?;
                for (i, row) in rows.iter_mut().enumerate() {
                    if has(i) {
                        row.1.insert(name.clone(), Value::Bool(d[i] != 0));
                    }
                }
            }
            T_STR => {
                let offs = r.take((n + 1) * 4)?;
                let off = |i: usize| u32::from_le_bytes(offs[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
                let bytes = r.take(off(n))?;
                for (i, row) in rows.iter_mut().enumerate() {
                    if has(i) {
                        let s = std::str::from_utf8(&bytes[off(i)..off(i + 1)]).map_err(|e| e.to_string())?;
                        row.1.insert(name.clone(), Value::String(s.to_string()));
                    }
                }
            }
            T_MIXED => {
                for (i, row) in rows.iter_mut().enumerate() {
                    if has(i) {
                        let l = r.u32()? as usize;
                        let v: Value = serde_json::from_slice(r.take(l)?).map_err(|e| e.to_string())?;
                        row.1.insert(name.clone(), v);
                    }
                }
            }
            t => return Err(format!("JWB1 unknown column tag {t}")),
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_types_and_absent() {
        let rows: Vec<(RecordId, HashMap<String, Value>)> = (0..1000u64)
            .map(|i| {
                let mut m = HashMap::new();
                m.insert("i".into(), Value::Int(i as i64 - 500));
                if i % 3 != 0 {
                    m.insert("f".into(), Value::Float(i as f64 / 7.0));
                }
                if i % 5 == 0 {
                    m.insert("s".into(), Value::String(format!("v{i} ünï")));
                }
                m.insert("b".into(), Value::Bool(i % 2 == 0));
                if i % 7 == 0 {
                    m.insert("z".into(), Value::Null);
                }
                m.insert(
                    "x".into(),
                    if i % 2 == 0 { Value::Int(1) } else { Value::String("a".into()) },
                );
                if i == 9 {
                    m.insert("a".into(), Value::Array(vec![Value::Int(1), Value::Null]));
                }
                (i * 3 + 1, m)
            })
            .collect();
        let enc = encode_chunk(&rows);
        assert!(is_binary(&enc));
        assert_eq!(decode_chunk(&enc).unwrap(), rows);
        let mut bad = enc.clone();
        bad[20] ^= 1;
        assert!(decode_chunk(&bad).is_err());
        assert!(decode_chunk(&enc[..enc.len() - 3]).is_err());
    }
}
