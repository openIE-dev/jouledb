//! Out-of-core disk-backed columnar segment storage (`.jcol`).
//!
//! Persists large analytical tables (`> JOULEDB_COL_SPILL_ROWS`, default `262,144`
//! rows) as immutable self-describing segment files (`<= 262,144` rows per segment)
//! under `<db_path>/col_segments/<table_hex>/seg_<seq:06>.jcol`.
//!
//! Only lightweight per-segment ZoneMap + byte-offset headers (`~1.6 MB` total for
//! `100,000,000` rows × `105` columns) stay resident in RAM. Analytical queries
//! prune whole segments via ZoneMaps and `pread` (`read_exact_at`) **only** the
//! `1–8` referenced columns per surviving segment, keeping process RSS bounded
//! (`< 4 GB`) even on full `100M`-row wide Parquet datasets.

#![deny(clippy::unwrap_used)]

use crate::columnar_exec::{Col, Ints, TableColumns};
use joule_db_amorphic::column_batch::{ColData, ColumnBatch};
use joule_db_amorphic::{AmorphicRecord, Value as AmorphicValue};
use joule_db_query::ast::Value as AstValue;
use joule_db_query::executor::RowData;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Maximum rows stored in a single `.jcol` segment file (`256 Ki` rows).
pub const MAX_SEGMENT_ROWS: usize = 262_144;

const SEG_MAGIC_HEAD: &[u8; 8] = b"JCOL\x01\x00\x00\x00";
const SEG_MAGIC_FOOT: &[u8; 4] = b"JCL1";
const NULL_CODE_U32: u32 = u32::MAX;

/// Ensure GNU `ptmalloc2` uses `mmap`/`munmap` for `>= 1 MiB` bulk Parquet/Arrow buffers
/// without starving 32-thread Rayon query workers on arena locks or 32 KiB TLB shootdowns.
#[inline]
pub fn ensure_linux_mmap_allocator() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        INIT.get_or_init(|| {
            const M_MMAP_THRESHOLD: i32 = -3;
            const M_MMAP_MAX: i32 = -4;
            unsafe extern "C" {
                fn mallopt(param: i32, value: i32) -> i32;
            }
            // SAFETY: Called at startup before worker threads allocate; keeps per-thread arenas
            // lock-free for `<= 1 MiB` segment scans while `munmap`ping multi-MB ingest buffers.
            unsafe {
                let _ = mallopt(M_MMAP_MAX, 1_048_576);
                let _ = mallopt(M_MMAP_THRESHOLD, 1_048_576);
            }
        });
    }
}

/// Return unused multi-MB Rayon thread-arena pages to the OS on GNU/Linux so
/// multi-million-row streaming `COPY` and out-of-core scans keep `VmRSS` bounded.
#[inline]
pub fn trim_unused_heap() {
    ensure_linux_mmap_allocator();
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: `malloc_trim(0)` is thread-safe in glibc and releases unused top-of-heap/arena pages.
        unsafe {
            let _ = malloc_trim(0);
        }
    }
}

/// Default row threshold (`128 Ki` rows) above which bulk loads spill directly
/// into durable `.jcol` disk segments instead of materializing in-memory `AmorphicRecord`s.
pub const DEFAULT_COL_SPILL_ROWS: usize = 131_072;

/// Row threshold above which streaming `COPY` / `insert_column_batch` spills
/// a table into durable `.jcol` disk segments instead of holding all rows in RAM.
pub fn disk_segment_spill_rows() -> usize {
    ensure_linux_mmap_allocator();
    static THRESH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THRESH.get_or_init(|| {
        std::env::var("JOULEDB_COL_SPILL_ROWS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_COL_SPILL_ROWS)
    })
}

/// Physical storage encoding of a column slice inside a `.jcol` segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DiskColKind {
    AllNull = 0,
    I8 = 1,
    I16 = 2,
    I32 = 3,
    I64 = 4,
    F64 = 5,
    Text = 6,
    Other = 7,
}

/// Per-column metadata kept in RAM for a single `.jcol` segment.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ColSliceMeta {
    pub kind: DiskColKind,
    pub has_nulls: bool,
    pub null_count: u32,
    /// `i64 as u64` for integer columns, `f64::to_bits()` for float columns.
    pub min_bits: u64,
    pub max_bits: u64,
    pub offset: u64,
    pub byte_len: u32,
    pub crc32: u32,
    /// Exact sum of non-null integer values in this segment (`None` for legacy `.jcol` or non-int columns).
    #[serde(default)]
    pub sum_i128: Option<i128>,
}

impl ColSliceMeta {
    #[inline]
    pub fn min_int(&self, seg_len: u32) -> Option<i64> {
        if self.null_count >= seg_len {
            return None;
        }
        match self.kind {
            DiskColKind::I8 | DiskColKind::I16 | DiskColKind::I32 | DiskColKind::I64 => {
                Some(self.min_bits as i64)
            }
            _ => None,
        }
    }

    #[inline]
    pub fn max_int(&self, seg_len: u32) -> Option<i64> {
        if self.null_count >= seg_len {
            return None;
        }
        match self.kind {
            DiskColKind::I8 | DiskColKind::I16 | DiskColKind::I32 | DiskColKind::I64 => {
                Some(self.max_bits as i64)
            }
            _ => None,
        }
    }

    #[inline]
    pub fn min_float(&self, seg_len: u32) -> Option<f64> {
        if self.null_count >= seg_len {
            return None;
        }
        match self.kind {
            DiskColKind::F64 => {
                let f = f64::from_bits(self.min_bits);
                (!f.is_nan()).then_some(f)
            }
            _ => None,
        }
    }

    #[inline]
    pub fn max_float(&self, seg_len: u32) -> Option<f64> {
        if self.null_count >= seg_len {
            return None;
        }
        match self.kind {
            DiskColKind::F64 => {
                let f = f64::from_bits(self.max_bits);
                (!f.is_nan()).then_some(f)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SegmentFooter {
    seq: u64,
    first_id: u64,
    len: u32,
    col_names: Vec<String>,
    cols: Vec<ColSliceMeta>,
}

/// In-memory handle + ZoneMap header for one immutable `.jcol` segment file.
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    pub seq: u64,
    pub first_id: u64,
    pub len: u32,
    pub path: PathBuf,
    /// Aligned 1:1 with `DiskColumnTable::col_order`.
    pub cols: Vec<ColSliceMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TableManifest {
    table: String,
    col_order: Vec<String>,
    next_seq: u64,
    max_record_id: u64,
    segment_files: Vec<String>,
}

/// Out-of-core segmented columnar table handle.
#[derive(Debug, Clone)]
pub struct DiskColumnTable {
    pub table: String,
    pub dir: PathBuf,
    pub col_order: Vec<String>,
    pub col_index: HashMap<String, usize>,
    pub segments: Vec<SegmentMeta>,
    pub total_rows: usize,
    pub next_seq: u64,
    pub max_record_id: u64,
}

fn table_dir_path(db_path: &Path, table: &str) -> PathBuf {
    let mut hex = String::with_capacity(table.len() * 2);
    for &b in table.as_bytes() {
        if b.is_ascii_alphanumeric() || b == b'_' {
            hex.push(b as char);
        } else {
            hex.push_str(&format!("_{b:02x}"));
        }
    }
    if hex.is_empty() {
        hex.push_str("_empty");
    }
    db_path.join("col_segments").join(hex)
}

#[inline]
fn crc32_bytes(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

fn read_slice_at(file: &File, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(&mut buf, offset)?;
    }
    #[cfg(not(unix))]
    {
        let mut clone = file.try_clone()?;
        clone.seek(SeekFrom::Start(offset))?;
        clone.read_exact(&mut buf)?;
    }
    Ok(buf)
}

fn pack_nulls(nulls: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; nulls.len().div_ceil(8)];
    for (i, &is_n) in nulls.iter().enumerate() {
        if is_n {
            out[i >> 3] |= 1u8 << (i & 7);
        }
    }
    out
}

fn unpack_nulls(bytes: &[u8], len: usize) -> Vec<bool> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let is_n = (bytes[i >> 3] >> (i & 7)) & 1 != 0;
        out.push(is_n);
    }
    out
}

struct EncodedCol {
    meta: ColSliceMeta,
    payload: Vec<u8>,
}

fn encode_int_values(vals: &[i64], nulls: &[bool]) -> EncodedCol {
    let len = vals.len();
    let has_nulls = !nulls.is_empty();
    let mut null_count = 0u32;
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    let mut isum: i128 = 0;
    let mut any = false;
    for (i, &x) in vals.iter().enumerate() {
        if has_nulls && nulls[i] {
            null_count += 1;
        } else {
            lo = lo.min(x);
            hi = hi.max(x);
            isum += x as i128;
            any = true;
        }
    }
    if !any {
        return EncodedCol {
            meta: ColSliceMeta {
                kind: DiskColKind::AllNull,
                has_nulls: false,
                null_count: len as u32,
                min_bits: 0,
                max_bits: 0,
                offset: 0,
                byte_len: 0,
                crc32: 0,
                sum_i128: Some(0),
            },
            payload: Vec::new(),
        };
    }

    let kind = if lo >= i8::MIN as i64 && hi <= i8::MAX as i64 {
        DiskColKind::I8
    } else if lo >= i16::MIN as i64 && hi <= i16::MAX as i64 {
        DiskColKind::I16
    } else if lo >= i32::MIN as i64 && hi <= i32::MAX as i64 {
        DiskColKind::I32
    } else {
        DiskColKind::I64
    };

    let width = match kind {
        DiskColKind::I8 => 1,
        DiskColKind::I16 => 2,
        DiskColKind::I32 => 4,
        _ => 8,
    };
    let null_bytes = if has_nulls && null_count > 0 {
        pack_nulls(nulls)
    } else {
        Vec::new()
    };
    let actual_has_nulls = !null_bytes.is_empty();
    let mut payload = Vec::with_capacity(null_bytes.len() + len * width);
    payload.extend_from_slice(&null_bytes);
    match kind {
        DiskColKind::I8 => {
            for &x in vals {
                payload.push(x as i8 as u8);
            }
        }
        DiskColKind::I16 => {
            for &x in vals {
                payload.extend_from_slice(&(x as i16).to_le_bytes());
            }
        }
        DiskColKind::I32 => {
            for &x in vals {
                payload.extend_from_slice(&(x as i32).to_le_bytes());
            }
        }
        _ => {
            for &x in vals {
                payload.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    let crc32 = crc32_bytes(&payload);
    EncodedCol {
        meta: ColSliceMeta {
            kind,
            has_nulls: actual_has_nulls,
            null_count,
            min_bits: lo as u64,
            max_bits: hi as u64,
            offset: 0,
            byte_len: payload.len() as u32,
            crc32,
            sum_i128: Some(isum),
        },
        payload,
    }
}

fn encode_float_values(vals: &[f64], nulls: &[bool]) -> EncodedCol {
    let len = vals.len();
    let has_nulls = !nulls.is_empty();
    let mut null_count = 0u32;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut any = false;
    let mut saw_nan = false;
    for (i, &x) in vals.iter().enumerate() {
        if has_nulls && nulls[i] {
            null_count += 1;
        } else {
            if x.is_nan() {
                saw_nan = true;
            } else {
                lo = lo.min(x);
                hi = hi.max(x);
            }
            any = true;
        }
    }
    if !any {
        return EncodedCol {
            meta: ColSliceMeta {
                kind: DiskColKind::AllNull,
                has_nulls: false,
                null_count: len as u32,
                min_bits: 0,
                max_bits: 0,
                offset: 0,
                byte_len: 0,
                crc32: 0,
                sum_i128: None,
            },
            payload: Vec::new(),
        };
    }
    let null_bytes = if has_nulls && null_count > 0 {
        pack_nulls(nulls)
    } else {
        Vec::new()
    };
    let actual_has_nulls = !null_bytes.is_empty();
    let mut payload = Vec::with_capacity(null_bytes.len() + len * 8);
    payload.extend_from_slice(&null_bytes);
    for &x in vals {
        payload.extend_from_slice(&x.to_le_bytes());
    }
    let crc32 = crc32_bytes(&payload);
    let (min_bits, max_bits) = if saw_nan {
        (f64::NAN.to_bits(), f64::NAN.to_bits())
    } else {
        (lo.to_bits(), hi.to_bits())
    };
    EncodedCol {
        meta: ColSliceMeta {
            kind: DiskColKind::F64,
            has_nulls: actual_has_nulls,
            null_count,
            min_bits,
            max_bits,
            offset: 0,
            byte_len: payload.len() as u32,
            crc32,
            sum_i128: None,
        },
        payload,
    }
}

fn encode_text_iter<'a>(len: usize, mut get: impl FnMut(usize) -> Option<&'a str>) -> EncodedCol {
    let mut idx: rustc_hash::FxHashMap<&'a str, u32> = rustc_hash::FxHashMap::default();
    let mut dict: Vec<&'a str> = Vec::new();
    let mut codes: Vec<u32> = Vec::with_capacity(len);
    let mut null_count = 0u32;
    let mut arena_bytes = 0usize;

    for i in 0..len {
        match get(i) {
            None => {
                null_count += 1;
                codes.push(NULL_CODE_U32);
            }
            Some(s) => {
                if let Some(&c) = idx.get(s) {
                    codes.push(c);
                } else {
                    let c = dict.len() as u32;
                    dict.push(s);
                    idx.insert(s, c);
                    arena_bytes += s.len();
                    codes.push(c);
                }
            }
        }
    }

    if null_count as usize == len {
        return EncodedCol {
            meta: ColSliceMeta {
                kind: DiskColKind::AllNull,
                has_nulls: false,
                null_count,
                min_bits: 0,
                max_bits: 0,
                offset: 0,
                byte_len: 0,
                crc32: 0,
                sum_i128: None,
            },
            payload: Vec::new(),
        };
    }

    let dict_len = dict.len();
    let code_width: u8 = if dict_len <= 254 {
        1
    } else if dict_len <= 65_534 {
        2
    } else {
        4
    };
    let cap = 5 + (dict_len + 1) * 4 + arena_bytes + len * (code_width as usize);
    let mut payload = Vec::with_capacity(cap);
    payload.extend_from_slice(&(dict_len as u32).to_le_bytes());
    payload.push(code_width);

    let mut cur_off = 0u32;
    payload.extend_from_slice(&0u32.to_le_bytes());
    for s in &dict {
        cur_off = cur_off.saturating_add(s.len() as u32);
        payload.extend_from_slice(&cur_off.to_le_bytes());
    }
    for s in &dict {
        payload.extend_from_slice(s.as_bytes());
    }
    match code_width {
        1 => {
            for &c in &codes {
                payload.push(if c == NULL_CODE_U32 { 0xFF } else { c as u8 });
            }
        }
        2 => {
            for &c in &codes {
                let u = if c == NULL_CODE_U32 {
                    0xFFFFu16
                } else {
                    c as u16
                };
                payload.extend_from_slice(&u.to_le_bytes());
            }
        }
        _ => {
            for &c in &codes {
                payload.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    let crc32 = crc32_bytes(&payload);
    EncodedCol {
        meta: ColSliceMeta {
            kind: DiskColKind::Text,
            has_nulls: null_count > 0,
            null_count,
            min_bits: 0,
            max_bits: 0,
            offset: 0,
            byte_len: payload.len() as u32,
            crc32,
            sum_i128: None,
        },
        payload,
    }
}

fn encode_batch_col(batch: &ColumnBatch, col_name: &str, s: usize, e: usize) -> EncodedCol {
    let len = e - s;
    let Some(bc) = batch.cols.iter().find(|c| c.name == col_name) else {
        return EncodedCol {
            meta: ColSliceMeta {
                kind: DiskColKind::AllNull,
                has_nulls: false,
                null_count: len as u32,
                min_bits: 0,
                max_bits: 0,
                offset: 0,
                byte_len: 0,
                crc32: 0,
                sum_i128: None,
            },
            payload: Vec::new(),
        };
    };

    let nulls: Vec<bool> = match &bc.valid {
        Some(v) if v[s..e].iter().any(|&ok| !ok) => v[s..e].iter().map(|&ok| !ok).collect(),
        _ => Vec::new(),
    };

    match &bc.data {
        ColData::Int8(v) => {
            let vals: Vec<i64> = (s..e)
                .map(|i| if bc.is_null(i) { 0 } else { v[i] as i64 })
                .collect();
            encode_int_values(&vals, &nulls)
        }
        ColData::Int16(v) => {
            let vals: Vec<i64> = (s..e)
                .map(|i| if bc.is_null(i) { 0 } else { v[i] as i64 })
                .collect();
            encode_int_values(&vals, &nulls)
        }
        ColData::Int32(v) => {
            let vals: Vec<i64> = (s..e)
                .map(|i| if bc.is_null(i) { 0 } else { v[i] as i64 })
                .collect();
            encode_int_values(&vals, &nulls)
        }
        ColData::Int(v) => {
            let vals: Vec<i64> = (s..e)
                .map(|i| if bc.is_null(i) { 0 } else { v[i] })
                .collect();
            encode_int_values(&vals, &nulls)
        }
        ColData::Float(v) => {
            let vals: Vec<f64> = (s..e)
                .map(|i| if bc.is_null(i) { 0.0 } else { v[i] })
                .collect();
            encode_float_values(&vals, &nulls)
        }
        ColData::Str { .. } => {
            encode_text_iter(len, |k| (!bc.is_null(s + k)).then(|| bc.str_at(s + k)))
        }
        ColData::Bool(_) => EncodedCol {
            meta: ColSliceMeta {
                kind: DiskColKind::Other,
                has_nulls: false,
                null_count: 0,
                min_bits: 0,
                max_bits: 0,
                offset: 0,
                byte_len: 0,
                crc32: 0,
                sum_i128: None,
            },
            payload: Vec::new(),
        },
    }
}

fn encode_records_col(records: &[&AmorphicRecord], col_name: &str) -> EncodedCol {
    let n = records.len();
    let (mut ints, mut floats, mut texts, mut other) = (0usize, 0usize, 0usize, false);
    for r in records {
        match r.get(col_name) {
            None | Some(AmorphicValue::Null) => {}
            Some(AmorphicValue::Int(_)) => ints += 1,
            Some(AmorphicValue::Float(_)) => floats += 1,
            Some(AmorphicValue::String(_)) => texts += 1,
            Some(_) => other = true,
        }
    }
    if other || (texts > 0 && ints + floats > 0) {
        return EncodedCol {
            meta: ColSliceMeta {
                kind: DiskColKind::Other,
                has_nulls: false,
                null_count: 0,
                min_bits: 0,
                max_bits: 0,
                offset: 0,
                byte_len: 0,
                crc32: 0,
                sum_i128: None,
            },
            payload: Vec::new(),
        };
    }
    if texts > 0 {
        return encode_text_iter(n, |i| match records[i].get(col_name) {
            Some(AmorphicValue::String(s)) => Some(s.as_str()),
            _ => None,
        });
    }
    if floats > 0 {
        let mut vals = Vec::with_capacity(n);
        let mut nulls = Vec::with_capacity(n);
        for r in records {
            match r.get(col_name) {
                Some(AmorphicValue::Float(f)) => {
                    vals.push(*f);
                    nulls.push(false);
                }
                Some(AmorphicValue::Int(i)) => {
                    vals.push(*i as f64);
                    nulls.push(false);
                }
                _ => {
                    vals.push(0.0);
                    nulls.push(true);
                }
            }
        }
        if !nulls.iter().any(|&b| b) {
            nulls.clear();
        }
        return encode_float_values(&vals, &nulls);
    }
    let mut vals = Vec::with_capacity(n);
    let mut nulls = Vec::with_capacity(n);
    for r in records {
        match r.get(col_name) {
            Some(AmorphicValue::Int(i)) => {
                vals.push(*i);
                nulls.push(false);
            }
            _ => {
                vals.push(0);
                nulls.push(true);
            }
        }
    }
    if !nulls.iter().any(|&b| b) {
        nulls.clear();
    }
    encode_int_values(&vals, &nulls)
}

fn decode_col_payload(
    meta: &ColSliceMeta,
    seg_len: usize,
    raw: &[u8],
    inferred_kind: DiskColKind,
) -> io::Result<Col> {
    if meta.byte_len > 0 && crc32_bytes(raw) != meta.crc32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "corrupt .jcol column slice (CRC32 mismatch)",
        ));
    }
    match meta.kind {
        DiskColKind::AllNull => match inferred_kind {
            DiskColKind::Text => Ok(Col::Text(vec![NULL_CODE_U32; seg_len], Vec::new())),
            DiskColKind::F64 => Ok(Col::Float(vec![0.0; seg_len], vec![true; seg_len])),
            DiskColKind::Other => Ok(Col::Other),
            _ => Ok(Col::Int(Ints::I8(vec![0; seg_len]), vec![true; seg_len])),
        },
        DiskColKind::Other => Ok(Col::Other),
        DiskColKind::I8 | DiskColKind::I16 | DiskColKind::I32 | DiskColKind::I64 => {
            let null_bytes = if meta.has_nulls {
                seg_len.div_ceil(8)
            } else {
                0
            };
            if raw.len() < null_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated .jcol null bitmap",
                ));
            }
            let nulls = if meta.has_nulls {
                unpack_nulls(&raw[..null_bytes], seg_len)
            } else {
                Vec::new()
            };
            let body = &raw[null_bytes..];
            let ints = match meta.kind {
                DiskColKind::I8 => {
                    if body.len() < seg_len {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "short i8 slice",
                        ));
                    }
                    Ints::I8(body[..seg_len].iter().map(|&b| b as i8).collect())
                }
                DiskColKind::I16 => {
                    if body.len() < seg_len * 2 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "short i16 slice",
                        ));
                    }
                    let mut v = Vec::with_capacity(seg_len);
                    for ch in body[..seg_len * 2].chunks_exact(2) {
                        v.push(i16::from_le_bytes([ch[0], ch[1]]));
                    }
                    Ints::I16(v)
                }
                DiskColKind::I32 => {
                    if body.len() < seg_len * 4 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "short i32 slice",
                        ));
                    }
                    let mut v = Vec::with_capacity(seg_len);
                    for ch in body[..seg_len * 4].chunks_exact(4) {
                        v.push(i32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                    }
                    Ints::I32(v)
                }
                _ => {
                    if body.len() < seg_len * 8 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "short i64 slice",
                        ));
                    }
                    let mut v = Vec::with_capacity(seg_len);
                    for ch in body[..seg_len * 8].chunks_exact(8) {
                        v.push(i64::from_le_bytes([
                            ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7],
                        ]));
                    }
                    Ints::I64(v)
                }
            };
            Ok(Col::Int(ints, nulls))
        }
        DiskColKind::F64 => {
            let null_bytes = if meta.has_nulls {
                seg_len.div_ceil(8)
            } else {
                0
            };
            if raw.len() < null_bytes + seg_len * 8 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short f64 slice",
                ));
            }
            let nulls = if meta.has_nulls {
                unpack_nulls(&raw[..null_bytes], seg_len)
            } else {
                Vec::new()
            };
            let body = &raw[null_bytes..null_bytes + seg_len * 8];
            let mut v = Vec::with_capacity(seg_len);
            for ch in body.chunks_exact(8) {
                v.push(f64::from_le_bytes([
                    ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7],
                ]));
            }
            Ok(Col::Float(v, nulls))
        }
        DiskColKind::Text => {
            if raw.len() < 9 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short text header",
                ));
            }
            let dict_len = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
            let code_width = raw[4];
            let offs_start = 5usize;
            let offs_end = offs_start + (dict_len + 1) * 4;
            if raw.len() < offs_end {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short text offsets",
                ));
            }
            let mut offs = Vec::with_capacity(dict_len + 1);
            for ch in raw[offs_start..offs_end].chunks_exact(4) {
                offs.push(u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]) as usize);
            }
            let arena_len = *offs.last().unwrap_or(&0);
            let arena_end = offs_end + arena_len;
            if raw.len() < arena_end {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short text arena",
                ));
            }
            let arena = &raw[offs_end..arena_end];
            let mut dict = Vec::with_capacity(dict_len);
            for i in 0..dict_len {
                let (a, b) = (offs[i], offs[i + 1]);
                if b < a || b > arena.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid text dictionary offset",
                    ));
                }
                let s = std::str::from_utf8(&arena[a..b])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                dict.push(Arc::from(s));
            }
            let codes_raw = &raw[arena_end..];
            let mut codes = Vec::with_capacity(seg_len);
            match code_width {
                1 => {
                    if codes_raw.len() < seg_len {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "short u8 text codes",
                        ));
                    }
                    for &b in &codes_raw[..seg_len] {
                        codes.push(if b == 0xFF { NULL_CODE_U32 } else { b as u32 });
                    }
                }
                2 => {
                    if codes_raw.len() < seg_len * 2 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "short u16 text codes",
                        ));
                    }
                    for ch in codes_raw[..seg_len * 2].chunks_exact(2) {
                        let u = u16::from_le_bytes([ch[0], ch[1]]);
                        codes.push(if u == 0xFFFF { NULL_CODE_U32 } else { u as u32 });
                    }
                }
                _ => {
                    if codes_raw.len() < seg_len * 4 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "short u32 text codes",
                        ));
                    }
                    for ch in codes_raw[..seg_len * 4].chunks_exact(4) {
                        codes.push(u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                    }
                }
            }
            Ok(Col::Text(codes, dict))
        }
    }
}

/// Sparse dictionary-compacted decoder for `.jcol` `Text` slices when only a small
/// subset of rows (`sel.len() <= 16_384`) survived `WHERE` or `ORDER BY ... LIMIT k`!
/// Avoids allocating `100,000+` `Arc<str>` strings when only `1..1000` rows are read.
fn decode_col_payload_selected(
    meta: &ColSliceMeta,
    seg_len: usize,
    raw: &[u8],
    inferred_kind: DiskColKind,
    sel: Option<&[u32]>,
) -> io::Result<Col> {
    let Some(rows) = sel.filter(|s| s.len() <= 16_384) else {
        return decode_col_payload(meta, seg_len, raw, inferred_kind);
    };
    if meta.kind != DiskColKind::Text {
        return decode_col_payload(meta, seg_len, raw, inferred_kind);
    }
    if meta.byte_len > 0 && crc32_bytes(raw) != meta.crc32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "corrupt .jcol slice (CRC32 mismatch)",
        ));
    }
    if raw.len() < 9 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short text header",
        ));
    }
    let dict_len = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
    if dict_len <= rows.len().saturating_mul(2).max(64) {
        return decode_col_payload(meta, seg_len, raw, inferred_kind);
    }
    let code_width = raw[4];
    let offs_start = 5usize;
    let offs_end = offs_start + (dict_len + 1) * 4;
    if raw.len() < offs_end {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short text offsets",
        ));
    }
    let arena_len = u32::from_le_bytes([
        raw[offs_end - 4],
        raw[offs_end - 3],
        raw[offs_end - 2],
        raw[offs_end - 1],
    ]) as usize;
    let arena_end = offs_end + arena_len;
    if raw.len() < arena_end {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short text arena",
        ));
    }
    let arena = &raw[offs_end..arena_end];
    let codes_raw = &raw[arena_end..];
    let need_code_bytes = match code_width {
        1 => seg_len,
        2 => seg_len * 2,
        _ => seg_len * 4,
    };
    if codes_raw.len() < need_code_bytes {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short selected text codes",
        ));
    }

    let mut codes = vec![NULL_CODE_U32; seg_len];
    let mut compact_dict: Vec<Arc<str>> = Vec::with_capacity(rows.len().min(dict_len));
    let mut remap: rustc_hash::FxHashMap<u32, u32> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(rows.len().min(dict_len), Default::default());

    for &ri_u32 in rows {
        let ri = ri_u32 as usize;
        if ri >= seg_len {
            continue;
        }
        let orig_c = match code_width {
            1 => {
                let b = codes_raw[ri];
                if b == 0xFF { NULL_CODE_U32 } else { b as u32 }
            }
            2 => {
                let o = ri * 2;
                let u = u16::from_le_bytes([codes_raw[o], codes_raw[o + 1]]);
                if u == 0xFFFF { NULL_CODE_U32 } else { u as u32 }
            }
            _ => {
                let o = ri * 4;
                u32::from_le_bytes([
                    codes_raw[o],
                    codes_raw[o + 1],
                    codes_raw[o + 2],
                    codes_raw[o + 3],
                ])
            }
        };
        if orig_c == NULL_CODE_U32 {
            codes[ri] = NULL_CODE_U32;
            continue;
        }
        if let Some(&nc) = remap.get(&orig_c) {
            codes[ri] = nc;
            continue;
        }
        let di = orig_c as usize;
        if di >= dict_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "text dictionary code out of bounds",
            ));
        }
        let oa = offs_start + di * 4;
        let a = u32::from_le_bytes([raw[oa], raw[oa + 1], raw[oa + 2], raw[oa + 3]]) as usize;
        let b =
            u32::from_le_bytes([raw[oa + 4], raw[oa + 5], raw[oa + 6], raw[oa + 7]]) as usize;
        if b < a || b > arena.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid text dictionary offset",
            ));
        }
        let s = std::str::from_utf8(&arena[a..b])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let nc = compact_dict.len() as u32;
        compact_dict.push(Arc::from(s));
        remap.insert(orig_c, nc);
        codes[ri] = nc;
    }
    Ok(Col::Text(codes, compact_dict))
}

fn write_segment_file_raw(
    dir: &Path,
    seq: u64,
    first_id: u64,
    len: u32,
    col_order: &[String],
    mut encoded: Vec<EncodedCol>,
    sync_data: bool,
) -> io::Result<()> {
    let fname = format!("seg_{seq:06}.jcol");
    let tmp_path = dir.join(format!("seg_{seq:06}.jcol.tmp"));
    let final_path = dir.join(&fname);

    {
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)?;
        let mut w = BufWriter::with_capacity(4 * 1024 * 1024, f);
        w.write_all(SEG_MAGIC_HEAD)?;
        let mut cursor = SEG_MAGIC_HEAD.len() as u64;
        for ec in &mut encoded {
            ec.meta.offset = cursor;
            ec.meta.byte_len = ec.payload.len() as u32;
            if !ec.payload.is_empty() {
                w.write_all(&ec.payload)?;
                cursor += ec.payload.len() as u32 as u64;
            }
            ec.payload = Vec::new();
        }
        let cols: Vec<ColSliceMeta> = encoded.into_iter().map(|e| e.meta).collect();
        let footer = SegmentFooter {
            seq,
            first_id,
            len,
            col_names: col_order.to_vec(),
            cols,
        };
        let footer_bytes = serde_json::to_vec(&footer)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let footer_crc = crc32_bytes(&footer_bytes);
        w.write_all(&footer_bytes)?;
        w.write_all(&(footer_bytes.len() as u32).to_le_bytes())?;
        w.write_all(&footer_crc.to_le_bytes())?;
        w.write_all(SEG_MAGIC_FOOT)?;
        w.flush()?;
        let inner = w.into_inner().map_err(|e| e.into_error())?;
        if sync_data {
            inner.sync_data()?;
        }
    }
    fs::rename(&tmp_path, &final_path)?;
    Ok(())
}

fn write_segment_file(
    dir: &Path,
    seq: u64,
    first_id: u64,
    len: u32,
    col_order: &[String],
    encoded: Vec<EncodedCol>,
    sync_data: bool,
) -> io::Result<(String, SegmentMeta)> {
    write_segment_file_raw(dir, seq, first_id, len, col_order, encoded, sync_data)?;
    let fname = format!("seg_{seq:06}.jcol");
    let final_path = dir.join(&fname);
    let footer = read_segment_footer(&final_path)?;
    Ok((
        fname,
        SegmentMeta {
            seq: footer.seq,
            first_id: footer.first_id,
            len: footer.len,
            path: final_path,
            cols: footer.cols,
        },
    ))
}

fn read_segment_footer(path: &Path) -> io::Result<SegmentFooter> {
    let mut f = File::open(path)?;
    let file_len = f.metadata()?.len();
    if file_len < 20 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file too small for .jcol",
        ));
    }
    let mut tail = [0u8; 12];
    f.seek(SeekFrom::End(-12))?;
    f.read_exact(&mut tail)?;
    if &tail[8..12] != SEG_MAGIC_FOOT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing JCL1 footer magic",
        ));
    }
    let footer_len = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as u64;
    let footer_crc = u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]);
    if footer_len + 20 > file_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid .jcol footer length",
        ));
    }
    f.seek(SeekFrom::End(-(12 + footer_len as i64)))?;
    let mut buf = vec![0u8; footer_len as usize];
    f.read_exact(&mut buf)?;
    if crc32_bytes(&buf) != footer_crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "corrupt .jcol footer CRC32",
        ));
    }
    serde_json::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

impl DiskColumnTable {
    pub fn create_new(db_path: &Path, table: &str, col_order: &[String]) -> io::Result<Self> {
        let dir = table_dir_path(db_path, table);
        fs::create_dir_all(&dir)?;
        let col_index = col_order
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        let dt = Self {
            table: table.to_string(),
            dir,
            col_order: col_order.to_vec(),
            col_index,
            segments: Vec::new(),
            total_rows: 0,
            next_seq: 0,
            max_record_id: 0,
        };
        dt.save_manifest(false)?;
        Ok(dt)
    }

    pub fn open_all(db_path: &Path) -> HashMap<String, Arc<Self>> {
        let mut out = HashMap::new();
        let root = db_path.join("col_segments");
        let Ok(entries) = fs::read_dir(&root) else {
            return out;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            if let Ok(dt) = Self::open_table_dir(&dir)
                && dt.total_rows > 0
            {
                out.insert(dt.table.clone(), Arc::new(dt));
            }
        }
        out
    }

    fn open_table_dir(dir: &Path) -> io::Result<Self> {
        let manifest_path = dir.join("meta.json");
        let raw = fs::read(&manifest_path)?;
        let manifest: TableManifest = serde_json::from_slice(&raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let col_index: HashMap<String, usize> = manifest
            .col_order
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        let mut segments = Vec::with_capacity(manifest.segment_files.len());
        let mut total_rows = 0usize;
        for sf in &manifest.segment_files {
            let p = dir.join(sf);
            let footer = read_segment_footer(&p)?;
            // Realign columns if schema order changed
            let cols = if footer.col_names == manifest.col_order {
                footer.cols
            } else {
                let mut aligned = Vec::with_capacity(manifest.col_order.len());
                for cn in &manifest.col_order {
                    if let Some(pos) = footer.col_names.iter().position(|x| x == cn) {
                        aligned.push(footer.cols[pos]);
                    } else {
                        aligned.push(ColSliceMeta {
                            kind: DiskColKind::AllNull,
                            has_nulls: false,
                            null_count: footer.len,
                            min_bits: 0,
                            max_bits: 0,
                            offset: 0,
                            byte_len: 0,
                            crc32: 0,
                            sum_i128: Some(0),
                        });
                    }
                }
                aligned
            };
            total_rows += footer.len as usize;
            segments.push(SegmentMeta {
                seq: footer.seq,
                first_id: footer.first_id,
                len: footer.len,
                path: p,
                cols,
            });
        }
        Ok(Self {
            table: manifest.table,
            dir: dir.to_path_buf(),
            col_order: manifest.col_order,
            col_index,
            segments,
            total_rows,
            next_seq: manifest.next_seq,
            max_record_id: manifest.max_record_id,
        })
    }

    pub fn remove_table_files(db_path: &Path, table: &str) -> io::Result<()> {
        let dir = table_dir_path(db_path, table);
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    fn save_manifest(&self, sync_data: bool) -> io::Result<()> {
        let manifest = TableManifest {
            table: self.table.clone(),
            col_order: self.col_order.clone(),
            next_seq: self.next_seq,
            max_record_id: self.max_record_id,
            segment_files: self
                .segments
                .iter()
                .filter_map(|s| s.path.file_name().and_then(|n| n.to_str()).map(String::from))
                .collect(),
        };
        let bytes = serde_json::to_vec(&manifest)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let tmp = self.dir.join("meta.json.tmp");
        let dst = self.dir.join("meta.json");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            if sync_data {
                f.sync_data()?;
            }
        }
        fs::rename(&tmp, &dst)?;
        Ok(())
    }

    /// Like [`Self::append_column_batch_from`], taking ownership of `batch` so all Rayon worker
    /// buffers and `ColumnBatch` vectors are dropped and `malloc_trim(0)` runs **before**
    /// long-lived `SegmentMeta` headers are allocated on the caller thread.
    pub fn append_column_batch_owned(
        &mut self,
        batch: ColumnBatch,
        row_offset: usize,
        first_id_base: u64,
        sync_data: bool,
    ) -> io::Result<usize> {
        use rayon::prelude::*;
        if row_offset >= batch.len {
            drop(batch);
            trim_unused_heap();
            return Ok(0);
        }
        let mut ranges: Vec<(u64, usize, usize)> = Vec::new();
        let mut cur = row_offset;
        let mut seq = self.next_seq;
        while cur < batch.len {
            let end = (cur + MAX_SEGMENT_ROWS).min(batch.len);
            ranges.push((seq, cur, end));
            seq += 1;
            cur = end;
        }
        let dir = &self.dir;
        let col_order = &self.col_order;
        // Return ONLY `u64` segment sequence numbers from Rayon workers so ZERO long-lived heap
        // allocations (`PathBuf` / `Vec<ColSliceMeta>`) pin Rayon thread `sub_heap` arenas!
        let built_seqs: Vec<io::Result<u64>> = ranges
            .par_iter()
            .map(|&(seg_seq, s, e)| {
                let seg_len = (e - s) as u32;
                let seg_first_id = first_id_base + (s - row_offset) as u64;
                let encoded: Vec<EncodedCol> = col_order
                    .par_iter()
                    .map(|cn| encode_batch_col(&batch, cn, s, e))
                    .collect();
                let _ = write_segment_file_raw(
                    dir,
                    seg_seq,
                    seg_first_id,
                    seg_len,
                    col_order,
                    encoded,
                    sync_data,
                )?;
                Ok(seg_seq)
            })
            .collect();

        // Drop `batch` and release all temporary Rayon/main-arena pages BEFORE allocating `SegmentMeta`!
        drop(batch);
        drop(ranges);
        trim_unused_heap();

        let mut written = 0usize;
        for res in built_seqs {
            let seg_seq = res?;
            let final_path = self.dir.join(format!("seg_{seg_seq:06}.jcol"));
            let footer = read_segment_footer(&final_path)?;
            let meta = SegmentMeta {
                seq: footer.seq,
                first_id: footer.first_id,
                len: footer.len,
                path: final_path,
                cols: footer.cols,
            };
            written += meta.len as usize;
            self.total_rows += meta.len as usize;
            self.max_record_id = self
                .max_record_id
                .max(meta.first_id.saturating_add(meta.len as u64));
            self.segments.push(meta);
        }
        self.next_seq = seq;
        self.save_manifest(sync_data)?;
        Ok(written)
    }

    /// Append rows `[row_offset .. batch.len]` of `batch` as one or more
    /// `<= MAX_SEGMENT_ROWS` `.jcol` segments (encoding & writing multiple
    /// segments in parallel when a multi-row-group Parquet chunk arrives).
    pub fn append_column_batch_from(
        &mut self,
        batch: &ColumnBatch,
        row_offset: usize,
        first_id_base: u64,
        sync_data: bool,
    ) -> io::Result<usize> {
        use rayon::prelude::*;
        if row_offset >= batch.len {
            return Ok(0);
        }
        let mut ranges: Vec<(u64, usize, usize)> = Vec::new();
        let mut cur = row_offset;
        let mut seq = self.next_seq;
        while cur < batch.len {
            let end = (cur + MAX_SEGMENT_ROWS).min(batch.len);
            ranges.push((seq, cur, end));
            seq += 1;
            cur = end;
        }
        let dir = &self.dir;
        let col_order = &self.col_order;
        let built: Vec<io::Result<SegmentMeta>> = ranges
            .par_iter()
            .map(|&(seg_seq, s, e)| {
                let seg_len = (e - s) as u32;
                let seg_first_id = first_id_base + (s - row_offset) as u64;
                let encoded: Vec<EncodedCol> = col_order
                    .par_iter()
                    .map(|cn| encode_batch_col(batch, cn, s, e))
                    .collect();
                let (_fname, meta) = write_segment_file(
                    dir,
                    seg_seq,
                    seg_first_id,
                    seg_len,
                    col_order,
                    encoded,
                    sync_data,
                )?;
                Ok(meta)
            })
            .collect();

        let mut written = 0usize;
        for res in built {
            let meta = res?;
            written += meta.len as usize;
            self.total_rows += meta.len as usize;
            self.max_record_id = self
                .max_record_id
                .max(meta.first_id.saturating_add(meta.len as u64));
            self.segments.push(meta);
        }
        self.next_seq = seq;
        self.save_manifest(sync_data)?;
        trim_unused_heap();
        Ok(written)
    }

    /// Exact global integer sum if every non-empty segment carries `sum_i128`.
    pub fn global_int_sum(&self, col_name: &str) -> Option<(u64, i128)> {
        let &ci = self.col_index.get(col_name)?;
        let kind = self.inferred_col_kind(ci);
        if !matches!(
            kind,
            DiskColKind::I8 | DiskColKind::I16 | DiskColKind::I32 | DiskColKind::I64
        ) {
            return None;
        }
        let mut non_null = 0u64;
        let mut total_sum: i128 = 0;
        for seg in &self.segments {
            let m = &seg.cols[ci];
            let seg_non_null = (seg.len as u64).saturating_sub(m.null_count as u64);
            non_null += seg_non_null;
            if seg_non_null > 0 {
                total_sum = total_sum.checked_add(m.sum_i128?)?;
            }
        }
        Some((non_null, total_sum))
    }

    /// Read a `.jcol` `Text` column slice in zero-`Arc<str>` raw byte-arena form
    /// so `COUNT(DISTINCT text)` and `GROUP BY text` never heap-allocate `dict_len` `Arc<str>`s.
    pub fn read_raw_text_slice(
        &self,
        seg_idx: usize,
        col_name: &str,
    ) -> io::Result<Option<RawTextSlice>> {
        let seg = self
            .segments
            .get(seg_idx)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "segment index OOB"))?;
        let seg_len = seg.len as usize;
        let Some(&ci) = self.col_index.get(col_name) else {
            return Ok(None);
        };
        let meta = &seg.cols[ci];
        match meta.kind {
            DiskColKind::AllNull => return Ok(Some(RawTextSlice::all_null(seg_len))),
            DiskColKind::Text => {}
            _ => return Ok(None),
        }
        let file = File::open(&seg.path)?;
        let raw = read_slice_at(&file, meta.offset, meta.byte_len as usize)?;
        if crc32_bytes(&raw) != meta.crc32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt .jcol text slice (CRC32 mismatch)",
            ));
        }
        RawTextSlice::from_raw(raw, seg_len).map(Some)
    }

    /// Spill already-materialized `AmorphicRecord`s into `.jcol` segments.
    pub fn append_amorphic_records(
        &mut self,
        records: &[&AmorphicRecord],
        sync_data: bool,
    ) -> io::Result<usize> {
        use rayon::prelude::*;
        if records.is_empty() {
            return Ok(0);
        }
        let mut cur = 0usize;
        let mut written = 0usize;
        while cur < records.len() {
            let end = (cur + MAX_SEGMENT_ROWS).min(records.len());
            let slice = &records[cur..end];
            let seg_len = slice.len() as u32;
            let seg_first_id = slice.first().map_or(1, |r| r.id);
            let seg_max_id = slice.iter().map(|r| r.id).max().unwrap_or(seg_first_id) + 1;
            let col_order = &self.col_order;
            let encoded: Vec<EncodedCol> = col_order
                .par_iter()
                .map(|cn| encode_records_col(slice, cn))
                .collect();
            let seq = self.next_seq;
            self.next_seq += 1;
            let (_fname, meta) = write_segment_file(
                &self.dir,
                seq,
                seg_first_id,
                seg_len,
                col_order,
                encoded,
                sync_data,
            )?;
            self.total_rows += seg_len as usize;
            self.max_record_id = self.max_record_id.max(seg_max_id);
            self.segments.push(meta);
            written += seg_len as usize;
            cur = end;
        }
        self.save_manifest(sync_data)?;
        Ok(written)
    }

    /// Infer global non-null column kind across all segments (defaults to `I8` if all segments are null).
    pub fn inferred_col_kind(&self, col_idx: usize) -> DiskColKind {
        let mut saw_int = false;
        let mut saw_float = false;
        let mut saw_text = false;
        for seg in &self.segments {
            if let Some(m) = seg.cols.get(col_idx) {
                match m.kind {
                    DiskColKind::AllNull => {}
                    DiskColKind::I8 | DiskColKind::I16 | DiskColKind::I32 | DiskColKind::I64 => {
                        saw_int = true;
                    }
                    DiskColKind::F64 => saw_float = true,
                    DiskColKind::Text => saw_text = true,
                    DiskColKind::Other => return DiskColKind::Other,
                }
            }
        }
        if saw_text && (saw_int || saw_float) {
            DiskColKind::Other
        } else if saw_text {
            DiskColKind::Text
        } else if saw_float {
            DiskColKind::F64
        } else {
            DiskColKind::I64
        }
    }

    /// Global `(kind, total_nulls, min_i64, max_i64, min_f64, max_f64)` across all segments.
    pub fn global_col_summary(
        &self,
        col_name: &str,
    ) -> Option<(
        DiskColKind,
        u64,
        Option<i64>,
        Option<i64>,
        Option<f64>,
        Option<f64>,
    )> {
        let &ci = self.col_index.get(col_name)?;
        let kind = self.inferred_col_kind(ci);
        let mut total_nulls = 0u64;
        let (mut min_i, mut max_i): (Option<i64>, Option<i64>) = (None, None);
        let (mut min_f, mut max_f): (Option<f64>, Option<f64>) = (None, None);
        let mut saw_nan = false;

        for seg in &self.segments {
            let m = &seg.cols[ci];
            total_nulls += m.null_count as u64;
            if let (Some(lo), Some(hi)) = (m.min_int(seg.len), m.max_int(seg.len)) {
                min_i = Some(min_i.map_or(lo, |x| x.min(lo)));
                max_i = Some(max_i.map_or(hi, |x| x.max(hi)));
            } else if m.kind == DiskColKind::F64 && m.null_count < seg.len {
                match (m.min_float(seg.len), m.max_float(seg.len)) {
                    (Some(lo), Some(hi)) => {
                        min_f = Some(min_f.map_or(lo, |x| x.min(lo)));
                        max_f = Some(max_f.map_or(hi, |x| x.max(hi)));
                    }
                    _ => saw_nan = true,
                }
            }
        }
        if saw_nan {
            min_f = None;
            max_f = None;
        }
        Some((kind, total_nulls, min_i, max_i, min_f, max_f))
    }

    /// Build a zero-row schema prototype `TableColumns` carrying global column types
    /// and global min/max ZoneMaps so `columnar_exec` can validate expressions and
    /// coerce ISO date literals before reading any segment bytes from disk.
    pub fn proto_table_columns(&self) -> TableColumns {
        let mut cols = HashMap::with_capacity(self.col_order.len());
        let mut int_mm = HashMap::new();
        let mut float_mm = HashMap::new();

        for (ci, name) in self.col_order.iter().enumerate() {
            let kind = self.inferred_col_kind(ci);
            match kind {
                DiskColKind::Text => {
                    cols.insert(name.clone(), Col::Text(Vec::new(), Vec::new()));
                }
                DiskColKind::F64 => {
                    cols.insert(name.clone(), Col::Float(Vec::new(), Vec::new()));
                    if let Some((_, _, _, _, Some(lo), Some(hi))) = self.global_col_summary(name) {
                        float_mm.insert(name.clone(), (lo, hi));
                    }
                }
                DiskColKind::Other => {
                    cols.insert(name.clone(), Col::Other);
                }
                _ => {
                    cols.insert(name.clone(), Col::Int(Ints::I8(Vec::new()), Vec::new()));
                    if let Some((_, _, Some(lo), Some(hi), _, _)) = self.global_col_summary(name) {
                        int_mm.insert(name.clone(), (lo, hi));
                    }
                }
            }
        }
        TableColumns::from_proto_cols(self.col_order.clone(), cols, int_mm, float_mm)
    }

    /// Read and decode **only** `cols_needed` from segment `seg_idx` into `target`
    /// (allowing Two-Phase `WHERE`-first loading to add remaining columns into the
    /// same `TableColumns` only if `>= 1` rows matched!).
    pub fn load_columns_into(
        &self,
        seg_idx: usize,
        cols_needed: &[&str],
        target: &mut TableColumns,
    ) -> io::Result<()> {
        self.load_columns_into_selected(seg_idx, cols_needed, None, target)
    }

    /// Read and decode `cols_needed` from segment `seg_idx` into `target`, sparsely
    /// decoding only the dictionary entries referenced by `sel` when `sel` is small!
    pub fn load_columns_into_selected(
        &self,
        seg_idx: usize,
        cols_needed: &[&str],
        sel: Option<&[u32]>,
        target: &mut TableColumns,
    ) -> io::Result<()> {
        let seg = self
            .segments
            .get(seg_idx)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "segment index OOB"))?;
        let seg_len = seg.len as usize;
        if cols_needed.is_empty() {
            return Ok(());
        }
        let file = File::open(&seg.path)?;
        let mut decoded = Vec::with_capacity(cols_needed.len());
        for &cn in cols_needed {
            if target.cols.contains_key(cn) {
                continue;
            }
            let Some(&ci) = self.col_index.get(cn) else {
                continue;
            };
            let meta = &seg.cols[ci];
            let inferred = self.inferred_col_kind(ci);
            let raw = if meta.byte_len > 0 {
                read_slice_at(&file, meta.offset, meta.byte_len as usize)?
            } else {
                Vec::new()
            };
            let mut col = decode_col_payload_selected(meta, seg_len, &raw, inferred, sel)?;
            // Widen segment int to f64 if a later segment widened the column globally to F64
            if inferred == DiskColKind::F64
                && let Col::Int(ints, nulls) = col
            {
                let fvals: Vec<f64> = (0..seg_len).map(|i| ints.at(i) as f64).collect();
                col = Col::Float(fvals, nulls);
            }
            decoded.push((cn.to_string(), col));
        }
        target.insert_decoded_columns(seg_len, decoded);
        Ok(())
    }

    pub fn load_segment_columns(
        &self,
        seg_idx: usize,
        cols_needed: &[&str],
    ) -> io::Result<TableColumns> {
        self.load_segment_columns_selected(seg_idx, cols_needed, None)
    }

    pub fn load_segment_columns_selected(
        &self,
        seg_idx: usize,
        cols_needed: &[&str],
        sel: Option<&[u32]>,
    ) -> io::Result<TableColumns> {
        let seg = self
            .segments
            .get(seg_idx)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "segment index OOB"))?;
        let mut tc = TableColumns::empty_with_len(seg.len as usize, self.col_order.clone());
        self.load_columns_into_selected(seg_idx, cols_needed, sel, &mut tc)?;
        Ok(tc)
    }

    /// Fallback row scan for non-vectorized queries on segmented tables.
    pub fn scan_rows(&self, columns: &[String]) -> io::Result<Vec<RowData>> {
        let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let mut rows = Vec::with_capacity(self.total_rows.min(131_072));
        for si in 0..self.segments.len() {
            let tc = self.load_segment_columns(si, &col_refs)?;
            for ri in 0..tc.len {
                let mut vals = Vec::with_capacity(columns.len());
                for cn in columns {
                    let v = match tc.cols.get(cn) {
                        None | Some(Col::Other) => AstValue::Null,
                        Some(Col::Int(ints, nulls)) => {
                            if !nulls.is_empty() && nulls[ri] {
                                AstValue::Null
                            } else {
                                AstValue::Int(ints.at(ri))
                            }
                        }
                        Some(Col::Float(vs, nulls)) => {
                            if !nulls.is_empty() && nulls[ri] {
                                AstValue::Null
                            } else {
                                AstValue::Float(vs[ri])
                            }
                        }
                        Some(Col::Text(codes, dict)) => {
                            let c = codes[ri];
                            if c == NULL_CODE_U32 {
                                AstValue::Null
                            } else {
                                AstValue::String(dict[c as usize].to_string())
                            }
                        }
                    };
                    vals.push(v);
                }
            rows.push(RowData::new(columns.to_vec(), vals));
            }
        }
        Ok(rows)
    }
}

/// Zero-`Arc<str>` view over a single `.jcol` `Text` column slice (`dict` offsets + UTF-8 arena + packed codes).
pub struct RawTextSlice {
    pub seg_len: usize,
    pub dict_len: usize,
    pub code_width: u8,
    pub offs: Vec<u32>,
    pub raw: Vec<u8>,
    pub arena_off: usize,
    pub codes_off: usize,
}

impl RawTextSlice {
    pub fn all_null(seg_len: usize) -> Self {
        Self {
            seg_len,
            dict_len: 0,
            code_width: 0,
            offs: vec![0],
            raw: Vec::new(),
            arena_off: 0,
            codes_off: 0,
        }
    }

    pub fn from_raw(raw: Vec<u8>, seg_len: usize) -> io::Result<Self> {
        if raw.len() < 9 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short text header",
            ));
        }
        let dict_len = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
        let code_width = raw[4];
        let offs_start = 5usize;
        let offs_end = offs_start + (dict_len + 1) * 4;
        if raw.len() < offs_end {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short text offsets",
            ));
        }
        let mut offs = Vec::with_capacity(dict_len + 1);
        for ch in raw[offs_start..offs_end].chunks_exact(4) {
            offs.push(u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
        }
        let arena_len = *offs.last().unwrap_or(&0) as usize;
        let arena_end = offs_end + arena_len;
        let need_codes = seg_len * (code_width as usize);
        if raw.len() < arena_end + need_codes {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short text arena or codes",
            ));
        }
        Ok(Self {
            seg_len,
            dict_len,
            code_width,
            offs,
            raw,
            arena_off: offs_end,
            codes_off: arena_end,
        })
    }

    #[inline]
    pub fn dict_bytes(&self, ci: usize) -> &[u8] {
        let a = self.arena_off + self.offs[ci] as usize;
        let b = self.arena_off + self.offs[ci + 1] as usize;
        &self.raw[a..b]
    }

    #[inline]
    pub fn code_at(&self, ri: usize) -> Option<u32> {
        match self.code_width {
            0 => None,
            1 => {
                let b = self.raw[self.codes_off + ri];
                (b != 0xFF).then_some(b as u32)
            }
            2 => {
                let o = self.codes_off + ri * 2;
                let u = u16::from_le_bytes([self.raw[o], self.raw[o + 1]]);
                (u != 0xFFFF).then_some(u as u32)
            }
            _ => {
                let o = self.codes_off + ri * 4;
                let u = u32::from_le_bytes([
                    self.raw[o],
                    self.raw[o + 1],
                    self.raw[o + 2],
                    self.raw[o + 3],
                ]);
                (u != NULL_CODE_U32).then_some(u)
            }
        }
    }
}
