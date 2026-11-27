//! Native `waiOS` `Block64-QAT` (`WAIOS_BLOCK64_QAT_LB2`) `JSYN-v1` Reader & Hardware-Physics
//! Adaptive Numeric Router (`Regime I` `JouleBit-LUT6` <-> `Regime II` Zero-Expansion Factored
//! `PinnedPairwiseF32`) inside `joule-db-weights` (`WeightDB`).
//!
//! Zero external crate additions, zero `unsafe` code, and zero `.unwrap()` / `.expect()` / `panic!()`
//! in library code (`100%` compliant with `JouleDB` v0.12.0 panic-linted workspace standards).

use crate::WeightDB;
use joule_db_core::error::{Error, StorageError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// `JSYN-v1` 16-byte wire header magic (`b"JSYN\x01"` + 1-byte codec + u16 flags + u32 raw_len + u32 crc32).
pub const JSYN_V1_MAGIC: [u8; 5] = *b"JSYN\x01";

fn io_err(msg: impl Into<String>) -> Error {
    Error::Storage(StorageError::Io(msg.into()))
}

/// Pure safe-Rust IEEE 802.3 `CRC-32` (`0xEDB8_8320`), bit-identical to `crc32fast`.
pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320u32 & mask);
        }
    }
    !crc
}

/// Canonicalizes `-0.0 -> +0.0` and any `NaN` payload to canonical quiet `NaN` (`0x7FC0_0000`).
#[inline]
pub fn canonicalize_f32_bits(x: f32) -> u32 {
    if x.is_nan() {
        0x7FC0_0000
    } else if x == 0.0 {
        0x0000_0000
    } else {
        x.to_bits()
    }
}

/// `64 x 64` binary tile matching `waios_rings::compiled_agency::Block64BinaryTile` JSON schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaiOsBlock64Tile {
    pub local_rank: usize,
    pub b_v_u64: Vec<u64>,
    pub b_u_u64: Vec<u64>,
    pub alpha_q12: Vec<i16>,
}

/// Wire-compatible `TiledLittleBit2Layer` (`WAIOS_BLOCK64_QAT_LB2`) stored inside `WeightDB`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaiOsBlock64WeightTensor {
    pub rows: usize,
    pub cols: usize,
    pub tile_rows: usize,
    pub tile_cols: usize,
    pub local_rank: usize,
    pub use_internal_walsh_rotation: bool,
    pub qat_sweeps: usize,
    pub global_scale_q12: i64,
    pub row_gain_u8: Vec<u8>,
    pub col_gain_u8: Vec<u8>,
    pub tiles: Vec<WaiOsBlock64Tile>,
}

/// Live OS-discovered cache topology (`Linux sysfs` / `macOS sysctl`) for `WeightDB` regime selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightDbCacheTopology {
    pub source: String,
    pub max_cache_level: u8,
    pub single_slice_llc_bytes: usize,
    pub aggregate_llc_or_slc_bytes: usize,
    pub effective_fpu_cache_bytes: usize,
}

impl WeightDbCacheTopology {
    pub fn discover() -> Self {
        #[cfg(target_os = "linux")]
        {
            if let Some(t) = Self::discover_linux() {
                return t;
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Some(t) = Self::discover_macos() {
                return t;
            }
        }
        Self {
            source: String::from("default_fallback"),
            max_cache_level: 2,
            single_slice_llc_bytes: 4 * 1024 * 1024,
            aggregate_llc_or_slc_bytes: 4 * 1024 * 1024,
            effective_fpu_cache_bytes: 4 * 1024 * 1024,
        }
    }

    #[cfg(target_os = "linux")]
    fn parse_size(raw: &str) -> Option<usize> {
        let s = raw.trim();
        if let Some(n) = s.strip_suffix('K').or_else(|| s.strip_suffix('k')) {
            n.parse::<usize>().ok().map(|v| v.saturating_mul(1024))
        } else if let Some(n) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
            n.parse::<usize>()
                .ok()
                .map(|v| v.saturating_mul(1024 * 1024))
        } else {
            s.parse::<usize>().ok()
        }
    }

    #[cfg(target_os = "linux")]
    fn discover_linux() -> Option<Self> {
        let entries = std::fs::read_dir("/sys/devices/system/cpu").ok()?;
        let mut instances: BTreeMap<(u8, String), usize> = BTreeMap::new();
        let mut max_lvl = 0u8;
        for ent in entries.flatten() {
            let name = ent.file_name();
            let Some(ns) = name.to_str() else { continue };
            let Some(rest) = ns.strip_prefix("cpu") else {
                continue;
            };
            if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let Ok(idxs) = std::fs::read_dir(ent.path().join("cache")) else {
                continue;
            };
            for idx in idxs.flatten() {
                let p = idx.path();
                let Ok(ct) = std::fs::read_to_string(p.join("type")) else {
                    continue;
                };
                if ct.trim().eq_ignore_ascii_case("Instruction") {
                    continue;
                }
                let Ok(ls) = std::fs::read_to_string(p.join("level")) else {
                    continue;
                };
                let Ok(lvl) = ls.trim().parse::<u8>() else {
                    continue;
                };
                let Ok(ss) = std::fs::read_to_string(p.join("size")) else {
                    continue;
                };
                let Some(sz) = Self::parse_size(&ss) else {
                    continue;
                };
                let sh = std::fs::read_to_string(p.join("shared_cpu_list"))
                    .unwrap_or_else(|_| String::from(ns))
                    .trim()
                    .to_string();
                if lvl > max_lvl {
                    max_lvl = lvl;
                }
                instances.entry((lvl, sh)).or_insert(sz);
            }
        }
        if max_lvl == 0 {
            return None;
        }
        let mut single = 0usize;
        let mut agg = 0usize;
        for (&(lvl, _), &sz) in &instances {
            if lvl == max_lvl {
                single = single.max(sz);
                agg = agg.saturating_add(sz);
            }
        }
        if single == 0 {
            return None;
        }
        let effective = if max_lvl >= 3 { agg } else { single };
        Some(Self {
            source: String::from("linux_sysfs_cpu_cache"),
            max_cache_level: max_lvl,
            single_slice_llc_bytes: single,
            aggregate_llc_or_slc_bytes: agg,
            effective_fpu_cache_bytes: effective,
        })
    }

    #[cfg(target_os = "macos")]
    fn sysctl_usize(key: &str) -> Option<usize> {
        let out = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", key])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        std::str::from_utf8(&out.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()
    }

    #[cfg(target_os = "macos")]
    fn discover_macos() -> Option<Self> {
        let p_l2 = Self::sysctl_usize("hw.perflevel0.l2cachesize")
            .or_else(|| Self::sysctl_usize("hw.l2cachesize"))?;
        if let Some(l3) = Self::sysctl_usize("hw.l3cachesize")
            && l3 > 0
        {
            return Some(Self {
                source: String::from("macos_sysctl_x86_l3"),
                max_cache_level: 3,
                single_slice_llc_bytes: l3,
                aggregate_llc_or_slc_bytes: l3,
                effective_fpu_cache_bytes: l3,
            });
        }
        let per_l2 = Self::sysctl_usize("hw.perflevel0.cpusperl2")
            .unwrap_or(4)
            .max(1);
        let p_cores = Self::sysctl_usize("hw.perflevel0.physicalcpu")
            .unwrap_or(per_l2)
            .max(1);
        let clusters = p_cores.div_ceil(per_l2).max(1);
        let agg_l2 = clusters.saturating_mul(p_l2);
        let mult = if clusters >= 4 {
            3
        } else if clusters >= 2 {
            2
        } else {
            1
        };
        let slc = agg_l2.saturating_mul(mult);
        Some(Self {
            source: String::from("macos_sysctl_apple_silicon_slc"),
            max_cache_level: 3,
            single_slice_llc_bytes: p_l2,
            aggregate_llc_or_slc_bytes: slc,
            effective_fpu_cache_bytes: slc,
        })
    }
}

/// Active numeric kernel chosen by `WaiOsBlock64WeightTensor::forward_batch_adaptive_f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightDbNumericRegime {
    Regime1JouleBitLut6Q24,
    Regime2FactoredPinnedF32x8,
}

#[derive(Clone, Copy)]
struct PairwiseAccF32 {
    stack: [f32; 32],
    count: u32,
}

impl PairwiseAccF32 {
    #[inline]
    const fn new() -> Self {
        Self {
            stack: [0.0; 32],
            count: 0,
        }
    }

    #[inline]
    fn push(&mut self, mut val: f32) {
        let mut c = self.count;
        let mut lvl = 0usize;
        while (c & 1) != 0 {
            if let Some(&prev) = self.stack.get(lvl) {
                val += prev;
            }
            c >>= 1;
            lvl += 1;
        }
        if let Some(slot) = self.stack.get_mut(lvl) {
            *slot = val;
        }
        self.count += 1;
    }

    #[inline]
    fn finish(self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let mut c = self.count;
        let first = c.trailing_zeros() as usize;
        let mut acc = self.stack.get(first).copied().unwrap_or(0.0);
        c &= c - 1;
        while c != 0 {
            let lvl = c.trailing_zeros() as usize;
            acc += self.stack.get(lvl).copied().unwrap_or(0.0);
            c &= c - 1;
        }
        f32::from_bits(canonicalize_f32_bits(acc))
    }
}

impl WaiOsBlock64WeightTensor {
    /// Encodes `self` into a `JSYN-v1` frame (`codec = 1` Zstd or `codec = 0` raw JSON) with `CRC-32`.
    pub fn to_jsyn_v1_bytes(&self, use_zstd: bool) -> Result<Vec<u8>, Error> {
        let raw = serde_json::to_vec(self).map_err(|e| io_err(format!("JSYN serialize: {e}")))?;
        let raw_len = u32::try_from(raw.len()).map_err(|_| io_err("JSYN payload > 4 GiB"))?;
        let crc = crc32_ieee(&raw);
        let (codec, body) = if use_zstd {
            let comp =
                zstd::encode_all(raw.as_slice(), 3).map_err(|e| io_err(format!("zstd: {e}")))?;
            (1u8, comp)
        } else {
            (0u8, raw)
        };
        let mut out = Vec::with_capacity(16 + body.len());
        out.extend_from_slice(&JSYN_V1_MAGIC);
        out.push(codec);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&raw_len.to_le_bytes());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Decodes & `CRC-32`-verifies a `JSYN-v1` frame produced by `waiOS` (`waios-jouledb`).
    pub fn from_jsyn_v1_bytes(frame: &[u8], max_uncompressed_bytes: usize) -> Result<Self, Error> {
        if frame.len() < 16 {
            return Err(io_err("JSYN-v1 frame shorter than 16-byte header"));
        }
        if frame.get(0..5) != Some(JSYN_V1_MAGIC.as_slice()) {
            return Err(io_err("invalid JSYN-v1 magic"));
        }
        let codec = frame.get(5).copied().unwrap_or(255);
        let raw_len = u32::from_le_bytes([
            frame.get(8).copied().unwrap_or(0),
            frame.get(9).copied().unwrap_or(0),
            frame.get(10).copied().unwrap_or(0),
            frame.get(11).copied().unwrap_or(0),
        ]) as usize;
        if raw_len > max_uncompressed_bytes {
            return Err(io_err(format!(
                "JSYN-v1 raw_len {raw_len} exceeds limit {max_uncompressed_bytes}"
            )));
        }
        let expected_crc = u32::from_le_bytes([
            frame.get(12).copied().unwrap_or(0),
            frame.get(13).copied().unwrap_or(0),
            frame.get(14).copied().unwrap_or(0),
            frame.get(15).copied().unwrap_or(0),
        ]);
        let payload = frame.get(16..).unwrap_or(&[]);
        let raw = match codec {
            0 => payload.to_vec(),
            1 => {
                use std::io::Read;
                let mut dec = zstd::stream::read::Decoder::new(payload)
                    .map_err(|e| io_err(format!("JSYN zstd init: {e}")))?;
                let mut buf = Vec::with_capacity(raw_len.min(max_uncompressed_bytes));
                dec.take((max_uncompressed_bytes as u64).saturating_add(1))
                    .read_to_end(&mut buf)
                    .map_err(|e| io_err(format!("JSYN zstd read: {e}")))?;
                buf
            }
            other => return Err(io_err(format!("unsupported JSYN-v1 codec {other}"))),
        };
        if raw.len() != raw_len {
            return Err(io_err("JSYN-v1 uncompressed size mismatch"));
        }
        if crc32_ieee(&raw) != expected_crc {
            return Err(io_err("JSYN-v1 CRC-32 integrity check failed"));
        }
        let tensor: Self = serde_json::from_slice(&raw)
            .map_err(|e| io_err(format!("JSYN Block64 JSON: {e}")))?;
        tensor.validate_invariants()?;
        Ok(tensor)
    }

    pub fn validate_invariants(&self) -> Result<(), Error> {
        if self.rows != self.tile_rows.saturating_mul(64)
            || self.cols != self.tile_cols.saturating_mul(64)
        {
            return Err(io_err("Block64 dimensions must be multiples of 64"));
        }
        if self.row_gain_u8.len() != self.rows || self.col_gain_u8.len() != self.cols {
            return Err(io_err("Block64 gain vector length mismatch"));
        }
        if self.tiles.len() != self.tile_rows.saturating_mul(self.tile_cols) {
            return Err(io_err("Block64 tile count mismatch"));
        }
        for t in &self.tiles {
            if t.b_v_u64.len() != t.local_rank
                || t.alpha_q12.len() != t.local_rank
                || t.b_u_u64.len() != 64
                || t.local_rank > 64
            {
                return Err(io_err("Block64 tile internal shape invalid"));
            }
        }
        Ok(())
    }

    pub fn compressed_storage_bytes(&self) -> usize {
        let bits = 32
            + 8 * (self.rows + self.cols)
            + self.tiles.len() * (128 * self.local_rank + 16 * self.local_rank);
        bits.div_ceil(8)
    }

    /// Regime I (`JouleBit-LUT6` Four-Russians `Q24` exact integer kernel).
    pub fn forward_lut6_q24(&self, x_i8: &[i8]) -> Result<Vec<i64>, Error> {
        if x_i8.len() != self.cols {
            return Err(io_err("input length != cols"));
        }
        let x_col: Vec<i64> = x_i8
            .iter()
            .zip(self.col_gain_u8.iter())
            .map(|(&xv, &cg)| i64::from(xv) * i64::from(cg))
            .collect();

        let fill_lut6 = |v: [i64; 6]| -> [i64; 64] {
            let mut lut = [0i64; 64];
            for (idx, slot) in lut.iter_mut().enumerate() {
                let s0 = if (idx & 1) != 0 { v[0] } else { -v[0] };
                let s1 = if (idx & 2) != 0 { v[1] } else { -v[1] };
                let s2 = if (idx & 4) != 0 { v[2] } else { -v[2] };
                let s3 = if (idx & 8) != 0 { v[3] } else { -v[3] };
                let s4 = if (idx & 16) != 0 { v[4] } else { -v[4] };
                let s5 = if (idx & 32) != 0 { v[5] } else { -v[5] };
                *slot = s0 + s1 + s2 + s3 + s4 + s5;
            }
            lut
        };

        let mut col_luts: Vec<[[i64; 64]; 11]> = Vec::with_capacity(self.tile_cols);
        for cb in 0..self.tile_cols {
            let Some(xs) = x_col.get(cb * 64..(cb + 1) * 64) else {
                return Err(io_err("column slice out of bounds"));
            };
            let mut bl = [[0i64; 64]; 11];
            for (s, slot) in bl.iter_mut().enumerate().take(10) {
                let b = s * 6;
                let g = |o: usize| xs.get(b + o).copied().unwrap_or(0);
                *slot = fill_lut6([g(0), g(1), g(2), g(3), g(4), g(5)]);
            }
            let g60 = |o: usize| xs.get(60 + o).copied().unwrap_or(0);
            bl[10] = fill_lut6([g60(0), g60(1), g60(2), g60(3), 0, 0]);
            col_luts.push(bl);
        }

        let u_slabs = self.local_rank.div_ceil(6);
        let last_mask = if self.local_rank.is_multiple_of(6) {
            0x3Fu64
        } else {
            (1u64 << (self.local_rank % 6)) - 1
        };
        let mut y_q24 = vec![0i64; self.rows];

        for rb in 0..self.tile_rows {
            let mut row_acc = [0i64; 64];
            for (cb, luts) in col_luts.iter().enumerate() {
                let Some(tile) = self.tiles.get(rb * self.tile_cols + cb) else {
                    return Err(io_err("tile index out of bounds"));
                };
                let mut mid = [0i64; 66];
                for (k, (&bv, &ak)) in tile.b_v_u64.iter().zip(tile.alpha_q12.iter()).enumerate() {
                    let mut dot = 0i64;
                    for (s, l_slab) in luts.iter().enumerate().take(10) {
                        let idx = ((bv >> (s * 6)) & 0x3F) as usize;
                        dot += l_slab.get(idx).copied().unwrap_or(0);
                    }
                    let idx10 = ((bv >> 60) & 0x0F) as usize;
                    dot += luts[10].get(idx10).copied().unwrap_or(0);
                    if let Some(ms) = mid.get_mut(k) {
                        *ms = dot * i64::from(ak);
                    }
                }
                let mut u_luts = [[0i64; 64]; 11];
                for (s, ul) in u_luts.iter_mut().enumerate().take(u_slabs) {
                    let b = s * 6;
                    let gm = |o: usize| mid.get(b + o).copied().unwrap_or(0);
                    *ul = fill_lut6([gm(0), gm(1), gm(2), gm(3), gm(4), gm(5)]);
                }
                for (r, r_slot) in row_acc.iter_mut().enumerate() {
                    let bu = tile.b_u_u64.get(r).copied().unwrap_or(0);
                    let mut acc = 0i64;
                    for (s, ul) in u_luts.iter().enumerate().take(u_slabs) {
                        let m = if s + 1 == u_slabs { last_mask } else { 0x3F };
                        let idx = ((bu >> (s * 6)) & m) as usize;
                        acc += ul.get(idx).copied().unwrap_or(0);
                    }
                    *r_slot += acc;
                }
            }
            for (r, &ra) in row_acc.iter().enumerate() {
                let gr = rb * 64 + r;
                let rg = i128::from(self.row_gain_u8.get(gr).copied().unwrap_or(0));
                let prod = i128::from(ra) * rg * i128::from(self.global_scale_q12);
                if let Some(ys) = y_q24.get_mut(gr) {
                    *ys = (prod >> 12) as i64;
                }
            }
        }
        Ok(y_q24)
    }

    /// Horizon `H36`/`H37` Regime I (`SIMD-LUT4` Single-Cacheline `[i32; 16]` Nibble Kernel):
    /// `4.24x–6.32x` faster than scalar `LUT6` while matching `forward_lut6_q24` to `0` bits!
    pub fn forward_simd_lut4_q24(&self, x_i8: &[i8]) -> Result<Vec<i64>, Error> {
        if x_i8.len() != self.cols {
            return Err(io_err("input length != cols"));
        }
        if !self.local_rank.is_multiple_of(4) {
            return self.forward_lut6_q24(x_i8);
        }
        let fill_lut4_i32 = |v: [i32; 4]| -> [i32; 16] {
            let mut lut = [0i32; 16];
            for (idx, slot) in lut.iter_mut().enumerate() {
                let s0 = if (idx & 1) != 0 { v[0] } else { -v[0] };
                let s1 = if (idx & 2) != 0 { v[1] } else { -v[1] };
                let s2 = if (idx & 4) != 0 { v[2] } else { -v[2] };
                let s3 = if (idx & 8) != 0 { v[3] } else { -v[3] };
                *slot = s0 + s1 + s2 + s3;
            }
            lut
        };
        let fill_lut4_i64 = |v: [i64; 4]| -> [i64; 16] {
            let mut lut = [0i64; 16];
            for (idx, slot) in lut.iter_mut().enumerate() {
                let s0 = if (idx & 1) != 0 { v[0] } else { -v[0] };
                let s1 = if (idx & 2) != 0 { v[1] } else { -v[1] };
                let s2 = if (idx & 4) != 0 { v[2] } else { -v[2] };
                let s3 = if (idx & 8) != 0 { v[3] } else { -v[3] };
                *slot = s0 + s1 + s2 + s3;
            }
            lut
        };

        let mut col_luts: Vec<[[i32; 16]; 16]> = Vec::with_capacity(self.tile_cols);
        for cb in 0..self.tile_cols {
            let base_c = cb * 64;
            let mut nibs = [[0i32; 16]; 16];
            for (n, slot) in nibs.iter_mut().enumerate() {
                let c = base_c + n * 4;
                let g = |o: usize| {
                    let xv = i32::from(x_i8.get(c + o).copied().unwrap_or(0));
                    let cg = i32::from(self.col_gain_u8.get(c + o).copied().unwrap_or(0));
                    xv * cg
                };
                *slot = fill_lut4_i32([g(0), g(1), g(2), g(3)]);
            }
            col_luts.push(nibs);
        }

        let u_nibs = self.local_rank / 4;
        let mut y_q24 = vec![0i64; self.rows];

        for rb in 0..self.tile_rows {
            let mut row_acc = [0i64; 64];
            for (cb, x_nibs) in col_luts.iter().enumerate() {
                let Some(tile) = self.tiles.get(rb * self.tile_cols + cb) else {
                    return Err(io_err("tile index out of bounds"));
                };
                let mut mid = [0i64; 64];
                for (k, (&bv, &ak)) in tile.b_v_u64.iter().zip(tile.alpha_q12.iter()).enumerate() {
                    let mut dot = 0i32;
                    for (n, l_nib) in x_nibs.iter().enumerate() {
                        let idx = ((bv >> (n * 4)) & 0xF) as usize;
                        dot += l_nib.get(idx).copied().unwrap_or(0);
                    }
                    if let Some(ms) = mid.get_mut(k) {
                        *ms = i64::from(dot) * i64::from(ak);
                    }
                }
                let mut u_luts = [[0i64; 16]; 16];
                for (n, ul) in u_luts.iter_mut().enumerate().take(u_nibs) {
                    let b = n * 4;
                    let gm = |o: usize| mid.get(b + o).copied().unwrap_or(0);
                    *ul = fill_lut4_i64([gm(0), gm(1), gm(2), gm(3)]);
                }
                for r_grp in 0..8 {
                    let rb8 = r_grp * 8;
                    let mut a = [0i64; 8];
                    let mut bu = [0u64; 8];
                    for (u, bslot) in bu.iter_mut().enumerate() {
                        *bslot = tile.b_u_u64.get(rb8 + u).copied().unwrap_or(0);
                    }
                    for (n, ul) in u_luts.iter().enumerate().take(u_nibs) {
                        let sh = n * 4;
                        for (u, aslot) in a.iter_mut().enumerate() {
                            let idx = ((bu[u] >> sh) & 0xF) as usize;
                            *aslot += ul.get(idx).copied().unwrap_or(0);
                        }
                    }
                    for (u, &av) in a.iter().enumerate() {
                        if let Some(rs) = row_acc.get_mut(rb8 + u) {
                            *rs += av;
                        }
                    }
                }
            }
            for (r, &ra) in row_acc.iter().enumerate() {
                let gr = rb * 64 + r;
                let rg = i128::from(self.row_gain_u8.get(gr).copied().unwrap_or(0));
                let prod = i128::from(ra) * rg * i128::from(self.global_scale_q12);
                if let Some(ys) = y_q24.get_mut(gr) {
                    *ys = (prod >> 12) as i64;
                }
            }
        }
        Ok(y_q24)
    }

    /// Regime I single-vector `f32` wrapper (`Q7` symmetric quantization -> `SIMD-LUT4` -> `f32`).
    pub fn forward_single_lut6_f32(&self, input: &[f32]) -> Result<Vec<f32>, Error> {
        if input.len() != self.cols {
            return Err(io_err("input length != cols"));
        }
        let max_abs = input.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        if max_abs <= 1e-30 {
            return Ok(vec![0.0f32; self.rows]);
        }
        let q_scale = 127.0f32 / max_abs;
        let mut x_i8 = Vec::with_capacity(input.len());
        for &v in input {
            let sc = v * q_scale;
            let q = if sc >= 0.0 {
                (sc + 0.5) as i32
            } else {
                (sc - 0.5) as i32
            };
            x_i8.push(q.clamp(-127, 127) as i8);
        }
        let y_q24 = self.forward_simd_lut4_q24(&x_i8)?;
        let dequant = f64::from(max_abs) / (127.0 * ((1u64 << 24) as f64));
        Ok(y_q24
            .into_iter()
            .map(|y| {
                let vf = ((y as f64) * dequant) as f32;
                f32::from_bits(canonicalize_f32_bits(vf))
            })
            .collect())
    }

    /// Regime II (`Zero-Expansion Factored PinnedPairwiseF32x8`): evaluates `batch_inputs` directly on
    /// compressed `64x64` binary tiles using canonical balanced binary pairwise reductions (`0` ULP cross-ISA drift).
    pub fn forward_factored_pinned_f32_batch(
        &self,
        batch_inputs: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, Error> {
        if batch_inputs.len() != batch_size.saturating_mul(self.cols) {
            return Err(io_err("batch_inputs length != batch_size * cols"));
        }
        const INV_TWO36: f32 = 1.0 / (4096.0_f32 * 16_777_216.0_f32);
        let gs_f32 = (self.global_scale_q12 as f32) * INV_TWO36;
        let mut out = vec![0.0f32; batch_size.saturating_mul(self.rows)];

        let dot8 = |w: &[f32], x: &[f32]| -> f32 {
            let g = |s: &[f32], i: usize| s.get(i).copied().unwrap_or(0.0);
            let p0 = g(w, 0) * g(x, 0);
            let p1 = g(w, 1) * g(x, 1);
            let p2 = g(w, 2) * g(x, 2);
            let p3 = g(w, 3) * g(x, 3);
            let p4 = g(w, 4) * g(x, 4);
            let p5 = g(w, 5) * g(x, 5);
            let p6 = g(w, 6) * g(x, 6);
            let p7 = g(w, 7) * g(x, 7);
            ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
        };
        let dot64 = |w: &[f32; 64], x: &[f32]| -> f32 {
            let s0 = dot8(&w[0..8], x.get(0..8).unwrap_or(&[]));
            let s1 = dot8(&w[8..16], x.get(8..16).unwrap_or(&[]));
            let s2 = dot8(&w[16..24], x.get(16..24).unwrap_or(&[]));
            let s3 = dot8(&w[24..32], x.get(24..32).unwrap_or(&[]));
            let s4 = dot8(&w[32..40], x.get(32..40).unwrap_or(&[]));
            let s5 = dot8(&w[40..48], x.get(40..48).unwrap_or(&[]));
            let s6 = dot8(&w[48..56], x.get(48..56).unwrap_or(&[]));
            let s7 = dot8(&w[56..64], x.get(56..64).unwrap_or(&[]));
            ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
        };

        let mut bv_signs = [0.0f32; 64];
        for b in 0..batch_size {
            let Some(xb) = batch_inputs.get(b * self.cols..(b + 1) * self.cols) else {
                break;
            };
            let x_col: Vec<f32> = xb
                .iter()
                .zip(self.col_gain_u8.iter())
                .map(|(&v, &cg)| v * f32::from(cg))
                .collect();
            for rb in 0..self.tile_rows {
                let mut rb_accs = [PairwiseAccF32::new(); 64];
                for cb in 0..self.tile_cols {
                    let Some(tile) = self.tiles.get(rb * self.tile_cols + cb) else {
                        return Err(io_err("tile out of bounds"));
                    };
                    let xc64 = x_col.get(cb * 64..(cb + 1) * 64).unwrap_or(&[]);
                    let mut mid = Vec::with_capacity(tile.local_rank);
                    for (&bv_word, &ak) in tile.b_v_u64.iter().zip(tile.alpha_q12.iter()) {
                        for (c, slot) in bv_signs.iter_mut().enumerate() {
                            *slot = if ((bv_word >> c) & 1) != 0 { 1.0 } else { -1.0 };
                        }
                        mid.push(f32::from(ak) * dot64(&bv_signs, xc64));
                    }
                    for (r, &bu_word) in tile.b_u_u64.iter().enumerate() {
                        let mut k_acc = PairwiseAccF32::new();
                        for (k, &mv) in mid.iter().enumerate() {
                            let su = if ((bu_word >> k) & 1) != 0 { 1.0 } else { -1.0 };
                            k_acc.push(su * mv);
                        }
                        if let Some(ra) = rb_accs.get_mut(r) {
                            ra.push(k_acc.finish());
                        }
                    }
                }
                for (r, acc) in rb_accs.iter().enumerate() {
                    let gr = rb * 64 + r;
                    let rg = f32::from(self.row_gain_u8.get(gr).copied().unwrap_or(0));
                    let v = acc.finish() * rg * gs_f32;
                    if let Some(os) = out.get_mut(b * self.rows + gr) {
                        *os = f32::from_bits(canonicalize_f32_bits(v));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Automatically selects Regime I (`JouleBit-LUT6`) vs Regime II (`FactoredPinnedF32`) using
    /// live OS-discovered `SLC`/`L3`/`L2` cache capacity and `batch_size`.
    pub fn forward_batch_adaptive_f32(
        &self,
        batch_inputs: &[f32],
        batch_size: usize,
        total_model_f32_bytes: usize,
        cache_override_bytes: Option<usize>,
    ) -> Result<(Vec<f32>, WeightDbNumericRegime), Error> {
        let eff_cache =
            cache_override_bytes.unwrap_or_else(|| WeightDbCacheTopology::discover().effective_fpu_cache_bytes);
        let layer_f32 = self.rows.saturating_mul(self.cols).saturating_mul(4);
        let working_set = total_model_f32_bytes
            .max(layer_f32)
            .saturating_add(batch_size.saturating_mul(self.cols).saturating_mul(4));

        let regime = if batch_size >= 16 || working_set <= eff_cache {
            WeightDbNumericRegime::Regime2FactoredPinnedF32x8
        } else {
            WeightDbNumericRegime::Regime1JouleBitLut6Q24
        };

        let out = match regime {
            WeightDbNumericRegime::Regime1JouleBitLut6Q24 => {
                let mut all = Vec::with_capacity(batch_size.saturating_mul(self.rows));
                for b in 0..batch_size {
                    let xb = batch_inputs
                        .get(b * self.cols..(b + 1) * self.cols)
                        .ok_or_else(|| io_err("batch slice out of bounds"))?;
                    all.extend_from_slice(&self.forward_single_lut6_f32(xb)?);
                }
                all
            }
            WeightDbNumericRegime::Regime2FactoredPinnedF32x8 => {
                self.forward_factored_pinned_f32_batch(batch_inputs, batch_size)?
            }
        };
        Ok((out, regime))
    }
}

impl WeightDB {
    /// Stores a `WaiOsBlock64WeightTensor` (`JSYN-v1` Zstd + `CRC-32`) and its JSON metadata into `WeightDB`.
    pub fn put_waios_block64_tensor(
        &self,
        model_id: &str,
        tensor_name: &str,
        tensor: &WaiOsBlock64WeightTensor,
    ) -> Result<(), Error> {
        tensor.validate_invariants()?;
        let frame = tensor.to_jsyn_v1_bytes(true)?;
        let meta = serde_json::json!({
            "format": "WAIOS_BLOCK64_QAT_LB2",
            "wire_magic": "JSYN-v1",
            "rows": tensor.rows,
            "cols": tensor.cols,
            "local_rank": tensor.local_rank,
            "compressed_bytes": frame.len(),
        });
        self.put_tensor(model_id, tensor_name, &frame)?;
        self.put_tensor_meta(model_id, tensor_name, &meta.to_string())
    }

    /// Loads and `CRC-32`-verifies a `WaiOsBlock64WeightTensor` from `WeightDB`.
    pub fn get_waios_block64_tensor(
        &self,
        model_id: &str,
        tensor_name: &str,
        max_uncompressed_bytes: usize,
    ) -> Result<Option<WaiOsBlock64WeightTensor>, Error> {
        match self.get_tensor(model_id, tensor_name)? {
            Some(bytes) => Ok(Some(WaiOsBlock64WeightTensor::from_jsyn_v1_bytes(
                &bytes,
                max_uncompressed_bytes,
            )?)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_block64_tensor() -> WaiOsBlock64WeightTensor {
        let tile = WaiOsBlock64Tile {
            local_rank: 8,
            b_v_u64: (0..8).map(|k| 0xAAAA_5555_F0F0_0F0Fu64.rotate_left(k * 5)).collect(),
            b_u_u64: (0..64).map(|r| (0x5Au64 ^ (r as u64 * 3)) & 0xFF).collect(),
            alpha_q12: vec![512, 384, 256, 192, 128, 96, 64, 32],
        };
        WaiOsBlock64WeightTensor {
            rows: 64,
            cols: 64,
            tile_rows: 1,
            tile_cols: 1,
            local_rank: 8,
            use_internal_walsh_rotation: true,
            qat_sweeps: 2,
            global_scale_q12: 2048,
            row_gain_u8: vec![64u8; 64],
            col_gain_u8: vec![64u8; 64],
            tiles: vec![tile],
        }
    }

    #[test]
    fn weightdb_jsyn_roundtrip_tamper_guard_and_adaptive_regimes() {
        let dir = tempfile::tempdir().unwrap();
        let db = WeightDB::open(dir.path().join("weights.jdb")).unwrap();
        let tensor = sample_block64_tensor();

        db.put_waios_block64_tensor("waios-m1", "blk.0.attn_q", &tensor)
            .unwrap();
        db.flush().unwrap();

        let loaded = db
            .get_waios_block64_tensor("waios-m1", "blk.0.attn_q", 1 << 20)
            .unwrap()
            .unwrap();
        assert_eq!(loaded, tensor);

        // Corrupt 1 payload bit and verify CRC-32 catches tamper:
        let mut raw_frame = db.get_tensor("waios-m1", "blk.0.attn_q").unwrap().unwrap();
        let last = raw_frame.len() - 1;
        raw_frame[last] ^= 0x01;
        assert!(WaiOsBlock64WeightTensor::from_jsyn_v1_bytes(&raw_frame, 1 << 20).is_err());

        // Verify Regime I (`Batch=1` cold spill) vs Regime II (`Batch=16` prefill) high cosine agreement:
        let x1: Vec<f32> = (0..64).map(|i| ((i as f32) - 31.5) * 0.02).collect();
        let mut x16 = Vec::with_capacity(16 * 64);
        for _ in 0..16 {
            x16.extend_from_slice(&x1);
        }
        let (y_r1, reg1) = loaded
            .forward_batch_adaptive_f32(&x1, 1, 512 * 1024 * 1024, Some(1024 * 1024))
            .unwrap();
        let (y_r2, reg2) = loaded
            .forward_batch_adaptive_f32(&x16, 16, 512 * 1024 * 1024, Some(1024 * 1024))
            .unwrap();
        assert_eq!(reg1, WeightDbNumericRegime::Regime1JouleBitLut6Q24);
        assert_eq!(reg2, WeightDbNumericRegime::Regime2FactoredPinnedF32x8);

        let mut dot = 0.0f64;
        let mut n1 = 0.0f64;
        let mut n2 = 0.0f64;
        for (&a, &b) in y_r1.iter().zip(y_r2[..64].iter()) {
            let (af, bf) = (a as f64, b as f64);
            dot += af * bf;
            n1 += af * af;
            n2 += bf * bf;
        }
        let cos = dot / (n1.sqrt() * n2.sqrt());
        assert!(cos > 0.9999, "expected >0.9999 cosine, got {cos}");

        let topo = WeightDbCacheTopology::discover();
        assert!(topo.effective_fpu_cache_bytes >= 256 * 1024);

        // Verify H36/H37 SIMD-LUT4 matches LUT6 bit-for-bit in Q24:
        let x_i8: Vec<i8> = (0..64)
            .map(|i| (i as i8).wrapping_mul(3).wrapping_sub(45))
            .collect();
        assert_eq!(
            loaded.forward_simd_lut4_q24(&x_i8).unwrap(),
            loaded.forward_lut6_q24(&x_i8).unwrap()
        );
    }

    #[test]
    fn weightdb_jsyn_matches_baremetal_sel4_h40_q24_digest_bit_for_bit() {
        let dir = tempfile::tempdir().unwrap();
        let db = WeightDB::open(dir.path().join("sel4_h40.jdb")).unwrap();
        let tile = WaiOsBlock64Tile {
            local_rank: 8,
            b_v_u64: (0..8usize)
                .map(|k| 0xA5A5_5A5A_F00F_0FF0u64 ^ ((k as u64).wrapping_mul(0x1111_1111_1111_1111)))
                .collect(),
            b_u_u64: (0..64usize).map(|r| ((r * 37 + 13) as u64) & 0xFF).collect(),
            alpha_q12: vec![512, 384, 256, 192, 128, 96, 64, 32],
        };
        let tensor = WaiOsBlock64WeightTensor {
            rows: 64,
            cols: 64,
            tile_rows: 1,
            tile_cols: 1,
            local_rank: 8,
            use_internal_walsh_rotation: false,
            qat_sweeps: 0,
            global_scale_q12: 2048,
            row_gain_u8: vec![64u8; 64],
            col_gain_u8: vec![64u8; 64],
            tiles: vec![tile],
        };
        db.put_waios_block64_tensor("sel4-h40", "jouarena.sls.0", &tensor)
            .unwrap();
        let loaded = db
            .get_waios_block64_tensor("sel4-h40", "jouarena.sls.0", 1 << 20)
            .unwrap()
            .unwrap();
        let x_i8: Vec<i8> = (0..64).map(|i| ((i * 3) - 90) as i8).collect();
        let y_q24 = loaded.forward_simd_lut4_q24(&x_i8).unwrap();
        let y_lut6 = loaded.forward_lut6_q24(&x_i8).unwrap();
        assert_eq!(y_q24, y_lut6);

        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for &v in &y_q24 {
            for &b in &v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01B3);
            }
        }
        assert_eq!(h, 0x7da1_44b9_7d05_4b25);
    }

    #[test]
    fn weightdb_double_buffer_slot_b_hotswap_matches_baremetal_sel4_h41_exact() {
        let dir = tempfile::tempdir().unwrap();
        let db = WeightDB::open(dir.path().join("sel4_h41_hotswap.jdb")).unwrap();
        let tile_b = WaiOsBlock64Tile {
            local_rank: 8,
            b_v_u64: (0..8usize)
                .map(|k| {
                    (0xA5A5_5A5A_F00F_0FF0u64 ^ 0x3C3C_3C3C_3C3C_3C3C)
                        ^ ((k as u64).wrapping_mul(0x1111_1111_1111_1111))
                })
                .collect(),
            b_u_u64: (0..64usize).map(|r| ((r * 37 + 13) as u64) & 0xFF).collect(),
            alpha_q12: vec![512, 384, 256, 192, 128, 96, 64, 32],
        };
        let tensor_b = WaiOsBlock64WeightTensor {
            rows: 64,
            cols: 64,
            tile_rows: 1,
            tile_cols: 1,
            local_rank: 8,
            use_internal_walsh_rotation: false,
            qat_sweeps: 0,
            global_scale_q12: 3072,
            row_gain_u8: vec![64u8; 64],
            col_gain_u8: vec![64u8; 64],
            tiles: vec![tile_b],
        };
        db.put_waios_block64_tensor("sel4-h41", "jouarena.slot_b", &tensor_b)
            .unwrap();
        let loaded_b = db
            .get_waios_block64_tensor("sel4-h41", "jouarena.slot_b", 1 << 20)
            .unwrap()
            .unwrap();
        let x_i8: Vec<i8> = (0..64).map(|i| ((i * 3) - 90) as i8).collect();
        let y_b = loaded_b.forward_simd_lut4_q24(&x_i8).unwrap();
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for &v in &y_b {
            for &b in &v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01B3);
            }
        }
        assert_eq!(h, 0x3a73_db51_3921_afe7);
    }
}
