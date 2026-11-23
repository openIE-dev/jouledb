//! Vortex / DuckDB-Class Lossless Columnar Compression (`ALP` + `DeltaVarint` + `SymbolDict`)
//!
//! Eliminates legacy bloated JSON and uncompressed `f64`/string page waste for high-rate
//! sensor telemetry, robotics trajectories, financial ticks, and agent event logs in pure
//! safe Rust (zero C/C++ build dependencies):
//!
//! 1. **[`AlpCodec`] (Adaptive Lossless Floating-Point)**: Discovers the shared decimal scale
//!    factor $10^e$ across a vector of `f64` values, Frame-of-Reference (`FOR`) shifts to `min`,
//!    packs integer deltas in 1/2/4/8 bytes, and patches non-decimal outliers bit-exact!
//! 2. **[`MonotonicTimestampCodec`]**: Delta-of-delta + ZigZag LEB128 varint encoding for
//!    nanosecond `HLC` / sensor clocks (typically 1-2 bytes per 64-bit timestamp!).
//! 3. **[`SymbolDictCodec`] (`FSST`-lite)**: Compresses repetitive topic/URI/JSON-key strings
//!    via a learned 1-byte frequent-token dictionary while preserving $O(1)$ random element decode!

use crate::error::CodecError;

const ALP_POW10: [f64; 10] = [
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
    10_000_000.0,
    100_000_000.0,
    1_000_000_000.0,
];

/// Adaptive Lossless Floating-Point (`ALP`) encoder/decoder for `f64` columns.
pub struct AlpCodec;

impl AlpCodec {
    /// Compress a slice of `f64` values losslessly (bit-exact `f64::to_bits` reconstruction).
    pub fn encode(values: &[f64]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"ALP1");
        out.extend_from_slice(&(values.len() as u32).to_le_bytes());
        if values.is_empty() {
            return out;
        }

        // Sample up to 64 values to pick best decimal exponent `exp in 0..10`
        // (counting both exact IEEE-754 decimal matches and +/-1..2 ULP sensor gain/offset values)
        let step = (values.len() / 64).max(1);
        let mut best_exp = 0usize;
        let mut best_hits = 0usize;
        for (exp, &scale) in ALP_POW10.iter().enumerate() {
            let mut hits = 0usize;
            for i in (0..values.len()).step_by(step) {
                let v = values[i];
                if v.is_finite() && (v * scale).abs() < (i64::MAX as f64) * 0.25 {
                    let scaled = (v * scale).round() as i64;
                    let recon = (scaled as f64) / scale;
                    let ulp_diff = (v.to_bits() as i128) - (recon.to_bits() as i128);
                    if (-1..=2).contains(&ulp_diff) {
                        hits += 1;
                    }
                }
            }
            if hits > best_hits {
                best_hits = hits;
                best_exp = exp;
            }
        }

        let scale = ALP_POW10[best_exp];
        let mut encoded_ints = Vec::with_capacity(values.len());
        let mut ulp_crumbs = Vec::with_capacity(values.len());
        let mut any_nonzero_crumb = false;
        let mut patches: Vec<(u32, u64)> = Vec::new();
        let mut min_int = i64::MAX;
        let mut max_int = i64::MIN;

        for (idx, &v) in values.iter().enumerate() {
            if v.is_finite() && (v * scale).abs() < (i64::MAX as f64) * 0.25 {
                let s = (v * scale).round() as i64;
                let recon = (s as f64) / scale;
                let ulp_diff = (v.to_bits() as i128) - (recon.to_bits() as i128);
                if let Some(crumb) = match ulp_diff {
                    0 => Some(0u8),
                    1 => Some(1u8),
                    -1 => Some(2u8),
                    2 => Some(3u8),
                    _ => None,
                } {
                    if crumb != 0 {
                        any_nonzero_crumb = true;
                    }
                    min_int = min_int.min(s);
                    max_int = max_int.max(s);
                    encoded_ints.push(s);
                    ulp_crumbs.push(crumb);
                    continue;
                }
            }
            patches.push((idx as u32, v.to_bits()));
            encoded_ints.push(0);
            ulp_crumbs.push(0);
        }

        if min_int > max_int {
            min_int = 0;
            max_int = 0;
        }
        // Replace patched placeholder slots with `min_int` so `delta = 0`!
        for &(idx, _) in &patches {
            encoded_ints[idx as usize] = min_int;
        }

        let span = (max_int as i128 - min_int as i128).max(0) as u64;
        let width: u8 = if span <= u8::MAX as u64 {
            1
        } else if span <= u16::MAX as u64 {
            2
        } else if span <= u32::MAX as u64 {
            4
        } else {
            8
        };

        out.push(best_exp as u8);
        let flags_width = width | if any_nonzero_crumb { 0x80 } else { 0 };
        out.push(flags_width);
        out.extend_from_slice(&min_int.to_le_bytes());

        for &s in &encoded_ints {
            let delta = (s as i128 - min_int as i128) as u64;
            match width {
                1 => out.push(delta as u8),
                2 => out.extend_from_slice(&(delta as u16).to_le_bytes()),
                4 => out.extend_from_slice(&(delta as u32).to_le_bytes()),
                _ => out.extend_from_slice(&delta.to_le_bytes()),
            }
        }

        if any_nonzero_crumb {
            for chunk in ulp_crumbs.chunks(4) {
                let mut packed = 0u8;
                for (lane, &c) in chunk.iter().enumerate() {
                    packed |= (c & 0x03) << (lane * 2);
                }
                out.push(packed);
            }
        }

        out.extend_from_slice(&(patches.len() as u32).to_le_bytes());
        for (idx, raw_bits) in patches {
            out.extend_from_slice(&idx.to_le_bytes());
            out.extend_from_slice(&raw_bits.to_le_bytes());
        }
        out
    }

    /// Decode an [`AlpCodec::encode`] buffer back into bit-exact `f64` values.
    pub fn decode(bytes: &[u8]) -> Result<Vec<f64>, CodecError> {
        if bytes.len() < 8 || &bytes[0..4] != b"ALP1" {
            return Err(CodecError::InvalidFormat {
                reason: "invalid ALP1 header".into(),
            });
        }
        let mut len_b = [0u8; 4];
        len_b.copy_from_slice(&bytes[4..8]);
        let count = u32::from_le_bytes(len_b) as usize;
        if count == 0 {
            return Ok(Vec::new());
        }
        if bytes.len() < 18 {
            return Err(CodecError::UnexpectedEof {
                expected: 18,
                actual: bytes.len(),
            });
        }

        let exp = bytes[8] as usize;
        let has_crumbs = (bytes[9] & 0x80) != 0;
        let width = (bytes[9] & 0x7F) as usize;
        if exp >= ALP_POW10.len() || !matches!(width, 1 | 2 | 4 | 8) {
            return Err(CodecError::InvalidFormat {
                reason: "invalid ALP exponent or byte width".into(),
            });
        }
        let scale = ALP_POW10[exp];
        let mut min_b = [0u8; 8];
        min_b.copy_from_slice(&bytes[10..18]);
        let min_int = i64::from_le_bytes(min_b);

        let crumb_bytes = if has_crumbs { count.div_ceil(4) } else { 0 };
        let data_end = count
            .checked_mul(width)
            .and_then(|n| n.checked_add(18))
            .and_then(|n| n.checked_add(crumb_bytes))
            .ok_or_else(|| CodecError::InvalidFormat {
                reason: "ALP count/width overflow".into(),
            })?;
        let min_required = data_end.checked_add(4).ok_or_else(|| CodecError::InvalidFormat {
            reason: "ALP header size overflow".into(),
        })?;
        if bytes.len() < min_required {
            return Err(CodecError::UnexpectedEof {
                expected: min_required,
                actual: bytes.len(),
            });
        }

        let mut out = Vec::with_capacity(count);
        let mut pos = 18usize;
        for _ in 0..count {
            let delta: u64 = match width {
                1 => bytes[pos] as u64,
                2 => {
                    let mut b = [0u8; 2];
                    b.copy_from_slice(&bytes[pos..pos + 2]);
                    u16::from_le_bytes(b) as u64
                }
                4 => {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&bytes[pos..pos + 4]);
                    u32::from_le_bytes(b) as u64
                }
                _ => {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&bytes[pos..pos + 8]);
                    u64::from_le_bytes(b)
                }
            };
            pos += width;
            let val_int = (min_int as i128 + delta as i128) as i64;
            out.push((val_int as f64) / scale);
        }

        if has_crumbs {
            for (i, slot) in out.iter_mut().enumerate() {
                let packed = bytes[pos + (i / 4)];
                let crumb = (packed >> ((i % 4) * 2)) & 0x03;
                let ulp_delta: i64 = match crumb {
                    0 => 0,
                    1 => 1,
                    2 => -1,
                    _ => 2,
                };
                if ulp_delta != 0 {
                    *slot = f64::from_bits(slot.to_bits().wrapping_add_signed(ulp_delta));
                }
            }
            pos += crumb_bytes;
        }

        let mut pc_b = [0u8; 4];
        pc_b.copy_from_slice(&bytes[pos..pos + 4]);
        let patch_count = u32::from_le_bytes(pc_b) as usize;
        pos += 4;
        let patch_bytes = patch_count
            .checked_mul(12)
            .ok_or_else(|| CodecError::InvalidFormat {
                reason: "ALP patch count overflow".into(),
            })?;
        if bytes.len().saturating_sub(pos) < patch_bytes {
            return Err(CodecError::UnexpectedEof {
                expected: pos.saturating_add(patch_bytes),
                actual: bytes.len(),
            });
        }
        for _ in 0..patch_count {
            let mut idx_b = [0u8; 4];
            idx_b.copy_from_slice(&bytes[pos..pos + 4]);
            let idx = u32::from_le_bytes(idx_b) as usize;
            let mut raw_b = [0u8; 8];
            raw_b.copy_from_slice(&bytes[pos + 4..pos + 12]);
            pos += 12;
            if idx >= out.len() {
                return Err(CodecError::InvalidFormat {
                    reason: format!("ALP patch index {idx} out of bounds for len {}", out.len()),
                });
            }
            out[idx] = f64::from_bits(u64::from_le_bytes(raw_b));
        }
        Ok(out)
    }
}

/// Delta-of-delta + ZigZag LEB128 varint codec for nanosecond timestamps.
pub struct MonotonicTimestampCodec;

impl MonotonicTimestampCodec {
    fn push_varint_zigzag(out: &mut Vec<u8>, v: i128) {
        let mut zz = ((v << 1) ^ (v >> 127)) as u128;
        loop {
            let mut byte = (zz & 0x7F) as u8;
            zz >>= 7;
            if zz != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if zz == 0 {
                break;
            }
        }
    }

    fn read_varint_zigzag(bytes: &[u8], pos: &mut usize) -> Result<i128, CodecError> {
        let mut zz = 0u128;
        let mut shift = 0u32;
        loop {
            if *pos >= bytes.len() {
                return Err(CodecError::UnexpectedEof {
                    expected: *pos + 1,
                    actual: bytes.len(),
                });
            }
            let b = bytes[*pos];
            *pos += 1;
            zz |= ((b & 0x7F) as u128) << shift;
            if (b & 0x80) == 0 {
                break;
            }
            shift += 7;
            if shift >= 128 {
                return Err(CodecError::InvalidFormat {
                    reason: "varint overflow".into(),
                });
            }
        }
        Ok(((zz >> 1) as i128) ^ (-((zz & 1) as i128)))
    }

    /// Compress `timestamps` via delta-of-delta ZigZag varints (~1 byte/sample on periodic clocks!).
    pub fn encode(timestamps: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(timestamps.len() as u32).to_le_bytes());
        if timestamps.is_empty() {
            return out;
        }
        out.extend_from_slice(&timestamps[0].to_le_bytes());
        let mut prev_ts = timestamps[0] as i128;
        let mut prev_delta = 0i128;
        for &ts in &timestamps[1..] {
            let delta = (ts as i128) - prev_ts;
            let dod = delta - prev_delta;
            Self::push_varint_zigzag(&mut out, dod);
            prev_ts = ts as i128;
            prev_delta = delta;
        }
        out
    }

    /// Reconstruct exact `u64` timestamps from [`MonotonicTimestampCodec::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Vec<u64>, CodecError> {
        if bytes.len() < 4 {
            return Err(CodecError::UnexpectedEof {
                expected: 4,
                actual: bytes.len(),
            });
        }
        let mut len_b = [0u8; 4];
        len_b.copy_from_slice(&bytes[0..4]);
        let count = u32::from_le_bytes(len_b) as usize;
        if count == 0 {
            return Ok(Vec::new());
        }
        if bytes.len() < 12 {
            return Err(CodecError::UnexpectedEof {
                expected: 12,
                actual: bytes.len(),
            });
        }
        let mut first_b = [0u8; 8];
        first_b.copy_from_slice(&bytes[4..12]);
        let first = u64::from_le_bytes(first_b);
        // Each remaining timestamp consumes at least 1 varint byte; reject OOM bombs upfront!
        let remaining_varint_bytes = bytes.len() - 12;
        if count - 1 > remaining_varint_bytes {
            return Err(CodecError::UnexpectedEof {
                expected: 12usize.saturating_add(count - 1),
                actual: bytes.len(),
            });
        }
        let mut out = Vec::with_capacity(count);
        out.push(first);

        let mut pos = 12usize;
        let mut prev_ts = first as i128;
        let mut prev_delta = 0i128;
        for _ in 1..count {
            let dod = Self::read_varint_zigzag(bytes, &mut pos)?;
            let delta = prev_delta.wrapping_add(dod);
            let ts = prev_ts.wrapping_add(delta);
            out.push(ts as u64);
            prev_ts = ts;
            prev_delta = delta;
        }
        Ok(out)
    }
}

/// `FSST`-inspired 1-byte symbol table compressor for string columns with $O(1)$ random access.
pub struct SymbolDictCodec {
    symbols: Vec<Vec<u8>>,
    offsets: Vec<u32>,
    data: Vec<u8>,
}

impl SymbolDictCodec {
    /// Train up to 254 frequent 4-8 byte substring symbols and compress `strings` with random-access offsets.
    pub fn compress(strings: &[&str]) -> Self {
        use std::collections::HashMap;
        let mut freq: HashMap<&[u8], usize> = HashMap::new();
        for s in strings.iter().take(512) {
            let b = s.as_bytes();
            let mut i = 0usize;
            while i + 8 <= b.len() {
                *freq.entry(&b[i..i + 8]).or_insert(0) += 1;
                i += 4;
            }
        }
        let mut ranked: Vec<(&[u8], usize)> = freq.into_iter().filter(|&(_, c)| c >= 2).collect();
        ranked.sort_by_key(|b| std::cmp::Reverse(b.1));
        let symbols: Vec<Vec<u8>> = ranked
            .into_iter()
            .take(254)
            .map(|(s, _)| s.to_vec())
            .collect();

        let mut offsets = Vec::with_capacity(strings.len() + 1);
        let mut data = Vec::new();
        for s in strings {
            offsets.push(data.len() as u32);
            let b = s.as_bytes();
            let mut pos = 0usize;
            while pos < b.len() {
                if pos + 8 <= b.len()
                    && let Some(code) = symbols.iter().position(|sym| sym.as_slice() == &b[pos..pos + 8])
                {
                    data.push(code as u8);
                    pos += 8;
                    continue;
                }
                // Escape literal byte via 0xFF
                data.push(0xFF);
                data.push(b[pos]);
                pos += 1;
            }
        }
        offsets.push(data.len() as u32);
        Self {
            symbols,
            offsets,
            data,
        }
    }

    /// Random-access decode of string `index` in $O(|s|)$ without decompressing surrounding rows!
    pub fn get(&self, index: usize) -> Option<String> {
        if index.saturating_add(1) >= self.offsets.len() {
            return None;
        }
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        if start > end || end > self.data.len() {
            return None;
        }
        let mut out = Vec::with_capacity(end - start);
        let mut pos = start;
        while pos < end {
            let code = self.data[pos];
            pos += 1;
            if code == 0xFF {
                if pos < end {
                    out.push(self.data[pos]);
                    pos += 1;
                }
            } else if let Some(sym) = self.symbols.get(code as usize) {
                out.extend_from_slice(sym);
            }
        }
        String::from_utf8(out).ok()
    }

    /// Total compressed footprint in bytes (dictionary + offset array + symbol stream).
    pub fn compressed_byte_size(&self) -> usize {
        let dict_bytes: usize = self.symbols.iter().map(|s| s.len() + 1).sum();
        dict_bytes + self.offsets.len() * 4 + self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alp_lossless_sensor_compression_and_nan_outlier_patching() {
        let mut readings: Vec<f64> = (0..1_000)
            .map(|i| 9.80 + ((i % 25) as f64) * 0.01)
            .collect();
        readings[42] = std::f64::consts::PI; // Irrational non-decimal outlier!
        readings[777] = f64::NAN; // NaN outlier!

        let raw_bytes = readings.len() * 8;
        let compressed = AlpCodec::encode(&readings);
        assert!(
            compressed.len() * 4 < raw_bytes,
            "ALP should achieve >4x compression on sensor decimals: {} vs {}",
            compressed.len(),
            raw_bytes
        );

        let decoded = AlpCodec::decode(&compressed).unwrap();
        assert_eq!(decoded.len(), readings.len());
        for (i, (&orig, &dec)) in readings.iter().zip(decoded.iter()).enumerate() {
            if orig.is_nan() {
                assert!(dec.is_nan(), "NaN preserved at {i}");
            } else {
                assert_eq!(orig.to_bits(), dec.to_bits(), "bit-exact mismatch at {i}");
            }
        }
    }

    #[test]
    fn test_monotonic_1khz_timestamp_delta_of_delta_compression() {
        // 1,000 samples at ~1 kHz (1_000_000 ns period +/- 20 ns jitter)
        let ts: Vec<u64> = (0..1_000)
            .map(|i| 1_700_000_000_000_000_000 + i * 1_000_000 + (i % 5) * 7)
            .collect();
        let compressed = MonotonicTimestampCodec::encode(&ts);
        assert!(
            compressed.len() * 6 < ts.len() * 8,
            "Delta-of-delta should achieve >6x compression: {} vs {}",
            compressed.len(),
            ts.len() * 8
        );
        assert_eq!(MonotonicTimestampCodec::decode(&compressed).unwrap(), ts);
    }

    #[test]
    fn test_fsst_symbol_dict_random_access_topic_compression() {
        let topics: Vec<String> = (0..500)
            .map(|i| format!("jouledb://robot/humanoid_01/arm_left/joint_{:02}/torque", i % 7))
            .collect();
        let refs: Vec<&str> = topics.iter().map(String::as_str).collect();
        let raw_len: usize = refs.iter().map(|s| s.len()).sum();

        let dict = SymbolDictCodec::compress(&refs);
        assert!(
            dict.compressed_byte_size() * 2 < raw_len,
            "SymbolDict should achieve >2x compression on repetitive URIs: {} vs {}",
            dict.compressed_byte_size(),
            raw_len
        );
        assert_eq!(dict.get(0).unwrap(), topics[0]);
        assert_eq!(dict.get(253).unwrap(), topics[253]);
        assert_eq!(dict.get(499).unwrap(), topics[499]);
    }
}
