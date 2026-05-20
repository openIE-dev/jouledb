//! Run-length encoding — byte-level RLE and PackBits variant.
//!
//! Implements run-length encoding for data with repeated byte sequences,
//! including the PackBits format (TIFF/Photoshop). Replaces JavaScript
//! RLE libraries with a pure Rust implementation.

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during RLE operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RleError {
    #[error("unexpected end of encoded data")]
    UnexpectedEof,
    #[error("invalid run length: {0}")]
    InvalidRunLength(usize),
    #[error("element size must be non-zero")]
    ZeroElementSize,
    #[error("data length {0} is not a multiple of element size {1}")]
    UnalignedData(usize, usize),
}

// ── Compression Statistics ──────────────────────────────────────────

/// Statistics from an RLE encode operation.
#[derive(Debug, Clone, PartialEq)]
pub struct RleStats {
    /// Original data size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes.
    pub compressed_size: usize,
    /// Number of run segments in the output.
    pub run_count: usize,
    /// Number of literal segments in the output.
    pub literal_count: usize,
    /// Compression ratio (compressed / original). Less than 1.0 means savings.
    pub ratio: f64,
}

// ── Basic Byte-Level RLE ────────────────────────────────────────────
//
// Format: [count, byte] pairs. Count is 1-based (1..=255).
// For runs longer than 255, multiple pairs are emitted.

/// Encode bytes using simple RLE: [count, byte] pairs.
pub fn rle_encode(data: &[u8]) -> Vec<u8> {
    rle_encode_min_run(data, 1)
}

/// Encode bytes using RLE with a configurable minimum run length.
/// Runs shorter than `min_run` are emitted as individual [1, byte] pairs.
pub fn rle_encode_min_run(data: &[u8], min_run: usize) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let min_run = min_run.max(1);
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let byte = data[i];
        let mut run_len = 1usize;
        while i + run_len < data.len() && data[i + run_len] == byte {
            run_len += 1;
        }

        if run_len >= min_run {
            // Emit as run(s).
            let mut remaining = run_len;
            while remaining > 0 {
                let chunk = remaining.min(255);
                result.push(chunk as u8);
                result.push(byte);
                remaining -= chunk;
            }
        } else {
            // Emit individual bytes as length-1 runs.
            for j in 0..run_len {
                result.push(1);
                result.push(data[i + j]);
            }
        }
        i += run_len;
    }
    result
}

/// Decode simple RLE encoded data.
pub fn rle_decode(data: &[u8]) -> Result<Vec<u8>, RleError> {
    if data.len() % 2 != 0 {
        return Err(RleError::UnexpectedEof);
    }
    let mut result = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let count = data[i] as usize;
        if count == 0 {
            return Err(RleError::InvalidRunLength(0));
        }
        let byte = data[i + 1];
        for _ in 0..count {
            result.push(byte);
        }
        i += 2;
    }
    Ok(result)
}

/// Get statistics for a simple RLE encode operation.
pub fn rle_stats(data: &[u8]) -> RleStats {
    let encoded = rle_encode(data);
    let compressed_size = encoded.len();
    let run_count = compressed_size / 2;
    let original_size = data.len();
    let ratio = if original_size == 0 {
        0.0
    } else {
        compressed_size as f64 / original_size as f64
    };
    RleStats {
        original_size,
        compressed_size,
        run_count,
        literal_count: 0,
        ratio,
    }
}

// ── PackBits RLE ────────────────────────────────────────────────────
//
// PackBits format (TIFF/Photoshop):
//   Header byte N:
//     0..=127 → copy next N+1 literal bytes
//     -1..-127 (129..=255 unsigned) → repeat next byte 2..128 times (-N+1)
//     -128 (128 unsigned) → no-op (skip)

/// Encode bytes using PackBits (TIFF) RLE.
pub fn packbits_encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Look for a run of identical bytes.
        let mut run_len = 1usize;
        while i + run_len < data.len()
            && data[i + run_len] == data[i]
            && run_len < 128
        {
            run_len += 1;
        }

        if run_len >= 3 {
            // Emit a run: header = -(run_len - 1) as i8, then the byte.
            // Compute in usize first to avoid signed overflow when run_len == 128.
            let header = (-((run_len as isize) - 1)) as i8;
            result.push(header as u8);
            result.push(data[i]);
            i += run_len;
        } else {
            // Collect literal bytes (non-repeating or short runs).
            let start = i;
            let mut lit_len = 0usize;
            while i + lit_len < data.len() && lit_len < 128 {
                // Check if a run of 3+ starts here.
                let mut ahead_run = 1;
                while i + lit_len + ahead_run < data.len()
                    && data[i + lit_len + ahead_run] == data[i + lit_len]
                    && ahead_run < 128
                {
                    ahead_run += 1;
                }
                if ahead_run >= 3 {
                    break;
                }
                lit_len += 1;
            }
            if lit_len == 0 {
                lit_len = 1;
            }
            // Emit literal: header = lit_len - 1, then the bytes.
            result.push((lit_len - 1) as u8);
            result.extend_from_slice(&data[start..start + lit_len]);
            i = start + lit_len;
        }
    }
    result
}

/// Decode PackBits (TIFF) RLE encoded data.
pub fn packbits_decode(data: &[u8]) -> Result<Vec<u8>, RleError> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let header = data[i] as i8;
        i += 1;

        if header >= 0 {
            // Literal: copy next (header + 1) bytes.
            let count = header as usize + 1;
            if i + count > data.len() {
                return Err(RleError::UnexpectedEof);
            }
            result.extend_from_slice(&data[i..i + count]);
            i += count;
        } else if header == -128 {
            // No-op, skip.
            continue;
        } else {
            // Run: repeat next byte (-header + 1) times.
            let count = (-header as usize) + 1;
            if i >= data.len() {
                return Err(RleError::UnexpectedEof);
            }
            let byte = data[i];
            i += 1;
            for _ in 0..count {
                result.push(byte);
            }
        }
    }
    Ok(result)
}

/// Get statistics for a PackBits encode operation.
pub fn packbits_stats(data: &[u8]) -> RleStats {
    let encoded = packbits_encode(data);
    let compressed_size = encoded.len();
    let original_size = data.len();

    // Count run vs literal segments.
    let mut run_count = 0;
    let mut literal_count = 0;
    let mut i = 0;
    while i < encoded.len() {
        let header = encoded[i] as i8;
        i += 1;
        if header >= 0 {
            literal_count += 1;
            i += header as usize + 1;
        } else if header == -128 {
            // no-op
        } else {
            run_count += 1;
            i += 1;
        }
    }

    let ratio = if original_size == 0 {
        0.0
    } else {
        compressed_size as f64 / original_size as f64
    };

    RleStats {
        original_size,
        compressed_size,
        run_count,
        literal_count,
        ratio,
    }
}

// ── Pixel-Level RLE ─────────────────────────────────────────────────
//
// RLE for multi-byte elements (e.g., RGB pixels = 3 bytes, RGBA = 4).
// Format: [count, element_bytes...] where count is 1..=255.

/// Encode data using element-based RLE with a configurable element size.
/// For example, `element_size=3` treats every 3 bytes as one pixel.
pub fn pixel_rle_encode(data: &[u8], element_size: usize) -> Result<Vec<u8>, RleError> {
    if element_size == 0 {
        return Err(RleError::ZeroElementSize);
    }
    if data.len() % element_size != 0 {
        return Err(RleError::UnalignedData(data.len(), element_size));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    let elements: Vec<&[u8]> = data.chunks_exact(element_size).collect();
    let mut i = 0;

    while i < elements.len() {
        let elem = elements[i];
        let mut run_len = 1usize;
        while i + run_len < elements.len()
            && elements[i + run_len] == elem
        {
            run_len += 1;
        }

        let mut remaining = run_len;
        while remaining > 0 {
            let chunk = remaining.min(255);
            result.push(chunk as u8);
            result.extend_from_slice(elem);
            remaining -= chunk;
        }
        i += run_len;
    }
    Ok(result)
}

/// Decode element-based RLE data.
pub fn pixel_rle_decode(data: &[u8], element_size: usize) -> Result<Vec<u8>, RleError> {
    if element_size == 0 {
        return Err(RleError::ZeroElementSize);
    }
    let record_size = 1 + element_size;
    if !data.is_empty() && data.len() % record_size != 0 {
        return Err(RleError::UnexpectedEof);
    }

    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let count = data[i] as usize;
        if count == 0 {
            return Err(RleError::InvalidRunLength(0));
        }
        i += 1;
        if i + element_size > data.len() {
            return Err(RleError::UnexpectedEof);
        }
        let elem = &data[i..i + element_size];
        for _ in 0..count {
            result.extend_from_slice(elem);
        }
        i += element_size;
    }
    Ok(result)
}

/// Get statistics for a pixel RLE encode operation.
pub fn pixel_rle_stats(data: &[u8], element_size: usize) -> Result<RleStats, RleError> {
    let encoded = pixel_rle_encode(data, element_size)?;
    let compressed_size = encoded.len();
    let original_size = data.len();
    let record_size = 1 + element_size;
    let run_count = if record_size > 0 {
        compressed_size / record_size
    } else {
        0
    };
    let ratio = if original_size == 0 {
        0.0
    } else {
        compressed_size as f64 / original_size as f64
    };
    Ok(RleStats {
        original_size,
        compressed_size,
        run_count,
        literal_count: 0,
        ratio,
    })
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic RLE ───────────────────────────────────────────────────

    #[test]
    fn rle_empty() {
        assert_eq!(rle_encode(b""), Vec::<u8>::new());
        assert_eq!(rle_decode(&[]).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn rle_single_byte() {
        let data = [42u8];
        let enc = rle_encode(&data);
        assert_eq!(enc, [1, 42]);
        assert_eq!(rle_decode(&enc).unwrap(), data);
    }

    #[test]
    fn rle_simple_run() {
        let data = [0xAA; 10];
        let enc = rle_encode(&data);
        assert_eq!(enc, [10, 0xAA]);
        assert_eq!(rle_decode(&enc).unwrap(), data);
    }

    #[test]
    fn rle_no_runs() {
        let data = [1u8, 2, 3, 4, 5];
        let enc = rle_encode(&data);
        assert_eq!(enc, [1, 1, 1, 2, 1, 3, 1, 4, 1, 5]);
        assert_eq!(rle_decode(&enc).unwrap(), data);
    }

    #[test]
    fn rle_mixed() {
        let data = [1u8, 1, 1, 2, 3, 3];
        let enc = rle_encode(&data);
        let dec = rle_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn rle_long_run() {
        let data = [0xFFu8; 300];
        let enc = rle_encode(&data);
        // Should be [255, 0xFF, 45, 0xFF]
        assert_eq!(enc, [255, 0xFF, 45, 0xFF]);
        assert_eq!(rle_decode(&enc).unwrap(), data);
    }

    #[test]
    fn rle_min_run() {
        let data = [1u8, 1, 2, 2, 2, 3];
        let enc = rle_encode_min_run(&data, 3);
        let dec = rle_decode(&enc).unwrap();
        assert_eq!(dec, data);
        // Runs of length < 3 are emitted as individual pairs.
        // 1,1 -> two [1,1] pairs; 2,2,2 -> one [3,2]; 3 -> [1,3]
        assert_eq!(enc, [1, 1, 1, 1, 3, 2, 1, 3]);
    }

    #[test]
    fn rle_decode_odd_length() {
        assert!(matches!(rle_decode(&[1, 2, 3]), Err(RleError::UnexpectedEof)));
    }

    #[test]
    fn rle_decode_zero_count() {
        assert!(matches!(rle_decode(&[0, 5]), Err(RleError::InvalidRunLength(0))));
    }

    #[test]
    fn rle_stats_check() {
        let data = [0xAA; 100];
        let stats = rle_stats(&data);
        assert_eq!(stats.original_size, 100);
        assert_eq!(stats.compressed_size, 2);
        assert!(stats.ratio < 0.1);
        assert_eq!(stats.run_count, 1);
    }

    // ── PackBits ────────────────────────────────────────────────────

    #[test]
    fn packbits_empty() {
        assert_eq!(packbits_encode(b""), Vec::<u8>::new());
        assert_eq!(packbits_decode(&[]).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn packbits_run() {
        let data = [0xAA; 5];
        let enc = packbits_encode(&data);
        let dec = packbits_decode(&enc).unwrap();
        assert_eq!(dec, data);
        // Header should be -(5-1) = -4 = 0xFC, then 0xAA
        assert_eq!(enc, [0xFC, 0xAA]);
    }

    #[test]
    fn packbits_literals() {
        let data = [1u8, 2, 3, 4, 5];
        let enc = packbits_encode(&data);
        let dec = packbits_decode(&enc).unwrap();
        assert_eq!(dec, data);
        // Header = 4 (lit_len - 1 = 5 - 1 = 4), then 5 bytes
        assert_eq!(enc, [4, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn packbits_mixed() {
        let data = [1u8, 2, 3, 0xAA, 0xAA, 0xAA, 0xAA, 4, 5];
        let enc = packbits_encode(&data);
        let dec = packbits_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn packbits_long_run() {
        let data = [0xBBu8; 128];
        let enc = packbits_encode(&data);
        let dec = packbits_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn packbits_roundtrip_random_like() {
        // Pattern with short runs and literals mixed.
        let data: Vec<u8> = (0..200u8).flat_map(|i| {
            if i % 7 == 0 {
                vec![i; 5]
            } else {
                vec![i]
            }
        }).collect();
        let enc = packbits_encode(&data);
        let dec = packbits_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn packbits_noop_header() {
        // Header 0x80 (-128) is a no-op.
        let enc = [0x80u8, 0, 42]; // noop, then literal of 1 byte
        let dec = packbits_decode(&enc).unwrap();
        assert_eq!(dec, [42]);
    }

    #[test]
    fn packbits_stats_check() {
        let data = [0xCC; 50];
        let stats = packbits_stats(&data);
        assert_eq!(stats.original_size, 50);
        assert!(stats.compressed_size < 50);
        assert_eq!(stats.run_count, 1);
    }

    #[test]
    fn packbits_decode_truncated() {
        // Literal header says 3 bytes follow, but only 2 present.
        assert!(matches!(packbits_decode(&[2, 10, 20]), Err(RleError::UnexpectedEof)));
    }

    // ── Pixel RLE ───────────────────────────────────────────────────

    #[test]
    fn pixel_rle_rgb() {
        // 3-byte elements (RGB pixels)
        let data = [
            255, 0, 0,   // red
            255, 0, 0,   // red
            255, 0, 0,   // red
            0, 255, 0,   // green
            0, 0, 255,   // blue
        ];
        let enc = pixel_rle_encode(&data, 3).unwrap();
        let dec = pixel_rle_decode(&enc, 3).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn pixel_rle_rgba() {
        // 4-byte elements (RGBA)
        let pixel = [255u8, 128, 64, 255];
        let data: Vec<u8> = pixel.iter().copied().cycle().take(4 * 20).collect();
        let enc = pixel_rle_encode(&data, 4).unwrap();
        let dec = pixel_rle_decode(&enc, 4).unwrap();
        assert_eq!(dec, data);
        // Should compress well: 20 identical pixels -> [20, 255, 128, 64, 255]
        assert_eq!(enc.len(), 5);
    }

    #[test]
    fn pixel_rle_zero_element_size() {
        assert!(matches!(
            pixel_rle_encode(&[1, 2, 3], 0),
            Err(RleError::ZeroElementSize)
        ));
    }

    #[test]
    fn pixel_rle_unaligned() {
        assert!(matches!(
            pixel_rle_encode(&[1, 2, 3, 4], 3),
            Err(RleError::UnalignedData(4, 3))
        ));
    }

    #[test]
    fn pixel_rle_stats_check() {
        let pixel = [0u8, 0, 0];
        let data: Vec<u8> = pixel.iter().copied().cycle().take(3 * 100).collect();
        let stats = pixel_rle_stats(&data, 3).unwrap();
        assert_eq!(stats.original_size, 300);
        assert!(stats.compressed_size < 300);
        assert!(stats.ratio < 1.0);
    }

    #[test]
    fn pixel_rle_single_element() {
        let data = [10u8, 20];
        let enc = pixel_rle_encode(&data, 2).unwrap();
        let dec = pixel_rle_decode(&enc, 2).unwrap();
        assert_eq!(dec, data);
    }
}
