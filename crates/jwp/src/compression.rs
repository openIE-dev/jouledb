//! Per-frame compression strategies for JWP.
//!
//! Compression is the highest-impact per-frame adaptive decision:
//! CPU joules spent compressing vs. network joules saved transmitting.
//! The [`CompressionStrategy`] trait lets the [`AdaptiveCodec`] swap
//! algorithms per-frame based on payload size and energy budget.

use crate::error::JwpError;

// ── Compression identifier ───────────────────────────────────────

/// Identifies the compression algorithm used for a frame's payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum CompressionId {
    #[default]
    None = 0x00,
    Zstd = 0x01,
    Lz4 = 0x02,
}

impl CompressionId {
    /// Parse from wire byte.
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Self::None),
            0x01 => Some(Self::Zstd),
            0x02 => Some(Self::Lz4),
            _ => None,
        }
    }
}

// ── Trait ─────────────────────────────────────────────────────────

/// Trait for pluggable per-frame compression.
///
/// Implementations decide whether to compress based on payload size
/// and energy budget, enabling energy-aware compression decisions.
pub trait CompressionStrategy: Send + Sync {
    /// Compress raw payload bytes.
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, JwpError>;

    /// Decompress payload bytes. `max_output` caps the decompressed
    /// size to prevent decompression bombs.
    fn decompress(&self, data: &[u8], max_output: usize) -> Result<Vec<u8>, JwpError>;

    /// The compression identifier for this strategy.
    fn compression_id(&self) -> CompressionId;

    /// Decide whether to compress this specific payload.
    ///
    /// Returns `false` when:
    /// - Payload is too small (overhead exceeds savings)
    /// - Energy budget is tight (compression costs CPU joules)
    fn should_compress(&self, payload_len: usize, energy_budget_uwh: Option<u64>) -> bool;
}

// ── No compression ───────────────────────────────────────────────

/// Pass-through: no compression applied.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCompression;

impl CompressionStrategy for NoCompression {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, JwpError> {
        Ok(data.to_vec())
    }

    fn decompress(&self, data: &[u8], _max_output: usize) -> Result<Vec<u8>, JwpError> {
        Ok(data.to_vec())
    }

    fn compression_id(&self) -> CompressionId {
        CompressionId::None
    }

    fn should_compress(&self, _payload_len: usize, _energy_budget_uwh: Option<u64>) -> bool {
        false
    }
}

// ── Zstd ─────────────────────────────────────────────────────────

/// Zstandard compression — good ratio, moderate CPU cost.
///
/// Best for payloads > 256 bytes on non-local networks where
/// bandwidth savings outweigh CPU energy.
#[derive(Debug, Clone)]
pub struct ZstdCompression {
    /// Zstd compression level (1-22, default 3).
    pub level: i32,
    /// Minimum payload size to bother compressing (default 64).
    pub min_payload_bytes: usize,
}

impl Default for ZstdCompression {
    fn default() -> Self {
        Self {
            level: 3,
            min_payload_bytes: 64,
        }
    }
}

impl CompressionStrategy for ZstdCompression {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, JwpError> {
        zstd::encode_all(data, self.level).map_err(|e| JwpError::CompressionError(e.to_string()))
    }

    fn decompress(&self, data: &[u8], max_output: usize) -> Result<Vec<u8>, JwpError> {
        let decoder =
            zstd::Decoder::new(data).map_err(|e| JwpError::CompressionError(e.to_string()))?;

        let mut output = Vec::with_capacity(data.len().min(max_output));
        use std::io::Read;
        decoder
            .take(max_output as u64)
            .read_to_end(&mut output)
            .map_err(|e| JwpError::CompressionError(e.to_string()))?;

        Ok(output)
    }

    fn compression_id(&self) -> CompressionId {
        CompressionId::Zstd
    }

    fn should_compress(&self, payload_len: usize, energy_budget_uwh: Option<u64>) -> bool {
        if payload_len < self.min_payload_bytes {
            return false;
        }
        // If energy budget is very tight (< 10 µWh), skip compression
        // to save CPU joules.
        if let Some(budget) = energy_budget_uwh
            && budget < 10
        {
            return false;
        }
        true
    }
}

// ── LZ4 ──────────────────────────────────────────────────────────

/// LZ4 compression — fast decompression, lower ratio than Zstd.
///
/// Preferred on local networks (RTT < 5ms) where speed matters
/// more than compression ratio.
#[derive(Debug, Clone)]
pub struct Lz4Compression {
    /// Minimum payload size to bother compressing (default 64).
    pub min_payload_bytes: usize,
}

impl Default for Lz4Compression {
    fn default() -> Self {
        Self {
            min_payload_bytes: 64,
        }
    }
}

impl CompressionStrategy for Lz4Compression {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, JwpError> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    fn decompress(&self, data: &[u8], max_output: usize) -> Result<Vec<u8>, JwpError> {
        let decompressed = lz4_flex::decompress_size_prepended(data)
            .map_err(|e| JwpError::CompressionError(e.to_string()))?;

        if decompressed.len() > max_output {
            return Err(JwpError::CompressionError(format!(
                "decompressed size {} exceeds max {}",
                decompressed.len(),
                max_output
            )));
        }
        Ok(decompressed)
    }

    fn compression_id(&self) -> CompressionId {
        CompressionId::Lz4
    }

    fn should_compress(&self, payload_len: usize, energy_budget_uwh: Option<u64>) -> bool {
        if payload_len < self.min_payload_bytes {
            return false;
        }
        if let Some(budget) = energy_budget_uwh
            && budget < 10
        {
            return false;
        }
        true
    }
}

// ── Factory ──────────────────────────────────────────────────────

/// Create a boxed compression strategy from a [`CompressionId`].
pub fn strategy_for(id: CompressionId) -> Box<dyn CompressionStrategy> {
    match id {
        CompressionId::None => Box::new(NoCompression),
        CompressionId::Zstd => Box::new(ZstdCompression::default()),
        CompressionId::Lz4 => Box::new(Lz4Compression::default()),
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MAX_PAYLOAD_LEN;

    const TEST_DATA: &[u8] = b"The quick brown fox jumps over the lazy dog. \
        This is a reasonably long string that should compress well because \
        it contains repeated English language patterns and common words.";

    #[test]
    fn no_compression_passthrough() {
        let c = NoCompression;
        let compressed = c.compress(TEST_DATA).unwrap();
        assert_eq!(compressed, TEST_DATA);
        let decompressed = c.decompress(&compressed, MAX_PAYLOAD_LEN).unwrap();
        assert_eq!(decompressed, TEST_DATA);
    }

    #[test]
    fn no_compression_never_compresses() {
        let c = NoCompression;
        assert!(!c.should_compress(1_000_000, None));
        assert!(!c.should_compress(1_000_000, Some(999_999)));
    }

    #[test]
    fn zstd_roundtrip() {
        let c = ZstdCompression::default();
        let compressed = c.compress(TEST_DATA).unwrap();
        // Zstd should actually compress this data
        assert!(compressed.len() < TEST_DATA.len());
        let decompressed = c.decompress(&compressed, MAX_PAYLOAD_LEN).unwrap();
        assert_eq!(decompressed, TEST_DATA);
    }

    #[test]
    fn lz4_roundtrip() {
        let c = Lz4Compression::default();
        let compressed = c.compress(TEST_DATA).unwrap();
        let decompressed = c.decompress(&compressed, MAX_PAYLOAD_LEN).unwrap();
        assert_eq!(decompressed, TEST_DATA);
    }

    #[test]
    fn should_compress_respects_min_payload() {
        let c = ZstdCompression {
            min_payload_bytes: 64,
            ..Default::default()
        };
        assert!(!c.should_compress(32, None)); // too small
        assert!(c.should_compress(128, None)); // large enough

        let c2 = Lz4Compression {
            min_payload_bytes: 64,
        };
        assert!(!c2.should_compress(32, None));
        assert!(c2.should_compress(128, None));
    }

    #[test]
    fn should_compress_respects_energy_budget() {
        let c = ZstdCompression::default();
        assert!(c.should_compress(256, Some(1000))); // plenty of budget
        assert!(!c.should_compress(256, Some(5))); // tight budget

        let c2 = Lz4Compression::default();
        assert!(c2.should_compress(256, Some(1000)));
        assert!(!c2.should_compress(256, Some(5)));
    }

    #[test]
    fn compression_id_from_u8() {
        assert_eq!(CompressionId::from_u8(0x00), Some(CompressionId::None));
        assert_eq!(CompressionId::from_u8(0x01), Some(CompressionId::Zstd));
        assert_eq!(CompressionId::from_u8(0x02), Some(CompressionId::Lz4));
        assert_eq!(CompressionId::from_u8(0xFF), None);
    }

    #[test]
    fn strategy_for_creates_correct_type() {
        assert_eq!(
            strategy_for(CompressionId::None).compression_id(),
            CompressionId::None
        );
        assert_eq!(
            strategy_for(CompressionId::Zstd).compression_id(),
            CompressionId::Zstd
        );
        assert_eq!(
            strategy_for(CompressionId::Lz4).compression_id(),
            CompressionId::Lz4
        );
    }

    #[test]
    fn lz4_rejects_oversized_decompression() {
        let c = Lz4Compression::default();
        let compressed = c.compress(TEST_DATA).unwrap();
        // Try to decompress with a very small max — should fail
        let result = c.decompress(&compressed, 10);
        assert!(result.is_err());
    }
}
