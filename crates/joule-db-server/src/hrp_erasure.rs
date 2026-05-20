//! HRP Phase 4: Reed-Solomon erasure coding for large payload transfer.
//!
//! When a Raft message payload exceeds a size threshold (e.g., InstallSnapshot),
//! it can be split into k data shards + m parity shards. Any k of the k+m shards
//! are sufficient to reconstruct the original data, providing fault tolerance
//! with reduced peak bandwidth compared to full replication.
//!
//! Uses the `reed-solomon-erasure` crate (GF(2^8), pure Rust).

use reed_solomon_erasure::galois_8::ReedSolomon;
use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for erasure coding.
#[derive(Debug, Clone)]
pub struct ErasureConfig {
    /// Number of data shards (k)
    pub data_shards: usize,
    /// Number of parity shards (m)
    pub parity_shards: usize,
    /// Only apply erasure coding to payloads larger than this (bytes)
    pub threshold_bytes: usize,
}

impl Default for ErasureConfig {
    fn default() -> Self {
        Self {
            data_shards: 2,
            parity_shards: 1,
            threshold_bytes: 64 * 1024, // 64 KB
        }
    }
}

// ============================================================================
// Shard
// ============================================================================

/// A single shard from an erasure-coded payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureShard {
    /// Shard index (0..data_shards+parity_shards)
    pub index: usize,
    /// Shard data
    pub data: Vec<u8>,
    /// Total number of shards (k + m)
    pub total_shards: usize,
    /// Original payload length (for trimming padding after reconstruction)
    pub original_len: usize,
    /// Number of data shards (k) — needed for reconstruction
    pub data_shards: usize,
    /// Number of parity shards (m) — needed for reconstruction
    pub parity_shards: usize,
}

// ============================================================================
// Encode / Decode
// ============================================================================

/// Encode a payload into k data shards + m parity shards.
///
/// Returns `data_shards + parity_shards` shards. Any `data_shards` of them
/// are sufficient to reconstruct the original payload.
pub fn encode(data: &[u8], config: &ErasureConfig) -> Result<Vec<ErasureShard>, ErasureError> {
    if data.is_empty() {
        return Err(ErasureError::EmptyPayload);
    }
    if config.data_shards == 0 || config.parity_shards == 0 {
        return Err(ErasureError::InvalidConfig(
            "data_shards and parity_shards must be > 0".to_string(),
        ));
    }

    let rs = ReedSolomon::new(config.data_shards, config.parity_shards)
        .map_err(|e| ErasureError::Internal(format!("ReedSolomon::new: {}", e)))?;

    let total = config.data_shards + config.parity_shards;
    let original_len = data.len();

    // Compute shard size (pad so data divides evenly into k shards)
    let shard_size = (data.len() + config.data_shards - 1) / config.data_shards;

    // Build data shards (pad last shard with zeros if needed)
    let mut shards: Vec<Vec<u8>> = Vec::with_capacity(total);
    for i in 0..config.data_shards {
        let start = i * shard_size;
        let end = std::cmp::min(start + shard_size, data.len());
        let mut shard = Vec::with_capacity(shard_size);
        if start < data.len() {
            shard.extend_from_slice(&data[start..end]);
        }
        // Pad to shard_size
        shard.resize(shard_size, 0);
        shards.push(shard);
    }

    // Add empty parity shards
    for _ in 0..config.parity_shards {
        shards.push(vec![0u8; shard_size]);
    }

    // Compute parity
    rs.encode(&mut shards)
        .map_err(|e| ErasureError::Internal(format!("encode: {}", e)))?;

    // Wrap in ErasureShard structs
    let result = shards
        .into_iter()
        .enumerate()
        .map(|(index, shard_data)| ErasureShard {
            index,
            data: shard_data,
            total_shards: total,
            original_len,
            data_shards: config.data_shards,
            parity_shards: config.parity_shards,
        })
        .collect();

    Ok(result)
}

/// Reconstruct the original payload from a set of shards.
///
/// Requires at least `data_shards` (k) shards. Missing shards are
/// represented as `None` in the input slice, indexed by shard position.
pub fn decode(shards: &[Option<ErasureShard>]) -> Result<Vec<u8>, ErasureError> {
    if shards.is_empty() {
        return Err(ErasureError::EmptyPayload);
    }

    // Extract config from the first available shard
    let first = shards
        .iter()
        .flatten()
        .next()
        .ok_or(ErasureError::InsufficientShards {
            available: 0,
            required: 0,
        })?;

    let data_shards = first.data_shards;
    let parity_shards = first.parity_shards;
    let total = data_shards + parity_shards;
    let original_len = first.original_len;

    if shards.len() != total {
        return Err(ErasureError::InvalidConfig(format!(
            "expected {} shard slots, got {}",
            total,
            shards.len()
        )));
    }

    let available = shards.iter().filter(|s| s.is_some()).count();
    if available < data_shards {
        return Err(ErasureError::InsufficientShards {
            available,
            required: data_shards,
        });
    }

    let rs = ReedSolomon::new(data_shards, parity_shards)
        .map_err(|e| ErasureError::Internal(format!("ReedSolomon::new: {}", e)))?;

    // Build the shard matrix (Option<Vec<u8>>)
    let mut shard_data: Vec<Option<Vec<u8>>> = shards
        .iter()
        .map(|s| s.as_ref().map(|es| es.data.clone()))
        .collect();

    // Reconstruct missing shards
    rs.reconstruct(&mut shard_data)
        .map_err(|e| ErasureError::Internal(format!("reconstruct: {}", e)))?;

    // Concatenate data shards and trim to original length
    let mut result = Vec::with_capacity(original_len);
    for shard in shard_data.iter().take(data_shards) {
        if let Some(data) = shard {
            result.extend_from_slice(data);
        }
    }
    result.truncate(original_len);

    Ok(result)
}

// ============================================================================
// Errors
// ============================================================================

/// Erasure coding errors.
#[derive(Debug, Clone)]
pub enum ErasureError {
    EmptyPayload,
    InvalidConfig(String),
    InsufficientShards { available: usize, required: usize },
    Internal(String),
}

impl std::fmt::Display for ErasureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErasureError::EmptyPayload => write!(f, "empty payload"),
            ErasureError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            ErasureError::InsufficientShards {
                available,
                required,
            } => {
                write!(
                    f,
                    "insufficient shards: {} available, {} required",
                    available, required
                )
            }
            ErasureError::Internal(msg) => write!(f, "internal erasure error: {}", msg),
        }
    }
}

impl std::error::Error for ErasureError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erasure_encode_decode_all_shards() {
        let config = ErasureConfig {
            data_shards: 3,
            parity_shards: 2,
            threshold_bytes: 0,
        };
        let original = b"Hello, erasure coding! This is a test payload for JouleDB HRP.";

        let shards = encode(original, &config).unwrap();
        assert_eq!(shards.len(), 5);
        assert_eq!(shards[0].total_shards, 5);
        assert_eq!(shards[0].original_len, original.len());

        // Decode with all shards present
        let shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
        let recovered = decode(&shard_opts).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_erasure_missing_parity_shards() {
        let config = ErasureConfig {
            data_shards: 3,
            parity_shards: 2,
            threshold_bytes: 0,
        };
        let original = b"Test data for erasure coding recovery";

        let shards = encode(original, &config).unwrap();

        // Drop both parity shards (indices 3 and 4) — still have k=3 data shards
        let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
        shard_opts[3] = None;
        shard_opts[4] = None;

        let recovered = decode(&shard_opts).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_erasure_missing_data_shard() {
        let config = ErasureConfig {
            data_shards: 2,
            parity_shards: 1,
            threshold_bytes: 0,
        };
        let original = b"Recovery test: can we survive losing a data shard?";

        let shards = encode(original, &config).unwrap();

        // Drop one data shard (index 0) — have 1 data + 1 parity = k shards
        let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
        shard_opts[0] = None;

        let recovered = decode(&shard_opts).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_erasure_too_many_missing() {
        let config = ErasureConfig {
            data_shards: 2,
            parity_shards: 1,
            threshold_bytes: 0,
        };
        let original = b"This should fail to decode";

        let shards = encode(original, &config).unwrap();

        // Drop 2 shards (more than m=1 parity can handle)
        let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
        shard_opts[0] = None;
        shard_opts[1] = None;

        let result = decode(&shard_opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_erasure_empty_payload() {
        let config = ErasureConfig::default();
        let result = encode(b"", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_erasure_single_byte() {
        let config = ErasureConfig {
            data_shards: 2,
            parity_shards: 1,
            threshold_bytes: 0,
        };
        let original = &[0x42u8];

        let shards = encode(original, &config).unwrap();
        let shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
        let recovered = decode(&shard_opts).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_erasure_large_payload() {
        let config = ErasureConfig {
            data_shards: 4,
            parity_shards: 2,
            threshold_bytes: 0,
        };
        // 100KB payload
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let shards = encode(&original, &config).unwrap();
        assert_eq!(shards.len(), 6);

        // Drop 2 shards (equal to parity count m=2)
        let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
        shard_opts[1] = None;
        shard_opts[4] = None;

        let recovered = decode(&shard_opts).unwrap();
        assert_eq!(recovered, original);
    }
}
