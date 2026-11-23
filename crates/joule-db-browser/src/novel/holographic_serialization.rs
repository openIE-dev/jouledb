//! Serialization for TurboHolographic stores
//!
//! Provides efficient binary serialization for OPFS persistence.
//!
//! Strategy: Store key-value pairs and reconstruct the holographic store.
//! This avoids needing to access private fields in TurboHolographic.

use joule_db_hdc::turbo_holographic::{TurboConfig, TurboHolographic};
use std::collections::HashMap;

/// Magic number for holographic store format
const MAGIC: &[u8] = b"THOLO"; // TurboHOLOgraphic
const VERSION: u8 = 1;

/// Serialized holographic store data
#[derive(Debug, Clone)]
pub struct SerializedHolographicStore {
    /// Dimension
    pub dimension: usize,
    /// Configuration
    pub config: TurboConfig,
    /// Key-value pairs (stored for reconstruction)
    pub key_value_pairs: Vec<(Vec<u8>, Vec<u8>)>,
}

/// Serialize TurboHolographic to bytes
///
/// Note: Since we can't access the internal bundle state directly,
/// we'll need to store key-value pairs separately and reconstruct.
/// For now, this is a placeholder that stores metadata.
pub fn serialize_turbo_holographic(
    store: &TurboHolographic,
    key_value_pairs: &[(Vec<u8>, Vec<u8>)],
) -> Vec<u8> {
    let mut result = Vec::new();

    // Magic number and version
    result.extend_from_slice(MAGIC);
    result.push(VERSION);

    // Dimension (u32)
    let dimension = store.dimension() as u32;
    result.extend_from_slice(&dimension.to_le_bytes());

    // Config - using the store's actual config
    let config = store.config();
    result.extend_from_slice(&config.seed.to_le_bytes());
    result.extend_from_slice(&(config.max_value_size as u32).to_le_bytes());
    result.extend_from_slice(&(config.position_shifts as u32).to_le_bytes());
    result.push(if config.enable_deletion { 1 } else { 0 });

    // Item count
    result.extend_from_slice(&(store.len() as u32).to_le_bytes());

    // Key-value pairs
    result.extend_from_slice(&(key_value_pairs.len() as u32).to_le_bytes());
    for (key, value) in key_value_pairs {
        // Key length and data
        result.extend_from_slice(&(key.len() as u32).to_le_bytes());
        result.extend_from_slice(key);
        // Value length and data
        result.extend_from_slice(&(value.len() as u32).to_le_bytes());
        result.extend_from_slice(value);
    }

    result
}

/// Deserialize TurboHolographic from bytes
pub fn deserialize_turbo_holographic(data: &[u8]) -> Result<SerializedHolographicStore, String> {
    if data.len() < MAGIC.len() + 1 {
        return Err("Data too short".to_string());
    }

    // Check magic number
    if &data[0..MAGIC.len()] != MAGIC {
        return Err("Invalid magic number".to_string());
    }

    let mut offset = MAGIC.len();

    // Version
    let version = data[offset];
    offset += 1;
    if version != VERSION {
        return Err(format!("Unsupported version: {}", version));
    }

    // Dimension
    if data.len() < offset + 4 {
        return Err("Data too short for dimension".to_string());
    }
    let dimension = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    offset += 4;

    // Config
    if data.len() < offset + 13 {
        return Err("Data too short for config".to_string());
    }
    let seed = u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]);
    offset += 8;

    let max_value_size = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    offset += 4;

    let position_shifts = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    offset += 4;

    let enable_deletion = data[offset] != 0;
    offset += 1;

    let config = TurboConfig {
        seed,
        max_value_size,
        position_shifts,
        enable_deletion,
    };

    // Item count (for validation)
    if data.len() < offset + 4 {
        return Err("Data too short for item_count".to_string());
    }
    let _item_count = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    offset += 4;

    // Key-value pairs
    if data.len() < offset + 4 {
        return Err("Data too short for pair count".to_string());
    }
    let pair_count = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    offset += 4;

    let mut key_value_pairs = Vec::with_capacity(pair_count);
    for _ in 0..pair_count {
        // Key
        if data.len() < offset + 4 {
            return Err("Data too short for key length".to_string());
        }
        let key_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if data.len() < offset + key_len {
            return Err("Data too short for key data".to_string());
        }
        let key = data[offset..offset + key_len].to_vec();
        offset += key_len;

        // Value
        if data.len() < offset + 4 {
            return Err("Data too short for value length".to_string());
        }
        let value_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if data.len() < offset + value_len {
            return Err("Data too short for value data".to_string());
        }
        let value = data[offset..offset + value_len].to_vec();
        offset += value_len;

        key_value_pairs.push((key, value));
    }

    Ok(SerializedHolographicStore {
        dimension,
        config,
        key_value_pairs,
    })
}

/// Reconstruct TurboHolographic from serialized data
pub fn reconstruct_turbo_holographic(serialized: &SerializedHolographicStore) -> TurboHolographic {
    let mut store = TurboHolographic::with_config(serialized.dimension, serialized.config.clone());

    // Reconstruct by re-adding all key-value pairs
    for (key, value) in &serialized.key_value_pairs {
        store.put(key, value);
    }

    store
}
