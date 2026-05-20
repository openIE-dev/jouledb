//! GPU-Friendly B-tree Node Serialization
//!
//! Provides a fixed-layout serialization format optimized for GPU parallel access.
//! Uses fixed-size structures with offset tables for efficient GPU memory access.

use joule_db_core::storage::PageId;

/// Maximum keys per node (must match B-tree MAX_KEYS in joule-db-local)
pub const GPU_MAX_KEYS: usize = 256;

/// Maximum key size (bytes) - keys larger than this are stored in overflow
pub const GPU_MAX_KEY_SIZE: usize = 256;

/// Maximum value size (bytes) - values larger than this are stored in overflow
pub const GPU_MAX_VALUE_SIZE: usize = 1024;

/// Actual serialized size of key entry (4 + 2 + 2 = 8 bytes)
pub const GPU_KEY_ENTRY_SIZE: usize = 8;

/// Actual serialized size of value entry (4 + 2 + 1 + 1 = 8 bytes)
pub const GPU_VALUE_ENTRY_SIZE: usize = 8;

/// Actual serialized size of child entry (8 bytes)
pub const GPU_CHILD_ENTRY_SIZE: usize = 8;

/// GPU B-tree node header (64 bytes, aligned for GPU)
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct GpuBTreeNodeHeader {
    /// Page ID (8 bytes)
    pub page_id: u64,
    /// Node type: 0 = internal, 1 = leaf (1 byte)
    pub is_leaf: u8,
    /// Number of keys (1 byte)
    pub num_keys: u8,
    /// Reserved (6 bytes)
    _reserved: [u8; 6],
}

/// GPU B-tree node key entry
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GpuBTreeKeyEntry {
    /// Offset into key data buffer (4 bytes)
    pub key_offset: u32,
    /// Key length (2 bytes)
    pub key_len: u16,
    /// Reserved (2 bytes)
    _reserved: u16,
}

/// GPU B-tree node value entry (leaf nodes only)
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GpuBTreeValueEntry {
    /// Offset into value data buffer (4 bytes)
    pub value_offset: u32,
    /// Value length (2 bytes)
    pub value_len: u16,
    /// Value marker: 0 = None, 1 = Empty, 2 = Data (1 byte)
    pub value_marker: u8,
    /// Reserved (1 byte)
    _reserved: u8,
}

/// GPU B-tree node child entry (internal nodes only)
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct GpuBTreeChildEntry {
    /// Child page ID (8 bytes)
    pub child_page_id: u64,
}

/// GPU-friendly B-tree node serialization
///
/// Format:
/// - Header (64 bytes, aligned)
/// - Key entries array (32 entries * 8 bytes = 256 bytes)
/// - Value entries array (32 entries * 8 bytes = 256 bytes, leaf only)
/// - Child entries array (33 entries * 8 bytes = 264 bytes, internal only)
/// - Key data buffer (variable, up to 32 * 256 = 8192 bytes)
/// - Value data buffer (variable, up to 32 * 1024 = 32768 bytes, leaf only)
///
/// Total size: ~34KB per node (worst case)
pub struct GpuBTreeNodeSerializer;

impl GpuBTreeNodeSerializer {
    /// Serialize B-tree node to GPU-friendly format
    ///
    /// # Arguments
    /// * `page_id` - Page ID
    /// * `is_leaf` - Whether this is a leaf node
    /// * `keys` - Key data (Vec<Vec<u8>>)
    /// * `values` - Value data (Vec<Option<Vec<u8>>>, leaf only)
    /// * `children` - Child page IDs (Vec<PageId>, internal only)
    ///
    /// # Returns
    /// Serialized node data and metadata
    pub fn serialize(
        page_id: PageId,
        is_leaf: bool,
        keys: &[Vec<u8>],
        values: &[Option<Vec<u8>>],
        children: &[PageId],
    ) -> Result<GpuSerializedNode, String> {
        if keys.len() > GPU_MAX_KEYS {
            return Err(format!("Too many keys: {} > {}", keys.len(), GPU_MAX_KEYS));
        }

        let num_keys = keys.len() as u8;

        // Build key data buffer
        let mut key_data = Vec::new();
        let mut key_entries = Vec::with_capacity(keys.len());

        for key in keys {
            if key.len() > GPU_MAX_KEY_SIZE {
                return Err(format!(
                    "Key too large: {} > {}",
                    key.len(),
                    GPU_MAX_KEY_SIZE
                ));
            }

            let offset = key_data.len() as u32;
            let len = key.len() as u16;

            key_entries.push(GpuBTreeKeyEntry {
                key_offset: offset,
                key_len: len,
                _reserved: 0,
            });

            key_data.extend_from_slice(key);
            // Pad to 16-byte alignment
            let padding = (16 - (key_data.len() % 16)) % 16;
            key_data.extend_from_slice(&vec![0u8; padding]);
        }

        // Build value data buffer (leaf only)
        let mut value_data = Vec::new();
        let mut value_entries = Vec::new();

        if is_leaf {
            value_entries.reserve(values.len());
            for value in values {
                let (offset, len, marker) = match value {
                    None => {
                        value_entries.push(GpuBTreeValueEntry {
                            value_offset: 0,
                            value_len: 0,
                            value_marker: 0, // VALUE_MARKER_NONE
                            _reserved: 0,
                        });
                        continue;
                    }
                    Some(v) if v.is_empty() => {
                        value_entries.push(GpuBTreeValueEntry {
                            value_offset: 0,
                            value_len: 0,
                            value_marker: 1, // VALUE_MARKER_EMPTY
                            _reserved: 0,
                        });
                        continue;
                    }
                    Some(v) => {
                        if v.len() > GPU_MAX_VALUE_SIZE {
                            return Err(format!(
                                "Value too large: {} > {}",
                                v.len(),
                                GPU_MAX_VALUE_SIZE
                            ));
                        }

                        let offset = value_data.len() as u32;
                        let len = v.len() as u16;

                        value_data.extend_from_slice(v);
                        // Pad to 16-byte alignment
                        let padding = (16 - (value_data.len() % 16)) % 16;
                        value_data.extend_from_slice(&vec![0u8; padding]);

                        (offset, len, 2u8) // VALUE_MARKER_DATA
                    }
                };

                value_entries.push(GpuBTreeValueEntry {
                    value_offset: offset,
                    value_len: len,
                    value_marker: marker,
                    _reserved: 0,
                });
            }
        }

        // Build child entries (internal only)
        let mut child_entries = Vec::new();
        if !is_leaf {
            child_entries.reserve(children.len());
            for &child_id in children {
                child_entries.push(GpuBTreeChildEntry {
                    child_page_id: child_id,
                });
            }
        }

        // Create header
        let header = GpuBTreeNodeHeader {
            page_id,
            is_leaf: if is_leaf { 1 } else { 0 },
            num_keys,
            _reserved: [0; 6],
        };

        // Serialize to bytes
        let mut buffer = Vec::with_capacity(4096); // Pre-allocate reasonable size

        // Header (64 bytes, aligned)
        buffer.extend_from_slice(&header.page_id.to_le_bytes()); // 8 bytes
        buffer.push(header.is_leaf); // 1 byte
        buffer.push(header.num_keys); // 1 byte
        buffer.extend_from_slice(&header._reserved); // 6 bytes
        // Pad to 64 bytes to match struct alignment
        buffer.extend_from_slice(&vec![0u8; 48]); // 16 bytes written, pad to 64

        // Key entries (pad to 256 bytes for 32 entries)
        for (i, entry) in key_entries.iter().enumerate() {
            buffer.extend_from_slice(&entry.key_offset.to_le_bytes());
            buffer.extend_from_slice(&entry.key_len.to_le_bytes());
            buffer.extend_from_slice(&entry._reserved.to_le_bytes());
        }
        // Pad to 32 entries
        let padding = (GPU_MAX_KEYS - key_entries.len()) * GPU_KEY_ENTRY_SIZE;
        buffer.extend_from_slice(&vec![0u8; padding]);

        // Value entries (leaf only, pad to 256 bytes)
        if is_leaf {
            for entry in &value_entries {
                buffer.extend_from_slice(&entry.value_offset.to_le_bytes());
                buffer.extend_from_slice(&entry.value_len.to_le_bytes());
                buffer.push(entry.value_marker);
                buffer.push(entry._reserved);
            }
            let padding = (GPU_MAX_KEYS - value_entries.len()) * GPU_VALUE_ENTRY_SIZE;
            buffer.extend_from_slice(&vec![0u8; padding]);
        } else {
            // Internal nodes: child entries (33 entries max, 264 bytes)
            for entry in &child_entries {
                buffer.extend_from_slice(&entry.child_page_id.to_le_bytes());
            }
            // Pad to 33 entries
            let padding = (33 - child_entries.len()) * GPU_CHILD_ENTRY_SIZE;
            buffer.extend_from_slice(&vec![0u8; padding]);
        }

        // Key data offset
        let key_data_offset = buffer.len() as u32;
        buffer.extend_from_slice(&key_data);

        // Value data offset (leaf only)
        let value_data_offset = if is_leaf { buffer.len() as u32 } else { 0 };
        if is_leaf {
            buffer.extend_from_slice(&value_data);
        }

        Ok(GpuSerializedNode {
            buffer,
            key_data_offset,
            value_data_offset,
            num_keys,
            is_leaf,
        })
    }
}

/// Serialized GPU B-tree node
pub struct GpuSerializedNode {
    /// Complete serialized buffer
    pub buffer: Vec<u8>,
    /// Offset to key data section
    pub key_data_offset: u32,
    /// Offset to value data section (0 for internal nodes)
    pub value_data_offset: u32,
    /// Number of keys
    pub num_keys: u8,
    /// Whether this is a leaf node
    pub is_leaf: bool,
}

impl GpuSerializedNode {
    /// Get the serialized buffer
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.buffer.len()
    }
}

/// Deserialize GPU node (for verification/testing)
pub struct GpuBTreeNodeDeserializer;

impl GpuBTreeNodeDeserializer {
    /// Deserialize GPU node back to Rust structures
    pub fn deserialize(
        data: &[u8],
    ) -> Result<
        (
            PageId,
            bool,
            Vec<Vec<u8>>,
            Vec<Option<Vec<u8>>>,
            Vec<PageId>,
        ),
        String,
    > {
        let header_size = std::mem::size_of::<GpuBTreeNodeHeader>();
        if data.len() < header_size {
            return Err("Buffer too small for header".to_string());
        }

        // Read header fields safely without pointer alignment issues
        let page_id = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let is_leaf_byte = data[8];
        let num_keys_byte = data[9];

        // Create header for reference
        let header = GpuBTreeNodeHeader {
            page_id,
            is_leaf: is_leaf_byte,
            num_keys: num_keys_byte,
            _reserved: [0; 6],
        };

        let is_leaf = header.is_leaf == 1;
        let num_keys = header.num_keys as usize;

        let key_entry_size = GPU_KEY_ENTRY_SIZE;
        let value_entry_size = GPU_VALUE_ENTRY_SIZE;
        let child_entry_size = GPU_CHILD_ENTRY_SIZE;

        // Calculate data buffer start offsets
        // Format: Header + Key entries (32) + Value entries (32, leaf) OR Child entries (33, internal)
        let key_data_base = if is_leaf {
            header_size + (GPU_MAX_KEYS * key_entry_size) + (GPU_MAX_KEYS * value_entry_size)
        } else {
            header_size + (GPU_MAX_KEYS * key_entry_size) + (33 * child_entry_size)
        };

        let mut offset = header_size;

        // First pass: read all key entries to determine key data buffer size
        let mut key_entries = Vec::with_capacity(num_keys);
        for i in 0..num_keys {
            if offset + key_entry_size > data.len() {
                return Err("Key entries truncated".to_string());
            }

            let key_offset = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            let key_len = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);

            key_entries.push((key_offset, key_len));
            offset += key_entry_size;
        }

        // Skip key entry padding
        offset += (GPU_MAX_KEYS - num_keys) * key_entry_size;

        // Read value or child entries and calculate value_data_base
        let mut value_entries = Vec::new();
        let mut children = Vec::new();

        if is_leaf {
            // Calculate where value data starts (after key data)
            // Find the end of key data by looking at key entries
            let mut key_data_end = 0usize;
            for (ko, kl) in &key_entries {
                let end = *ko as usize + *kl as usize;
                // Account for 16-byte alignment padding
                let aligned_end = ((end + 15) / 16) * 16;
                if aligned_end > key_data_end {
                    key_data_end = aligned_end;
                }
            }
            let value_data_base = key_data_base + key_data_end;

            // Read value entries
            for _i in 0..num_keys {
                if offset + value_entry_size > data.len() {
                    return Err("Value entries truncated".to_string());
                }

                let value_offset = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                let value_len = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);
                let value_marker = data[offset + 6];

                value_entries.push((value_offset, value_len, value_marker));
                offset += value_entry_size;
            }

            // Now extract key data using key_data_base
            let mut keys = Vec::with_capacity(num_keys);
            for (i, (key_offset, key_len)) in key_entries.iter().enumerate() {
                let abs_offset = key_data_base + *key_offset as usize;
                if abs_offset + *key_len as usize > data.len() {
                    return Err("Key data out of bounds".to_string());
                }
                let key_data = &data[abs_offset..abs_offset + *key_len as usize];
                keys.push(key_data.to_vec());
            }

            // Extract value data using value_data_base
            let mut values = Vec::with_capacity(num_keys);
            for (value_offset, value_len, value_marker) in value_entries {
                let value = match value_marker {
                    0 => None,             // VALUE_MARKER_NONE
                    1 => Some(Vec::new()), // VALUE_MARKER_EMPTY
                    2 => {
                        let abs_offset = value_data_base + value_offset as usize;
                        if abs_offset + value_len as usize > data.len() {
                            return Err("Value data out of bounds".to_string());
                        }
                        let value_data = &data[abs_offset..abs_offset + value_len as usize];
                        Some(value_data.to_vec())
                    }
                    _ => return Err("Invalid value marker".to_string()),
                };
                values.push(value);
            }

            return Ok((page_id, is_leaf, keys, values, children));
        } else {
            // Read child entries
            let num_children = num_keys + 1;
            for _ in 0..num_children {
                if offset + child_entry_size > data.len() {
                    return Err("Child entries truncated".to_string());
                }

                let child_page_id = u64::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]);
                offset += child_entry_size;
                children.push(child_page_id);
            }

            // Extract key data using key_data_base
            let mut keys = Vec::with_capacity(num_keys);
            for (key_offset, key_len) in key_entries {
                let abs_offset = key_data_base + key_offset as usize;
                if abs_offset + key_len as usize > data.len() {
                    return Err("Key data out of bounds".to_string());
                }
                let key_data = &data[abs_offset..abs_offset + key_len as usize];
                keys.push(key_data.to_vec());
            }

            return Ok((page_id, is_leaf, keys, Vec::new(), children));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize() {
        let keys = vec![b"key1".to_vec(), b"key2".to_vec(), b"key3".to_vec()];
        let values = vec![Some(b"value1".to_vec()), None, Some(b"value3".to_vec())];

        let serialized = GpuBTreeNodeSerializer::serialize(1, true, &keys, &values, &[]).unwrap();

        let (page_id, is_leaf, deserialized_keys, deserialized_values, _) =
            GpuBTreeNodeDeserializer::deserialize(serialized.as_bytes()).unwrap();

        assert_eq!(page_id, 1);
        assert!(is_leaf);
        assert_eq!(deserialized_keys, keys);
        assert_eq!(deserialized_values, values);
    }
}
