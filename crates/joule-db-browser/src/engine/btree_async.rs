//! Async B-tree implementation for browser backends
//!
//! Full B-tree structure with async I/O operations.

use std::sync::{Arc, RwLock};
use joule_db_core::{
    Error as CoreError,
    StorageError,
    storage::PageId,
};
use crate::persistence::AsyncPersistenceBackend;

/// Async B-tree node
pub struct AsyncBTreeNode {
    page_id: PageId,
    keys: Vec<Vec<u8>>,
    values: Vec<Option<Vec<u8>>>, // None for internal nodes
    children: Vec<PageId>, // Only for internal nodes
    is_leaf: bool,
    min_degree: usize, // Minimum degree (t)
}

impl AsyncBTreeNode {
    /// Create new leaf node
    pub fn new_leaf(page_id: PageId, min_degree: usize) -> Self {
        Self {
            page_id,
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: true,
            min_degree,
        }
    }

    /// Create new internal node
    pub fn new_internal(page_id: PageId, min_degree: usize) -> Self {
        Self {
            page_id,
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: false,
            min_degree,
        }
    }

    /// Check if node is full
    pub fn is_full(&self) -> bool {
        self.keys.len() >= 2 * self.min_degree - 1
    }

    /// Check if node is underfull
    pub fn is_underfull(&self) -> bool {
        self.keys.len() < self.min_degree - 1
    }

    /// Serialize node to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        
        // is_leaf (1 byte)
        buf.push(if self.is_leaf { 1 } else { 0 });
        
        // num_keys (4 bytes)
        buf.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());
        
        // Keys
        for key in &self.keys {
            buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
            buf.extend_from_slice(key);
        }
        
        // Values (leaf only)
        if self.is_leaf {
            for value in &self.values {
                match value {
                    None => buf.push(0), // NULL marker
                    Some(v) if v.is_empty() => buf.push(1), // Empty marker
                    Some(v) => {
                        buf.push(2); // Data marker
                        buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                        buf.extend_from_slice(v);
                    }
                }
            }
        }
        
        // Children (internal only)
        if !self.is_leaf {
            buf.extend_from_slice(&(self.children.len() as u32).to_le_bytes());
            for child in &self.children {
                buf.extend_from_slice(&child.to_le_bytes());
            }
        }
        
        buf
    }

    /// Deserialize node from bytes
    pub fn deserialize(page_id: PageId, data: &[u8], min_degree: usize) -> Result<Self, StorageError> {
        if data.is_empty() {
            return Err(StorageError::Corrupted {
                page_id,
                reason: "Empty node data".to_string(),
            });
        }

        let mut cursor = 0;

        // is_leaf
        let is_leaf = data[cursor] == 1;
        cursor += 1;

        // num_keys
        if cursor + 4 > data.len() {
            return Err(StorageError::Corrupted {
                page_id,
                reason: "Missing key count".to_string(),
            });
        }
        let num_keys = u32::from_le_bytes([
            data[cursor], data[cursor + 1], data[cursor + 2], data[cursor + 3]
        ]) as usize;
        cursor += 4;

        // Keys
        let mut keys = Vec::with_capacity(num_keys);
        for _ in 0..num_keys {
            if cursor + 4 > data.len() {
                return Err(StorageError::Corrupted {
                    page_id,
                    reason: "Missing key length".to_string(),
                });
            }
            let key_len = u32::from_le_bytes([
                data[cursor], data[cursor + 1], data[cursor + 2], data[cursor + 3]
            ]) as usize;
            cursor += 4;

            if cursor + key_len > data.len() {
                return Err(StorageError::Corrupted {
                    page_id,
                    reason: "Key data truncated".to_string(),
                });
            }
            keys.push(data[cursor..cursor + key_len].to_vec());
            cursor += key_len;
        }

        // Values (leaf only)
        let mut values = Vec::new();
        if is_leaf {
            for _ in 0..num_keys {
                if cursor >= data.len() {
                    return Err(StorageError::Corrupted {
                        page_id,
                        reason: "Missing value marker".to_string(),
                    });
                }
                let marker = data[cursor];
                cursor += 1;

                match marker {
                    0 => values.push(None), // NULL
                    1 => values.push(Some(Vec::new())), // Empty
                    2 => {
                        if cursor + 4 > data.len() {
                            return Err(StorageError::Corrupted {
                                page_id,
                                reason: "Missing value length".to_string(),
                            });
                        }
                        let val_len = u32::from_le_bytes([
                            data[cursor], data[cursor + 1], data[cursor + 2], data[cursor + 3]
                        ]) as usize;
                        cursor += 4;

                        if cursor + val_len > data.len() {
                            return Err(StorageError::Corrupted {
                                page_id,
                                reason: "Value data truncated".to_string(),
                            });
                        }
                        values.push(Some(data[cursor..cursor + val_len].to_vec()));
                        cursor += val_len;
                    }
                    _ => {
                        return Err(StorageError::Corrupted {
                            page_id,
                            reason: format!("Invalid value marker: {}", marker),
                        });
                    }
                }
            }
        }

        // Children (internal only)
        let mut children = Vec::new();
        if !is_leaf {
            if cursor + 4 > data.len() {
                return Err(StorageError::Corrupted {
                    page_id,
                    reason: "Missing children count".to_string(),
                });
            }
            let num_children = u32::from_le_bytes([
                data[cursor], data[cursor + 1], data[cursor + 2], data[cursor + 3]
            ]) as usize;
            cursor += 4;

            for _ in 0..num_children {
                if cursor + 8 > data.len() {
                    return Err(StorageError::Corrupted {
                        page_id,
                        reason: "Child pointer truncated".to_string(),
                    });
                }
                let child_id = u64::from_le_bytes([
                    data[cursor], data[cursor + 1], data[cursor + 2], data[cursor + 3],
                    data[cursor + 4], data[cursor + 5], data[cursor + 6], data[cursor + 7],
                ]);
                cursor += 8;
                children.push(child_id);
            }
        }

        Ok(Self {
            page_id,
            keys,
            values,
            children,
            is_leaf,
            min_degree,
        })
    }
}

/// Async B-tree implementation
pub struct AsyncBTree {
    backend: Arc<RwLock<dyn AsyncPersistenceBackend + Send + Sync>>,
    root_page_id: Arc<RwLock<PageId>>,
    next_page_id: Arc<RwLock<PageId>>,
    min_degree: usize,
}

impl AsyncBTree {
    /// Create new B-tree
    pub fn new(backend: Arc<RwLock<dyn AsyncPersistenceBackend + Send + Sync>>, min_degree: usize) -> Self {
        Self {
            backend,
            root_page_id: Arc::new(RwLock::new(1)),
            next_page_id: Arc::new(RwLock::new(2)),
            min_degree,
        }
    }

    /// Search for a key
    pub async fn search(&self, key: &[u8]) -> Result<Option<Vec<u8>>, CoreError> {
        let root_id = *self.root_page_id.read().unwrap();
        self.search_node(root_id, key).await
    }

    /// Search in a specific node
    async fn search_node(&self, page_id: PageId, key: &[u8]) -> Result<Option<Vec<u8>>, CoreError> {
        // Load node
        let backend = self.backend.read().unwrap();
        let data = backend.read_page(page_id).await
            .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
        drop(backend);

        let node = AsyncBTreeNode::deserialize(page_id, &data, self.min_degree)
            .map_err(|e| CoreError::Storage(e))?;

        // Binary search for key
        let idx = node.keys.binary_search_by(|k| k.as_slice().cmp(key));

        if node.is_leaf {
            match idx {
                Ok(i) => Ok(node.values[i].clone()),
                Err(_) => Ok(None),
            }
        } else {
            // Internal node - recurse to child
            match idx {
                Ok(i) => {
                    // Found in this node (shouldn't happen in internal node, but handle it)
                    Ok(None)
                }
                Err(i) => {
                    // Search in child
                    let child_id = node.children[i];
                    self.search_node(child_id, key).await
                }
            }
        }
    }

    /// Insert key-value pair
    pub async fn insert(&self, key: &[u8], value: &[u8]) -> Result<(), CoreError> {
        let root_id = *self.root_page_id.read().unwrap();
        
        // Load root
        let backend = self.backend.read().unwrap();
        let root_data = backend.read_page(root_id).await
            .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
        drop(backend);

        let mut root = AsyncBTreeNode::deserialize(root_id, &root_data, self.min_degree)
            .map_err(|e| CoreError::Storage(e))?;

        // If root is full, split it
        if root.is_full() {
            let new_root_id = {
                let mut next = self.next_page_id.write().unwrap();
                let id = *next;
                *next += 1;
                id
            };

            let mut new_root = AsyncBTreeNode::new_internal(new_root_id, self.min_degree);
            new_root.children.push(root_id);
            
            // Split old root
            let (right, median_key) = root.split(new_root_id);
            
            // Update new root
            new_root.keys.push(median_key);
            new_root.children.push(right.page_id);

            // Save nodes
            let backend = self.backend.read().unwrap();
            backend.write_page(new_root.page_id, &new_root.serialize()).await
                .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
            backend.write_page(root.page_id, &root.serialize()).await
                .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
            backend.write_page(right.page_id, &right.serialize()).await
                .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
            drop(backend);

            *self.root_page_id.write().unwrap() = new_root_id;
            
            // Insert into new root
            self.insert_non_full(new_root_id, key, value).await
        } else {
            self.insert_non_full(root_id, key, value).await
        }
    }

    /// Insert into non-full node
    async fn insert_non_full(&self, page_id: PageId, key: &[u8], value: &[u8]) -> Result<(), CoreError> {
        // Load node
        let backend = self.backend.read().unwrap();
        let data = backend.read_page(page_id).await
            .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
        drop(backend);

        let mut node = AsyncBTreeNode::deserialize(page_id, &data, self.min_degree)
            .map_err(|e| CoreError::Storage(e))?;

        if node.is_leaf {
            // Insert into leaf
            let idx = node.keys.binary_search_by(|k| k.as_slice().cmp(key))
                .unwrap_or_else(|i| i);
            
            if idx < node.keys.len() && node.keys[idx] == key {
                // Update existing
                node.values[idx] = Some(value.to_vec());
            } else {
                // Insert new
                node.keys.insert(idx, key.to_vec());
                node.values.insert(idx, Some(value.to_vec()));
            }

            // Save node
            let backend = self.backend.read().unwrap();
            backend.write_page(page_id, &node.serialize()).await
                .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
            Ok(())
        } else {
            // Internal node - find child to insert into
            let idx = node.keys.binary_search_by(|k| k.as_slice().cmp(key))
                .unwrap_or_else(|i| i);
            
            let child_id = node.children[idx];
            
            // Load child
            let backend = self.backend.read().unwrap();
            let child_data = backend.read_page(child_id).await
                .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
            drop(backend);

            let child = AsyncBTreeNode::deserialize(child_id, &child_data, self.min_degree)
                .map_err(|e| CoreError::Storage(e))?;

            // If child is full, split it
            if child.is_full() {
                let new_child_id = {
                    let mut next = self.next_page_id.write().unwrap();
                    let id = *next;
                    *next += 1;
                    id
                };

                let (mut right, median_key) = child.split(new_child_id);

                // Insert median into current node
                node.keys.insert(idx, median_key);
                node.children.insert(idx + 1, new_child_id);

                // Save nodes
                let backend = self.backend.read().unwrap();
                backend.write_page(page_id, &node.serialize()).await
                    .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
                backend.write_page(child_id, &child.serialize()).await
                    .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
                backend.write_page(new_child_id, &right.serialize()).await
                    .map_err(|e| CoreError::Storage(StorageError::Backend(e.to_string())))?;
                drop(backend);

                // Determine which child to insert into
                if key < &node.keys[idx] {
                    self.insert_non_full(child_id, key, value).await
                } else {
                    self.insert_non_full(new_child_id, key, value).await
                }
            } else {
                self.insert_non_full(child_id, key, value).await
            }
        }
    }
}

impl AsyncBTreeNode {
    /// Split node, returns (new_node, median_key)
    fn split(&mut self, new_page_id: PageId) -> (Self, Vec<u8>) {
        let mid = self.keys.len() / 2;
        let median_key = self.keys[mid].clone();

        let mut right = if self.is_leaf {
            Self::new_leaf(new_page_id, self.min_degree)
        } else {
            Self::new_internal(new_page_id, self.min_degree)
        };

        if self.is_leaf {
            right.keys = self.keys.split_off(mid);
            right.values = self.values.split_off(mid);
        } else {
            right.keys = self.keys.split_off(mid + 1);
            right.children = self.children.split_off(mid + 1);
            self.keys.pop(); // Remove median from left
        }

        (right, median_key)
    }
}

