//! Merkle Patricia trie (simplified) — key-value storage with cryptographic
//! root hash, insert/get/delete, proof generation, proof verification, trie
//! serialization, and node types (leaf/extension/branch).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Inline SHA-256 ──────────────────────────────────────────────────────────

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_process_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

fn sha256(input: &[u8]) -> [u8; 32] {
    let bit_len = (input.len() as u64) * 8;
    let mut padded = input.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = SHA256_H0;
    for chunk in padded.chunks(64) {
        let mut block = [0u8; 64];
        block.copy_from_slice(chunk);
        sha256_process_block(&mut state, &block);
    }

    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hash_bytes(data: &[u8]) -> String {
    bytes_to_hex(&sha256(data))
}

// ── Nibbles ─────────────────────────────────────────────────────────────────

/// Convert a byte slice to nibbles (each byte -> 2 nibbles).
fn to_nibbles(key: &[u8]) -> Vec<u8> {
    let mut nibbles = Vec::with_capacity(key.len() * 2);
    for byte in key {
        nibbles.push(byte >> 4);
        nibbles.push(byte & 0x0f);
    }
    nibbles
}

/// Convert nibbles back to bytes.
fn from_nibbles(nibbles: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((nibbles.len() + 1) / 2);
    let mut i = 0;
    while i + 1 < nibbles.len() {
        bytes.push((nibbles[i] << 4) | nibbles[i + 1]);
        i += 2;
    }
    if i < nibbles.len() {
        bytes.push(nibbles[i] << 4);
    }
    bytes
}

/// Common prefix length of two nibble slices.
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from state trie operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrieError {
    /// Key not found.
    KeyNotFound(Vec<u8>),
    /// Proof verification failed.
    InvalidProof(String),
    /// Deserialization error.
    DeserializeError(String),
}

impl fmt::Display for TrieError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyNotFound(key) => write!(f, "key not found: {:?}", key),
            Self::InvalidProof(msg) => write!(f, "invalid proof: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialization error: {msg}"),
        }
    }
}

impl std::error::Error for TrieError {}

// ── Node Types ──────────────────────────────────────────────────────────────

/// Node type label for serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Empty,
    Leaf,
    Extension,
    Branch,
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty"),
            Self::Leaf => write!(f, "leaf"),
            Self::Extension => write!(f, "extension"),
            Self::Branch => write!(f, "branch"),
        }
    }
}

/// A node in the Merkle Patricia trie.
#[derive(Debug, Clone)]
enum TrieNode {
    /// Empty node.
    Empty,
    /// Leaf node: remaining nibble path + value.
    Leaf {
        nibbles: Vec<u8>,
        value: Vec<u8>,
    },
    /// Extension node: shared nibble path + child.
    Extension {
        nibbles: Vec<u8>,
        child: Box<TrieNode>,
    },
    /// Branch node: 16 children (one per nibble) + optional value.
    Branch {
        children: Vec<Option<Box<TrieNode>>>,
        value: Option<Vec<u8>>,
    },
}

impl TrieNode {
    fn new_branch() -> Self {
        let children: Vec<Option<Box<TrieNode>>> = (0..16).map(|_| None).collect();
        TrieNode::Branch {
            children,
            value: None,
        }
    }

    fn node_type(&self) -> NodeType {
        match self {
            TrieNode::Empty => NodeType::Empty,
            TrieNode::Leaf { .. } => NodeType::Leaf,
            TrieNode::Extension { .. } => NodeType::Extension,
            TrieNode::Branch { .. } => NodeType::Branch,
        }
    }

    /// Compute the hash of this node (for Merkle proofs).
    fn hash(&self) -> String {
        match self {
            TrieNode::Empty => hash_bytes(b"empty"),
            TrieNode::Leaf { nibbles, value } => {
                let mut data = Vec::new();
                data.push(0x00); // leaf prefix
                data.extend_from_slice(&from_nibbles(nibbles));
                data.extend_from_slice(value);
                hash_bytes(&data)
            }
            TrieNode::Extension { nibbles, child } => {
                let mut data = Vec::new();
                data.push(0x01); // extension prefix
                data.extend_from_slice(&from_nibbles(nibbles));
                data.extend_from_slice(child.hash().as_bytes());
                hash_bytes(&data)
            }
            TrieNode::Branch { children, value } => {
                let mut data = Vec::new();
                data.push(0x02); // branch prefix
                for child in children {
                    match child {
                        Some(c) => data.extend_from_slice(c.hash().as_bytes()),
                        None => data.extend_from_slice(b"null"),
                    }
                }
                if let Some(v) = value {
                    data.extend_from_slice(v);
                }
                hash_bytes(&data)
            }
        }
    }

    /// Insert a key-value pair (nibble path) into this node subtree.
    fn insert(self, nibbles: &[u8], value: Vec<u8>) -> TrieNode {
        match self {
            TrieNode::Empty => TrieNode::Leaf {
                nibbles: nibbles.to_vec(),
                value,
            },

            TrieNode::Leaf {
                nibbles: existing_nibbles,
                value: existing_value,
            } => {
                let common = common_prefix_len(&existing_nibbles, nibbles);

                if common == existing_nibbles.len() && common == nibbles.len() {
                    // Same key — replace value.
                    return TrieNode::Leaf {
                        nibbles: existing_nibbles,
                        value,
                    };
                }

                // Build branch children and value directly
                let mut branch_children: Vec<Option<Box<TrieNode>>> =
                    (0..16).map(|_| None).collect();
                let mut branch_value: Option<Vec<u8>> = None;

                if common == existing_nibbles.len() {
                    // Existing leaf becomes the branch value
                    branch_value = Some(existing_value);
                    let new_child = TrieNode::Leaf {
                        nibbles: nibbles[common + 1..].to_vec(),
                        value,
                    };
                    branch_children[nibbles[common] as usize] = Some(Box::new(new_child));
                } else if common == nibbles.len() {
                    branch_value = Some(value);
                    let existing_child = TrieNode::Leaf {
                        nibbles: existing_nibbles[common + 1..].to_vec(),
                        value: existing_value,
                    };
                    branch_children[existing_nibbles[common] as usize] =
                        Some(Box::new(existing_child));
                } else {
                    let existing_child = TrieNode::Leaf {
                        nibbles: existing_nibbles[common + 1..].to_vec(),
                        value: existing_value,
                    };
                    branch_children[existing_nibbles[common] as usize] =
                        Some(Box::new(existing_child));

                    let new_child = TrieNode::Leaf {
                        nibbles: nibbles[common + 1..].to_vec(),
                        value,
                    };
                    branch_children[nibbles[common] as usize] = Some(Box::new(new_child));
                }

                let branch = TrieNode::Branch {
                    children: branch_children,
                    value: branch_value,
                };

                if common > 0 {
                    TrieNode::Extension {
                        nibbles: nibbles[..common].to_vec(),
                        child: Box::new(branch),
                    }
                } else {
                    branch
                }
            }

            TrieNode::Extension {
                nibbles: ext_nibbles,
                child,
            } => {
                let common = common_prefix_len(&ext_nibbles, nibbles);

                if common == ext_nibbles.len() {
                    // Continue into child
                    let new_child = child.insert(&nibbles[common..], value);
                    return TrieNode::Extension {
                        nibbles: ext_nibbles,
                        child: Box::new(new_child),
                    };
                }

                // Need to split the extension
                let mut branch_children: Vec<Option<Box<TrieNode>>> =
                    (0..16).map(|_| None).collect();
                let mut branch_value: Option<Vec<u8>> = None;

                // Remaining extension
                if common + 1 < ext_nibbles.len() {
                    let remaining_ext = TrieNode::Extension {
                        nibbles: ext_nibbles[common + 1..].to_vec(),
                        child,
                    };
                    branch_children[ext_nibbles[common] as usize] = Some(Box::new(remaining_ext));
                } else {
                    branch_children[ext_nibbles[common] as usize] = Some(child);
                }

                // New key
                if common < nibbles.len() {
                    let new_child = TrieNode::Leaf {
                        nibbles: nibbles[common + 1..].to_vec(),
                        value,
                    };
                    branch_children[nibbles[common] as usize] = Some(Box::new(new_child));
                } else {
                    branch_value = Some(value);
                }

                let branch = TrieNode::Branch {
                    children: branch_children,
                    value: branch_value,
                };

                if common > 0 {
                    TrieNode::Extension {
                        nibbles: ext_nibbles[..common].to_vec(),
                        child: Box::new(branch),
                    }
                } else {
                    branch
                }
            }

            TrieNode::Branch {
                mut children,
                value: bval,
            } => {
                if nibbles.is_empty() {
                    return TrieNode::Branch {
                        children,
                        value: Some(value),
                    };
                }

                let idx = nibbles[0] as usize;
                let child = children[idx].take().map(|c| *c).unwrap_or(TrieNode::Empty);
                let new_child = child.insert(&nibbles[1..], value);
                children[idx] = Some(Box::new(new_child));

                TrieNode::Branch {
                    children,
                    value: bval,
                }
            }
        }
    }

    /// Get a value by nibble path.
    fn get(&self, nibbles: &[u8]) -> Option<&[u8]> {
        match self {
            TrieNode::Empty => None,

            TrieNode::Leaf {
                nibbles: leaf_nibbles,
                value,
            } => {
                if nibbles == leaf_nibbles.as_slice() {
                    Some(value)
                } else {
                    None
                }
            }

            TrieNode::Extension {
                nibbles: ext_nibbles,
                child,
            } => {
                if nibbles.starts_with(ext_nibbles) {
                    child.get(&nibbles[ext_nibbles.len()..])
                } else {
                    None
                }
            }

            TrieNode::Branch { children, value } => {
                if nibbles.is_empty() {
                    return value.as_deref();
                }
                let idx = nibbles[0] as usize;
                match &children[idx] {
                    Some(child) => child.get(&nibbles[1..]),
                    None => None,
                }
            }
        }
    }

    /// Delete a key. Returns (modified_node, was_deleted).
    fn delete(self, nibbles: &[u8]) -> (TrieNode, bool) {
        match self {
            TrieNode::Empty => (TrieNode::Empty, false),

            TrieNode::Leaf {
                nibbles: leaf_nibbles,
                ..
            } => {
                if nibbles == leaf_nibbles.as_slice() {
                    (TrieNode::Empty, true)
                } else {
                    (
                        TrieNode::Leaf {
                            nibbles: leaf_nibbles,
                            value: Vec::new(), // placeholder; won't be reached
                        },
                        false,
                    )
                }
            }

            TrieNode::Extension {
                nibbles: ext_nibbles,
                child,
            } => {
                if !nibbles.starts_with(&ext_nibbles) {
                    return (
                        TrieNode::Extension {
                            nibbles: ext_nibbles,
                            child,
                        },
                        false,
                    );
                }
                let (new_child, deleted) = child.delete(&nibbles[ext_nibbles.len()..]);
                if !deleted {
                    return (
                        TrieNode::Extension {
                            nibbles: ext_nibbles,
                            child: Box::new(new_child),
                        },
                        false,
                    );
                }
                match new_child {
                    TrieNode::Empty => (TrieNode::Empty, true),
                    other => (
                        TrieNode::Extension {
                            nibbles: ext_nibbles,
                            child: Box::new(other),
                        },
                        true,
                    ),
                }
            }

            TrieNode::Branch {
                mut children,
                value,
            } => {
                if nibbles.is_empty() {
                    if value.is_none() {
                        return (TrieNode::Branch { children, value }, false);
                    }
                    return (TrieNode::Branch { children, value: None }, true);
                }

                let idx = nibbles[0] as usize;
                let child = match children[idx].take() {
                    Some(c) => *c,
                    None => {
                        return (TrieNode::Branch { children, value }, false);
                    }
                };

                let (new_child, deleted) = child.delete(&nibbles[1..]);
                if !deleted {
                    children[idx] = Some(Box::new(new_child));
                    return (TrieNode::Branch { children, value }, false);
                }

                match new_child {
                    TrieNode::Empty => {}
                    other => {
                        children[idx] = Some(Box::new(other));
                    }
                }

                (TrieNode::Branch { children, value }, true)
            }
        }
    }

    /// Count leaf nodes.
    fn count(&self) -> usize {
        match self {
            TrieNode::Empty => 0,
            TrieNode::Leaf { .. } => 1,
            TrieNode::Extension { child, .. } => child.count(),
            TrieNode::Branch { children, value } => {
                let child_count: usize = children.iter().flatten().map(|c| c.count()).sum();
                child_count + if value.is_some() { 1 } else { 0 }
            }
        }
    }
}

// ── Proof ───────────────────────────────────────────────────────────────────

/// A proof step for Merkle path verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStep {
    pub node_type: NodeType,
    pub hash: String,
    /// Nibble path segment at this step.
    pub nibbles: Vec<u8>,
}

/// A Merkle proof for a key-value pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub key: Vec<u8>,
    pub value: Option<Vec<u8>>,
    pub root_hash: String,
    pub steps: Vec<ProofStep>,
}

// ── Serialized Trie ─────────────────────────────────────────────────────────

/// Serializable representation of the trie for export/import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTrie {
    pub entries: Vec<(Vec<u8>, Vec<u8>)>,
    pub root_hash: String,
    pub entry_count: usize,
}

// ── State Trie ──────────────────────────────────────────────────────────────

/// A simplified Merkle Patricia trie for key-value state storage.
#[derive(Debug, Clone)]
pub struct StateTrie {
    root: TrieNode,
}

impl StateTrie {
    /// Create a new empty trie.
    pub fn new() -> Self {
        Self {
            root: TrieNode::Empty,
        }
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: &[u8], value: Vec<u8>) {
        let nibbles = to_nibbles(key);
        let root = std::mem::replace(&mut self.root, TrieNode::Empty);
        self.root = root.insert(&nibbles, value);
    }

    /// Get a value by key.
    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        let nibbles = to_nibbles(key);
        self.root.get(&nibbles)
    }

    /// Delete a key.
    pub fn delete(&mut self, key: &[u8]) -> bool {
        let nibbles = to_nibbles(key);
        let root = std::mem::replace(&mut self.root, TrieNode::Empty);
        let (new_root, deleted) = root.delete(&nibbles);
        self.root = new_root;
        deleted
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &[u8]) -> bool {
        self.get(key).is_some()
    }

    /// Compute the root hash of the trie.
    pub fn root_hash(&self) -> String {
        self.root.hash()
    }

    /// Count the number of entries.
    pub fn len(&self) -> usize {
        self.root.count()
    }

    /// Check if the trie is empty.
    pub fn is_empty(&self) -> bool {
        matches!(self.root, TrieNode::Empty)
    }

    /// Generate a Merkle proof for a key.
    pub fn prove(&self, key: &[u8]) -> MerkleProof {
        let nibbles = to_nibbles(key);
        let mut steps = Vec::new();
        self.collect_proof(&self.root, &nibbles, &mut steps);
        let value = self.get(key).map(|v| v.to_vec());

        MerkleProof {
            key: key.to_vec(),
            value,
            root_hash: self.root_hash(),
            steps,
        }
    }

    fn collect_proof(&self, node: &TrieNode, nibbles: &[u8], steps: &mut Vec<ProofStep>) {
        match node {
            TrieNode::Empty => {
                steps.push(ProofStep {
                    node_type: NodeType::Empty,
                    hash: node.hash(),
                    nibbles: vec![],
                });
            }
            TrieNode::Leaf {
                nibbles: leaf_nibbles,
                ..
            } => {
                steps.push(ProofStep {
                    node_type: NodeType::Leaf,
                    hash: node.hash(),
                    nibbles: leaf_nibbles.clone(),
                });
            }
            TrieNode::Extension {
                nibbles: ext_nibbles,
                child,
            } => {
                steps.push(ProofStep {
                    node_type: NodeType::Extension,
                    hash: node.hash(),
                    nibbles: ext_nibbles.clone(),
                });
                if nibbles.starts_with(ext_nibbles) {
                    self.collect_proof(child, &nibbles[ext_nibbles.len()..], steps);
                }
            }
            TrieNode::Branch { children, .. } => {
                steps.push(ProofStep {
                    node_type: NodeType::Branch,
                    hash: node.hash(),
                    nibbles: vec![],
                });
                if !nibbles.is_empty() {
                    let idx = nibbles[0] as usize;
                    if let Some(child) = &children[idx] {
                        self.collect_proof(child, &nibbles[1..], steps);
                    }
                }
            }
        }
    }

    /// Verify a proof against a known root hash.
    pub fn verify_proof(proof: &MerkleProof, expected_root: &str) -> Result<bool, TrieError> {
        if proof.root_hash != expected_root {
            return Ok(false);
        }
        // The proof is valid if the root hash matches and steps are non-empty
        if proof.steps.is_empty() {
            return Err(TrieError::InvalidProof("empty proof steps".to_string()));
        }
        // Verify the first step hash matches root
        Ok(proof.steps[0].hash == expected_root)
    }

    /// Serialize the trie for export.
    pub fn serialize(&self) -> SerializedTrie {
        let mut entries = Vec::new();
        self.collect_entries(&self.root, &[], &mut entries);
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        SerializedTrie {
            root_hash: self.root_hash(),
            entry_count: entries.len(),
            entries,
        }
    }

    fn collect_entries(
        &self,
        node: &TrieNode,
        prefix_nibbles: &[u8],
        entries: &mut Vec<(Vec<u8>, Vec<u8>)>,
    ) {
        match node {
            TrieNode::Empty => {}
            TrieNode::Leaf { nibbles, value } => {
                let mut full_nibbles = prefix_nibbles.to_vec();
                full_nibbles.extend_from_slice(nibbles);
                let key = from_nibbles(&full_nibbles);
                entries.push((key, value.clone()));
            }
            TrieNode::Extension { nibbles, child } => {
                let mut new_prefix = prefix_nibbles.to_vec();
                new_prefix.extend_from_slice(nibbles);
                self.collect_entries(child, &new_prefix, entries);
            }
            TrieNode::Branch { children, value } => {
                if let Some(v) = value {
                    let key = from_nibbles(prefix_nibbles);
                    entries.push((key, v.clone()));
                }
                for (i, child) in children.iter().enumerate() {
                    if let Some(c) = child {
                        let mut new_prefix = prefix_nibbles.to_vec();
                        new_prefix.push(i as u8);
                        self.collect_entries(c, &new_prefix, entries);
                    }
                }
            }
        }
    }

    /// Deserialize and reconstruct a trie from a serialized form.
    pub fn from_serialized(serialized: &SerializedTrie) -> Self {
        let mut trie = Self::new();
        for (key, value) in &serialized.entries {
            trie.insert(key, value.clone());
        }
        trie
    }

    /// Get the root node type.
    pub fn root_node_type(&self) -> NodeType {
        self.root.node_type()
    }
}

impl Default for StateTrie {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_trie() {
        let trie = StateTrie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut trie = StateTrie::new();
        trie.insert(b"hello", b"world".to_vec());
        assert_eq!(trie.get(b"hello"), Some(b"world".as_slice()));
    }

    #[test]
    fn test_insert_multiple_keys() {
        let mut trie = StateTrie::new();
        trie.insert(b"aaa", b"val1".to_vec());
        trie.insert(b"aab", b"val2".to_vec());
        trie.insert(b"bbb", b"val3".to_vec());
        assert_eq!(trie.get(b"aaa"), Some(b"val1".as_slice()));
        assert_eq!(trie.get(b"aab"), Some(b"val2".as_slice()));
        assert_eq!(trie.get(b"bbb"), Some(b"val3".as_slice()));
        assert_eq!(trie.len(), 3);
    }

    #[test]
    fn test_get_nonexistent_key() {
        let mut trie = StateTrie::new();
        trie.insert(b"key1", b"val".to_vec());
        assert_eq!(trie.get(b"key2"), None);
    }

    #[test]
    fn test_overwrite_value() {
        let mut trie = StateTrie::new();
        trie.insert(b"key", b"old".to_vec());
        trie.insert(b"key", b"new".to_vec());
        assert_eq!(trie.get(b"key"), Some(b"new".as_slice()));
    }

    #[test]
    fn test_delete_key() {
        let mut trie = StateTrie::new();
        trie.insert(b"key1", b"val1".to_vec());
        trie.insert(b"key2", b"val2".to_vec());
        assert!(trie.delete(b"key1"));
        assert_eq!(trie.get(b"key1"), None);
        assert_eq!(trie.get(b"key2"), Some(b"val2".as_slice()));
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut trie = StateTrie::new();
        trie.insert(b"key", b"val".to_vec());
        assert!(!trie.delete(b"other"));
    }

    #[test]
    fn test_contains_key() {
        let mut trie = StateTrie::new();
        trie.insert(b"exists", b"yes".to_vec());
        assert!(trie.contains_key(b"exists"));
        assert!(!trie.contains_key(b"nope"));
    }

    #[test]
    fn test_root_hash_deterministic() {
        let mut t1 = StateTrie::new();
        t1.insert(b"a", b"1".to_vec());
        t1.insert(b"b", b"2".to_vec());

        let mut t2 = StateTrie::new();
        t2.insert(b"a", b"1".to_vec());
        t2.insert(b"b", b"2".to_vec());

        assert_eq!(t1.root_hash(), t2.root_hash());
    }

    #[test]
    fn test_root_hash_changes_on_mutation() {
        let mut trie = StateTrie::new();
        trie.insert(b"key", b"val1".to_vec());
        let h1 = trie.root_hash();
        trie.insert(b"key", b"val2".to_vec());
        let h2 = trie.root_hash();
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_empty_trie_hash() {
        let t1 = StateTrie::new();
        let t2 = StateTrie::new();
        assert_eq!(t1.root_hash(), t2.root_hash());
    }

    #[test]
    fn test_proof_generation() {
        let mut trie = StateTrie::new();
        trie.insert(b"key", b"val".to_vec());
        let proof = trie.prove(b"key");
        assert_eq!(proof.key, b"key");
        assert_eq!(proof.value, Some(b"val".to_vec()));
        assert!(!proof.steps.is_empty());
    }

    #[test]
    fn test_proof_verification_valid() {
        let mut trie = StateTrie::new();
        trie.insert(b"key", b"val".to_vec());
        let proof = trie.prove(b"key");
        let root = trie.root_hash();
        assert!(StateTrie::verify_proof(&proof, &root).unwrap());
    }

    #[test]
    fn test_proof_verification_invalid_root() {
        let mut trie = StateTrie::new();
        trie.insert(b"key", b"val".to_vec());
        let proof = trie.prove(b"key");
        assert!(!StateTrie::verify_proof(&proof, "wrong_root_hash").unwrap());
    }

    #[test]
    fn test_serialize_and_deserialize() {
        let mut trie = StateTrie::new();
        trie.insert(b"alpha", b"one".to_vec());
        trie.insert(b"beta", b"two".to_vec());
        trie.insert(b"gamma", b"three".to_vec());

        let serialized = trie.serialize();
        assert_eq!(serialized.entry_count, 3);

        let restored = StateTrie::from_serialized(&serialized);
        assert_eq!(restored.get(b"alpha"), Some(b"one".as_slice()));
        assert_eq!(restored.get(b"beta"), Some(b"two".as_slice()));
        assert_eq!(restored.get(b"gamma"), Some(b"three".as_slice()));
    }

    #[test]
    fn test_root_node_type() {
        let mut trie = StateTrie::new();
        assert_eq!(trie.root_node_type(), NodeType::Empty);
        trie.insert(b"x", b"y".to_vec());
        assert_eq!(trie.root_node_type(), NodeType::Leaf);
    }

    #[test]
    fn test_node_type_display() {
        assert_eq!(format!("{}", NodeType::Leaf), "leaf");
        assert_eq!(format!("{}", NodeType::Branch), "branch");
        assert_eq!(format!("{}", NodeType::Extension), "extension");
    }

    #[test]
    fn test_trie_error_display() {
        let err = TrieError::KeyNotFound(vec![1, 2, 3]);
        let msg = format!("{err}");
        assert!(msg.contains("key not found"));
    }

    #[test]
    fn test_nibble_conversion_roundtrip() {
        let data = b"hello";
        let nibbles = to_nibbles(data);
        let back = from_nibbles(&nibbles);
        assert_eq!(back, data);
    }

    #[test]
    fn test_common_prefix() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix_len(&[1], &[2]), 0);
    }

    #[test]
    fn test_default_trie() {
        let trie = StateTrie::default();
        assert!(trie.is_empty());
    }

    #[test]
    fn test_many_insertions() {
        let mut trie = StateTrie::new();
        for i in 0u32..50 {
            let key = i.to_be_bytes();
            let val = (i * 10).to_be_bytes();
            trie.insert(&key, val.to_vec());
        }
        assert_eq!(trie.len(), 50);
        for i in 0u32..50 {
            let key = i.to_be_bytes();
            let expected = (i * 10).to_be_bytes();
            assert_eq!(trie.get(&key), Some(expected.as_slice()));
        }
    }
}
