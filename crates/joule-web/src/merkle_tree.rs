//! Merkle tree — binary hash tree with inclusion proofs, root computation,
//! partial tree updates, depth/leaf queries, proof verification, visualization,
//! and serialization.

use serde::{Deserialize, Serialize};

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

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = SHA256_H0;
    let total_len = data.len() as u64;
    let mut buf = data.to_vec();
    buf.push(0x80);
    while buf.len() % 64 != 56 {
        buf.push(0x00);
    }
    buf.extend_from_slice(&(total_len * 8).to_be_bytes());
    for chunk in buf.chunks_exact(64) {
        let block: [u8; 64] = chunk.try_into().unwrap();
        sha256_process_block(&mut state, &block);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(left);
    combined[32..].copy_from_slice(right);
    sha256(&combined)
}

fn hash_leaf(data: &[u8]) -> [u8; 32] {
    // Leaf prefix 0x00 to prevent second-preimage attacks.
    let mut prefixed = Vec::with_capacity(1 + data.len());
    prefixed.push(0x00);
    prefixed.extend_from_slice(data);
    sha256(&prefixed)
}

fn hash_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    // Node prefix 0x01 to distinguish from leaves.
    let mut combined = Vec::with_capacity(65);
    combined.push(0x01);
    combined.extend_from_slice(left);
    combined.extend_from_slice(right);
    sha256(&combined)
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Merkle tree errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MerkleError {
    /// Empty data set.
    EmptyTree,
    /// Leaf index out of range.
    IndexOutOfRange { index: usize, count: usize },
    /// Invalid proof.
    InvalidProof(String),
}

impl std::fmt::Display for MerkleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTree => write!(f, "cannot build tree from empty data"),
            Self::IndexOutOfRange { index, count } => {
                write!(f, "leaf index {index} out of range (count={count})")
            }
            Self::InvalidProof(s) => write!(f, "invalid proof: {s}"),
        }
    }
}

impl std::error::Error for MerkleError {}

// ── Proof ───────────────────────────────────────────────────────────────────

/// Direction of a sibling in the proof path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofDirection {
    /// Sibling is on the left.
    Left,
    /// Sibling is on the right.
    Right,
}

/// A single step in a Merkle inclusion proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofStep {
    /// The sibling hash at this level.
    pub hash: Vec<u8>,
    /// Whether the sibling is on the left or right.
    pub direction: ProofDirection,
}

/// A Merkle inclusion proof for a specific leaf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Leaf index (0-based).
    pub leaf_index: usize,
    /// Leaf hash.
    pub leaf_hash: Vec<u8>,
    /// Proof path from leaf to root.
    pub steps: Vec<ProofStep>,
    /// Root hash that this proof targets.
    pub root_hash: Vec<u8>,
}

impl MerkleProof {
    /// Verify this proof: recompute root from leaf + path and compare.
    pub fn verify(&self) -> bool {
        let mut current = [0u8; 32];
        if self.leaf_hash.len() != 32 {
            return false;
        }
        current.copy_from_slice(&self.leaf_hash);

        for step in &self.steps {
            if step.hash.len() != 32 {
                return false;
            }
            let sibling: [u8; 32] = step.hash.as_slice().try_into().unwrap_or([0u8; 32]);
            current = match step.direction {
                ProofDirection::Left => hash_node(&sibling, &current),
                ProofDirection::Right => hash_node(&current, &sibling),
            };
        }

        if self.root_hash.len() != 32 {
            return false;
        }
        current[..] == self.root_hash[..]
    }

    /// Verify this proof against a specific root hash.
    pub fn verify_against(&self, root: &[u8; 32]) -> bool {
        let mut current = [0u8; 32];
        if self.leaf_hash.len() != 32 {
            return false;
        }
        current.copy_from_slice(&self.leaf_hash);

        for step in &self.steps {
            if step.hash.len() != 32 {
                return false;
            }
            let sibling: [u8; 32] = step.hash.as_slice().try_into().unwrap_or([0u8; 32]);
            current = match step.direction {
                ProofDirection::Left => hash_node(&sibling, &current),
                ProofDirection::Right => hash_node(&current, &sibling),
            };
        }
        current == *root
    }
}

// ── Merkle Tree ─────────────────────────────────────────────────────────────

/// A binary Merkle tree stored as a flat array.
///
/// Nodes are stored bottom-up: leaves at the end, root at index 0.
/// For N leaves, the total node count is 2*next_power_of_two(N) - 1.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    /// All nodes in level order. Root at 0.
    nodes: Vec<[u8; 32]>,
    /// Number of actual leaf data items (not padded).
    leaf_count: usize,
    /// Total number of leaves including padding to power-of-2.
    padded_leaf_count: usize,
}

impl MerkleTree {
    /// Build a Merkle tree from a list of leaf data items.
    pub fn build(leaves: &[&[u8]]) -> Result<Self, MerkleError> {
        if leaves.is_empty() {
            return Err(MerkleError::EmptyTree);
        }

        let leaf_count = leaves.len();
        // Pad to next power of 2.
        let padded = leaf_count.next_power_of_two();
        let total_nodes = 2 * padded - 1;

        let mut nodes = vec![[0u8; 32]; total_nodes];

        // Fill leaves (last `padded` entries).
        let leaf_start = padded - 1;
        for (i, data) in leaves.iter().enumerate() {
            nodes[leaf_start + i] = hash_leaf(data);
        }
        // Duplicate the last real leaf for padding slots.
        if leaf_count < padded {
            let last_leaf = nodes[leaf_start + leaf_count - 1];
            for i in leaf_count..padded {
                nodes[leaf_start + i] = last_leaf;
            }
        }

        // Build internal nodes bottom-up.
        for i in (0..leaf_start).rev() {
            let left = nodes[2 * i + 1];
            let right = nodes[2 * i + 2];
            nodes[i] = hash_node(&left, &right);
        }

        Ok(Self {
            nodes,
            leaf_count,
            padded_leaf_count: padded,
        })
    }

    /// Build from pre-hashed leaves.
    pub fn build_from_hashes(leaf_hashes: &[[u8; 32]]) -> Result<Self, MerkleError> {
        if leaf_hashes.is_empty() {
            return Err(MerkleError::EmptyTree);
        }

        let leaf_count = leaf_hashes.len();
        let padded = leaf_count.next_power_of_two();
        let total_nodes = 2 * padded - 1;

        let mut nodes = vec![[0u8; 32]; total_nodes];

        let leaf_start = padded - 1;
        for (i, h) in leaf_hashes.iter().enumerate() {
            nodes[leaf_start + i] = *h;
        }
        if leaf_count < padded {
            let last = nodes[leaf_start + leaf_count - 1];
            for i in leaf_count..padded {
                nodes[leaf_start + i] = last;
            }
        }

        for i in (0..leaf_start).rev() {
            let left = nodes[2 * i + 1];
            let right = nodes[2 * i + 2];
            nodes[i] = hash_node(&left, &right);
        }

        Ok(Self {
            nodes,
            leaf_count,
            padded_leaf_count: padded,
        })
    }

    /// Root hash.
    pub fn root(&self) -> &[u8; 32] {
        &self.nodes[0]
    }

    /// Root hash as hex string.
    pub fn root_hex(&self) -> String {
        bytes_to_hex(&self.nodes[0])
    }

    /// Number of actual leaves (not padded).
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Tree depth (number of levels excluding root).
    pub fn depth(&self) -> usize {
        if self.padded_leaf_count <= 1 {
            return 0;
        }
        // depth = log2(padded_leaf_count)
        let mut d = 0;
        let mut n = self.padded_leaf_count;
        while n > 1 {
            n >>= 1;
            d += 1;
        }
        d
    }

    /// Total number of nodes (internal + leaves).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the hash at a specific node index.
    pub fn node_hash(&self, index: usize) -> Option<&[u8; 32]> {
        self.nodes.get(index)
    }

    /// Get leaf hash by leaf index (0-based).
    pub fn leaf_hash(&self, index: usize) -> Result<&[u8; 32], MerkleError> {
        if index >= self.leaf_count {
            return Err(MerkleError::IndexOutOfRange {
                index,
                count: self.leaf_count,
            });
        }
        let leaf_start = self.padded_leaf_count - 1;
        Ok(&self.nodes[leaf_start + index])
    }

    /// Generate an inclusion proof for the leaf at `index`.
    pub fn proof(&self, index: usize) -> Result<MerkleProof, MerkleError> {
        if index >= self.leaf_count {
            return Err(MerkleError::IndexOutOfRange {
                index,
                count: self.leaf_count,
            });
        }

        let leaf_start = self.padded_leaf_count - 1;
        let mut node_idx = leaf_start + index;
        let leaf_hash = self.nodes[node_idx].to_vec();
        let mut steps = Vec::new();

        while node_idx > 0 {
            let sibling_idx = if node_idx % 2 == 1 {
                node_idx + 1
            } else {
                node_idx - 1
            };
            let direction = if node_idx % 2 == 1 {
                ProofDirection::Right
            } else {
                ProofDirection::Left
            };
            steps.push(ProofStep {
                hash: self.nodes[sibling_idx].to_vec(),
                direction,
            });
            node_idx = (node_idx - 1) / 2;
        }

        Ok(MerkleProof {
            leaf_index: index,
            leaf_hash,
            steps,
            root_hash: self.nodes[0].to_vec(),
        })
    }

    /// Update a single leaf and recompute affected hashes.
    pub fn update_leaf(&mut self, index: usize, data: &[u8]) -> Result<(), MerkleError> {
        if index >= self.leaf_count {
            return Err(MerkleError::IndexOutOfRange {
                index,
                count: self.leaf_count,
            });
        }

        let leaf_start = self.padded_leaf_count - 1;
        let mut node_idx = leaf_start + index;
        self.nodes[node_idx] = hash_leaf(data);

        // Propagate changes up to root.
        while node_idx > 0 {
            let parent = (node_idx - 1) / 2;
            let left = self.nodes[2 * parent + 1];
            let right = self.nodes[2 * parent + 2];
            self.nodes[parent] = hash_node(&left, &right);
            node_idx = parent;
        }

        Ok(())
    }

    /// Produce an ASCII visualization of the tree (compact).
    pub fn visualize(&self) -> String {
        let mut lines = Vec::new();
        let depth = self.depth();
        let mut level_start = 0;
        let mut level_size = 1;

        for level in 0..=depth {
            let indent = " ".repeat((depth - level) * 4);
            let mut level_line = String::new();
            for i in 0..level_size {
                let idx = level_start + i;
                if idx < self.nodes.len() {
                    let hex = bytes_to_hex(&self.nodes[idx]);
                    let short = &hex[..8];
                    if i > 0 {
                        level_line.push_str("  ");
                    }
                    level_line.push_str(short);
                }
            }
            lines.push(format!("{indent}{level_line}"));
            level_start += level_size;
            level_size *= 2;
        }
        lines.join("\n")
    }

    /// Verify the entire tree is internally consistent.
    pub fn verify_integrity(&self) -> bool {
        let leaf_start = self.padded_leaf_count - 1;
        for i in 0..leaf_start {
            let left = self.nodes[2 * i + 1];
            let right = self.nodes[2 * i + 2];
            let expected = hash_node(&left, &right);
            if self.nodes[i] != expected {
                return false;
            }
        }
        true
    }

    /// Serialize the tree to a JSON-compatible structure.
    pub fn to_serializable(&self) -> SerializableMerkleTree {
        SerializableMerkleTree {
            nodes: self.nodes.iter().map(|n| bytes_to_hex(n)).collect(),
            leaf_count: self.leaf_count,
            padded_leaf_count: self.padded_leaf_count,
        }
    }

    /// Deserialize from the serializable structure.
    pub fn from_serializable(s: &SerializableMerkleTree) -> Result<Self, MerkleError> {
        let mut nodes = Vec::with_capacity(s.nodes.len());
        for hex in &s.nodes {
            if hex.len() != 64 {
                return Err(MerkleError::InvalidProof(format!("bad hex length: {}", hex.len())));
            }
            let mut arr = [0u8; 32];
            for i in 0..32 {
                arr[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                    .map_err(|_| MerkleError::InvalidProof("bad hex".to_string()))?;
            }
            nodes.push(arr);
        }
        Ok(Self {
            nodes,
            leaf_count: s.leaf_count,
            padded_leaf_count: s.padded_leaf_count,
        })
    }
}

/// JSON-serializable representation of a Merkle tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableMerkleTree {
    pub nodes: Vec<String>,
    pub leaf_count: usize,
    pub padded_leaf_count: usize,
}

// ── Standalone verification ─────────────────────────────────────────────────

/// Verify a Merkle proof against a known root hash and leaf data.
pub fn verify_proof(
    root: &[u8; 32],
    leaf_data: &[u8],
    proof: &MerkleProof,
) -> bool {
    let leaf_h = hash_leaf(leaf_data);
    if proof.leaf_hash.len() != 32 || proof.leaf_hash[..] != leaf_h[..] {
        return false;
    }
    proof.verify_against(root)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_single_leaf() {
        let tree = MerkleTree::build(&[b"hello"]).unwrap();
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.depth(), 0);
        assert_eq!(tree.node_count(), 1);
    }

    #[test]
    fn test_build_two_leaves() {
        let tree = MerkleTree::build(&[b"a", b"b"]).unwrap();
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.depth(), 1);
        assert_eq!(tree.node_count(), 3);
    }

    #[test]
    fn test_build_four_leaves() {
        let tree = MerkleTree::build(&[b"a", b"b", b"c", b"d"]).unwrap();
        assert_eq!(tree.leaf_count(), 4);
        assert_eq!(tree.depth(), 2);
        assert_eq!(tree.node_count(), 7);
    }

    #[test]
    fn test_build_non_power_of_two() {
        // 3 leaves => padded to 4
        let tree = MerkleTree::build(&[b"x", b"y", b"z"]).unwrap();
        assert_eq!(tree.leaf_count(), 3);
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    fn test_build_empty_errors() {
        let empty: Vec<&[u8]> = vec![];
        assert!(MerkleTree::build(&empty).is_err());
    }

    #[test]
    fn test_root_deterministic() {
        let t1 = MerkleTree::build(&[b"a", b"b", b"c", b"d"]).unwrap();
        let t2 = MerkleTree::build(&[b"a", b"b", b"c", b"d"]).unwrap();
        assert_eq!(t1.root(), t2.root());
    }

    #[test]
    fn test_root_changes_with_data() {
        let t1 = MerkleTree::build(&[b"a", b"b"]).unwrap();
        let t2 = MerkleTree::build(&[b"a", b"c"]).unwrap();
        assert_ne!(t1.root(), t2.root());
    }

    #[test]
    fn test_proof_verify_two_leaves() {
        let tree = MerkleTree::build(&[b"alpha", b"beta"]).unwrap();
        let proof = tree.proof(0).unwrap();
        assert!(proof.verify());
        let proof1 = tree.proof(1).unwrap();
        assert!(proof1.verify());
    }

    #[test]
    fn test_proof_verify_four_leaves() {
        let tree = MerkleTree::build(&[b"a", b"b", b"c", b"d"]).unwrap();
        for i in 0..4 {
            let proof = tree.proof(i).unwrap();
            assert!(proof.verify(), "proof for leaf {i} should verify");
        }
    }

    #[test]
    fn test_proof_out_of_range() {
        let tree = MerkleTree::build(&[b"only"]).unwrap();
        assert!(tree.proof(1).is_err());
    }

    #[test]
    fn test_proof_verify_against_root() {
        let tree = MerkleTree::build(&[b"x", b"y", b"z"]).unwrap();
        let proof = tree.proof(2).unwrap();
        assert!(proof.verify_against(tree.root()));
    }

    #[test]
    fn test_standalone_verify() {
        let tree = MerkleTree::build(&[b"foo", b"bar"]).unwrap();
        let proof = tree.proof(0).unwrap();
        assert!(verify_proof(tree.root(), b"foo", &proof));
        // Wrong data should fail.
        assert!(!verify_proof(tree.root(), b"baz", &proof));
    }

    #[test]
    fn test_update_leaf() {
        let mut tree = MerkleTree::build(&[b"a", b"b", b"c", b"d"]).unwrap();
        let old_root = *tree.root();
        tree.update_leaf(2, b"C").unwrap();
        assert_ne!(*tree.root(), old_root);
        assert!(tree.verify_integrity());
    }

    #[test]
    fn test_update_leaf_out_of_range() {
        let mut tree = MerkleTree::build(&[b"a"]).unwrap();
        assert!(tree.update_leaf(5, b"x").is_err());
    }

    #[test]
    fn test_verify_integrity() {
        let tree = MerkleTree::build(&[b"a", b"b", b"c", b"d"]).unwrap();
        assert!(tree.verify_integrity());
    }

    #[test]
    fn test_leaf_hash() {
        let tree = MerkleTree::build(&[b"hello", b"world"]).unwrap();
        let h = tree.leaf_hash(0).unwrap();
        assert_eq!(*h, hash_leaf(b"hello"));
        assert!(tree.leaf_hash(10).is_err());
    }

    #[test]
    fn test_visualize() {
        let tree = MerkleTree::build(&[b"a", b"b"]).unwrap();
        let viz = tree.visualize();
        assert!(!viz.is_empty());
        // Should have multiple lines for depth > 0.
        assert!(viz.lines().count() >= 2);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let tree = MerkleTree::build(&[b"alpha", b"beta", b"gamma"]).unwrap();
        let s = tree.to_serializable();
        let json = serde_json::to_string(&s).unwrap();
        let s2: SerializableMerkleTree = serde_json::from_str(&json).unwrap();
        let tree2 = MerkleTree::from_serializable(&s2).unwrap();
        assert_eq!(tree.root(), tree2.root());
        assert_eq!(tree.leaf_count(), tree2.leaf_count());
    }

    #[test]
    fn test_build_from_hashes() {
        let leaves = vec![hash_leaf(b"a"), hash_leaf(b"b")];
        let tree = MerkleTree::build_from_hashes(&leaves).unwrap();
        assert_eq!(tree.leaf_count(), 2);
        assert!(tree.verify_integrity());
    }

    #[test]
    fn test_root_hex() {
        let tree = MerkleTree::build(&[b"test"]).unwrap();
        let hex = tree.root_hex();
        assert_eq!(hex.len(), 64);
    }

    #[test]
    fn test_update_and_reprove() {
        let mut tree = MerkleTree::build(&[b"a", b"b", b"c", b"d"]).unwrap();
        tree.update_leaf(1, b"B").unwrap();
        let proof = tree.proof(1).unwrap();
        assert!(proof.verify());
    }

    #[test]
    fn test_many_leaves() {
        let data: Vec<Vec<u8>> = (0u32..32).map(|i| i.to_le_bytes().to_vec()).collect();
        let refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let tree = MerkleTree::build(&refs).unwrap();
        assert_eq!(tree.leaf_count(), 32);
        assert_eq!(tree.depth(), 5);
        for i in 0..32 {
            let proof = tree.proof(i).unwrap();
            assert!(proof.verify());
        }
    }
}
