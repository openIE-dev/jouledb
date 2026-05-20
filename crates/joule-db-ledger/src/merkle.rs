use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A SHA-256 binary Merkle tree built from leaf hashes.
///
/// Leaves are stored at the bottom level; interior nodes are computed
/// as `SHA-256(left_child || right_child)`. If the leaf count is not
/// a power of 2, zero-hash padding is added.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    /// Flat array of all tree nodes. Index 0 = root.
    /// For a tree with N leaves (padded to power-of-2), total nodes = 2*N - 1.
    /// Children of node i: left = 2*i + 1, right = 2*i + 2.
    nodes: Vec<[u8; 32]>,
    /// Number of original (non-padded) leaves.
    leaf_count: usize,
    /// Total leaves including padding (always a power of 2).
    padded_count: usize,
}

/// A Merkle inclusion proof for a single leaf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Index of the leaf in the original array.
    pub leaf_index: usize,
    /// Hex-encoded hash of the leaf.
    pub leaf_hash: String,
    /// Sibling hashes from leaf to root, with direction.
    pub siblings: Vec<ProofNode>,
    /// Hex-encoded Merkle root this proof resolves to.
    pub root: String,
}

/// A sibling node in a Merkle proof path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    /// Hex-encoded hash of the sibling.
    pub hash: String,
    /// Position of the sibling relative to the current node.
    pub position: Position,
}

/// Position of a sibling in the proof path.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Position {
    Left,
    Right,
}

const ZERO_HASH: [u8; 32] = [0u8; 32];

/// Compute SHA-256(left || right).
fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Round up to the next power of 2.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n.next_power_of_two()
}

impl MerkleTree {
    /// Build a Merkle tree from an ordered slice of leaf hashes.
    ///
    /// If the count is not a power of 2, pads with zero-hashes.
    /// Returns a tree with a single root even for 0 or 1 leaves.
    pub fn from_leaves(leaves: &[[u8; 32]]) -> Self {
        let leaf_count = leaves.len();
        let padded_count = if leaf_count == 0 {
            1
        } else {
            next_power_of_two(leaf_count)
        };

        let total_nodes = 2 * padded_count - 1;
        let mut nodes = vec![ZERO_HASH; total_nodes];

        // Place leaves at the bottom level.
        // Bottom level starts at index (padded_count - 1).
        let leaf_start = padded_count - 1;
        for (i, leaf) in leaves.iter().enumerate() {
            nodes[leaf_start + i] = *leaf;
        }
        // Remaining positions stay as ZERO_HASH (padding).

        // Build interior nodes bottom-up.
        if padded_count > 1 {
            for i in (0..leaf_start).rev() {
                let left = nodes[2 * i + 1];
                let right = nodes[2 * i + 2];
                nodes[i] = hash_pair(&left, &right);
            }
        }

        MerkleTree {
            nodes,
            leaf_count,
            padded_count,
        }
    }

    /// Return the Merkle root hash.
    pub fn root(&self) -> [u8; 32] {
        self.nodes[0]
    }

    /// Return the hex-encoded Merkle root.
    pub fn root_hex(&self) -> String {
        hex::encode(self.root())
    }

    /// Number of original (non-padded) leaves.
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Generate an inclusion proof for the leaf at `index`.
    ///
    /// Returns `None` if `index >= leaf_count`.
    pub fn proof(&self, index: usize) -> Option<MerkleProof> {
        if index >= self.leaf_count {
            return None;
        }

        let leaf_start = self.padded_count - 1;
        let mut current = leaf_start + index;
        let leaf_hash = hex::encode(self.nodes[current]);
        let mut siblings = Vec::new();

        while current > 0 {
            let parent = (current - 1) / 2;
            let left_child = 2 * parent + 1;
            let right_child = 2 * parent + 2;

            if current == left_child {
                // Current is left child, sibling is on the right
                siblings.push(ProofNode {
                    hash: hex::encode(self.nodes[right_child]),
                    position: Position::Right,
                });
            } else {
                // Current is right child, sibling is on the left
                siblings.push(ProofNode {
                    hash: hex::encode(self.nodes[left_child]),
                    position: Position::Left,
                });
            }
            current = parent;
        }

        Some(MerkleProof {
            leaf_index: index,
            leaf_hash,
            siblings,
            root: self.root_hex(),
        })
    }
}

impl MerkleProof {
    /// Verify this proof against the expected root hash.
    pub fn verify(&self, expected_root: &[u8; 32]) -> bool {
        let mut current = match hex::decode(&self.leaf_hash) {
            Ok(bytes) if bytes.len() == 32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                arr
            }
            _ => return false,
        };

        for sibling in &self.siblings {
            let sibling_hash = match hex::decode(&sibling.hash) {
                Ok(bytes) if bytes.len() == 32 => {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&bytes);
                    arr
                }
                _ => return false,
            };

            current = match sibling.position {
                Position::Right => hash_pair(&current, &sibling_hash),
                Position::Left => hash_pair(&sibling_hash, &current),
            };
        }

        current == *expected_root
    }

    /// Verify against the root stored in this proof.
    pub fn verify_self(&self) -> bool {
        match hex::decode(&self.root) {
            Ok(bytes) if bytes.len() == 32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                self.verify(&arr)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_leaf(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    #[test]
    fn single_leaf() {
        let leaf = make_leaf(b"hello");
        let tree = MerkleTree::from_leaves(&[leaf]);
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.root(), leaf);
    }

    #[test]
    fn two_leaves() {
        let a = make_leaf(b"a");
        let b = make_leaf(b"b");
        let tree = MerkleTree::from_leaves(&[a, b]);
        assert_eq!(tree.leaf_count(), 2);
        let expected_root = hash_pair(&a, &b);
        assert_eq!(tree.root(), expected_root);
    }

    #[test]
    fn four_leaves() {
        let leaves: Vec<[u8; 32]> = (0..4).map(|i| make_leaf(&[i])).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        assert_eq!(tree.leaf_count(), 4);

        let h01 = hash_pair(&leaves[0], &leaves[1]);
        let h23 = hash_pair(&leaves[2], &leaves[3]);
        let expected = hash_pair(&h01, &h23);
        assert_eq!(tree.root(), expected);
    }

    #[test]
    fn three_leaves_padded() {
        // 3 leaves -> padded to 4 with zero-hash
        let leaves: Vec<[u8; 32]> = (0..3).map(|i| make_leaf(&[i])).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        assert_eq!(tree.leaf_count(), 3);

        let h01 = hash_pair(&leaves[0], &leaves[1]);
        let h2z = hash_pair(&leaves[2], &ZERO_HASH);
        let expected = hash_pair(&h01, &h2z);
        assert_eq!(tree.root(), expected);
    }

    #[test]
    fn empty_tree() {
        let tree = MerkleTree::from_leaves(&[]);
        assert_eq!(tree.leaf_count(), 0);
        assert_eq!(tree.root(), ZERO_HASH);
    }

    #[test]
    fn proof_verifies_all_leaves() {
        let leaves: Vec<[u8; 32]> = (0..8).map(|i| make_leaf(&[i])).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let root = tree.root();

        for i in 0..8 {
            let proof = tree.proof(i).expect("proof should exist");
            assert!(proof.verify(&root), "proof for leaf {} should verify", i);
            assert!(
                proof.verify_self(),
                "self-verify for leaf {} should pass",
                i
            );
        }
    }

    #[test]
    fn proof_fails_wrong_root() {
        let leaves: Vec<[u8; 32]> = (0..4).map(|i| make_leaf(&[i])).collect();
        let tree = MerkleTree::from_leaves(&leaves);

        let proof = tree.proof(0).unwrap();
        let wrong_root = make_leaf(b"wrong");
        assert!(!proof.verify(&wrong_root));
    }

    #[test]
    fn proof_out_of_bounds() {
        let leaves: Vec<[u8; 32]> = (0..4).map(|i| make_leaf(&[i])).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        assert!(tree.proof(4).is_none());
        assert!(tree.proof(100).is_none());
    }

    #[test]
    fn proof_serialization_roundtrip() {
        let leaves: Vec<[u8; 32]> = (0..4).map(|i| make_leaf(&[i])).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let proof = tree.proof(2).unwrap();

        let json = serde_json::to_string(&proof).unwrap();
        let proof2: MerkleProof = serde_json::from_str(&json).unwrap();
        assert_eq!(proof.leaf_index, proof2.leaf_index);
        assert_eq!(proof.leaf_hash, proof2.leaf_hash);
        assert_eq!(proof.root, proof2.root);
        assert!(proof2.verify_self());
    }

    #[test]
    fn large_tree_16_leaves() {
        let leaves: Vec<[u8; 32]> = (0..16).map(|i| make_leaf(&[i as u8])).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        assert_eq!(tree.leaf_count(), 16);

        // All proofs should verify
        let root = tree.root();
        for i in 0..16 {
            let proof = tree.proof(i).unwrap();
            assert!(proof.verify(&root));
            // Depth should be log2(16) = 4
            assert_eq!(proof.siblings.len(), 4);
        }
    }
}
