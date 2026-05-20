//! Basic blockchain — blocks, chain validation, proof-of-work mining,
//! genesis block creation, integrity verification, and chain statistics.

use serde::{Deserialize, Serialize};
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

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from blockchain operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockchainError {
    /// Block index does not match expected sequence.
    InvalidIndex { expected: u64, got: u64 },
    /// Block's previous hash does not match.
    InvalidPrevHash { block_index: u64 },
    /// Block hash does not satisfy difficulty target.
    InsufficientWork { block_index: u64, difficulty: u32 },
    /// Block hash does not match computed hash.
    InvalidHash { block_index: u64 },
    /// Chain is empty (no genesis block).
    EmptyChain,
    /// Genesis block has non-zero index.
    InvalidGenesis,
}

impl fmt::Display for BlockchainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIndex { expected, got } => {
                write!(f, "invalid block index: expected {expected}, got {got}")
            }
            Self::InvalidPrevHash { block_index } => {
                write!(f, "block {block_index}: previous hash mismatch")
            }
            Self::InsufficientWork { block_index, difficulty } => {
                write!(f, "block {block_index}: hash does not meet difficulty {difficulty}")
            }
            Self::InvalidHash { block_index } => {
                write!(f, "block {block_index}: hash mismatch")
            }
            Self::EmptyChain => write!(f, "chain is empty"),
            Self::InvalidGenesis => write!(f, "genesis block must have index 0"),
        }
    }
}

impl std::error::Error for BlockchainError {}

// ── Block ───────────────────────────────────────────────────────────────────

/// A single block in the blockchain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Block {
    /// Sequential block index (0 = genesis).
    pub index: u64,
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Arbitrary block data payload.
    pub data: String,
    /// Hex-encoded hash of the previous block.
    pub prev_hash: String,
    /// Hex-encoded hash of this block.
    pub hash: String,
    /// Proof-of-work nonce.
    pub nonce: u64,
}

impl Block {
    /// Compute the hash for this block's content (excluding the stored hash).
    pub fn compute_hash(&self) -> String {
        let input = format!(
            "{}:{}:{}:{}:{}",
            self.index, self.timestamp, self.data, self.prev_hash, self.nonce,
        );
        bytes_to_hex(&sha256(input.as_bytes()))
    }

    /// Check whether this block's hash satisfies the given difficulty
    /// (number of leading hex zeros required).
    pub fn satisfies_difficulty(&self, difficulty: u32) -> bool {
        let prefix: String = std::iter::repeat('0').take(difficulty as usize).collect();
        self.hash.starts_with(&prefix)
    }

    /// Create a new block (not yet mined — nonce = 0, hash = empty).
    pub fn new(index: u64, timestamp: u64, data: impl Into<String>, prev_hash: impl Into<String>) -> Self {
        let mut block = Self {
            index,
            timestamp,
            data: data.into(),
            prev_hash: prev_hash.into(),
            hash: String::new(),
            nonce: 0,
        };
        block.hash = block.compute_hash();
        block
    }

    /// Mine this block by incrementing the nonce until the hash
    /// starts with `difficulty` leading zeros (hex).
    pub fn mine(&mut self, difficulty: u32) {
        let prefix: String = std::iter::repeat('0').take(difficulty as usize).collect();
        loop {
            self.hash = self.compute_hash();
            if self.hash.starts_with(&prefix) {
                break;
            }
            self.nonce += 1;
        }
    }

    /// Verify that the stored hash matches the computed hash.
    pub fn verify_hash(&self) -> bool {
        self.hash == self.compute_hash()
    }
}

// ── Genesis Block ───────────────────────────────────────────────────────────

/// Create the genesis (first) block with difficulty-0 mining.
pub fn create_genesis_block(timestamp: u64) -> Block {
    let mut block = Block::new(0, timestamp, "Genesis Block", "0");
    block.hash = block.compute_hash();
    block
}

// ── Blockchain ──────────────────────────────────────────────────────────────

/// Statistics about a blockchain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStats {
    /// Total number of blocks including genesis.
    pub block_count: u64,
    /// Total characters of data stored across all blocks.
    pub total_data_bytes: u64,
    /// The current difficulty setting.
    pub difficulty: u32,
    /// The highest nonce found in any block.
    pub max_nonce: u64,
    /// Average nonce across all blocks.
    pub avg_nonce: f64,
}

/// A proof-of-work blockchain.
#[derive(Debug, Clone)]
pub struct Blockchain {
    /// The ordered list of blocks.
    pub blocks: Vec<Block>,
    /// Mining difficulty (leading hex zeros).
    pub difficulty: u32,
}

impl Blockchain {
    /// Create a new blockchain with a genesis block.
    pub fn new(difficulty: u32) -> Self {
        let genesis = create_genesis_block(0);
        Self {
            blocks: vec![genesis],
            difficulty,
        }
    }

    /// Create a blockchain with a custom genesis timestamp.
    pub fn with_genesis_timestamp(difficulty: u32, timestamp: u64) -> Self {
        let genesis = create_genesis_block(timestamp);
        Self {
            blocks: vec![genesis],
            difficulty,
        }
    }

    /// Get the latest (most recent) block.
    pub fn latest_block(&self) -> Option<&Block> {
        self.blocks.last()
    }

    /// Mine and add a new block with the given data.
    pub fn add_block(&mut self, timestamp: u64, data: impl Into<String>) {
        let prev_hash = self
            .blocks
            .last()
            .map(|b| b.hash.clone())
            .unwrap_or_else(|| "0".to_string());
        let index = self.blocks.len() as u64;
        let mut block = Block::new(index, timestamp, data, prev_hash);
        block.mine(self.difficulty);
        self.blocks.push(block);
    }

    /// Get the chain length (number of blocks including genesis).
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if the chain is empty (should never be true after construction).
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Set the mining difficulty.
    pub fn set_difficulty(&mut self, difficulty: u32) {
        self.difficulty = difficulty;
    }

    /// Validate the entire chain for integrity.
    ///
    /// Checks: genesis index, sequential indices, prev_hash linkage,
    /// hash correctness, and difficulty satisfaction.
    pub fn validate(&self) -> Result<(), BlockchainError> {
        if self.blocks.is_empty() {
            return Err(BlockchainError::EmptyChain);
        }

        let genesis = &self.blocks[0];
        if genesis.index != 0 {
            return Err(BlockchainError::InvalidGenesis);
        }
        if !genesis.verify_hash() {
            return Err(BlockchainError::InvalidHash { block_index: 0 });
        }

        for i in 1..self.blocks.len() {
            let block = &self.blocks[i];
            let prev = &self.blocks[i - 1];

            if block.index != i as u64 {
                return Err(BlockchainError::InvalidIndex {
                    expected: i as u64,
                    got: block.index,
                });
            }

            if block.prev_hash != prev.hash {
                return Err(BlockchainError::InvalidPrevHash {
                    block_index: block.index,
                });
            }

            if !block.verify_hash() {
                return Err(BlockchainError::InvalidHash {
                    block_index: block.index,
                });
            }

            if !block.satisfies_difficulty(self.difficulty) {
                return Err(BlockchainError::InsufficientWork {
                    block_index: block.index,
                    difficulty: self.difficulty,
                });
            }
        }

        Ok(())
    }

    /// Check if a specific block at `index` has been tampered with.
    pub fn is_block_valid(&self, index: usize) -> bool {
        if index >= self.blocks.len() {
            return false;
        }
        let block = &self.blocks[index];
        if !block.verify_hash() {
            return false;
        }
        if index > 0 {
            let prev = &self.blocks[index - 1];
            if block.prev_hash != prev.hash {
                return false;
            }
        }
        true
    }

    /// Gather chain statistics.
    pub fn stats(&self) -> ChainStats {
        let block_count = self.blocks.len() as u64;
        let total_data_bytes: u64 = self.blocks.iter().map(|b| b.data.len() as u64).sum();
        let max_nonce = self.blocks.iter().map(|b| b.nonce).max().unwrap_or(0);
        let sum_nonce: u64 = self.blocks.iter().map(|b| b.nonce).sum();
        let avg_nonce = if block_count == 0 {
            0.0
        } else {
            sum_nonce as f64 / block_count as f64
        };

        ChainStats {
            block_count,
            total_data_bytes,
            difficulty: self.difficulty,
            max_nonce,
            avg_nonce,
        }
    }

    /// Get a block by its index.
    pub fn get_block(&self, index: u64) -> Option<&Block> {
        self.blocks.get(index as usize)
    }

    /// Search for blocks containing the given data substring.
    pub fn find_blocks_by_data(&self, needle: &str) -> Vec<&Block> {
        self.blocks.iter().filter(|b| b.data.contains(needle)).collect()
    }

    /// Replace the chain with a longer valid one (longest chain wins).
    /// Returns true if the chain was replaced.
    pub fn replace_chain(&mut self, new_blocks: Vec<Block>) -> Result<bool, BlockchainError> {
        if new_blocks.len() <= self.blocks.len() {
            return Ok(false);
        }
        // Validate the incoming chain
        let candidate = Blockchain {
            blocks: new_blocks.clone(),
            difficulty: self.difficulty,
        };
        candidate.validate()?;
        self.blocks = new_blocks;
        Ok(true)
    }
}

// ── Display ─────────────────────────────────────────────────────────────────

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Block #{} [nonce={}] hash={}...",
            self.index,
            self.nonce,
            &self.hash[..8.min(self.hash.len())]
        )
    }
}

impl fmt::Display for Blockchain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Blockchain (difficulty={}, blocks={})", self.difficulty, self.blocks.len())?;
        for block in &self.blocks {
            writeln!(f, "  {block}")?;
        }
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_block_creation() {
        let genesis = create_genesis_block(1000);
        assert_eq!(genesis.index, 0);
        assert_eq!(genesis.timestamp, 1000);
        assert_eq!(genesis.data, "Genesis Block");
        assert_eq!(genesis.prev_hash, "0");
        assert!(genesis.verify_hash());
    }

    #[test]
    fn test_block_compute_hash_deterministic() {
        let block = Block::new(1, 1234, "test data", "abc123");
        let h1 = block.compute_hash();
        let h2 = block.compute_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_block_hash_changes_with_data() {
        let b1 = Block::new(1, 1234, "data A", "prev");
        let b2 = Block::new(1, 1234, "data B", "prev");
        assert_ne!(b1.hash, b2.hash);
    }

    #[test]
    fn test_block_mining_difficulty_1() {
        let mut block = Block::new(1, 100, "mine me", "0");
        block.mine(1);
        assert!(block.hash.starts_with("0"));
        assert!(block.verify_hash());
    }

    #[test]
    fn test_block_mining_difficulty_2() {
        let mut block = Block::new(1, 200, "harder", "0");
        block.mine(2);
        assert!(block.hash.starts_with("00"));
        assert!(block.verify_hash());
    }

    #[test]
    fn test_block_satisfies_difficulty() {
        let mut block = Block::new(1, 100, "test", "0");
        block.mine(2);
        assert!(block.satisfies_difficulty(2));
        assert!(block.satisfies_difficulty(1));
    }

    #[test]
    fn test_blockchain_new_has_genesis() {
        let chain = Blockchain::new(1);
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
        let genesis = chain.get_block(0).unwrap();
        assert_eq!(genesis.index, 0);
    }

    #[test]
    fn test_blockchain_add_block() {
        let mut chain = Blockchain::new(1);
        chain.add_block(100, "Block 1");
        chain.add_block(200, "Block 2");
        assert_eq!(chain.len(), 3);
        let b1 = chain.get_block(1).unwrap();
        assert_eq!(b1.data, "Block 1");
        let b2 = chain.get_block(2).unwrap();
        assert_eq!(b2.data, "Block 2");
    }

    #[test]
    fn test_blockchain_validation_passes() {
        let mut chain = Blockchain::new(1);
        chain.add_block(100, "tx1");
        chain.add_block(200, "tx2");
        assert!(chain.validate().is_ok());
    }

    #[test]
    fn test_blockchain_validation_fails_tampered_data() {
        let mut chain = Blockchain::new(1);
        chain.add_block(100, "tx1");
        chain.add_block(200, "tx2");
        // Tamper with block 1's data
        chain.blocks[1].data = "tampered".to_string();
        assert!(chain.validate().is_err());
    }

    #[test]
    fn test_blockchain_validation_fails_broken_link() {
        let mut chain = Blockchain::new(1);
        chain.add_block(100, "tx1");
        chain.add_block(200, "tx2");
        // Break the prev_hash link
        chain.blocks[2].prev_hash = "wrong".to_string();
        assert!(chain.validate().is_err());
    }

    #[test]
    fn test_blockchain_latest_block() {
        let mut chain = Blockchain::new(1);
        chain.add_block(100, "latest");
        let latest = chain.latest_block().unwrap();
        assert_eq!(latest.data, "latest");
    }

    #[test]
    fn test_blockchain_is_block_valid() {
        let mut chain = Blockchain::new(1);
        chain.add_block(100, "valid");
        assert!(chain.is_block_valid(0));
        assert!(chain.is_block_valid(1));
        assert!(!chain.is_block_valid(99));
    }

    #[test]
    fn test_blockchain_stats() {
        let mut chain = Blockchain::new(1);
        chain.add_block(100, "abc");
        chain.add_block(200, "defgh");
        let stats = chain.stats();
        assert_eq!(stats.block_count, 3);
        // genesis = "Genesis Block" (13) + "abc" (3) + "defgh" (5) = 21
        assert_eq!(stats.total_data_bytes, 21);
        assert_eq!(stats.difficulty, 1);
    }

    #[test]
    fn test_blockchain_find_blocks_by_data() {
        let mut chain = Blockchain::new(1);
        chain.add_block(100, "payment to Alice");
        chain.add_block(200, "payment to Bob");
        chain.add_block(300, "refund to Alice");
        let alice_blocks = chain.find_blocks_by_data("Alice");
        assert_eq!(alice_blocks.len(), 2);
    }

    #[test]
    fn test_blockchain_set_difficulty() {
        let mut chain = Blockchain::new(1);
        chain.set_difficulty(2);
        chain.add_block(100, "harder block");
        let block = chain.get_block(1).unwrap();
        assert!(block.hash.starts_with("00"));
    }

    #[test]
    fn test_block_display() {
        let block = Block::new(5, 1000, "test", "prev");
        let display = format!("{block}");
        assert!(display.contains("Block #5"));
    }

    #[test]
    fn test_blockchain_display() {
        let chain = Blockchain::new(1);
        let display = format!("{chain}");
        assert!(display.contains("difficulty=1"));
    }

    #[test]
    fn test_replace_chain_longer() {
        let mut chain1 = Blockchain::new(1);
        chain1.add_block(100, "a");

        let mut chain2 = Blockchain::new(1);
        chain2.add_block(100, "x");
        chain2.add_block(200, "y");
        chain2.add_block(300, "z");

        let replaced = chain1.replace_chain(chain2.blocks.clone()).unwrap();
        assert!(replaced);
        assert_eq!(chain1.len(), 4);
    }

    #[test]
    fn test_replace_chain_shorter_rejected() {
        let mut chain1 = Blockchain::new(1);
        chain1.add_block(100, "a");
        chain1.add_block(200, "b");

        let chain2 = Blockchain::new(1);
        let replaced = chain1.replace_chain(chain2.blocks.clone()).unwrap();
        assert!(!replaced);
        assert_eq!(chain1.len(), 3);
    }

    #[test]
    fn test_genesis_with_custom_timestamp() {
        let chain = Blockchain::with_genesis_timestamp(1, 999_999);
        let genesis = chain.get_block(0).unwrap();
        assert_eq!(genesis.timestamp, 999_999);
    }

    #[test]
    fn test_block_serialization() {
        let block = Block::new(1, 100, "ser test", "prev");
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: Block = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_empty_chain_validation_error() {
        let chain = Blockchain {
            blocks: vec![],
            difficulty: 1,
        };
        assert_eq!(chain.validate().unwrap_err(), BlockchainError::EmptyChain);
    }

    #[test]
    fn test_blockchain_error_display() {
        let err = BlockchainError::InvalidIndex { expected: 1, got: 5 };
        let msg = format!("{err}");
        assert!(msg.contains("expected 1"));
        assert!(msg.contains("got 5"));
    }
}
