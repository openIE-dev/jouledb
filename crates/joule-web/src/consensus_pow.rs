//! Proof-of-work consensus — mining (nonce search), difficulty adjustment
//! (target block time), hash target comparison, block validation, chain
//! selection (longest chain), uncle/ommer blocks, and mining reward calculation.

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

fn hash_hex(input: &str) -> String {
    bytes_to_hex(&sha256(input.as_bytes()))
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from PoW consensus operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowError {
    /// Block hash does not meet the target difficulty.
    InsufficientWork { block_height: u64, difficulty: u32 },
    /// Block hash does not match computed hash.
    InvalidHash { block_height: u64 },
    /// Previous hash does not match parent block.
    InvalidPrevHash { block_height: u64 },
    /// Block height does not match expected sequence.
    InvalidHeight { expected: u64, got: u64 },
    /// Chain is empty.
    EmptyChain,
    /// Uncle block not valid (too old or duplicate).
    InvalidUncle(String),
    /// Difficulty cannot be zero.
    ZeroDifficulty,
}

impl fmt::Display for PowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientWork { block_height, difficulty } => {
                write!(f, "block {block_height}: insufficient work for difficulty {difficulty}")
            }
            Self::InvalidHash { block_height } => {
                write!(f, "block {block_height}: hash mismatch")
            }
            Self::InvalidPrevHash { block_height } => {
                write!(f, "block {block_height}: previous hash mismatch")
            }
            Self::InvalidHeight { expected, got } => {
                write!(f, "invalid height: expected {expected}, got {got}")
            }
            Self::EmptyChain => write!(f, "empty chain"),
            Self::InvalidUncle(msg) => write!(f, "invalid uncle: {msg}"),
            Self::ZeroDifficulty => write!(f, "difficulty cannot be zero"),
        }
    }
}

impl std::error::Error for PowError {}

// ── Difficulty Target ───────────────────────────────────────────────────────

/// Difficulty target — represented as the number of leading hex zeros
/// the hash must have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DifficultyTarget {
    /// Number of leading hex zeros required.
    pub leading_zeros: u32,
}

impl DifficultyTarget {
    /// Create a new difficulty target.
    pub fn new(leading_zeros: u32) -> Result<Self, PowError> {
        if leading_zeros == 0 {
            return Err(PowError::ZeroDifficulty);
        }
        Ok(Self { leading_zeros })
    }

    /// Check if a hash meets this target.
    pub fn is_satisfied_by(&self, hash: &str) -> bool {
        let prefix: String = std::iter::repeat('0')
            .take(self.leading_zeros as usize)
            .collect();
        hash.starts_with(&prefix)
    }

    /// Numerical difficulty value (higher = harder).
    pub fn difficulty_value(&self) -> u64 {
        16u64.pow(self.leading_zeros)
    }
}

impl fmt::Display for DifficultyTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "difficulty(leading_zeros={})", self.leading_zeros)
    }
}

// ── PoW Block ───────────────────────────────────────────────────────────────

/// A block in the proof-of-work chain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PowBlock {
    /// Block height (0 = genesis).
    pub height: u64,
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Block data/payload.
    pub data: String,
    /// Hash of the previous block.
    pub prev_hash: String,
    /// This block's hash.
    pub hash: String,
    /// Proof-of-work nonce.
    pub nonce: u64,
    /// Difficulty at which this block was mined.
    pub difficulty: u32,
    /// Miner address.
    pub miner: String,
    /// Uncle block hashes included in this block.
    pub uncle_hashes: Vec<String>,
}

impl PowBlock {
    /// Compute the hash for this block (excluding the stored hash field).
    pub fn compute_hash(&self) -> String {
        let input = format!(
            "{}:{}:{}:{}:{}:{}",
            self.height, self.timestamp, self.data, self.prev_hash, self.nonce, self.difficulty,
        );
        hash_hex(&input)
    }

    /// Verify that the stored hash matches the computed hash.
    pub fn verify_hash(&self) -> bool {
        self.hash == self.compute_hash()
    }

    /// Check if this block satisfies a difficulty target.
    pub fn meets_target(&self, target: &DifficultyTarget) -> bool {
        target.is_satisfied_by(&self.hash)
    }

    /// Mine this block — search for a valid nonce.
    pub fn mine(&mut self, target: &DifficultyTarget) {
        let prefix: String = std::iter::repeat('0')
            .take(target.leading_zeros as usize)
            .collect();
        self.difficulty = target.leading_zeros;
        loop {
            self.hash = self.compute_hash();
            if self.hash.starts_with(&prefix) {
                break;
            }
            self.nonce += 1;
        }
    }

    /// Create a new unmined block.
    pub fn new(
        height: u64,
        timestamp: u64,
        data: impl Into<String>,
        prev_hash: impl Into<String>,
        miner: impl Into<String>,
    ) -> Self {
        let mut block = Self {
            height,
            timestamp,
            data: data.into(),
            prev_hash: prev_hash.into(),
            hash: String::new(),
            nonce: 0,
            difficulty: 0,
            miner: miner.into(),
            uncle_hashes: Vec::new(),
        };
        block.hash = block.compute_hash();
        block
    }
}

impl fmt::Display for PowBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PowBlock(height={}, miner={}, nonce={}, hash={}...)",
            self.height,
            self.miner,
            self.nonce,
            &self.hash[..8.min(self.hash.len())]
        )
    }
}

// ── Uncle/Ommer Block ───────────────────────────────────────────────────────

/// An uncle (ommer) block — a valid block that was not included in the main chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncleBlock {
    /// The uncle block's hash.
    pub hash: String,
    /// Height at which it was mined.
    pub height: u64,
    /// Miner of the uncle block.
    pub miner: String,
    /// Timestamp.
    pub timestamp: u64,
}

// ── Mining Reward ───────────────────────────────────────────────────────────

/// Mining reward configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Base block reward (in minor units).
    pub base_reward: u64,
    /// Uncle inclusion reward as fraction of base (numerator).
    pub uncle_inclusion_reward_num: u64,
    /// Uncle inclusion reward fraction (denominator).
    pub uncle_inclusion_reward_den: u64,
    /// Uncle mining reward as fraction of base (numerator).
    pub uncle_mining_reward_num: u64,
    /// Uncle mining reward fraction (denominator).
    pub uncle_mining_reward_den: u64,
    /// Number of blocks between reward halvings (0 = no halving).
    pub halving_interval: u64,
}

impl RewardConfig {
    /// Create a Bitcoin-like reward config.
    pub fn bitcoin_like(base_reward: u64, halving_interval: u64) -> Self {
        Self {
            base_reward,
            uncle_inclusion_reward_num: 0,
            uncle_inclusion_reward_den: 1,
            uncle_mining_reward_num: 0,
            uncle_mining_reward_den: 1,
            halving_interval,
        }
    }

    /// Create an Ethereum-like reward config with uncle rewards.
    pub fn ethereum_like(base_reward: u64) -> Self {
        Self {
            base_reward,
            uncle_inclusion_reward_num: 1,
            uncle_inclusion_reward_den: 32,
            uncle_mining_reward_num: 7,
            uncle_mining_reward_den: 8,
            halving_interval: 0,
        }
    }

    /// Calculate the base reward at a given block height (with halving).
    pub fn reward_at_height(&self, height: u64) -> u64 {
        if self.halving_interval == 0 {
            return self.base_reward;
        }
        let halvings = height / self.halving_interval;
        if halvings >= 64 {
            return 0; // Reward halved to zero
        }
        self.base_reward >> halvings
    }

    /// Calculate total miner reward for a block (base + uncle inclusion bonuses).
    pub fn total_block_reward(&self, height: u64, uncle_count: usize) -> u64 {
        let base = self.reward_at_height(height);
        let uncle_bonus = if self.uncle_inclusion_reward_den > 0 {
            uncle_count as u64 * base * self.uncle_inclusion_reward_num
                / self.uncle_inclusion_reward_den
        } else {
            0
        };
        base + uncle_bonus
    }

    /// Calculate the reward for mining an uncle block.
    pub fn uncle_reward(&self, uncle_height: u64, nephew_height: u64) -> u64 {
        let base = self.reward_at_height(nephew_height);
        if self.uncle_mining_reward_den == 0 {
            return 0;
        }
        // Ethereum formula: (uncle_height + 8 - nephew_height) * base / 8
        let height_diff = nephew_height.saturating_sub(uncle_height);
        if height_diff > 7 {
            return 0; // Too old
        }
        let numerator = (8 - height_diff) as u128 * base as u128;
        (numerator / 8) as u64
    }
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self::bitcoin_like(5_000_000_000, 210_000) // 50 BTC in satoshis
    }
}

// ── Difficulty Adjustment ───────────────────────────────────────────────────

/// Configuration for difficulty adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyConfig {
    /// Target time between blocks (in seconds).
    pub target_block_time: u64,
    /// Number of blocks in the adjustment window.
    pub adjustment_window: u64,
    /// Minimum difficulty (leading zeros).
    pub min_difficulty: u32,
    /// Maximum difficulty (leading zeros).
    pub max_difficulty: u32,
}

impl DifficultyConfig {
    /// Create a new difficulty configuration.
    pub fn new(target_block_time: u64, adjustment_window: u64) -> Self {
        Self {
            target_block_time,
            adjustment_window,
            min_difficulty: 1,
            max_difficulty: 8, // 8 leading hex zeros is very hard
        }
    }

    /// Calculate the new difficulty based on the elapsed time for the last window.
    pub fn adjust_difficulty(&self, current_difficulty: u32, actual_time: u64) -> u32 {
        let expected_time = self.target_block_time * self.adjustment_window;

        if actual_time == 0 {
            // Blocks are instant — increase difficulty
            return (current_difficulty + 1).min(self.max_difficulty);
        }

        // If blocks are too fast, increase difficulty; if too slow, decrease
        let ratio_x100 = (expected_time * 100) / actual_time;

        let new_diff = if ratio_x100 > 150 {
            // Blocks are >1.5x too fast
            current_difficulty + 1
        } else if ratio_x100 < 67 {
            // Blocks are >1.5x too slow
            current_difficulty.saturating_sub(1)
        } else {
            current_difficulty
        };

        new_diff.clamp(self.min_difficulty, self.max_difficulty)
    }
}

impl Default for DifficultyConfig {
    fn default() -> Self {
        Self::new(10, 10) // 10-second blocks, 10-block window
    }
}

// ── PoW Chain ───────────────────────────────────────────────────────────────

/// A proof-of-work blockchain with consensus rules.
#[derive(Debug, Clone)]
pub struct PowChain {
    /// The ordered blocks.
    pub blocks: Vec<PowBlock>,
    /// Current difficulty target.
    pub target: DifficultyTarget,
    /// Difficulty adjustment config.
    pub diff_config: DifficultyConfig,
    /// Mining reward config.
    pub reward_config: RewardConfig,
    /// Uncle blocks included in the chain.
    pub uncles: Vec<UncleBlock>,
    /// Total work done (sum of difficulties).
    total_work: u64,
}

impl PowChain {
    /// Create a new chain with a genesis block.
    pub fn new(
        initial_difficulty: u32,
        diff_config: DifficultyConfig,
        reward_config: RewardConfig,
    ) -> Result<Self, PowError> {
        let target = DifficultyTarget::new(initial_difficulty)?;
        let genesis = PowBlock::new(0, 0, "genesis", "0", "system");
        Ok(Self {
            blocks: vec![genesis],
            target,
            diff_config,
            reward_config,
            uncles: Vec::new(),
            total_work: target.difficulty_value(),
        })
    }

    /// Get the chain length.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Get the latest block.
    pub fn tip(&self) -> Option<&PowBlock> {
        self.blocks.last()
    }

    /// Get the total accumulated work.
    pub fn total_work(&self) -> u64 {
        self.total_work
    }

    /// Current difficulty value.
    pub fn current_difficulty(&self) -> u32 {
        self.target.leading_zeros
    }

    /// Mine and append a new block.
    pub fn mine_block(
        &mut self,
        timestamp: u64,
        data: impl Into<String>,
        miner: impl Into<String>,
    ) {
        let prev_hash = self
            .blocks
            .last()
            .map(|b| b.hash.clone())
            .unwrap_or_else(|| "0".to_string());
        let height = self.blocks.len() as u64;
        let mut block = PowBlock::new(height, timestamp, data, prev_hash, miner);
        block.mine(&self.target);
        self.total_work += self.target.difficulty_value();
        self.blocks.push(block);

        // Maybe adjust difficulty
        self.maybe_adjust_difficulty();
    }

    /// Validate the entire chain.
    pub fn validate(&self) -> Result<(), PowError> {
        if self.blocks.is_empty() {
            return Err(PowError::EmptyChain);
        }

        let genesis = &self.blocks[0];
        if genesis.height != 0 {
            return Err(PowError::InvalidHeight {
                expected: 0,
                got: genesis.height,
            });
        }

        for i in 1..self.blocks.len() {
            let block = &self.blocks[i];
            let prev = &self.blocks[i - 1];

            if block.height != i as u64 {
                return Err(PowError::InvalidHeight {
                    expected: i as u64,
                    got: block.height,
                });
            }

            if block.prev_hash != prev.hash {
                return Err(PowError::InvalidPrevHash {
                    block_height: block.height,
                });
            }

            if !block.verify_hash() {
                return Err(PowError::InvalidHash {
                    block_height: block.height,
                });
            }
        }

        Ok(())
    }

    /// Validate a single block against a target.
    pub fn validate_block(block: &PowBlock, target: &DifficultyTarget) -> Result<(), PowError> {
        if !block.verify_hash() {
            return Err(PowError::InvalidHash {
                block_height: block.height,
            });
        }
        if !block.meets_target(target) {
            return Err(PowError::InsufficientWork {
                block_height: block.height,
                difficulty: target.leading_zeros,
            });
        }
        Ok(())
    }

    /// Add an uncle block.
    pub fn add_uncle(&mut self, uncle: UncleBlock) -> Result<(), PowError> {
        // Uncle must be within the last 7 blocks
        let current_height = self.blocks.len() as u64;
        if uncle.height + 7 < current_height {
            return Err(PowError::InvalidUncle("uncle too old".to_string()));
        }
        if uncle.height >= current_height {
            return Err(PowError::InvalidUncle("uncle height >= chain height".to_string()));
        }
        // Check for duplicates
        if self.uncles.iter().any(|u| u.hash == uncle.hash) {
            return Err(PowError::InvalidUncle("duplicate uncle".to_string()));
        }
        self.uncles.push(uncle);
        Ok(())
    }

    /// Calculate reward for the current tip.
    pub fn tip_reward(&self) -> u64 {
        let height = self.blocks.len().saturating_sub(1) as u64;
        let uncle_count = self
            .blocks
            .last()
            .map(|b| b.uncle_hashes.len())
            .unwrap_or(0);
        self.reward_config.total_block_reward(height, uncle_count)
    }

    /// Compare this chain against another for longest-chain selection.
    /// Returns true if this chain should be preferred.
    pub fn is_preferred_over(&self, other: &PowChain) -> bool {
        self.total_work > other.total_work
    }

    /// Adjust difficulty if we've completed an adjustment window.
    fn maybe_adjust_difficulty(&mut self) {
        let height = self.blocks.len() as u64;
        let window = self.diff_config.adjustment_window;
        if window == 0 || height < window + 1 {
            return;
        }
        if height % window != 0 {
            return;
        }

        let end_idx = self.blocks.len() - 1;
        let start_idx = end_idx - window as usize;

        let end_time = self.blocks[end_idx].timestamp;
        let start_time = self.blocks[start_idx].timestamp;
        let elapsed = end_time.saturating_sub(start_time);

        let new_diff = self
            .diff_config
            .adjust_difficulty(self.target.leading_zeros, elapsed);
        // Use checked new to avoid zero difficulty
        if let Ok(target) = DifficultyTarget::new(new_diff) {
            self.target = target;
        }
    }

    /// Get block by height.
    pub fn get_block(&self, height: u64) -> Option<&PowBlock> {
        self.blocks.get(height as usize)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain() -> PowChain {
        PowChain::new(
            1,
            DifficultyConfig::new(10, 5),
            RewardConfig::bitcoin_like(5000, 100),
        )
        .unwrap()
    }

    #[test]
    fn test_difficulty_target_creation() {
        let target = DifficultyTarget::new(2).unwrap();
        assert_eq!(target.leading_zeros, 2);
        assert_eq!(target.difficulty_value(), 256); // 16^2
    }

    #[test]
    fn test_zero_difficulty_error() {
        let err = DifficultyTarget::new(0).unwrap_err();
        assert_eq!(err, PowError::ZeroDifficulty);
    }

    #[test]
    fn test_target_satisfaction() {
        let target = DifficultyTarget::new(2).unwrap();
        assert!(target.is_satisfied_by("00abcdef1234"));
        assert!(!target.is_satisfied_by("0abcdef12345"));
    }

    #[test]
    fn test_block_creation() {
        let block = PowBlock::new(1, 100, "data", "prev", "miner1");
        assert_eq!(block.height, 1);
        assert_eq!(block.miner, "miner1");
        assert!(block.verify_hash());
    }

    #[test]
    fn test_block_mining() {
        let target = DifficultyTarget::new(1).unwrap();
        let mut block = PowBlock::new(1, 100, "mine me", "0", "miner");
        block.mine(&target);
        assert!(block.hash.starts_with("0"));
        assert!(block.verify_hash());
        assert!(block.meets_target(&target));
    }

    #[test]
    fn test_chain_creation() {
        let chain = make_chain();
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
        assert_eq!(chain.current_difficulty(), 1);
    }

    #[test]
    fn test_mine_and_append_block() {
        let mut chain = make_chain();
        chain.mine_block(100, "block 1", "miner1");
        assert_eq!(chain.len(), 2);
        let block = chain.get_block(1).unwrap();
        assert_eq!(block.miner, "miner1");
    }

    #[test]
    fn test_chain_validation_passes() {
        let mut chain = make_chain();
        chain.mine_block(100, "b1", "m1");
        chain.mine_block(200, "b2", "m2");
        assert!(chain.validate().is_ok());
    }

    #[test]
    fn test_chain_validation_tampered() {
        let mut chain = make_chain();
        chain.mine_block(100, "b1", "m1");
        chain.blocks[1].data = "tampered".to_string();
        assert!(chain.validate().is_err());
    }

    #[test]
    fn test_total_work_accumulates() {
        let mut chain = make_chain();
        let initial = chain.total_work();
        chain.mine_block(100, "b1", "m1");
        assert!(chain.total_work() > initial);
    }

    #[test]
    fn test_chain_preference_by_work() {
        let mut chain1 = make_chain();
        chain1.mine_block(100, "a", "m");

        let mut chain2 = make_chain();
        chain2.mine_block(100, "a", "m");
        chain2.mine_block(200, "b", "m");

        assert!(chain2.is_preferred_over(&chain1));
        assert!(!chain1.is_preferred_over(&chain2));
    }

    #[test]
    fn test_reward_at_height_no_halving() {
        let config = RewardConfig::ethereum_like(2000);
        assert_eq!(config.reward_at_height(0), 2000);
        assert_eq!(config.reward_at_height(999_999), 2000);
    }

    #[test]
    fn test_reward_halving() {
        let config = RewardConfig::bitcoin_like(5000, 100);
        assert_eq!(config.reward_at_height(0), 5000);
        assert_eq!(config.reward_at_height(99), 5000);
        assert_eq!(config.reward_at_height(100), 2500);
        assert_eq!(config.reward_at_height(200), 1250);
    }

    #[test]
    fn test_total_block_reward_with_uncles() {
        let config = RewardConfig::ethereum_like(2000);
        // Base reward + uncle inclusion bonus
        let reward_no_uncles = config.total_block_reward(1, 0);
        let reward_with_uncles = config.total_block_reward(1, 2);
        assert!(reward_with_uncles > reward_no_uncles);
    }

    #[test]
    fn test_uncle_reward() {
        let config = RewardConfig::ethereum_like(2000);
        // Uncle mined at height 5, nephew at height 6
        let reward = config.uncle_reward(5, 6);
        assert!(reward > 0);
        assert!(reward < 2000);
    }

    #[test]
    fn test_uncle_too_old() {
        let config = RewardConfig::ethereum_like(2000);
        // Uncle at height 0, nephew at height 10 — too old
        let reward = config.uncle_reward(0, 10);
        assert_eq!(reward, 0);
    }

    #[test]
    fn test_add_uncle() {
        let mut chain = make_chain();
        chain.mine_block(100, "b1", "m1");
        chain.mine_block(200, "b2", "m2");
        let uncle = UncleBlock {
            hash: "uncle_hash_1".to_string(),
            height: 1,
            miner: "uncle_miner".to_string(),
            timestamp: 150,
        };
        chain.add_uncle(uncle).unwrap();
        assert_eq!(chain.uncles.len(), 1);
    }

    #[test]
    fn test_add_uncle_too_old() {
        let mut chain = make_chain();
        for i in 1..=10 {
            chain.mine_block(i * 100, format!("b{i}"), "m");
        }
        let uncle = UncleBlock {
            hash: "old".to_string(),
            height: 1,
            miner: "m".to_string(),
            timestamp: 100,
        };
        let err = chain.add_uncle(uncle).unwrap_err();
        assert!(matches!(err, PowError::InvalidUncle(_)));
    }

    #[test]
    fn test_duplicate_uncle() {
        let mut chain = make_chain();
        chain.mine_block(100, "b1", "m1");
        let uncle = UncleBlock {
            hash: "uncle1".to_string(),
            height: 0,
            miner: "m".to_string(),
            timestamp: 50,
        };
        chain.add_uncle(uncle.clone()).unwrap();
        let err = chain.add_uncle(uncle).unwrap_err();
        assert!(matches!(err, PowError::InvalidUncle(_)));
    }

    #[test]
    fn test_difficulty_adjustment_increase() {
        let config = DifficultyConfig::new(10, 10);
        // Expected window time = 10 * 10 = 100 seconds
        // Actual = 30 seconds (too fast, ratio = 333%)
        let new_diff = config.adjust_difficulty(2, 30);
        assert_eq!(new_diff, 3); // Increased
    }

    #[test]
    fn test_difficulty_adjustment_decrease() {
        let config = DifficultyConfig::new(10, 10);
        // Actual = 300 seconds (too slow, ratio = 33%)
        let new_diff = config.adjust_difficulty(3, 300);
        assert_eq!(new_diff, 2); // Decreased
    }

    #[test]
    fn test_difficulty_adjustment_stable() {
        let config = DifficultyConfig::new(10, 10);
        // Actual = 100 seconds (perfect, ratio = 100%)
        let new_diff = config.adjust_difficulty(2, 100);
        assert_eq!(new_diff, 2); // No change
    }

    #[test]
    fn test_difficulty_clamped_to_min() {
        let config = DifficultyConfig::new(10, 10);
        let new_diff = config.adjust_difficulty(1, 10000);
        assert_eq!(new_diff, 1); // Can't go below min
    }

    #[test]
    fn test_block_display() {
        let block = PowBlock::new(5, 100, "data", "prev", "miner");
        let display = format!("{block}");
        assert!(display.contains("height=5"));
        assert!(display.contains("miner=miner"));
    }

    #[test]
    fn test_target_display() {
        let target = DifficultyTarget::new(3).unwrap();
        let display = format!("{target}");
        assert!(display.contains("leading_zeros=3"));
    }

    #[test]
    fn test_pow_error_display() {
        let err = PowError::InsufficientWork { block_height: 5, difficulty: 3 };
        let msg = format!("{err}");
        assert!(msg.contains("block 5"));
        assert!(msg.contains("difficulty 3"));
    }

    #[test]
    fn test_validate_block_static() {
        let target = DifficultyTarget::new(1).unwrap();
        let mut block = PowBlock::new(1, 100, "test", "0", "m");
        block.mine(&target);
        assert!(PowChain::validate_block(&block, &target).is_ok());
    }

    #[test]
    fn test_default_configs() {
        let dc = DifficultyConfig::default();
        assert_eq!(dc.target_block_time, 10);
        let rc = RewardConfig::default();
        assert_eq!(rc.base_reward, 5_000_000_000);
    }
}
