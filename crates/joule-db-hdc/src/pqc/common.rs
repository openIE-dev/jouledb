//! Common cryptographic utilities for PQC
//!
//! Implements:
//! - SHA-3 and SHAKE extendable output functions (Keccak-based)
//! - Constant-time comparison and selection
//! - Secure memory zeroization

use std::ops::Drop;

// ============================================================================
// Keccak Permutation (SHA-3 / SHAKE foundation)
// ============================================================================

/// Keccak-f[1600] round constants
const KECCAK_RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Keccak rotation offsets
const KECCAK_RHO: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];

/// Keccak pi permutation indices
const KECCAK_PI: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

/// Keccak state (5x5 matrix of 64-bit words = 1600 bits)
#[derive(Clone)]
pub struct KeccakState {
    state: [u64; 25],
    /// Current position within the squeeze block (bytes consumed since last permute)
    squeeze_offset: usize,
}

impl KeccakState {
    /// Create new zeroed state
    pub fn new() -> Self {
        Self {
            state: [0u64; 25],
            squeeze_offset: 0,
        }
    }

    /// Keccak-f[1600] permutation
    pub fn permute(&mut self) {
        let mut state = self.state;

        for round in 0..24 {
            // θ step
            let mut c = [0u64; 5];
            for x in 0..5 {
                c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
            }

            let mut d = [0u64; 5];
            for x in 0..5 {
                d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            }

            for i in 0..25 {
                state[i] ^= d[i % 5];
            }

            // ρ and π steps
            let mut last = state[1];
            for i in 0..24 {
                let j = KECCAK_PI[i];
                let temp = state[j];
                state[j] = last.rotate_left(KECCAK_RHO[i]);
                last = temp;
            }

            // χ step
            for y in 0..5 {
                let mut row = [0u64; 5];
                for x in 0..5 {
                    row[x] = state[y * 5 + x];
                }
                for x in 0..5 {
                    state[y * 5 + x] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
                }
            }

            // ι step
            state[0] ^= KECCAK_RC[round];
        }

        self.state = state;
    }

    /// Absorb data into state (rate bytes at a time)
    pub fn absorb(&mut self, data: &[u8], rate: usize) {
        let mut offset = 0;

        while offset < data.len() {
            let block_size = (data.len() - offset).min(rate);

            // XOR data into state
            for i in 0..block_size {
                let byte_pos = i / 8;
                let bit_pos = (i % 8) * 8;
                self.state[byte_pos] ^= (data[offset + i] as u64) << bit_pos;
            }

            offset += block_size;

            if block_size == rate {
                self.permute();
            }
        }
    }

    /// Apply padding and final permutation
    pub fn finalize(&mut self, rate: usize, absorbed: usize, domain_sep: u8) {
        let pad_pos = absorbed % rate;

        // Padding: domain separator || 10*1
        let byte_pos = pad_pos / 8;
        let bit_pos = (pad_pos % 8) * 8;
        self.state[byte_pos] ^= (domain_sep as u64) << bit_pos;

        // Final bit at end of rate
        let final_byte = (rate - 1) / 8;
        let final_bit = ((rate - 1) % 8) * 8;
        self.state[final_byte] ^= 0x80u64 << final_bit;

        self.permute();
        self.squeeze_offset = 0;
    }

    /// Squeeze output from state
    pub fn squeeze(&mut self, output: &mut [u8], rate: usize) {
        let mut offset = 0;

        while offset < output.len() {
            // How many bytes remain in the current block
            let remaining_in_block = rate - self.squeeze_offset;
            let to_copy = (output.len() - offset).min(remaining_in_block);

            // Extract from state at current squeeze_offset
            for i in 0..to_copy {
                let pos = self.squeeze_offset + i;
                let byte_pos = pos / 8;
                let bit_pos = (pos % 8) * 8;
                output[offset + i] = ((self.state[byte_pos] >> bit_pos) & 0xFF) as u8;
            }

            offset += to_copy;
            self.squeeze_offset += to_copy;

            // If we've consumed the full block, permute for next block
            if self.squeeze_offset >= rate {
                self.permute();
                self.squeeze_offset = 0;
            }
        }
    }
}

impl Default for KeccakState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SHA-3 Hash Functions
// ============================================================================

/// SHA3-256 hash function
pub struct Sha3_256 {
    state: KeccakState,
    absorbed: usize,
}

impl Sha3_256 {
    /// Rate in bytes (1600 - 2*256) / 8 = 136
    pub const RATE: usize = 136;
    /// Output length in bytes
    pub const OUTPUT_LEN: usize = 32;

    /// Create new hasher
    pub fn new() -> Self {
        Self {
            state: KeccakState::new(),
            absorbed: 0,
        }
    }

    /// Update with data
    pub fn update(&mut self, data: &[u8]) {
        let mut offset = 0;

        while offset < data.len() {
            let remaining_in_block = Self::RATE - (self.absorbed % Self::RATE);
            let to_absorb = (data.len() - offset).min(remaining_in_block);

            // XOR into state
            for i in 0..to_absorb {
                let pos = self.absorbed % Self::RATE;
                let byte_pos = pos / 8;
                let bit_pos = (pos % 8) * 8;
                self.state.state[byte_pos] ^= (data[offset + i] as u64) << bit_pos;
                self.absorbed += 1;
            }

            offset += to_absorb;

            if self.absorbed % Self::RATE == 0 {
                self.state.permute();
            }
        }
    }

    /// Finalize and return hash
    pub fn finalize(mut self) -> [u8; Self::OUTPUT_LEN] {
        // SHA-3 domain separator: 0x06
        self.state.finalize(Self::RATE, self.absorbed, 0x06);

        let mut output = [0u8; Self::OUTPUT_LEN];
        self.state.squeeze(&mut output, Self::RATE);
        output
    }

    /// One-shot hash
    pub fn hash(data: &[u8]) -> [u8; Self::OUTPUT_LEN] {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }
}

impl Default for Sha3_256 {
    fn default() -> Self {
        Self::new()
    }
}

/// SHA3-512 hash function
pub struct Sha3_512 {
    state: KeccakState,
    absorbed: usize,
}

impl Sha3_512 {
    /// Rate in bytes (1600 - 2*512) / 8 = 72
    pub const RATE: usize = 72;
    /// Output length in bytes
    pub const OUTPUT_LEN: usize = 64;

    /// Create new hasher
    pub fn new() -> Self {
        Self {
            state: KeccakState::new(),
            absorbed: 0,
        }
    }

    /// Update with data
    pub fn update(&mut self, data: &[u8]) {
        let mut offset = 0;

        while offset < data.len() {
            let remaining_in_block = Self::RATE - (self.absorbed % Self::RATE);
            let to_absorb = (data.len() - offset).min(remaining_in_block);

            for i in 0..to_absorb {
                let pos = self.absorbed % Self::RATE;
                let byte_pos = pos / 8;
                let bit_pos = (pos % 8) * 8;
                self.state.state[byte_pos] ^= (data[offset + i] as u64) << bit_pos;
                self.absorbed += 1;
            }

            offset += to_absorb;

            if self.absorbed % Self::RATE == 0 {
                self.state.permute();
            }
        }
    }

    /// Finalize and return hash
    pub fn finalize(mut self) -> [u8; Self::OUTPUT_LEN] {
        self.state.finalize(Self::RATE, self.absorbed, 0x06);

        let mut output = [0u8; Self::OUTPUT_LEN];
        self.state.squeeze(&mut output, Self::RATE);
        output
    }

    /// One-shot hash
    pub fn hash(data: &[u8]) -> [u8; Self::OUTPUT_LEN] {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }
}

impl Default for Sha3_512 {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SHAKE Extendable Output Functions
// ============================================================================

/// SHAKE128 extendable output function
pub struct Shake128 {
    state: KeccakState,
    absorbed: usize,
    finalized: bool,
}

impl Shake128 {
    /// Rate in bytes (1600 - 2*128) / 8 = 168
    pub const RATE: usize = 168;

    /// Create new SHAKE128 instance
    pub fn new() -> Self {
        Self {
            state: KeccakState::new(),
            absorbed: 0,
            finalized: false,
        }
    }

    /// Absorb data
    pub fn absorb(&mut self, data: &[u8]) {
        assert!(!self.finalized, "Cannot absorb after squeezing");

        let mut offset = 0;

        while offset < data.len() {
            let remaining_in_block = Self::RATE - (self.absorbed % Self::RATE);
            let to_absorb = (data.len() - offset).min(remaining_in_block);

            for i in 0..to_absorb {
                let pos = self.absorbed % Self::RATE;
                let byte_pos = pos / 8;
                let bit_pos = (pos % 8) * 8;
                self.state.state[byte_pos] ^= (data[offset + i] as u64) << bit_pos;
                self.absorbed += 1;
            }

            offset += to_absorb;

            if self.absorbed % Self::RATE == 0 {
                self.state.permute();
            }
        }
    }

    /// Squeeze output (can be called multiple times for XOF)
    pub fn squeeze(&mut self, output: &mut [u8]) {
        if !self.finalized {
            // SHAKE domain separator: 0x1F
            self.state.finalize(Self::RATE, self.absorbed, 0x1F);
            self.finalized = true;
        }

        self.state.squeeze(output, Self::RATE);
    }

    /// One-shot XOF
    pub fn xof(data: &[u8], output_len: usize) -> Vec<u8> {
        let mut shake = Self::new();
        shake.absorb(data);
        let mut output = vec![0u8; output_len];
        shake.squeeze(&mut output);
        output
    }
}

impl Default for Shake128 {
    fn default() -> Self {
        Self::new()
    }
}

/// SHAKE256 extendable output function
pub struct Shake256 {
    state: KeccakState,
    absorbed: usize,
    finalized: bool,
}

impl Shake256 {
    /// Rate in bytes (1600 - 2*256) / 8 = 136
    pub const RATE: usize = 136;

    /// Create new SHAKE256 instance
    pub fn new() -> Self {
        Self {
            state: KeccakState::new(),
            absorbed: 0,
            finalized: false,
        }
    }

    /// Absorb data
    pub fn absorb(&mut self, data: &[u8]) {
        assert!(!self.finalized, "Cannot absorb after squeezing");

        let mut offset = 0;

        while offset < data.len() {
            let remaining_in_block = Self::RATE - (self.absorbed % Self::RATE);
            let to_absorb = (data.len() - offset).min(remaining_in_block);

            for i in 0..to_absorb {
                let pos = self.absorbed % Self::RATE;
                let byte_pos = pos / 8;
                let bit_pos = (pos % 8) * 8;
                self.state.state[byte_pos] ^= (data[offset + i] as u64) << bit_pos;
                self.absorbed += 1;
            }

            offset += to_absorb;

            if self.absorbed % Self::RATE == 0 {
                self.state.permute();
            }
        }
    }

    /// Squeeze output
    pub fn squeeze(&mut self, output: &mut [u8]) {
        if !self.finalized {
            self.state.finalize(Self::RATE, self.absorbed, 0x1F);
            self.finalized = true;
        }

        self.state.squeeze(output, Self::RATE);
    }

    /// One-shot XOF
    pub fn xof(data: &[u8], output_len: usize) -> Vec<u8> {
        let mut shake = Self::new();
        shake.absorb(data);
        let mut output = vec![0u8; output_len];
        shake.squeeze(&mut output);
        output
    }
}

impl Default for Shake256 {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Constant-Time Operations
// ============================================================================

/// Constant-time utilities to prevent timing attacks
pub struct ConstantTime;

impl ConstantTime {
    /// Constant-time byte comparison (returns 0 if equal, non-zero otherwise)
    #[inline]
    pub fn ct_compare(a: &[u8], b: &[u8]) -> u8 {
        if a.len() != b.len() {
            return 1;
        }

        let mut result = 0u8;
        for (x, y) in a.iter().zip(b.iter()) {
            result |= x ^ y;
        }
        result
    }

    /// Constant-time equality check
    #[inline]
    pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
        Self::ct_compare(a, b) == 0
    }

    /// Constant-time conditional select: returns a if condition is 0, b if condition is non-zero
    #[inline]
    pub fn ct_select(condition: u8, a: &[u8], b: &[u8]) -> Vec<u8> {
        assert_eq!(a.len(), b.len());

        let mask = (condition.wrapping_neg()) as u8; // 0x00 or 0xFF

        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x & !mask) | (y & mask))
            .collect()
    }

    /// Constant-time conditional swap
    #[inline]
    pub fn ct_swap(condition: u8, a: &mut [u8], b: &mut [u8]) {
        assert_eq!(a.len(), b.len());

        let mask = (condition.wrapping_neg()) as u8;

        for (x, y) in a.iter_mut().zip(b.iter_mut()) {
            let t = mask & (*x ^ *y);
            *x ^= t;
            *y ^= t;
        }
    }

    /// Constant-time u16 comparison
    #[inline]
    pub fn ct_lt_u16(a: u16, b: u16) -> u16 {
        // Returns 0xFFFF if a < b, 0x0000 otherwise
        let diff = (a as i32) - (b as i32);
        ((diff >> 31) as u16) & 0xFFFF
    }

    /// Constant-time conditional move for u16
    #[inline]
    pub fn ct_cmov_u16(condition: u16, a: u16, b: u16) -> u16 {
        let mask = condition.wrapping_neg();
        (a & !mask) | (b & mask)
    }
}

// ============================================================================
// Secure Memory
// ============================================================================

/// Vector that securely zeros its contents on drop
#[derive(Clone)]
pub struct SecureZeroingVec {
    data: Vec<u8>,
}

impl std::fmt::Debug for SecureZeroingVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't reveal actual contents for security
        write!(f, "SecureZeroingVec({} bytes)", self.data.len())
    }
}

impl SecureZeroingVec {
    /// Create new secure vector
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
        }
    }

    /// Create from existing data (takes ownership)
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Get slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Drop for SecureZeroingVec {
    fn drop(&mut self) {
        // Securely zero the memory
        for byte in &mut self.data {
            // Use volatile write to prevent compiler optimization
            unsafe {
                std::ptr::write_volatile(byte, 0);
            }
        }
        // Memory barrier to ensure the writes complete
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
}

impl std::ops::Deref for SecureZeroingVec {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for SecureZeroingVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

// ============================================================================
// Byte encoding utilities
// ============================================================================

/// Encode integer to bytes (little-endian)
#[inline]
pub fn encode_le_u16(val: u16) -> [u8; 2] {
    val.to_le_bytes()
}

/// Decode bytes to integer (little-endian)
#[inline]
pub fn decode_le_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

/// Encode integer to bytes (little-endian)
#[inline]
pub fn encode_le_u32(val: u32) -> [u8; 4] {
    val.to_le_bytes()
}

/// Decode bytes to integer (little-endian)
#[inline]
pub fn decode_le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha3_256() {
        // Test vector: empty string
        let hash = Sha3_256::hash(b"");
        // SHA3-256("") = a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a
        assert_eq!(hash[0], 0xa7);
        assert_eq!(hash[1], 0xff);
    }

    #[test]
    fn test_sha3_256_abc() {
        let hash = Sha3_256::hash(b"abc");
        // SHA3-256("abc") = 3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532
        assert_eq!(hash[0], 0x3a);
        assert_eq!(hash[1], 0x98);
    }

    #[test]
    fn test_shake128() {
        let output = Shake128::xof(b"", 32);
        // SHAKE128("", 32) starts with 7f9c2ba4e88f827d...
        assert_eq!(output[0], 0x7f);
        assert_eq!(output[1], 0x9c);
    }

    #[test]
    fn test_shake256() {
        let output = Shake256::xof(b"", 32);
        // SHAKE256("", 32) starts with 46b9dd2b0ba88d13...
        assert_eq!(output[0], 0x46);
        assert_eq!(output[1], 0xb9);
    }

    #[test]
    fn test_constant_time_compare() {
        let a = [1u8, 2, 3, 4];
        let b = [1u8, 2, 3, 4];
        let c = [1u8, 2, 3, 5];

        assert!(ConstantTime::ct_eq(&a, &b));
        assert!(!ConstantTime::ct_eq(&a, &c));
    }

    #[test]
    fn test_constant_time_select() {
        let a = vec![1u8, 2, 3];
        let b = vec![4u8, 5, 6];

        let result0 = ConstantTime::ct_select(0, &a, &b);
        let result1 = ConstantTime::ct_select(1, &a, &b);

        assert_eq!(result0, a);
        assert_eq!(result1, b);
    }

    #[test]
    fn test_secure_zeroing() {
        let mut secure = SecureZeroingVec::new(32);
        secure.as_mut_slice().copy_from_slice(&[0xFFu8; 32]);
        assert_eq!(secure.as_slice()[0], 0xFF);
        // Drop happens automatically, zeroing the memory
    }

    #[test]
    fn test_keccak_permutation() {
        // Test that permutation is deterministic
        let mut state1 = KeccakState::new();
        let mut state2 = KeccakState::new();

        state1.state[0] = 0x123456789ABCDEF0;
        state2.state[0] = 0x123456789ABCDEF0;

        state1.permute();
        state2.permute();

        assert_eq!(state1.state, state2.state);
    }
}
