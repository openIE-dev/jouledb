//! AES block cipher — pure-Rust SubBytes, ShiftRows, MixColumns, key expansion.
//!
//! Replaces Node.js `crypto.createCipheriv('aes-256-cbc', ...)` and browser
//! SubtleCrypto AES with a zero-dependency AES implementation supporting
//! ECB, CBC, and CTR modes with PKCS7 padding.

use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────

/// AES S-Box lookup table.
const SBOX: [u8; 256] = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
];

/// Inverse S-Box for decryption.
const INV_SBOX: [u8; 256] = [
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d,
];

/// Round constants for key expansion.
const RCON: [u8; 10] = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36];

// ── Errors ─────────────────────────────────────────────────────

/// AES domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AesError {
    /// Invalid key length (must be 16, 24, or 32 bytes).
    InvalidKeyLength(usize),
    /// Invalid IV length (must be 16 bytes).
    InvalidIvLength(usize),
    /// Invalid ciphertext length (not a multiple of 16, or empty).
    InvalidCiphertextLength(usize),
    /// Invalid PKCS7 padding.
    InvalidPadding,
    /// Invalid nonce length for CTR mode (must be 16 bytes).
    InvalidNonceLength(usize),
}

impl std::fmt::Display for AesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKeyLength(n) => write!(f, "invalid key length: {n} (expected 16, 24, or 32)"),
            Self::InvalidIvLength(n) => write!(f, "invalid IV length: {n} (expected 16)"),
            Self::InvalidCiphertextLength(n) => write!(f, "invalid ciphertext length: {n}"),
            Self::InvalidPadding => write!(f, "invalid PKCS7 padding"),
            Self::InvalidNonceLength(n) => write!(f, "invalid nonce length: {n} (expected 16)"),
        }
    }
}

impl std::error::Error for AesError {}

// ── Key size enum ──────────────────────────────────────────────

/// AES key size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AesKeySize {
    Aes128,
    Aes192,
    Aes256,
}

impl AesKeySize {
    pub fn key_bytes(self) -> usize {
        match self {
            Self::Aes128 => 16,
            Self::Aes192 => 24,
            Self::Aes256 => 32,
        }
    }

    pub fn num_rounds(self) -> usize {
        match self {
            Self::Aes128 => 10,
            Self::Aes192 => 12,
            Self::Aes256 => 14,
        }
    }

    pub fn from_key_len(len: usize) -> Result<Self, AesError> {
        match len {
            16 => Ok(Self::Aes128),
            24 => Ok(Self::Aes192),
            32 => Ok(Self::Aes256),
            _ => Err(AesError::InvalidKeyLength(len)),
        }
    }
}

// ── AES Cipher ─────────────────────────────────────────────────

/// AES cipher with expanded round keys.
#[derive(Debug, Clone)]
pub struct AesCipher {
    round_keys: Vec<[u8; 16]>,
    num_rounds: usize,
}

impl AesCipher {
    /// Create a new AES cipher from a key (16, 24, or 32 bytes).
    pub fn new(key: &[u8]) -> Result<Self, AesError> {
        let key_size = AesKeySize::from_key_len(key.len())?;
        let round_keys = key_expansion(key, key_size);
        Ok(Self {
            round_keys,
            num_rounds: key_size.num_rounds(),
        })
    }

    /// Encrypt a single 16-byte block (ECB).
    pub fn encrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let mut state = *block;
        // Initial round key addition.
        xor_block(&mut state, &self.round_keys[0]);

        for round in 1..self.num_rounds {
            sub_bytes(&mut state);
            shift_rows(&mut state);
            mix_columns(&mut state);
            xor_block(&mut state, &self.round_keys[round]);
        }

        // Final round (no MixColumns).
        sub_bytes(&mut state);
        shift_rows(&mut state);
        xor_block(&mut state, &self.round_keys[self.num_rounds]);

        state
    }

    /// Decrypt a single 16-byte block (ECB).
    pub fn decrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let mut state = *block;
        xor_block(&mut state, &self.round_keys[self.num_rounds]);

        for round in (1..self.num_rounds).rev() {
            inv_shift_rows(&mut state);
            inv_sub_bytes(&mut state);
            xor_block(&mut state, &self.round_keys[round]);
            inv_mix_columns(&mut state);
        }

        inv_shift_rows(&mut state);
        inv_sub_bytes(&mut state);
        xor_block(&mut state, &self.round_keys[0]);

        state
    }

    /// CBC encrypt with PKCS7 padding.
    pub fn cbc_encrypt(&self, plaintext: &[u8], iv: &[u8]) -> Result<Vec<u8>, AesError> {
        if iv.len() != 16 {
            return Err(AesError::InvalidIvLength(iv.len()));
        }
        let padded = pkcs7_pad(plaintext);
        let mut result = Vec::with_capacity(padded.len());
        let mut prev: [u8; 16] = iv.try_into().unwrap();

        for chunk in padded.chunks_exact(16) {
            let mut block: [u8; 16] = chunk.try_into().unwrap();
            xor_block(&mut block, &prev);
            prev = self.encrypt_block(&block);
            result.extend_from_slice(&prev);
        }
        Ok(result)
    }

    /// CBC decrypt with PKCS7 unpadding.
    pub fn cbc_decrypt(&self, ciphertext: &[u8], iv: &[u8]) -> Result<Vec<u8>, AesError> {
        if iv.len() != 16 {
            return Err(AesError::InvalidIvLength(iv.len()));
        }
        if ciphertext.is_empty() || ciphertext.len() % 16 != 0 {
            return Err(AesError::InvalidCiphertextLength(ciphertext.len()));
        }

        let mut result = Vec::with_capacity(ciphertext.len());
        let mut prev: [u8; 16] = iv.try_into().unwrap();

        for chunk in ciphertext.chunks_exact(16) {
            let ct_block: [u8; 16] = chunk.try_into().unwrap();
            let mut decrypted = self.decrypt_block(&ct_block);
            xor_block(&mut decrypted, &prev);
            result.extend_from_slice(&decrypted);
            prev = ct_block;
        }

        pkcs7_unpad(&result).map(|s| s.to_vec())
    }

    /// CTR mode encrypt/decrypt (symmetric).
    pub fn ctr_crypt(&self, data: &[u8], nonce: &[u8]) -> Result<Vec<u8>, AesError> {
        if nonce.len() != 16 {
            return Err(AesError::InvalidNonceLength(nonce.len()));
        }
        let mut counter: [u8; 16] = nonce.try_into().unwrap();
        let mut result = Vec::with_capacity(data.len());

        for chunk in data.chunks(16) {
            let keystream = self.encrypt_block(&counter);
            for (i, &byte) in chunk.iter().enumerate() {
                result.push(byte ^ keystream[i]);
            }
            // Increment counter (big-endian last 4 bytes).
            increment_counter(&mut counter);
        }
        Ok(result)
    }
}

// ── PKCS7 padding ──────────────────────────────────────────────

/// Apply PKCS7 padding to make data a multiple of 16 bytes.
pub fn pkcs7_pad(data: &[u8]) -> Vec<u8> {
    let pad_len = 16 - (data.len() % 16);
    let mut padded = Vec::with_capacity(data.len() + pad_len);
    padded.extend_from_slice(data);
    padded.extend(std::iter::repeat(pad_len as u8).take(pad_len));
    padded
}

/// Remove PKCS7 padding.
pub fn pkcs7_unpad(data: &[u8]) -> Result<&[u8], AesError> {
    if data.is_empty() {
        return Err(AesError::InvalidPadding);
    }
    let pad_len = *data.last().unwrap() as usize;
    if pad_len == 0 || pad_len > 16 || pad_len > data.len() {
        return Err(AesError::InvalidPadding);
    }
    // Verify all padding bytes are correct.
    for &b in &data[data.len() - pad_len..] {
        if b != pad_len as u8 {
            return Err(AesError::InvalidPadding);
        }
    }
    Ok(&data[..data.len() - pad_len])
}

// ── Internal helpers ───────────────────────────────────────────

fn sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

fn inv_sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = INV_SBOX[*b as usize];
    }
}

fn shift_rows(state: &mut [u8; 16]) {
    // State is column-major: state[row + 4*col]
    // Row 1: shift left by 1
    let tmp = state[1];
    state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = tmp;
    // Row 2: shift left by 2
    let (t0, t1) = (state[2], state[6]);
    state[2] = state[10]; state[6] = state[14]; state[10] = t0; state[14] = t1;
    // Row 3: shift left by 3 (= right by 1)
    let tmp = state[15];
    state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = tmp;
}

fn inv_shift_rows(state: &mut [u8; 16]) {
    // Row 1: shift right by 1
    let tmp = state[13];
    state[13] = state[9]; state[9] = state[5]; state[5] = state[1]; state[1] = tmp;
    // Row 2: shift right by 2
    let (t0, t1) = (state[2], state[6]);
    state[2] = state[10]; state[6] = state[14]; state[10] = t0; state[14] = t1;
    // Row 3: shift right by 3 (= left by 1)
    let tmp = state[3];
    state[3] = state[7]; state[7] = state[11]; state[11] = state[15]; state[15] = tmp;
}

fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut result = 0u8;
    while b > 0 {
        if b & 1 != 0 {
            result ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b; // irreducible polynomial
        }
        b >>= 1;
    }
    result
}

fn mix_columns(state: &mut [u8; 16]) {
    for col in 0..4 {
        let i = col * 4;
        let (s0, s1, s2, s3) = (state[i], state[i + 1], state[i + 2], state[i + 3]);
        state[i]     = gf_mul(s0, 2) ^ gf_mul(s1, 3) ^ s2 ^ s3;
        state[i + 1] = s0 ^ gf_mul(s1, 2) ^ gf_mul(s2, 3) ^ s3;
        state[i + 2] = s0 ^ s1 ^ gf_mul(s2, 2) ^ gf_mul(s3, 3);
        state[i + 3] = gf_mul(s0, 3) ^ s1 ^ s2 ^ gf_mul(s3, 2);
    }
}

fn inv_mix_columns(state: &mut [u8; 16]) {
    for col in 0..4 {
        let i = col * 4;
        let (s0, s1, s2, s3) = (state[i], state[i + 1], state[i + 2], state[i + 3]);
        state[i]     = gf_mul(s0, 14) ^ gf_mul(s1, 11) ^ gf_mul(s2, 13) ^ gf_mul(s3, 9);
        state[i + 1] = gf_mul(s0, 9) ^ gf_mul(s1, 14) ^ gf_mul(s2, 11) ^ gf_mul(s3, 13);
        state[i + 2] = gf_mul(s0, 13) ^ gf_mul(s1, 9) ^ gf_mul(s2, 14) ^ gf_mul(s3, 11);
        state[i + 3] = gf_mul(s0, 11) ^ gf_mul(s1, 13) ^ gf_mul(s2, 9) ^ gf_mul(s3, 14);
    }
}

fn xor_block(state: &mut [u8; 16], key: &[u8; 16]) {
    for i in 0..16 {
        state[i] ^= key[i];
    }
}

fn increment_counter(counter: &mut [u8; 16]) {
    for i in (0..16).rev() {
        counter[i] = counter[i].wrapping_add(1);
        if counter[i] != 0 {
            break;
        }
    }
}

fn key_expansion(key: &[u8], key_size: AesKeySize) -> Vec<[u8; 16]> {
    let nk = key.len() / 4; // Number of 32-bit words in key
    let nr = key_size.num_rounds();
    let total_words = 4 * (nr + 1);

    let mut w = vec![0u32; total_words];
    // Copy key into first Nk words.
    for i in 0..nk {
        w[i] = u32::from_be_bytes([key[4*i], key[4*i+1], key[4*i+2], key[4*i+3]]);
    }

    for i in nk..total_words {
        let mut temp = w[i - 1];
        if i % nk == 0 {
            // RotWord + SubWord + Rcon
            temp = temp.rotate_left(8);
            let bytes = temp.to_be_bytes();
            temp = u32::from_be_bytes([
                SBOX[bytes[0] as usize],
                SBOX[bytes[1] as usize],
                SBOX[bytes[2] as usize],
                SBOX[bytes[3] as usize],
            ]);
            temp ^= (RCON[i / nk - 1] as u32) << 24;
        } else if nk > 6 && i % nk == 4 {
            let bytes = temp.to_be_bytes();
            temp = u32::from_be_bytes([
                SBOX[bytes[0] as usize],
                SBOX[bytes[1] as usize],
                SBOX[bytes[2] as usize],
                SBOX[bytes[3] as usize],
            ]);
        }
        w[i] = w[i - nk] ^ temp;
    }

    // Pack into 16-byte round keys.
    let mut round_keys = Vec::with_capacity(nr + 1);
    for i in 0..=nr {
        let mut rk = [0u8; 16];
        for j in 0..4 {
            rk[j*4..j*4+4].copy_from_slice(&w[i * 4 + j].to_be_bytes());
        }
        round_keys.push(rk);
    }
    round_keys
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes128_ecb_encrypt() {
        // NIST FIPS 197 Appendix B
        let key = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let plaintext: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
            0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
        ];
        let expected: [u8; 16] = [
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
            0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32,
        ];
        let cipher = AesCipher::new(&key).unwrap();
        let ct = cipher.encrypt_block(&plaintext);
        assert_eq!(ct, expected);
    }

    #[test]
    fn test_aes128_ecb_decrypt() {
        let key = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let ciphertext: [u8; 16] = [
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
            0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32,
        ];
        let expected: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
            0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
        ];
        let cipher = AesCipher::new(&key).unwrap();
        let pt = cipher.decrypt_block(&ciphertext);
        assert_eq!(pt, expected);
    }

    #[test]
    fn test_aes256_ecb_roundtrip() {
        let key = [0x42u8; 32];
        let plaintext: [u8; 16] = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10];
        let cipher = AesCipher::new(&key).unwrap();
        let ct = cipher.encrypt_block(&plaintext);
        let pt = cipher.decrypt_block(&ct);
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_cbc_encrypt_decrypt() {
        let key = [0x00u8; 16];
        let iv = [0x00u8; 16];
        let plaintext = b"Hello, AES-CBC!!"; // exactly 16 bytes
        let cipher = AesCipher::new(&key).unwrap();
        let ct = cipher.cbc_encrypt(plaintext, &iv).unwrap();
        let pt = cipher.cbc_decrypt(&ct, &iv).unwrap();
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_cbc_multi_block() {
        let key = [0xABu8; 16];
        let iv = [0xCDu8; 16];
        let plaintext = b"This is a longer message that spans multiple AES blocks easily.";
        let cipher = AesCipher::new(&key).unwrap();
        let ct = cipher.cbc_encrypt(plaintext, &iv).unwrap();
        let pt = cipher.cbc_decrypt(&ct, &iv).unwrap();
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_ctr_encrypt_decrypt() {
        let key = [0x11u8; 16];
        let nonce = [0x22u8; 16];
        let plaintext = b"CTR mode is symmetric!";
        let cipher = AesCipher::new(&key).unwrap();
        let ct = cipher.ctr_crypt(plaintext, &nonce).unwrap();
        assert_ne!(&ct, plaintext);
        let pt = cipher.ctr_crypt(&ct, &nonce).unwrap();
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_pkcs7_pad_unpad() {
        let data = b"hello";
        let padded = pkcs7_pad(data);
        assert_eq!(padded.len(), 16);
        assert_eq!(padded[5..], [11u8; 11]);
        let unpadded = pkcs7_unpad(&padded).unwrap();
        assert_eq!(unpadded, data);
    }

    #[test]
    fn test_pkcs7_full_block_pad() {
        // When input is already 16 bytes, a full block of padding is added.
        let data = [0x42u8; 16];
        let padded = pkcs7_pad(&data);
        assert_eq!(padded.len(), 32);
        assert_eq!(padded[16..], [16u8; 16]);
    }

    #[test]
    fn test_invalid_key_length() {
        assert!(AesCipher::new(&[0u8; 15]).is_err());
        assert!(AesCipher::new(&[0u8; 17]).is_err());
    }

    #[test]
    fn test_invalid_iv_length() {
        let cipher = AesCipher::new(&[0u8; 16]).unwrap();
        assert!(cipher.cbc_encrypt(b"test", &[0u8; 15]).is_err());
    }

    #[test]
    fn test_invalid_padding() {
        assert!(pkcs7_unpad(&[]).is_err());
        assert!(pkcs7_unpad(&[0x00]).is_err()); // pad byte 0 is invalid
        assert!(pkcs7_unpad(&[0x05, 0x05, 0x05]).is_err()); // claims 5 but only 3 bytes
    }

    #[test]
    fn test_aes192_roundtrip() {
        let key = [0x33u8; 24];
        let plaintext: [u8; 16] = *b"AES-192 test!!!!";
        let cipher = AesCipher::new(&key).unwrap();
        let ct = cipher.encrypt_block(&plaintext);
        let pt = cipher.decrypt_block(&ct);
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_ctr_empty() {
        let cipher = AesCipher::new(&[0u8; 16]).unwrap();
        let ct = cipher.ctr_crypt(b"", &[0u8; 16]).unwrap();
        assert!(ct.is_empty());
    }
}
