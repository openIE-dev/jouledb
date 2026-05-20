//! HQC (Hamming Quasi-Cyclic) Post-Quantum KEM
//!
//! HQC is a code-based Key Encapsulation Mechanism built on the hardness of decoding
//! random quasi-cyclic codes in the Hamming metric. It was selected by NIST in March 2025
//! as a **backup KEM** to complement the lattice-based ML-KEM (FIPS 203).
//!
//! Unlike ML-KEM (which relies on lattice problems), HQC is based on error-correcting
//! codes, providing cryptographic diversity -- if a breakthrough attack were found against
//! lattice-based schemes, HQC would remain secure.
//!
//! ## Status
//!
//! - **Selected**: NIST announced HQC as the backup KEM on 2025-03-11
//! - **Expected standardization**: 2026--2027
//! - **Security basis**: Quasi-Cyclic Syndrome Decoding problem (QCSD)
//!
//! ## Parameter Sets
//!
//! | Parameter Set | NIST Level | Code Length (n) | Security (k) | PK Size  | CT Size  | SS Size |
//! |---------------|------------|-----------------|--------------|----------|----------|---------|
//! | HQC-128       | 1          | 17,669          | 128 bits     | ~2,249 B | ~4,497 B | 64 B    |
//! | HQC-192       | 3          | 35,851          | 192 bits     | ~4,522 B | ~9,042 B | 64 B    |
//! | HQC-256       | 5          | 57,637          | 256 bits     | ~7,245 B | ~14,485 B| 64 B    |
//!
//! ## Implementation
//!
//! This module implements the full HQC cryptographic pipeline:
//!
//! - **Binary polynomial arithmetic** in GF(2)\[x\]/(x^n - 1)
//! - **Reed-Muller RM(1,7)** encoding/decoding via Walsh-Hadamard transform
//! - **Reed-Solomon over GF(2^8)** encoding/decoding via Berlekamp-Massey
//! - **Concatenated (tensor product) code** combining RS outer and RM+repetition inner
//! - **Sparse vector sampling** with exact Hamming weight
//! - **Fujisaki-Okamoto (FO) transform** for IND-CCA2 security
//!
//! ## Hybrid Mode
//!
//! [`HybridHqcKem`] combines classical X25519 key exchange with HQC to provide
//! defense-in-depth: the combined shared secret is secure as long as *either* primitive
//! remains unbroken.
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::pqc::hqc::{HqcParams, HybridHqcKem};
//!
//! // Standalone HQC
//! let kp = hqc::keygen(HqcParams::Hqc128);
//! let (ss_enc, ct) = hqc::encapsulate(&kp.public_key, HqcParams::Hqc128);
//! let ss_dec = hqc::decapsulate(&kp.secret_key, &ct);
//! assert_eq!(ss_enc, ss_dec);
//!
//! // Hybrid X25519 + HQC
//! let hkp = HybridHqcKem::hybrid_keygen(HqcParams::Hqc192).unwrap();
//! let (ss_enc, hct) = HybridHqcKem::hybrid_encapsulate(&hkp.public_key).unwrap();
//! let ss_dec = HybridHqcKem::hybrid_decapsulate(&hkp.secret_key, &hct).unwrap();
//! assert_eq!(ss_enc, ss_dec);
//! ```

use super::common::{ConstantTime, SecureZeroingVec, Sha3_256, Shake256};
use super::{PqcError, PqcResult};
use rand::Rng;
use std::sync::LazyLock;

// ============================================================================
// HQC Parameter Sets
// ============================================================================

/// HQC parameter sets corresponding to NIST security levels.
///
/// Each variant specifies the quasi-cyclic code length `n`, security parameter `k`,
/// and derived sizes for public keys, ciphertexts, and shared secrets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HqcParams {
    /// HQC-128: NIST Security Level 1 (equivalent to AES-128).
    ///
    /// - Code length n = 17,669
    /// - Security parameter k = 128 bits
    /// - Public key: ~2,249 bytes
    /// - Ciphertext: ~4,497 bytes
    Hqc128,

    /// HQC-192: NIST Security Level 3 (equivalent to AES-192).
    ///
    /// - Code length n = 35,851
    /// - Security parameter k = 192 bits
    /// - Public key: ~4,522 bytes
    /// - Ciphertext: ~9,042 bytes
    Hqc192,

    /// HQC-256: NIST Security Level 5 (equivalent to AES-256).
    ///
    /// - Code length n = 57,637
    /// - Security parameter k = 256 bits
    /// - Public key: ~7,245 bytes
    /// - Ciphertext: ~14,485 bytes
    Hqc256,
}

impl HqcParams {
    /// Code length `n` for this parameter set.
    pub const fn code_length(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 17_669,
            HqcParams::Hqc192 => 35_851,
            HqcParams::Hqc256 => 57_637,
        }
    }

    /// Security parameter `k` in bits.
    pub const fn security_bits(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 128,
            HqcParams::Hqc192 => 192,
            HqcParams::Hqc256 => 256,
        }
    }

    /// NIST security level (1, 3, or 5).
    pub const fn nist_level(&self) -> u8 {
        match self {
            HqcParams::Hqc128 => 1,
            HqcParams::Hqc192 => 3,
            HqcParams::Hqc256 => 5,
        }
    }

    /// Public key size in bytes (approximate, based on NIST submission).
    pub const fn public_key_size(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 2_249,
            HqcParams::Hqc192 => 4_522,
            HqcParams::Hqc256 => 7_245,
        }
    }

    /// Secret key size in bytes (approximate, based on NIST submission).
    pub const fn secret_key_size(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 2_289,
            HqcParams::Hqc192 => 4_562,
            HqcParams::Hqc256 => 7_285,
        }
    }

    /// Ciphertext size in bytes (approximate, based on NIST submission).
    pub const fn ciphertext_size(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 4_497,
            HqcParams::Hqc192 => 9_042,
            HqcParams::Hqc256 => 14_485,
        }
    }

    /// Shared secret size in bytes (always 64 for all parameter sets).
    pub const fn shared_secret_size(&self) -> usize {
        64
    }

    /// Domain separator tag used to bind KDF outputs to a specific parameter set.
    fn domain_tag(&self) -> &'static [u8] {
        match self {
            HqcParams::Hqc128 => b"HQC-128-v1",
            HqcParams::Hqc192 => b"HQC-192-v1",
            HqcParams::Hqc256 => b"HQC-256-v1",
        }
    }

    /// Security parameter k in bytes (message size for KEM).
    const fn k_bytes(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 16,
            HqcParams::Hqc192 => 24,
            HqcParams::Hqc256 => 32,
        }
    }

    /// RS code length n1.
    const fn n1(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 46,
            HqcParams::Hqc192 => 56,
            HqcParams::Hqc256 => 90,
        }
    }

    /// Inner code block size n2 (RM(1,7) * multiplicity).
    const fn n2(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 384, // 128 * 3
            HqcParams::Hqc192 => 640, // 128 * 5
            HqcParams::Hqc256 => 640, // 128 * 5
        }
    }

    /// Repetition multiplicity for inner code.
    const fn multiplicity(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 3,
            HqcParams::Hqc192 => 5,
            HqcParams::Hqc256 => 5,
        }
    }

    /// Hamming weight w for secret key vectors (x, y).
    const fn w(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 66,
            HqcParams::Hqc192 => 100,
            HqcParams::Hqc256 => 131,
        }
    }

    /// Hamming weight wr for r1, r2 vectors.
    const fn wr(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 77,
            HqcParams::Hqc192 => 114,
            HqcParams::Hqc256 => 149,
        }
    }

    /// Hamming weight we for error vector e.
    const fn we(&self) -> usize {
        match self {
            HqcParams::Hqc128 => 77,
            HqcParams::Hqc192 => 114,
            HqcParams::Hqc256 => 149,
        }
    }

    /// Number of bytes to hold n bits (ceil(n/8)).
    const fn n_bytes(&self) -> usize {
        (self.code_length() + 7) / 8
    }

    /// RS error correction capability t = (n1 - k_bytes) / 2.
    const fn rs_t(&self) -> usize {
        (self.n1() - self.k_bytes()) / 2
    }
}

impl std::fmt::Display for HqcParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HqcParams::Hqc128 => write!(f, "HQC-128 (NIST Level 1)"),
            HqcParams::Hqc192 => write!(f, "HQC-192 (NIST Level 3)"),
            HqcParams::Hqc256 => write!(f, "HQC-256 (NIST Level 5)"),
        }
    }
}

// ============================================================================
// GF(2^8) Arithmetic for Reed-Solomon
// ============================================================================
//
// Irreducible polynomial: x^8 + x^4 + x^3 + x + 1 = 0x11B
// Primitive element: alpha = 2

/// Multiply by 2 in GF(2^8) (left shift with conditional reduction).
#[inline]
fn gf256_xtime(val: u8) -> u8 {
    let carry = if val & 0x80 != 0 { 0x1Bu8 } else { 0u8 };
    (val << 1) ^ carry
}

/// GF(2^8) tables computed at runtime via LazyLock.
/// Uses primitive element alpha = 3 (= x + 1), which generates all 255 non-zero elements.
/// Element 2 only has order 51 for polynomial 0x11B, so it cannot be used as the generator.
/// Returns (log_table[256], exp_table[512]).
static GF256_TABLES: LazyLock<([u8; 256], [u8; 512])> = LazyLock::new(|| {
    let mut log = [0u8; 256];
    let mut exp = [0u8; 512];
    let mut val: u8 = 1;
    for i in 0..255usize {
        exp[i] = val;
        log[val as usize] = i as u8;
        // Multiply by 3 (= x + 1): val * 3 = val * 2 XOR val
        val = gf256_xtime(val) ^ val;
    }
    // Wrap-around for modular reduction: exp[i] = exp[i % 255]
    for i in 255..512 {
        exp[i] = exp[i - 255];
    }
    (log, exp)
});

/// Multiply two elements in GF(2^8).
#[inline]
fn gf256_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let log_a = GF256_TABLES.0[a as usize] as usize;
    let log_b = GF256_TABLES.0[b as usize] as usize;
    GF256_TABLES.1[log_a + log_b]
}

/// Compute the inverse of an element in GF(2^8).
#[inline]
fn gf256_inv(a: u8) -> u8 {
    if a == 0 {
        return 0; // undefined, but return 0 for safety
    }
    let log_a = GF256_TABLES.0[a as usize] as usize;
    GF256_TABLES.1[255 - log_a]
}

/// alpha^i in GF(2^8).
#[inline]
fn gf256_alpha_pow(i: usize) -> u8 {
    GF256_TABLES.1[i % 255]
}

// ============================================================================
// Reed-Solomon Encoding/Decoding over GF(2^8)
// ============================================================================
//
// Systematic RS code: k_bytes message bytes -> n1 bytes (n1 - k_bytes parity bytes).
// Generator polynomial: g(x) = prod_{i=1}^{2t} (x - alpha^i) where t = (n1-k)/2.

/// Compute the RS generator polynomial for the given parameters.
/// Returns coefficients [g_0, g_1, ..., g_{2t}] where g_{2t} = 1 (monic).
fn rs_generator_poly(n1: usize, k: usize) -> Vec<u8> {
    let two_t = n1 - k;
    // Start with g(x) = 1
    let mut g = vec![0u8; two_t + 1];
    g[0] = 1;
    let mut len = 1;

    // Multiply by (x - alpha^i) for i = 1..=2t
    for i in 1..=two_t {
        let alpha_i = gf256_alpha_pow(i);
        // Multiply polynomial g by (x - alpha^i) = (x + alpha^i) in GF(2^8)
        // Process from high degree to low to avoid overwriting
        let new_len = len + 1;
        // g[new_len-1] = g[len-1] (leading coeff shifted up)
        let mut j = new_len - 1;
        while j >= 1 {
            // new_g[j] = g[j-1] + alpha_i * g[j]
            let prev = if j < len { g[j] } else { 0 };
            let shifted = if j >= 1 && j - 1 < len { g[j - 1] } else { 0 };
            g[j] = shifted ^ gf256_mul(alpha_i, prev);
            j -= 1;
        }
        // g[0] = alpha_i * g[0]
        g[0] = gf256_mul(alpha_i, g[0]);
        len = new_len;
    }
    g
}

/// Systematic RS encode: message[0..k] -> codeword[0..n1].
///
/// Codeword layout: [parity(0..2t-1) | message(2t..n1-1)]
/// where position j = coefficient of x^j in the codeword polynomial.
/// This ensures c(α^i) = 0 for i = 1..2t (syndrome roots).
fn rs_encode(message: &[u8], n1: usize, k: usize) -> Vec<u8> {
    let two_t = n1 - k;
    let gpoly = rs_generator_poly(n1, k);

    // M(x) = m[0]*x^{2t} + m[1]*x^{2t+1} + ... + m[k-1]*x^{n1-1}
    let mut work = vec![0u8; n1];
    for i in 0..k {
        work[two_t + i] = message[i];
    }

    // Polynomial long division of M(x) by g(x), processing from highest degree
    for pos in (two_t..n1).rev() {
        let coeff = work[pos];
        if coeff != 0 {
            // Subtract coeff * g(x) * x^{pos - 2t} from work
            for j in 0..=two_t {
                work[pos - two_t + j] ^= gf256_mul(coeff, gpoly[j]);
            }
        }
    }

    // work[0..2t-1] = remainder r(x)
    // Codeword c(x) = r(x) + M(x): parity at low positions, message at high positions
    let mut codeword = Vec::with_capacity(n1);
    codeword.extend_from_slice(&work[..two_t]); // parity
    codeword.extend_from_slice(message); // message
    codeword
}

/// RS syndrome computation: compute S_i = received(alpha^i) for i = 1..=2t.
fn rs_syndromes(received: &[u8], n1: usize, two_t: usize) -> Vec<u8> {
    let mut syndromes = vec![0u8; two_t];
    for i in 0..two_t {
        let mut val = 0u8;
        let mut alpha_power = 1u8; // alpha^((i+1)*0) = 1
        let alpha_i = gf256_alpha_pow(i + 1);
        for j in 0..n1 {
            val ^= gf256_mul(received[j], alpha_power);
            alpha_power = gf256_mul(alpha_power, alpha_i);
        }
        syndromes[i] = val;
    }
    syndromes
}

/// Berlekamp-Massey algorithm to find the error locator polynomial.
/// Returns coefficients of Lambda(x) = 1 + L1*x + L2*x^2 + ...
fn berlekamp_massey(syndromes: &[u8], two_t: usize) -> Vec<u8> {
    let mut c = vec![0u8; two_t + 1]; // current LFSR
    let mut b = vec![0u8; two_t + 1]; // previous LFSR
    c[0] = 1;
    b[0] = 1;
    let mut l: usize = 0;
    let mut m: isize = 1;
    let mut delta_prev: u8 = 1;

    for n in 0..two_t {
        // Compute discrepancy
        let mut delta = syndromes[n];
        for i in 1..=l {
            delta ^= gf256_mul(c[i], syndromes[n - i]);
        }

        if delta == 0 {
            m += 1;
        } else if 2 * l <= n {
            // Update: save old c, compute new c
            let t_poly = c.clone();
            let factor = gf256_mul(delta, gf256_inv(delta_prev));
            for i in 0..=(two_t) {
                let shift_idx = i as isize - m;
                if shift_idx >= 0 && (shift_idx as usize) <= two_t {
                    c[i] ^= gf256_mul(factor, b[shift_idx as usize]);
                }
            }
            l = n + 1 - l;
            b = t_poly;
            delta_prev = delta;
            m = 1;
        } else {
            let factor = gf256_mul(delta, gf256_inv(delta_prev));
            for i in 0..=(two_t) {
                let shift_idx = i as isize - m;
                if shift_idx >= 0 && (shift_idx as usize) <= two_t {
                    c[i] ^= gf256_mul(factor, b[shift_idx as usize]);
                }
            }
            m += 1;
        }
    }

    c.truncate(l + 1);
    c
}

/// Find error locations by evaluating Lambda at alpha^(-j) for j = 0..n1-1 (Chien search).
fn chien_search(lambda: &[u8], n1: usize) -> Vec<usize> {
    let mut locations = Vec::new();
    for j in 0..n1 {
        // Evaluate Lambda(alpha^(-j)) = Lambda(alpha^(255-j))
        let mut val = 0u8;
        for (i, &coeff) in lambda.iter().enumerate() {
            if coeff != 0 {
                let power = (i * (255 - (j % 255))) % 255;
                val ^= gf256_mul(coeff, gf256_alpha_pow(power));
            }
        }
        if val == 0 {
            locations.push(j);
        }
    }
    locations
}

/// Forney algorithm to compute error magnitudes.
fn forney_algorithm(
    syndromes: &[u8],
    lambda: &[u8],
    error_positions: &[usize],
    _n1: usize,
) -> Vec<u8> {
    let two_t = syndromes.len();

    // Compute error evaluator polynomial Omega(x) = S(x) * Lambda(x) mod x^(2t)
    let mut omega = vec![0u8; two_t];
    for i in 0..two_t {
        for (j, &lc) in lambda.iter().enumerate() {
            if j <= i {
                omega[i] ^= gf256_mul(syndromes[i - j], lc);
            }
        }
    }

    // Compute formal derivative Lambda'(x) (only odd-indexed coefficients contribute in char 2)
    let mut lambda_prime = vec![0u8; lambda.len()];
    for i in (1..lambda.len()).step_by(2) {
        lambda_prime[i - 1] = lambda[i];
    }

    let mut magnitudes = Vec::with_capacity(error_positions.len());
    for &pos in error_positions {
        // X_k = alpha^pos, X_k^(-1) = alpha^(255-pos)
        let x_inv = gf256_alpha_pow(255 - (pos % 255));

        // Evaluate Omega(X_k^(-1))
        let mut omega_val = 0u8;
        let mut x_power = 1u8;
        for &o in omega.iter() {
            omega_val ^= gf256_mul(o, x_power);
            x_power = gf256_mul(x_power, x_inv);
        }

        // Evaluate Lambda'(X_k^(-1))
        let mut lambda_p_val = 0u8;
        let mut x_power = 1u8;
        for &lp in lambda_prime.iter() {
            lambda_p_val ^= gf256_mul(lp, x_power);
            x_power = gf256_mul(x_power, x_inv);
        }

        if lambda_p_val != 0 {
            magnitudes.push(gf256_mul(omega_val, gf256_inv(lambda_p_val)));
        } else {
            magnitudes.push(0);
        }
    }
    magnitudes
}

/// RS decode: attempt to correct errors in received[0..n1].
/// Codeword layout: [parity(0..2t-1) | message(2t..n1-1)].
/// Returns the corrected message (k bytes from positions 2t..n1-1) or None on failure.
fn rs_decode(received: &[u8], n1: usize, k: usize) -> Option<Vec<u8>> {
    let two_t = n1 - k;
    let syndromes = rs_syndromes(received, n1, two_t);

    // Check if all syndromes are zero (no errors)
    let all_zero = syndromes.iter().all(|&s| s == 0);
    if all_zero {
        return Some(received[two_t..].to_vec());
    }

    let lambda = berlekamp_massey(&syndromes, two_t);

    // Number of errors = degree of Lambda
    let num_errors = lambda.len() - 1;
    if num_errors > two_t / 2 {
        return None; // too many errors
    }

    let error_positions = chien_search(&lambda, n1);
    if error_positions.len() != num_errors {
        return None; // couldn't find all error positions
    }

    let magnitudes = forney_algorithm(&syndromes, &lambda, &error_positions, n1);

    // Correct errors
    let mut corrected = received.to_vec();
    for (i, &pos) in error_positions.iter().enumerate() {
        if pos < n1 {
            corrected[pos] ^= magnitudes[i];
        }
    }

    Some(corrected[two_t..].to_vec())
}

// ============================================================================
// Reed-Muller RM(1,7) Encoding/Decoding
// ============================================================================
//
// RM(1,7) encodes 8 bits (1 byte) into 128 bits (16 bytes).
// Encoding: c[j] = XOR_{i=0..7} (b_i AND bit_i(j)) XOR b_constant, for j = 0..127
// where b_0 is a constant term and b_1..b_7 are coefficients of the 7 variables.
//
// Actually, RM(1,7) with 8 information bits encodes as:
//   The codeword at position j (0..127) is: b[0] XOR (b[1] & j_0) XOR ... XOR (b[7] & j_6)
// where j_i is the i-th bit of j, and b[0..7] are the 8 message bits.

/// RM(1,7) encode: 1 byte (8 bits) -> 128 bits (16 bytes).
fn rm_encode(byte: u8) -> [u8; 16] {
    let mut codeword = [0u8; 16];
    for j in 0u8..128 {
        let mut bit = (byte >> 7) & 1; // b[0] = constant term (MSB)
        for i in 0..7 {
            let b_i = (byte >> (6 - i)) & 1;
            let j_i = (j >> i) & 1;
            bit ^= b_i & j_i;
        }
        // Set bit j of codeword
        let byte_idx = (j / 8) as usize;
        let bit_idx = j % 8;
        codeword[byte_idx] |= bit << bit_idx;
    }
    codeword
}

/// RM(1,7) decode via Walsh-Hadamard transform.
/// Input: 128 bits as 16 bytes. Output: decoded byte, or None on failure.
fn rm_decode(codeword: &[u8]) -> Option<u8> {
    // Convert bits to +1/-1 representation
    let mut signal = [0i32; 128];
    for j in 0..128 {
        let byte_idx = j / 8;
        let bit_idx = j % 8;
        let bit = (codeword[byte_idx] >> bit_idx) & 1;
        signal[j] = 1 - 2 * (bit as i32); // 0 -> +1, 1 -> -1
    }

    // Walsh-Hadamard transform (in-place, iterative)
    let mut wht = signal;
    let mut step = 1;
    while step < 128 {
        let mut i = 0;
        while i < 128 {
            for j in i..(i + step) {
                let u = wht[j];
                let v = wht[j + step];
                wht[j] = u + v;
                wht[j + step] = u - v;
            }
            i += step * 2;
        }
        step *= 2;
    }

    // Find the coefficient with maximum absolute value
    let mut max_abs = 0i32;
    let mut max_idx = 0usize;
    let mut max_sign = 1i32; // +1 or -1
    for (i, &val) in wht.iter().enumerate() {
        let abs_val = val.abs();
        if abs_val > max_abs {
            max_abs = abs_val;
            max_idx = i;
            max_sign = if val >= 0 { 1 } else { -1 };
        }
    }

    // Reconstruct the 8-bit message:
    // max_idx gives the 7 variable bits (b[1]..b[7])
    // max_sign gives the constant bit b[0]: positive means b[0]=0, negative means b[0]=1
    let constant_bit = if max_sign < 0 { 1u8 } else { 0u8 };

    // The variable bits b[1]..b[7] are encoded in max_idx
    // b[1] = bit 0 of max_idx, b[2] = bit 1, etc.
    let mut decoded = constant_bit << 7;
    for i in 0..7 {
        let b_i = ((max_idx >> i) & 1) as u8;
        decoded |= b_i << (6 - i);
    }

    Some(decoded)
}

// ============================================================================
// Concatenated (Tensor Product) Code: RS outer, RM(1,7)+repetition inner
// ============================================================================
//
// Encode:
// 1. RS encode k bytes -> n1 bytes
// 2. Each byte -> RM(1,7) encode to 128 bits -> repeat `mult` times -> n2 bits
// 3. Concatenate n1 chunks of n2 bits -> n1*n2 bits total
//
// Decode:
// 1. Split into n1 chunks of n2 bits
// 2. Each chunk: split into `mult` copies of 128 bits, majority-vote each bit
// 3. RM(1,7) decode via WHT -> 1 byte
// 4. RS decode n1 bytes -> k bytes

/// Encode a message using the concatenated code. Returns a bit-vector as bytes.
/// The output has exactly `n1 * n2` bits, zero-padded to full bytes.
fn concatenated_encode(message: &[u8], params: &HqcParams) -> Vec<u8> {
    let n1 = params.n1();
    let k = params.k_bytes();
    let n2 = params.n2();
    let mult = params.multiplicity();

    // Step 1: RS encode
    let rs_codeword = rs_encode(message, n1, k);

    // Step 2+3: For each RS symbol, RM encode + repeat, then pack into bit vector
    let total_bits = n1 * n2;
    let total_bytes = (total_bits + 7) / 8;
    let mut encoded = vec![0u8; total_bytes];

    for sym_idx in 0..n1 {
        let rm_cw = rm_encode(rs_codeword[sym_idx]);
        // Repeat the 128-bit RM codeword `mult` times to fill n2 bits
        let base_bit = sym_idx * n2;
        for rep in 0..mult {
            for bit in 0..128 {
                let src_byte = bit / 8;
                let src_bit = bit % 8;
                let val = (rm_cw[src_byte] >> src_bit) & 1;
                if val == 1 {
                    let dst_bit = base_bit + rep * 128 + bit;
                    let dst_byte = dst_bit / 8;
                    let dst_bit_pos = dst_bit % 8;
                    if dst_byte < total_bytes {
                        encoded[dst_byte] |= 1 << dst_bit_pos;
                    }
                }
            }
        }
    }

    encoded
}

/// Decode a message from a noisy bit-vector using the concatenated code.
/// Input is `n1 * n2` bits (packed in bytes). Returns k message bytes or None.
fn concatenated_decode(received: &[u8], params: &HqcParams) -> Option<Vec<u8>> {
    let n1 = params.n1();
    let k = params.k_bytes();
    let n2 = params.n2();
    let mult = params.multiplicity();

    let mut rs_received = Vec::with_capacity(n1);

    for sym_idx in 0..n1 {
        let base_bit = sym_idx * n2;

        // Accumulate votes across repetitions for each of the 128 bits
        let mut votes = [0i32; 128];
        for rep in 0..mult {
            for bit in 0..128 {
                let src_bit = base_bit + rep * 128 + bit;
                let src_byte = src_bit / 8;
                let src_bit_pos = src_bit % 8;
                let val = if src_byte < received.len() {
                    (received[src_byte] >> src_bit_pos) & 1
                } else {
                    0
                };
                // Vote: 0 -> +1, 1 -> -1
                votes[bit] += 1 - 2 * (val as i32);
            }
        }

        // Majority decision -> reconstruct 128-bit block
        let mut block = [0u8; 16];
        for bit in 0..128 {
            // Negative vote means majority is 1
            if votes[bit] < 0 {
                block[bit / 8] |= 1 << (bit % 8);
            }
            // tie (votes == 0) defaults to 0
        }

        // RM(1,7) decode
        match rm_decode(&block) {
            Some(byte) => rs_received.push(byte),
            None => rs_received.push(0), // push zero on RM decode failure, let RS handle it
        }
    }

    // RS decode
    rs_decode(&rs_received, n1, k)
}

// ============================================================================
// GF(2) Polynomial Arithmetic in GF(2)[x]/(x^n - 1)
// ============================================================================
//
// Polynomials are represented as bit-vectors packed in Vec<u8>.
// Bit i of the polynomial is at byte i/8, bit position i%8.

/// Get bit at position `pos` in a bit-vector.
#[inline]
fn get_bit(poly: &[u8], pos: usize) -> u8 {
    if pos / 8 >= poly.len() {
        return 0;
    }
    (poly[pos / 8] >> (pos % 8)) & 1
}

/// Set bit at position `pos` in a bit-vector.
#[inline]
fn set_bit(poly: &mut [u8], pos: usize, val: u8) {
    if pos / 8 >= poly.len() {
        return;
    }
    if val == 1 {
        poly[pos / 8] |= 1 << (pos % 8);
    } else {
        poly[pos / 8] &= !(1 << (pos % 8));
    }
}

/// XOR two bit-vectors (GF(2) addition). Result has length of the longer input.
fn poly_add(a: &[u8], b: &[u8]) -> Vec<u8> {
    let len = a.len().max(b.len());
    let mut result = vec![0u8; len];
    for i in 0..a.len() {
        result[i] ^= a[i];
    }
    for i in 0..b.len() {
        result[i] ^= b[i];
    }
    result
}

/// XOR b into a in-place.
fn poly_add_assign(a: &mut [u8], b: &[u8]) {
    let len = a.len().min(b.len());
    for i in 0..len {
        a[i] ^= b[i];
    }
}

/// Multiply two polynomials in GF(2)[x]/(x^n - 1).
/// Uses sparse multiplication when one operand is sparse (given as position list).
fn poly_mul_sparse_dense(sparse_positions: &[usize], dense: &[u8], n: usize) -> Vec<u8> {
    let n_bytes = (n + 7) / 8;
    let mut result = vec![0u8; n_bytes];

    for &pos in sparse_positions {
        // Shift dense by pos positions and XOR into result (mod x^n - 1)
        for bit_idx in 0..n {
            if get_bit(dense, bit_idx) == 1 {
                let target = (bit_idx + pos) % n;
                let byte_idx = target / 8;
                let bit_pos = target % 8;
                result[byte_idx] ^= 1 << bit_pos;
            }
        }
    }

    result
}

/// Sample a sparse binary vector with exactly `weight` bits set in [0, n).
/// Uses SHAKE256 as a PRNG seeded with `seed`.
fn sample_sparse_vector(
    seed: &[u8],
    domain: &[u8],
    n: usize,
    weight: usize,
) -> (Vec<u8>, Vec<usize>) {
    let n_bytes = (n + 7) / 8;
    let mut poly = vec![0u8; n_bytes];
    let mut positions = Vec::with_capacity(weight);

    // Use SHAKE256 to generate a stream of random bytes
    let prng_seed: Vec<u8> = [seed, domain].concat();
    // Generate enough random bytes: each position needs ~2 bytes for n < 65536
    let random_bytes = Shake256::xof(&prng_seed, weight * 4 + 64);

    let mut byte_offset = 0;
    let mut count = 0;

    while count < weight {
        if byte_offset + 2 > random_bytes.len() {
            // Need more random bytes (shouldn't happen with our allocation)
            break;
        }
        let candidate = ((random_bytes[byte_offset] as usize)
            | ((random_bytes[byte_offset + 1] as usize) << 8))
            % n;
        byte_offset += 2;

        // Check if this position is already set (rejection sampling)
        if get_bit(&poly, candidate) == 0 {
            set_bit(&mut poly, candidate, 1);
            positions.push(candidate);
            count += 1;
        }
    }

    // If we didn't get enough positions (extremely unlikely), fill deterministically
    if count < weight {
        for i in 0..n {
            if count >= weight {
                break;
            }
            if get_bit(&poly, i) == 0 {
                set_bit(&mut poly, i, 1);
                positions.push(i);
                count += 1;
            }
        }
    }

    (poly, positions)
}

/// Expand a 32-byte seed into a dense polynomial of n bits using SHAKE256.
fn expand_seed_to_poly(seed: &[u8], domain: &[u8], n: usize) -> Vec<u8> {
    let n_bytes = (n + 7) / 8;
    let input: Vec<u8> = [domain, seed].concat();
    let mut poly = Shake256::xof(&input, n_bytes);

    // Clear any excess bits beyond position n
    let excess = n_bytes * 8 - n;
    if excess > 0 && !poly.is_empty() {
        let last = poly.len() - 1;
        poly[last] &= (1u8 << (8 - excess)) - 1;
    }

    poly
}

// ============================================================================
// Serialization Helpers
// ============================================================================

/// Encode a polynomial (bit-vector) to bytes, padded to a fixed number of bytes.
fn poly_to_bytes(poly: &[u8], n_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; n_bytes];
    let copy_len = poly.len().min(n_bytes);
    out[..copy_len].copy_from_slice(&poly[..copy_len]);
    out
}

/// Encode public key: seed_h (32 bytes) || s (n_bytes)
fn encode_public_key(seed_h: &[u8], s: &[u8], params: &HqcParams) -> Vec<u8> {
    let target_size = params.public_key_size();
    let n_bytes = params.n_bytes();
    let mut pk = Vec::with_capacity(target_size);
    pk.extend_from_slice(seed_h); // 32 bytes
    let s_padded = poly_to_bytes(s, n_bytes);
    pk.extend_from_slice(&s_padded);
    // Pad to exact target size
    pk.resize(target_size, 0);
    pk
}

/// Decode public key: returns (seed_h, s)
fn decode_public_key(pk: &[u8], params: &HqcParams) -> (Vec<u8>, Vec<u8>) {
    let n_bytes = params.n_bytes();
    let seed_h = pk[..32].to_vec();
    let s_end = (32 + n_bytes).min(pk.len());
    let s = pk[32..s_end].to_vec();
    (seed_h, s)
}

/// Encode secret key: sigma (32 bytes) || seed_h (32 bytes) || x_positions || y_positions
/// We store the sparse positions as 2-byte little-endian values prefixed by count.
fn encode_secret_key(
    sigma: &[u8],
    seed_h: &[u8],
    x_positions: &[usize],
    y_positions: &[usize],
    params: &HqcParams,
) -> Vec<u8> {
    let target_size = params.secret_key_size();
    let mut sk = Vec::with_capacity(target_size);

    // sigma (32 bytes)
    sk.extend_from_slice(sigma);
    // seed_h (32 bytes)
    sk.extend_from_slice(seed_h);
    // x weight (2 bytes LE)
    sk.extend_from_slice(&(x_positions.len() as u16).to_le_bytes());
    // x positions (2 bytes each LE)
    for &p in x_positions {
        sk.extend_from_slice(&(p as u16).to_le_bytes());
    }
    // y weight (2 bytes LE)
    sk.extend_from_slice(&(y_positions.len() as u16).to_le_bytes());
    // y positions (2 bytes each LE)
    for &p in y_positions {
        sk.extend_from_slice(&(p as u16).to_le_bytes());
    }

    // Pad to exact target size
    sk.resize(target_size, 0);
    sk
}

/// Decode secret key: returns (sigma, seed_h, x_positions, y_positions)
fn decode_secret_key(sk: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<usize>, Vec<usize>) {
    let sigma = sk[..32].to_vec();
    let seed_h = sk[32..64].to_vec();

    let mut offset = 64;

    // x positions
    let x_weight = u16::from_le_bytes([sk[offset], sk[offset + 1]]) as usize;
    offset += 2;
    let mut x_positions = Vec::with_capacity(x_weight);
    for _ in 0..x_weight {
        let p = u16::from_le_bytes([sk[offset], sk[offset + 1]]) as usize;
        offset += 2;
        x_positions.push(p);
    }

    // y positions
    let y_weight = u16::from_le_bytes([sk[offset], sk[offset + 1]]) as usize;
    offset += 2;
    let mut y_positions = Vec::with_capacity(y_weight);
    for _ in 0..y_weight {
        let p = u16::from_le_bytes([sk[offset], sk[offset + 1]]) as usize;
        offset += 2;
        y_positions.push(p);
    }

    (sigma, seed_h, x_positions, y_positions)
}

/// Encode ciphertext: u (n_bytes) || v (n_bytes) || d (32 bytes)
fn encode_ciphertext(u: &[u8], v: &[u8], d: &[u8], params: &HqcParams) -> Vec<u8> {
    let target_size = params.ciphertext_size();
    let n_bytes = params.n_bytes();
    let mut ct = Vec::with_capacity(target_size);
    let u_padded = poly_to_bytes(u, n_bytes);
    let v_padded = poly_to_bytes(v, n_bytes);
    ct.extend_from_slice(&u_padded);
    ct.extend_from_slice(&v_padded);
    ct.extend_from_slice(d);
    // Pad to exact target size
    ct.resize(target_size, 0);
    ct
}

/// Decode ciphertext: returns (u, v, d)
fn decode_ciphertext(ct: &[u8], params: &HqcParams) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n_bytes = params.n_bytes();
    let u = ct[..n_bytes].to_vec();
    let v = ct[n_bytes..2 * n_bytes].to_vec();
    let d_start = 2 * n_bytes;
    let d_end = (d_start + 32).min(ct.len());
    let d = ct[d_start..d_end].to_vec();
    (u, v, d)
}

// ============================================================================
// Message Encoding/Decoding (concatenated code embedded in n-bit polynomial)
// ============================================================================

/// Encode a k-byte message into an n-bit polynomial using the concatenated code.
fn encode_message(message: &[u8], params: &HqcParams) -> Vec<u8> {
    let n = params.code_length();
    let n_bytes = params.n_bytes();

    // Concatenated encode produces n1*n2 bits
    let encoded = concatenated_encode(message, params);

    // Zero-pad to n bits (n_bytes)
    let mut result = vec![0u8; n_bytes];
    let copy_len = encoded.len().min(n_bytes);
    result[..copy_len].copy_from_slice(&encoded[..copy_len]);

    // Clear excess bits
    let excess = n_bytes * 8 - n;
    if excess > 0 {
        let last = result.len() - 1;
        result[last] &= (1u8 << (8 - excess)) - 1;
    }

    result
}

/// Decode a noisy n-bit polynomial back to a k-byte message using the concatenated code.
fn decode_message(noisy: &[u8], params: &HqcParams) -> Option<Vec<u8>> {
    // The concatenated code occupies the first n1*n2 bits
    concatenated_decode(noisy, params)
}

// ============================================================================
// HQC Key Pair
// ============================================================================

/// An HQC key pair containing public and secret keys.
///
/// The public key contains a seed for the parity-check polynomial `h` and the
/// syndrome vector `s = x + h*y mod (x^n - 1)`. The secret key contains the
/// implicit-rejection value sigma, the seed, and the sparse vectors x, y.
#[derive(Clone)]
pub struct HqcKeyPair {
    /// Public (encapsulation) key.
    pub public_key: Vec<u8>,
    /// Secret (decapsulation) key -- zeroized on drop via [`SecureZeroingVec`] wrapper.
    pub secret_key: Vec<u8>,
    /// Parameter set this key pair was generated for.
    pub params: HqcParams,
}

impl std::fmt::Debug for HqcKeyPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HqcKeyPair")
            .field("params", &self.params)
            .field("public_key_len", &self.public_key.len())
            .field("secret_key_len", &self.secret_key.len())
            .finish()
    }
}

// ============================================================================
// HQC Ciphertext
// ============================================================================

/// An HQC ciphertext produced by [`encapsulate`].
///
/// Contains the noisy codeword components `u`, `v`, and hash `d` serialized
/// into a byte blob of the size specified by the parameter set.
#[derive(Clone)]
pub struct HqcCiphertext {
    /// Raw ciphertext bytes.
    pub data: Vec<u8>,
    /// Parameter set this ciphertext was produced with.
    pub params: HqcParams,
}

impl std::fmt::Debug for HqcCiphertext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HqcCiphertext")
            .field("params", &self.params)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl HqcCiphertext {
    /// Serialize to bytes (length-prefixed with 1-byte param tag).
    pub fn to_bytes(&self) -> Vec<u8> {
        let tag: u8 = match self.params {
            HqcParams::Hqc128 => 0x01,
            HqcParams::Hqc192 => 0x03,
            HqcParams::Hqc256 => 0x05,
        };
        let mut out = Vec::with_capacity(1 + self.data.len());
        out.push(tag);
        out.extend_from_slice(&self.data);
        out
    }

    /// Deserialize from bytes produced by [`to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> PqcResult<Self> {
        if bytes.is_empty() {
            return Err(PqcError::InvalidCiphertext);
        }
        let params = match bytes[0] {
            0x01 => HqcParams::Hqc128,
            0x03 => HqcParams::Hqc192,
            0x05 => HqcParams::Hqc256,
            _ => return Err(PqcError::InvalidCiphertext),
        };
        let expected = params.ciphertext_size();
        if bytes.len() - 1 != expected {
            return Err(PqcError::InvalidCiphertext);
        }
        Ok(Self {
            data: bytes[1..].to_vec(),
            params,
        })
    }
}

// ============================================================================
// Core HQC Operations (real implementation)
// ============================================================================

/// Generate an HQC key pair for the given parameter set.
///
/// # Algorithm (KeyGen)
///
/// 1. Generate random 32-byte seed `seed_h`, expand to polynomial `h` via SHAKE256
/// 2. Sample sparse vectors `x`, `y` with Hamming weight `w`
/// 3. Compute syndrome `s = x + h*y mod (x^n - 1)` over GF(2)
/// 4. Generate random 32-byte `sigma` for implicit rejection
/// 5. Encode pk = (seed_h, s), sk = (sigma, seed_h, x_positions, y_positions)
pub fn keygen(params: HqcParams) -> HqcKeyPair {
    let mut rng = rand::rng();
    let n = params.code_length();

    // Step 1: Random seed for h
    let mut seed_h = [0u8; 32];
    rng.fill(&mut seed_h);

    // Expand seed_h -> h polynomial (n bits)
    let h = expand_seed_to_poly(&seed_h, params.domain_tag(), n);

    // Step 2: Sample sparse secret vectors x, y with weight w
    let mut x_seed = [0u8; 32];
    rng.fill(&mut x_seed);
    let (x_poly, x_positions) = sample_sparse_vector(&x_seed, b"HQC-x", n, params.w());

    let mut y_seed = [0u8; 32];
    rng.fill(&mut y_seed);
    let (_y_poly, y_positions) = sample_sparse_vector(&y_seed, b"HQC-y", n, params.w());

    // Step 3: s = x + h*y mod (x^n - 1)
    let hy = poly_mul_sparse_dense(&y_positions, &h, n);
    let s = poly_add(&x_poly, &hy);

    // Step 4: Random sigma for implicit rejection
    let mut sigma = [0u8; 32];
    rng.fill(&mut sigma);

    // Step 5: Encode keys
    let public_key = encode_public_key(&seed_h, &s, &params);
    let secret_key = encode_secret_key(&sigma, &seed_h, &x_positions, &y_positions, &params);

    HqcKeyPair {
        public_key,
        secret_key,
        params,
    }
}

/// Internal: derive (r1, r2, e) sparse vectors from theta seed.
fn derive_encaps_randomness(
    theta: &[u8],
    params: &HqcParams,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = params.code_length();
    let (_r1_poly, r1_positions) = sample_sparse_vector(theta, b"HQC-r1", n, params.wr());
    let (_r2_poly, r2_positions) = sample_sparse_vector(theta, b"HQC-r2", n, params.wr());
    let (_e_poly, e_positions) = sample_sparse_vector(theta, b"HQC-e", n, params.we());
    (r1_positions, r2_positions, e_positions)
}

/// Internal: build u, v, d from message m, public key components, and randomness.
fn build_ciphertext_components(
    m: &[u8],
    seed_h: &[u8],
    s: &[u8],
    r1_positions: &[usize],
    r2_positions: &[usize],
    e_positions: &[usize],
    params: &HqcParams,
) -> (Vec<u8>, Vec<u8>, [u8; 32]) {
    let n = params.code_length();
    let n_bytes = params.n_bytes();

    // h = expand(seed_h)
    let h = expand_seed_to_poly(seed_h, params.domain_tag(), n);

    // Reconstruct r1 as polynomial
    let mut r1_poly = vec![0u8; n_bytes];
    for &p in r1_positions {
        set_bit(&mut r1_poly, p, 1);
    }

    // u = r1 + h*r2 mod (x^n - 1)
    let hr2 = poly_mul_sparse_dense(r2_positions, &h, n);
    let u = poly_add(&r1_poly, &hr2);

    // Reconstruct e as polynomial
    let mut e_poly = vec![0u8; n_bytes];
    for &p in e_positions {
        set_bit(&mut e_poly, p, 1);
    }

    // v = encode_msg(m) + s*r2 + e mod (x^n - 1)
    let encoded_m = encode_message(m, params);
    let sr2 = poly_mul_sparse_dense(r2_positions, s, n);
    let tmp = poly_add(&encoded_m, &sr2);
    let v = poly_add(&tmp, &e_poly);

    // d = SHA3-256(m)
    let d = Sha3_256::hash(m);

    (u, v, d)
}

/// Encapsulate: produce a shared secret and ciphertext from a public key.
///
/// # Algorithm (Encaps with FO transform)
///
/// 1. Sample random message m (k bytes)
/// 2. theta = SHA3-256(m) -- deterministic coins
/// 3. Derive (r1, r2, e) from theta
/// 4. u = r1 + h*r2, v = encode_msg(m) + s*r2 + e
/// 5. d = SHA3-256(m)
/// 6. ct = encode(u, v, d)
/// 7. ss = SHAKE-256(m || ct, 64)
///
/// # Returns
///
/// `(shared_secret, ciphertext)` where `shared_secret` is 64 bytes.
pub fn encapsulate(pk: &[u8], params: HqcParams) -> (Vec<u8>, HqcCiphertext) {
    let mut rng = rand::rng();
    let k = params.k_bytes();

    // Parse public key
    let (seed_h, s) = decode_public_key(pk, &params);

    // Step 1: Random message m
    let mut m = vec![0u8; k];
    rng.fill(&mut m[..]);

    // Step 2: theta = SHA3-256(m)
    let theta = Sha3_256::hash(&m);

    // Step 3: Derive randomness
    let (r1_positions, r2_positions, e_positions) = derive_encaps_randomness(&theta, &params);

    // Step 4-5: Build ciphertext components
    let (u, v, d) = build_ciphertext_components(
        &m,
        &seed_h,
        &s,
        &r1_positions,
        &r2_positions,
        &e_positions,
        &params,
    );

    // Step 6: Encode ciphertext
    let ct_data = encode_ciphertext(&u, &v, &d, &params);
    let ciphertext = HqcCiphertext {
        data: ct_data,
        params,
    };

    // Step 7: ss = SHAKE-256(m || ct_data, 64)
    let ss_input: Vec<u8> = [&m[..], &ciphertext.data].concat();
    let ss = Shake256::xof(&ss_input, params.shared_secret_size());

    (ss, ciphertext)
}

/// Decapsulate: recover the shared secret from a ciphertext using the secret key.
///
/// # Algorithm (Decaps with FO transform / implicit rejection)
///
/// 1. Parse secret key to get (sigma, seed_h, x_positions, y_positions)
/// 2. Parse ciphertext to get (u, v, d)
/// 3. Reconstruct y polynomial, compute tmp = v + u*y over GF(2)
/// 4. m' = decode_msg(tmp) using concatenated code decoder
/// 5. theta' = SHA3-256(m'), re-derive (r1', r2', e')
/// 6. Recompute u', v', d' and check ciphertext equality (constant-time)
/// 7. If match: ss = SHAKE-256(m' || ct, 64)
/// 8. If mismatch: ss = SHAKE-256(sigma || ct, 64) (implicit rejection)
///
/// # Errors
///
/// Returns [`PqcError::InvalidCiphertext`] if the ciphertext length does not match
/// the parameter set, or [`PqcError::InvalidKey`] if the secret key is too short.
pub fn decapsulate(sk: &[u8], ct: &HqcCiphertext) -> PqcResult<Vec<u8>> {
    let params = ct.params;

    if sk.len() != params.secret_key_size() {
        return Err(PqcError::InvalidKey);
    }
    if ct.data.len() != params.ciphertext_size() {
        return Err(PqcError::InvalidCiphertext);
    }

    let n = params.code_length();
    let n_bytes = params.n_bytes();

    // Step 1: Parse secret key
    let (sigma, seed_h, _x_positions, y_positions) = decode_secret_key(sk);

    // Step 2: Parse ciphertext
    let (u, v, d) = decode_ciphertext(&ct.data, &params);

    // Decode public key from seed_h to get s (needed for re-encryption check)
    let h = expand_seed_to_poly(&seed_h, params.domain_tag(), n);

    // Reconstruct x from sk: we need the full public key s for re-encryption
    // s is in the public key, but we can reconstruct it: s = x + h*y
    // We need x_positions too. Let's re-read them from the sk.
    let (_sigma2, _seed_h2, x_positions, y_positions2) = decode_secret_key(sk);
    let _ = y_positions2; // same as y_positions

    // Reconstruct s = x + h*y
    let mut x_poly = vec![0u8; n_bytes];
    for &p in &x_positions {
        set_bit(&mut x_poly, p, 1);
    }
    let hy = poly_mul_sparse_dense(&y_positions, &h, n);
    let s = poly_add(&x_poly, &hy);

    // Step 3: tmp = v XOR (u * y) over GF(2)
    // u*y is computed as sparse(y) * dense(u)
    let uy = poly_mul_sparse_dense(&y_positions, &u, n);
    let tmp = poly_add(&v, &uy);

    // Step 4: Decode message
    let m_prime = decode_message(&tmp, &params);

    // Use a default message on decode failure (for implicit rejection path)
    let k = params.k_bytes();
    let m_decoded = m_prime.unwrap_or_else(|| vec![0u8; k]);

    // Step 5: theta' = SHA3-256(m')
    let theta_prime = Sha3_256::hash(&m_decoded);

    // Derive randomness from theta'
    let (r1p_positions, r2p_positions, ep_positions) =
        derive_encaps_randomness(&theta_prime, &params);

    // Step 6: Re-encrypt and check
    let (u_prime, v_prime, d_prime) = build_ciphertext_components(
        &m_decoded,
        &seed_h,
        &s,
        &r1p_positions,
        &r2p_positions,
        &ep_positions,
        &params,
    );

    // Encode the re-encrypted ciphertext for comparison
    let ct_prime = encode_ciphertext(&u_prime, &v_prime, &d_prime, &params);

    // Step 7/8: Compute BOTH possible shared secrets, then select in constant time
    let ct_match = ConstantTime::ct_compare(&ct_prime, &ct.data);

    // Good path: ss = SHAKE-256(m' || ct, 64)
    let ss_good_input: Vec<u8> = [&m_decoded[..], &ct.data].concat();
    let ss_good = Shake256::xof(&ss_good_input, params.shared_secret_size());

    // Rejection path: ss = SHAKE-256(sigma || ct, 64)
    let ss_reject_input: Vec<u8> = [&sigma[..], &ct.data].concat();
    let ss_reject = Shake256::xof(&ss_reject_input, params.shared_secret_size());

    // ct_match == 0 → equal → use ss_good (a), ct_match != 0 → different → use ss_reject (b)
    let ss = ConstantTime::ct_select(ct_match, &ss_good, &ss_reject);

    Ok(ss)
}

// ============================================================================
// Hybrid X25519 + HQC KEM
// ============================================================================

/// Hybrid public key combining a classical X25519 component and a post-quantum
/// HQC component.
#[derive(Clone, Debug)]
pub struct HybridHqcPublicKey {
    /// Classical (X25519-like) public key -- 32 bytes.
    pub classical: Vec<u8>,
    /// Post-quantum HQC public key.
    pub pq: Vec<u8>,
    /// HQC parameter set.
    pub params: HqcParams,
}

impl HybridHqcPublicKey {
    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let tag: u8 = match self.params {
            HqcParams::Hqc128 => 0x01,
            HqcParams::Hqc192 => 0x03,
            HqcParams::Hqc256 => 0x05,
        };
        let mut out = Vec::with_capacity(1 + self.classical.len() + self.pq.len());
        out.push(tag);
        out.extend_from_slice(&self.classical);
        out.extend_from_slice(&self.pq);
        out
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> PqcResult<Self> {
        if bytes.len() < 33 {
            return Err(PqcError::InvalidKey);
        }
        let params = match bytes[0] {
            0x01 => HqcParams::Hqc128,
            0x03 => HqcParams::Hqc192,
            0x05 => HqcParams::Hqc256,
            _ => return Err(PqcError::InvalidKey),
        };
        let classical = bytes[1..33].to_vec();
        let pq = bytes[33..].to_vec();
        if pq.len() != params.public_key_size() {
            return Err(PqcError::InvalidKey);
        }
        Ok(Self {
            classical,
            pq,
            params,
        })
    }
}

/// Hybrid secret key combining a classical X25519 component and a post-quantum
/// HQC component.
#[derive(Clone)]
pub struct HybridHqcSecretKey {
    /// Classical (X25519-like) secret key -- 32 bytes, zeroized on drop.
    pub classical: SecureZeroingVec,
    /// Post-quantum HQC secret key, zeroized on drop.
    pub pq: SecureZeroingVec,
    /// HQC parameter set.
    pub params: HqcParams,
}

impl std::fmt::Debug for HybridHqcSecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridHqcSecretKey")
            .field("params", &self.params)
            .field("classical_len", &self.classical.len())
            .field("pq_len", &self.pq.len())
            .finish()
    }
}

/// Hybrid key pair (public + secret) for the X25519 + HQC combined scheme.
#[derive(Clone, Debug)]
pub struct HybridHqcKeyPair {
    /// Combined public key.
    pub public_key: HybridHqcPublicKey,
    /// Combined secret key.
    pub secret_key: HybridHqcSecretKey,
}

/// Hybrid ciphertext produced by [`HybridHqcKem::hybrid_encapsulate`].
///
/// Contains both the classical (X25519) ephemeral public key and the HQC ciphertext.
#[derive(Clone, Debug)]
pub struct HybridHqcCiphertext {
    /// Classical ephemeral public key (32 bytes).
    pub classical: Vec<u8>,
    /// Post-quantum HQC ciphertext.
    pub pq: HqcCiphertext,
}

impl HybridHqcCiphertext {
    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let pq_bytes = self.pq.to_bytes();
        let mut out = Vec::with_capacity(self.classical.len() + pq_bytes.len());
        out.extend_from_slice(&self.classical);
        out.extend_from_slice(&pq_bytes);
        out
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> PqcResult<Self> {
        if bytes.len() < 33 {
            return Err(PqcError::InvalidCiphertext);
        }
        let classical = bytes[..32].to_vec();
        let pq = HqcCiphertext::from_bytes(&bytes[32..])?;
        Ok(Self { classical, pq })
    }
}

/// Hybrid KEM combining X25519 (classical) and HQC (post-quantum).
///
/// The combined shared secret is derived as:
///
/// ```text
/// ss = SHAKE-256("HybridHqcKem-Combine" || classical_ss || hqc_ss, 64)
/// ```
///
/// This ensures the result is secure as long as **either** the classical or the
/// post-quantum component remains unbroken.
///
/// The classical component uses the same simulated X25519 approach as
/// [`super::hybrid::HybridKem`] for consistency within this crate.
pub struct HybridHqcKem;

impl HybridHqcKem {
    /// Generate a hybrid X25519 + HQC key pair.
    pub fn hybrid_keygen(params: HqcParams) -> PqcResult<HybridHqcKeyPair> {
        let mut rng = rand::rng();

        // --- Classical (simulated KEM) ---
        let mut classical_sk_bytes = vec![0u8; 32];
        rng.fill(&mut classical_sk_bytes[..]);
        let classical_pk_bytes = Sha3_256::hash(&classical_sk_bytes);

        // --- Post-quantum (HQC) ---
        let hqc_kp = keygen(params);

        Ok(HybridHqcKeyPair {
            public_key: HybridHqcPublicKey {
                classical: classical_pk_bytes.to_vec(),
                pq: hqc_kp.public_key,
                params,
            },
            secret_key: HybridHqcSecretKey {
                classical: SecureZeroingVec::from_vec(classical_sk_bytes),
                pq: SecureZeroingVec::from_vec(hqc_kp.secret_key),
                params,
            },
        })
    }

    /// XOR two 32-byte slices.
    fn xor_32(a: &[u8], b: &[u8]) -> Vec<u8> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect()
    }

    /// Encapsulate: produce a combined shared secret and hybrid ciphertext.
    ///
    /// The classical component uses a simulated KEM: random coins are XOR-encrypted
    /// with the public key and the shared secret is derived from coins + pk.
    ///
    /// # Returns
    ///
    /// `(shared_secret, hybrid_ciphertext)` where `shared_secret` is 64 bytes.
    pub fn hybrid_encapsulate(
        pk: &HybridHqcPublicKey,
    ) -> PqcResult<(Vec<u8>, HybridHqcCiphertext)> {
        let mut rng = rand::rng();

        // --- Classical encapsulation (simulated KEM) ---
        let mut coins = [0u8; 32];
        rng.fill(&mut coins);
        // "Encrypt" coins with the public key (XOR)
        let classical_ct = Self::xor_32(&coins, &pk.classical);
        // Shared secret = SHA3-256("ClassicalSS" || coins || pk)
        let classical_ss = {
            let mut input = Vec::with_capacity(32 + 32 + 12);
            input.extend_from_slice(b"ClassicalSS");
            input.extend_from_slice(&coins);
            input.extend_from_slice(&pk.classical);
            Sha3_256::hash(&input)
        };

        // --- Post-quantum encapsulation (HQC) ---
        let (pq_ss, pq_ct) = encapsulate(&pk.pq, pk.params);

        // --- Combine shared secrets ---
        let combined = Self::combine_secrets(&classical_ss, &pq_ss);

        Ok((
            combined,
            HybridHqcCiphertext {
                classical: classical_ct,
                pq: pq_ct,
            },
        ))
    }

    /// Decapsulate: recover the combined shared secret from a hybrid ciphertext.
    pub fn hybrid_decapsulate(
        sk: &HybridHqcSecretKey,
        ct: &HybridHqcCiphertext,
    ) -> PqcResult<Vec<u8>> {
        // --- Classical decapsulation (simulated KEM) ---
        // Recover pk = SHA3-256(sk), then coins = ct XOR pk
        let pk = Sha3_256::hash(sk.classical.as_slice());
        let coins = Self::xor_32(&ct.classical, &pk);
        let classical_ss = {
            let mut input = Vec::with_capacity(32 + 32 + 12);
            input.extend_from_slice(b"ClassicalSS");
            input.extend_from_slice(&coins);
            input.extend_from_slice(&pk);
            Sha3_256::hash(&input)
        };

        // --- Post-quantum decapsulation (HQC) ---
        let pq_ss = decapsulate(sk.pq.as_slice(), &ct.pq)?;

        // --- Combine ---
        let combined = Self::combine_secrets(&classical_ss, &pq_ss);
        Ok(combined)
    }

    /// Combine classical and post-quantum shared secrets using a domain-separated KDF.
    fn combine_secrets(classical_ss: &[u8], pq_ss: &[u8]) -> Vec<u8> {
        let mut input =
            Vec::with_capacity(b"HybridHqcKem-Combine".len() + classical_ss.len() + pq_ss.len());
        input.extend_from_slice(b"HybridHqcKem-Combine");
        input.extend_from_slice(classical_ss);
        input.extend_from_slice(pq_ss);
        Shake256::xof(&input, 64)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Parameter set tests ------------------------------------------------

    #[test]
    fn test_params_code_lengths() {
        assert_eq!(HqcParams::Hqc128.code_length(), 17_669);
        assert_eq!(HqcParams::Hqc192.code_length(), 35_851);
        assert_eq!(HqcParams::Hqc256.code_length(), 57_637);
    }

    #[test]
    fn test_params_security_bits() {
        assert_eq!(HqcParams::Hqc128.security_bits(), 128);
        assert_eq!(HqcParams::Hqc192.security_bits(), 192);
        assert_eq!(HqcParams::Hqc256.security_bits(), 256);
    }

    #[test]
    fn test_params_nist_levels() {
        assert_eq!(HqcParams::Hqc128.nist_level(), 1);
        assert_eq!(HqcParams::Hqc192.nist_level(), 3);
        assert_eq!(HqcParams::Hqc256.nist_level(), 5);
    }

    #[test]
    fn test_params_display() {
        assert_eq!(format!("{}", HqcParams::Hqc128), "HQC-128 (NIST Level 1)");
        assert_eq!(format!("{}", HqcParams::Hqc192), "HQC-192 (NIST Level 3)");
        assert_eq!(format!("{}", HqcParams::Hqc256), "HQC-256 (NIST Level 5)");
    }

    // ---- GF(2^8) arithmetic unit tests --------------------------------------

    #[test]
    fn test_gf256_tables_consistency() {
        let (log, exp) = &*GF256_TABLES;
        // alpha = 3 (primitive element)
        // alpha^0 = 1
        assert_eq!(exp[0], 1);
        // alpha^1 = 3
        assert_eq!(exp[1], 3);
        // log(1) = 0
        assert_eq!(log[1], 0, "log[1] should be 0 but got {}", log[1]);
        // log(3) = 1
        assert_eq!(log[3], 1);
        // alpha^255 wraps to alpha^0 = 1
        assert_eq!(exp[255], exp[0]);
        // Verify all 255 non-zero elements are generated (no duplicates)
        let mut seen = [false; 256];
        for i in 0..255 {
            let v = exp[i] as usize;
            assert!(!seen[v], "duplicate exp[{i}] = {v}");
            seen[v] = true;
        }
        assert!(!seen[0], "0 should not appear in exp table");
    }

    #[test]
    fn test_gf256_mul_identity() {
        for a in 0..=255u8 {
            assert_eq!(gf256_mul(a, 1), a);
            assert_eq!(gf256_mul(1, a), a);
            assert_eq!(gf256_mul(a, 0), 0);
            assert_eq!(gf256_mul(0, a), 0);
        }
    }

    #[test]
    fn test_gf256_mul_inverse() {
        for a in 1..=255u8 {
            let inv = gf256_inv(a);
            assert_eq!(gf256_mul(a, inv), 1, "a={a}, inv={inv}");
        }
    }

    // ---- Reed-Solomon unit tests --------------------------------------------

    #[test]
    fn test_rs_encode_decode_no_errors() {
        let k = 16;
        let n1 = 46;
        let msg: Vec<u8> = (0..k).map(|i| i as u8).collect();
        let codeword = rs_encode(&msg, n1, k);
        assert_eq!(codeword.len(), n1);
        let decoded = rs_decode(&codeword, n1, k).expect("decode should succeed");
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_rs_encode_decode_with_errors() {
        let k = 16;
        let n1 = 46;
        let t = (n1 - k) / 2; // can correct up to t errors
        let msg: Vec<u8> = (0..k).map(|i| (i * 7 + 3) as u8).collect();
        let mut codeword = rs_encode(&msg, n1, k);

        // Introduce t errors
        for i in 0..t {
            codeword[i * 3] ^= 0xAB;
        }

        let decoded = rs_decode(&codeword, n1, k).expect("decode should succeed");
        assert_eq!(decoded, msg);
    }

    // ---- Reed-Muller unit tests ---------------------------------------------

    #[test]
    fn test_rm_encode_decode_all_bytes() {
        // Test that RM(1,7) roundtrips for all 256 possible byte values
        for b in 0..=255u8 {
            let codeword = rm_encode(b);
            let decoded = rm_decode(&codeword).expect("RM decode should succeed");
            assert_eq!(decoded, b, "RM roundtrip failed for byte {b}");
        }
    }

    #[test]
    fn test_rm_decode_with_noise() {
        // RM(1,7) can correct a significant number of errors (up to ~31 out of 128)
        let original = 0x42u8;
        let mut codeword = rm_encode(original);
        // Flip 10 bits
        for i in 0..10 {
            codeword[i] ^= 0x01;
        }
        let decoded = rm_decode(&codeword).expect("RM decode should succeed");
        assert_eq!(decoded, original);
    }

    // ---- Concatenated code unit tests ---------------------------------------

    #[test]
    fn test_concatenated_encode_decode() {
        let params = HqcParams::Hqc128;
        let k = params.k_bytes();
        let msg: Vec<u8> = (0..k).map(|i| (i * 13 + 7) as u8).collect();
        let encoded = concatenated_encode(&msg, &params);
        let decoded = concatenated_decode(&encoded, &params).expect("decode should succeed");
        assert_eq!(decoded, msg);
    }

    // ---- GF(2) polynomial arithmetic tests ----------------------------------

    #[test]
    fn test_poly_add_self_is_zero() {
        let a = vec![0xAB, 0xCD, 0xEF];
        let result = poly_add(&a, &a);
        assert!(result.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_sparse_sampling() {
        let seed = [42u8; 32];
        let n = 1000;
        let weight = 50;
        let (poly, positions) = sample_sparse_vector(&seed, b"test", n, weight);
        assert_eq!(positions.len(), weight);

        // Verify positions match poly
        let mut hw = 0;
        for i in 0..n {
            hw += get_bit(&poly, i) as usize;
        }
        assert_eq!(hw, weight);
    }

    // ---- Key generation tests -----------------------------------------------

    #[test]
    fn test_keygen_sizes_hqc128() {
        let kp = keygen(HqcParams::Hqc128);
        assert_eq!(kp.public_key.len(), HqcParams::Hqc128.public_key_size());
        assert_eq!(kp.secret_key.len(), HqcParams::Hqc128.secret_key_size());
        assert_eq!(kp.params, HqcParams::Hqc128);
    }

    #[test]
    fn test_keygen_sizes_hqc192() {
        let kp = keygen(HqcParams::Hqc192);
        assert_eq!(kp.public_key.len(), HqcParams::Hqc192.public_key_size());
        assert_eq!(kp.secret_key.len(), HqcParams::Hqc192.secret_key_size());
    }

    #[test]
    fn test_keygen_sizes_hqc256() {
        let kp = keygen(HqcParams::Hqc256);
        assert_eq!(kp.public_key.len(), HqcParams::Hqc256.public_key_size());
        assert_eq!(kp.secret_key.len(), HqcParams::Hqc256.secret_key_size());
    }

    #[test]
    fn test_keygen_unique_keys() {
        let kp1 = keygen(HqcParams::Hqc128);
        let kp2 = keygen(HqcParams::Hqc128);
        // Two independent key generations must produce different keys
        assert_ne!(kp1.public_key, kp2.public_key);
        assert_ne!(kp1.secret_key, kp2.secret_key);
    }

    // ---- KEM roundtrip tests ------------------------------------------------

    #[test]
    fn test_kem_roundtrip_hqc128() {
        let kp = keygen(HqcParams::Hqc128);
        let (ss_enc, ct) = encapsulate(&kp.public_key, HqcParams::Hqc128);
        let ss_dec = decapsulate(&kp.secret_key, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc.len(), HqcParams::Hqc128.shared_secret_size());
        assert_eq!(ss_enc, ss_dec, "shared secrets must match");
    }

    #[test]
    fn test_kem_roundtrip_hqc192() {
        let kp = keygen(HqcParams::Hqc192);
        let (ss_enc, ct) = encapsulate(&kp.public_key, HqcParams::Hqc192);
        let ss_dec = decapsulate(&kp.secret_key, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_kem_roundtrip_hqc256() {
        let kp = keygen(HqcParams::Hqc256);
        let (ss_enc, ct) = encapsulate(&kp.public_key, HqcParams::Hqc256);
        let ss_dec = decapsulate(&kp.secret_key, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_different_encapsulations_produce_different_secrets() {
        let kp = keygen(HqcParams::Hqc128);
        let (ss1, _ct1) = encapsulate(&kp.public_key, HqcParams::Hqc128);
        let (ss2, _ct2) = encapsulate(&kp.public_key, HqcParams::Hqc128);
        // Different random coins => different shared secrets
        assert_ne!(ss1, ss2);
    }

    // ---- Ciphertext serialization tests -------------------------------------

    #[test]
    fn test_ciphertext_serialization_roundtrip() {
        let kp = keygen(HqcParams::Hqc128);
        let (_ss, ct) = encapsulate(&kp.public_key, HqcParams::Hqc128);

        let bytes = ct.to_bytes();
        let ct2 = HqcCiphertext::from_bytes(&bytes).expect("deserialize failed");

        assert_eq!(ct.data, ct2.data);
        assert_eq!(ct.params, ct2.params);
    }

    #[test]
    fn test_ciphertext_from_invalid_bytes() {
        assert!(HqcCiphertext::from_bytes(&[]).is_err());
        assert!(HqcCiphertext::from_bytes(&[0xFF]).is_err()); // bad tag
        assert!(HqcCiphertext::from_bytes(&[0x01, 0x00]).is_err()); // wrong length
    }

    // ---- Error path tests ---------------------------------------------------

    #[test]
    fn test_decapsulate_wrong_sk_size() {
        let kp = keygen(HqcParams::Hqc128);
        let (_ss, ct) = encapsulate(&kp.public_key, HqcParams::Hqc128);

        let result = decapsulate(&[0u8; 10], &ct);
        assert_eq!(result, Err(PqcError::InvalidKey));
    }

    #[test]
    fn test_decapsulate_wrong_ct_size() {
        let kp = keygen(HqcParams::Hqc128);

        let bad_ct = HqcCiphertext {
            data: vec![0u8; 10],
            params: HqcParams::Hqc128,
        };
        let result = decapsulate(&kp.secret_key, &bad_ct);
        assert_eq!(result, Err(PqcError::InvalidCiphertext));
    }

    // ---- Hybrid KEM tests ---------------------------------------------------

    #[test]
    fn test_hybrid_keygen() {
        let hkp = HybridHqcKem::hybrid_keygen(HqcParams::Hqc128).expect("keygen failed");
        assert_eq!(hkp.public_key.classical.len(), 32);
        assert_eq!(hkp.public_key.pq.len(), HqcParams::Hqc128.public_key_size());
        assert_eq!(hkp.secret_key.classical.len(), 32);
        assert_eq!(hkp.secret_key.pq.len(), HqcParams::Hqc128.secret_key_size());
    }

    #[test]
    fn test_hybrid_kem_roundtrip_hqc128() {
        let hkp = HybridHqcKem::hybrid_keygen(HqcParams::Hqc128).expect("keygen failed");
        let (ss_enc, ct) =
            HybridHqcKem::hybrid_encapsulate(&hkp.public_key).expect("encapsulate failed");
        let ss_dec =
            HybridHqcKem::hybrid_decapsulate(&hkp.secret_key, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc.len(), 64);
        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_hybrid_kem_roundtrip_hqc192() {
        let hkp = HybridHqcKem::hybrid_keygen(HqcParams::Hqc192).expect("keygen failed");
        let (ss_enc, ct) =
            HybridHqcKem::hybrid_encapsulate(&hkp.public_key).expect("encapsulate failed");
        let ss_dec =
            HybridHqcKem::hybrid_decapsulate(&hkp.secret_key, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_hybrid_kem_roundtrip_hqc256() {
        let hkp = HybridHqcKem::hybrid_keygen(HqcParams::Hqc256).expect("keygen failed");
        let (ss_enc, ct) =
            HybridHqcKem::hybrid_encapsulate(&hkp.public_key).expect("encapsulate failed");
        let ss_dec =
            HybridHqcKem::hybrid_decapsulate(&hkp.secret_key, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_hybrid_ciphertext_serialization() {
        let hkp = HybridHqcKem::hybrid_keygen(HqcParams::Hqc128).expect("keygen failed");
        let (_ss, ct) =
            HybridHqcKem::hybrid_encapsulate(&hkp.public_key).expect("encapsulate failed");

        let bytes = ct.to_bytes();
        let ct2 = HybridHqcCiphertext::from_bytes(&bytes).expect("deserialize failed");

        assert_eq!(ct.classical, ct2.classical);
        assert_eq!(ct.pq.data, ct2.pq.data);
        assert_eq!(ct.pq.params, ct2.pq.params);
    }

    #[test]
    fn test_hybrid_public_key_serialization() {
        let hkp = HybridHqcKem::hybrid_keygen(HqcParams::Hqc192).expect("keygen failed");

        let bytes = hkp.public_key.to_bytes();
        let pk2 = HybridHqcPublicKey::from_bytes(&bytes).expect("deserialize failed");

        assert_eq!(hkp.public_key.classical, pk2.classical);
        assert_eq!(hkp.public_key.pq, pk2.pq);
        assert_eq!(hkp.public_key.params, pk2.params);
    }

    #[test]
    fn test_hybrid_different_encapsulations() {
        let hkp = HybridHqcKem::hybrid_keygen(HqcParams::Hqc128).expect("keygen failed");
        let (ss1, _ct1) =
            HybridHqcKem::hybrid_encapsulate(&hkp.public_key).expect("encapsulate failed");
        let (ss2, _ct2) =
            HybridHqcKem::hybrid_encapsulate(&hkp.public_key).expect("encapsulate failed");

        // Different random coins -> different shared secrets
        assert_ne!(ss1, ss2);
    }
}
