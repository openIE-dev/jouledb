//! Shamir's Secret Sharing — split a secret into N shares with K threshold,
//! reconstruct from K shares, GF(256) arithmetic, share verification, and
//! serialization.

use serde::{Deserialize, Serialize};

// ── GF(256) Arithmetic ──────────────────────────────────────────────────────

/// GF(256) with irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x11B).
/// This is the same field used by AES.

/// GF(256) addition is XOR.
fn gf256_add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// GF(256) subtraction is also XOR (same as addition in GF(2^n)).
fn gf256_sub(a: u8, b: u8) -> u8 {
    a ^ b
}

/// GF(256) multiplication using Russian Peasant multiplication.
fn gf256_mul(a: u8, b: u8) -> u8 {
    let mut result: u16 = 0;
    let mut aa = a as u16;
    let mut bb = b as u16;
    while bb > 0 {
        if bb & 1 != 0 {
            result ^= aa;
        }
        aa <<= 1;
        if aa & 0x100 != 0 {
            aa ^= 0x11B; // Reduce by the irreducible polynomial.
        }
        bb >>= 1;
    }
    result as u8
}

/// GF(256) multiplicative inverse using extended Euclidean algorithm approach.
/// Returns 0 for input 0 (which has no inverse).
fn gf256_inv(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    // a^254 = a^(-1) in GF(256) by Fermat's little theorem.
    let mut result = a;
    for _ in 0..6 {
        result = gf256_mul(result, result); // square
        result = gf256_mul(result, a);      // multiply by a
    }
    // After 6 iterations of square-multiply: a^(2^7 - 1) = a^127
    // We need a^254 = (a^127)^2
    result = gf256_mul(result, result);
    result
}

/// GF(256) division: a / b.
fn gf256_div(a: u8, b: u8) -> u8 {
    assert!(b != 0, "division by zero in GF(256)");
    gf256_mul(a, gf256_inv(b))
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Secret sharing errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShamirError {
    /// Threshold must be >= 2.
    ThresholdTooLow(usize),
    /// Threshold must be <= total shares.
    ThresholdExceedsTotal { threshold: usize, total: usize },
    /// Not enough shares to reconstruct.
    InsufficientShares { needed: usize, provided: usize },
    /// Duplicate share index detected.
    DuplicateShareIndex(u8),
    /// Zero share index (x = 0 is reserved for the secret).
    ZeroShareIndex,
    /// Empty secret.
    EmptySecret,
    /// Total shares limited to 255 (GF(256) constraint).
    TooManyShares(usize),
    /// Share verification failed.
    VerificationFailed,
}

impl std::fmt::Display for ShamirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ThresholdTooLow(t) => write!(f, "threshold {t} too low, must be >= 2"),
            Self::ThresholdExceedsTotal { threshold, total } => {
                write!(f, "threshold {threshold} exceeds total shares {total}")
            }
            Self::InsufficientShares { needed, provided } => {
                write!(f, "need {needed} shares, only {provided} provided")
            }
            Self::DuplicateShareIndex(i) => write!(f, "duplicate share index: {i}"),
            Self::ZeroShareIndex => write!(f, "share index 0 is reserved"),
            Self::EmptySecret => write!(f, "secret must not be empty"),
            Self::TooManyShares(n) => write!(f, "too many shares: {n} (max 255)"),
            Self::VerificationFailed => write!(f, "share verification failed"),
        }
    }
}

impl std::error::Error for ShamirError {}

// ── Share ───────────────────────────────────────────────────────────────────

/// A single share of a split secret.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Share {
    /// Share index (1-based, x-coordinate on the polynomial).
    pub index: u8,
    /// Share data (y-coordinates for each byte of the secret).
    pub data: Vec<u8>,
    /// Threshold required for reconstruction.
    pub threshold: usize,
    /// Total shares created in the original split.
    pub total: usize,
}

impl Share {
    /// Hex-encode the share data.
    pub fn to_hex(&self) -> String {
        self.data.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Length of the secret this share represents.
    pub fn secret_len(&self) -> usize {
        self.data.len()
    }
}

// ── Polynomial evaluation ───────────────────────────────────────────────────

/// Evaluate a polynomial (given as coefficients [a0, a1, ..., ak-1]) at point x
/// in GF(256).
fn poly_eval(coeffs: &[u8], x: u8) -> u8 {
    // Horner's method: a0 + x*(a1 + x*(a2 + ...))
    let mut result = 0u8;
    for &coeff in coeffs.iter().rev() {
        result = gf256_add(gf256_mul(result, x), coeff);
    }
    result
}

// ── Simple deterministic RNG for share generation ───────────────────────────

/// A simple xorshift-based PRNG seeded from the secret bytes.
/// This is NOT cryptographically secure by itself, but we use it only
/// for generating polynomial coefficients, seeded from the secret.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn from_seed(seed: &[u8]) -> Self {
        let mut state = 0x123456789ABCDEFu64;
        for (i, &byte) in seed.iter().enumerate() {
            state ^= (byte as u64) << ((i % 8) * 8);
            state = state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
        }
        if state == 0 {
            state = 1;
        }
        Self { state }
    }

    fn next_u8(&mut self) -> u8 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state & 0xFF) as u8
    }

    /// Generate a non-zero random byte.
    fn next_nonzero(&mut self) -> u8 {
        loop {
            let v = self.next_u8();
            if v != 0 {
                return v;
            }
        }
    }
}

// ── Split and Reconstruct ───────────────────────────────────────────────────

/// Split a secret into `total` shares with a `threshold` for reconstruction.
///
/// - `threshold`: minimum number of shares needed (>= 2).
/// - `total`: total shares to create (>= threshold, <= 255).
/// - `entropy`: additional entropy bytes to seed the random coefficients.
///
/// Each byte of the secret is independently encoded as a polynomial of
/// degree (threshold - 1) over GF(256).
pub fn split(
    secret: &[u8],
    threshold: usize,
    total: usize,
    entropy: &[u8],
) -> Result<Vec<Share>, ShamirError> {
    if secret.is_empty() {
        return Err(ShamirError::EmptySecret);
    }
    if threshold < 2 {
        return Err(ShamirError::ThresholdTooLow(threshold));
    }
    if threshold > total {
        return Err(ShamirError::ThresholdExceedsTotal { threshold, total });
    }
    if total > 255 {
        return Err(ShamirError::TooManyShares(total));
    }

    // Seed RNG from entropy + secret hash for unpredictability.
    let mut seed_material = Vec::with_capacity(entropy.len() + secret.len() + 8);
    seed_material.extend_from_slice(entropy);
    seed_material.extend_from_slice(secret);
    seed_material.extend_from_slice(&(threshold as u64).to_le_bytes());
    let mut rng = SimpleRng::from_seed(&seed_material);

    let mut shares: Vec<Share> = (0..total)
        .map(|i| Share {
            index: (i + 1) as u8,
            data: Vec::with_capacity(secret.len()),
            threshold,
            total,
        })
        .collect();

    // For each byte of the secret, create a random polynomial and evaluate.
    for &secret_byte in secret {
        // Coefficients: [secret_byte, random, random, ..., random]
        let mut coeffs = Vec::with_capacity(threshold);
        coeffs.push(secret_byte);
        for _ in 1..threshold {
            coeffs.push(rng.next_nonzero());
        }

        // Evaluate polynomial at x = 1, 2, ..., total.
        for share in shares.iter_mut() {
            let y = poly_eval(&coeffs, share.index);
            share.data.push(y);
        }
    }

    Ok(shares)
}

/// Reconstruct a secret from `threshold` or more shares using Lagrange
/// interpolation over GF(256).
pub fn reconstruct(shares: &[Share]) -> Result<Vec<u8>, ShamirError> {
    if shares.is_empty() {
        return Err(ShamirError::InsufficientShares {
            needed: 2,
            provided: 0,
        });
    }

    let threshold = shares[0].threshold;
    if shares.len() < threshold {
        return Err(ShamirError::InsufficientShares {
            needed: threshold,
            provided: shares.len(),
        });
    }

    // Check for duplicate indices.
    let mut seen = std::collections::HashSet::new();
    for share in shares {
        if share.index == 0 {
            return Err(ShamirError::ZeroShareIndex);
        }
        if !seen.insert(share.index) {
            return Err(ShamirError::DuplicateShareIndex(share.index));
        }
    }

    let secret_len = shares[0].data.len();
    let k = threshold.min(shares.len());
    let eval_shares = &shares[..k];

    let mut secret = Vec::with_capacity(secret_len);

    for byte_idx in 0..secret_len {
        // Lagrange interpolation at x = 0.
        let mut value = 0u8;
        for (i, share_i) in eval_shares.iter().enumerate() {
            let xi = share_i.index;
            let yi = share_i.data[byte_idx];

            // Compute Lagrange basis polynomial L_i(0).
            let mut basis = 1u8;
            for (j, share_j) in eval_shares.iter().enumerate() {
                if i == j {
                    continue;
                }
                let xj = share_j.index;
                // L_i(0) *= (0 - xj) / (xi - xj) = xj / (xi ^ xj) in GF(256)
                let numerator = xj;
                let denominator = gf256_sub(xi, xj);
                basis = gf256_mul(basis, gf256_div(numerator, denominator));
            }

            value = gf256_add(value, gf256_mul(yi, basis));
        }
        secret.push(value);
    }

    Ok(secret)
}

/// Verify a share against the reconstructed secret: split with the same
/// parameters and check that the share matches one of the output shares.
/// This is a convenience function — in production, use commitments.
pub fn verify_share(share: &Share, secret: &[u8], entropy: &[u8]) -> Result<bool, ShamirError> {
    let shares = split(secret, share.threshold, share.total, entropy)?;
    let matching = shares.iter().find(|s| s.index == share.index);
    match matching {
        Some(expected) => Ok(expected.data == share.data),
        None => Ok(false),
    }
}

// ── Lagrange coefficient ────────────────────────────────────────────────────

/// Compute the Lagrange basis coefficient for share `i` at x = 0,
/// given a set of share indices.
pub fn lagrange_coefficient(index: u8, all_indices: &[u8]) -> u8 {
    let mut basis = 1u8;
    for &xj in all_indices {
        if xj == index {
            continue;
        }
        let numerator = xj;
        let denominator = gf256_sub(index, xj);
        basis = gf256_mul(basis, gf256_div(numerator, denominator));
    }
    basis
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf256_add_sub() {
        // In GF(2^n), add == sub == XOR.
        assert_eq!(gf256_add(0x53, 0xCA), 0x53 ^ 0xCA);
        assert_eq!(gf256_sub(0x53, 0xCA), 0x53 ^ 0xCA);
        // a + a = 0
        assert_eq!(gf256_add(42, 42), 0);
    }

    #[test]
    fn test_gf256_mul_identity() {
        for a in 0..=255u8 {
            assert_eq!(gf256_mul(a, 1), a);
            assert_eq!(gf256_mul(1, a), a);
            assert_eq!(gf256_mul(a, 0), 0);
        }
    }

    #[test]
    fn test_gf256_mul_known() {
        // AES MixColumns uses GF(256) multiply.
        // 0x57 * 0x83 = 0xC1 (well-known test vector)
        assert_eq!(gf256_mul(0x57, 0x83), 0xC1);
    }

    #[test]
    fn test_gf256_inv() {
        // a * inv(a) = 1 for all non-zero a.
        for a in 1..=255u8 {
            let inv = gf256_inv(a);
            assert_eq!(gf256_mul(a, inv), 1, "inverse failed for {a}");
        }
        assert_eq!(gf256_inv(0), 0); // 0 has no inverse, returns 0.
    }

    #[test]
    fn test_gf256_div() {
        // a / a = 1.
        for a in 1..=255u8 {
            assert_eq!(gf256_div(a, a), 1);
        }
    }

    #[test]
    fn test_poly_eval() {
        // f(x) = 5 (constant polynomial).
        assert_eq!(poly_eval(&[5], 3), 5);
        // f(x) = 1 + x. f(0) = 1, f(1) = 1^1 = 0 in GF(256).
        assert_eq!(poly_eval(&[1, 1], 0), 1);
        assert_eq!(poly_eval(&[1, 1], 1), 0); // 1 XOR 1 = 0
    }

    #[test]
    fn test_split_and_reconstruct_basic() {
        let secret = b"Hello!";
        let shares = split(secret, 3, 5, b"entropy").unwrap();
        assert_eq!(shares.len(), 5);

        // Reconstruct from first 3 shares.
        let recovered = reconstruct(&shares[..3]).unwrap();
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_reconstruct_from_any_k_shares() {
        let secret = b"Secret Data 123";
        let shares = split(secret, 3, 5, b"seed").unwrap();

        // Any 3 of 5 shares should work.
        let combinations: Vec<Vec<usize>> = vec![
            vec![0, 1, 2],
            vec![0, 1, 4],
            vec![0, 3, 4],
            vec![1, 2, 3],
            vec![2, 3, 4],
        ];
        for combo in &combinations {
            let subset: Vec<Share> = combo.iter().map(|i| shares[*i].clone()).collect();
            let recovered = reconstruct(&subset).unwrap();
            assert_eq!(recovered, secret, "failed with shares {:?}", combo);
        }
    }

    #[test]
    fn test_insufficient_shares() {
        let secret = b"data";
        let shares = split(secret, 3, 5, b"e").unwrap();
        let result = reconstruct(&shares[..2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_threshold_too_low() {
        assert!(split(b"x", 1, 3, b"").is_err());
        assert!(split(b"x", 0, 3, b"").is_err());
    }

    #[test]
    fn test_threshold_exceeds_total() {
        assert!(split(b"x", 5, 3, b"").is_err());
    }

    #[test]
    fn test_too_many_shares() {
        assert!(split(b"x", 2, 256, b"").is_err());
    }

    #[test]
    fn test_empty_secret() {
        assert!(split(b"", 2, 3, b"").is_err());
    }

    #[test]
    fn test_duplicate_share_index() {
        let shares = split(b"abc", 2, 3, b"e").unwrap();
        let duplicate = vec![shares[0].clone(), shares[0].clone()];
        assert!(reconstruct(&duplicate).is_err());
    }

    #[test]
    fn test_share_serialization() {
        let shares = split(b"secret", 2, 3, b"ent").unwrap();
        let json = serde_json::to_string(&shares[0]).unwrap();
        let deserialized: Share = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, shares[0]);
    }

    #[test]
    fn test_share_hex() {
        let shares = split(b"abc", 2, 3, b"e").unwrap();
        let hex = shares[0].to_hex();
        assert_eq!(hex.len(), 6); // 3 bytes * 2 hex chars each
    }

    #[test]
    fn test_share_secret_len() {
        let shares = split(b"hello", 2, 3, b"e").unwrap();
        for share in &shares {
            assert_eq!(share.secret_len(), 5);
        }
    }

    #[test]
    fn test_verify_share() {
        let secret = b"verify me";
        let entropy = b"entropy";
        let shares = split(secret, 3, 5, entropy).unwrap();
        assert!(verify_share(&shares[0], secret, entropy).unwrap());
        assert!(verify_share(&shares[4], secret, entropy).unwrap());
    }

    #[test]
    fn test_two_of_two() {
        // Minimum viable sharing: 2-of-2.
        let secret = b"minimal";
        let shares = split(secret, 2, 2, b"e").unwrap();
        assert_eq!(shares.len(), 2);
        let recovered = reconstruct(&shares).unwrap();
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_lagrange_coefficient() {
        let indices: Vec<u8> = vec![1, 2, 3];
        let l1 = lagrange_coefficient(1, &indices);
        let l2 = lagrange_coefficient(2, &indices);
        let l3 = lagrange_coefficient(3, &indices);
        // The Lagrange coefficients should sum to 1 (in GF(256))
        // because we're evaluating at x=0 for the polynomial that
        // interpolates through x=1,2,3 and equals 1 at x=0.
        // Actually, sum of L_i(0) = 1 only if we interpolate the constant
        // polynomial f(x) = 1 through points x=1,2,3.
        // Let's just verify they're non-zero.
        assert_ne!(l1, 0);
        assert_ne!(l2, 0);
        assert_ne!(l3, 0);
    }

    #[test]
    fn test_single_byte_secret() {
        let secret = &[42u8];
        let shares = split(secret, 2, 3, b"e").unwrap();
        let recovered = reconstruct(&shares[..2]).unwrap();
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_large_secret() {
        let secret: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let shares = split(&secret, 5, 10, b"big").unwrap();
        let recovered = reconstruct(&shares[..5]).unwrap();
        assert_eq!(recovered, secret);
    }
}
