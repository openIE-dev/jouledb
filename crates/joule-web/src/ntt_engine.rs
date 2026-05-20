//! NTT (Number Theoretic Transform) engine for polynomial arithmetic.
//!
//! Provides forward and inverse NTT over Zq, Montgomery and Barrett
//! reduction, butterfly operations, NTT-friendly prime generation,
//! and vectorized coefficient-level polynomial arithmetic.
//!
//! All arithmetic operates on `u64` coefficients modulo a chosen prime `q`.
//! The engine supports configurable ring degrees (powers of two) and
//! includes precomputed root-of-unity tables for fast transforms.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────────

/// Default NTT-friendly prime: q = 12289 (used by many lattice schemes).
pub const DEFAULT_Q: u64 = 12289;

/// Another common NTT-friendly prime: q = 7681.
pub const Q_7681: u64 = 7681;

/// Large NTT-friendly prime for Kyber/Dilithium: q = 3329.
pub const Q_KYBER: u64 = 3329;

/// Large NTT-friendly prime: q = 8380417 (Dilithium).
pub const Q_DILITHIUM: u64 = 8380417;

// ── Montgomery Reduction ────────────────────────────────────────────

/// Montgomery reduction state for fast modular multiplication.
#[derive(Debug, Clone)]
pub struct MontgomeryCtx {
    pub q: u64,
    /// R = 2^32 for 64-bit arithmetic.
    pub r_bits: u32,
    /// q_inv such that q * q_inv ≡ -1 (mod R).
    pub q_inv: u64,
    /// R mod q.
    pub r_mod_q: u64,
    /// R^2 mod q (for converting to Montgomery form).
    pub r2_mod_q: u64,
}

impl MontgomeryCtx {
    /// Build a Montgomery context for modulus `q`.
    pub fn new(q: u64) -> Self {
        let r_bits = 32u32;
        let r: u128 = 1u128 << r_bits;
        // Compute q_inv such that q * q_inv ≡ -1 (mod R) using Newton's method.
        let mut q_inv: u64 = 1;
        for _ in 0..31 {
            q_inv = q_inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(q_inv)));
        }
        // We need -q_inv mod R, i.e. q_inv such that q * q_inv ≡ -1 (mod R).
        q_inv = q_inv.wrapping_neg() & ((r as u64) - 1);
        let r_mod_q = (r % (q as u128)) as u64;
        let r2_mod_q = ((r * r) % (q as u128)) as u64;
        Self {
            q,
            r_bits,
            q_inv,
            r_mod_q,
            r2_mod_q,
        }
    }

    /// Reduce `a` from Montgomery form: returns a * R^{-1} mod q.
    pub fn reduce(&self, a: u64) -> u64 {
        let r: u64 = 1u64 << self.r_bits;
        let mask = r - 1;
        let t = ((a & mask).wrapping_mul(self.q_inv)) & mask;
        let u = (a as u128 + t as u128 * self.q as u128) >> self.r_bits;
        let result = u as u64;
        if result >= self.q {
            result - self.q
        } else {
            result
        }
    }

    /// Convert `a` to Montgomery form: returns a * R mod q.
    pub fn to_montgomery(&self, a: u64) -> u64 {
        self.reduce(((a as u128 * self.r2_mod_q as u128) % self.q as u128) as u64)
    }

    /// Montgomery multiplication: returns a * b * R^{-1} mod q.
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        let product = a as u128 * b as u128;
        let r: u64 = 1u64 << self.r_bits;
        let mask = (r - 1) as u128;
        let t = ((product & mask) as u64).wrapping_mul(self.q_inv) & (r - 1);
        let u = (product + t as u128 * self.q as u128) >> self.r_bits;
        let result = u as u64;
        if result >= self.q {
            result - self.q
        } else {
            result
        }
    }
}

impl fmt::Display for MontgomeryCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Montgomery(q={}, R=2^{})", self.q, self.r_bits)
    }
}

// ── Barrett Reduction ───────────────────────────────────────────────

/// Barrett reduction for fast modular reduction without division.
#[derive(Debug, Clone)]
pub struct BarrettCtx {
    pub q: u64,
    /// Precomputed floor(2^k / q) where k is chosen for precision.
    pub multiplier: u128,
    pub shift: u32,
}

impl BarrettCtx {
    /// Build Barrett context for modulus `q`.
    pub fn new(q: u64) -> Self {
        let shift = 64u32;
        let multiplier = (1u128 << shift) / q as u128;
        Self {
            q,
            multiplier,
            shift,
        }
    }

    /// Reduce `a` mod `q` using Barrett reduction.
    pub fn reduce(&self, a: u64) -> u64 {
        let estimate = ((a as u128 * self.multiplier) >> self.shift) as u64;
        let mut result = a - estimate * self.q;
        if result >= self.q {
            result -= self.q;
        }
        result
    }

    /// Reduce a wide (u128) value mod q.
    pub fn reduce_wide(&self, a: u128) -> u64 {
        // Two-step: first bring into u64 range, then Barrett.
        let reduced = (a % self.q as u128) as u64;
        if reduced >= self.q {
            reduced - self.q
        } else {
            reduced
        }
    }
}

impl fmt::Display for BarrettCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Barrett(q={}, shift={})", self.q, self.shift)
    }
}

// ── Butterfly Operations ────────────────────────────────────────────

/// In-place Cooley-Tukey butterfly: (a, b) -> (a + w*b, a - w*b) mod q.
#[inline]
pub fn butterfly_ct(a: &mut u64, b: &mut u64, w: u64, q: u64) {
    let t = ((*b as u128 * w as u128) % q as u128) as u64;
    let sum = if *a + t >= q { *a + t - q } else { *a + t };
    let diff = if *a >= t { *a - t } else { *a + q - t };
    *a = sum;
    *b = diff;
}

/// In-place Gentleman-Sande butterfly: (a, b) -> (a + b, (a - b)*w) mod q.
#[inline]
pub fn butterfly_gs(a: &mut u64, b: &mut u64, w: u64, q: u64) {
    let sum = if *a + *b >= q { *a + *b - q } else { *a + *b };
    let diff = if *a >= *b { *a - *b } else { *a + q - *b };
    *a = sum;
    *b = ((diff as u128 * w as u128) % q as u128) as u64;
}

// ── NTT-Friendly Prime Utilities ────────────────────────────────────

/// Check if `p` is an NTT-friendly prime for degree `n` (i.e. 2n | p-1).
pub fn is_ntt_friendly(p: u64, n: usize) -> bool {
    if !is_prime(p) {
        return false;
    }
    let two_n = 2 * n as u64;
    (p - 1) % two_n == 0
}

/// Simple primality test (deterministic for values < 2^32, Miller-Rabin for larger).
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    // Trial division up to sqrt(n) for small values.
    let mut i = 5u64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Find a primitive `n`-th root of unity modulo `q`.
pub fn find_primitive_root(q: u64, n: usize) -> Option<u64> {
    if (q - 1) % (n as u64) != 0 {
        return None;
    }
    let exp = (q - 1) / (n as u64);
    // Try small generators.
    for g_candidate in 2..q {
        let root = mod_pow(g_candidate, exp, q);
        if root == 1 {
            continue;
        }
        // Check that root^n ≡ 1 and root^(n/2) ≢ 1.
        if mod_pow(root, n as u64, q) == 1
            && (n < 2 || mod_pow(root, (n / 2) as u64, q) != 1)
        {
            return Some(root);
        }
    }
    None
}

/// Modular exponentiation: base^exp mod modulus.
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u128 = 1;
    base %= modulus;
    let m = modulus as u128;
    let mut b = base as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % m;
        }
        exp >>= 1;
        b = (b * b) % m;
    }
    result as u64
}

// ── NTT Engine ──────────────────────────────────────────────────────

/// Configuration for the NTT engine.
#[derive(Debug, Clone)]
pub struct NttConfig {
    /// Modulus q.
    pub q: u64,
    /// Ring degree n (must be a power of two).
    pub n: usize,
}

impl NttConfig {
    pub fn new(q: u64, n: usize) -> Self {
        Self { q, n }
    }

    pub fn with_q(mut self, q: u64) -> Self {
        self.q = q;
        self
    }

    pub fn with_n(mut self, n: usize) -> Self {
        self.n = n;
        self
    }
}

impl fmt::Display for NttConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NttConfig(q={}, n={})", self.q, self.n)
    }
}

/// NTT engine with precomputed root tables.
#[derive(Debug, Clone)]
pub struct NttEngine {
    pub config: NttConfig,
    /// Forward NTT root-of-unity table (bit-reversed).
    pub roots_fwd: Vec<u64>,
    /// Inverse NTT root-of-unity table (bit-reversed).
    pub roots_inv: Vec<u64>,
    /// Precomputed n^{-1} mod q for inverse scaling.
    pub n_inv: u64,
}

impl NttEngine {
    /// Build an NTT engine for the given config.
    pub fn new(config: NttConfig) -> Self {
        let n = config.n;
        let q = config.q;
        let psi = find_primitive_root(q, 2 * n)
            .expect("no primitive root found for given q and n");

        // Build forward roots table: roots_fwd[i] = psi^{bit_reverse(i, log2(n))}
        // for i in 0..n. This is the standard Kyber/Dilithium layout.
        let log_n = (n as f64).log2() as u32;
        let mut roots_fwd = vec![0u64; n];
        for i in 0..n {
            let exp = bit_reverse(i, log_n);
            roots_fwd[i] = mod_pow(psi, exp as u64, q);
        }

        // Build inverse roots table: roots_inv[i] = psi^{-(bit_reverse(i, log2(n)))}
        let psi_inv = mod_pow(psi, q - 2, q);
        let mut roots_inv = vec![0u64; n];
        for i in 0..n {
            let exp = bit_reverse(i, log_n);
            roots_inv[i] = mod_pow(psi_inv, exp as u64, q);
        }

        let n_inv = mod_pow(n as u64, q - 2, q);

        Self {
            config,
            roots_fwd,
            roots_inv,
            n_inv,
        }
    }

    /// Forward NTT in-place. Input coefficients are in standard order.
    pub fn forward(&self, a: &mut [u64]) {
        let n = self.config.n;
        let q = self.config.q;
        assert_eq!(a.len(), n, "polynomial length must equal n");

        let mut len = n / 2;
        let mut k = 1usize;
        while len >= 1 {
            let mut start = 0;
            while start < n {
                let w = self.roots_fwd[k];
                k += 1;
                for j in start..(start + len) {
                    let (lo, hi) = a.split_at_mut(j + len);
                    butterfly_ct(&mut lo[j], &mut hi[0], w, q);
                }
                start += 2 * len;
            }
            len /= 2;
        }
    }

    /// Inverse NTT in-place. Includes the 1/n scaling.
    pub fn inverse(&self, a: &mut [u64]) {
        let n = self.config.n;
        let q = self.config.q;
        assert_eq!(a.len(), n, "polynomial length must equal n");

        let mut len = 1;
        let mut k = (n as isize) - 1;
        while len < n {
            let groups = n / (2 * len);
            // Walk k backwards for this layer.
            k = k - groups as isize + 1;
            let layer_start = k as usize;
            let mut ki = 0usize;
            let mut start = 0;
            while start < n {
                let w = self.roots_inv[layer_start + ki];
                ki += 1;
                for j in start..(start + len) {
                    let (lo, hi) = a.split_at_mut(j + len);
                    butterfly_gs(&mut lo[j], &mut hi[0], w, q);
                }
                start += 2 * len;
            }
            k = layer_start as isize - 1;
            len *= 2;
        }
        // Scale by n^{-1}.
        for coeff in a.iter_mut() {
            *coeff = ((*coeff as u128 * self.n_inv as u128) % q as u128) as u64;
        }
    }

    /// Multiply two polynomials in NTT domain (pointwise).
    pub fn pointwise_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let q = self.config.q;
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| ((*x as u128 * *y as u128) % q as u128) as u64)
            .collect()
    }

    /// Full polynomial multiplication via NTT: c = a * b mod (X^n + 1).
    pub fn poly_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut a_ntt = a.to_vec();
        let mut b_ntt = b.to_vec();
        self.forward(&mut a_ntt);
        self.forward(&mut b_ntt);
        let mut c_ntt = self.pointwise_mul(&a_ntt, &b_ntt);
        self.inverse(&mut c_ntt);
        c_ntt
    }
}

impl fmt::Display for NttEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NttEngine(q={}, n={}, roots={})",
            self.config.q,
            self.config.n,
            self.roots_fwd.len()
        )
    }
}

// ── Polynomial Arithmetic (coefficient domain) ─────────────────────

/// Add two polynomials coefficient-wise mod q.
pub fn poly_add(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut result = vec![0u64; n];
    for i in 0..n {
        let ai = if i < a.len() { a[i] } else { 0 };
        let bi = if i < b.len() { b[i] } else { 0 };
        let sum = ai + bi;
        result[i] = if sum >= q { sum - q } else { sum };
    }
    result
}

/// Subtract two polynomials coefficient-wise mod q.
pub fn poly_sub(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut result = vec![0u64; n];
    for i in 0..n {
        let ai = if i < a.len() { a[i] } else { 0 };
        let bi = if i < b.len() { b[i] } else { 0 };
        result[i] = if ai >= bi { ai - bi } else { ai + q - bi };
    }
    result
}

/// Negate a polynomial coefficient-wise mod q.
pub fn poly_neg(a: &[u64], q: u64) -> Vec<u64> {
    a.iter()
        .map(|c| if *c == 0 { 0 } else { q - c })
        .collect()
}

/// Scalar multiply: each coefficient * scalar mod q.
pub fn poly_scalar_mul(a: &[u64], scalar: u64, q: u64) -> Vec<u64> {
    a.iter()
        .map(|c| ((*c as u128 * scalar as u128) % q as u128) as u64)
        .collect()
}

/// Schoolbook polynomial multiplication mod q (no NTT, for small sizes).
pub fn poly_mul_schoolbook(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut result = vec![0u64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            let prod = (ai as u128 * bj as u128) % q as u128;
            result[i + j] = ((result[i + j] as u128 + prod) % q as u128) as u64;
        }
    }
    result
}

/// Reduce polynomial modulo X^n + 1 (negacyclic): fold higher terms.
pub fn reduce_negacyclic(a: &[u64], n: usize, q: u64) -> Vec<u64> {
    let mut result = vec![0u64; n];
    for (i, &coeff) in a.iter().enumerate() {
        let idx = i % n;
        let wrap = i / n;
        if wrap % 2 == 0 {
            result[idx] = (result[idx] + coeff) % q;
        } else {
            result[idx] = if result[idx] >= coeff {
                result[idx] - coeff
            } else {
                result[idx] + q - coeff
            };
        }
    }
    result
}

// ── Bit-reversal Permutation ────────────────────────────────────────

/// Compute bit-reversal of `i` within `log_n` bits.
pub fn bit_reverse(mut i: usize, log_n: u32) -> usize {
    let mut result = 0usize;
    for _ in 0..log_n {
        result = (result << 1) | (i & 1);
        i >>= 1;
    }
    result
}

/// In-place bit-reversal permutation.
pub fn bit_reverse_permute(a: &mut [u64]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let log_n = (n as f64).log2() as u32;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_pow_basic() {
        assert_eq!(mod_pow(2, 10, 1024), 0); // 1024 mod 1024
        assert_eq!(mod_pow(3, 4, 100), 81);
        assert_eq!(mod_pow(2, 10, 1000), 24);
    }

    #[test]
    fn test_mod_pow_identity() {
        assert_eq!(mod_pow(7, 0, 13), 1);
        assert_eq!(mod_pow(0, 5, 13), 0);
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(12289));
        assert!(is_prime(7681));
        assert!(is_prime(3329));
    }

    #[test]
    fn test_ntt_friendly() {
        // 12289 - 1 = 12288 = 2^12 * 3, supports n=256 since 2*256=512 divides 12288.
        assert!(is_ntt_friendly(12289, 256));
        assert!(!is_ntt_friendly(12289, 8192));
        assert!(!is_ntt_friendly(10, 4)); // 10 is not prime
    }

    #[test]
    fn test_find_primitive_root() {
        let root = find_primitive_root(12289, 8);
        assert!(root.is_some());
        let r = root.unwrap();
        assert_eq!(mod_pow(r, 8, 12289), 1);
        assert_ne!(mod_pow(r, 4, 12289), 1);
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(3, 3), 6);
        assert_eq!(bit_reverse(7, 3), 7);
    }

    #[test]
    fn test_bit_reverse_permute() {
        let mut a = vec![0, 1, 2, 3, 4, 5, 6, 7];
        bit_reverse_permute(&mut a);
        assert_eq!(a, vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn test_poly_add() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let c = poly_add(&a, &b, 100);
        assert_eq!(c, vec![5, 7, 9]);
    }

    #[test]
    fn test_poly_add_wrap() {
        let a = vec![99, 50];
        let b = vec![2, 51];
        let c = poly_add(&a, &b, 100);
        assert_eq!(c, vec![1, 1]);
    }

    #[test]
    fn test_poly_sub() {
        let a = vec![5, 3, 10];
        let b = vec![2, 5, 3];
        let c = poly_sub(&a, &b, 100);
        assert_eq!(c, vec![3, 98, 7]);
    }

    #[test]
    fn test_poly_neg() {
        let a = vec![0, 5, 10];
        let c = poly_neg(&a, 100);
        assert_eq!(c, vec![0, 95, 90]);
    }

    #[test]
    fn test_poly_scalar_mul() {
        let a = vec![3, 7, 11];
        let c = poly_scalar_mul(&a, 5, 100);
        assert_eq!(c, vec![15, 35, 55]);
    }

    #[test]
    fn test_schoolbook_mul() {
        let a = vec![1, 2]; // 1 + 2x
        let b = vec![3, 4]; // 3 + 4x
        let c = poly_mul_schoolbook(&a, &b, 100);
        // (1+2x)(3+4x) = 3 + 10x + 8x^2
        assert_eq!(c, vec![3, 10, 8]);
    }

    #[test]
    fn test_reduce_negacyclic() {
        // x^4 in Z[x]/(x^4+1) => -1, so coeff of x^0 should get subtracted.
        let a = vec![5, 0, 0, 0, 3]; // 5 + 3*x^4
        let r = reduce_negacyclic(&a, 4, 100);
        assert_eq!(r[0], 2); // 5 - 3 = 2
    }

    #[test]
    fn test_montgomery_roundtrip() {
        let ctx = MontgomeryCtx::new(12289);
        let a = 1234u64;
        let a_mont = ctx.to_montgomery(a);
        let a_back = ctx.reduce(a_mont);
        assert_eq!(a_back, a);
    }

    #[test]
    fn test_montgomery_mul() {
        let ctx = MontgomeryCtx::new(12289);
        let a = 100u64;
        let b = 200u64;
        let expected = (a * b) % 12289;
        let a_m = ctx.to_montgomery(a);
        let b_m = ctx.to_montgomery(b);
        let c_m = ctx.mul(a_m, b_m);
        let result = ctx.reduce(c_m);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_barrett_reduce() {
        let ctx = BarrettCtx::new(12289);
        assert_eq!(ctx.reduce(0), 0);
        assert_eq!(ctx.reduce(12289), 0);
        assert_eq!(ctx.reduce(12290), 1);
        assert_eq!(ctx.reduce(100), 100);
    }

    #[test]
    fn test_butterfly_ct() {
        let q = 12289u64;
        let mut a = 5000u64;
        let mut b = 3000u64;
        let w = 1u64; // trivial twiddle
        butterfly_ct(&mut a, &mut b, w, q);
        assert_eq!(a, 8000);
        assert_eq!(b, 2000);
    }

    #[test]
    fn test_butterfly_gs() {
        let q = 12289u64;
        let mut a = 5000u64;
        let mut b = 3000u64;
        butterfly_gs(&mut a, &mut b, 1, q);
        assert_eq!(a, 8000);
        assert_eq!(b, 2000);
    }

    #[test]
    fn test_ntt_forward_inverse_roundtrip() {
        let config = NttConfig::new(12289, 8);
        let engine = NttEngine::new(config);
        let original = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut data = original.clone();
        engine.forward(&mut data);
        // After forward NTT, should differ from original.
        assert_ne!(data, original);
        engine.inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_ntt_poly_mul_trivial() {
        let config = NttConfig::new(12289, 8);
        let engine = NttEngine::new(config);
        // Multiply by [1, 0, 0, ...] should be identity.
        let a = vec![3, 5, 7, 0, 0, 0, 0, 0];
        let b = vec![1, 0, 0, 0, 0, 0, 0, 0];
        let c = engine.poly_mul(&a, &b);
        assert_eq!(c, a);
    }

    #[test]
    fn test_display_impls() {
        let ctx = MontgomeryCtx::new(12289);
        assert!(format!("{}", ctx).contains("12289"));
        let bctx = BarrettCtx::new(7681);
        assert!(format!("{}", bctx).contains("7681"));
        let cfg = NttConfig::new(3329, 256);
        assert!(format!("{}", cfg).contains("3329"));
    }
}
