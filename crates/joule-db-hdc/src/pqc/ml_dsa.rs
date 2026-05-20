//! ML-DSA (Module Lattice Digital Signature Algorithm) - FIPS 204
//!
//! Implementation of the NIST-standardized post-quantum signature scheme
//! based on Module Learning With Errors and Short Integer Solution problems.
//!
//! ## Parameter Sets
//!
//! | Parameter   | (k,l) | Security | Classical | Quantum |
//! |-------------|-------|----------|-----------|---------|
//! | ML-DSA-44   | (4,4) | Level 2  | AES-128+  | NIST-2  |
//! | ML-DSA-65   | (6,5) | Level 3  | AES-192   | NIST-3  |
//! | ML-DSA-87   | (8,7) | Level 5  | AES-256   | NIST-5  |

use super::common::{ConstantTime, SecureZeroingVec, Sha3_256, Sha3_512, Shake128, Shake256};
use super::{PqcError, PqcResult};
use rand::Rng;

// ============================================================================
// Constants (same ring as ML-KEM)
// ============================================================================

/// Polynomial degree
pub const N: usize = 256;

/// Prime modulus q = 8380417 = 2^23 - 2^13 + 1 (NTT-friendly)
pub const Q: i32 = 8380417;

/// Dropped bits from t: d = 13
pub const D: usize = 13;

/// NTT root of unity
pub const ZETA: i32 = 1753;

// ============================================================================
// NTT Tables for ML-DSA (different from ML-KEM due to different q)
// ============================================================================

/// Precomputed zetas for forward NTT (bit-reversed order per FIPS 204)
const ZETAS: [i32; 256] = compute_zetas();

/// Reverse 8 bits
const fn bit_reverse_8(mut x: u8) -> u8 {
    let mut result = 0u8;
    let mut i = 0;
    while i < 8 {
        result = (result << 1) | (x & 1);
        x >>= 1;
        i += 1;
    }
    result
}

/// Compute zetas in bit-reversed order at compile time
const fn compute_zetas() -> [i32; 256] {
    let mut zetas = [0i32; 256];
    // Compute all powers of ζ mod q
    let mut powers = [0i64; 512];
    powers[0] = 1;
    let mut i = 1;
    while i < 512 {
        powers[i] = (powers[i - 1] * ZETA as i64) % Q as i64;
        i += 1;
    }
    // Assign in bit-reversed order: zetas[i] = ζ^{brv(i)} mod q
    let mut i = 0;
    while i < 256 {
        zetas[i] = powers[bit_reverse_8(i as u8) as usize] as i32;
        i += 1;
    }
    zetas
}

// ============================================================================
// ML-DSA Parameters
// ============================================================================

/// ML-DSA parameter set
#[derive(Clone, Copy, Debug)]
pub struct MlDsaParams {
    /// Security level name
    pub name: &'static str,
    /// Module dimensions (k, l)
    pub k: usize,
    pub l: usize,
    /// Max coefficient of secret key
    pub eta: usize,
    /// Challenge weight (number of ±1s)
    pub tau: usize,
    /// Coefficient range for hint
    pub gamma1: i32,
    /// Rounding range
    pub gamma2: i32,
    /// Max hint weight
    pub omega: usize,
    /// Number of dropped bits for commitment
    pub beta: i32,
}

/// ML-DSA-44 parameters (NIST Level 2)
pub const ML_DSA_44_PARAMS: MlDsaParams = MlDsaParams {
    name: "ML-DSA-44",
    k: 4,
    l: 4,
    eta: 2,
    tau: 39,
    gamma1: 1 << 17,
    gamma2: (Q - 1) / 88,
    omega: 80,
    beta: 78,
};

/// ML-DSA-65 parameters (NIST Level 3)
pub const ML_DSA_65_PARAMS: MlDsaParams = MlDsaParams {
    name: "ML-DSA-65",
    k: 6,
    l: 5,
    eta: 4,
    tau: 49,
    gamma1: 1 << 19,
    gamma2: (Q - 1) / 32,
    omega: 55,
    beta: 196,
};

/// ML-DSA-87 parameters (NIST Level 5)
pub const ML_DSA_87_PARAMS: MlDsaParams = MlDsaParams {
    name: "ML-DSA-87",
    k: 8,
    l: 7,
    eta: 2,
    tau: 60,
    gamma1: 1 << 19,
    gamma2: (Q - 1) / 32,
    omega: 75,
    beta: 120,
};

impl MlDsaParams {
    /// Verification key size in bytes
    pub const fn verification_key_size(&self) -> usize {
        // pk = ρ || t1
        32 + 320 * self.k
    }

    /// Signing key size in bytes
    pub const fn signing_key_size(&self) -> usize {
        // sk = ρ || K || tr || s1 || s2 || t0
        32 + 32
            + 64
            + 32 * self.l * (1 + self.eta.ilog2() as usize)
            + 32 * self.k * (1 + self.eta.ilog2() as usize)
            + 416 * self.k
    }

    /// Signature size in bytes
    pub const fn signature_size(&self) -> usize {
        // σ = c_tilde || z || h
        let lambda = match self.k {
            4 => 128,
            6 => 192,
            _ => 256,
        };
        lambda / 4 + 32 * self.l * (1 + self.gamma1.ilog2() as usize) + self.omega + self.k
    }
}

// ============================================================================
// Polynomial Operations
// ============================================================================

/// Polynomial in Z_q[X]/(X^256 + 1)
#[derive(Clone)]
pub struct Polynomial {
    coeffs: [i32; N],
}

impl Polynomial {
    /// Create zero polynomial
    pub fn zero() -> Self {
        Self { coeffs: [0; N] }
    }

    /// Reduce coefficient modulo q to centered representation
    #[inline]
    fn reduce(a: i32) -> i32 {
        let t = a % Q;
        if t > Q / 2 {
            t - Q
        } else if t < -Q / 2 {
            t + Q
        } else {
            t
        }
    }

    /// Montgomery reduction for ML-DSA
    #[inline]
    fn montgomery_reduce(a: i64) -> i32 {
        const QINV: i64 = 58728449; // q^(-1) mod 2^32
        const R: i64 = 1 << 32;

        let t = ((a as i32 as i64).wrapping_mul(QINV) as i32) as i64;
        let r = (a - t.wrapping_mul(Q as i64)) >> 32;
        r as i32
    }

    /// Add two polynomials
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            result.coeffs[i] = Self::reduce(self.coeffs[i] + other.coeffs[i]);
        }
        result
    }

    /// Subtract two polynomials
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            result.coeffs[i] = Self::reduce(self.coeffs[i] - other.coeffs[i]);
        }
        result
    }

    /// NTT (Number Theoretic Transform) — FIPS 204 Algorithm 41
    pub fn ntt(&mut self) {
        let mut k = 0usize;
        let mut len = 128;

        while len >= 1 {
            let mut start = 0;
            while start < N {
                k += 1;
                let zeta = ZETAS[k] as i64;

                for j in start..(start + len) {
                    let t = ((zeta * self.coeffs[j + len] as i64) % Q as i64) as i32;
                    self.coeffs[j + len] = Self::reduce(self.coeffs[j] - t);
                    self.coeffs[j] = Self::reduce(self.coeffs[j] + t);
                }
                start += 2 * len;
            }
            len >>= 1;
        }
    }

    /// Inverse NTT — FIPS 204 Algorithm 42
    pub fn inv_ntt(&mut self) {
        let mut k = 256usize;
        let mut len = 1;

        while len <= 128 {
            let mut start = 0;
            while start < N {
                k -= 1;
                let zeta = -(ZETAS[k] as i64);

                for j in start..(start + len) {
                    let t = self.coeffs[j];
                    self.coeffs[j] = Self::reduce(t + self.coeffs[j + len]);
                    self.coeffs[j + len] =
                        ((zeta * Self::reduce(t - self.coeffs[j + len]) as i64) % Q as i64) as i32;
                }
                start += 2 * len;
            }
            len <<= 1;
        }

        // Multiply by n^(-1) mod q
        const N_INV: i64 = 8347681; // 256^(-1) mod q
        for coeff in &mut self.coeffs {
            *coeff = Self::reduce(((*coeff as i64 * N_INV) % Q as i64) as i32);
        }
    }

    /// Pointwise multiplication in NTT domain
    pub fn pointwise_mul(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            result.coeffs[i] =
                Self::reduce(((self.coeffs[i] as i64 * other.coeffs[i] as i64) % Q as i64) as i32);
        }
        result
    }

    /// Check if infinity norm is below bound
    pub fn check_norm(&self, bound: i32) -> bool {
        for coeff in &self.coeffs {
            let val = Self::reduce(*coeff);
            if val.abs() >= bound {
                return false;
            }
        }
        true
    }

    /// Power2Round: decompose into high and low bits
    pub fn power2round(&self) -> (Self, Self) {
        let mut a1 = Self::zero();
        let mut a0 = Self::zero();

        for i in 0..N {
            let a = Self::reduce(self.coeffs[i]);
            let a_plus = if a < 0 { a + Q } else { a };

            a1.coeffs[i] = (a_plus + (1 << (D - 1)) - 1) >> D;
            a0.coeffs[i] = a - (a1.coeffs[i] << D);
        }

        (a1, a0)
    }

    /// Decompose into high and low parts
    pub fn decompose(&self, gamma2: i32) -> (Self, Self) {
        let mut a1 = Self::zero();
        let mut a0 = Self::zero();

        for i in 0..N {
            let a = Self::reduce(self.coeffs[i]);
            let a_plus = if a < 0 { a + Q } else { a };

            a0.coeffs[i] = a_plus % (2 * gamma2);
            if a0.coeffs[i] > gamma2 {
                a0.coeffs[i] -= 2 * gamma2;
            }

            if a_plus - a0.coeffs[i] == Q - 1 {
                a1.coeffs[i] = 0;
                a0.coeffs[i] -= 1;
            } else {
                a1.coeffs[i] = (a_plus - a0.coeffs[i]) / (2 * gamma2);
            }
        }

        (a1, a0)
    }

    /// Compute hint polynomial
    pub fn make_hint(&self, other: &Self, gamma2: i32) -> (Self, usize) {
        let mut hint = Self::zero();
        let mut count = 0;

        for i in 0..N {
            let a = Self::reduce(self.coeffs[i]);
            let b = Self::reduce(other.coeffs[i]);

            let a_plus = if a < 0 { a + Q } else { a };
            let b_plus = if b < 0 { b + Q } else { b };

            let a1 = Self::high_bits(a_plus, gamma2);
            let ab = (a_plus + b_plus) % Q;
            let ab1 = Self::high_bits(ab, gamma2);

            if a1 != ab1 {
                hint.coeffs[i] = 1;
                count += 1;
            }
        }

        (hint, count)
    }

    /// Use hint to recover high bits (FIPS 204 Algorithm 38)
    pub fn use_hint(&self, hint: &Self, gamma2: i32) -> Self {
        let m = (Q - 1) / (2 * gamma2); // FIPS 204: m = (q-1)/(2γ2)
        let mut result = Self::zero();

        for i in 0..N {
            let a = Self::reduce(self.coeffs[i]);
            let h = hint.coeffs[i];

            let a_plus = if a < 0 { a + Q } else { a };

            // Use Decompose to get r1 and r0 (matches FIPS 204 Algorithm 37)
            let mut a0 = a_plus % (2 * gamma2);
            if a0 > gamma2 {
                a0 -= 2 * gamma2;
            }
            let (r1, a0) = if a_plus - a0 == Q - 1 {
                (0, a0 - 1)
            } else {
                ((a_plus - a0) / (2 * gamma2), a0)
            };

            if h == 0 {
                result.coeffs[i] = r1;
            } else if a0 > 0 {
                result.coeffs[i] = (r1 + 1) % m;
            } else {
                result.coeffs[i] = (r1 + m - 1) % m;
            }
        }

        result
    }

    /// Compute HighBits using the same logic as Decompose (FIPS 204 Algorithm 37)
    fn high_bits(a_plus: i32, gamma2: i32) -> i32 {
        let mut a0 = a_plus % (2 * gamma2);
        if a0 > gamma2 {
            a0 -= 2 * gamma2;
        }
        if a_plus - a0 == Q - 1 {
            0
        } else {
            (a_plus - a0) / (2 * gamma2)
        }
    }

    /// Sample polynomial with coefficients in [-eta, eta]
    pub fn sample_uniform_eta(seed: &[u8], nonce: u16, eta: usize) -> PqcResult<Self> {
        let mut poly = Self::zero();
        let mut shake = Shake256::new();
        shake.absorb(seed);
        shake.absorb(&nonce.to_le_bytes());

        let mut buf = [0u8; 1];
        let mut ctr = 0;

        while ctr < N {
            shake.squeeze(&mut buf);

            match eta {
                2 => {
                    let t0 = buf[0] & 0x0F;
                    let t1 = buf[0] >> 4;

                    if t0 < 15 {
                        poly.coeffs[ctr] = 2 - (t0 % 5) as i32;
                        ctr += 1;
                    }
                    if ctr < N && t1 < 15 {
                        poly.coeffs[ctr] = 2 - (t1 % 5) as i32;
                        ctr += 1;
                    }
                }
                4 => {
                    let t0 = buf[0] & 0x0F;
                    let t1 = buf[0] >> 4;

                    if t0 < 9 {
                        poly.coeffs[ctr] = 4 - t0 as i32;
                        ctr += 1;
                    }
                    if ctr < N && t1 < 9 {
                        poly.coeffs[ctr] = 4 - t1 as i32;
                        ctr += 1;
                    }
                }
                _ => {
                    return Err(PqcError::InvalidParameter(format!(
                        "unsupported ML-DSA eta: {eta} (expected 2 or 4)"
                    )));
                }
            }
        }

        Ok(poly)
    }

    /// Sample polynomial uniformly from Z_q
    pub fn sample_uniform(seed: &[u8], nonce_i: u8, nonce_j: u8) -> Self {
        let mut poly = Self::zero();
        let mut shake = Shake128::new();
        shake.absorb(seed);
        shake.absorb(&[nonce_j, nonce_i]);

        let mut buf = [0u8; 3];
        let mut ctr = 0;

        while ctr < N {
            shake.squeeze(&mut buf);

            let t = ((buf[0] as i32) | ((buf[1] as i32) << 8) | ((buf[2] as i32) << 16)) & 0x7FFFFF;

            if t < Q {
                poly.coeffs[ctr] = t;
                ctr += 1;
            }
        }

        poly
    }

    /// Sample challenge polynomial with τ ones and τ minus ones
    pub fn sample_challenge(seed: &[u8], tau: usize) -> Self {
        let mut poly = Self::zero();
        let mut shake = Shake256::new();
        shake.absorb(seed);

        let mut signs = [0u8; 8];
        shake.squeeze(&mut signs);
        let signs = u64::from_le_bytes(signs);

        let mut pos = [0u8; 1];
        let mut ctr = 0;
        let mut sign_idx = 0;

        while ctr < tau {
            shake.squeeze(&mut pos);
            let j = pos[0] as usize;

            if j <= ctr + (N - tau) {
                let idx = ctr + (N - tau);
                poly.coeffs[idx] = poly.coeffs[j];
                poly.coeffs[j] = if (signs >> sign_idx) & 1 == 1 { -1 } else { 1 };
                sign_idx += 1;
                ctr += 1;
            }
        }

        poly
    }

    /// Sample polynomial with coefficients from [-gamma1+1, gamma1]
    pub fn sample_mask(seed: &[u8], nonce: u16, gamma1: i32) -> Self {
        let mut poly = Self::zero();
        let gamma1_bits = gamma1.ilog2() as usize + 1;

        let bytes_needed = (N * gamma1_bits + 7) / 8;
        let expanded = Shake256::xof(&[seed, &nonce.to_le_bytes()].concat(), bytes_needed);

        let mask = (1i32 << gamma1_bits) - 1;

        for i in 0..N {
            let bit_pos = i * gamma1_bits;
            let byte_pos = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            let mut val = (expanded[byte_pos] >> bit_offset) as i32;
            if bit_offset + gamma1_bits > 8 && byte_pos + 1 < expanded.len() {
                val |= (expanded[byte_pos + 1] as i32) << (8 - bit_offset);
            }
            if bit_offset + gamma1_bits > 16 && byte_pos + 2 < expanded.len() {
                val |= (expanded[byte_pos + 2] as i32) << (16 - bit_offset);
            }
            val &= mask;

            poly.coeffs[i] = gamma1 - val;
        }

        poly
    }

    /// Encode polynomial to bytes (for t1, 10 bits per coeff)
    pub fn encode_t1(&self) -> Vec<u8> {
        let mut result = vec![0u8; 320];

        for i in 0..N / 4 {
            let c0 = self.coeffs[4 * i] as u32;
            let c1 = self.coeffs[4 * i + 1] as u32;
            let c2 = self.coeffs[4 * i + 2] as u32;
            let c3 = self.coeffs[4 * i + 3] as u32;

            result[5 * i] = c0 as u8;
            result[5 * i + 1] = ((c0 >> 8) | (c1 << 2)) as u8;
            result[5 * i + 2] = ((c1 >> 6) | (c2 << 4)) as u8;
            result[5 * i + 3] = ((c2 >> 4) | (c3 << 6)) as u8;
            result[5 * i + 4] = (c3 >> 2) as u8;
        }

        result
    }

    /// Decode polynomial from bytes (for t1)
    pub fn decode_t1(bytes: &[u8]) -> Self {
        let mut poly = Self::zero();

        for i in 0..N / 4 {
            poly.coeffs[4 * i] = (bytes[5 * i] as i32) | ((bytes[5 * i + 1] as i32 & 0x03) << 8);
            poly.coeffs[4 * i + 1] =
                ((bytes[5 * i + 1] as i32) >> 2) | ((bytes[5 * i + 2] as i32 & 0x0F) << 6);
            poly.coeffs[4 * i + 2] =
                ((bytes[5 * i + 2] as i32) >> 4) | ((bytes[5 * i + 3] as i32 & 0x3F) << 4);
            poly.coeffs[4 * i + 3] =
                ((bytes[5 * i + 3] as i32) >> 6) | ((bytes[5 * i + 4] as i32) << 2);
        }

        poly
    }
}

impl Default for Polynomial {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// Vector/Matrix Operations
// ============================================================================

/// Vector of polynomials
#[derive(Clone)]
pub struct PolyVec {
    polys: Vec<Polynomial>,
}

impl PolyVec {
    /// Create zero vector
    pub fn new(len: usize) -> Self {
        Self {
            polys: (0..len).map(|_| Polynomial::zero()).collect(),
        }
    }

    /// Length
    pub fn len(&self) -> usize {
        self.polys.len()
    }

    /// NTT all polynomials
    pub fn ntt(&mut self) {
        for poly in &mut self.polys {
            poly.ntt();
        }
    }

    /// Inverse NTT
    pub fn inv_ntt(&mut self) {
        for poly in &mut self.polys {
            poly.inv_ntt();
        }
    }

    /// Add vectors
    pub fn add(&self, other: &Self) -> Self {
        Self {
            polys: self
                .polys
                .iter()
                .zip(&other.polys)
                .map(|(a, b)| a.add(b))
                .collect(),
        }
    }

    /// Subtract vectors
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            polys: self
                .polys
                .iter()
                .zip(&other.polys)
                .map(|(a, b)| a.sub(b))
                .collect(),
        }
    }

    /// Check all polynomials have norm below bound
    pub fn check_norm(&self, bound: i32) -> bool {
        self.polys.iter().all(|p| p.check_norm(bound))
    }

    /// Power2Round for each polynomial
    pub fn power2round(&self) -> (Self, Self) {
        let (highs, lows): (Vec<_>, Vec<_>) = self.polys.iter().map(|p| p.power2round()).unzip();
        (Self { polys: highs }, Self { polys: lows })
    }

    /// Decompose for each polynomial
    pub fn decompose(&self, gamma2: i32) -> (Self, Self) {
        let (highs, lows): (Vec<_>, Vec<_>) =
            self.polys.iter().map(|p| p.decompose(gamma2)).unzip();
        (Self { polys: highs }, Self { polys: lows })
    }

    /// Make hint
    pub fn make_hint(&self, other: &Self, gamma2: i32) -> (Self, usize) {
        let mut total = 0;
        let mut hints = Vec::new();

        for (a, b) in self.polys.iter().zip(&other.polys) {
            let (h, c) = a.make_hint(b, gamma2);
            hints.push(h);
            total += c;
        }

        (Self { polys: hints }, total)
    }

    /// Use hint
    pub fn use_hint(&self, hint: &Self, gamma2: i32) -> Self {
        Self {
            polys: self
                .polys
                .iter()
                .zip(&hint.polys)
                .map(|(a, h)| a.use_hint(h, gamma2))
                .collect(),
        }
    }

    /// Encode t1 vector
    pub fn encode_t1(&self) -> Vec<u8> {
        self.polys.iter().flat_map(|p| p.encode_t1()).collect()
    }

    /// Decode t1 vector
    pub fn decode_t1(bytes: &[u8], k: usize) -> Self {
        Self {
            polys: (0..k)
                .map(|i| Polynomial::decode_t1(&bytes[320 * i..]))
                .collect(),
        }
    }
}

/// Matrix multiplication A * s (in NTT domain)
pub fn matrix_mul(a: &[PolyVec], s: &PolyVec) -> PolyVec {
    let k = a.len();
    let mut result = PolyVec::new(k);

    for i in 0..k {
        for j in 0..s.len() {
            let prod = a[i].polys[j].pointwise_mul(&s.polys[j]);
            result.polys[i] = result.polys[i].add(&prod);
        }
    }

    result
}

/// Generate matrix A from seed
pub fn expand_a(seed: &[u8; 32], k: usize, l: usize) -> Vec<PolyVec> {
    let mut a = Vec::with_capacity(k);

    for i in 0..k {
        let mut row = PolyVec::new(l);
        for j in 0..l {
            row.polys[j] = Polynomial::sample_uniform(seed, i as u8, j as u8);
        }
        a.push(row);
    }

    a
}

// ============================================================================
// Key Types
// ============================================================================

/// ML-DSA Verification Key (public key)
#[derive(Clone)]
pub struct MlDsaVerificationKey {
    data: Vec<u8>,
    params: MlDsaParams,
}

impl MlDsaVerificationKey {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], params: MlDsaParams) -> PqcResult<Self> {
        Ok(Self {
            data: bytes.to_vec(),
            params,
        })
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get parameters
    pub fn params(&self) -> MlDsaParams {
        self.params
    }
}

/// ML-DSA Signing Key (secret key)
#[derive(Clone)]
pub struct MlDsaSigningKey {
    data: SecureZeroingVec,
    params: MlDsaParams,
}

impl MlDsaSigningKey {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], params: MlDsaParams) -> PqcResult<Self> {
        Ok(Self {
            data: SecureZeroingVec::from_vec(bytes.to_vec()),
            params,
        })
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Get parameters
    pub fn params(&self) -> MlDsaParams {
        self.params
    }
}

/// ML-DSA Signature
#[derive(Clone)]
pub struct MlDsaSignature {
    data: Vec<u8>,
    params: MlDsaParams,
}

impl MlDsaSignature {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], params: MlDsaParams) -> PqcResult<Self> {
        Ok(Self {
            data: bytes.to_vec(),
            params,
        })
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get parameters
    pub fn params(&self) -> MlDsaParams {
        self.params
    }
}

// ============================================================================
// ML-DSA Core Algorithm
// ============================================================================

/// Core ML-DSA implementation
pub struct MlDsaCore;

impl MlDsaCore {
    /// Key generation (Algorithm 1: ML-DSA.KeyGen)
    pub fn keygen(params: MlDsaParams) -> PqcResult<(MlDsaVerificationKey, MlDsaSigningKey)> {
        let mut rng = rand::rng();
        let mut seed = [0u8; 32];
        rng.fill(&mut seed);

        Self::keygen_internal(params, &seed)
    }

    /// Internal keygen with explicit randomness
    pub fn keygen_internal(
        params: MlDsaParams,
        seed: &[u8; 32],
    ) -> PqcResult<(MlDsaVerificationKey, MlDsaSigningKey)> {
        let k = params.k;
        let l = params.l;

        // Expand seed: H(seed) = (ρ, ρ', K)
        let expanded = Shake256::xof(seed, 128);
        let rho: [u8; 32] = expanded[..32].try_into().unwrap();
        let rho_prime: [u8; 64] = expanded[32..96].try_into().unwrap();
        let kappa: [u8; 32] = expanded[96..128].try_into().unwrap();

        // Expand A from ρ
        let a = expand_a(&rho, k, l);

        // Sample s1, s2
        let mut s1 = PolyVec::new(l);
        let mut s2 = PolyVec::new(k);

        for i in 0..l {
            s1.polys[i] = Polynomial::sample_uniform_eta(&rho_prime, i as u16, params.eta)?;
        }
        for i in 0..k {
            s2.polys[i] = Polynomial::sample_uniform_eta(&rho_prime, (l + i) as u16, params.eta)?;
        }

        // Compute t = A*s1 + s2
        let mut s1_ntt = s1.clone();
        s1_ntt.ntt();

        let mut t = matrix_mul(&a, &s1_ntt);
        t.inv_ntt();
        t = t.add(&s2);

        // Power2Round: t = t1*2^d + t0
        let (t1, t0) = t.power2round();

        // Encode verification key: vk = ρ || t1
        let mut vk_bytes = Vec::with_capacity(params.verification_key_size());
        vk_bytes.extend_from_slice(&rho);
        vk_bytes.extend(t1.encode_t1());

        // tr = H(vk)
        let tr = Sha3_512::hash(&vk_bytes);

        // Encode signing key (simplified encoding)
        let mut sk_bytes = Vec::new();
        sk_bytes.extend_from_slice(&rho);
        sk_bytes.extend_from_slice(&kappa);
        sk_bytes.extend_from_slice(&tr);
        // Simplified: just store the seed for s1, s2, t0
        sk_bytes.extend_from_slice(&rho_prime);

        Ok((
            MlDsaVerificationKey {
                data: vk_bytes,
                params,
            },
            MlDsaSigningKey {
                data: SecureZeroingVec::from_vec(sk_bytes),
                params,
            },
        ))
    }

    /// Sign message (Algorithm 2: ML-DSA.Sign)
    pub fn sign(sk: &MlDsaSigningKey, message: &[u8]) -> PqcResult<MlDsaSignature> {
        let mut rng = rand::rng();
        let mut rnd = [0u8; 32];
        rng.fill(&mut rnd);

        Self::sign_internal(sk, message, &rnd)
    }

    /// Internal sign with explicit randomness
    pub fn sign_internal(
        sk: &MlDsaSigningKey,
        message: &[u8],
        rnd: &[u8; 32],
    ) -> PqcResult<MlDsaSignature> {
        let params = sk.params;
        let k = params.k;
        let l = params.l;

        // Parse signing key
        let sk_bytes = sk.as_bytes();
        let rho: [u8; 32] = sk_bytes[..32].try_into().unwrap();
        let kappa: [u8; 32] = sk_bytes[32..64].try_into().unwrap();
        let tr: [u8; 64] = sk_bytes[64..128].try_into().unwrap();
        let rho_prime: [u8; 64] = sk_bytes[128..192].try_into().unwrap();

        // Reconstruct s1, s2
        let mut s1 = PolyVec::new(l);
        let mut s2 = PolyVec::new(k);

        for i in 0..l {
            s1.polys[i] = Polynomial::sample_uniform_eta(&rho_prime, i as u16, params.eta)?;
        }
        for i in 0..k {
            s2.polys[i] = Polynomial::sample_uniform_eta(&rho_prime, (l + i) as u16, params.eta)?;
        }

        // Expand A
        let a = expand_a(&rho, k, l);

        // Compute t = A*s1 + s2 and t0
        let mut s1_ntt = s1.clone();
        s1_ntt.ntt();

        let mut t = matrix_mul(&a, &s1_ntt);
        t.inv_ntt();
        t = t.add(&s2);
        let (_, t0) = t.power2round();

        // μ = H(tr || M)
        let mu = Sha3_512::hash(&[&tr[..], message].concat());

        // ρ'' = H(K || rnd || μ)
        let rho_double_prime = Sha3_512::hash(&[&kappa[..], rnd, &mu[..]].concat());

        // Signing loop
        let mut nonce = 0u16;
        loop {
            // Sample y
            let mut y = PolyVec::new(l);
            for i in 0..l {
                y.polys[i] =
                    Polynomial::sample_mask(&rho_double_prime, nonce + i as u16, params.gamma1);
            }

            // w = A*y
            let mut y_ntt = y.clone();
            y_ntt.ntt();
            let mut w = matrix_mul(&a, &y_ntt);
            w.inv_ntt();

            // Decompose w
            let (w1, _) = w.decompose(params.gamma2);

            // c_tilde = H(μ || w1)
            let w1_encoded = w1.encode_t1(); // Simplified encoding
            let c_input = [&mu[..], &w1_encoded].concat();
            let c_tilde = Sha3_256::hash(&c_input);

            // Sample c
            let c = Polynomial::sample_challenge(&c_tilde, params.tau);

            // z = y + c*s1
            let mut c_ntt = c.clone();
            c_ntt.ntt();

            let mut z = PolyVec::new(l);
            for i in 0..l {
                let mut cs1 = c_ntt.pointwise_mul(&s1_ntt.polys[i]);
                cs1.inv_ntt();
                z.polys[i] = y.polys[i].add(&cs1);
            }

            // Check ||z||∞ < γ1 - β
            if !z.check_norm(params.gamma1 - params.beta) {
                nonce += l as u16;
                continue;
            }

            // r = w - c*s2
            let mut s2_ntt = s2.clone();
            s2_ntt.ntt();

            let mut r = PolyVec::new(k);
            for i in 0..k {
                let mut cs2 = c_ntt.pointwise_mul(&s2_ntt.polys[i]);
                cs2.inv_ntt();
                r.polys[i] = w.polys[i].sub(&cs2);
            }

            // Decompose r
            let (r1, r0) = r.decompose(params.gamma2);

            // Check ||r0||∞ < γ2 - β
            if !r0.check_norm(params.gamma2 - params.beta) {
                nonce += l as u16;
                continue;
            }

            // Compute ct0
            let mut t0_ntt = t0.clone();
            t0_ntt.ntt();

            let mut ct0 = PolyVec::new(k);
            for i in 0..k {
                let mut prod = c_ntt.pointwise_mul(&t0_ntt.polys[i]);
                prod.inv_ntt();
                ct0.polys[i] = prod;
            }

            // Check ||ct0||∞ < γ2 (FIPS 204 line 25)
            if !ct0.check_norm(params.gamma2) {
                nonce += l as u16;
                continue;
            }

            // Compute hint: MakeHint checks HighBits(w-cs2) ≠ HighBits(w-cs2+ct0)
            // Must use full r (w-cs2), NOT r0 (low bits only)
            let (hint, hint_count) = r.make_hint(&ct0, params.gamma2);

            if hint_count > params.omega {
                nonce += l as u16;
                continue;
            }

            // Encode signature: c_tilde || z || hint
            let mut sig_bytes = Vec::new();
            sig_bytes.extend_from_slice(&c_tilde);
            // Encode z (4 bytes per coefficient)
            for poly in &z.polys {
                for coeff in &poly.coeffs {
                    sig_bytes.extend_from_slice(&coeff.to_le_bytes());
                }
            }
            // Encode hint as packed bits (32 bytes per polynomial)
            for poly in &hint.polys {
                let mut hint_bytes = [0u8; 32];
                for j in 0..N {
                    if poly.coeffs[j] != 0 {
                        hint_bytes[j / 8] |= 1 << (j % 8);
                    }
                }
                sig_bytes.extend_from_slice(&hint_bytes);
            }

            return Ok(MlDsaSignature {
                data: sig_bytes,
                params,
            });
        }
    }

    /// Verify signature (Algorithm 3: ML-DSA.Verify)
    pub fn verify(vk: &MlDsaVerificationKey, message: &[u8], sig: &MlDsaSignature) -> bool {
        let params = vk.params;
        let k = params.k;
        let l = params.l;

        // Parse verification key
        let vk_bytes = vk.as_bytes();
        let rho: [u8; 32] = vk_bytes[..32].try_into().unwrap();
        let t1 = PolyVec::decode_t1(&vk_bytes[32..], k);

        // Parse signature: c_tilde || z || hint
        let sig_bytes = sig.as_bytes();
        if sig_bytes.len() < 32 {
            return false;
        }
        let c_tilde: [u8; 32] = sig_bytes[..32].try_into().unwrap();

        // Decode z
        let mut z = PolyVec::new(l);
        let z_start = 32;
        for i in 0..l {
            for j in 0..N {
                let idx = z_start + (i * N + j) * 4;
                if idx + 4 > sig_bytes.len() {
                    return false;
                }
                z.polys[i].coeffs[j] =
                    i32::from_le_bytes(sig_bytes[idx..idx + 4].try_into().unwrap());
            }
        }

        // Decode hint (packed bits, 32 bytes per polynomial)
        let hint_start = z_start + l * N * 4;
        let mut hint = PolyVec::new(k);
        for i in 0..k {
            let base = hint_start + i * 32;
            if base + 32 > sig_bytes.len() {
                return false;
            }
            for j in 0..N {
                if (sig_bytes[base + j / 8] >> (j % 8)) & 1 == 1 {
                    hint.polys[i].coeffs[j] = 1;
                }
            }
        }

        // Check ||z||∞ < γ1 - β
        if !z.check_norm(params.gamma1 - params.beta) {
            return false;
        }

        // Expand A
        let a = expand_a(&rho, k, l);

        // Sample c from c_tilde
        let c = Polynomial::sample_challenge(&c_tilde, params.tau);

        // Compute w' = A*z - c*t1*2^d (all in NTT domain)
        let mut z_ntt = z.clone();
        z_ntt.ntt();

        let mut w_prime = matrix_mul(&a, &z_ntt);

        let mut c_ntt = c.clone();
        c_ntt.ntt();

        let mut t1_ntt = t1.clone();
        t1_ntt.ntt();

        for i in 0..k {
            let ct1 = c_ntt.pointwise_mul(&t1_ntt.polys[i]);
            // Scale ct1 by 2^d in NTT domain (direct modular reduction, not Montgomery)
            let mut ct1_scaled = Polynomial::zero();
            for j in 0..N {
                ct1_scaled.coeffs[j] =
                    Polynomial::reduce(((ct1.coeffs[j] as i64) * (1i64 << D) % Q as i64) as i32);
            }
            w_prime.polys[i] = w_prime.polys[i].sub(&ct1_scaled);
        }

        w_prime.inv_ntt();

        // Apply hints to recover w1
        let w1_prime = w_prime.use_hint(&hint, params.gamma2);

        // Recompute c_tilde and compare
        let tr = Sha3_512::hash(vk_bytes);
        let mu = Sha3_512::hash(&[&tr[..], message].concat());
        let w1_encoded = w1_prime.encode_t1();
        let c_tilde_prime = Sha3_256::hash(&[&mu[..], &w1_encoded].concat());

        ConstantTime::ct_eq(&c_tilde, &c_tilde_prime)
    }
}

// ============================================================================
// ML-DSA Variants
// ============================================================================

/// ML-DSA-44 (NIST Security Level 2)
pub struct MlDsa44;

impl MlDsa44 {
    /// Parameters
    pub const PARAMS: MlDsaParams = ML_DSA_44_PARAMS;

    /// Generate key pair
    pub fn keygen() -> PqcResult<(MlDsaVerificationKey, MlDsaSigningKey)> {
        MlDsaCore::keygen(Self::PARAMS)
    }

    /// Sign message
    pub fn sign(sk: &MlDsaSigningKey, message: &[u8]) -> PqcResult<MlDsaSignature> {
        MlDsaCore::sign(sk, message)
    }

    /// Verify signature
    pub fn verify(vk: &MlDsaVerificationKey, message: &[u8], sig: &MlDsaSignature) -> bool {
        MlDsaCore::verify(vk, message, sig)
    }
}

/// ML-DSA-65 (NIST Security Level 3)
pub struct MlDsa65;

impl MlDsa65 {
    /// Parameters
    pub const PARAMS: MlDsaParams = ML_DSA_65_PARAMS;

    /// Generate key pair
    pub fn keygen() -> PqcResult<(MlDsaVerificationKey, MlDsaSigningKey)> {
        MlDsaCore::keygen(Self::PARAMS)
    }

    /// Sign message
    pub fn sign(sk: &MlDsaSigningKey, message: &[u8]) -> PqcResult<MlDsaSignature> {
        MlDsaCore::sign(sk, message)
    }

    /// Verify signature
    pub fn verify(vk: &MlDsaVerificationKey, message: &[u8], sig: &MlDsaSignature) -> bool {
        MlDsaCore::verify(vk, message, sig)
    }
}

/// ML-DSA-87 (NIST Security Level 5)
pub struct MlDsa87;

impl MlDsa87 {
    /// Parameters
    pub const PARAMS: MlDsaParams = ML_DSA_87_PARAMS;

    /// Generate key pair
    pub fn keygen() -> PqcResult<(MlDsaVerificationKey, MlDsaSigningKey)> {
        MlDsaCore::keygen(Self::PARAMS)
    }

    /// Sign message
    pub fn sign(sk: &MlDsaSigningKey, message: &[u8]) -> PqcResult<MlDsaSignature> {
        MlDsaCore::sign(sk, message)
    }

    /// Verify signature
    pub fn verify(vk: &MlDsaVerificationKey, message: &[u8], sig: &MlDsaSignature) -> bool {
        MlDsaCore::verify(vk, message, sig)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_ntt_roundtrip() {
        let mut poly = Polynomial::zero();
        for i in 0..N {
            poly.coeffs[i] = ((i * 7) % Q as usize) as i32;
        }

        let original = poly.clone();
        poly.ntt();
        poly.inv_ntt();

        for i in 0..N {
            let diff = (poly.coeffs[i] - original.coeffs[i]).abs();
            assert!(
                diff < 100,
                "NTT roundtrip error at {}: {} vs {}",
                i,
                poly.coeffs[i],
                original.coeffs[i]
            );
        }
    }

    #[test]
    fn test_ml_dsa_44_sign_verify() {
        let (vk, sk) = MlDsa44::keygen().expect("keygen failed");
        let message = b"Hello, post-quantum world!";

        let sig = MlDsa44::sign(&sk, message).expect("signing failed");
        assert!(MlDsa44::verify(&vk, message, &sig), "verification failed");
    }

    #[test]
    fn test_ml_dsa_65_sign_verify() {
        let (vk, sk) = MlDsa65::keygen().expect("keygen failed");
        let message = b"Testing ML-DSA-65";

        let sig = MlDsa65::sign(&sk, message).expect("signing failed");
        assert!(MlDsa65::verify(&vk, message, &sig), "verification failed");
    }

    #[test]
    fn test_ml_dsa_wrong_message() {
        let (vk, sk) = MlDsa44::keygen().expect("keygen failed");
        let message = b"Original message";
        let wrong_message = b"Wrong message";

        let sig = MlDsa44::sign(&sk, message).expect("signing failed");
        assert!(
            !MlDsa44::verify(&vk, wrong_message, &sig),
            "should reject wrong message"
        );
    }

    #[test]
    fn test_challenge_polynomial() {
        let seed = [0u8; 32];
        let c = Polynomial::sample_challenge(&seed, 39);

        // Count non-zero coefficients
        let non_zero: usize = c.coeffs.iter().filter(|&&x| x != 0).count();
        assert_eq!(
            non_zero, 39,
            "Challenge should have tau non-zero coefficients"
        );

        // All non-zero should be ±1
        for coeff in &c.coeffs {
            assert!(
                *coeff == 0 || *coeff == 1 || *coeff == -1,
                "Challenge coefficients should be 0, 1, or -1"
            );
        }
    }

    #[test]
    fn test_ntt_polynomial_multiplication() {
        // Test: (1+X) * (1+X) = 1 + 2X + X^2
        let mut a = Polynomial::zero();
        a.coeffs[0] = 1;
        a.coeffs[1] = 1;
        let mut b = Polynomial::zero();
        b.coeffs[0] = 1;
        b.coeffs[1] = 1;

        let mut a_ntt = a.clone();
        let mut b_ntt = b.clone();
        a_ntt.ntt();
        b_ntt.ntt();
        let mut c = a_ntt.pointwise_mul(&b_ntt);
        c.inv_ntt();

        assert_eq!(Polynomial::reduce(c.coeffs[0]), 1, "coeff 0 should be 1");
        assert_eq!(Polynomial::reduce(c.coeffs[1]), 2, "coeff 1 should be 2");
        assert_eq!(Polynomial::reduce(c.coeffs[2]), 1, "coeff 2 should be 1");
        for i in 3..N {
            assert_eq!(
                Polynomial::reduce(c.coeffs[i]),
                0,
                "coeff {} should be 0, got {}",
                i,
                c.coeffs[i]
            );
        }
    }

    #[test]
    fn test_ntt_mul_wraparound() {
        // Test: X^255 * X = X^256 = -1 mod (X^256 + 1)
        let mut a = Polynomial::zero();
        a.coeffs[255] = 1;
        let mut b = Polynomial::zero();
        b.coeffs[1] = 1;

        let mut a_ntt = a.clone();
        let mut b_ntt = b.clone();
        a_ntt.ntt();
        b_ntt.ntt();
        let mut c = a_ntt.pointwise_mul(&b_ntt);
        c.inv_ntt();

        let c0 = Polynomial::reduce(c.coeffs[0]);
        assert!(c0 == -1 || c0 == Q - 1, "coeff 0 should be -1, got {}", c0);
        for i in 1..N {
            assert_eq!(
                Polynomial::reduce(c.coeffs[i]),
                0,
                "coeff {} should be 0, got {}",
                i,
                c.coeffs[i]
            );
        }
    }

    #[test]
    fn test_deterministic_keygen() {
        let seed = [42u8; 32];

        let (vk1, sk1) = MlDsaCore::keygen_internal(ML_DSA_44_PARAMS, &seed).unwrap();
        let (vk2, sk2) = MlDsaCore::keygen_internal(ML_DSA_44_PARAMS, &seed).unwrap();

        assert_eq!(vk1.as_bytes(), vk2.as_bytes());
        assert_eq!(sk1.as_bytes(), sk2.as_bytes());
    }
}
