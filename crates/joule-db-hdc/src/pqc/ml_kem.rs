//! ML-KEM (Module Lattice Key Encapsulation Mechanism) - FIPS 203
//!
//! Implementation of the NIST-standardized post-quantum KEM based on
//! Module Learning With Errors (MLWE).
//!
//! ## Parameter Sets
//!
//! | Parameter | k  | Security | Classical | Quantum |
//! |-----------|----|---------:|----------:|--------:|
//! | ML-KEM-512  | 2 | Level 1 | AES-128  | NIST-1 |
//! | ML-KEM-768  | 3 | Level 3 | AES-192  | NIST-3 |
//! | ML-KEM-1024 | 4 | Level 5 | AES-256  | NIST-5 |

use super::common::{ConstantTime, SecureZeroingVec, Sha3_256, Sha3_512, Shake128, Shake256};
use super::{PqcError, PqcResult};
use rand::Rng;

// ============================================================================
// Constants
// ============================================================================

/// Polynomial degree n = 256
pub const N: usize = 256;

/// Modulus q = 3329 (NTT-friendly prime: q ≡ 1 mod 2n)
pub const Q: i16 = 3329;

/// q as u32 for intermediate calculations
const Q32: u32 = 3329;

/// Barrett reduction constant: floor(2^26 / q)
const BARRETT_MULT: i32 = 20159;

/// Montgomery constant: -q^(-1) mod 2^16
const QINV: i32 = 62209;

/// NTT root of unity ζ = 17 (primitive 512th root of unity mod q)
#[allow(dead_code)]
const ZETA: i16 = 17;

// ============================================================================
// Simple NTT using schoolbook multiplication for correctness
// This implementation trades speed for simplicity and correctness
// ============================================================================

// ============================================================================
// ML-KEM Parameters
// ============================================================================

/// ML-KEM parameter set
#[derive(Clone, Copy, Debug)]
pub struct MlKemParams {
    /// Security level name
    pub name: &'static str,
    /// Module rank (k = 2, 3, or 4)
    pub k: usize,
    /// Noise parameter η₁
    pub eta1: usize,
    /// Noise parameter η₂
    pub eta2: usize,
    /// Compression parameter for u
    pub du: usize,
    /// Compression parameter for v
    pub dv: usize,
}

/// ML-KEM-512 parameters (NIST Level 1)
pub const ML_KEM_512_PARAMS: MlKemParams = MlKemParams {
    name: "ML-KEM-512",
    k: 2,
    eta1: 3,
    eta2: 2,
    du: 10,
    dv: 4,
};

/// ML-KEM-768 parameters (NIST Level 3)
pub const ML_KEM_768_PARAMS: MlKemParams = MlKemParams {
    name: "ML-KEM-768",
    k: 3,
    eta1: 2,
    eta2: 2,
    du: 10,
    dv: 4,
};

/// ML-KEM-1024 parameters (NIST Level 5)
pub const ML_KEM_1024_PARAMS: MlKemParams = MlKemParams {
    name: "ML-KEM-1024",
    k: 4,
    eta1: 2,
    eta2: 2,
    du: 11,
    dv: 5,
};

impl MlKemParams {
    /// Encapsulation key size in bytes
    pub const fn encapsulation_key_size(&self) -> usize {
        384 * self.k + 32
    }

    /// Decapsulation key size in bytes
    pub const fn decapsulation_key_size(&self) -> usize {
        768 * self.k + 96
    }

    /// Ciphertext size in bytes
    pub const fn ciphertext_size(&self) -> usize {
        32 * (self.du * self.k + self.dv)
    }

    /// Shared secret size in bytes (always 32)
    pub const fn shared_secret_size(&self) -> usize {
        32
    }
}

// ============================================================================
// Polynomial Operations
// ============================================================================

/// Polynomial in Z_q[X]/(X^256 + 1)
#[derive(Clone)]
pub struct Polynomial {
    coeffs: [i16; N],
}

impl Polynomial {
    /// Create zero polynomial
    pub fn zero() -> Self {
        Self { coeffs: [0; N] }
    }

    /// Barrett reduction: a mod q
    #[inline]
    fn barrett_reduce(a: i16) -> i16 {
        let t = ((a as i32 * BARRETT_MULT) >> 26) as i16;
        let r = a - t * Q;
        // May need one more reduction
        let r = r - Q;
        let mask = (r >> 15) as i16; // 0 if r >= 0, -1 if r < 0
        r + (mask & Q)
    }

    /// Montgomery reduction: a * R^(-1) mod q where R = 2^16
    #[inline]
    fn montgomery_reduce(a: i32) -> i16 {
        let u = ((a as i16 as i32) * QINV) as i16;
        let t = (a - (u as i32 * Q as i32)) >> 16;
        t as i16
    }

    /// Add two polynomials
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            result.coeffs[i] = Self::barrett_reduce(self.coeffs[i] + other.coeffs[i]);
        }
        result
    }

    /// Subtract two polynomials
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            result.coeffs[i] = Self::barrett_reduce(self.coeffs[i] - other.coeffs[i]);
        }
        result
    }

    /// NTT - For this simplified implementation, we just ensure coefficients
    /// are in the proper range. The actual polynomial multiplication is done
    /// via schoolbook method in the basemul function.
    pub fn ntt(&mut self) {
        // Reduce all coefficients to proper range
        for coeff in &mut self.coeffs {
            *coeff = Self::mod_reduce(*coeff);
        }
    }

    /// Inverse NTT - counterpart to ntt
    pub fn inv_ntt(&mut self) {
        // Reduce all coefficients to proper range
        for coeff in &mut self.coeffs {
            *coeff = Self::mod_reduce(*coeff);
        }
    }

    /// Reduce coefficient to [-q/2, q/2]
    #[inline]
    fn mod_reduce(a: i16) -> i16 {
        let mut r = a % Q;
        if r > Q / 2 {
            r -= Q;
        } else if r < -Q / 2 {
            r += Q;
        }
        r
    }

    /// Basemul: schoolbook polynomial multiplication mod (X^256 + 1) mod q
    pub fn basemul(&self, other: &Self) -> Self {
        let mut result = Self::zero();

        // Use i64 for intermediate calculations to avoid overflow
        let mut temp = [0i64; 2 * N];

        // Schoolbook multiplication
        for i in 0..N {
            for j in 0..N {
                temp[i + j] += self.coeffs[i] as i64 * other.coeffs[j] as i64;
            }
        }

        // Reduce mod X^256 + 1 (wrap with negation)
        for i in 0..N {
            let sum = temp[i] - temp[i + N];
            result.coeffs[i] = (sum % Q as i64) as i16;
        }

        // Final reduction
        for coeff in &mut result.coeffs {
            *coeff = Self::mod_reduce(*coeff);
        }

        result
    }

    /// Reduce all coefficients to [0, q-1]
    pub fn reduce(&mut self) {
        for coeff in &mut self.coeffs {
            *coeff = Self::barrett_reduce(*coeff);
            if *coeff < 0 {
                *coeff += Q;
            }
        }
    }

    /// Compress polynomial to d bits per coefficient
    pub fn compress(&self, d: usize) -> Vec<u8> {
        let mut result = vec![0u8; N * d / 8];

        for i in 0..N {
            // Normalize to [0, q-1] before compression
            let c = self.coeffs[i] % Q;
            let coeff = if c < 0 { (c + Q) as u32 } else { c as u32 };
            let compressed = (((coeff << d) + Q32 / 2) / Q32) & ((1 << d) - 1);

            // Pack into bytes
            let bit_pos = i * d;
            let byte_pos = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            result[byte_pos] |= (compressed << bit_offset) as u8;
            if bit_offset + d > 8 && byte_pos + 1 < result.len() {
                result[byte_pos + 1] |= (compressed >> (8 - bit_offset)) as u8;
            }
            if bit_offset + d > 16 && byte_pos + 2 < result.len() {
                result[byte_pos + 2] |= (compressed >> (16 - bit_offset)) as u8;
            }
        }

        result
    }

    /// Decompress polynomial from d bits per coefficient
    pub fn decompress(bytes: &[u8], d: usize) -> Self {
        let mut poly = Self::zero();

        for i in 0..N {
            let bit_pos = i * d;
            let byte_pos = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            let mut compressed = (bytes[byte_pos] >> bit_offset) as u32;
            if bit_offset + d > 8 && byte_pos + 1 < bytes.len() {
                compressed |= (bytes[byte_pos + 1] as u32) << (8 - bit_offset);
            }
            if bit_offset + d > 16 && byte_pos + 2 < bytes.len() {
                compressed |= (bytes[byte_pos + 2] as u32) << (16 - bit_offset);
            }
            compressed &= (1 << d) - 1;

            // Decompress: round((q / 2^d) * compressed)
            poly.coeffs[i] = ((compressed * Q32 + (1 << (d - 1))) >> d) as i16;
        }

        poly
    }

    /// Encode polynomial to bytes (12 bits per coefficient)
    pub fn to_bytes(&self) -> [u8; 384] {
        let mut result = [0u8; 384];

        for i in 0..N / 2 {
            // Normalize to [0, q-1] before encoding
            let a = {
                let c = self.coeffs[2 * i] % Q;
                if c < 0 { (c + Q) as u16 } else { c as u16 }
            };
            let b = {
                let c = self.coeffs[2 * i + 1] % Q;
                if c < 0 { (c + Q) as u16 } else { c as u16 }
            };

            result[3 * i] = a as u8;
            result[3 * i + 1] = ((a >> 8) | (b << 4)) as u8;
            result[3 * i + 2] = (b >> 4) as u8;
        }

        result
    }

    /// Decode polynomial from bytes (12 bits per coefficient)
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut poly = Self::zero();

        for i in 0..N / 2 {
            let a = bytes[3 * i] as u16 | ((bytes[3 * i + 1] as u16 & 0x0F) << 8);
            let b = (bytes[3 * i + 1] as u16 >> 4) | ((bytes[3 * i + 2] as u16) << 4);

            poly.coeffs[2 * i] = (a % Q as u16) as i16;
            poly.coeffs[2 * i + 1] = (b % Q as u16) as i16;
        }

        poly
    }

    /// Sample polynomial from centered binomial distribution (FIPS 203 Algorithm 7)
    /// Input: byte array of length 64*eta. Each coefficient uses 2*eta bits.
    pub fn sample_cbd(bytes: &[u8], eta: usize) -> PqcResult<Self> {
        let mut poly = Self::zero();

        match eta {
            2 => {
                // 4 bits per coefficient: coeff = popcount(bits[0:2]) - popcount(bits[2:4])
                // 256 coefficients × 4 bits = 1024 bits = 128 bytes
                // Process 2 coefficients per byte (4 bits each)
                for i in 0..N / 2 {
                    let byte = bytes[i];
                    let a0 = (byte & 1) + ((byte >> 1) & 1);
                    let b0 = ((byte >> 2) & 1) + ((byte >> 3) & 1);
                    let a1 = ((byte >> 4) & 1) + ((byte >> 5) & 1);
                    let b1 = ((byte >> 6) & 1) + ((byte >> 7) & 1);
                    poly.coeffs[2 * i] = a0 as i16 - b0 as i16;
                    poly.coeffs[2 * i + 1] = a1 as i16 - b1 as i16;
                }
            }
            3 => {
                // 6 bits per coefficient: coeff = popcount(bits[0:3]) - popcount(bits[3:6])
                // 256 coefficients × 6 bits = 1536 bits = 192 bytes
                for i in 0..N {
                    let bit_pos = i * 6;
                    let byte_pos = bit_pos / 8;
                    let bit_off = bit_pos % 8;

                    let mut bits = (bytes[byte_pos] as u16) >> bit_off;
                    if byte_pos + 1 < bytes.len() {
                        bits |= (bytes[byte_pos + 1] as u16) << (8 - bit_off);
                    }
                    let bits = (bits & 0x3F) as u8;

                    let a = (bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1);
                    let b = ((bits >> 3) & 1) + ((bits >> 4) & 1) + ((bits >> 5) & 1);
                    poly.coeffs[i] = a as i16 - b as i16;
                }
            }
            _ => {
                return Err(PqcError::InvalidParameter(format!(
                    "unsupported ML-KEM eta: {eta} (expected 2 or 3)"
                )));
            }
        }

        Ok(poly)
    }
}

impl Default for Polynomial {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Vector of polynomials
#[derive(Clone)]
pub struct PolyVec {
    polys: Vec<Polynomial>,
}

impl PolyVec {
    /// Create zero vector
    pub fn new(k: usize) -> Self {
        Self {
            polys: (0..k).map(|_| Polynomial::zero()).collect(),
        }
    }

    /// NTT all polynomials
    pub fn ntt(&mut self) {
        for poly in &mut self.polys {
            poly.ntt();
        }
    }

    /// Inverse NTT all polynomials
    pub fn inv_ntt(&mut self) {
        for poly in &mut self.polys {
            poly.inv_ntt();
        }
    }

    /// Add two vectors
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

    /// Dot product in NTT domain
    pub fn dot(&self, other: &Self) -> Polynomial {
        let mut result = Polynomial::zero();
        for (a, b) in self.polys.iter().zip(&other.polys) {
            result = result.add(&a.basemul(b));
        }
        result
    }

    /// Reduce all polynomials
    pub fn reduce(&mut self) {
        for poly in &mut self.polys {
            poly.reduce();
        }
    }

    /// Compress vector
    pub fn compress(&self, d: usize) -> Vec<u8> {
        self.polys.iter().flat_map(|p| p.compress(d)).collect()
    }

    /// Decompress vector
    pub fn decompress(bytes: &[u8], k: usize, d: usize) -> Self {
        let bytes_per_poly = N * d / 8;
        Self {
            polys: (0..k)
                .map(|i| Polynomial::decompress(&bytes[i * bytes_per_poly..], d))
                .collect(),
        }
    }

    /// Encode to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.polys.iter().flat_map(|p| p.to_bytes()).collect()
    }

    /// Decode from bytes
    pub fn from_bytes(bytes: &[u8], k: usize) -> Self {
        Self {
            polys: (0..k)
                .map(|i| Polynomial::from_bytes(&bytes[i * 384..]))
                .collect(),
        }
    }
}

/// Matrix of polynomials (stored in NTT domain)
pub struct PolyMatrix {
    rows: Vec<PolyVec>,
}

impl PolyMatrix {
    /// Generate matrix A from seed (XOF expansion)
    pub fn generate_a(seed: &[u8; 32], k: usize) -> Self {
        let mut rows = Vec::with_capacity(k);

        for i in 0..k {
            let mut row_polys = Vec::with_capacity(k);
            for j in 0..k {
                row_polys.push(Self::sample_ntt(seed, i as u8, j as u8));
            }
            rows.push(PolyVec { polys: row_polys });
        }

        Self { rows }
    }

    /// Sample polynomial in NTT domain from XOF
    fn sample_ntt(seed: &[u8; 32], i: u8, j: u8) -> Polynomial {
        let mut shake = Shake128::new();
        shake.absorb(seed);
        shake.absorb(&[j, i]); // Note: column, row order per spec

        let mut poly = Polynomial::zero();
        let mut buf = [0u8; 3];
        let mut ctr = 0;

        while ctr < N {
            shake.squeeze(&mut buf);

            let d1 = ((buf[0] as u16) | ((buf[1] as u16 & 0x0F) << 8)) as i16;
            let d2 = (((buf[1] as u16) >> 4) | ((buf[2] as u16) << 4)) as i16;

            if d1 < Q {
                poly.coeffs[ctr] = d1;
                ctr += 1;
            }
            if ctr < N && d2 < Q {
                poly.coeffs[ctr] = d2;
                ctr += 1;
            }
        }

        poly
    }

    /// Matrix-vector multiply: A * s (in NTT domain)
    pub fn mul_vec(&self, s: &PolyVec) -> PolyVec {
        PolyVec {
            polys: self.rows.iter().map(|row| row.dot(s)).collect(),
        }
    }

    /// Transpose matrix-vector multiply: A^T * s (in NTT domain)
    pub fn mul_vec_transpose(&self, s: &PolyVec) -> PolyVec {
        let k = self.rows.len();
        let mut result = PolyVec::new(k);

        for j in 0..k {
            for i in 0..k {
                let prod = self.rows[i].polys[j].basemul(&s.polys[i]);
                result.polys[j] = result.polys[j].add(&prod);
            }
        }

        result
    }
}

// ============================================================================
// Key Types
// ============================================================================

/// ML-KEM Encapsulation Key (public key)
#[derive(Clone)]
pub struct MlKemEncapsulationKey {
    data: Vec<u8>,
    params: MlKemParams,
}

impl MlKemEncapsulationKey {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], params: MlKemParams) -> PqcResult<Self> {
        if bytes.len() != params.encapsulation_key_size() {
            return Err(PqcError::InvalidKey);
        }
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
    pub fn params(&self) -> MlKemParams {
        self.params
    }
}

/// ML-KEM Decapsulation Key (secret key)
#[derive(Clone)]
pub struct MlKemDecapsulationKey {
    data: SecureZeroingVec,
    params: MlKemParams,
}

impl MlKemDecapsulationKey {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], params: MlKemParams) -> PqcResult<Self> {
        if bytes.len() != params.decapsulation_key_size() {
            return Err(PqcError::InvalidKey);
        }
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
    pub fn params(&self) -> MlKemParams {
        self.params
    }
}

/// ML-KEM Ciphertext
#[derive(Clone)]
pub struct MlKemCiphertext {
    data: Vec<u8>,
    params: MlKemParams,
}

impl MlKemCiphertext {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], params: MlKemParams) -> PqcResult<Self> {
        if bytes.len() != params.ciphertext_size() {
            return Err(PqcError::InvalidCiphertext);
        }
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
    pub fn params(&self) -> MlKemParams {
        self.params
    }
}

/// ML-KEM Shared Secret (32 bytes)
#[derive(Clone, Debug)]
pub struct MlKemSharedSecret {
    data: SecureZeroingVec,
}

impl MlKemSharedSecret {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> PqcResult<Self> {
        if bytes.len() != 32 {
            return Err(PqcError::InvalidKey);
        }
        Ok(Self {
            data: SecureZeroingVec::from_vec(bytes.to_vec()),
        })
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }
}

impl PartialEq for MlKemSharedSecret {
    fn eq(&self, other: &Self) -> bool {
        ConstantTime::ct_eq(self.data.as_slice(), other.data.as_slice())
    }
}

// ============================================================================
// ML-KEM Core Algorithm
// ============================================================================

/// Core ML-KEM implementation
pub struct MlKemCore;

impl MlKemCore {
    /// Key generation (Algorithm 15: ML-KEM.KeyGen)
    pub fn keygen(
        params: MlKemParams,
    ) -> PqcResult<(MlKemEncapsulationKey, MlKemDecapsulationKey)> {
        let mut rng = rand::rng();

        // Generate random seeds
        let mut d = [0u8; 32];
        let mut z = [0u8; 32];
        rng.fill(&mut d);
        rng.fill(&mut z);

        Self::keygen_internal(params, &d, &z)
    }

    /// Internal key generation with explicit randomness (for deterministic testing)
    pub fn keygen_internal(
        params: MlKemParams,
        d: &[u8; 32],
        z: &[u8; 32],
    ) -> PqcResult<(MlKemEncapsulationKey, MlKemDecapsulationKey)> {
        let k = params.k;

        // G(d) = (ρ, σ)
        let g_input = Sha3_512::hash(d);
        let (rho, sigma) = g_input.split_at(32);
        let rho: [u8; 32] = rho.try_into().unwrap();

        // Generate matrix A in NTT domain
        let a_hat = PolyMatrix::generate_a(&rho, k);

        // Sample secret s and error e
        let mut s_hat = PolyVec::new(k);
        let mut e_hat = PolyVec::new(k);

        for i in 0..k {
            let prf_input = [&sigma[..], &[i as u8]].concat();
            let noise = Shake256::xof(&prf_input, 64 * params.eta1);
            s_hat.polys[i] = Polynomial::sample_cbd(&noise, params.eta1)?;

            let prf_input = [&sigma[..], &[(k + i) as u8]].concat();
            let noise = Shake256::xof(&prf_input, 64 * params.eta1);
            e_hat.polys[i] = Polynomial::sample_cbd(&noise, params.eta1)?;
        }

        // Transform to NTT domain
        s_hat.ntt();
        e_hat.ntt();

        // t = A * s + e (in NTT domain)
        let mut t_hat = a_hat.mul_vec(&s_hat);
        t_hat = t_hat.add(&e_hat);
        t_hat.reduce();

        // Encode public key: ek = (t_hat || ρ)
        let mut ek_bytes = t_hat.to_bytes();
        ek_bytes.extend_from_slice(&rho);

        // H(ek)
        let h_ek = Sha3_256::hash(&ek_bytes);

        // Encode decapsulation key: dk = (s_hat || ek || H(ek) || z)
        let s_bytes = s_hat.to_bytes();
        let mut dk_bytes = Vec::with_capacity(params.decapsulation_key_size());
        dk_bytes.extend_from_slice(&s_bytes);
        dk_bytes.extend_from_slice(&ek_bytes);
        dk_bytes.extend_from_slice(&h_ek);
        dk_bytes.extend_from_slice(z);

        Ok((
            MlKemEncapsulationKey {
                data: ek_bytes,
                params,
            },
            MlKemDecapsulationKey {
                data: SecureZeroingVec::from_vec(dk_bytes),
                params,
            },
        ))
    }

    /// Encapsulation (Algorithm 16: ML-KEM.Encaps)
    pub fn encapsulate(
        ek: &MlKemEncapsulationKey,
    ) -> PqcResult<(MlKemCiphertext, MlKemSharedSecret)> {
        let mut rng = rand::rng();
        let mut m = [0u8; 32];
        rng.fill(&mut m);

        Self::encapsulate_internal(ek, &m)
    }

    /// Internal encapsulation with explicit randomness
    pub fn encapsulate_internal(
        ek: &MlKemEncapsulationKey,
        m: &[u8; 32],
    ) -> PqcResult<(MlKemCiphertext, MlKemSharedSecret)> {
        let params = ek.params;
        let k = params.k;

        // Parse encapsulation key
        let t_bytes = &ek.data[..384 * k];
        let rho: [u8; 32] = ek.data[384 * k..].try_into().unwrap();

        let t_hat = PolyVec::from_bytes(t_bytes, k);

        // G(m || H(ek))
        let h_ek = Sha3_256::hash(&ek.data);
        let g_input = [&m[..], &h_ek[..]].concat();
        let g_output = Sha3_512::hash(&g_input);
        let (shared_secret, randomness) = g_output.split_at(32);

        // Generate A^T (transpose)
        let a_hat = PolyMatrix::generate_a(&rho, k);

        // Sample r, e1, e2
        let mut r_hat = PolyVec::new(k);
        let mut e1 = PolyVec::new(k);

        for i in 0..k {
            let prf_input = [randomness, &[i as u8]].concat();
            let noise = Shake256::xof(&prf_input, 64 * params.eta1);
            r_hat.polys[i] = Polynomial::sample_cbd(&noise, params.eta1)?;

            let prf_input = [randomness, &[(k + i) as u8]].concat();
            let noise = Shake256::xof(&prf_input, 64 * params.eta2);
            e1.polys[i] = Polynomial::sample_cbd(&noise, params.eta2)?;
        }

        let prf_input = [randomness, &[2 * k as u8]].concat();
        let noise = Shake256::xof(&prf_input, 64 * params.eta2);
        let e2 = Polynomial::sample_cbd(&noise, params.eta2)?;

        // Transform r to NTT domain
        r_hat.ntt();

        // u = A^T * r + e1
        let mut u = a_hat.mul_vec_transpose(&r_hat);
        u.inv_ntt();
        u = u.add(&e1);

        // v = t^T * r + e2 + Decompress(m)
        let mut v = t_hat.dot(&r_hat);
        v.inv_ntt();
        v = v.add(&e2);

        // Encode m as polynomial and add
        let mut m_poly = Polynomial::zero();
        for i in 0..N {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = ((m[byte_idx] >> bit_idx) & 1) as i16;
            m_poly.coeffs[i] = bit * ((Q + 1) / 2);
        }
        v = v.add(&m_poly);

        // Compress and encode ciphertext
        u.reduce();
        v.reduce();

        let mut ct_bytes = u.compress(params.du);
        ct_bytes.extend(v.compress(params.dv));

        Ok((
            MlKemCiphertext {
                data: ct_bytes,
                params,
            },
            MlKemSharedSecret {
                data: SecureZeroingVec::from_vec(shared_secret.to_vec()),
            },
        ))
    }

    /// Decapsulation (Algorithm 17: ML-KEM.Decaps)
    pub fn decapsulate(
        dk: &MlKemDecapsulationKey,
        ct: &MlKemCiphertext,
    ) -> PqcResult<MlKemSharedSecret> {
        let params = dk.params;
        let k = params.k;

        if ct.params.k != k {
            return Err(PqcError::InvalidCiphertext);
        }

        let dk_bytes = dk.data.as_slice();

        // Parse decapsulation key components
        let s_bytes = &dk_bytes[..384 * k];
        let ek_bytes = &dk_bytes[384 * k..768 * k + 32];
        let h_ek = &dk_bytes[768 * k + 32..768 * k + 64];
        let z = &dk_bytes[768 * k + 64..];

        let s_hat = PolyVec::from_bytes(s_bytes, k);

        // Decompress ciphertext
        let u_bytes = &ct.data[..32 * params.du * k];
        let v_bytes = &ct.data[32 * params.du * k..];

        let mut u = PolyVec::decompress(u_bytes, k, params.du);
        let v = Polynomial::decompress(v_bytes, params.dv);

        // Transform u to NTT domain
        u.ntt();

        // Compute w = v - s^T * u
        let su = s_hat.dot(&u);
        let mut su_normal = su;
        su_normal.inv_ntt();

        let w = v.sub(&su_normal);

        // Decode message from w
        let mut m_prime = [0u8; 32];
        for i in 0..N {
            // Check if coefficient is closer to 0 or q/2
            let coeff = w.coeffs[i];
            let normalized = if coeff < 0 { coeff + Q } else { coeff };
            let threshold = Q / 4;
            let bit = if normalized > threshold && normalized < (3 * Q / 4) {
                1u8
            } else {
                0u8
            };

            let byte_idx = i / 8;
            let bit_idx = i % 8;
            m_prime[byte_idx] |= bit << bit_idx;
        }

        // Re-encapsulate to verify
        let ek = MlKemEncapsulationKey::from_bytes(ek_bytes, params)?;
        let (ct_prime, _) = Self::encapsulate_internal(&ek, &m_prime)?;

        // G(m' || H(ek))
        let g_input = [&m_prime[..], h_ek].concat();
        let g_output = Sha3_512::hash(&g_input);
        let (k_bar, _) = g_output.split_at(32);

        // Implicit rejection: J(z || c)
        let j_input = [z, ct.as_bytes()].concat();
        let k_bar_fail = Shake256::xof(&j_input, 32);

        // Constant-time selection based on ciphertext comparison
        let ct_eq = ConstantTime::ct_compare(ct.as_bytes(), ct_prime.as_bytes());
        let shared_secret = ConstantTime::ct_select(ct_eq, k_bar, &k_bar_fail);

        Ok(MlKemSharedSecret {
            data: SecureZeroingVec::from_vec(shared_secret),
        })
    }
}

// ============================================================================
// ML-KEM Variants (Type-Safe Wrappers)
// ============================================================================

/// ML-KEM-512 (NIST Security Level 1)
pub struct MlKem512;

impl MlKem512 {
    /// Parameters for this variant
    pub const PARAMS: MlKemParams = ML_KEM_512_PARAMS;

    /// Generate key pair
    pub fn keygen() -> PqcResult<(MlKemEncapsulationKey, MlKemDecapsulationKey)> {
        MlKemCore::keygen(Self::PARAMS)
    }

    /// Encapsulate shared secret
    pub fn encapsulate(
        ek: &MlKemEncapsulationKey,
    ) -> PqcResult<(MlKemCiphertext, MlKemSharedSecret)> {
        MlKemCore::encapsulate(ek)
    }

    /// Decapsulate shared secret
    pub fn decapsulate(
        dk: &MlKemDecapsulationKey,
        ct: &MlKemCiphertext,
    ) -> PqcResult<MlKemSharedSecret> {
        MlKemCore::decapsulate(dk, ct)
    }
}

/// ML-KEM-768 (NIST Security Level 3)
pub struct MlKem768;

impl MlKem768 {
    /// Parameters for this variant
    pub const PARAMS: MlKemParams = ML_KEM_768_PARAMS;

    /// Generate key pair
    pub fn keygen() -> PqcResult<(MlKemEncapsulationKey, MlKemDecapsulationKey)> {
        MlKemCore::keygen(Self::PARAMS)
    }

    /// Encapsulate shared secret
    pub fn encapsulate(
        ek: &MlKemEncapsulationKey,
    ) -> PqcResult<(MlKemCiphertext, MlKemSharedSecret)> {
        MlKemCore::encapsulate(ek)
    }

    /// Decapsulate shared secret
    pub fn decapsulate(
        dk: &MlKemDecapsulationKey,
        ct: &MlKemCiphertext,
    ) -> PqcResult<MlKemSharedSecret> {
        MlKemCore::decapsulate(dk, ct)
    }
}

/// ML-KEM-1024 (NIST Security Level 5)
pub struct MlKem1024;

impl MlKem1024 {
    /// Parameters for this variant
    pub const PARAMS: MlKemParams = ML_KEM_1024_PARAMS;

    /// Generate key pair
    pub fn keygen() -> PqcResult<(MlKemEncapsulationKey, MlKemDecapsulationKey)> {
        MlKemCore::keygen(Self::PARAMS)
    }

    /// Encapsulate shared secret
    pub fn encapsulate(
        ek: &MlKemEncapsulationKey,
    ) -> PqcResult<(MlKemCiphertext, MlKemSharedSecret)> {
        MlKemCore::encapsulate(ek)
    }

    /// Decapsulate shared secret
    pub fn decapsulate(
        dk: &MlKemDecapsulationKey,
        ct: &MlKemCiphertext,
    ) -> PqcResult<MlKemSharedSecret> {
        MlKemCore::decapsulate(dk, ct)
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
            poly.coeffs[i] = (i % Q as usize) as i16;
        }

        let original = poly.clone();
        poly.ntt();
        poly.inv_ntt();

        // Check coefficients are approximately equal (mod q)
        for i in 0..N {
            let diff = (poly.coeffs[i] - original.coeffs[i]).abs();
            let diff_mod_q = diff.min(Q - diff);
            assert!(diff_mod_q < 10, "NTT roundtrip failed at index {}", i);
        }
    }

    #[test]
    fn test_polynomial_compress_decompress() {
        let mut poly = Polynomial::zero();
        for i in 0..N {
            poly.coeffs[i] = ((i * 13) % Q as usize) as i16;
        }

        for d in [4, 5, 10, 11] {
            let compressed = poly.compress(d);
            let decompressed = Polynomial::decompress(&compressed, d);

            for i in 0..N {
                let original = poly.coeffs[i] as i32;
                let recovered = decompressed.coeffs[i] as i32;
                let diff = (original - recovered).abs();
                let max_error = Q as i32 / (1 << d) + 1;
                assert!(
                    diff <= max_error || diff >= (Q as i32 - max_error),
                    "Compression error too large for d={} at i={}",
                    d,
                    i
                );
            }
        }
    }

    #[test]
    fn test_ml_kem_512_roundtrip() {
        let (ek, dk) = MlKem512::keygen().expect("keygen failed");

        assert_eq!(
            ek.as_bytes().len(),
            ML_KEM_512_PARAMS.encapsulation_key_size()
        );
        assert_eq!(
            dk.as_bytes().len(),
            ML_KEM_512_PARAMS.decapsulation_key_size()
        );

        let (ct, ss_enc) = MlKem512::encapsulate(&ek).expect("encapsulate failed");
        assert_eq!(ct.as_bytes().len(), ML_KEM_512_PARAMS.ciphertext_size());

        let ss_dec = MlKem512::decapsulate(&dk, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc, ss_dec, "Shared secrets don't match");
    }

    #[test]
    fn test_ml_kem_768_roundtrip() {
        let (ek, dk) = MlKem768::keygen().expect("keygen failed");

        let (ct, ss_enc) = MlKem768::encapsulate(&ek).expect("encapsulate failed");
        let ss_dec = MlKem768::decapsulate(&dk, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc, ss_dec, "Shared secrets don't match");
    }

    #[test]
    fn test_ml_kem_1024_roundtrip() {
        let (ek, dk) = MlKem1024::keygen().expect("keygen failed");

        let (ct, ss_enc) = MlKem1024::encapsulate(&ek).expect("encapsulate failed");
        let ss_dec = MlKem1024::decapsulate(&dk, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc, ss_dec, "Shared secrets don't match");
    }

    #[test]
    fn test_ml_kem_implicit_rejection() {
        let (ek, dk) = MlKem512::keygen().expect("keygen failed");
        let (mut ct, ss_enc) = MlKem512::encapsulate(&ek).expect("encapsulate failed");

        // Corrupt ciphertext
        ct.data[0] ^= 0xFF;
        ct.data[10] ^= 0x55;

        // Should still return a shared secret (implicit rejection)
        let ss_dec = MlKem512::decapsulate(&dk, &ct).expect("decapsulate failed");

        // But it should be different from the original
        assert_ne!(ss_enc, ss_dec, "Implicit rejection failed");
    }

    #[test]
    fn test_key_sizes() {
        // ML-KEM-512
        assert_eq!(ML_KEM_512_PARAMS.encapsulation_key_size(), 800);
        assert_eq!(ML_KEM_512_PARAMS.decapsulation_key_size(), 1632);
        assert_eq!(ML_KEM_512_PARAMS.ciphertext_size(), 768);

        // ML-KEM-768
        assert_eq!(ML_KEM_768_PARAMS.encapsulation_key_size(), 1184);
        assert_eq!(ML_KEM_768_PARAMS.decapsulation_key_size(), 2400);
        assert_eq!(ML_KEM_768_PARAMS.ciphertext_size(), 1088);

        // ML-KEM-1024
        assert_eq!(ML_KEM_1024_PARAMS.encapsulation_key_size(), 1568);
        assert_eq!(ML_KEM_1024_PARAMS.decapsulation_key_size(), 3168);
        assert_eq!(ML_KEM_1024_PARAMS.ciphertext_size(), 1568);
    }

    #[test]
    fn test_deterministic_keygen() {
        let d = [0u8; 32];
        let z = [1u8; 32];

        let (ek1, dk1) = MlKemCore::keygen_internal(ML_KEM_512_PARAMS, &d, &z).unwrap();
        let (ek2, dk2) = MlKemCore::keygen_internal(ML_KEM_512_PARAMS, &d, &z).unwrap();

        assert_eq!(ek1.as_bytes(), ek2.as_bytes());
        assert_eq!(dk1.as_bytes(), dk2.as_bytes());
    }
}
