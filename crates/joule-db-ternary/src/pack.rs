//! Packed trit storage: 5 trits per byte (3^5 = 243 ≤ 255).
//!
//! LUT-based decode avoids div/mod at runtime. NEON SIMD dot product
//! on aarch64 for hardware-accelerated similarity computation.

use crate::Trit;

// ============================================================================
// LUT Tables
// ============================================================================

/// Decode table: byte → [trit0, trit1, trit2, trit3, trit4] as i8.
const DECODE_LUT: [[i8; 5]; 243] = {
    let mut table = [[0i8; 5]; 243];
    let mut i = 0u16;
    while i < 243 {
        let mut val = i;
        let mut j = 0;
        while j < 5 {
            table[i as usize][j] = (val % 3) as i8 - 1;
            val /= 3;
            j += 1;
        }
        i += 1;
    }
    table
};

/// Encode table: [t0+1][t1+1][t2+1][t3+1][t4+1] → byte.
const ENCODE_LUT: [[[[[u8; 3]; 3]; 3]; 3]; 3] = {
    let mut table = [[[[[0u8; 3]; 3]; 3]; 3]; 3];
    let mut a = 0usize;
    while a < 3 {
        let mut b = 0usize;
        while b < 3 {
            let mut c = 0usize;
            while c < 3 {
                let mut d = 0usize;
                while d < 3 {
                    let mut e = 0usize;
                    while e < 3 {
                        table[a][b][c][d][e] = (a + b * 3 + c * 9 + d * 27 + e * 81) as u8;
                        e += 1;
                    }
                    d += 1;
                }
                c += 1;
            }
            b += 1;
        }
        a += 1;
    }
    table
};

/// Dot product LUT: byte_a × byte_b → sum of element-wise products.
const DOT_LUT: [[i8; 243]; 243] = {
    let mut table = [[0i8; 243]; 243];
    let mut a = 0usize;
    while a < 243 {
        let mut b = 0usize;
        while b < 243 {
            let da = DECODE_LUT[a];
            let db = DECODE_LUT[b];
            let mut sum = 0i8;
            let mut k = 0;
            while k < 5 {
                sum += da[k] * db[k];
                k += 1;
            }
            table[a][b] = sum;
            b += 1;
        }
        a += 1;
    }
    table
};

// ============================================================================
// xorshift64* PRNG (matches BinaryHV pattern — no rand dependency)
// ============================================================================

/// Deterministic xorshift64* PRNG for reproducible random vectors.
#[inline]
fn xorshift64star(state: &mut u64) -> u64 {
    let mut s = *state;
    if s == 0 {
        s = 0xDEADBEEF;
    }
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    *state = s;
    s.wrapping_mul(0x2545F4914F6CDD1D)
}

// ============================================================================
// PackedTrits
// ============================================================================

/// A vector of balanced ternary digits packed at 5 trits per byte.
///
/// Storage: ceil(len/5) bytes. For a 10,000-dimensional hypervector,
/// this is 2,000 bytes — compact enough for in-memory HDC operations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PackedTrits {
    bytes: Vec<u8>,
    len: usize,
}

impl PackedTrits {
    /// Pack a slice of i8 values (-1, 0, +1) into packed trits.
    pub fn from_i8(trits: &[i8]) -> Self {
        let num_bytes = (trits.len() + 4) / 5;
        let mut bytes = Vec::with_capacity(num_bytes);
        let mut i = 0;
        while i < trits.len() {
            let remaining = trits.len() - i;
            let count = remaining.min(5);
            let mut vals = [1usize; 5]; // default 0 → index 1
            for j in 0..count {
                vals[j] = (trits[i + j] + 1) as usize;
            }
            bytes.push(ENCODE_LUT[vals[0]][vals[1]][vals[2]][vals[3]][vals[4]]);
            i += 5;
        }
        Self {
            bytes,
            len: trits.len(),
        }
    }

    /// Create a zero vector (all trits = 0).
    pub fn zeros(len: usize) -> Self {
        let num_bytes = (len + 4) / 5;
        // 0+1=1 for all 5 positions → ENCODE_LUT[1][1][1][1][1] = 1+3+9+27+81 = 121
        let bytes = vec![121u8; num_bytes];
        Self { bytes, len }
    }

    /// Create a random ternary vector with target sparsity (fraction of zeros).
    ///
    /// Uses deterministic xorshift64* seeded from dimension and sparsity.
    pub fn random(len: usize, sparsity: f64, seed: u64) -> Self {
        let mut state = seed;
        if state == 0 {
            state = 0xCAFEBABE;
        }
        let mut trits = Vec::with_capacity(len);
        for _ in 0..len {
            let r = xorshift64star(&mut state);
            let frac = (r >> 11) as f64 / (1u64 << 53) as f64; // uniform [0, 1)
            let trit = if frac < sparsity {
                0i8
            } else if frac < sparsity + (1.0 - sparsity) / 2.0 {
                1
            } else {
                -1
            };
            trits.push(trit);
        }
        Self::from_i8(&trits)
    }

    /// Number of trits.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of packed bytes.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Raw byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get the trit at a position.
    pub fn get(&self, index: usize) -> Trit {
        assert!(
            index < self.len,
            "index {index} out of bounds (len={})",
            self.len
        );
        let byte_idx = index / 5;
        let bit_idx = index % 5;
        let bv = self.bytes[byte_idx];
        assert!(bv < 243, "invalid packed byte {bv}");
        Trit::from_i8(DECODE_LUT[bv as usize][bit_idx])
    }

    /// Set the trit at a position.
    pub fn set(&mut self, index: usize, value: Trit) {
        assert!(index < self.len);
        let byte_idx = index / 5;
        let bit_idx = index % 5;
        let bv = self.bytes[byte_idx];
        let mut decoded = DECODE_LUT[bv as usize];
        decoded[bit_idx] = value.to_i8();
        let vals: [usize; 5] = std::array::from_fn(|i| (decoded[i] + 1) as usize);
        self.bytes[byte_idx] = ENCODE_LUT[vals[0]][vals[1]][vals[2]][vals[3]][vals[4]];
    }

    /// Decode all trits to i8 buffer (for SIMD operations).
    pub fn decode_to_i8(&self, out: &mut [i8]) {
        assert!(out.len() >= self.len);
        let mut pos = 0;
        for &b in &self.bytes {
            let decoded = &DECODE_LUT[b as usize];
            let remaining = self.len - pos;
            let count = remaining.min(5);
            out[pos..pos + count].copy_from_slice(&decoded[..count]);
            pos += count;
        }
    }

    /// Count positive (+1) trits.
    pub fn count_pos(&self) -> usize {
        let mut decoded = vec![0i8; self.len];
        self.decode_to_i8(&mut decoded);
        decoded.iter().filter(|&&v| v == 1).count()
    }

    /// Count negative (-1) trits.
    pub fn count_neg(&self) -> usize {
        let mut decoded = vec![0i8; self.len];
        self.decode_to_i8(&mut decoded);
        decoded.iter().filter(|&&v| v == -1).count()
    }

    /// Count zero trits.
    pub fn count_zero(&self) -> usize {
        self.len - self.count_pos() - self.count_neg()
    }

    /// Sparsity: fraction of zero elements.
    pub fn sparsity(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        self.count_zero() as f64 / self.len as f64
    }

    /// Dot product of two packed trit vectors (LUT-based).
    pub fn dot(&self, other: &PackedTrits) -> i32 {
        assert_eq!(self.len, other.len);
        let mut sum = 0i32;
        for (&a, &b) in self.bytes.iter().zip(other.bytes.iter()) {
            sum += DOT_LUT[a as usize][b as usize] as i32;
        }
        sum
    }

    /// Element-wise multiply (bind operation for ternary HDC).
    ///
    /// Ternary × ternary stays ternary: {-1,0,+1} × {-1,0,+1} → {-1,0,+1}.
    pub fn multiply(&self, other: &PackedTrits) -> PackedTrits {
        assert_eq!(self.len, other.len);
        let mut result_trits = vec![0i8; self.len];
        let mut a_decoded = vec![0i8; self.len];
        let mut b_decoded = vec![0i8; self.len];
        self.decode_to_i8(&mut a_decoded);
        other.decode_to_i8(&mut b_decoded);
        for i in 0..self.len {
            result_trits[i] = a_decoded[i] * b_decoded[i];
        }
        PackedTrits::from_i8(&result_trits)
    }

    /// Cyclic permutation (rotate right by `shift` positions).
    pub fn permute(&self, shift: usize) -> PackedTrits {
        if self.len == 0 || shift % self.len == 0 {
            return self.clone();
        }
        let shift = shift % self.len;
        let mut decoded = vec![0i8; self.len];
        self.decode_to_i8(&mut decoded);
        let mut result = vec![0i8; self.len];
        for i in 0..self.len {
            result[(i + shift) % self.len] = decoded[i];
        }
        PackedTrits::from_i8(&result)
    }
}

/// Pack a slice of i8 trits (convenience function).
pub fn pack_i8(trits: &[i8]) -> PackedTrits {
    PackedTrits::from_i8(trits)
}

/// Dot product of i8 weights × f32 input (for matvec).
pub fn dot_i8_f32(weights: &[i8], input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if weights.len() >= 16 {
            return unsafe { dot_i8_f32_neon(weights, input) };
        }
    }
    dot_i8_f32_scalar(weights, input)
}

fn dot_i8_f32_scalar(weights: &[i8], input: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&w, &x) in weights.iter().zip(input.iter()) {
        sum += w as f32 * x;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_f32_neon(weights: &[i8], input: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = weights.len();
    let chunks = n / 16;
    let mut acc = vdupq_n_f32(0.0);

    for c in 0..chunks {
        let base = c * 16;
        // SAFETY: bounds checked by chunks = n/16, so base+16 <= n.
        let w = unsafe { vld1q_s8(weights.as_ptr().add(base)) };
        let w_lo = vmovl_s8(vget_low_s8(w));
        let w_hi = vmovl_s8(vget_high_s8(w));
        let w32_0 = vmovl_s16(vget_low_s16(w_lo));
        let w32_1 = vmovl_s16(vget_high_s16(w_lo));
        let w32_2 = vmovl_s16(vget_low_s16(w_hi));
        let w32_3 = vmovl_s16(vget_high_s16(w_hi));
        let wf0 = vcvtq_f32_s32(w32_0);
        let wf1 = vcvtq_f32_s32(w32_1);
        let wf2 = vcvtq_f32_s32(w32_2);
        let wf3 = vcvtq_f32_s32(w32_3);
        // SAFETY: input.len() == weights.len(), base+16 <= n.
        let i0 = unsafe { vld1q_f32(input.as_ptr().add(base)) };
        let i1 = unsafe { vld1q_f32(input.as_ptr().add(base + 4)) };
        let i2 = unsafe { vld1q_f32(input.as_ptr().add(base + 8)) };
        let i3 = unsafe { vld1q_f32(input.as_ptr().add(base + 12)) };
        acc = vfmaq_f32(acc, wf0, i0);
        acc = vfmaq_f32(acc, wf1, i1);
        acc = vfmaq_f32(acc, wf2, i2);
        acc = vfmaq_f32(acc, wf3, i3);
    }

    let mut sum = vaddvq_f32(acc);
    for i in (chunks * 16)..n {
        sum += weights[i] as f32 * input[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let trits: Vec<i8> = vec![1, 0, -1, 1, -1, 0, 0, 1];
        let packed = PackedTrits::from_i8(&trits);
        let mut decoded = vec![0i8; trits.len()];
        packed.decode_to_i8(&mut decoded);
        assert_eq!(decoded, trits);
    }

    #[test]
    fn get_set() {
        let mut p = PackedTrits::zeros(10);
        assert_eq!(p.get(3), Trit::Zero);
        p.set(3, Trit::Pos);
        assert_eq!(p.get(3), Trit::Pos);
        p.set(3, Trit::Neg);
        assert_eq!(p.get(3), Trit::Neg);
    }

    #[test]
    fn zeros_are_zero() {
        let p = PackedTrits::zeros(100);
        for i in 0..100 {
            assert_eq!(p.get(i), Trit::Zero);
        }
        assert!((p.sparsity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn dot_product_identity() {
        let a = PackedTrits::from_i8(&[1, 0, -1, 1, -1]);
        let b = PackedTrits::from_i8(&[1, 0, -1, 1, -1]);
        // 1*1 + 0*0 + (-1)*(-1) + 1*1 + (-1)*(-1) = 1+0+1+1+1 = 4
        assert_eq!(a.dot(&b), 4);
    }

    #[test]
    fn dot_product_orthogonal() {
        let a = PackedTrits::from_i8(&[1, -1, 0, 0, 0]);
        let b = PackedTrits::from_i8(&[0, 0, 1, -1, 0]);
        assert_eq!(a.dot(&b), 0);
    }

    #[test]
    fn multiply_binding() {
        let a = PackedTrits::from_i8(&[1, -1, 0, 1, -1]);
        let b = PackedTrits::from_i8(&[1, 1, 1, -1, 0]);
        let c = a.multiply(&b);
        // 1*1=1, -1*1=-1, 0*1=0, 1*-1=-1, -1*0=0
        let mut decoded = vec![0i8; 5];
        c.decode_to_i8(&mut decoded);
        assert_eq!(decoded, vec![1, -1, 0, -1, 0]);
    }

    #[test]
    fn permute_roundtrip() {
        let p = PackedTrits::from_i8(&[1, -1, 0, 1, -1, 0, 1]);
        let shifted = p.permute(3);
        let back = shifted.permute(p.len() - 3);
        let mut orig = vec![0i8; p.len()];
        let mut round = vec![0i8; p.len()];
        p.decode_to_i8(&mut orig);
        back.decode_to_i8(&mut round);
        assert_eq!(orig, round);
    }

    #[test]
    fn random_sparsity() {
        let p = PackedTrits::random(10000, 0.5, 42);
        let s = p.sparsity();
        assert!(s > 0.4 && s < 0.6, "sparsity {s} should be ~0.5");
    }

    #[test]
    fn random_deterministic() {
        let a = PackedTrits::random(1000, 0.3, 42);
        let b = PackedTrits::random(1000, 0.3, 42);
        assert_eq!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn counts() {
        let p = PackedTrits::from_i8(&[1, 0, -1, 1, 0, -1, 0]);
        assert_eq!(p.count_pos(), 2);
        assert_eq!(p.count_neg(), 2);
        assert_eq!(p.count_zero(), 3);
    }

    #[test]
    fn dot_i8_f32_correctness() {
        let weights: Vec<i8> = vec![1, -1, 0, 1, -1, 1, 0, -1, 1, 1, -1, 0, 1, -1, 1, 0, 1, -1];
        let input: Vec<f32> = (0..18).map(|i| i as f32 * 0.1).collect();
        let result = dot_i8_f32(&weights, &input);
        let expected: f32 = weights
            .iter()
            .zip(input.iter())
            .map(|(&w, &x)| w as f32 * x)
            .sum();
        assert!(
            (result - expected).abs() < 1e-4,
            "got {result}, expected {expected}"
        );
    }
}
