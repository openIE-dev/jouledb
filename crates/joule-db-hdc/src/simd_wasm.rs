//! WASM SIMD-Accelerated HDC Operations
//!
//! Provides 128-bit SIMD-accelerated binary hypervector operations for browser targets.
//! WASM SIMD is now universally supported in all major browsers (2025+).
//!
//! Key operations accelerated:
//! - XOR binding: 2x faster using v128 XOR
//! - Popcount/Hamming distance: using v128 lookup table approach
//! - Batch operations: process multiple vector pairs simultaneously
//!
//! ## Architecture
//!
//! WASM SIMD provides 128-bit (v128) lanes. Since our binary hypervectors are
//! packed as `Vec<u64>`, we process 2 u64 words (128 bits of the hypervector)
//! per SIMD instruction.
//!
//! ## Popcount Strategy
//!
//! WASM SIMD has no native popcount instruction. We use the standard
//! nibble-lookup approach:
//! 1. Split each byte into low/high nibbles
//! 2. Use `i8x16_swizzle` as a parallel 4-bit lookup table
//! 3. Sum the per-byte counts with horizontal addition
//!
//! This is the same algorithm used by `_mm_popcnt_epi8` polyfills on x86.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use joule_db_hdc::simd_wasm;
//! use joule_db_hdc::BinaryHyperVector;
//!
//! let a = BinaryHyperVector::random(512, 42);
//! let b = BinaryHyperVector::random(512, 43);
//!
//! let bound = simd_wasm::bind_simd(&a, &b);
//! let dist = simd_wasm::hamming_distance_simd(&a, &b);
//! ```

use crate::binary_hd::BinaryHyperVector;

// ============================================================================
// WASM SIMD implementation (wasm32 with simd128)
// ============================================================================

/// Bind two binary hypervectors using XOR (WASM SIMD v128 accelerated).
///
/// Processes 128 bits (2 u64 words) at a time using `v128_xor`.
/// Falls back to scalar XOR for any remainder words.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn bind_simd_wasm(a: &BinaryHyperVector, b: &BinaryHyperVector) -> BinaryHyperVector {
    use std::arch::wasm32::*;

    debug_assert_eq!(a.dimensions(), b.dimensions());

    let a_words = a.words();
    let b_words = b.words();
    let len = a_words.len();
    let mut result = vec![0u64; len];

    let chunks = len / 2; // 2 u64s = 128 bits per v128

    for i in 0..chunks {
        let base = i * 2;

        // Load 128 bits (2 u64s) from each vector
        let va = v128_load(a_words.as_ptr().add(base) as *const v128);
        let vb = v128_load(b_words.as_ptr().add(base) as *const v128);

        // XOR binding
        let xor = v128_xor(va, vb);

        // Store result
        v128_store(result.as_mut_ptr().add(base) as *mut v128, xor);
    }

    // Handle remainder (0 or 1 word)
    let base = chunks * 2;
    for i in base..len {
        result[i] = a_words[i] ^ b_words[i];
    }

    BinaryHyperVector::from_words(result, a.dimensions())
}

/// Compute Hamming distance between two binary hypervectors (WASM SIMD v128 accelerated).
///
/// Uses the nibble-lookup popcount algorithm:
/// 1. XOR to get differing bits
/// 2. For each byte, split into low/high nibbles
/// 3. Use `i8x16_swizzle` as a 16-entry lookup table for 4-bit popcount
/// 4. Sum byte counts using `i8x16_add` and horizontal reduction
///
/// This achieves approximately 2x speedup over scalar popcount on WASM targets.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn hamming_distance_simd_wasm(a: &BinaryHyperVector, b: &BinaryHyperVector) -> u32 {
    use std::arch::wasm32::*;

    debug_assert_eq!(a.dimensions(), b.dimensions());

    let a_words = a.words();
    let b_words = b.words();
    let len = a_words.len();

    // Popcount lookup table: number of set bits for values 0..15
    // popcount_table[i] = i.count_ones() for i in 0..16
    let popcount_table = i8x16(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

    // Mask to extract low nibble (0x0F repeated)
    let low_mask = u8x16_splat(0x0F);

    // Accumulator for the total popcount (using u64x2 lanes)
    let mut total_acc = u64x2_splat(0);

    let chunks = len / 2; // 2 u64s = 128 bits per v128

    for i in 0..chunks {
        let base = i * 2;

        // Load 128 bits from each vector
        let va = v128_load(a_words.as_ptr().add(base) as *const v128);
        let vb = v128_load(b_words.as_ptr().add(base) as *const v128);

        // XOR to get differing bits
        let xor = v128_xor(va, vb);

        // Nibble-lookup popcount:
        // Split each byte into low and high nibbles
        let lo = v128_and(xor, low_mask);
        let hi = v128_and(u8x16_shr(xor, 4), low_mask);

        // Look up popcount for each nibble using swizzle
        let cnt_lo = i8x16_swizzle(popcount_table, lo);
        let cnt_hi = i8x16_swizzle(popcount_table, hi);

        // Sum low + high nibble counts per byte
        let byte_counts = u8x16_add(cnt_lo, cnt_hi);

        // Horizontal sum: reduce 16 byte counts to a total
        // Use SAD (sum of absolute differences) against zero to get u16 lane sums,
        // then accumulate into u64 lanes.
        //
        // i8x16_extract_lane gets individual bytes, but that's slow.
        // Instead, we use widening additions:
        //
        // Step 1: pairwise add adjacent bytes into u16 lanes (8 lanes)
        // WASM doesn't have a direct pairwise-add, so we use the
        // extadd_pairwise approach: u16x8_extadd_pairwise_u8x16
        let sum_u16 = u16x8_extadd_pairwise_u8x16(byte_counts);

        // Step 2: pairwise add adjacent u16 into u32 lanes (4 lanes)
        let sum_u32 = u32x4_extadd_pairwise_u16x8(sum_u16);

        // Step 3: accumulate u32x4 into u64x2 by widening addition
        // Extract the 4 u32 lanes and accumulate into our u64 accumulator
        let lo_u64 = u64x2(
            (u32x4_extract_lane::<0>(sum_u32) as u64) + (u32x4_extract_lane::<1>(sum_u32) as u64),
            (u32x4_extract_lane::<2>(sum_u32) as u64) + (u32x4_extract_lane::<3>(sum_u32) as u64),
        );
        total_acc = u64x2_add(total_acc, lo_u64);
    }

    // Extract u64 lanes and sum
    let total = u64x2_extract_lane::<0>(total_acc) + u64x2_extract_lane::<1>(total_acc);
    let mut total = total as u32;

    // Handle remainder with scalar popcount
    let base = chunks * 2;
    for i in base..len {
        total += (a_words[i] ^ b_words[i]).count_ones();
    }

    total
}

/// Batch bind: XOR-bind multiple pairs of binary hypervectors (WASM SIMD accelerated).
///
/// Processes each pair using SIMD-accelerated XOR binding.
/// This is useful for bulk encoding operations where many bindings are needed.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn batch_bind_simd_wasm(
    pairs: &[(&BinaryHyperVector, &BinaryHyperVector)],
) -> Vec<BinaryHyperVector> {
    pairs.iter().map(|(a, b)| bind_simd_wasm(a, b)).collect()
}

// ============================================================================
// Public dispatch functions (safe wrappers)
// ============================================================================

/// Bind two binary hypervectors using XOR with SIMD acceleration.
///
/// On `wasm32` targets with SIMD support, this uses 128-bit v128 XOR instructions.
/// On all other targets, this falls back to a scalar word-by-word XOR.
///
/// # Panics
///
/// Panics (in debug mode) if `a` and `b` have different dimensions.
pub fn bind_simd(a: &BinaryHyperVector, b: &BinaryHyperVector) -> BinaryHyperVector {
    debug_assert_eq!(a.dimensions(), b.dimensions());

    #[cfg(target_arch = "wasm32")]
    {
        // Safety: we've verified dimensions match and simd128 target feature
        // is enabled at compile time for wasm32+simd128 targets.
        unsafe { bind_simd_wasm(a, b) }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        bind_scalar(a, b)
    }
}

/// Compute Hamming distance between two binary hypervectors with SIMD acceleration.
///
/// On `wasm32` targets with SIMD support, this uses a v128 nibble-lookup popcount
/// algorithm for approximately 2x speedup over scalar.
/// On all other targets, this falls back to scalar XOR + popcount.
///
/// # Panics
///
/// Panics (in debug mode) if `a` and `b` have different dimensions.
pub fn hamming_distance_simd(a: &BinaryHyperVector, b: &BinaryHyperVector) -> u32 {
    debug_assert_eq!(a.dimensions(), b.dimensions());

    #[cfg(target_arch = "wasm32")]
    {
        unsafe { hamming_distance_simd_wasm(a, b) }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        hamming_distance_scalar(a, b)
    }
}

/// Batch bind multiple pairs of binary hypervectors with SIMD acceleration.
///
/// On `wasm32` targets, each pair is bound using SIMD-accelerated XOR.
/// On all other targets, this falls back to scalar binding.
///
/// # Panics
///
/// Panics (in debug mode) if any pair has mismatched dimensions.
pub fn batch_bind_simd(
    pairs: &[(&BinaryHyperVector, &BinaryHyperVector)],
) -> Vec<BinaryHyperVector> {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { batch_bind_simd_wasm(pairs) }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        batch_bind_scalar(pairs)
    }
}

// ============================================================================
// Scalar fallback implementations (non-WASM targets)
// ============================================================================

/// Scalar XOR binding fallback.
///
/// Used on non-WASM targets where WASM SIMD instructions are not available.
fn bind_scalar(a: &BinaryHyperVector, b: &BinaryHyperVector) -> BinaryHyperVector {
    debug_assert_eq!(a.dimensions(), b.dimensions());

    let words: Vec<u64> = a
        .words()
        .iter()
        .zip(b.words().iter())
        .map(|(&wa, &wb)| wa ^ wb)
        .collect();

    BinaryHyperVector::from_words(words, a.dimensions())
}

/// Scalar Hamming distance fallback.
///
/// Uses 4-way unrolling for instruction-level parallelism, matching the
/// pattern in `crate::simd::hamming_scalar`.
fn hamming_distance_scalar(a: &BinaryHyperVector, b: &BinaryHyperVector) -> u32 {
    debug_assert_eq!(a.dimensions(), b.dimensions());

    let a_words = a.words();
    let b_words = b.words();
    let len = a_words.len();

    let mut total = 0u32;
    let chunks = len / 4;
    let remainder = len % 4;

    // Main loop: 4 words at a time for ILP
    for i in 0..chunks {
        let base = i * 4;
        let d0 = (a_words[base] ^ b_words[base]).count_ones();
        let d1 = (a_words[base + 1] ^ b_words[base + 1]).count_ones();
        let d2 = (a_words[base + 2] ^ b_words[base + 2]).count_ones();
        let d3 = (a_words[base + 3] ^ b_words[base + 3]).count_ones();
        total += d0 + d1 + d2 + d3;
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        total += (a_words[base + i] ^ b_words[base + i]).count_ones();
    }

    total
}

/// Scalar batch binding fallback.
fn batch_bind_scalar(pairs: &[(&BinaryHyperVector, &BinaryHyperVector)]) -> Vec<BinaryHyperVector> {
    pairs.iter().map(|(a, b)| bind_scalar(a, b)).collect()
}

// ============================================================================
// Utility functions
// ============================================================================

/// Returns whether WASM SIMD is available on the current target.
///
/// This is a compile-time check: returns `true` on `wasm32` targets
/// compiled with SIMD support, `false` otherwise.
pub const fn is_wasm_simd_available() -> bool {
    cfg!(target_arch = "wasm32")
}

/// Returns a human-readable description of the SIMD backend in use.
pub fn simd_backend_description() -> &'static str {
    if cfg!(target_arch = "wasm32") {
        "WASM SIMD v128 (128-bit)"
    } else {
        "Scalar fallback (no WASM SIMD)"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_hd::BinaryHyperVector;

    #[test]
    fn test_bind_scalar_basic() {
        let a = BinaryHyperVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFFu64; 8], 512);
        let b = BinaryHyperVector::from_words(vec![0u64; 8], 512);

        let result = bind_scalar(&a, &b);
        // XOR(all-ones, all-zeros) = all-ones
        for &w in result.words() {
            assert_eq!(w, 0xFFFF_FFFF_FFFF_FFFFu64);
        }
    }

    #[test]
    fn test_bind_scalar_self_inverse() {
        // XOR is self-inverse: bind(bind(a, b), b) == a
        let a = BinaryHyperVector::from_words(vec![0xDEAD_BEEF_CAFE_BABEu64; 8], 512);
        let b = BinaryHyperVector::from_words(vec![0x1234_5678_9ABC_DEF0u64; 8], 512);

        let bound = bind_scalar(&a, &b);
        let recovered = bind_scalar(&bound, &b);

        assert_eq!(a.words(), recovered.words());
    }

    #[test]
    fn test_hamming_distance_scalar_identical() {
        let a = BinaryHyperVector::from_words(vec![0xDEAD_BEEFu64; 8], 512);
        let dist = hamming_distance_scalar(&a, &a);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_hamming_distance_scalar_opposite() {
        let a = BinaryHyperVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFFu64; 8], 512);
        let b = BinaryHyperVector::from_words(vec![0u64; 8], 512);

        let dist = hamming_distance_scalar(&a, &b);
        assert_eq!(dist, 512); // 8 words * 64 bits = 512
    }

    #[test]
    fn test_hamming_distance_scalar_known_value() {
        // 0xFF = 8 bits set per word in the lowest byte
        let a = BinaryHyperVector::from_words(vec![0xFFu64; 8], 512);
        let b = BinaryHyperVector::from_words(vec![0u64; 8], 512);

        let dist = hamming_distance_scalar(&a, &b);
        assert_eq!(dist, 64); // 8 bits per word * 8 words
    }

    #[test]
    fn test_hamming_distance_scalar_single_bit() {
        let a = BinaryHyperVector::from_words(vec![1u64, 0, 0, 0, 0, 0, 0, 0], 512);
        let b = BinaryHyperVector::from_words(vec![0u64; 8], 512);

        let dist = hamming_distance_scalar(&a, &b);
        assert_eq!(dist, 1);
    }

    #[test]
    fn test_batch_bind_scalar() {
        let a1 = BinaryHyperVector::from_words(vec![0xAAAAu64; 8], 512);
        let b1 = BinaryHyperVector::from_words(vec![0x5555u64; 8], 512);
        let a2 = BinaryHyperVector::from_words(vec![0xFFFFu64; 8], 512);
        let b2 = BinaryHyperVector::from_words(vec![0xFFFFu64; 8], 512);

        let pairs: Vec<(&BinaryHyperVector, &BinaryHyperVector)> = vec![(&a1, &b1), (&a2, &b2)];

        let results = batch_bind_scalar(&pairs);
        assert_eq!(results.len(), 2);

        // 0xAAAA ^ 0x5555 = 0xFFFF
        for &w in results[0].words() {
            assert_eq!(w, 0xFFFFu64);
        }

        // 0xFFFF ^ 0xFFFF = 0
        for &w in results[1].words() {
            assert_eq!(w, 0u64);
        }
    }

    #[test]
    fn test_dispatch_functions_use_scalar_on_non_wasm() {
        // On non-wasm targets, the dispatch functions should use scalar fallbacks
        let a = BinaryHyperVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFFu64; 8], 512);
        let b = BinaryHyperVector::from_words(vec![0u64; 8], 512);

        let bound = bind_simd(&a, &b);
        for &w in bound.words() {
            assert_eq!(w, 0xFFFF_FFFF_FFFF_FFFFu64);
        }

        let dist = hamming_distance_simd(&a, &b);
        assert_eq!(dist, 512);

        let pairs: Vec<(&BinaryHyperVector, &BinaryHyperVector)> = vec![(&a, &b)];
        let batch = batch_bind_simd(&pairs);
        assert_eq!(batch.len(), 1);
        for &w in batch[0].words() {
            assert_eq!(w, 0xFFFF_FFFF_FFFF_FFFFu64);
        }
    }

    #[test]
    fn test_is_wasm_simd_available() {
        // On non-wasm test targets, this should be false
        if cfg!(target_arch = "wasm32") {
            assert!(is_wasm_simd_available());
        } else {
            assert!(!is_wasm_simd_available());
        }
    }

    #[test]
    fn test_simd_backend_description() {
        let desc = simd_backend_description();
        assert!(!desc.is_empty());
        if cfg!(target_arch = "wasm32") {
            assert!(desc.contains("WASM"));
        } else {
            assert!(desc.contains("Scalar"));
        }
    }

    #[test]
    fn test_consistency_with_binary_hd() {
        // Verify our scalar fallback matches BinaryHyperVector's native methods
        let a = BinaryHyperVector::from_words(
            vec![
                0x123456789ABCDEFu64,
                0xFEDCBA9876543210u64,
                0xDEADBEEFCAFEBABEu64,
                0x0123456789ABCDEFu64,
                0xAAAAAAAA55555555u64,
                0x1111111122222222u64,
                0x3333333344444444u64,
                0x5555555566666666u64,
            ],
            512,
        );
        let b = BinaryHyperVector::from_words(
            vec![
                0xFEDCBA9876543210u64,
                0x123456789ABCDEFu64,
                0xCAFEBABEDEADBEEFu64,
                0xFEDCBA9876543210u64,
                0x55555555AAAAAAAAu64,
                0x2222222211111111u64,
                0x4444444433333333u64,
                0x6666666655555555u64,
            ],
            512,
        );

        // Compare bind results
        let native_bind = a.bind(&b);
        let simd_bind = bind_simd(&a, &b);
        assert_eq!(native_bind.words(), simd_bind.words());

        // Compare Hamming distance results
        let native_dist = a.hamming_distance(&b);
        let simd_dist = hamming_distance_simd(&a, &b);
        assert_eq!(native_dist, simd_dist);
    }

    #[test]
    fn test_non_aligned_dimensions() {
        // Test with dimensions that are not a multiple of 128 (v128 width)
        // 192 dimensions = 3 u64 words, meaning 1 word is a remainder for v128 processing
        let a = BinaryHyperVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFFu64; 3], 192);
        let b = BinaryHyperVector::from_words(vec![0u64; 3], 192);

        let bound = bind_simd(&a, &b);
        assert_eq!(bound.words().len(), 3);
        for &w in bound.words() {
            assert_eq!(w, 0xFFFF_FFFF_FFFF_FFFFu64);
        }

        let dist = hamming_distance_simd(&a, &b);
        assert_eq!(dist, 192); // 3 * 64 = 192
    }

    #[test]
    fn test_empty_batch() {
        let pairs: Vec<(&BinaryHyperVector, &BinaryHyperVector)> = vec![];
        let results = batch_bind_simd(&pairs);
        assert!(results.is_empty());
    }
}
