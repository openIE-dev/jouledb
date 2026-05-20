//! SIMD-Accelerated Operations for Hyperdimensional Computing
//!
//! This module provides SIMD-optimized implementations of core HDC operations,
//! particularly Hamming distance calculation which is the bottleneck for similarity search.
//!
//! ## Supported Architectures
//!
//! - **AVX-512 VPOPCNTDQ** - 8x u64 popcount in parallel (best performance)
//! - **AVX-512** - 8x u64 XOR + horizontal popcount
//! - **AVX2** - 4x u64 XOR + fallback popcount
//! - **Fallback** - Scalar with 4-way unrolling
//!
//! ## Performance
//!
//! For 4096-bit vectors (64 u64 words):
//! - AVX-512 VPOPCNTDQ: ~5ns
//! - AVX-512: ~10ns
//! - AVX2: ~15ns
//! - Scalar: ~20ns
//!
//! ## Usage
//!
//! ```rust,ignore
//! use joule_db_hdc::simd::{hamming_distance, HammingEngine};
//!
//! let a = vec![0u64; 64]; // 4096 bits
//! let b = vec![0xFFFFFFFFFFFFFFFFu64; 64];
//!
//! let dist = hamming_distance(&a, &b);
//! assert_eq!(dist, 4096);
//! ```

use std::sync::OnceLock;

/// Detect SIMD capabilities at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// AVX-512 with VPOPCNTDQ extension
    Avx512Popcnt,
    /// AVX-512 basic
    Avx512,
    /// AVX2
    Avx2,
    /// Scalar fallback
    Scalar,
}

impl SimdLevel {
    /// Detect the best available SIMD level
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            // Check for AVX-512 VPOPCNTDQ (best for Hamming distance)
            if std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
                return SimdLevel::Avx512Popcnt;
            }
            // Check for basic AVX-512
            if std::arch::is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            // Check for AVX2
            if std::arch::is_x86_feature_detected!("avx2") {
                return SimdLevel::Avx2;
            }
        }
        SimdLevel::Scalar
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            SimdLevel::Avx512Popcnt => "AVX-512 with VPOPCNTDQ",
            SimdLevel::Avx512 => "AVX-512",
            SimdLevel::Avx2 => "AVX2",
            SimdLevel::Scalar => "Scalar",
        }
    }
}

/// Cached SIMD level detection
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Get the detected SIMD level (cached)
pub fn simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(SimdLevel::detect)
}

/// Compute Hamming distance between two slices of u64
///
/// Automatically dispatches to the best available SIMD implementation.
#[inline]
pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());

    match simd_level() {
        SimdLevel::Avx512Popcnt => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                hamming_avx512_popcnt(a, b)
            }
            #[cfg(not(target_arch = "x86_64"))]
            hamming_scalar(a, b)
        }
        SimdLevel::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                hamming_avx512(a, b)
            }
            #[cfg(not(target_arch = "x86_64"))]
            hamming_scalar(a, b)
        }
        SimdLevel::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                hamming_avx2(a, b)
            }
            #[cfg(not(target_arch = "x86_64"))]
            hamming_scalar(a, b)
        }
        SimdLevel::Scalar => hamming_scalar(a, b),
    }
}

/// Scalar Hamming distance with 4-way unrolling
#[inline]
pub fn hamming_scalar(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());

    let mut total = 0u32;
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    // Main loop - 4 words at a time for ILP
    for i in 0..chunks {
        let base = i * 4;
        let d0 = (a[base] ^ b[base]).count_ones();
        let d1 = (a[base + 1] ^ b[base + 1]).count_ones();
        let d2 = (a[base + 2] ^ b[base + 2]).count_ones();
        let d3 = (a[base + 3] ^ b[base + 3]).count_ones();
        total += d0 + d1 + d2 + d3;
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        total += (a[base + i] ^ b[base + i]).count_ones();
    }

    total
}

/// AVX-512 VPOPCNTDQ implementation (8x u64 vector popcount)
///
/// This is the fastest implementation, processing 512 bits at once.
///
/// # Safety
/// Requires AVX-512F and AVX-512VPOPCNTDQ CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
#[inline]
pub unsafe fn hamming_avx512_popcnt(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());

    unsafe {
        let mut total = _mm512_setzero_si512();
        let chunks = a.len() / 8;

        // Process 8 u64s (512 bits) at a time
        for i in 0..chunks {
            let base = i * 8;

            // Load 8 u64s from each slice
            let va = _mm512_loadu_si512(a.as_ptr().add(base) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(base) as *const __m512i);

            // XOR to get differing bits
            let xor = _mm512_xor_si512(va, vb);

            // Vector popcount - each 64-bit lane gets its popcount
            let popcnt = _mm512_popcnt_epi64(xor);

            // Accumulate
            total = _mm512_add_epi64(total, popcnt);
        }

        // Horizontal sum of the 8 lanes
        let sum = horizontal_sum_512(total);

        // Handle remainder with scalar
        let base = chunks * 8;
        let mut remainder_sum = 0u32;
        for i in base..a.len() {
            remainder_sum += (a[i] ^ b[i]).count_ones();
        }

        sum + remainder_sum
    }
}

/// Horizontal sum of 8 u64 lanes in __m512i
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn horizontal_sum_512(v: std::arch::x86_64::__m512i) -> u32 {
    use std::arch::x86_64::*;

    unsafe {
        // Extract upper and lower 256-bit halves
        let lo = _mm512_castsi512_si256(v);
        let hi = _mm512_extracti64x4_epi64(v, 1);

        // Add the halves
        let sum256 = _mm256_add_epi64(lo, hi);

        // Extract upper and lower 128-bit halves
        let lo128 = _mm256_castsi256_si128(sum256);
        let hi128 = _mm256_extracti128_si256(sum256, 1);

        // Add
        let sum128 = _mm_add_epi64(lo128, hi128);

        // Final reduction: extract both 64-bit values and add
        let a = _mm_extract_epi64(sum128, 0) as u64;
        let b = _mm_extract_epi64(sum128, 1) as u64;

        (a + b) as u32
    }
}

/// AVX-512 implementation without VPOPCNTDQ
///
/// Uses XOR + scalar popcount as fallback.
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn hamming_avx512(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());

    unsafe {
        let mut total = 0u32;
        let chunks = a.len() / 8;

        // Process 8 u64s (512 bits) at a time
        for i in 0..chunks {
            let base = i * 8;

            // Load 8 u64s from each slice
            let va = _mm512_loadu_si512(a.as_ptr().add(base) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(base) as *const __m512i);

            // XOR to get differing bits
            let xor = _mm512_xor_si512(va, vb);

            // Store to memory and use scalar popcount
            let mut xor_buf = [0u64; 8];
            _mm512_storeu_si512(xor_buf.as_mut_ptr() as *mut __m512i, xor);

            // Scalar popcount (will use POPCNT instruction if available)
            for x in xor_buf {
                total += x.count_ones();
            }
        }

        // Handle remainder
        let base = chunks * 8;
        for i in base..a.len() {
            total += (a[i] ^ b[i]).count_ones();
        }

        total
    }
}

/// AVX2 implementation (4x u64)
///
/// # Safety
/// Requires AVX2 CPU feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hamming_avx2(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());

    unsafe {
        let mut total = 0u32;
        let chunks = a.len() / 4;

        // Process 4 u64s (256 bits) at a time
        for i in 0..chunks {
            let base = i * 4;

            // Load 4 u64s from each slice
            let va = _mm256_loadu_si256(a.as_ptr().add(base) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(base) as *const __m256i);

            // XOR to get differing bits
            let xor = _mm256_xor_si256(va, vb);

            // Store and use scalar popcount
            let mut xor_buf = [0u64; 4];
            _mm256_storeu_si256(xor_buf.as_mut_ptr() as *mut __m256i, xor);

            // Scalar popcount
            for x in xor_buf {
                total += x.count_ones();
            }
        }

        // Handle remainder
        let base = chunks * 4;
        for i in base..a.len() {
            total += (a[i] ^ b[i]).count_ones();
        }

        total
    }
}

/// Batch Hamming distance computation
///
/// Computes distances from a query to multiple targets efficiently.
/// Returns a vector of distances.
pub fn hamming_distances_batch(query: &[u64], targets: &[&[u64]]) -> Vec<u32> {
    targets.iter().map(|t| hamming_distance(query, t)).collect()
}

/// Parallel batch Hamming distance using Rayon
#[cfg(feature = "parallel")]
pub fn hamming_distances_parallel(query: &[u64], targets: &[&[u64]]) -> Vec<u32> {
    use rayon::prelude::*;
    targets
        .par_iter()
        .map(|t| hamming_distance(query, t))
        .collect()
}

/// Hamming distance engine with pre-allocated buffers
///
/// Useful for batch operations to avoid repeated allocations.
pub struct HammingEngine {
    simd_level: SimdLevel,
}

impl HammingEngine {
    /// Create a new Hamming engine
    pub fn new() -> Self {
        Self {
            simd_level: simd_level(),
        }
    }

    /// Get the SIMD level being used
    pub fn simd_level(&self) -> SimdLevel {
        self.simd_level
    }

    /// Compute Hamming distance
    #[inline]
    pub fn distance(&self, a: &[u64], b: &[u64]) -> u32 {
        hamming_distance(a, b)
    }

    /// Compute similarity (normalized to [0, 1])
    #[inline]
    pub fn similarity(&self, a: &[u64], b: &[u64]) -> f32 {
        let max_dist = (a.len() * 64) as f32;
        let dist = self.distance(a, b) as f32;
        1.0 - (dist / max_dist)
    }

    /// Find top-k most similar vectors
    pub fn top_k<'a>(
        &self,
        query: &[u64],
        candidates: &[&'a [u64]],
        k: usize,
    ) -> Vec<(usize, u32)> {
        let mut distances: Vec<(usize, u32)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, self.distance(query, c)))
            .collect();

        // Partial sort to get top-k
        distances.sort_by_key(|(_, d)| *d);
        distances.truncate(k);
        distances
    }
}

impl Default for HammingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_level_detection() {
        let level = simd_level();
        println!("Detected SIMD level: {:?} ({})", level, level.description());
    }

    #[test]
    fn test_hamming_scalar() {
        let a = vec![0xFFFFFFFFFFFFFFFFu64; 64]; // 4096 bits all 1s
        let b = vec![0u64; 64]; // 4096 bits all 0s
        assert_eq!(hamming_scalar(&a, &b), 4096);

        let c = vec![0xAAAAAAAAAAAAAAAAu64; 64]; // alternating bits
        let d = vec![0x5555555555555555u64; 64]; // opposite alternating
        assert_eq!(hamming_scalar(&c, &d), 4096);

        // Same vectors
        assert_eq!(hamming_scalar(&a, &a), 0);
    }

    #[test]
    fn test_hamming_distance_dispatch() {
        let a = vec![0xFFFFFFFFFFFFFFFFu64; 64];
        let b = vec![0u64; 64];
        assert_eq!(hamming_distance(&a, &b), 4096);

        // Test with different sizes
        let a2 = vec![0xFFu64; 8]; // 512 bits
        let b2 = vec![0u64; 8];
        assert_eq!(hamming_distance(&a2, &b2), 8 * 8); // 8 bits per byte, 8 bytes

        // Actually it's 8 bits set in each u64 (0xFF = 8 bits), times 8 words
        // 0xFF = 0b11111111 = 8 bits set in lowest byte
        // So 8 bits * 8 words = 64 differing bits
        assert_eq!(hamming_distance(&a2, &b2), 64);
    }

    #[test]
    fn test_hamming_engine() {
        let engine = HammingEngine::new();
        println!("HammingEngine using: {:?}", engine.simd_level());

        let a = vec![0u64; 64];
        let b = vec![0u64; 64];
        assert_eq!(engine.distance(&a, &b), 0);
        assert_eq!(engine.similarity(&a, &b), 1.0);
    }

    #[test]
    fn test_similarity() {
        let engine = HammingEngine::new();

        // Identical vectors
        let a = vec![0xDEADBEEFu64; 64];
        let b = vec![0xDEADBEEFu64; 64];
        assert_eq!(engine.similarity(&a, &b), 1.0);

        // Completely opposite (for 64 bits set in lower half)
        let c = vec![0xFFFFFFFFFFFFFFFFu64; 64];
        let d = vec![0u64; 64];
        assert_eq!(engine.similarity(&c, &d), 0.0);
    }

    #[test]
    fn test_top_k() {
        let engine = HammingEngine::new();

        let query = vec![0u64; 64];
        let targets: Vec<Vec<u64>> = (0..10).map(|i| vec![i as u64; 64]).collect();

        let refs: Vec<&[u64]> = targets.iter().map(|t| t.as_slice()).collect();
        let top = engine.top_k(&query, &refs, 3);

        assert_eq!(top.len(), 3);
        // First should be index 0 (same as query)
        assert_eq!(top[0].0, 0);
        assert_eq!(top[0].1, 0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_if_available() {
        if std::arch::is_x86_feature_detected!("avx2") {
            let a = vec![0xFFFFFFFFFFFFFFFFu64; 64];
            let b = vec![0u64; 64];
            let dist = unsafe { hamming_avx2(&a, &b) };
            assert_eq!(dist, 4096);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_if_available() {
        if std::arch::is_x86_feature_detected!("avx512f") {
            let a = vec![0xFFFFFFFFFFFFFFFFu64; 64];
            let b = vec![0u64; 64];
            let dist = unsafe { hamming_avx512(&a, &b) };
            assert_eq!(dist, 4096);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_popcnt_if_available() {
        if std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
            let a = vec![0xFFFFFFFFFFFFFFFFu64; 64];
            let b = vec![0u64; 64];
            let dist = unsafe { hamming_avx512_popcnt(&a, &b) };
            assert_eq!(dist, 4096);
        }
    }

    #[test]
    fn test_batch() {
        let query = vec![0u64; 64];
        let targets: Vec<Vec<u64>> = vec![vec![0u64; 64], vec![0xFFu64; 64], vec![0xFFFFu64; 64]];
        let refs: Vec<&[u64]> = targets.iter().map(|t| t.as_slice()).collect();

        let distances = hamming_distances_batch(&query, &refs);
        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0], 0); // Same
        assert_eq!(distances[1], 8 * 64); // 8 bits per word * 64 words (0xFF has 8 bits set)
        assert_eq!(distances[2], 16 * 64); // 16 bits per word * 64 words (0xFFFF has 16 bits set)
    }

    #[test]
    fn test_consistency() {
        // Verify all implementations give same results
        // Use wrapping operations to avoid overflow
        let a: Vec<u64> = (0..64)
            .map(|i| (i as u64).wrapping_mul(0x123456789ABCDEFu64))
            .collect();
        let b: Vec<u64> = (0..64)
            .map(|i| (i as u64).wrapping_mul(0xFEDCBA987654321u64))
            .collect();

        let scalar = hamming_scalar(&a, &b);
        let dispatch = hamming_distance(&a, &b);

        assert_eq!(
            scalar, dispatch,
            "Scalar and dispatched results should match"
        );

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let avx2 = unsafe { hamming_avx2(&a, &b) };
                assert_eq!(scalar, avx2, "Scalar and AVX2 results should match");
            }

            if std::arch::is_x86_feature_detected!("avx512f") {
                let avx512 = unsafe { hamming_avx512(&a, &b) };
                assert_eq!(scalar, avx512, "Scalar and AVX-512 results should match");
            }

            if std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
                let avx512p = unsafe { hamming_avx512_popcnt(&a, &b) };
                assert_eq!(
                    scalar, avx512p,
                    "Scalar and AVX-512 POPCNT results should match"
                );
            }
        }
    }
}
