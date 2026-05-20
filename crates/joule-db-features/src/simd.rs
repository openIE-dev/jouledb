//! SIMD-optimized distance calculations for vector similarity search
//!
//! This module provides highly optimized implementations of common distance
//! metrics using SIMD (Single Instruction, Multiple Data) instructions.
//!
//! ## Supported Platforms
//!
//! - **x86_64**: Uses AVX2 (256-bit) or SSE (128-bit) instructions
//! - **aarch64**: Uses NEON (128-bit) instructions
//! - **Fallback**: Auto-vectorizable scalar code for other platforms
//!
//! ## Performance
//!
//! SIMD implementations can process 4-8 floats per instruction, providing
//! 4-8x speedup over naive scalar implementations for large vectors.
//!
//! ## Example
//!
//! ```rust
//! use joule_db_features::simd::{euclidean_distance, cosine_similarity, dot_product};
//!
//! let a = vec![1.0, 2.0, 3.0, 4.0];
//! let b = vec![4.0, 3.0, 2.0, 1.0];
//!
//! let dist = euclidean_distance(&a, &b);
//! let sim = cosine_similarity(&a, &b);
//! let dot = dot_product(&a, &b);
//! ```

// ============================================================================
// SIMD Euclidean Distance
// ============================================================================

/// Calculate squared Euclidean distance between two vectors (SIMD-optimized).
///
/// Returns the sum of squared differences: Σ(a[i] - b[i])²
///
/// This is faster than `euclidean_distance` when you only need to compare
/// distances (avoids the sqrt).
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { euclidean_distance_squared_avx2(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "avx2")
    ))]
    {
        unsafe { euclidean_distance_squared_sse(a, b) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { euclidean_distance_squared_neon(a, b) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        euclidean_distance_squared_scalar(a, b)
    }
}

/// Calculate Euclidean (L2) distance between two vectors (SIMD-optimized).
///
/// Returns √(Σ(a[i] - b[i])²)
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared(a, b).sqrt()
}

/// Scalar fallback with auto-vectorization hints
#[inline]
fn euclidean_distance_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    // Process in chunks of 8 for better auto-vectorization
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    let mut sum = 0.0f32;

    // Main loop - processes 8 elements at a time
    for i in 0..chunks {
        let base = i * 8;
        let mut chunk_sum = 0.0f32;

        // Unrolled loop for auto-vectorization
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        let d4 = a[base + 4] - b[base + 4];
        let d5 = a[base + 5] - b[base + 5];
        let d6 = a[base + 6] - b[base + 6];
        let d7 = a[base + 7] - b[base + 7];

        chunk_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        chunk_sum += d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;

        sum += chunk_sum;
    }

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }

    sum
}

// x86_64 AVX2 implementation (256-bit, 8 floats at a time)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn euclidean_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum of 8 floats
    let sum128_lo = _mm256_castps256_ps128(sum);
    let sum128_hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    let base = chunks * 8;
    for i in base..len {
        let d = a[i] - b[i];
        result += d * d;
    }

    result
}

// x86_64 SSE implementation (128-bit, 4 floats at a time)
#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline]
unsafe fn euclidean_distance_squared_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 4;
    let mut sum = _mm_setzero_ps();

    for i in 0..chunks {
        let base = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(base));
        let vb = _mm_loadu_ps(b.as_ptr().add(base));
        let diff = _mm_sub_ps(va, vb);
        let sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
    }

    // Horizontal sum of 4 floats
    let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    let base = chunks * 4;
    for i in base..len {
        let d = a[i] - b[i];
        result += d * d;
    }

    result
}

// aarch64 NEON implementation (128-bit, 4 floats at a time)
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn euclidean_distance_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 4;
        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let base = i * 4;
            let va = vld1q_f32(a.as_ptr().add(base));
            let vb = vld1q_f32(b.as_ptr().add(base));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }

        // Horizontal sum
        let mut result = vaddvq_f32(sum);

        // Handle remainder
        let base = chunks * 4;
        for i in base..len {
            let d = a[i] - b[i];
            result += d * d;
        }

        result
    }
}

// ============================================================================
// SIMD Dot Product
// ============================================================================

/// Calculate dot product (inner product) of two vectors (SIMD-optimized).
///
/// Returns Σ(a[i] * b[i])
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { dot_product_avx2(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "avx2")
    ))]
    {
        unsafe { dot_product_sse(a, b) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { dot_product_neon(a, b) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        dot_product_scalar(a, b)
    }
}

/// Scalar fallback with auto-vectorization hints
#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    let mut sum = 0.0f32;

    for i in 0..chunks {
        let base = i * 8;
        let mut chunk_sum = 0.0f32;

        chunk_sum += a[base] * b[base];
        chunk_sum += a[base + 1] * b[base + 1];
        chunk_sum += a[base + 2] * b[base + 2];
        chunk_sum += a[base + 3] * b[base + 3];
        chunk_sum += a[base + 4] * b[base + 4];
        chunk_sum += a[base + 5] * b[base + 5];
        chunk_sum += a[base + 6] * b[base + 6];
        chunk_sum += a[base + 7] * b[base + 7];

        sum += chunk_sum;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }

    sum
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    let sum128_lo = _mm256_castps256_ps128(sum);
    let sum128_hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    let base = chunks * 8;
    for i in base..len {
        result += a[i] * b[i];
    }

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 4;
    let mut sum = _mm_setzero_ps();

    for i in 0..chunks {
        let base = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(base));
        let vb = _mm_loadu_ps(b.as_ptr().add(base));
        let prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
    }

    // Horizontal sum
    let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    let base = chunks * 4;
    for i in base..len {
        result += a[i] * b[i];
    }

    result
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 4;
        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let base = i * 4;
            let va = vld1q_f32(a.as_ptr().add(base));
            let vb = vld1q_f32(b.as_ptr().add(base));
            sum = vfmaq_f32(sum, va, vb);
        }

        let mut result = vaddvq_f32(sum);

        let base = chunks * 4;
        for i in base..len {
            result += a[i] * b[i];
        }

        result
    }
}

// ============================================================================
// SIMD L2 Norm (magnitude)
// ============================================================================

/// Calculate the L2 norm (magnitude) of a vector (SIMD-optimized).
///
/// Returns √(Σ(a[i]²))
#[inline]
pub fn l2_norm(a: &[f32]) -> f32 {
    l2_norm_squared(a).sqrt()
}

/// Calculate the squared L2 norm of a vector (SIMD-optimized).
///
/// Returns Σ(a[i]²)
#[inline]
pub fn l2_norm_squared(a: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { l2_norm_squared_avx2(a) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "avx2")
    ))]
    {
        unsafe { l2_norm_squared_sse(a) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { l2_norm_squared_neon(a) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        l2_norm_squared_scalar(a)
    }
}

#[inline]
fn l2_norm_squared_scalar(a: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    let mut sum = 0.0f32;

    for i in 0..chunks {
        let base = i * 8;
        let mut chunk_sum = 0.0f32;

        chunk_sum += a[base] * a[base];
        chunk_sum += a[base + 1] * a[base + 1];
        chunk_sum += a[base + 2] * a[base + 2];
        chunk_sum += a[base + 3] * a[base + 3];
        chunk_sum += a[base + 4] * a[base + 4];
        chunk_sum += a[base + 5] * a[base + 5];
        chunk_sum += a[base + 6] * a[base + 6];
        chunk_sum += a[base + 7] * a[base + 7];

        sum += chunk_sum;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        sum += a[base + i] * a[base + i];
    }

    sum
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn l2_norm_squared_avx2(a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        sum = _mm256_fmadd_ps(va, va, sum);
    }

    let sum128_lo = _mm256_castps256_ps128(sum);
    let sum128_hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    let base = chunks * 8;
    for i in base..len {
        result += a[i] * a[i];
    }

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline]
unsafe fn l2_norm_squared_sse(a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 4;
    let mut sum = _mm_setzero_ps();

    for i in 0..chunks {
        let base = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(base));
        let sq = _mm_mul_ps(va, va);
        sum = _mm_add_ps(sum, sq);
    }

    let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    let base = chunks * 4;
    for i in base..len {
        result += a[i] * a[i];
    }

    result
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn l2_norm_squared_neon(a: &[f32]) -> f32 {
    unsafe {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 4;
        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let base = i * 4;
            let va = vld1q_f32(a.as_ptr().add(base));
            sum = vfmaq_f32(sum, va, va);
        }

        let mut result = vaddvq_f32(sum);

        let base = chunks * 4;
        for i in base..len {
            result += a[i] * a[i];
        }

        result
    }
}

// ============================================================================
// SIMD Cosine Similarity
// ============================================================================

/// Calculate cosine similarity between two vectors (SIMD-optimized).
///
/// Returns (a · b) / (||a|| * ||b||)
///
/// Range: [-1, 1] where 1 = identical direction, 0 = orthogonal, -1 = opposite
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    // Compute all three values in one pass for better cache utilization
    let (dot, norm_a_sq, norm_b_sq) = dot_and_norms(a, b);

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Compute dot product and both norms in a single pass (cache-efficient).
#[inline]
fn dot_and_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { dot_and_norms_avx2(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "avx2")
    ))]
    {
        unsafe { dot_and_norms_sse(a, b) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { dot_and_norms_neon(a, b) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        dot_and_norms_scalar(a, b)
    }
}

#[inline]
fn dot_and_norms_scalar(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;

        dot += a[base] * b[base];
        dot += a[base + 1] * b[base + 1];
        dot += a[base + 2] * b[base + 2];
        dot += a[base + 3] * b[base + 3];

        norm_a += a[base] * a[base];
        norm_a += a[base + 1] * a[base + 1];
        norm_a += a[base + 2] * a[base + 2];
        norm_a += a[base + 3] * a[base + 3];

        norm_b += b[base] * b[base];
        norm_b += b[base + 1] * b[base + 1];
        norm_b += b[base + 2] * b[base + 2];
        norm_b += b[base + 3] * b[base + 3];
    }

    let base = chunks * 4;
    for i in 0..remainder {
        dot += a[base + i] * b[base + i];
        norm_a += a[base + i] * a[base + i];
        norm_b += b[base + i] * b[base + i];
    }

    (dot, norm_a, norm_b)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn dot_and_norms_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;

    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));

        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
    }

    // Horizontal sums
    let hsum = |v: __m256| -> f32 {
        let sum128_lo = _mm256_castps256_ps128(v);
        let sum128_hi = _mm256_extractf128_ps(v, 1);
        let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    };

    let mut dot = hsum(dot_sum);
    let mut norm_a = hsum(norm_a_sum);
    let mut norm_b = hsum(norm_b_sum);

    // Handle remainder
    let base = chunks * 8;
    for i in base..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    (dot, norm_a, norm_b)
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline]
unsafe fn dot_and_norms_sse(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 4;

    let mut dot_sum = _mm_setzero_ps();
    let mut norm_a_sum = _mm_setzero_ps();
    let mut norm_b_sum = _mm_setzero_ps();

    for i in 0..chunks {
        let base = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(base));
        let vb = _mm_loadu_ps(b.as_ptr().add(base));

        dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
        norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
        norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));
    }

    let hsum = |v: __m128| -> f32 {
        let sum64 = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    };

    let mut dot = hsum(dot_sum);
    let mut norm_a = hsum(norm_a_sum);
    let mut norm_b = hsum(norm_b_sum);

    let base = chunks * 4;
    for i in base..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    (dot, norm_a, norm_b)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn dot_and_norms_neon(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    unsafe {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 4;

        let mut dot_sum = vdupq_n_f32(0.0);
        let mut norm_a_sum = vdupq_n_f32(0.0);
        let mut norm_b_sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let base = i * 4;
            let va = vld1q_f32(a.as_ptr().add(base));
            let vb = vld1q_f32(b.as_ptr().add(base));

            dot_sum = vfmaq_f32(dot_sum, va, vb);
            norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
            norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
        }

        let mut dot = vaddvq_f32(dot_sum);
        let mut norm_a = vaddvq_f32(norm_a_sum);
        let mut norm_b = vaddvq_f32(norm_b_sum);

        let base = chunks * 4;
        for i in base..len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        (dot, norm_a, norm_b)
    }
}

// ============================================================================
// SIMD Manhattan Distance
// ============================================================================

/// Calculate Manhattan (L1) distance between two vectors (SIMD-optimized).
///
/// Returns Σ|a[i] - b[i]|
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { manhattan_distance_avx2(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "avx2")
    ))]
    {
        unsafe { manhattan_distance_sse(a, b) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { manhattan_distance_neon(a, b) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        manhattan_distance_scalar(a, b)
    }
}

#[inline]
fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    let mut sum = 0.0f32;

    for i in 0..chunks {
        let base = i * 8;
        let mut chunk_sum = 0.0f32;

        chunk_sum += (a[base] - b[base]).abs();
        chunk_sum += (a[base + 1] - b[base + 1]).abs();
        chunk_sum += (a[base + 2] - b[base + 2]).abs();
        chunk_sum += (a[base + 3] - b[base + 3]).abs();
        chunk_sum += (a[base + 4] - b[base + 4]).abs();
        chunk_sum += (a[base + 5] - b[base + 5]).abs();
        chunk_sum += (a[base + 6] - b[base + 6]).abs();
        chunk_sum += (a[base + 7] - b[base + 7]).abs();

        sum += chunk_sum;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        sum += (a[base + i] - b[base + i]).abs();
    }

    sum
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let mut sum = _mm256_setzero_ps();

    // Mask for clearing sign bit (absolute value)
    let sign_mask = _mm256_set1_ps(-0.0);

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);
        sum = _mm256_add_ps(sum, abs_diff);
    }

    let sum128_lo = _mm256_castps256_ps128(sum);
    let sum128_hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    let base = chunks * 8;
    for i in base..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline]
unsafe fn manhattan_distance_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 4;
    let mut sum = _mm_setzero_ps();

    let sign_mask = _mm_set1_ps(-0.0);

    for i in 0..chunks {
        let base = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(base));
        let vb = _mm_loadu_ps(b.as_ptr().add(base));
        let diff = _mm_sub_ps(va, vb);
        let abs_diff = _mm_andnot_ps(sign_mask, diff);
        sum = _mm_add_ps(sum, abs_diff);
    }

    let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    let base = chunks * 4;
    for i in base..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn manhattan_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    for i in 0..chunks {
        let base = i * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(base)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(base)) };
        let diff = unsafe { vsubq_f32(va, vb) };
        let abs_diff = unsafe { vabsq_f32(diff) };
        sum = unsafe { vaddq_f32(sum, abs_diff) };
    }

    let mut result = unsafe { vaddvq_f32(sum) };

    let base = chunks * 4;
    for i in base..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Calculate distances from a query vector to multiple target vectors (SIMD-optimized).
///
/// This is more efficient than calling distance functions individually due to
/// better cache utilization and reduced function call overhead.
pub fn batch_euclidean_distances(query: &[f32], targets: &[&[f32]]) -> Vec<f32> {
    targets
        .iter()
        .map(|t| euclidean_distance(query, t))
        .collect()
}

/// Calculate cosine similarities from a query vector to multiple targets.
pub fn batch_cosine_similarities(query: &[f32], targets: &[&[f32]]) -> Vec<f32> {
    // Pre-compute query norm for efficiency
    let query_norm = l2_norm(query);

    if query_norm == 0.0 {
        return vec![0.0; targets.len()];
    }

    targets
        .iter()
        .map(|t| {
            let dot = dot_product(query, t);
            let t_norm = l2_norm(t);
            if t_norm == 0.0 {
                0.0
            } else {
                dot / (query_norm * t_norm)
            }
        })
        .collect()
}

/// Calculate dot products from a query vector to multiple targets.
pub fn batch_dot_products(query: &[f32], targets: &[&[f32]]) -> Vec<f32> {
    targets.iter().map(|t| dot_product(query, t)).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];

        let dist = euclidean_distance(&a, &b);
        // sqrt((4-1)^2 + (3-2)^2 + (2-3)^2 + (1-4)^2) = sqrt(9+1+1+9) = sqrt(20)
        assert!(approx_eq(dist, 20.0_f32.sqrt()));
    }

    #[test]
    fn test_euclidean_distance_same_vector() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(euclidean_distance(&a, &a), 0.0));
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];

        // 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
        assert!(approx_eq(dot_product(&a, &b), 20.0));
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert!(approx_eq(dot_product(&a, &b), 0.0));
    }

    #[test]
    fn test_l2_norm() {
        let a = vec![3.0, 4.0];
        assert!(approx_eq(l2_norm(&a), 5.0)); // 3-4-5 triangle
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(cosine_similarity(&a, &a), 1.0));
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.0));
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![-1.0, -2.0, -3.0, -4.0];
        assert!(approx_eq(cosine_similarity(&a, &b), -1.0));
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];

        // |4-1| + |3-2| + |2-3| + |1-4| = 3 + 1 + 1 + 3 = 8
        assert!(approx_eq(manhattan_distance(&a, &b), 8.0));
    }

    #[test]
    fn test_large_vector() {
        // Test with a vector that triggers SIMD paths
        let size = 256;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        // Each element differs by 1, so Manhattan = 256, Euclidean = sqrt(256) = 16
        assert!(approx_eq(manhattan_distance(&a, &b), 256.0));
        assert!(approx_eq(euclidean_distance(&a, &b), 16.0));
    }

    #[test]
    fn test_non_aligned_vector() {
        // Test vectors that don't align to SIMD boundaries
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let dot = dot_product(&a, &b);
        // 1*7 + 2*6 + 3*5 + 4*4 + 5*3 + 6*2 + 7*1 = 7+12+15+16+15+12+7 = 84
        assert!(approx_eq(dot, 84.0));
    }

    #[test]
    fn test_empty_vector() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];

        assert!(approx_eq(euclidean_distance(&a, &b), 0.0));
        assert!(approx_eq(dot_product(&a, &b), 0.0));
        assert!(approx_eq(manhattan_distance(&a, &b), 0.0));
    }

    #[test]
    fn test_batch_operations() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let targets: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];
        let target_refs: Vec<&[f32]> = targets.iter().map(|v| v.as_slice()).collect();

        let distances = batch_euclidean_distances(&query, &target_refs);
        assert_eq!(distances.len(), 3);
        assert!(approx_eq(distances[0], 0.0)); // Same vector
        assert!(approx_eq(distances[1], 2.0_f32.sqrt())); // Orthogonal

        let similarities = batch_cosine_similarities(&query, &target_refs);
        assert!(approx_eq(similarities[0], 1.0)); // Identical
        assert!(approx_eq(similarities[1], 0.0)); // Orthogonal
    }

    #[test]
    fn test_zero_vector_cosine() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.0));
        assert!(approx_eq(cosine_similarity(&b, &a), 0.0));
    }
}
