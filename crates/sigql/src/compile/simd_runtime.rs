//! SIMD Runtime Execution
//!
//! Pre-compiled SIMD functions for common signal processing operations.
//! These functions are used at runtime to execute generated SIMD code.

use super::codegen::SimdFeature;

/// Type alias for SIMD processing functions
pub type SimdProcessFn = fn(&[f64], &mut [f64]);

/// SIMD operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimdOp {
    /// Element-wise absolute value
    Abs,
    /// Element-wise square
    Square,
    /// Element-wise square root
    Sqrt,
    /// Element-wise natural logarithm
    Log,
    /// Element-wise exponential
    Exp,
    /// Scale by constant
    Scale,
    /// Add constant offset
    Offset,
    /// Negate values
    Negate,
    /// Copy values (identity)
    Copy,
    /// Sum reduction
    Sum,
    /// Mean reduction
    Mean,
    /// Max reduction
    Max,
    /// Min reduction
    Min,
    /// RMS (root mean square)
    Rms,
    /// Variance
    Variance,
    /// Standard deviation
    Std,
}

/// Runtime SIMD executor
pub struct SimdRuntime {
    feature: SimdFeature,
}

impl SimdRuntime {
    /// Create a new SIMD runtime with auto-detected features
    pub fn new() -> Self {
        Self {
            feature: SimdFeature::detect(),
        }
    }

    /// Create with specific feature set
    pub fn with_feature(feature: SimdFeature) -> Self {
        Self { feature }
    }

    /// Get the execute function for a given operation
    pub fn get_execute_fn(&self, op: SimdOp) -> SimdProcessFn {
        match op {
            SimdOp::Abs => self.select_abs_fn(),
            SimdOp::Square => self.select_square_fn(),
            SimdOp::Sqrt => self.select_sqrt_fn(),
            SimdOp::Log => self.select_log_fn(),
            SimdOp::Exp => self.select_exp_fn(),
            SimdOp::Negate => self.select_negate_fn(),
            SimdOp::Copy => copy_scalar,
            SimdOp::Sum
            | SimdOp::Mean
            | SimdOp::Max
            | SimdOp::Min
            | SimdOp::Rms
            | SimdOp::Variance
            | SimdOp::Std => copy_scalar, // Reductions handled separately
            SimdOp::Scale | SimdOp::Offset => copy_scalar, // Need parameter
        }
    }

    /// Execute an operation with SIMD acceleration
    pub fn execute(&self, op: SimdOp, input: &[f64], output: &mut [f64]) {
        let func = self.get_execute_fn(op);
        func(input, output);
    }

    /// Execute a reduction operation
    pub fn reduce(&self, op: SimdOp, input: &[f64]) -> f64 {
        match op {
            SimdOp::Sum => self.simd_sum(input),
            SimdOp::Mean => self.simd_mean(input),
            SimdOp::Max => self.simd_max(input),
            SimdOp::Min => self.simd_min(input),
            SimdOp::Rms => self.simd_rms(input),
            SimdOp::Variance => self.simd_variance(input),
            SimdOp::Std => self.simd_std(input),
            _ => 0.0,
        }
    }

    /// Execute scale operation with parameter
    pub fn scale(&self, input: &[f64], output: &mut [f64], factor: f64) {
        match self.feature {
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Avx2 => unsafe { scale_avx2(input, output, factor) },
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Sse42 => unsafe { scale_sse42(input, output, factor) },
            #[cfg(target_arch = "aarch64")]
            SimdFeature::Neon => unsafe { scale_neon(input, output, factor) },
            _ => scale_scalar(input, output, factor),
        }
    }

    /// Execute offset operation with parameter
    pub fn offset(&self, input: &[f64], output: &mut [f64], value: f64) {
        match self.feature {
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Avx2 => unsafe { offset_avx2(input, output, value) },
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Sse42 => unsafe { offset_sse42(input, output, value) },
            #[cfg(target_arch = "aarch64")]
            SimdFeature::Neon => unsafe { offset_neon(input, output, value) },
            _ => offset_scalar(input, output, value),
        }
    }

    // Feature selection functions
    fn select_abs_fn(&self) -> SimdProcessFn {
        match self.feature {
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Avx2 => abs_avx2_wrapper,
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Sse42 => abs_sse42_wrapper,
            #[cfg(target_arch = "aarch64")]
            SimdFeature::Neon => abs_neon_wrapper,
            _ => abs_scalar,
        }
    }

    fn select_square_fn(&self) -> SimdProcessFn {
        match self.feature {
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Avx2 => square_avx2_wrapper,
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Sse42 => square_sse42_wrapper,
            #[cfg(target_arch = "aarch64")]
            SimdFeature::Neon => square_neon_wrapper,
            _ => square_scalar,
        }
    }

    fn select_sqrt_fn(&self) -> SimdProcessFn {
        match self.feature {
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Avx2 => sqrt_avx2_wrapper,
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Sse42 => sqrt_sse42_wrapper,
            #[cfg(target_arch = "aarch64")]
            SimdFeature::Neon => sqrt_neon_wrapper,
            _ => sqrt_scalar,
        }
    }

    fn select_log_fn(&self) -> SimdProcessFn {
        // Log is typically not available as SIMD intrinsic, use scalar
        log_scalar
    }

    fn select_exp_fn(&self) -> SimdProcessFn {
        // Exp is typically not available as SIMD intrinsic, use scalar
        exp_scalar
    }

    fn select_negate_fn(&self) -> SimdProcessFn {
        match self.feature {
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Avx2 => negate_avx2_wrapper,
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Sse42 => negate_sse42_wrapper,
            #[cfg(target_arch = "aarch64")]
            SimdFeature::Neon => negate_neon_wrapper,
            _ => negate_scalar,
        }
    }

    // SIMD reduction implementations
    fn simd_sum(&self, input: &[f64]) -> f64 {
        match self.feature {
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Avx2 => unsafe { sum_avx2(input) },
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Sse42 => unsafe { sum_sse42(input) },
            _ => input.iter().sum(),
        }
    }

    fn simd_mean(&self, input: &[f64]) -> f64 {
        if input.is_empty() {
            return 0.0;
        }
        self.simd_sum(input) / input.len() as f64
    }

    fn simd_max(&self, input: &[f64]) -> f64 {
        match self.feature {
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Avx2 => unsafe { max_avx2(input) },
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Sse42 => unsafe { max_sse42(input) },
            _ => input.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }

    fn simd_min(&self, input: &[f64]) -> f64 {
        match self.feature {
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Avx2 => unsafe { min_avx2(input) },
            #[cfg(target_arch = "x86_64")]
            SimdFeature::Sse42 => unsafe { min_sse42(input) },
            _ => input.iter().cloned().fold(f64::INFINITY, f64::min),
        }
    }

    fn simd_rms(&self, input: &[f64]) -> f64 {
        if input.is_empty() {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for &x in input {
            sum_sq += x * x;
        }
        (sum_sq / input.len() as f64).sqrt()
    }

    fn simd_variance(&self, input: &[f64]) -> f64 {
        if input.len() < 2 {
            return 0.0;
        }
        let mean = self.simd_mean(input);
        let sum_sq: f64 = input.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_sq / (input.len() - 1) as f64
    }

    fn simd_std(&self, input: &[f64]) -> f64 {
        self.simd_variance(input).sqrt()
    }
}

impl Default for SimdRuntime {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Scalar implementations (fallback)
// ============================================================================

fn copy_scalar(input: &[f64], output: &mut [f64]) {
    let n = input.len().min(output.len());
    output[..n].copy_from_slice(&input[..n]);
}

fn abs_scalar(input: &[f64], output: &mut [f64]) {
    let n = input.len().min(output.len());
    for i in 0..n {
        output[i] = input[i].abs();
    }
}

fn square_scalar(input: &[f64], output: &mut [f64]) {
    let n = input.len().min(output.len());
    for i in 0..n {
        output[i] = input[i] * input[i];
    }
}

fn sqrt_scalar(input: &[f64], output: &mut [f64]) {
    let n = input.len().min(output.len());
    for i in 0..n {
        output[i] = input[i].sqrt();
    }
}

fn log_scalar(input: &[f64], output: &mut [f64]) {
    let n = input.len().min(output.len());
    for i in 0..n {
        output[i] = input[i].ln();
    }
}

fn exp_scalar(input: &[f64], output: &mut [f64]) {
    let n = input.len().min(output.len());
    for i in 0..n {
        output[i] = input[i].exp();
    }
}

fn negate_scalar(input: &[f64], output: &mut [f64]) {
    let n = input.len().min(output.len());
    for i in 0..n {
        output[i] = -input[i];
    }
}

fn scale_scalar(input: &[f64], output: &mut [f64], factor: f64) {
    let n = input.len().min(output.len());
    for i in 0..n {
        output[i] = input[i] * factor;
    }
}

fn offset_scalar(input: &[f64], output: &mut [f64], value: f64) {
    let n = input.len().min(output.len());
    for i in 0..n {
        output[i] = input[i] + value;
    }
}

// ============================================================================
// x86_64 AVX2 implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn abs_avx2_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { abs_avx2(input, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn abs_avx2(input: &[f64], output: &mut [f64]) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 4);

    let sign_mask = _mm256_set1_pd(-0.0);

    let mut i = 0;
    while i < simd_end {
        let v = _mm256_loadu_pd(input.as_ptr().add(i));
        let result = _mm256_andnot_pd(sign_mask, v);
        _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    // Scalar tail
    for j in simd_end..n {
        output[j] = input[j].abs();
    }
}

#[cfg(target_arch = "x86_64")]
fn square_avx2_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { square_avx2(input, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn square_avx2(input: &[f64], output: &mut [f64]) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 4);

    let mut i = 0;
    while i < simd_end {
        let v = _mm256_loadu_pd(input.as_ptr().add(i));
        let result = _mm256_mul_pd(v, v);
        _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    for j in simd_end..n {
        output[j] = input[j] * input[j];
    }
}

#[cfg(target_arch = "x86_64")]
fn sqrt_avx2_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { sqrt_avx2(input, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sqrt_avx2(input: &[f64], output: &mut [f64]) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 4);

    let mut i = 0;
    while i < simd_end {
        let v = _mm256_loadu_pd(input.as_ptr().add(i));
        let result = _mm256_sqrt_pd(v);
        _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    for j in simd_end..n {
        output[j] = input[j].sqrt();
    }
}

#[cfg(target_arch = "x86_64")]
fn negate_avx2_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { negate_avx2(input, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn negate_avx2(input: &[f64], output: &mut [f64]) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 4);

    let sign_mask = _mm256_set1_pd(-0.0);

    let mut i = 0;
    while i < simd_end {
        let v = _mm256_loadu_pd(input.as_ptr().add(i));
        let result = _mm256_xor_pd(v, sign_mask);
        _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    for j in simd_end..n {
        output[j] = -input[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2(input: &[f64], output: &mut [f64], factor: f64) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 4);

    let scale_v = _mm256_set1_pd(factor);

    let mut i = 0;
    while i < simd_end {
        let v = _mm256_loadu_pd(input.as_ptr().add(i));
        let result = _mm256_mul_pd(v, scale_v);
        _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    for j in simd_end..n {
        output[j] = input[j] * factor;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn offset_avx2(input: &[f64], output: &mut [f64], value: f64) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 4);

    let offset_v = _mm256_set1_pd(value);

    let mut i = 0;
    while i < simd_end {
        let v = _mm256_loadu_pd(input.as_ptr().add(i));
        let result = _mm256_add_pd(v, offset_v);
        _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    for j in simd_end..n {
        output[j] = input[j] + value;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(input: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let n = input.len();
    let simd_end = n - (n % 4);

    let mut acc = _mm256_setzero_pd();

    let mut i = 0;
    while i < simd_end {
        let v = _mm256_loadu_pd(input.as_ptr().add(i));
        acc = _mm256_add_pd(acc, v);
        i += 4;
    }

    // Horizontal sum
    let hi = _mm256_extractf128_pd(acc, 1);
    let lo = _mm256_castpd256_pd128(acc);
    let sum128 = _mm_add_pd(hi, lo);
    let sum64 = _mm_add_sd(sum128, _mm_unpackhi_pd(sum128, sum128));
    let mut result = _mm_cvtsd_f64(sum64);

    // Add scalar tail
    for j in simd_end..n {
        result += input[j];
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_avx2(input: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    if input.is_empty() {
        return f64::NEG_INFINITY;
    }

    let n = input.len();
    let simd_end = n - (n % 4);

    let mut acc = _mm256_set1_pd(f64::NEG_INFINITY);

    let mut i = 0;
    while i < simd_end {
        let v = _mm256_loadu_pd(input.as_ptr().add(i));
        acc = _mm256_max_pd(acc, v);
        i += 4;
    }

    // Horizontal max
    let hi = _mm256_extractf128_pd(acc, 1);
    let lo = _mm256_castpd256_pd128(acc);
    let max128 = _mm_max_pd(hi, lo);
    let max64 = _mm_max_sd(max128, _mm_unpackhi_pd(max128, max128));
    let mut result = _mm_cvtsd_f64(max64);

    // Check scalar tail
    for j in simd_end..n {
        if input[j] > result {
            result = input[j];
        }
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn min_avx2(input: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    if input.is_empty() {
        return f64::INFINITY;
    }

    let n = input.len();
    let simd_end = n - (n % 4);

    let mut acc = _mm256_set1_pd(f64::INFINITY);

    let mut i = 0;
    while i < simd_end {
        let v = _mm256_loadu_pd(input.as_ptr().add(i));
        acc = _mm256_min_pd(acc, v);
        i += 4;
    }

    // Horizontal min
    let hi = _mm256_extractf128_pd(acc, 1);
    let lo = _mm256_castpd256_pd128(acc);
    let min128 = _mm_min_pd(hi, lo);
    let min64 = _mm_min_sd(min128, _mm_unpackhi_pd(min128, min128));
    let mut result = _mm_cvtsd_f64(min64);

    // Check scalar tail
    for j in simd_end..n {
        if input[j] < result {
            result = input[j];
        }
    }

    result
}

// ============================================================================
// x86_64 SSE4.2 implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn abs_sse42_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { abs_sse42(input, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn abs_sse42(input: &[f64], output: &mut [f64]) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 2);

    let sign_mask = _mm_set1_pd(-0.0);

    let mut i = 0;
    while i < simd_end {
        let v = _mm_loadu_pd(input.as_ptr().add(i));
        let result = _mm_andnot_pd(sign_mask, v);
        _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 2;
    }

    for j in simd_end..n {
        output[j] = input[j].abs();
    }
}

#[cfg(target_arch = "x86_64")]
fn square_sse42_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { square_sse42(input, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn square_sse42(input: &[f64], output: &mut [f64]) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 2);

    let mut i = 0;
    while i < simd_end {
        let v = _mm_loadu_pd(input.as_ptr().add(i));
        let result = _mm_mul_pd(v, v);
        _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 2;
    }

    for j in simd_end..n {
        output[j] = input[j] * input[j];
    }
}

#[cfg(target_arch = "x86_64")]
fn sqrt_sse42_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { sqrt_sse42(input, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn sqrt_sse42(input: &[f64], output: &mut [f64]) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 2);

    let mut i = 0;
    while i < simd_end {
        let v = _mm_loadu_pd(input.as_ptr().add(i));
        let result = _mm_sqrt_pd(v);
        _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 2;
    }

    for j in simd_end..n {
        output[j] = input[j].sqrt();
    }
}

#[cfg(target_arch = "x86_64")]
fn negate_sse42_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { negate_sse42(input, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn negate_sse42(input: &[f64], output: &mut [f64]) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 2);

    let sign_mask = _mm_set1_pd(-0.0);

    let mut i = 0;
    while i < simd_end {
        let v = _mm_loadu_pd(input.as_ptr().add(i));
        let result = _mm_xor_pd(v, sign_mask);
        _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 2;
    }

    for j in simd_end..n {
        output[j] = -input[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn scale_sse42(input: &[f64], output: &mut [f64], factor: f64) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 2);

    let scale_v = _mm_set1_pd(factor);

    let mut i = 0;
    while i < simd_end {
        let v = _mm_loadu_pd(input.as_ptr().add(i));
        let result = _mm_mul_pd(v, scale_v);
        _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 2;
    }

    for j in simd_end..n {
        output[j] = input[j] * factor;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn offset_sse42(input: &[f64], output: &mut [f64], value: f64) {
    use std::arch::x86_64::*;

    let n = input.len().min(output.len());
    let simd_end = n - (n % 2);

    let offset_v = _mm_set1_pd(value);

    let mut i = 0;
    while i < simd_end {
        let v = _mm_loadu_pd(input.as_ptr().add(i));
        let result = _mm_add_pd(v, offset_v);
        _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        i += 2;
    }

    for j in simd_end..n {
        output[j] = input[j] + value;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn sum_sse42(input: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let n = input.len();
    let simd_end = n - (n % 2);

    let mut acc = _mm_setzero_pd();

    let mut i = 0;
    while i < simd_end {
        let v = _mm_loadu_pd(input.as_ptr().add(i));
        acc = _mm_add_pd(acc, v);
        i += 2;
    }

    // Horizontal sum
    let sum = _mm_add_sd(acc, _mm_unpackhi_pd(acc, acc));
    let mut result = _mm_cvtsd_f64(sum);

    for j in simd_end..n {
        result += input[j];
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn max_sse42(input: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    if input.is_empty() {
        return f64::NEG_INFINITY;
    }

    let n = input.len();
    let simd_end = n - (n % 2);

    let mut acc = _mm_set1_pd(f64::NEG_INFINITY);

    let mut i = 0;
    while i < simd_end {
        let v = _mm_loadu_pd(input.as_ptr().add(i));
        acc = _mm_max_pd(acc, v);
        i += 2;
    }

    let max = _mm_max_sd(acc, _mm_unpackhi_pd(acc, acc));
    let mut result = _mm_cvtsd_f64(max);

    for j in simd_end..n {
        if input[j] > result {
            result = input[j];
        }
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn min_sse42(input: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    if input.is_empty() {
        return f64::INFINITY;
    }

    let n = input.len();
    let simd_end = n - (n % 2);

    let mut acc = _mm_set1_pd(f64::INFINITY);

    let mut i = 0;
    while i < simd_end {
        let v = _mm_loadu_pd(input.as_ptr().add(i));
        acc = _mm_min_pd(acc, v);
        i += 2;
    }

    let min = _mm_min_sd(acc, _mm_unpackhi_pd(acc, acc));
    let mut result = _mm_cvtsd_f64(min);

    for j in simd_end..n {
        if input[j] < result {
            result = input[j];
        }
    }

    result
}

// ============================================================================
// ARM NEON implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
fn abs_neon_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { abs_neon(input, output) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn abs_neon(input: &[f64], output: &mut [f64]) {
    unsafe {
        use std::arch::aarch64::*;

        let n = input.len().min(output.len());
        let simd_end = n - (n % 2);

        let mut i = 0;
        while i < simd_end {
            let v = vld1q_f64(input.as_ptr().add(i));
            let result = vabsq_f64(v);
            vst1q_f64(output.as_mut_ptr().add(i), result);
            i += 2;
        }

        for j in simd_end..n {
            output[j] = input[j].abs();
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn square_neon_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { square_neon(input, output) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn square_neon(input: &[f64], output: &mut [f64]) {
    unsafe {
        use std::arch::aarch64::*;

        let n = input.len().min(output.len());
        let simd_end = n - (n % 2);

        let mut i = 0;
        while i < simd_end {
            let v = vld1q_f64(input.as_ptr().add(i));
            let result = vmulq_f64(v, v);
            vst1q_f64(output.as_mut_ptr().add(i), result);
            i += 2;
        }

        for j in simd_end..n {
            output[j] = input[j] * input[j];
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn sqrt_neon_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { sqrt_neon(input, output) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sqrt_neon(input: &[f64], output: &mut [f64]) {
    unsafe {
        use std::arch::aarch64::*;

        let n = input.len().min(output.len());
        let simd_end = n - (n % 2);

        let mut i = 0;
        while i < simd_end {
            let v = vld1q_f64(input.as_ptr().add(i));
            let result = vsqrtq_f64(v);
            vst1q_f64(output.as_mut_ptr().add(i), result);
            i += 2;
        }

        for j in simd_end..n {
            output[j] = input[j].sqrt();
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn negate_neon_wrapper(input: &[f64], output: &mut [f64]) {
    unsafe { negate_neon(input, output) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn negate_neon(input: &[f64], output: &mut [f64]) {
    unsafe {
        use std::arch::aarch64::*;

        let n = input.len().min(output.len());
        let simd_end = n - (n % 2);

        let mut i = 0;
        while i < simd_end {
            let v = vld1q_f64(input.as_ptr().add(i));
            let result = vnegq_f64(v);
            vst1q_f64(output.as_mut_ptr().add(i), result);
            i += 2;
        }

        for j in simd_end..n {
            output[j] = -input[j];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_neon(input: &[f64], output: &mut [f64], factor: f64) {
    unsafe {
        use std::arch::aarch64::*;

        let n = input.len().min(output.len());
        let simd_end = n - (n % 2);

        let scale_v = vdupq_n_f64(factor);

        let mut i = 0;
        while i < simd_end {
            let v = vld1q_f64(input.as_ptr().add(i));
            let result = vmulq_f64(v, scale_v);
            vst1q_f64(output.as_mut_ptr().add(i), result);
            i += 2;
        }

        for j in simd_end..n {
            output[j] = input[j] * factor;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn offset_neon(input: &[f64], output: &mut [f64], value: f64) {
    unsafe {
        use std::arch::aarch64::*;

        let n = input.len().min(output.len());
        let simd_end = n - (n % 2);

        let offset_v = vdupq_n_f64(value);

        let mut i = 0;
        while i < simd_end {
            let v = vld1q_f64(input.as_ptr().add(i));
            let result = vaddq_f64(v, offset_v);
            vst1q_f64(output.as_mut_ptr().add(i), result);
            i += 2;
        }

        for j in simd_end..n {
            output[j] = input[j] + value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_runtime_creation() {
        let runtime = SimdRuntime::new();
        // Should not panic
        let _fn = runtime.get_execute_fn(SimdOp::Abs);
    }

    #[test]
    fn test_abs_scalar() {
        let input = vec![-1.0, 2.0, -3.0, 4.0];
        let mut output = vec![0.0; 4];
        abs_scalar(&input, &mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_square_scalar() {
        let input = vec![2.0, 3.0, 4.0];
        let mut output = vec![0.0; 3];
        square_scalar(&input, &mut output);
        assert_eq!(output, vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_runtime_execute() {
        let runtime = SimdRuntime::new();
        let input = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
        let mut output = vec![0.0; 8];

        runtime.execute(SimdOp::Abs, &input, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, input[i].abs());
        }
    }

    #[test]
    fn test_runtime_reduce() {
        let runtime = SimdRuntime::new();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((runtime.reduce(SimdOp::Sum, &input) - 15.0).abs() < 1e-10);
        assert!((runtime.reduce(SimdOp::Mean, &input) - 3.0).abs() < 1e-10);
        assert!((runtime.reduce(SimdOp::Max, &input) - 5.0).abs() < 1e-10);
        assert!((runtime.reduce(SimdOp::Min, &input) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale() {
        let runtime = SimdRuntime::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        runtime.scale(&input, &mut output, 2.0);

        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_offset() {
        let runtime = SimdRuntime::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        runtime.offset(&input, &mut output, 10.0);

        assert_eq!(output, vec![11.0, 12.0, 13.0, 14.0]);
    }
}
