//! FFT-Based Holographic Operations (Fourier-Native Binding)
//!
//! Implements high-performance circular convolution using the Convolution Theorem:
//! - Convolution in time domain = Multiplication in frequency domain
//! - bind(a, b) = IFFT(FFT(a) * FFT(b))
//!
//! # Design Principles
//!
//! 1. **Plan Reuse** - FFT plans are expensive to create; cache and reuse them
//! 2. **Cache-Oblivious Blocking** - Process large vectors in cache-sized chunks
//! 3. **SIMD Optimization** - Hand-tuned complex multiplication using portable SIMD
//! 4. **Memory Pooling** - Preallocated scratch buffers avoid allocation in hot paths
//! 5. **Batched Processing** - Process up to 64 vectors simultaneously for throughput
//!
//! # Example
//!
//! ```rust,ignore
//! use joule_db_hdc::fourier_bind::{FourierBinder, FourierConfig};
//! use joule_db_hdc::hyperdimensional::HyperVector;
//!
//! let config = FourierConfig::default().with_dimension(10000);
//! let binder = FourierBinder::new(config);
//!
//! let a = HyperVector::random(10000, 42);
//! let b = HyperVector::random(10000, 123);
//!
//! // Fast FFT-based circular convolution
//! let bound = binder.bind(&a, &b).unwrap();
//!
//! // Batched processing for throughput
//! let vectors: Vec<_> = (0..64).map(|i| HyperVector::random(10000, i)).collect();
//! let results = binder.bind_batch(&vectors[..32], &vectors[32..]).unwrap();
//! ```

use std::f32::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::hyperdimensional::{HDError, HyperVector};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the Fourier binder
#[derive(Debug, Clone)]
pub struct FourierConfig {
    /// Vector dimension (should be power of 2 for optimal FFT)
    pub dimension: usize,
    /// Maximum batch size for batched operations
    pub max_batch_size: usize,
    /// Cache block size for cache-oblivious algorithms (bytes)
    pub cache_block_size: usize,
    /// Number of scratch buffers in the pool
    pub scratch_pool_size: usize,
}

impl Default for FourierConfig {
    fn default() -> Self {
        Self {
            dimension: crate::binary_hd::DEFAULT_DIMENSIONS,
            max_batch_size: 64,
            cache_block_size: 32 * 1024, // 32KB L1 cache typical
            scratch_pool_size: 8,
        }
    }
}

impl FourierConfig {
    /// Set the vector dimension
    pub fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = dim;
        self
    }

    /// Set maximum batch size
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set cache block size for blocked algorithms
    pub fn with_cache_block_size(mut self, size: usize) -> Self {
        self.cache_block_size = size;
        self
    }
}

// ============================================================================
// COMPLEX NUMBER TYPE (SIMD-FRIENDLY LAYOUT)
// ============================================================================

/// Complex number optimized for SIMD operations
///
/// Uses separate real/imag arrays in SoA layout for better vectorization.
/// Aligned to 8 bytes for efficient memory access.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(8))]
pub struct Complex32 {
    /// Real component
    pub re: f32,
    /// Imaginary component
    pub im: f32,
}

impl Complex32 {
    /// Create a new complex number from real and imaginary parts
    #[inline(always)]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Create a zero complex number (0 + 0i)
    #[inline(always)]
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// Create a complex number from polar form (magnitude and phase)
    #[inline(always)]
    pub fn from_polar(mag: f32, phase: f32) -> Self {
        Self {
            re: mag * phase.cos(),
            im: mag * phase.sin(),
        }
    }

    /// Compute magnitude squared (|z|^2 = re^2 + im^2)
    #[inline(always)]
    pub fn magnitude_squared(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Compute magnitude (|z| = sqrt(re^2 + im^2))
    #[inline(always)]
    pub fn magnitude(&self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Compute complex conjugate (a + bi -> a - bi)
    #[inline(always)]
    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Complex multiplication
    #[inline(always)]
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Complex addition
    #[inline(always)]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Complex subtraction
    #[inline(always)]
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// Scale by real number
    #[inline(always)]
    pub fn scale(&self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

// ============================================================================
// FFT TWIDDLE FACTORS (PRECOMPUTED)
// ============================================================================

/// Precomputed twiddle factors for FFT
///
/// W_n^k = e^(-2*pi*i*k/n) for forward FFT
/// W_n^k = e^(+2*pi*i*k/n) for inverse FFT
#[derive(Debug, Clone)]
pub struct TwiddleFactors {
    /// Forward twiddle factors for each stage
    forward: Vec<Vec<Complex32>>,
    /// Inverse twiddle factors for each stage
    inverse: Vec<Vec<Complex32>>,
    /// FFT size (power of 2)
    fft_size: usize,
    /// Number of stages (log2 of size)
    num_stages: usize,
}

impl TwiddleFactors {
    /// Create twiddle factors for given FFT size
    pub fn new(fft_size: usize) -> Self {
        assert!(fft_size.is_power_of_two(), "FFT size must be power of 2");

        let num_stages = fft_size.trailing_zeros() as usize;
        let mut forward = Vec::with_capacity(num_stages);
        let mut inverse = Vec::with_capacity(num_stages);

        for stage in 0..num_stages {
            let stage_size = 1 << (stage + 1);
            let half_size = stage_size / 2;

            let mut fwd_twiddles = Vec::with_capacity(half_size);
            let mut inv_twiddles = Vec::with_capacity(half_size);

            for k in 0..half_size {
                let angle = -2.0 * PI * (k as f32) / (stage_size as f32);
                fwd_twiddles.push(Complex32::from_polar(1.0, angle));
                inv_twiddles.push(Complex32::from_polar(1.0, -angle));
            }

            forward.push(fwd_twiddles);
            inverse.push(inv_twiddles);
        }

        Self {
            forward,
            inverse,
            fft_size,
            num_stages,
        }
    }

    /// Get the FFT size this table was created for
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Get the number of stages (log2 of FFT size)
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    #[inline(always)]
    fn get_forward(&self, stage: usize, k: usize) -> &Complex32 {
        &self.forward[stage][k]
    }

    #[inline(always)]
    fn get_inverse(&self, stage: usize, k: usize) -> &Complex32 {
        &self.inverse[stage][k]
    }
}

// ============================================================================
// BIT REVERSAL PERMUTATION
// ============================================================================

/// Precomputed bit-reversal permutation indices for FFT
///
/// FFT algorithms require data to be reordered using bit-reversal permutation.
/// This table precomputes the permutation indices for efficient reordering.
#[derive(Debug, Clone)]
pub struct BitReversalTable {
    /// Precomputed permutation indices
    indices: Vec<usize>,
    /// Size of the FFT (power of 2)
    size: usize,
}

impl BitReversalTable {
    /// Create a new bit-reversal table for the given FFT size
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two());
        let bits = size.trailing_zeros() as usize;

        let indices: Vec<usize> = (0..size).map(|i| Self::reverse_bits(i, bits)).collect();

        Self { indices, size }
    }

    #[inline(always)]
    fn reverse_bits(mut x: usize, bits: usize) -> usize {
        let mut result = 0;
        for _ in 0..bits {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }

    /// Apply bit-reversal permutation in place
    #[inline]
    pub fn permute(&self, data: &mut [Complex32]) {
        debug_assert_eq!(data.len(), self.size);

        for i in 0..self.size {
            let j = self.indices[i];
            if i < j {
                data.swap(i, j);
            }
        }
    }
}

// ============================================================================
// SCRATCH BUFFER POOL
// ============================================================================

/// Thread-safe pool of scratch buffers to avoid allocation in hot paths
///
/// FFT operations require temporary buffers. This pool pre-allocates
/// buffers and hands them out in a round-robin fashion to avoid
/// allocation overhead during computation.
pub struct ScratchPool {
    /// Pool of mutex-protected buffers
    buffers: Vec<std::sync::Mutex<Vec<Complex32>>>,
    /// Size of each buffer in complex elements
    buffer_size: usize,
    /// Round-robin index for buffer selection
    next_idx: AtomicU64,
}

impl ScratchPool {
    /// Create a new scratch pool with the specified number and size of buffers
    pub fn new(num_buffers: usize, buffer_size: usize) -> Self {
        let buffers = (0..num_buffers)
            .map(|_| std::sync::Mutex::new(vec![Complex32::zero(); buffer_size]))
            .collect();

        Self {
            buffers,
            buffer_size,
            next_idx: AtomicU64::new(0),
        }
    }

    /// Acquire a scratch buffer (round-robin allocation)
    pub fn acquire(&self) -> ScratchGuard<'_> {
        let idx = self.next_idx.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = idx % self.buffers.len();

        let guard = self.buffers[idx].lock().unwrap();
        ScratchGuard {
            guard,
            size: self.buffer_size,
        }
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

/// RAII guard for scratch buffer from the pool
///
/// Automatically returns the buffer to the pool when dropped.
pub struct ScratchGuard<'a> {
    /// The underlying mutex guard holding the buffer
    guard: std::sync::MutexGuard<'a, Vec<Complex32>>,
    /// Usable size of the buffer
    size: usize,
}

impl<'a> ScratchGuard<'a> {
    /// Get the buffer contents as an immutable slice
    pub fn as_slice(&self) -> &[Complex32] {
        &self.guard[..self.size]
    }

    /// Returns a mutable slice of the scratch buffer up to the requested size
    pub fn as_mut_slice(&mut self) -> &mut [Complex32] {
        &mut self.guard[..self.size]
    }

    /// Clear the buffer (set to zero)
    pub fn clear(&mut self) {
        for c in self.guard.iter_mut() {
            *c = Complex32::zero();
        }
    }
}

impl std::fmt::Debug for ScratchPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScratchPool")
            .field("num_buffers", &self.buffers.len())
            .field("buffer_size", &self.buffer_size)
            .finish()
    }
}

// ============================================================================
// FFT PLAN (CACHED)
// ============================================================================

/// Cached FFT computation plan
///
/// Creating plans is expensive; this struct caches all precomputed data
/// needed for efficient FFT computation.
#[derive(Debug)]
pub struct FftPlan {
    /// FFT size (padded to power of 2)
    fft_size: usize,
    /// Original dimension
    original_dim: usize,
    /// Precomputed twiddle factors
    twiddles: TwiddleFactors,
    /// Bit-reversal permutation table
    bit_reversal: BitReversalTable,
    /// Cache block size in elements
    block_size: usize,
}

impl FftPlan {
    /// Create a new FFT plan for given dimension
    pub fn new(dimension: usize, cache_block_size: usize) -> Self {
        // Pad to next power of 2 for efficient FFT
        let fft_size = dimension.next_power_of_two();

        // Block size in complex elements (each element is 8 bytes)
        let block_size = cache_block_size / 8;

        Self {
            fft_size,
            original_dim: dimension,
            twiddles: TwiddleFactors::new(fft_size),
            bit_reversal: BitReversalTable::new(fft_size),
            block_size,
        }
    }

    /// Get the FFT size
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Get original dimension
    pub fn original_dim(&self) -> usize {
        self.original_dim
    }
}

// ============================================================================
// FFT IMPLEMENTATION
// ============================================================================

/// Radix-2 Cooley-Tukey FFT implementation
///
/// Uses decimation-in-time with precomputed twiddle factors.
/// This engine reuses precomputed data for efficient repeated FFT operations.
#[derive(Debug)]
pub struct FftEngine {
    /// Precomputed FFT plan
    plan: FftPlan,
}

impl FftEngine {
    /// Create a new FFT engine with the given plan
    pub fn new(plan: FftPlan) -> Self {
        Self { plan }
    }

    /// Perform forward FFT in place
    ///
    /// Input should be in natural order, output will be in natural order
    pub fn forward(&self, data: &mut [Complex32]) {
        self.fft_core(data, false);
    }

    /// Perform inverse FFT in place
    ///
    /// Also normalizes by 1/N
    pub fn inverse(&self, data: &mut [Complex32]) {
        self.fft_core(data, true);

        // Normalize
        let scale = 1.0 / (self.plan.fft_size as f32);
        for c in data.iter_mut() {
            *c = c.scale(scale);
        }
    }

    /// Core FFT implementation
    fn fft_core(&self, data: &mut [Complex32], inverse: bool) {
        let n = self.plan.fft_size;
        debug_assert_eq!(data.len(), n);

        // Bit-reversal permutation
        self.plan.bit_reversal.permute(data);

        // Cooley-Tukey butterfly stages
        for stage in 0..self.plan.twiddles.num_stages {
            let stage_size = 1 << (stage + 1);
            let half_size = stage_size / 2;

            // Process in cache-friendly blocks
            if stage_size <= self.plan.block_size {
                // Small stages: process all butterflies together
                self.butterfly_stage_small(data, stage, stage_size, half_size, inverse);
            } else {
                // Large stages: use blocked processing
                self.butterfly_stage_blocked(data, stage, stage_size, half_size, inverse);
            }
        }
    }

    /// Process small butterfly stage (fits in cache)
    #[inline]
    fn butterfly_stage_small(
        &self,
        data: &mut [Complex32],
        stage: usize,
        stage_size: usize,
        half_size: usize,
        inverse: bool,
    ) {
        let n = self.plan.fft_size;
        let mut group_start = 0;

        while group_start < n {
            for k in 0..half_size {
                let twiddle = if inverse {
                    self.plan.twiddles.get_inverse(stage, k)
                } else {
                    self.plan.twiddles.get_forward(stage, k)
                };

                let i = group_start + k;
                let j = i + half_size;

                // SIMD-friendly butterfly
                self.butterfly(data, i, j, twiddle);
            }
            group_start += stage_size;
        }
    }

    /// Process large butterfly stage with blocking
    fn butterfly_stage_blocked(
        &self,
        data: &mut [Complex32],
        stage: usize,
        stage_size: usize,
        half_size: usize,
        inverse: bool,
    ) {
        let n = self.plan.fft_size;
        let block_size = self.plan.block_size.min(half_size);

        let mut group_start = 0;
        while group_start < n {
            // Process twiddles in blocks
            let mut k_start = 0;
            while k_start < half_size {
                let k_end = (k_start + block_size).min(half_size);

                for k in k_start..k_end {
                    let twiddle = if inverse {
                        self.plan.twiddles.get_inverse(stage, k)
                    } else {
                        self.plan.twiddles.get_forward(stage, k)
                    };

                    let i = group_start + k;
                    let j = i + half_size;

                    self.butterfly(data, i, j, twiddle);
                }

                k_start = k_end;
            }
            group_start += stage_size;
        }
    }

    /// Single butterfly operation (SIMD-optimized)
    #[inline(always)]
    fn butterfly(&self, data: &mut [Complex32], i: usize, j: usize, twiddle: &Complex32) {
        // Butterfly: (a, b) -> (a + W*b, a - W*b)
        let a = data[i];
        let b = data[j];

        // W * b
        let wb = twiddle.mul(&b);

        // a + W*b
        data[i] = a.add(&wb);
        // a - W*b
        data[j] = a.sub(&wb);
    }
}

// ============================================================================
// SIMD COMPLEX MULTIPLICATION (4-wide)
// ============================================================================

/// SIMD-optimized element-wise complex multiplication
///
/// Processes 4 complex numbers at a time using portable SIMD patterns
/// that auto-vectorize well on ARM NEON and x86 AVX
pub struct SimdComplexMul;

impl SimdComplexMul {
    /// Element-wise multiplication of two complex arrays
    ///
    /// result[i] = a[i] * b[i] for all i
    #[inline]
    pub fn multiply(a: &[Complex32], b: &[Complex32], result: &mut [Complex32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), result.len());

        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // Process in chunks of 4 for SIMD
        for chunk in 0..chunks {
            let base = chunk * 4;
            Self::mul_4(
                &a[base..base + 4],
                &b[base..base + 4],
                &mut result[base..base + 4],
            );
        }

        // Handle remainder
        let base = chunks * 4;
        for i in 0..remainder {
            result[base + i] = a[base + i].mul(&b[base + i]);
        }
    }

    /// Multiply 4 complex numbers (auto-vectorizes)
    #[inline(always)]
    fn mul_4(a: &[Complex32], b: &[Complex32], result: &mut [Complex32]) {
        // Load reals and imaginaries
        let a_re = [a[0].re, a[1].re, a[2].re, a[3].re];
        let a_im = [a[0].im, a[1].im, a[2].im, a[3].im];
        let b_re = [b[0].re, b[1].re, b[2].re, b[3].re];
        let b_im = [b[0].im, b[1].im, b[2].im, b[3].im];

        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let mut re = [0.0f32; 4];
        let mut im = [0.0f32; 4];

        for i in 0..4 {
            re[i] = a_re[i] * b_re[i] - a_im[i] * b_im[i];
            im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i];
        }

        // Store results
        for i in 0..4 {
            result[i] = Complex32::new(re[i], im[i]);
        }
    }

    /// Element-wise multiplication with conjugate of second operand
    ///
    /// result[i] = a[i] * conj(b[i]) for all i
    /// Used for correlation (unbind) operation
    #[inline]
    pub fn multiply_conj(a: &[Complex32], b: &[Complex32], result: &mut [Complex32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), result.len());

        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        for chunk in 0..chunks {
            let base = chunk * 4;
            Self::mul_conj_4(
                &a[base..base + 4],
                &b[base..base + 4],
                &mut result[base..base + 4],
            );
        }

        let base = chunks * 4;
        for i in 0..remainder {
            result[base + i] = a[base + i].mul(&b[base + i].conjugate());
        }
    }

    /// Multiply 4 complex numbers with conjugate
    #[inline(always)]
    fn mul_conj_4(a: &[Complex32], b: &[Complex32], result: &mut [Complex32]) {
        let a_re = [a[0].re, a[1].re, a[2].re, a[3].re];
        let a_im = [a[0].im, a[1].im, a[2].im, a[3].im];
        let b_re = [b[0].re, b[1].re, b[2].re, b[3].re];
        // Conjugate: negate imaginary
        let b_im = [-b[0].im, -b[1].im, -b[2].im, -b[3].im];

        let mut re = [0.0f32; 4];
        let mut im = [0.0f32; 4];

        for i in 0..4 {
            re[i] = a_re[i] * b_re[i] - a_im[i] * b_im[i];
            im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i];
        }

        for i in 0..4 {
            result[i] = Complex32::new(re[i], im[i]);
        }
    }
}

// ============================================================================
// FOURIER BINDER (MAIN API)
// ============================================================================

/// High-performance FFT-based holographic binder
///
/// Implements binding and unbinding operations using the convolution theorem:
/// - bind(a, b) = IFFT(FFT(a) * FFT(b))
/// - unbind(ab, b) = IFFT(FFT(ab) * conj(FFT(b)))
///
/// # Performance Features
///
/// - Reuses FFT plans (expensive to create)
/// - Preallocated scratch buffers avoid allocation
/// - SIMD-optimized complex multiplication
/// - Cache-oblivious blocked FFT for large vectors
/// - Batched processing for throughput
#[derive(Debug)]
pub struct FourierBinder {
    config: FourierConfig,
    fft_engine: FftEngine,
    scratch_pool: ScratchPool,
    stats: FourierBinderStats,
}

/// Statistics for FourierBinder operations
///
/// Tracks the number of bind, unbind, and batch operations performed.
#[derive(Debug, Default)]
pub struct FourierBinderStats {
    /// Number of single bind operations
    pub binds: AtomicU64,
    /// Number of single unbind operations
    pub unbinds: AtomicU64,
    /// Number of batched operations
    pub batched_ops: AtomicU64,
    /// Total number of vectors processed
    pub total_vectors_processed: AtomicU64,
}

impl FourierBinderStats {
    /// Get the number of bind operations
    pub fn binds(&self) -> u64 {
        self.binds.load(Ordering::Relaxed)
    }

    /// Get the number of unbind operations
    pub fn unbinds(&self) -> u64 {
        self.unbinds.load(Ordering::Relaxed)
    }

    /// Get the number of batched operations
    pub fn batched_ops(&self) -> u64 {
        self.batched_ops.load(Ordering::Relaxed)
    }

    /// Get the total number of vectors processed
    pub fn total_vectors_processed(&self) -> u64 {
        self.total_vectors_processed.load(Ordering::Relaxed)
    }
}

impl FourierBinder {
    /// Create a new FourierBinder with given configuration
    pub fn new(config: FourierConfig) -> Self {
        let plan = FftPlan::new(config.dimension, config.cache_block_size);
        let fft_size = plan.fft_size();
        let fft_engine = FftEngine::new(plan);

        // Create scratch pool with buffers for FFT operations
        let scratch_pool = ScratchPool::new(config.scratch_pool_size, fft_size);

        Self {
            config,
            fft_engine,
            scratch_pool,
            stats: FourierBinderStats::default(),
        }
    }

    /// Create with default configuration
    pub fn with_dimension(dimension: usize) -> Self {
        Self::new(FourierConfig::default().with_dimension(dimension))
    }

    /// Get the FFT size (padded dimension)
    pub fn fft_size(&self) -> usize {
        self.fft_engine.plan.fft_size()
    }

    /// Get the original dimension
    pub fn dimension(&self) -> usize {
        self.fft_engine.plan.original_dim()
    }

    /// Get statistics
    pub fn stats(&self) -> &FourierBinderStats {
        &self.stats
    }

    /// Bind two HyperVectors using FFT-based circular convolution
    ///
    /// bind(a, b) = IFFT(FFT(a) * FFT(b))
    pub fn bind(&self, a: &HyperVector, b: &HyperVector) -> Result<HyperVector, HDError> {
        self.check_dimensions(a, b)?;

        // Acquire scratch buffers
        let mut scratch_a = self.scratch_pool.acquire();
        let mut scratch_b = self.scratch_pool.acquire();
        let mut scratch_result = self.scratch_pool.acquire();

        // Convert to complex and zero-pad
        self.real_to_complex_padded(a.components(), scratch_a.as_mut_slice());
        self.real_to_complex_padded(b.components(), scratch_b.as_mut_slice());

        // Forward FFT
        self.fft_engine.forward(scratch_a.as_mut_slice());
        self.fft_engine.forward(scratch_b.as_mut_slice());

        // Element-wise multiplication in frequency domain
        SimdComplexMul::multiply(
            scratch_a.as_slice(),
            scratch_b.as_slice(),
            scratch_result.as_mut_slice(),
        );

        // Inverse FFT
        self.fft_engine.inverse(scratch_result.as_mut_slice());

        // Convert back to real
        let result = self.complex_to_real_truncate(scratch_result.as_slice(), self.dimension());

        self.stats.binds.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_vectors_processed
            .fetch_add(2, Ordering::Relaxed);

        Ok(HyperVector::from_components(result))
    }

    /// Unbind (inverse of bind) using FFT-based correlation
    ///
    /// unbind(ab, b) = IFFT(FFT(ab) * conj(FFT(b)))
    ///
    /// This recovers approximately `a` from `ab` given `b`
    pub fn unbind(&self, bound: &HyperVector, key: &HyperVector) -> Result<HyperVector, HDError> {
        self.check_dimensions(bound, key)?;

        let mut scratch_bound = self.scratch_pool.acquire();
        let mut scratch_key = self.scratch_pool.acquire();
        let mut scratch_result = self.scratch_pool.acquire();

        // Convert to complex and zero-pad
        self.real_to_complex_padded(bound.components(), scratch_bound.as_mut_slice());
        self.real_to_complex_padded(key.components(), scratch_key.as_mut_slice());

        // Forward FFT
        self.fft_engine.forward(scratch_bound.as_mut_slice());
        self.fft_engine.forward(scratch_key.as_mut_slice());

        // Element-wise multiplication with conjugate (correlation)
        SimdComplexMul::multiply_conj(
            scratch_bound.as_slice(),
            scratch_key.as_slice(),
            scratch_result.as_mut_slice(),
        );

        // Inverse FFT
        self.fft_engine.inverse(scratch_result.as_mut_slice());

        // Convert back to real
        let result = self.complex_to_real_truncate(scratch_result.as_slice(), self.dimension());

        self.stats.unbinds.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_vectors_processed
            .fetch_add(2, Ordering::Relaxed);

        Ok(HyperVector::from_components(result))
    }

    /// Batched bind operation for high throughput
    ///
    /// Processes multiple vector pairs simultaneously, reusing FFT computations
    /// when possible. Returns a vector of bound results.
    pub fn bind_batch(
        &self,
        a_vectors: &[HyperVector],
        b_vectors: &[HyperVector],
    ) -> Result<Vec<HyperVector>, HDError> {
        if a_vectors.is_empty() {
            return Err(HDError::EmptyInput);
        }

        if a_vectors.len() != b_vectors.len() {
            return Err(HDError::DimensionMismatch {
                expected: a_vectors.len(),
                actual: b_vectors.len(),
            });
        }

        let batch_size = a_vectors.len().min(self.config.max_batch_size);
        let fft_size = self.fft_size();
        let dim = self.dimension();

        // Verify all dimensions match
        for (a, b) in a_vectors.iter().zip(b_vectors.iter()) {
            self.check_dimensions(a, b)?;
        }

        let mut results = Vec::with_capacity(a_vectors.len());

        // Process in batches
        for chunk_start in (0..a_vectors.len()).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(a_vectors.len());
            let chunk_size = chunk_end - chunk_start;

            // Allocate batch buffers
            let mut a_fft: Vec<Vec<Complex32>> = (0..chunk_size)
                .map(|_| vec![Complex32::zero(); fft_size])
                .collect();
            let mut b_fft: Vec<Vec<Complex32>> = (0..chunk_size)
                .map(|_| vec![Complex32::zero(); fft_size])
                .collect();

            // Parallel FFT preparation
            for i in 0..chunk_size {
                let idx = chunk_start + i;
                self.real_to_complex_padded(a_vectors[idx].components(), &mut a_fft[i]);
                self.real_to_complex_padded(b_vectors[idx].components(), &mut b_fft[i]);
            }

            // FFT all vectors
            for i in 0..chunk_size {
                self.fft_engine.forward(&mut a_fft[i]);
                self.fft_engine.forward(&mut b_fft[i]);
            }

            // Element-wise multiplication and IFFT
            for i in 0..chunk_size {
                let mut result = vec![Complex32::zero(); fft_size];
                SimdComplexMul::multiply(&a_fft[i], &b_fft[i], &mut result);
                self.fft_engine.inverse(&mut result);

                let real_result = self.complex_to_real_truncate(&result, dim);
                results.push(HyperVector::from_components(real_result));
            }
        }

        self.stats.batched_ops.fetch_add(1, Ordering::Relaxed);
        self.stats
            .binds
            .fetch_add(a_vectors.len() as u64, Ordering::Relaxed);
        self.stats
            .total_vectors_processed
            .fetch_add((a_vectors.len() * 2) as u64, Ordering::Relaxed);

        Ok(results)
    }

    /// Batched unbind operation
    pub fn unbind_batch(
        &self,
        bound_vectors: &[HyperVector],
        key_vectors: &[HyperVector],
    ) -> Result<Vec<HyperVector>, HDError> {
        if bound_vectors.is_empty() {
            return Err(HDError::EmptyInput);
        }

        if bound_vectors.len() != key_vectors.len() {
            return Err(HDError::DimensionMismatch {
                expected: bound_vectors.len(),
                actual: key_vectors.len(),
            });
        }

        let batch_size = bound_vectors.len().min(self.config.max_batch_size);
        let fft_size = self.fft_size();
        let dim = self.dimension();

        for (bound, key) in bound_vectors.iter().zip(key_vectors.iter()) {
            self.check_dimensions(bound, key)?;
        }

        let mut results = Vec::with_capacity(bound_vectors.len());

        for chunk_start in (0..bound_vectors.len()).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(bound_vectors.len());
            let chunk_size = chunk_end - chunk_start;

            let mut bound_fft: Vec<Vec<Complex32>> = (0..chunk_size)
                .map(|_| vec![Complex32::zero(); fft_size])
                .collect();
            let mut key_fft: Vec<Vec<Complex32>> = (0..chunk_size)
                .map(|_| vec![Complex32::zero(); fft_size])
                .collect();

            for i in 0..chunk_size {
                let idx = chunk_start + i;
                self.real_to_complex_padded(bound_vectors[idx].components(), &mut bound_fft[i]);
                self.real_to_complex_padded(key_vectors[idx].components(), &mut key_fft[i]);
            }

            for i in 0..chunk_size {
                self.fft_engine.forward(&mut bound_fft[i]);
                self.fft_engine.forward(&mut key_fft[i]);
            }

            for i in 0..chunk_size {
                let mut result = vec![Complex32::zero(); fft_size];
                SimdComplexMul::multiply_conj(&bound_fft[i], &key_fft[i], &mut result);
                self.fft_engine.inverse(&mut result);

                let real_result = self.complex_to_real_truncate(&result, dim);
                results.push(HyperVector::from_components(real_result));
            }
        }

        self.stats.batched_ops.fetch_add(1, Ordering::Relaxed);
        self.stats
            .unbinds
            .fetch_add(bound_vectors.len() as u64, Ordering::Relaxed);
        self.stats
            .total_vectors_processed
            .fetch_add((bound_vectors.len() * 2) as u64, Ordering::Relaxed);

        Ok(results)
    }

    /// Bind with a single key against multiple vectors
    ///
    /// Useful when binding the same key to many values (e.g., role-filler bindings)
    pub fn bind_one_to_many(
        &self,
        key: &HyperVector,
        values: &[HyperVector],
    ) -> Result<Vec<HyperVector>, HDError> {
        if values.is_empty() {
            return Err(HDError::EmptyInput);
        }

        let fft_size = self.fft_size();
        let dim = self.dimension();

        // FFT the key once
        let mut key_fft = vec![Complex32::zero(); fft_size];
        self.real_to_complex_padded(key.components(), &mut key_fft);
        self.fft_engine.forward(&mut key_fft);

        let mut results = Vec::with_capacity(values.len());

        for value in values {
            if value.dimension() != dim {
                return Err(HDError::DimensionMismatch {
                    expected: dim,
                    actual: value.dimension(),
                });
            }

            let mut value_fft = vec![Complex32::zero(); fft_size];
            self.real_to_complex_padded(value.components(), &mut value_fft);
            self.fft_engine.forward(&mut value_fft);

            let mut result = vec![Complex32::zero(); fft_size];
            SimdComplexMul::multiply(&key_fft, &value_fft, &mut result);
            self.fft_engine.inverse(&mut result);

            let real_result = self.complex_to_real_truncate(&result, dim);
            results.push(HyperVector::from_components(real_result));
        }

        self.stats
            .binds
            .fetch_add(values.len() as u64, Ordering::Relaxed);
        self.stats
            .total_vectors_processed
            .fetch_add((values.len() + 1) as u64, Ordering::Relaxed);

        Ok(results)
    }

    /// Unbind with a single key from multiple bound vectors
    pub fn unbind_one_from_many(
        &self,
        key: &HyperVector,
        bound_vectors: &[HyperVector],
    ) -> Result<Vec<HyperVector>, HDError> {
        if bound_vectors.is_empty() {
            return Err(HDError::EmptyInput);
        }

        let fft_size = self.fft_size();
        let dim = self.dimension();

        // FFT the key once (conjugate for correlation)
        let mut key_fft = vec![Complex32::zero(); fft_size];
        self.real_to_complex_padded(key.components(), &mut key_fft);
        self.fft_engine.forward(&mut key_fft);

        let mut results = Vec::with_capacity(bound_vectors.len());

        for bound in bound_vectors {
            if bound.dimension() != dim {
                return Err(HDError::DimensionMismatch {
                    expected: dim,
                    actual: bound.dimension(),
                });
            }

            let mut bound_fft = vec![Complex32::zero(); fft_size];
            self.real_to_complex_padded(bound.components(), &mut bound_fft);
            self.fft_engine.forward(&mut bound_fft);

            let mut result = vec![Complex32::zero(); fft_size];
            SimdComplexMul::multiply_conj(&bound_fft, &key_fft, &mut result);
            self.fft_engine.inverse(&mut result);

            let real_result = self.complex_to_real_truncate(&result, dim);
            results.push(HyperVector::from_components(real_result));
        }

        self.stats
            .unbinds
            .fetch_add(bound_vectors.len() as u64, Ordering::Relaxed);
        self.stats
            .total_vectors_processed
            .fetch_add((bound_vectors.len() + 1) as u64, Ordering::Relaxed);

        Ok(results)
    }

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    fn check_dimensions(&self, a: &HyperVector, b: &HyperVector) -> Result<(), HDError> {
        let dim = self.dimension();

        if a.dimension() != dim {
            return Err(HDError::DimensionMismatch {
                expected: dim,
                actual: a.dimension(),
            });
        }

        if b.dimension() != dim {
            return Err(HDError::DimensionMismatch {
                expected: dim,
                actual: b.dimension(),
            });
        }

        Ok(())
    }

    /// Convert real vector to complex with zero imaginary, zero-padded to FFT size
    fn real_to_complex_padded(&self, real: &[f32], complex: &mut [Complex32]) {
        let n = real.len();

        for i in 0..n {
            complex[i] = Complex32::new(real[i], 0.0);
        }

        // Zero-pad the rest
        for c in complex.iter_mut().skip(n) {
            *c = Complex32::zero();
        }
    }

    /// Extract real parts and truncate to original dimension
    fn complex_to_real_truncate(&self, complex: &[Complex32], dim: usize) -> Vec<f32> {
        complex.iter().take(dim).map(|c| c.re).collect()
    }
}

// ============================================================================
// INTEGRATION WITH HYPERVECTOR (EXTENSION TRAIT)
// ============================================================================

/// Extension trait for HyperVector to use Fourier-based operations
pub trait FourierBindable {
    /// Bind using FFT-based circular convolution
    fn fft_bind(&self, other: &Self, binder: &FourierBinder) -> Result<Self, HDError>
    where
        Self: Sized;

    /// Unbind using FFT-based correlation
    fn fft_unbind(&self, key: &Self, binder: &FourierBinder) -> Result<Self, HDError>
    where
        Self: Sized;
}

impl FourierBindable for HyperVector {
    fn fft_bind(&self, other: &Self, binder: &FourierBinder) -> Result<Self, HDError> {
        binder.bind(self, other)
    }

    fn fft_unbind(&self, key: &Self, binder: &FourierBinder) -> Result<Self, HDError> {
        binder.unbind(self, key)
    }
}

// ============================================================================
// BENCHMARK HELPERS
// ============================================================================

/// Benchmark-friendly interface for performance testing
pub struct FourierBenchmark {
    binder: FourierBinder,
    test_vectors: Vec<HyperVector>,
}

impl FourierBenchmark {
    /// Create benchmark with N random vectors
    pub fn new(dimension: usize, num_vectors: usize) -> Self {
        let binder = FourierBinder::with_dimension(dimension);
        let test_vectors: Vec<_> = (0..num_vectors as u64)
            .map(|i| HyperVector::random(dimension, i * 12345))
            .collect();

        Self {
            binder,
            test_vectors,
        }
    }

    /// Benchmark single bind operation
    pub fn bench_bind(&self) -> HyperVector {
        self.binder
            .bind(&self.test_vectors[0], &self.test_vectors[1])
            .unwrap()
    }

    /// Benchmark single unbind operation
    pub fn bench_unbind(&self) -> HyperVector {
        let bound = self
            .binder
            .bind(&self.test_vectors[0], &self.test_vectors[1])
            .unwrap();
        self.binder.unbind(&bound, &self.test_vectors[1]).unwrap()
    }

    /// Benchmark batched bind
    pub fn bench_batch_bind(&self, batch_size: usize) -> Vec<HyperVector> {
        let half = self.test_vectors.len() / 2;
        let a = &self.test_vectors[..batch_size.min(half)];
        let b = &self.test_vectors[half..half + batch_size.min(half)];
        self.binder.bind_batch(a, b).unwrap()
    }

    /// Get binder reference
    pub fn binder(&self) -> &FourierBinder {
        &self.binder
    }

    /// Get test vectors
    pub fn test_vectors(&self) -> &[HyperVector] {
        &self.test_vectors
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Basic FFT Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_complex32_operations() {
        let a = Complex32::new(3.0, 4.0);
        let b = Complex32::new(1.0, 2.0);

        // Magnitude
        assert!((a.magnitude() - 5.0).abs() < 1e-6);

        // Multiplication: (3 + 4i)(1 + 2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
        let c = a.mul(&b);
        assert!((c.re - (-5.0)).abs() < 1e-6);
        assert!((c.im - 10.0).abs() < 1e-6);

        // Conjugate
        let conj = a.conjugate();
        assert_eq!(conj.re, 3.0);
        assert_eq!(conj.im, -4.0);
    }

    #[test]
    fn test_twiddle_factors() {
        let twiddles = TwiddleFactors::new(8);

        // First twiddle should be 1 + 0i
        let w0 = twiddles.get_forward(0, 0);
        assert!((w0.re - 1.0).abs() < 1e-6);
        assert!(w0.im.abs() < 1e-6);
    }

    #[test]
    fn test_bit_reversal() {
        let table = BitReversalTable::new(8);

        // Expected bit-reversal for n=8:
        // 0 -> 0, 1 -> 4, 2 -> 2, 3 -> 6, 4 -> 1, 5 -> 5, 6 -> 3, 7 -> 7
        assert_eq!(table.indices[0], 0);
        assert_eq!(table.indices[1], 4);
        assert_eq!(table.indices[2], 2);
        assert_eq!(table.indices[3], 6);
        assert_eq!(table.indices[4], 1);
        assert_eq!(table.indices[5], 5);
        assert_eq!(table.indices[6], 3);
        assert_eq!(table.indices[7], 7);
    }

    #[test]
    fn test_fft_roundtrip() {
        let plan = FftPlan::new(16, 4096);
        let engine = FftEngine::new(plan);

        // Create a simple test signal
        let mut data: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new((i as f32).sin(), 0.0))
            .collect();
        let original = data.clone();

        // Forward then inverse should give back original
        engine.forward(&mut data);
        engine.inverse(&mut data);

        for (orig, result) in original.iter().zip(data.iter()) {
            assert!(
                (orig.re - result.re).abs() < 1e-5,
                "FFT roundtrip failed: {} vs {}",
                orig.re,
                result.re
            );
        }
    }

    // ------------------------------------------------------------------------
    // FourierBinder Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_fourier_binder_creation() {
        let binder = FourierBinder::with_dimension(1024);
        assert_eq!(binder.dimension(), 1024);
        assert_eq!(binder.fft_size(), 1024); // 1024 is already power of 2
    }

    #[test]
    fn test_fourier_binder_non_power_of_two() {
        let binder = FourierBinder::with_dimension(1000);
        assert_eq!(binder.dimension(), 1000);
        assert_eq!(binder.fft_size(), 1024); // Padded to next power of 2
    }

    #[test]
    fn test_bind_self_inverse() {
        // Circular convolution of a vector with itself should give non-zero result
        let binder = FourierBinder::with_dimension(256);
        let v = HyperVector::random(256, 42);

        let bound = binder.bind(&v, &v).unwrap();

        // Result should be non-trivial
        let norm: f32 = bound.components().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_bind_unbind_recovery() {
        let binder = FourierBinder::with_dimension(512);

        let a = HyperVector::random(512, 1);
        let b = HyperVector::random(512, 2);

        let bound = binder.bind(&a, &b).unwrap();
        let recovered = binder.unbind(&bound, &b).unwrap();

        // Recovered should be similar to original
        let similarity = recovered.similarity(&a).unwrap();
        assert!(
            similarity > 0.3,
            "Recovery similarity too low: {}",
            similarity
        );
    }

    #[test]
    fn test_bind_commutativity() {
        // Circular convolution is commutative
        let binder = FourierBinder::with_dimension(256);

        let a = HyperVector::random(256, 10);
        let b = HyperVector::random(256, 20);

        let ab = binder.bind(&a, &b).unwrap();
        let ba = binder.bind(&b, &a).unwrap();

        let similarity = ab.similarity(&ba).unwrap();
        assert!(
            (similarity - 1.0).abs() < 0.01,
            "bind(a,b) != bind(b,a): similarity = {}",
            similarity
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let binder = FourierBinder::with_dimension(256);

        let a = HyperVector::random(256, 1);
        let b = HyperVector::random(512, 2);

        let result = binder.bind(&a, &b);
        assert!(matches!(result, Err(HDError::DimensionMismatch { .. })));
    }

    // ------------------------------------------------------------------------
    // Batched Operations Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_batch_bind() {
        let binder = FourierBinder::with_dimension(256);

        let a_vecs: Vec<_> = (0..8).map(|i| HyperVector::random(256, i)).collect();
        let b_vecs: Vec<_> = (8..16).map(|i| HyperVector::random(256, i)).collect();

        let results = binder.bind_batch(&a_vecs, &b_vecs).unwrap();

        assert_eq!(results.len(), 8);

        // Each result should match individual bind
        for (i, result) in results.iter().enumerate() {
            let expected = binder.bind(&a_vecs[i], &b_vecs[i]).unwrap();
            let sim = result.similarity(&expected).unwrap();
            assert!(
                (sim - 1.0).abs() < 0.01,
                "Batch result {} doesn't match individual: sim = {}",
                i,
                sim
            );
        }
    }

    #[test]
    fn test_batch_unbind() {
        let binder = FourierBinder::with_dimension(256);

        let keys: Vec<_> = (0..4).map(|i| HyperVector::random(256, i)).collect();
        let values: Vec<_> = (4..8).map(|i| HyperVector::random(256, i)).collect();

        // First bind
        let bound = binder.bind_batch(&keys, &values).unwrap();

        // Then unbind
        let recovered = binder.unbind_batch(&bound, &keys).unwrap();

        // Check recovery
        for (i, rec) in recovered.iter().enumerate() {
            let sim = rec.similarity(&values[i]).unwrap();
            assert!(sim > 0.2, "Batch unbind recovery {} too low: {}", i, sim);
        }
    }

    #[test]
    fn test_batch_empty_input() {
        let binder = FourierBinder::with_dimension(256);

        let result = binder.bind_batch(&[], &[]);
        assert!(matches!(result, Err(HDError::EmptyInput)));
    }

    #[test]
    fn test_batch_mismatched_lengths() {
        let binder = FourierBinder::with_dimension(256);

        let a_vecs: Vec<_> = (0..4).map(|i| HyperVector::random(256, i)).collect();
        let b_vecs: Vec<_> = (0..8).map(|i| HyperVector::random(256, i)).collect();

        let result = binder.bind_batch(&a_vecs, &b_vecs);
        assert!(matches!(result, Err(HDError::DimensionMismatch { .. })));
    }

    // ------------------------------------------------------------------------
    // One-to-Many Operations Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_bind_one_to_many() {
        let binder = FourierBinder::with_dimension(256);

        let key = HyperVector::random(256, 100);
        let values: Vec<_> = (0..10).map(|i| HyperVector::random(256, i)).collect();

        let results = binder.bind_one_to_many(&key, &values).unwrap();

        assert_eq!(results.len(), 10);

        // Each should match individual bind
        for (i, result) in results.iter().enumerate() {
            let expected = binder.bind(&key, &values[i]).unwrap();
            let sim = result.similarity(&expected).unwrap();
            assert!((sim - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_unbind_one_from_many() {
        let binder = FourierBinder::with_dimension(256);

        let key = HyperVector::random(256, 100);
        let values: Vec<_> = (0..10).map(|i| HyperVector::random(256, i)).collect();

        let bound = binder.bind_one_to_many(&key, &values).unwrap();
        let recovered = binder.unbind_one_from_many(&key, &bound).unwrap();

        for (i, rec) in recovered.iter().enumerate() {
            let sim = rec.similarity(&values[i]).unwrap();
            assert!(
                sim > 0.2,
                "One-to-many unbind recovery {} too low: {}",
                i,
                sim
            );
        }
    }

    // ------------------------------------------------------------------------
    // SIMD Multiplication Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_multiply() {
        let a: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new(i as f32, (i * 2) as f32))
            .collect();
        let b: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new((i + 1) as f32, (i + 3) as f32))
            .collect();
        let mut result = vec![Complex32::zero(); 16];

        SimdComplexMul::multiply(&a, &b, &mut result);

        // Verify against scalar multiplication
        for i in 0..16 {
            let expected = a[i].mul(&b[i]);
            assert!((result[i].re - expected.re).abs() < 1e-5);
            assert!((result[i].im - expected.im).abs() < 1e-5);
        }
    }

    #[test]
    fn test_simd_multiply_conj() {
        let a: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new(i as f32, (i * 2) as f32))
            .collect();
        let b: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new((i + 1) as f32, (i + 3) as f32))
            .collect();
        let mut result = vec![Complex32::zero(); 16];

        SimdComplexMul::multiply_conj(&a, &b, &mut result);

        for i in 0..16 {
            let expected = a[i].mul(&b[i].conjugate());
            assert!((result[i].re - expected.re).abs() < 1e-5);
            assert!((result[i].im - expected.im).abs() < 1e-5);
        }
    }

    // ------------------------------------------------------------------------
    // Statistics Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_statistics_tracking() {
        let binder = FourierBinder::with_dimension(128);

        let a = HyperVector::random(128, 1);
        let b = HyperVector::random(128, 2);

        // Initial stats should be zero
        assert_eq!(binder.stats().binds(), 0);
        assert_eq!(binder.stats().unbinds(), 0);

        // Do some operations
        let _ = binder.bind(&a, &b).unwrap();
        assert_eq!(binder.stats().binds(), 1);

        let bound = binder.bind(&a, &b).unwrap();
        assert_eq!(binder.stats().binds(), 2);

        let _ = binder.unbind(&bound, &b).unwrap();
        assert_eq!(binder.stats().unbinds(), 1);
    }

    // ------------------------------------------------------------------------
    // Extension Trait Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_fourier_bindable_trait() {
        let binder = FourierBinder::with_dimension(256);

        let a = HyperVector::random(256, 42);
        let b = HyperVector::random(256, 123);

        // Use extension trait
        let bound = a.fft_bind(&b, &binder).unwrap();
        let recovered = bound.fft_unbind(&b, &binder).unwrap();

        let sim = recovered.similarity(&a).unwrap();
        assert!(sim > 0.3);
    }

    // ------------------------------------------------------------------------
    // Benchmark Helper Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_benchmark_helper() {
        let bench = FourierBenchmark::new(256, 128);

        assert_eq!(bench.test_vectors().len(), 128);

        let _bound = bench.bench_bind();
        let _unbound = bench.bench_unbind();
        let batch_results = bench.bench_batch_bind(32);

        assert_eq!(batch_results.len(), 32);
    }

    // ------------------------------------------------------------------------
    // Large Vector Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_large_dimension() {
        // Test with typical HD computing dimension
        let binder = FourierBinder::with_dimension(10000);

        let a = HyperVector::random(10000, 1);
        let b = HyperVector::random(10000, 2);

        let bound = binder.bind(&a, &b).unwrap();
        let recovered = binder.unbind(&bound, &b).unwrap();

        // Should still recover reasonably well
        let sim = recovered.similarity(&a).unwrap();
        assert!(sim > 0.1, "Large dimension recovery too low: {}", sim);
    }

    #[test]
    fn test_scratch_pool() {
        let pool = ScratchPool::new(4, 1024);

        // Acquire multiple buffers
        let mut guard1 = pool.acquire();
        let mut guard2 = pool.acquire();

        // Should be able to use independently
        guard1.as_mut_slice()[0] = Complex32::new(1.0, 2.0);
        guard2.as_mut_slice()[0] = Complex32::new(3.0, 4.0);

        assert!((guard1.as_slice()[0].re - 1.0).abs() < 1e-6);
        assert!((guard2.as_slice()[0].re - 3.0).abs() < 1e-6);
    }
}
