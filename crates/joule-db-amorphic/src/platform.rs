//! Platform-specific optimizations for JouleDB
//!
//! This module provides runtime detection of hardware capabilities and
//! automatically selects the best implementation for the current platform.
//!
//! ## Detected Features
//!
//! - **CPU**: Core count, SIMD support (AVX-512, AVX2, SSE4.2)
//! - **Memory**: Available RAM for cache sizing
//! - **Architecture**: x86_64, ARM64/Apple Silicon
//!
//! ## Automatic Optimizations
//!
//! - Parallel query execution using all available cores
//! - SIMD-accelerated Hamming distance calculations
//! - Cache-friendly data layouts
//! - Platform-specific memory allocation hints

use std::sync::OnceLock;

/// Cached platform capabilities (computed once at startup)
static PLATFORM: OnceLock<PlatformCapabilities> = OnceLock::new();

/// Get the platform capabilities (cached)
pub fn platform() -> &'static PlatformCapabilities {
    PLATFORM.get_or_init(PlatformCapabilities::detect)
}

/// Detected platform capabilities
#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    /// Number of physical CPU cores
    pub cpu_cores: usize,
    /// Number of logical threads (with hyperthreading)
    pub cpu_threads: usize,
    /// CPU architecture
    pub arch: CpuArch,
    /// SIMD capabilities
    pub simd: SimdCapabilities,
    /// Recommended parallelism level
    pub recommended_parallelism: usize,
    /// Recommended shard count for ShardedAmorphicStore
    pub recommended_shard_count: usize,
    /// Recommended batch size for vector operations
    pub recommended_batch_size: usize,
}

impl PlatformCapabilities {
    /// Detect capabilities of the current platform
    pub fn detect() -> Self {
        let cpu_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        // Estimate physical cores (assume hyperthreading on x86)
        let cpu_cores = if cfg!(target_arch = "x86_64") {
            cpu_threads / 2
        } else {
            cpu_threads
        }
        .max(1);

        let arch = CpuArch::detect();
        let simd = SimdCapabilities::detect();

        // Recommendations based on detected hardware
        let recommended_parallelism = cpu_threads;
        let recommended_shard_count = (cpu_cores * 2).max(4).min(64);

        // Batch size based on cache and SIMD width
        let recommended_batch_size = match simd.best_level() {
            SimdLevel::Avx512 => 512,
            SimdLevel::Avx2 => 256,
            SimdLevel::Sse42 => 128,
            SimdLevel::Neon => 128,
            SimdLevel::Scalar => 64,
        };

        Self {
            cpu_cores,
            cpu_threads,
            arch,
            simd,
            recommended_parallelism,
            recommended_shard_count,
            recommended_batch_size,
        }
    }

    /// Get a summary string for logging
    pub fn summary(&self) -> String {
        format!(
            "CPU: {} cores/{} threads, Arch: {:?}, SIMD: {:?}, Shards: {}, Batch: {}",
            self.cpu_cores,
            self.cpu_threads,
            self.arch,
            self.simd.best_level(),
            self.recommended_shard_count,
            self.recommended_batch_size,
        )
    }
}

impl Default for PlatformCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

/// CPU architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArch {
    /// x86-64 (Intel/AMD)
    X86_64,
    /// ARM64 (Apple Silicon, AWS Graviton, etc.)
    Aarch64,
    /// WebAssembly
    Wasm32,
    /// Unknown architecture
    Unknown,
}

impl CpuArch {
    /// Detect the current CPU architecture
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            CpuArch::X86_64
        }

        #[cfg(target_arch = "aarch64")]
        {
            CpuArch::Aarch64
        }

        #[cfg(target_arch = "wasm32")]
        {
            CpuArch::Wasm32
        }

        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )))]
        {
            CpuArch::Unknown
        }
    }

    /// Check if this is Apple Silicon
    pub fn is_apple_silicon(&self) -> bool {
        *self == CpuArch::Aarch64 && cfg!(target_os = "macos")
    }
}

/// SIMD capability level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD, scalar operations only
    Scalar = 0,
    /// ARM NEON (128-bit)
    Neon = 1,
    /// x86 SSE4.2 (128-bit)
    Sse42 = 2,
    /// x86 AVX2 (256-bit)
    Avx2 = 3,
    /// x86 AVX-512 (512-bit)
    Avx512 = 4,
}

impl SimdLevel {
    /// Get the vector width in bits
    pub fn width_bits(&self) -> usize {
        match self {
            SimdLevel::Scalar => 64,
            SimdLevel::Neon => 128,
            SimdLevel::Sse42 => 128,
            SimdLevel::Avx2 => 256,
            SimdLevel::Avx512 => 512,
        }
    }

    /// Get the number of u64 elements per vector
    pub fn u64_per_vector(&self) -> usize {
        self.width_bits() / 64
    }
}

/// Detected SIMD capabilities
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    /// AVX-512F support
    pub avx512f: bool,
    /// AVX-512 VPOPCNTDQ support (fast popcount)
    pub avx512_vpopcntdq: bool,
    /// AVX2 support
    pub avx2: bool,
    /// SSE4.2 support
    pub sse42: bool,
    /// ARM NEON support
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities of the current CPU
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                avx512_vpopcntdq: std::arch::is_x86_feature_detected!("avx512vpopcntdq"),
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                sse42: std::arch::is_x86_feature_detected!("sse4.2"),
                neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx512f: false,
                avx512_vpopcntdq: false,
                avx2: false,
                sse42: false,
                // NEON is always available on aarch64
                neon: true,
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx512f: false,
                avx512_vpopcntdq: false,
                avx2: false,
                sse42: false,
                neon: false,
            }
        }
    }

    /// Get the best available SIMD level
    pub fn best_level(&self) -> SimdLevel {
        if self.avx512_vpopcntdq || self.avx512f {
            SimdLevel::Avx512
        } else if self.avx2 {
            SimdLevel::Avx2
        } else if self.sse42 {
            SimdLevel::Sse42
        } else if self.neon {
            SimdLevel::Neon
        } else {
            SimdLevel::Scalar
        }
    }

    /// Check if hardware popcount is available
    pub fn has_fast_popcount(&self) -> bool {
        self.avx512_vpopcntdq || self.sse42 || self.neon
    }
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

// =============================================================================
// OPTIMIZED OPERATIONS
// =============================================================================

/// Parallel iterator configuration
pub struct ParallelConfig {
    /// Number of threads to use
    pub num_threads: usize,
    /// Minimum items per thread before parallelizing
    pub min_items_per_thread: usize,
}

impl ParallelConfig {
    /// Create a new parallel config based on platform detection
    pub fn auto() -> Self {
        let platform = platform();
        Self {
            num_threads: platform.recommended_parallelism,
            min_items_per_thread: 100,
        }
    }

    /// Check if work should be parallelized
    pub fn should_parallelize(&self, item_count: usize) -> bool {
        item_count >= self.num_threads * self.min_items_per_thread
    }

    /// Calculate chunk size for parallel processing
    pub fn chunk_size(&self, item_count: usize) -> usize {
        (item_count / self.num_threads).max(self.min_items_per_thread)
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::auto()
    }
}

/// Execute a function in parallel across chunks if beneficial
pub fn parallel_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    let config = ParallelConfig::auto();

    if !config.should_parallelize(items.len()) {
        // Serial execution for small workloads
        return items.iter().map(f).collect();
    }

    // Parallel execution using scoped threads
    let chunk_size = config.chunk_size(items.len());
    let chunks: Vec<_> = items.chunks(chunk_size).collect();
    let mut results = Vec::with_capacity(items.len());

    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk| s.spawn(|| chunk.iter().map(&f).collect::<Vec<_>>()))
            .collect();

        for handle in handles {
            results.extend(handle.join().unwrap());
        }
    });

    results
}

/// Execute a reduction in parallel if beneficial
pub fn parallel_reduce<T, R, M, C>(items: &[T], map_fn: M, combine_fn: C, identity: R) -> R
where
    T: Sync,
    R: Send + Clone,
    M: Fn(&T) -> R + Sync,
    C: Fn(R, R) -> R + Sync,
{
    let config = ParallelConfig::auto();

    if !config.should_parallelize(items.len()) {
        // Serial reduction
        return items
            .iter()
            .fold(identity, |acc, item| combine_fn(acc, map_fn(item)));
    }

    // Parallel reduction using references to closures
    let chunk_size = config.chunk_size(items.len());
    let chunks: Vec<_> = items.chunks(chunk_size).collect();
    let map_ref = &map_fn;
    let combine_ref = &combine_fn;

    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                let identity = identity.clone();
                s.spawn(move || {
                    chunk
                        .iter()
                        .fold(identity, |acc, item| combine_ref(acc, map_ref(item)))
                })
            })
            .collect();

        let final_identity = identity.clone();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .fold(final_identity, |a, b| combine_fn(a, b))
    })
}

// =============================================================================
// PLATFORM-SPECIFIC HAMMING DISTANCE
// =============================================================================

/// Compute Hamming distance using the best available method
#[inline]
pub fn hamming_distance_optimized(a: &[u64], b: &[u64]) -> u32 {
    // Delegate to the HDC simd module which handles platform detection
    joule_db_hdc::simd::hamming_distance(a, b)
}

/// Compute batch Hamming distances using the best available method
#[inline]
pub fn hamming_distances_batch_optimized(query: &[u64], targets: &[&[u64]]) -> Vec<u32> {
    joule_db_hdc::simd::hamming_distances_batch(query, targets)
}

/// Find top-k nearest neighbors by Hamming distance
#[inline]
pub fn hamming_top_k_optimized(query: &[u64], targets: &[&[u64]], k: usize) -> Vec<(usize, u32)> {
    // Compute all distances
    let distances = joule_db_hdc::simd::hamming_distances_batch(query, targets);

    // Create (index, distance) pairs
    let mut indexed: Vec<(usize, u32)> = distances.into_iter().enumerate().collect();

    // Partial sort to get top-k (smallest distances)
    let k = k.min(indexed.len());
    if k == 0 {
        return vec![];
    }

    // Use partial_sort for efficiency on large datasets
    indexed.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
    indexed.truncate(k);
    indexed.sort_by_key(|&(_, d)| d);

    indexed
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = platform();

        // Should detect something reasonable
        assert!(platform.cpu_cores >= 1);
        assert!(platform.cpu_threads >= 1);
        assert!(platform.recommended_parallelism >= 1);
        assert!(platform.recommended_shard_count >= 4);

        println!("Platform: {}", platform.summary());
    }

    #[test]
    fn test_simd_detection() {
        let simd = SimdCapabilities::detect();
        let level = simd.best_level();

        println!("SIMD Level: {:?}", level);
        println!("SIMD Width: {} bits", level.width_bits());
        println!("Fast popcount: {}", simd.has_fast_popcount());

        // Level should be valid
        assert!(level.width_bits() >= 64);
    }

    #[test]
    fn test_parallel_map() {
        let items: Vec<i32> = (0..1000).collect();
        let results = parallel_map(&items, |x| x * 2);

        assert_eq!(results.len(), items.len());
        for (i, r) in results.iter().enumerate() {
            assert_eq!(*r, (i as i32) * 2);
        }
    }

    #[test]
    fn test_parallel_reduce() {
        let items: Vec<i32> = (1..=100).collect();
        let sum = parallel_reduce(&items, |x| *x as i64, |a, b| a + b, 0i64);

        // Sum of 1..=100 = 5050
        assert_eq!(sum, 5050);
    }

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::auto();

        // Small workloads shouldn't parallelize
        assert!(!config.should_parallelize(10));

        // Large workloads should
        assert!(config.should_parallelize(100_000));
    }

    #[test]
    fn test_arch_detection() {
        let arch = CpuArch::detect();

        #[cfg(target_arch = "x86_64")]
        assert_eq!(arch, CpuArch::X86_64);

        #[cfg(target_arch = "aarch64")]
        assert_eq!(arch, CpuArch::Aarch64);

        println!("Architecture: {:?}", arch);
    }
}
