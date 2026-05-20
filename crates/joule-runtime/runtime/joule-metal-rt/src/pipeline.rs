//! Metal Compute Pipeline
//!
//! This module provides compute pipeline management for Metal.
//! A compute pipeline encapsulates a compiled compute kernel and
//! its execution configuration.
//!
//! ## Pipeline Creation
//!
//! ```ignore
//! let device = MetalDevice::default()?;
//!
//! // Create library from compiled shader bytecode
//! let library = device.create_library(&bytecode)?;
//!
//! // Create compute pipeline from library function
//! let pipeline = device.create_compute_pipeline(&library, "my_kernel")?;
//!
//! // Get execution limits
//! let max_threads = pipeline.max_total_threads_per_threadgroup();
//! ```

/// Threadgroup size for compute dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadgroupSize {
    /// Width (x dimension)
    pub width: u32,
    /// Height (y dimension)
    pub height: u32,
    /// Depth (z dimension)
    pub depth: u32,
}

impl ThreadgroupSize {
    /// Create a new threadgroup size
    #[must_use]
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    /// Create a 1D threadgroup size
    #[must_use]
    pub fn d1(width: u32) -> Self {
        Self::new(width, 1, 1)
    }

    /// Create a 2D threadgroup size
    #[must_use]
    pub fn d2(width: u32, height: u32) -> Self {
        Self::new(width, height, 1)
    }

    /// Total number of threads in the threadgroup
    #[must_use]
    pub fn total(&self) -> u32 {
        self.width * self.height * self.depth
    }

    /// Convert to tuple format
    #[must_use]
    pub fn as_tuple(&self) -> (u32, u32, u32) {
        (self.width, self.height, self.depth)
    }
}

impl Default for ThreadgroupSize {
    fn default() -> Self {
        Self::d1(32)
    }
}

impl From<(u32, u32, u32)> for ThreadgroupSize {
    fn from((width, height, depth): (u32, u32, u32)) -> Self {
        Self::new(width, height, depth)
    }
}

/// A compiled Metal library
///
/// Contains compiled shader functions that can be used to create
/// compute pipelines.
pub struct MetalLibrary {
    /// Function names in this library
    function_names: Vec<String>,
    /// Real Metal library handle (macOS only)
    #[cfg(target_os = "macos")]
    handle: metal::Library,
    #[cfg(not(target_os = "macos"))]
    _marker: std::marker::PhantomData<()>,
}

impl std::fmt::Debug for MetalLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalLibrary")
            .field("function_count", &self.function_names.len())
            .finish()
    }
}

impl MetalLibrary {
    /// Create a MetalLibrary from a real metal::Library (macOS only).
    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal(library: metal::Library) -> Self {
        let function_names = library.function_names().iter().map(|s| s.to_string()).collect();
        Self {
            function_names,
            handle: library,
        }
    }

    /// Get all function names in this library
    #[must_use]
    pub fn function_names(&self) -> &[String] {
        &self.function_names
    }

    /// Check if a function exists in this library
    #[must_use]
    pub fn has_function(&self, name: &str) -> bool {
        self.function_names.iter().any(|n| n == name)
    }

    /// Get a metal::Function by name (macOS only).
    #[cfg(target_os = "macos")]
    pub(crate) fn get_metal_function(
        &self,
        name: &str,
    ) -> Result<metal::Function, String> {
        self.handle.get_function(name, None)
    }

    /// Get library handle ID (for debugging)
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        self.function_names.len() as u64
    }

    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        0
    }
}

/// A compute pipeline state
///
/// Encapsulates a compiled compute kernel with its execution configuration.
/// Create pipelines once and reuse them for optimal performance.
pub struct MetalComputePipeline {
    /// Function name
    function_name: String,
    /// Maximum total threads per threadgroup
    max_total_threads_per_threadgroup: u32,
    /// Thread execution width (SIMD width)
    thread_execution_width: u32,
    /// Static threadgroup memory length
    static_threadgroup_memory_length: u32,
    /// Real Metal pipeline state (macOS only)
    #[cfg(target_os = "macos")]
    handle: metal::ComputePipelineState,
    #[cfg(not(target_os = "macos"))]
    _marker: std::marker::PhantomData<()>,
}

impl std::fmt::Debug for MetalComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalComputePipeline")
            .field("function_name", &self.function_name)
            .field(
                "max_threads_per_threadgroup",
                &self.max_total_threads_per_threadgroup,
            )
            .finish()
    }
}

impl MetalComputePipeline {
    /// Create a MetalComputePipeline from a real metal::ComputePipelineState (macOS only).
    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal(
        pipeline: metal::ComputePipelineState,
        function_name: String,
    ) -> Self {
        let max_total = pipeline.max_total_threads_per_threadgroup() as u32;
        let exec_width = pipeline.thread_execution_width() as u32;
        let static_mem = pipeline.static_threadgroup_memory_length() as u32;

        Self {
            function_name,
            max_total_threads_per_threadgroup: max_total,
            thread_execution_width: exec_width,
            static_threadgroup_memory_length: static_mem,
            handle: pipeline,
        }
    }

    /// Get the function name
    #[must_use]
    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    /// Get maximum total threads per threadgroup
    #[must_use]
    pub fn max_total_threads_per_threadgroup(&self) -> u32 {
        self.max_total_threads_per_threadgroup
    }

    /// Get thread execution width (SIMD width)
    #[must_use]
    pub fn thread_execution_width(&self) -> u32 {
        self.thread_execution_width
    }

    /// Get static threadgroup memory length
    #[must_use]
    pub fn static_threadgroup_memory_length(&self) -> u32 {
        self.static_threadgroup_memory_length
    }

    /// Calculate optimal threadgroup size for a 1D dispatch
    #[must_use]
    pub fn optimal_threadgroup_size_1d(&self) -> ThreadgroupSize {
        let width = self.max_total_threads_per_threadgroup.min(256);
        ThreadgroupSize::d1(width)
    }

    /// Calculate optimal threadgroup size for a 2D dispatch
    #[must_use]
    pub fn optimal_threadgroup_size_2d(&self) -> ThreadgroupSize {
        let max = self.max_total_threads_per_threadgroup;
        let side = (max as f32).sqrt() as u32;
        let width = side.min(16);
        let height = (max / width).min(16);
        ThreadgroupSize::d2(width, height)
    }

    /// Calculate grid size from problem size
    #[must_use]
    pub fn grid_size_for_elements(
        total_elements: (u32, u32, u32),
        threadgroup_size: ThreadgroupSize,
    ) -> (u32, u32, u32) {
        let div_ceil = |a: u32, b: u32| (a + b - 1) / b;

        (
            div_ceil(total_elements.0, threadgroup_size.width),
            div_ceil(total_elements.1, threadgroup_size.height),
            div_ceil(total_elements.2, threadgroup_size.depth),
        )
    }

    /// Get a reference to the underlying Metal pipeline state (macOS only).
    #[cfg(target_os = "macos")]
    pub(crate) fn metal_pipeline(&self) -> &metal::ComputePipelineStateRef {
        &self.handle
    }

    /// Get pipeline handle ID (for debugging)
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        self.thread_execution_width as u64
    }

    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        0
    }
}

// Safety: Metal pipeline state is thread-safe
unsafe impl Send for MetalLibrary {}
unsafe impl Sync for MetalLibrary {}
unsafe impl Send for MetalComputePipeline {}
unsafe impl Sync for MetalComputePipeline {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threadgroup_size() {
        let size = ThreadgroupSize::new(32, 8, 1);
        assert_eq!(size.total(), 256);
        assert_eq!(size.as_tuple(), (32, 8, 1));
    }

    #[test]
    fn test_threadgroup_size_1d() {
        let size = ThreadgroupSize::d1(64);
        assert_eq!(size.width, 64);
        assert_eq!(size.height, 1);
        assert_eq!(size.depth, 1);
    }

    #[test]
    fn test_threadgroup_size_from_tuple() {
        let size: ThreadgroupSize = (16, 16, 1).into();
        assert_eq!(size.total(), 256);
    }

    #[test]
    fn test_grid_size_calculation() {
        let elements = (1000u32, 1u32, 1u32);
        let tg = ThreadgroupSize::d1(256);
        let grid = MetalComputePipeline::grid_size_for_elements(elements, tg);
        assert_eq!(grid.0, 4);
        assert_eq!(grid.1, 1);
        assert_eq!(grid.2, 1);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_library_from_source() {
        let device = crate::MetalDevice::system_default().unwrap();
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void add_arrays(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                uint idx [[thread_position_in_grid]]
            ) {
                c[idx] = a[idx] + b[idx];
            }
        "#;
        let library = device.create_library_from_source(source).unwrap();
        assert!(library.has_function("add_arrays"));

        let pipeline = device
            .create_compute_pipeline(&library, "add_arrays")
            .unwrap();
        assert_eq!(pipeline.function_name(), "add_arrays");
        assert!(pipeline.max_total_threads_per_threadgroup() > 0);
        assert!(pipeline.thread_execution_width() > 0);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_device_not_available_for_library() {
        let result = crate::MetalDevice::system_default();
        assert!(result.is_err());
    }
}
