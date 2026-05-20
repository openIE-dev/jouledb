//! Metal Device Wrapper
//!
//! This module provides a safe wrapper around `MTLDevice`, representing a
//! GPU that can execute Metal commands.
//!
//! ## Device Selection
//!
//! On systems with multiple GPUs, you can enumerate available devices and
//! select the most appropriate one:
//!
//! ```ignore
//! use joule_metal_rt::MetalDevice;
//!
//! // Get the default device (usually the best available)
//! let device = MetalDevice::system_default()?;
//!
//! // Or enumerate all devices
//! for device in MetalDevice::all_devices()? {
//!     println!("Found device: {}", device.name());
//! }
//! ```

use crate::error::{MetalError, MetalResult};
use crate::{MetalBuffer, MetalCommandQueue, MetalComputePipeline, MetalLibrary};

/// Information about a Metal device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Registry ID (unique identifier)
    pub registry_id: u64,
    /// Whether this is a low-power device
    pub is_low_power: bool,
    /// Whether this is a headless device (no display)
    pub is_headless: bool,
    /// Whether this device is removable (eGPU)
    pub is_removable: bool,
    /// Maximum buffer length in bytes
    pub max_buffer_length: usize,
    /// Maximum threads per threadgroup
    pub max_threads_per_threadgroup: u32,
    /// Maximum threadgroup memory length in bytes
    pub max_threadgroup_memory_length: u32,
    /// Recommended max working set size in bytes
    pub recommended_max_working_set_size: u64,
    /// Whether the device supports unified memory
    pub has_unified_memory: bool,
    /// GPU family/architecture
    pub gpu_family: GpuFamily,
}

/// GPU family/architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFamily {
    /// Apple Silicon (M1, M2, M3, M4 series)
    AppleSilicon,
    /// Intel integrated graphics
    Intel,
    /// AMD discrete graphics
    Amd,
    /// Unknown GPU family
    Unknown,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            name: "Unknown Device".to_string(),
            registry_id: 0,
            is_low_power: false,
            is_headless: false,
            is_removable: false,
            max_buffer_length: 256 * 1024 * 1024, // 256MB default
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory_length: 32768,
            recommended_max_working_set_size: 1024 * 1024 * 1024,
            has_unified_memory: false,
            gpu_family: GpuFamily::Unknown,
        }
    }
}

/// A Metal GPU device
///
/// This struct wraps a Metal device and provides methods for creating
/// resources (buffers, textures) and command queues.
///
/// ## Platform Support
///
/// On non-macOS platforms, all methods return `MetalError::NotAvailable`.
pub struct MetalDevice {
    /// Device information
    info: DeviceInfo,
    /// Internal device handle — real Metal device on macOS
    #[cfg(target_os = "macos")]
    handle: metal::Device,
    #[cfg(not(target_os = "macos"))]
    _handle: (),
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("name", &self.info.name)
            .field("registry_id", &self.info.registry_id)
            .finish()
    }
}

/// Detect GPU family from device name.
#[cfg(target_os = "macos")]
fn detect_gpu_family(name: &str) -> GpuFamily {
    if name.contains("Apple") {
        GpuFamily::AppleSilicon
    } else if name.contains("Intel") {
        GpuFamily::Intel
    } else if name.contains("AMD") || name.contains("Radeon") {
        GpuFamily::Amd
    } else {
        GpuFamily::Unknown
    }
}

/// Build DeviceInfo from a Metal device.
#[cfg(target_os = "macos")]
fn build_device_info(device: &metal::DeviceRef) -> DeviceInfo {
    let name = device.name().to_string();
    let gpu_family = detect_gpu_family(&name);
    let max_tpg = device.max_threads_per_threadgroup();

    DeviceInfo {
        name,
        registry_id: device.registry_id(),
        is_low_power: device.is_low_power(),
        is_headless: device.is_headless(),
        is_removable: device.is_removable(),
        max_buffer_length: device.max_buffer_length() as usize,
        max_threads_per_threadgroup: max_tpg.width as u32,
        max_threadgroup_memory_length: device.max_threadgroup_memory_length() as u32,
        recommended_max_working_set_size: device.recommended_max_working_set_size(),
        has_unified_memory: device.has_unified_memory(),
        gpu_family,
    }
}

impl MetalDevice {
    /// Get the default Metal device
    ///
    /// Returns the system default GPU, which is typically the most
    /// capable discrete GPU or the integrated GPU if no discrete GPU
    /// is available.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not running on macOS
    /// - No Metal-capable GPU is found
    /// - Device creation fails
    #[cfg(target_os = "macos")]
    pub fn system_default() -> MetalResult<Self> {
        let device = metal::Device::system_default().ok_or(MetalError::NoDevice)?;
        let info = build_device_info(&device);
        Ok(Self {
            info,
            handle: device,
        })
    }

    /// Get the default Metal device (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn system_default() -> MetalResult<Self> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Enumerate all available Metal devices
    ///
    /// Returns a list of all Metal-capable GPUs in the system.
    /// This includes integrated, discrete, and external GPUs.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not running on macOS
    /// - Device enumeration fails
    #[cfg(target_os = "macos")]
    pub fn all_devices() -> MetalResult<Vec<Self>> {
        let devices = metal::Device::all();
        if devices.is_empty() {
            return Err(MetalError::NoDevice);
        }
        Ok(devices
            .into_iter()
            .map(|device| {
                let info = build_device_info(&device);
                Self {
                    info,
                    handle: device,
                }
            })
            .collect())
    }

    /// Enumerate all available Metal devices (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn all_devices() -> MetalResult<Vec<Self>> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get device information
    #[must_use]
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Get the device name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Check if this is an Apple Silicon GPU
    #[must_use]
    pub fn is_apple_silicon(&self) -> bool {
        self.info.gpu_family == GpuFamily::AppleSilicon
    }

    /// Check if this device has unified memory
    #[must_use]
    pub fn has_unified_memory(&self) -> bool {
        self.info.has_unified_memory
    }

    /// Get the maximum buffer size in bytes
    #[must_use]
    pub fn max_buffer_length(&self) -> usize {
        self.info.max_buffer_length
    }

    /// Get the maximum threads per threadgroup
    #[must_use]
    pub fn max_threads_per_threadgroup(&self) -> u32 {
        self.info.max_threads_per_threadgroup
    }

    /// Create a buffer with the specified size
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the buffer in bytes
    /// * `usage` - How the buffer will be used
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Size exceeds device maximum
    /// - Allocation fails
    #[cfg(target_os = "macos")]
    pub fn create_buffer(
        &self,
        size: usize,
        usage: crate::BufferUsage,
    ) -> MetalResult<MetalBuffer> {
        if size > self.info.max_buffer_length {
            return Err(MetalError::BufferTooLarge {
                size,
                max_size: self.info.max_buffer_length,
            });
        }
        let options = usage.metal_resource_options();
        let buffer = self.handle.new_buffer(size as u64, options);
        Ok(MetalBuffer::from_metal(buffer, size, usage))
    }

    /// Create a buffer (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn create_buffer(
        &self,
        _size: usize,
        _usage: crate::BufferUsage,
    ) -> MetalResult<MetalBuffer> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Create a typed buffer for a specific element type
    ///
    /// This is a convenience method that creates a buffer sized for
    /// the specified number of elements.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Element type (must be `Copy` + `Sized`)
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of data to upload
    /// * `usage` - How the buffer will be used
    #[cfg(target_os = "macos")]
    pub fn create_buffer_with_data<T: Copy>(
        &self,
        data: &[T],
        usage: crate::BufferUsage,
    ) -> MetalResult<MetalBuffer> {
        let size = std::mem::size_of_val(data);
        if size > self.info.max_buffer_length {
            return Err(MetalError::BufferTooLarge {
                size,
                max_size: self.info.max_buffer_length,
            });
        }
        let options = usage.metal_resource_options();
        let buffer = self.handle.new_buffer_with_data(
            data.as_ptr().cast::<std::ffi::c_void>(),
            size as u64,
            options,
        );
        Ok(MetalBuffer::from_metal(buffer, size, usage))
    }

    /// Create a typed buffer with data (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn create_buffer_with_data<T: Copy>(
        &self,
        _data: &[T],
        _usage: crate::BufferUsage,
    ) -> MetalResult<MetalBuffer> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Create a command queue
    ///
    /// Command queues are used to submit command buffers to the GPU.
    /// Multiple command queues can be created for parallel submission.
    ///
    /// # Errors
    ///
    /// Returns an error if command queue creation fails.
    #[cfg(target_os = "macos")]
    pub fn create_command_queue(&self) -> MetalResult<MetalCommandQueue> {
        let queue = self.handle.new_command_queue();
        Ok(MetalCommandQueue::from_metal(queue))
    }

    /// Create a command queue (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn create_command_queue(&self) -> MetalResult<MetalCommandQueue> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Create a library from compiled Metal shader bytecode
    ///
    /// The bytecode should be in AIR (Apple Intermediate Representation)
    /// format, typically produced by the Metal compiler.
    ///
    /// # Arguments
    ///
    /// * `bytecode` - Compiled Metal shader bytecode
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Bytecode is invalid
    /// - Library creation fails
    #[cfg(target_os = "macos")]
    pub fn create_library(&self, bytecode: &[u8]) -> MetalResult<MetalLibrary> {
        if bytecode.is_empty() {
            return Err(MetalError::LibraryCompilation {
                message: "empty bytecode".to_string(),
            });
        }
        let library = self
            .handle
            .new_library_with_data(bytecode)
            .map_err(|e| MetalError::LibraryCompilation { message: e })?;
        Ok(MetalLibrary::from_metal(library))
    }

    /// Create a library (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn create_library(&self, _bytecode: &[u8]) -> MetalResult<MetalLibrary> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Create a library from Metal Shading Language source code
    ///
    /// This compiles MSL source at runtime. For production use,
    /// prefer pre-compiled bytecode with `create_library`.
    ///
    /// # Arguments
    ///
    /// * `source` - Metal Shading Language source code
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Source code has syntax errors
    /// - Compilation fails
    #[cfg(target_os = "macos")]
    pub fn create_library_from_source(&self, source: &str) -> MetalResult<MetalLibrary> {
        if source.is_empty() {
            return Err(MetalError::LibraryCompilation {
                message: "empty source code".to_string(),
            });
        }
        let options = metal::CompileOptions::new();
        let library = self
            .handle
            .new_library_with_source(source, &options)
            .map_err(|e| MetalError::LibraryCompilation { message: e })?;
        Ok(MetalLibrary::from_metal(library))
    }

    /// Create a library from source (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn create_library_from_source(&self, _source: &str) -> MetalResult<MetalLibrary> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Create a compute pipeline from a library function
    ///
    /// # Arguments
    ///
    /// * `library` - Metal library containing the function
    /// * `function_name` - Name of the compute kernel function
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Function not found in library
    /// - Pipeline creation fails
    #[cfg(target_os = "macos")]
    pub fn create_compute_pipeline(
        &self,
        library: &MetalLibrary,
        function_name: &str,
    ) -> MetalResult<MetalComputePipeline> {
        let function = library
            .get_metal_function(function_name)
            .map_err(|e| MetalError::FunctionNotFound {
                name: format!("{function_name}: {e}"),
            })?;
        let pipeline = self
            .handle
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::PipelineCreation { message: e })?;
        Ok(MetalComputePipeline::from_metal(
            pipeline,
            function_name.to_string(),
        ))
    }

    /// Create a compute pipeline (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn create_compute_pipeline(
        &self,
        _library: &MetalLibrary,
        _function_name: &str,
    ) -> MetalResult<MetalComputePipeline> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get internal device ID (registry ID on macOS)
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn device_id(&self) -> u64 {
        self.info.registry_id
    }

    /// Get internal device ID (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn device_id(&self) -> u64 {
        0
    }

    /// Get a reference to the underlying Metal device (macOS only).
    #[cfg(target_os = "macos")]
    pub(crate) fn metal_device(&self) -> &metal::DeviceRef {
        &self.handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_default() {
        let info = DeviceInfo::default();
        assert!(!info.name.is_empty());
        assert!(info.max_threads_per_threadgroup > 0);
    }

    #[test]
    fn test_gpu_family() {
        assert_eq!(GpuFamily::AppleSilicon, GpuFamily::AppleSilicon);
        assert_ne!(GpuFamily::AppleSilicon, GpuFamily::Intel);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_system_default_succeeds_on_macos() {
        // On macOS, Metal should be available.
        let device = MetalDevice::system_default();
        assert!(device.is_ok(), "Metal device should be available on macOS");
        let device = device.unwrap();
        assert!(!device.name().is_empty());
        assert!(device.max_buffer_length() > 0);
        assert!(device.max_threads_per_threadgroup() > 0);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_all_devices_succeeds_on_macos() {
        let devices = MetalDevice::all_devices();
        assert!(devices.is_ok());
        let devices = devices.unwrap();
        assert!(!devices.is_empty());
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_system_default_returns_not_available() {
        let result = MetalDevice::system_default();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, MetalError::NotAvailable { .. }));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_all_devices_returns_not_available() {
        let result = MetalDevice::all_devices();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, MetalError::NotAvailable { .. }));
    }
}
