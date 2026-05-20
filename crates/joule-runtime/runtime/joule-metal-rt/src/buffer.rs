//! Metal Buffer Management
//!
//! This module provides GPU buffer allocation and management for Metal.
//! Buffers are used to transfer data between the CPU and GPU and to provide
//! input/output for compute kernels.
//!
//! ## Buffer Usage
//!
//! Different buffer usage modes affect memory placement and synchronization:
//!
//! - `Shared` - CPU and GPU can both access; automatic synchronization
//! - `Private` - GPU-only access; fastest for GPU computation
//! - `Managed` - Explicit synchronization for optimal performance
//!
//! ## Example
//!
//! ```ignore
//! let device = MetalDevice::default()?;
//!
//! // Create a shared buffer
//! let buffer = device.create_buffer(1024, BufferUsage::Shared)?;
//!
//! // Write data from CPU
//! buffer.write(&[1.0f32, 2.0, 3.0, 4.0])?;
//!
//! // Use buffer in compute kernel
//! // ...
//!
//! // Read results back to CPU
//! let results: Vec<f32> = buffer.read(4)?;
//! ```

use crate::error::{MetalError, MetalResult};

/// Buffer storage mode
///
/// Determines where the buffer is stored and how CPU/GPU synchronization works.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BufferUsage {
    /// Shared between CPU and GPU
    ///
    /// - CPU and GPU can both read/write
    /// - Automatic synchronization
    /// - Best for frequently updated data
    /// - On Apple Silicon (unified memory), no copy is needed
    #[default]
    Shared,

    /// Private to GPU
    ///
    /// - Only GPU can access
    /// - Fastest for GPU-only data
    /// - Must use blit commands to transfer data
    Private,

    /// Managed synchronization
    ///
    /// - CPU and GPU can both access
    /// - Explicit synchronization required
    /// - Best performance for large, infrequently updated data
    Managed,

    /// Memoryless (render targets only)
    ///
    /// - Tile memory only, not backed by system memory
    /// - Contents discarded after render pass
    /// - Only valid for render targets
    Memoryless,
}

impl BufferUsage {
    /// Get the Metal resource options for this usage mode
    #[must_use]
    pub fn resource_options(&self) -> u32 {
        // These values correspond to MTLResourceStorageModeShift and friends
        match self {
            Self::Shared => 0,          // MTLResourceStorageModeShared
            Self::Private => 2 << 4,    // MTLResourceStorageModePrivate
            Self::Managed => 1 << 4,    // MTLResourceStorageModeManaged
            Self::Memoryless => 3 << 4, // MTLResourceStorageModeMemoryless
        }
    }

    /// Convert to real Metal resource options.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn metal_resource_options(&self) -> metal::MTLResourceOptions {
        match self {
            Self::Shared => metal::MTLResourceOptions::StorageModeShared,
            Self::Private => metal::MTLResourceOptions::StorageModePrivate,
            Self::Managed => metal::MTLResourceOptions::StorageModeManaged,
            Self::Memoryless => metal::MTLResourceOptions::StorageModeMemoryless,
        }
    }
}

/// A GPU buffer
///
/// Represents a region of GPU memory that can be used for computation.
/// Depending on the usage mode, the buffer may be accessible from both
/// CPU and GPU.
pub struct MetalBuffer {
    /// Size in bytes
    size: usize,
    /// Usage mode
    usage: BufferUsage,
    /// Real Metal buffer handle (macOS only)
    #[cfg(target_os = "macos")]
    handle: metal::Buffer,
    #[cfg(not(target_os = "macos"))]
    _marker: std::marker::PhantomData<()>,
}

// Safety: MetalBuffer is Send because the underlying MTLBuffer is thread-safe
unsafe impl Send for MetalBuffer {}
// Safety: MetalBuffer is Sync because the underlying MTLBuffer is thread-safe
unsafe impl Sync for MetalBuffer {}

impl std::fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("size", &self.size)
            .field("usage", &self.usage)
            .finish()
    }
}

impl MetalBuffer {
    /// Create a MetalBuffer from a real metal::Buffer (macOS only).
    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal(buffer: metal::Buffer, size: usize, usage: BufferUsage) -> Self {
        Self {
            size,
            usage,
            handle: buffer,
        }
    }

    /// Get the size of the buffer in bytes
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the usage mode
    #[must_use]
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Write data to the buffer
    ///
    /// # Type Parameters
    ///
    /// * `T` - Element type (must be `Copy`)
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of data to write
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer is Private (use blit commands instead)
    /// - Data exceeds buffer size
    #[cfg(target_os = "macos")]
    pub fn write<T: Copy>(&mut self, data: &[T]) -> MetalResult<()> {
        if self.usage == BufferUsage::Private {
            return Err(MetalError::InvalidBufferAccess {
                message: "cannot write to Private buffer from CPU; use blit commands".to_string(),
            });
        }
        let byte_len = std::mem::size_of_val(data);
        if byte_len > self.size {
            return Err(MetalError::InvalidBufferAccess {
                message: format!(
                    "data size ({byte_len} bytes) exceeds buffer size ({} bytes)",
                    self.size
                ),
            });
        }
        let dst = self.handle.contents().cast::<u8>();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), dst, byte_len);
        }
        Ok(())
    }

    /// Write data (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn write<T: Copy>(&mut self, _data: &[T]) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Write data at a specific offset
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset in the buffer
    /// * `data` - Slice of data to write
    #[cfg(target_os = "macos")]
    pub fn write_at_offset<T: Copy>(
        &mut self,
        offset: usize,
        data: &[T],
    ) -> MetalResult<()> {
        if self.usage == BufferUsage::Private {
            return Err(MetalError::InvalidBufferAccess {
                message: "cannot write to Private buffer from CPU; use blit commands".to_string(),
            });
        }
        let byte_len = std::mem::size_of_val(data);
        if offset + byte_len > self.size {
            return Err(MetalError::InvalidBufferAccess {
                message: format!(
                    "write at offset {offset} + {byte_len} bytes exceeds buffer size ({} bytes)",
                    self.size
                ),
            });
        }
        let dst = unsafe { (self.handle.contents().cast::<u8>()).add(offset) };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), dst, byte_len);
        }
        Ok(())
    }

    /// Write data at offset (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn write_at_offset<T: Copy>(&mut self, _offset: usize, _data: &[T]) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Read data from the buffer
    ///
    /// # Type Parameters
    ///
    /// * `T` - Element type (must be `Copy` + `Default`)
    ///
    /// # Arguments
    ///
    /// * `count` - Number of elements to read
    ///
    /// # Returns
    ///
    /// Vector containing the read data
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer is Private (use blit commands instead)
    /// - Requested data exceeds buffer size
    #[cfg(target_os = "macos")]
    pub fn read<T: Copy + Default>(&self, count: usize) -> MetalResult<Vec<T>> {
        if self.usage == BufferUsage::Private {
            return Err(MetalError::InvalidBufferAccess {
                message: "cannot read from Private buffer on CPU; use blit commands".to_string(),
            });
        }
        let byte_len = count * std::mem::size_of::<T>();
        if byte_len > self.size {
            return Err(MetalError::InvalidBufferAccess {
                message: format!(
                    "read of {byte_len} bytes exceeds buffer size ({} bytes)",
                    self.size
                ),
            });
        }
        let mut result = vec![T::default(); count];
        let src = self.handle.contents().cast::<u8>().cast_const();
        unsafe {
            std::ptr::copy_nonoverlapping(src, result.as_mut_ptr().cast::<u8>(), byte_len);
        }
        Ok(result)
    }

    /// Read data (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn read<T: Copy + Default>(&self, _count: usize) -> MetalResult<Vec<T>> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get a raw pointer to the buffer contents
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The buffer is not Private
    /// - No other code is accessing the buffer
    /// - The pointer is not used after the buffer is dropped
    #[cfg(target_os = "macos")]
    #[must_use]
    pub unsafe fn contents_ptr(&self) -> Option<*mut u8> {
        if self.usage == BufferUsage::Private {
            return None;
        }
        Some(self.handle.contents().cast::<u8>())
    }

    /// Get contents pointer (non-macOS stub)
    ///
    /// Always returns `None` because Metal is only supported on macOS/iOS.
    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub unsafe fn contents_ptr(&self) -> Option<*mut u8> {
        None
    }

    /// Synchronize managed buffer from GPU to CPU
    ///
    /// This must be called after GPU writes before reading on CPU
    /// for managed buffers.
    #[cfg(target_os = "macos")]
    pub fn did_modify_range(&self, offset: usize, size: usize) -> MetalResult<()> {
        if self.usage != BufferUsage::Managed {
            return Err(MetalError::InvalidBufferAccess {
                message: "did_modify_range is only valid for Managed buffers".to_string(),
            });
        }
        let range = metal::NSRange::new(offset as u64, size as u64);
        self.handle.did_modify_range(range);
        Ok(())
    }

    /// Synchronize managed buffer (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn did_modify_range(&self, _offset: usize, _size: usize) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get a reference to the underlying metal::Buffer (macOS only).
    #[cfg(target_os = "macos")]
    pub(crate) fn metal_buffer(&self) -> &metal::BufferRef {
        &self.handle
    }

    /// Get buffer handle ID (for debugging)
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        self.handle.gpu_address()
    }

    /// Get buffer handle ID (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_usage_default() {
        let usage = BufferUsage::default();
        assert_eq!(usage, BufferUsage::Shared);
    }

    #[test]
    fn test_buffer_usage_resource_options() {
        assert_eq!(BufferUsage::Shared.resource_options(), 0);
        assert!(BufferUsage::Private.resource_options() > 0);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_buffer_create_write_read() {
        let device = crate::MetalDevice::system_default().unwrap();
        let mut buffer = device
            .create_buffer(256, BufferUsage::Shared)
            .unwrap();
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        buffer.write(&data).unwrap();
        let result: Vec<f32> = buffer.read(4).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_device_not_available_for_buffer() {
        let result = crate::MetalDevice::system_default();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Metal"));
    }
}
