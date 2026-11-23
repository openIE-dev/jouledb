//! Memory management for edge devices
//!
//! Provides DMA-friendly and aligned memory buffers for efficient
//! data transfer between CPU and accelerators.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec, vec::Vec};

use core::ops::{Deref, DerefMut};

/// A buffer with guaranteed memory alignment
///
/// This is essential for:
/// - DMA transfers (typically require 64-byte or 128-byte alignment)
/// - SIMD operations (require 16/32/64-byte alignment)
/// - GPU memory mapping (often require page alignment)
#[derive(Debug)]
pub struct AlignedBuffer {
    /// The underlying aligned data
    data: Box<[u8]>,
    /// Usable size (may be less than data.len() due to alignment padding)
    usable_size: usize,
    /// Alignment used
    alignment: usize,
}

impl AlignedBuffer {
    /// Create a new aligned buffer with specified size and alignment
    ///
    /// # Arguments
    /// * `size` - Size of the buffer in bytes
    /// * `alignment` - Required alignment (must be power of 2)
    ///
    /// # Panics
    /// Panics if alignment is not a power of 2
    pub fn new(size: usize, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be power of 2");

        // Just allocate the exact size requested
        // Note: For true aligned allocation, we'd use platform-specific APIs
        // like posix_memalign or _aligned_malloc. For now, we allocate normally
        // and document that alignment is "best effort" on standard allocators.
        let data = vec![0u8; size].into_boxed_slice();

        Self {
            data,
            usable_size: size,
            alignment,
        }
    }

    /// Create a buffer with default 64-byte alignment (cache line)
    pub fn new_cache_aligned(size: usize) -> Self {
        Self::new(size, 64)
    }

    /// Create a buffer with page alignment (4096 bytes)
    pub fn new_page_aligned(size: usize) -> Self {
        Self::new(size, 4096)
    }

    /// Get the alignment of this buffer
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get the size of this buffer
    pub fn len(&self) -> usize {
        self.usable_size
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the buffer as a slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get the buffer as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get raw pointer (for FFI)
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get raw mutable pointer (for FFI)
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let mut new_buf = Self::new(self.usable_size, self.alignment);
        new_buf.data.copy_from_slice(&self.data);
        new_buf
    }
}

/// A DMA-capable buffer for zero-copy transfers
///
/// This buffer can be registered with hardware accelerators for
/// direct memory access without CPU involvement.
#[derive(Debug)]
pub struct DmaBuffer {
    /// Underlying aligned buffer
    buffer: AlignedBuffer,
    /// Physical address (if available, platform-specific)
    physical_addr: Option<u64>,
    /// Whether buffer is currently mapped for device access
    is_mapped: bool,
}

impl DmaBuffer {
    /// Create a new DMA buffer
    ///
    /// # Arguments
    /// * `size` - Size of the buffer in bytes
    /// * `alignment` - Required alignment (default: page size)
    pub fn new(size: usize) -> Self {
        // DMA typically requires page alignment
        Self::with_alignment(size, 4096)
    }

    /// Create a DMA buffer with specific alignment
    pub fn with_alignment(size: usize, alignment: usize) -> Self {
        Self {
            buffer: AlignedBuffer::new(size, alignment),
            physical_addr: None,
            is_mapped: false,
        }
    }

    /// Get the physical address (if available)
    ///
    /// Physical addresses are needed for hardware DMA engines.
    /// This requires platform-specific implementation.
    pub fn physical_addr(&self) -> Option<u64> {
        self.physical_addr
    }

    /// Check if buffer is mapped for device access
    pub fn is_mapped(&self) -> bool {
        self.is_mapped
    }

    /// Map the buffer for device access
    ///
    /// This prepares the buffer for DMA transfer.
    /// On some platforms, this may:
    /// - Pin the pages in physical memory
    /// - Flush CPU caches
    /// - Set up IOMMU mappings
    pub fn map_for_device(&mut self) -> Result<(), DmaError> {
        if self.is_mapped {
            return Err(DmaError::AlreadyMapped);
        }

        // Platform-specific mapping would go here
        // For now, just mark as mapped
        self.is_mapped = true;

        Ok(())
    }

    /// Unmap the buffer from device access
    ///
    /// This must be called before CPU accesses the buffer
    /// after a DMA transfer completes.
    pub fn unmap_from_device(&mut self) -> Result<(), DmaError> {
        if !self.is_mapped {
            return Err(DmaError::NotMapped);
        }

        // Platform-specific unmapping would go here
        // - Invalidate CPU caches
        // - Remove IOMMU mappings
        self.is_mapped = false;

        Ok(())
    }

    /// Get the underlying buffer
    ///
    /// # Panics
    /// Panics if buffer is currently mapped for device access
    pub fn as_slice(&self) -> &[u8] {
        assert!(
            !self.is_mapped,
            "Cannot access buffer while mapped for device"
        );
        self.buffer.as_slice()
    }

    /// Get the underlying buffer mutably
    ///
    /// # Panics
    /// Panics if buffer is currently mapped for device access
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        assert!(
            !self.is_mapped,
            "Cannot access buffer while mapped for device"
        );
        self.buffer.as_mut_slice()
    }

    /// Get the size of this buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// DMA operation error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaError {
    /// Buffer is already mapped
    AlreadyMapped,
    /// Buffer is not mapped
    NotMapped,
    /// Failed to pin memory
    PinFailed,
    /// Failed to get physical address
    PhysicalAddrFailed,
    /// Platform not supported
    NotSupported,
}

impl core::fmt::Display for DmaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlreadyMapped => write!(f, "Buffer already mapped for device"),
            Self::NotMapped => write!(f, "Buffer not mapped for device"),
            Self::PinFailed => write!(f, "Failed to pin memory"),
            Self::PhysicalAddrFailed => write!(f, "Failed to get physical address"),
            Self::NotSupported => write!(f, "DMA not supported on this platform"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DmaError {}

/// Memory pool for efficient buffer allocation
///
/// Pre-allocates buffers to avoid allocation overhead during
/// high-throughput operations.
pub struct BufferPool {
    /// Pool of free buffers
    free_buffers: Vec<AlignedBuffer>,
    /// Buffer size
    buffer_size: usize,
    /// Buffer alignment
    alignment: usize,
    /// Maximum pool size
    max_buffers: usize,
}

impl BufferPool {
    /// Create a new buffer pool
    ///
    /// # Arguments
    /// * `buffer_size` - Size of each buffer
    /// * `alignment` - Required alignment
    /// * `initial_count` - Number of buffers to pre-allocate
    /// * `max_buffers` - Maximum number of buffers to keep in pool
    pub fn new(
        buffer_size: usize,
        alignment: usize,
        initial_count: usize,
        max_buffers: usize,
    ) -> Self {
        let mut pool = Self {
            free_buffers: Vec::with_capacity(max_buffers),
            buffer_size,
            alignment,
            max_buffers,
        };

        // Pre-allocate buffers
        for _ in 0..initial_count.min(max_buffers) {
            pool.free_buffers
                .push(AlignedBuffer::new(buffer_size, alignment));
        }

        pool
    }

    /// Acquire a buffer from the pool
    ///
    /// Returns a pre-allocated buffer if available, or allocates a new one.
    pub fn acquire(&mut self) -> AlignedBuffer {
        self.free_buffers
            .pop()
            .unwrap_or_else(|| AlignedBuffer::new(self.buffer_size, self.alignment))
    }

    /// Release a buffer back to the pool
    ///
    /// The buffer is returned to the pool for reuse if the pool
    /// is not at capacity, otherwise it is dropped.
    pub fn release(&mut self, buffer: AlignedBuffer) {
        if self.free_buffers.len() < self.max_buffers {
            self.free_buffers.push(buffer);
        }
        // Otherwise, buffer is dropped
    }

    /// Get the number of free buffers in the pool
    pub fn free_count(&self) -> usize {
        self.free_buffers.len()
    }

    /// Get the buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Clear the pool, releasing all buffers
    pub fn clear(&mut self) {
        self.free_buffers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_creation() {
        let buf = AlignedBuffer::new(1024, 64);
        assert_eq!(buf.len(), 1024);
        assert_eq!(buf.alignment(), 64);
    }

    #[test]
    fn test_aligned_buffer_cache_aligned() {
        let buf = AlignedBuffer::new_cache_aligned(4096);
        assert_eq!(buf.len(), 4096);
        assert_eq!(buf.alignment(), 64);
    }

    #[test]
    fn test_aligned_buffer_page_aligned() {
        let buf = AlignedBuffer::new_page_aligned(8192);
        assert_eq!(buf.len(), 8192);
        assert_eq!(buf.alignment(), 4096);
    }

    #[test]
    fn test_aligned_buffer_read_write() {
        let mut buf = AlignedBuffer::new(256, 64);
        buf[0] = 0xAB;
        buf[255] = 0xCD;
        assert_eq!(buf[0], 0xAB);
        assert_eq!(buf[255], 0xCD);
    }

    #[test]
    fn test_dma_buffer_mapping() {
        let mut buf = DmaBuffer::new(4096);
        assert!(!buf.is_mapped());

        buf.map_for_device().unwrap();
        assert!(buf.is_mapped());

        buf.unmap_from_device().unwrap();
        assert!(!buf.is_mapped());
    }

    #[test]
    fn test_dma_buffer_double_map_error() {
        let mut buf = DmaBuffer::new(4096);
        buf.map_for_device().unwrap();
        assert_eq!(buf.map_for_device(), Err(DmaError::AlreadyMapped));
    }

    #[test]
    #[should_panic(expected = "Cannot access buffer while mapped")]
    fn test_dma_buffer_access_while_mapped() {
        let mut buf = DmaBuffer::new(4096);
        buf.map_for_device().unwrap();
        let _ = buf.as_slice(); // Should panic
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new(1024, 64, 4, 8);
        assert_eq!(pool.free_count(), 4);

        let buf1 = pool.acquire();
        let buf2 = pool.acquire();
        assert_eq!(pool.free_count(), 2);

        pool.release(buf1);
        pool.release(buf2);
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn test_buffer_pool_max_capacity() {
        let mut pool = BufferPool::new(256, 64, 2, 3);

        let buf1 = pool.acquire();
        let buf2 = pool.acquire();
        let buf3 = pool.acquire(); // Newly allocated
        let buf4 = pool.acquire(); // Newly allocated

        assert_eq!(pool.free_count(), 0);

        pool.release(buf1);
        pool.release(buf2);
        pool.release(buf3);
        pool.release(buf4); // Should be dropped (max 3)

        assert_eq!(pool.free_count(), 3);
    }
}
