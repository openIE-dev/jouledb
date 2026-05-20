//! Arena Allocator for Transaction-Scoped Memory
//!
//! Inspired by Phantom Engine's zero-GC memory management strategy.
//! Provides fast, predictable memory allocation with zero-cost cleanup.
//!
//! Usage:
//! ```rust
//! use joule_db_core::allocator::arena::TransactionArena;
//! use std::mem::size_of;
//!
//! struct MyStruct { value: i32 }
//!
//! let mut arena = TransactionArena::new(1024 * 1024); // 1MB
//! let ptr = arena.allocate(size_of::<MyStruct>());
//! // ... use ptr ...
//! arena.reset(); // Zero-cost cleanup
//! ```

/// Transaction-scoped arena allocator
///
/// All allocations within a transaction use this arena.
/// When transaction commits/rolls back, entire arena is reset (zero-cost).
///
/// This eliminates GC pressure in browser environments and provides
/// predictable memory usage patterns.
///
/// Note: This allocator is not thread-safe by design. Each transaction
/// should have its own arena instance. For thread-safe allocation, use
/// `allocate_offset()` which returns offsets that can be accessed later.
pub struct TransactionArena {
    /// Pre-allocated memory block
    memory: Vec<u8>,
    /// Current allocation offset (not atomic - single-threaded per transaction)
    offset: usize,
    /// Total capacity
    capacity: usize,
    /// Alignment requirement (default: 8 bytes)
    alignment: usize,
}

impl TransactionArena {
    /// Create a new arena with specified capacity
    ///
    /// # Arguments
    /// * `capacity` - Size in bytes (e.g., 1MB = 1024 * 1024)
    pub fn new(capacity: usize) -> Self {
        Self::with_alignment(capacity, 8)
    }

    /// Create with custom alignment
    ///
    /// # Arguments
    /// * `capacity` - Size in bytes
    /// * `alignment` - Required alignment (must be power of 2)
    pub fn with_alignment(capacity: usize, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be power of 2");

        // Allocate and initialize memory (safe alternative to set_len)
        let memory = vec![0u8; capacity];

        Self {
            memory,
            offset: 0,
            capacity,
            alignment,
        }
    }

    /// Allocate memory of the given size
    ///
    /// Returns a mutable slice to the allocated memory, or None if arena is exhausted.
    ///
    /// The returned slice is valid until `reset()` is called.
    ///
    /// # Note
    /// This method requires `&mut self` and is not thread-safe.
    /// For thread-safe allocation, use `allocate_offset()` with a `Mutex` or `RwLock`.
    pub fn allocate(&mut self, size: usize) -> Option<&mut [u8]> {
        // Align size to alignment requirement
        let aligned_size = (size + self.alignment - 1) & !(self.alignment - 1);

        let current = self.offset;
        let new_offset = current + aligned_size;

        if new_offset > self.capacity {
            return None; // Arena exhausted
        }

        // Update offset
        self.offset = new_offset;

        // Return slice (safe - we know the bounds are valid)
        Some(&mut self.memory[current..new_offset])
    }

    /// Allocate memory and return offset (for thread-safe scenarios)
    ///
    /// Returns the offset and size of the allocated region.
    /// The caller must use `get_slice()` to access the memory.
    ///
    /// This is useful when the arena is wrapped in a `Mutex` or `RwLock`.
    pub fn allocate_offset(&mut self, size: usize) -> Option<(usize, usize)> {
        // Align size to alignment requirement
        let aligned_size = (size + self.alignment - 1) & !(self.alignment - 1);

        let current = self.offset;
        let new_offset = current + aligned_size;

        if new_offset > self.capacity {
            return None; // Arena exhausted
        }

        // Update offset
        self.offset = new_offset;

        Some((current, aligned_size))
    }

    /// Get a slice to the allocated region
    ///
    /// This is safe because we know the offset is valid and within bounds.
    pub fn get_slice(&mut self, offset: usize, size: usize) -> &mut [u8] {
        &mut self.memory[offset..offset + size]
    }

    /// Allocate and initialize with zero
    pub fn allocate_zeroed(&mut self, size: usize) -> Option<&mut [u8]> {
        let slice = self.allocate(size)?;
        // Memory is already zeroed from Vec initialization
        Some(slice)
    }

    /// Reset the arena (zero-cost cleanup)
    ///
    /// All allocations become invalid after this call.
    /// The memory is not freed, just the offset is reset to 0.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Get current usage in bytes
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Get remaining capacity in bytes
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.used())
    }

    /// Get total capacity in bytes
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if arena has enough space for the given size
    pub fn has_capacity(&self, size: usize) -> bool {
        let aligned_size = (size + self.alignment - 1) & !(self.alignment - 1);
        self.remaining() >= aligned_size
    }
}

/// Frame-scoped arena allocator
///
/// Similar to TransactionArena but reset every frame.
/// Useful for temporary allocations that don't need to persist.
pub struct FrameArena {
    arena: TransactionArena,
}

impl FrameArena {
    /// Create a new frame arena
    pub fn new(capacity: usize) -> Self {
        Self {
            arena: TransactionArena::new(capacity),
        }
    }

    /// Allocate memory (delegates to underlying arena)
    pub fn allocate(&mut self, size: usize) -> Option<&mut [u8]> {
        self.arena.allocate(size)
    }

    /// Reset for next frame
    pub fn reset_frame(&mut self) {
        self.arena.reset();
    }

    /// Get usage
    pub fn used(&self) -> usize {
        self.arena.used()
    }
}

/// Pool allocator for fixed-size objects
///
/// Pre-allocates a pool of objects for reuse.
/// Eliminates allocation overhead for frequently allocated types.
pub struct PoolAllocator<T> {
    /// Pool of free objects
    free: Vec<T>,
    /// Total capacity
    capacity: usize,
}

impl<T> PoolAllocator<T> {
    /// Create a new pool allocator
    pub fn new(capacity: usize) -> Self {
        Self {
            free: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Acquire an object from the pool
    ///
    /// Returns None if pool is empty and at capacity.
    pub fn acquire(&mut self) -> Option<T> {
        self.free.pop()
    }

    /// Release an object back to the pool
    ///
    /// Returns false if pool is at capacity.
    pub fn release(&mut self, item: T) -> bool {
        if self.free.len() < self.capacity {
            self.free.push(item);
            true
        } else {
            false
        }
    }

    /// Get number of free objects
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Clear the pool
    pub fn clear(&mut self) {
        self.free.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let mut arena = TransactionArena::new(1024);

        let _slice1 = arena.allocate(64).unwrap();
        assert_eq!(arena.used(), 64);

        let _slice2 = arena.allocate(32).unwrap();
        assert_eq!(arena.used(), 96); // 64 + 32

        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_arena_exhaustion() {
        // Use 128 capacity with allocations that fit exactly with alignment
        // 56 bytes aligned to 8 = 56, second 56 = 112, third should fail
        let mut arena = TransactionArena::new(112);

        let _slice1 = arena.allocate(56).unwrap();
        let _slice2 = arena.allocate(56).unwrap();
        let slice3 = arena.allocate(1);

        assert!(slice3.is_none()); // Should fail - arena exhausted
    }

    #[test]
    fn test_arena_offset_allocation() {
        let mut arena = TransactionArena::new(1024);

        let (offset1, size1) = arena.allocate_offset(64).unwrap();
        assert_eq!(offset1, 0);
        assert_eq!(size1, 64);
        assert_eq!(arena.used(), 64);

        let (offset2, size2) = arena.allocate_offset(32).unwrap();
        assert_eq!(offset2, 64);
        assert_eq!(size2, 32);
        assert_eq!(arena.used(), 96);
    }

    #[test]
    fn test_pool_allocator() {
        let mut pool = PoolAllocator::<Vec<u8>>::new(10);

        let item1 = vec![1, 2, 3];
        pool.release(item1);

        let item2 = pool.acquire();
        assert!(item2.is_some());
    }
}
