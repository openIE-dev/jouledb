//! Metal Synchronization Primitives
//!
//! This module provides synchronization primitives for coordinating
//! GPU execution and CPU/GPU communication.
//!
//! ## Synchronization Types
//!
//! - `MetalFence` - GPU-internal synchronization between encoders
//! - `MetalEvent` - Binary signal for simple coordination
//! - `MetalSharedEvent` - Cross-process event with counter
//!
//! ## Usage
//!
//! ```ignore
//! // Create a shared event for CPU/GPU synchronization
//! let event = device.create_shared_event()?;
//!
//! // Signal from GPU
//! cmd_buffer.encode_signal_event(&event, 1)?;
//!
//! // Wait on CPU
//! event.wait_until_signaled(1, 1000)?;
//! ```

use crate::device::MetalDevice;
use crate::error::{MetalError, MetalResult};

/// A fence for GPU-internal synchronization
///
/// Fences are used to ensure ordering between command encoders
/// within the same command buffer or across command buffers
/// on the same device.
pub struct MetalFence {
    /// Device this fence belongs to
    device_id: u64,
    /// Real Metal fence handle (macOS only)
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    handle: metal::Fence,
    #[cfg(not(target_os = "macos"))]
    _marker: std::marker::PhantomData<()>,
}

impl std::fmt::Debug for MetalFence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalFence")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl MetalFence {
    /// Create a new fence
    #[cfg(target_os = "macos")]
    pub fn new(device: &MetalDevice) -> MetalResult<Self> {
        let fence = device.metal_device().new_fence();
        Ok(Self {
            device_id: device.device_id(),
            handle: fence,
        })
    }

    /// Create fence (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn new(_device: &MetalDevice) -> MetalResult<Self> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get fence handle ID (for debugging)
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        self.device_id
    }
}

/// A binary event for simple synchronization
///
/// Events can be signaled from the GPU and waited on from the CPU,
/// or vice versa. They are binary (signaled/not signaled) and
/// auto-reset after waiting.
pub struct MetalEvent {
    /// Device this event belongs to
    device_id: u64,
    /// Whether the event is signaled (CPU-side tracking)
    signaled: std::sync::atomic::AtomicBool,
    /// Real Metal event handle (macOS only)
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    handle: metal::Event,
    #[cfg(not(target_os = "macos"))]
    _marker: std::marker::PhantomData<()>,
}

impl std::fmt::Debug for MetalEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalEvent")
            .field("device_id", &self.device_id)
            .field(
                "signaled",
                &self.signaled.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl MetalEvent {
    /// Create a new event
    #[cfg(target_os = "macos")]
    pub fn new(device: &MetalDevice) -> MetalResult<Self> {
        let event = device.metal_device().new_event();
        Ok(Self {
            device_id: device.device_id(),
            signaled: std::sync::atomic::AtomicBool::new(false),
            handle: event,
        })
    }

    /// Create event (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn new(_device: &MetalDevice) -> MetalResult<Self> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Check if the event is signaled
    #[must_use]
    pub fn is_signaled(&self) -> bool {
        self.signaled.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Signal the event (CPU-side)
    pub fn signal(&self) {
        self.signaled
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Reset the event
    pub fn reset(&self) {
        self.signaled
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Wait for the event to be signaled
    ///
    /// Blocks until the event is signaled or timeout expires.
    ///
    /// # Arguments
    ///
    /// * `timeout_ms` - Maximum time to wait in milliseconds
    #[cfg(target_os = "macos")]
    pub fn wait(&self, timeout_ms: u64) -> MetalResult<()> {
        let start = std::time::Instant::now();
        let deadline = std::time::Duration::from_millis(timeout_ms);
        while !self.signaled.load(std::sync::atomic::Ordering::Acquire) {
            if start.elapsed() >= deadline {
                return Err(MetalError::Timeout { timeout_ms });
            }
            std::thread::yield_now();
        }
        // Auto-reset after successful wait.
        self.signaled
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Wait (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn wait(&self, _timeout_ms: u64) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get event handle ID (for debugging)
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        self.device_id
    }
}

/// A shared event with counter for fine-grained synchronization
///
/// Shared events have a 64-bit counter that can be signaled
/// and waited on from both CPU and GPU. They can also be
/// shared across processes for multi-process GPU coordination.
pub struct MetalSharedEvent {
    /// Device this event belongs to
    device_id: u64,
    /// CPU-side counter mirror
    counter: std::sync::atomic::AtomicU64,
    /// Real Metal shared event handle (macOS only)
    #[cfg(target_os = "macos")]
    handle: metal::SharedEvent,
    #[cfg(not(target_os = "macos"))]
    _marker: std::marker::PhantomData<()>,
}

impl std::fmt::Debug for MetalSharedEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalSharedEvent")
            .field("device_id", &self.device_id)
            .field(
                "counter",
                &self.counter.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl MetalSharedEvent {
    /// Create a new shared event
    #[cfg(target_os = "macos")]
    pub fn new(device: &MetalDevice) -> MetalResult<Self> {
        let event = device.metal_device().new_shared_event();
        Ok(Self {
            device_id: device.device_id(),
            counter: std::sync::atomic::AtomicU64::new(0),
            handle: event,
        })
    }

    /// Create shared event (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn new(_device: &MetalDevice) -> MetalResult<Self> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get the current counter value
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn signaled_value(&self) -> u64 {
        self.handle.signaled_value()
    }

    /// Get the current counter value (non-macOS)
    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn signaled_value(&self) -> u64 {
        self.counter.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Signal the event with a new counter value (CPU-side)
    ///
    /// The counter can only move forward; attempting to signal
    /// a lower value is ignored.
    #[cfg(target_os = "macos")]
    pub fn signal(&self, value: u64) {
        self.handle.set_signaled_value(value);
        // Mirror to our atomic for local reads.
        let mut current = self.counter.load(std::sync::atomic::Ordering::Relaxed);
        while value > current {
            match self.counter.compare_exchange_weak(
                current,
                value,
                std::sync::atomic::Ordering::Release,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_current) => current = new_current,
            }
        }
    }

    /// Signal (non-macOS)
    #[cfg(not(target_os = "macos"))]
    pub fn signal(&self, value: u64) {
        let mut current = self.counter.load(std::sync::atomic::Ordering::Relaxed);
        while value > current {
            match self.counter.compare_exchange_weak(
                current,
                value,
                std::sync::atomic::Ordering::Release,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_current) => current = new_current,
            }
        }
    }

    /// Wait until the counter reaches or exceeds the specified value
    #[cfg(target_os = "macos")]
    pub fn wait_until_signaled(&self, value: u64, timeout_ms: u64) -> MetalResult<()> {
        let start = std::time::Instant::now();
        let deadline = std::time::Duration::from_millis(timeout_ms);
        loop {
            if self.handle.signaled_value() >= value {
                return Ok(());
            }
            if start.elapsed() >= deadline {
                return Err(MetalError::Timeout { timeout_ms });
            }
            std::thread::yield_now();
        }
    }

    /// Wait until signaled (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn wait_until_signaled(&self, _value: u64, _timeout_ms: u64) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Add a listener for when the counter reaches a specific value
    ///
    /// Implemented by spawning a thread that polls the counter.
    #[cfg(target_os = "macos")]
    pub fn notify_listener<F>(&self, value: u64, callback: F) -> MetalResult<()>
    where
        F: FnOnce(u64) + Send + 'static,
    {
        let handle = self.handle.to_owned();
        std::thread::spawn(move || {
            loop {
                let current = handle.signaled_value();
                if current >= value {
                    callback(current);
                    break;
                }
                std::thread::yield_now();
            }
        });
        Ok(())
    }

    /// Add listener (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn notify_listener<F>(&self, _value: u64, _callback: F) -> MetalResult<()>
    where
        F: FnOnce(u64) + Send + 'static,
    {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get a shared event handle for cross-process sharing
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn shared_handle(&self) -> u64 {
        self.handle.signaled_value()
    }

    /// Get shared handle (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn shared_handle(&self) -> u64 {
        0
    }

    /// Get event handle ID (for debugging)
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        self.device_id
    }
}

// Safety: These sync primitives are thread-safe
unsafe impl Send for MetalFence {}
unsafe impl Sync for MetalFence {}
unsafe impl Send for MetalEvent {}
unsafe impl Sync for MetalEvent {}
unsafe impl Send for MetalSharedEvent {}
unsafe impl Sync for MetalSharedEvent {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_fence_creation() {
        let device = MetalDevice::system_default().unwrap();
        let fence = MetalFence::new(&device);
        assert!(fence.is_ok());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_event_creation_and_signal() {
        let device = MetalDevice::system_default().unwrap();
        let event = MetalEvent::new(&device).unwrap();
        assert!(!event.is_signaled());
        event.signal();
        assert!(event.is_signaled());
        event.reset();
        assert!(!event.is_signaled());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_shared_event_creation() {
        let device = MetalDevice::system_default().unwrap();
        let event = MetalSharedEvent::new(&device).unwrap();
        assert_eq!(event.signaled_value(), 0);
        event.signal(42);
        assert!(event.signaled_value() >= 42);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_fence_not_available() {
        let result = MetalDevice::system_default();
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_event_not_available() {
        let result = MetalDevice::system_default();
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_shared_event_not_available() {
        let result = MetalDevice::system_default();
        assert!(result.is_err());
    }
}
