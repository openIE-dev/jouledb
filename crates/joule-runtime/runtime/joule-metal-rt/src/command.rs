//! Metal Command Queue and Buffer
//!
//! This module provides command submission infrastructure for Metal.
//! Commands are recorded into command buffers and submitted via command queues.
//!
//! ## Command Execution Model
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
//! │ MetalCommandQueue│────►│MetalCommandBuffer│────►│   GPU Execution │
//! └─────────────────┘     └─────────────────┘     └─────────────────┘
//!         │                       │
//!         │                       ├── Compute Commands
//!         │                       ├── Blit Commands
//!         │                       └── Render Commands
//!         │
//!         └── Multiple buffers can be in flight
//! ```

use crate::error::{MetalError, MetalResult};
use crate::pipeline::MetalComputePipeline;
use crate::{BlockSize, GridSize, MetalBuffer};

/// Command buffer status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferStatus {
    /// Not yet enqueued
    NotEnqueued,
    /// Enqueued but not yet scheduled
    Enqueued,
    /// Scheduled for execution
    Committed,
    /// Currently executing on GPU
    Scheduled,
    /// Execution completed successfully
    Completed,
    /// Execution failed with error
    Error,
}

/// Command queue for submitting GPU work
///
/// A command queue manages command buffer submission to the GPU.
/// Multiple command buffers can be submitted concurrently for
/// maximum throughput.
pub struct MetalCommandQueue {
    /// Real Metal command queue (macOS only)
    #[cfg(target_os = "macos")]
    handle: metal::CommandQueue,
    #[cfg(not(target_os = "macos"))]
    _marker: std::marker::PhantomData<()>,
}

impl std::fmt::Debug for MetalCommandQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalCommandQueue").finish()
    }
}

impl MetalCommandQueue {
    /// Create a MetalCommandQueue from a real metal::CommandQueue (macOS only).
    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal(queue: metal::CommandQueue) -> Self {
        Self { handle: queue }
    }

    /// Create a new command buffer
    ///
    /// Command buffers are used to record commands for GPU execution.
    /// They are one-shot: once committed, they cannot be reused.
    #[cfg(target_os = "macos")]
    pub fn create_command_buffer(&self) -> MetalResult<MetalCommandBuffer> {
        let cmd_buf = self.handle.new_command_buffer().to_owned();
        Ok(MetalCommandBuffer::from_metal(cmd_buf))
    }

    /// Create command buffer (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn create_command_buffer(&self) -> MetalResult<MetalCommandBuffer> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Launch a compute kernel
    ///
    /// This is a convenience method that creates a command buffer,
    /// encodes the kernel dispatch, and submits it synchronously.
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The compute pipeline to execute
    /// * `grid_size` - Total number of threads (width, height, depth)
    /// * `block_size` - Threads per threadgroup (width, height, depth)
    /// * `buffers` - Buffers to bind (bound in order: 0, 1, 2, ...)
    #[cfg(target_os = "macos")]
    pub fn launch_kernel(
        &self,
        pipeline: &MetalComputePipeline,
        grid_size: GridSize,
        block_size: BlockSize,
        buffers: &[&MetalBuffer],
    ) -> MetalResult<()> {
        let mut cmd_buf = self.create_command_buffer()?;
        cmd_buf.encode_compute(pipeline, grid_size, block_size, buffers)?;
        cmd_buf.commit()?;
        cmd_buf.wait_until_completed()
    }

    /// Launch kernel (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn launch_kernel(
        &self,
        _pipeline: &MetalComputePipeline,
        _grid_size: GridSize,
        _block_size: BlockSize,
        _buffers: &[&MetalBuffer],
    ) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Launch kernel asynchronously
    ///
    /// Returns immediately after submission. Use the returned command
    /// buffer to check status or wait for completion.
    #[cfg(target_os = "macos")]
    pub fn launch_kernel_async(
        &self,
        pipeline: &MetalComputePipeline,
        grid_size: GridSize,
        block_size: BlockSize,
        buffers: &[&MetalBuffer],
    ) -> MetalResult<MetalCommandBuffer> {
        let mut cmd_buf = self.create_command_buffer()?;
        cmd_buf.encode_compute(pipeline, grid_size, block_size, buffers)?;
        cmd_buf.commit()?;
        Ok(cmd_buf)
    }

    /// Launch kernel async (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn launch_kernel_async(
        &self,
        _pipeline: &MetalComputePipeline,
        _grid_size: GridSize,
        _block_size: BlockSize,
        _buffers: &[&MetalBuffer],
    ) -> MetalResult<MetalCommandBuffer> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get queue handle ID (for debugging)
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        // CommandQueue has no direct ID in the Metal API.
        1
    }

    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        0
    }
}

/// A command buffer for recording GPU commands
///
/// Command buffers are created from a command queue and used to
/// record commands (compute dispatches, memory copies, etc.).
/// Once recording is complete, the buffer is committed for execution.
pub struct MetalCommandBuffer {
    /// Current status
    status: CommandBufferStatus,
    /// Real Metal command buffer (macOS only)
    #[cfg(target_os = "macos")]
    handle: metal::CommandBuffer,
    #[cfg(not(target_os = "macos"))]
    _marker: std::marker::PhantomData<()>,
}

impl std::fmt::Debug for MetalCommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalCommandBuffer")
            .field("status", &self.status)
            .finish()
    }
}

/// Convert Metal SDK status to our enum.
#[cfg(target_os = "macos")]
fn translate_status(status: metal::MTLCommandBufferStatus) -> CommandBufferStatus {
    match status {
        metal::MTLCommandBufferStatus::NotEnqueued => CommandBufferStatus::NotEnqueued,
        metal::MTLCommandBufferStatus::Enqueued => CommandBufferStatus::Enqueued,
        metal::MTLCommandBufferStatus::Committed => CommandBufferStatus::Committed,
        metal::MTLCommandBufferStatus::Scheduled => CommandBufferStatus::Scheduled,
        metal::MTLCommandBufferStatus::Completed => CommandBufferStatus::Completed,
        metal::MTLCommandBufferStatus::Error => CommandBufferStatus::Error,
    }
}

impl MetalCommandBuffer {
    /// Create a MetalCommandBuffer from a real metal::CommandBuffer (macOS only).
    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal(cmd_buf: metal::CommandBuffer) -> Self {
        Self {
            status: CommandBufferStatus::NotEnqueued,
            handle: cmd_buf,
        }
    }

    /// Get the current status
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn status(&self) -> CommandBufferStatus {
        translate_status(self.handle.status())
    }

    /// Get the current status (non-macOS)
    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn status(&self) -> CommandBufferStatus {
        self.status
    }

    /// Encode a compute dispatch
    ///
    /// Records a compute kernel dispatch to be executed when the
    /// command buffer is committed.
    #[cfg(target_os = "macos")]
    pub fn encode_compute(
        &mut self,
        pipeline: &MetalComputePipeline,
        grid_size: GridSize,
        block_size: BlockSize,
        buffers: &[&MetalBuffer],
    ) -> MetalResult<()> {
        let encoder = self.handle.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline.metal_pipeline());

        for (i, buf) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buf.metal_buffer()), 0);
        }

        let grid = metal::MTLSize::new(
            grid_size.0 as u64,
            grid_size.1 as u64,
            grid_size.2 as u64,
        );
        let block = metal::MTLSize::new(
            block_size.0 as u64,
            block_size.1 as u64,
            block_size.2 as u64,
        );
        encoder.dispatch_thread_groups(grid, block);
        encoder.end_encoding();
        Ok(())
    }

    /// Encode compute (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn encode_compute(
        &mut self,
        _pipeline: &MetalComputePipeline,
        _grid_size: GridSize,
        _block_size: BlockSize,
        _buffers: &[&MetalBuffer],
    ) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Encode a memory copy (blit) operation
    ///
    /// Copies data between buffers on the GPU.
    #[cfg(target_os = "macos")]
    pub fn encode_copy(
        &mut self,
        source: &MetalBuffer,
        source_offset: usize,
        dest: &MetalBuffer,
        dest_offset: usize,
        size: usize,
    ) -> MetalResult<()> {
        let encoder = self.handle.new_blit_command_encoder();
        encoder.copy_from_buffer(
            source.metal_buffer(),
            source_offset as u64,
            dest.metal_buffer(),
            dest_offset as u64,
            size as u64,
        );
        encoder.end_encoding();
        Ok(())
    }

    /// Encode copy (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn encode_copy(
        &mut self,
        _source: &MetalBuffer,
        _source_offset: usize,
        _dest: &MetalBuffer,
        _dest_offset: usize,
        _size: usize,
    ) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Commit the command buffer for execution
    ///
    /// After this call, no more commands can be encoded.
    /// The GPU will begin executing when resources are available.
    #[cfg(target_os = "macos")]
    pub fn commit(&mut self) -> MetalResult<()> {
        self.handle.commit();
        self.status = CommandBufferStatus::Committed;
        Ok(())
    }

    /// Commit (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn commit(&mut self) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Wait for the command buffer to complete
    ///
    /// Blocks until GPU execution is complete.
    #[cfg(target_os = "macos")]
    pub fn wait_until_completed(&mut self) -> MetalResult<()> {
        self.handle.wait_until_completed();
        let status = self.handle.status();
        self.status = translate_status(status);
        if status == metal::MTLCommandBufferStatus::Error {
            return Err(MetalError::CommandExecution {
                message: "command buffer execution failed on GPU".to_string(),
            });
        }
        Ok(())
    }

    /// Wait until completed (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn wait_until_completed(&mut self) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Wait with timeout
    ///
    /// # Arguments
    ///
    /// * `timeout_ms` - Maximum time to wait in milliseconds
    #[cfg(target_os = "macos")]
    pub fn wait_until_completed_timeout(&mut self, timeout_ms: u64) -> MetalResult<()> {
        let start = std::time::Instant::now();
        let deadline = std::time::Duration::from_millis(timeout_ms);
        loop {
            let status = self.handle.status();
            self.status = translate_status(status);
            match status {
                metal::MTLCommandBufferStatus::Completed => return Ok(()),
                metal::MTLCommandBufferStatus::Error => {
                    return Err(MetalError::CommandExecution {
                        message: "command buffer execution failed on GPU".to_string(),
                    });
                }
                _ => {
                    if start.elapsed() >= deadline {
                        return Err(MetalError::Timeout { timeout_ms });
                    }
                    std::thread::yield_now();
                }
            }
        }
    }

    /// Wait with timeout (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn wait_until_completed_timeout(&mut self, _timeout_ms: u64) -> MetalResult<()> {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Check if execution is complete without blocking
    #[must_use]
    pub fn is_completed(&self) -> bool {
        matches!(
            self.status(),
            CommandBufferStatus::Completed | CommandBufferStatus::Error
        )
    }

    /// Add a completion handler
    ///
    /// The callback will be invoked when execution completes.
    /// Implemented by spawning a thread that waits for completion.
    #[cfg(target_os = "macos")]
    pub fn add_completed_handler<F>(&mut self, handler: F) -> MetalResult<()>
    where
        F: FnOnce(CommandBufferStatus) + Send + 'static,
    {
        // Clone the command buffer status query into a background thread
        // that polls for completion and calls the handler.
        let handle = self.handle.to_owned();
        std::thread::spawn(move || {
            handle.wait_until_completed();
            let status = translate_status(handle.status());
            handler(status);
        });
        Ok(())
    }

    /// Add completion handler (non-macOS stub)
    #[cfg(not(target_os = "macos"))]
    pub fn add_completed_handler<F>(&mut self, _handler: F) -> MetalResult<()>
    where
        F: FnOnce(CommandBufferStatus) + Send + 'static,
    {
        Err(MetalError::NotAvailable {
            reason: "Metal is only supported on macOS/iOS".to_string(),
        })
    }

    /// Get buffer handle ID (for debugging)
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        1
    }

    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn handle_id(&self) -> u64 {
        0
    }
}

// Safety: Metal command queue and command buffer are thread-safe
unsafe impl Send for MetalCommandQueue {}
unsafe impl Sync for MetalCommandQueue {}
unsafe impl Send for MetalCommandBuffer {}
unsafe impl Sync for MetalCommandBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_buffer_status() {
        assert_ne!(
            CommandBufferStatus::NotEnqueued,
            CommandBufferStatus::Completed
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_command_queue_and_buffer() {
        let device = crate::MetalDevice::system_default().unwrap();
        let queue = device.create_command_queue().unwrap();
        let cmd_buf = queue.create_command_buffer().unwrap();
        assert_eq!(cmd_buf.status(), CommandBufferStatus::NotEnqueued);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_device_not_available_for_command_queue() {
        let result = crate::MetalDevice::system_default();
        assert!(result.is_err());
    }
}
