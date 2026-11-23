//! Data transformation for visualization
//!
//! Converts query results to GPU-friendly formats with zero-copy when possible.

use joule_db_core::persistence::compute::{BufferHandle, BufferUsage, ComputeBackend};
use joule_db_query::ResultSet;

/// Transforms query results to visualization-ready data
pub struct QueryResultTransformer;

impl QueryResultTransformer {
    /// Transform to GPU buffer (zero-copy when possible)
    pub fn to_gpu_buffer(
        backend: &mut dyn ComputeBackend,
        results: &ResultSet,
    ) -> Result<BufferHandle, String> {
        // Extract numeric data
        let mut data = Vec::new();
        for row in results.rows() {
            for value in row.values() {
                if let Some(num) = value.as_f64() {
                    data.push(num as f32);
                }
            }
        }

        // Create GPU buffer
        let buffer = backend
            .create_buffer(
                (data.len() * 4) as u64, // f32 = 4 bytes
                BufferUsage::STORAGE_READ,
                Some("viz_data"),
            )
            .map_err(|e| format!("Buffer creation failed: {}", e))?;

        // Write data
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4).to_vec()
        };
        backend
            .write_buffer(buffer, 0, &bytes)
            .map_err(|e| format!("Buffer write failed: {}", e))?;

        Ok(buffer)
    }

    /// Transform to CPU array (for fallback)
    pub fn to_array(results: &ResultSet) -> Vec<f32> {
        let mut data = Vec::new();
        for row in results.rows() {
            for value in row.values() {
                if let Some(num) = value.as_f64() {
                    data.push(num as f32);
                }
            }
        }
        data
    }
}
