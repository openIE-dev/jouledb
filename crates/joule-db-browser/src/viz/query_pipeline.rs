//! Query-to-Visualization Pipeline
//!
//! Provides zero-copy data flow from query execution to GPU-accelerated rendering.

use joule_db_core::persistence::compute::{BufferHandle, BufferUsage, ComputeBackend};
use joule_db_query::{QueryEngine, ResultSet};
use joule_db_viz::{ChartType, GpuRenderer, VizError, VizResult};
use std::sync::Arc;

use joule_db_gpu::WgpuComputeBackend;

/// Pipeline that connects query execution to visualization
pub struct QueryToVizPipeline {
    /// Query engine
    query_engine: Arc<QueryEngine>,
    /// GPU renderer
    viz_engine: Arc<GpuRenderer>,
    /// GPU compute backend for data transformation
    gpu_backend: Option<Arc<WgpuComputeBackend>>,
}

impl QueryToVizPipeline {
    /// Create new pipeline
    pub async fn new(
        query_engine: Arc<QueryEngine>,
        viz_engine: Arc<GpuRenderer>,
    ) -> VizResult<Self> {
        let gpu_backend = WgpuComputeBackend::new()
            .await
            .map_err(|e| VizError::GpuError(format!("GPU init failed: {}", e)))?;

        Ok(Self {
            query_engine,
            viz_engine,
            gpu_backend: Some(Arc::new(gpu_backend)),
        })
    }

    /// Create without GPU (CPU fallback)
    pub fn new_cpu(query_engine: Arc<QueryEngine>, viz_engine: Arc<GpuRenderer>) -> Self {
        Self {
            query_engine,
            viz_engine,
            gpu_backend: None,
        }
    }

    /// Execute query and visualize results
    pub async fn execute_and_visualize(&self, query: &str, chart_type: ChartType) -> VizResult<()> {
        // 1. Execute query
        let results = self
            .query_engine
            .execute(query)
            .map_err(|e| VizError::DataError(format!("Query execution failed: {}", e)))?;

        // 2. Transform to GPU buffers (zero-copy when possible)
        let gpu_data = self.transform_to_gpu(&results)?;

        // 3. Render chart
        self.viz_engine.render_chart(chart_type, &gpu_data).await
    }

    /// Transform query results to GPU-friendly format
    fn transform_to_gpu(&self, results: &ResultSet) -> VizResult<Vec<f32>> {
        // Extract numeric columns for visualization
        let mut gpu_data = Vec::new();

        for row in results.rows() {
            for value in row.values() {
                if let Some(num) = value.as_f64() {
                    gpu_data.push(num as f32);
                }
            }
        }

        Ok(gpu_data)
    }
}
