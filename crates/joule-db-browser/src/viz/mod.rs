//! Visualization pipeline for query results
//!
//! Provides direct data flow from queries to GPU-accelerated visualization.

pub mod query_pipeline;
pub mod streaming;
pub mod transform;

pub use query_pipeline::QueryToVizPipeline;
pub use streaming::RealtimeQueryStream;
pub use transform::QueryResultTransformer;
