//! HTAP (Hybrid Transaction-Analytical Processing) Module
//!
//! Implements browser-native HTAP architecture combining:
//! - OLTP: B-tree storage engine (transactional)
//! - OLAP: GPU-accelerated columnar processing (analytical)
//! - Zero-copy format conversion
//! - Automatic query routing

pub mod columnar;
#[cfg(feature = "compression-test")]
pub mod compression_test;
pub mod converter;
pub mod router;
pub mod storage;

pub use columnar::{ColumnarFormat, ColumnarStore};
#[cfg(feature = "compression-test")]
pub use compression_test::{
    CompressionResults, CompressionTestRunner, DataGenerator, DataType, SpecificDataGenerator,
    analyze_5gb_compression, format_results_table, generate_compression_report,
};
pub use converter::{ColumnDef, ColumnarSchema, ConversionStrategy, FormatConverter};
pub use router::{QueryRouter, QueryType, RoutingDecision};
pub use storage::{HTAPConfig, HTAPStorage};
