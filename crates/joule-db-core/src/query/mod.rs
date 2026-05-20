//! Query execution and optimization

pub mod triple_buffer;

pub use triple_buffer::{QueryHandle, TripleBufferedQueries, TripleBufferedQueryManager};
