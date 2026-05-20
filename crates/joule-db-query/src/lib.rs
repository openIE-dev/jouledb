//! # JouleDB Query
//!
//! Query language parsers and execution engine for JouleDB.
//!
//! ## Supported Query Languages
//!
//! - **SQL** - Standard SQL subset for relational queries
//! - **Cypher** - Graph query language (Neo4j compatible)
//! - **CQL** - Cassandra Query Language for wide-column stores
//! - **GraphQL** - GraphQL for API queries
//! - **InfluxQL/PromQL** - Time series query languages
//!
//! ## Example
//!
//! ```ignore
//! use joule_db_query::{QueryEngine, sql::SqlParser};
//!
//! let parser = SqlParser::new();
//! let query = parser.parse("SELECT * FROM users WHERE id = 1")?;
//! ```

pub mod ast;
pub mod error;

#[cfg(feature = "sql")]
pub mod sql;

#[cfg(feature = "cypher")]
pub mod cypher;

#[cfg(feature = "cql")]
pub mod cql;

#[cfg(feature = "graphql")]
pub mod graphql;

#[cfg(feature = "datalog")]
pub mod datalog;

#[cfg(feature = "sparql")]
pub mod sparql;

#[cfg(feature = "gremlin")]
pub mod gremlin;

#[cfg(feature = "timeseries")]
pub mod timeseries;

pub mod adaptive;
pub mod columnar;
pub mod execution;
pub mod executor;
pub mod functions;
pub mod geo;
pub mod graph;
pub mod jsonb;
pub mod planner;
pub mod prepared;
pub mod timeout;
pub mod triggers;
pub mod vector;
pub mod wcoj;
pub mod wcoj_cost;

#[cfg(feature = "analytical")]
pub mod analytical;

#[cfg(feature = "storage-executor")]
pub mod storage_executor;

#[cfg(feature = "arrow-execution")]
pub mod arrow;

#[cfg(feature = "datafusion-backend")]
pub mod datafusion_backend;

#[cfg(feature = "amorphic")]
pub mod amorphic_executor;

#[cfg(feature = "hdc")]
pub mod hdc;

// Re-exports
pub use adaptive::{
    AdaptiveOptimizer, AdaptiveOptimizerConfig, CacheStatistics, CardinalityEstimator,
    CostCoefficients, CostModel, ExecutionPathPreference, FeedbackSummary, IndexAdvisor,
    IndexRecommendation, IndexType, PlanArm, PlanCache, PlanCacheConfig, PlanSelector,
    PlanStatistics, QueryFingerprint, RuntimeFeedbackConfig, RuntimeFeedbackLoop,
};
pub use ast::{Expression, Operator, Query, QueryType, Value};
pub use error::{QueryError, QueryResult};
pub use execution::{ExecutionPlan, NoOpStorage, QueryContext, QueryEngine};
pub use executor::{RowData, StorageExecutor, TableStorage};
pub use planner::{PlanNode, QueryPlanner, VectorMetric};
pub use prepared::{
    BoundStatement, CacheStatsSnapshot, PreparedCacheConfig, PreparedError, PreparedStatement,
    PreparedStatementCache,
};
pub use timeout::{
    CancellationReason, CancellationToken, CheckpointContext, QueryTimeout, TimeoutConfig,
    TimeoutStatistics,
};

#[cfg(feature = "storage-executor")]
pub use storage_executor::{
    ExecutionResult, RowData as StorageRowData, SerializedRow,
    StorageExecutor as BTreeStorageExecutor,
};

#[cfg(feature = "arrow-execution")]
pub use arrow::{AggregateOp, ArrowExecutor, FilterOp, RecordBatchBuilder};

#[cfg(feature = "datafusion-backend")]
pub use datafusion_backend::{
    ColumnType, DataFusionBackend, DataFusionConfig, DataFusionError, DataFusionResult,
    DataFusionStats, RegisteredTable,
};

#[cfg(feature = "sql")]
pub use sql::{InsertSource, SqlParser, SqlQuery, SqlStatement, TriggerEvent, TriggerTiming};

#[cfg(feature = "cypher")]
pub use cypher::{CypherParser, CypherPattern, CypherQuery};

#[cfg(feature = "cql")]
pub use cql::{CqlParser, CqlQuery, CqlStatement};

#[cfg(feature = "graphql")]
pub use graphql::{GraphqlField, GraphqlParser, GraphqlQuery};

#[cfg(feature = "timeseries")]
pub use timeseries::{InfluxParser, PromqlParser, TimeSeriesQuery};

#[cfg(feature = "amorphic")]
pub use amorphic_executor::{AmorphicTableStorage, TableSchema};
