//! SigQL Query AST
//!
//! Represents a complete SigQL query with all clauses.

use smol_str::SmolStr;

use super::expr::{
    AggregateOp, CorrelateOp, Literal, ScalarExpr, SignalExpr, SourceRef, WindowSpec,
};
use crate::types::Seconds;

/// A complete SigQL query
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    /// WITH clause - common table expressions / signal definitions
    pub with: Vec<WithClause>,

    /// FROM clause - signal sources
    pub from: Vec<FromClause>,

    /// LET clause - intermediate variable bindings
    pub let_bindings: Vec<LetBinding>,

    /// WHERE clause - temporal and conditional filters
    pub where_clause: Option<WhereClause>,

    /// TRANSFORM clause - DSP operations (can be multiple, applied in order)
    pub transforms: Vec<TransformClause>,

    /// WINDOW clause - temporal windowing
    pub window: Option<WindowClause>,

    /// CORRELATE clause - cross-signal operations
    pub correlate: Option<CorrelateClause>,

    /// AGGREGATE clause - reduction operations
    pub aggregate: Option<AggregateClause>,

    /// INTERPRET clause - clinical/domain interpretation
    pub interpret: Option<InterpretClause>,

    /// RETURNING clause - output specification with uncertainty
    pub returning: ReturningClause,
}

impl Query {
    /// Create a new empty query
    pub fn new() -> Self {
        Self {
            with: Vec::new(),
            from: Vec::new(),
            let_bindings: Vec::new(),
            where_clause: None,
            transforms: Vec::new(),
            window: None,
            correlate: None,
            aggregate: None,
            interpret: None,
            returning: ReturningClause::default(),
        }
    }

    /// Add a source to FROM clause
    pub fn from_source(mut self, source: SourceRef) -> Self {
        self.from.push(FromClause::Signal(source));
        self
    }
}

impl Default for Query {
    fn default() -> Self {
        Self::new()
    }
}

/// WITH clause for reusable signal definitions
#[derive(Debug, Clone, PartialEq)]
pub struct WithClause {
    /// Name of the CTE
    pub name: SmolStr,
    /// Signal expression
    pub expr: SignalExpr,
    /// Whether this is materialized (computed once) or inline
    pub materialized: bool,
}

/// FROM clause sources
#[derive(Debug, Clone, PartialEq)]
pub enum FromClause {
    /// Direct signal reference
    Signal(SourceRef),
    /// Session reference with patient/timestamp
    Session {
        session_id: SmolStr,
        patient: Option<SmolStr>,
        timestamp: Option<i64>,
    },
    /// Subquery
    Subquery { query: Box<Query>, alias: SmolStr },
    /// Table/view reference
    Table {
        name: SmolStr,
        alias: Option<SmolStr>,
    },
    /// Media source (image, audio, video) with automatic ingest
    Media {
        source: MediaSourceRef,
        alias: SmolStr,
    },
    /// Knowledge graph traversal (subsumes SigSPARQL)
    Graph {
        start_node: SmolStr,
        edge_type: Option<SmolStr>,
        depth: usize,
        alias: SmolStr,
    },
}

/// Reference to a media source for MediaQL queries
#[derive(Debug, Clone, PartialEq)]
pub enum MediaSourceRef {
    /// File path or URL
    Path(SmolStr),
    /// Reference to stored media in amorphic engine
    Stored { collection: SmolStr, id: SmolStr },
    /// Inline bytes with format hint
    Bytes { format: SmolStr },
}

/// LET binding for intermediate values
#[derive(Debug, Clone, PartialEq)]
pub struct LetBinding {
    pub name: SmolStr,
    pub expr: SignalExpr,
}

/// WHERE clause with temporal and value filters
#[derive(Debug, Clone, PartialEq)]
pub struct WhereClause {
    pub conditions: Vec<WhereCondition>,
    pub combinator: LogicalCombinator,
}

/// Individual WHERE conditions
#[derive(Debug, Clone, PartialEq)]
pub enum WhereCondition {
    /// Scalar comparison
    Scalar(ScalarExpr),
    /// Temporal range filter
    TimeRange {
        start: Option<TimeSpec>,
        end: Option<TimeSpec>,
    },
    /// Session filter
    Session(SessionFilter),
    /// Task/phase filter
    TaskPhase {
        task: Option<SmolStr>,
        phase: Option<SmolStr>,
    },
    /// Quality filter
    Quality {
        min_snr: Option<f64>,
        max_artifacts: Option<f64>,
    },
    /// Nested conditions
    Nested {
        conditions: Vec<WhereCondition>,
        combinator: LogicalCombinator,
    },
}

/// Time specification
#[derive(Debug, Clone, PartialEq)]
pub enum TimeSpec {
    Absolute(i64),     // nanoseconds since epoch
    Relative(Seconds), // relative to session start
    Named(SmolStr),    // named marker (e.g., "stimulus_onset")
}

/// Session filter
#[derive(Debug, Clone, PartialEq)]
pub struct SessionFilter {
    pub patient_id: Option<SmolStr>,
    pub session_id: Option<SmolStr>,
    pub date_range: Option<(i64, i64)>,
    pub diagnosis: Option<Vec<SmolStr>>,
}

/// Logical combinator for conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogicalCombinator {
    #[default]
    And,
    Or,
}

/// TRANSFORM clause
#[derive(Debug, Clone, PartialEq)]
pub struct TransformClause {
    pub transforms: Vec<TransformItem>,
}

/// Individual transform item (potentially with name)
#[derive(Debug, Clone, PartialEq)]
pub struct TransformItem {
    pub op: super::expr::TransformOp,
    pub alias: Option<SmolStr>,
}

/// WINDOW clause
#[derive(Debug, Clone, PartialEq)]
pub struct WindowClause {
    pub spec: WindowSpec,
    pub partition_by: Vec<SmolStr>,
    pub order_by: Option<OrderSpec>,
}

/// Order specification
#[derive(Debug, Clone, PartialEq)]
pub struct OrderSpec {
    pub field: SmolStr,
    pub direction: OrderDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrderDirection {
    #[default]
    Asc,
    Desc,
}

/// CORRELATE clause
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelateClause {
    /// Signal pairs to correlate
    pub pairs: Vec<CorrelatePair>,
    /// Operations to perform
    pub operations: Vec<CorrelateItem>,
    /// Approximation mode for large datasets
    pub approximation: Option<ApproximationSpec>,
}

/// A pair of signals to correlate
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelatePair {
    pub signal_a: SmolStr,
    pub signal_b: SmolStr,
}

/// Individual correlation operation
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelateItem {
    pub name: SmolStr,
    pub op: CorrelateOp,
}

/// Approximation specification for large datasets
#[derive(Debug, Clone, PartialEq)]
pub struct ApproximationSpec {
    pub error_rate: f64, // e.g., 0.01 for 1% error
}

/// AGGREGATE clause
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateClause {
    pub aggregations: Vec<AggregateItem>,
}

/// Individual aggregation
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateItem {
    pub name: SmolStr,
    pub op: AggregateOp,
    pub input: Option<SmolStr>, // Optional input signal reference
}

/// INTERPRET clause for domain-specific interpretation
#[derive(Debug, Clone, PartialEq)]
pub struct InterpretClause {
    pub rules: Vec<InterpretRule>,
}

/// Interpretation rule
#[derive(Debug, Clone, PartialEq)]
pub struct InterpretRule {
    pub condition: ScalarExpr,
    pub interpretation: SmolStr,
    pub severity: Option<Severity>,
}

/// Severity levels for interpretations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

/// RETURNING clause - output specification
#[derive(Debug, Clone, PartialEq)]
pub struct ReturningClause {
    /// Confidence level (default 0.95)
    pub confidence: f64,
    /// Uncertainty method
    pub uncertainty_method: UncertaintyMethod,
    /// Output format
    pub format: OutputFormat,
    /// Export formats
    pub export: Vec<ExportFormat>,
    /// Fields to include in output
    pub fields: Vec<OutputField>,
}

impl Default for ReturningClause {
    fn default() -> Self {
        Self {
            confidence: 0.95,
            uncertainty_method: UncertaintyMethod::default(),
            format: OutputFormat::default(),
            export: Vec::new(),
            fields: Vec::new(),
        }
    }
}

/// Uncertainty quantification method
#[derive(Debug, Clone, PartialEq, Default)]
pub enum UncertaintyMethod {
    /// Analytical error propagation
    #[default]
    Analytical,
    /// Bootstrap resampling
    Bootstrap { replicates: usize },
    /// Monte Carlo
    MonteCarlo { samples: usize },
    /// Bayesian inference
    Bayesian { prior: Option<SmolStr> },
}

/// Output format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Value,
    NaturalLanguage,
    TimeFrequencyPlot,
    Table,
    Json,
}

/// Export format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Csv,
    Json,
    Parquet,
    FhirObservation,
    Hl7,
    Edf,
}

/// Output field specification
#[derive(Debug, Clone, PartialEq)]
pub struct OutputField {
    pub name: SmolStr,
    pub include_bounds: bool,
    pub include_metadata: bool,
}

/// Materialized view definition
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedView {
    pub name: SmolStr,
    pub query: Query,
    pub refresh: RefreshPolicy,
}

/// Refresh policy for materialized views
#[derive(Debug, Clone, PartialEq)]
pub enum RefreshPolicy {
    Manual,
    OnWrite,
    Periodic { interval: Seconds },
    Continuous,
}

/// Statement types (beyond queries)
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Query(Query),
    CreateView(MaterializedView),
    DropView(SmolStr),
    Explain(Box<Statement>),
    Set { key: SmolStr, value: Literal },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::expr::TransformOp;

    #[test]
    fn test_query_builder() {
        let query = Query::new().from_source(SourceRef::new("controller.imu.accel"));

        assert_eq!(query.from.len(), 1);
    }

    #[test]
    fn test_default_returning() {
        let returning = ReturningClause::default();
        assert!((returning.confidence - 0.95).abs() < 0.001);
    }
}
