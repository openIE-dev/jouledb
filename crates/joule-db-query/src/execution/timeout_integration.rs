//! Timeout Integration for Query Execution
//!
//! Integrates the timeout system with query execution to provide
//! automatic timeout handling and cancellation.

use crate::error::QueryResult;
use crate::execution::{QueryContext, ResultSet};
use crate::planner::PlanNode;
use crate::timeout::{CancellationToken, CheckpointContext, QueryTimeout, TimeoutConfig};

/// Execute a plan node with timeout and cancellation support
pub fn execute_with_timeout<F>(
    node: &PlanNode,
    context: &QueryContext,
    executor: F,
) -> QueryResult<ResultSet>
where
    F: FnOnce(&PlanNode, &QueryContext) -> QueryResult<ResultSet>,
{
    // Create cancellation token
    let token = CancellationToken::new();

    // Create timeout config from context
    let timeout_config = context
        .timeout
        .map(|d| TimeoutConfig::new(d))
        .unwrap_or_else(|| TimeoutConfig::default());

    // Create query timeout wrapper
    let query_timeout = QueryTimeout::new(timeout_config, token.clone());

    // Execute with checkpoint support
    query_timeout.execute_with_checkpoints(|checkpoint_ctx| {
        // Execute the plan node
        let result = executor(node, context)?;

        // Check for cancellation periodically
        checkpoint_ctx.checkpoint()?;

        Ok(result)
    })
}

/// Execute a plan node with periodic checkpoint checks
pub fn execute_with_checkpoints<F>(
    node: &PlanNode,
    context: &QueryContext,
    checkpoint_ctx: &CheckpointContext,
    executor: F,
) -> QueryResult<ResultSet>
where
    F: Fn(&PlanNode, &QueryContext) -> QueryResult<ResultSet>,
{
    // Check cancellation before starting
    checkpoint_ctx.checkpoint()?;

    // Execute
    let result = executor(node, context)?;

    // Check cancellation after execution
    checkpoint_ctx.checkpoint()?;

    Ok(result)
}

/// Add timeout support to QueryContext
impl QueryContext {
    /// Create cancellation token for this context
    pub fn cancellation_token(&self) -> CancellationToken {
        CancellationToken::new()
    }

    /// Create timeout config from context
    pub fn timeout_config(&self) -> TimeoutConfig {
        self.timeout
            .map(|d| TimeoutConfig::new(d))
            .unwrap_or_else(|| TimeoutConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::PlanNode;
    use std::time::Duration;

    #[test]
    fn test_timeout_integration() {
        let context = QueryContext::new().with_timeout(Duration::from_millis(100));

        let token = context.cancellation_token();
        let config = context.timeout_config();

        assert_eq!(config.query_timeout, Duration::from_millis(100));
        assert!(!token.is_cancelled());
    }
}
