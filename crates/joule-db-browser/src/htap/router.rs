//! Query Router: Automatic OLTP vs OLAP Routing
//!
//! Routes queries to appropriate engine based on query characteristics

use wasm_bindgen::prelude::*;

/// Query type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// OLTP: Transactional query (point lookups, updates)
    OLTP,
    /// OLAP: Analytical query (aggregations, scans)
    OLAP,
    /// Hybrid: Requires both OLTP and OLAP
    Hybrid,
}

/// Routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub query_type: QueryType,
    pub use_btree: bool,
    pub use_columnar: bool,
    pub explanation: String,
}

/// Query router

pub struct QueryRouter {
    /// Threshold for OLAP classification (number of rows)
    olap_threshold: usize,
}

impl QueryRouter {
    /// Create new query router

    pub fn new(olap_threshold: usize) -> Self {
        Self { olap_threshold }
    }

    /// Route query to appropriate engine
    pub fn route(&self, query: &str) -> RoutingDecision {
        // Parse query to determine type
        let query_type = self.classify_query(query);

        let (use_btree, use_columnar) = match query_type {
            QueryType::OLTP => (true, false),
            QueryType::OLAP => (false, true),
            QueryType::Hybrid => (true, true),
        };

        RoutingDecision {
            query_type,
            use_btree,
            use_columnar,
            explanation: format!("Query classified as {:?}", query_type),
        }
    }

    /// Classify query type
    fn classify_query(&self, query: &str) -> QueryType {
        let query_lower = query.to_lowercase();

        // OLAP indicators
        let olap_keywords = [
            "select", "sum(", "avg(", "count(", "max(", "min(", "group by", "order by", "having",
            "join",
        ];
        let is_olap = olap_keywords.iter().any(|kw| query_lower.contains(kw));

        // OLTP indicators
        let oltp_keywords = ["insert", "update", "delete", "where id =", "where key ="];
        let is_oltp = oltp_keywords.iter().any(|kw| query_lower.contains(kw));

        // Hybrid if both present
        if is_olap && is_oltp {
            QueryType::Hybrid
        } else if is_olap {
            QueryType::OLAP
        } else if is_oltp {
            QueryType::OLTP
        } else {
            // Default to OLTP for safety
            QueryType::OLTP
        }
    }
}
