//! HTAP Storage: Unified Storage for OLTP and OLAP
//!
//! Manages both B-tree (OLTP) and columnar (OLAP) storage

use super::{ColumnarStore, FormatConverter, QueryRouter};
use crate::engine_enhanced::EnhancedAsyncEngine;
use std::sync::{Arc, RwLock};

/// HTAP storage configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HTAPConfig {
    /// Enable columnar storage
    pub enable_columnar: bool,
    /// OLAP threshold (rows)
    pub olap_threshold: usize,
    /// Conversion strategy
    pub conversion_strategy: super::converter::ConversionStrategy,
}

impl Default for HTAPConfig {
    fn default() -> Self {
        Self {
            enable_columnar: true,
            olap_threshold: 1000,
            conversion_strategy: super::converter::ConversionStrategy::ZeroCopy,
        }
    }
}

/// HTAP storage engine (internal, no WASM bindings)
pub struct HTAPStorage {
    /// OLTP engine (B-tree)
    oltp_engine: Arc<RwLock<EnhancedAsyncEngine>>,
    /// OLAP store (columnar)
    olap_store: Option<Arc<ColumnarStore>>,
    /// Query router
    router: Arc<QueryRouter>,
    /// Format converter
    converter: Arc<FormatConverter>,
    /// Configuration
    config: HTAPConfig,
}

impl HTAPStorage {
    /// Create new HTAP storage
    pub async fn new(config: HTAPConfig) -> Result<HTAPStorage, String> {
        // Create base async engine first
        use crate::engine::AsyncEngine;
        use crate::persistence::{AsyncPersistenceBackend, OPFSVirtualBlockDevice};
        use std::sync::Arc;
        use std::sync::RwLock;

        let mut backend = OPFSVirtualBlockDevice::new("htap_db");
        backend
            .init()
            .await
            .map_err(|e| format!("OPFS VBD init failed: {}", e))?;

        let backend_arc: Arc<RwLock<dyn AsyncPersistenceBackend + Send + Sync>> =
            Arc::new(RwLock::new(backend));

        let async_engine = AsyncEngine::open(backend_arc)
            .await
            .map_err(|e| format!("Failed to open engine: {}", e))?;

        let async_engine_arc = Arc::new(RwLock::new(async_engine));
        let oltp_engine = Arc::new(RwLock::new(EnhancedAsyncEngine::new(async_engine_arc)));

        // Create query router
        let router = Arc::new(QueryRouter::new(config.olap_threshold));

        // Create format converter
        let converter = Arc::new(FormatConverter::new(config.conversion_strategy));

        // Create OLAP store if enabled
        let olap_store = if config.enable_columnar {
            Some(Arc::new(ColumnarStore::new(vec![], vec![])))
        } else {
            None
        };

        Ok(Self {
            oltp_engine,
            olap_store,
            router,
            converter,
            config,
        })
    }

    /// Execute query (automatic routing)
    pub async fn query(&self, query: &str) -> Result<String, String> {
        let decision = self.router.route(query);

        match decision.query_type {
            super::router::QueryType::OLTP => {
                // Execute on B-tree
                self.execute_oltp(query).await
            }
            super::router::QueryType::OLAP => {
                // Execute on columnar
                if let Some(ref olap) = self.olap_store {
                    olap.query(query).await
                } else {
                    // Fallback to OLTP
                    self.execute_oltp(query).await
                }
            }
            super::router::QueryType::Hybrid => {
                // Execute on both and combine
                self.execute_hybrid(query, &decision).await
            }
        }
    }

    /// Execute OLTP query
    async fn execute_oltp(&self, query: &str) -> Result<String, String> {
        // Use existing EnhancedAsyncEngine
        let engine = self
            .oltp_engine
            .read()
            .map_err(|_| "Failed to lock engine".to_string())?;

        // Execute query via query engine
        // This is a simplified implementation - full query execution would go here
        Ok(format!("Query executed: {}", query))
    }

    /// Execute hybrid query
    async fn execute_hybrid(
        &self,
        query: &str,
        _decision: &super::router::RoutingDecision,
    ) -> Result<String, String> {
        // For hybrid queries, execute OLTP part first, then convert to columnar for OLAP
        // This is a simplified implementation
        self.execute_oltp(query).await
    }
}
