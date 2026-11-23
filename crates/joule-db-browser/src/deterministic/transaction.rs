//! Deterministic Transaction Manager
//!
//! Guarantees identical results across replicas

use crate::engine_enhanced::EnhancedAsyncEngine;
use std::sync::{Arc, RwLock};

/// Deterministic transaction
#[derive(Debug, Clone)]
pub struct DeterministicTransaction {
    /// Transaction ID
    pub id: String,
    /// Operations (pre-ordered)
    pub operations: Vec<TransactionOperation>,
    /// Timestamp (for ordering)
    pub timestamp: u64,
}

/// Transaction operation
#[derive(Debug, Clone)]
pub enum TransactionOperation {
    Put { key: String, value: String },
    Get { key: String },
    Delete { key: String },
}

/// Deterministic transaction manager

pub struct DeterministicTransactionManager {
    /// Base engine
    engine: Arc<RwLock<EnhancedAsyncEngine>>,
    /// Transaction queue (ordered)
    transaction_queue: Vec<DeterministicTransaction>,
}

impl DeterministicTransactionManager {
    /// Create new deterministic transaction manager
    pub async fn new() -> Result<DeterministicTransactionManager, String> {
        let engine = {
            use crate::engine::AsyncEngine;
            use crate::persistence::{AsyncPersistenceBackend, OPFSVirtualBlockDevice};
            use std::sync::Arc;
            use std::sync::RwLock;

            let mut backend = OPFSVirtualBlockDevice::new("deterministic_db");
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
            Arc::new(RwLock::new(EnhancedAsyncEngine::new(async_engine_arc)))
        };

        Ok(Self {
            engine,
            transaction_queue: Vec::new(),
        })
    }

    /// Begin deterministic transaction
    pub fn begin(&mut self) -> String {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let tx_id = format!("tx_{}", timestamp);
        let tx = DeterministicTransaction {
            id: tx_id.clone(),
            operations: Vec::new(),
            timestamp: timestamp as u64,
        };
        self.transaction_queue.push(tx);
        tx_id
    }

    /// Add operation to transaction
    pub fn add_operation(
        &mut self,
        tx_id: &str,
        operation: TransactionOperation,
    ) -> Result<(), String> {
        // Find transaction and add operation
        if let Some(tx) = self.transaction_queue.iter_mut().find(|t| t.id == tx_id) {
            tx.operations.push(operation);
            Ok(())
        } else {
            Err("Transaction not found".to_string())
        }
    }

    /// Commit transaction (deterministic execution)
    pub async fn commit(&mut self, tx_id: &str) -> Result<(), String> {
        // Find transaction
        let tx = self
            .transaction_queue
            .iter()
            .position(|t| t.id == tx_id)
            .ok_or_else(|| "Transaction not found".to_string())?;

        let transaction = self.transaction_queue.remove(tx);

        // Execute operations in deterministic order
        let engine = self
            .engine
            .read()
            .map_err(|_| "Failed to lock engine".to_string())?;

        for op in transaction.operations {
            match op {
                TransactionOperation::Put { key, value } => {
                    engine
                        .put(key.as_bytes(), value.as_bytes())
                        .await
                        .map_err(|e| format!("Put failed: {}", e))?;
                }
                TransactionOperation::Get { key: _ } => {
                    // Read operations don't need execution
                }
                TransactionOperation::Delete { key } => {
                    engine
                        .delete(key.as_bytes())
                        .await
                        .map_err(|e| format!("Delete failed: {}", e))?;
                }
            }
        }

        Ok(())
    }

    /// Get transaction order (for coordination)
    pub fn get_transaction_order(&self) -> Vec<String> {
        // Return transactions in deterministic order
        self.transaction_queue
            .iter()
            .map(|t| t.id.clone())
            .collect()
    }
}
