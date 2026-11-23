//! Unit tests for Deterministic Execution

#![cfg(feature = "deterministic")]

#[cfg(test)]
mod tests {
    use joule_db_browser::deterministic::{CoordinationStrategy, MultiTabCoordinator};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_multitab_coordinator_creation() {
        let coordinator = MultiTabCoordinator::new("weblocks".to_string()).await;
        assert!(
            coordinator.is_ok(),
            "Coordinator creation should succeed in browser: {:?}",
            coordinator.err()
        );
    }

    #[wasm_bindgen_test]
    async fn test_coordination_strategy_parsing() {
        // Test Web Locks strategy
        let coordinator_wl = MultiTabCoordinator::new("weblocks".to_string()).await;
        assert!(
            coordinator_wl.is_ok(),
            "WebLocks coordinator should succeed: {:?}",
            coordinator_wl.err()
        );

        // Test SharedArrayBuffer strategy
        let coordinator_sab = MultiTabCoordinator::new("sab".to_string()).await;
        assert!(
            coordinator_sab.is_ok(),
            "SAB coordinator should succeed: {:?}",
            coordinator_sab.err()
        );

        // Test BroadcastChannel strategy
        let coordinator_bc = MultiTabCoordinator::new("broadcast".to_string()).await;
        assert!(
            coordinator_bc.is_ok(),
            "BroadcastChannel coordinator should succeed: {:?}",
            coordinator_bc.err()
        );
    }

    #[wasm_bindgen_test]
    async fn test_lock_acquisition() {
        let coord = MultiTabCoordinator::new("weblocks".to_string())
            .await
            .expect("Coordinator creation should succeed");
        let result = coord.acquire_lock("test_lock").await;
        assert!(
            result.is_ok(),
            "Lock acquisition should succeed: {:?}",
            result.err()
        );
    }

    #[wasm_bindgen_test]
    async fn test_transaction_coordination() {
        let coord = MultiTabCoordinator::new("weblocks".to_string())
            .await
            .expect("Coordinator creation should succeed");
        let result = coord.coordinate_transaction("tx_123").await;
        assert!(
            result.is_ok(),
            "Transaction coordination should succeed: {:?}",
            result.err()
        );
    }

    #[wasm_bindgen_test]
    async fn test_coordination_strategy_enum() {
        // Test enum variants
        assert_eq!(
            CoordinationStrategy::WebLocks,
            CoordinationStrategy::WebLocks
        );
        assert_ne!(
            CoordinationStrategy::WebLocks,
            CoordinationStrategy::SharedArrayBuffer
        );
    }
}
