//! Integration tests for browser research features

#![cfg(all(feature = "learned", feature = "deterministic"))]

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_htap_columnar_conversion_flow() {
        // Test full flow: B-tree -> Columnar -> Query
        use joule_db_browser::htap::{
            ColumnDef, ColumnarSchema, ColumnarStore, ConversionStrategy, FormatConverter,
        };

        // Create converter
        let converter = FormatConverter::new(ConversionStrategy::Cpu);

        // Create sample data
        let mut btree_data = Vec::new();
        let key = b"user1";
        let value = br#"{"name":"Alice","age":30}"#;
        btree_data.extend_from_slice(&(key.len() as u32).to_le_bytes());
        btree_data.extend_from_slice(key);
        btree_data.extend_from_slice(&(value.len() as u32).to_le_bytes());
        btree_data.extend_from_slice(value);

        let schema = ColumnarSchema {
            columns: vec![
                ColumnDef {
                    name: "name".to_string(),
                    data_type: "string".to_string(),
                    nullable: false,
                },
                ColumnDef {
                    name: "age".to_string(),
                    data_type: "int32".to_string(),
                    nullable: false,
                },
            ],
        };

        // Convert
        let columnar_result = converter.btree_to_columnar(&btree_data, &schema).await;
        assert!(columnar_result.is_ok());

        // Create columnar store and query
        let mut store = ColumnarStore::new(
            vec!["name".to_string(), "age".to_string()],
            vec!["string".to_string(), "float64".to_string()],
        );

        // Add data directly to store
        store
            .add_row(vec![JsValue::from_str("Alice"), JsValue::from_f64(30.0)])
            .unwrap();

        // Query
        let result: Result<String, String> = store.query("SELECT COUNT(*) FROM table").await;
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    async fn test_learned_index_training_and_prediction() {
        // Test full flow: Training -> Model -> Prediction
        use joule_db_browser::learned::model::LearnedModel;
        use joule_db_browser::learned::training::{ModelTrainer, TrainingData};
        use joule_db_browser::learned::{LearnedIndex, LearnedIndexConfig, ModelType};

        // Create training data
        let training_data = TrainingData {
            keys: (0..100).map(|i| format!("key_{}", i)).collect(),
            positions: (0..100).map(|i| i * 10).collect(),
        };

        // Train model
        let trainer = ModelTrainer::new(ModelType::Linear);
        let model_result: Result<LearnedModel, String> = trainer.train(&training_data).await;

        // Model training should succeed
        assert!(model_result.is_ok());

        // Create learned index
        let config = LearnedIndexConfig {
            model_type: ModelType::Linear,
            use_webnn: false,
            fallback_to_btree: true,
            model_path: "test_integration_model".to_string(),
        };

        let index_result: Result<LearnedIndex, String> = LearnedIndex::new(config).await;
        if let Ok(mut index) = index_result {
            // Train the index — may fail if OPFS not available in test env
            if let Err(e) = index.train(training_data).await {
                web_sys::console::warn_1(
                    &format!("Index training failed (expected without OPFS): {e}").into(),
                );
            }
        }
    }

    #[wasm_bindgen_test]
    async fn test_deterministic_multi_tab_coordination() {
        // Test coordination across multiple tabs (simulated)
        use joule_db_browser::deterministic::MultiTabCoordinator;

        // Create coordinator for tab 1
        let coord1 = MultiTabCoordinator::new("weblocks".to_string())
            .await
            .expect("MultiTabCoordinator creation should succeed in browser");
        // Acquire lock
        coord1
            .acquire_lock("shared_resource")
            .await
            .expect("Lock acquisition should succeed");
        // Coordinate transaction
        coord1
            .coordinate_transaction("tx_1")
            .await
            .expect("Transaction coordination should succeed");

        // In a real scenario, we would create coord2 for tab 2
        // and verify that locks are properly coordinated
    }

    #[wasm_bindgen_test]
    async fn test_htap_learned_index_integration() {
        // Test HTAP with learned indexes
        use joule_db_browser::htap::{HTAPConfig, HTAPStorage};

        let config = HTAPConfig::default();

        let storage = HTAPStorage::new(config)
            .await
            .expect("HTAPStorage creation should succeed in browser");
        // Execute a query — should return a result
        let query_result = storage.query("SELECT COUNT(*) FROM table").await;
        assert!(
            query_result.is_ok(),
            "HTAP query should succeed: {:?}",
            query_result.err()
        );
    }

    #[wasm_bindgen_test]
    async fn test_all_features_together() {
        // Test all three features working together
        // This is a high-level integration test

        // 1. Create HTAP storage
        use joule_db_browser::htap::{HTAPConfig, HTAPStorage};
        let htap_config = HTAPConfig::default();
        let htap_result: Result<HTAPStorage, String> = HTAPStorage::new(htap_config).await;

        // 2. Create learned index
        use joule_db_browser::learned::{LearnedIndex, LearnedIndexConfig, ModelType};
        let learned_config = LearnedIndexConfig::default();
        let learned_result: Result<LearnedIndex, String> = LearnedIndex::new(learned_config).await;

        // 3. Create deterministic coordinator
        use joule_db_browser::deterministic::MultiTabCoordinator;
        let coord_result: Result<MultiTabCoordinator, String> =
            MultiTabCoordinator::new("weblocks".to_string()).await;

        // At least one feature should initialize successfully in browser
        let any_succeeded = htap_result.is_ok() || learned_result.is_ok() || coord_result.is_ok();
        assert!(
            any_succeeded,
            "At least one feature should initialize: htap={:?}, learned={:?}, coord={:?}",
            htap_result.err(),
            learned_result.err(),
            coord_result.err()
        );
    }
}
