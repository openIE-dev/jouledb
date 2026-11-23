//! Unit tests for Learned Indexes

#![cfg(feature = "learned")]

#[cfg(test)]
mod tests {
    use joule_db_browser::learned::{LearnedIndex, LearnedIndexConfig, ModelType};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_learned_index_creation() {
        let config = LearnedIndexConfig {
            model_type: ModelType::Linear,
            use_webnn: false,
            fallback_to_btree: true,
            model_path: "test_model".to_string(),
        };

        let index = LearnedIndex::new(config).await;
        assert!(
            index.is_ok(),
            "LearnedIndex creation should succeed: {:?}",
            index.err()
        );
    }

    #[wasm_bindgen_test]
    async fn test_model_training_data_structure() {
        use joule_db_browser::learned::training::TrainingData;

        let training_data = TrainingData {
            keys: vec!["key1".to_string(), "key2".to_string(), "key3".to_string()],
            positions: vec![0, 100, 200],
        };

        assert_eq!(training_data.keys.len(), 3);
        assert_eq!(training_data.positions.len(), 3);
    }

    #[wasm_bindgen_test]
    async fn test_model_trainer_creation() {
        use joule_db_browser::learned::training::ModelTrainer;

        let trainer_linear = ModelTrainer::new(ModelType::Linear);
        assert!(matches!(trainer_linear.model_type, ModelType::Linear));

        let trainer_neural = ModelTrainer::new(ModelType::Neural);
        assert!(matches!(trainer_neural.model_type, ModelType::Neural));
    }

    #[wasm_bindgen_test]
    async fn test_learned_index_config_default() {
        let config = LearnedIndexConfig::default();

        assert_eq!(config.model_path, "learned_index_model");
        assert!(config.fallback_to_btree);
    }

    #[wasm_bindgen_test]
    async fn test_model_type_equality() {
        assert_eq!(ModelType::Linear, ModelType::Linear);
        assert_ne!(ModelType::Linear, ModelType::Neural);
    }
}
