//! Unit tests for HTAP Systems
//!
//! These tests require a WASM target (wasm32) since they use JsValue and wasm_bindgen_test.

#[cfg(target_arch = "wasm32")]
#[cfg(test)]
mod tests {
    use joule_db_browser::htap::{
        ColumnDef, ColumnarSchema, ColumnarStore, ConversionStrategy, FormatConverter, HTAPConfig,
        HTAPStorage,
    };
    use wasm_bindgen::JsValue;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_format_converter_cpu() {
        let converter = FormatConverter::new(ConversionStrategy::Cpu);

        // Create sample B-tree data (simplified format)
        let mut btree_data = Vec::new();

        // Row 1: key="user1", value={"name":"Alice","age":30}
        let key1 = b"user1";
        let value1 = br#"{"name":"Alice","age":30}"#;
        btree_data.extend_from_slice(&(key1.len() as u32).to_le_bytes());
        btree_data.extend_from_slice(key1);
        btree_data.extend_from_slice(&(value1.len() as u32).to_le_bytes());
        btree_data.extend_from_slice(value1);

        // Row 2: key="user2", value={"name":"Bob","age":25}
        let key2 = b"user2";
        let value2 = br#"{"name":"Bob","age":25}"#;
        btree_data.extend_from_slice(&(key2.len() as u32).to_le_bytes());
        btree_data.extend_from_slice(key2);
        btree_data.extend_from_slice(&(value2.len() as u32).to_le_bytes());
        btree_data.extend_from_slice(value2);

        // Create schema
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
        let result = converter.btree_to_columnar(&btree_data, &schema).await;
        assert!(result.is_ok());
        let columnar_data = result.unwrap();

        // Verify header
        assert!(columnar_data.len() >= 8); // At least header
        let num_rows = u32::from_le_bytes([
            columnar_data[0],
            columnar_data[1],
            columnar_data[2],
            columnar_data[3],
        ]);
        let num_columns = u32::from_le_bytes([
            columnar_data[4],
            columnar_data[5],
            columnar_data[6],
            columnar_data[7],
        ]);

        assert_eq!(num_rows, 2);
        assert_eq!(num_columns, 2);
    }

    #[wasm_bindgen_test]
    async fn test_columnar_store_add_row() {
        let mut store = ColumnarStore::new(
            vec!["name".to_string(), "age".to_string()],
            vec!["string".to_string(), "int32".to_string()],
        );

        // Add first row
        let row1 = vec![
            serde_json::Value::String("Alice".to_string()),
            serde_json::json!(30),
        ];
        assert!(store.add_row(row1).is_ok());

        // Add second row
        let row2 = vec![
            serde_json::Value::String("Bob".to_string()),
            serde_json::json!(25),
        ];
        assert!(store.add_row(row2).is_ok());
    }

    #[wasm_bindgen_test]
    async fn test_columnar_store_query_count() {
        let mut store = ColumnarStore::new(
            vec!["name".to_string(), "age".to_string()],
            vec!["string".to_string(), "float64".to_string()],
        );

        // Add rows
        for i in 0..10 {
            let row = vec![
                serde_json::Value::String(format!("user{}", i)),
                serde_json::json!(i as f64 * 10.0),
            ];
            store.add_row(row).unwrap();
        }

        // Query COUNT(*)
        let result: Result<String, String> = store.query("SELECT COUNT(*) FROM table").await;
        assert!(result.is_ok());
        let count_str = result.unwrap();
        let count: f64 = count_str.parse().unwrap();
        assert_eq!(count, 10.0);
    }

    #[wasm_bindgen_test]
    async fn test_columnar_store_query_sum() {
        let mut store = ColumnarStore::new(
            vec!["name".to_string(), "age".to_string()],
            vec!["string".to_string(), "float64".to_string()],
        );

        // Add rows with ages
        for i in 0..5 {
            let row = vec![
                serde_json::Value::String(format!("user{}", i)),
                serde_json::json!(i as f64 * 10.0),
            ];
            store.add_row(row).unwrap();
        }

        // Query SUM(age)
        let result: Result<String, String> = store.query("SELECT SUM(age) FROM table").await;
        assert!(result.is_ok());
        let sum_str = result.unwrap();
        let sum: f64 = sum_str.parse().unwrap();
        // Sum of 0, 10, 20, 30, 40 = 100
        assert_eq!(sum, 100.0);
    }

    #[wasm_bindgen_test]
    async fn test_htap_storage_creation() {
        let config = HTAPConfig::default();

        let storage = HTAPStorage::new(config).await;
        assert!(
            storage.is_ok(),
            "HTAPStorage creation should succeed: {:?}",
            storage.err()
        );
    }

    #[wasm_bindgen_test]
    async fn test_conversion_strategies() {
        // Test zero-copy strategy
        let converter_zc = FormatConverter::new(ConversionStrategy::ZeroCopy);
        assert!(matches!(
            converter_zc.strategy(),
            ConversionStrategy::ZeroCopy
        ));

        // Test GPU strategy
        let converter_gpu = FormatConverter::new(ConversionStrategy::GpuAccelerated);
        assert!(matches!(
            converter_gpu.strategy(),
            ConversionStrategy::GpuAccelerated
        ));

        // Test CPU strategy
        let converter_cpu = FormatConverter::new(ConversionStrategy::Cpu);
        assert!(matches!(converter_cpu.strategy(), ConversionStrategy::Cpu));
    }
}
