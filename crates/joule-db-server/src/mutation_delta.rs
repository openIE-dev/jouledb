//! HRP Phase 2: Mutation Delta — row-level change capture for efficient replication.
//!
//! Instead of replicating raw SQL strings (which followers must parse + plan + execute),
//! the leader captures a structured `MutationDelta` from the SQL AST and proposes it
//! through Raft. Followers apply the delta directly to `AmorphicTableStorage`, bypassing
//! the SQL parser and query planner entirely.
//!
//! This eliminates the follower-side parse + plan overhead, reducing replication lag.

use serde::{Deserialize, Serialize};

/// A bincode-compatible value type for row data in mutation deltas.
/// Unlike `serde_json::Value`, this supports bincode's serialization
/// (which doesn't support `deserialize_any`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeltaValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Blob(Vec<u8>),
    Array(Vec<DeltaValue>),
}

impl DeltaValue {
    /// Convert to serde_json::Value for storage layer interop.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            DeltaValue::Null => serde_json::Value::Null,
            DeltaValue::Bool(b) => serde_json::json!(*b),
            DeltaValue::Int(i) => serde_json::json!(*i),
            DeltaValue::Float(f) => serde_json::json!(*f),
            DeltaValue::Text(s) => serde_json::json!(s),
            DeltaValue::Blob(b) => serde_json::json!(base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                b
            )),
            DeltaValue::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(|v| v.to_json()).collect())
            }
        }
    }

    /// Convert from serde_json::Value.
    pub fn from_json(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::Null => DeltaValue::Null,
            serde_json::Value::Bool(b) => DeltaValue::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    DeltaValue::Int(i)
                } else if let Some(f) = n.as_f64() {
                    DeltaValue::Float(f)
                } else {
                    DeltaValue::Text(n.to_string())
                }
            }
            serde_json::Value::String(s) => DeltaValue::Text(s.clone()),
            serde_json::Value::Array(arr) => {
                DeltaValue::Array(arr.iter().map(DeltaValue::from_json).collect())
            }
            serde_json::Value::Object(map) => {
                // Encode objects as JSON text for simplicity
                DeltaValue::Text(serde_json::Value::Object(map.clone()).to_string())
            }
        }
    }
}

/// A mutation delta captures the row-level changes for a single SQL write statement.
///
/// Serialized with bincode and proposed through Raft as `Command::MutationDelta`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationDelta {
    /// INSERT INTO table (columns...) VALUES (values...)
    InsertRows {
        table: String,
        columns: Vec<String>,
        /// Each inner Vec is one row of bincode-compatible values.
        rows: Vec<Vec<DeltaValue>>,
    },
    /// CREATE TABLE name (column_defs...)
    CreateTable {
        name: String,
        columns: Vec<String>,
        column_defs: Vec<ColumnDef>,
        if_not_exists: bool,
    },
    /// DROP TABLE name
    DropTable { name: String, if_exists: bool },
    /// Fallback: raw SQL for operations not yet captured as deltas
    /// (UPDATE with WHERE, DELETE with WHERE, ALTER TABLE, etc.)
    RawSql { sql: String },
}

/// Serializable column definition (mirrors SqlColumnDef but is serde-friendly).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub auto_increment: bool,
}

impl MutationDelta {
    /// Serialize the delta to bytes using bincode.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| format!("MutationDelta serialize: {}", e))
    }

    /// Deserialize the delta from bytes using bincode.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        bincode::serde::decode_from_slice(data, bincode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| format!("MutationDelta deserialize: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_rows_roundtrip() {
        let delta = MutationDelta::InsertRows {
            table: "users".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![
                vec![DeltaValue::Int(1), DeltaValue::Text("Alice".to_string())],
                vec![DeltaValue::Int(2), DeltaValue::Text("Bob".to_string())],
            ],
        };
        let bytes = delta.to_bytes().unwrap();
        let recovered = MutationDelta::from_bytes(&bytes).unwrap();
        match recovered {
            MutationDelta::InsertRows {
                table,
                columns,
                rows,
            } => {
                assert_eq!(table, "users");
                assert_eq!(columns, vec!["id", "name"]);
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][0], DeltaValue::Int(1));
                assert_eq!(rows[0][1], DeltaValue::Text("Alice".to_string()));
                assert_eq!(rows[1][0], DeltaValue::Int(2));
            }
            _ => panic!("Expected InsertRows"),
        }
    }

    #[test]
    fn test_create_table_roundtrip() {
        let delta = MutationDelta::CreateTable {
            name: "products".to_string(),
            columns: vec!["id".to_string(), "price".to_string()],
            column_defs: vec![
                ColumnDef {
                    name: "id".to_string(),
                    data_type: "INT".to_string(),
                    nullable: false,
                    primary_key: true,
                    unique: false,
                    auto_increment: false,
                },
                ColumnDef {
                    name: "price".to_string(),
                    data_type: "FLOAT".to_string(),
                    nullable: true,
                    primary_key: false,
                    unique: false,
                    auto_increment: false,
                },
            ],
            if_not_exists: false,
        };
        let bytes = delta.to_bytes().unwrap();
        let recovered = MutationDelta::from_bytes(&bytes).unwrap();
        match recovered {
            MutationDelta::CreateTable {
                name,
                columns,
                column_defs,
                if_not_exists,
            } => {
                assert_eq!(name, "products");
                assert_eq!(columns.len(), 2);
                assert_eq!(column_defs[0].data_type, "INT");
                assert!(!if_not_exists);
            }
            _ => panic!("Expected CreateTable"),
        }
    }

    #[test]
    fn test_drop_table_roundtrip() {
        let delta = MutationDelta::DropTable {
            name: "old_table".to_string(),
            if_exists: true,
        };
        let bytes = delta.to_bytes().unwrap();
        let recovered = MutationDelta::from_bytes(&bytes).unwrap();
        match recovered {
            MutationDelta::DropTable { name, if_exists } => {
                assert_eq!(name, "old_table");
                assert!(if_exists);
            }
            _ => panic!("Expected DropTable"),
        }
    }

    #[test]
    fn test_raw_sql_roundtrip() {
        let delta = MutationDelta::RawSql {
            sql: "UPDATE users SET name = 'Charlie' WHERE id = 3".to_string(),
        };
        let bytes = delta.to_bytes().unwrap();
        let recovered = MutationDelta::from_bytes(&bytes).unwrap();
        match recovered {
            MutationDelta::RawSql { sql } => {
                assert!(sql.starts_with("UPDATE"));
            }
            _ => panic!("Expected RawSql"),
        }
    }

    #[test]
    fn test_delta_value_json_roundtrip() {
        let values = vec![
            DeltaValue::Null,
            DeltaValue::Bool(true),
            DeltaValue::Int(42),
            DeltaValue::Float(3.14),
            DeltaValue::Text("hello".to_string()),
            DeltaValue::Array(vec![DeltaValue::Int(1), DeltaValue::Int(2)]),
        ];
        for v in &values {
            let json = v.to_json();
            let recovered = DeltaValue::from_json(&json);
            assert_eq!(*v, recovered);
        }
    }
}
