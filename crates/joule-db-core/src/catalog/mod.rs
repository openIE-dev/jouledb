//! Schema Catalog for JouleDB
//!
//! Provides persistent storage of table schemas, column definitions,
//! and index metadata using the B-tree storage engine.
//!
//! ## Key Format
//!
//! - Table schema: `__catalog__::table::{name}`
//! - Index definition: `__catalog__::index::{table}::{name}`
//! - Sequence: `__catalog__::seq::{name}`

use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::types::spatial::Spatial3dKind;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::collections::HashMap;
use std::sync::Arc;

/// Prefix for catalog keys
const CATALOG_PREFIX: &[u8] = b"__catalog__::";
const TABLE_PREFIX: &[u8] = b"__catalog__::table::";
const INDEX_PREFIX: &[u8] = b"__catalog__::index::";
const SEQ_PREFIX: &[u8] = b"__catalog__::seq::";
/// Key for the global monotonic object-ID counter
const NEXT_ID_KEY: &[u8] = b"__catalog__::next_id";
/// Prefix for ID-to-name mapping
const ID_PREFIX: &[u8] = b"__catalog__::id::";

/// SQL data types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// Boolean
    Boolean,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// Variable-length string
    String,
    /// Variable-length string with max length
    Varchar(usize),
    /// Fixed-length string
    Char(usize),
    /// Variable-length binary
    Binary,
    /// Timestamp with timezone
    Timestamp,
    /// Date only
    Date,
    /// Time only
    Time,
    /// UUID
    Uuid,
    /// JSON document
    Json,
    /// Array of another type
    Array(Box<DataType>),
    /// Fixed-dimension vector of f32 values (for similarity search)
    Vector(usize),
    /// 3D spatial type — Point3, Quat, Pose6, or Bbox3.
    /// First-class so spatial indexes can key on these directly.
    Spatial3d(Spatial3dKind),
}

impl DataType {
    /// Get the SQL type name
    pub fn sql_name(&self) -> String {
        match self {
            DataType::Boolean => "BOOLEAN".to_string(),
            DataType::Int8 => "TINYINT".to_string(),
            DataType::Int16 => "SMALLINT".to_string(),
            DataType::Int32 => "INTEGER".to_string(),
            DataType::Int64 => "BIGINT".to_string(),
            DataType::Float32 => "REAL".to_string(),
            DataType::Float64 => "DOUBLE".to_string(),
            DataType::String => "TEXT".to_string(),
            DataType::Varchar(n) => format!("VARCHAR({})", n),
            DataType::Char(n) => format!("CHAR({})", n),
            DataType::Binary => "BLOB".to_string(),
            DataType::Timestamp => "TIMESTAMP".to_string(),
            DataType::Date => "DATE".to_string(),
            DataType::Time => "TIME".to_string(),
            DataType::Uuid => "UUID".to_string(),
            DataType::Json => "JSON".to_string(),
            DataType::Array(inner) => format!("{}[]", inner.sql_name()),
            DataType::Vector(n) => format!("VECTOR({})", n),
            DataType::Spatial3d(k) => k.sql_name().to_string(),
        }
    }

    /// Parse from SQL type name
    pub fn from_sql(s: &str) -> Option<Self> {
        let s = s.to_uppercase();
        match s.as_str() {
            "BOOLEAN" | "BOOL" => Some(DataType::Boolean),
            "TINYINT" | "INT1" => Some(DataType::Int8),
            "SMALLINT" | "INT2" => Some(DataType::Int16),
            "INTEGER" | "INT" | "INT4" => Some(DataType::Int32),
            "BIGINT" | "INT8" => Some(DataType::Int64),
            "REAL" | "FLOAT4" => Some(DataType::Float32),
            "DOUBLE" | "FLOAT8" | "DOUBLE PRECISION" => Some(DataType::Float64),
            "TEXT" => Some(DataType::String),
            "BLOB" | "BYTEA" => Some(DataType::Binary),
            "TIMESTAMP" | "TIMESTAMPTZ" => Some(DataType::Timestamp),
            "DATE" => Some(DataType::Date),
            "TIME" => Some(DataType::Time),
            "UUID" => Some(DataType::Uuid),
            "JSON" | "JSONB" => Some(DataType::Json),
            _ => {
                // Try VARCHAR(n) or CHAR(n)
                if s.starts_with("VARCHAR(") && s.ends_with(')') {
                    let n: usize = s[8..s.len() - 1].parse().ok()?;
                    Some(DataType::Varchar(n))
                } else if s.starts_with("CHAR(") && s.ends_with(')') {
                    let n: usize = s[5..s.len() - 1].parse().ok()?;
                    Some(DataType::Char(n))
                } else if s.starts_with("VECTOR(") && s.ends_with(')') {
                    let n: usize = s[7..s.len() - 1].parse().ok()?;
                    Some(DataType::Vector(n))
                } else if let Some(k) = Spatial3dKind::from_sql(&s) {
                    Some(DataType::Spatial3d(k))
                } else {
                    None
                }
            }
        }
    }
}

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Whether NULL is allowed
    pub nullable: bool,
    /// Default value (serialized)
    pub default: Option<Vec<u8>>,
    /// Is this column part of the primary key
    pub primary_key: bool,
    /// Is this column unique
    pub unique: bool,
    /// Column position (0-indexed)
    pub position: usize,
}

impl ColumnDef {
    /// Create a new column definition
    pub fn new(name: &str, data_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            nullable: true,
            default: None,
            primary_key: false,
            unique: false,
            position: 0,
        }
    }

    /// Set nullable
    pub fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Set as primary key
    pub fn primary_key(mut self) -> Self {
        self.primary_key = true;
        self.nullable = false; // PK is never null
        self
    }

    /// Set as unique
    pub fn unique(mut self) -> Self {
        self.unique = true;
        self
    }

    /// Set default value
    pub fn with_default(mut self, default: Vec<u8>) -> Self {
        self.default = Some(default);
        self
    }
}

/// Index type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// B-tree index (default)
    BTree,
    /// Hash index
    Hash,
    /// Full-text search index
    FullText,
    /// Vector similarity index
    Vector,
    /// Spatial/GiST index
    Spatial,
    /// Adaptive Radix Tree (cache-friendly ordered index)
    ART,
    /// MinHash-LSH (near-duplicate detection, Jaccard similarity)
    MinHashLSH,
}

/// Index definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDef {
    /// Stable numeric identifier (assigned on creation)
    #[serde(default)]
    pub id: u32,
    /// Index name
    pub name: String,
    /// Table name
    pub table: String,
    /// Indexed columns
    pub columns: Vec<String>,
    /// Whether this is a unique index
    pub unique: bool,
    /// Index type
    pub index_type: IndexType,
    /// Partial index predicate (SQL WHERE clause)
    pub predicate: Option<String>,
    /// Index method (e.g., "HNSW", "IVF", "LSH", "GIN", "GIST")
    #[serde(default)]
    pub method: Option<String>,
    /// Distance/similarity metric (e.g., "euclidean", "cosine", "hamming")
    #[serde(default)]
    pub metric: Option<String>,
    /// Arbitrary index options (e.g., m=16, ef_construction=200)
    #[serde(default)]
    pub options: std::collections::HashMap<String, String>,
}

impl IndexDef {
    /// Create a new index definition
    pub fn new(name: &str, table: &str, columns: Vec<String>) -> Self {
        Self {
            id: 0, // assigned by Catalog::create_index
            name: name.to_string(),
            table: table.to_string(),
            columns,
            unique: false,
            index_type: IndexType::BTree,
            predicate: None,
            method: None,
            metric: None,
            options: std::collections::HashMap::new(),
        }
    }

    /// Set as unique index
    pub fn unique(mut self) -> Self {
        self.unique = true;
        self
    }

    /// Set index type
    pub fn index_type(mut self, index_type: IndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Set partial index predicate
    pub fn with_predicate(mut self, predicate: &str) -> Self {
        self.predicate = Some(predicate.to_string());
        self
    }
}

/// Table schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    /// Stable numeric identifier (assigned on creation)
    #[serde(default)]
    pub id: u32,
    /// Table name
    pub name: String,
    /// Column definitions (ordered)
    pub columns: Vec<ColumnDef>,
    /// Primary key column names
    pub primary_key: Vec<String>,
    /// Indexes on this table
    pub indexes: Vec<IndexDef>,
    /// Table options/metadata
    pub options: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: i64,
    /// Last modified timestamp  
    pub updated_at: i64,
}

impl TableSchema {
    /// Create a new table schema
    pub fn new(name: &str) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Self {
            id: 0, // assigned by Catalog::create_table
            name: name.to_string(),
            columns: Vec::new(),
            primary_key: Vec::new(),
            indexes: Vec::new(),
            options: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a column
    pub fn add_column(mut self, mut column: ColumnDef) -> Self {
        column.position = self.columns.len();
        if column.primary_key {
            self.primary_key.push(column.name.clone());
        }
        self.columns.push(column);
        self
    }

    /// Add an index
    pub fn add_index(mut self, index: IndexDef) -> Self {
        self.indexes.push(index);
        self
    }

    /// Set an option
    pub fn with_option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Get column by name
    pub fn get_column(&self, name: &str) -> Option<&ColumnDef> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Get column index by name
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Get all column names
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| {
            Error::Storage(crate::error::StorageError::Backend(format!(
                "Failed to serialize schema: {}",
                e
            )))
        })
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| {
            Error::Storage(crate::error::StorageError::Backend(format!(
                "Failed to deserialize schema: {}",
                e
            )))
        })
    }
}

/// Schema catalog - persistent storage of table definitions
pub struct Catalog {
    engine: Arc<Engine>,
}

impl Catalog {
    /// Create a new catalog backed by the given engine
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine }
    }

    /// Allocate the next monotonic object ID.
    fn next_object_id(&self) -> Result<u32> {
        let current = match self.engine.get(NEXT_ID_KEY)? {
            Some(data) if data.len() == 4 => {
                u32::from_le_bytes(data.try_into().expect("exact 4-byte vec"))
            }
            _ => 0,
        };
        let next = current + 1;
        self.engine.put(NEXT_ID_KEY, &next.to_le_bytes())?;
        Ok(next)
    }

    /// Store an ID→name mapping.
    fn store_id_mapping(&self, id: u32, name: &str) -> Result<()> {
        let mut key = ID_PREFIX.to_vec();
        key.extend_from_slice(&id.to_le_bytes());
        self.engine.put(&key, name.as_bytes())?;
        Ok(())
    }

    /// Resolve an ID to a name.
    pub fn resolve_id(&self, id: u32) -> Result<Option<String>> {
        let mut key = ID_PREFIX.to_vec();
        key.extend_from_slice(&id.to_le_bytes());
        match self.engine.get(&key)? {
            Some(data) => Ok(Some(String::from_utf8_lossy(&data).to_string())),
            None => Ok(None),
        }
    }

    /// Get a table's numeric ID by name.
    pub fn get_table_id(&self, name: &str) -> Result<Option<u32>> {
        self.get_table(name).map(|opt| opt.map(|s| s.id))
    }

    /// Get a table schema by its numeric ID.
    pub fn get_table_by_id(&self, id: u32) -> Result<Option<TableSchema>> {
        match self.resolve_id(id)? {
            Some(name) => self.get_table(&name),
            None => Ok(None),
        }
    }

    /// Create a table
    pub fn create_table(&self, mut schema: TableSchema) -> Result<()> {
        let key = self.table_key(&schema.name);

        // Check if table already exists
        if self.engine.get(&key)?.is_some() {
            return Err(Error::Storage(crate::error::StorageError::Backend(
                format!("Table '{}' already exists", schema.name),
            )));
        }

        // Assign stable ID if not already set
        if schema.id == 0 {
            schema.id = self.next_object_id()?;
        }

        let value = schema.serialize()?;
        self.engine.put(&key, &value)?;

        // Dual-write: store ID→name mapping
        self.store_id_mapping(schema.id, &schema.name)?;

        // Create indexes (assign IDs)
        for index in &mut schema.indexes {
            if index.id == 0 {
                index.id = self.next_object_id()?;
                self.store_id_mapping(index.id, &index.name)?;
            }
            self.create_index_internal(index)?;
        }

        Ok(())
    }

    /// Create a table if it doesn't exist
    pub fn create_table_if_not_exists(&self, schema: TableSchema) -> Result<bool> {
        let key = self.table_key(&schema.name);

        if self.engine.get(&key)?.is_some() {
            return Ok(false);
        }

        self.create_table(schema)?;
        Ok(true)
    }

    /// Get a table schema
    pub fn get_table(&self, name: &str) -> Result<Option<TableSchema>> {
        let key = self.table_key(name);
        match self.engine.get(&key)? {
            Some(data) => Ok(Some(TableSchema::deserialize(&data)?)),
            None => Ok(None),
        }
    }

    /// Drop a table
    pub fn drop_table(&self, name: &str) -> Result<bool> {
        let key = self.table_key(name);

        // Get schema to find indexes
        if let Some(schema) = self.get_table(name)? {
            // Drop all indexes
            for index in &schema.indexes {
                self.drop_index_internal(&index.name)?;
            }
        }

        self.engine.delete(&key)
    }

    /// Drop a table if it exists
    pub fn drop_table_if_exists(&self, name: &str) -> Result<bool> {
        if self.get_table(name)?.is_none() {
            return Ok(false);
        }
        self.drop_table(name)
    }

    /// List all tables
    pub fn list_tables(&self) -> Result<Vec<String>> {
        let prefix = TABLE_PREFIX;
        let mut tables = Vec::new();

        let mut iter = self.engine.prefix_scan(prefix)?;
        while let Some(result) = iter.next() {
            let entry = result.map_err(|e| Error::Index(e))?;
            let key = entry.key;
            // Extract table name from key
            if key.starts_with(prefix) {
                let name = String::from_utf8_lossy(&key[prefix.len()..]).to_string();
                tables.push(name);
            }
        }

        Ok(tables)
    }

    /// Check if a table exists
    pub fn table_exists(&self, name: &str) -> Result<bool> {
        let key = self.table_key(name);
        Ok(self.engine.get(&key)?.is_some())
    }

    /// Alter table - add column
    pub fn add_column(&self, table: &str, column: ColumnDef) -> Result<()> {
        let mut schema = self.get_table(table)?.ok_or_else(|| {
            Error::Storage(crate::error::StorageError::Backend(format!(
                "Table '{}' not found",
                table
            )))
        })?;

        // Check column doesn't already exist
        if schema.get_column(&column.name).is_some() {
            return Err(Error::Storage(crate::error::StorageError::Backend(
                format!(
                    "Column '{}' already exists in table '{}'",
                    column.name, table
                ),
            )));
        }

        let mut col = column;
        col.position = schema.columns.len();
        schema.columns.push(col);
        schema.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let key = self.table_key(table);
        let value = schema.serialize()?;
        self.engine.put(&key, &value)?;

        Ok(())
    }

    /// Alter table - drop column
    pub fn drop_column(&self, table: &str, column: &str) -> Result<()> {
        let mut schema = self.get_table(table)?.ok_or_else(|| {
            Error::Storage(crate::error::StorageError::Backend(format!(
                "Table '{}' not found",
                table
            )))
        })?;

        // Check column exists
        let pos = schema.column_index(column).ok_or_else(|| {
            Error::Storage(crate::error::StorageError::Backend(format!(
                "Column '{}' not found in table '{}'",
                column, table
            )))
        })?;

        // Don't allow dropping primary key columns
        if schema.primary_key.contains(&column.to_string()) {
            return Err(Error::Storage(crate::error::StorageError::Backend(
                format!("Cannot drop primary key column '{}'", column),
            )));
        }

        schema.columns.remove(pos);

        // Update positions
        for (i, col) in schema.columns.iter_mut().enumerate() {
            col.position = i;
        }

        schema.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let key = self.table_key(table);
        let value = schema.serialize()?;
        self.engine.put(&key, &value)?;

        Ok(())
    }

    /// Create an index
    pub fn create_index(&self, mut index: IndexDef) -> Result<()> {
        // Verify table exists
        let schema = self.get_table(&index.table)?.ok_or_else(|| {
            Error::Storage(crate::error::StorageError::Backend(format!(
                "Table '{}' not found",
                index.table
            )))
        })?;

        // Verify columns exist
        for col in &index.columns {
            if schema.get_column(col).is_none() {
                return Err(Error::Storage(crate::error::StorageError::Backend(
                    format!("Column '{}' not found in table '{}'", col, index.table),
                )));
            }
        }

        // Assign stable ID if not already set
        if index.id == 0 {
            index.id = self.next_object_id()?;
            self.store_id_mapping(index.id, &index.name)?;
        }

        self.create_index_internal(&index)?;

        // Add to table schema
        let mut schema = schema;
        schema.indexes.push(index);
        schema.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let key = self.table_key(&schema.name);
        let value = schema.serialize()?;
        self.engine.put(&key, &value)?;

        Ok(())
    }

    /// Drop an index
    pub fn drop_index(&self, name: &str) -> Result<bool> {
        // Find which table has this index
        for table_name in self.list_tables()? {
            if let Some(mut schema) = self.get_table(&table_name)? {
                if let Some(pos) = schema.indexes.iter().position(|i| i.name == name) {
                    schema.indexes.remove(pos);
                    schema.updated_at = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64;

                    let key = self.table_key(&table_name);
                    let value = schema.serialize()?;
                    self.engine.put(&key, &value)?;

                    return self.drop_index_internal(name);
                }
            }
        }

        Ok(false)
    }

    /// Get an index definition
    pub fn get_index(&self, name: &str) -> Result<Option<IndexDef>> {
        let key = self.index_key(name);
        match self.engine.get(&key)? {
            Some(data) => {
                let index: IndexDef = serde_json::from_slice(&data).map_err(|e| {
                    Error::Storage(crate::error::StorageError::Backend(format!(
                        "Failed to deserialize index: {}",
                        e
                    )))
                })?;
                Ok(Some(index))
            }
            None => Ok(None),
        }
    }

    /// Get next sequence value
    pub fn next_sequence(&self, name: &str) -> Result<i64> {
        let key = self.seq_key(name);
        let current = match self.engine.get(&key)? {
            Some(data) if data.len() == 8 => {
                i64::from_le_bytes(data.try_into().expect("exact 8-byte vec"))
            }
            _ => 0,
        };

        let next = current + 1;
        self.engine.put(&key, &next.to_le_bytes())?;
        Ok(next)
    }

    /// Reset sequence
    pub fn reset_sequence(&self, name: &str, value: i64) -> Result<()> {
        let key = self.seq_key(name);
        self.engine.put(&key, &value.to_le_bytes())?;
        Ok(())
    }

    // Internal helpers

    fn table_key(&self, name: &str) -> Vec<u8> {
        let mut key = TABLE_PREFIX.to_vec();
        key.extend_from_slice(name.as_bytes());
        key
    }

    fn index_key(&self, name: &str) -> Vec<u8> {
        let mut key = INDEX_PREFIX.to_vec();
        key.extend_from_slice(name.as_bytes());
        key
    }

    fn seq_key(&self, name: &str) -> Vec<u8> {
        let mut key = SEQ_PREFIX.to_vec();
        key.extend_from_slice(name.as_bytes());
        key
    }

    fn create_index_internal(&self, index: &IndexDef) -> Result<()> {
        let key = self.index_key(&index.name);
        let value = serde_json::to_vec(index).map_err(|e| {
            Error::Storage(crate::error::StorageError::Backend(format!(
                "Failed to serialize index: {}",
                e
            )))
        })?;
        self.engine.put(&key, &value)?;
        Ok(())
    }

    fn drop_index_internal(&self, name: &str) -> Result<bool> {
        let key = self.index_key(name);
        self.engine.delete(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryBackend;

    fn create_test_catalog() -> Catalog {
        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());
        Catalog::new(engine)
    }

    #[test]
    fn test_create_and_get_table() {
        let catalog = create_test_catalog();

        let schema = TableSchema::new("users")
            .add_column(ColumnDef::new("id", DataType::Int64).primary_key())
            .add_column(ColumnDef::new("name", DataType::String).nullable(false))
            .add_column(ColumnDef::new("email", DataType::String).unique());

        catalog.create_table(schema.clone()).unwrap();

        let retrieved = catalog.get_table("users").unwrap().unwrap();
        assert_eq!(retrieved.name, "users");
        assert_eq!(retrieved.columns.len(), 3);
        assert_eq!(retrieved.primary_key, vec!["id"]);
    }

    #[test]
    fn test_table_already_exists() {
        let catalog = create_test_catalog();

        let schema = TableSchema::new("users")
            .add_column(ColumnDef::new("id", DataType::Int64).primary_key());

        catalog.create_table(schema.clone()).unwrap();
        assert!(catalog.create_table(schema).is_err());
    }

    #[test]
    fn test_create_if_not_exists() {
        let catalog = create_test_catalog();

        let schema = TableSchema::new("users")
            .add_column(ColumnDef::new("id", DataType::Int64).primary_key());

        assert!(catalog.create_table_if_not_exists(schema.clone()).unwrap());
        assert!(!catalog.create_table_if_not_exists(schema).unwrap());
    }

    #[test]
    fn test_drop_table() {
        let catalog = create_test_catalog();

        let schema = TableSchema::new("users")
            .add_column(ColumnDef::new("id", DataType::Int64).primary_key());

        catalog.create_table(schema).unwrap();
        assert!(catalog.table_exists("users").unwrap());

        assert!(catalog.drop_table("users").unwrap());
        assert!(!catalog.table_exists("users").unwrap());
    }

    #[test]
    fn test_list_tables() {
        let catalog = create_test_catalog();

        catalog
            .create_table(
                TableSchema::new("users")
                    .add_column(ColumnDef::new("id", DataType::Int64).primary_key()),
            )
            .unwrap();
        catalog
            .create_table(
                TableSchema::new("orders")
                    .add_column(ColumnDef::new("id", DataType::Int64).primary_key()),
            )
            .unwrap();
        catalog
            .create_table(
                TableSchema::new("products")
                    .add_column(ColumnDef::new("id", DataType::Int64).primary_key()),
            )
            .unwrap();

        let tables = catalog.list_tables().unwrap();
        assert_eq!(tables.len(), 3);
        assert!(tables.contains(&"users".to_string()));
        assert!(tables.contains(&"orders".to_string()));
        assert!(tables.contains(&"products".to_string()));
    }

    #[test]
    fn test_add_column() {
        let catalog = create_test_catalog();

        let schema = TableSchema::new("users")
            .add_column(ColumnDef::new("id", DataType::Int64).primary_key());

        catalog.create_table(schema).unwrap();

        catalog
            .add_column("users", ColumnDef::new("name", DataType::String))
            .unwrap();

        let retrieved = catalog.get_table("users").unwrap().unwrap();
        assert_eq!(retrieved.columns.len(), 2);
        assert_eq!(retrieved.columns[1].name, "name");
    }

    #[test]
    fn test_drop_column() {
        let catalog = create_test_catalog();

        let schema = TableSchema::new("users")
            .add_column(ColumnDef::new("id", DataType::Int64).primary_key())
            .add_column(ColumnDef::new("name", DataType::String))
            .add_column(ColumnDef::new("email", DataType::String));

        catalog.create_table(schema).unwrap();

        catalog.drop_column("users", "email").unwrap();

        let retrieved = catalog.get_table("users").unwrap().unwrap();
        assert_eq!(retrieved.columns.len(), 2);
        assert!(retrieved.get_column("email").is_none());
    }

    #[test]
    fn test_cannot_drop_pk_column() {
        let catalog = create_test_catalog();

        let schema = TableSchema::new("users")
            .add_column(ColumnDef::new("id", DataType::Int64).primary_key());

        catalog.create_table(schema).unwrap();

        assert!(catalog.drop_column("users", "id").is_err());
    }

    #[test]
    fn test_create_index() {
        let catalog = create_test_catalog();

        let schema = TableSchema::new("users")
            .add_column(ColumnDef::new("id", DataType::Int64).primary_key())
            .add_column(ColumnDef::new("email", DataType::String));

        catalog.create_table(schema).unwrap();

        let index = IndexDef::new("idx_users_email", "users", vec!["email".to_string()]).unique();
        catalog.create_index(index).unwrap();

        let retrieved = catalog.get_index("idx_users_email").unwrap().unwrap();
        assert!(retrieved.unique);
        assert_eq!(retrieved.columns, vec!["email"]);
    }

    #[test]
    fn test_sequence() {
        let catalog = create_test_catalog();

        assert_eq!(catalog.next_sequence("user_id").unwrap(), 1);
        assert_eq!(catalog.next_sequence("user_id").unwrap(), 2);
        assert_eq!(catalog.next_sequence("user_id").unwrap(), 3);

        catalog.reset_sequence("user_id", 100).unwrap();
        assert_eq!(catalog.next_sequence("user_id").unwrap(), 101);
    }

    #[test]
    fn test_data_type_sql_names() {
        assert_eq!(DataType::Int64.sql_name(), "BIGINT");
        assert_eq!(DataType::Varchar(255).sql_name(), "VARCHAR(255)");
        assert_eq!(
            DataType::Array(Box::new(DataType::Int32)).sql_name(),
            "INTEGER[]"
        );
    }

    #[test]
    fn test_data_type_from_sql() {
        assert_eq!(DataType::from_sql("INTEGER"), Some(DataType::Int32));
        assert_eq!(
            DataType::from_sql("VARCHAR(100)"),
            Some(DataType::Varchar(100))
        );
        assert_eq!(DataType::from_sql("BIGINT"), Some(DataType::Int64));
        assert_eq!(DataType::from_sql("UNKNOWN"), None);
    }
}
