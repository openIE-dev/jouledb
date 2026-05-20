//! GIN index manager — wraps joule_db_core's GinIndex for use in the query executor.
//!
//! Provides inverted-index pre-filtering for JSONB `@>` containment,
//! array `&&` overlap, full-text `@@`, and trigram `LIKE/ILIKE` queries.
//! This is a coarse pre-filter; exact predicates are still evaluated
//! on the candidate set for correctness.

use std::collections::{BTreeSet, HashMap};
use std::sync::RwLock;

use joule_db_core::index::gin::{GinConfig, GinIndex, GinStrategy};

/// Metadata about a GIN index.
struct GinIndexInfo {
    index: GinIndex,
    table: String,
    column: String,
    /// Map from amorphic record ID (String) → internal u64 ID used by GinIndex.
    id_to_u64: HashMap<String, u64>,
    /// Reverse map: u64 → amorphic record ID.
    u64_to_id: HashMap<u64, String>,
    /// Next internal ID to assign.
    next_id: u64,
}

impl GinIndexInfo {
    fn alloc_id(&mut self, record_id: &str) -> u64 {
        if let Some(&id) = self.id_to_u64.get(record_id) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.id_to_u64.insert(record_id.to_string(), id);
        self.u64_to_id.insert(id, record_id.to_string());
        id
    }

    fn resolve_ids(&self, internal_ids: &BTreeSet<u64>) -> Vec<String> {
        internal_ids
            .iter()
            .filter_map(|id| self.u64_to_id.get(id).cloned())
            .collect()
    }
}

/// Manages live GIN inverted indexes, analogous to SpatialIndexManager.
pub struct GinIndexManager {
    indexes: RwLock<HashMap<String, GinIndexInfo>>,
}

impl GinIndexManager {
    pub fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
        }
    }

    /// Build a GIN index from existing table data.
    ///
    /// `rows` contains `(record_id, column_value)` pairs extracted from the table.
    pub fn build_index(
        &self,
        name: &str,
        table: &str,
        column: &str,
        strategy: GinStrategy,
        rows: Vec<(String, serde_json::Value)>,
    ) {
        let config = GinConfig {
            strategy,
            ..GinConfig::default()
        };
        let mut gin = GinIndex::new(config);
        let mut id_to_u64 = HashMap::new();
        let mut u64_to_id = HashMap::new();
        let mut next_id: u64 = 0;

        for (record_id, value) in &rows {
            let id = next_id;
            next_id += 1;
            id_to_u64.insert(record_id.clone(), id);
            u64_to_id.insert(id, record_id.clone());
            gin.insert(id, value);
        }

        let info = GinIndexInfo {
            index: gin,
            table: table.to_string(),
            column: column.to_string(),
            id_to_u64,
            u64_to_id,
            next_id,
        };
        crate::lock_util::write_lock(&self.indexes).insert(name.to_string(), info);
    }

    /// Find which GIN index covers a given table + column.
    pub fn find_index_for(&self, table: &str, column: &str) -> Option<String> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        for (name, info) in indexes.iter() {
            if info.table.eq_ignore_ascii_case(table)
                && info.column.eq_ignore_ascii_case(column)
            {
                return Some(name.clone());
            }
        }
        None
    }

    /// List all GIN indexes for a given table, returning `(name, column)` pairs.
    pub fn indexes_for_table(&self, table: &str) -> Vec<(String, String)> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        indexes
            .iter()
            .filter(|(_, info)| info.table.eq_ignore_ascii_case(table))
            .map(|(name, info)| (name.clone(), info.column.clone()))
            .collect()
    }

    /// Search: JSONB `@>` containment.  Returns matching amorphic record IDs.
    pub fn search_contains(
        &self,
        index_name: &str,
        query: &serde_json::Value,
    ) -> Vec<String> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        match indexes.get(index_name) {
            Some(info) => {
                let internal = info.index.search_jsonb_contains(query);
                info.resolve_ids(&internal)
            }
            None => Vec::new(),
        }
    }

    /// Search: array overlap `&&`.  Returns matching amorphic record IDs.
    pub fn search_overlap(
        &self,
        index_name: &str,
        elements: &[serde_json::Value],
    ) -> Vec<String> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        match indexes.get(index_name) {
            Some(info) => {
                let internal = info.index.search_array_overlap(elements);
                info.resolve_ids(&internal)
            }
            None => Vec::new(),
        }
    }

    /// Search: full-text query.  Returns matching amorphic record IDs.
    pub fn search_text(
        &self,
        index_name: &str,
        query: &str,
    ) -> Vec<String> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        match indexes.get(index_name) {
            Some(info) => {
                let internal = info.index.search_text(query);
                info.resolve_ids(&internal)
            }
            None => Vec::new(),
        }
    }

    /// Search: trigram LIKE/ILIKE.  Returns matching amorphic record IDs.
    pub fn search_trigram(
        &self,
        index_name: &str,
        pattern: &str,
    ) -> Vec<String> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        match indexes.get(index_name) {
            Some(info) => {
                let internal = info.index.search_trigram(pattern);
                info.resolve_ids(&internal)
            }
            None => Vec::new(),
        }
    }

    /// Insert a single value into an existing GIN index (called on INSERT).
    pub fn insert_into_index(
        &self,
        index_name: &str,
        record_id: String,
        value: &serde_json::Value,
    ) {
        if let Some(info) = crate::lock_util::write_lock(&self.indexes).get_mut(index_name) {
            let id = info.alloc_id(&record_id);
            info.index.insert(id, value);
        }
    }

    /// Remove a record from a GIN index (called on DELETE).
    pub fn remove_from_index(&self, index_name: &str, record_id: &str) {
        if let Some(info) = crate::lock_util::write_lock(&self.indexes).get_mut(index_name) {
            if let Some(&id) = info.id_to_u64.get(record_id) {
                info.index.remove(id);
                info.id_to_u64.remove(record_id);
                info.u64_to_id.remove(&id);
            }
        }
    }

    /// Drop an index entirely.
    pub fn drop_index(&self, name: &str) -> bool {
        crate::lock_util::write_lock(&self.indexes)
            .remove(name)
            .is_some()
    }

    /// Rebuild GIN indexes from `__indexes__` metadata at startup.
    pub fn rebuild_from_metadata(&self, amorphic: &crate::amorphic_adapter::AmorphicTableStorage) {
        use joule_db_query::ast::Value;
        use joule_db_query::executor::TableStorage;

        let index_records = amorphic.scan("__indexes__").unwrap_or_default();

        let mut to_build: Vec<(String, String, String, GinStrategy)> = Vec::new();
        for row in &index_records {
            let get_str = |col: &str| -> Option<String> {
                let pos = row.columns.iter().position(|c| c == col)?;
                match row.values.get(pos)? {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                }
            };

            let idx_type = get_str("__index_type__");
            if idx_type.as_deref() != Some("gin") {
                continue;
            }

            let name = match get_str("__index_name__") {
                Some(n) => n,
                None => continue,
            };
            let table = match get_str("__index_table_ref__") {
                Some(t) => t,
                None => continue,
            };
            let column_str = get_str("__index_columns__").unwrap_or_default();
            let column = column_str
                .trim_matches(|c: char| c == '[' || c == ']' || c == '"')
                .to_string();

            let strategy_str = get_str("__index_strategy__").unwrap_or_default();
            let strategy = match strategy_str.as_str() {
                "jsonb_path_ops" => GinStrategy::JsonbPathOps,
                "array_ops" => GinStrategy::ArrayOps,
                "text_search" => GinStrategy::TextSearchOps,
                "trigram" => GinStrategy::TrigramOps,
                _ => GinStrategy::JsonbOps,
            };

            if !column.is_empty() {
                to_build.push((name, table, column, strategy));
            }
        }

        // Build each index from table data
        for (name, table, column, strategy) in to_build {
            if let Ok(rows_with_ids) = amorphic.scan_with_record_ids(&table) {
                let gin_rows: Vec<(String, serde_json::Value)> = rows_with_ids
                    .iter()
                    .filter_map(|(record_id, row)| {
                        let col_idx = row.columns.iter().position(|c| c == &column)?;
                        let val = row.values.get(col_idx)?;
                        Some((record_id.clone(), crate::json_ops::ast_value_to_json(val)))
                    })
                    .collect();
                self.build_index(&name, &table, &column, strategy, gin_rows);
            }
        }
    }
}

// ==================== Unit Tests ====================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_gin_manager_build_and_search_jsonb() {
        let manager = GinIndexManager::new();
        let rows = vec![
            ("r1".into(), json!({"name": "alice", "age": 30})),
            ("r2".into(), json!({"name": "bob", "age": 25})),
            ("r3".into(), json!({"name": "alice", "age": 25})),
        ];
        manager.build_index("idx_data", "users", "data", GinStrategy::JsonbPathOps, rows);

        // find_index_for
        assert_eq!(
            manager.find_index_for("users", "data"),
            Some("idx_data".to_string())
        );
        assert!(manager.find_index_for("users", "other").is_none());

        // @> {"name": "alice"}
        let mut results = manager.search_contains("idx_data", &json!({"name": "alice"}));
        results.sort();
        assert_eq!(results, vec!["r1", "r3"]);

        // @> {"age": 25}
        let mut results = manager.search_contains("idx_data", &json!({"age": 25}));
        results.sort();
        assert_eq!(results, vec!["r2", "r3"]);
    }

    #[test]
    fn test_gin_manager_insert_and_remove() {
        let manager = GinIndexManager::new();
        manager.build_index(
            "idx",
            "t",
            "data",
            GinStrategy::JsonbPathOps,
            vec![],
        );

        manager.insert_into_index("idx", "r1".into(), &json!({"color": "red"}));
        manager.insert_into_index("idx", "r2".into(), &json!({"color": "blue"}));

        let results = manager.search_contains("idx", &json!({"color": "red"}));
        assert_eq!(results, vec!["r1"]);

        manager.remove_from_index("idx", "r1");
        let results = manager.search_contains("idx", &json!({"color": "red"}));
        assert!(results.is_empty());
    }

    #[test]
    fn test_gin_manager_drop() {
        let manager = GinIndexManager::new();
        manager.build_index("idx", "t", "c", GinStrategy::JsonbOps, vec![]);
        assert!(manager.drop_index("idx"));
        assert!(!manager.drop_index("idx"));
        assert!(manager.find_index_for("t", "c").is_none());
    }

    #[test]
    fn test_gin_manager_indexes_for_table() {
        let manager = GinIndexManager::new();
        manager.build_index("idx1", "t", "a", GinStrategy::JsonbOps, vec![]);
        manager.build_index("idx2", "t", "b", GinStrategy::ArrayOps, vec![]);
        manager.build_index("idx3", "other", "c", GinStrategy::TextSearchOps, vec![]);

        let mut idxs = manager.indexes_for_table("t");
        idxs.sort();
        assert_eq!(idxs.len(), 2);
    }

    #[test]
    fn test_gin_manager_array_overlap() {
        let manager = GinIndexManager::new();
        let rows = vec![
            ("r1".into(), json!([1, 2, 3])),
            ("r2".into(), json!([3, 4, 5])),
            ("r3".into(), json!([5, 6, 7])),
        ];
        manager.build_index("idx", "t", "arr", GinStrategy::ArrayOps, rows);

        let mut results = manager.search_overlap("idx", &[json!(3)]);
        results.sort();
        assert_eq!(results, vec!["r1", "r2"]);
    }

    #[test]
    fn test_gin_manager_text_search() {
        let manager = GinIndexManager::new();
        let rows = vec![
            ("r1".into(), json!("the quick brown fox")),
            ("r2".into(), json!("a fast brown car")),
            ("r3".into(), json!("quick brown foxes")),
        ];
        manager.build_index("idx", "t", "text", GinStrategy::TextSearchOps, rows);

        let mut results = manager.search_text("idx", "quick brown");
        results.sort();
        assert_eq!(results, vec!["r1", "r3"]);
    }

    #[test]
    fn test_gin_manager_case_insensitive_find() {
        let manager = GinIndexManager::new();
        manager.build_index("idx", "MyTable", "myCol", GinStrategy::JsonbOps, vec![]);
        assert!(manager.find_index_for("mytable", "mycol").is_some());
        assert!(manager.find_index_for("MYTABLE", "MYCOL").is_some());
    }
}
