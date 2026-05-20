//! Full-text index management: create, drop, search, term freq, doc count, incremental updates.

use super::*;
use super::conversions::*;

impl AmorphicTableStorage {
    /// Create a full-text index on the specified columns of a table.
    pub fn create_fulltext_index(
        &self,
        name: &str,
        table: &str,
        columns: &[String],
        analyzer_name: &str,
    ) -> QueryResult<()> {
        // Verify table exists
        if !self.table_exists(table)? {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' does not exist",
                table
            )));
        }

        // Check if fulltext index already exists
        let store = self.store.read().map_err(lock_error)?;
        let existing =
            store.query_equals(INDEX_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let exists = existing
            .records()
            .iter()
            .any(|r| r.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string())));
        drop(store);

        if exists {
            return Err(QueryError::ExecutionError(format!(
                "Fulltext index '{}' already exists",
                name
            )));
        }

        // Store fulltext index metadata (including analyzer name)
        let columns_json: Vec<serde_json::Value> = columns
            .iter()
            .map(|c| serde_json::Value::String(c.clone()))
            .collect();

        let index_json = serde_json::json!({
            TABLE_FIELD: INDEX_TABLE,
            INDEX_NAME_FIELD: name,
            INDEX_TABLE_REF: table,
            INDEX_COLUMNS_FIELD: columns_json,
            "__index_type__": "fulltext",
            "__ft_analyzer__": analyzer_name,
        });

        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&index_json.to_string())
            .map_err(amorphic_error)?;

        // Build the index: scan existing rows, compute avg_dl, index text content
        let schema_columns = self.get_schema_columns(&store, table).unwrap_or_default();
        let rows = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));

        let analyzer = crate::fts_analyzer::create_analyzer(analyzer_name);
        let mut total_tokens = 0_u64;
        let mut doc_count = 0_u64;

        for record in rows.records() {
            let doc_id = record.id.to_string();
            let mut text_parts: Vec<String> = Vec::new();
            for col in columns {
                if let Some(val) = record.get(col) {
                    match val {
                        AmorphicValue::String(s) => text_parts.push(s.clone()),
                        _ => text_parts.push(format!("{:?}", val)),
                    }
                }
            }
            if !text_parts.is_empty() {
                let content = text_parts.join(" ");
                let tokens = analyzer.tokenize(&content);
                total_tokens += tokens.len() as u64;
                doc_count += 1;
                // Store the indexed document content for search
                let ft_doc = serde_json::json!({
                    TABLE_FIELD: format!("__ft_{}__", name),
                    "__ft_doc_id__": doc_id,
                    "__ft_content__": content,
                    "__ft_source_table__": table,
                });
                let _ = store.ingest_json(&ft_doc.to_string());
            }
        }

        // Store avg_dl metadata for BM25
        let avg_dl = if doc_count > 0 {
            total_tokens as f64 / doc_count as f64
        } else {
            100.0
        };
        let meta_doc = serde_json::json!({
            TABLE_FIELD: format!("__ft_{}_meta__", name),
            "__ft_avg_dl__": avg_dl,
            "__ft_doc_count__": doc_count,
        });
        let _ = store.ingest_json(&meta_doc.to_string());

        let _ = schema_columns; // used above in get_schema_columns

        Ok(())
    }

    /// Drop a fulltext index by name.
    pub fn drop_fulltext_index(&self, name: &str) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;

        // Remove index metadata
        let result = store.query_equals(INDEX_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let index_ids: Vec<RecordId> = result
            .records()
            .iter()
            .filter(|r| {
                r.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string()))
                    && r.get("__index_type__")
                        == Some(&AmorphicValue::String("fulltext".to_string()))
            })
            .map(|r| r.id)
            .collect();

        if index_ids.is_empty() {
            return Err(QueryError::ExecutionError(format!(
                "Fulltext index '{}' does not exist",
                name
            )));
        }

        for id in &index_ids {
            let _ = store.delete(*id);
        }

        // Remove indexed documents
        let ft_table = format!("__ft_{}__", name);
        let ft_docs = store.query_equals(TABLE_FIELD, &AmorphicValue::String(ft_table));
        let ft_ids: Vec<RecordId> = ft_docs.records().iter().map(|r| r.id).collect();
        for id in &ft_ids {
            let _ = store.delete(*id);
        }

        Ok(())
    }

    /// Get fulltext index metadata: (table, columns) for a given index name.
    pub fn get_fulltext_index(&self, name: &str) -> QueryResult<Option<(String, Vec<String>)>> {
        let store = self.store.read().map_err(lock_error)?;
        let result = store.query_equals(INDEX_NAME_FIELD, &AmorphicValue::String(name.to_string()));

        for record in result.records() {
            if record.get(TABLE_FIELD) != Some(&AmorphicValue::String(INDEX_TABLE.to_string())) {
                continue;
            }
            if record.get("__index_type__") != Some(&AmorphicValue::String("fulltext".to_string()))
            {
                continue;
            }
            let table = match record.get(INDEX_TABLE_REF) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let columns = match record.get(INDEX_COLUMNS_FIELD) {
                Some(AmorphicValue::Array(arr)) => arr
                    .iter()
                    .filter_map(|v| match v {
                        AmorphicValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
                _ => continue,
            };
            return Ok(Some((table, columns)));
        }
        Ok(None)
    }

    /// List all fulltext indexes, returning (name, table, columns).
    pub fn list_fulltext_indexes(&self) -> QueryResult<Vec<(String, String, Vec<String>)>> {
        let store = self.store.read().map_err(lock_error)?;
        let result = store.query_equals(
            "__index_type__",
            &AmorphicValue::String("fulltext".to_string()),
        );
        let mut indexes = Vec::new();

        for record in result.records() {
            if record.get(TABLE_FIELD) != Some(&AmorphicValue::String(INDEX_TABLE.to_string())) {
                continue;
            }
            let name = match record.get(INDEX_NAME_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let table = match record.get(INDEX_TABLE_REF) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let columns = match record.get(INDEX_COLUMNS_FIELD) {
                Some(AmorphicValue::Array(arr)) => arr
                    .iter()
                    .filter_map(|v| match v {
                        AmorphicValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
                _ => continue,
            };
            indexes.push((name, table, columns));
        }
        Ok(indexes)
    }

    /// Get the analyzer name for a fulltext index.
    pub fn get_fulltext_analyzer(&self, index_name: &str) -> QueryResult<String> {
        let store = self.store.read().map_err(lock_error)?;
        let result = store.query_equals(
            INDEX_NAME_FIELD,
            &AmorphicValue::String(index_name.to_string()),
        );
        for record in result.records() {
            if record.get(TABLE_FIELD) != Some(&AmorphicValue::String(INDEX_TABLE.to_string())) {
                continue;
            }
            if let Some(AmorphicValue::String(a)) = record.get("__ft_analyzer__") {
                return Ok(a.clone());
            }
        }
        Ok("standard".to_string())
    }

    /// Search a fulltext index, returning (doc_id, score, content) tuples
    /// sorted by relevance score descending. Uses proper BM25 with correct
    /// IDF formula and tracked average document length.
    pub fn search_fulltext_index(
        &self,
        index_name: &str,
        query: &str,
        limit: Option<usize>,
    ) -> QueryResult<Vec<(String, f64, String)>> {
        let store = self.store.read().map_err(lock_error)?;
        let ft_table = format!("__ft_{}__", index_name);
        let docs = store.query_equals(TABLE_FIELD, &AmorphicValue::String(ft_table));

        // Load analyzer for this index
        let analyzer_name = {
            let idx_result = store.query_equals(
                INDEX_NAME_FIELD,
                &AmorphicValue::String(index_name.to_string()),
            );
            let mut name = "standard".to_string();
            for record in idx_result.records() {
                if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string()))
                {
                    if let Some(AmorphicValue::String(a)) = record.get("__ft_analyzer__") {
                        name = a.clone();
                    }
                    break;
                }
            }
            name
        };
        let analyzer = crate::fts_analyzer::create_analyzer(&analyzer_name);

        // Load avg_dl from metadata
        let meta_table = format!("__ft_{}_meta__", index_name);
        let meta_docs = store.query_equals(TABLE_FIELD, &AmorphicValue::String(meta_table));
        let avg_dl = meta_docs
            .records()
            .first()
            .and_then(|r| match r.get("__ft_avg_dl__") {
                Some(AmorphicValue::Float(f)) => Some(*f),
                _ => None,
            })
            .unwrap_or(100.0);

        // Tokenize query using the same analyzer
        let query_tokens = analyzer.tokenize(query);
        let query_terms: Vec<String> = query_tokens.iter().map(|t| t.text.clone()).collect();

        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let total_docs = docs.records().len() as f64;

        // Compute document frequency for each query term
        let mut doc_freqs: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        for record in docs.records() {
            let content = match record.get("__ft_content__") {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let doc_tokens = analyzer.tokenize(&content);
            let doc_term_set: std::collections::HashSet<&str> =
                doc_tokens.iter().map(|t| t.text.as_str()).collect();
            for term in &query_terms {
                if doc_term_set.contains(term.as_str()) {
                    *doc_freqs.entry(term.clone()).or_insert(0.0) += 1.0;
                }
            }
        }

        let mut results: Vec<(String, f64, String)> = Vec::new();

        for record in docs.records() {
            let doc_id = match record.get("__ft_doc_id__") {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let content = match record.get("__ft_content__") {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };

            let doc_tokens = analyzer.tokenize(&content);
            let doc_len = doc_tokens.len() as f64;

            // Count term frequencies
            let mut tf_map: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
            for tok in &doc_tokens {
                *tf_map.entry(tok.text.as_str()).or_insert(0.0) += 1.0;
            }

            let mut score = 0.0_f64;
            for term in &query_terms {
                let tf = tf_map.get(term.as_str()).copied().unwrap_or(0.0);
                if tf > 0.0 {
                    let df = doc_freqs.get(term).copied().unwrap_or(1.0);
                    let idf = crate::fts_analyzer::bm25_idf(total_docs, df);
                    score += crate::fts_analyzer::bm25_term_score(tf, doc_len, avg_dl, idf);
                }
            }

            if score > 0.0 {
                results.push((doc_id, score, content));
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(limit) = limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Search a fulltext index with boolean query semantics and per-field boosting.
    pub fn search_fulltext_index_boosted(
        &self,
        index_name: &str,
        query: &str,
        limit: Option<usize>,
        field_boosts: &std::collections::HashMap<String, f64>,
    ) -> QueryResult<Vec<(String, f64, String)>> {
        use crate::fts_analyzer::{
            BooleanClause, boolean_query_scoring_terms, matches_boolean_query, parse_boolean_query,
        };

        let store = self.store.read().map_err(lock_error)?;
        let ft_table = format!("__ft_{}__", index_name);
        let docs = store.query_equals(TABLE_FIELD, &AmorphicValue::String(ft_table));

        // Load analyzer
        let analyzer_name = {
            let idx_result = store.query_equals(
                INDEX_NAME_FIELD,
                &AmorphicValue::String(index_name.to_string()),
            );
            let mut name = "standard".to_string();
            for record in idx_result.records() {
                if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string()))
                {
                    if let Some(AmorphicValue::String(a)) = record.get("__ft_analyzer__") {
                        name = a.clone();
                    }
                    break;
                }
            }
            name
        };
        let analyzer = crate::fts_analyzer::create_analyzer(&analyzer_name);

        // Load avg_dl
        let meta_table = format!("__ft_{}_meta__", index_name);
        let meta_docs = store.query_equals(TABLE_FIELD, &AmorphicValue::String(meta_table));
        let avg_dl = meta_docs
            .records()
            .first()
            .and_then(|r| match r.get("__ft_avg_dl__") {
                Some(AmorphicValue::Float(f)) => Some(*f),
                _ => None,
            })
            .unwrap_or(100.0);

        // Parse boolean query
        let clauses = parse_boolean_query(query);
        let has_boolean_ops = clauses.iter().any(|c| {
            matches!(
                c,
                BooleanClause::Required(_) | BooleanClause::Excluded(_) | BooleanClause::Phrase(_)
            )
        });

        // Get scoring terms (required + optional + phrase words)
        let scoring_terms = if has_boolean_ops {
            boolean_query_scoring_terms(&clauses)
        } else {
            // Plain query: tokenize normally
            analyzer
                .tokenize(query)
                .iter()
                .map(|t| t.text.clone())
                .collect()
        };

        if scoring_terms.is_empty() && !has_boolean_ops {
            return Ok(Vec::new());
        }

        let total_docs = docs.records().len() as f64;

        // Load index columns for per-field scoring
        let index_columns: Vec<String> = {
            let idx_result = store.query_equals(
                INDEX_NAME_FIELD,
                &AmorphicValue::String(index_name.to_string()),
            );
            let mut cols = Vec::new();
            for record in idx_result.records() {
                if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string()))
                {
                    if let Some(AmorphicValue::Array(arr)) = record.get(INDEX_COLUMNS_FIELD) {
                        for v in arr {
                            if let AmorphicValue::String(s) = v {
                                cols.push(s.clone());
                            }
                        }
                    }
                    break;
                }
            }
            cols
        };

        // Compute doc_freqs
        let mut doc_freqs: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        for record in docs.records() {
            let content = match record.get("__ft_content__") {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let doc_tokens = analyzer.tokenize(&content);
            let doc_term_set: std::collections::HashSet<&str> =
                doc_tokens.iter().map(|t| t.text.as_str()).collect();
            for term in &scoring_terms {
                if doc_term_set.contains(term.as_str()) {
                    *doc_freqs.entry(term.clone()).or_insert(0.0) += 1.0;
                }
            }
        }

        let mut results: Vec<(String, f64, String)> = Vec::new();

        for record in docs.records() {
            let doc_id = match record.get("__ft_doc_id__") {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let content = match record.get("__ft_content__") {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };

            let doc_tokens = analyzer.tokenize(&content);
            let doc_term_set: std::collections::HashSet<&str> =
                doc_tokens.iter().map(|t| t.text.as_str()).collect();

            // Boolean filter: check required/excluded/phrase constraints
            if has_boolean_ops
                && !matches_boolean_query(&clauses, &doc_term_set, &content, analyzer.as_ref())
            {
                continue;
            }

            let doc_len = doc_tokens.len() as f64;

            // Per-field boosted scoring if we have multiple columns
            let score = if !field_boosts.is_empty() && index_columns.len() > 1 {
                let field_contents: Vec<&str> = content.splitn(index_columns.len(), ' ').collect();
                let mut total_score = 0.0_f64;
                for (col_idx, col_name) in index_columns.iter().enumerate() {
                    let boost = field_boosts.get(col_name).copied().unwrap_or(1.0);
                    let field_text = field_contents.get(col_idx).unwrap_or(&"");
                    let field_tokens = analyzer.tokenize(field_text);
                    let field_len = field_tokens.len() as f64;
                    let mut tf_map: std::collections::HashMap<&str, f64> =
                        std::collections::HashMap::new();
                    for tok in &field_tokens {
                        *tf_map.entry(tok.text.as_str()).or_insert(0.0) += 1.0;
                    }
                    for term in &scoring_terms {
                        let tf = tf_map.get(term.as_str()).copied().unwrap_or(0.0);
                        if tf > 0.0 {
                            let df = doc_freqs.get(term).copied().unwrap_or(1.0);
                            let idf = crate::fts_analyzer::bm25_idf(total_docs, df);
                            total_score += boost
                                * crate::fts_analyzer::bm25_term_score(
                                    tf,
                                    field_len.max(1.0),
                                    avg_dl,
                                    idf,
                                );
                        }
                    }
                }
                total_score
            } else {
                // Standard single-field scoring
                let mut tf_map: std::collections::HashMap<&str, f64> =
                    std::collections::HashMap::new();
                for tok in &doc_tokens {
                    *tf_map.entry(tok.text.as_str()).or_insert(0.0) += 1.0;
                }
                let mut score = 0.0_f64;
                for term in &scoring_terms {
                    let tf = tf_map.get(term.as_str()).copied().unwrap_or(0.0);
                    if tf > 0.0 {
                        let df = doc_freqs.get(term).copied().unwrap_or(1.0);
                        let idf = crate::fts_analyzer::bm25_idf(total_docs, df);
                        score += crate::fts_analyzer::bm25_term_score(tf, doc_len, avg_dl, idf);
                    }
                }
                score
            };

            if score > 0.0 || has_boolean_ops {
                // For boolean queries with only excluded/phrase, score may be 0 -- still include
                results.push((doc_id, score.max(0.001), content));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(limit) = limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Get the term frequency of a specific term across all docs in a fulltext index.
    pub fn fts_term_freq(&self, index_name: &str, term: &str) -> QueryResult<u64> {
        let store = self.store.read().map_err(lock_error)?;
        let ft_table = format!("__ft_{}__", index_name);
        let _docs = store.query_equals(TABLE_FIELD, &AmorphicValue::String(ft_table));

        let analyzer_name = self
            .get_fulltext_analyzer(index_name)
            .unwrap_or_else(|_| "standard".to_string());
        drop(store);

        let analyzer = crate::fts_analyzer::create_analyzer(&analyzer_name);
        let stemmed = crate::fts_analyzer::porter_stem(&term.to_lowercase());

        let store = self.store.read().map_err(lock_error)?;
        let ft_table = format!("__ft_{}__", index_name);
        let docs = store.query_equals(TABLE_FIELD, &AmorphicValue::String(ft_table));

        let mut count = 0_u64;
        for record in docs.records() {
            let content = match record.get("__ft_content__") {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let doc_tokens = analyzer.tokenize(&content);
            if doc_tokens.iter().any(|t| t.text == stemmed) {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Get the total document count in a fulltext index.
    pub fn fts_doc_count(&self, index_name: &str) -> QueryResult<u64> {
        let store = self.store.read().map_err(lock_error)?;
        let ft_table = format!("__ft_{}__", index_name);
        let docs = store.query_equals(TABLE_FIELD, &AmorphicValue::String(ft_table));
        Ok(docs.records().len() as u64)
    }

    /// Update fulltext indexes for a table after INSERT.
    pub fn update_fulltext_indexes_on_insert(
        &self,
        table: &str,
        record_id: u64,
        record: &serde_json::Value,
    ) -> QueryResult<()> {
        // Find all fulltext indexes for this table
        let ft_indexes = self.list_fulltext_indexes()?;
        let relevant: Vec<_> = ft_indexes
            .iter()
            .filter(|(_, tbl, _)| tbl == table)
            .collect();

        if relevant.is_empty() {
            return Ok(());
        }

        let mut store = self.store.write().map_err(lock_error)?;
        for (idx_name, _, columns) in relevant {
            let mut text_parts: Vec<String> = Vec::new();
            for col in columns {
                if let Some(val) = record.get(col) {
                    if let Some(s) = val.as_str() {
                        text_parts.push(s.to_string());
                    } else if !val.is_null() {
                        text_parts.push(val.to_string());
                    }
                }
            }
            if !text_parts.is_empty() {
                let content = text_parts.join(" ");
                let ft_doc = serde_json::json!({
                    TABLE_FIELD: format!("__ft_{}__", idx_name),
                    "__ft_doc_id__": record_id.to_string(),
                    "__ft_content__": content,
                    "__ft_source_table__": table,
                });
                let _ = store.ingest_json(&ft_doc.to_string());
            }
        }
        Ok(())
    }

    /// Remove fulltext index entries for a deleted record.
    pub fn update_fulltext_indexes_on_delete(
        &self,
        table: &str,
        record_id: u64,
    ) -> QueryResult<()> {
        let ft_indexes = self.list_fulltext_indexes()?;
        let relevant: Vec<_> = ft_indexes
            .iter()
            .filter(|(_, tbl, _)| tbl == table)
            .collect();

        if relevant.is_empty() {
            return Ok(());
        }

        let mut store = self.store.write().map_err(lock_error)?;
        for (idx_name, _, _) in relevant {
            let ft_table = format!("__ft_{}__", idx_name);
            let docs = store.query_equals(
                "__ft_doc_id__",
                &AmorphicValue::String(record_id.to_string()),
            );
            let ft_ids: Vec<RecordId> = docs
                .records()
                .iter()
                .filter(|r| r.get(TABLE_FIELD) == Some(&AmorphicValue::String(ft_table.clone())))
                .map(|r| r.id)
                .collect();
            for id in &ft_ids {
                let _ = store.delete(*id);
            }
        }
        Ok(())
    }

    /// Update fulltext indexes for a table after UPDATE.
    pub fn update_fulltext_indexes_on_update(
        &self,
        table: &str,
        record_id: u64,
        record: &serde_json::Value,
    ) -> QueryResult<()> {
        // Delete old FTS entries and re-index
        self.update_fulltext_indexes_on_delete(table, record_id)?;
        self.update_fulltext_indexes_on_insert(table, record_id, record)?;
        Ok(())
    }
}
