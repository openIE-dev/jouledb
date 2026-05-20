//! Trigger management and catalog operations (references, buckets, API endpoints).

use super::*;
use super::conversions::*;

impl AmorphicTableStorage {
    /// Create a new trigger stored in __triggers__ metadata.
    pub fn create_trigger(
        &self,
        name: &str,
        table: &str,
        timing: &str,
        event: &str,
        body: &str,
        or_replace: bool,
    ) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;

        // Check if trigger already exists
        let existing =
            store.query_equals(TRIGGER_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let existing_ids: Vec<RecordId> = existing
            .records()
            .iter()
            .filter(|r| {
                r.get(TABLE_FIELD) == Some(&AmorphicValue::String(TRIGGER_TABLE.to_string()))
            })
            .map(|r| r.id)
            .collect();

        if !existing_ids.is_empty() {
            if or_replace {
                for id in &existing_ids {
                    let _ = store.delete(*id);
                }
            } else {
                return Err(QueryError::ExecutionError(format!(
                    "Trigger '{}' already exists",
                    name
                )));
            }
        }

        // Verify target table exists
        let table_schema =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        let table_exists = table_schema
            .records()
            .iter()
            .any(|r| r.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string())));
        if !table_exists {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' does not exist (CREATE TRIGGER)",
                table
            )));
        }

        // Store trigger metadata
        let trigger_json = serde_json::json!({
            TABLE_FIELD: TRIGGER_TABLE,
            TRIGGER_NAME_FIELD: name,
            TRIGGER_TABLE_REF: table,
            TRIGGER_TIMING_FIELD: timing,
            TRIGGER_EVENT_FIELD: event,
            TRIGGER_BODY_FIELD: body,
        });
        store
            .ingest_json(&trigger_json.to_string())
            .map_err(amorphic_error)?;

        Ok(())
    }

    /// Drop a trigger by name.
    pub fn drop_trigger(&self, name: &str, if_exists: bool) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;

        let result =
            store.query_equals(TRIGGER_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let trigger_ids: Vec<RecordId> = result
            .records()
            .iter()
            .filter(|r| {
                r.get(TABLE_FIELD) == Some(&AmorphicValue::String(TRIGGER_TABLE.to_string()))
            })
            .map(|r| r.id)
            .collect();

        if trigger_ids.is_empty() {
            if if_exists {
                return Ok(());
            }
            return Err(QueryError::ExecutionError(format!(
                "Trigger '{}' does not exist",
                name
            )));
        }

        for id in &trigger_ids {
            let _ = store.delete(*id);
        }
        Ok(())
    }

    /// List all triggers (returned from the list_triggers method).
    pub fn list_triggers(&self) -> Vec<TriggerInfo> {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(TRIGGER_TABLE.to_string()),
        );
        result
            .records()
            .iter()
            .filter_map(|r| {
                let name = match r.get(TRIGGER_NAME_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };
                let table = match r.get(TRIGGER_TABLE_REF) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };
                let timing = match r.get(TRIGGER_TIMING_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };
                let event = match r.get(TRIGGER_EVENT_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };
                let body = match r.get(TRIGGER_BODY_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };
                Some(TriggerInfo {
                    name,
                    table,
                    timing,
                    event,
                    body,
                })
            })
            .collect()
    }

    /// Get all triggers for a given table, timing, and event.
    pub fn get_triggers(&self, table: &str, timing: &str, event: &str) -> Vec<TriggerInfo> {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(TRIGGER_TABLE.to_string()),
        );
        result
            .records()
            .iter()
            .filter_map(|r| {
                let t_table = match r.get(TRIGGER_TABLE_REF) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };
                if t_table != table {
                    return None;
                }

                let t_timing = match r.get(TRIGGER_TIMING_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };
                if t_timing != timing {
                    return None;
                }

                let t_event = match r.get(TRIGGER_EVENT_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };
                if t_event != event {
                    return None;
                }

                let name = match r.get(TRIGGER_NAME_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };
                let body = match r.get(TRIGGER_BODY_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => return None,
                };

                Some(TriggerInfo {
                    name,
                    table: t_table,
                    timing: t_timing,
                    event: t_event,
                    body,
                })
            })
            .collect()
    }

    /// Store a reference definition in the catalog.
    pub fn create_reference(
        &self,
        name: &str,
        src_table: &str,
        src_column: &str,
        tgt_table: &str,
        tgt_column: &str,
    ) -> QueryResult<()> {
        let json = serde_json::json!({
            TABLE_FIELD: REFERENCE_TABLE,
            REFERENCE_NAME_FIELD: name,
            REFERENCE_SRC_TABLE_FIELD: src_table,
            REFERENCE_SRC_COLUMN_FIELD: src_column,
            REFERENCE_TGT_TABLE_FIELD: tgt_table,
            REFERENCE_TGT_COLUMN_FIELD: tgt_column,
        });
        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&json.to_string())
            .map_err(amorphic_error)?;
        Ok(())
    }

    /// Retrieve a reference definition by name.
    pub fn get_reference(&self, name: &str) -> QueryResult<Option<ReferenceInfo>> {
        let store = self.store.read().map_err(lock_error)?;
        let result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(REFERENCE_TABLE.to_string()),
        );
        for record in result.records() {
            let ref_name = match record.get(REFERENCE_NAME_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            if ref_name != name {
                continue;
            }
            let src_table = match record.get(REFERENCE_SRC_TABLE_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let src_column = match record.get(REFERENCE_SRC_COLUMN_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let tgt_table = match record.get(REFERENCE_TGT_TABLE_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let tgt_column = match record.get(REFERENCE_TGT_COLUMN_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            return Ok(Some(ReferenceInfo {
                name: ref_name,
                src_table,
                src_column,
                tgt_table,
                tgt_column,
            }));
        }
        Ok(None)
    }

    // ==================== Bucket catalog ====================

    /// Store a bucket definition in the catalog.
    pub fn create_bucket(&self, name: &str, max_size: Option<u64>) -> QueryResult<()> {
        let mut json = serde_json::json!({
            TABLE_FIELD: BUCKET_TABLE,
            BUCKET_NAME_FIELD: name,
        });
        if let Some(sz) = max_size {
            json[BUCKET_MAX_SIZE_FIELD] = serde_json::Value::Number(sz.into());
        }
        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&json.to_string())
            .map_err(amorphic_error)?;
        Ok(())
    }

    /// Retrieve a bucket definition by name.
    pub fn get_bucket(&self, name: &str) -> QueryResult<Option<BucketInfo>> {
        let store = self.store.read().map_err(lock_error)?;
        let result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(BUCKET_TABLE.to_string()),
        );
        for record in result.records() {
            let bucket_name = match record.get(BUCKET_NAME_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            if bucket_name != name {
                continue;
            }
            let max_size = match record.get(BUCKET_MAX_SIZE_FIELD) {
                Some(AmorphicValue::Int(n)) => Some(*n as u64),
                _ => None,
            };
            return Ok(Some(BucketInfo {
                name: bucket_name,
                max_size,
            }));
        }
        Ok(None)
    }

    // ==================== API endpoint catalog ====================

    /// Store an API endpoint definition in the catalog.
    pub fn create_api_endpoint(
        &self,
        path: &str,
        method: &str,
        handler_sql: &str,
    ) -> QueryResult<()> {
        let json = serde_json::json!({
            TABLE_FIELD: API_TABLE,
            API_PATH_FIELD: path,
            API_METHOD_FIELD: method,
            API_HANDLER_FIELD: handler_sql,
        });
        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&json.to_string())
            .map_err(amorphic_error)?;
        Ok(())
    }

    /// Retrieve an API endpoint definition by path and method.
    pub fn get_api_endpoint(
        &self,
        path: &str,
        method: &str,
    ) -> QueryResult<Option<ApiEndpointInfo>> {
        let store = self.store.read().map_err(lock_error)?;
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(API_TABLE.to_string()));
        for record in result.records() {
            let api_path = match record.get(API_PATH_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            if api_path != path {
                continue;
            }
            let api_method = match record.get(API_METHOD_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            if api_method != method {
                continue;
            }
            let handler_sql = match record.get(API_HANDLER_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            return Ok(Some(ApiEndpointInfo {
                path: api_path,
                method: api_method,
                handler_sql,
            }));
        }
        Ok(None)
    }

    /// List all API endpoint definitions.
    pub fn list_api_endpoints(&self) -> QueryResult<Vec<ApiEndpointInfo>> {
        let store = self.store.read().map_err(lock_error)?;
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(API_TABLE.to_string()));
        let mut endpoints = Vec::new();
        for record in result.records() {
            let path = match record.get(API_PATH_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let method = match record.get(API_METHOD_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let handler_sql = match record.get(API_HANDLER_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            endpoints.push(ApiEndpointInfo {
                path,
                method,
                handler_sql,
            });
        }
        Ok(endpoints)
    }
}
