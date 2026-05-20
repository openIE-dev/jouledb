//! Constraint checking: PRIMARY KEY, UNIQUE, FOREIGN KEY.

use super::*;
use super::conversions::*;

impl AmorphicTableStorage {
    /// Check PRIMARY KEY uniqueness for a row about to be inserted.
    /// Returns Ok(()) if no duplicate, Err with message if duplicate found.
    pub fn check_primary_key_unique(&self, table: &str, row: &RowData) -> QueryResult<()> {
        let column_defs = self.get_column_defs(table)?;
        if column_defs.is_empty() {
            return Ok(()); // No column defs -- skip check
        }

        // Find primary key columns
        let pk_cols: Vec<&ColumnDefInfo> = column_defs.iter().filter(|d| d.primary_key).collect();
        if pk_cols.is_empty() {
            return Ok(()); // No primary key defined
        }

        // Get PK values from the row
        let pk_values: Vec<(&str, &AstValue)> = pk_cols
            .iter()
            .filter_map(|def| {
                row.columns
                    .iter()
                    .position(|c| c == &def.name)
                    .map(|idx| (def.name.as_str(), &row.values[idx]))
            })
            .collect();

        if pk_values.is_empty() {
            return Ok(()); // PK columns not in row -- skip
        }

        // Scan table and check for duplicate PK
        let store = self.store.read().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));

        for record in result.records() {
            let existing_row = Self::record_to_row(record, &columns);
            let all_match = pk_values.iter().all(|(col, val)| {
                existing_row
                    .columns
                    .iter()
                    .position(|c| c == *col)
                    .map(|idx| &existing_row.values[idx] == *val)
                    .unwrap_or(false)
            });
            if all_match {
                let pk_desc: Vec<String> = pk_values
                    .iter()
                    .map(|(col, val)| format!("{} = {:?}", col, val))
                    .collect();
                return Err(QueryError::ExecutionError(format!(
                    "UNIQUE constraint failed: duplicate primary key ({})",
                    pk_desc.join(", ")
                )));
            }
        }

        Ok(())
    }

    /// Check UNIQUE column constraints (not primary key, which is checked separately).
    pub fn check_unique_constraints(&self, table: &str, row: &RowData) -> QueryResult<()> {
        let column_defs = self.get_column_defs(table)?;
        if column_defs.is_empty() {
            return Ok(());
        }

        // Find UNIQUE columns (excluding primary keys, which are already checked)
        let unique_cols: Vec<&ColumnDefInfo> = column_defs
            .iter()
            .filter(|d| d.unique && !d.primary_key)
            .collect();
        if unique_cols.is_empty() {
            return Ok(());
        }

        let store = self.store.read().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));

        for unique_def in &unique_cols {
            // Get the value for this unique column from the new row
            let row_val = row
                .columns
                .iter()
                .position(|c| c == &unique_def.name)
                .map(|idx| &row.values[idx]);

            let Some(new_val) = row_val else { continue };
            if matches!(new_val, AstValue::Null) {
                continue;
            } // NULL is always unique

            // Scan existing rows for duplicates
            for record in result.records() {
                let existing_row = Self::record_to_row(record, &columns);
                if let Some(idx) = existing_row
                    .columns
                    .iter()
                    .position(|c| c == &unique_def.name)
                {
                    if &existing_row.values[idx] == new_val {
                        return Err(QueryError::ExecutionError(format!(
                            "UNIQUE constraint failed: column '{}' value {:?} already exists",
                            unique_def.name, new_val
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Check UNIQUE constraints against the store, optionally excluding a specific record (for UPDATE).
    pub(super) fn check_unique_in_store(
        store: &DurableAmorphicStore,
        table: &str,
        columns: &[String],
        column_defs: &[ColumnDefInfo],
        row: &RowData,
        exclude_record_id: Option<RecordId>,
    ) -> QueryResult<()> {
        let unique_cols: Vec<&ColumnDefInfo> = column_defs
            .iter()
            .filter(|d| d.unique || d.primary_key)
            .collect();
        if unique_cols.is_empty() {
            return Ok(());
        }

        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));

        for unique_def in &unique_cols {
            let row_val = row
                .columns
                .iter()
                .position(|c| c == &unique_def.name)
                .map(|idx| &row.values[idx]);
            let Some(new_val) = row_val else { continue };
            if matches!(new_val, AstValue::Null) {
                continue;
            }

            for record in result.records() {
                if let Some(exclude_id) = exclude_record_id {
                    if record.id == exclude_id {
                        continue;
                    }
                }
                let existing_row = Self::record_to_row(record, columns);
                if let Some(idx) = existing_row
                    .columns
                    .iter()
                    .position(|c| c == &unique_def.name)
                {
                    if &existing_row.values[idx] == new_val {
                        let constraint_type = if unique_def.primary_key {
                            "PRIMARY KEY"
                        } else {
                            "UNIQUE"
                        };
                        return Err(QueryError::ExecutionError(format!(
                            "{} constraint failed: column '{}' value {:?} already exists",
                            constraint_type, unique_def.name, new_val
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate FOREIGN KEY constraints for an INSERT/UPDATE row.
    /// Checks that each FK column value exists in the referenced table's referenced column.
    pub fn check_foreign_keys(&self, table: &str, row: &RowData) -> QueryResult<()> {
        let column_defs = self.get_column_defs(table)?;
        let store = self.store.read().map_err(lock_error)?;
        for def in &column_defs {
            if let Some(ref fk) = def.foreign_key {
                // Get the value of the FK column in the row
                let fk_val = row
                    .columns
                    .iter()
                    .position(|c| c == &def.name)
                    .map(|idx| &row.values[idx]);
                let Some(val) = fk_val else { continue };
                if matches!(val, AstValue::Null) {
                    continue;
                } // NULL FK is allowed

                // Check if the referenced value exists in the referenced table
                let ref_columns = self
                    .get_schema_columns(&store, &fk.ref_table)
                    .map_err(|_| {
                        QueryError::ExecutionError(format!(
                            "Referenced table '{}' does not exist (FOREIGN KEY on '{}.{}')",
                            fk.ref_table, table, def.name
                        ))
                    })?;
                let ref_result =
                    store.query_equals(TABLE_FIELD, &AmorphicValue::String(fk.ref_table.clone()));
                let found = ref_result.records().iter().any(|record| {
                    let ref_row = Self::record_to_row(record, &ref_columns);
                    ref_row
                        .columns
                        .iter()
                        .position(|c| c == &fk.ref_column)
                        .map(|idx| &ref_row.values[idx] == val)
                        .unwrap_or(false)
                });
                if !found {
                    return Err(QueryError::ExecutionError(format!(
                        "FOREIGN KEY constraint failed: value {:?} in column '{}.{}' does not exist in '{}.{}'",
                        val, table, def.name, fk.ref_table, fk.ref_column
                    )));
                }
            }
        }
        Ok(())
    }

    /// Check if any other table has a FK referencing a value in this table that would be violated by deletion.
    /// Returns an error if any FK would be violated (unless ON DELETE CASCADE/SET NULL is configured).
    pub fn check_foreign_key_references_on_delete(
        &self,
        table: &str,
        row: &RowData,
    ) -> QueryResult<Option<Vec<(String, String, String)>>> {
        let store = self.store.read().map_err(lock_error)?;
        let mut cascade_actions: Vec<(String, String, String)> = Vec::new(); // (child_table, child_col, action)

        // Scan all schemas to find FK references to this table
        let schema_result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(SCHEMA_TABLE.to_string()),
        );
        for schema_record in schema_result.records() {
            let child_table = schema_record.get(SCHEMA_NAME_FIELD).and_then(|v| {
                if let AmorphicValue::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            });
            let Some(child_table) = child_table else {
                continue;
            };
            if child_table == table {
                continue;
            } // skip self-references for now

            let child_defs = self
                .get_column_defs_from_store(&store, &child_table)
                .unwrap_or_default();
            for def in &child_defs {
                if let Some(ref fk) = def.foreign_key {
                    if fk.ref_table == table {
                        // This child FK references our table -- check if the deleted row's value is referenced
                        let ref_val = row
                            .columns
                            .iter()
                            .position(|c| c == &fk.ref_column)
                            .map(|idx| &row.values[idx]);
                        let Some(val) = ref_val else { continue };
                        if matches!(val, AstValue::Null) {
                            continue;
                        }

                        let child_columns = self
                            .get_schema_columns(&store, &child_table)
                            .unwrap_or_default();
                        let child_result = store
                            .query_equals(TABLE_FIELD, &AmorphicValue::String(child_table.clone()));
                        let has_ref = child_result.records().iter().any(|record| {
                            let child_row = Self::record_to_row(record, &child_columns);
                            child_row
                                .columns
                                .iter()
                                .position(|c| c == &def.name)
                                .map(|idx| &child_row.values[idx] == val)
                                .unwrap_or(false)
                        });

                        if has_ref {
                            let action = fk.on_delete.as_deref().unwrap_or("RESTRICT");
                            match action {
                                "CASCADE" => {
                                    cascade_actions.push((
                                        child_table.clone(),
                                        def.name.clone(),
                                        "CASCADE".to_string(),
                                    ));
                                }
                                "SET NULL" => {
                                    cascade_actions.push((
                                        child_table.clone(),
                                        def.name.clone(),
                                        "SET NULL".to_string(),
                                    ));
                                }
                                _ => {
                                    // RESTRICT / NO ACTION
                                    return Err(QueryError::ExecutionError(format!(
                                        "FOREIGN KEY constraint failed: cannot delete from '{}' because '{}' references it via column '{}'",
                                        table, child_table, def.name
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }

        if cascade_actions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(cascade_actions))
        }
    }

    /// Check if any other table has a foreign key referencing the given table.
    /// Returns the first (child_table, child_column) found, or None.
    pub fn has_foreign_key_references(&self, table: &str) -> QueryResult<Option<(String, String)>> {
        let store = self.store.read().map_err(lock_error)?;
        let schema_result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(SCHEMA_TABLE.to_string()),
        );
        for schema_record in schema_result.records() {
            let child_table = schema_record.get(SCHEMA_NAME_FIELD).and_then(|v| {
                if let AmorphicValue::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            });
            let Some(child_table) = child_table else {
                continue;
            };
            if child_table == table {
                continue;
            }
            let child_defs = self
                .get_column_defs_from_store(&store, &child_table)
                .unwrap_or_default();
            for def in &child_defs {
                if let Some(ref fk) = def.foreign_key {
                    if fk.ref_table == table {
                        return Ok(Some((child_table, def.name.clone())));
                    }
                }
            }
        }
        Ok(None)
    }
}
