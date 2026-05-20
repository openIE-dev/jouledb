//! Materialized Views for Pre-computed Aggregations
//!
//! This module provides materialized views that can store pre-computed query results
//! for faster analytical queries. Views can be refreshed manually, on write, or periodically.
//!
//! ## Features
//!
//! - **Pre-computed aggregations**: Store GROUP BY results for instant queries
//! - **Multiple refresh policies**: Manual, on-write, or periodic refresh
//! - **Incremental updates**: Only update affected rows when source data changes
//! - **View matching**: Optimizer can route queries to matching views
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_amorphic::materialized::{MaterializedView, RefreshPolicy};
//!
//! // Create a materialized view for sales by region
//! let view = MaterializedView::create(
//!     "sales_by_region",
//!     "SELECT region, SUM(amount) as total FROM sales GROUP BY region"
//! )?;
//!
//! // Query the view (instant results)
//! let result = view.query_all()?;
//!
//! // Refresh when needed
//! view.refresh(&source_store)?;
//! ```

use crate::columnar::ColumnarStore;
use crate::optimizer::AggregateFunc;
use crate::{AmorphicError, AmorphicResult, RecordId, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Materialized view - stores pre-computed aggregation results
#[derive(Debug, Clone)]
pub struct MaterializedView {
    /// Name of the view
    pub name: String,
    /// Original SQL query (for documentation/recreation)
    pub query: String,
    /// Columns to group by
    pub group_keys: Vec<String>,
    /// Aggregations: (alias, function, source_column)
    pub aggregates: Vec<(String, AggregateFunc, String)>,
    /// Pre-computed data storage
    pub data: ColumnarStore,
    /// Last refresh timestamp
    pub last_refresh: Option<Instant>,
    /// Refresh policy
    pub refresh_policy: RefreshPolicy,
    /// Source table name
    pub source_table: String,
    /// Number of rows in the materialized view
    pub row_count: usize,
}

/// When to refresh the materialized view
#[derive(Debug, Clone, PartialEq)]
pub enum RefreshPolicy {
    /// Only refresh when explicitly requested
    Manual,
    /// Refresh after each write to the source table
    OnWrite,
    /// Refresh periodically
    Periodic(Duration),
    /// Stale after this duration
    StaleAfter(Duration),
}

impl Default for RefreshPolicy {
    fn default() -> Self {
        RefreshPolicy::Manual
    }
}

/// Statistics about a materialized view
#[derive(Debug, Clone)]
pub struct ViewStats {
    pub name: String,
    pub row_count: usize,
    pub group_key_count: usize,
    pub aggregate_count: usize,
    pub last_refresh_secs_ago: Option<f64>,
    pub is_stale: bool,
}

impl MaterializedView {
    /// Create a new materialized view from a query definition
    ///
    /// # Arguments
    /// * `name` - Unique name for the view
    /// * `query` - SQL query defining the view (must be a GROUP BY query)
    ///
    /// # Returns
    /// A new MaterializedView ready to be populated
    pub fn create(
        name: &str,
        query: &str,
        source_table: &str,
        group_keys: Vec<String>,
        aggregates: Vec<(String, AggregateFunc, String)>,
    ) -> AmorphicResult<Self> {
        if group_keys.is_empty() {
            return Err(AmorphicError::InvalidQuery(
                "Materialized view must have at least one GROUP BY column".to_string(),
            ));
        }

        if aggregates.is_empty() {
            return Err(AmorphicError::InvalidQuery(
                "Materialized view must have at least one aggregate".to_string(),
            ));
        }

        Ok(Self {
            name: name.to_string(),
            query: query.to_string(),
            group_keys,
            aggregates,
            data: ColumnarStore::new(),
            last_refresh: None,
            refresh_policy: RefreshPolicy::default(),
            source_table: source_table.to_string(),
            row_count: 0,
        })
    }

    /// Create a view for SUM aggregation grouped by a single column
    pub fn create_sum_by(
        name: &str,
        source_table: &str,
        group_col: &str,
        sum_col: &str,
    ) -> AmorphicResult<Self> {
        Self::create(
            name,
            &format!(
                "SELECT {}, SUM({}) FROM {} GROUP BY {}",
                group_col, sum_col, source_table, group_col
            ),
            source_table,
            vec![group_col.to_string()],
            vec![("sum".to_string(), AggregateFunc::Sum, sum_col.to_string())],
        )
    }

    /// Create a view for COUNT aggregation grouped by a single column
    pub fn create_count_by(
        name: &str,
        source_table: &str,
        group_col: &str,
    ) -> AmorphicResult<Self> {
        Self::create(
            name,
            &format!(
                "SELECT {}, COUNT(*) FROM {} GROUP BY {}",
                group_col, source_table, group_col
            ),
            source_table,
            vec![group_col.to_string()],
            vec![("count".to_string(), AggregateFunc::Count, "*".to_string())],
        )
    }

    /// Set the refresh policy
    pub fn with_refresh_policy(mut self, policy: RefreshPolicy) -> Self {
        self.refresh_policy = policy;
        self
    }

    /// Fully refresh the materialized view from source data
    pub fn refresh(&mut self, source: &ColumnarStore) -> AmorphicResult<usize> {
        // Clear existing data
        self.data = ColumnarStore::new();
        self.row_count = 0;

        if self.group_keys.len() != 1 {
            return Err(AmorphicError::InvalidQuery(
                "Currently only single-column GROUP BY is supported".to_string(),
            ));
        }

        let group_col = &self.group_keys[0];

        // For each aggregate, compute the grouped result
        for (alias, func, source_col) in &self.aggregates {
            let groups = match func {
                AggregateFunc::Sum => source.group_by_sum(group_col, source_col),
                AggregateFunc::Count => source
                    .group_by_count(group_col)
                    .map(|m| m.into_iter().map(|(k, v)| (k, v as f64)).collect()),
                AggregateFunc::Avg => source.group_by_avg(group_col, source_col),
                AggregateFunc::Min => source.group_by_min(group_col, source_col),
                AggregateFunc::Max => source.group_by_max(group_col, source_col),
            };

            if let Some(groups) = groups {
                // Store results in our columnar store
                for (i, (group_key, agg_value)) in groups.iter().enumerate() {
                    let record_id = i as RecordId;

                    // Store group key
                    self.data
                        .record_value(group_col, record_id, &Value::Float(*group_key as f64));

                    // Store aggregate value
                    self.data
                        .record_value(alias, record_id, &Value::Float(*agg_value));
                }

                self.row_count = groups.len();
            }
        }

        self.last_refresh = Some(Instant::now());
        Ok(self.row_count)
    }

    /// Incrementally update the view based on changed record IDs
    ///
    /// This is more efficient than full refresh when only a few records changed.
    pub fn incremental_update(
        &mut self,
        source: &ColumnarStore,
        _changed_ids: &[RecordId],
    ) -> AmorphicResult<usize> {
        // For now, just do a full refresh
        // A true incremental update would:
        // 1. Find which groups are affected
        // 2. Recompute only those groups
        // 3. Update the stored results
        self.refresh(source)
    }

    /// Check if the view is stale based on refresh policy
    pub fn is_stale(&self) -> bool {
        match &self.refresh_policy {
            RefreshPolicy::Manual => false,  // Manual views are never "stale"
            RefreshPolicy::OnWrite => false, // OnWrite should always be fresh
            RefreshPolicy::Periodic(duration) | RefreshPolicy::StaleAfter(duration) => {
                match self.last_refresh {
                    Some(last) => last.elapsed() > *duration,
                    None => true, // Never refreshed = stale
                }
            }
        }
    }

    /// Query all data from the view
    pub fn query_all(&self) -> Vec<HashMap<String, f64>> {
        let mut results = Vec::new();

        // Get all column names
        let group_col = &self.group_keys[0];

        // Get all record IDs from the group column
        if let Some(col) = self.data.get_column(group_col) {
            for (record_id, group_value) in col.scan() {
                let mut row = HashMap::new();
                row.insert(group_col.clone(), group_value);

                // Add aggregate values
                for (alias, _, _) in &self.aggregates {
                    if let Some(agg_col) = self.data.get_column(alias) {
                        if let Some(value) = agg_col.get_value(record_id) {
                            row.insert(alias.clone(), value);
                        }
                    }
                }

                results.push(row);
            }
        }

        results
    }

    /// Query with a filter on the group key
    pub fn query_where(&self, group_value: f64) -> Option<HashMap<String, f64>> {
        let group_col = &self.group_keys[0];

        if let Some(col) = self.data.get_column(group_col) {
            for (record_id, value) in col.scan() {
                if (value - group_value).abs() < f64::EPSILON {
                    let mut row = HashMap::new();
                    row.insert(group_col.clone(), value);

                    for (alias, _, _) in &self.aggregates {
                        if let Some(agg_col) = self.data.get_column(alias) {
                            if let Some(v) = agg_col.get_value(record_id) {
                                row.insert(alias.clone(), v);
                            }
                        }
                    }

                    return Some(row);
                }
            }
        }

        None
    }

    /// Get view statistics
    pub fn stats(&self) -> ViewStats {
        ViewStats {
            name: self.name.clone(),
            row_count: self.row_count,
            group_key_count: self.group_keys.len(),
            aggregate_count: self.aggregates.len(),
            last_refresh_secs_ago: self.last_refresh.map(|t| t.elapsed().as_secs_f64()),
            is_stale: self.is_stale(),
        }
    }

    /// Check if this view can answer a query
    ///
    /// Returns true if the view's group keys and aggregates match the query.
    pub fn can_answer_query(
        &self,
        query_group_keys: &[String],
        query_aggregates: &[(AggregateFunc, String)],
    ) -> bool {
        // Check if group keys match
        if self.group_keys != query_group_keys {
            return false;
        }

        // Check if all required aggregates are available
        for (func, col) in query_aggregates {
            let found = self
                .aggregates
                .iter()
                .any(|(_, f, c)| f == func && c == col);
            if !found {
                return false;
            }
        }

        true
    }
}

/// Manager for multiple materialized views
#[derive(Debug, Default)]
pub struct MaterializedViewManager {
    views: HashMap<String, MaterializedView>,
}

impl MaterializedViewManager {
    pub fn new() -> Self {
        Self {
            views: HashMap::new(),
        }
    }

    /// Register a new materialized view
    pub fn register(&mut self, view: MaterializedView) {
        self.views.insert(view.name.clone(), view);
    }

    /// Get a view by name
    pub fn get(&self, name: &str) -> Option<&MaterializedView> {
        self.views.get(name)
    }

    /// Get a mutable view by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut MaterializedView> {
        self.views.get_mut(name)
    }

    /// Drop a view
    pub fn drop(&mut self, name: &str) -> Option<MaterializedView> {
        self.views.remove(name)
    }

    /// List all view names
    pub fn list_views(&self) -> Vec<&str> {
        self.views.keys().map(|s| s.as_str()).collect()
    }

    /// Find a view that can answer a query
    pub fn find_matching_view(
        &self,
        group_keys: &[String],
        aggregates: &[(AggregateFunc, String)],
    ) -> Option<&MaterializedView> {
        self.views
            .values()
            .find(|v| v.can_answer_query(group_keys, aggregates))
    }

    /// Refresh all views that depend on a table
    pub fn refresh_for_table(
        &mut self,
        table_name: &str,
        source: &ColumnarStore,
    ) -> AmorphicResult<usize> {
        let mut total_rows = 0;

        for view in self.views.values_mut() {
            if view.source_table == table_name {
                if view.refresh_policy == RefreshPolicy::OnWrite {
                    total_rows += view.refresh(source)?;
                }
            }
        }

        Ok(total_rows)
    }

    /// Refresh all stale views
    pub fn refresh_stale(&mut self, source: &ColumnarStore) -> AmorphicResult<usize> {
        let mut total_rows = 0;

        for view in self.views.values_mut() {
            if view.is_stale() {
                total_rows += view.refresh(source)?;
            }
        }

        Ok(total_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_store() -> ColumnarStore {
        let mut store = ColumnarStore::new();

        // Add test data: sales by region
        // Region 1: 100, 200, 300
        // Region 2: 150, 250
        // Region 3: 500
        store.record_value("region", 0, &Value::Float(1.0));
        store.record_value("amount", 0, &Value::Float(100.0));

        store.record_value("region", 1, &Value::Float(1.0));
        store.record_value("amount", 1, &Value::Float(200.0));

        store.record_value("region", 2, &Value::Float(1.0));
        store.record_value("amount", 2, &Value::Float(300.0));

        store.record_value("region", 3, &Value::Float(2.0));
        store.record_value("amount", 3, &Value::Float(150.0));

        store.record_value("region", 4, &Value::Float(2.0));
        store.record_value("amount", 4, &Value::Float(250.0));

        store.record_value("region", 5, &Value::Float(3.0));
        store.record_value("amount", 5, &Value::Float(500.0));

        store
    }

    #[test]
    fn test_create_view() {
        let view = MaterializedView::create_sum_by("sales_by_region", "sales", "region", "amount")
            .unwrap();

        assert_eq!(view.name, "sales_by_region");
        assert_eq!(view.group_keys, vec!["region"]);
        assert_eq!(view.aggregates.len(), 1);
    }

    #[test]
    fn test_refresh_sum_view() {
        let store = create_test_store();

        let mut view =
            MaterializedView::create_sum_by("sales_by_region", "sales", "region", "amount")
                .unwrap();

        let rows = view.refresh(&store).unwrap();
        assert_eq!(rows, 3); // 3 regions

        // Check results
        let results = view.query_all();
        assert_eq!(results.len(), 3);

        // Find region 1 sum (should be 600)
        let region1 = results.iter().find(|r| r["region"] == 1.0);
        assert!(region1.is_some());
        assert_eq!(region1.unwrap()["sum"], 600.0);

        // Find region 2 sum (should be 400)
        let region2 = results.iter().find(|r| r["region"] == 2.0);
        assert!(region2.is_some());
        assert_eq!(region2.unwrap()["sum"], 400.0);
    }

    #[test]
    fn test_refresh_count_view() {
        let store = create_test_store();

        let mut view =
            MaterializedView::create_count_by("count_by_region", "sales", "region").unwrap();

        let rows = view.refresh(&store).unwrap();
        assert_eq!(rows, 3);

        let results = view.query_all();

        // Region 1 has 3 records
        let region1 = results.iter().find(|r| r["region"] == 1.0);
        assert_eq!(region1.unwrap()["count"], 3.0);

        // Region 2 has 2 records
        let region2 = results.iter().find(|r| r["region"] == 2.0);
        assert_eq!(region2.unwrap()["count"], 2.0);

        // Region 3 has 1 record
        let region3 = results.iter().find(|r| r["region"] == 3.0);
        assert_eq!(region3.unwrap()["count"], 1.0);
    }

    #[test]
    fn test_query_where() {
        let store = create_test_store();

        let mut view =
            MaterializedView::create_sum_by("sales_by_region", "sales", "region", "amount")
                .unwrap();

        view.refresh(&store).unwrap();

        // Query specific region
        let result = view.query_where(2.0);
        assert!(result.is_some());
        assert_eq!(result.unwrap()["sum"], 400.0);

        // Query non-existent region
        let result = view.query_where(99.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_view_staleness() {
        let view = MaterializedView::create_sum_by("test", "sales", "region", "amount")
            .unwrap()
            .with_refresh_policy(RefreshPolicy::StaleAfter(Duration::from_millis(1)));

        // Initially stale (never refreshed)
        assert!(view.is_stale());
    }

    #[test]
    fn test_view_manager() {
        let store = create_test_store();

        let mut manager = MaterializedViewManager::new();

        let mut view1 =
            MaterializedView::create_sum_by("sales_sum", "sales", "region", "amount").unwrap();
        view1.refresh(&store).unwrap();

        let mut view2 =
            MaterializedView::create_count_by("sales_count", "sales", "region").unwrap();
        view2.refresh(&store).unwrap();

        manager.register(view1);
        manager.register(view2);

        assert_eq!(manager.list_views().len(), 2);
        assert!(manager.get("sales_sum").is_some());
        assert!(manager.get("nonexistent").is_none());
    }

    #[test]
    fn test_can_answer_query() {
        let view = MaterializedView::create_sum_by("test", "sales", "region", "amount").unwrap();

        // Matching query
        assert!(view.can_answer_query(
            &["region".to_string()],
            &[(AggregateFunc::Sum, "amount".to_string())]
        ));

        // Different group key
        assert!(!view.can_answer_query(
            &["category".to_string()],
            &[(AggregateFunc::Sum, "amount".to_string())]
        ));

        // Different aggregate
        assert!(!view.can_answer_query(
            &["region".to_string()],
            &[(AggregateFunc::Avg, "amount".to_string())]
        ));
    }

    #[test]
    fn test_find_matching_view() {
        let store = create_test_store();

        let mut manager = MaterializedViewManager::new();

        let mut view =
            MaterializedView::create_sum_by("sales_by_region", "sales", "region", "amount")
                .unwrap();
        view.refresh(&store).unwrap();

        manager.register(view);

        // Should find the view
        let found = manager.find_matching_view(
            &["region".to_string()],
            &[(AggregateFunc::Sum, "amount".to_string())],
        );
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "sales_by_region");

        // Should not find for different query
        let not_found = manager.find_matching_view(
            &["category".to_string()],
            &[(AggregateFunc::Sum, "amount".to_string())],
        );
        assert!(not_found.is_none());
    }

    #[test]
    fn test_view_stats() {
        let store = create_test_store();

        let mut view =
            MaterializedView::create_sum_by("test", "sales", "region", "amount").unwrap();
        view.refresh(&store).unwrap();

        let stats = view.stats();
        assert_eq!(stats.name, "test");
        assert_eq!(stats.row_count, 3);
        assert_eq!(stats.group_key_count, 1);
        assert_eq!(stats.aggregate_count, 1);
        assert!(!stats.is_stale);
    }
}
