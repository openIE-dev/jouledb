//! Spatial index manager — grid-based spatial index for point data.
//!
//! Provides bounding-box and radius pre-filtering for spatial queries
//! like `WHERE ST_DISTANCE(geom, ST_POINT(x,y)) < radius`.
//! This is a coarse pre-filter; exact predicates are still evaluated
//! on the candidate set for correctness.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::amorphic_adapter::AmorphicTableStorage;

// ==================== GridIndex ====================

/// Simple grid-based spatial index for point data.
/// Divides 2D space into cells of `cell_size` and stores points in cells.
pub struct GridIndex {
    cells: HashMap<(i64, i64), Vec<GridEntry>>,
    cell_size: f64,
    count: usize,
}

/// A point entry in the grid index.
#[derive(Clone, Debug)]
pub struct GridEntry {
    pub row_id: String,
    pub x: f64,
    pub y: f64,
}

impl GridIndex {
    /// Create a new grid index with the given cell size.
    /// Default cell_size of 1.0 degree ≈ ~111 km at the equator.
    pub fn new(cell_size: f64) -> Self {
        Self {
            cells: HashMap::new(),
            cell_size: if cell_size > 0.0 { cell_size } else { 1.0 },
            count: 0,
        }
    }

    fn cell_key(&self, x: f64, y: f64) -> (i64, i64) {
        let cx = (x / self.cell_size).floor() as i64;
        let cy = (y / self.cell_size).floor() as i64;
        (cx, cy)
    }

    /// Insert a point into the grid index.
    pub fn insert(&mut self, row_id: String, x: f64, y: f64) {
        let key = self.cell_key(x, y);
        self.cells
            .entry(key)
            .or_default()
            .push(GridEntry { row_id, x, y });
        self.count += 1;
    }

    /// Query all points within a bounding box.
    pub fn query_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<&GridEntry> {
        let min_cx = (min_x / self.cell_size).floor() as i64;
        let max_cx = (max_x / self.cell_size).floor() as i64;
        let min_cy = (min_y / self.cell_size).floor() as i64;
        let max_cy = (max_y / self.cell_size).floor() as i64;

        let mut results = Vec::new();
        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                if let Some(entries) = self.cells.get(&(cx, cy)) {
                    for entry in entries {
                        if entry.x >= min_x
                            && entry.x <= max_x
                            && entry.y >= min_y
                            && entry.y <= max_y
                        {
                            results.push(entry);
                        }
                    }
                }
            }
        }
        results
    }

    /// Query all points within a given radius of (cx, cy).
    /// Uses bounding-box pre-filter then exact distance check.
    pub fn query_radius(&self, cx: f64, cy: f64, radius: f64) -> Vec<&GridEntry> {
        let bbox_results = self.query_bbox(cx - radius, cy - radius, cx + radius, cy + radius);
        let r2 = radius * radius;
        bbox_results
            .into_iter()
            .filter(|e| {
                let dx = e.x - cx;
                let dy = e.y - cy;
                dx * dx + dy * dy <= r2
            })
            .collect()
    }

    /// Number of indexed points.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// ==================== SpatialIndexManager ====================

/// Manages live spatial grid indexes, analogous to VectorIndexManager.
pub struct SpatialIndexManager {
    indexes: RwLock<HashMap<String, SpatialIndexInfo>>,
}

struct SpatialIndexInfo {
    grid: GridIndex,
    pub table: String,
    pub column: String,
}

impl SpatialIndexManager {
    pub fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
        }
    }

    /// Build a spatial index from existing table data.
    /// `rows` is a list of (record_id, x, y) tuples parsed from geometry values.
    pub fn build_index(
        &self,
        name: &str,
        table: &str,
        column: &str,
        rows: Vec<(String, f64, f64)>,
    ) {
        // Auto-detect cell size from data spread
        let cell_size = if rows.len() >= 2 {
            let xs: Vec<f64> = rows.iter().map(|(_, x, _)| *x).collect();
            let ys: Vec<f64> = rows.iter().map(|(_, _, y)| *y).collect();
            let x_range = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let y_range = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - ys.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = x_range.max(y_range);
            // Target ~10-20 cells across the data range
            (range / 10.0).max(0.001)
        } else {
            1.0
        };

        let mut grid = GridIndex::new(cell_size);
        for (row_id, x, y) in rows {
            grid.insert(row_id, x, y);
        }

        let info = SpatialIndexInfo {
            grid,
            table: table.to_string(),
            column: column.to_string(),
        };
        crate::lock_util::write_lock(&self.indexes).insert(name.to_string(), info);
    }

    /// Insert a single point into an existing spatial index.
    pub fn insert_into_index(&self, index_name: &str, row_id: String, x: f64, y: f64) {
        if let Some(info) = crate::lock_util::write_lock(&self.indexes).get_mut(index_name) {
            info.grid.insert(row_id, x, y);
        }
    }

    /// Remove a spatial index.
    pub fn remove_index(&self, name: &str) -> bool {
        crate::lock_util::write_lock(&self.indexes)
            .remove(name)
            .is_some()
    }

    /// List all spatial indexes for a given table, returning (name, column) pairs.
    pub fn indexes_for_table(&self, table: &str) -> Vec<(String, String)> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        indexes
            .iter()
            .filter(|(_, info)| info.table == table)
            .map(|(name, info)| (name.clone(), info.column.clone()))
            .collect()
    }

    /// Find a spatial index for a given table and column.
    pub fn find_index_for(&self, table: &str, column: &str) -> Option<String> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        for (name, info) in indexes.iter() {
            if info.table == table && info.column == column {
                return Some(name.clone());
            }
        }
        None
    }

    /// Query a radius search on a named index.
    /// Returns the row_ids of points within the radius.
    pub fn query_radius(
        &self,
        index_name: &str,
        cx: f64,
        cy: f64,
        radius: f64,
    ) -> Option<Vec<String>> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        let info = indexes.get(index_name)?;
        let results = info.grid.query_radius(cx, cy, radius);
        Some(results.iter().map(|e| e.row_id.clone()).collect())
    }

    /// Query a bounding box on a named index.
    /// Returns the row_ids of points within the bbox.
    pub fn query_bbox(
        &self,
        index_name: &str,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Option<Vec<String>> {
        let indexes = crate::lock_util::read_lock(&self.indexes);
        let info = indexes.get(index_name)?;
        let results = info.grid.query_bbox(min_x, min_y, max_x, max_y);
        Some(results.iter().map(|e| e.row_id.clone()).collect())
    }

    /// Rebuild all spatial indexes from `__indexes__` metadata at startup.
    pub fn rebuild_from_metadata(&self, amorphic: &AmorphicTableStorage) {
        use joule_db_query::ast::Value;
        use joule_db_query::executor::TableStorage;

        // Scan __indexes__ meta-table for spatial index records
        let index_records = amorphic.scan("__indexes__").unwrap_or_default();

        let mut to_build: Vec<(String, String, String)> = Vec::new(); // (name, table, column)
        for row in &index_records {
            let get_str = |col: &str| -> Option<String> {
                let pos = row.columns.iter().position(|c| c == col)?;
                match row.values.get(pos)? {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                }
            };

            let idx_type = get_str("__index_type__");
            if idx_type.as_deref() != Some("spatial") {
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

            if !column.is_empty() {
                to_build.push((name, table, column));
            }
        }

        // Build each index from table data
        for (name, table, column) in to_build {
            if let Ok(rows_with_ids) = amorphic.scan_with_record_ids(&table) {
                let points: Vec<(String, f64, f64)> = rows_with_ids
                    .iter()
                    .filter_map(|(record_id, row)| {
                        let col_idx = row.columns.iter().position(|c| c == &column)?;
                        let val = row.values.get(col_idx)?;
                        parse_point_from_value(val).map(|(x, y)| (record_id.clone(), x, y))
                    })
                    .collect();
                self.build_index(&name, &table, &column, points);
            }
        }
    }
}

/// Try to parse (x, y) coordinates from a SQL Value.
/// Handles:
/// - `POINT(x y)` WKT format (stored as string)
/// - Numeric values interpreted as single-dimensional (returns (val, 0.0))
pub fn parse_point_from_value(val: &joule_db_query::ast::Value) -> Option<(f64, f64)> {
    match val {
        joule_db_query::ast::Value::String(s) => parse_point_wkt(s),
        joule_db_query::ast::Value::Float(f) => Some((*f, 0.0)),
        joule_db_query::ast::Value::Int(i) => Some((*i as f64, 0.0)),
        _ => None,
    }
}

/// Parse POINT(x y) WKT string.
pub fn parse_point_wkt(s: &str) -> Option<(f64, f64)> {
    let s = s.trim();
    let upper = s.to_uppercase();
    if upper.starts_with("POINT") {
        // POINT(x y) or POINT (x y)
        let paren_start = s.find('(')?;
        let paren_end = s.find(')')?;
        let inner = s[paren_start + 1..paren_end].trim();
        let parts: Vec<&str> = inner.split_whitespace().collect();
        if parts.len() >= 2 {
            let x = parts[0].parse::<f64>().ok()?;
            let y = parts[1].parse::<f64>().ok()?;
            return Some((x, y));
        }
    }
    None
}

// ==================== Unit Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_index_basic() {
        let mut grid = GridIndex::new(1.0);
        grid.insert("a".into(), 0.5, 0.5);
        grid.insert("b".into(), 1.5, 1.5);
        grid.insert("c".into(), 10.0, 10.0);
        assert_eq!(grid.len(), 3);
    }

    #[test]
    fn test_grid_index_bbox_query() {
        let mut grid = GridIndex::new(1.0);
        grid.insert("a".into(), 0.5, 0.5);
        grid.insert("b".into(), 1.5, 1.5);
        grid.insert("c".into(), 10.0, 10.0);

        let results = grid.query_bbox(0.0, 0.0, 2.0, 2.0);
        let ids: Vec<&str> = results.iter().map(|e| e.row_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(!ids.contains(&"c"));
    }

    #[test]
    fn test_grid_index_radius_query() {
        let mut grid = GridIndex::new(1.0);
        grid.insert("center".into(), 0.0, 0.0);
        grid.insert("near".into(), 0.5, 0.5);
        grid.insert("far".into(), 5.0, 5.0);

        // radius 1.0 from origin: center (0,0) and near (0.5,0.5) should be included
        // distance of near = sqrt(0.25+0.25) = ~0.707
        let results = grid.query_radius(0.0, 0.0, 1.0);
        let ids: Vec<&str> = results.iter().map(|e| e.row_id.as_str()).collect();
        assert!(ids.contains(&"center"));
        assert!(ids.contains(&"near"));
        assert!(!ids.contains(&"far"));
    }

    #[test]
    fn test_grid_index_no_false_negatives() {
        // Insert points on cell boundaries to verify no points are missed
        let mut grid = GridIndex::new(1.0);
        for i in 0..10 {
            for j in 0..10 {
                let x = i as f64;
                let y = j as f64;
                grid.insert(format!("{},{}", i, j), x, y);
            }
        }

        // Query the entire range — should get all 100 points
        let results = grid.query_bbox(0.0, 0.0, 9.0, 9.0);
        assert_eq!(results.len(), 100);
    }

    #[test]
    fn test_grid_index_empty() {
        let grid = GridIndex::new(1.0);
        assert!(grid.is_empty());
        let results = grid.query_bbox(-1.0, -1.0, 1.0, 1.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_grid_index_negative_coords() {
        let mut grid = GridIndex::new(1.0);
        grid.insert("a".into(), -5.0, -3.0);
        grid.insert("b".into(), -4.5, -2.5);

        let results = grid.query_bbox(-6.0, -4.0, -4.0, -2.0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_parse_point_wkt() {
        assert_eq!(parse_point_wkt("POINT(1.5 2.5)"), Some((1.5, 2.5)));
        assert_eq!(parse_point_wkt("POINT (10 20)"), Some((10.0, 20.0)));
        assert_eq!(
            parse_point_wkt("point(-73.9857 40.7484)"),
            Some((-73.9857, 40.7484))
        );
        assert_eq!(parse_point_wkt("not a point"), None);
        assert_eq!(parse_point_wkt(""), None);
    }

    #[test]
    fn test_parse_point_from_value() {
        use joule_db_query::ast::Value;
        assert_eq!(
            parse_point_from_value(&Value::String("POINT(1.0 2.0)".into())),
            Some((1.0, 2.0))
        );
        assert_eq!(
            parse_point_from_value(&Value::Float(3.14)),
            Some((3.14, 0.0))
        );
        assert_eq!(parse_point_from_value(&Value::Int(42)), Some((42.0, 0.0)));
        assert_eq!(parse_point_from_value(&Value::Null), None);
    }

    #[test]
    fn test_spatial_index_manager_build_and_query() {
        let manager = SpatialIndexManager::new();
        let points = vec![
            ("r1".into(), 0.0, 0.0),
            ("r2".into(), 1.0, 1.0),
            ("r3".into(), 5.0, 5.0),
        ];
        manager.build_index("idx_geo", "locations", "geom", points);

        // Find index
        assert!(manager.find_index_for("locations", "geom").is_some());
        assert!(manager.find_index_for("locations", "other").is_none());

        // Radius query
        let results = manager.query_radius("idx_geo", 0.0, 0.0, 2.0).unwrap();
        assert!(results.contains(&"r1".to_string()));
        assert!(results.contains(&"r2".to_string()));
        assert!(!results.contains(&"r3".to_string()));
    }

    #[test]
    fn test_spatial_index_manager_insert_and_remove() {
        let manager = SpatialIndexManager::new();
        manager.build_index("idx", "t", "geom", vec![]);

        manager.insert_into_index("idx", "r1".into(), 1.0, 2.0);
        let results = manager.query_radius("idx", 1.0, 2.0, 0.5).unwrap();
        assert_eq!(results.len(), 1);

        assert!(manager.remove_index("idx"));
        assert!(!manager.remove_index("idx")); // already removed
        assert!(manager.find_index_for("t", "geom").is_none());
    }
}
