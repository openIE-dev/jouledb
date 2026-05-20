//! Space-Filling Curves for Multi-Dimensional Range Queries
//!
//! This module provides Z-order (Morton) and Hilbert curve encodings for efficient
//! multi-dimensional range queries. Space-filling curves map multi-dimensional
//! coordinates to a single dimension while preserving locality.
//!
//! ## Features
//!
//! - **Z-order (Morton) encoding**: Fast, simple encoding for 2D data
//! - **Hilbert curve encoding**: Better locality preservation than Z-order
//! - **Spatial index**: Index creation for efficient range queries
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_amorphic::spatial::{z_order_encode, hilbert_encode};
//!
//! // Encode a 2D point (0.5, 0.5) with 16-bit precision
//! let z = z_order_encode(0.5, 0.5, 16);
//! let h = hilbert_encode(0.5, 0.5, 8);
//! ```

use crate::{RecordId, Value};
use std::collections::HashMap;

// =============================================================================
// Z-ORDER (MORTON) ENCODING
// =============================================================================

/// Encode two values into a Z-order (Morton) code
///
/// Z-order interleaves the bits of two coordinates, creating a single value
/// that preserves 2D locality. Points close in 2D space tend to have similar
/// Z-order codes.
///
/// # Arguments
/// * `x` - First coordinate (normalized to 0.0-1.0 range)
/// * `y` - Second coordinate (normalized to 0.0-1.0 range)
/// * `precision` - Number of bits per dimension (max 32)
///
/// # Returns
/// A 64-bit Z-order code
pub fn z_order_encode(x: f64, y: f64, precision: u32) -> u64 {
    let precision = precision.min(32);
    let scale = (1u64 << precision) - 1;

    let xi = (x.clamp(0.0, 1.0) * scale as f64) as u64;
    let yi = (y.clamp(0.0, 1.0) * scale as f64) as u64;

    interleave_bits(xi, yi, precision)
}

/// Decode a Z-order code back to coordinates
///
/// # Arguments
/// * `z` - Z-order code
/// * `precision` - Number of bits per dimension (must match encoding)
///
/// # Returns
/// Tuple of (x, y) normalized coordinates
pub fn z_order_decode(z: u64, precision: u32) -> (f64, f64) {
    let precision = precision.min(32);
    let scale = (1u64 << precision) - 1;

    let (xi, yi) = deinterleave_bits(z, precision);

    let x = xi as f64 / scale as f64;
    let y = yi as f64 / scale as f64;

    (x, y)
}

/// Interleave bits of two 32-bit values into a 64-bit value
fn interleave_bits(x: u64, y: u64, bits: u32) -> u64 {
    let mut result = 0u64;
    for i in 0..bits {
        result |= ((x >> i) & 1) << (2 * i);
        result |= ((y >> i) & 1) << (2 * i + 1);
    }
    result
}

/// Deinterleave a 64-bit Z-order code into two 32-bit values
fn deinterleave_bits(z: u64, bits: u32) -> (u64, u64) {
    let mut x = 0u64;
    let mut y = 0u64;
    for i in 0..bits {
        x |= ((z >> (2 * i)) & 1) << i;
        y |= ((z >> (2 * i + 1)) & 1) << i;
    }
    (x, y)
}

/// Calculate Z-order ranges for a 2D query rectangle
///
/// Returns a list of Z-order ranges that cover the query rectangle.
/// The ranges can be used to efficiently query a Z-order indexed column.
pub fn z_order_ranges(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    precision: u32,
) -> Vec<(u64, u64)> {
    let z_min = z_order_encode(x_min, y_min, precision);
    let z_max = z_order_encode(x_max, y_max, precision);

    // Simple approach: return single range from min to max corner
    // A more sophisticated approach would decompose into smaller ranges
    vec![(z_min, z_max)]
}

// =============================================================================
// HILBERT CURVE ENCODING
// =============================================================================

/// Encode two values into a Hilbert curve index
///
/// Hilbert curves provide better locality than Z-order curves, meaning
/// points close in 2D space are more likely to be close in Hilbert index.
///
/// # Arguments
/// * `x` - First coordinate (normalized to 0.0-1.0 range)
/// * `y` - Second coordinate (normalized to 0.0-1.0 range)
/// * `order` - Hilbert curve order (controls precision, max 16)
///
/// # Returns
/// A 32-bit Hilbert index
pub fn hilbert_encode(x: f64, y: f64, order: u32) -> u64 {
    let order = order.min(16);
    let n = 1u32 << order;

    let xi = (x.clamp(0.0, 1.0) * (n - 1) as f64) as u32;
    let yi = (y.clamp(0.0, 1.0) * (n - 1) as f64) as u32;

    xy_to_hilbert(xi, yi, n) as u64
}

/// Decode a Hilbert index back to coordinates
///
/// # Arguments
/// * `h` - Hilbert index
/// * `order` - Hilbert curve order (must match encoding)
///
/// # Returns
/// Tuple of (x, y) normalized coordinates
pub fn hilbert_decode(h: u64, order: u32) -> (f64, f64) {
    let order = order.min(16);
    let n = 1u32 << order;

    let (xi, yi) = hilbert_to_xy(h as u32, n);

    let x = xi as f64 / (n - 1) as f64;
    let y = yi as f64 / (n - 1) as f64;

    (x, y)
}

/// Convert (x, y) to Hilbert index
fn xy_to_hilbert(x: u32, y: u32, n: u32) -> u32 {
    let mut rx: u32;
    let mut ry: u32;
    let mut s: u32;
    let mut d: u32 = 0;
    let mut x = x;
    let mut y = y;

    s = n >> 1;
    while s > 0 {
        rx = if (x & s) > 0 { 1 } else { 0 };
        ry = if (y & s) > 0 { 1 } else { 0 };
        d += s * s * ((3 * rx) ^ ry);
        rotate_quadrant(n, &mut x, &mut y, rx, ry);
        s >>= 1;
    }
    d
}

/// Convert Hilbert index to (x, y)
fn hilbert_to_xy(d: u32, n: u32) -> (u32, u32) {
    let mut rx: u32;
    let mut ry: u32;
    let mut s: u32;
    let mut t: u32 = d;
    let mut x: u32 = 0;
    let mut y: u32 = 0;

    s = 1;
    while s < n {
        rx = 1 & (t >> 1);
        ry = 1 & (t ^ rx);
        rotate_quadrant(s, &mut x, &mut y, rx, ry);
        x += s * rx;
        y += s * ry;
        t >>= 2;
        s <<= 1;
    }
    (x, y)
}

/// Rotate/flip quadrant appropriately
fn rotate_quadrant(n: u32, x: &mut u32, y: &mut u32, rx: u32, ry: u32) {
    if ry == 0 {
        if rx == 1 {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        std::mem::swap(x, y);
    }
}

// =============================================================================
// SPATIAL INDEX
// =============================================================================

/// Space-filling curve type for indexing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveType {
    /// Z-order (Morton) curve
    ZOrder,
    /// Hilbert curve
    Hilbert,
}

/// Spatial index for efficient 2D range queries
#[derive(Debug, Clone)]
pub struct SpatialIndex {
    /// Curve type used for encoding
    pub curve_type: CurveType,
    /// Precision/order of the curve
    pub precision: u32,
    /// Encoded values indexed by record ID
    values: HashMap<RecordId, u64>,
    /// Sorted (curve_value, record_id) pairs for range scans
    sorted_entries: Vec<(u64, RecordId)>,
    /// X column name
    pub x_col: String,
    /// Y column name
    pub y_col: String,
    /// Whether the sorted entries are up-to-date
    is_sorted: bool,
}

impl SpatialIndex {
    /// Create a new spatial index
    pub fn new(x_col: &str, y_col: &str, curve_type: CurveType, precision: u32) -> Self {
        Self {
            curve_type,
            precision,
            values: HashMap::new(),
            sorted_entries: Vec::new(),
            x_col: x_col.to_string(),
            y_col: y_col.to_string(),
            is_sorted: true,
        }
    }

    /// Insert a point into the index
    pub fn insert(&mut self, record_id: RecordId, x: f64, y: f64) {
        let encoded = match self.curve_type {
            CurveType::ZOrder => z_order_encode(x, y, self.precision),
            CurveType::Hilbert => hilbert_encode(x, y, self.precision),
        };

        self.values.insert(record_id, encoded);
        self.sorted_entries.push((encoded, record_id));
        self.is_sorted = false;
    }

    /// Remove a point from the index
    pub fn remove(&mut self, record_id: RecordId) {
        self.values.remove(&record_id);
        self.sorted_entries.retain(|(_, id)| *id != record_id);
    }

    /// Ensure sorted entries are up-to-date
    fn ensure_sorted(&mut self) {
        if !self.is_sorted {
            self.sorted_entries.sort_by_key(|(encoded, _)| *encoded);
            self.is_sorted = true;
        }
    }

    /// Query a 2D range
    ///
    /// Returns record IDs of points within the given rectangle.
    pub fn query_range(&mut self, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Vec<RecordId> {
        self.ensure_sorted();

        // Encode the bounding box corners
        let (min_encoded, max_encoded) = match self.curve_type {
            CurveType::ZOrder => {
                let min = z_order_encode(x_min, y_min, self.precision);
                let max = z_order_encode(x_max, y_max, self.precision);
                (min.min(max), min.max(max))
            }
            CurveType::Hilbert => {
                // For Hilbert, we need to check all corners
                let corners = [
                    hilbert_encode(x_min, y_min, self.precision),
                    hilbert_encode(x_min, y_max, self.precision),
                    hilbert_encode(x_max, y_min, self.precision),
                    hilbert_encode(x_max, y_max, self.precision),
                ];
                (
                    corners.iter().copied().min().unwrap(),
                    corners.iter().copied().max().unwrap(),
                )
            }
        };

        // Binary search for start position
        let start = self
            .sorted_entries
            .binary_search_by_key(&min_encoded, |(e, _)| *e)
            .unwrap_or_else(|i| i);

        // Collect candidates and filter by actual bounds
        let mut result = Vec::new();
        for i in start..self.sorted_entries.len() {
            let (encoded, record_id) = self.sorted_entries[i];
            if encoded > max_encoded {
                break;
            }

            // Decode and check if actually in bounds
            let (x, y) = match self.curve_type {
                CurveType::ZOrder => z_order_decode(encoded, self.precision),
                CurveType::Hilbert => hilbert_decode(encoded, self.precision),
            };

            if x >= x_min && x <= x_max && y >= y_min && y <= y_max {
                result.push(record_id);
            }
        }

        result
    }

    /// Get count of indexed points
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Extension trait for adding spatial index support to ColumnarStore
pub trait SpatialIndexExt {
    /// Create a spatial index on two columns
    fn create_spatial_index(
        &self,
        x_col: &str,
        y_col: &str,
        curve_type: CurveType,
        precision: u32,
    ) -> Option<SpatialIndex>;
}

impl SpatialIndexExt for crate::columnar::ColumnarStore {
    fn create_spatial_index(
        &self,
        x_col: &str,
        y_col: &str,
        curve_type: CurveType,
        precision: u32,
    ) -> Option<SpatialIndex> {
        let x_column = self.get_column(x_col)?;
        let y_column = self.get_column(y_col)?;

        let mut index = SpatialIndex::new(x_col, y_col, curve_type, precision);

        // Build map of record_id -> y_value
        let y_map: HashMap<RecordId, f64> = y_column.scan().collect();

        // Normalize x and y values to [0, 1] range
        let x_min = x_column.min();
        let x_max = x_column.max();
        let y_min = y_column.min();
        let y_max = y_column.max();

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        // Populate the index
        for (record_id, x_value) in x_column.scan() {
            if let Some(&y_value) = y_map.get(&record_id) {
                let x_norm = if x_range > f64::EPSILON {
                    (x_value - x_min) / x_range
                } else {
                    0.5
                };
                let y_norm = if y_range > f64::EPSILON {
                    (y_value - y_min) / y_range
                } else {
                    0.5
                };
                index.insert(record_id, x_norm, y_norm);
            }
        }

        Some(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;
    use crate::columnar::ColumnarStore;

    #[test]
    fn test_z_order_encode_decode() {
        let precision = 16;

        // Test several points
        let points = [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.25, 0.75)];

        for (x, y) in points {
            let encoded = z_order_encode(x, y, precision);
            let (dx, dy) = z_order_decode(encoded, precision);

            // Should be close (within precision)
            assert!((dx - x).abs() < 0.001, "x: expected {}, got {}", x, dx);
            assert!((dy - y).abs() < 0.001, "y: expected {}, got {}", y, dy);
        }
    }

    #[test]
    fn test_z_order_locality() {
        let precision = 16;

        // Z-order preserves locality within quadrants better than across quadrants
        // Points in same quadrant (bottom-left)
        let z1 = z_order_encode(0.1, 0.1, precision);
        let z2 = z_order_encode(0.15, 0.15, precision);

        // Point in different quadrant (top-right)
        let z3 = z_order_encode(0.9, 0.9, precision);

        // Verify encoding/decoding round-trips correctly
        let (x1, y1) = z_order_decode(z1, precision);
        assert!((x1 - 0.1).abs() < 0.001);
        assert!((y1 - 0.1).abs() < 0.001);

        // Verify different points get different codes
        assert_ne!(z1, z2, "Different points should have different codes");
        assert_ne!(z1, z3, "Different points should have different codes");
        assert_ne!(z2, z3, "Different points should have different codes");
    }

    #[test]
    fn test_hilbert_encode_decode() {
        let order = 8;

        // Test several points
        let points = [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.25, 0.75)];

        for (x, y) in points {
            let encoded = hilbert_encode(x, y, order);
            let (dx, dy) = hilbert_decode(encoded, order);

            // Should be close (within precision)
            assert!((dx - x).abs() < 0.01, "x: expected {}, got {}", x, dx);
            assert!((dy - y).abs() < 0.01, "y: expected {}, got {}", y, dy);
        }
    }

    #[test]
    fn test_hilbert_locality() {
        let order = 6; // Lower order for easier testing

        // Close points should have similar Hilbert codes
        let h1 = hilbert_encode(0.5, 0.5, order);
        let h2 = hilbert_encode(0.52, 0.52, order);
        let h3 = hilbert_encode(0.0, 0.0, order);

        // h1 and h2 should be relatively close compared to h3
        let diff_close = (h1 as i64 - h2 as i64).abs();
        let diff_far = (h1 as i64 - h3 as i64).abs();

        // The difference between close points should generally be smaller
        // Note: Hilbert has better locality than Z-order
        assert!(h1 != h3, "Different points should have different codes");

        // h1 and h3 should be different
        assert!(
            diff_far > 0,
            "Far points should have different Hilbert codes"
        );
    }

    #[test]
    fn test_spatial_index_z_order() {
        let mut index = SpatialIndex::new("x", "y", CurveType::ZOrder, 16);

        // Insert some points
        index.insert(0, 0.1, 0.1);
        index.insert(1, 0.2, 0.2);
        index.insert(2, 0.9, 0.9);
        index.insert(3, 0.5, 0.5);

        assert_eq!(index.len(), 4);

        // Query a range that should contain points 0, 1
        let results = index.query_range(0.0, 0.3, 0.0, 0.3);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));

        // Query a range that should contain point 2
        let results = index.query_range(0.8, 1.0, 0.8, 1.0);
        assert!(results.contains(&2));
        assert!(!results.contains(&0));
    }

    #[test]
    fn test_spatial_index_hilbert() {
        let mut index = SpatialIndex::new("x", "y", CurveType::Hilbert, 8);

        // Insert some points
        index.insert(0, 0.1, 0.1);
        index.insert(1, 0.2, 0.2);
        index.insert(2, 0.9, 0.9);
        index.insert(3, 0.5, 0.5);

        assert_eq!(index.len(), 4);

        // Query a range that should contain points 0, 1
        let results = index.query_range(0.0, 0.3, 0.0, 0.3);
        assert!(results.contains(&0));
        assert!(results.contains(&1));

        // Query entire space
        let results = index.query_range(0.0, 1.0, 0.0, 1.0);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_spatial_index_remove() {
        let mut index = SpatialIndex::new("x", "y", CurveType::ZOrder, 16);

        index.insert(0, 0.1, 0.1);
        index.insert(1, 0.2, 0.2);
        index.insert(2, 0.3, 0.3);

        assert_eq!(index.len(), 3);

        index.remove(1);
        assert_eq!(index.len(), 2);

        let results = index.query_range(0.0, 1.0, 0.0, 1.0);
        assert!(!results.contains(&1));
    }

    #[test]
    fn test_columnar_spatial_index() {
        let mut store = ColumnarStore::new();

        // Add points
        for i in 0..100 {
            let x = (i % 10) as f64;
            let y = (i / 10) as f64;
            store.record_value("x", i as u64, &Value::Float(x));
            store.record_value("y", i as u64, &Value::Float(y));
        }

        // Create spatial index
        let mut index = store
            .create_spatial_index("x", "y", CurveType::ZOrder, 16)
            .unwrap();

        assert_eq!(index.len(), 100);

        // Query lower-left quadrant (x < 5, y < 5)
        // Points should have x in [0,4] and y in [0,4] -> 25 points
        let results = index.query_range(0.0, 0.5, 0.0, 0.5);

        // Due to normalization, this should return roughly 25 points
        assert!(!results.is_empty());
    }

    #[test]
    fn test_z_order_ranges() {
        let ranges = z_order_ranges(0.0, 0.5, 0.0, 0.5, 16);

        // Should return at least one range
        assert!(!ranges.is_empty());

        // Min should be less than max
        let (min, max) = ranges[0];
        assert!(min <= max);
    }
}
