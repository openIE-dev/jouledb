//! Map algebra: local ops (add/subtract/multiply/divide grids), focal ops
//! (neighborhood stats: mean/max/min/std), zonal stats, conditional reclassify,
//! and weighted overlay.

use std::fmt;

// ── Error ───────────────────────────────────────────────────────

/// Errors from raster algebra operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AlgebraError {
    DimensionMismatch { a: (usize, usize), b: (usize, usize) },
    DivisionByZero { row: usize, col: usize },
    EmptyGrid,
    InvalidWeights,
    NoZones,
}

impl fmt::Display for AlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlgebraError::DimensionMismatch { a, b } => {
                write!(f, "dimension mismatch: {}x{} vs {}x{}", a.0, a.1, b.0, b.1)
            }
            AlgebraError::DivisionByZero { row, col } => {
                write!(f, "division by zero at ({row}, {col})")
            }
            AlgebraError::EmptyGrid => write!(f, "empty grid"),
            AlgebraError::InvalidWeights => write!(f, "invalid weights"),
            AlgebraError::NoZones => write!(f, "no zones found"),
        }
    }
}

// ── Grid (local representation) ─────────────────────────────────

/// Lightweight grid for algebra operations (row-major f64).
#[derive(Debug, Clone)]
pub struct AlgGrid {
    pub rows: usize,
    pub cols: usize,
    pub nodata: f64,
    data: Vec<f64>,
}

impl AlgGrid {
    pub fn new(rows: usize, cols: usize, nodata: f64) -> Self {
        Self { rows, cols, nodata, data: vec![nodata; rows * cols] }
    }

    pub fn from_data(rows: usize, cols: usize, nodata: f64, data: Vec<f64>) -> Option<Self> {
        if data.len() != rows * cols { return None; }
        Some(Self { rows, cols, nodata, data })
    }

    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Self { rows, cols, nodata: -9999.0, data: vec![value; rows * cols] }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn is_nodata(&self, v: f64) -> bool {
        (v - self.nodata).abs() < f64::EPSILON
    }

    pub fn is_valid(&self, r: usize, c: usize) -> bool {
        !self.is_nodata(self.get(r, c))
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn check_dims(&self, other: &AlgGrid) -> Result<(), AlgebraError> {
        if self.dims() != other.dims() {
            Err(AlgebraError::DimensionMismatch { a: self.dims(), b: other.dims() })
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for AlgGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let valid = self.data.iter().filter(|v| !self.is_nodata(**v)).count();
        write!(f, "AlgGrid({}x{}, valid={}/{})", self.rows, self.cols, valid, self.data.len())
    }
}

// ── Local Operations ────────────────────────────────────────────

/// Local (cell-by-cell) operation type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl fmt::Display for LocalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocalOp::Add => write!(f, "Add"),
            LocalOp::Subtract => write!(f, "Subtract"),
            LocalOp::Multiply => write!(f, "Multiply"),
            LocalOp::Divide => write!(f, "Divide"),
        }
    }
}

/// Apply a local operation between two grids cell by cell.
pub fn local_op(a: &AlgGrid, b: &AlgGrid, op: LocalOp) -> Result<AlgGrid, AlgebraError> {
    a.check_dims(b)?;
    let mut out = AlgGrid::new(a.rows, a.cols, a.nodata);
    for r in 0..a.rows {
        for c in 0..a.cols {
            let va = a.get(r, c);
            let vb = b.get(r, c);
            if a.is_nodata(va) || b.is_nodata(vb) {
                continue; // stays nodata
            }
            let result = match op {
                LocalOp::Add => va + vb,
                LocalOp::Subtract => va - vb,
                LocalOp::Multiply => va * vb,
                LocalOp::Divide => {
                    if vb.abs() < f64::EPSILON {
                        return Err(AlgebraError::DivisionByZero { row: r, col: c });
                    }
                    va / vb
                }
            };
            out.set(r, c, result);
        }
    }
    Ok(out)
}

/// Apply a scalar operation to every valid cell.
pub fn local_scalar(grid: &AlgGrid, scalar: f64, op: LocalOp) -> AlgGrid {
    let mut out = AlgGrid::new(grid.rows, grid.cols, grid.nodata);
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            let v = grid.get(r, c);
            if grid.is_nodata(v) { continue; }
            let result = match op {
                LocalOp::Add => v + scalar,
                LocalOp::Subtract => v - scalar,
                LocalOp::Multiply => v * scalar,
                LocalOp::Divide => if scalar.abs() < f64::EPSILON { grid.nodata } else { v / scalar },
            };
            out.set(r, c, result);
        }
    }
    out
}

// ── Focal Operations ────────────────────────────────────────────

/// Focal (neighborhood) statistic type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FocalStat {
    Mean,
    Max,
    Min,
    StdDev,
    Sum,
    Count,
}

impl fmt::Display for FocalStat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FocalStat::Mean => write!(f, "Mean"),
            FocalStat::Max => write!(f, "Max"),
            FocalStat::Min => write!(f, "Min"),
            FocalStat::StdDev => write!(f, "StdDev"),
            FocalStat::Sum => write!(f, "Sum"),
            FocalStat::Count => write!(f, "Count"),
        }
    }
}

/// Apply a focal statistic with a square neighborhood of given radius.
pub fn focal(grid: &AlgGrid, radius: usize, stat: FocalStat) -> AlgGrid {
    let mut out = AlgGrid::new(grid.rows, grid.cols, grid.nodata);
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            if !grid.is_valid(r, c) { continue; }
            let mut vals = Vec::new();
            let r_start = r.saturating_sub(radius);
            let r_end = (r + radius + 1).min(grid.rows);
            let c_start = c.saturating_sub(radius);
            let c_end = (c + radius + 1).min(grid.cols);
            for nr in r_start..r_end {
                for nc in c_start..c_end {
                    if grid.is_valid(nr, nc) {
                        vals.push(grid.get(nr, nc));
                    }
                }
            }
            if vals.is_empty() { continue; }
            let result = match stat {
                FocalStat::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
                FocalStat::Max => vals.iter().cloned().fold(f64::MIN, f64::max),
                FocalStat::Min => vals.iter().cloned().fold(f64::MAX, f64::min),
                FocalStat::Sum => vals.iter().sum(),
                FocalStat::Count => vals.len() as f64,
                FocalStat::StdDev => {
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                    var.sqrt()
                }
            };
            out.set(r, c, result);
        }
    }
    out
}

// ── Zonal Statistics ────────────────────────────────────────────

/// Result of zonal statistics for one zone.
#[derive(Debug, Clone)]
pub struct ZonalResult {
    pub zone_id: i64,
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
}

impl fmt::Display for ZonalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Zone {}: n={}, mean={:.4}, min={:.4}, max={:.4}",
            self.zone_id, self.count, self.mean, self.min, self.max,
        )
    }
}

/// Compute zonal statistics: the `zones` grid contains integer zone IDs,
/// `values` grid contains the data to summarize.
pub fn zonal_stats(zones: &AlgGrid, values: &AlgGrid) -> Result<Vec<ZonalResult>, AlgebraError> {
    zones.check_dims(values)?;
    let mut zone_map: std::collections::BTreeMap<i64, Vec<f64>> = std::collections::BTreeMap::new();
    for r in 0..zones.rows {
        for c in 0..zones.cols {
            if !zones.is_valid(r, c) || !values.is_valid(r, c) { continue; }
            let zid = zones.get(r, c) as i64;
            zone_map.entry(zid).or_default().push(values.get(r, c));
        }
    }
    if zone_map.is_empty() {
        return Err(AlgebraError::NoZones);
    }
    let results = zone_map.into_iter().map(|(zid, vals)| {
        let count = vals.len();
        let sum: f64 = vals.iter().sum();
        let mean = sum / count as f64;
        let min = vals.iter().cloned().fold(f64::MAX, f64::min);
        let max = vals.iter().cloned().fold(f64::MIN, f64::max);
        ZonalResult { zone_id: zid, count, sum, mean, min, max }
    }).collect();
    Ok(results)
}

// ── Conditional Reclassify ──────────────────────────────────────

/// A reclassification rule: values in [min, max) map to `new_value`.
#[derive(Debug, Clone)]
pub struct ReclassRule {
    pub min: f64,
    pub max: f64,
    pub new_value: f64,
}

/// Reclassify grid values using a set of rules. Unmatched cells become nodata.
pub fn reclassify(grid: &AlgGrid, rules: &[ReclassRule]) -> AlgGrid {
    let mut out = AlgGrid::new(grid.rows, grid.cols, grid.nodata);
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            let v = grid.get(r, c);
            if grid.is_nodata(v) { continue; }
            for rule in rules {
                if v >= rule.min && v < rule.max {
                    out.set(r, c, rule.new_value);
                    break;
                }
            }
        }
    }
    out
}

// ── Weighted Overlay ────────────────────────────────────────────

/// Compute a weighted overlay (suitability model) from multiple grids.
/// Each grid is paired with a weight. Weights are normalized to sum to 1.
pub fn weighted_overlay(layers: &[(&AlgGrid, f64)]) -> Result<AlgGrid, AlgebraError> {
    if layers.is_empty() {
        return Err(AlgebraError::EmptyGrid);
    }
    let first = layers[0].0;
    for (g, _) in &layers[1..] {
        first.check_dims(g)?;
    }
    let wsum: f64 = layers.iter().map(|(_, w)| w).sum();
    if wsum.abs() < f64::EPSILON {
        return Err(AlgebraError::InvalidWeights);
    }
    let norm: Vec<f64> = layers.iter().map(|(_, w)| w / wsum).collect();
    let mut out = AlgGrid::new(first.rows, first.cols, first.nodata);
    for r in 0..first.rows {
        for c in 0..first.cols {
            let mut val = 0.0;
            let mut all_valid = true;
            for (i, (g, _)) in layers.iter().enumerate() {
                let v = g.get(r, c);
                if g.is_nodata(v) {
                    all_valid = false;
                    break;
                }
                val += v * norm[i];
            }
            if all_valid {
                out.set(r, c, val);
            }
        }
    }
    Ok(out)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn grid2x2(a: f64, b: f64, c: f64, d: f64) -> AlgGrid {
        AlgGrid::from_data(2, 2, -9999.0, vec![a, b, c, d]).unwrap()
    }

    #[test]
    fn test_local_add() {
        let a = grid2x2(1.0, 2.0, 3.0, 4.0);
        let b = grid2x2(10.0, 20.0, 30.0, 40.0);
        let r = local_op(&a, &b, LocalOp::Add).unwrap();
        assert!((r.get(0, 0) - 11.0).abs() < 1e-9);
        assert!((r.get(1, 1) - 44.0).abs() < 1e-9);
    }

    #[test]
    fn test_local_subtract() {
        let a = grid2x2(10.0, 20.0, 30.0, 40.0);
        let b = grid2x2(1.0, 2.0, 3.0, 4.0);
        let r = local_op(&a, &b, LocalOp::Subtract).unwrap();
        assert!((r.get(0, 0) - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_local_multiply() {
        let a = grid2x2(2.0, 3.0, 4.0, 5.0);
        let b = grid2x2(10.0, 10.0, 10.0, 10.0);
        let r = local_op(&a, &b, LocalOp::Multiply).unwrap();
        assert!((r.get(0, 0) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_local_divide() {
        let a = grid2x2(10.0, 20.0, 30.0, 40.0);
        let b = grid2x2(2.0, 4.0, 5.0, 8.0);
        let r = local_op(&a, &b, LocalOp::Divide).unwrap();
        assert!((r.get(0, 0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_local_divide_by_zero() {
        let a = grid2x2(1.0, 2.0, 3.0, 4.0);
        let b = grid2x2(0.0, 1.0, 1.0, 1.0);
        let r = local_op(&a, &b, LocalOp::Divide);
        assert!(matches!(r, Err(AlgebraError::DivisionByZero { .. })));
    }

    #[test]
    fn test_local_dimension_mismatch() {
        let a = AlgGrid::filled(2, 2, 1.0);
        let b = AlgGrid::filled(3, 3, 1.0);
        let r = local_op(&a, &b, LocalOp::Add);
        assert!(matches!(r, Err(AlgebraError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_local_scalar_add() {
        let g = grid2x2(1.0, 2.0, 3.0, 4.0);
        let r = local_scalar(&g, 100.0, LocalOp::Add);
        assert!((r.get(0, 0) - 101.0).abs() < 1e-9);
    }

    #[test]
    fn test_local_nodata_propagation() {
        let mut a = grid2x2(1.0, 2.0, 3.0, 4.0);
        a.set(0, 0, a.nodata);
        let b = grid2x2(10.0, 20.0, 30.0, 40.0);
        let r = local_op(&a, &b, LocalOp::Add).unwrap();
        assert!(r.is_nodata(r.get(0, 0)));
        assert!((r.get(0, 1) - 22.0).abs() < 1e-9);
    }

    #[test]
    fn test_focal_mean() {
        let g = AlgGrid::from_data(3, 3, -9999.0, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let r = focal(&g, 1, FocalStat::Mean);
        // center cell has all 9 neighbors -> mean = 5.0
        assert!((r.get(1, 1) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_focal_max() {
        let g = AlgGrid::from_data(3, 3, -9999.0,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let r = focal(&g, 1, FocalStat::Max);
        assert!((r.get(1, 1) - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_focal_min() {
        let g = AlgGrid::from_data(3, 3, -9999.0,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let r = focal(&g, 1, FocalStat::Min);
        assert!((r.get(1, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_focal_stddev() {
        let g = AlgGrid::from_data(3, 3, -9999.0,
            vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]).unwrap();
        let r = focal(&g, 1, FocalStat::StdDev);
        assert!(r.get(1, 1).abs() < 1e-9); // all same -> stddev = 0
    }

    #[test]
    fn test_zonal_stats() {
        let zones = AlgGrid::from_data(2, 2, -9999.0, vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let values = AlgGrid::from_data(2, 2, -9999.0, vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let stats = zonal_stats(&zones, &values).unwrap();
        assert_eq!(stats.len(), 2);
        let z1 = &stats[0];
        assert_eq!(z1.zone_id, 1);
        assert_eq!(z1.count, 2);
        assert!((z1.mean - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_reclassify() {
        let g = AlgGrid::from_data(2, 2, -9999.0, vec![5.0, 15.0, 25.0, 35.0]).unwrap();
        let rules = vec![
            ReclassRule { min: 0.0, max: 10.0, new_value: 1.0 },
            ReclassRule { min: 10.0, max: 20.0, new_value: 2.0 },
            ReclassRule { min: 20.0, max: 30.0, new_value: 3.0 },
            ReclassRule { min: 30.0, max: 40.0, new_value: 4.0 },
        ];
        let r = reclassify(&g, &rules);
        assert!((r.get(0, 0) - 1.0).abs() < 1e-9);
        assert!((r.get(0, 1) - 2.0).abs() < 1e-9);
        assert!((r.get(1, 0) - 3.0).abs() < 1e-9);
        assert!((r.get(1, 1) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_reclassify_unmatched_becomes_nodata() {
        let g = AlgGrid::from_data(1, 1, -9999.0, vec![100.0]).unwrap();
        let rules = vec![ReclassRule { min: 0.0, max: 10.0, new_value: 1.0 }];
        let r = reclassify(&g, &rules);
        assert!(r.is_nodata(r.get(0, 0)));
    }

    #[test]
    fn test_weighted_overlay() {
        let a = grid2x2(1.0, 2.0, 3.0, 4.0);
        let b = grid2x2(10.0, 20.0, 30.0, 40.0);
        let r = weighted_overlay(&[(&a, 0.5), (&b, 0.5)]).unwrap();
        // (1*0.5 + 10*0.5) = 5.5
        assert!((r.get(0, 0) - 5.5).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_overlay_normalization() {
        let a = grid2x2(10.0, 10.0, 10.0, 10.0);
        let b = grid2x2(20.0, 20.0, 20.0, 20.0);
        // weights 2:8 => normalized 0.2:0.8
        let r = weighted_overlay(&[(&a, 2.0), (&b, 8.0)]).unwrap();
        // 10*0.2 + 20*0.8 = 2 + 16 = 18
        assert!((r.get(0, 0) - 18.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_overlay_empty() {
        let r = weighted_overlay(&[]);
        assert!(matches!(r, Err(AlgebraError::EmptyGrid)));
    }

    #[test]
    fn test_display_formats() {
        assert_eq!(format!("{}", LocalOp::Add), "Add");
        assert_eq!(format!("{}", FocalStat::Mean), "Mean");
        let g = AlgGrid::filled(2, 2, 1.0);
        assert!(format!("{g}").contains("2x2"));
    }
}
