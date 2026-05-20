//! Raster grid: 2D grid with nodata handling, cell size/origin/CRS metadata,
//! pixel-to-coordinate transform, resampling (nearest/bilinear/cubic),
//! grid subsetting, and RasterConfig builder.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────

const DEFAULT_NODATA: f64 = -9999.0;

// ── CRS ─────────────────────────────────────────────────────────

/// Coordinate reference system identifier.
#[derive(Debug, Clone, PartialEq)]
pub enum Crs {
    /// EPSG code (e.g. 4326 for WGS84, 32610 for UTM 10N).
    Epsg(u32),
    /// Well-Known Text representation.
    Wkt(String),
    /// Unknown / unset.
    Unknown,
}

impl fmt::Display for Crs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Crs::Epsg(code) => write!(f, "EPSG:{code}"),
            Crs::Wkt(s) => write!(f, "WKT({} chars)", s.len()),
            Crs::Unknown => write!(f, "Unknown CRS"),
        }
    }
}

// ── Resampling ──────────────────────────────────────────────────

/// Resampling method for grid transformations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Resampling {
    Nearest,
    Bilinear,
    Cubic,
}

impl fmt::Display for Resampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Resampling::Nearest => write!(f, "Nearest"),
            Resampling::Bilinear => write!(f, "Bilinear"),
            Resampling::Cubic => write!(f, "Cubic"),
        }
    }
}

// ── RasterConfig ────────────────────────────────────────────────

/// Builder for raster grid configuration.
#[derive(Debug, Clone)]
pub struct RasterConfig {
    pub rows: usize,
    pub cols: usize,
    pub cell_width: f64,
    pub cell_height: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub nodata: f64,
    pub crs: Crs,
}

impl RasterConfig {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            cell_width: 1.0,
            cell_height: 1.0,
            origin_x: 0.0,
            origin_y: 0.0,
            nodata: DEFAULT_NODATA,
            crs: Crs::Unknown,
        }
    }

    pub fn with_cell_size(mut self, width: f64, height: f64) -> Self {
        self.cell_width = width;
        self.cell_height = height;
        self
    }

    pub fn with_origin(mut self, x: f64, y: f64) -> Self {
        self.origin_x = x;
        self.origin_y = y;
        self
    }

    pub fn with_nodata(mut self, nodata: f64) -> Self {
        self.nodata = nodata;
        self
    }

    pub fn with_crs(mut self, crs: Crs) -> Self {
        self.crs = crs;
        self
    }

    /// Build a `RasterGrid` filled with the nodata value.
    pub fn build(&self) -> RasterGrid {
        RasterGrid {
            rows: self.rows,
            cols: self.cols,
            cell_width: self.cell_width,
            cell_height: self.cell_height,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            nodata: self.nodata,
            crs: self.crs.clone(),
            data: vec![self.nodata; self.rows * self.cols],
        }
    }
}

impl fmt::Display for RasterConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RasterConfig({}x{}, cell={}x{}, origin=({}, {}), nodata={}, {})",
            self.rows, self.cols, self.cell_width, self.cell_height,
            self.origin_x, self.origin_y, self.nodata, self.crs,
        )
    }
}

// ── RasterGrid ──────────────────────────────────────────────────

/// A 2D raster grid with geospatial metadata and nodata handling.
#[derive(Debug, Clone)]
pub struct RasterGrid {
    pub rows: usize,
    pub cols: usize,
    pub cell_width: f64,
    pub cell_height: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub nodata: f64,
    pub crs: Crs,
    data: Vec<f64>,
}

impl RasterGrid {
    /// Create a grid filled with a constant value.
    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        RasterConfig::new(rows, cols).build_with_fill(value)
    }

    /// Create from raw data (row-major).
    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Option<Self> {
        if data.len() != rows * cols {
            return None;
        }
        let mut grid = RasterConfig::new(rows, cols).build();
        grid.data = data;
        Some(grid)
    }

    /// Total number of cells.
    pub fn cell_count(&self) -> usize {
        self.rows * self.cols
    }

    /// Get value at (row, col). Returns `None` for out-of-bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Set value at (row, col). Returns `false` if out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> bool {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = value;
            true
        } else {
            false
        }
    }

    /// Check if a value is nodata.
    pub fn is_nodata(&self, value: f64) -> bool {
        (value - self.nodata).abs() < f64::EPSILON
    }

    /// Check if cell at (row, col) is nodata.
    pub fn is_nodata_at(&self, row: usize, col: usize) -> bool {
        self.get(row, col).map_or(true, |v| self.is_nodata(v))
    }

    /// Count of valid (non-nodata) cells.
    pub fn valid_count(&self) -> usize {
        self.data.iter().filter(|v| !self.is_nodata(**v)).count()
    }

    /// Raw data slice.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Mutable raw data slice.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    // ── Coordinate transforms ───────────────────────────────────

    /// Convert pixel (row, col) to world coordinate (x, y) at cell center.
    pub fn pixel_to_coord(&self, row: usize, col: usize) -> (f64, f64) {
        let x = self.origin_x + (col as f64 + 0.5) * self.cell_width;
        let y = self.origin_y + (self.rows as f64 - row as f64 - 0.5) * self.cell_height;
        (x, y)
    }

    /// Convert world coordinate (x, y) to pixel (row, col).
    pub fn coord_to_pixel(&self, x: f64, y: f64) -> (usize, usize) {
        let col = ((x - self.origin_x) / self.cell_width).floor() as usize;
        let row = ((self.origin_y + self.rows as f64 * self.cell_height - y) / self.cell_height)
            .floor() as usize;
        (row.min(self.rows.saturating_sub(1)), col.min(self.cols.saturating_sub(1)))
    }

    /// Bounding box as (min_x, min_y, max_x, max_y).
    pub fn extent(&self) -> (f64, f64, f64, f64) {
        let min_x = self.origin_x;
        let min_y = self.origin_y;
        let max_x = self.origin_x + self.cols as f64 * self.cell_width;
        let max_y = self.origin_y + self.rows as f64 * self.cell_height;
        (min_x, min_y, max_x, max_y)
    }

    // ── Subsetting ──────────────────────────────────────────────

    /// Extract a rectangular sub-grid.
    pub fn subset(&self, start_row: usize, start_col: usize, nrows: usize, ncols: usize) -> Option<RasterGrid> {
        if start_row + nrows > self.rows || start_col + ncols > self.cols {
            return None;
        }
        let mut sub = RasterConfig::new(nrows, ncols)
            .with_cell_size(self.cell_width, self.cell_height)
            .with_origin(
                self.origin_x + start_col as f64 * self.cell_width,
                self.origin_y + (self.rows - start_row - nrows) as f64 * self.cell_height,
            )
            .with_nodata(self.nodata)
            .with_crs(self.crs.clone())
            .build();
        for r in 0..nrows {
            for c in 0..ncols {
                let v = self.data[(start_row + r) * self.cols + (start_col + c)];
                sub.data[r * ncols + c] = v;
            }
        }
        Some(sub)
    }

    // ── Resampling ──────────────────────────────────────────────

    /// Resample the grid to new dimensions using the specified method.
    pub fn resample(&self, new_rows: usize, new_cols: usize, method: Resampling) -> RasterGrid {
        let new_cell_w = (self.cols as f64 * self.cell_width) / new_cols as f64;
        let new_cell_h = (self.rows as f64 * self.cell_height) / new_rows as f64;
        let mut out = RasterConfig::new(new_rows, new_cols)
            .with_cell_size(new_cell_w, new_cell_h)
            .with_origin(self.origin_x, self.origin_y)
            .with_nodata(self.nodata)
            .with_crs(self.crs.clone())
            .build();

        let scale_r = self.rows as f64 / new_rows as f64;
        let scale_c = self.cols as f64 / new_cols as f64;

        for r in 0..new_rows {
            for c in 0..new_cols {
                let src_r = (r as f64 + 0.5) * scale_r - 0.5;
                let src_c = (c as f64 + 0.5) * scale_c - 0.5;
                let val = match method {
                    Resampling::Nearest => self.sample_nearest(src_r, src_c),
                    Resampling::Bilinear => self.sample_bilinear(src_r, src_c),
                    Resampling::Cubic => self.sample_cubic(src_r, src_c),
                };
                out.data[r * new_cols + c] = val;
            }
        }
        out
    }

    fn sample_nearest(&self, r: f64, c: f64) -> f64 {
        let ri = r.round() as isize;
        let ci = c.round() as isize;
        if ri >= 0 && ri < self.rows as isize && ci >= 0 && ci < self.cols as isize {
            self.data[ri as usize * self.cols + ci as usize]
        } else {
            self.nodata
        }
    }

    fn sample_bilinear(&self, r: f64, c: f64) -> f64 {
        let r0 = r.floor() as isize;
        let c0 = c.floor() as isize;
        let fr = r - r.floor();
        let fc = c - c.floor();

        let get = |ri: isize, ci: isize| -> f64 {
            if ri >= 0 && ri < self.rows as isize && ci >= 0 && ci < self.cols as isize {
                let v = self.data[ri as usize * self.cols + ci as usize];
                if self.is_nodata(v) { return f64::NAN; }
                v
            } else {
                f64::NAN
            }
        };

        let v00 = get(r0, c0);
        let v01 = get(r0, c0 + 1);
        let v10 = get(r0 + 1, c0);
        let v11 = get(r0 + 1, c0 + 1);

        if v00.is_nan() || v01.is_nan() || v10.is_nan() || v11.is_nan() {
            return self.nodata;
        }

        let top = v00 * (1.0 - fc) + v01 * fc;
        let bot = v10 * (1.0 - fc) + v11 * fc;
        top * (1.0 - fr) + bot * fr
    }

    fn sample_cubic(&self, r: f64, c: f64) -> f64 {
        let ri = r.floor() as isize;
        let ci = c.floor() as isize;
        let fr = r - r.floor();
        let fc = c - c.floor();

        let get = |ri: isize, ci: isize| -> f64 {
            let ri = ri.clamp(0, self.rows as isize - 1) as usize;
            let ci = ci.clamp(0, self.cols as isize - 1) as usize;
            let v = self.data[ri * self.cols + ci];
            if self.is_nodata(v) { f64::NAN } else { v }
        };

        let cubic_weight = |t: f64| -> [f64; 4] {
            let t2 = t * t;
            let t3 = t2 * t;
            [
                -0.5 * t3 + t2 - 0.5 * t,
                1.5 * t3 - 2.5 * t2 + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                0.5 * t3 - 0.5 * t2,
            ]
        };

        let wr = cubic_weight(fr);
        let wc = cubic_weight(fc);
        let mut sum = 0.0;
        let mut wsum = 0.0;

        for dr in 0..4_isize {
            for dc in 0..4_isize {
                let v = get(ri - 1 + dr, ci - 1 + dc);
                if v.is_nan() {
                    return self.nodata;
                }
                let w = wr[dr as usize] * wc[dc as usize];
                sum += w * v;
                wsum += w;
            }
        }

        if wsum.abs() < f64::EPSILON { self.nodata } else { sum / wsum }
    }

    // ── Statistics ──────────────────────────────────────────────

    /// Compute min, max, mean of valid cells.
    pub fn statistics(&self) -> Option<(f64, f64, f64)> {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum = 0.0;
        let mut count = 0usize;

        for &v in &self.data {
            if !self.is_nodata(v) {
                if v < min { min = v; }
                if v > max { max = v; }
                sum += v;
                count += 1;
            }
        }

        if count == 0 {
            None
        } else {
            Some((min, max, sum / count as f64))
        }
    }
}

impl RasterConfig {
    /// Build a grid filled with a specific value.
    pub fn build_with_fill(&self, value: f64) -> RasterGrid {
        let mut g = self.build();
        for v in g.data.iter_mut() {
            *v = value;
        }
        g
    }
}

impl fmt::Display for RasterGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let valid = self.valid_count();
        write!(
            f,
            "RasterGrid({}x{}, cell={}x{}, valid={}/{}, {})",
            self.rows, self.cols, self.cell_width, self.cell_height,
            valid, self.cell_count(), self.crs,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let cfg = RasterConfig::new(10, 20)
            .with_cell_size(0.5, 0.5)
            .with_origin(100.0, 200.0)
            .with_nodata(-1.0)
            .with_crs(Crs::Epsg(4326));
        assert_eq!(cfg.rows, 10);
        assert_eq!(cfg.cols, 20);
        assert_eq!(cfg.nodata, -1.0);
    }

    #[test]
    fn test_build_grid() {
        let g = RasterConfig::new(3, 4).build();
        assert_eq!(g.cell_count(), 12);
        assert_eq!(g.valid_count(), 0); // all nodata
    }

    #[test]
    fn test_filled_grid() {
        let g = RasterGrid::filled(5, 5, 42.0);
        assert_eq!(g.valid_count(), 25);
        assert_eq!(g.get(2, 3), Some(42.0));
    }

    #[test]
    fn test_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let g = RasterGrid::from_data(2, 3, data).unwrap();
        assert_eq!(g.get(0, 0), Some(1.0));
        assert_eq!(g.get(1, 2), Some(6.0));
    }

    #[test]
    fn test_from_data_wrong_size() {
        let data = vec![1.0, 2.0];
        assert!(RasterGrid::from_data(3, 3, data).is_none());
    }

    #[test]
    fn test_set_and_get() {
        let mut g = RasterGrid::filled(3, 3, 0.0);
        assert!(g.set(1, 1, 99.0));
        assert_eq!(g.get(1, 1), Some(99.0));
        assert!(!g.set(10, 10, 1.0));
    }

    #[test]
    fn test_nodata_handling() {
        let mut g = RasterGrid::filled(2, 2, 5.0);
        g.set(0, 0, g.nodata);
        assert!(g.is_nodata_at(0, 0));
        assert!(!g.is_nodata_at(0, 1));
        assert_eq!(g.valid_count(), 3);
    }

    #[test]
    fn test_pixel_to_coord() {
        let g = RasterConfig::new(4, 4)
            .with_cell_size(10.0, 10.0)
            .with_origin(100.0, 200.0)
            .build();
        let (x, y) = g.pixel_to_coord(0, 0);
        assert!((x - 105.0).abs() < 1e-9);
        assert!((y - 235.0).abs() < 1e-9);
    }

    #[test]
    fn test_coord_to_pixel() {
        let g = RasterConfig::new(4, 4)
            .with_cell_size(10.0, 10.0)
            .with_origin(100.0, 200.0)
            .build();
        let (r, c) = g.coord_to_pixel(105.0, 235.0);
        assert_eq!(r, 0);
        assert_eq!(c, 0);
    }

    #[test]
    fn test_extent() {
        let g = RasterConfig::new(10, 20)
            .with_cell_size(5.0, 5.0)
            .with_origin(0.0, 0.0)
            .build();
        let (min_x, min_y, max_x, max_y) = g.extent();
        assert!((min_x - 0.0).abs() < 1e-9);
        assert!((max_x - 100.0).abs() < 1e-9);
        assert!((max_y - 50.0).abs() < 1e-9);
        assert!((min_y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_subset() {
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let g = RasterGrid::from_data(4, 4, data).unwrap();
        let sub = g.subset(1, 1, 2, 2).unwrap();
        assert_eq!(sub.rows, 2);
        assert_eq!(sub.cols, 2);
        assert_eq!(sub.get(0, 0), Some(5.0));
        assert_eq!(sub.get(1, 1), Some(10.0));
    }

    #[test]
    fn test_subset_out_of_bounds() {
        let g = RasterGrid::filled(3, 3, 1.0);
        assert!(g.subset(2, 2, 3, 3).is_none());
    }

    #[test]
    fn test_resample_nearest() {
        let g = RasterGrid::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let r = g.resample(4, 4, Resampling::Nearest);
        assert_eq!(r.rows, 4);
        assert_eq!(r.cols, 4);
        // corners should preserve values
        assert_eq!(r.get(0, 0), Some(1.0));
        assert_eq!(r.get(3, 3), Some(4.0));
    }

    #[test]
    fn test_resample_bilinear() {
        let g = RasterGrid::from_data(2, 2, vec![0.0, 10.0, 10.0, 20.0]).unwrap();
        let r = g.resample(3, 3, Resampling::Bilinear);
        assert_eq!(r.rows, 3);
        // center should be ~10.0 (average of all 4)
        let center = r.get(1, 1).unwrap();
        assert!((center - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_resample_cubic() {
        let g = RasterGrid::from_data(4, 4, (0..16).map(|i| i as f64).collect()).unwrap();
        let r = g.resample(4, 4, Resampling::Cubic);
        assert_eq!(r.rows, 4);
        // same-size resample should roughly preserve values
        let v = r.get(1, 1).unwrap();
        assert!((v - 5.0).abs() < 2.0);
    }

    #[test]
    fn test_statistics() {
        let g = RasterGrid::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let (mn, mx, avg) = g.statistics().unwrap();
        assert!((mn - 1.0).abs() < 1e-9);
        assert!((mx - 4.0).abs() < 1e-9);
        assert!((avg - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_statistics_with_nodata() {
        let mut g = RasterGrid::from_data(2, 2, vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        g.set(0, 0, g.nodata);
        let (mn, mx, avg) = g.statistics().unwrap();
        assert!((mn - 20.0).abs() < 1e-9);
        assert!((mx - 40.0).abs() < 1e-9);
        assert!((avg - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_crs() {
        assert_eq!(format!("{}", Crs::Epsg(4326)), "EPSG:4326");
        assert_eq!(format!("{}", Crs::Unknown), "Unknown CRS");
    }

    #[test]
    fn test_display_grid() {
        let g = RasterGrid::filled(3, 4, 1.0);
        let s = format!("{g}");
        assert!(s.contains("3x4"));
        assert!(s.contains("valid=12/12"));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let g = RasterGrid::filled(2, 2, 1.0);
        assert!(g.get(5, 5).is_none());
    }
}
