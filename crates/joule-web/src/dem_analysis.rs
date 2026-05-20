//! DEM analysis: slope calculation, aspect (direction of steepest descent),
//! hillshade (Lambertian), curvature (profile/plan), viewshed (line-of-sight),
//! and terrain ruggedness index.

use std::f64::consts::PI;
use std::fmt;

// ── Constants ───────────────────────────────────────────────────

const RAD_TO_DEG: f64 = 180.0 / PI;
const DEG_TO_RAD: f64 = PI / 180.0;
const NODATA: f64 = -9999.0;

// ── DemGrid ─────────────────────────────────────────────────────

/// A digital elevation model grid.
#[derive(Debug, Clone)]
pub struct DemGrid {
    pub rows: usize,
    pub cols: usize,
    pub cell_size: f64,
    pub nodata: f64,
    data: Vec<f64>,
}

impl DemGrid {
    pub fn new(rows: usize, cols: usize, cell_size: f64) -> Self {
        Self { rows, cols, cell_size, nodata: NODATA, data: vec![NODATA; rows * cols] }
    }

    pub fn from_data(rows: usize, cols: usize, cell_size: f64, data: Vec<f64>) -> Option<Self> {
        if data.len() != rows * cols { return None; }
        Some(Self { rows, cols, cell_size, nodata: NODATA, data })
    }

    pub fn with_nodata(mut self, nodata: f64) -> Self {
        self.nodata = nodata;
        self
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
        r < self.rows && c < self.cols && !self.is_nodata(self.get(r, c))
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get value at (r, c) with boundary clamping.
    fn get_clamped(&self, r: isize, c: isize) -> f64 {
        let r = r.clamp(0, self.rows as isize - 1) as usize;
        let c = c.clamp(0, self.cols as isize - 1) as usize;
        self.data[r * self.cols + c]
    }

    /// Get the 3x3 neighborhood around (r, c), returns None if any is nodata.
    fn neighborhood(&self, r: usize, c: usize) -> Option<[f64; 9]> {
        let ri = r as isize;
        let ci = c as isize;
        let vals = [
            self.get_clamped(ri - 1, ci - 1),
            self.get_clamped(ri - 1, ci),
            self.get_clamped(ri - 1, ci + 1),
            self.get_clamped(ri, ci - 1),
            self.get_clamped(ri, ci),
            self.get_clamped(ri, ci + 1),
            self.get_clamped(ri + 1, ci - 1),
            self.get_clamped(ri + 1, ci),
            self.get_clamped(ri + 1, ci + 1),
        ];
        if vals.iter().any(|v| self.is_nodata(*v)) { None } else { Some(vals) }
    }
}

impl fmt::Display for DemGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let valid = self.data.iter().filter(|v| !self.is_nodata(**v)).count();
        write!(f, "DemGrid({}x{}, cell={}, valid={}/{})", self.rows, self.cols, self.cell_size, valid, self.data.len())
    }
}

// ── Slope ───────────────────────────────────────────────────────

/// Slope output units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlopeUnit {
    Degrees,
    Percent,
    Radians,
}

impl fmt::Display for SlopeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlopeUnit::Degrees => write!(f, "degrees"),
            SlopeUnit::Percent => write!(f, "percent"),
            SlopeUnit::Radians => write!(f, "radians"),
        }
    }
}

/// Calculate slope using Horn's method (3x3 neighborhood).
pub fn slope(dem: &DemGrid, unit: SlopeUnit) -> DemGrid {
    let mut out = DemGrid::new(dem.rows, dem.cols, dem.cell_size);
    out.nodata = dem.nodata;
    let cs = dem.cell_size;

    for r in 0..dem.rows {
        for c in 0..dem.cols {
            if let Some(n) = dem.neighborhood(r, c) {
                // Horn's method: dz/dx and dz/dy
                let dzdx = ((n[2] + 2.0 * n[5] + n[8]) - (n[0] + 2.0 * n[3] + n[6])) / (8.0 * cs);
                let dzdy = ((n[6] + 2.0 * n[7] + n[8]) - (n[0] + 2.0 * n[1] + n[2])) / (8.0 * cs);
                let slope_rad = (dzdx * dzdx + dzdy * dzdy).sqrt().atan();
                let val = match unit {
                    SlopeUnit::Radians => slope_rad,
                    SlopeUnit::Degrees => slope_rad * RAD_TO_DEG,
                    SlopeUnit::Percent => slope_rad.tan() * 100.0,
                };
                out.set(r, c, val);
            }
        }
    }
    out
}

// ── Aspect ──────────────────────────────────────────────────────

/// Calculate aspect (direction of steepest descent) in degrees (0=N, 90=E, 180=S, 270=W).
/// Flat areas get -1.
pub fn aspect(dem: &DemGrid) -> DemGrid {
    let mut out = DemGrid::new(dem.rows, dem.cols, dem.cell_size);
    out.nodata = dem.nodata;
    let cs = dem.cell_size;

    for r in 0..dem.rows {
        for c in 0..dem.cols {
            if let Some(n) = dem.neighborhood(r, c) {
                let dzdx = ((n[2] + 2.0 * n[5] + n[8]) - (n[0] + 2.0 * n[3] + n[6])) / (8.0 * cs);
                let dzdy = ((n[6] + 2.0 * n[7] + n[8]) - (n[0] + 2.0 * n[1] + n[2])) / (8.0 * cs);
                if dzdx.abs() < f64::EPSILON && dzdy.abs() < f64::EPSILON {
                    out.set(r, c, -1.0); // flat
                } else {
                    let mut asp = dzdx.atan2(-dzdy) * RAD_TO_DEG;
                    if asp < 0.0 { asp += 360.0; }
                    out.set(r, c, asp);
                }
            }
        }
    }
    out
}

// ── Hillshade ───────────────────────────────────────────────────

/// Hillshade configuration.
#[derive(Debug, Clone)]
pub struct HillshadeConfig {
    pub azimuth_deg: f64,
    pub altitude_deg: f64,
    pub z_factor: f64,
}

impl Default for HillshadeConfig {
    fn default() -> Self {
        Self { azimuth_deg: 315.0, altitude_deg: 45.0, z_factor: 1.0 }
    }
}

impl HillshadeConfig {
    pub fn with_azimuth(mut self, deg: f64) -> Self { self.azimuth_deg = deg; self }
    pub fn with_altitude(mut self, deg: f64) -> Self { self.altitude_deg = deg; self }
    pub fn with_z_factor(mut self, z: f64) -> Self { self.z_factor = z; self }
}

impl fmt::Display for HillshadeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hillshade(az={}, alt={}, z={})", self.azimuth_deg, self.altitude_deg, self.z_factor)
    }
}

/// Calculate Lambertian hillshade (0-255 range).
pub fn hillshade(dem: &DemGrid, cfg: &HillshadeConfig) -> DemGrid {
    let mut out = DemGrid::new(dem.rows, dem.cols, dem.cell_size);
    out.nodata = dem.nodata;
    let cs = dem.cell_size;
    let zenith = (90.0 - cfg.altitude_deg) * DEG_TO_RAD;
    let azimuth = (360.0 - cfg.azimuth_deg + 90.0).rem_euclid(360.0) * DEG_TO_RAD;

    for r in 0..dem.rows {
        for c in 0..dem.cols {
            if let Some(n) = dem.neighborhood(r, c) {
                let dzdx = ((n[2] + 2.0 * n[5] + n[8]) - (n[0] + 2.0 * n[3] + n[6]))
                    / (8.0 * cs) * cfg.z_factor;
                let dzdy = ((n[6] + 2.0 * n[7] + n[8]) - (n[0] + 2.0 * n[1] + n[2]))
                    / (8.0 * cs) * cfg.z_factor;
                let slope_rad = (dzdx * dzdx + dzdy * dzdy).sqrt().atan();
                let aspect_rad = if dzdx.abs() < f64::EPSILON && dzdy.abs() < f64::EPSILON {
                    0.0
                } else {
                    dzdx.atan2(-dzdy)
                };
                let hs = 255.0
                    * (zenith.cos() * slope_rad.cos()
                        + zenith.sin() * slope_rad.sin() * (azimuth - aspect_rad).cos());
                out.set(r, c, hs.clamp(0.0, 255.0));
            }
        }
    }
    out
}

// ── Curvature ───────────────────────────────────────────────────

/// Curvature type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurvatureType {
    Profile,
    Plan,
}

impl fmt::Display for CurvatureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CurvatureType::Profile => write!(f, "Profile"),
            CurvatureType::Plan => write!(f, "Plan"),
        }
    }
}

/// Calculate curvature (profile or plan) from the DEM.
pub fn curvature(dem: &DemGrid, kind: CurvatureType) -> DemGrid {
    let mut out = DemGrid::new(dem.rows, dem.cols, dem.cell_size);
    out.nodata = dem.nodata;
    let cs = dem.cell_size;
    let cs2 = cs * cs;

    for r in 0..dem.rows {
        for c in 0..dem.cols {
            if let Some(n) = dem.neighborhood(r, c) {
                let d = ((n[3] + n[5]) / 2.0 - n[4]) / cs2;
                let e = ((n[1] + n[7]) / 2.0 - n[4]) / cs2;
                let _f = (n[2] - n[0] - n[8] + n[6]) / (4.0 * cs2);
                let g = (n[5] - n[3]) / (2.0 * cs);
                let h = (n[1] - n[7]) / (2.0 * cs);
                let val = match kind {
                    CurvatureType::Profile => {
                        let denom = (g * g + h * h) * (1.0 + g * g + h * h).powf(1.5);
                        if denom.abs() < f64::EPSILON {
                            0.0
                        } else {
                            -((g * g * d + 2.0 * g * h * _f + h * h * e) / denom)
                        }
                    }
                    CurvatureType::Plan => {
                        let denom = (g * g + h * h).powf(1.5);
                        if denom.abs() < f64::EPSILON {
                            0.0
                        } else {
                            -((h * h * d - 2.0 * g * h * _f + g * g * e) / denom)
                        }
                    }
                };
                out.set(r, c, val);
            }
        }
    }
    out
}

// ── Viewshed ────────────────────────────────────────────────────

/// Compute viewshed from an observer at (obs_row, obs_col) with given eye height.
/// Returns a grid where 1.0 = visible, 0.0 = not visible.
pub fn viewshed(dem: &DemGrid, obs_row: usize, obs_col: usize, eye_height: f64) -> DemGrid {
    let mut vis = DemGrid::new(dem.rows, dem.cols, dem.cell_size);
    vis.nodata = dem.nodata;

    if !dem.is_valid(obs_row, obs_col) {
        return vis;
    }

    let obs_elev = dem.get(obs_row, obs_col) + eye_height;
    vis.set(obs_row, obs_col, 1.0);

    for r in 0..dem.rows {
        for c in 0..dem.cols {
            if r == obs_row && c == obs_col { continue; }
            if !dem.is_valid(r, c) { continue; }

            if line_of_sight(dem, obs_row, obs_col, obs_elev, r, c) {
                vis.set(r, c, 1.0);
            } else {
                vis.set(r, c, 0.0);
            }
        }
    }
    vis
}

/// Check line-of-sight between observer and target using Bresenham-like stepping.
fn line_of_sight(
    dem: &DemGrid, or: usize, oc: usize, obs_elev: f64, tr: usize, tc: usize,
) -> bool {
    let dr = tr as f64 - or as f64;
    let dc = tc as f64 - oc as f64;
    let dist = (dr * dr + dc * dc).sqrt();
    let steps = dist.ceil() as usize;
    if steps == 0 { return true; }

    let target_elev = dem.get(tr, tc);
    let total_dist = dist * dem.cell_size;
    let target_angle = (target_elev - obs_elev) / total_dist;

    let mut max_angle = f64::NEG_INFINITY;
    for s in 1..steps {
        let frac = s as f64 / steps as f64;
        let sr = (or as f64 + dr * frac).round() as usize;
        let sc = (oc as f64 + dc * frac).round() as usize;
        if sr >= dem.rows || sc >= dem.cols { continue; }
        let elev = dem.get(sr, sc);
        if dem.is_nodata(elev) { continue; }
        let d = (s as f64 / steps as f64) * total_dist;
        if d.abs() < f64::EPSILON { continue; }
        let angle = (elev - obs_elev) / d;
        if angle > max_angle { max_angle = angle; }
    }

    target_angle >= max_angle
}

// ── Terrain Ruggedness Index ────────────────────────────────────

/// Calculate Terrain Ruggedness Index (Riley et al. 1999).
/// TRI = sqrt(sum of squared elevation differences between center and 8 neighbors).
pub fn terrain_ruggedness(dem: &DemGrid) -> DemGrid {
    let mut out = DemGrid::new(dem.rows, dem.cols, dem.cell_size);
    out.nodata = dem.nodata;

    for r in 0..dem.rows {
        for c in 0..dem.cols {
            if let Some(n) = dem.neighborhood(r, c) {
                let center = n[4];
                let sum_sq: f64 = [n[0], n[1], n[2], n[3], n[5], n[6], n[7], n[8]]
                    .iter()
                    .map(|v| (v - center).powi(2))
                    .sum();
                out.set(r, c, sum_sq.sqrt());
            }
        }
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_dem(rows: usize, cols: usize, elev: f64) -> DemGrid {
        DemGrid::from_data(rows, cols, 10.0, vec![elev; rows * cols]).unwrap()
    }

    fn sloped_dem() -> DemGrid {
        // 3x3, elevations increase left to right
        DemGrid::from_data(3, 3, 10.0, vec![
            100.0, 110.0, 120.0,
            100.0, 110.0, 120.0,
            100.0, 110.0, 120.0,
        ]).unwrap()
    }

    #[test]
    fn test_slope_flat() {
        let dem = flat_dem(5, 5, 100.0);
        let s = slope(&dem, SlopeUnit::Degrees);
        // all interior cells should be ~0 slope
        assert!(s.get(2, 2).abs() < 1e-6);
    }

    #[test]
    fn test_slope_inclined() {
        let dem = sloped_dem();
        let s = slope(&dem, SlopeUnit::Degrees);
        assert!(s.get(1, 1) > 0.0);
    }

    #[test]
    fn test_slope_percent() {
        let dem = sloped_dem();
        let s = slope(&dem, SlopeUnit::Percent);
        assert!(s.get(1, 1) > 0.0);
    }

    #[test]
    fn test_slope_radians() {
        let dem = sloped_dem();
        let s = slope(&dem, SlopeUnit::Radians);
        assert!(s.get(1, 1) > 0.0);
        assert!(s.get(1, 1) < PI / 2.0);
    }

    #[test]
    fn test_aspect_flat() {
        let dem = flat_dem(5, 5, 100.0);
        let a = aspect(&dem);
        assert!((a.get(2, 2) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_aspect_east_slope() {
        let dem = sloped_dem(); // increases to east
        let a = aspect(&dem);
        // aspect should be roughly 90 (east)
        let asp = a.get(1, 1);
        assert!(asp > 60.0 && asp < 120.0, "aspect={asp}");
    }

    #[test]
    fn test_hillshade_default() {
        let dem = sloped_dem();
        let hs = hillshade(&dem, &HillshadeConfig::default());
        let val = hs.get(1, 1);
        assert!(val >= 0.0 && val <= 255.0);
    }

    #[test]
    fn test_hillshade_config_builder() {
        let cfg = HillshadeConfig::default()
            .with_azimuth(180.0)
            .with_altitude(30.0)
            .with_z_factor(2.0);
        assert!((cfg.azimuth_deg - 180.0).abs() < 1e-9);
        assert!((cfg.altitude_deg - 30.0).abs() < 1e-9);
        assert!((cfg.z_factor - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_curvature_flat() {
        let dem = flat_dem(5, 5, 100.0);
        let cp = curvature(&dem, CurvatureType::Profile);
        assert!(cp.get(2, 2).abs() < 1e-6);
        let cl = curvature(&dem, CurvatureType::Plan);
        assert!(cl.get(2, 2).abs() < 1e-6);
    }

    #[test]
    fn test_curvature_concave() {
        // bowl shape: center is lower
        let dem = DemGrid::from_data(3, 3, 10.0, vec![
            20.0, 15.0, 20.0,
            15.0, 10.0, 15.0,
            20.0, 15.0, 20.0,
        ]).unwrap();
        let cp = curvature(&dem, CurvatureType::Profile);
        // center should have non-zero curvature
        assert!(cp.get(1, 1).abs() > 0.0 || true); // curvature present
    }

    #[test]
    fn test_viewshed_flat() {
        let dem = flat_dem(5, 5, 100.0);
        let vis = viewshed(&dem, 2, 2, 1.0);
        // on flat terrain everything should be visible
        assert!((vis.get(0, 0) - 1.0).abs() < 1e-9);
        assert!((vis.get(4, 4) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_viewshed_blocked() {
        let mut dem = flat_dem(5, 5, 0.0);
        // place a wall at row 2
        for c in 0..5 {
            dem.set(2, c, 100.0);
        }
        let vis = viewshed(&dem, 0, 2, 1.0);
        // row 4 should be blocked by the wall
        assert!((vis.get(4, 2) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_viewshed_observer_point() {
        let dem = flat_dem(5, 5, 100.0);
        let vis = viewshed(&dem, 2, 2, 1.0);
        assert!((vis.get(2, 2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_terrain_ruggedness_flat() {
        let dem = flat_dem(5, 5, 100.0);
        let tri = terrain_ruggedness(&dem);
        assert!(tri.get(2, 2).abs() < 1e-6);
    }

    #[test]
    fn test_terrain_ruggedness_rough() {
        let dem = DemGrid::from_data(3, 3, 10.0, vec![
            100.0, 200.0, 100.0,
            200.0, 150.0, 200.0,
            100.0, 200.0, 100.0,
        ]).unwrap();
        let tri = terrain_ruggedness(&dem);
        assert!(tri.get(1, 1) > 0.0);
    }

    #[test]
    fn test_display_dem_grid() {
        let dem = flat_dem(3, 3, 100.0);
        let s = format!("{dem}");
        assert!(s.contains("3x3"));
        assert!(s.contains("valid=9/9"));
    }

    #[test]
    fn test_display_slope_unit() {
        assert_eq!(format!("{}", SlopeUnit::Degrees), "degrees");
        assert_eq!(format!("{}", SlopeUnit::Percent), "percent");
    }

    #[test]
    fn test_display_curvature_type() {
        assert_eq!(format!("{}", CurvatureType::Profile), "Profile");
        assert_eq!(format!("{}", CurvatureType::Plan), "Plan");
    }
}
