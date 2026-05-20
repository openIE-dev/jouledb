//! H3-like hexagonal indexing — hex grid generation, lat/lon to hex cell,
//! hex neighbors (k-ring), hex distance, hierarchical resolution
//! (parent/child), hex boundary vertices.
//!
//! Pure-Rust hexagonal spatial index inspired by H3's hierarchical
//! grid system with flat-top hexagons, cube coordinates, and
//! multi-resolution parent/child relationships.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum H3Error {
    InvalidResolution(u8),
    InvalidCoordinate { lat: f64, lon: f64 },
    InvalidIndex(String),
}

impl fmt::Display for H3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResolution(r) => write!(f, "invalid resolution: {r} (must be 0..=15)"),
            Self::InvalidCoordinate { lat, lon } => {
                write!(f, "invalid coordinate: lat={lat}, lon={lon}")
            }
            Self::InvalidIndex(s) => write!(f, "invalid hex index: {s}"),
        }
    }
}

impl std::error::Error for H3Error {}

// ── Constants ───────────────────────────────────────────────────

const MAX_RESOLUTION: u8 = 15;
const SQRT3: f64 = 1.7320508075688772;

// ── Cube coordinates ────────────────────────────────────────────

/// Cube coordinates for a hex cell (q + r + s = 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeCoord {
    pub q: i64,
    pub r: i64,
    pub s: i64,
}

impl CubeCoord {
    pub fn new(q: i64, r: i64) -> Self {
        Self { q, r, s: -q - r }
    }

    pub fn origin() -> Self { Self::new(0, 0) }

    pub fn distance(&self, other: &CubeCoord) -> u64 {
        let dq = (self.q - other.q).unsigned_abs();
        let dr = (self.r - other.r).unsigned_abs();
        let ds = (self.s - other.s).unsigned_abs();
        dq.max(dr).max(ds)
    }

    /// The six immediate neighbors in cube coordinates.
    pub fn neighbors(&self) -> [CubeCoord; 6] {
        [
            CubeCoord::new(self.q + 1, self.r),
            CubeCoord::new(self.q + 1, self.r - 1),
            CubeCoord::new(self.q, self.r - 1),
            CubeCoord::new(self.q - 1, self.r),
            CubeCoord::new(self.q - 1, self.r + 1),
            CubeCoord::new(self.q, self.r + 1),
        ]
    }
}

impl fmt::Display for CubeCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CubeCoord({}, {}, {})", self.q, self.r, self.s)
    }
}

// ── HexCell ─────────────────────────────────────────────────────

/// A hex cell at a given resolution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HexCell {
    pub coord: CubeCoord,
    pub resolution: u8,
}

impl HexCell {
    pub fn new(coord: CubeCoord, resolution: u8) -> Self {
        Self { coord, resolution }
    }

    /// Cell edge length in degrees (approximate) at this resolution.
    pub fn edge_length_deg(&self) -> f64 {
        180.0 / (3.0_f64.powi(self.resolution as i32))
    }

    /// Compute the hex distance to another cell at the same resolution.
    pub fn distance(&self, other: &HexCell) -> Option<u64> {
        if self.resolution != other.resolution {
            return None;
        }
        Some(self.coord.distance(&other.coord))
    }

    /// Get the parent cell at coarser resolution.
    pub fn parent(&self) -> Option<HexCell> {
        if self.resolution == 0 {
            return None;
        }
        // Map to parent: divide coordinates by 3 (floor toward zero)
        let pq = div_floor(self.coord.q, 3);
        let pr = div_floor(self.coord.r, 3);
        Some(HexCell::new(CubeCoord::new(pq, pr), self.resolution - 1))
    }

    /// Get children cells at finer resolution (7 children per parent in H3-like scheme).
    pub fn children(&self) -> Option<Vec<HexCell>> {
        if self.resolution >= MAX_RESOLUTION {
            return None;
        }
        let child_res = self.resolution + 1;
        let cq = self.coord.q * 3;
        let cr = self.coord.r * 3;
        // Center child + 6 surrounding
        let mut kids = Vec::with_capacity(7);
        kids.push(HexCell::new(CubeCoord::new(cq, cr), child_res));
        for nb in CubeCoord::new(cq, cr).neighbors() {
            kids.push(HexCell::new(nb, child_res));
        }
        Some(kids)
    }

    /// Compute boundary vertices (6 corners) in pixel/coordinate space.
    pub fn boundary_vertices(&self, center_x: f64, center_y: f64, size: f64) -> [(f64, f64); 6] {
        let mut vertices = [(0.0, 0.0); 6];
        for i in 0..6 {
            let angle_deg = 60.0 * i as f64;
            let angle_rad = angle_deg.to_radians();
            vertices[i] = (
                center_x + size * angle_rad.cos(),
                center_y + size * angle_rad.sin(),
            );
        }
        vertices
    }
}

impl fmt::Display for HexCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HexCell(res={}, {})", self.resolution, self.coord)
    }
}

fn div_floor(a: i64, b: i64) -> i64 {
    let d = a / b;
    if (a ^ b) < 0 && a % b != 0 { d - 1 } else { d }
}

// ── HexGrid ─────────────────────────────────────────────────────

/// Hex grid manager for lat/lon to hex cell conversion and ring queries.
pub struct HexGrid {
    resolution: u8,
}

impl HexGrid {
    pub fn new(resolution: u8) -> Result<Self, H3Error> {
        if resolution > MAX_RESOLUTION {
            return Err(H3Error::InvalidResolution(resolution));
        }
        Ok(Self { resolution })
    }

    pub fn resolution(&self) -> u8 { self.resolution }

    /// Convert lat/lon to hex cell coordinates.
    pub fn lat_lon_to_cell(&self, lat: f64, lon: f64) -> Result<HexCell, H3Error> {
        if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
            return Err(H3Error::InvalidCoordinate { lat, lon });
        }
        let size = self.cell_size_deg();
        // Convert to fractional hex coordinates (flat-top)
        let fx = (2.0 / 3.0 * lon) / size;
        let fy = (-1.0 / 3.0 * lon + SQRT3 / 3.0 * lat) / size;
        let (q, r) = cube_round(fx, fy);
        Ok(HexCell::new(CubeCoord::new(q, r), self.resolution))
    }

    /// Convert hex cell to approximate lat/lon center.
    pub fn cell_to_lat_lon(&self, cell: &HexCell) -> (f64, f64) {
        let size = self.cell_size_deg();
        let lon = size * 1.5 * cell.coord.q as f64;
        let lat = size * (SQRT3 * (cell.coord.r as f64 + cell.coord.q as f64 * 0.5));
        (lat.clamp(-90.0, 90.0), lon.clamp(-180.0, 180.0))
    }

    /// Cell size in degrees at this resolution.
    pub fn cell_size_deg(&self) -> f64 {
        180.0 / (3.0_f64.powi(self.resolution as i32))
    }

    /// K-ring: all cells within distance k of center.
    pub fn k_ring(&self, center: &CubeCoord, k: u32) -> Vec<HexCell> {
        let mut cells = Vec::new();
        let ki = k as i64;
        for q in -ki..=ki {
            let r_min = (-ki).max(-q - ki);
            let r_max = ki.min(-q + ki);
            for r in r_min..=r_max {
                cells.push(HexCell::new(CubeCoord::new(center.q + q, center.r + r), self.resolution));
            }
        }
        cells
    }

    /// Hex ring: cells exactly at distance k from center.
    pub fn hex_ring(&self, center: &CubeCoord, k: u32) -> Vec<HexCell> {
        if k == 0 {
            return vec![HexCell::new(*center, self.resolution)];
        }
        let mut results = Vec::new();
        let ki = k as i64;
        // Walk around the ring
        let directions = [
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        ];
        let mut cur = CubeCoord::new(center.q - ki, center.r + ki);
        for &(dq, dr) in &directions {
            for _ in 0..k {
                results.push(HexCell::new(cur, self.resolution));
                cur = CubeCoord::new(cur.q + dq, cur.r + dr);
            }
        }
        results
    }

    /// Generate all hex cells covering a bounding box.
    pub fn cells_in_bbox(&self, lat_min: f64, lon_min: f64, lat_max: f64, lon_max: f64) -> Result<Vec<HexCell>, H3Error> {
        let c1 = self.lat_lon_to_cell(lat_min, lon_min)?;
        let c2 = self.lat_lon_to_cell(lat_max, lon_max)?;
        let q_min = c1.coord.q.min(c2.coord.q) - 1;
        let q_max = c1.coord.q.max(c2.coord.q) + 1;
        let r_min = c1.coord.r.min(c2.coord.r) - 1;
        let r_max = c1.coord.r.max(c2.coord.r) + 1;
        let mut cells = Vec::new();
        let pad = self.cell_size_deg();
        for q in q_min..=q_max {
            for r in r_min..=r_max {
                let cell = HexCell::new(CubeCoord::new(q, r), self.resolution);
                let (lat, lon) = self.cell_to_lat_lon(&cell);
                if lat >= lat_min - pad && lat <= lat_max + pad && lon >= lon_min - pad && lon <= lon_max + pad {
                    cells.push(cell);
                }
            }
        }
        Ok(cells)
    }
}

impl fmt::Display for HexGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HexGrid(res={}, cell_size={:.4}°)", self.resolution, self.cell_size_deg())
    }
}

// ── Hex index encoding ──────────────────────────────────────────

/// Encode a HexCell into a compact u64 index.
pub fn cell_to_index(cell: &HexCell) -> u64 {
    let res = cell.resolution as u64;
    // Bias coordinates to be non-negative
    let bias = 1_i64 << 28;
    let q = (cell.coord.q + bias) as u64;
    let r = (cell.coord.r + bias) as u64;
    (res << 58) | (q << 29) | r
}

/// Decode a u64 index back to a HexCell.
pub fn index_to_cell(index: u64) -> Result<HexCell, H3Error> {
    let res = (index >> 58) as u8;
    if res > MAX_RESOLUTION {
        return Err(H3Error::InvalidIndex(format!("resolution {res} out of range")));
    }
    let bias = 1_i64 << 28;
    let mask = (1_u64 << 29) - 1;
    let q = ((index >> 29) & mask) as i64 - bias;
    let r = (index & mask) as i64 - bias;
    Ok(HexCell::new(CubeCoord::new(q, r), res))
}

// ── Cube rounding ───────────────────────────────────────────────

fn cube_round(fq: f64, fr: f64) -> (i64, i64) {
    let fs = -fq - fr;
    let mut q = fq.round();
    let mut r = fr.round();
    let s = fs.round();
    let q_diff = (q - fq).abs();
    let r_diff = (r - fr).abs();
    let s_diff = (s - fs).abs();
    if q_diff > r_diff && q_diff > s_diff {
        q = -r - s;
    } else if r_diff > s_diff {
        r = -q - s;
    }
    (q as i64, r as i64)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_coord_new() {
        let c = CubeCoord::new(1, -1);
        assert_eq!(c.s, 0);
        assert_eq!(c.q + c.r + c.s, 0);
    }

    #[test]
    fn test_cube_coord_distance() {
        let a = CubeCoord::origin();
        let b = CubeCoord::new(3, -2);
        assert_eq!(a.distance(&b), 3);
    }

    #[test]
    fn test_cube_coord_neighbors() {
        let c = CubeCoord::origin();
        let nbrs = c.neighbors();
        assert_eq!(nbrs.len(), 6);
        for nb in &nbrs {
            assert_eq!(c.distance(nb), 1);
        }
    }

    #[test]
    fn test_cube_coord_display() {
        let c = CubeCoord::new(1, 2);
        let s = format!("{c}");
        assert!(s.contains("CubeCoord"));
    }

    #[test]
    fn test_hex_cell_display() {
        let cell = HexCell::new(CubeCoord::new(1, 2), 5);
        let s = format!("{cell}");
        assert!(s.contains("HexCell"));
        assert!(s.contains("res=5"));
    }

    #[test]
    fn test_hex_cell_parent() {
        let cell = HexCell::new(CubeCoord::new(9, -6), 3);
        let parent = cell.parent().unwrap();
        assert_eq!(parent.resolution, 2);
    }

    #[test]
    fn test_hex_cell_parent_at_zero() {
        let cell = HexCell::new(CubeCoord::origin(), 0);
        assert!(cell.parent().is_none());
    }

    #[test]
    fn test_hex_cell_children() {
        let cell = HexCell::new(CubeCoord::origin(), 3);
        let children = cell.children().unwrap();
        assert_eq!(children.len(), 7);
        assert!(children.iter().all(|c| c.resolution == 4));
    }

    #[test]
    fn test_hex_cell_children_max_res() {
        let cell = HexCell::new(CubeCoord::origin(), MAX_RESOLUTION);
        assert!(cell.children().is_none());
    }

    #[test]
    fn test_hex_cell_boundary_vertices() {
        let cell = HexCell::new(CubeCoord::origin(), 5);
        let verts = cell.boundary_vertices(0.0, 0.0, 1.0);
        assert_eq!(verts.len(), 6);
        // First vertex at angle 0 should be (1.0, 0.0)
        assert!((verts[0].0 - 1.0).abs() < 1e-9);
        assert!(verts[0].1.abs() < 1e-9);
    }

    #[test]
    fn test_hex_grid_new() {
        assert!(HexGrid::new(5).is_ok());
        assert!(HexGrid::new(16).is_err());
    }

    #[test]
    fn test_hex_grid_display() {
        let grid = HexGrid::new(5).unwrap();
        let s = format!("{grid}");
        assert!(s.contains("HexGrid"));
    }

    #[test]
    fn test_lat_lon_to_cell() {
        let grid = HexGrid::new(3).unwrap();
        let cell = grid.lat_lon_to_cell(48.8566, 2.3522).unwrap();
        assert_eq!(cell.resolution, 3);
    }

    #[test]
    fn test_lat_lon_roundtrip() {
        let grid = HexGrid::new(5).unwrap();
        let cell = grid.lat_lon_to_cell(40.0, -74.0).unwrap();
        let (lat, lon) = grid.cell_to_lat_lon(&cell);
        assert!((lat - 40.0).abs() < grid.cell_size_deg() * 2.0);
        assert!((lon - (-74.0)).abs() < grid.cell_size_deg() * 2.0);
    }

    #[test]
    fn test_lat_lon_invalid() {
        let grid = HexGrid::new(3).unwrap();
        assert!(grid.lat_lon_to_cell(91.0, 0.0).is_err());
    }

    #[test]
    fn test_k_ring() {
        let grid = HexGrid::new(5).unwrap();
        let center = CubeCoord::origin();
        let ring0 = grid.k_ring(&center, 0);
        assert_eq!(ring0.len(), 1);
        let ring1 = grid.k_ring(&center, 1);
        assert_eq!(ring1.len(), 7);  // 1 + 6
        let ring2 = grid.k_ring(&center, 2);
        assert_eq!(ring2.len(), 19); // 1 + 6 + 12
    }

    #[test]
    fn test_hex_ring() {
        let grid = HexGrid::new(5).unwrap();
        let center = CubeCoord::origin();
        let ring1 = grid.hex_ring(&center, 1);
        assert_eq!(ring1.len(), 6);
        let ring2 = grid.hex_ring(&center, 2);
        assert_eq!(ring2.len(), 12);
    }

    #[test]
    fn test_hex_distance() {
        let a = HexCell::new(CubeCoord::origin(), 5);
        let b = HexCell::new(CubeCoord::new(3, -1), 5);
        assert_eq!(a.distance(&b), Some(3));
    }

    #[test]
    fn test_hex_distance_different_res() {
        let a = HexCell::new(CubeCoord::origin(), 3);
        let b = HexCell::new(CubeCoord::origin(), 5);
        assert_eq!(a.distance(&b), None);
    }

    #[test]
    fn test_cell_index_roundtrip() {
        let cell = HexCell::new(CubeCoord::new(42, -17), 7);
        let idx = cell_to_index(&cell);
        let decoded = index_to_cell(idx).unwrap();
        assert_eq!(decoded.coord, cell.coord);
        assert_eq!(decoded.resolution, cell.resolution);
    }

    #[test]
    fn test_cells_in_bbox() {
        let grid = HexGrid::new(2).unwrap();
        let cells = grid.cells_in_bbox(48.0, 2.0, 49.0, 3.0).unwrap();
        assert!(!cells.is_empty());
    }

    #[test]
    fn test_edge_length_decreases() {
        let c3 = HexCell::new(CubeCoord::origin(), 3);
        let c5 = HexCell::new(CubeCoord::origin(), 5);
        assert!(c3.edge_length_deg() > c5.edge_length_deg());
    }
}
