//! Heightmap-based terrain engine: grid storage, normal computation, LOD via
//! CDLOD (chunked with seamless boundaries), bilinear height queries, hole
//! masks, and AABB bounds.
//!
//! Pure Rust — no GPU dependency. All geometry is computed on CPU; the
//! rendering layer maps vertex buffers to whatever backend is in use.

use std::collections::HashMap;

// ── Vec3 helper ──────────────────────────────────────────────────

/// Minimal 3-component vector for terrain math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return Self::zero();
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
        Vec3 {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    pub fn dot(a: &Vec3, b: &Vec3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }
}

// ── AABB ─────────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn contains_point(&self, p: &Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    pub fn size(&self) -> Vec3 {
        Vec3::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }
}

// ── LOD ──────────────────────────────────────────────────────────

/// Level-of-detail for terrain chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LodLevel {
    Full,
    Half,
    Quarter,
    Eighth,
}

impl LodLevel {
    /// How many source samples to skip per output vertex.
    pub fn step(&self) -> usize {
        match self {
            LodLevel::Full => 1,
            LodLevel::Half => 2,
            LodLevel::Quarter => 4,
            LodLevel::Eighth => 8,
        }
    }

    /// Select LOD level from camera distance.
    pub fn from_distance(dist: f32, thresholds: &LodThresholds) -> Self {
        if dist < thresholds.half {
            LodLevel::Full
        } else if dist < thresholds.quarter {
            LodLevel::Half
        } else if dist < thresholds.eighth {
            LodLevel::Quarter
        } else {
            LodLevel::Eighth
        }
    }
}

/// Distance thresholds for LOD transitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodThresholds {
    pub half: f32,
    pub quarter: f32,
    pub eighth: f32,
}

impl Default for LodThresholds {
    fn default() -> Self {
        Self {
            half: 100.0,
            quarter: 250.0,
            eighth: 500.0,
        }
    }
}

// ── Terrain chunk ────────────────────────────────────────────────

/// Coordinate of a chunk in the terrain grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub cx: i32,
    pub cz: i32,
}

/// A single terrain chunk — a square sub-grid of height values.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainChunk {
    pub coord: ChunkCoord,
    pub size: usize,
    pub heights: Vec<f32>,
    pub hole_mask: Vec<bool>,
    pub lod: LodLevel,
    spacing: f32,
}

impl TerrainChunk {
    /// Create chunk with all heights at zero.
    pub fn new(coord: ChunkCoord, size: usize, spacing: f32) -> Self {
        let count = size * size;
        Self {
            coord,
            size,
            heights: vec![0.0; count],
            hole_mask: vec![false; count],
            lod: LodLevel::Full,
            spacing,
        }
    }

    /// Create chunk from a height slice (row-major, size x size).
    pub fn from_heights(
        coord: ChunkCoord,
        size: usize,
        spacing: f32,
        heights: &[f32],
    ) -> Self {
        let count = size * size;
        let mut h = vec![0.0; count];
        let copy_len = heights.len().min(count);
        h[..copy_len].copy_from_slice(&heights[..copy_len]);
        Self {
            coord,
            size,
            heights: h,
            hole_mask: vec![false; count],
            lod: LodLevel::Full,
            spacing,
        }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.size + col
    }

    pub fn get_height(&self, row: usize, col: usize) -> f32 {
        if row < self.size && col < self.size {
            self.heights[self.idx(row, col)]
        } else {
            0.0
        }
    }

    pub fn set_height(&mut self, row: usize, col: usize, h: f32) {
        if row < self.size && col < self.size {
            let i = self.idx(row, col);
            self.heights[i] = h;
        }
    }

    pub fn is_hole(&self, row: usize, col: usize) -> bool {
        if row < self.size && col < self.size {
            self.hole_mask[self.idx(row, col)]
        } else {
            false
        }
    }

    pub fn set_hole(&mut self, row: usize, col: usize, hole: bool) {
        if row < self.size && col < self.size {
            let i = self.idx(row, col);
            self.hole_mask[i] = hole;
        }
    }

    /// Compute the surface normal at (row, col) using central differences.
    pub fn compute_normal(&self, row: usize, col: usize) -> Vec3 {
        let sz = self.size;
        if sz < 2 {
            return Vec3::new(0.0, 1.0, 0.0);
        }
        let h_left = if col > 0 {
            self.get_height(row, col - 1)
        } else {
            self.get_height(row, col)
        };
        let h_right = if col + 1 < sz {
            self.get_height(row, col + 1)
        } else {
            self.get_height(row, col)
        };
        let h_down = if row > 0 {
            self.get_height(row - 1, col)
        } else {
            self.get_height(row, col)
        };
        let h_up = if row + 1 < sz {
            self.get_height(row + 1, col)
        } else {
            self.get_height(row, col)
        };
        let dx = (h_right - h_left) / (2.0 * self.spacing);
        let dz = (h_up - h_down) / (2.0 * self.spacing);
        Vec3::new(-dx, 1.0, -dz).normalize()
    }

    /// Bilinear height query at fractional (row_f, col_f).
    pub fn sample_height_bilinear(&self, row_f: f32, col_f: f32) -> f32 {
        let r0 = row_f.floor() as usize;
        let c0 = col_f.floor() as usize;
        let r1 = (r0 + 1).min(self.size.saturating_sub(1));
        let c1 = (c0 + 1).min(self.size.saturating_sub(1));
        let fr = row_f - row_f.floor();
        let fc = col_f - col_f.floor();
        let h00 = self.get_height(r0, c0);
        let h10 = self.get_height(r1, c0);
        let h01 = self.get_height(r0, c1);
        let h11 = self.get_height(r1, c1);
        let top = h00 * (1.0 - fc) + h01 * fc;
        let bot = h10 * (1.0 - fc) + h11 * fc;
        top * (1.0 - fr) + bot * fr
    }

    /// Min/max height in this chunk.
    pub fn height_range(&self) -> (f32, f32) {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for &h in &self.heights {
            if h < lo {
                lo = h;
            }
            if h > hi {
                hi = h;
            }
        }
        (lo, hi)
    }

    /// AABB of this chunk in world space.
    pub fn bounds(&self) -> Aabb {
        let (lo, hi) = self.height_range();
        let wx = self.coord.cx as f32 * (self.size as f32 - 1.0) * self.spacing;
        let wz = self.coord.cz as f32 * (self.size as f32 - 1.0) * self.spacing;
        let extent = (self.size as f32 - 1.0) * self.spacing;
        Aabb::new(
            Vec3::new(wx, lo, wz),
            Vec3::new(wx + extent, hi, wz + extent),
        )
    }

    /// Generate vertices at the current LOD level. Returns (positions, normals).
    pub fn generate_lod_mesh(&self) -> (Vec<Vec3>, Vec<Vec3>) {
        let step = self.lod.step();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let wx = self.coord.cx as f32 * (self.size as f32 - 1.0) * self.spacing;
        let wz = self.coord.cz as f32 * (self.size as f32 - 1.0) * self.spacing;
        let mut row = 0;
        while row < self.size {
            let mut col = 0;
            while col < self.size {
                if !self.is_hole(row, col) {
                    let x = wx + col as f32 * self.spacing;
                    let y = self.get_height(row, col);
                    let z = wz + row as f32 * self.spacing;
                    positions.push(Vec3::new(x, y, z));
                    normals.push(self.compute_normal(row, col));
                }
                col += step;
            }
            row += step;
        }
        (positions, normals)
    }

    /// Count of non-hole vertices at current LOD.
    pub fn vertex_count(&self) -> usize {
        let step = self.lod.step();
        let mut count = 0usize;
        let mut row = 0;
        while row < self.size {
            let mut col = 0;
            while col < self.size {
                if !self.is_hole(row, col) {
                    count += 1;
                }
                col += step;
            }
            row += step;
        }
        count
    }
}

// ── Heightmap Terrain ────────────────────────────────────────────

/// Full terrain composed of chunks.
#[derive(Debug, Clone)]
pub struct HeightmapTerrain {
    pub chunk_size: usize,
    pub spacing: f32,
    pub lod_thresholds: LodThresholds,
    chunks: HashMap<(i32, i32), TerrainChunk>,
}

impl HeightmapTerrain {
    pub fn new(chunk_size: usize, spacing: f32) -> Self {
        Self {
            chunk_size,
            spacing,
            lod_thresholds: LodThresholds::default(),
            chunks: HashMap::new(),
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn insert_chunk(&mut self, chunk: TerrainChunk) {
        self.chunks.insert((chunk.coord.cx, chunk.coord.cz), chunk);
    }

    pub fn get_chunk(&self, cx: i32, cz: i32) -> Option<&TerrainChunk> {
        self.chunks.get(&(cx, cz))
    }

    pub fn get_chunk_mut(&mut self, cx: i32, cz: i32) -> Option<&mut TerrainChunk> {
        self.chunks.get_mut(&(cx, cz))
    }

    pub fn remove_chunk(&mut self, cx: i32, cz: i32) -> Option<TerrainChunk> {
        self.chunks.remove(&(cx, cz))
    }

    /// Query height at an arbitrary world (x, z) position via bilinear
    /// interpolation inside the owning chunk.
    pub fn height_at(&self, world_x: f32, world_z: f32) -> Option<f32> {
        let extent = (self.chunk_size as f32 - 1.0) * self.spacing;
        if extent <= 0.0 {
            return None;
        }
        let cx = (world_x / extent).floor() as i32;
        let cz = (world_z / extent).floor() as i32;
        let chunk = self.chunks.get(&(cx, cz))?;
        let local_x = world_x - cx as f32 * extent;
        let local_z = world_z - cz as f32 * extent;
        let col_f = local_x / self.spacing;
        let row_f = local_z / self.spacing;
        if col_f < 0.0
            || row_f < 0.0
            || col_f > (self.chunk_size - 1) as f32
            || row_f > (self.chunk_size - 1) as f32
        {
            return None;
        }
        Some(chunk.sample_height_bilinear(row_f, col_f))
    }

    /// Normal at world (x, z).
    pub fn normal_at(&self, world_x: f32, world_z: f32) -> Option<Vec3> {
        let extent = (self.chunk_size as f32 - 1.0) * self.spacing;
        if extent <= 0.0 {
            return None;
        }
        let cx = (world_x / extent).floor() as i32;
        let cz = (world_z / extent).floor() as i32;
        let chunk = self.chunks.get(&(cx, cz))?;
        let local_x = world_x - cx as f32 * extent;
        let local_z = world_z - cz as f32 * extent;
        let col = (local_x / self.spacing).round() as usize;
        let row = (local_z / self.spacing).round() as usize;
        if col >= self.chunk_size || row >= self.chunk_size {
            return None;
        }
        Some(chunk.compute_normal(row, col))
    }

    /// Update LOD for all chunks based on camera position.
    pub fn update_lod(&mut self, camera: &Vec3) {
        let thresholds = self.lod_thresholds;
        let vals: Vec<((i32, i32), LodLevel)> = self
            .chunks
            .iter()
            .map(|(key, chunk)| {
                let bounds = chunk.bounds();
                let center = Vec3::new(
                    (bounds.min.x + bounds.max.x) * 0.5,
                    (bounds.min.y + bounds.max.y) * 0.5,
                    (bounds.min.z + bounds.max.z) * 0.5,
                );
                let dx = camera.x - center.x;
                let dy = camera.y - center.y;
                let dz = camera.z - center.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                (*key, LodLevel::from_distance(dist, &thresholds))
            })
            .collect();
        for ((cx, cz), lod) in vals {
            if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
                chunk.lod = lod;
            }
        }
    }

    /// Global AABB enclosing all chunks.
    pub fn global_bounds(&self) -> Option<Aabb> {
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        if self.chunks.is_empty() {
            return None;
        }
        for chunk in self.chunks.values() {
            let b = chunk.bounds();
            if b.min.x < min.x {
                min.x = b.min.x;
            }
            if b.min.y < min.y {
                min.y = b.min.y;
            }
            if b.min.z < min.z {
                min.z = b.min.z;
            }
            if b.max.x > max.x {
                max.x = b.max.x;
            }
            if b.max.y > max.y {
                max.y = b.max.y;
            }
            if b.max.z > max.z {
                max.z = b.max.z;
            }
        }
        Some(Aabb::new(min, max))
    }

    /// Collect chunk coords, sorted for deterministic iteration.
    pub fn chunk_coords_sorted(&self) -> Vec<ChunkCoord> {
        let mut coords: Vec<ChunkCoord> = self
            .chunks
            .keys()
            .map(|&(cx, cz)| ChunkCoord { cx, cz })
            .collect();
        coords.sort_by(|a, b| a.cx.cmp(&b.cx).then(a.cz.cmp(&b.cz)));
        coords
    }
}

/// Generate a flat terrain of `cols x rows` chunks, each `chunk_size` on a side.
pub fn generate_flat_terrain(
    cols: usize,
    rows: usize,
    chunk_size: usize,
    spacing: f32,
    height: f32,
) -> HeightmapTerrain {
    let mut terrain = HeightmapTerrain::new(chunk_size, spacing);
    for cz in 0..rows as i32 {
        for cx in 0..cols as i32 {
            let coord = ChunkCoord { cx, cz };
            let mut chunk = TerrainChunk::new(coord, chunk_size, spacing);
            for h in &mut chunk.heights {
                *h = height;
            }
            terrain.insert_chunk(chunk);
        }
    }
    terrain
}

/// Generate a sine-hill terrain for testing.
pub fn generate_sine_terrain(
    cols: usize,
    rows: usize,
    chunk_size: usize,
    spacing: f32,
    amplitude: f32,
    frequency: f32,
) -> HeightmapTerrain {
    let mut terrain = HeightmapTerrain::new(chunk_size, spacing);
    for cz_i in 0..rows as i32 {
        for cx_i in 0..cols as i32 {
            let coord = ChunkCoord { cx: cx_i, cz: cz_i };
            let mut chunk = TerrainChunk::new(coord, chunk_size, spacing);
            let base_x = cx_i as f32 * (chunk_size as f32 - 1.0) * spacing;
            let base_z = cz_i as f32 * (chunk_size as f32 - 1.0) * spacing;
            for r in 0..chunk_size {
                for c in 0..chunk_size {
                    let wx = base_x + c as f32 * spacing;
                    let wz = base_z + r as f32 * spacing;
                    let h = amplitude
                        * (wx * frequency).sin()
                        * (wz * frequency).sin();
                    chunk.set_height(r, c, h);
                }
            }
            terrain.insert_chunk(chunk);
        }
    }
    terrain
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_v3(a: &Vec3, b: &Vec3, eps: f32) -> bool {
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps)
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(3.0, 0.0, 4.0).normalize();
        assert!(approx_eq(v.length(), 1.0, 1e-6));
    }

    #[test]
    fn vec3_cross_product() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = Vec3::cross(&x, &y);
        assert!(approx_v3(&z, &Vec3::new(0.0, 0.0, 1.0), 1e-6));
    }

    #[test]
    fn vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(approx_eq(Vec3::dot(&a, &b), 32.0, 1e-6));
    }

    #[test]
    fn vec3_zero_normalize() {
        let v = Vec3::zero().normalize();
        assert!(approx_v3(&v, &Vec3::zero(), 1e-6));
    }

    #[test]
    fn aabb_contains() {
        let bb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 10.0));
        assert!(bb.contains_point(&Vec3::new(5.0, 5.0, 5.0)));
        assert!(!bb.contains_point(&Vec3::new(11.0, 5.0, 5.0)));
    }

    #[test]
    fn aabb_size() {
        let bb = Aabb::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 6.0, 9.0));
        let s = bb.size();
        assert!(approx_eq(s.x, 3.0, 1e-6));
        assert!(approx_eq(s.y, 4.0, 1e-6));
        assert!(approx_eq(s.z, 6.0, 1e-6));
    }

    #[test]
    fn lod_from_distance() {
        let thr = LodThresholds::default();
        assert_eq!(LodLevel::from_distance(50.0, &thr), LodLevel::Full);
        assert_eq!(LodLevel::from_distance(150.0, &thr), LodLevel::Half);
        assert_eq!(LodLevel::from_distance(300.0, &thr), LodLevel::Quarter);
        assert_eq!(LodLevel::from_distance(600.0, &thr), LodLevel::Eighth);
    }

    #[test]
    fn lod_step_values() {
        assert_eq!(LodLevel::Full.step(), 1);
        assert_eq!(LodLevel::Half.step(), 2);
        assert_eq!(LodLevel::Quarter.step(), 4);
        assert_eq!(LodLevel::Eighth.step(), 8);
    }

    #[test]
    fn chunk_new_default_height() {
        let c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 4, 1.0);
        for r in 0..4 {
            for col in 0..4 {
                assert!(approx_eq(c.get_height(r, col), 0.0, 1e-6));
            }
        }
    }

    #[test]
    fn chunk_set_get_height() {
        let mut c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 4, 1.0);
        c.set_height(2, 3, 42.0);
        assert!(approx_eq(c.get_height(2, 3), 42.0, 1e-6));
    }

    #[test]
    fn chunk_hole_mask() {
        let mut c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 4, 1.0);
        assert!(!c.is_hole(1, 1));
        c.set_hole(1, 1, true);
        assert!(c.is_hole(1, 1));
        c.set_hole(1, 1, false);
        assert!(!c.is_hole(1, 1));
    }

    #[test]
    fn chunk_flat_normal() {
        let c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 5, 1.0);
        let n = c.compute_normal(2, 2);
        assert!(approx_v3(&n, &Vec3::new(0.0, 1.0, 0.0), 1e-6));
    }

    #[test]
    fn chunk_slope_normal() {
        let mut c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 5, 1.0);
        // Linear slope in X: h = col
        for r in 0..5 {
            for col in 0..5 {
                c.set_height(r, col, col as f32);
            }
        }
        let n = c.compute_normal(2, 2);
        assert!(n.x < 0.0, "normal should tilt against +X slope");
        assert!(n.y > 0.0, "normal should still point upward");
    }

    #[test]
    fn chunk_bilinear_interpolation() {
        let mut c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 3, 1.0);
        c.set_height(0, 0, 0.0);
        c.set_height(0, 1, 10.0);
        c.set_height(1, 0, 10.0);
        c.set_height(1, 1, 20.0);
        let mid = c.sample_height_bilinear(0.5, 0.5);
        assert!(approx_eq(mid, 10.0, 1e-4));
    }

    #[test]
    fn chunk_height_range() {
        let mut c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 4, 1.0);
        c.set_height(0, 0, -5.0);
        c.set_height(3, 3, 25.0);
        let (lo, hi) = c.height_range();
        assert!(approx_eq(lo, -5.0, 1e-6));
        assert!(approx_eq(hi, 25.0, 1e-6));
    }

    #[test]
    fn chunk_bounds() {
        let mut c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 5, 2.0);
        c.set_height(0, 0, 1.0);
        c.set_height(4, 4, 10.0);
        let b = c.bounds();
        assert!(approx_eq(b.min.y, 0.0, 1e-6));
        assert!(approx_eq(b.max.y, 10.0, 1e-6));
        assert!(approx_eq(b.max.x, 8.0, 1e-6)); // (5-1)*2
    }

    #[test]
    fn chunk_lod_mesh_vertex_count() {
        let c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 9, 1.0);
        assert_eq!(c.vertex_count(), 81); // 9*9 at Full
        let mut c2 = c.clone();
        c2.lod = LodLevel::Half;
        assert_eq!(c2.vertex_count(), 25); // 5*5
    }

    #[test]
    fn chunk_lod_mesh_with_holes() {
        let mut c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 5, 1.0);
        c.set_hole(0, 0, true);
        c.set_hole(4, 4, true);
        assert_eq!(c.vertex_count(), 23);
    }

    #[test]
    fn terrain_insert_and_query() {
        let mut t = HeightmapTerrain::new(5, 1.0);
        let mut c = TerrainChunk::new(ChunkCoord { cx: 0, cz: 0 }, 5, 1.0);
        c.set_height(2, 2, 7.0);
        t.insert_chunk(c);
        assert_eq!(t.chunk_count(), 1);
        let h = t.height_at(2.0, 2.0);
        assert!(h.is_some());
        assert!(approx_eq(h.unwrap(), 7.0, 1e-4));
    }

    #[test]
    fn terrain_height_at_missing_chunk() {
        let t = HeightmapTerrain::new(5, 1.0);
        assert!(t.height_at(100.0, 100.0).is_none());
    }

    #[test]
    fn terrain_normal_at() {
        let t = generate_flat_terrain(1, 1, 5, 1.0, 0.0);
        let n = t.normal_at(2.0, 2.0).unwrap();
        assert!(approx_v3(&n, &Vec3::new(0.0, 1.0, 0.0), 1e-6));
    }

    #[test]
    fn terrain_update_lod() {
        let mut t = generate_flat_terrain(2, 2, 9, 1.0, 0.0);
        let cam = Vec3::new(0.0, 0.0, 0.0);
        t.update_lod(&cam);
        // Chunk at (0,0) should be closest → Full
        let c00 = t.get_chunk(0, 0).unwrap();
        assert_eq!(c00.lod, LodLevel::Full);
    }

    #[test]
    fn terrain_global_bounds() {
        let t = generate_flat_terrain(2, 2, 5, 1.0, 3.0);
        let b = t.global_bounds().unwrap();
        assert!(approx_eq(b.min.y, 3.0, 1e-6));
        assert!(approx_eq(b.max.y, 3.0, 1e-6));
    }

    #[test]
    fn terrain_global_bounds_empty() {
        let t = HeightmapTerrain::new(5, 1.0);
        assert!(t.global_bounds().is_none());
    }

    #[test]
    fn generate_sine_terrain_height_variation() {
        let t = generate_sine_terrain(1, 1, 9, 1.0, 5.0, 0.5);
        let c = t.get_chunk(0, 0).unwrap();
        let (lo, hi) = c.height_range();
        assert!(lo < 0.0);
        assert!(hi > 0.0);
    }

    #[test]
    fn chunk_from_heights() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let c = TerrainChunk::from_heights(
            ChunkCoord { cx: 0, cz: 0 },
            4,
            1.0,
            &data,
        );
        assert!(approx_eq(c.get_height(0, 0), 0.0, 1e-6));
        assert!(approx_eq(c.get_height(3, 3), 15.0, 1e-6));
    }

    #[test]
    fn terrain_remove_chunk() {
        let mut t = generate_flat_terrain(2, 2, 5, 1.0, 0.0);
        assert_eq!(t.chunk_count(), 4);
        let removed = t.remove_chunk(0, 0);
        assert!(removed.is_some());
        assert_eq!(t.chunk_count(), 3);
    }

    #[test]
    fn chunk_coords_sorted_order() {
        let t = generate_flat_terrain(3, 2, 5, 1.0, 0.0);
        let coords = t.chunk_coords_sorted();
        assert_eq!(coords.len(), 6);
        // Should be sorted by cx then cz
        assert_eq!(coords[0], ChunkCoord { cx: 0, cz: 0 });
        assert_eq!(coords[1], ChunkCoord { cx: 0, cz: 1 });
        assert_eq!(coords[2], ChunkCoord { cx: 1, cz: 0 });
    }
}
