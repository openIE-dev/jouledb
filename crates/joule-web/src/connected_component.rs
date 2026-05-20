//! Connected components — labeling (4/8-connectivity), component extraction,
//! area/centroid/bounding box, flood fill, largest component, filtering by size.
//!
//! Replaces OpenCV.js connectedComponents with a pure-Rust union-find
//! implementation that works on native and WASM.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Binary Image ─────────────────────────────────────────────────

/// A binary image for connected component analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<bool>,
}

impl BinaryImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![false; width as usize * height as usize] }
    }

    pub fn from_data(width: u32, height: u32, data: Vec<bool>) -> Self {
        assert_eq!(data.len(), width as usize * height as usize);
        Self { width, height, data }
    }

    pub fn get(&self, x: u32, y: u32) -> bool {
        self.data[y as usize * self.width as usize + x as usize]
    }

    pub fn set(&mut self, x: u32, y: u32, v: bool) {
        self.data[y as usize * self.width as usize + x as usize] = v;
    }
}

// ── Connectivity ─────────────────────────────────────────────────

/// Connectivity type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Connectivity {
    Four,
    Eight,
}

impl Connectivity {
    fn neighbor_offsets(self) -> &'static [(i32, i32)] {
        match self {
            Self::Four => &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            Self::Eight => &[
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),           (1, 0),
                (-1, 1),  (0, 1),  (1, 1),
            ],
        }
    }
}

// ── Union-Find ───────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb { return; }
        if self.rank[ra as usize] < self.rank[rb as usize] {
            self.parent[ra as usize] = rb;
        } else if self.rank[ra as usize] > self.rank[rb as usize] {
            self.parent[rb as usize] = ra;
        } else {
            self.parent[rb as usize] = ra;
            self.rank[ra as usize] += 1;
        }
    }
}

// ── Label Map ────────────────────────────────────────────────────

/// Labeled image — each pixel has a component label (0 = background).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelMap {
    pub width: u32,
    pub height: u32,
    pub labels: Vec<u32>,
    pub num_labels: u32,
}

impl LabelMap {
    pub fn get(&self, x: u32, y: u32) -> u32 {
        self.labels[y as usize * self.width as usize + x as usize]
    }
}

// ── Component Stats ──────────────────────────────────────────────

/// Statistics for a single connected component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentStats {
    pub label: u32,
    pub area: u32,
    pub centroid_x: f64,
    pub centroid_y: f64,
    pub bbox_x: u32,
    pub bbox_y: u32,
    pub bbox_w: u32,
    pub bbox_h: u32,
}

// ── Labeling ─────────────────────────────────────────────────────

/// Label connected components using a two-pass union-find algorithm.
pub fn label_components(img: &BinaryImage, connectivity: Connectivity) -> LabelMap {
    let w = img.width as usize;
    let h = img.height as usize;
    let n = w * h;

    let mut uf = UnionFind::new(n);
    let mut temp_labels = vec![0u32; n];
    let mut next_label = 1u32;

    // Pass 1: assign provisional labels
    for y in 0..h {
        for x in 0..w {
            if !img.data[y * w + x] {
                continue;
            }

            // Collect labels of already-visited neighbors
            let mut neighbor_labels = Vec::new();
            let offsets: &[(i32, i32)] = match connectivity {
                Connectivity::Four => &[(-1, 0), (0, -1)],
                Connectivity::Eight => &[(-1, -1), (0, -1), (1, -1), (-1, 0)],
            };

            for &(dx, dy) in offsets {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let ni = ny as usize * w + nx as usize;
                    if temp_labels[ni] > 0 {
                        neighbor_labels.push(temp_labels[ni]);
                    }
                }
            }

            if neighbor_labels.is_empty() {
                temp_labels[y * w + x] = next_label;
                next_label += 1;
            } else {
                let min_label = *neighbor_labels.iter().min().unwrap();
                temp_labels[y * w + x] = min_label;
                for &nl in &neighbor_labels {
                    uf.union(min_label, nl);
                }
            }
        }
    }

    // Pass 2: resolve labels
    let mut label_remap: HashMap<u32, u32> = HashMap::new();
    let mut final_label = 1u32;
    let mut labels = vec![0u32; n];

    for i in 0..n {
        if temp_labels[i] == 0 {
            continue;
        }
        let root = uf.find(temp_labels[i]);
        let mapped = *label_remap.entry(root).or_insert_with(|| {
            let l = final_label;
            final_label += 1;
            l
        });
        labels[i] = mapped;
    }

    LabelMap {
        width: img.width,
        height: img.height,
        labels,
        num_labels: final_label - 1,
    }
}

/// Extract statistics for each component.
pub fn component_stats(label_map: &LabelMap) -> Vec<ComponentStats> {
    let mut stats: HashMap<u32, (u32, f64, f64, u32, u32, u32, u32)> = HashMap::new();

    for y in 0..label_map.height {
        for x in 0..label_map.width {
            let label = label_map.get(x, y);
            if label == 0 { continue; }
            let entry = stats.entry(label).or_insert((0, 0.0, 0.0, x, y, x, y));
            entry.0 += 1;
            entry.1 += x as f64;
            entry.2 += y as f64;
            entry.3 = entry.3.min(x);
            entry.4 = entry.4.min(y);
            entry.5 = entry.5.max(x);
            entry.6 = entry.6.max(y);
        }
    }

    let mut result: Vec<_> = stats.into_iter().map(|(label, (area, sx, sy, minx, miny, maxx, maxy))| {
        ComponentStats {
            label,
            area,
            centroid_x: sx / area as f64,
            centroid_y: sy / area as f64,
            bbox_x: minx,
            bbox_y: miny,
            bbox_w: maxx - minx + 1,
            bbox_h: maxy - miny + 1,
        }
    }).collect();
    result.sort_by_key(|s| s.label);
    result
}

/// Find the largest component by area.
pub fn largest_component(stats: &[ComponentStats]) -> Option<&ComponentStats> {
    stats.iter().max_by_key(|s| s.area)
}

/// Filter components by minimum area, returning a new binary image.
pub fn filter_by_size(label_map: &LabelMap, stats: &[ComponentStats], min_area: u32) -> BinaryImage {
    let keep: std::collections::HashSet<u32> = stats
        .iter()
        .filter(|s| s.area >= min_area)
        .map(|s| s.label)
        .collect();
    let mut out = BinaryImage::new(label_map.width, label_map.height);
    for i in 0..label_map.labels.len() {
        out.data[i] = keep.contains(&label_map.labels[i]);
    }
    out
}

// ── Flood Fill ───────────────────────────────────────────────────

/// Flood fill from a seed point, returning the filled binary image.
pub fn flood_fill(
    img: &BinaryImage,
    seed_x: u32,
    seed_y: u32,
    connectivity: Connectivity,
) -> BinaryImage {
    let mut out = BinaryImage::new(img.width, img.height);
    if !img.get(seed_x, seed_y) {
        return out;
    }

    let mut stack = vec![(seed_x, seed_y)];
    let mut visited = vec![false; img.width as usize * img.height as usize];

    while let Some((x, y)) = stack.pop() {
        let idx = y as usize * img.width as usize + x as usize;
        if visited[idx] { continue; }
        visited[idx] = true;

        if !img.get(x, y) { continue; }
        out.set(x, y, true);

        for &(dx, dy) in connectivity.neighbor_offsets() {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 && (nx as u32) < img.width && (ny as u32) < img.height {
                let ni = ny as usize * img.width as usize + nx as usize;
                if !visited[ni] {
                    stack.push((nx as u32, ny as u32));
                }
            }
        }
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_blobs() -> BinaryImage {
        // 7x5 with two separate 2x2 blobs
        let mut img = BinaryImage::new(7, 5);
        // Blob 1 at (0,0)
        img.set(0, 0, true);
        img.set(1, 0, true);
        img.set(0, 1, true);
        img.set(1, 1, true);
        // Blob 2 at (4,3)
        img.set(4, 3, true);
        img.set(5, 3, true);
        img.set(4, 4, true);
        img.set(5, 4, true);
        img
    }

    #[test]
    fn test_label_two_components_4conn() {
        let img = make_two_blobs();
        let lm = label_components(&img, Connectivity::Four);
        assert_eq!(lm.num_labels, 2);
        assert_ne!(lm.get(0, 0), 0);
        assert_ne!(lm.get(4, 3), 0);
        assert_ne!(lm.get(0, 0), lm.get(4, 3));
    }

    #[test]
    fn test_label_background_is_zero() {
        let img = make_two_blobs();
        let lm = label_components(&img, Connectivity::Four);
        assert_eq!(lm.get(3, 2), 0);
    }

    #[test]
    fn test_component_stats_area() {
        let img = make_two_blobs();
        let lm = label_components(&img, Connectivity::Four);
        let stats = component_stats(&lm);
        assert_eq!(stats.len(), 2);
        assert!(stats.iter().all(|s| s.area == 4));
    }

    #[test]
    fn test_component_centroid() {
        let img = make_two_blobs();
        let lm = label_components(&img, Connectivity::Four);
        let stats = component_stats(&lm);
        // Blob 1 centroid should be (0.5, 0.5)
        let s1 = stats.iter().find(|s| s.bbox_x == 0).unwrap();
        assert!((s1.centroid_x - 0.5).abs() < 0.01);
        assert!((s1.centroid_y - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_component_bbox() {
        let img = make_two_blobs();
        let lm = label_components(&img, Connectivity::Four);
        let stats = component_stats(&lm);
        let s2 = stats.iter().find(|s| s.bbox_x == 4).unwrap();
        assert_eq!(s2.bbox_w, 2);
        assert_eq!(s2.bbox_h, 2);
    }

    #[test]
    fn test_largest_component() {
        let mut img = BinaryImage::new(10, 5);
        // Small blob
        img.set(0, 0, true);
        // Big blob
        for y in 2..5 {
            for x in 3..8 {
                img.set(x, y, true);
            }
        }
        let lm = label_components(&img, Connectivity::Four);
        let stats = component_stats(&lm);
        let largest = largest_component(&stats).unwrap();
        assert_eq!(largest.area, 15);
    }

    #[test]
    fn test_filter_by_size() {
        let mut img = BinaryImage::new(10, 5);
        img.set(0, 0, true); // 1-pixel blob
        for y in 2..5 {
            for x in 3..8 {
                img.set(x, y, true);
            }
        }
        let lm = label_components(&img, Connectivity::Four);
        let stats = component_stats(&lm);
        let filtered = filter_by_size(&lm, &stats, 5);
        assert!(!filtered.get(0, 0)); // small blob removed
        assert!(filtered.get(5, 3)); // big blob kept
    }

    #[test]
    fn test_flood_fill_4conn() {
        let mut img = BinaryImage::new(5, 5);
        // L-shape
        img.set(0, 0, true);
        img.set(0, 1, true);
        img.set(0, 2, true);
        img.set(1, 2, true);
        img.set(2, 2, true);
        // Disconnected pixel
        img.set(4, 4, true);

        let filled = flood_fill(&img, 0, 0, Connectivity::Four);
        assert!(filled.get(0, 0));
        assert!(filled.get(2, 2));
        assert!(!filled.get(4, 4)); // Not connected
    }

    #[test]
    fn test_flood_fill_empty_seed() {
        let img = BinaryImage::new(5, 5);
        let filled = flood_fill(&img, 2, 2, Connectivity::Four);
        assert_eq!(filled.data.iter().filter(|&&v| v).count(), 0);
    }

    #[test]
    fn test_8_connectivity() {
        let mut img = BinaryImage::new(3, 3);
        img.set(0, 0, true);
        img.set(1, 1, true); // diagonal neighbor
        let lm = label_components(&img, Connectivity::Eight);
        assert_eq!(lm.num_labels, 1); // Connected via diagonal
    }

    #[test]
    fn test_4_connectivity_diagonal_separate() {
        let mut img = BinaryImage::new(3, 3);
        img.set(0, 0, true);
        img.set(1, 1, true);
        let lm = label_components(&img, Connectivity::Four);
        assert_eq!(lm.num_labels, 2); // Not connected in 4-conn
    }

    #[test]
    fn test_single_component() {
        let mut img = BinaryImage::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                img.set(x, y, true);
            }
        }
        let lm = label_components(&img, Connectivity::Four);
        assert_eq!(lm.num_labels, 1);
    }

    #[test]
    fn test_empty_image() {
        let img = BinaryImage::new(5, 5);
        let lm = label_components(&img, Connectivity::Four);
        assert_eq!(lm.num_labels, 0);
        let stats = component_stats(&lm);
        assert!(stats.is_empty());
    }

    #[test]
    fn test_flood_fill_8conn() {
        let mut img = BinaryImage::new(3, 3);
        img.set(0, 0, true);
        img.set(1, 1, true);
        img.set(2, 2, true);
        let filled = flood_fill(&img, 0, 0, Connectivity::Eight);
        assert!(filled.get(1, 1));
        assert!(filled.get(2, 2));
    }
}
