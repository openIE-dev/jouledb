//! Semantic segmentation masks and operations.
//!
//! Argmax classification, mask overlay, connected-component labelling
//! (union-find), contour extraction, and IoU between masks.

// ── Class definition ────────────────────────────────────────────

/// Definition of a segmentation class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDef {
    pub id: u8,
    pub name: String,
    /// Display colour as [R, G, B].
    pub color: [u8; 3],
}

impl ClassDef {
    pub fn new(id: u8, name: impl Into<String>, color: [u8; 3]) -> Self {
        Self { id, name: name.into(), color }
    }
}

// ── SegmentationMask ────────────────────────────────────────────

/// A per-pixel class-label mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentationMask {
    pub width: usize,
    pub height: usize,
    /// One class label per pixel (row-major).
    pub labels: Vec<u8>,
}

impl SegmentationMask {
    /// Create a mask filled with class 0.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            labels: vec![0; width * height],
        }
    }

    /// Create from existing labels. Panics on length mismatch.
    pub fn from_labels(width: usize, height: usize, labels: Vec<u8>) -> Self {
        assert_eq!(labels.len(), width * height, "label count mismatch");
        Self { width, height, labels }
    }

    /// Get the class label at (x, y).
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.labels[y * self.width + x]
    }

    /// Set the class label at (x, y).
    pub fn set(&mut self, x: usize, y: usize, label: u8) {
        self.labels[y * self.width + x] = label;
    }

    /// Count of pixels assigned to a given class.
    pub fn class_pixel_count(&self, class_id: u8) -> usize {
        self.labels.iter().filter(|l| **l == class_id).count()
    }

    /// Total number of pixels.
    pub fn total_pixels(&self) -> usize {
        self.width * self.height
    }
}

// ── Argmax ──────────────────────────────────────────────────────

/// Produce a segmentation mask by taking argmax across class channels.
///
/// `logits` is shaped `[num_classes][height][width]` in row-major order.
pub fn argmax_mask(
    logits: &[f32],
    num_classes: usize,
    height: usize,
    width: usize,
) -> SegmentationMask {
    assert_eq!(
        logits.len(),
        num_classes * height * width,
        "logits size mismatch"
    );
    let hw = height * width;
    let mut labels = vec![0u8; hw];
    for pixel in 0..hw {
        let mut best_class = 0u8;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let val = logits[c * hw + pixel];
            if val > best_val {
                best_val = val;
                best_class = c as u8;
            }
        }
        labels[pixel] = best_class;
    }
    SegmentationMask { width, height, labels }
}

// ── Overlay blending ────────────────────────────────────────────

/// Blend a segmentation mask onto an RGB image buffer.
///
/// `image` is `[height][width][3]` (RGB). Returns a new blended image.
pub fn overlay_blend(
    image: &[u8],
    mask: &SegmentationMask,
    class_defs: &[ClassDef],
    alpha: f32,
) -> Vec<u8> {
    let n = mask.width * mask.height;
    assert_eq!(image.len(), n * 3, "image size mismatch");
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let label = mask.labels[i];
        let color = class_defs
            .iter()
            .find(|c| c.id == label)
            .map(|c| c.color)
            .unwrap_or([0, 0, 0]);
        for ch in 0..3 {
            let img_val = image[i * 3 + ch] as f32;
            let mask_val = color[ch] as f32;
            out[i * 3 + ch] = (img_val * (1.0 - alpha) + mask_val * alpha) as u8;
        }
    }
    out
}

// ── Connected component labelling (union-find) ──────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Connected component labelling for a given class.
/// Returns a label map where each pixel in the target class gets
/// a component ID (starting from 1). Background pixels get 0.
pub fn connected_components(mask: &SegmentationMask, target_class: u8) -> Vec<u32> {
    let w = mask.width;
    let h = mask.height;
    let n = w * h;
    let mut uf = UnionFind::new(n);

    // Pass 1: union adjacent pixels of target class.
    for y in 0..h {
        for x in 0..w {
            if mask.get(x, y) != target_class {
                continue;
            }
            let idx = y * w + x;
            // Right neighbor.
            if x + 1 < w && mask.get(x + 1, y) == target_class {
                uf.union(idx, idx + 1);
            }
            // Down neighbor.
            if y + 1 < h && mask.get(x, y + 1) == target_class {
                uf.union(idx, idx + w);
            }
        }
    }

    // Pass 2: assign sequential component IDs.
    let mut root_to_id = std::collections::HashMap::new();
    let mut next_id = 1u32;
    let mut result = vec![0u32; n];

    for y in 0..h {
        for x in 0..w {
            if mask.get(x, y) != target_class {
                continue;
            }
            let idx = y * w + x;
            let root = uf.find(idx);
            let cid = root_to_id.entry(root).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            result[idx] = *cid;
        }
    }
    result
}

/// Count distinct connected components for a class.
pub fn component_count(mask: &SegmentationMask, target_class: u8) -> u32 {
    let labels = connected_components(mask, target_class);
    let max = labels.iter().copied().max().unwrap_or(0);
    max
}

// ── Contour extraction ──────────────────────────────────────────

/// Extract border pixels for a given class.
/// A pixel is a border pixel if it belongs to the target class and has
/// at least one 4-connected neighbor that does NOT belong to the class.
pub fn contour_pixels(mask: &SegmentationMask, target_class: u8) -> Vec<(usize, usize)> {
    let w = mask.width;
    let h = mask.height;
    let mut border = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if mask.get(x, y) != target_class {
                continue;
            }
            let is_border = x == 0
                || y == 0
                || x == w - 1
                || y == h - 1
                || mask.get(x.wrapping_sub(1), y) != target_class
                || mask.get(x + 1, y) != target_class
                || mask.get(x, y.wrapping_sub(1)) != target_class
                || mask.get(x, y + 1) != target_class;
            // Guard wrapping: if x==0 or y==0 we already flagged as border.
            if is_border {
                border.push((x, y));
            }
        }
    }
    border
}

// ── IoU between masks ───────────────────────────────────────────

/// Intersection-over-union between two masks for a given class.
pub fn mask_iou(a: &SegmentationMask, b: &SegmentationMask, class_id: u8) -> f64 {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    let mut intersection = 0usize;
    let mut union = 0usize;
    for i in 0..a.labels.len() {
        let in_a = a.labels[i] == class_id;
        let in_b = b.labels[i] == class_id;
        if in_a && in_b {
            intersection += 1;
        }
        if in_a || in_b {
            union += 1;
        }
    }
    if union == 0 { 1.0 } else { intersection as f64 / union as f64 }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_new() {
        let m = SegmentationMask::new(10, 10);
        assert_eq!(m.total_pixels(), 100);
        assert_eq!(m.class_pixel_count(0), 100);
    }

    #[test]
    fn test_mask_get_set() {
        let mut m = SegmentationMask::new(5, 5);
        m.set(2, 3, 7);
        assert_eq!(m.get(2, 3), 7);
        assert_eq!(m.get(0, 0), 0);
    }

    #[test]
    fn test_argmax_mask() {
        // 2 classes, 2x2 image
        // class 0 logits: [1.0, 0.5, 0.2, 0.8]
        // class 1 logits: [0.5, 1.0, 0.9, 0.3]
        let logits = vec![
            1.0, 0.5, 0.2, 0.8, // class 0
            0.5, 1.0, 0.9, 0.3, // class 1
        ];
        let mask = argmax_mask(&logits, 2, 2, 2);
        assert_eq!(mask.get(0, 0), 0); // 1.0 > 0.5
        assert_eq!(mask.get(1, 0), 1); // 0.5 < 1.0
        assert_eq!(mask.get(0, 1), 1); // 0.2 < 0.9
        assert_eq!(mask.get(1, 1), 0); // 0.8 > 0.3
    }

    #[test]
    fn test_overlay_blend() {
        let mask = SegmentationMask::from_labels(2, 1, vec![0, 1]);
        let image = vec![100, 100, 100, 200, 200, 200];
        let classes = vec![
            ClassDef::new(0, "bg", [0, 0, 0]),
            ClassDef::new(1, "fg", [255, 0, 0]),
        ];
        let blended = overlay_blend(&image, &mask, &classes, 0.5);
        assert_eq!(blended.len(), 6);
        // Pixel 0: bg → (100*0.5 + 0*0.5, ...) = 50
        assert_eq!(blended[0], 50);
        // Pixel 1: fg → (200*0.5 + 255*0.5) = 227
        assert_eq!(blended[3], 227);
    }

    #[test]
    fn test_connected_components() {
        // 4x3 mask with two separate class-1 blobs.
        let labels = vec![
            1, 1, 0, 0,
            0, 0, 0, 0,
            0, 0, 1, 1,
        ];
        let mask = SegmentationMask::from_labels(4, 3, labels);
        let count = component_count(&mask, 1);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_connected_components_single() {
        let labels = vec![
            1, 1,
            1, 1,
        ];
        let mask = SegmentationMask::from_labels(2, 2, labels);
        assert_eq!(component_count(&mask, 1), 1);
    }

    #[test]
    fn test_contour_pixels() {
        // 3x3 block of class 1 — only border pixels.
        let labels = vec![
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
        ];
        let mask = SegmentationMask::from_labels(3, 3, labels);
        let contour = contour_pixels(&mask, 1);
        // 8 pixels are on image boundary (edges/corners). Center (1,1) has all 4
        // neighbors = 1 and is not on the image edge, so it's not a contour pixel.
        assert_eq!(contour.len(), 8);

        // Now test a 5x5 with a 3x3 interior block
        let mut mask2 = SegmentationMask::new(5, 5);
        for y in 1..4 {
            for x in 1..4 {
                mask2.set(x, y, 1);
            }
        }
        let contour2 = contour_pixels(&mask2, 1);
        // 9 pixels of class 1, but (2,2) center has all neighbors = 1
        assert_eq!(contour2.len(), 8);
    }

    #[test]
    fn test_mask_iou_identical() {
        let m = SegmentationMask::from_labels(3, 3, vec![1; 9]);
        assert!((mask_iou(&m, &m, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mask_iou_no_overlap() {
        let a = SegmentationMask::from_labels(2, 1, vec![1, 0]);
        let b = SegmentationMask::from_labels(2, 1, vec![0, 1]);
        assert_eq!(mask_iou(&a, &b, 1), 0.0);
    }

    #[test]
    fn test_mask_iou_partial() {
        let a = SegmentationMask::from_labels(4, 1, vec![1, 1, 0, 0]);
        let b = SegmentationMask::from_labels(4, 1, vec![0, 1, 1, 0]);
        // intersection=1, union=3
        assert!((mask_iou(&a, &b, 1) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_class_def() {
        let cd = ClassDef::new(5, "road", [128, 64, 128]);
        assert_eq!(cd.id, 5);
        assert_eq!(cd.name, "road");
        assert_eq!(cd.color, [128, 64, 128]);
    }
}
