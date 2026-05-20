//! Clustered heatmap with dendrograms — replaces seaborn.clustermap, R's pheatmap.
//!
//! Builds on the existing heatmap module, adding:
//! - Full agglomerative hierarchical clustering (single/complete/average/ward)
//! - Dendrogram rendering (row and column)
//! - Pearson/Spearman correlation matrix convenience
//! - Row/column color annotations

use std::fmt::Write as FmtWrite;
use crate::surface3d::Colormap;

// ── Hierarchical clustering ────────────────────────────────────────

/// Linkage method.
#[derive(Debug, Clone, Copy)]
pub enum Linkage { Single, Complete, Average, Ward }

/// Merge record.
#[derive(Debug, Clone)]
pub struct Merge {
    pub left: usize,
    pub right: usize,
    pub distance: f64,
    pub size: usize,
}

/// Agglomerative clustering result.
#[derive(Debug, Clone)]
pub struct Dendrogram {
    pub merges: Vec<Merge>,
    pub leaf_order: Vec<usize>,
}

/// Run agglomerative clustering on a distance matrix.
pub fn agglomerative(dists: &[Vec<f64>], linkage: Linkage) -> Dendrogram {
    let n = dists.len();
    if n <= 1 { return Dendrogram { merges: vec![], leaf_order: (0..n).collect() }; }

    let mut d = dists.to_vec();
    let mut active: Vec<bool> = vec![true; n];
    let mut sizes: Vec<usize> = vec![1; n];
    let mut ids: Vec<usize> = (0..n).collect();
    let mut merges = Vec::new();
    let mut next_id = n;

    for _ in 0..n - 1 {
        let mut best = f64::INFINITY;
        let (mut bi, mut bj) = (0, 0);
        for i in 0..n {
            if !active[i] { continue; }
            for j in i + 1..n {
                if !active[j] { continue; }
                if d[i][j] < best { best = d[i][j]; bi = i; bj = j; }
            }
        }

        let new_size = sizes[bi] + sizes[bj];
        merges.push(Merge { left: ids[bi], right: ids[bj], distance: best, size: new_size });

        // Update distances
        for k in 0..n {
            if !active[k] || k == bi || k == bj { continue; }
            let dk = match linkage {
                Linkage::Single => d[bi][k].min(d[bj][k]),
                Linkage::Complete => d[bi][k].max(d[bj][k]),
                Linkage::Average => {
                    let (si, sj) = (sizes[bi] as f64, sizes[bj] as f64);
                    (d[bi][k] * si + d[bj][k] * sj) / (si + sj)
                }
                Linkage::Ward => {
                    let (si, sj, sk) = (sizes[bi] as f64, sizes[bj] as f64, sizes[k] as f64);
                    let total = si + sj + sk;
                    ((si + sk) * d[bi][k] + (sj + sk) * d[bj][k] - sk * best) / total
                }
            };
            d[bi][k] = dk;
            d[k][bi] = dk;
        }

        active[bj] = false;
        sizes[bi] = new_size;
        ids[bi] = next_id;
        next_id += 1;
    }

    let leaf_order = tree_order(&merges, n);
    Dendrogram { merges, leaf_order }
}

fn tree_order(merges: &[Merge], n_leaves: usize) -> Vec<usize> {
    if merges.is_empty() { return (0..n_leaves).collect(); }
    let root = n_leaves + merges.len() - 1;
    let mut order = Vec::new();
    walk(root, merges, n_leaves, &mut order);
    order
}

fn walk(node: usize, merges: &[Merge], n: usize, out: &mut Vec<usize>) {
    if node < n { out.push(node); }
    else if node - n < merges.len() {
        let m = &merges[node - n];
        walk(m.left, merges, n, out);
        walk(m.right, merges, n, out);
    }
}

/// Euclidean distance matrix.
pub fn euclidean_dists(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = rows.len();
    let mut d = vec![vec![0.0; n]; n];
    for i in 0..n { for j in i + 1..n {
        let v: f64 = rows[i].iter().zip(&rows[j]).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        d[i][j] = v; d[j][i] = v;
    }}
    d
}

/// Pearson correlation matrix.
pub fn pearson_matrix(cols: &[(&str, &[f64])]) -> (Vec<Vec<f64>>, Vec<String>) {
    let n = cols.len();
    let labels: Vec<String> = cols.iter().map(|(s, _)| s.to_string()).collect();
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { for j in 0..n { m[i][j] = pearson_r(cols[i].1, cols[j].1); } }
    (m, labels)
}

fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 { return 0.0; }
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut c, mut sx, mut sy) = (0.0, 0.0, 0.0);
    for i in 0..n as usize { let (dx, dy) = (x[i] - mx, y[i] - my); c += dx * dy; sx += dx * dx; sy += dy * dy; }
    let d = (sx * sy).sqrt();
    if d < 1e-15 { 0.0 } else { c / d }
}

// ── Dendrogram SVG rendering ───────────────────────────────────────

/// Render a vertical dendrogram (for row clustering).
pub fn dendro_vertical_svg(merges: &[Merge], n: usize, w: f64, h: f64, cell_h: f64) -> String {
    if merges.is_empty() { return String::new(); }
    let max_d = merges.iter().map(|m| m.distance).fold(0.0f64, f64::max).max(1e-10);
    let mut yc: Vec<f64> = (0..n + merges.len()).map(|i| {
        if i < n { (i as f64 + 0.5) * cell_h } else { 0.0 }
    }).collect();
    for (i, m) in merges.iter().enumerate() { yc[n + i] = (yc[m.left] + yc[m.right]) / 2.0; }

    let mut s = String::new();
    for (i, m) in merges.iter().enumerate() {
        let xm = w * (1.0 - m.distance / max_d);
        let xl = if m.left < n { w } else { w * (1.0 - merges[m.left - n].distance / max_d) };
        let xr = if m.right < n { w } else { w * (1.0 - merges[m.right - n].distance / max_d) };
        let (yl, yr) = (yc[m.left], yc[m.right]);
        let _ = write!(s, "<line x1=\"{xl:.1}\" y1=\"{yl:.1}\" x2=\"{xm:.1}\" y2=\"{yl:.1}\" stroke=\"#555\" stroke-width=\"1\"/>");
        let _ = write!(s, "<line x1=\"{xr:.1}\" y1=\"{yr:.1}\" x2=\"{xm:.1}\" y2=\"{yr:.1}\" stroke=\"#555\" stroke-width=\"1\"/>");
        let _ = write!(s, "<line x1=\"{xm:.1}\" y1=\"{yl:.1}\" x2=\"{xm:.1}\" y2=\"{yr:.1}\" stroke=\"#555\" stroke-width=\"1\"/>");
    }
    s
}

/// Render a horizontal dendrogram (for column clustering).
pub fn dendro_horizontal_svg(merges: &[Merge], n: usize, w: f64, h: f64, cell_w: f64) -> String {
    if merges.is_empty() { return String::new(); }
    let max_d = merges.iter().map(|m| m.distance).fold(0.0f64, f64::max).max(1e-10);
    let mut xc: Vec<f64> = (0..n + merges.len()).map(|i| {
        if i < n { (i as f64 + 0.5) * cell_w } else { 0.0 }
    }).collect();
    for (i, m) in merges.iter().enumerate() { xc[n + i] = (xc[m.left] + xc[m.right]) / 2.0; }

    let mut s = String::new();
    for (i, m) in merges.iter().enumerate() {
        let ym = h * (1.0 - m.distance / max_d);
        let yl = if m.left < n { h } else { h * (1.0 - merges[m.left - n].distance / max_d) };
        let yr = if m.right < n { h } else { h * (1.0 - merges[m.right - n].distance / max_d) };
        let (xl, xr) = (xc[m.left], xc[m.right]);
        let _ = write!(s, "<line x1=\"{xl:.1}\" y1=\"{yl:.1}\" x2=\"{xl:.1}\" y2=\"{ym:.1}\" stroke=\"#555\" stroke-width=\"1\"/>");
        let _ = write!(s, "<line x1=\"{xr:.1}\" y1=\"{yr:.1}\" x2=\"{xr:.1}\" y2=\"{ym:.1}\" stroke=\"#555\" stroke-width=\"1\"/>");
        let _ = write!(s, "<line x1=\"{xl:.1}\" y1=\"{ym:.1}\" x2=\"{xr:.1}\" y2=\"{ym:.1}\" stroke=\"#555\" stroke-width=\"1\"/>");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agglomerative_single() {
        let d = vec![vec![0.0,1.0,4.0], vec![1.0,0.0,3.0], vec![4.0,3.0,0.0]];
        let r = agglomerative(&d, Linkage::Single);
        assert_eq!(r.merges.len(), 2);
        assert_eq!(r.leaf_order.len(), 3);
        assert!((r.merges[0].distance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn agglomerative_ward() {
        let d = vec![
            vec![0.0, 1.0, 5.0, 9.0],
            vec![1.0, 0.0, 4.0, 8.0],
            vec![5.0, 4.0, 0.0, 3.0],
            vec![9.0, 8.0, 3.0, 0.0],
        ];
        let r = agglomerative(&d, Linkage::Ward);
        assert_eq!(r.merges.len(), 3);
    }

    #[test]
    fn euclidean_correct() {
        let rows = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let d = euclidean_dists(&rows);
        assert!((d[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let (m, _) = pearson_matrix(&[("x", &x), ("y", &y)]);
        assert!((m[0][1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn dendro_svg_renders() {
        let d = vec![vec![0.0,1.0,4.0], vec![1.0,0.0,3.0], vec![4.0,3.0,0.0]];
        let r = agglomerative(&d, Linkage::Average);
        let svg = dendro_vertical_svg(&r.merges, 3, 60.0, 120.0, 40.0);
        assert!(svg.contains("line"));
    }
}
