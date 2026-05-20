//! Point operations — point clustering (DBSCAN-like), nearest point on line,
//! point-in-bbox test, point density estimation (kernel), point thinning,
//! convex hull of points.
//!
//! Pure-Rust spatial point algorithms. All coordinates are `f64`.

use std::fmt;

// ── Point2 ─────────────────────────────────────────────────────

/// A 2-D point in Cartesian space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn dist_sq(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    pub fn dist(self, other: Self) -> f64 {
        self.dist_sq(other).sqrt()
    }
}

impl fmt::Display for Point2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6}, {:.6})", self.x, self.y)
    }
}

// ── Errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PointError {
    TooFewPoints(usize),
    InvalidParameter(String),
    EmptyInput,
}

impl fmt::Display for PointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewPoints(n) => write!(f, "need >=1 point, got {n}"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for PointError {}

// ── BBox ───────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    /// Build a bbox from a set of points.
    pub fn from_points(pts: &[Point2]) -> Option<Self> {
        if pts.is_empty() {
            return None;
        }
        let mut b = Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        };
        for p in pts {
            if p.x < b.min_x { b.min_x = p.x; }
            if p.y < b.min_y { b.min_y = p.y; }
            if p.x > b.max_x { b.max_x = p.x; }
            if p.y > b.max_y { b.max_y = p.y; }
        }
        Some(b)
    }

    /// Test whether a point lies inside (inclusive) the bounding box.
    pub fn contains(&self, p: Point2) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }

    /// Width of the box.
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height of the box.
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Area of the box.
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Center point.
    pub fn center(&self) -> Point2 {
        Point2::new(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )
    }

    /// Expand the box by `margin` on every side.
    pub fn expand(&self, margin: f64) -> Self {
        Self {
            min_x: self.min_x - margin,
            min_y: self.min_y - margin,
            max_x: self.max_x + margin,
            max_y: self.max_y + margin,
        }
    }
}

impl fmt::Display for BBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BBox[({:.4},{:.4})–({:.4},{:.4})]",
            self.min_x, self.min_y, self.max_x, self.max_y
        )
    }
}

// ── DBSCAN-like clustering ─────────────────────────────────────

/// Label returned per point after clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterLabel {
    /// Assigned to cluster with this id (0-based).
    Cluster(usize),
    /// Noise point (not reachable within eps by min_pts neighbours).
    Noise,
}

impl fmt::Display for ClusterLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cluster(id) => write!(f, "cluster-{id}"),
            Self::Noise => write!(f, "noise"),
        }
    }
}

/// DBSCAN-style density-based clustering.
///
/// Returns a label per input point.
pub fn dbscan(points: &[Point2], eps: f64, min_pts: usize) -> Result<Vec<ClusterLabel>, PointError> {
    if points.is_empty() {
        return Err(PointError::EmptyInput);
    }
    if eps <= 0.0 {
        return Err(PointError::InvalidParameter(format!("eps must be >0, got {eps}")));
    }
    if min_pts == 0 {
        return Err(PointError::InvalidParameter("min_pts must be >0".into()));
    }
    let n = points.len();
    let eps_sq = eps * eps;
    let mut labels = vec![None; n];
    let mut cluster_id: usize = 0;

    for i in 0..n {
        if labels[i].is_some() {
            continue;
        }
        let neighbours: Vec<usize> = (0..n)
            .filter(|j| points[i].dist_sq(points[*j]) <= eps_sq)
            .collect();

        if neighbours.len() < min_pts {
            labels[i] = Some(ClusterLabel::Noise);
            continue;
        }

        let cid = cluster_id;
        cluster_id += 1;
        labels[i] = Some(ClusterLabel::Cluster(cid));

        let mut queue = neighbours;
        let mut qi = 0;
        while qi < queue.len() {
            let q = queue[qi];
            qi += 1;
            if labels[q] == Some(ClusterLabel::Noise) {
                labels[q] = Some(ClusterLabel::Cluster(cid));
            }
            if labels[q].is_some() {
                continue;
            }
            labels[q] = Some(ClusterLabel::Cluster(cid));
            let q_neighbours: Vec<usize> = (0..n)
                .filter(|j| points[q].dist_sq(points[*j]) <= eps_sq)
                .collect();
            if q_neighbours.len() >= min_pts {
                for nb in q_neighbours {
                    if !queue.contains(&nb) {
                        queue.push(nb);
                    }
                }
            }
        }
    }

    Ok(labels.into_iter().map(|l| l.unwrap()).collect())
}

/// Count the number of distinct clusters (excluding noise).
pub fn cluster_count(labels: &[ClusterLabel]) -> usize {
    let mut max_id = None;
    for l in labels {
        if let ClusterLabel::Cluster(id) = l {
            match max_id {
                Some(m) if *id > m => max_id = Some(*id),
                None => max_id = Some(*id),
                _ => {}
            }
        }
    }
    max_id.map_or(0, |m| m + 1)
}

// ── Point density estimation (Gaussian kernel) ─────────────────

/// Gaussian kernel density estimate at query point `q`.
pub fn kernel_density(points: &[Point2], q: Point2, bandwidth: f64) -> Result<f64, PointError> {
    if points.is_empty() {
        return Err(PointError::EmptyInput);
    }
    if bandwidth <= 0.0 {
        return Err(PointError::InvalidParameter(format!(
            "bandwidth must be >0, got {bandwidth}"
        )));
    }
    let n = points.len() as f64;
    let h2 = bandwidth * bandwidth;
    let norm = 1.0 / (n * 2.0 * std::f64::consts::PI * h2);
    let mut sum = 0.0;
    for p in points {
        let d2 = q.dist_sq(*p);
        sum += (-d2 / (2.0 * h2)).exp();
    }
    Ok(norm * sum)
}

/// Evaluate density on a grid, returning `(grid_points, densities)`.
pub fn density_grid(
    points: &[Point2],
    bbox: &BBox,
    resolution: usize,
    bandwidth: f64,
) -> Result<(Vec<Point2>, Vec<f64>), PointError> {
    if points.is_empty() {
        return Err(PointError::EmptyInput);
    }
    if resolution == 0 {
        return Err(PointError::InvalidParameter("resolution must be >0".into()));
    }
    let dx = bbox.width() / resolution as f64;
    let dy = bbox.height() / resolution as f64;
    let mut grid_pts = Vec::new();
    let mut densities = Vec::new();
    for iy in 0..=resolution {
        for ix in 0..=resolution {
            let q = Point2::new(bbox.min_x + ix as f64 * dx, bbox.min_y + iy as f64 * dy);
            let d = kernel_density(points, q, bandwidth)?;
            grid_pts.push(q);
            densities.push(d);
        }
    }
    Ok((grid_pts, densities))
}

// ── Point thinning ─────────────────────────────────────────────

/// Thin a point set so no two retained points are closer than `min_dist`.
/// Greedy: first point accepted wins.
pub fn thin_points(points: &[Point2], min_dist: f64) -> Result<Vec<Point2>, PointError> {
    if points.is_empty() {
        return Err(PointError::EmptyInput);
    }
    if min_dist <= 0.0 {
        return Err(PointError::InvalidParameter(format!(
            "min_dist must be >0, got {min_dist}"
        )));
    }
    let min_sq = min_dist * min_dist;
    let mut kept: Vec<Point2> = Vec::new();
    for &p in points {
        let too_close = kept.iter().any(|k| k.dist_sq(p) < min_sq);
        if !too_close {
            kept.push(p);
        }
    }
    Ok(kept)
}

// ── Convex hull (Andrew monotone chain) ────────────────────────

fn hull_cross(o: Point2, a: Point2, b: Point2) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Convex hull of a point set using Andrew's monotone chain.
/// Returns vertices in counter-clockwise order.
pub fn convex_hull(points: &[Point2]) -> Result<Vec<Point2>, PointError> {
    if points.len() < 3 {
        return Err(PointError::TooFewPoints(points.len()));
    }
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));

    let mut lower: Vec<Point2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2
            && hull_cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Point2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2
            && hull_cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.append(&mut upper);
    Ok(lower)
}

// ── Nearest point on segment ───────────────────────────────────

/// Nearest point on segment `a–b` to query `p`.
pub fn nearest_on_segment(p: Point2, a: Point2, b: Point2) -> Point2 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-15 {
        return a;
    }
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    Point2::new(a.x + t * dx, a.y + t * dy)
}

/// Nearest point on a polyline (sequence of segments) to query `p`.
pub fn nearest_on_polyline(p: Point2, line: &[Point2]) -> Result<Point2, PointError> {
    if line.len() < 2 {
        return Err(PointError::TooFewPoints(line.len()));
    }
    let mut best = line[0];
    let mut best_d = f64::INFINITY;
    for i in 0..line.len() - 1 {
        let np = nearest_on_segment(p, line[i], line[i + 1]);
        let d = p.dist_sq(np);
        if d < best_d {
            best_d = d;
            best = np;
        }
    }
    Ok(best)
}

// ── PointSet builder ───────────────────────────────────────────

/// Builder for assembling point collections.
#[derive(Debug, Clone)]
pub struct PointSetBuilder {
    points: Vec<Point2>,
}

impl PointSetBuilder {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn with_point(mut self, x: f64, y: f64) -> Self {
        self.points.push(Point2::new(x, y));
        self
    }

    pub fn with_points(mut self, pts: &[(f64, f64)]) -> Self {
        for &(x, y) in pts {
            self.points.push(Point2::new(x, y));
        }
        self
    }

    pub fn with_grid(mut self, bbox: &BBox, nx: usize, ny: usize) -> Self {
        let dx = if nx > 1 { bbox.width() / (nx - 1) as f64 } else { 0.0 };
        let dy = if ny > 1 { bbox.height() / (ny - 1) as f64 } else { 0.0 };
        for iy in 0..ny {
            for ix in 0..nx {
                self.points.push(Point2::new(
                    bbox.min_x + ix as f64 * dx,
                    bbox.min_y + iy as f64 * dy,
                ));
            }
        }
        self
    }

    pub fn build(self) -> Vec<Point2> {
        self.points
    }
}

impl Default for PointSetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PointSetBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PointSetBuilder({} points)", self.points.len())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_bbox_contains() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(b.contains(Point2::new(5.0, 5.0)));
        assert!(!b.contains(Point2::new(11.0, 5.0)));
    }

    #[test]
    fn test_bbox_from_points() {
        let pts = vec![Point2::new(1.0, 2.0), Point2::new(5.0, 8.0)];
        let b = BBox::from_points(&pts).unwrap();
        assert!(approx(b.min_x, 1.0));
        assert!(approx(b.max_y, 8.0));
    }

    #[test]
    fn test_bbox_area() {
        let b = BBox::new(0.0, 0.0, 3.0, 4.0);
        assert!(approx(b.area(), 12.0));
    }

    #[test]
    fn test_bbox_center() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        let c = b.center();
        assert!(approx(c.x, 5.0));
    }

    #[test]
    fn test_bbox_expand() {
        let b = BBox::new(1.0, 1.0, 5.0, 5.0);
        let e = b.expand(1.0);
        assert!(approx(e.min_x, 0.0));
        assert!(approx(e.max_x, 6.0));
    }

    #[test]
    fn test_bbox_display() {
        let b = BBox::new(0.0, 0.0, 1.0, 1.0);
        assert!(format!("{b}").contains("BBox"));
    }

    #[test]
    fn test_dbscan_two_clusters() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.1, 0.0),
            Point2::new(0.0, 0.1),
            Point2::new(10.0, 10.0),
            Point2::new(10.1, 10.0),
            Point2::new(10.0, 10.1),
        ];
        let labels = dbscan(&pts, 0.5, 2).unwrap();
        assert_eq!(cluster_count(&labels), 2);
    }

    #[test]
    fn test_dbscan_noise() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(100.0, 100.0),
        ];
        let labels = dbscan(&pts, 1.0, 2).unwrap();
        assert!(labels.iter().all(|l| *l == ClusterLabel::Noise));
    }

    #[test]
    fn test_dbscan_empty() {
        assert!(dbscan(&[], 1.0, 2).is_err());
    }

    #[test]
    fn test_dbscan_invalid_eps() {
        let pts = vec![Point2::new(0.0, 0.0)];
        assert!(dbscan(&pts, -1.0, 2).is_err());
    }

    #[test]
    fn test_kernel_density_basic() {
        let pts = vec![Point2::new(0.0, 0.0)];
        let d = kernel_density(&pts, Point2::new(0.0, 0.0), 1.0).unwrap();
        assert!(d > 0.0);
    }

    #[test]
    fn test_kernel_density_empty() {
        assert!(kernel_density(&[], Point2::new(0.0, 0.0), 1.0).is_err());
    }

    #[test]
    fn test_density_grid_basic() {
        let pts = vec![Point2::new(0.5, 0.5)];
        let bbox = BBox::new(0.0, 0.0, 1.0, 1.0);
        let (gp, ds) = density_grid(&pts, &bbox, 2, 0.5).unwrap();
        assert_eq!(gp.len(), 9); // (2+1)^2
        assert!(ds.iter().all(|d| *d >= 0.0));
    }

    #[test]
    fn test_thin_points() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.01, 0.0),
            Point2::new(1.0, 0.0),
        ];
        let kept = thin_points(&pts, 0.5).unwrap();
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_thin_points_empty() {
        assert!(thin_points(&[], 1.0).is_err());
    }

    #[test]
    fn test_convex_hull_basic() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 0.5),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let hull = convex_hull(&pts).unwrap();
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_convex_hull_too_few() {
        assert!(convex_hull(&[Point2::new(0.0, 0.0)]).is_err());
    }

    #[test]
    fn test_nearest_on_segment() {
        let np = nearest_on_segment(
            Point2::new(0.5, 1.0),
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
        );
        assert!(approx(np.x, 0.5));
        assert!(approx(np.y, 0.0));
    }

    #[test]
    fn test_nearest_on_polyline() {
        let line = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
        ];
        let np = nearest_on_polyline(Point2::new(3.0, 1.0), &line).unwrap();
        assert!(approx(np.x, 2.0));
        assert!(approx(np.y, 1.0));
    }

    #[test]
    fn test_point_set_builder() {
        let pts = PointSetBuilder::new()
            .with_point(0.0, 0.0)
            .with_points(&[(1.0, 0.0), (2.0, 0.0)])
            .build();
        assert_eq!(pts.len(), 3);
    }

    #[test]
    fn test_point_set_builder_grid() {
        let bbox = BBox::new(0.0, 0.0, 1.0, 1.0);
        let pts = PointSetBuilder::new().with_grid(&bbox, 3, 3).build();
        assert_eq!(pts.len(), 9);
    }

    #[test]
    fn test_cluster_label_display() {
        assert_eq!(format!("{}", ClusterLabel::Cluster(0)), "cluster-0");
        assert_eq!(format!("{}", ClusterLabel::Noise), "noise");
    }
}
