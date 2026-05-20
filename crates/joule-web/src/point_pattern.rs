//! Point Pattern Analysis — quadrat analysis, nearest-neighbor index
//! (Clark-Evans), Ripley's K function, kernel density estimation,
//! point pattern simulation (CSR), MonteCarloEnvelope.
//!
//! Pure-Rust spatial point pattern tools, f64 precision, std-only.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PointPatternError {
    TooFewPoints(usize),
    InvalidExtent(String),
    InvalidParameter(String),
    InvalidSimulations(usize),
}

impl fmt::Display for PointPatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewPoints(n) => write!(f, "need >= 2 points, got {n}"),
            Self::InvalidExtent(s) => write!(f, "invalid study area extent: {s}"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::InvalidSimulations(n) => write!(f, "simulations must be > 0, got {n}"),
        }
    }
}

impl std::error::Error for PointPatternError {}

// ── Point ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4})", self.x, self.y)
    }
}

// ── Study area extent ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extent {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl Extent {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Result<Self, PointPatternError> {
        if x_max <= x_min || y_max <= y_min {
            return Err(PointPatternError::InvalidExtent(format!(
                "({x_min},{y_min})-({x_max},{y_max})"
            )));
        }
        Ok(Self { x_min, y_min, x_max, y_max })
    }

    pub fn area(&self) -> f64 {
        (self.x_max - self.x_min) * (self.y_max - self.y_min)
    }

    pub fn width(&self) -> f64 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> f64 {
        self.y_max - self.y_min
    }

    /// Fit an extent tightly around a set of points with optional buffer.
    pub fn from_points(points: &[Point], buffer: f64) -> Result<Self, PointPatternError> {
        if points.len() < 2 {
            return Err(PointPatternError::TooFewPoints(points.len()));
        }
        let x_min = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) - buffer;
        let y_min = points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) - buffer;
        let x_max = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) + buffer;
        let y_max = points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) + buffer;
        Self::new(x_min, y_min, x_max, y_max)
    }

    pub fn contains(&self, p: &Point) -> bool {
        p.x >= self.x_min && p.x <= self.x_max && p.y >= self.y_min && p.y <= self.y_max
    }
}

impl fmt::Display for Extent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Extent({:.2}..{:.2}, {:.2}..{:.2})",
            self.x_min, self.x_max, self.y_min, self.y_max
        )
    }
}

// ── Simple PRNG ─────────────────────────────────────────────────

struct Prng {
    state: u64,
}

impl Prng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ── Quadrat analysis ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QuadratResult {
    pub rows: usize,
    pub cols: usize,
    pub counts: Vec<usize>,
    pub chi_squared: f64,
    pub df: usize,
    pub mean_count: f64,
    pub variance: f64,
    pub vmi_ratio: f64,
}

impl fmt::Display for QuadratResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Quadrat({}x{}, chi2={:.4}, VMI={:.4})",
            self.rows, self.cols, self.chi_squared, self.vmi_ratio
        )
    }
}

pub fn quadrat_analysis(
    points: &[Point],
    extent: &Extent,
    rows: usize,
    cols: usize,
) -> Result<QuadratResult, PointPatternError> {
    if points.len() < 2 {
        return Err(PointPatternError::TooFewPoints(points.len()));
    }
    if rows == 0 || cols == 0 {
        return Err(PointPatternError::InvalidParameter("rows and cols must be > 0".into()));
    }

    let cell_w = extent.width() / cols as f64;
    let cell_h = extent.height() / rows as f64;
    let k = rows * cols;
    let mut counts = vec![0usize; k];

    for p in points {
        if !extent.contains(p) {
            continue;
        }
        let c = ((p.x - extent.x_min) / cell_w).min(cols as f64 - 1.0) as usize;
        let r = ((p.y - extent.y_min) / cell_h).min(rows as f64 - 1.0) as usize;
        counts[r * cols + c] += 1;
    }

    let n = points.len() as f64;
    let expected = n / k as f64;
    let chi2: f64 = counts
        .iter()
        .map(|c| {
            let diff = *c as f64 - expected;
            diff * diff / expected
        })
        .sum();
    let mean_count = expected;
    let variance = counts.iter().map(|c| (*c as f64 - mean_count).powi(2)).sum::<f64>() / k as f64;
    let vmi = if mean_count > 1e-15 { variance / mean_count } else { 0.0 };

    Ok(QuadratResult {
        rows,
        cols,
        counts,
        chi_squared: chi2,
        df: k - 1,
        mean_count,
        variance,
        vmi_ratio: vmi,
    })
}

// ── Clark-Evans nearest-neighbor index ──────────────────────────

#[derive(Debug, Clone)]
pub struct ClarkEvansResult {
    pub observed_mean_nnd: f64,
    pub expected_mean_nnd: f64,
    pub r_index: f64,
    pub z_score: f64,
    pub pattern: PatternType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternType {
    Clustered,
    Random,
    Dispersed,
}

impl fmt::Display for PatternType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clustered => write!(f, "clustered"),
            Self::Random => write!(f, "random"),
            Self::Dispersed => write!(f, "dispersed"),
        }
    }
}

impl fmt::Display for ClarkEvansResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClarkEvans(R={:.4}, z={:.4}, pattern={})",
            self.r_index, self.z_score, self.pattern
        )
    }
}

pub fn clark_evans(
    points: &[Point],
    extent: &Extent,
) -> Result<ClarkEvansResult, PointPatternError> {
    let n = points.len();
    if n < 2 {
        return Err(PointPatternError::TooFewPoints(n));
    }
    let area = extent.area();

    let mut nnd_sum = 0.0;
    for i in 0..n {
        let mut min_d = f64::INFINITY;
        for j in 0..n {
            if i != j {
                let d = points[i].distance_to(&points[j]);
                if d < min_d {
                    min_d = d;
                }
            }
        }
        nnd_sum += min_d;
    }
    let observed = nnd_sum / n as f64;
    let density = n as f64 / area;
    let expected = 0.5 / density.sqrt();
    let se = 0.26136 / (n as f64 * density).sqrt();
    let r = observed / expected;
    let z = (observed - expected) / se;

    let pattern = if z < -1.96 {
        PatternType::Clustered
    } else if z > 1.96 {
        PatternType::Dispersed
    } else {
        PatternType::Random
    };

    Ok(ClarkEvansResult {
        observed_mean_nnd: observed,
        expected_mean_nnd: expected,
        r_index: r,
        z_score: z,
        pattern,
    })
}

// ── Ripley's K function ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RipleyKResult {
    pub distances: Vec<f64>,
    pub k_observed: Vec<f64>,
    pub k_expected: Vec<f64>,
    pub l_observed: Vec<f64>,
}

impl fmt::Display for RipleyKResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RipleyK({} distance bands)", self.distances.len())
    }
}

pub fn ripleys_k(
    points: &[Point],
    extent: &Extent,
    max_dist: f64,
    n_bins: usize,
) -> Result<RipleyKResult, PointPatternError> {
    let n = points.len();
    if n < 2 {
        return Err(PointPatternError::TooFewPoints(n));
    }
    if n_bins == 0 || max_dist <= 0.0 {
        return Err(PointPatternError::InvalidParameter("max_dist and n_bins must be > 0".into()));
    }

    let area = extent.area();
    let lambda = n as f64 / area;
    let step = max_dist / n_bins as f64;

    let mut distances = Vec::with_capacity(n_bins);
    let mut k_obs = Vec::with_capacity(n_bins);
    let mut k_exp = Vec::with_capacity(n_bins);
    let mut l_obs = Vec::with_capacity(n_bins);

    for b in 1..=n_bins {
        let d = step * b as f64;
        distances.push(d);

        let mut count = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j && points[i].distance_to(&points[j]) <= d {
                    count += 1.0;
                }
            }
        }
        let k = area * count / (n as f64 * n as f64);
        k_obs.push(k);
        let ke = std::f64::consts::PI * d * d;
        k_exp.push(ke);
        l_obs.push((k / std::f64::consts::PI).sqrt() - d);
    }

    Ok(RipleyKResult {
        distances,
        k_observed: k_obs,
        k_expected: k_exp,
        l_observed: l_obs,
    })
}

// ── Kernel density estimation ───────────────────────────────────

#[derive(Debug, Clone)]
pub struct KdeResult {
    pub grid_x: Vec<f64>,
    pub grid_y: Vec<f64>,
    pub density: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl fmt::Display for KdeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KDE({}x{} grid)", self.rows, self.cols)
    }
}

pub fn kernel_density(
    points: &[Point],
    extent: &Extent,
    bandwidth: f64,
    grid_size: usize,
) -> Result<KdeResult, PointPatternError> {
    if points.len() < 2 {
        return Err(PointPatternError::TooFewPoints(points.len()));
    }
    if bandwidth <= 0.0 || grid_size == 0 {
        return Err(PointPatternError::InvalidParameter(
            "bandwidth and grid_size must be > 0".into(),
        ));
    }

    let step_x = extent.width() / grid_size as f64;
    let step_y = extent.height() / grid_size as f64;
    let mut grid_x = Vec::with_capacity(grid_size);
    let mut grid_y = Vec::with_capacity(grid_size);
    for c in 0..grid_size {
        grid_x.push(extent.x_min + (c as f64 + 0.5) * step_x);
    }
    for r in 0..grid_size {
        grid_y.push(extent.y_min + (r as f64 + 0.5) * step_y);
    }

    let h2 = bandwidth * bandwidth;
    let norm = 1.0 / (2.0 * std::f64::consts::PI * h2 * points.len() as f64);
    let mut density = vec![0.0; grid_size * grid_size];

    for r in 0..grid_size {
        for c in 0..grid_size {
            let gx = grid_x[c];
            let gy = grid_y[r];
            let mut sum = 0.0;
            for p in points {
                let dx = gx - p.x;
                let dy = gy - p.y;
                sum += (-(dx * dx + dy * dy) / (2.0 * h2)).exp();
            }
            density[r * grid_size + c] = sum * norm;
        }
    }

    Ok(KdeResult { grid_x, grid_y, density, rows: grid_size, cols: grid_size })
}

// ── CSR simulation ──────────────────────────────────────────────

pub fn simulate_csr(extent: &Extent, n: usize, seed: u64) -> Vec<Point> {
    let mut rng = Prng::new(seed);
    let w = extent.width();
    let h = extent.height();
    (0..n)
        .map(|_| {
            Point::new(
                extent.x_min + rng.next_f64() * w,
                extent.y_min + rng.next_f64() * h,
            )
        })
        .collect()
}

// ── Monte Carlo envelope ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MonteCarloEnvelope {
    pub distances: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub observed: Vec<f64>,
}

impl fmt::Display for MonteCarloEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MonteCarloEnvelope({} bands)", self.distances.len())
    }
}

pub fn monte_carlo_envelope(
    points: &[Point],
    extent: &Extent,
    max_dist: f64,
    n_bins: usize,
    simulations: usize,
    seed: u64,
) -> Result<MonteCarloEnvelope, PointPatternError> {
    if simulations == 0 {
        return Err(PointPatternError::InvalidSimulations(simulations));
    }
    let observed = ripleys_k(points, extent, max_dist, n_bins)?;
    let n = points.len();
    let mut rng = Prng::new(seed);

    let mut all_k: Vec<Vec<f64>> = vec![Vec::with_capacity(simulations); n_bins];

    for _ in 0..simulations {
        let sim_pts = simulate_csr(extent, n, rng.next_u64());
        if let Ok(sim_k) = ripleys_k(&sim_pts, extent, max_dist, n_bins) {
            for (b, kv) in sim_k.k_observed.iter().enumerate() {
                all_k[b].push(*kv);
            }
        }
    }

    let mut lower = Vec::with_capacity(n_bins);
    let mut upper = Vec::with_capacity(n_bins);
    for band in &mut all_k {
        band.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = band.first().copied().unwrap_or(0.0);
        let hi = band.last().copied().unwrap_or(0.0);
        lower.push(lo);
        upper.push(hi);
    }

    Ok(MonteCarloEnvelope {
        distances: observed.distances,
        lower,
        upper,
        observed: observed.k_observed,
    })
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_points() -> Vec<Point> {
        let mut pts = Vec::new();
        for r in 0..5 {
            for c in 0..5 {
                pts.push(Point::new(c as f64, r as f64));
            }
        }
        pts
    }

    #[test]
    fn test_point_display() {
        let p = Point::new(1.0, 2.0);
        assert!(format!("{p}").contains("1.0"));
    }

    #[test]
    fn test_point_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_extent_new() {
        let e = Extent::new(0.0, 0.0, 10.0, 10.0).unwrap();
        assert!((e.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_extent_invalid() {
        assert!(Extent::new(10.0, 0.0, 5.0, 5.0).is_err());
    }

    #[test]
    fn test_extent_from_points() {
        let pts = grid_points();
        let e = Extent::from_points(&pts, 0.5).unwrap();
        assert!(e.x_min < 0.0);
    }

    #[test]
    fn test_extent_contains() {
        let e = Extent::new(0.0, 0.0, 10.0, 10.0).unwrap();
        assert!(e.contains(&Point::new(5.0, 5.0)));
        assert!(!e.contains(&Point::new(11.0, 5.0)));
    }

    #[test]
    fn test_quadrat_analysis() {
        let pts = grid_points();
        let e = Extent::new(-0.5, -0.5, 4.5, 4.5).unwrap();
        let r = quadrat_analysis(&pts, &e, 5, 5).unwrap();
        assert_eq!(r.rows, 5);
        assert_eq!(r.cols, 5);
        // Regular grid → each cell should have 1 point
        for &c in &r.counts {
            assert_eq!(c, 1);
        }
    }

    #[test]
    fn test_quadrat_display() {
        let pts = grid_points();
        let e = Extent::new(-0.5, -0.5, 4.5, 4.5).unwrap();
        let r = quadrat_analysis(&pts, &e, 3, 3).unwrap();
        assert!(format!("{r}").contains("Quadrat"));
    }

    #[test]
    fn test_clark_evans_regular() {
        let pts = grid_points();
        let e = Extent::new(-0.5, -0.5, 4.5, 4.5).unwrap();
        let r = clark_evans(&pts, &e).unwrap();
        assert!(r.r_index > 0.9); // Regular pattern → R > 1
    }

    #[test]
    fn test_clark_evans_display() {
        let pts = grid_points();
        let e = Extent::new(-0.5, -0.5, 4.5, 4.5).unwrap();
        let r = clark_evans(&pts, &e).unwrap();
        assert!(format!("{r}").contains("ClarkEvans"));
    }

    #[test]
    fn test_clark_evans_too_few() {
        let pts = vec![Point::new(0.0, 0.0)];
        let e = Extent::new(-1.0, -1.0, 1.0, 1.0).unwrap();
        assert!(clark_evans(&pts, &e).is_err());
    }

    #[test]
    fn test_ripleys_k() {
        let pts = grid_points();
        let e = Extent::new(-0.5, -0.5, 4.5, 4.5).unwrap();
        let r = ripleys_k(&pts, &e, 2.0, 4).unwrap();
        assert_eq!(r.distances.len(), 4);
        assert_eq!(r.k_observed.len(), 4);
    }

    #[test]
    fn test_ripleys_k_display() {
        let pts = grid_points();
        let e = Extent::new(-0.5, -0.5, 4.5, 4.5).unwrap();
        let r = ripleys_k(&pts, &e, 2.0, 4).unwrap();
        assert!(format!("{r}").contains("RipleyK"));
    }

    #[test]
    fn test_kernel_density() {
        let pts = grid_points();
        let e = Extent::new(-1.0, -1.0, 5.0, 5.0).unwrap();
        let r = kernel_density(&pts, &e, 1.0, 10).unwrap();
        assert_eq!(r.density.len(), 100);
        assert!(r.density.iter().all(|d| *d >= 0.0));
    }

    #[test]
    fn test_kde_display() {
        let pts = grid_points();
        let e = Extent::new(-1.0, -1.0, 5.0, 5.0).unwrap();
        let r = kernel_density(&pts, &e, 1.0, 5).unwrap();
        assert!(format!("{r}").contains("KDE"));
    }

    #[test]
    fn test_simulate_csr() {
        let e = Extent::new(0.0, 0.0, 10.0, 10.0).unwrap();
        let pts = simulate_csr(&e, 100, 42);
        assert_eq!(pts.len(), 100);
        for p in &pts {
            assert!(e.contains(p));
        }
    }

    #[test]
    fn test_monte_carlo_envelope() {
        let pts = grid_points();
        let e = Extent::new(-0.5, -0.5, 4.5, 4.5).unwrap();
        let env = monte_carlo_envelope(&pts, &e, 2.0, 3, 10, 42).unwrap();
        assert_eq!(env.distances.len(), 3);
        assert_eq!(env.lower.len(), 3);
        assert_eq!(env.upper.len(), 3);
    }

    #[test]
    fn test_envelope_display() {
        let pts = grid_points();
        let e = Extent::new(-0.5, -0.5, 4.5, 4.5).unwrap();
        let env = monte_carlo_envelope(&pts, &e, 2.0, 3, 10, 42).unwrap();
        assert!(format!("{env}").contains("MonteCarloEnvelope"));
    }

    #[test]
    fn test_pattern_type_display() {
        assert_eq!(format!("{}", PatternType::Clustered), "clustered");
        assert_eq!(format!("{}", PatternType::Random), "random");
        assert_eq!(format!("{}", PatternType::Dispersed), "dispersed");
    }

    #[test]
    fn test_extent_display() {
        let e = Extent::new(0.0, 0.0, 10.0, 10.0).unwrap();
        let s = format!("{e}");
        assert!(s.contains("Extent"));
    }
}
