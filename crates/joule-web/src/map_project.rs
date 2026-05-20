//! Map projections: Mercator, Transverse Mercator, Lambert Conformal Conic,
//! Albers Equal-Area, Stereographic — forward/inverse transforms, scale factor,
//! ProjectionConfig builder.

use std::f64::consts::PI;
use std::fmt;

// ── Constants ──────────────────────────────────────────────────

const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257_223_563;
const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
const DEG: f64 = PI / 180.0;

// ── Error ──────────────────────────────────────────────────────

/// Errors produced by projection operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionError {
    /// Input coordinate out of the projection's valid domain.
    DomainError(String),
    /// Convergence failure in iterative inverse.
    ConvergenceFailed,
    /// Unknown or unsupported projection.
    UnsupportedProjection(String),
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DomainError(s) => write!(f, "domain error: {s}"),
            Self::ConvergenceFailed => write!(f, "iterative inverse did not converge"),
            Self::UnsupportedProjection(s) => write!(f, "unsupported projection: {s}"),
        }
    }
}

// ── ProjectedPoint ─────────────────────────────────────────────

/// A 2-D point in projected (easting, northing) space, in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedPoint {
    pub easting: f64,
    pub northing: f64,
}

impl ProjectedPoint {
    pub fn new(easting: f64, northing: f64) -> Self {
        Self { easting, northing }
    }
}

impl fmt::Display for ProjectedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E {:.2} N {:.2}", self.easting, self.northing)
    }
}

// ── ProjectionKind ─────────────────────────────────────────────

/// Supported projection types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectionKind {
    Mercator,
    TransverseMercator,
    LambertConformalConic,
    AlbersEqualArea,
    Stereographic,
}

impl fmt::Display for ProjectionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Mercator => "Mercator",
            Self::TransverseMercator => "Transverse Mercator",
            Self::LambertConformalConic => "Lambert Conformal Conic",
            Self::AlbersEqualArea => "Albers Equal-Area",
            Self::Stereographic => "Stereographic",
        };
        write!(f, "{name}")
    }
}

// ── ProjectionConfig ───────────────────────────────────────────

/// Configuration for a map projection instance.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionConfig {
    pub kind: ProjectionKind,
    /// Central meridian in degrees.
    pub central_meridian: f64,
    /// Latitude of origin in degrees.
    pub lat_origin: f64,
    /// First standard parallel (conic projections).
    pub std_parallel_1: f64,
    /// Second standard parallel (conic projections).
    pub std_parallel_2: f64,
    /// Scale factor on the central meridian.
    pub scale_factor: f64,
    /// False easting in metres.
    pub false_easting: f64,
    /// False northing in metres.
    pub false_northing: f64,
    /// Semi-major axis override.
    pub semi_major: f64,
    /// Eccentricity squared override.
    pub ecc_sq: f64,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            kind: ProjectionKind::Mercator,
            central_meridian: 0.0,
            lat_origin: 0.0,
            std_parallel_1: 33.0,
            std_parallel_2: 45.0,
            scale_factor: 1.0,
            false_easting: 0.0,
            false_northing: 0.0,
            semi_major: WGS84_A,
            ecc_sq: WGS84_E2,
        }
    }
}

impl ProjectionConfig {
    pub fn new(kind: ProjectionKind) -> Self {
        Self { kind, ..Default::default() }
    }

    pub fn with_central_meridian(mut self, cm: f64) -> Self { self.central_meridian = cm; self }
    pub fn with_lat_origin(mut self, lo: f64) -> Self { self.lat_origin = lo; self }
    pub fn with_std_parallels(mut self, p1: f64, p2: f64) -> Self {
        self.std_parallel_1 = p1;
        self.std_parallel_2 = p2;
        self
    }
    pub fn with_scale_factor(mut self, k: f64) -> Self { self.scale_factor = k; self }
    pub fn with_false_easting(mut self, fe: f64) -> Self { self.false_easting = fe; self }
    pub fn with_false_northing(mut self, fn_: f64) -> Self { self.false_northing = fn_; self }
    pub fn with_ellipsoid(mut self, a: f64, e2: f64) -> Self {
        self.semi_major = a;
        self.ecc_sq = e2;
        self
    }
}

// ── Forward transforms ─────────────────────────────────────────

/// Forward projection: geographic (degrees) → projected (metres).
pub fn forward(cfg: &ProjectionConfig, lat_deg: f64, lon_deg: f64) -> Result<ProjectedPoint, ProjectionError> {
    match cfg.kind {
        ProjectionKind::Mercator => forward_mercator(cfg, lat_deg, lon_deg),
        ProjectionKind::TransverseMercator => forward_tm(cfg, lat_deg, lon_deg),
        ProjectionKind::LambertConformalConic => forward_lcc(cfg, lat_deg, lon_deg),
        ProjectionKind::AlbersEqualArea => forward_albers(cfg, lat_deg, lon_deg),
        ProjectionKind::Stereographic => forward_stereo(cfg, lat_deg, lon_deg),
    }
}

/// Inverse projection: projected (metres) → geographic (degrees).
pub fn inverse(cfg: &ProjectionConfig, easting: f64, northing: f64) -> Result<(f64, f64), ProjectionError> {
    match cfg.kind {
        ProjectionKind::Mercator => inverse_mercator(cfg, easting, northing),
        ProjectionKind::TransverseMercator => inverse_tm(cfg, easting, northing),
        ProjectionKind::LambertConformalConic => inverse_lcc(cfg, easting, northing),
        ProjectionKind::AlbersEqualArea => inverse_albers(cfg, easting, northing),
        ProjectionKind::Stereographic => inverse_stereo(cfg, easting, northing),
    }
}

/// Compute the projection scale factor at a given latitude.
pub fn scale_at(cfg: &ProjectionConfig, lat_deg: f64) -> f64 {
    match cfg.kind {
        ProjectionKind::Mercator => 1.0 / (lat_deg * DEG).cos(),
        ProjectionKind::TransverseMercator => cfg.scale_factor,
        _ => cfg.scale_factor,
    }
}

// ── Mercator ───────────────────────────────────────────────────

fn forward_mercator(cfg: &ProjectionConfig, lat_deg: f64, lon_deg: f64) -> Result<ProjectedPoint, ProjectionError> {
    if lat_deg.abs() > 85.051_129 {
        return Err(ProjectionError::DomainError("latitude beyond Mercator limit".into()));
    }
    let lat = lat_deg * DEG;
    let lon = lon_deg * DEG;
    let cm = cfg.central_meridian * DEG;
    let e = cfg.ecc_sq.sqrt();
    let esin = e * lat.sin();
    let x = cfg.semi_major * cfg.scale_factor * (lon - cm) + cfg.false_easting;
    let y = cfg.semi_major * cfg.scale_factor
        * ((PI / 4.0 + lat / 2.0).tan() * ((1.0 - esin) / (1.0 + esin)).powf(e / 2.0)).ln()
        + cfg.false_northing;
    Ok(ProjectedPoint::new(x, y))
}

fn inverse_mercator(cfg: &ProjectionConfig, easting: f64, northing: f64) -> Result<(f64, f64), ProjectionError> {
    let x = (easting - cfg.false_easting) / (cfg.semi_major * cfg.scale_factor);
    let y = (northing - cfg.false_northing) / (cfg.semi_major * cfg.scale_factor);
    let lon = x / DEG + cfg.central_meridian;
    let e = cfg.ecc_sq.sqrt();
    let mut lat = PI / 2.0 - 2.0 * (-y).exp().atan();
    for _ in 0..10 {
        let esin = e * lat.sin();
        lat = PI / 2.0 - 2.0 * ((-y).exp() * ((1.0 - esin) / (1.0 + esin)).powf(e / 2.0)).atan();
    }
    Ok((lat / DEG, lon))
}

// ── Transverse Mercator ────────────────────────────────────────

fn forward_tm(cfg: &ProjectionConfig, lat_deg: f64, lon_deg: f64) -> Result<ProjectedPoint, ProjectionError> {
    let lat = lat_deg * DEG;
    let lon = lon_deg * DEG;
    let cm = cfg.central_meridian * DEG;
    let dl = lon - cm;
    let e2 = cfg.ecc_sq;
    let ep2 = e2 / (1.0 - e2);
    let n_val = cfg.semi_major / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    let t = lat.tan();
    let c = ep2 * lat.cos().powi(2);
    let a_val = dl * lat.cos();
    let m = meridional_arc(cfg.semi_major, e2, lat);
    let m0 = meridional_arc(cfg.semi_major, e2, cfg.lat_origin * DEG);

    let x = cfg.false_easting
        + cfg.scale_factor * n_val
            * (a_val
                + a_val.powi(3) / 6.0 * (1.0 - t * t + c)
                + a_val.powi(5) / 120.0 * (5.0 - 18.0 * t * t + t.powi(4)));

    let y = cfg.false_northing
        + cfg.scale_factor
            * (m - m0
                + n_val * t
                    * (a_val * a_val / 2.0
                        + a_val.powi(4) / 24.0 * (5.0 - t * t + 9.0 * c + 4.0 * c * c)));

    Ok(ProjectedPoint::new(x, y))
}

fn inverse_tm(cfg: &ProjectionConfig, easting: f64, northing: f64) -> Result<(f64, f64), ProjectionError> {
    let e2 = cfg.ecc_sq;
    let ep2 = e2 / (1.0 - e2);
    let m0 = meridional_arc(cfg.semi_major, e2, cfg.lat_origin * DEG);
    let m = m0 + (northing - cfg.false_northing) / cfg.scale_factor;
    let mu = m / (cfg.semi_major * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2.powi(3) / 256.0));
    let e1 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());
    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin();
    let n1 = cfg.semi_major / (1.0 - e2 * phi1.sin().powi(2)).sqrt();
    let t1 = phi1.tan();
    let c1 = ep2 * phi1.cos().powi(2);
    let r1 = cfg.semi_major * (1.0 - e2) / (1.0 - e2 * phi1.sin().powi(2)).powf(1.5);
    let d = (easting - cfg.false_easting) / (n1 * cfg.scale_factor);

    let lat = phi1
        - n1 * t1 / r1
            * (d * d / 2.0 - d.powi(4) / 24.0 * (5.0 + 3.0 * t1 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * ep2));
    let lon = cfg.central_meridian * DEG
        + (d - d.powi(3) / 6.0 * (1.0 + 2.0 * t1 * t1 + c1)) / phi1.cos();

    Ok((lat / DEG, lon / DEG))
}

fn meridional_arc(a: f64, e2: f64, lat: f64) -> f64 {
    let e4 = e2 * e2;
    let e6 = e4 * e2;
    a * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * lat).sin()
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * lat).sin()
        - (35.0 * e6 / 3072.0) * (6.0 * lat).sin())
}

// ── Lambert Conformal Conic ────────────────────────────────────

fn lcc_t(lat: f64, e: f64) -> f64 {
    let esin = e * lat.sin();
    (PI / 4.0 - lat / 2.0).tan() / ((1.0 - esin) / (1.0 + esin)).powf(e / 2.0)
}

fn lcc_m(lat: f64, e2: f64) -> f64 {
    lat.cos() / (1.0 - e2 * lat.sin().powi(2)).sqrt()
}

fn forward_lcc(cfg: &ProjectionConfig, lat_deg: f64, lon_deg: f64) -> Result<ProjectedPoint, ProjectionError> {
    let e = cfg.ecc_sq.sqrt();
    let phi = lat_deg * DEG;
    let lam = lon_deg * DEG;
    let phi1 = cfg.std_parallel_1 * DEG;
    let phi2 = cfg.std_parallel_2 * DEG;
    let phi0 = cfg.lat_origin * DEG;
    let lam0 = cfg.central_meridian * DEG;

    let m1 = lcc_m(phi1, cfg.ecc_sq);
    let m2 = lcc_m(phi2, cfg.ecc_sq);
    let t0 = lcc_t(phi0, e);
    let t1 = lcc_t(phi1, e);
    let t2 = lcc_t(phi2, e);
    let t = lcc_t(phi, e);
    let n = (m1.ln() - m2.ln()) / (t1.ln() - t2.ln());
    let ff = m1 / (n * t1.powf(n));
    let rho0 = cfg.semi_major * ff * t0.powf(n);
    let rho = cfg.semi_major * ff * t.powf(n);
    let theta = n * (lam - lam0);

    let x = cfg.false_easting + rho * theta.sin();
    let y = cfg.false_northing + rho0 - rho * theta.cos();
    Ok(ProjectedPoint::new(x, y))
}

fn inverse_lcc(cfg: &ProjectionConfig, easting: f64, northing: f64) -> Result<(f64, f64), ProjectionError> {
    let e = cfg.ecc_sq.sqrt();
    let phi1 = cfg.std_parallel_1 * DEG;
    let phi2 = cfg.std_parallel_2 * DEG;
    let phi0 = cfg.lat_origin * DEG;
    let lam0 = cfg.central_meridian * DEG;

    let m1 = lcc_m(phi1, cfg.ecc_sq);
    let m2 = lcc_m(phi2, cfg.ecc_sq);
    let t0 = lcc_t(phi0, e);
    let t1 = lcc_t(phi1, e);
    let t2 = lcc_t(phi2, e);
    let n = (m1.ln() - m2.ln()) / (t1.ln() - t2.ln());
    let ff = m1 / (n * t1.powf(n));
    let rho0 = cfg.semi_major * ff * t0.powf(n);

    let dx = easting - cfg.false_easting;
    let dy = rho0 - (northing - cfg.false_northing);
    let rho = n.signum() * (dx * dx + dy * dy).sqrt();
    let t = (rho / (cfg.semi_major * ff)).powf(1.0 / n);
    let theta = dx.atan2(dy);

    let mut lat = PI / 2.0 - 2.0 * t.atan();
    for _ in 0..10 {
        let esin = e * lat.sin();
        lat = PI / 2.0 - 2.0 * (t * ((1.0 - esin) / (1.0 + esin)).powf(e / 2.0)).atan();
    }
    let lon = theta / n + lam0;
    Ok((lat / DEG, lon / DEG))
}

// ── Albers Equal-Area ──────────────────────────────────────────

fn albers_q(sin_lat: f64, e: f64) -> f64 {
    let esin = e * sin_lat;
    (1.0 - e * e) * (sin_lat / (1.0 - esin * esin)
        - (1.0 / (2.0 * e)) * ((1.0 - esin) / (1.0 + esin)).ln())
}

fn forward_albers(cfg: &ProjectionConfig, lat_deg: f64, lon_deg: f64) -> Result<ProjectedPoint, ProjectionError> {
    let e = cfg.ecc_sq.sqrt();
    let phi = lat_deg * DEG;
    let lam = lon_deg * DEG;
    let phi1 = cfg.std_parallel_1 * DEG;
    let phi2 = cfg.std_parallel_2 * DEG;
    let phi0 = cfg.lat_origin * DEG;
    let lam0 = cfg.central_meridian * DEG;

    let m1 = lcc_m(phi1, cfg.ecc_sq);
    let m2 = lcc_m(phi2, cfg.ecc_sq);
    let q0 = albers_q(phi0.sin(), e);
    let q1 = albers_q(phi1.sin(), e);
    let q2 = albers_q(phi2.sin(), e);
    let q = albers_q(phi.sin(), e);
    let n = (m1 * m1 - m2 * m2) / (q2 - q1);
    let cc = m1 * m1 + n * q1;
    let rho0 = cfg.semi_major * (cc - n * q0).abs().sqrt() / n;
    let rho = cfg.semi_major * (cc - n * q).abs().sqrt() / n;
    let theta = n * (lam - lam0);

    let x = cfg.false_easting + rho * theta.sin();
    let y = cfg.false_northing + rho0 - rho * theta.cos();
    Ok(ProjectedPoint::new(x, y))
}

fn inverse_albers(cfg: &ProjectionConfig, easting: f64, northing: f64) -> Result<(f64, f64), ProjectionError> {
    let e = cfg.ecc_sq.sqrt();
    let phi1 = cfg.std_parallel_1 * DEG;
    let phi2 = cfg.std_parallel_2 * DEG;
    let phi0 = cfg.lat_origin * DEG;
    let lam0 = cfg.central_meridian * DEG;

    let m1 = lcc_m(phi1, cfg.ecc_sq);
    let m2 = lcc_m(phi2, cfg.ecc_sq);
    let q0 = albers_q(phi0.sin(), e);
    let q1 = albers_q(phi1.sin(), e);
    let q2 = albers_q(phi2.sin(), e);
    let n = (m1 * m1 - m2 * m2) / (q2 - q1);
    let cc = m1 * m1 + n * q1;
    let rho0 = cfg.semi_major * (cc - n * q0).abs().sqrt() / n;

    let dx = easting - cfg.false_easting;
    let dy = rho0 - (northing - cfg.false_northing);
    let rho = (dx * dx + dy * dy).sqrt();
    let theta = dx.atan2(dy);
    let q = (cc - (rho * rho * n * n) / (cfg.semi_major * cfg.semi_major)) / n;

    let mut lat = (q / 2.0).asin();
    for _ in 0..10 {
        let sin_lat = lat.sin();
        let esin = e * sin_lat;
        let denom = 1.0 - esin * esin;
        let dq = (1.0 - cfg.ecc_sq)
            * (sin_lat / denom - (1.0 / (2.0 * e)) * ((1.0 - esin) / (1.0 + esin)).ln());
        let ddq = denom * denom / (2.0 * lat.cos());
        lat += ddq * (q - dq);
    }
    let lon = lam0 + theta / n;
    Ok((lat / DEG, lon / DEG))
}

// ── Stereographic ──────────────────────────────────────────────

fn forward_stereo(cfg: &ProjectionConfig, lat_deg: f64, lon_deg: f64) -> Result<ProjectedPoint, ProjectionError> {
    let phi = lat_deg * DEG;
    let lam = lon_deg * DEG;
    let phi0 = cfg.lat_origin * DEG;
    let lam0 = cfg.central_meridian * DEG;
    let k0 = cfg.scale_factor;
    let r = 2.0 * cfg.semi_major * k0;

    let denom = 1.0 + phi0.sin() * phi.sin() + phi0.cos() * phi.cos() * (lam - lam0).cos();
    if denom.abs() < 1e-12 {
        return Err(ProjectionError::DomainError("antipodal point".into()));
    }
    let k = r / denom;
    let x = cfg.false_easting + k * phi.cos() * (lam - lam0).sin();
    let y = cfg.false_northing + k * (phi0.cos() * phi.sin() - phi0.sin() * phi.cos() * (lam - lam0).cos());
    Ok(ProjectedPoint::new(x, y))
}

fn inverse_stereo(cfg: &ProjectionConfig, easting: f64, northing: f64) -> Result<(f64, f64), ProjectionError> {
    let phi0 = cfg.lat_origin * DEG;
    let lam0 = cfg.central_meridian * DEG;
    let k0 = cfg.scale_factor;
    let r = 2.0 * cfg.semi_major * k0;

    let dx = easting - cfg.false_easting;
    let dy = northing - cfg.false_northing;
    let rho = (dx * dx + dy * dy).sqrt();
    if rho < 1e-12 {
        return Ok((cfg.lat_origin, cfg.central_meridian));
    }
    let c = 2.0 * (rho / r).atan();
    let lat = (c.cos() * phi0.sin() + dy * c.sin() * phi0.cos() / rho).asin();
    let lon = lam0
        + (dx * c.sin()).atan2(rho * phi0.cos() * c.cos() - dy * phi0.sin() * c.sin());
    Ok((lat / DEG, lon / DEG))
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL_M: f64 = 1.0;
    const TOL_DEG: f64 = 1e-4;

    fn roundtrip(cfg: &ProjectionConfig, lat: f64, lon: f64) {
        let p = forward(cfg, lat, lon).unwrap();
        let (lat2, lon2) = inverse(cfg, p.easting, p.northing).unwrap();
        assert!((lat - lat2).abs() < TOL_DEG, "lat {lat} != {lat2}");
        assert!((lon - lon2).abs() < TOL_DEG, "lon {lon} != {lon2}");
    }

    #[test]
    fn test_mercator_origin() {
        let cfg = ProjectionConfig::new(ProjectionKind::Mercator);
        let p = forward(&cfg, 0.0, 0.0).unwrap();
        assert!(p.easting.abs() < TOL_M);
        assert!(p.northing.abs() < TOL_M);
    }

    #[test]
    fn test_mercator_roundtrip() {
        let cfg = ProjectionConfig::new(ProjectionKind::Mercator);
        roundtrip(&cfg, 45.0, 90.0);
    }

    #[test]
    fn test_mercator_domain_error() {
        let cfg = ProjectionConfig::new(ProjectionKind::Mercator);
        assert!(forward(&cfg, 86.0, 0.0).is_err());
    }

    #[test]
    fn test_mercator_scale_factor() {
        let cfg = ProjectionConfig::new(ProjectionKind::Mercator);
        let s0 = scale_at(&cfg, 0.0);
        let s60 = scale_at(&cfg, 60.0);
        assert!((s0 - 1.0).abs() < 0.01);
        assert!(s60 > 1.5);
    }

    #[test]
    fn test_tm_roundtrip() {
        let cfg = ProjectionConfig::new(ProjectionKind::TransverseMercator)
            .with_central_meridian(-87.0)
            .with_scale_factor(0.9996);
        roundtrip(&cfg, 40.0, -85.0);
    }

    #[test]
    fn test_tm_roundtrip_south() {
        let cfg = ProjectionConfig::new(ProjectionKind::TransverseMercator)
            .with_central_meridian(15.0)
            .with_scale_factor(0.9996);
        roundtrip(&cfg, -34.0, 18.0);
    }

    #[test]
    fn test_lcc_roundtrip() {
        let cfg = ProjectionConfig::new(ProjectionKind::LambertConformalConic)
            .with_std_parallels(33.0, 45.0)
            .with_central_meridian(-96.0)
            .with_lat_origin(23.0);
        roundtrip(&cfg, 40.0, -90.0);
    }

    #[test]
    fn test_lcc_symmetric() {
        let cfg = ProjectionConfig::new(ProjectionKind::LambertConformalConic)
            .with_std_parallels(33.0, 45.0)
            .with_central_meridian(0.0)
            .with_lat_origin(39.0);
        let pe = forward(&cfg, 45.0, 10.0).unwrap();
        let pw = forward(&cfg, 45.0, -10.0).unwrap();
        assert!((pe.easting + pw.easting).abs() < TOL_M);
    }

    #[test]
    fn test_albers_roundtrip() {
        let cfg = ProjectionConfig::new(ProjectionKind::AlbersEqualArea)
            .with_std_parallels(29.5, 45.5)
            .with_central_meridian(-96.0)
            .with_lat_origin(23.0);
        roundtrip(&cfg, 35.0, -80.0);
    }

    #[test]
    fn test_albers_equal_area_property() {
        let cfg = ProjectionConfig::new(ProjectionKind::AlbersEqualArea)
            .with_std_parallels(29.5, 45.5)
            .with_central_meridian(-96.0)
            .with_lat_origin(23.0);
        // two small rectangles same angular extent at different lats
        // Equal-area means projected area is proportional to true surface area.
        // True area of a 1x1 degree cell scales with cos(lat).
        // Verify the ratio of projected areas matches the ratio of true areas.
        let c1 = [forward(&cfg, 30.0, -90.0).unwrap(), forward(&cfg, 30.0, -89.0).unwrap(),
                  forward(&cfg, 31.0, -89.0).unwrap(), forward(&cfg, 31.0, -90.0).unwrap()];
        let c2 = [forward(&cfg, 40.0, -90.0).unwrap(), forward(&cfg, 40.0, -89.0).unwrap(),
                  forward(&cfg, 41.0, -89.0).unwrap(), forward(&cfg, 41.0, -90.0).unwrap()];
        let shoelace = |pts: &[ProjectedPoint]| -> f64 {
            let n = pts.len();
            let mut s = 0.0;
            for i in 0..n {
                let j = (i + 1) % n;
                s += pts[i].easting * pts[j].northing - pts[j].easting * pts[i].northing;
            }
            s.abs() / 2.0
        };
        let area1 = shoelace(&c1);
        let area2 = shoelace(&c2);
        let projected_ratio = area1 / area2;
        let true_ratio = (30.5_f64 * DEG).cos() / (40.5_f64 * DEG).cos();
        assert!((projected_ratio - true_ratio).abs() / true_ratio < 0.02,
            "projected ratio {projected_ratio:.4} vs true ratio {true_ratio:.4}");
    }

    #[test]
    fn test_stereo_roundtrip() {
        let cfg = ProjectionConfig::new(ProjectionKind::Stereographic)
            .with_lat_origin(90.0)
            .with_central_meridian(0.0);
        roundtrip(&cfg, 70.0, 45.0);
    }

    #[test]
    fn test_stereo_origin() {
        let cfg = ProjectionConfig::new(ProjectionKind::Stereographic)
            .with_lat_origin(45.0)
            .with_central_meridian(10.0);
        let p = forward(&cfg, 45.0, 10.0).unwrap();
        assert!(p.easting.abs() < TOL_M);
        assert!(p.northing.abs() < TOL_M);
    }

    #[test]
    fn test_projected_point_display() {
        let p = ProjectedPoint::new(500000.0, 4000000.0);
        let s = format!("{p}");
        assert!(s.contains("500000"));
    }

    #[test]
    fn test_projection_kind_display() {
        assert_eq!(format!("{}", ProjectionKind::Mercator), "Mercator");
        assert_eq!(format!("{}", ProjectionKind::TransverseMercator), "Transverse Mercator");
    }

    #[test]
    fn test_config_builder() {
        let cfg = ProjectionConfig::new(ProjectionKind::Mercator)
            .with_central_meridian(10.0)
            .with_false_easting(500000.0);
        assert!((cfg.central_meridian - 10.0).abs() < 1e-10);
        assert!((cfg.false_easting - 500000.0).abs() < 1e-10);
    }

    #[test]
    fn test_mercator_false_easting() {
        let cfg = ProjectionConfig::new(ProjectionKind::Mercator)
            .with_false_easting(500000.0);
        let p = forward(&cfg, 0.0, 0.0).unwrap();
        assert!((p.easting - 500000.0).abs() < TOL_M);
    }

    #[test]
    fn test_tm_equator() {
        let cfg = ProjectionConfig::new(ProjectionKind::TransverseMercator)
            .with_central_meridian(0.0)
            .with_scale_factor(0.9996)
            .with_false_easting(500000.0);
        let p = forward(&cfg, 0.0, 0.0).unwrap();
        assert!((p.easting - 500000.0).abs() < TOL_M);
    }

    #[test]
    fn test_error_display() {
        let e = ProjectionError::DomainError("test".into());
        assert!(format!("{e}").contains("test"));
    }
}
