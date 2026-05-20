//! Map projections: Mercator, Equirectangular, Robinson, Albers Equal-Area Conic,
//! Stereographic. Great-circle distance (Vincenty), bearing, midpoint.

use std::f64::consts::PI;

// ── Projected Point ─────────────────────────────────────────────

/// A projected 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedPoint {
    pub x: f64,
    pub y: f64,
}

impl ProjectedPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

// ── Projection Trait ────────────────────────────────────────────

/// Common interface for map projections.
pub trait Projection {
    /// Project geographic coordinates (lat, lng in degrees) to (x, y).
    fn project(&self, lat: f64, lng: f64) -> ProjectedPoint;

    /// Inverse: (x, y) back to (lat, lng) in degrees.
    fn inverse(&self, x: f64, y: f64) -> (f64, f64);

    /// Scale factor at a given latitude.
    fn scale_factor(&self, lat: f64) -> f64;

    /// Name of the projection.
    fn name(&self) -> &'static str;
}

// ── Mercator ────────────────────────────────────────────────────

/// Mercator projection (conformal, cylindrical).
#[derive(Debug, Clone, Copy)]
pub struct Mercator;

impl Projection for Mercator {
    fn project(&self, lat: f64, lng: f64) -> ProjectedPoint {
        let lat_r = lat.to_radians().clamp(-1.4844, 1.4844);
        let lng_r = lng.to_radians();
        let x = lng_r;
        let y = (PI / 4.0 + lat_r / 2.0).tan().ln();
        ProjectedPoint::new(x, y)
    }

    fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let lng = x.to_degrees();
        let lat = (2.0 * y.exp().atan() - PI / 2.0).to_degrees();
        (lat, lng)
    }

    fn scale_factor(&self, lat: f64) -> f64 {
        1.0 / lat.to_radians().cos()
    }

    fn name(&self) -> &'static str {
        "Mercator"
    }
}

// ── Equirectangular ─────────────────────────────────────────────

/// Equirectangular (Plate Carrée) projection.
#[derive(Debug, Clone, Copy)]
pub struct Equirectangular {
    /// Standard parallel in degrees (default 0).
    pub standard_parallel: f64,
}

impl Default for Equirectangular {
    fn default() -> Self {
        Self { standard_parallel: 0.0 }
    }
}

impl Projection for Equirectangular {
    fn project(&self, lat: f64, lng: f64) -> ProjectedPoint {
        let cos_sp = self.standard_parallel.to_radians().cos();
        ProjectedPoint::new(lng.to_radians() * cos_sp, lat.to_radians())
    }

    fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let cos_sp = self.standard_parallel.to_radians().cos();
        let lat = y.to_degrees();
        let lng = (x / cos_sp).to_degrees();
        (lat, lng)
    }

    fn scale_factor(&self, _lat: f64) -> f64 {
        1.0
    }

    fn name(&self) -> &'static str {
        "Equirectangular"
    }
}

// ── Robinson ────────────────────────────────────────────────────

/// Robinson projection (compromise, neither conformal nor equal-area).
#[derive(Debug, Clone, Copy)]
pub struct Robinson;

// Robinson lookup table: [latitude_degrees, PLEN (x multiplier), PDFE (y multiplier)]
const ROBINSON_TABLE: [(f64, f64, f64); 19] = [
    (0.0, 1.0000, 0.0000),
    (5.0, 0.9986, 0.0620),
    (10.0, 0.9954, 0.1240),
    (15.0, 0.9900, 0.1860),
    (20.0, 0.9822, 0.2480),
    (25.0, 0.9730, 0.3100),
    (30.0, 0.9600, 0.3720),
    (35.0, 0.9427, 0.4340),
    (40.0, 0.9216, 0.4958),
    (45.0, 0.8962, 0.5571),
    (50.0, 0.8679, 0.6176),
    (55.0, 0.8350, 0.6769),
    (60.0, 0.7986, 0.7346),
    (65.0, 0.7597, 0.7903),
    (70.0, 0.7186, 0.8435),
    (75.0, 0.6732, 0.8936),
    (80.0, 0.6213, 0.9394),
    (85.0, 0.5722, 0.9761),
    (90.0, 0.5322, 1.0000),
];

impl Robinson {
    fn interpolate(abs_lat: f64) -> (f64, f64) {
        let abs_lat = abs_lat.clamp(0.0, 90.0);
        let idx = (abs_lat / 5.0).floor() as usize;
        if idx >= ROBINSON_TABLE.len() - 1 {
            let (_, plen, pdfe) = ROBINSON_TABLE[ROBINSON_TABLE.len() - 1];
            return (plen, pdfe);
        }
        let frac = (abs_lat - ROBINSON_TABLE[idx].0) / 5.0;
        let plen = ROBINSON_TABLE[idx].1 + frac * (ROBINSON_TABLE[idx + 1].1 - ROBINSON_TABLE[idx].1);
        let pdfe = ROBINSON_TABLE[idx].2 + frac * (ROBINSON_TABLE[idx + 1].2 - ROBINSON_TABLE[idx].2);
        (plen, pdfe)
    }
}

impl Projection for Robinson {
    fn project(&self, lat: f64, lng: f64) -> ProjectedPoint {
        let (plen, pdfe) = Self::interpolate(lat.abs());
        let x = 0.8487 * plen * lng.to_radians();
        let y = 1.3523 * pdfe * lat.signum();
        ProjectedPoint::new(x, y)
    }

    fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        // Iterative inverse
        let y_sign = y.signum();
        let target_pdfe = (y.abs() / 1.3523).min(1.0);
        // Find interval
        let mut lat_guess = target_pdfe * 90.0;
        for _ in 0..20 {
            let (plen, pdfe) = Self::interpolate(lat_guess);
            let err = pdfe - target_pdfe;
            if err.abs() < 1e-10 {
                let lng = x / (0.8487 * plen);
                return (lat_guess * y_sign, lng.to_degrees());
            }
            // Numerical derivative
            let (_, pdfe2) = Self::interpolate(lat_guess + 0.01);
            let deriv = (pdfe2 - pdfe) / 0.01;
            if deriv.abs() < 1e-15 {
                break;
            }
            lat_guess -= err / deriv;
            lat_guess = lat_guess.clamp(0.0, 90.0);
        }
        let (plen, _) = Self::interpolate(lat_guess);
        let lng = x / (0.8487 * plen);
        (lat_guess * y_sign, lng.to_degrees())
    }

    fn scale_factor(&self, lat: f64) -> f64 {
        let (plen, _) = Self::interpolate(lat.abs());
        plen
    }

    fn name(&self) -> &'static str {
        "Robinson"
    }
}

// ── Albers Equal-Area Conic ─────────────────────────────────────

/// Albers equal-area conic projection.
#[derive(Debug, Clone, Copy)]
pub struct AlbersEqualArea {
    /// First standard parallel (degrees).
    pub parallel1: f64,
    /// Second standard parallel (degrees).
    pub parallel2: f64,
    /// Central meridian (degrees).
    pub central_meridian: f64,
    /// Latitude of origin (degrees).
    pub origin_lat: f64,
}

impl Default for AlbersEqualArea {
    fn default() -> Self {
        // US contiguous defaults
        Self {
            parallel1: 29.5,
            parallel2: 45.5,
            central_meridian: -96.0,
            origin_lat: 37.5,
        }
    }
}

impl AlbersEqualArea {
    fn params(&self) -> (f64, f64, f64) {
        let p1 = self.parallel1.to_radians();
        let p2 = self.parallel2.to_radians();
        let n = (p1.sin() + p2.sin()) / 2.0;
        let c = p1.cos().powi(2) + 2.0 * n * p1.sin();
        let rho0_sq = c - 2.0 * n * self.origin_lat.to_radians().sin();
        let rho0 = if rho0_sq > 0.0 { (rho0_sq / n).sqrt() } else { 0.0 };
        (n, c, rho0)
    }
}

impl Projection for AlbersEqualArea {
    fn project(&self, lat: f64, lng: f64) -> ProjectedPoint {
        let (n, c, rho0) = self.params();
        let theta = n * (lng - self.central_meridian).to_radians();
        let rho_sq = c - 2.0 * n * lat.to_radians().sin();
        let rho = if rho_sq > 0.0 { (rho_sq / n).sqrt() } else { 0.0 };
        let x = rho * theta.sin();
        let y = rho0 - rho * theta.cos();
        ProjectedPoint::new(x, y)
    }

    fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let (n, c, rho0) = self.params();
        let rho = (x * x + (rho0 - y) * (rho0 - y)).sqrt();
        let theta = (x / rho).asin();
        let sin_lat = (c - rho * rho * n) / (2.0 * n);
        let lat = sin_lat.clamp(-1.0, 1.0).asin().to_degrees();
        let lng = (theta / n).to_degrees() + self.central_meridian;
        (lat, lng)
    }

    fn scale_factor(&self, lat: f64) -> f64 {
        let (n, c, _) = self.params();
        let rho_sq = c - 2.0 * n * lat.to_radians().sin();
        if rho_sq > 0.0 {
            let rho = (rho_sq / n).sqrt();
            let h = rho * n / lat.to_radians().cos();
            h
        } else {
            1.0
        }
    }

    fn name(&self) -> &'static str {
        "Albers Equal-Area Conic"
    }
}

// ── Stereographic ───────────────────────────────────────────────

/// Stereographic (azimuthal, conformal) projection centered on a given point.
#[derive(Debug, Clone, Copy)]
pub struct Stereographic {
    pub center_lat: f64,
    pub center_lng: f64,
}

impl Default for Stereographic {
    fn default() -> Self {
        Self { center_lat: 90.0, center_lng: 0.0 }
    }
}

impl Projection for Stereographic {
    fn project(&self, lat: f64, lng: f64) -> ProjectedPoint {
        let lat_r = lat.to_radians();
        let lng_r = lng.to_radians();
        let lat0 = self.center_lat.to_radians();
        let lng0 = self.center_lng.to_radians();
        let dlng = lng_r - lng0;

        let k = 2.0
            / (1.0 + lat0.sin() * lat_r.sin() + lat0.cos() * lat_r.cos() * dlng.cos());
        let x = k * lat_r.cos() * dlng.sin();
        let y = k * (lat0.cos() * lat_r.sin() - lat0.sin() * lat_r.cos() * dlng.cos());
        ProjectedPoint::new(x, y)
    }

    fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let lat0 = self.center_lat.to_radians();
        let lng0 = self.center_lng.to_radians();
        let rho = (x * x + y * y).sqrt();
        if rho < 1e-15 {
            return (self.center_lat, self.center_lng);
        }
        let c = 2.0 * (rho / 2.0).atan();
        let lat = (c.cos() * lat0.sin() + y * c.sin() * lat0.cos() / rho).asin();
        let lng = lng0
            + (x * c.sin()).atan2(rho * lat0.cos() * c.cos() - y * lat0.sin() * c.sin());
        (lat.to_degrees(), lng.to_degrees())
    }

    fn scale_factor(&self, lat: f64) -> f64 {
        let lat_r = lat.to_radians();
        let lat0 = self.center_lat.to_radians();
        let cos_c = lat0.sin() * lat_r.sin() + lat0.cos() * lat_r.cos();
        2.0 / (1.0 + cos_c)
    }

    fn name(&self) -> &'static str {
        "Stereographic"
    }
}

// ── Great Circle Utilities ──────────────────────────────────────

const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// WGS-84 ellipsoid semi-major axis (m).
const WGS84_A: f64 = 6_378_137.0;
/// WGS-84 ellipsoid flattening.
const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// WGS-84 semi-minor axis.
const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);

/// Great-circle distance using Vincenty formula (meters).
/// More accurate than haversine on the WGS-84 ellipsoid.
pub fn vincenty_distance(lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> f64 {
    let u1 = ((1.0 - WGS84_F) * lat1.to_radians().tan()).atan();
    let u2 = ((1.0 - WGS84_F) * lat2.to_radians().tan()).atan();
    let l = (lng2 - lng1).to_radians();

    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let mut lambda = l;
    let mut _cos2_alpha;
    let mut sin_sigma;
    let mut cos_sigma;
    let mut sigma;
    let mut cos_2sigma_m;

    for _ in 0..100 {
        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();

        sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
        .sqrt();

        if sin_sigma < 1e-15 {
            return 0.0; // coincident points
        }

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        sigma = sin_sigma.atan2(cos_sigma);

        let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        _cos2_alpha = 1.0 - sin_alpha * sin_alpha;

        cos_2sigma_m = if _cos2_alpha.abs() < 1e-15 {
            0.0
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / _cos2_alpha
        };

        let c = WGS84_F / 16.0 * _cos2_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * _cos2_alpha));
        let lambda_prev = lambda;
        lambda = l
            + (1.0 - c) * WGS84_F * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))));

        if (lambda - lambda_prev).abs() < 1e-12 {
            // Converged
            let u_sq = _cos2_alpha * (WGS84_A * WGS84_A - WGS84_B * WGS84_B)
                / (WGS84_B * WGS84_B);
            let a_coeff =
                1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
            let b_coeff =
                u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
            let delta_sigma = b_coeff
                * sin_sigma
                * (cos_2sigma_m
                    + b_coeff / 4.0
                        * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))
                            - b_coeff / 6.0
                                * cos_2sigma_m
                                * (-3.0 + 4.0 * sin_sigma.powi(2))
                                * (-3.0 + 4.0 * cos_2sigma_m.powi(2))));
            return WGS84_B * a_coeff * (sigma - delta_sigma);
        }
    }

    // Fallback to haversine on sphere if Vincenty didn't converge (antipodal points)
    haversine_distance(lat1, lng1, lat2, lng2)
}

/// Haversine distance (meters) on a sphere.
pub fn haversine_distance(lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> f64 {
    let dlat = (lat2 - lat1).to_radians();
    let dlng = (lng2 - lng1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlng / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * a.sqrt().asin()
}

/// Initial bearing from point 1 to point 2 (degrees, 0 = north, clockwise).
pub fn bearing(lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> f64 {
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();
    let dlng = (lng2 - lng1).to_radians();
    let y = dlng.sin() * lat2_r.cos();
    let x = lat1_r.cos() * lat2_r.sin() - lat1_r.sin() * lat2_r.cos() * dlng.cos();
    (y.atan2(x).to_degrees() + 360.0) % 360.0
}

/// Midpoint on the great circle between two points.
pub fn midpoint(lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> (f64, f64) {
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();
    let dlng = (lng2 - lng1).to_radians();

    let bx = lat2_r.cos() * dlng.cos();
    let by = lat2_r.cos() * dlng.sin();

    let lat = (lat1_r.sin() + lat2_r.sin()).atan2(
        ((lat1_r.cos() + bx).powi(2) + by.powi(2)).sqrt(),
    );
    let lng = lng1.to_radians() + by.atan2(lat1_r.cos() + bx);

    (lat.to_degrees(), lng.to_degrees())
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mercator_project_origin() {
        let m = Mercator;
        let p = m.project(0.0, 0.0);
        assert!((p.x).abs() < 1e-10);
        assert!((p.y).abs() < 1e-10);
    }

    #[test]
    fn test_mercator_roundtrip() {
        let m = Mercator;
        let p = m.project(45.0, -90.0);
        let (lat, lng) = m.inverse(p.x, p.y);
        assert!((lat - 45.0).abs() < 0.01);
        assert!((lng - (-90.0)).abs() < 0.01);
    }

    #[test]
    fn test_equirectangular_roundtrip() {
        let eq = Equirectangular::default();
        let p = eq.project(30.0, 60.0);
        let (lat, lng) = eq.inverse(p.x, p.y);
        assert!((lat - 30.0).abs() < 0.01);
        assert!((lng - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_robinson_origin() {
        let r = Robinson;
        let p = r.project(0.0, 0.0);
        assert!((p.x).abs() < 1e-10);
        assert!((p.y).abs() < 1e-10);
    }

    #[test]
    fn test_robinson_roundtrip() {
        let r = Robinson;
        let p = r.project(40.0, -74.0);
        let (lat, lng) = r.inverse(p.x, p.y);
        assert!((lat - 40.0).abs() < 0.5);
        assert!((lng - (-74.0)).abs() < 0.5);
    }

    #[test]
    fn test_albers_project_origin() {
        let a = AlbersEqualArea::default();
        let p = a.project(a.origin_lat, a.central_meridian);
        assert!((p.x).abs() < 0.01);
        assert!((p.y).abs() < 0.01);
    }

    #[test]
    fn test_stereographic_roundtrip() {
        let s = Stereographic { center_lat: 90.0, center_lng: 0.0 };
        let p = s.project(80.0, 30.0);
        let (lat, lng) = s.inverse(p.x, p.y);
        assert!((lat - 80.0).abs() < 0.01);
        assert!((lng - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_stereographic_center() {
        let s = Stereographic { center_lat: 45.0, center_lng: -90.0 };
        let p = s.project(45.0, -90.0);
        assert!((p.x).abs() < 1e-10);
        assert!((p.y).abs() < 1e-10);
    }

    #[test]
    fn test_vincenty_distance_known() {
        // New York to London: approximately 5570 km
        let d = vincenty_distance(40.7128, -74.0060, 51.5074, -0.1278);
        assert!((d / 1000.0 - 5570.0).abs() < 50.0);
    }

    #[test]
    fn test_vincenty_zero_distance() {
        let d = vincenty_distance(45.0, 90.0, 45.0, 90.0);
        assert!(d < 0.01);
    }

    #[test]
    fn test_haversine_vs_vincenty() {
        let h = haversine_distance(48.8566, 2.3522, 35.6762, 139.6503);
        let v = vincenty_distance(48.8566, 2.3522, 35.6762, 139.6503);
        // Should be roughly similar (within 0.5%)
        assert!((h - v).abs() / v < 0.005);
    }

    #[test]
    fn test_bearing_north() {
        let b = bearing(0.0, 0.0, 10.0, 0.0);
        assert!((b - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_bearing_east() {
        let b = bearing(0.0, 0.0, 0.0, 10.0);
        assert!((b - 90.0).abs() < 0.1);
    }

    #[test]
    fn test_midpoint() {
        let (lat, lng) = midpoint(0.0, 0.0, 0.0, 10.0);
        assert!((lat - 0.0).abs() < 0.01);
        assert!((lng - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_projection_names() {
        assert_eq!(Mercator.name(), "Mercator");
        assert_eq!(Equirectangular::default().name(), "Equirectangular");
        assert_eq!(Robinson.name(), "Robinson");
        assert_eq!(AlbersEqualArea::default().name(), "Albers Equal-Area Conic");
        assert_eq!(Stereographic::default().name(), "Stereographic");
    }
}
