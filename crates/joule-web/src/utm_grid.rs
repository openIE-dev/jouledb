//! UTM grid: zone calculation from longitude, MGRS grid reference, UTM ↔ lat/lon
//! conversion, zone boundary handling, hemisphere detection, grid convergence angle.

use std::f64::consts::PI;
use std::fmt;

// ── WGS84 Constants ────────────────────────────────────────────

const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257_223_563;
const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
const DEG: f64 = PI / 180.0;
const UTM_K0: f64 = 0.9996;
const UTM_FE: f64 = 500_000.0;
const UTM_FN_SOUTH: f64 = 10_000_000.0;

// ── Error ──────────────────────────────────────────────────────

/// Errors from UTM operations.
#[derive(Debug, Clone, PartialEq)]
pub enum UtmError {
    /// Latitude outside UTM range (80°S – 84°N).
    LatitudeOutOfRange(f64),
    /// Invalid zone number (1–60).
    InvalidZone(u8),
    /// Invalid MGRS string.
    InvalidMgrs(String),
    /// Generic error.
    Other(String),
}

impl fmt::Display for UtmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LatitudeOutOfRange(v) => write!(f, "latitude {v}° outside UTM range [-80, 84]"),
            Self::InvalidZone(z) => write!(f, "invalid UTM zone {z}"),
            Self::InvalidMgrs(s) => write!(f, "invalid MGRS: {s}"),
            Self::Other(s) => write!(f, "UTM error: {s}"),
        }
    }
}

// ── Hemisphere ─────────────────────────────────────────────────

/// Northern or Southern hemisphere indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hemisphere {
    North,
    South,
}

impl fmt::Display for Hemisphere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::North => write!(f, "N"),
            Self::South => write!(f, "S"),
        }
    }
}

// ── UtmCoord ───────────────────────────────────────────────────

/// A UTM coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtmCoord {
    /// Zone number 1..60.
    pub zone: u8,
    /// Hemisphere.
    pub hemisphere: Hemisphere,
    /// Easting in metres (relative to 500 000 m false easting).
    pub easting: f64,
    /// Northing in metres (0 at equator in north, 10 000 000 at equator in south).
    pub northing: f64,
}

impl UtmCoord {
    /// Central meridian of this UTM zone.
    pub fn central_meridian(&self) -> f64 {
        (self.zone as f64 - 1.0) * 6.0 - 180.0 + 3.0
    }

    /// Grid convergence angle in degrees (approximate).
    pub fn convergence_angle(&self, lat_deg: f64) -> f64 {
        let dl = (self.easting - UTM_FE) / (UTM_K0 * WGS84_A);
        let lat = lat_deg * DEG;
        (dl * lat.sin()).to_degrees()
    }
}

impl fmt::Display for UtmCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{} {:.0}m E {:.0}m N",
            self.zone, self.hemisphere, self.easting, self.northing
        )
    }
}

// ── Zone calculation ───────────────────────────────────────────

/// Calculate UTM zone number from longitude (degrees).
pub fn zone_from_lon(lon: f64) -> u8 {
    let z = ((lon + 180.0) / 6.0).floor() as u8 + 1;
    z.min(60).max(1)
}

/// Determine the UTM zone number accounting for Norway/Svalbard exceptions.
pub fn zone_from_latlon(lat: f64, lon: f64) -> u8 {
    let base = zone_from_lon(lon);
    // Norway exception: zone 32 for lat 56-64 lon 3-12
    if (56.0..64.0).contains(&lat) && (3.0..12.0).contains(&lon) {
        return 32;
    }
    // Svalbard exceptions
    if (72.0..84.0).contains(&lat) {
        if (0.0..9.0).contains(&lon) {
            return 31;
        } else if (9.0..21.0).contains(&lon) {
            return 33;
        } else if (21.0..33.0).contains(&lon) {
            return 35;
        } else if (33.0..42.0).contains(&lon) {
            return 37;
        }
    }
    base
}

/// Detect hemisphere from latitude.
pub fn hemisphere_from_lat(lat: f64) -> Hemisphere {
    if lat >= 0.0 {
        Hemisphere::North
    } else {
        Hemisphere::South
    }
}

/// Central meridian for a given UTM zone.
pub fn central_meridian(zone: u8) -> Result<f64, UtmError> {
    if zone < 1 || zone > 60 {
        return Err(UtmError::InvalidZone(zone));
    }
    Ok((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0)
}

// ── Lat/Lon → UTM ──────────────────────────────────────────────

/// Convert geographic coordinates (degrees) to UTM.
pub fn latlon_to_utm(lat_deg: f64, lon_deg: f64) -> Result<UtmCoord, UtmError> {
    if lat_deg < -80.0 || lat_deg > 84.0 {
        return Err(UtmError::LatitudeOutOfRange(lat_deg));
    }
    let zone = zone_from_latlon(lat_deg, lon_deg);
    latlon_to_utm_zone(lat_deg, lon_deg, zone)
}

/// Convert geographic coordinates to a specific UTM zone.
pub fn latlon_to_utm_zone(lat_deg: f64, lon_deg: f64, zone: u8) -> Result<UtmCoord, UtmError> {
    if zone < 1 || zone > 60 {
        return Err(UtmError::InvalidZone(zone));
    }
    let cm = central_meridian(zone)?;
    let lat = lat_deg * DEG;
    let dl = (lon_deg - cm) * DEG;
    let e2 = WGS84_E2;
    let ep2 = e2 / (1.0 - e2);
    let n = WGS84_A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    let t = lat.tan();
    let c = ep2 * lat.cos().powi(2);
    let a = dl * lat.cos();
    let m = meridional_arc(lat);

    let easting = UTM_FE
        + UTM_K0 * n
            * (a + a.powi(3) / 6.0 * (1.0 - t * t + c)
                + a.powi(5) / 120.0 * (5.0 - 18.0 * t * t + t.powi(4) + 72.0 * c - 58.0 * ep2));

    let mut northing = UTM_K0
        * (m + n * t
            * (a * a / 2.0
                + a.powi(4) / 24.0 * (5.0 - t * t + 9.0 * c + 4.0 * c * c)
                + a.powi(6) / 720.0
                    * (61.0 - 58.0 * t * t + t.powi(4) + 600.0 * c - 330.0 * ep2)));

    let hemi = hemisphere_from_lat(lat_deg);
    if hemi == Hemisphere::South {
        northing += UTM_FN_SOUTH;
    }

    Ok(UtmCoord { zone, hemisphere: hemi, easting, northing })
}

// ── UTM → Lat/Lon ──────────────────────────────────────────────

/// Convert UTM to geographic coordinates (degrees).
pub fn utm_to_latlon(utm: &UtmCoord) -> Result<(f64, f64), UtmError> {
    if utm.zone < 1 || utm.zone > 60 {
        return Err(UtmError::InvalidZone(utm.zone));
    }
    let cm = central_meridian(utm.zone)?;
    let e2 = WGS84_E2;
    let ep2 = e2 / (1.0 - e2);
    let e1 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());

    let mut northing = utm.northing;
    if utm.hemisphere == Hemisphere::South {
        northing -= UTM_FN_SOUTH;
    }
    let m = northing / UTM_K0;
    let mu = m / (WGS84_A * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2.powi(3) / 256.0));

    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

    let n1 = WGS84_A / (1.0 - e2 * phi1.sin().powi(2)).sqrt();
    let t1 = phi1.tan();
    let c1 = ep2 * phi1.cos().powi(2);
    let r1 = WGS84_A * (1.0 - e2) / (1.0 - e2 * phi1.sin().powi(2)).powf(1.5);
    let d = (utm.easting - UTM_FE) / (n1 * UTM_K0);

    let lat = phi1
        - n1 * t1 / r1
            * (d * d / 2.0
                - d.powi(4) / 24.0
                    * (5.0 + 3.0 * t1 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * ep2)
                + d.powi(6) / 720.0
                    * (61.0 + 90.0 * t1 * t1 + 298.0 * c1 + 45.0 * t1.powi(4)
                        - 252.0 * ep2
                        - 3.0 * c1 * c1));

    let lon_rad = cm * DEG
        + (d - d.powi(3) / 6.0 * (1.0 + 2.0 * t1 * t1 + c1)
            + d.powi(5) / 120.0
                * (5.0 - 2.0 * c1 + 28.0 * t1 * t1 - 3.0 * c1 * c1 + 8.0 * ep2
                    + 24.0 * t1.powi(4)))
            / phi1.cos();

    Ok((lat / DEG, lon_rad / DEG))
}

// ── MGRS ───────────────────────────────────────────────────────

/// UTM latitude band letter for a given latitude.
pub fn lat_band(lat: f64) -> char {
    const BANDS: &[u8] = b"CDEFGHJKLMNPQRSTUVWX";
    if lat < -80.0 || lat > 84.0 {
        return 'Z'; // outside UTM
    }
    let idx = ((lat + 80.0) / 8.0).floor() as usize;
    BANDS[idx.min(BANDS.len() - 1)] as char
}

/// Format a UTM coordinate as an MGRS 1-metre grid reference.
pub fn to_mgrs(utm: &UtmCoord, lat_deg: f64) -> String {
    let band = lat_band(lat_deg);
    let set = ((utm.zone - 1) % 6) as usize;
    let col_letters: &[u8] = match set % 3 {
        0 => b"ABCDEFGH",
        1 => b"JKLMNPQR",
        _ => b"STUVWXYZ",
    };
    let row_letters: &[u8] = if set % 2 == 0 {
        b"ABCDEFGHJKLMNPQRSTUV"
    } else {
        b"FGHJKLMNPQRSTUVABCDE"
    };
    let e100k = ((utm.easting / 100_000.0).floor() as usize).saturating_sub(1);
    let n100k = ((utm.northing % 2_000_000.0) / 100_000.0).floor() as usize;
    let col = col_letters[e100k % col_letters.len()] as char;
    let row = row_letters[n100k % row_letters.len()] as char;
    let e5 = (utm.easting % 100_000.0) as u32;
    let n5 = (utm.northing % 100_000.0) as u32;
    format!("{:02}{}{}{} {:05} {:05}", utm.zone, band, col, row, e5, n5)
}

/// Check whether a point lies within the boundaries of the specified zone.
pub fn is_in_zone(lat: f64, lon: f64, zone: u8) -> bool {
    if zone < 1 || zone > 60 {
        return false;
    }
    let west = (zone as f64 - 1.0) * 6.0 - 180.0;
    let east = west + 6.0;
    lon >= west && lon < east && lat >= -80.0 && lat <= 84.0
}

/// Grid convergence angle at a point (degrees).
pub fn grid_convergence(lat_deg: f64, lon_deg: f64) -> f64 {
    let zone = zone_from_latlon(lat_deg, lon_deg);
    let cm = (zone as f64 - 1.0) * 6.0 - 180.0 + 3.0;
    let dl = (lon_deg - cm) * DEG;
    let lat = lat_deg * DEG;
    (dl * lat.sin()).to_degrees()
}

/// Point scale factor for a UTM coordinate.
pub fn point_scale(utm: &UtmCoord) -> f64 {
    let dx = utm.easting - UTM_FE;
    let r = WGS84_A;
    UTM_K0 * (1.0 + (dx * dx) / (2.0 * r * r))
}

// ── Internal helpers ───────────────────────────────────────────

fn meridional_arc(lat: f64) -> f64 {
    let e2 = WGS84_E2;
    let e4 = e2 * e2;
    let e6 = e4 * e2;
    WGS84_A
        * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
            - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * lat).sin()
            + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * lat).sin()
            - (35.0 * e6 / 3072.0) * (6.0 * lat).sin())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-4;
    const TOL_M: f64 = 1.0;

    #[test]
    fn test_zone_from_lon_london() {
        assert_eq!(zone_from_lon(-0.1278), 30);
    }

    #[test]
    fn test_zone_from_lon_new_york() {
        assert_eq!(zone_from_lon(-74.006), 18);
    }

    #[test]
    fn test_zone_from_lon_tokyo() {
        assert_eq!(zone_from_lon(139.6917), 54);
    }

    #[test]
    fn test_zone_from_lon_boundary() {
        // -180 → zone 1, 174 → zone 60
        assert_eq!(zone_from_lon(-180.0), 1);
        assert_eq!(zone_from_lon(174.0), 60);
    }

    #[test]
    fn test_norway_exception() {
        assert_eq!(zone_from_latlon(60.0, 5.0), 32);
    }

    #[test]
    fn test_svalbard_exception() {
        assert_eq!(zone_from_latlon(78.0, 15.0), 33);
    }

    #[test]
    fn test_hemisphere() {
        assert_eq!(hemisphere_from_lat(45.0), Hemisphere::North);
        assert_eq!(hemisphere_from_lat(-30.0), Hemisphere::South);
        assert_eq!(hemisphere_from_lat(0.0), Hemisphere::North);
    }

    #[test]
    fn test_central_meridian() {
        assert!((central_meridian(1).unwrap() - (-177.0)).abs() < TOL);
        assert!((central_meridian(30).unwrap() - (-3.0)).abs() < TOL);
        assert!(central_meridian(0).is_err());
        assert!(central_meridian(61).is_err());
    }

    #[test]
    fn test_latlon_to_utm_equator() {
        let u = latlon_to_utm(0.0, 3.0).unwrap();
        assert_eq!(u.zone, 31);
        assert!((u.easting - UTM_FE).abs() < 1000.0); // near CM
    }

    #[test]
    fn test_roundtrip_north() {
        let lat = 40.7128;
        let lon = -74.006;
        let u = latlon_to_utm(lat, lon).unwrap();
        let (lat2, lon2) = utm_to_latlon(&u).unwrap();
        assert!((lat - lat2).abs() < TOL, "lat {lat} != {lat2}");
        assert!((lon - lon2).abs() < TOL, "lon {lon} != {lon2}");
    }

    #[test]
    fn test_roundtrip_south() {
        let lat = -33.8688;
        let lon = 151.2093;
        let u = latlon_to_utm(lat, lon).unwrap();
        assert_eq!(u.hemisphere, Hemisphere::South);
        let (lat2, lon2) = utm_to_latlon(&u).unwrap();
        assert!((lat - lat2).abs() < TOL);
        assert!((lon - lon2).abs() < TOL);
    }

    #[test]
    fn test_out_of_range() {
        assert!(latlon_to_utm(85.0, 0.0).is_err());
        assert!(latlon_to_utm(-81.0, 0.0).is_err());
    }

    #[test]
    fn test_lat_band() {
        assert_eq!(lat_band(0.0), 'N');
        assert_eq!(lat_band(-45.0), 'G');
        assert_eq!(lat_band(80.0), 'X');
    }

    #[test]
    fn test_mgrs_format() {
        let u = latlon_to_utm(40.7128, -74.006).unwrap();
        let m = to_mgrs(&u, 40.7128);
        assert!(m.len() > 10);
        assert!(m.starts_with("18"));
    }

    #[test]
    fn test_is_in_zone() {
        assert!(is_in_zone(45.0, 3.0, 31));
        assert!(!is_in_zone(45.0, 3.0, 32));
    }

    #[test]
    fn test_grid_convergence_cm() {
        // on central meridian, convergence should be near zero
        let gc = grid_convergence(45.0, 3.0);
        assert!(gc.abs() < 0.5);
    }

    #[test]
    fn test_point_scale_cm() {
        let u = UtmCoord {
            zone: 31,
            hemisphere: Hemisphere::North,
            easting: UTM_FE,
            northing: 5_000_000.0,
        };
        let s = point_scale(&u);
        assert!((s - UTM_K0).abs() < 0.0001);
    }

    #[test]
    fn test_display() {
        let u = latlon_to_utm(51.5074, -0.1278).unwrap();
        let s = format!("{u}");
        assert!(s.contains("30"));
    }

    #[test]
    fn test_convergence_angle() {
        let u = latlon_to_utm(48.8566, 2.3522).unwrap();
        let angle = u.convergence_angle(48.8566);
        assert!(angle.abs() < 5.0);
    }
}
