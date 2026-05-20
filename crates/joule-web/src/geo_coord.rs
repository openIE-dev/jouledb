//! Geographic coordinates: lat/lon/alt representation, WGS84 ellipsoid constants,
//! geodetic-to-ECEF conversion, coordinate validation, DMS parsing, GeoCoord builder.

use std::f64::consts::PI;
use std::fmt;

// ── WGS84 Constants ────────────────────────────────────────────

/// WGS84 semi-major axis (equatorial radius) in metres.
pub const WGS84_A: f64 = 6_378_137.0;

/// WGS84 semi-minor axis (polar radius) in metres.
pub const WGS84_B: f64 = 6_356_752.314_245;

/// WGS84 flattening  f = (a - b) / a.
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;

/// WGS84 first eccentricity squared  e² = 2f - f².
pub const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;

/// WGS84 second eccentricity squared  e'² = (a²-b²)/b².
pub const WGS84_EP2: f64 =
    (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);

// ── Error ──────────────────────────────────────────────────────

/// Errors produced by coordinate operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CoordError {
    /// Latitude out of range -90..=90.
    LatitudeOutOfRange(f64),
    /// Longitude out of range -180..=180.
    LongitudeOutOfRange(f64),
    /// Malformed DMS string.
    InvalidDms(String),
    /// Generic conversion failure.
    ConversionFailed(String),
}

impl fmt::Display for CoordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LatitudeOutOfRange(v) => write!(f, "latitude {v} out of range [-90, 90]"),
            Self::LongitudeOutOfRange(v) => write!(f, "longitude {v} out of range [-180, 180]"),
            Self::InvalidDms(s) => write!(f, "invalid DMS string: {s}"),
            Self::ConversionFailed(s) => write!(f, "conversion failed: {s}"),
        }
    }
}

// ── GeoCoord ───────────────────────────────────────────────────

/// A geographic coordinate in geodetic form (lat, lon, optional altitude).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCoord {
    /// Latitude in decimal degrees, north-positive.
    pub lat: f64,
    /// Longitude in decimal degrees, east-positive.
    pub lon: f64,
    /// Altitude above ellipsoid in metres (optional).
    pub alt: Option<f64>,
}

impl GeoCoord {
    /// Create a new coordinate after validation.
    pub fn new(lat: f64, lon: f64) -> Result<Self, CoordError> {
        validate_lat(lat)?;
        validate_lon(lon)?;
        Ok(Self { lat, lon, alt: None })
    }

    /// Create with altitude.
    pub fn with_alt(mut self, alt: f64) -> Self {
        self.alt = Some(alt);
        self
    }

    /// Latitude in radians.
    pub fn lat_rad(&self) -> f64 {
        self.lat.to_radians()
    }

    /// Longitude in radians.
    pub fn lon_rad(&self) -> f64 {
        self.lon.to_radians()
    }

    /// Convert to Earth-Centred Earth-Fixed (ECEF) cartesian coordinates.
    pub fn to_ecef(&self) -> EcefCoord {
        geodetic_to_ecef(self.lat, self.lon, self.alt.unwrap_or(0.0))
    }

    /// Create from ECEF coordinates.
    pub fn from_ecef(ecef: &EcefCoord) -> Self {
        ecef_to_geodetic(ecef.x, ecef.y, ecef.z)
    }

    /// Convert to DMS representation.
    pub fn to_dms(&self) -> DmsCoord {
        DmsCoord {
            lat: decimal_to_dms(self.lat),
            lat_hem: if self.lat >= 0.0 { 'N' } else { 'S' },
            lon: decimal_to_dms(self.lon),
            lon_hem: if self.lon >= 0.0 { 'E' } else { 'W' },
        }
    }
}

impl fmt::Display for GeoCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.alt {
            Some(a) => write!(f, "({:.6}°, {:.6}°, {:.1}m)", self.lat, self.lon, a),
            None => write!(f, "({:.6}°, {:.6}°)", self.lat, self.lon),
        }
    }
}

// ── GeoCoordBuilder ────────────────────────────────────────────

/// Builder for constructing a `GeoCoord` incrementally.
#[derive(Debug, Clone, Default)]
pub struct GeoCoordBuilder {
    lat: Option<f64>,
    lon: Option<f64>,
    alt: Option<f64>,
}

impl GeoCoordBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_lat(mut self, lat: f64) -> Self {
        self.lat = Some(lat);
        self
    }

    pub fn with_lon(mut self, lon: f64) -> Self {
        self.lon = Some(lon);
        self
    }

    pub fn with_alt(mut self, alt: f64) -> Self {
        self.alt = Some(alt);
        self
    }

    /// Build the coordinate, returning an error if lat/lon are missing or invalid.
    pub fn build(self) -> Result<GeoCoord, CoordError> {
        let lat = self
            .lat
            .ok_or_else(|| CoordError::ConversionFailed("latitude not set".into()))?;
        let lon = self
            .lon
            .ok_or_else(|| CoordError::ConversionFailed("longitude not set".into()))?;
        let mut c = GeoCoord::new(lat, lon)?;
        c.alt = self.alt;
        Ok(c)
    }
}

// ── ECEF ───────────────────────────────────────────────────────

/// Earth-Centred Earth-Fixed cartesian coordinate (metres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcefCoord {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl EcefCoord {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance to another ECEF point.
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl fmt::Display for EcefCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ECEF({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

// ── DMS ────────────────────────────────────────────────────────

/// Degrees-minutes-seconds triple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dms {
    pub degrees: i32,
    pub minutes: u32,
    pub seconds: f64,
}

impl fmt::Display for Dms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}°{:02}'{:06.3}\"", self.degrees.abs(), self.minutes, self.seconds)
    }
}

/// A full DMS coordinate pair with hemisphere indicators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmsCoord {
    pub lat: Dms,
    pub lat_hem: char,
    pub lon: Dms,
    pub lon_hem: char,
}

impl fmt::Display for DmsCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}, {} {}", self.lat, self.lat_hem, self.lon, self.lon_hem)
    }
}

// ── Validation ─────────────────────────────────────────────────

/// Validate latitude is within -90..=90.
pub fn validate_lat(lat: f64) -> Result<(), CoordError> {
    if lat.is_nan() || lat < -90.0 || lat > 90.0 {
        Err(CoordError::LatitudeOutOfRange(lat))
    } else {
        Ok(())
    }
}

/// Validate longitude is within -180..=180.
pub fn validate_lon(lon: f64) -> Result<(), CoordError> {
    if lon.is_nan() || lon < -180.0 || lon > 180.0 {
        Err(CoordError::LongitudeOutOfRange(lon))
    } else {
        Ok(())
    }
}

/// Normalise longitude to -180..180.
pub fn normalize_lon(lon: f64) -> f64 {
    let mut l = lon % 360.0;
    if l > 180.0 {
        l -= 360.0;
    } else if l < -180.0 {
        l += 360.0;
    }
    l
}

// ── Geodetic ↔ ECEF ────────────────────────────────────────────

/// Convert geodetic (deg, deg, m) to ECEF.
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, alt: f64) -> EcefCoord {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    EcefCoord {
        x: (n + alt) * cos_lat * lon.cos(),
        y: (n + alt) * cos_lat * lon.sin(),
        z: (n * (1.0 - WGS84_E2) + alt) * sin_lat,
    }
}

/// Convert ECEF to geodetic (Bowring iterative method).
pub fn ecef_to_geodetic(x: f64, y: f64, z: f64) -> GeoCoord {
    let lon = y.atan2(x);
    let p = (x * x + y * y).sqrt();
    // initial estimate using parametric latitude
    let mut lat = z.atan2(p * (1.0 - WGS84_E2));
    for _ in 0..10 {
        let sin_lat = lat.sin();
        let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
        lat = (z + WGS84_E2 * n * sin_lat).atan2(p);
    }
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    let alt = if cos_lat.abs() > 1e-10 {
        p / cos_lat - n
    } else {
        z.abs() / sin_lat.abs() - n * (1.0 - WGS84_E2)
    };
    GeoCoord {
        lat: lat.to_degrees(),
        lon: lon.to_degrees(),
        alt: Some(alt),
    }
}

// ── DMS helpers ────────────────────────────────────────────────

/// Convert decimal degrees to DMS.
pub fn decimal_to_dms(dd: f64) -> Dms {
    let total = dd.abs();
    let deg = total.floor() as i32;
    let min_f = (total - deg as f64) * 60.0;
    let min = min_f.floor() as u32;
    let sec = (min_f - min as f64) * 60.0;
    Dms {
        degrees: if dd < 0.0 { -deg } else { deg },
        minutes: min,
        seconds: sec,
    }
}

/// Convert DMS to decimal degrees.
pub fn dms_to_decimal(deg: i32, min: u32, sec: f64) -> f64 {
    let sign = if deg < 0 { -1.0 } else { 1.0 };
    sign * ((deg.unsigned_abs() as f64) + (min as f64) / 60.0 + sec / 3600.0)
}

/// Parse a DMS string like `40°26'46.3"N` or `40 26 46.3 N`.
pub fn parse_dms(s: &str) -> Result<f64, CoordError> {
    let s = s.trim();
    // Detect hemisphere suffix
    let (body, hem) = if let Some(last) = s.chars().last() {
        match last {
            'N' | 'S' | 'E' | 'W' | 'n' | 's' | 'e' | 'w' => {
                (&s[..s.len() - 1], Some(last.to_ascii_uppercase()))
            }
            _ => (s, None),
        }
    } else {
        return Err(CoordError::InvalidDms(s.to_string()));
    };

    // Replace symbols with spaces
    let cleaned: String = body
        .chars()
        .map(|c| match c {
            '°' | '\'' | '"' | '′' | '″' => ' ',
            _ => c,
        })
        .collect();

    let parts: Vec<&str> = cleaned.split_whitespace().collect();
    if parts.len() < 2 || parts.len() > 3 {
        return Err(CoordError::InvalidDms(s.to_string()));
    }

    let deg: i32 = parts[0]
        .parse()
        .map_err(|_| CoordError::InvalidDms(s.to_string()))?;
    let min: u32 = parts[1]
        .parse()
        .map_err(|_| CoordError::InvalidDms(s.to_string()))?;
    let sec: f64 = if parts.len() == 3 {
        parts[2]
            .parse()
            .map_err(|_| CoordError::InvalidDms(s.to_string()))?
    } else {
        0.0
    };

    let mut dd = dms_to_decimal(deg.abs(), min, sec);
    if hem == Some('S') || hem == Some('W') || deg < 0 {
        dd = -dd;
    }
    Ok(dd)
}

/// Compute the prime vertical radius of curvature N for a given latitude.
pub fn prime_vertical_radius(lat_deg: f64) -> f64 {
    let sin_lat = lat_deg.to_radians().sin();
    WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt()
}

/// Compute the meridional radius of curvature M for a given latitude.
pub fn meridional_radius(lat_deg: f64) -> f64 {
    let sin_lat = lat_deg.to_radians().sin();
    let denom = 1.0 - WGS84_E2 * sin_lat * sin_lat;
    WGS84_A * (1.0 - WGS84_E2) / (denom * denom.sqrt())
}

/// Angular distance (radians) between two coordinates on a unit sphere.
pub fn angular_distance(c1: &GeoCoord, c2: &GeoCoord) -> f64 {
    let lat1 = c1.lat_rad();
    let lat2 = c2.lat_rad();
    let dlon = c2.lon_rad() - c1.lon_rad();
    let cos1 = lat1.cos();
    let cos2 = lat2.cos();
    let sin1 = lat1.sin();
    let sin2 = lat2.sin();
    let num = ((cos2 * dlon.sin()).powi(2) + (cos1 * sin2 - sin1 * cos2 * dlon.cos()).powi(2))
        .sqrt();
    let den = sin1 * sin2 + cos1 * cos2 * dlon.cos();
    num.atan2(den)
}

/// Approximate radius at a given latitude in metres (ellipsoidal).
pub fn earth_radius_at_lat(lat_deg: f64) -> f64 {
    let lat = lat_deg.to_radians();
    let cos_lat = lat.cos();
    let sin_lat = lat.sin();
    let a2 = WGS84_A * WGS84_A;
    let b2 = WGS84_B * WGS84_B;
    let num = (a2 * cos_lat).powi(2) + (b2 * sin_lat).powi(2);
    let den = (a2 * cos_lat * cos_lat) + (b2 * sin_lat * sin_lat);
    (num / den).sqrt()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;
    const TOL_M: f64 = 1.0; // 1 metre tolerance for ECEF round-trips

    #[test]
    fn test_wgs84_constants() {
        assert!((WGS84_F - 1.0 / 298.257_223_563).abs() < 1e-15);
        assert!(WGS84_E2 > 0.0 && WGS84_E2 < 0.01);
    }

    #[test]
    fn test_geo_coord_valid() {
        let c = GeoCoord::new(40.0, -74.0).unwrap();
        assert!((c.lat - 40.0).abs() < TOL);
    }

    #[test]
    fn test_geo_coord_invalid_lat() {
        assert!(GeoCoord::new(91.0, 0.0).is_err());
        assert!(GeoCoord::new(-91.0, 0.0).is_err());
    }

    #[test]
    fn test_geo_coord_invalid_lon() {
        assert!(GeoCoord::new(0.0, 181.0).is_err());
        assert!(GeoCoord::new(0.0, -181.0).is_err());
    }

    #[test]
    fn test_geo_coord_with_alt() {
        let c = GeoCoord::new(0.0, 0.0).unwrap().with_alt(100.0);
        assert_eq!(c.alt, Some(100.0));
    }

    #[test]
    fn test_builder() {
        let c = GeoCoordBuilder::new()
            .with_lat(51.5074)
            .with_lon(-0.1278)
            .with_alt(11.0)
            .build()
            .unwrap();
        assert!((c.lat - 51.5074).abs() < TOL);
        assert_eq!(c.alt, Some(11.0));
    }

    #[test]
    fn test_builder_missing_lat() {
        let r = GeoCoordBuilder::new().with_lon(0.0).build();
        assert!(r.is_err());
    }

    #[test]
    fn test_ecef_roundtrip_equator() {
        let orig = GeoCoord::new(0.0, 0.0).unwrap().with_alt(0.0);
        let ecef = orig.to_ecef();
        let back = GeoCoord::from_ecef(&ecef);
        assert!((back.lat - orig.lat).abs() < TOL);
        assert!((back.lon - orig.lon).abs() < TOL);
        assert!((back.alt.unwrap() - 0.0).abs() < TOL_M);
    }

    #[test]
    fn test_ecef_roundtrip_pole() {
        let orig = GeoCoord::new(90.0, 0.0).unwrap().with_alt(0.0);
        let ecef = orig.to_ecef();
        let back = GeoCoord::from_ecef(&ecef);
        assert!((back.lat - 90.0).abs() < TOL);
    }

    #[test]
    fn test_ecef_roundtrip_negative() {
        let orig = GeoCoord::new(-33.8688, 151.2093).unwrap().with_alt(50.0);
        let ecef = orig.to_ecef();
        let back = GeoCoord::from_ecef(&ecef);
        assert!((back.lat - orig.lat).abs() < TOL);
        assert!((back.lon - orig.lon).abs() < TOL);
        assert!((back.alt.unwrap() - 50.0).abs() < TOL_M);
    }

    #[test]
    fn test_ecef_distance() {
        let a = EcefCoord::new(1.0, 0.0, 0.0);
        let b = EcefCoord::new(0.0, 1.0, 0.0);
        assert!((a.distance_to(&b) - 2.0_f64.sqrt()).abs() < TOL);
    }

    #[test]
    fn test_decimal_to_dms() {
        let d = decimal_to_dms(40.446195);
        assert_eq!(d.degrees, 40);
        assert_eq!(d.minutes, 26);
        assert!((d.seconds - 46.302).abs() < 0.01);
    }

    #[test]
    fn test_dms_to_decimal() {
        let dd = dms_to_decimal(40, 26, 46.302);
        assert!((dd - 40.446195).abs() < 1e-4);
    }

    #[test]
    fn test_parse_dms_standard() {
        let dd = parse_dms("40°26'46.3\"N").unwrap();
        assert!((dd - 40.44619).abs() < 1e-3);
    }

    #[test]
    fn test_parse_dms_south() {
        let dd = parse_dms("33 51 54.0 S").unwrap();
        assert!(dd < 0.0);
        assert!((dd + 33.865).abs() < 1e-2);
    }

    #[test]
    fn test_normalize_lon() {
        assert!((normalize_lon(190.0) - (-170.0)).abs() < TOL);
        assert!((normalize_lon(-200.0) - 160.0).abs() < TOL);
        assert!((normalize_lon(0.0) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_display() {
        let c = GeoCoord::new(51.5074, -0.1278).unwrap();
        let s = format!("{c}");
        assert!(s.contains("51.507"));
    }

    #[test]
    fn test_prime_vertical_radius() {
        let n = prime_vertical_radius(0.0);
        assert!((n - WGS84_A).abs() < 1.0);
    }

    #[test]
    fn test_earth_radius_at_poles() {
        let r = earth_radius_at_lat(90.0);
        assert!((r - WGS84_B).abs() < 1.0);
    }

    #[test]
    fn test_angular_distance_same_point() {
        let c = GeoCoord::new(45.0, 90.0).unwrap();
        assert!(angular_distance(&c, &c).abs() < 1e-12);
    }
}
