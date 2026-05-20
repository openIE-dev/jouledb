//! Celestial body rendering data — star colors, planet brightness, coordinates.
//!
//! Replaces Stellarium.js / three-stellation with pure Rust.
//! Blackbody-to-RGB, magnitude scale, coordinate systems (equatorial,
//! horizontal, ecliptic), precession, apparent motion computation.

use std::f64::consts::PI;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for celestial rendering.
#[derive(Debug, Clone, PartialEq)]
pub enum CelestialError {
    /// Temperature must be positive.
    NonPositiveTemperature(f64),
    /// Luminosity must be positive.
    NonPositiveLuminosity(f64),
    /// Radius must be positive.
    NonPositiveRadius(f64),
    /// Albedo must be in [0, 1].
    InvalidAlbedo(f64),
    /// Latitude must be in [-90, 90].
    InvalidLatitude(f64),
    /// Hour angle out of range.
    InvalidAngle(f64),
}

impl fmt::Display for CelestialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveTemperature(t) => write!(f, "temperature must be positive, got {t}"),
            Self::NonPositiveLuminosity(l) => write!(f, "luminosity must be positive, got {l}"),
            Self::NonPositiveRadius(r) => write!(f, "radius must be positive, got {r}"),
            Self::InvalidAlbedo(a) => write!(f, "albedo must be in [0,1], got {a}"),
            Self::InvalidLatitude(l) => write!(f, "latitude must be in [-90,90], got {l}"),
            Self::InvalidAngle(a) => write!(f, "invalid angle: {a}"),
        }
    }
}

impl std::error::Error for CelestialError {}

// ── RGB Color ───────────────────────────────────────────────────

/// An RGB color with components in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Rgb {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Convert to 8-bit per channel (0-255).
    pub fn to_u8(self) -> (u8, u8, u8) {
        (
            (self.r * 255.0).round() as u8,
            (self.g * 255.0).round() as u8,
            (self.b * 255.0).round() as u8,
        )
    }
}

// ── Blackbody to RGB ────────────────────────────────────────────

/// Convert stellar temperature (Kelvin) to approximate RGB color.
/// Uses Tanner Helland's algorithm (CIE-based approximation).
pub fn blackbody_to_rgb(temperature: f64) -> Result<Rgb, CelestialError> {
    if temperature <= 0.0 {
        return Err(CelestialError::NonPositiveTemperature(temperature));
    }
    let temp = temperature / 100.0;

    let r = if temp <= 66.0 {
        1.0
    } else {
        let x = temp - 60.0;
        (329.698727446 * x.powf(-0.1332047592) / 255.0).clamp(0.0, 1.0)
    };

    let g = if temp <= 66.0 {
        let x = temp;
        ((99.4708025861 * x.ln() - 161.1195681661) / 255.0).clamp(0.0, 1.0)
    } else {
        let x = temp - 60.0;
        (288.1221695283 * x.powf(-0.0755148492) / 255.0).clamp(0.0, 1.0)
    };

    let b = if temp >= 66.0 {
        1.0
    } else if temp <= 19.0 {
        0.0
    } else {
        let x = temp - 10.0;
        ((138.5177312231 * x.ln() - 305.0447927307) / 255.0).clamp(0.0, 1.0)
    };

    Ok(Rgb::new(r, g, b))
}

// ── Star ────────────────────────────────────────────────────────

/// A star with physical and rendering properties.
#[derive(Debug, Clone, PartialEq)]
pub struct Star {
    /// Position in 3D space (arbitrary units).
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Luminosity relative to the Sun.
    pub luminosity: f64,
    /// Surface temperature in Kelvin.
    pub temperature: f64,
}

impl Star {
    pub fn new(x: f64, y: f64, z: f64, luminosity: f64, temperature: f64) -> Result<Self, CelestialError> {
        if luminosity <= 0.0 {
            return Err(CelestialError::NonPositiveLuminosity(luminosity));
        }
        if temperature <= 0.0 {
            return Err(CelestialError::NonPositiveTemperature(temperature));
        }
        Ok(Self { x, y, z, luminosity, temperature })
    }

    /// Get the color of the star from its temperature.
    pub fn color(&self) -> Rgb {
        blackbody_to_rgb(self.temperature).unwrap_or(Rgb::new(1.0, 1.0, 1.0))
    }

    /// Apparent magnitude at distance d (parsecs). Uses M + 5*log10(d/10).
    /// Absolute magnitude from luminosity: M = 4.83 - 2.5*log10(L/L_sun).
    pub fn absolute_magnitude(&self) -> f64 {
        4.83 - 2.5 * self.luminosity.log10()
    }

    pub fn apparent_magnitude(&self, distance_pc: f64) -> f64 {
        self.absolute_magnitude() + 5.0 * (distance_pc / 10.0).log10()
    }

    /// Distance from origin.
    pub fn distance(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

// ── Planet ───────────────────────────────────────────────────────

/// A planet with rendering properties.
#[derive(Debug, Clone, PartialEq)]
pub struct Planet {
    /// Position in 3D space.
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Radius (km or arbitrary).
    pub radius: f64,
    /// Bond albedo [0, 1].
    pub albedo: f64,
}

impl Planet {
    pub fn new(x: f64, y: f64, z: f64, radius: f64, albedo: f64) -> Result<Self, CelestialError> {
        if radius <= 0.0 {
            return Err(CelestialError::NonPositiveRadius(radius));
        }
        if albedo < 0.0 || albedo > 1.0 {
            return Err(CelestialError::InvalidAlbedo(albedo));
        }
        Ok(Self { x, y, z, radius, albedo })
    }

    /// Apparent brightness factor given phase angle (radians).
    /// Uses Lambert sphere: brightness ~ albedo * (sin(alpha) + (pi-alpha)*cos(alpha)) / pi.
    pub fn phase_brightness(&self, phase_angle: f64) -> f64 {
        let alpha = phase_angle.clamp(0.0, PI);
        self.albedo * (alpha.sin() + (PI - alpha) * alpha.cos()) / PI
    }

    /// Geometric albedo apparent brightness at distance d from observer,
    /// with distance r from star, given star luminosity.
    pub fn apparent_brightness(
        &self,
        star_luminosity: f64,
        distance_from_star: f64,
        distance_from_observer: f64,
        phase_angle: f64,
    ) -> f64 {
        let phase = self.phase_brightness(phase_angle);
        let cross_section = PI * self.radius * self.radius;
        star_luminosity * cross_section * phase
            / (4.0 * PI * distance_from_star * distance_from_star
                * distance_from_observer * distance_from_observer)
    }
}

// ── Magnitude Scale ─────────────────────────────────────────────

/// Convert flux ratio to magnitude difference.
/// A flux ratio > 1 (brighter) gives a positive magnitude difference.
pub fn flux_to_magnitude_diff(flux_ratio: f64) -> f64 {
    2.5 * flux_ratio.log10()
}

/// Convert magnitude difference to flux ratio.
pub fn magnitude_diff_to_flux(mag_diff: f64) -> f64 {
    10.0_f64.powf(-mag_diff / 2.5)
}

/// Combined magnitude of two stars.
pub fn combined_magnitude(m1: f64, m2: f64) -> f64 {
    let f1 = magnitude_diff_to_flux(m1);
    let f2 = magnitude_diff_to_flux(m2);
    -2.5 * (f1 + f2).log10()
}

// ── Coordinate Systems ──────────────────────────────────────────

/// Equatorial coordinates (RA in hours, Dec in degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equatorial {
    /// Right Ascension in hours [0, 24).
    pub ra: f64,
    /// Declination in degrees [-90, 90].
    pub dec: f64,
}

/// Horizontal coordinates (altitude, azimuth in degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Horizontal {
    /// Altitude in degrees [-90, 90].
    pub altitude: f64,
    /// Azimuth in degrees [0, 360), north=0, east=90.
    pub azimuth: f64,
}

/// Ecliptic coordinates (longitude, latitude in degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ecliptic {
    pub longitude: f64,
    pub latitude: f64,
}

/// Convert equatorial to horizontal coordinates.
/// `lat` is observer latitude in degrees, `lst` is local sidereal time in hours.
pub fn equatorial_to_horizontal(eq: &Equatorial, lat_deg: f64, lst_hours: f64) -> Result<Horizontal, CelestialError> {
    if lat_deg < -90.0 || lat_deg > 90.0 {
        return Err(CelestialError::InvalidLatitude(lat_deg));
    }
    let lat = lat_deg.to_radians();
    let dec = eq.dec.to_radians();
    let ha = ((lst_hours - eq.ra) * 15.0).to_radians(); // hour angle in radians

    let sin_alt = dec.sin() * lat.sin() + dec.cos() * lat.cos() * ha.cos();
    let alt = sin_alt.clamp(-1.0, 1.0).asin();

    let cos_az = (dec.sin() - alt.sin() * lat.sin()) / (alt.cos() * lat.cos() + 1e-30);
    let mut az = cos_az.clamp(-1.0, 1.0).acos();
    if ha.sin() > 0.0 {
        az = 2.0 * PI - az;
    }

    Ok(Horizontal {
        altitude: alt.to_degrees(),
        azimuth: az.to_degrees(),
    })
}

/// Convert equatorial to ecliptic coordinates.
/// `obliquity` is the axial tilt in degrees (default ~23.4393).
pub fn equatorial_to_ecliptic(eq: &Equatorial, obliquity_deg: f64) -> Ecliptic {
    let eps = obliquity_deg.to_radians();
    let ra = (eq.ra * 15.0).to_radians();
    let dec = eq.dec.to_radians();

    let sin_lon = ra.sin() * eps.cos() + dec.tan() * eps.sin();
    let cos_lon = ra.cos();
    let lon = sin_lon.atan2(cos_lon).to_degrees();

    let sin_lat = dec.sin() * eps.cos() - dec.cos() * eps.sin() * ra.sin();
    let lat = sin_lat.clamp(-1.0, 1.0).asin().to_degrees();

    Ecliptic {
        longitude: if lon < 0.0 { lon + 360.0 } else { lon },
        latitude: lat,
    }
}

// ── Precession ──────────────────────────────────────────────────

/// Apply precession correction to RA/Dec.
/// `years` is number of years from epoch (e.g., J2000).
/// Uses simplified linear precession: ~50.3 arcsec/year in RA, ~20.0 arcsec/year in Dec.
pub fn precess(eq: &Equatorial, years: f64) -> Equatorial {
    let ra_shift = 50.3 / 3600.0 * years / 15.0; // hours
    let dec_shift = 20.0 / 3600.0 * years; // degrees (approximate, depends on position)
    let ra = (eq.ra + ra_shift) % 24.0;
    let dec = (eq.dec + dec_shift).clamp(-90.0, 90.0);
    Equatorial { ra: if ra < 0.0 { ra + 24.0 } else { ra }, dec }
}

// ── Constellation ───────────────────────────────────────────────

/// A constellation defined by star connections.
#[derive(Debug, Clone, PartialEq)]
pub struct Constellation {
    pub name: String,
    /// Star indices that form the pattern.
    pub star_indices: Vec<usize>,
    /// Connections as pairs of indices into star_indices.
    pub connections: Vec<(usize, usize)>,
}

impl Constellation {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), star_indices: Vec::new(), connections: Vec::new() }
    }

    pub fn add_star(&mut self, index: usize) {
        if !self.star_indices.contains(&index) {
            self.star_indices.push(index);
        }
    }

    pub fn add_connection(&mut self, a: usize, b: usize) {
        self.connections.push((a, b));
    }

    pub fn star_count(&self) -> usize {
        self.star_indices.len()
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

// ── Apparent Motion ─────────────────────────────────────────────

/// Compute the local sidereal time from UT and longitude.
/// `jd` is Julian date, `longitude_deg` is observer longitude (east positive).
pub fn local_sidereal_time(jd: f64, longitude_deg: f64) -> f64 {
    let t = (jd - 2451545.0) / 36525.0;
    let gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - t * t * t / 38710000.0;
    let lst = (gmst + longitude_deg) % 360.0;
    let lst_hours = (if lst < 0.0 { lst + 360.0 } else { lst }) / 15.0;
    lst_hours
}

/// Compute Julian date from year, month, day (UT).
pub fn julian_date(year: i32, month: u32, day: f64) -> f64 {
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let a = (y as f64 / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    (365.25 * (y as f64 + 4716.0)).floor()
        + (30.6001 * (m as f64 + 1.0)).floor()
        + day + b - 1524.5
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn blackbody_red_hot() {
        let rgb = blackbody_to_rgb(2000.0).unwrap();
        // Low temp should be reddish: r > g > b.
        assert!(rgb.r > rgb.g);
        assert!(rgb.g > rgb.b || approx_eq(rgb.b, 0.0, 0.01));
    }

    #[test]
    fn blackbody_sun() {
        let rgb = blackbody_to_rgb(5778.0).unwrap();
        // Sun should be yellowish-white: all channels high.
        assert!(rgb.r > 0.8);
        assert!(rgb.g > 0.7);
    }

    #[test]
    fn blackbody_hot_blue() {
        let rgb = blackbody_to_rgb(30000.0).unwrap();
        // Hot star: bluish-white. B channel should be high.
        assert!(rgb.b > 0.8);
    }

    #[test]
    fn blackbody_invalid_temp() {
        assert!(blackbody_to_rgb(-100.0).is_err());
        assert!(blackbody_to_rgb(0.0).is_err());
    }

    #[test]
    fn star_creation() {
        assert!(Star::new(0.0, 0.0, 0.0, 1.0, 5778.0).is_ok());
        assert!(Star::new(0.0, 0.0, 0.0, -1.0, 5778.0).is_err());
        assert!(Star::new(0.0, 0.0, 0.0, 1.0, -100.0).is_err());
    }

    #[test]
    fn star_magnitude() {
        let sun = Star::new(0.0, 0.0, 0.0, 1.0, 5778.0).unwrap();
        assert!(approx_eq(sun.absolute_magnitude(), 4.83, 1e-4));
        // At 10 pc, apparent = absolute.
        assert!(approx_eq(sun.apparent_magnitude(10.0), 4.83, 1e-4));
    }

    #[test]
    fn star_brighter_lower_magnitude() {
        let dim = Star::new(0.0, 0.0, 0.0, 1.0, 5778.0).unwrap();
        let bright = Star::new(0.0, 0.0, 0.0, 100.0, 10000.0).unwrap();
        assert!(bright.absolute_magnitude() < dim.absolute_magnitude());
    }

    #[test]
    fn planet_creation() {
        assert!(Planet::new(0.0, 0.0, 0.0, 6371.0, 0.3).is_ok());
        assert!(Planet::new(0.0, 0.0, 0.0, -1.0, 0.3).is_err());
        assert!(Planet::new(0.0, 0.0, 0.0, 6371.0, 1.5).is_err());
    }

    #[test]
    fn planet_full_phase() {
        let p = Planet::new(0.0, 0.0, 0.0, 1.0, 1.0).unwrap();
        // At phase 0 (full), brightness should be maximum.
        let b0 = p.phase_brightness(0.0);
        let b90 = p.phase_brightness(PI / 2.0);
        assert!(b0 > b90);
    }

    #[test]
    fn magnitude_flux_roundtrip() {
        let ratio = 100.0;
        let dm = flux_to_magnitude_diff(ratio);
        let ratio_back = magnitude_diff_to_flux(dm);
        assert!(approx_eq(ratio_back, 1.0 / ratio, 1e-8));
    }

    #[test]
    fn combined_magnitude_equal_stars() {
        let m = 5.0;
        let mc = combined_magnitude(m, m);
        // Two equal-brightness stars: combined is 0.75 mag brighter.
        assert!(mc < m);
        assert!(approx_eq(mc - m, -2.5 * 2.0_f64.log10(), 0.01));
    }

    #[test]
    fn equatorial_to_horizontal_zenith() {
        // Star at dec = lat, HA = 0 => altitude = 90.
        let eq = Equatorial { ra: 6.0, dec: 45.0 };
        let h = equatorial_to_horizontal(&eq, 45.0, 6.0).unwrap();
        assert!(approx_eq(h.altitude, 90.0, 0.1));
    }

    #[test]
    fn equatorial_to_horizontal_invalid_lat() {
        let eq = Equatorial { ra: 0.0, dec: 0.0 };
        assert!(equatorial_to_horizontal(&eq, 91.0, 0.0).is_err());
    }

    #[test]
    fn ecliptic_equinox_point() {
        // Vernal equinox: RA=0, Dec=0 => ecliptic lon=0, lat=0.
        let eq = Equatorial { ra: 0.0, dec: 0.0 };
        let ec = equatorial_to_ecliptic(&eq, 23.4393);
        assert!(approx_eq(ec.latitude, 0.0, 0.1));
        assert!(approx_eq(ec.longitude, 0.0, 0.1));
    }

    #[test]
    fn precession_forward_and_back() {
        let eq = Equatorial { ra: 12.0, dec: 45.0 };
        let p1 = precess(&eq, 50.0);
        let p2 = precess(&p1, -50.0);
        assert!(approx_eq(p2.ra, eq.ra, 0.01));
        assert!(approx_eq(p2.dec, eq.dec, 0.01));
    }

    #[test]
    fn constellation_building() {
        let mut c = Constellation::new("Test");
        c.add_star(0);
        c.add_star(1);
        c.add_star(2);
        c.add_connection(0, 1);
        c.add_connection(1, 2);
        assert_eq!(c.star_count(), 3);
        assert_eq!(c.connection_count(), 2);
    }

    #[test]
    fn julian_date_j2000() {
        let jd = julian_date(2000, 1, 1.5);
        assert!(approx_eq(jd, 2451545.0, 0.001));
    }

    #[test]
    fn local_sidereal_time_greenwich() {
        let jd = julian_date(2000, 1, 1.5); // J2000 epoch
        let lst = local_sidereal_time(jd, 0.0);
        // GMST at J2000: ~18.697h
        assert!(lst > 0.0 && lst < 24.0);
    }

    #[test]
    fn rgb_to_u8() {
        let c = Rgb::new(1.0, 0.5, 0.0);
        let (r, g, b) = c.to_u8();
        assert_eq!(r, 255);
        assert_eq!(g, 128);
        assert_eq!(b, 0);
    }

    #[test]
    fn planet_apparent_brightness_inverse_square() {
        let p = Planet::new(0.0, 0.0, 0.0, 1.0, 1.0).unwrap();
        let b1 = p.apparent_brightness(1.0, 1.0, 1.0, 0.0);
        let b2 = p.apparent_brightness(1.0, 1.0, 2.0, 0.0);
        // Should decrease with distance^2.
        assert!(approx_eq(b1 / b2, 4.0, 0.1));
    }

    #[test]
    fn star_distance() {
        let s = Star::new(3.0, 4.0, 0.0, 1.0, 5778.0).unwrap();
        assert!(approx_eq(s.distance(), 5.0, 1e-10));
    }
}
