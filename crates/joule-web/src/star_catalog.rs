//! Star catalog data structure — catalog operations, spectral types, HR diagram.
//!
//! Replaces HYG database / astroquery.js with pure Rust.
//! Star entries with RA, Dec, magnitude, spectral type, distance.
//! Cone search, magnitude filtering, nearest-N, HR diagram, proper motion,
//! binary star systems.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for star catalog.
#[derive(Debug, Clone, PartialEq)]
pub enum CatalogError {
    /// Invalid spectral class.
    InvalidSpectralType(String),
    /// Magnitude range inverted.
    InvalidMagnitudeRange { min_val: f64, max_val: f64 },
    /// RA must be in [0, 360) degrees.
    InvalidRA(f64),
    /// Dec must be in [-90, 90] degrees.
    InvalidDec(f64),
    /// Search radius must be positive.
    NonPositiveRadius(f64),
    /// Distance must be positive.
    NonPositiveDistance(f64),
    /// N must be at least 1.
    InvalidN(usize),
    /// Star not found.
    StarNotFound(String),
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpectralType(s) => write!(f, "invalid spectral type: {s}"),
            Self::InvalidMagnitudeRange { min_val, max_val } => {
                write!(f, "magnitude range inverted: {min_val} > {max_val}")
            }
            Self::InvalidRA(ra) => write!(f, "RA must be in [0, 360), got {ra}"),
            Self::InvalidDec(dec) => write!(f, "Dec must be in [-90, 90], got {dec}"),
            Self::NonPositiveRadius(r) => write!(f, "radius must be positive, got {r}"),
            Self::NonPositiveDistance(d) => write!(f, "distance must be positive, got {d}"),
            Self::InvalidN(n) => write!(f, "N must be >= 1, got {n}"),
            Self::StarNotFound(s) => write!(f, "star not found: {s}"),
        }
    }
}

impl std::error::Error for CatalogError {}

// ── Spectral Type ───────────────────────────────────────────────

/// MK spectral classification (Harvard sequence).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpectralClass {
    O,
    B,
    A,
    F,
    G,
    K,
    M,
}

impl SpectralClass {
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'O' => Some(Self::O),
            'B' => Some(Self::B),
            'A' => Some(Self::A),
            'F' => Some(Self::F),
            'G' => Some(Self::G),
            'K' => Some(Self::K),
            'M' => Some(Self::M),
            _ => None,
        }
    }

    pub fn as_char(self) -> char {
        match self {
            Self::O => 'O',
            Self::B => 'B',
            Self::A => 'A',
            Self::F => 'F',
            Self::G => 'G',
            Self::K => 'K',
            Self::M => 'M',
        }
    }

    /// Approximate effective temperature (K) for the class midpoint.
    pub fn temperature(self) -> f64 {
        match self {
            Self::O => 40000.0,
            Self::B => 20000.0,
            Self::A => 8500.0,
            Self::F => 6500.0,
            Self::G => 5500.0,
            Self::K => 4000.0,
            Self::M => 3000.0,
        }
    }
}

impl fmt::Display for SpectralClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

/// Parsed spectral type: class + subclass (0-9).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralType {
    pub class: SpectralClass,
    pub subclass: u8,
}

impl SpectralType {
    pub fn parse(s: &str) -> Result<Self, CatalogError> {
        let mut chars = s.chars();
        let class_char = chars.next().ok_or_else(|| CatalogError::InvalidSpectralType(s.to_string()))?;
        let class = SpectralClass::from_char(class_char)
            .ok_or_else(|| CatalogError::InvalidSpectralType(s.to_string()))?;
        let subclass = match chars.next() {
            Some(c) if c.is_ascii_digit() => c as u8 - b'0',
            _ => 5, // default subclass
        };
        Ok(Self { class, subclass })
    }

    /// Approximate temperature from spectral type (linear interpolation within class).
    pub fn temperature(&self) -> f64 {
        let t_base = self.class.temperature();
        let t_next = match self.class {
            SpectralClass::O => 40000.0,
            SpectralClass::B => 10000.0, // B9 -> A0
            SpectralClass::A => 7500.0,
            SpectralClass::F => 6000.0,
            SpectralClass::G => 5000.0,
            SpectralClass::K => 3500.0,
            SpectralClass::M => 2300.0,
        };
        let frac = self.subclass as f64 / 10.0;
        t_base + (t_next - t_base) * frac
    }
}

impl fmt::Display for SpectralType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.class, self.subclass)
    }
}

// ── Star Entry ──────────────────────────────────────────────────

/// A star catalog entry.
#[derive(Debug, Clone, PartialEq)]
pub struct StarEntry {
    /// Common name (e.g., "Sirius"). May be empty.
    pub name: String,
    /// Catalog designation (e.g., "Alpha CMa").
    pub designation: String,
    /// Right ascension in degrees [0, 360).
    pub ra_deg: f64,
    /// Declination in degrees [-90, 90].
    pub dec_deg: f64,
    /// Apparent visual magnitude.
    pub magnitude: f64,
    /// Spectral type.
    pub spectral_type: SpectralType,
    /// Distance in parsecs.
    pub distance_pc: f64,
    /// Proper motion in RA (mas/yr).
    pub pm_ra: f64,
    /// Proper motion in Dec (mas/yr).
    pub pm_dec: f64,
}

impl StarEntry {
    /// Absolute magnitude: M = m - 5*log10(d/10).
    pub fn absolute_magnitude(&self) -> f64 {
        self.magnitude - 5.0 * (self.distance_pc / 10.0).log10()
    }

    /// Luminosity relative to the Sun from absolute magnitude.
    pub fn luminosity(&self) -> f64 {
        10.0_f64.powf((4.83 - self.absolute_magnitude()) / 2.5)
    }

    /// Total proper motion in mas/yr.
    pub fn total_proper_motion(&self) -> f64 {
        (self.pm_ra * self.pm_ra + self.pm_dec * self.pm_dec).sqrt()
    }

    /// Tangential velocity in km/s (approximate).
    pub fn tangential_velocity(&self) -> f64 {
        // v_t = 4.74047 * mu(arcsec/yr) * d(pc)
        let mu_arcsec = self.total_proper_motion() / 1000.0;
        4.74047 * mu_arcsec * self.distance_pc
    }
}

// ── Binary Star System ──────────────────────────────────────────

/// A binary or multiple star system.
#[derive(Debug, Clone, PartialEq)]
pub struct BinarySystem {
    pub primary: StarEntry,
    pub companions: Vec<StarEntry>,
    /// Orbital period in years (if known).
    pub orbital_period_yr: Option<f64>,
    /// Separation in arcseconds (if known).
    pub separation_arcsec: Option<f64>,
}

impl BinarySystem {
    pub fn new(primary: StarEntry) -> Self {
        Self { primary, companions: Vec::new(), orbital_period_yr: None, separation_arcsec: None }
    }

    pub fn add_companion(&mut self, star: StarEntry) {
        self.companions.push(star);
    }

    pub fn component_count(&self) -> usize {
        1 + self.companions.len()
    }

    /// Combined apparent magnitude.
    pub fn combined_magnitude(&self) -> f64 {
        let mut total_flux = 10.0_f64.powf(-self.primary.magnitude / 2.5);
        for c in &self.companions {
            total_flux += 10.0_f64.powf(-c.magnitude / 2.5);
        }
        -2.5 * total_flux.log10()
    }
}

// ── Star Catalog ────────────────────────────────────────────────

/// A collection of star entries with query operations.
#[derive(Debug, Clone)]
pub struct StarCatalog {
    pub entries: Vec<StarEntry>,
}

impl StarCatalog {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn add(&mut self, entry: StarEntry) {
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find star by name.
    pub fn find_by_name(&self, name: &str) -> Option<&StarEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Query stars within a magnitude range.
    pub fn by_magnitude_range(
        &self,
        min_mag: f64,
        max_mag: f64,
    ) -> Result<Vec<&StarEntry>, CatalogError> {
        if min_mag > max_mag {
            return Err(CatalogError::InvalidMagnitudeRange { min_val: min_mag, max_val: max_mag });
        }
        Ok(self.entries.iter().filter(|e| e.magnitude >= min_mag && e.magnitude <= max_mag).collect())
    }

    /// Cone search: stars within `radius_deg` of (ra, dec).
    pub fn cone_search(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        radius_deg: f64,
    ) -> Result<Vec<&StarEntry>, CatalogError> {
        if ra_deg < 0.0 || ra_deg >= 360.0 {
            return Err(CatalogError::InvalidRA(ra_deg));
        }
        if dec_deg < -90.0 || dec_deg > 90.0 {
            return Err(CatalogError::InvalidDec(dec_deg));
        }
        if radius_deg <= 0.0 {
            return Err(CatalogError::NonPositiveRadius(radius_deg));
        }
        Ok(self
            .entries
            .iter()
            .filter(|e| angular_distance(ra_deg, dec_deg, e.ra_deg, e.dec_deg) <= radius_deg)
            .collect())
    }

    /// Nearest N stars to a given position.
    pub fn nearest(&self, ra_deg: f64, dec_deg: f64, n: usize) -> Result<Vec<&StarEntry>, CatalogError> {
        if n < 1 {
            return Err(CatalogError::InvalidN(n));
        }
        let mut with_dist: Vec<(&StarEntry, f64)> = self
            .entries
            .iter()
            .map(|e| (e, angular_distance(ra_deg, dec_deg, e.ra_deg, e.dec_deg)))
            .collect();
        with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(with_dist.into_iter().take(n).map(|(e, _)| e).collect())
    }

    /// Stars of a given spectral class.
    pub fn by_spectral_class(&self, class: SpectralClass) -> Vec<&StarEntry> {
        self.entries.iter().filter(|e| e.spectral_type.class == class).collect()
    }

    /// Stars closer than a given distance in parsecs.
    pub fn within_distance(&self, max_pc: f64) -> Result<Vec<&StarEntry>, CatalogError> {
        if max_pc <= 0.0 {
            return Err(CatalogError::NonPositiveDistance(max_pc));
        }
        Ok(self.entries.iter().filter(|e| e.distance_pc <= max_pc).collect())
    }

    /// Get HR diagram data: (log10(temperature), absolute_magnitude).
    pub fn hr_diagram_data(&self) -> Vec<(f64, f64)> {
        self.entries
            .iter()
            .map(|e| (e.spectral_type.temperature().log10(), e.absolute_magnitude()))
            .collect()
    }

    /// Brightest N stars (lowest apparent magnitude).
    pub fn brightest(&self, n: usize) -> Vec<&StarEntry> {
        let mut sorted: Vec<&StarEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }
}

// ── Angular Distance ────────────────────────────────────────────

/// Angular distance between two sky positions (degrees) using Vincenty formula.
pub fn angular_distance(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let ra1 = ra1.to_radians();
    let dec1 = dec1.to_radians();
    let ra2 = ra2.to_radians();
    let dec2 = dec2.to_radians();
    let dra = ra2 - ra1;
    let num = ((dec2.cos() * dra.sin()).powi(2)
        + (dec1.cos() * dec2.sin() - dec1.sin() * dec2.cos() * dra.cos()).powi(2))
    .sqrt();
    let den = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * dra.cos();
    num.atan2(den).to_degrees()
}

// ── Helper: sample catalog ──────────────────────────────────────

/// Create a small sample catalog for testing.
pub fn sample_catalog() -> StarCatalog {
    let mut cat = StarCatalog::new();
    cat.add(StarEntry {
        name: "Sirius".to_string(),
        designation: "Alpha CMa".to_string(),
        ra_deg: 101.287,
        dec_deg: -16.716,
        magnitude: -1.46,
        spectral_type: SpectralType::parse("A1").unwrap(),
        distance_pc: 2.64,
        pm_ra: -546.05,
        pm_dec: -1223.14,
    });
    cat.add(StarEntry {
        name: "Canopus".to_string(),
        designation: "Alpha Car".to_string(),
        ra_deg: 95.988,
        dec_deg: -52.696,
        magnitude: -0.74,
        spectral_type: SpectralType::parse("F0").unwrap(),
        distance_pc: 95.0,
        pm_ra: 19.93,
        pm_dec: 23.24,
    });
    cat.add(StarEntry {
        name: "Arcturus".to_string(),
        designation: "Alpha Boo".to_string(),
        ra_deg: 213.915,
        dec_deg: 19.182,
        magnitude: -0.05,
        spectral_type: SpectralType::parse("K1").unwrap(),
        distance_pc: 11.26,
        pm_ra: -1093.45,
        pm_dec: -1999.4,
    });
    cat.add(StarEntry {
        name: "Vega".to_string(),
        designation: "Alpha Lyr".to_string(),
        ra_deg: 279.235,
        dec_deg: 38.784,
        magnitude: 0.03,
        spectral_type: SpectralType::parse("A0").unwrap(),
        distance_pc: 7.68,
        pm_ra: 200.94,
        pm_dec: 286.23,
    });
    cat.add(StarEntry {
        name: "Betelgeuse".to_string(),
        designation: "Alpha Ori".to_string(),
        ra_deg: 88.793,
        dec_deg: 7.407,
        magnitude: 0.5,
        spectral_type: SpectralType::parse("M1").unwrap(),
        distance_pc: 197.0,
        pm_ra: 27.33,
        pm_dec: 10.86,
    });
    cat
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn make_catalog() -> StarCatalog {
        sample_catalog()
    }

    #[test]
    fn spectral_type_parse_valid() {
        let st = SpectralType::parse("G2").unwrap();
        assert_eq!(st.class, SpectralClass::G);
        assert_eq!(st.subclass, 2);
    }

    #[test]
    fn spectral_type_parse_class_only() {
        let st = SpectralType::parse("O").unwrap();
        assert_eq!(st.class, SpectralClass::O);
        assert_eq!(st.subclass, 5); // default
    }

    #[test]
    fn spectral_type_invalid() {
        assert!(SpectralType::parse("X5").is_err());
        assert!(SpectralType::parse("").is_err());
    }

    #[test]
    fn spectral_class_temperature_order() {
        let temps: Vec<f64> = [
            SpectralClass::O,
            SpectralClass::B,
            SpectralClass::A,
            SpectralClass::F,
            SpectralClass::G,
            SpectralClass::K,
            SpectralClass::M,
        ]
        .iter()
        .map(|c| c.temperature())
        .collect();
        for w in temps.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn spectral_type_display() {
        let st = SpectralType::parse("K5").unwrap();
        assert_eq!(format!("{st}"), "K5");
    }

    #[test]
    fn catalog_add_and_len() {
        let cat = make_catalog();
        assert_eq!(cat.len(), 5);
        assert!(!cat.is_empty());
    }

    #[test]
    fn find_by_name() {
        let cat = make_catalog();
        let s = cat.find_by_name("Sirius").unwrap();
        assert!(approx_eq(s.magnitude, -1.46, 0.01));
    }

    #[test]
    fn find_by_name_missing() {
        let cat = make_catalog();
        assert!(cat.find_by_name("Nonexistent").is_none());
    }

    #[test]
    fn magnitude_range_query() {
        let cat = make_catalog();
        let stars = cat.by_magnitude_range(-2.0, 0.0).unwrap();
        // Sirius (-1.46), Canopus (-0.74), Arcturus (-0.05)
        assert_eq!(stars.len(), 3);
    }

    #[test]
    fn magnitude_range_inverted() {
        let cat = make_catalog();
        assert!(cat.by_magnitude_range(5.0, -5.0).is_err());
    }

    #[test]
    fn cone_search() {
        let cat = make_catalog();
        // Search around Sirius with 30 deg radius.
        let stars = cat.cone_search(101.287, -16.716, 30.0).unwrap();
        assert!(stars.len() >= 1); // At least Sirius itself.
    }

    #[test]
    fn cone_search_invalid_ra() {
        let cat = make_catalog();
        assert!(cat.cone_search(400.0, 0.0, 10.0).is_err());
    }

    #[test]
    fn cone_search_invalid_dec() {
        let cat = make_catalog();
        assert!(cat.cone_search(0.0, 100.0, 10.0).is_err());
    }

    #[test]
    fn nearest_stars() {
        let cat = make_catalog();
        let nearest = cat.nearest(100.0, -20.0, 2).unwrap();
        assert_eq!(nearest.len(), 2);
        // First should be Sirius (closest to query point).
        assert_eq!(nearest[0].name, "Sirius");
    }

    #[test]
    fn nearest_invalid_n() {
        let cat = make_catalog();
        assert!(cat.nearest(0.0, 0.0, 0).is_err());
    }

    #[test]
    fn by_spectral_class() {
        let cat = make_catalog();
        let a_stars = cat.by_spectral_class(SpectralClass::A);
        assert_eq!(a_stars.len(), 2); // Sirius A1, Vega A0
    }

    #[test]
    fn within_distance() {
        let cat = make_catalog();
        let nearby = cat.within_distance(10.0).unwrap();
        // Sirius (2.64), Vega (7.68) within 10 pc.
        assert_eq!(nearby.len(), 2);
    }

    #[test]
    fn brightest_stars() {
        let cat = make_catalog();
        let top = cat.brightest(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].name, "Sirius"); // brightest
    }

    #[test]
    fn absolute_magnitude() {
        let cat = make_catalog();
        let sirius = cat.find_by_name("Sirius").unwrap();
        let abs_mag = sirius.absolute_magnitude();
        // Sirius: m=−1.46, d=2.64 pc => M ≈ 1.42
        assert!(approx_eq(abs_mag, 1.42, 0.1));
    }

    #[test]
    fn luminosity_from_magnitude() {
        let cat = make_catalog();
        let sirius = cat.find_by_name("Sirius").unwrap();
        let lum = sirius.luminosity();
        // Sirius ~25 L_sun
        assert!(lum > 10.0 && lum < 50.0);
    }

    #[test]
    fn proper_motion() {
        let cat = make_catalog();
        let sirius = cat.find_by_name("Sirius").unwrap();
        let pm = sirius.total_proper_motion();
        assert!(pm > 1000.0); // Sirius has large proper motion.
    }

    #[test]
    fn tangential_velocity() {
        let cat = make_catalog();
        let arc = cat.find_by_name("Arcturus").unwrap();
        let vt = arc.tangential_velocity();
        // Arcturus ~120 km/s tangential velocity
        assert!(vt > 50.0 && vt < 200.0);
    }

    #[test]
    fn hr_diagram_data() {
        let cat = make_catalog();
        let data = cat.hr_diagram_data();
        assert_eq!(data.len(), 5);
        for (log_t, _mag) in &data {
            assert!(log_t.is_finite());
        }
    }

    #[test]
    fn angular_distance_same_point() {
        let d = angular_distance(100.0, 45.0, 100.0, 45.0);
        assert!(approx_eq(d, 0.0, 1e-10));
    }

    #[test]
    fn angular_distance_poles() {
        let d = angular_distance(0.0, 90.0, 0.0, -90.0);
        assert!(approx_eq(d, 180.0, 1e-6));
    }

    #[test]
    fn binary_system() {
        let primary = sample_catalog().entries[0].clone(); // Sirius A
        let mut sys = BinarySystem::new(primary);
        sys.add_companion(StarEntry {
            name: "Sirius B".to_string(),
            designation: "Alpha CMa B".to_string(),
            ra_deg: 101.287,
            dec_deg: -16.716,
            magnitude: 8.44,
            spectral_type: SpectralType::parse("A2").unwrap(),
            distance_pc: 2.64,
            pm_ra: -546.05,
            pm_dec: -1223.14,
        });
        assert_eq!(sys.component_count(), 2);
        let cm = sys.combined_magnitude();
        // Combined should be very close to primary (A is way brighter).
        assert!(approx_eq(cm, -1.46, 0.1));
    }
}
