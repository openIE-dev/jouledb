//! Geohash — encode lat/lon to geohash string, decode to bounding box,
//! neighbor computation (8 directions), precision control, geohash range
//! for bounding box coverage.
//!
//! Pure-Rust geohash encoding/decoding for spatial indexing, proximity
//! queries, and bounding-box coverage computation.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum GeohashError {
    InvalidPrecision(usize),
    InvalidGeohash(String),
    InvalidCoordinate { lat: f64, lon: f64 },
}

impl fmt::Display for GeohashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrecision(p) => write!(f, "invalid precision: {p} (must be 1..=12)"),
            Self::InvalidGeohash(s) => write!(f, "invalid geohash: {s}"),
            Self::InvalidCoordinate { lat, lon } => {
                write!(f, "invalid coordinate: lat={lat}, lon={lon}")
            }
        }
    }
}

impl std::error::Error for GeohashError {}

// ── Constants ───────────────────────────────────────────────────

const BASE32_CHARS: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";
const MAX_PRECISION: usize = 12;

// ── Direction ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
}

impl Direction {
    pub fn all() -> [Direction; 8] {
        [
            Direction::North, Direction::NorthEast, Direction::East,
            Direction::SouthEast, Direction::South, Direction::SouthWest,
            Direction::West, Direction::NorthWest,
        ]
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::North => write!(f, "N"),
            Self::South => write!(f, "S"),
            Self::East => write!(f, "E"),
            Self::West => write!(f, "W"),
            Self::NorthEast => write!(f, "NE"),
            Self::NorthWest => write!(f, "NW"),
            Self::SouthEast => write!(f, "SE"),
            Self::SouthWest => write!(f, "SW"),
        }
    }
}

// ── Bounding box ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct GeoBBox {
    pub lat_min: f64,
    pub lat_max: f64,
    pub lon_min: f64,
    pub lon_max: f64,
}

impl GeoBBox {
    pub fn new(lat_min: f64, lat_max: f64, lon_min: f64, lon_max: f64) -> Self {
        Self { lat_min, lat_max, lon_min, lon_max }
    }

    pub fn center_lat(&self) -> f64 { (self.lat_min + self.lat_max) * 0.5 }
    pub fn center_lon(&self) -> f64 { (self.lon_min + self.lon_max) * 0.5 }

    pub fn lat_span(&self) -> f64 { self.lat_max - self.lat_min }
    pub fn lon_span(&self) -> f64 { self.lon_max - self.lon_min }

    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        lat >= self.lat_min && lat <= self.lat_max && lon >= self.lon_min && lon <= self.lon_max
    }
}

impl fmt::Display for GeoBBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "GeoBBox(lat: {:.6}..{:.6}, lon: {:.6}..{:.6})",
            self.lat_min, self.lat_max, self.lon_min, self.lon_max
        )
    }
}

// ── Encode ──────────────────────────────────────────────────────

/// Encode latitude/longitude to a geohash string of given precision (1..=12).
pub fn encode(lat: f64, lon: f64, precision: usize) -> Result<String, GeohashError> {
    if precision == 0 || precision > MAX_PRECISION {
        return Err(GeohashError::InvalidPrecision(precision));
    }
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return Err(GeohashError::InvalidCoordinate { lat, lon });
    }

    let mut lat_range = (-90.0_f64, 90.0_f64);
    let mut lon_range = (-180.0_f64, 180.0_f64);
    let mut result = String::with_capacity(precision);
    let mut bit = 0u8;
    let mut ch_value = 0u8;
    let mut is_lon = true;

    let total_bits = precision * 5;
    for _ in 0..total_bits {
        if is_lon {
            let mid = (lon_range.0 + lon_range.1) * 0.5;
            if lon >= mid {
                ch_value = (ch_value << 1) | 1;
                lon_range.0 = mid;
            } else {
                ch_value <<= 1;
                lon_range.1 = mid;
            }
        } else {
            let mid = (lat_range.0 + lat_range.1) * 0.5;
            if lat >= mid {
                ch_value = (ch_value << 1) | 1;
                lat_range.0 = mid;
            } else {
                ch_value <<= 1;
                lat_range.1 = mid;
            }
        }
        is_lon = !is_lon;
        bit += 1;
        if bit == 5 {
            result.push(BASE32_CHARS[ch_value as usize] as char);
            bit = 0;
            ch_value = 0;
        }
    }
    Ok(result)
}

// ── Decode ──────────────────────────────────────────────────────

/// Decode a geohash string into a bounding box.
pub fn decode(geohash: &str) -> Result<GeoBBox, GeohashError> {
    if geohash.is_empty() || geohash.len() > MAX_PRECISION {
        return Err(GeohashError::InvalidGeohash(geohash.to_string()));
    }

    let mut lat_range = (-90.0_f64, 90.0_f64);
    let mut lon_range = (-180.0_f64, 180.0_f64);
    let mut is_lon = true;

    for ch in geohash.chars() {
        let idx = char_to_index(ch)?;
        for bit in (0..5).rev() {
            let b = (idx >> bit) & 1;
            if is_lon {
                let mid = (lon_range.0 + lon_range.1) * 0.5;
                if b == 1 { lon_range.0 = mid; } else { lon_range.1 = mid; }
            } else {
                let mid = (lat_range.0 + lat_range.1) * 0.5;
                if b == 1 { lat_range.0 = mid; } else { lat_range.1 = mid; }
            }
            is_lon = !is_lon;
        }
    }
    Ok(GeoBBox::new(lat_range.0, lat_range.1, lon_range.0, lon_range.1))
}

/// Decode a geohash to its center point.
pub fn decode_center(geohash: &str) -> Result<(f64, f64), GeohashError> {
    let bbox = decode(geohash)?;
    Ok((bbox.center_lat(), bbox.center_lon()))
}

fn char_to_index(c: char) -> Result<u8, GeohashError> {
    let b = c as u8;
    for (i, &ch) in BASE32_CHARS.iter().enumerate() {
        if ch == b {
            return Ok(i as u8);
        }
    }
    Err(GeohashError::InvalidGeohash(format!("invalid character: {c}")))
}

// ── Neighbors ───────────────────────────────────────────────────

/// Compute the neighbor geohash in the given direction.
pub fn neighbor(geohash: &str, direction: Direction) -> Result<String, GeohashError> {
    let bbox = decode(geohash)?;
    let precision = geohash.len();
    let lat_step = bbox.lat_span();
    let lon_step = bbox.lon_span();
    let center_lat = bbox.center_lat();
    let center_lon = bbox.center_lon();

    let (dlat, dlon) = match direction {
        Direction::North     => ( lat_step,  0.0),
        Direction::South     => (-lat_step,  0.0),
        Direction::East      => ( 0.0,       lon_step),
        Direction::West      => ( 0.0,      -lon_step),
        Direction::NorthEast => ( lat_step,  lon_step),
        Direction::NorthWest => ( lat_step, -lon_step),
        Direction::SouthEast => (-lat_step,  lon_step),
        Direction::SouthWest => (-lat_step, -lon_step),
    };

    let new_lat = (center_lat + dlat).clamp(-90.0, 90.0);
    let mut new_lon = center_lon + dlon;
    // Wrap longitude
    if new_lon > 180.0 { new_lon -= 360.0; }
    if new_lon < -180.0 { new_lon += 360.0; }

    encode(new_lat, new_lon, precision)
}

/// Compute all 8 neighbors of a geohash.
pub fn neighbors(geohash: &str) -> Result<Vec<(Direction, String)>, GeohashError> {
    let mut result = Vec::with_capacity(8);
    for dir in Direction::all() {
        let n = neighbor(geohash, dir)?;
        result.push((dir, n));
    }
    Ok(result)
}

// ── Bounding box coverage ───────────────────────────────────────

/// Compute the set of geohashes at given precision that cover a bounding box.
pub fn bbox_coverage(bbox: &GeoBBox, precision: usize) -> Result<Vec<String>, GeohashError> {
    if precision == 0 || precision > MAX_PRECISION {
        return Err(GeohashError::InvalidPrecision(precision));
    }

    // Determine step sizes from a sample geohash at the center
    let sample = encode(bbox.center_lat(), bbox.center_lon(), precision)?;
    let sample_bbox = decode(&sample)?;
    let lat_step = sample_bbox.lat_span();
    let lon_step = sample_bbox.lon_span();

    let mut results = Vec::new();
    let mut lat = bbox.lat_min;
    while lat <= bbox.lat_max {
        let mut lon = bbox.lon_min;
        while lon <= bbox.lon_max {
            let clamped_lat = lat.clamp(-90.0, 90.0);
            let clamped_lon = lon.clamp(-180.0, 180.0);
            let gh = encode(clamped_lat, clamped_lon, precision)?;
            if !results.contains(&gh) {
                results.push(gh);
            }
            lon += lon_step;
        }
        lat += lat_step;
    }
    Ok(results)
}

// ── Precision estimation ────────────────────────────────────────

/// Estimate the geohash precision needed for a given error tolerance in meters.
pub fn precision_for_error(meters: f64) -> usize {
    // Approximate error per precision level (at equator)
    let errors_m = [
        5_009_400.0, // 1
        1_252_300.0, // 2
          156_500.0, // 3
           39_100.0, // 4
            4_900.0, // 5
            1_200.0, // 6
              153.0, // 7
               38.0, // 8
                4.8, // 9
                1.2, // 10
                0.15,// 11
                0.04,// 12
    ];
    for (i, &e) in errors_m.iter().enumerate() {
        if e <= meters {
            return i + 1;
        }
    }
    MAX_PRECISION
}

// ── Display helper ──────────────────────────────────────────────

/// Format a geohash with its bounding box for debugging.
pub fn describe(geohash: &str) -> Result<String, GeohashError> {
    let bbox = decode(geohash)?;
    Ok(format!("{geohash} -> {bbox}"))
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let lat = 48.8566;
        let lon = 2.3522;
        let gh = encode(lat, lon, 8).unwrap();
        let bbox = decode(&gh).unwrap();
        assert!(bbox.contains(lat, lon));
    }

    #[test]
    fn test_encode_precision() {
        let gh5 = encode(40.7128, -74.0060, 5).unwrap();
        let gh8 = encode(40.7128, -74.0060, 8).unwrap();
        assert_eq!(gh5.len(), 5);
        assert_eq!(gh8.len(), 8);
        assert!(gh8.starts_with(&gh5));
    }

    #[test]
    fn test_encode_invalid_precision() {
        assert!(encode(0.0, 0.0, 0).is_err());
        assert!(encode(0.0, 0.0, 13).is_err());
    }

    #[test]
    fn test_encode_invalid_coordinate() {
        assert!(encode(91.0, 0.0, 5).is_err());
        assert!(encode(0.0, 181.0, 5).is_err());
    }

    #[test]
    fn test_decode_known() {
        // "s0000" should be near lat=0, lon=0
        let bbox = decode("s0000").unwrap();
        assert!(bbox.lat_min < 1.0 && bbox.lat_max > -1.0);
        assert!(bbox.lon_min < 1.0 && bbox.lon_max > -1.0);
    }

    #[test]
    fn test_decode_invalid() {
        assert!(decode("").is_err());
        assert!(decode("invalid!@#").is_err());
    }

    #[test]
    fn test_decode_center() {
        let (lat, lon) = decode_center("u4pruydqqvj").unwrap();
        assert!((lat - 57.64911).abs() < 0.001);
        assert!((lon - 10.40744).abs() < 0.001);
    }

    #[test]
    fn test_neighbor_north() {
        let gh = encode(48.8566, 2.3522, 5).unwrap();
        let n = neighbor(&gh, Direction::North).unwrap();
        let n_bbox = decode(&n).unwrap();
        let orig_bbox = decode(&gh).unwrap();
        assert!(n_bbox.center_lat() > orig_bbox.center_lat());
    }

    #[test]
    fn test_neighbor_south() {
        let gh = encode(48.8566, 2.3522, 5).unwrap();
        let n = neighbor(&gh, Direction::South).unwrap();
        let n_bbox = decode(&n).unwrap();
        let orig_bbox = decode(&gh).unwrap();
        assert!(n_bbox.center_lat() < orig_bbox.center_lat());
    }

    #[test]
    fn test_all_neighbors() {
        let gh = encode(48.8566, 2.3522, 5).unwrap();
        let nbrs = neighbors(&gh).unwrap();
        assert_eq!(nbrs.len(), 8);
        let unique: std::collections::HashSet<String> = nbrs.iter().map(|(_, s)| s.clone()).collect();
        assert_eq!(unique.len(), 8);
    }

    #[test]
    fn test_bbox_coverage() {
        let bbox = GeoBBox::new(48.8, 48.9, 2.3, 2.4);
        let hashes = bbox_coverage(&bbox, 5).unwrap();
        assert!(!hashes.is_empty());
    }

    #[test]
    fn test_bbox_coverage_invalid() {
        let bbox = GeoBBox::new(48.8, 48.9, 2.3, 2.4);
        assert!(bbox_coverage(&bbox, 0).is_err());
    }

    #[test]
    fn test_precision_for_error() {
        assert!(precision_for_error(10_000.0) <= 5);
        assert!(precision_for_error(1.0) >= 10);
    }

    #[test]
    fn test_geo_bbox_display() {
        let bb = GeoBBox::new(1.0, 2.0, 3.0, 4.0);
        let s = format!("{bb}");
        assert!(s.contains("GeoBBox"));
    }

    #[test]
    fn test_direction_display() {
        assert_eq!(format!("{}", Direction::North), "N");
        assert_eq!(format!("{}", Direction::SouthWest), "SW");
    }

    #[test]
    fn test_direction_all() {
        let dirs = Direction::all();
        assert_eq!(dirs.len(), 8);
    }

    #[test]
    fn test_describe() {
        let desc = describe("u4pruydqqvj").unwrap();
        assert!(desc.contains("u4pruydqqvj"));
        assert!(desc.contains("GeoBBox"));
    }

    #[test]
    fn test_encode_edge_cases() {
        // Encode at exact boundaries
        let gh = encode(90.0, 180.0, 5).unwrap();
        assert_eq!(gh.len(), 5);
        let gh = encode(-90.0, -180.0, 5).unwrap();
        assert_eq!(gh.len(), 5);
    }

    #[test]
    fn test_geo_bbox_contains() {
        let bb = GeoBBox::new(40.0, 50.0, -10.0, 10.0);
        assert!(bb.contains(45.0, 0.0));
        assert!(!bb.contains(55.0, 0.0));
    }
}
