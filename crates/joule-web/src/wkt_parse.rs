//! WKT (Well-Known Text) parser and serializer — POINT, LINESTRING, POLYGON,
//! MULTIPOINT, MULTILINESTRING, MULTIPOLYGON, GEOMETRYCOLLECTION with EMPTY
//! handling and 2D/3D/4D coordinate support.
//!
//! Pure-Rust implementation of OGC Simple Features WKT encoding. All coordinate
//! values use `f64`. Supports Z (3D) and ZM (4D) variants.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum WktError {
    ParseError(String),
    UnexpectedToken(String),
    UnsupportedType(String),
    EmptyInput,
    InvalidCoordinate(String),
}

impl fmt::Display for WktError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(s) => write!(f, "WKT parse error: {s}"),
            Self::UnexpectedToken(s) => write!(f, "unexpected token: {s}"),
            Self::UnsupportedType(s) => write!(f, "unsupported type: {s}"),
            Self::EmptyInput => write!(f, "empty input"),
            Self::InvalidCoordinate(s) => write!(f, "invalid coordinate: {s}"),
        }
    }
}

impl std::error::Error for WktError {}

// ── Coordinate Dimension ────────────────────────────────────────

/// Coordinate dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordDim {
    XY,
    XYZ,
    XYZM,
}

impl fmt::Display for CoordDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoordDim::XY => write!(f, "2D"),
            CoordDim::XYZ => write!(f, "3D"),
            CoordDim::XYZM => write!(f, "4D"),
        }
    }
}

// ── WKT Coordinate ──────────────────────────────────────────────

/// A coordinate with 2 to 4 ordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WktCoord {
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>,
    pub m: Option<f64>,
}

impl WktCoord {
    pub fn xy(x: f64, y: f64) -> Self {
        Self { x, y, z: None, m: None }
    }

    pub fn xyz(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z: Some(z), m: None }
    }

    pub fn xyzm(x: f64, y: f64, z: f64, m: f64) -> Self {
        Self { x, y, z: Some(z), m: Some(m) }
    }

    pub fn dimension(&self) -> CoordDim {
        match (self.z, self.m) {
            (Some(_), Some(_)) => CoordDim::XYZM,
            (Some(_), None) => CoordDim::XYZ,
            _ => CoordDim::XY,
        }
    }

    /// Euclidean distance to another coordinate (2D only).
    pub fn distance_2d(&self, other: &WktCoord) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl fmt::Display for WktCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.x, self.y)?;
        if let Some(z) = self.z {
            write!(f, " {z}")?;
        }
        if let Some(m) = self.m {
            write!(f, " {m}")?;
        }
        Ok(())
    }
}

// ── WKT Geometry ────────────────────────────────────────────────

/// WKT geometry types.
#[derive(Debug, Clone, PartialEq)]
pub enum WktGeometry {
    Point(Option<WktCoord>),
    LineString(Vec<WktCoord>),
    Polygon(Vec<Vec<WktCoord>>),
    MultiPoint(Vec<Option<WktCoord>>),
    MultiLineString(Vec<Vec<WktCoord>>),
    MultiPolygon(Vec<Vec<Vec<WktCoord>>>),
    GeometryCollection(Vec<WktGeometry>),
}

impl WktGeometry {
    /// Return the type keyword used in WKT.
    pub fn type_keyword(&self) -> &'static str {
        match self {
            WktGeometry::Point(_) => "POINT",
            WktGeometry::LineString(_) => "LINESTRING",
            WktGeometry::Polygon(_) => "POLYGON",
            WktGeometry::MultiPoint(_) => "MULTIPOINT",
            WktGeometry::MultiLineString(_) => "MULTILINESTRING",
            WktGeometry::MultiPolygon(_) => "MULTIPOLYGON",
            WktGeometry::GeometryCollection(_) => "GEOMETRYCOLLECTION",
        }
    }

    /// Check if this geometry is EMPTY.
    pub fn is_empty(&self) -> bool {
        match self {
            WktGeometry::Point(c) => c.is_none(),
            WktGeometry::LineString(cs) => cs.is_empty(),
            WktGeometry::Polygon(rings) => rings.is_empty(),
            WktGeometry::MultiPoint(pts) => pts.is_empty(),
            WktGeometry::MultiLineString(lines) => lines.is_empty(),
            WktGeometry::MultiPolygon(polys) => polys.is_empty(),
            WktGeometry::GeometryCollection(geoms) => geoms.is_empty(),
        }
    }

    /// Collect all coordinates from this geometry.
    pub fn all_coords(&self) -> Vec<WktCoord> {
        match self {
            WktGeometry::Point(Some(c)) => vec![*c],
            WktGeometry::Point(None) => vec![],
            WktGeometry::LineString(cs) => cs.clone(),
            WktGeometry::Polygon(rings) => {
                rings.iter().flat_map(|r| r.iter().copied()).collect()
            }
            WktGeometry::MultiPoint(pts) => {
                pts.iter().filter_map(|p| *p).collect()
            }
            WktGeometry::MultiLineString(lines) => {
                lines.iter().flat_map(|l| l.iter().copied()).collect()
            }
            WktGeometry::MultiPolygon(polys) => {
                polys.iter()
                    .flat_map(|p| p.iter().flat_map(|r| r.iter().copied()))
                    .collect()
            }
            WktGeometry::GeometryCollection(geoms) => {
                geoms.iter().flat_map(|g| g.all_coords()).collect()
            }
        }
    }

    /// Get the coordinate dimension from the first coordinate found.
    pub fn dimension(&self) -> Option<CoordDim> {
        self.all_coords().first().map(|c| c.dimension())
    }

    /// Serialize to WKT string.
    pub fn to_wkt(&self) -> String {
        if self.is_empty() {
            return format!("{} EMPTY", self.type_keyword());
        }
        match self {
            WktGeometry::Point(Some(c)) => format!("POINT ({})", c),
            WktGeometry::Point(None) => "POINT EMPTY".to_string(),
            WktGeometry::LineString(cs) => {
                let coords = Self::format_coord_list(cs);
                format!("LINESTRING ({coords})")
            }
            WktGeometry::Polygon(rings) => {
                let ring_strs: Vec<String> = rings.iter()
                    .map(|r| format!("({})", Self::format_coord_list(r)))
                    .collect();
                format!("POLYGON ({})", ring_strs.join(", "))
            }
            WktGeometry::MultiPoint(pts) => {
                let pt_strs: Vec<String> = pts.iter()
                    .map(|p| match p {
                        Some(c) => format!("({c})"),
                        None => "EMPTY".to_string(),
                    })
                    .collect();
                format!("MULTIPOINT ({})", pt_strs.join(", "))
            }
            WktGeometry::MultiLineString(lines) => {
                let line_strs: Vec<String> = lines.iter()
                    .map(|l| format!("({})", Self::format_coord_list(l)))
                    .collect();
                format!("MULTILINESTRING ({})", line_strs.join(", "))
            }
            WktGeometry::MultiPolygon(polys) => {
                let poly_strs: Vec<String> = polys.iter().map(|rings| {
                    let ring_strs: Vec<String> = rings.iter()
                        .map(|r| format!("({})", Self::format_coord_list(r)))
                        .collect();
                    format!("({})", ring_strs.join(", "))
                }).collect();
                format!("MULTIPOLYGON ({})", poly_strs.join(", "))
            }
            WktGeometry::GeometryCollection(geoms) => {
                let geom_strs: Vec<String> = geoms.iter().map(|g| g.to_wkt()).collect();
                format!("GEOMETRYCOLLECTION ({})", geom_strs.join(", "))
            }
        }
    }

    fn format_coord_list(coords: &[WktCoord]) -> String {
        coords.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", ")
    }
}

impl fmt::Display for WktGeometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_wkt())
    }
}

// ── WKT Parser ──────────────────────────────────────────────────

/// Stateful parser for WKT strings.
pub struct WktParser {
    input: Vec<char>,
    pos: usize,
}

impl WktParser {
    pub fn new(input: &str) -> Self {
        Self { input: input.chars().collect(), pos: 0 }
    }

    /// Parse WKT from a string.
    pub fn parse(input: &str) -> Result<WktGeometry, WktError> {
        if input.trim().is_empty() {
            return Err(WktError::EmptyInput);
        }
        let mut parser = WktParser::new(input);
        parser.parse_geometry()
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn expect_char(&mut self, ch: char) -> Result<(), WktError> {
        self.skip_whitespace();
        if self.peek() == Some(ch) {
            self.pos += 1;
            Ok(())
        } else {
            Err(WktError::UnexpectedToken(
                format!("expected '{}', got {:?} at pos {}", ch, self.peek(), self.pos)
            ))
        }
    }

    fn read_keyword(&mut self) -> String {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos].is_alphabetic() {
            self.pos += 1;
        }
        self.input[start..self.pos].iter().collect::<String>().to_uppercase()
    }

    fn read_number(&mut self) -> Result<f64, WktError> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s: String = self.input[start..self.pos].iter().collect();
        s.parse::<f64>().map_err(|e| WktError::InvalidCoordinate(format!("{e}: '{s}'")))
    }

    fn check_empty(&mut self) -> bool {
        self.skip_whitespace();
        let saved = self.pos;
        let kw = self.read_keyword();
        if kw == "EMPTY" {
            true
        } else {
            self.pos = saved;
            false
        }
    }

    fn parse_coord(&mut self) -> Result<WktCoord, WktError> {
        let x = self.read_number()?;
        let y = self.read_number()?;
        self.skip_whitespace();
        let next = self.peek();
        if next == Some(',') || next == Some(')') || next.is_none() {
            return Ok(WktCoord::xy(x, y));
        }
        let z = self.read_number()?;
        self.skip_whitespace();
        let next = self.peek();
        if next == Some(',') || next == Some(')') || next.is_none() {
            return Ok(WktCoord::xyz(x, y, z));
        }
        let m = self.read_number()?;
        Ok(WktCoord::xyzm(x, y, z, m))
    }

    fn parse_coord_list(&mut self) -> Result<Vec<WktCoord>, WktError> {
        self.expect_char('(')?;
        let mut coords = Vec::new();
        loop {
            coords.push(self.parse_coord()?);
            self.skip_whitespace();
            if self.peek() == Some(',') {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.expect_char(')')?;
        Ok(coords)
    }

    fn parse_ring_list(&mut self) -> Result<Vec<Vec<WktCoord>>, WktError> {
        self.expect_char('(')?;
        let mut rings = Vec::new();
        loop {
            rings.push(self.parse_coord_list()?);
            self.skip_whitespace();
            if self.peek() == Some(',') {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.expect_char(')')?;
        Ok(rings)
    }

    fn parse_geometry(&mut self) -> Result<WktGeometry, WktError> {
        let keyword = self.read_keyword();
        // Skip optional Z/M/ZM suffix
        self.skip_whitespace();
        let saved = self.pos;
        let suffix = self.read_keyword();
        if suffix != "Z" && suffix != "M" && suffix != "ZM" && suffix != "EMPTY" {
            self.pos = saved;
        } else if suffix == "EMPTY" {
            return Self::empty_for_type(&keyword);
        }

        if self.check_empty() {
            return Self::empty_for_type(&keyword);
        }

        match keyword.as_str() {
            "POINT" => {
                self.expect_char('(')?;
                let c = self.parse_coord()?;
                self.expect_char(')')?;
                Ok(WktGeometry::Point(Some(c)))
            }
            "LINESTRING" => {
                let coords = self.parse_coord_list()?;
                Ok(WktGeometry::LineString(coords))
            }
            "POLYGON" => {
                let rings = self.parse_ring_list()?;
                Ok(WktGeometry::Polygon(rings))
            }
            "MULTIPOINT" => {
                self.expect_char('(')?;
                let mut pts = Vec::new();
                loop {
                    self.skip_whitespace();
                    if self.peek() == Some('(') {
                        self.pos += 1;
                        let c = self.parse_coord()?;
                        self.expect_char(')')?;
                        pts.push(Some(c));
                    } else {
                        let c = self.parse_coord()?;
                        pts.push(Some(c));
                    }
                    self.skip_whitespace();
                    if self.peek() == Some(',') { self.pos += 1; } else { break; }
                }
                self.expect_char(')')?;
                Ok(WktGeometry::MultiPoint(pts))
            }
            "MULTILINESTRING" => {
                self.expect_char('(')?;
                let mut lines = Vec::new();
                loop {
                    lines.push(self.parse_coord_list()?);
                    self.skip_whitespace();
                    if self.peek() == Some(',') { self.pos += 1; } else { break; }
                }
                self.expect_char(')')?;
                Ok(WktGeometry::MultiLineString(lines))
            }
            "MULTIPOLYGON" => {
                self.expect_char('(')?;
                let mut polys = Vec::new();
                loop {
                    polys.push(self.parse_ring_list()?);
                    self.skip_whitespace();
                    if self.peek() == Some(',') { self.pos += 1; } else { break; }
                }
                self.expect_char(')')?;
                Ok(WktGeometry::MultiPolygon(polys))
            }
            "GEOMETRYCOLLECTION" => {
                self.expect_char('(')?;
                let mut geoms = Vec::new();
                loop {
                    geoms.push(self.parse_geometry()?);
                    self.skip_whitespace();
                    if self.peek() == Some(',') { self.pos += 1; } else { break; }
                }
                self.expect_char(')')?;
                Ok(WktGeometry::GeometryCollection(geoms))
            }
            other => Err(WktError::UnsupportedType(other.to_string())),
        }
    }

    fn empty_for_type(keyword: &str) -> Result<WktGeometry, WktError> {
        match keyword {
            "POINT" => Ok(WktGeometry::Point(None)),
            "LINESTRING" => Ok(WktGeometry::LineString(vec![])),
            "POLYGON" => Ok(WktGeometry::Polygon(vec![])),
            "MULTIPOINT" => Ok(WktGeometry::MultiPoint(vec![])),
            "MULTILINESTRING" => Ok(WktGeometry::MultiLineString(vec![])),
            "MULTIPOLYGON" => Ok(WktGeometry::MultiPolygon(vec![])),
            "GEOMETRYCOLLECTION" => Ok(WktGeometry::GeometryCollection(vec![])),
            other => Err(WktError::UnsupportedType(other.to_string())),
        }
    }
}

impl fmt::Display for WktParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WktParser(pos={}/{})", self.pos, self.input.len())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_point_2d() {
        let g = WktParser::parse("POINT (1.5 2.5)").unwrap();
        assert_eq!(g, WktGeometry::Point(Some(WktCoord::xy(1.5, 2.5))));
    }

    #[test]
    fn test_parse_point_3d() {
        let g = WktParser::parse("POINT Z (1 2 3)").unwrap();
        if let WktGeometry::Point(Some(c)) = g {
            assert_eq!(c.z, Some(3.0));
        } else {
            panic!("expected Point");
        }
    }

    #[test]
    fn test_parse_point_4d() {
        let g = WktParser::parse("POINT (1 2 3 4)").unwrap();
        if let WktGeometry::Point(Some(c)) = g {
            assert_eq!(c.m, Some(4.0));
        } else {
            panic!("expected Point");
        }
    }

    #[test]
    fn test_parse_point_empty() {
        let g = WktParser::parse("POINT EMPTY").unwrap();
        assert!(g.is_empty());
    }

    #[test]
    fn test_parse_linestring() {
        let g = WktParser::parse("LINESTRING (0 0, 1 1, 2 2)").unwrap();
        assert_eq!(g.all_coords().len(), 3);
    }

    #[test]
    fn test_parse_linestring_empty() {
        let g = WktParser::parse("LINESTRING EMPTY").unwrap();
        assert!(g.is_empty());
    }

    #[test]
    fn test_parse_polygon() {
        let g = WktParser::parse("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
        if let WktGeometry::Polygon(rings) = &g {
            assert_eq!(rings.len(), 1);
            assert_eq!(rings[0].len(), 5);
        } else {
            panic!("expected Polygon");
        }
    }

    #[test]
    fn test_parse_polygon_with_hole() {
        let g = WktParser::parse(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2))"
        ).unwrap();
        if let WktGeometry::Polygon(rings) = &g {
            assert_eq!(rings.len(), 2);
        } else {
            panic!("expected Polygon");
        }
    }

    #[test]
    fn test_parse_multipoint() {
        let g = WktParser::parse("MULTIPOINT ((0 0), (1 1))").unwrap();
        if let WktGeometry::MultiPoint(pts) = &g {
            assert_eq!(pts.len(), 2);
        } else {
            panic!("expected MultiPoint");
        }
    }

    #[test]
    fn test_parse_multilinestring() {
        let g = WktParser::parse("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))").unwrap();
        if let WktGeometry::MultiLineString(lines) = &g {
            assert_eq!(lines.len(), 2);
        } else {
            panic!("expected MultiLineString");
        }
    }

    #[test]
    fn test_parse_multipolygon() {
        let g = WktParser::parse(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))"
        ).unwrap();
        if let WktGeometry::MultiPolygon(polys) = &g {
            assert_eq!(polys.len(), 2);
        } else {
            panic!("expected MultiPolygon");
        }
    }

    #[test]
    fn test_parse_geometry_collection() {
        let g = WktParser::parse(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))"
        ).unwrap();
        if let WktGeometry::GeometryCollection(geoms) = &g {
            assert_eq!(geoms.len(), 2);
        } else {
            panic!("expected GeometryCollection");
        }
    }

    #[test]
    fn test_roundtrip_point() {
        let wkt = "POINT (1 2)";
        let g = WktParser::parse(wkt).unwrap();
        assert_eq!(g.to_wkt(), wkt);
    }

    #[test]
    fn test_roundtrip_linestring() {
        let wkt = "LINESTRING (0 0, 1 1, 2 2)";
        let g = WktParser::parse(wkt).unwrap();
        assert_eq!(g.to_wkt(), wkt);
    }

    #[test]
    fn test_empty_input_error() {
        assert!(matches!(WktParser::parse(""), Err(WktError::EmptyInput)));
    }

    #[test]
    fn test_unsupported_type() {
        assert!(matches!(
            WktParser::parse("CURVE (0 0, 1 1)"),
            Err(WktError::UnsupportedType(_))
        ));
    }

    #[test]
    fn test_coord_distance_2d() {
        let a = WktCoord::xy(0.0, 0.0);
        let b = WktCoord::xy(3.0, 4.0);
        assert!((a.distance_2d(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_detection() {
        let g = WktParser::parse("POINT (1 2 3)").unwrap();
        assert_eq!(g.dimension(), Some(CoordDim::XYZ));
    }

    #[test]
    fn test_type_keyword() {
        let g = WktGeometry::MultiPolygon(vec![]);
        assert_eq!(g.type_keyword(), "MULTIPOLYGON");
    }
}
