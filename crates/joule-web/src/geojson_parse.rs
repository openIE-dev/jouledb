//! GeoJSON parser and serializer — Feature, FeatureCollection, Geometry types
//! (Point, LineString, Polygon, Multi*), properties as key-value map, bounding
//! box computation, and coordinate precision control.
//!
//! Implements RFC 7946 GeoJSON format using only the standard library. All
//! coordinates are `f64` with configurable decimal precision for serialization.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum GeoJsonError {
    ParseError(String),
    InvalidGeometry(String),
    InvalidCoordinate(String),
    MissingField(String),
    UnknownType(String),
}

impl fmt::Display for GeoJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(s) => write!(f, "parse error: {s}"),
            Self::InvalidGeometry(s) => write!(f, "invalid geometry: {s}"),
            Self::InvalidCoordinate(s) => write!(f, "invalid coordinate: {s}"),
            Self::MissingField(s) => write!(f, "missing field: {s}"),
            Self::UnknownType(s) => write!(f, "unknown type: {s}"),
        }
    }
}

impl std::error::Error for GeoJsonError {}

// ── Coordinate ──────────────────────────────────────────────────

/// A position with longitude, latitude, and optional altitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coord {
    pub lon: f64,
    pub lat: f64,
    pub alt: Option<f64>,
}

impl Coord {
    pub fn new(lon: f64, lat: f64) -> Self {
        Self { lon, lat, alt: None }
    }

    pub fn with_altitude(lon: f64, lat: f64, alt: f64) -> Self {
        Self { lon, lat, alt: Some(alt) }
    }

    /// Validate that coordinates are within WGS84 bounds.
    pub fn validate(&self) -> Result<(), GeoJsonError> {
        if self.lon < -180.0 || self.lon > 180.0 {
            return Err(GeoJsonError::InvalidCoordinate(format!(
                "longitude {} out of range [-180, 180]", self.lon
            )));
        }
        if self.lat < -90.0 || self.lat > 90.0 {
            return Err(GeoJsonError::InvalidCoordinate(format!(
                "latitude {} out of range [-90, 90]", self.lat
            )));
        }
        Ok(())
    }

    /// Round coordinates to given decimal places.
    pub fn round_to(self, decimals: u32) -> Self {
        let factor = 10_f64.powi(decimals as i32);
        Self {
            lon: (self.lon * factor).round() / factor,
            lat: (self.lat * factor).round() / factor,
            alt: self.alt.map(|a| (a * factor).round() / factor),
        }
    }

    fn serialize(&self, precision: u32) -> String {
        match self.alt {
            Some(alt) => format!(
                "[{:.prec$}, {:.prec$}, {:.prec$}]",
                self.lon, self.lat, alt, prec = precision as usize
            ),
            None => format!(
                "[{:.prec$}, {:.prec$}]",
                self.lon, self.lat, prec = precision as usize
            ),
        }
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.alt {
            Some(alt) => write!(f, "[{}, {}, {}]", self.lon, self.lat, alt),
            None => write!(f, "[{}, {}]", self.lon, self.lat),
        }
    }
}

// ── Bounding Box ────────────────────────────────────────────────

/// Axis-aligned bounding box: [west, south, east, north].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl BBox {
    pub fn new(west: f64, south: f64, east: f64, north: f64) -> Self {
        Self { west, south, east, north }
    }

    pub fn from_coords(coords: &[Coord]) -> Option<Self> {
        if coords.is_empty() {
            return None;
        }
        let mut west = f64::MAX;
        let mut south = f64::MAX;
        let mut east = f64::MIN;
        let mut north = f64::MIN;
        for c in coords {
            if c.lon < west { west = c.lon; }
            if c.lat < south { south = c.lat; }
            if c.lon > east { east = c.lon; }
            if c.lat > north { north = c.lat; }
        }
        Some(Self { west, south, east, north })
    }

    pub fn contains(&self, c: &Coord) -> bool {
        c.lon >= self.west && c.lon <= self.east
            && c.lat >= self.south && c.lat <= self.north
    }

    pub fn merge(&self, other: &BBox) -> BBox {
        BBox {
            west: self.west.min(other.west),
            south: self.south.min(other.south),
            east: self.east.max(other.east),
            north: self.north.max(other.north),
        }
    }

    fn serialize(&self, precision: u32) -> String {
        format!(
            "[{:.prec$}, {:.prec$}, {:.prec$}, {:.prec$}]",
            self.west, self.south, self.east, self.north,
            prec = precision as usize
        )
    }
}

impl fmt::Display for BBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.west, self.south, self.east, self.north)
    }
}

// ── Geometry ────────────────────────────────────────────────────

/// GeoJSON geometry types per RFC 7946.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    Point(Coord),
    LineString(Vec<Coord>),
    Polygon(Vec<Vec<Coord>>),
    MultiPoint(Vec<Coord>),
    MultiLineString(Vec<Vec<Coord>>),
    MultiPolygon(Vec<Vec<Vec<Coord>>>),
    GeometryCollection(Vec<Geometry>),
}

impl Geometry {
    /// Collect all coordinates in this geometry.
    pub fn all_coords(&self) -> Vec<Coord> {
        match self {
            Geometry::Point(c) => vec![*c],
            Geometry::LineString(cs) => cs.clone(),
            Geometry::Polygon(rings) => rings.iter().flat_map(|r| r.iter().copied()).collect(),
            Geometry::MultiPoint(cs) => cs.clone(),
            Geometry::MultiLineString(lines) => {
                lines.iter().flat_map(|l| l.iter().copied()).collect()
            }
            Geometry::MultiPolygon(polys) => {
                polys.iter().flat_map(|p| p.iter().flat_map(|r| r.iter().copied())).collect()
            }
            Geometry::GeometryCollection(geoms) => {
                geoms.iter().flat_map(|g| g.all_coords()).collect()
            }
        }
    }

    /// Compute the bounding box for this geometry.
    pub fn bbox(&self) -> Option<BBox> {
        BBox::from_coords(&self.all_coords())
    }

    /// Return the geometry type name as used in GeoJSON.
    pub fn type_name(&self) -> &'static str {
        match self {
            Geometry::Point(_) => "Point",
            Geometry::LineString(_) => "LineString",
            Geometry::Polygon(_) => "Polygon",
            Geometry::MultiPoint(_) => "MultiPoint",
            Geometry::MultiLineString(_) => "MultiLineString",
            Geometry::MultiPolygon(_) => "MultiPolygon",
            Geometry::GeometryCollection(_) => "GeometryCollection",
        }
    }

    /// Validate all coordinates in this geometry.
    pub fn validate(&self) -> Result<(), GeoJsonError> {
        for c in &self.all_coords() {
            c.validate()?;
        }
        match self {
            Geometry::Polygon(rings) => {
                for ring in rings {
                    if ring.len() < 4 {
                        return Err(GeoJsonError::InvalidGeometry(
                            "polygon ring must have at least 4 positions".into()
                        ));
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn serialize_coords(&self, precision: u32) -> String {
        match self {
            Geometry::Point(c) => c.serialize(precision),
            Geometry::LineString(cs) | Geometry::MultiPoint(cs) => {
                let parts: Vec<String> = cs.iter().map(|c| c.serialize(precision)).collect();
                format!("[{}]", parts.join(", "))
            }
            Geometry::Polygon(rings) | Geometry::MultiLineString(rings) => {
                let ring_strs: Vec<String> = rings.iter().map(|ring| {
                    let parts: Vec<String> = ring.iter().map(|c| c.serialize(precision)).collect();
                    format!("[{}]", parts.join(", "))
                }).collect();
                format!("[{}]", ring_strs.join(", "))
            }
            Geometry::MultiPolygon(polys) => {
                let poly_strs: Vec<String> = polys.iter().map(|poly| {
                    let ring_strs: Vec<String> = poly.iter().map(|ring| {
                        let parts: Vec<String> = ring.iter().map(|c| c.serialize(precision)).collect();
                        format!("[{}]", parts.join(", "))
                    }).collect();
                    format!("[{}]", ring_strs.join(", "))
                }).collect();
                format!("[{}]", poly_strs.join(", "))
            }
            Geometry::GeometryCollection(_) => String::new(),
        }
    }

    /// Serialize this geometry to a GeoJSON string.
    pub fn to_geojson(&self, precision: u32) -> String {
        match self {
            Geometry::GeometryCollection(geoms) => {
                let parts: Vec<String> = geoms.iter().map(|g| g.to_geojson(precision)).collect();
                format!(
                    "{{\"type\": \"GeometryCollection\", \"geometries\": [{}]}}",
                    parts.join(", ")
                )
            }
            _ => {
                format!(
                    "{{\"type\": \"{}\", \"coordinates\": {}}}",
                    self.type_name(),
                    self.serialize_coords(precision)
                )
            }
        }
    }
}

impl fmt::Display for Geometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(coords={})", self.type_name(), self.all_coords().len())
    }
}

// ── Property Value ──────────────────────────────────────────────

/// A value in the properties map.
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    Array(Vec<PropValue>),
}

impl fmt::Display for PropValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropValue::Null => write!(f, "null"),
            PropValue::Bool(b) => write!(f, "{b}"),
            PropValue::Number(n) => write!(f, "{n}"),
            PropValue::Text(s) => write!(f, "\"{s}\""),
            PropValue::Array(arr) => {
                let parts: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                write!(f, "[{}]", parts.join(", "))
            }
        }
    }
}

// ── Feature ─────────────────────────────────────────────────────

/// A GeoJSON Feature: geometry + properties.
#[derive(Debug, Clone, PartialEq)]
pub struct Feature {
    pub id: Option<String>,
    pub geometry: Option<Geometry>,
    pub properties: HashMap<String, PropValue>,
    pub bbox: Option<BBox>,
}

impl Feature {
    pub fn new(geometry: Geometry) -> Self {
        Self {
            id: None,
            geometry: Some(geometry),
            properties: HashMap::new(),
            bbox: None,
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    pub fn with_property(mut self, key: &str, value: PropValue) -> Self {
        self.properties.insert(key.to_string(), value);
        self
    }

    pub fn with_bbox(mut self, bbox: BBox) -> Self {
        self.bbox = Some(bbox);
        self
    }

    /// Compute bounding box from geometry if not set.
    pub fn compute_bbox(&mut self) {
        if self.bbox.is_none() {
            if let Some(ref geom) = self.geometry {
                self.bbox = geom.bbox();
            }
        }
    }

    pub fn to_geojson(&self, precision: u32) -> String {
        let mut parts = vec![format!("\"type\": \"Feature\"")];
        if let Some(ref id) = self.id {
            parts.push(format!("\"id\": \"{id}\""));
        }
        if let Some(ref bbox) = self.bbox {
            parts.push(format!("\"bbox\": {}", bbox.serialize(precision)));
        }
        match &self.geometry {
            Some(geom) => parts.push(format!("\"geometry\": {}", geom.to_geojson(precision))),
            None => parts.push("\"geometry\": null".to_string()),
        }
        let prop_parts: Vec<String> = self.properties.iter()
            .map(|(k, v)| format!("\"{k}\": {v}"))
            .collect();
        parts.push(format!("\"properties\": {{{}}}", prop_parts.join(", ")));
        format!("{{{}}}", parts.join(", "))
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let geom_str = match &self.geometry {
            Some(g) => g.to_string(),
            None => "null".to_string(),
        };
        write!(f, "Feature(id={:?}, geom={}, props={})", self.id, geom_str, self.properties.len())
    }
}

// ── FeatureCollection ───────────────────────────────────────────

/// A GeoJSON FeatureCollection.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureCollection {
    pub features: Vec<Feature>,
    pub bbox: Option<BBox>,
}

impl FeatureCollection {
    pub fn new() -> Self {
        Self { features: Vec::new(), bbox: None }
    }

    pub fn with_feature(mut self, feature: Feature) -> Self {
        self.features.push(feature);
        self
    }

    pub fn with_bbox(mut self, bbox: BBox) -> Self {
        self.bbox = Some(bbox);
        self
    }

    pub fn add_feature(&mut self, feature: Feature) {
        self.features.push(feature);
    }

    /// Compute the bounding box from all features.
    pub fn compute_bbox(&mut self) {
        let all_coords: Vec<Coord> = self.features.iter()
            .filter_map(|f| f.geometry.as_ref())
            .flat_map(|g| g.all_coords())
            .collect();
        self.bbox = BBox::from_coords(&all_coords);
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn to_geojson(&self, precision: u32) -> String {
        let feat_strs: Vec<String> = self.features.iter()
            .map(|f| f.to_geojson(precision))
            .collect();
        let mut parts = vec![format!("\"type\": \"FeatureCollection\"")];
        if let Some(ref bbox) = self.bbox {
            parts.push(format!("\"bbox\": {}", bbox.serialize(precision)));
        }
        parts.push(format!("\"features\": [{}]", feat_strs.join(", ")));
        format!("{{{}}}", parts.join(", "))
    }
}

impl Default for FeatureCollection {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for FeatureCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FeatureCollection(features={})", self.features.len())
    }
}

// ── GeoJSON Config ──────────────────────────────────────────────

/// Configuration for GeoJSON serialization.
#[derive(Debug, Clone)]
pub struct GeoJsonConfig {
    pub precision: u32,
    pub compute_bbox: bool,
    pub validate_coords: bool,
}

impl GeoJsonConfig {
    pub fn new() -> Self {
        Self { precision: 6, compute_bbox: false, validate_coords: true }
    }

    pub fn with_precision(mut self, precision: u32) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_compute_bbox(mut self, compute: bool) -> Self {
        self.compute_bbox = compute;
        self
    }

    pub fn with_validate_coords(mut self, validate: bool) -> Self {
        self.validate_coords = validate;
        self
    }
}

impl Default for GeoJsonConfig {
    fn default() -> Self { Self::new() }
}

// ── Simple JSON tokenizer for GeoJSON parsing ───────────────────

/// Minimal GeoJSON parser from string.
pub struct GeoJsonParser {
    config: GeoJsonConfig,
}

impl GeoJsonParser {
    pub fn new(config: GeoJsonConfig) -> Self {
        Self { config }
    }

    /// Parse a coordinate array like `[lon, lat]` or `[lon, lat, alt]`.
    pub fn parse_coord(s: &str) -> Result<Coord, GeoJsonError> {
        let trimmed = s.trim();
        if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
            return Err(GeoJsonError::ParseError("expected coordinate array".into()));
        }
        let inner = &trimmed[1..trimmed.len() - 1];
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(GeoJsonError::ParseError(
                format!("expected 2 or 3 values, got {}", parts.len())
            ));
        }
        let lon = parts[0].trim().parse::<f64>()
            .map_err(|e| GeoJsonError::ParseError(format!("bad lon: {e}")))?;
        let lat = parts[1].trim().parse::<f64>()
            .map_err(|e| GeoJsonError::ParseError(format!("bad lat: {e}")))?;
        let alt = if parts.len() == 3 {
            Some(parts[2].trim().parse::<f64>()
                .map_err(|e| GeoJsonError::ParseError(format!("bad alt: {e}")))?)
        } else {
            None
        };
        Ok(Coord { lon, lat, alt })
    }

    /// Parse a linear ring from bracket-delimited coordinate string.
    pub fn parse_coord_list(s: &str) -> Result<Vec<Coord>, GeoJsonError> {
        let trimmed = s.trim();
        if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
            return Err(GeoJsonError::ParseError("expected array of coords".into()));
        }
        let inner = &trimmed[1..trimmed.len() - 1];
        let mut coords = Vec::new();
        let mut depth = 0i32;
        let mut start = 0;
        for (i, ch) in inner.char_indices() {
            match ch {
                '[' => {
                    if depth == 0 { start = i; }
                    depth += 1;
                }
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        coords.push(Self::parse_coord(&inner[start..=i])?);
                    }
                }
                _ => {}
            }
        }
        Ok(coords)
    }

    /// Build a Feature from geometry and optional properties.
    pub fn build_feature(&self, geometry: Geometry) -> Result<Feature, GeoJsonError> {
        if self.config.validate_coords {
            geometry.validate()?;
        }
        let mut feature = Feature::new(geometry);
        if self.config.compute_bbox {
            feature.compute_bbox();
        }
        Ok(feature)
    }
}

impl fmt::Display for GeoJsonParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GeoJsonParser(precision={})", self.config.precision)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_new() {
        let c = Coord::new(-73.985, 40.748);
        assert_eq!(c.lon, -73.985);
        assert_eq!(c.lat, 40.748);
        assert!(c.alt.is_none());
    }

    #[test]
    fn test_coord_with_altitude() {
        let c = Coord::with_altitude(-73.985, 40.748, 100.5);
        assert_eq!(c.alt, Some(100.5));
    }

    #[test]
    fn test_coord_validate_ok() {
        let c = Coord::new(0.0, 0.0);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_coord_validate_lon_out_of_range() {
        let c = Coord::new(200.0, 0.0);
        assert!(matches!(c.validate(), Err(GeoJsonError::InvalidCoordinate(_))));
    }

    #[test]
    fn test_coord_validate_lat_out_of_range() {
        let c = Coord::new(0.0, -95.0);
        assert!(matches!(c.validate(), Err(GeoJsonError::InvalidCoordinate(_))));
    }

    #[test]
    fn test_coord_round_to() {
        let c = Coord::new(-73.985678, 40.748123).round_to(3);
        assert_eq!(c.lon, -73.986);
        assert_eq!(c.lat, 40.748);
    }

    #[test]
    fn test_bbox_from_coords() {
        let coords = vec![
            Coord::new(-74.0, 40.7),
            Coord::new(-73.9, 40.8),
        ];
        let bbox = BBox::from_coords(&coords).unwrap();
        assert_eq!(bbox.west, -74.0);
        assert_eq!(bbox.north, 40.8);
    }

    #[test]
    fn test_bbox_contains() {
        let bbox = BBox::new(-74.0, 40.7, -73.9, 40.8);
        assert!(bbox.contains(&Coord::new(-73.95, 40.75)));
        assert!(!bbox.contains(&Coord::new(-75.0, 40.75)));
    }

    #[test]
    fn test_bbox_merge() {
        let a = BBox::new(0.0, 0.0, 1.0, 1.0);
        let b = BBox::new(-1.0, -1.0, 0.5, 0.5);
        let merged = a.merge(&b);
        assert_eq!(merged.west, -1.0);
        assert_eq!(merged.north, 1.0);
    }

    #[test]
    fn test_geometry_point_all_coords() {
        let g = Geometry::Point(Coord::new(1.0, 2.0));
        assert_eq!(g.all_coords().len(), 1);
    }

    #[test]
    fn test_geometry_linestring_bbox() {
        let g = Geometry::LineString(vec![
            Coord::new(0.0, 0.0),
            Coord::new(10.0, 10.0),
        ]);
        let bbox = g.bbox().unwrap();
        assert_eq!(bbox.west, 0.0);
        assert_eq!(bbox.east, 10.0);
    }

    #[test]
    fn test_geometry_polygon_validate_too_few_points() {
        let g = Geometry::Polygon(vec![vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(0.0, 0.0),
        ]]);
        assert!(matches!(g.validate(), Err(GeoJsonError::InvalidGeometry(_))));
    }

    #[test]
    fn test_geometry_type_name() {
        assert_eq!(Geometry::Point(Coord::new(0.0, 0.0)).type_name(), "Point");
        assert_eq!(Geometry::MultiPolygon(vec![]).type_name(), "MultiPolygon");
    }

    #[test]
    fn test_feature_new() {
        let f = Feature::new(Geometry::Point(Coord::new(1.0, 2.0)));
        assert!(f.geometry.is_some());
        assert!(f.properties.is_empty());
    }

    #[test]
    fn test_feature_with_property() {
        let f = Feature::new(Geometry::Point(Coord::new(1.0, 2.0)))
            .with_property("name", PropValue::Text("test".into()));
        assert_eq!(f.properties.get("name"), Some(&PropValue::Text("test".into())));
    }

    #[test]
    fn test_feature_with_id() {
        let f = Feature::new(Geometry::Point(Coord::new(0.0, 0.0)))
            .with_id("abc");
        assert_eq!(f.id, Some("abc".to_string()));
    }

    #[test]
    fn test_feature_compute_bbox() {
        let mut f = Feature::new(Geometry::LineString(vec![
            Coord::new(-10.0, -20.0),
            Coord::new(30.0, 40.0),
        ]));
        f.compute_bbox();
        assert!(f.bbox.is_some());
        assert_eq!(f.bbox.unwrap().west, -10.0);
    }

    #[test]
    fn test_feature_collection_add() {
        let mut fc = FeatureCollection::new();
        fc.add_feature(Feature::new(Geometry::Point(Coord::new(0.0, 0.0))));
        fc.add_feature(Feature::new(Geometry::Point(Coord::new(1.0, 1.0))));
        assert_eq!(fc.len(), 2);
    }

    #[test]
    fn test_feature_collection_compute_bbox() {
        let mut fc = FeatureCollection::new()
            .with_feature(Feature::new(Geometry::Point(Coord::new(-5.0, -5.0))))
            .with_feature(Feature::new(Geometry::Point(Coord::new(5.0, 5.0))));
        fc.compute_bbox();
        let bbox = fc.bbox.unwrap();
        assert_eq!(bbox.west, -5.0);
        assert_eq!(bbox.east, 5.0);
    }

    #[test]
    fn test_geojson_serialize_point() {
        let g = Geometry::Point(Coord::new(1.5, 2.5));
        let json = g.to_geojson(2);
        assert!(json.contains("\"Point\""));
        assert!(json.contains("1.50"));
    }

    #[test]
    fn test_geojson_serialize_geometry_collection() {
        let gc = Geometry::GeometryCollection(vec![
            Geometry::Point(Coord::new(1.0, 2.0)),
        ]);
        let json = gc.to_geojson(1);
        assert!(json.contains("GeometryCollection"));
        assert!(json.contains("geometries"));
    }

    #[test]
    fn test_parse_coord() {
        let c = GeoJsonParser::parse_coord("[1.5, 2.5]").unwrap();
        assert_eq!(c.lon, 1.5);
        assert_eq!(c.lat, 2.5);
    }

    #[test]
    fn test_parse_coord_with_alt() {
        let c = GeoJsonParser::parse_coord("[1.0, 2.0, 100.0]").unwrap();
        assert_eq!(c.alt, Some(100.0));
    }

    #[test]
    fn test_config_builder() {
        let cfg = GeoJsonConfig::new()
            .with_precision(3)
            .with_compute_bbox(true)
            .with_validate_coords(false);
        assert_eq!(cfg.precision, 3);
        assert!(cfg.compute_bbox);
        assert!(!cfg.validate_coords);
    }

    #[test]
    fn test_prop_value_display() {
        assert_eq!(PropValue::Null.to_string(), "null");
        assert_eq!(PropValue::Bool(true).to_string(), "true");
        assert_eq!(PropValue::Number(3.14).to_string(), "3.14");
        assert_eq!(PropValue::Text("hi".into()).to_string(), "\"hi\"");
    }

    #[test]
    fn test_multi_polygon_coords() {
        let mp = Geometry::MultiPolygon(vec![
            vec![vec![Coord::new(0.0, 0.0), Coord::new(1.0, 0.0),
                      Coord::new(1.0, 1.0), Coord::new(0.0, 0.0)]],
        ]);
        assert_eq!(mp.all_coords().len(), 4);
    }
}
