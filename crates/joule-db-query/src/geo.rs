//! Spatial/Geometry support for JouleDB SQL
//!
//! Provides PostGIS-compatible geometry types and spatial computation functions.
//! Geometries are stored as WKT (Well-Known Text) strings in the database.

use std::f64::consts::PI;

/// Earth's mean radius in meters (WGS-84)
const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Geometry types supported by the spatial engine
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    Point { x: f64, y: f64 },
    LineString(Vec<(f64, f64)>),
    Polygon(Vec<Vec<(f64, f64)>>), // outer ring + holes
}

// ==================== WKT PARSING ====================

/// Parse a WKT (Well-Known Text) string into a Geometry
pub fn parse_wkt(text: &str) -> Option<Geometry> {
    let text = text.trim();
    let upper = text.to_uppercase();

    if upper.starts_with("POINT") {
        parse_wkt_point(text)
    } else if upper.starts_with("LINESTRING") {
        parse_wkt_linestring(text)
    } else if upper.starts_with("POLYGON") {
        parse_wkt_polygon(text)
    } else {
        None
    }
}

fn parse_wkt_point(text: &str) -> Option<Geometry> {
    // POINT(x y) or POINT (x y)
    let inner = extract_parens(text)?;
    let coords = parse_coord_pair(inner.trim())?;
    Some(Geometry::Point {
        x: coords.0,
        y: coords.1,
    })
}

fn parse_wkt_linestring(text: &str) -> Option<Geometry> {
    // LINESTRING(x1 y1, x2 y2, ...)
    let inner = extract_parens(text)?;
    let points = parse_coord_list(inner)?;
    if points.len() < 2 {
        return None;
    }
    Some(Geometry::LineString(points))
}

fn parse_wkt_polygon(text: &str) -> Option<Geometry> {
    // POLYGON((x1 y1, x2 y2, ..., x1 y1), (hole...))
    let inner = extract_parens(text)?;
    let mut rings = Vec::new();

    // Split by ")," to get individual rings
    let mut depth = 0;
    let mut start = 0;
    let chars: Vec<char> = inner.chars().collect();

    for i in 0..chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let ring_str: String = chars[start..=i].iter().collect();
                    let ring_inner = extract_parens(&ring_str)?;
                    let points = parse_coord_list(ring_inner)?;
                    if points.len() < 3 {
                        return None;
                    }
                    rings.push(points);
                    // Skip comma and whitespace
                    start = i + 1;
                    while start < chars.len() && (chars[start] == ',' || chars[start] == ' ') {
                        start += 1;
                    }
                }
            }
            _ => {}
        }
    }

    if rings.is_empty() {
        return None;
    }
    Some(Geometry::Polygon(rings))
}

fn extract_parens(text: &str) -> Option<&str> {
    let open = text.find('(')?;
    let close = text.rfind(')')?;
    if close <= open {
        return None;
    }
    Some(&text[open + 1..close])
}

fn parse_coord_pair(text: &str) -> Option<(f64, f64)> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let x = parts[0].parse::<f64>().ok()?;
    let y = parts[1].parse::<f64>().ok()?;
    Some((x, y))
}

fn parse_coord_list(text: &str) -> Option<Vec<(f64, f64)>> {
    let pairs: Vec<&str> = text.split(',').collect();
    let mut coords = Vec::new();
    for pair in pairs {
        let c = parse_coord_pair(pair.trim())?;
        coords.push(c);
    }
    Some(coords)
}

// ==================== WKT OUTPUT ====================

/// Convert a Geometry to WKT string
pub fn to_wkt(geom: &Geometry) -> String {
    match geom {
        Geometry::Point { x, y } => format!("POINT({} {})", x, y),
        Geometry::LineString(points) => {
            let coords: Vec<String> = points.iter().map(|(x, y)| format!("{} {}", x, y)).collect();
            format!("LINESTRING({})", coords.join(", "))
        }
        Geometry::Polygon(rings) => {
            let ring_strs: Vec<String> = rings
                .iter()
                .map(|ring| {
                    let coords: Vec<String> =
                        ring.iter().map(|(x, y)| format!("{} {}", x, y)).collect();
                    format!("({})", coords.join(", "))
                })
                .collect();
            format!("POLYGON({})", ring_strs.join(", "))
        }
    }
}

// ==================== GeoJSON PARSING ====================

/// Parse a GeoJSON string into a Geometry
pub fn parse_geojson(json: &str) -> Option<Geometry> {
    let val: serde_json::Value = serde_json::from_str(json).ok()?;
    parse_geojson_value(&val)
}

fn parse_geojson_value(val: &serde_json::Value) -> Option<Geometry> {
    let obj = val.as_object()?;
    let geom_type = obj.get("type")?.as_str()?;

    match geom_type {
        "Point" => {
            let coords = obj.get("coordinates")?.as_array()?;
            if coords.len() < 2 {
                return None;
            }
            Some(Geometry::Point {
                x: coords[0].as_f64()?,
                y: coords[1].as_f64()?,
            })
        }
        "LineString" => {
            let coords = obj.get("coordinates")?.as_array()?;
            let points: Option<Vec<(f64, f64)>> = coords
                .iter()
                .map(|c| {
                    let arr = c.as_array()?;
                    Some((arr.first()?.as_f64()?, arr.get(1)?.as_f64()?))
                })
                .collect();
            Some(Geometry::LineString(points?))
        }
        "Polygon" => {
            let rings_arr = obj.get("coordinates")?.as_array()?;
            let rings: Option<Vec<Vec<(f64, f64)>>> = rings_arr
                .iter()
                .map(|ring| {
                    let arr = ring.as_array()?;
                    arr.iter()
                        .map(|c| {
                            let a = c.as_array()?;
                            Some((a.first()?.as_f64()?, a.get(1)?.as_f64()?))
                        })
                        .collect()
                })
                .collect();
            Some(Geometry::Polygon(rings?))
        }
        // For Feature, extract geometry
        "Feature" => {
            let geometry = obj.get("geometry")?;
            parse_geojson_value(geometry)
        }
        _ => None,
    }
}

/// Convert a Geometry to GeoJSON string
pub fn to_geojson(geom: &Geometry) -> String {
    match geom {
        Geometry::Point { x, y } => {
            format!(r#"{{"type":"Point","coordinates":[{},{}]}}"#, x, y)
        }
        Geometry::LineString(points) => {
            let coords: Vec<String> = points
                .iter()
                .map(|(x, y)| format!("[{},{}]", x, y))
                .collect();
            format!(
                r#"{{"type":"LineString","coordinates":[{}]}}"#,
                coords.join(",")
            )
        }
        Geometry::Polygon(rings) => {
            let ring_strs: Vec<String> = rings
                .iter()
                .map(|ring| {
                    let coords: Vec<String> =
                        ring.iter().map(|(x, y)| format!("[{},{}]", x, y)).collect();
                    format!("[{}]", coords.join(","))
                })
                .collect();
            format!(
                r#"{{"type":"Polygon","coordinates":[{}]}}"#,
                ring_strs.join(",")
            )
        }
    }
}

// ==================== SPATIAL COMPUTATIONS ====================

fn to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// Haversine distance between two points (x=lng, y=lat) in meters
pub fn haversine_distance(p1: &Geometry, p2: &Geometry) -> Option<f64> {
    let (lng1, lat1) = point_coords(p1)?;
    let (lng2, lat2) = point_coords(p2)?;

    let dlat = to_radians(lat2 - lat1);
    let dlng = to_radians(lng2 - lng1);
    let lat1r = to_radians(lat1);
    let lat2r = to_radians(lat2);

    let a = (dlat / 2.0).sin().powi(2) + lat1r.cos() * lat2r.cos() * (dlng / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    Some(EARTH_RADIUS_M * c)
}

/// Distance between two geometries. For non-point geometries, uses closest points.
pub fn geometry_distance(g1: &Geometry, g2: &Geometry) -> Option<f64> {
    match (g1, g2) {
        (Geometry::Point { .. }, Geometry::Point { .. }) => haversine_distance(g1, g2),
        _ => {
            // For complex geometries, compute distance between centroids
            let c1 = centroid(g1)?;
            let c2 = centroid(g2)?;
            haversine_distance(&c1, &c2)
        }
    }
}

/// Area of a polygon in square meters (Shoelace formula on projected coords)
pub fn polygon_area(geom: &Geometry) -> Option<f64> {
    match geom {
        Geometry::Polygon(rings) if !rings.is_empty() => {
            let outer_area = ring_area_m2(&rings[0]);
            let hole_area: f64 = rings[1..].iter().map(|r| ring_area_m2(r)).sum();
            Some((outer_area - hole_area).abs())
        }
        _ => Some(0.0),
    }
}

/// Approximate ring area in square meters using spherical excess
fn ring_area_m2(ring: &[(f64, f64)]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    // Use the surveyor's formula on lat/lng with Earth radius scaling
    let n = ring.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let (lng1, lat1) = ring[i];
        let (lng2, lat2) = ring[j];
        area += to_radians(lng2 - lng1) * (2.0 + to_radians(lat1).sin() + to_radians(lat2).sin());
    }
    (area * EARTH_RADIUS_M * EARTH_RADIUS_M / 2.0).abs()
}

/// Length of a linestring in meters
pub fn linestring_length(geom: &Geometry) -> Option<f64> {
    match geom {
        Geometry::LineString(points) => {
            let mut total = 0.0;
            for i in 0..points.len().saturating_sub(1) {
                let p1 = Geometry::Point {
                    x: points[i].0,
                    y: points[i].1,
                };
                let p2 = Geometry::Point {
                    x: points[i + 1].0,
                    y: points[i + 1].1,
                };
                total += haversine_distance(&p1, &p2).unwrap_or(0.0);
            }
            Some(total)
        }
        _ => Some(0.0),
    }
}

/// Test if a point is inside a polygon (ray casting algorithm)
pub fn point_in_polygon(point: &Geometry, polygon: &Geometry) -> bool {
    let (px, py) = match point_coords(point) {
        Some(c) => c,
        None => return false,
    };
    match polygon {
        Geometry::Polygon(rings) if !rings.is_empty() => {
            // Must be inside outer ring
            if !point_in_ring(px, py, &rings[0]) {
                return false;
            }
            // Must not be inside any hole
            for hole in &rings[1..] {
                if point_in_ring(px, py, hole) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

fn point_in_ring(px: f64, py: f64, ring: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let n = ring.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (xi, yi) = ring[i];
        let (xj, yj) = ring[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Test if g1 contains g2
pub fn contains(g1: &Geometry, g2: &Geometry) -> bool {
    match (g1, g2) {
        (poly @ Geometry::Polygon(_), point @ Geometry::Point { .. }) => {
            point_in_polygon(point, poly)
        }
        (poly @ Geometry::Polygon(_), Geometry::LineString(points)) => points
            .iter()
            .all(|(x, y)| point_in_polygon(&Geometry::Point { x: *x, y: *y }, poly)),
        (Geometry::Polygon(_), Geometry::Polygon(inner_rings)) => {
            if inner_rings.is_empty() {
                return false;
            }
            // All vertices of inner polygon must be inside outer polygon
            inner_rings[0]
                .iter()
                .all(|(x, y)| point_in_polygon(&Geometry::Point { x: *x, y: *y }, g1))
        }
        _ => false,
    }
}

/// Test if two geometries intersect
pub fn intersects(g1: &Geometry, g2: &Geometry) -> bool {
    match (g1, g2) {
        (Geometry::Point { .. }, Geometry::Point { .. }) => {
            // Points intersect if they're the same
            let (x1, y1) = point_coords(g1).unwrap();
            let (x2, y2) = point_coords(g2).unwrap();
            (x1 - x2).abs() < 1e-10 && (y1 - y2).abs() < 1e-10
        }
        (point @ Geometry::Point { .. }, poly @ Geometry::Polygon(_))
        | (poly @ Geometry::Polygon(_), point @ Geometry::Point { .. }) => {
            point_in_polygon(point, poly)
        }
        (Geometry::Polygon(_), Geometry::Polygon(rings2)) => {
            // If any vertex of g2 is inside g1, or any vertex of g1 is inside g2
            if rings2.is_empty() {
                return false;
            }
            if rings2[0]
                .iter()
                .any(|(x, y)| point_in_polygon(&Geometry::Point { x: *x, y: *y }, g1))
            {
                return true;
            }
            if let Geometry::Polygon(rings1) = g1 {
                if !rings1.is_empty() {
                    return rings1[0]
                        .iter()
                        .any(|(x, y)| point_in_polygon(&Geometry::Point { x: *x, y: *y }, g2));
                }
            }
            false
        }
        (point @ Geometry::Point { .. }, Geometry::LineString(points))
        | (Geometry::LineString(points), point @ Geometry::Point { .. }) => {
            let (px, py) = point_coords(point).unwrap();
            // Point is on linestring if it's very close to any segment
            for i in 0..points.len().saturating_sub(1) {
                if point_on_segment(px, py, points[i], points[i + 1]) {
                    return true;
                }
            }
            false
        }
        _ => {
            // Fallback: bounding box intersection
            let (min1, max1) = bounding_box(g1);
            let (min2, max2) = bounding_box(g2);
            min1.0 <= max2.0 && max1.0 >= min2.0 && min1.1 <= max2.1 && max1.1 >= min2.1
        }
    }
}

fn point_on_segment(px: f64, py: f64, a: (f64, f64), b: (f64, f64)) -> bool {
    let cross = (py - a.1) * (b.0 - a.0) - (px - a.0) * (b.1 - a.1);
    if cross.abs() > 1e-10 {
        return false;
    }
    px >= a.0.min(b.0) && px <= a.0.max(b.0) && py >= a.1.min(b.1) && py <= a.1.max(b.1)
}

/// Compute centroid of a geometry
pub fn centroid(geom: &Geometry) -> Option<Geometry> {
    match geom {
        Geometry::Point { .. } => Some(geom.clone()),
        Geometry::LineString(points) if !points.is_empty() => {
            let n = points.len() as f64;
            let sx: f64 = points.iter().map(|(x, _)| x).sum();
            let sy: f64 = points.iter().map(|(_, y)| y).sum();
            Some(Geometry::Point {
                x: sx / n,
                y: sy / n,
            })
        }
        Geometry::Polygon(rings) if !rings.is_empty() && !rings[0].is_empty() => {
            // Centroid of outer ring (exclude closing point if it duplicates first)
            let ring = &rings[0];
            let pts = if ring.len() > 1 && ring.first() == ring.last() {
                &ring[..ring.len() - 1]
            } else {
                ring.as_slice()
            };
            let n = pts.len() as f64;
            let sx: f64 = pts.iter().map(|(x, _)| x).sum();
            let sy: f64 = pts.iter().map(|(_, y)| y).sum();
            Some(Geometry::Point {
                x: sx / n,
                y: sy / n,
            })
        }
        _ => None,
    }
}

/// Compute bounding box (envelope) as a polygon
pub fn envelope(geom: &Geometry) -> Option<Geometry> {
    let (min, max) = bounding_box(geom);
    Some(Geometry::Polygon(vec![vec![
        (min.0, min.1),
        (max.0, min.1),
        (max.0, max.1),
        (min.0, max.1),
        (min.0, min.1),
    ]]))
}

fn bounding_box(geom: &Geometry) -> ((f64, f64), (f64, f64)) {
    let points = all_coords(geom);
    if points.is_empty() {
        return ((0.0, 0.0), (0.0, 0.0));
    }
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for (x, y) in &points {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }
    ((min_x, min_y), (max_x, max_y))
}

fn all_coords(geom: &Geometry) -> Vec<(f64, f64)> {
    match geom {
        Geometry::Point { x, y } => vec![(*x, *y)],
        Geometry::LineString(pts) => pts.clone(),
        Geometry::Polygon(rings) => rings.iter().flat_map(|r| r.iter().copied()).collect(),
    }
}

/// Approximate buffer around a point (creates a polygon)
pub fn buffer(geom: &Geometry, distance_m: f64) -> Option<Geometry> {
    let (cx, cy) = point_coords(&centroid(geom)?)?;
    // Approximate: convert meters to degrees
    let lat_deg = distance_m / 111_320.0;
    let lng_deg = distance_m / (111_320.0 * to_radians(cy).cos());

    // Create an approximate circle with 32 segments
    let n = 32;
    let mut ring = Vec::with_capacity(n + 1);
    for i in 0..n {
        let angle = 2.0 * PI * (i as f64) / (n as f64);
        ring.push((cx + lng_deg * angle.cos(), cy + lat_deg * angle.sin()));
    }
    ring.push(ring[0]); // close ring
    Some(Geometry::Polygon(vec![ring]))
}

/// Check if two geometries are within a given distance (meters)
pub fn dwithin(g1: &Geometry, g2: &Geometry, distance_m: f64) -> bool {
    geometry_distance(g1, g2)
        .map(|d| d <= distance_m)
        .unwrap_or(false)
}

// ==================== HELPERS ====================

/// Extract (x, y) coordinates from a Point geometry
pub fn point_coords(geom: &Geometry) -> Option<(f64, f64)> {
    match geom {
        Geometry::Point { x, y } => Some((*x, *y)),
        _ => None,
    }
}

/// Try to parse a value as a geometry (WKT string, GeoJSON, or POINT coords)
pub fn value_to_geometry(val: &str) -> Option<Geometry> {
    let trimmed = val.trim();
    // Try WKT first
    if let Some(g) = parse_wkt(trimmed) {
        return Some(g);
    }
    // Try GeoJSON
    if trimmed.starts_with('{') {
        if let Some(g) = parse_geojson(trimmed) {
            return Some(g);
        }
    }
    None
}

// ==================== INTERVAL PARSING ====================

/// Parse an interval string like '5 minutes', '1 hour', '1 day', '30 seconds' to seconds as i64
pub fn parse_interval_to_seconds(s: &str) -> Option<i64> {
    let s = s.trim().to_lowercase();
    // Try simple number (seconds)
    if let Ok(n) = s.parse::<i64>() {
        return Some(n);
    }
    // Split into parts and parse pairs
    let parts: Vec<&str> = s.split_whitespace().collect();
    let mut total: i64 = 0;
    let mut i = 0;
    while i < parts.len() {
        // Try to parse number + unit pair
        if let Ok(n) = parts[i].parse::<i64>() {
            let unit = if i + 1 < parts.len() {
                i += 1;
                parts[i]
            } else {
                "seconds"
            };
            let multiplier = match unit.trim_end_matches('s') {
                "second" | "sec" => 1,
                "minute" | "min" => 60,
                "hour" | "hr" => 3600,
                "day" => 86400,
                "week" => 604800,
                "month" | "mon" => 2592000, // 30 days
                "year" | "yr" => 31536000,  // 365 days
                _ => return None,
            };
            total += n * multiplier;
        } else {
            return None;
        }
        i += 1;
    }
    if total > 0 { Some(total) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_wkt_point() {
        let g = parse_wkt("POINT(10.5 20.3)").unwrap();
        assert_eq!(g, Geometry::Point { x: 10.5, y: 20.3 });
    }

    #[test]
    fn test_parse_wkt_point_spaces() {
        let g = parse_wkt("POINT (  10.5  20.3  )").unwrap();
        assert_eq!(g, Geometry::Point { x: 10.5, y: 20.3 });
    }

    #[test]
    fn test_parse_wkt_linestring() {
        let g = parse_wkt("LINESTRING(0 0, 1 1, 2 0)").unwrap();
        assert_eq!(
            g,
            Geometry::LineString(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
        );
    }

    #[test]
    fn test_parse_wkt_polygon() {
        let g = parse_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
        match g {
            Geometry::Polygon(rings) => {
                assert_eq!(rings.len(), 1);
                assert_eq!(rings[0].len(), 5);
            }
            _ => panic!("Expected polygon"),
        }
    }

    #[test]
    fn test_wkt_roundtrip() {
        let point = Geometry::Point { x: 1.5, y: 2.5 };
        let wkt = to_wkt(&point);
        let parsed = parse_wkt(&wkt).unwrap();
        assert_eq!(point, parsed);
    }

    #[test]
    fn test_parse_geojson_point() {
        let g = parse_geojson(r#"{"type":"Point","coordinates":[10.5,20.3]}"#).unwrap();
        assert_eq!(g, Geometry::Point { x: 10.5, y: 20.3 });
    }

    #[test]
    fn test_geojson_feature() {
        let g =
            parse_geojson(r#"{"type":"Feature","geometry":{"type":"Point","coordinates":[1,2]}}"#)
                .unwrap();
        assert_eq!(g, Geometry::Point { x: 1.0, y: 2.0 });
    }

    #[test]
    fn test_geojson_roundtrip() {
        let point = Geometry::Point { x: 1.5, y: 2.5 };
        let json = to_geojson(&point);
        let parsed = parse_geojson(&json).unwrap();
        assert_eq!(point, parsed);
    }

    #[test]
    fn test_haversine_same_point() {
        let p = Geometry::Point { x: 0.0, y: 0.0 };
        let d = haversine_distance(&p, &p).unwrap();
        assert!(d < 0.001);
    }

    #[test]
    fn test_haversine_known_distance() {
        // NYC to London: ~5570 km
        let nyc = Geometry::Point {
            x: -74.006,
            y: 40.7128,
        };
        let london = Geometry::Point {
            x: -0.1278,
            y: 51.5074,
        };
        let d = haversine_distance(&nyc, &london).unwrap();
        assert!((d - 5_570_000.0).abs() < 50_000.0); // within 50km tolerance
    }

    #[test]
    fn test_point_in_polygon() {
        let poly = Geometry::Polygon(vec![vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]]);
        let inside = Geometry::Point { x: 5.0, y: 5.0 };
        let outside = Geometry::Point { x: 15.0, y: 5.0 };

        assert!(super::point_in_polygon(&inside, &poly));
        assert!(!super::point_in_polygon(&outside, &poly));
    }

    #[test]
    fn test_contains() {
        let poly = Geometry::Polygon(vec![vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]]);
        let point = Geometry::Point { x: 5.0, y: 5.0 };
        assert!(contains(&poly, &point));
        assert!(!contains(&point, &poly));
    }

    #[test]
    fn test_intersects_point_polygon() {
        let poly = Geometry::Polygon(vec![vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]]);
        let inside = Geometry::Point { x: 5.0, y: 5.0 };
        let outside = Geometry::Point { x: 15.0, y: 5.0 };
        assert!(intersects(&inside, &poly));
        assert!(!intersects(&outside, &poly));
    }

    #[test]
    fn test_centroid_polygon() {
        let poly = Geometry::Polygon(vec![vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]]);
        let c = centroid(&poly).unwrap();
        let (x, y) = point_coords(&c).unwrap();
        assert!((x - 5.0).abs() < 0.01);
        assert!((y - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_envelope() {
        let line = Geometry::LineString(vec![(1.0, 2.0), (5.0, 8.0), (3.0, 4.0)]);
        let env = envelope(&line).unwrap();
        let wkt = to_wkt(&env);
        assert!(wkt.contains("POLYGON"));
    }

    #[test]
    fn test_buffer() {
        let p = Geometry::Point { x: 0.0, y: 45.0 };
        let b = buffer(&p, 1000.0).unwrap();
        match b {
            Geometry::Polygon(rings) => {
                assert_eq!(rings.len(), 1);
                assert_eq!(rings[0].len(), 33); // 32 segments + closing
            }
            _ => panic!("Expected polygon"),
        }
    }

    #[test]
    fn test_dwithin() {
        let p1 = Geometry::Point { x: 0.0, y: 0.0 };
        let p2 = Geometry::Point { x: 0.001, y: 0.0 };
        assert!(dwithin(&p1, &p2, 200.0)); // ~111m apart
        assert!(!dwithin(&p1, &p2, 50.0));
    }

    #[test]
    fn test_polygon_area() {
        // A 1-degree x 1-degree square at equator ≈ 12,321 km²
        let poly = Geometry::Polygon(vec![vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]]);
        let area = polygon_area(&poly).unwrap();
        // Should be roughly 12,000 km² = 1.2e10 m²
        assert!(area > 1e10 && area < 1.5e10);
    }

    #[test]
    fn test_linestring_length() {
        // 1 degree of longitude at equator ≈ 111km
        let line = Geometry::LineString(vec![(0.0, 0.0), (1.0, 0.0)]);
        let len = linestring_length(&line).unwrap();
        assert!((len - 111_195.0).abs() < 1000.0);
    }

    #[test]
    fn test_value_to_geometry_wkt() {
        let g = value_to_geometry("POINT(1 2)").unwrap();
        assert_eq!(g, Geometry::Point { x: 1.0, y: 2.0 });
    }

    #[test]
    fn test_value_to_geometry_geojson() {
        let g = value_to_geometry(r#"{"type":"Point","coordinates":[1,2]}"#).unwrap();
        assert_eq!(g, Geometry::Point { x: 1.0, y: 2.0 });
    }

    // ==================== INTERVAL PARSING TESTS ====================

    #[test]
    fn test_parse_interval_seconds() {
        assert_eq!(parse_interval_to_seconds("30 seconds"), Some(30));
        assert_eq!(parse_interval_to_seconds("1 second"), Some(1));
    }

    #[test]
    fn test_parse_interval_minutes() {
        assert_eq!(parse_interval_to_seconds("5 minutes"), Some(300));
        assert_eq!(parse_interval_to_seconds("1 minute"), Some(60));
    }

    #[test]
    fn test_parse_interval_hours() {
        assert_eq!(parse_interval_to_seconds("1 hour"), Some(3600));
        assert_eq!(parse_interval_to_seconds("2 hours"), Some(7200));
    }

    #[test]
    fn test_parse_interval_days() {
        assert_eq!(parse_interval_to_seconds("1 day"), Some(86400));
        assert_eq!(parse_interval_to_seconds("7 days"), Some(604800));
    }

    #[test]
    fn test_parse_interval_weeks() {
        assert_eq!(parse_interval_to_seconds("1 week"), Some(604800));
    }

    #[test]
    fn test_parse_interval_plain_number() {
        assert_eq!(parse_interval_to_seconds("60"), Some(60));
        assert_eq!(parse_interval_to_seconds("3600"), Some(3600));
    }

    #[test]
    fn test_parse_interval_invalid() {
        assert_eq!(parse_interval_to_seconds("foo"), None);
        assert_eq!(parse_interval_to_seconds(""), None);
    }
}
