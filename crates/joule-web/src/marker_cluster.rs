//! Marker clustering: grid-based clustering, hierarchical supercluster,
//! expand/spiderfy for map markers.

use std::collections::HashMap;

// ── Point ───────────────────────────────────────────────────────

/// A geographic point with associated data.
#[derive(Debug, Clone)]
pub struct MarkerPoint {
    pub lat: f64,
    pub lng: f64,
    pub data: Option<String>,
    pub id: u64,
}

impl MarkerPoint {
    pub fn new(id: u64, lat: f64, lng: f64) -> Self {
        Self { lat, lng, data: None, id }
    }

    pub fn with_data(mut self, data: String) -> Self {
        self.data = Some(data);
        self
    }
}

// ── Cluster ─────────────────────────────────────────────────────

/// A cluster of markers.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Weighted center latitude.
    pub center_lat: f64,
    /// Weighted center longitude.
    pub center_lng: f64,
    /// Number of points in the cluster.
    pub point_count: usize,
    /// Bounding box: (min_lat, min_lng, max_lat, max_lng).
    pub bounds: (f64, f64, f64, f64),
    /// IDs of points in this cluster.
    pub point_ids: Vec<u64>,
    /// Unique cluster id (for hierarchical lookups).
    pub cluster_id: u64,
}

// ── Grid-based Clustering ───────────────────────────────────────

/// Grid cell key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GridCell {
    col: i64,
    row: i64,
}

/// Grid-based clustering at a given zoom level.
pub fn grid_cluster(points: &[MarkerPoint], zoom: u32, grid_size: f64) -> Vec<Cluster> {
    let scale = 2.0_f64.powi(zoom as i32);
    let cell_size = grid_size / scale;

    let mut cells: HashMap<GridCell, Vec<usize>> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        let col = (p.lng / cell_size).floor() as i64;
        let row = (p.lat / cell_size).floor() as i64;
        cells.entry(GridCell { col, row }).or_default().push(i);
    }

    let mut clusters = Vec::new();
    let mut cluster_id = 0u64;
    for indices in cells.values() {
        if indices.is_empty() {
            continue;
        }
        let mut sum_lat = 0.0;
        let mut sum_lng = 0.0;
        let mut min_lat = f64::MAX;
        let mut min_lng = f64::MAX;
        let mut max_lat = f64::MIN;
        let mut max_lng = f64::MIN;
        let mut ids = Vec::new();

        for &i in indices {
            let p = &points[i];
            sum_lat += p.lat;
            sum_lng += p.lng;
            min_lat = min_lat.min(p.lat);
            min_lng = min_lng.min(p.lng);
            max_lat = max_lat.max(p.lat);
            max_lng = max_lng.max(p.lng);
            ids.push(p.id);
        }

        let count = indices.len();
        clusters.push(Cluster {
            center_lat: sum_lat / count as f64,
            center_lng: sum_lng / count as f64,
            point_count: count,
            bounds: (min_lat, min_lng, max_lat, max_lng),
            point_ids: ids,
            cluster_id,
        });
        cluster_id += 1;
    }
    clusters
}

// ── Supercluster (hierarchical) ─────────────────────────────────

/// Pre-computed hierarchical clustering at all zoom levels.
#[derive(Debug, Clone)]
pub struct Supercluster {
    /// Clusters at each zoom level (index = zoom).
    levels: Vec<Vec<Cluster>>,
    /// Original points.
    points: Vec<MarkerPoint>,
    /// Grid size in pixels.
    grid_size: f64,
    /// Min zoom.
    min_zoom: u32,
    /// Max zoom.
    max_zoom: u32,
}

impl Supercluster {
    /// Build a supercluster from points.
    pub fn new(points: Vec<MarkerPoint>, min_zoom: u32, max_zoom: u32, grid_size: f64) -> Self {
        let mut levels = Vec::with_capacity((max_zoom - min_zoom + 1) as usize);
        for z in min_zoom..=max_zoom {
            levels.push(grid_cluster(&points, z, grid_size));
        }
        Self {
            levels,
            points,
            grid_size,
            min_zoom,
            max_zoom,
        }
    }

    /// Get clusters at a given zoom level.
    pub fn clusters_at_zoom(&self, zoom: u32) -> &[Cluster] {
        let zoom = zoom.clamp(self.min_zoom, self.max_zoom);
        let idx = (zoom - self.min_zoom) as usize;
        &self.levels[idx]
    }

    /// Get the original points.
    pub fn points(&self) -> &[MarkerPoint] {
        &self.points
    }

    /// Get the number of zoom levels.
    pub fn zoom_levels(&self) -> u32 {
        self.max_zoom - self.min_zoom + 1
    }

    /// Expand a cluster: return its children (sub-clusters or individual points)
    /// at the next zoom level.
    pub fn expand_cluster(&self, cluster: &Cluster, zoom: u32) -> Vec<Cluster> {
        if zoom >= self.max_zoom {
            // At max zoom, return individual points as single-point clusters
            return cluster
                .point_ids
                .iter()
                .filter_map(|id| {
                    self.points.iter().find(|p| p.id == *id).map(|p| Cluster {
                        center_lat: p.lat,
                        center_lng: p.lng,
                        point_count: 1,
                        bounds: (p.lat, p.lng, p.lat, p.lng),
                        point_ids: vec![p.id],
                        cluster_id: p.id,
                    })
                })
                .collect();
        }
        let next_zoom = zoom + 1;
        let next_clusters = self.clusters_at_zoom(next_zoom);
        next_clusters
            .iter()
            .filter(|c| {
                c.point_ids
                    .iter()
                    .any(|id| cluster.point_ids.contains(id))
            })
            .cloned()
            .collect()
    }
}

// ── Spiderfy ────────────────────────────────────────────────────

/// A point arranged in a spider pattern for display.
#[derive(Debug, Clone)]
pub struct SpiderfyLeg {
    /// Original point id.
    pub point_id: u64,
    /// Offset x (pixels from cluster center).
    pub offset_x: f64,
    /// Offset y (pixels from cluster center).
    pub offset_y: f64,
    /// Angle from center (radians).
    pub angle: f64,
}

/// Arrange overlapping markers in a circle around the cluster center.
pub fn spiderfy(cluster: &Cluster, leg_length: f64) -> Vec<SpiderfyLeg> {
    let count = cluster.point_ids.len();
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![SpiderfyLeg {
            point_id: cluster.point_ids[0],
            offset_x: 0.0,
            offset_y: 0.0,
            angle: 0.0,
        }];
    }

    let angle_step = 2.0 * std::f64::consts::PI / count as f64;
    // Use spiral for many points, circle for few
    let use_spiral = count > 10;

    cluster
        .point_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let angle = angle_step * i as f64;
            let radius = if use_spiral {
                leg_length * (1.0 + 0.3 * i as f64)
            } else {
                leg_length
            };
            SpiderfyLeg {
                point_id: id,
                offset_x: radius * angle.cos(),
                offset_y: radius * angle.sin(),
                angle,
            }
        })
        .collect()
}

// ── Distance between markers (for clustering quality) ───────────

/// Squared Euclidean distance in projected space (simple lat/lng).
fn dist_sq(a: &MarkerPoint, b: &MarkerPoint) -> f64 {
    let dlat = a.lat - b.lat;
    let dlng = a.lng - b.lng;
    dlat * dlat + dlng * dlng
}

/// Find the nearest point to a given location.
pub fn nearest_point(points: &[MarkerPoint], lat: f64, lng: f64) -> Option<&MarkerPoint> {
    let query = MarkerPoint::new(0, lat, lng);
    points.iter().min_by(|a, b| {
        dist_sq(a, &query)
            .partial_cmp(&dist_sq(b, &query))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_points() -> Vec<MarkerPoint> {
        vec![
            MarkerPoint::new(1, 40.7128, -74.0060),
            MarkerPoint::new(2, 40.7138, -74.0050),
            MarkerPoint::new(3, 40.7148, -74.0040),
            MarkerPoint::new(4, 51.5074, -0.1278),
            MarkerPoint::new(5, 51.5084, -0.1268),
            MarkerPoint::new(6, 35.6762, 139.6503),
            MarkerPoint::new(7, 48.8566, 2.3522),
            MarkerPoint::new(8, 48.8576, 2.3532),
        ]
    }

    #[test]
    fn test_grid_cluster_groups_nearby() {
        let points = sample_points();
        let clusters = grid_cluster(&points, 2, 256.0);
        // At low zoom, nearby points should cluster together
        assert!(!clusters.is_empty());
        let total: usize = clusters.iter().map(|c| c.point_count).sum();
        assert_eq!(total, points.len());
    }

    #[test]
    fn test_grid_cluster_more_clusters_at_high_zoom() {
        let points = sample_points();
        let low = grid_cluster(&points, 2, 256.0);
        let high = grid_cluster(&points, 16, 256.0);
        assert!(high.len() >= low.len());
    }

    #[test]
    fn test_cluster_bounds() {
        let points = sample_points();
        let clusters = grid_cluster(&points, 5, 256.0);
        for c in &clusters {
            assert!(c.bounds.0 <= c.bounds.2); // min_lat <= max_lat
            assert!(c.bounds.1 <= c.bounds.3); // min_lng <= max_lng
        }
    }

    #[test]
    fn test_supercluster_build() {
        let points = sample_points();
        let sc = Supercluster::new(points, 0, 16, 256.0);
        assert_eq!(sc.zoom_levels(), 17);
    }

    #[test]
    fn test_supercluster_clusters_at_zoom() {
        let points = sample_points();
        let sc = Supercluster::new(points.clone(), 0, 16, 256.0);
        let c0 = sc.clusters_at_zoom(0);
        let c16 = sc.clusters_at_zoom(16);
        assert!(c16.len() >= c0.len());
        // Total points should be conserved at each level
        let total_0: usize = c0.iter().map(|c| c.point_count).sum();
        let total_16: usize = c16.iter().map(|c| c.point_count).sum();
        assert_eq!(total_0, points.len());
        assert_eq!(total_16, points.len());
    }

    #[test]
    fn test_supercluster_expand() {
        let points = sample_points();
        let sc = Supercluster::new(points, 0, 16, 256.0);
        let clusters = sc.clusters_at_zoom(0);
        if let Some(c) = clusters.first() {
            let children = sc.expand_cluster(c, 0);
            assert!(!children.is_empty());
        }
    }

    #[test]
    fn test_spiderfy_single() {
        let cluster = Cluster {
            center_lat: 40.0,
            center_lng: -74.0,
            point_count: 1,
            bounds: (40.0, -74.0, 40.0, -74.0),
            point_ids: vec![1],
            cluster_id: 0,
        };
        let legs = spiderfy(&cluster, 30.0);
        assert_eq!(legs.len(), 1);
        assert_eq!(legs[0].offset_x, 0.0);
    }

    #[test]
    fn test_spiderfy_circle() {
        let cluster = Cluster {
            center_lat: 40.0,
            center_lng: -74.0,
            point_count: 6,
            bounds: (39.9, -74.1, 40.1, -73.9),
            point_ids: vec![1, 2, 3, 4, 5, 6],
            cluster_id: 0,
        };
        let legs = spiderfy(&cluster, 30.0);
        assert_eq!(legs.len(), 6);
        // All should be at the same distance from center (circle)
        for leg in &legs {
            let dist = (leg.offset_x.powi(2) + leg.offset_y.powi(2)).sqrt();
            assert!((dist - 30.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_spiderfy_spiral() {
        let ids: Vec<u64> = (1..=15).collect();
        let cluster = Cluster {
            center_lat: 40.0,
            center_lng: -74.0,
            point_count: 15,
            bounds: (39.9, -74.1, 40.1, -73.9),
            point_ids: ids,
            cluster_id: 0,
        };
        let legs = spiderfy(&cluster, 20.0);
        assert_eq!(legs.len(), 15);
        // Spiral: later legs should be farther from center
        let d0 = (legs[0].offset_x.powi(2) + legs[0].offset_y.powi(2)).sqrt();
        let d14 = (legs[14].offset_x.powi(2) + legs[14].offset_y.powi(2)).sqrt();
        assert!(d14 > d0);
    }

    #[test]
    fn test_nearest_point() {
        let points = sample_points();
        let nearest = nearest_point(&points, 40.7130, -74.0058).unwrap();
        assert_eq!(nearest.id, 1); // Closest to NYC point
    }

    #[test]
    fn test_empty_cluster() {
        let clusters = grid_cluster(&[], 5, 256.0);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_marker_point_with_data() {
        let p = MarkerPoint::new(42, 10.0, 20.0).with_data("test".to_string());
        assert_eq!(p.data, Some("test".to_string()));
        assert_eq!(p.id, 42);
    }
}
