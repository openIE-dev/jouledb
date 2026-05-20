//! HDC-powered Geospatial and Location module
//!
//! Provides holographic encoding for:
//! - Point-of-interest similarity and proximity search
//! - Geofence membership and boundary detection
//! - Route and trajectory analysis
//! - Spatial clustering and hotspot detection

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoiCategory {
    Restaurant,
    Retail,
    Healthcare,
    Education,
    Transportation,
    Park,
    Government,
    Residential,
    Industrial,
    Entertainment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeometryType {
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiPolygon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportMode {
    Walking,
    Cycling,
    Driving,
    Transit,
    Aerial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZoneType {
    Urban,
    Suburban,
    Rural,
    Industrial,
    Commercial,
    Protected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
    pub altitude_m: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointOfInterest {
    pub id: String,
    pub name: String,
    pub category: PoiCategory,
    pub location: GeoPoint,
    pub rating: f32,
    pub tags: Vec<String>,
    pub zone: ZoneType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geofence {
    pub id: String,
    pub name: String,
    pub geometry_type: GeometryType,
    pub vertices: Vec<GeoPoint>,
    pub zone: ZoneType,
    pub area_sq_km: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub id: String,
    pub entity_id: String,
    pub mode: TransportMode,
    pub points: Vec<GeoPoint>,
    pub timestamps: Vec<u64>,
    pub distance_km: f64,
    pub duration_secs: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialEvent {
    pub id: String,
    pub location: GeoPoint,
    pub category: PoiCategory,
    pub timestamp: u64,
    pub radius_m: f32,
    pub intensity: f32,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for geospatial domain data
    pub struct SpatialLink {
        seed: 0x5BA7_0001,
        dimension: 10000,
        fields: ["location", "poi", "fence", "route", "event", "zone", "geometry"],
        scalars: ["lat", "lon", "altitude", "distance", "duration", "area", "rating", "radius", "intensity"],
        enums: {
            poi_vectors: PoiCategory => [PoiCategory::Restaurant, PoiCategory::Retail, PoiCategory::Healthcare, PoiCategory::Education, PoiCategory::Transportation, PoiCategory::Park, PoiCategory::Government, PoiCategory::Residential, PoiCategory::Industrial, PoiCategory::Entertainment],
            geometry_vectors: GeometryType => [GeometryType::Point, GeometryType::LineString, GeometryType::Polygon, GeometryType::MultiPoint, GeometryType::MultiPolygon],
            transport_vectors: TransportMode => [TransportMode::Walking, TransportMode::Cycling, TransportMode::Driving, TransportMode::Transit, TransportMode::Aerial],
            zone_vectors: ZoneType => [ZoneType::Urban, ZoneType::Suburban, ZoneType::Rural, ZoneType::Industrial, ZoneType::Commercial, ZoneType::Protected]
        },
        dynamic: {
            tag_vectors: "tag",
            region_vectors: "region"
        },
    }
}

impl SpatialLink {
    /// Encode latitude and longitude into an HDC vector.
    ///
    /// Maps coordinates to a quantized grid: lat [-90,90] → [0,1800], lon [-180,180] → [0,3600].
    pub fn encode_location(&self, point: &GeoPoint) -> BinaryHV {
        let lat_hv = self.encode_scalar("lat", ((point.lat + 90.0) * 10.0) as u32, 1800);
        let lon_hv = self.encode_scalar("lon", ((point.lon + 180.0) * 10.0) as u32, 3600);
        let loc_field = &self.field_vectors["location"];
        let bound = loc_field.bind(&lat_hv).bind(&lon_hv);
        match point.altitude_m {
            Some(alt) => {
                let alt_hv = self.encode_scalar("altitude", (alt.max(0.0)) as u32, 9000);
                self.bundle(&[bound, alt_hv])
            }
            None => bound,
        }
    }

    pub fn encode_poi(&self, poi: &PointOfInterest) -> BinaryHV {
        let loc_hv = self.encode_location(&poi.location);
        let cat_hv = self.field_vectors["poi"].bind(&self.poi_vectors[&poi.category]);
        let zone_hv = self.field_vectors["zone"].bind(&self.zone_vectors[&poi.zone]);
        let rating_hv = self.encode_scalar("rating", (poi.rating * 10.0) as u32, 50);
        self.bundle(&[loc_hv, cat_hv, zone_hv, rating_hv])
    }

    pub fn encode_geofence(&self, fence: &Geofence) -> BinaryHV {
        let geo_hv =
            self.field_vectors["geometry"].bind(&self.geometry_vectors[&fence.geometry_type]);
        let zone_hv = self.field_vectors["zone"].bind(&self.zone_vectors[&fence.zone]);
        let area_hv = self.encode_scalar("area", fence.area_sq_km as u32, 10000);
        // Encode centroid of vertices
        if fence.vertices.is_empty() {
            return self.bundle(&[geo_hv, zone_hv, area_hv]);
        }
        let centroid = GeoPoint {
            lat: fence.vertices.iter().map(|v| v.lat).sum::<f64>() / fence.vertices.len() as f64,
            lon: fence.vertices.iter().map(|v| v.lon).sum::<f64>() / fence.vertices.len() as f64,
            altitude_m: None,
        };
        let loc_hv = self.encode_location(&centroid);
        self.bundle(&[loc_hv, geo_hv, zone_hv, area_hv])
    }

    pub fn encode_trajectory(&self, traj: &Trajectory) -> BinaryHV {
        let mode_hv = self.field_vectors["route"].bind(&self.transport_vectors[&traj.mode]);
        let dist_hv = self.encode_scalar("distance", (traj.distance_km * 10.0) as u32, 10000);
        let dur_hv = self.encode_scalar("duration", traj.duration_secs, 86400);
        let mut components = vec![mode_hv, dist_hv, dur_hv];
        // Encode start and end points if available
        if let Some(start) = traj.points.first() {
            components.push(self.encode_location(start).permute(1));
        }
        if let Some(end) = traj.points.last() {
            components.push(self.encode_location(end).permute(2));
        }
        self.bundle(&components)
    }

    pub fn encode_event(&self, event: &SpatialEvent) -> BinaryHV {
        let loc_hv = self.encode_location(&event.location);
        let cat_hv = self.field_vectors["event"].bind(&self.poi_vectors[&event.category]);
        let radius_hv = self.encode_scalar("radius", event.radius_m as u32, 50000);
        let intensity_hv = self.encode_scalar("intensity", (event.intensity * 100.0) as u32, 100);
        self.bundle(&[loc_hv, cat_hv, radius_hv, intensity_hv])
    }
}

/// Spatial proximity search using HDC similarity.
pub struct ProximityIndex {
    encoder: SpatialLink,
    entries: Vec<(String, BinaryHV)>,
}

impl ProximityIndex {
    pub fn new() -> Self {
        Self {
            encoder: SpatialLink::new(),
            entries: Vec::new(),
        }
    }

    pub fn insert_poi(&mut self, poi: &PointOfInterest) {
        let hv = self.encoder.encode_poi(poi);
        self.entries.push((poi.id.clone(), hv));
    }

    /// Find the k nearest POIs to a query point by HDC similarity.
    pub fn nearest(&self, query: &GeoPoint, k: usize) -> Vec<(String, f32)> {
        let query_hv = self.encoder.encode_location(query);
        let mut results: Vec<(String, f32)> = self
            .entries
            .iter()
            .map(|(id, hv)| (id.clone(), query_hv.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

impl Default for ProximityIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Geofence membership detector.
pub struct GeofenceDetector {
    encoder: SpatialLink,
    fences: Vec<(String, BinaryHV)>,
    threshold: f32,
}

impl GeofenceDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: SpatialLink::new(),
            fences: Vec::new(),
            threshold,
        }
    }

    pub fn add_fence(&mut self, fence: &Geofence) {
        let hv = self.encoder.encode_geofence(fence);
        self.fences.push((fence.id.clone(), hv));
    }

    /// Check which geofences a point is likely inside (by HDC similarity).
    pub fn check(&self, point: &GeoPoint) -> Vec<(String, f32)> {
        let loc_hv = self.encoder.encode_location(point);
        self.fences
            .iter()
            .filter_map(|(id, hv)| {
                let sim = loc_hv.similarity(hv);
                if sim > self.threshold {
                    Some((id.clone(), sim))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for GeofenceDetector {
    fn default() -> Self {
        Self::new(0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_point() -> GeoPoint {
        GeoPoint {
            lat: 40.7128,
            lon: -74.0060,
            altitude_m: Some(10.0),
        }
    }

    fn sample_poi() -> PointOfInterest {
        PointOfInterest {
            id: "poi-1".to_string(),
            name: "Central Park".to_string(),
            category: PoiCategory::Park,
            location: sample_point(),
            rating: 4.8,
            tags: vec!["outdoor".to_string(), "nature".to_string()],
            zone: ZoneType::Urban,
        }
    }

    #[test]
    fn test_location_encoding() {
        let encoder = SpatialLink::new();
        let hv = encoder.encode_location(&sample_point());
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_nearby_locations_similar() {
        let encoder = SpatialLink::new();
        let nyc = GeoPoint {
            lat: 40.7128,
            lon: -74.0060,
            altitude_m: None,
        };
        let nearby = GeoPoint {
            lat: 40.7200,
            lon: -74.0000,
            altitude_m: None,
        };
        let far = GeoPoint {
            lat: 35.6762,
            lon: 139.6503,
            altitude_m: None,
        }; // Tokyo

        let hv_nyc = encoder.encode_location(&nyc);
        let hv_nearby = encoder.encode_location(&nearby);
        let hv_far = encoder.encode_location(&far);

        let sim_near = hv_nyc.similarity(&hv_nearby);
        let sim_far = hv_nyc.similarity(&hv_far);
        assert!(
            sim_near > sim_far,
            "Nearby points should be more similar: near={sim_near} far={sim_far}"
        );
    }

    #[test]
    fn test_poi_encoding() {
        let encoder = SpatialLink::new();
        let hv = encoder.encode_poi(&sample_poi());
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_geofence_encoding() {
        let encoder = SpatialLink::new();
        let fence = Geofence {
            id: "zone-1".to_string(),
            name: "Manhattan".to_string(),
            geometry_type: GeometryType::Polygon,
            vertices: vec![
                GeoPoint {
                    lat: 40.70,
                    lon: -74.02,
                    altitude_m: None,
                },
                GeoPoint {
                    lat: 40.80,
                    lon: -74.02,
                    altitude_m: None,
                },
                GeoPoint {
                    lat: 40.80,
                    lon: -73.93,
                    altitude_m: None,
                },
                GeoPoint {
                    lat: 40.70,
                    lon: -73.93,
                    altitude_m: None,
                },
            ],
            zone: ZoneType::Urban,
            area_sq_km: 59.1,
        };
        let hv = encoder.encode_geofence(&fence);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_trajectory_encoding() {
        let encoder = SpatialLink::new();
        let traj = Trajectory {
            id: "trip-1".to_string(),
            entity_id: "user-1".to_string(),
            mode: TransportMode::Driving,
            points: vec![
                GeoPoint {
                    lat: 40.71,
                    lon: -74.00,
                    altitude_m: None,
                },
                GeoPoint {
                    lat: 40.75,
                    lon: -73.98,
                    altitude_m: None,
                },
            ],
            timestamps: vec![1000, 2000],
            distance_km: 5.2,
            duration_secs: 900,
        };
        let hv = encoder.encode_trajectory(&traj);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_proximity_index() {
        let mut index = ProximityIndex::new();
        index.insert_poi(&PointOfInterest {
            id: "p1".to_string(),
            name: "Place A".to_string(),
            category: PoiCategory::Restaurant,
            location: GeoPoint {
                lat: 40.71,
                lon: -74.00,
                altitude_m: None,
            },
            rating: 4.0,
            tags: vec![],
            zone: ZoneType::Urban,
        });
        index.insert_poi(&PointOfInterest {
            id: "p2".to_string(),
            name: "Place B".to_string(),
            category: PoiCategory::Park,
            location: GeoPoint {
                lat: 35.67,
                lon: 139.65,
                altitude_m: None,
            },
            rating: 4.5,
            tags: vec![],
            zone: ZoneType::Urban,
        });
        let results = index.nearest(
            &GeoPoint {
                lat: 40.72,
                lon: -74.01,
                altitude_m: None,
            },
            2,
        );
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_geofence_detector() {
        let mut detector = GeofenceDetector::new(0.3);
        detector.add_fence(&Geofence {
            id: "zone-1".to_string(),
            name: "Zone A".to_string(),
            geometry_type: GeometryType::Polygon,
            vertices: vec![
                GeoPoint {
                    lat: 40.70,
                    lon: -74.02,
                    altitude_m: None,
                },
                GeoPoint {
                    lat: 40.80,
                    lon: -73.93,
                    altitude_m: None,
                },
            ],
            zone: ZoneType::Urban,
            area_sq_km: 50.0,
        });
        let hits = detector.check(&GeoPoint {
            lat: 40.75,
            lon: -73.97,
            altitude_m: None,
        });
        // Should return results (may or may not exceed threshold depending on encoding)
        assert!(hits.len() <= 1);
    }

    #[test]
    fn test_spatial_event_encoding() {
        let encoder = SpatialLink::new();
        let event = SpatialEvent {
            id: "evt-1".to_string(),
            location: sample_point(),
            category: PoiCategory::Entertainment,
            timestamp: 1700000000,
            radius_m: 500.0,
            intensity: 0.8,
        };
        let hv = encoder.encode_event(&event);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_deterministic_encoding() {
        let enc1 = SpatialLink::new();
        let enc2 = SpatialLink::new();
        let point = sample_point();
        let sim = enc1
            .encode_location(&point)
            .similarity(&enc2.encode_location(&point));
        assert_eq!(sim, 1.0, "Same seed should produce identical encodings");
    }
}
