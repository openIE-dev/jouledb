//! Geolocation: position tracking, haversine distance, geofencing.

use chrono::{DateTime, Utc};

// ── Types ───────────────────────────────────────────────────────

/// A geographic position.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoPosition {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
    pub accuracy: f64,
    pub heading: Option<f64>,
    pub speed: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Geolocation errors.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GeoError {
    #[error("permission denied")]
    PermissionDenied,
    #[error("position unavailable")]
    PositionUnavailable,
    #[error("timeout")]
    Timeout,
}

/// Options for requesting a position.
#[derive(Debug, Clone)]
pub struct GeoOptions {
    pub enable_high_accuracy: bool,
    pub timeout_ms: Option<u64>,
    pub maximum_age_ms: Option<u64>,
}

impl Default for GeoOptions {
    fn default() -> Self {
        Self {
            enable_high_accuracy: false,
            timeout_ms: None,
            maximum_age_ms: None,
        }
    }
}

// ── GeoState ────────────────────────────────────────────────────

/// Stateful geolocation tracker with history.
#[derive(Debug, Clone)]
pub struct GeoState {
    last_position: Option<GeoPosition>,
    watching: bool,
    history: Vec<GeoPosition>,
    max_history: usize,
}

impl GeoState {
    pub fn new() -> Self {
        Self {
            last_position: None,
            watching: false,
            history: Vec::new(),
            max_history: 1000,
        }
    }

    pub fn update_position(&mut self, pos: GeoPosition) {
        self.last_position = Some(pos.clone());
        self.history.push(pos);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    pub fn last_position(&self) -> Option<&GeoPosition> {
        self.last_position.as_ref()
    }

    /// Total distance traveled across history, in meters (haversine sum).
    pub fn distance_traveled(&self) -> f64 {
        self.history
            .windows(2)
            .map(|w| {
                haversine_distance(
                    w[0].latitude, w[0].longitude,
                    w[1].latitude, w[1].longitude,
                )
            })
            .sum()
    }

    pub fn start_watch(&mut self) {
        self.watching = true;
    }

    pub fn stop_watch(&mut self) {
        self.watching = false;
    }

    pub fn is_watching(&self) -> bool {
        self.watching
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl Default for GeoState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Haversine ───────────────────────────────────────────────────

/// Earth radius in meters (WGS-84 mean).
const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Great-circle distance between two points in meters.
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();

    let a = (d_lat / 2.0).sin().powi(2)
        + lat1_r.cos() * lat2_r.cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    EARTH_RADIUS_M * c
}

// ── GeoFence ────────────────────────────────────────────────────

/// A circular geofence around a center point.
#[derive(Debug, Clone)]
pub struct GeoFence {
    pub center: GeoPosition,
    pub radius_meters: f64,
}

impl GeoFence {
    /// True if the position is inside the fence.
    pub fn contains(&self, pos: &GeoPosition) -> bool {
        let d = haversine_distance(
            self.center.latitude, self.center.longitude,
            pos.latitude, pos.longitude,
        );
        d <= self.radius_meters
    }

    /// Signed distance to the fence edge (negative = inside, positive = outside).
    pub fn distance_to_edge(&self, pos: &GeoPosition) -> f64 {
        let d = haversine_distance(
            self.center.latitude, self.center.longitude,
            pos.latitude, pos.longitude,
        );
        d - self.radius_meters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn pos(lat: f64, lon: f64) -> GeoPosition {
        GeoPosition {
            latitude: lat,
            longitude: lon,
            altitude: None,
            accuracy: 10.0,
            heading: None,
            speed: None,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn haversine_known_distance() {
        // New York (40.7128, -74.0060) to Los Angeles (34.0522, -118.2437)
        // Known distance: ~3,944 km
        let d = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437);
        assert!((d - 3_944_000.0).abs() < 50_000.0, "got {d}");
    }

    #[test]
    fn haversine_same_point() {
        let d = haversine_distance(51.5074, -0.1278, 51.5074, -0.1278);
        assert!(d.abs() < 0.01);
    }

    #[test]
    fn geofence_inside() {
        let fence = GeoFence {
            center: pos(40.7128, -74.0060),
            radius_meters: 1000.0,
        };
        // Very close point.
        let p = pos(40.7130, -74.0060);
        assert!(fence.contains(&p));
        assert!(fence.distance_to_edge(&p) < 0.0);
    }

    #[test]
    fn geofence_outside() {
        let fence = GeoFence {
            center: pos(40.7128, -74.0060),
            radius_meters: 100.0,
        };
        // ~3,944 km away
        let p = pos(34.0522, -118.2437);
        assert!(!fence.contains(&p));
        assert!(fence.distance_to_edge(&p) > 0.0);
    }

    #[test]
    fn distance_traveled() {
        let mut state = GeoState::new();
        state.update_position(pos(0.0, 0.0));
        state.update_position(pos(1.0, 0.0));
        state.update_position(pos(2.0, 0.0));

        let total = state.distance_traveled();
        // Each degree of latitude ≈ 111 km, so ~222 km total.
        assert!((total - 222_400.0).abs() < 5_000.0, "got {total}");
    }

    #[test]
    fn history_tracking() {
        let mut state = GeoState::new();
        assert!(state.last_position().is_none());
        state.update_position(pos(10.0, 20.0));
        assert!(state.last_position().is_some());
        assert!((state.last_position().unwrap().latitude - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn watch_state() {
        let mut state = GeoState::new();
        assert!(!state.is_watching());
        state.start_watch();
        assert!(state.is_watching());
        state.stop_watch();
        assert!(!state.is_watching());
    }

    #[test]
    fn clear_history_resets() {
        let mut state = GeoState::new();
        state.update_position(pos(1.0, 1.0));
        state.update_position(pos(2.0, 2.0));
        state.clear_history();
        assert!((state.distance_traveled() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn distance_traveled_empty() {
        let state = GeoState::new();
        assert!((state.distance_traveled() - 0.0).abs() < f64::EPSILON);
    }
}
