//! Map engine: Web Mercator projection, viewport management, tile coordinates,
//! pan/zoom, bounding box, and fly-to animation interpolation.

use std::f64::consts::PI;

// ── Constants ───────────────────────────────────────────────────

const TILE_SIZE: f64 = 256.0;
const MAX_LATITUDE: f64 = 85.051_128_78;

// ── MapState ────────────────────────────────────────────────────

/// The current state of the map view.
#[derive(Debug, Clone, PartialEq)]
pub struct MapState {
    /// Center latitude in degrees (-85.05..85.05).
    pub center_lat: f64,
    /// Center longitude in degrees (-180..180).
    pub center_lng: f64,
    /// Zoom level (0..22).
    pub zoom: f64,
    /// Bearing in degrees clockwise from north (0..360).
    pub bearing: f64,
    /// Pitch in degrees (0..60).
    pub pitch: f64,
}

impl Default for MapState {
    fn default() -> Self {
        Self {
            center_lat: 0.0,
            center_lng: 0.0,
            zoom: 2.0,
            bearing: 0.0,
            pitch: 0.0,
        }
    }
}

impl MapState {
    pub fn new(lat: f64, lng: f64, zoom: f64) -> Self {
        Self {
            center_lat: lat.clamp(-MAX_LATITUDE, MAX_LATITUDE),
            center_lng: wrap_lng(lng),
            zoom: zoom.clamp(0.0, 22.0),
            bearing: 0.0,
            pitch: 0.0,
        }
    }

    pub fn with_bearing(mut self, bearing: f64) -> Self {
        self.bearing = bearing % 360.0;
        self
    }

    pub fn with_pitch(mut self, pitch: f64) -> Self {
        self.pitch = pitch.clamp(0.0, 60.0);
        self
    }
}

// ── Viewport ────────────────────────────────────────────────────

/// Pixel dimensions of the visible map area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

// ── LatLng ──────────────────────────────────────────────────────

/// A geographic coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatLng {
    pub lat: f64,
    pub lng: f64,
}

impl LatLng {
    pub fn new(lat: f64, lng: f64) -> Self {
        Self { lat, lng }
    }
}

// ── Pixel ───────────────────────────────────────────────────────

/// A pixel coordinate on screen (origin = top-left of viewport).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pixel {
    pub x: f64,
    pub y: f64,
}

// ── BoundingBox ─────────────────────────────────────────────────

/// Geographic bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub sw: LatLng,
    pub ne: LatLng,
}

impl BoundingBox {
    pub fn contains(&self, p: &LatLng) -> bool {
        p.lat >= self.sw.lat && p.lat <= self.ne.lat && p.lng >= self.sw.lng && p.lng <= self.ne.lng
    }

    pub fn width(&self) -> f64 {
        self.ne.lng - self.sw.lng
    }

    pub fn height(&self) -> f64 {
        self.ne.lat - self.sw.lat
    }
}

// ── TileCoord ───────────────────────────────────────────────────

/// A map tile coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub z: u32,
    pub x: u32,
    pub y: u32,
}

// ── Web Mercator Projection (EPSG:3857) ─────────────────────────

/// Convert latitude/longitude to absolute pixel coordinates at a given zoom.
pub fn latlng_to_pixel_absolute(lat: f64, lng: f64, zoom: f64) -> Pixel {
    let scale = TILE_SIZE * 2.0_f64.powf(zoom);
    let x = (lng + 180.0) / 360.0 * scale;
    let lat_rad = lat.clamp(-MAX_LATITUDE, MAX_LATITUDE).to_radians();
    let y = (1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI) / 2.0 * scale;
    Pixel { x, y }
}

/// Convert absolute pixel coordinates back to latitude/longitude at a given zoom.
pub fn pixel_absolute_to_latlng(px: f64, py: f64, zoom: f64) -> LatLng {
    let scale = TILE_SIZE * 2.0_f64.powf(zoom);
    let lng = px / scale * 360.0 - 180.0;
    let n = PI - 2.0 * PI * py / scale;
    let lat = (0.5 * (n.exp() - (-n).exp())).atan().to_degrees();
    LatLng { lat, lng }
}

/// Convert lat/lng to viewport pixel coordinates relative to the map center.
pub fn latlng_to_viewport_pixel(
    point: &LatLng,
    state: &MapState,
    viewport: &Viewport,
) -> Pixel {
    let center_abs = latlng_to_pixel_absolute(state.center_lat, state.center_lng, state.zoom);
    let point_abs = latlng_to_pixel_absolute(point.lat, point.lng, state.zoom);
    Pixel {
        x: point_abs.x - center_abs.x + (viewport.width as f64) / 2.0,
        y: point_abs.y - center_abs.y + (viewport.height as f64) / 2.0,
    }
}

/// Convert viewport pixel coordinates back to lat/lng.
pub fn viewport_pixel_to_latlng(
    pixel: &Pixel,
    state: &MapState,
    viewport: &Viewport,
) -> LatLng {
    let center_abs = latlng_to_pixel_absolute(state.center_lat, state.center_lng, state.zoom);
    let abs_x = pixel.x + center_abs.x - (viewport.width as f64) / 2.0;
    let abs_y = pixel.y + center_abs.y - (viewport.height as f64) / 2.0;
    pixel_absolute_to_latlng(abs_x, abs_y, state.zoom)
}

// ── Tile Calculation ────────────────────────────────────────────

/// Calculate which tiles are visible in the viewport.
pub fn visible_tiles(state: &MapState, viewport: &Viewport) -> Vec<TileCoord> {
    let z = state.zoom.floor() as u32;
    let max_tile = 1u32 << z;
    let center_abs = latlng_to_pixel_absolute(state.center_lat, state.center_lng, state.zoom);
    let half_w = (viewport.width as f64) / 2.0;
    let half_h = (viewport.height as f64) / 2.0;

    // Scale factor from fractional zoom to integer zoom tiles
    let frac = state.zoom - (z as f64);
    let tile_scale = TILE_SIZE * 2.0_f64.powf(frac);

    let min_tx = ((center_abs.x - half_w) / tile_scale).floor() as i64;
    let max_tx = ((center_abs.x + half_w) / tile_scale).ceil() as i64;
    let min_ty = ((center_abs.y - half_h) / tile_scale).floor() as i64;
    let max_ty = ((center_abs.y + half_h) / tile_scale).ceil() as i64;

    let mut tiles = Vec::new();
    for ty in min_ty..=max_ty {
        if ty < 0 || ty >= max_tile as i64 {
            continue;
        }
        for tx in min_tx..=max_tx {
            let wrapped = ((tx % max_tile as i64) + max_tile as i64) as u32 % max_tile;
            tiles.push(TileCoord {
                z,
                x: wrapped,
                y: ty as u32,
            });
        }
    }
    tiles
}

// ── Pan ─────────────────────────────────────────────────────────

/// Pan the map by a pixel delta, returning a new MapState.
pub fn pan(state: &MapState, dx: f64, dy: f64) -> MapState {
    let center_abs = latlng_to_pixel_absolute(state.center_lat, state.center_lng, state.zoom);
    let new_center = pixel_absolute_to_latlng(
        center_abs.x + dx,
        center_abs.y + dy,
        state.zoom,
    );
    MapState {
        center_lat: new_center.lat.clamp(-MAX_LATITUDE, MAX_LATITUDE),
        center_lng: wrap_lng(new_center.lng),
        zoom: state.zoom,
        bearing: state.bearing,
        pitch: state.pitch,
    }
}

// ── Zoom ────────────────────────────────────────────────────────

/// Zoom toward a screen point (e.g., mouse position).
pub fn zoom_at(
    state: &MapState,
    viewport: &Viewport,
    screen_x: f64,
    screen_y: f64,
    new_zoom: f64,
) -> MapState {
    let new_zoom = new_zoom.clamp(0.0, 22.0);
    let target = viewport_pixel_to_latlng(
        &Pixel { x: screen_x, y: screen_y },
        state,
        viewport,
    );
    // After zoom, the target point should stay at the same screen position.
    let target_abs_new = latlng_to_pixel_absolute(target.lat, target.lng, new_zoom);
    let half_w = (viewport.width as f64) / 2.0;
    let half_h = (viewport.height as f64) / 2.0;
    let new_center_abs_x = target_abs_new.x - (screen_x - half_w);
    let new_center_abs_y = target_abs_new.y - (screen_y - half_h);
    let new_center = pixel_absolute_to_latlng(new_center_abs_x, new_center_abs_y, new_zoom);
    MapState {
        center_lat: new_center.lat.clamp(-MAX_LATITUDE, MAX_LATITUDE),
        center_lng: wrap_lng(new_center.lng),
        zoom: new_zoom,
        bearing: state.bearing,
        pitch: state.pitch,
    }
}

// ── Bounding Box ────────────────────────────────────────────────

/// Compute the geographic bounding box of the visible area.
pub fn visible_bounds(state: &MapState, viewport: &Viewport) -> BoundingBox {
    let sw = viewport_pixel_to_latlng(
        &Pixel { x: 0.0, y: viewport.height as f64 },
        state,
        viewport,
    );
    let ne = viewport_pixel_to_latlng(
        &Pixel { x: viewport.width as f64, y: 0.0 },
        state,
        viewport,
    );
    BoundingBox { sw, ne }
}

// ── Fly-To Animation ────────────────────────────────────────────

/// Interpolate between two map states for a fly-to animation.
/// `t` ranges from 0.0 (start) to 1.0 (end).
pub fn fly_to_interpolate(from: &MapState, to: &MapState, t: f64) -> MapState {
    let t = t.clamp(0.0, 1.0);
    // Use ease-in-out cubic for smooth animation
    let eased = if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0_f64).powi(3) / 2.0
    };

    MapState {
        center_lat: lerp(from.center_lat, to.center_lat, eased),
        center_lng: lerp_lng(from.center_lng, to.center_lng, eased),
        zoom: lerp(from.zoom, to.zoom, eased),
        bearing: lerp_angle(from.bearing, to.bearing, eased),
        pitch: lerp(from.pitch, to.pitch, eased),
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn wrap_lng(lng: f64) -> f64 {
    let mut l = lng;
    while l > 180.0 {
        l -= 360.0;
    }
    while l < -180.0 {
        l += 360.0;
    }
    l
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn lerp_lng(a: f64, b: f64, t: f64) -> f64 {
    let mut diff = b - a;
    if diff > 180.0 {
        diff -= 360.0;
    }
    if diff < -180.0 {
        diff += 360.0;
    }
    wrap_lng(a + diff * t)
}

fn lerp_angle(a: f64, b: f64, t: f64) -> f64 {
    let mut diff = b - a;
    if diff > 180.0 {
        diff -= 360.0;
    }
    if diff < -180.0 {
        diff += 360.0;
    }
    ((a + diff * t) % 360.0 + 360.0) % 360.0
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_map_state() {
        let s = MapState::default();
        assert_eq!(s.center_lat, 0.0);
        assert_eq!(s.center_lng, 0.0);
        assert_eq!(s.zoom, 2.0);
    }

    #[test]
    fn test_map_state_clamping() {
        let s = MapState::new(100.0, 200.0, 25.0);
        assert!(s.center_lat <= MAX_LATITUDE);
        assert!(s.center_lng >= -180.0 && s.center_lng <= 180.0);
        assert_eq!(s.zoom, 22.0);
    }

    #[test]
    fn test_latlng_to_pixel_zero() {
        let p = latlng_to_pixel_absolute(0.0, 0.0, 0.0);
        assert!((p.x - 128.0).abs() < 0.01);
        assert!((p.y - 128.0).abs() < 0.01);
    }

    #[test]
    fn test_pixel_roundtrip() {
        let lat = 40.7128;
        let lng = -74.0060;
        let zoom = 10.0;
        let px = latlng_to_pixel_absolute(lat, lng, zoom);
        let ll = pixel_absolute_to_latlng(px.x, px.y, zoom);
        assert!((ll.lat - lat).abs() < 0.0001);
        assert!((ll.lng - lng).abs() < 0.0001);
    }

    #[test]
    fn test_viewport_pixel_roundtrip() {
        let state = MapState::new(51.505, -0.09, 13.0);
        let vp = Viewport::new(800, 600);
        let point = LatLng::new(51.51, -0.08);
        let px = latlng_to_viewport_pixel(&point, &state, &vp);
        let back = viewport_pixel_to_latlng(&px, &state, &vp);
        assert!((back.lat - point.lat).abs() < 0.0001);
        assert!((back.lng - point.lng).abs() < 0.0001);
    }

    #[test]
    fn test_visible_tiles_not_empty() {
        let state = MapState::new(0.0, 0.0, 2.0);
        let vp = Viewport::new(512, 512);
        let tiles = visible_tiles(&state, &vp);
        assert!(!tiles.is_empty());
        for t in &tiles {
            assert_eq!(t.z, 2);
            assert!(t.x < 4);
            assert!(t.y < 4);
        }
    }

    #[test]
    fn test_pan() {
        let state = MapState::new(0.0, 0.0, 4.0);
        let panned = pan(&state, 100.0, 0.0);
        // Panning right increases longitude
        assert!(panned.center_lng > state.center_lng);
        assert!((panned.center_lat - state.center_lat).abs() < 0.5);
    }

    #[test]
    fn test_zoom_at_center() {
        let state = MapState::new(40.0, -74.0, 10.0);
        let vp = Viewport::new(800, 600);
        let zoomed = zoom_at(&state, &vp, 400.0, 300.0, 12.0);
        assert_eq!(zoomed.zoom, 12.0);
        // Zooming at center should keep center roughly the same
        assert!((zoomed.center_lat - state.center_lat).abs() < 0.01);
        assert!((zoomed.center_lng - state.center_lng).abs() < 0.01);
    }

    #[test]
    fn test_visible_bounds() {
        let state = MapState::new(0.0, 0.0, 4.0);
        let vp = Viewport::new(800, 600);
        let bounds = visible_bounds(&state, &vp);
        assert!(bounds.sw.lat < 0.0);
        assert!(bounds.ne.lat > 0.0);
        assert!(bounds.sw.lng < 0.0);
        assert!(bounds.ne.lng > 0.0);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bb = BoundingBox {
            sw: LatLng::new(-10.0, -10.0),
            ne: LatLng::new(10.0, 10.0),
        };
        assert!(bb.contains(&LatLng::new(0.0, 0.0)));
        assert!(!bb.contains(&LatLng::new(20.0, 0.0)));
    }

    #[test]
    fn test_fly_to_endpoints() {
        let from = MapState::new(0.0, 0.0, 2.0);
        let to = MapState::new(40.0, -74.0, 12.0);
        let start = fly_to_interpolate(&from, &to, 0.0);
        assert!((start.center_lat - from.center_lat).abs() < 0.001);
        let end = fly_to_interpolate(&from, &to, 1.0);
        assert!((end.center_lat - to.center_lat).abs() < 0.001);
        assert!((end.zoom - to.zoom).abs() < 0.001);
    }

    #[test]
    fn test_fly_to_midpoint() {
        let from = MapState::new(0.0, 0.0, 4.0);
        let to = MapState::new(50.0, 50.0, 10.0);
        let mid = fly_to_interpolate(&from, &to, 0.5);
        // At t=0.5, eased value is 0.5 for cubic ease-in-out
        assert!(mid.center_lat > from.center_lat && mid.center_lat < to.center_lat);
        assert!(mid.zoom > from.zoom && mid.zoom < to.zoom);
    }

    #[test]
    fn test_wrap_longitude() {
        assert_eq!(wrap_lng(181.0), -179.0);
        assert_eq!(wrap_lng(-181.0), 179.0);
        assert_eq!(wrap_lng(0.0), 0.0);
    }

    #[test]
    fn test_lerp_angle_shortest_path() {
        // From 350 to 10 should go through 0, not through 180
        let result = lerp_angle(350.0, 10.0, 0.5);
        assert!((result - 0.0).abs() < 0.01 || (result - 360.0).abs() < 0.01);
    }
}
