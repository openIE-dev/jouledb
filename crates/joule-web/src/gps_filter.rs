//! GPS filtering — WGS84/UTM coordinate transforms, Kalman smoothing for
//! position/velocity estimation, HDOP-based measurement weighting, and
//! waypoint management with arrival detection.
//!
//! Pure-Rust GPS processing for outdoor robotic navigation, suitable for
//! embedded systems without external dependencies.

use std::f64::consts::PI;
use std::fmt;

// ── WGS84 Constants ─────────────────────────────────────────────

/// WGS84 ellipsoid semi-major axis in meters.
const WGS84_A: f64 = 6_378_137.0;
/// WGS84 ellipsoid flattening.
const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// WGS84 first eccentricity squared.
const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
/// UTM scale factor at the central meridian.
const UTM_K0: f64 = 0.9996;
/// UTM false easting in meters.
const UTM_FE: f64 = 500_000.0;
/// UTM false northing for southern hemisphere in meters.
const UTM_FN_SOUTH: f64 = 10_000_000.0;

// ── Coordinate Types ────────────────────────────────────────────

/// Geographic coordinate (latitude, longitude, altitude).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCoord {
    /// Latitude in degrees (positive = North).
    pub lat_deg: f64,
    /// Longitude in degrees (positive = East).
    pub lon_deg: f64,
    /// Altitude above WGS84 ellipsoid in meters.
    pub alt_m: f64,
}

impl GeoCoord {
    pub fn new(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Self {
        Self { lat_deg, lon_deg, alt_m }
    }

    pub fn lat_rad(&self) -> f64 {
        self.lat_deg.to_radians()
    }

    pub fn lon_rad(&self) -> f64 {
        self.lon_deg.to_radians()
    }

    /// Haversine distance to another coordinate in meters (ignoring altitude).
    pub fn haversine_distance(&self, other: &GeoCoord) -> f64 {
        let dlat = (other.lat_deg - self.lat_deg).to_radians();
        let dlon = (other.lon_deg - self.lon_deg).to_radians();
        let lat1 = self.lat_rad();
        let lat2 = other.lat_rad();
        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();
        WGS84_A * c
    }

    /// Bearing to another coordinate in radians (0 = North, clockwise).
    pub fn bearing_to(&self, other: &GeoCoord) -> f64 {
        let lat1 = self.lat_rad();
        let lat2 = other.lat_rad();
        let dlon = (other.lon_deg - self.lon_deg).to_radians();
        let y = dlon.sin() * lat2.cos();
        let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
        let bearing = y.atan2(x);
        (bearing + 2.0 * PI) % (2.0 * PI)
    }

    /// UTM zone number for this coordinate.
    pub fn utm_zone(&self) -> u8 {
        ((self.lon_deg + 180.0) / 6.0).floor() as u8 + 1
    }
}

impl fmt::Display for GeoCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ns = if self.lat_deg >= 0.0 { 'N' } else { 'S' };
        let ew = if self.lon_deg >= 0.0 { 'E' } else { 'W' };
        write!(
            f,
            "{:.6}{}{}, {:.6}{}{}, alt={:.1}m",
            self.lat_deg.abs(), '\u{00B0}', ns,
            self.lon_deg.abs(), '\u{00B0}', ew,
            self.alt_m,
        )
    }
}

/// UTM coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtmCoord {
    pub easting: f64,
    pub northing: f64,
    pub zone: u8,
    pub northern_hemisphere: bool,
    pub alt_m: f64,
}

impl UtmCoord {
    pub fn new(easting: f64, northing: f64, zone: u8, northern: bool) -> Self {
        Self { easting, northing, zone, northern_hemisphere: northern, alt_m: 0.0 }
    }

    pub fn with_altitude(mut self, alt: f64) -> Self {
        self.alt_m = alt;
        self
    }

    /// Distance to another UTM coordinate (same zone only).
    pub fn distance_to(&self, other: &UtmCoord) -> f64 {
        let de = other.easting - self.easting;
        let dn = other.northing - self.northing;
        (de * de + dn * dn).sqrt()
    }
}

impl fmt::Display for UtmCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hem = if self.northern_hemisphere { 'N' } else { 'S' };
        write!(
            f,
            "UTM {}{}  E={:.2}  N={:.2}",
            self.zone, hem, self.easting, self.northing,
        )
    }
}

// ── Coordinate Transforms ───────────────────────────────────────

/// Convert WGS84 geographic coordinates to UTM.
pub fn geo_to_utm(coord: &GeoCoord) -> UtmCoord {
    let lat = coord.lat_rad();
    let lon = coord.lon_rad();
    let zone = coord.utm_zone();
    let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();

    let n = WGS84_A / (1.0 - WGS84_E2 * lat.sin().powi(2)).sqrt();
    let t = lat.tan();
    let c = WGS84_E2 * lat.cos().powi(2) / (1.0 - WGS84_E2);
    let a_coeff = (lon - lon0) * lat.cos();

    // Meridional arc length
    let e2 = WGS84_E2;
    let e4 = e2 * e2;
    let e6 = e4 * e2;
    let m = WGS84_A * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * lat).sin()
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * lat).sin()
        - (35.0 * e6 / 3072.0) * (6.0 * lat).sin()
    );

    let t2 = t * t;
    let a2 = a_coeff * a_coeff;

    let easting = UTM_K0 * n * (
        a_coeff
        + (1.0 - t2 + c) * a2 * a_coeff / 6.0
        + (5.0 - 18.0 * t2 + t2 * t2 + 72.0 * c - 58.0 * WGS84_E2 / (1.0 - WGS84_E2))
            * a2 * a2 * a_coeff / 120.0
    ) + UTM_FE;

    let northing_raw = UTM_K0 * (
        m + n * t * (
            a2 / 2.0
            + (5.0 - t2 + 9.0 * c + 4.0 * c * c) * a2 * a2 / 24.0
            + (61.0 - 58.0 * t2 + t2 * t2 + 600.0 * c - 330.0 * WGS84_E2 / (1.0 - WGS84_E2))
                * a2 * a2 * a2 / 720.0
        )
    );

    let northern = coord.lat_deg >= 0.0;
    let northing = if northern { northing_raw } else { northing_raw + UTM_FN_SOUTH };

    UtmCoord {
        easting,
        northing,
        zone,
        northern_hemisphere: northern,
        alt_m: coord.alt_m,
    }
}

/// Convert UTM coordinates back to WGS84 (approximate inverse).
pub fn utm_to_geo(utm: &UtmCoord) -> GeoCoord {
    let x = utm.easting - UTM_FE;
    let y = if utm.northern_hemisphere { utm.northing } else { utm.northing - UTM_FN_SOUTH };

    let m = y / UTM_K0;
    let mu = m / (WGS84_A * (1.0 - WGS84_E2 / 4.0 - 3.0 * WGS84_E2.powi(2) / 64.0
        - 5.0 * WGS84_E2.powi(3) / 256.0));

    let e1 = (1.0 - (1.0 - WGS84_E2).sqrt()) / (1.0 + (1.0 - WGS84_E2).sqrt());
    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin();

    let n1 = WGS84_A / (1.0 - WGS84_E2 * phi1.sin().powi(2)).sqrt();
    let t1 = phi1.tan();
    let c1 = WGS84_E2 * phi1.cos().powi(2) / (1.0 - WGS84_E2);
    let r1 = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * phi1.sin().powi(2)).powf(1.5);
    let d = x / (n1 * UTM_K0);
    let d2 = d * d;
    let t12 = t1 * t1;

    let lat = phi1 - (n1 * t1 / r1) * (
        d2 / 2.0
        - (5.0 + 3.0 * t12 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * WGS84_E2 / (1.0 - WGS84_E2)) * d2 * d2 / 24.0
        + (61.0 + 90.0 * t12 + 298.0 * c1 + 45.0 * t12 * t12
            - 252.0 * WGS84_E2 / (1.0 - WGS84_E2) - 3.0 * c1 * c1) * d2 * d2 * d2 / 720.0
    );

    let lon0 = ((utm.zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
    let lon = lon0 + (
        d - (1.0 + 2.0 * t12 + c1) * d2 * d / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t12 - 3.0 * c1 * c1
            + 8.0 * WGS84_E2 / (1.0 - WGS84_E2) + 24.0 * t12 * t12) * d2 * d2 * d / 120.0
    ) / phi1.cos();

    GeoCoord::new(lat.to_degrees(), lon.to_degrees(), utm.alt_m)
}

// ── GPS Fix ─────────────────────────────────────────────────────

/// GPS fix quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixQuality {
    NoFix,
    Gps,
    Dgps,
    Rtk,
    FloatRtk,
}

/// A single GPS fix with quality metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpsFix {
    pub coord: GeoCoord,
    pub hdop: f64,
    pub vdop: f64,
    pub num_satellites: u8,
    pub quality: FixQuality,
    pub speed_mps: f64,
    pub heading_deg: f64,
    pub timestamp_s: f64,
}

impl GpsFix {
    pub fn new(coord: GeoCoord, timestamp_s: f64) -> Self {
        Self {
            coord,
            hdop: 1.0,
            vdop: 1.5,
            num_satellites: 0,
            quality: FixQuality::Gps,
            speed_mps: 0.0,
            heading_deg: 0.0,
            timestamp_s,
        }
    }

    pub fn with_hdop(mut self, hdop: f64) -> Self {
        self.hdop = hdop;
        self
    }

    pub fn with_quality(mut self, quality: FixQuality) -> Self {
        self.quality = quality;
        self
    }

    pub fn with_speed(mut self, speed_mps: f64) -> Self {
        self.speed_mps = speed_mps;
        self
    }

    pub fn with_heading(mut self, heading_deg: f64) -> Self {
        self.heading_deg = heading_deg;
        self
    }

    /// Horizontal accuracy estimate in meters (CEP ~= HDOP * UERE).
    pub fn horizontal_accuracy_m(&self) -> f64 {
        self.hdop * 3.0 // Typical UERE ~3m for civilian GPS
    }
}

impl fmt::Display for GpsFix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fix({}, hdop={:.1}, sats={}, {:?})",
            self.coord, self.hdop, self.num_satellites, self.quality,
        )
    }
}

// ── GPS Kalman Filter ───────────────────────────────────────────

/// 2D Kalman filter for GPS position/velocity smoothing.
/// State: [x, y, vx, vy]
#[derive(Debug, Clone)]
pub struct GpsKalman {
    /// State vector: [x, y, vx, vy] in UTM meters.
    pub state: [f64; 4],
    /// State covariance (4x4 upper triangle stored as flat array).
    pub covariance: [[f64; 4]; 4],
    /// Process noise (acceleration variance in m^2/s^4).
    pub process_noise: f64,
    /// Base measurement noise in m^2 (scaled by HDOP).
    pub base_measurement_noise: f64,
    pub last_time: f64,
    pub initialized: bool,
    pub utm_zone: u8,
    pub northern: bool,
}

impl GpsKalman {
    pub fn new() -> Self {
        Self {
            state: [0.0; 4],
            covariance: [
                [100.0, 0.0, 0.0, 0.0],
                [0.0, 100.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0, 10.0],
            ],
            process_noise: 0.5,
            base_measurement_noise: 9.0,
            last_time: 0.0,
            initialized: false,
            utm_zone: 0,
            northern: true,
        }
    }

    pub fn with_process_noise(mut self, noise: f64) -> Self {
        self.process_noise = noise;
        self
    }

    pub fn with_measurement_noise(mut self, noise: f64) -> Self {
        self.base_measurement_noise = noise;
        self
    }

    /// Update the filter with a new GPS fix.
    pub fn update(&mut self, fix: &GpsFix) {
        let utm = geo_to_utm(&fix.coord);

        if !self.initialized {
            self.state = [utm.easting, utm.northing, 0.0, 0.0];
            self.last_time = fix.timestamp_s;
            self.utm_zone = utm.zone;
            self.northern = utm.northern_hemisphere;
            self.initialized = true;
            return;
        }

        let dt = fix.timestamp_s - self.last_time;
        if dt <= 0.0 {
            return;
        }

        // Predict step: F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
        self.state[0] += self.state[2] * dt;
        self.state[1] += self.state[3] * dt;

        // Predict covariance: P = F*P*F' + Q
        let q = self.process_noise * dt;
        let p = &mut self.covariance;
        // Propagate position rows by velocity
        p[0][0] += 2.0 * dt * p[0][2] + dt * dt * p[2][2] + q;
        p[0][1] += dt * p[0][3] + dt * p[2][1];
        p[1][1] += 2.0 * dt * p[1][3] + dt * dt * p[3][3] + q;
        p[1][0] = p[0][1];
        p[0][2] += dt * p[2][2];
        p[0][3] += dt * p[2][3];
        p[1][2] += dt * p[3][2];
        p[1][3] += dt * p[3][3];
        p[2][0] = p[0][2];
        p[3][0] = p[0][3];
        p[2][1] = p[1][2];
        p[3][1] = p[1][3];
        p[2][2] += q * 0.1;
        p[3][3] += q * 0.1;

        // Update step: measurement z = [easting, northing]
        let r = self.base_measurement_noise * fix.hdop * fix.hdop;
        let z = [utm.easting, utm.northing];

        // Innovation: y = z - H*x (H = [[1,0,0,0],[0,1,0,0]])
        let y = [z[0] - self.state[0], z[1] - self.state[1]];

        // Innovation covariance: S = H*P*H' + R
        let s00 = p[0][0] + r;
        let s01 = p[0][1];
        let s10 = p[1][0];
        let s11 = p[1][1] + r;

        // S inverse (2x2)
        let det = s00 * s11 - s01 * s10;
        if det.abs() < 1e-15 {
            self.last_time = fix.timestamp_s;
            return;
        }
        let si00 = s11 / det;
        let si01 = -s01 / det;
        let si10 = -s10 / det;
        let si11 = s00 / det;

        // Kalman gain: K = P*H'*S^-1 (4x2)
        let k = [
            [p[0][0] * si00 + p[0][1] * si10, p[0][0] * si01 + p[0][1] * si11],
            [p[1][0] * si00 + p[1][1] * si10, p[1][0] * si01 + p[1][1] * si11],
            [p[2][0] * si00 + p[2][1] * si10, p[2][0] * si01 + p[2][1] * si11],
            [p[3][0] * si00 + p[3][1] * si10, p[3][0] * si01 + p[3][1] * si11],
        ];

        // State update: x = x + K*y
        for i in 0..4 {
            self.state[i] += k[i][0] * y[0] + k[i][1] * y[1];
        }

        // Covariance update: P = (I - K*H)*P
        let mut new_p = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut ikh = 0.0;
                // (I - K*H)[i][j]
                if i == j { ikh = 1.0; }
                if j < 2 { ikh -= k[i][j]; }
                new_p[i][j] = 0.0;
                for m in 0..4 {
                    let mut ikm = 0.0;
                    if i == m { ikm = 1.0; }
                    if m < 2 { ikm -= k[i][m]; }
                    new_p[i][j] += ikm * p[m][j];
                }
            }
        }
        self.covariance = new_p;
        self.last_time = fix.timestamp_s;
    }

    /// Get smoothed position as a GeoCoord.
    pub fn position(&self) -> GeoCoord {
        let utm = UtmCoord {
            easting: self.state[0],
            northing: self.state[1],
            zone: self.utm_zone,
            northern_hemisphere: self.northern,
            alt_m: 0.0,
        };
        utm_to_geo(&utm)
    }

    /// Get estimated velocity (east, north) in m/s.
    pub fn velocity(&self) -> (f64, f64) {
        (self.state[2], self.state[3])
    }

    /// Get estimated speed in m/s.
    pub fn speed(&self) -> f64 {
        (self.state[2] * self.state[2] + self.state[3] * self.state[3]).sqrt()
    }
}

impl fmt::Display for GpsKalman {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GpsKalman(speed={:.2}m/s, zone={})",
            self.speed(), self.utm_zone,
        )
    }
}

// ── Waypoint Manager ────────────────────────────────────────────

/// A navigation waypoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waypoint {
    pub coord: GeoCoord,
    pub radius_m: f64,
    pub id: usize,
}

/// Waypoint manager with arrival detection.
#[derive(Debug, Clone)]
pub struct WaypointManager {
    pub waypoints: Vec<Waypoint>,
    pub current_index: usize,
    pub loop_mode: bool,
}

impl WaypointManager {
    pub fn new() -> Self {
        Self { waypoints: Vec::new(), current_index: 0, loop_mode: false }
    }

    pub fn with_loop(mut self, looping: bool) -> Self {
        self.loop_mode = looping;
        self
    }

    pub fn add_waypoint(&mut self, coord: GeoCoord, radius_m: f64) {
        let id = self.waypoints.len();
        self.waypoints.push(Waypoint { coord, radius_m, id });
    }

    pub fn current_waypoint(&self) -> Option<&Waypoint> {
        self.waypoints.get(self.current_index)
    }

    /// Check if the current position has arrived at the active waypoint.
    /// If so, advance to the next waypoint. Returns true if arrived.
    pub fn check_arrival(&mut self, position: &GeoCoord) -> bool {
        if let Some(wp) = self.waypoints.get(self.current_index) {
            let dist = position.haversine_distance(&wp.coord);
            if dist <= wp.radius_m {
                self.current_index += 1;
                if self.loop_mode && self.current_index >= self.waypoints.len() {
                    self.current_index = 0;
                }
                return true;
            }
        }
        false
    }

    /// Distance to current waypoint from given position.
    pub fn distance_to_current(&self, position: &GeoCoord) -> Option<f64> {
        self.current_waypoint().map(|wp| position.haversine_distance(&wp.coord))
    }

    /// Bearing to current waypoint from given position (radians).
    pub fn bearing_to_current(&self, position: &GeoCoord) -> Option<f64> {
        self.current_waypoint().map(|wp| position.bearing_to(&wp.coord))
    }

    pub fn is_complete(&self) -> bool {
        !self.loop_mode && self.current_index >= self.waypoints.len()
    }

    pub fn reset(&mut self) {
        self.current_index = 0;
    }
}

impl fmt::Display for WaypointManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WaypointMgr(wps={}, current={}, loop={})",
            self.waypoints.len(), self.current_index, self.loop_mode,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_same_point() {
        let p = GeoCoord::new(34.0, -118.0, 0.0);
        assert!(p.haversine_distance(&p) < 1e-6);
    }

    #[test]
    fn test_haversine_known_distance() {
        // New York to Los Angeles: ~3944 km
        let nyc = GeoCoord::new(40.7128, -74.0060, 0.0);
        let lax = GeoCoord::new(34.0522, -118.2437, 0.0);
        let dist = nyc.haversine_distance(&lax);
        assert!((dist - 3_944_000.0).abs() < 50_000.0); // Within 50km
    }

    #[test]
    fn test_bearing_east() {
        let a = GeoCoord::new(0.0, 0.0, 0.0);
        let b = GeoCoord::new(0.0, 1.0, 0.0);
        let bearing = a.bearing_to(&b);
        assert!((bearing - PI / 2.0).abs() < 0.01);
    }

    #[test]
    fn test_bearing_north() {
        let a = GeoCoord::new(0.0, 0.0, 0.0);
        let b = GeoCoord::new(1.0, 0.0, 0.0);
        let bearing = a.bearing_to(&b);
        assert!(bearing.abs() < 0.01 || (bearing - 2.0 * PI).abs() < 0.01);
    }

    #[test]
    fn test_utm_zone() {
        let p = GeoCoord::new(34.0, -118.0, 0.0);
        assert_eq!(p.utm_zone(), 11);
    }

    #[test]
    fn test_geo_coord_display() {
        let p = GeoCoord::new(34.0522, -118.2437, 100.0);
        let s = format!("{p}");
        assert!(s.contains("N"));
        assert!(s.contains("W"));
    }

    #[test]
    fn test_utm_roundtrip() {
        let original = GeoCoord::new(34.0522, -118.2437, 50.0);
        let utm = geo_to_utm(&original);
        let recovered = utm_to_geo(&utm);
        assert!((recovered.lat_deg - original.lat_deg).abs() < 0.0001);
        assert!((recovered.lon_deg - original.lon_deg).abs() < 0.0001);
    }

    #[test]
    fn test_utm_coord_display() {
        let utm = UtmCoord::new(500000.0, 3762000.0, 11, true);
        let s = format!("{utm}");
        assert!(s.contains("UTM"));
        assert!(s.contains("11N"));
    }

    #[test]
    fn test_utm_distance() {
        let a = UtmCoord::new(500000.0, 3762000.0, 11, true);
        let b = UtmCoord::new(500100.0, 3762000.0, 11, true);
        assert!((a.distance_to(&b) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_gps_fix_accuracy() {
        let fix = GpsFix::new(GeoCoord::new(34.0, -118.0, 0.0), 0.0)
            .with_hdop(1.0);
        assert!((fix.horizontal_accuracy_m() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_gps_fix_display() {
        let fix = GpsFix::new(GeoCoord::new(34.0, -118.0, 0.0), 0.0);
        let s = format!("{fix}");
        assert!(s.contains("Fix"));
    }

    #[test]
    fn test_kalman_initialization() {
        let mut kf = GpsKalman::new();
        let fix = GpsFix::new(GeoCoord::new(34.0, -118.0, 0.0), 0.0);
        kf.update(&fix);
        assert!(kf.initialized);
    }

    #[test]
    fn test_kalman_stationary() {
        let mut kf = GpsKalman::new();
        let coord = GeoCoord::new(34.0522, -118.2437, 0.0);
        for i in 0..20 {
            let fix = GpsFix::new(coord, i as f64).with_hdop(1.0);
            kf.update(&fix);
        }
        assert!(kf.speed() < 1.0);
    }

    #[test]
    fn test_kalman_display() {
        let kf = GpsKalman::new();
        let s = format!("{kf}");
        assert!(s.contains("GpsKalman"));
    }

    #[test]
    fn test_waypoint_arrival() {
        let mut mgr = WaypointManager::new();
        mgr.add_waypoint(GeoCoord::new(34.0, -118.0, 0.0), 10.0);
        mgr.add_waypoint(GeoCoord::new(34.001, -118.0, 0.0), 10.0);

        let arrived = mgr.check_arrival(&GeoCoord::new(34.0, -118.0, 0.0));
        assert!(arrived);
        assert_eq!(mgr.current_index, 1);
    }

    #[test]
    fn test_waypoint_loop() {
        let mut mgr = WaypointManager::new().with_loop(true);
        mgr.add_waypoint(GeoCoord::new(34.0, -118.0, 0.0), 100.0);

        mgr.check_arrival(&GeoCoord::new(34.0, -118.0, 0.0));
        assert_eq!(mgr.current_index, 0); // Wrapped around
    }

    #[test]
    fn test_waypoint_distance() {
        let mut mgr = WaypointManager::new();
        mgr.add_waypoint(GeoCoord::new(34.0, -118.0, 0.0), 10.0);
        let dist = mgr.distance_to_current(&GeoCoord::new(34.0, -118.0, 0.0)).unwrap();
        assert!(dist < 1.0);
    }

    #[test]
    fn test_waypoint_display() {
        let mgr = WaypointManager::new();
        let s = format!("{mgr}");
        assert!(s.contains("WaypointMgr"));
    }

    #[test]
    fn test_waypoint_complete() {
        let mut mgr = WaypointManager::new();
        mgr.add_waypoint(GeoCoord::new(34.0, -118.0, 0.0), 100.0);
        mgr.check_arrival(&GeoCoord::new(34.0, -118.0, 0.0));
        assert!(mgr.is_complete());
    }
}
