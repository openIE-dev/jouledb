//! Region/server selection for optimal latency — ping, load balancing, failover.
//!
//! Replaces region-select.js / AWS-GameLift-latency with pure Rust.
//! Region with id/name/location/capacity, RegionSelector picks best region
//! based on player pings, simulated ping measurement, load balancing,
//! preference override, multi-region match (minimize total latency),
//! region health status, failover, geo-distance estimation.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionError {
    RegionNotFound(String),
    NoHealthyRegion,
    NoRegionsAvailable,
    DuplicateRegion(String),
    AllRegionsOverloaded,
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegionNotFound(id) => write!(f, "region not found: {id}"),
            Self::NoHealthyRegion => write!(f, "no healthy region available"),
            Self::NoRegionsAvailable => write!(f, "no regions available"),
            Self::DuplicateRegion(id) => write!(f, "duplicate region: {id}"),
            Self::AllRegionsOverloaded => write!(f, "all regions overloaded"),
        }
    }
}

impl std::error::Error for RegionError {}

// ── Health Status ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Down,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Degraded => write!(f, "Degraded"),
            Self::Down => write!(f, "Down"),
        }
    }
}

// ── GeoLocation ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoLocation {
    pub lat: f64,
    pub lon: f64,
}

impl GeoLocation {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Great-circle distance in kilometers using Haversine formula.
    pub fn distance_km(&self, other: &GeoLocation) -> f64 {
        let r = 6371.0; // Earth radius in km
        let d_lat = (other.lat - self.lat).to_radians();
        let d_lon = (other.lon - self.lon).to_radians();
        let a = (d_lat / 2.0).sin().powi(2)
            + self.lat.to_radians().cos()
                * other.lat.to_radians().cos()
                * (d_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();
        r * c
    }

    /// Estimated ping in ms based on distance (speed of light in fiber ~ 200km/ms round trip).
    pub fn estimated_ping_ms(&self, other: &GeoLocation) -> f64 {
        let dist = self.distance_km(other);
        // Round trip through fiber: ~2/3 speed of light, round trip = 2x.
        (dist / 100.0).max(1.0)
    }
}

impl fmt::Display for GeoLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.lat, self.lon)
    }
}

// ── Region ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Region {
    pub id: String,
    pub name: String,
    pub location: GeoLocation,
    pub capacity: usize,
    pub current_load: usize,
    pub health: HealthStatus,
}

impl Region {
    pub fn new(id: &str, name: &str, lat: f64, lon: f64, capacity: usize) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            location: GeoLocation::new(lat, lon),
            capacity,
            current_load: 0,
            health: HealthStatus::Healthy,
        }
    }

    pub fn load_factor(&self) -> f64 {
        if self.capacity == 0 {
            return 1.0;
        }
        self.current_load as f64 / self.capacity as f64
    }

    pub fn has_capacity(&self) -> bool {
        self.current_load < self.capacity
    }

    pub fn is_healthy(&self) -> bool {
        self.health == HealthStatus::Healthy
    }

    pub fn is_available(&self) -> bool {
        self.health != HealthStatus::Down && self.has_capacity()
    }
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({}, {:.0}% load, {})",
            self.name,
            self.id,
            self.load_factor() * 100.0,
            self.health
        )
    }
}

// ── Player Ping Data ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PlayerPings {
    pub player_id: String,
    pub pings: HashMap<String, f64>,
    pub location: Option<GeoLocation>,
    pub preferred_region: Option<String>,
}

impl PlayerPings {
    pub fn new(player_id: &str) -> Self {
        Self {
            player_id: player_id.to_string(),
            pings: HashMap::new(),
            location: None,
            preferred_region: None,
        }
    }

    pub fn with_ping(mut self, region_id: &str, ping_ms: f64) -> Self {
        self.pings.insert(region_id.to_string(), ping_ms);
        self
    }

    pub fn with_location(mut self, lat: f64, lon: f64) -> Self {
        self.location = Some(GeoLocation::new(lat, lon));
        self
    }

    pub fn with_preference(mut self, region_id: &str) -> Self {
        self.preferred_region = Some(region_id.to_string());
        self
    }

    pub fn best_region(&self) -> Option<(&str, f64)> {
        self.pings
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, &ping)| (id.as_str(), ping))
    }
}

// ── Region Selection Result ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SelectionResult {
    pub region_id: String,
    pub reason: String,
    pub estimated_ping_ms: f64,
    pub alternatives: Vec<(String, f64)>,
}

impl fmt::Display for SelectionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Selected {} ({:.0}ms, {})",
            self.region_id, self.estimated_ping_ms, self.reason
        )
    }
}

// ── Region Selector ─────────────────────────────────────────────

#[derive(Debug)]
pub struct RegionSelector {
    regions: HashMap<String, Region>,
    load_weight: f64,
    ping_weight: f64,
}

impl RegionSelector {
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
            load_weight: 0.3,
            ping_weight: 0.7,
        }
    }

    pub fn with_weights(mut self, ping: f64, load: f64) -> Self {
        let total = ping + load;
        self.ping_weight = ping / total;
        self.load_weight = load / total;
        self
    }

    pub fn add_region(&mut self, region: Region) -> Result<(), RegionError> {
        if self.regions.contains_key(&region.id) {
            return Err(RegionError::DuplicateRegion(region.id.clone()));
        }
        self.regions.insert(region.id.clone(), region);
        Ok(())
    }

    pub fn get_region(&self, id: &str) -> Result<&Region, RegionError> {
        self.regions
            .get(id)
            .ok_or_else(|| RegionError::RegionNotFound(id.to_string()))
    }

    pub fn get_region_mut(&mut self, id: &str) -> Result<&mut Region, RegionError> {
        self.regions
            .get_mut(id)
            .ok_or_else(|| RegionError::RegionNotFound(id.to_string()))
    }

    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    pub fn healthy_regions(&self) -> Vec<&Region> {
        self.regions.values().filter(|r| r.is_healthy()).collect()
    }

    pub fn available_regions(&self) -> Vec<&Region> {
        self.regions.values().filter(|r| r.is_available()).collect()
    }

    /// Select best region for a single player.
    pub fn select_for_player(&self, player: &PlayerPings) -> Result<SelectionResult, RegionError> {
        // Check preference override first.
        if let Some(ref pref) = player.preferred_region {
            if let Some(region) = self.regions.get(pref) {
                if region.is_available() {
                    let ping = player.pings.get(pref).copied().unwrap_or(50.0);
                    let alts = self.ranked_alternatives(player, pref);
                    return Ok(SelectionResult {
                        region_id: pref.clone(),
                        reason: "player preference".into(),
                        estimated_ping_ms: ping,
                        alternatives: alts,
                    });
                }
            }
        }

        self.select_by_score(player)
    }

    /// Select best region for a group of players (minimize total latency).
    pub fn select_for_group(&self, players: &[PlayerPings]) -> Result<SelectionResult, RegionError> {
        let available = self.available_regions();
        if available.is_empty() {
            return Err(RegionError::NoRegionsAvailable);
        }

        let mut best_region = None;
        let mut best_total_ping = f64::MAX;

        for region in &available {
            let total: f64 = players
                .iter()
                .map(|p| {
                    p.pings.get(&region.id).copied().unwrap_or_else(|| {
                        p.location
                            .map(|loc| loc.estimated_ping_ms(&region.location))
                            .unwrap_or(100.0)
                    })
                })
                .sum();
            if total < best_total_ping {
                best_total_ping = total;
                best_region = Some(region);
            }
        }

        let region = best_region.ok_or(RegionError::NoRegionsAvailable)?;
        let avg_ping = best_total_ping / players.len().max(1) as f64;
        let alts: Vec<(String, f64)> = available
            .iter()
            .filter(|r| r.id != region.id)
            .map(|r| {
                let total: f64 = players
                    .iter()
                    .map(|p| p.pings.get(&r.id).copied().unwrap_or(100.0))
                    .sum();
                (r.id.clone(), total / players.len().max(1) as f64)
            })
            .collect();

        Ok(SelectionResult {
            region_id: region.id.clone(),
            reason: "lowest group latency".into(),
            estimated_ping_ms: avg_ping,
            alternatives: alts,
        })
    }

    /// Failover: return next-best region if primary is down.
    pub fn failover(&self, current_id: &str, player: &PlayerPings) -> Result<SelectionResult, RegionError> {
        let available: Vec<&Region> = self
            .regions
            .values()
            .filter(|r| r.is_available() && r.id != current_id)
            .collect();
        if available.is_empty() {
            return Err(RegionError::NoHealthyRegion);
        }

        let best = available
            .iter()
            .min_by(|a, b| {
                let pa = player.pings.get(&a.id).copied().unwrap_or(999.0);
                let pb = player.pings.get(&b.id).copied().unwrap_or(999.0);
                pa.partial_cmp(&pb).unwrap()
            })
            .unwrap();

        let ping = player.pings.get(&best.id).copied().unwrap_or(100.0);
        Ok(SelectionResult {
            region_id: best.id.clone(),
            reason: format!("failover from {current_id}"),
            estimated_ping_ms: ping,
            alternatives: Vec::new(),
        })
    }

    /// Simulate ping measurement from geo-distance.
    pub fn simulate_pings(&self, location: &GeoLocation) -> HashMap<String, f64> {
        self.regions
            .values()
            .map(|r| (r.id.clone(), location.estimated_ping_ms(&r.location)))
            .collect()
    }

    fn select_by_score(&self, player: &PlayerPings) -> Result<SelectionResult, RegionError> {
        let available = self.available_regions();
        if available.is_empty() {
            return Err(RegionError::NoRegionsAvailable);
        }

        let max_ping = available
            .iter()
            .filter_map(|r| player.pings.get(&r.id))
            .cloned()
            .fold(1.0f64, f64::max);

        let mut scored: Vec<(&Region, f64, f64)> = available
            .iter()
            .map(|r| {
                let ping = player.pings.get(&r.id).copied().unwrap_or_else(|| {
                    player
                        .location
                        .map(|loc| loc.estimated_ping_ms(&r.location))
                        .unwrap_or(max_ping)
                });
                let ping_norm = ping / max_ping.max(1.0);
                let load_norm = r.load_factor();
                let score = self.ping_weight * ping_norm + self.load_weight * load_norm;
                (*r, score, ping)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (best, _, ping) = &scored[0];
        let alts = scored[1..]
            .iter()
            .map(|(r, _, p)| (r.id.clone(), *p))
            .collect();

        Ok(SelectionResult {
            region_id: best.id.clone(),
            reason: "lowest combined score".into(),
            estimated_ping_ms: *ping,
            alternatives: alts,
        })
    }

    fn ranked_alternatives(&self, player: &PlayerPings, exclude: &str) -> Vec<(String, f64)> {
        let mut alts: Vec<(String, f64)> = self
            .regions
            .values()
            .filter(|r| r.is_available() && r.id != exclude)
            .map(|r| {
                let ping = player.pings.get(&r.id).copied().unwrap_or(100.0);
                (r.id.clone(), ping)
            })
            .collect();
        alts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        alts
    }
}

impl Default for RegionSelector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> RegionSelector {
        let mut sel = RegionSelector::new();
        sel.add_region(Region::new("us-east", "US East", 39.0, -77.0, 1000)).unwrap();
        sel.add_region(Region::new("us-west", "US West", 37.0, -122.0, 1000)).unwrap();
        sel.add_region(Region::new("eu-west", "EU West", 51.0, -0.1, 1000)).unwrap();
        sel
    }

    fn player_east() -> PlayerPings {
        PlayerPings::new("alice")
            .with_ping("us-east", 20.0)
            .with_ping("us-west", 80.0)
            .with_ping("eu-west", 120.0)
    }

    #[test]
    fn select_lowest_ping() {
        let sel = setup();
        let result = sel.select_for_player(&player_east()).unwrap();
        assert_eq!(result.region_id, "us-east");
    }

    #[test]
    fn preference_override() {
        let sel = setup();
        let player = player_east().with_preference("us-west");
        let result = sel.select_for_player(&player).unwrap();
        assert_eq!(result.region_id, "us-west");
        assert_eq!(result.reason, "player preference");
    }

    #[test]
    fn preference_fallback_when_down() {
        let mut sel = setup();
        sel.get_region_mut("us-west").unwrap().health = HealthStatus::Down;
        let player = player_east().with_preference("us-west");
        let result = sel.select_for_player(&player).unwrap();
        assert_ne!(result.region_id, "us-west");
    }

    #[test]
    fn group_selection() {
        let sel = setup();
        let p1 = PlayerPings::new("a").with_ping("us-east", 20.0).with_ping("us-west", 80.0);
        let p2 = PlayerPings::new("b").with_ping("us-east", 30.0).with_ping("us-west", 70.0);
        let result = sel.select_for_group(&[p1, p2]).unwrap();
        assert_eq!(result.region_id, "us-east");
    }

    #[test]
    fn failover() {
        let sel = setup();
        let player = player_east();
        let result = sel.failover("us-east", &player).unwrap();
        assert_eq!(result.region_id, "us-west");
        assert!(result.reason.contains("failover"));
    }

    #[test]
    fn no_regions_error() {
        let sel = RegionSelector::new();
        let player = PlayerPings::new("a");
        let err = sel.select_for_player(&player).unwrap_err();
        assert!(matches!(err, RegionError::NoRegionsAvailable));
    }

    #[test]
    fn all_down_failover_error() {
        let mut sel = setup();
        for r in sel.regions.values_mut() {
            r.health = HealthStatus::Down;
        }
        let player = player_east();
        let err = sel.failover("us-east", &player).unwrap_err();
        assert!(matches!(err, RegionError::NoHealthyRegion));
    }

    #[test]
    fn load_balancing_effect() {
        let mut sel = RegionSelector::new().with_weights(0.5, 0.5);
        sel.add_region(Region::new("r1", "R1", 0.0, 0.0, 100)).unwrap();
        sel.add_region(Region::new("r2", "R2", 0.0, 0.0, 100)).unwrap();
        sel.get_region_mut("r1").unwrap().current_load = 95;
        let player = PlayerPings::new("a")
            .with_ping("r1", 10.0)
            .with_ping("r2", 15.0);
        let result = sel.select_for_player(&player).unwrap();
        // Despite higher ping, r2 should win due to load.
        assert_eq!(result.region_id, "r2");
    }

    #[test]
    fn geo_distance() {
        let nyc = GeoLocation::new(40.7128, -74.0060);
        let london = GeoLocation::new(51.5074, -0.1278);
        let dist = nyc.distance_km(&london);
        assert!((dist - 5570.0).abs() < 50.0);
    }

    #[test]
    fn geo_distance_same_point() {
        let p = GeoLocation::new(0.0, 0.0);
        assert!(p.distance_km(&p) < 0.01);
    }

    #[test]
    fn estimated_ping_from_distance() {
        let nyc = GeoLocation::new(40.7128, -74.0060);
        let london = GeoLocation::new(51.5074, -0.1278);
        let ping = nyc.estimated_ping_ms(&london);
        assert!(ping > 30.0 && ping < 100.0);
    }

    #[test]
    fn simulate_pings() {
        let sel = setup();
        let loc = GeoLocation::new(39.0, -77.0); // Near US East
        let pings = sel.simulate_pings(&loc);
        assert!(pings["us-east"] < pings["eu-west"]);
    }

    #[test]
    fn region_load_factor() {
        let mut r = Region::new("r1", "R1", 0.0, 0.0, 100);
        r.current_load = 50;
        assert!((r.load_factor() - 0.5).abs() < 0.01);
    }

    #[test]
    fn region_availability() {
        let mut r = Region::new("r1", "R1", 0.0, 0.0, 100);
        assert!(r.is_available());
        r.health = HealthStatus::Down;
        assert!(!r.is_available());
        r.health = HealthStatus::Healthy;
        r.current_load = 100;
        assert!(!r.is_available());
    }

    #[test]
    fn duplicate_region_error() {
        let mut sel = setup();
        let err = sel.add_region(Region::new("us-east", "dup", 0.0, 0.0, 1)).unwrap_err();
        assert!(matches!(err, RegionError::DuplicateRegion(_)));
    }

    #[test]
    fn display_region() {
        let r = Region::new("us-east", "US East", 39.0, -77.0, 100);
        let s = r.to_string();
        assert!(s.contains("US East"));
        assert!(s.contains("Healthy"));
    }

    #[test]
    fn display_selection() {
        let result = SelectionResult {
            region_id: "us-east".into(),
            reason: "test".into(),
            estimated_ping_ms: 25.0,
            alternatives: vec![],
        };
        assert!(result.to_string().contains("us-east"));
    }

    #[test]
    fn healthy_regions_filter() {
        let mut sel = setup();
        sel.get_region_mut("eu-west").unwrap().health = HealthStatus::Down;
        assert_eq!(sel.healthy_regions().len(), 2);
    }

    #[test]
    fn alternatives_listed() {
        let sel = setup();
        let result = sel.select_for_player(&player_east()).unwrap();
        assert_eq!(result.alternatives.len(), 2);
    }
}
