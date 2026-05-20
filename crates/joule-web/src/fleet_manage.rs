//! # Fleet Management
//!
//! Fleet management system for multi-robot coordination. Handles vehicle
//! routing, dispatch scheduling, battery/fuel management, status monitoring,
//! and fleet rebalancing for autonomous vehicle fleets.

use std::fmt;
use std::collections::HashMap;

// ── Core Types ──

/// Status of a vehicle in the fleet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VehicleStatus {
    Idle,
    EnRoute,
    OnTask,
    Charging,
    Maintenance,
    Offline,
}

impl fmt::Display for VehicleStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::EnRoute => write!(f, "EnRoute"),
            Self::OnTask => write!(f, "OnTask"),
            Self::Charging => write!(f, "Charging"),
            Self::Maintenance => write!(f, "Maintenance"),
            Self::Offline => write!(f, "Offline"),
        }
    }
}

/// A vehicle in the fleet.
#[derive(Clone, Debug)]
pub struct Vehicle {
    pub id: usize,
    pub name: String,
    pub position: (f64, f64),
    pub status: VehicleStatus,
    pub battery_level: f64,
    pub max_battery: f64,
    pub speed: f64,
    pub current_task: Option<usize>,
    pub tasks_completed: usize,
    pub total_distance: f64,
}

impl Vehicle {
    pub fn new(id: usize, name: &str, position: (f64, f64)) -> Self {
        Self {
            id, name: name.to_string(), position,
            status: VehicleStatus::Idle,
            battery_level: 100.0, max_battery: 100.0,
            speed: 1.0, current_task: None,
            tasks_completed: 0, total_distance: 0.0,
        }
    }

    pub fn with_battery(mut self, level: f64, max: f64) -> Self {
        self.battery_level = level;
        self.max_battery = max;
        self
    }

    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    pub fn battery_percent(&self) -> f64 {
        if self.max_battery > 0.0 {
            (self.battery_level / self.max_battery) * 100.0
        } else {
            0.0
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self.status, VehicleStatus::Idle) && self.battery_level > 10.0
    }

    pub fn distance_to(&self, target: (f64, f64)) -> f64 {
        let dx = self.position.0 - target.0;
        let dy = self.position.1 - target.1;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn time_to(&self, target: (f64, f64)) -> f64 {
        if self.speed > 0.0 {
            self.distance_to(target) / self.speed
        } else {
            f64::MAX
        }
    }

    pub fn move_towards(&mut self, target: (f64, f64), dt: f64) -> bool {
        let dist = self.distance_to(target);
        let step = self.speed * dt;
        if step >= dist {
            self.total_distance += dist;
            self.position = target;
            true
        } else {
            let ratio = step / dist;
            let dx = target.0 - self.position.0;
            let dy = target.1 - self.position.1;
            self.position.0 += dx * ratio;
            self.position.1 += dy * ratio;
            self.total_distance += step;
            // Drain battery proportional to distance
            self.battery_level = (self.battery_level - step * 0.1).max(0.0);
            false
        }
    }
}

impl fmt::Display for Vehicle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vehicle({}, \"{}\", {}, bat={:.0}%)",
            self.id, self.name, self.status, self.battery_percent())
    }
}

// ── Dispatch Request ──

/// A dispatch request / delivery task.
#[derive(Clone, Debug)]
pub struct DispatchRequest {
    pub id: usize,
    pub pickup: (f64, f64),
    pub dropoff: (f64, f64),
    pub priority: f64,
    pub deadline: f64,
    pub assigned_vehicle: Option<usize>,
    pub completed: bool,
}

impl DispatchRequest {
    pub fn new(id: usize, pickup: (f64, f64), dropoff: (f64, f64)) -> Self {
        Self {
            id, pickup, dropoff,
            priority: 1.0, deadline: f64::MAX,
            assigned_vehicle: None, completed: false,
        }
    }

    pub fn with_priority(mut self, p: f64) -> Self {
        self.priority = p;
        self
    }

    pub fn with_deadline(mut self, d: f64) -> Self {
        self.deadline = d;
        self
    }

    pub fn trip_distance(&self) -> f64 {
        let dx = self.dropoff.0 - self.pickup.0;
        let dy = self.dropoff.1 - self.pickup.1;
        (dx * dx + dy * dy).sqrt()
    }
}

impl fmt::Display for DispatchRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Request({}, pri={:.1}, dist={:.2})", self.id, self.priority, self.trip_distance())
    }
}

// ── Fleet Manager ──

/// Central fleet management system.
#[derive(Clone, Debug)]
pub struct FleetManager {
    vehicles: Vec<Vehicle>,
    requests: Vec<DispatchRequest>,
    charging_stations: Vec<(f64, f64)>,
    low_battery_threshold: f64,
    dispatch_strategy: DispatchStrategy,
}

/// Strategy for dispatching vehicles to requests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DispatchStrategy {
    NearestFirst,
    HighestBattery,
    FastestEta,
}

impl fmt::Display for DispatchStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NearestFirst => write!(f, "NearestFirst"),
            Self::HighestBattery => write!(f, "HighestBattery"),
            Self::FastestEta => write!(f, "FastestEta"),
        }
    }
}

impl FleetManager {
    pub fn new() -> Self {
        Self {
            vehicles: Vec::new(),
            requests: Vec::new(),
            charging_stations: Vec::new(),
            low_battery_threshold: 20.0,
            dispatch_strategy: DispatchStrategy::NearestFirst,
        }
    }

    pub fn with_strategy(mut self, strategy: DispatchStrategy) -> Self {
        self.dispatch_strategy = strategy;
        self
    }

    pub fn with_low_battery_threshold(mut self, threshold: f64) -> Self {
        self.low_battery_threshold = threshold;
        self
    }

    pub fn add_vehicle(&mut self, vehicle: Vehicle) {
        self.vehicles.push(vehicle);
    }

    pub fn add_request(&mut self, request: DispatchRequest) {
        self.requests.push(request);
    }

    pub fn add_charging_station(&mut self, pos: (f64, f64)) {
        self.charging_stations.push(pos);
    }

    pub fn fleet_size(&self) -> usize {
        self.vehicles.len()
    }

    pub fn available_vehicles(&self) -> Vec<&Vehicle> {
        self.vehicles.iter().filter(|v| v.is_available()).collect()
    }

    pub fn pending_requests(&self) -> Vec<&DispatchRequest> {
        self.requests.iter().filter(|r| !r.completed && r.assigned_vehicle.is_none()).collect()
    }

    /// Dispatch available vehicles to pending requests.
    pub fn dispatch(&mut self) -> Vec<(usize, usize)> {
        let mut assignments = Vec::new();
        let pending: Vec<usize> = self.requests.iter()
            .enumerate()
            .filter(|(_, r)| !r.completed && r.assigned_vehicle.is_none())
            .map(|(i, _)| i)
            .collect();

        // Sort pending by priority (highest first)
        let mut sorted_pending = pending;
        sorted_pending.sort_by(|&a, &b|
            self.requests[b].priority.partial_cmp(&self.requests[a].priority)
                .unwrap_or(std::cmp::Ordering::Equal));

        for ri in sorted_pending {
            let pickup = self.requests[ri].pickup;
            let best_vehicle = self.select_vehicle(pickup);

            if let Some(vi) = best_vehicle {
                self.vehicles[vi].status = VehicleStatus::EnRoute;
                self.vehicles[vi].current_task = Some(self.requests[ri].id);
                self.requests[ri].assigned_vehicle = Some(self.vehicles[vi].id);
                assignments.push((self.vehicles[vi].id, self.requests[ri].id));
            }
        }

        assignments
    }

    fn select_vehicle(&self, target: (f64, f64)) -> Option<usize> {
        let available: Vec<usize> = self.vehicles.iter()
            .enumerate()
            .filter(|(_, v)| v.is_available())
            .map(|(i, _)| i)
            .collect();

        if available.is_empty() { return None; }

        match self.dispatch_strategy {
            DispatchStrategy::NearestFirst => {
                available.into_iter()
                    .min_by(|&a, &b| {
                        let da = self.vehicles[a].distance_to(target);
                        let db = self.vehicles[b].distance_to(target);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
            }
            DispatchStrategy::HighestBattery => {
                available.into_iter()
                    .max_by(|&a, &b| {
                        self.vehicles[a].battery_level.partial_cmp(&self.vehicles[b].battery_level)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            }
            DispatchStrategy::FastestEta => {
                available.into_iter()
                    .min_by(|&a, &b| {
                        let ta = self.vehicles[a].time_to(target);
                        let tb = self.vehicles[b].time_to(target);
                        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
                    })
            }
        }
    }

    /// Find nearest charging station for a vehicle.
    pub fn nearest_charger(&self, vehicle_idx: usize) -> Option<(f64, f64)> {
        let pos = self.vehicles[vehicle_idx].position;
        self.charging_stations.iter()
            .min_by(|a, b| {
                let da = (a.0 - pos.0).powi(2) + (a.1 - pos.1).powi(2);
                let db = (b.0 - pos.0).powi(2) + (b.1 - pos.1).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Check for vehicles needing to charge.
    pub fn vehicles_needing_charge(&self) -> Vec<usize> {
        self.vehicles.iter()
            .enumerate()
            .filter(|(_, v)| v.battery_percent() < self.low_battery_threshold
                && v.status != VehicleStatus::Charging)
            .map(|(i, _)| i)
            .collect()
    }

    /// Fleet statistics.
    pub fn stats(&self) -> FleetStats {
        let total = self.vehicles.len();
        let available = self.vehicles.iter().filter(|v| v.is_available()).count();
        let on_task = self.vehicles.iter().filter(|v| v.status == VehicleStatus::OnTask).count();
        let charging = self.vehicles.iter().filter(|v| v.status == VehicleStatus::Charging).count();
        let avg_battery = if total > 0 {
            self.vehicles.iter().map(|v| v.battery_percent()).sum::<f64>() / total as f64
        } else { 0.0 };
        let total_completed = self.vehicles.iter().map(|v| v.tasks_completed).sum();
        let pending = self.requests.iter().filter(|r| !r.completed && r.assigned_vehicle.is_none()).count();

        FleetStats { total, available, on_task, charging, avg_battery, total_completed, pending }
    }
}

impl fmt::Display for FleetManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FleetManager({} vehicles, {} requests, {})",
            self.vehicles.len(), self.requests.len(), self.dispatch_strategy)
    }
}

/// Fleet statistics summary.
#[derive(Clone, Debug)]
pub struct FleetStats {
    pub total: usize,
    pub available: usize,
    pub on_task: usize,
    pub charging: usize,
    pub avg_battery: f64,
    pub total_completed: usize,
    pub pending: usize,
}

impl fmt::Display for FleetStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fleet: {}/{} avail, {} on-task, {} charging, bat={:.0}%, {} completed, {} pending",
            self.available, self.total, self.on_task, self.charging,
            self.avg_battery, self.total_completed, self.pending)
    }
}

// ── Zone Rebalancing ──

/// Zone-based fleet rebalancing.
#[derive(Clone, Debug)]
pub struct RebalanceZone {
    pub id: usize,
    pub center: (f64, f64),
    pub radius: f64,
    pub demand_weight: f64,
}

impl RebalanceZone {
    pub fn new(id: usize, center: (f64, f64), radius: f64) -> Self {
        Self { id, center, radius, demand_weight: 1.0 }
    }

    pub fn with_demand(mut self, weight: f64) -> Self {
        self.demand_weight = weight;
        self
    }

    pub fn contains(&self, pos: (f64, f64)) -> bool {
        let dx = pos.0 - self.center.0;
        let dy = pos.1 - self.center.1;
        (dx * dx + dy * dy).sqrt() <= self.radius
    }
}

/// Compute rebalance moves: which idle vehicles should move to which zones.
pub fn compute_rebalance(
    vehicles: &[Vehicle],
    zones: &[RebalanceZone],
) -> HashMap<usize, (f64, f64)> {
    let mut moves = HashMap::new();

    // Count vehicles per zone
    let mut zone_counts = vec![0usize; zones.len()];
    for v in vehicles {
        for (zi, zone) in zones.iter().enumerate() {
            if zone.contains(v.position) {
                zone_counts[zi] += 1;
            }
        }
    }

    // Find undersupplied zones
    let total_demand: f64 = zones.iter().map(|z| z.demand_weight).sum();
    let total_idle: usize = vehicles.iter().filter(|v| v.is_available()).count();

    for (zi, zone) in zones.iter().enumerate() {
        let target_count = ((zone.demand_weight / total_demand.max(1.0)) * total_idle as f64).ceil() as usize;
        if zone_counts[zi] < target_count {
            // Find nearest idle vehicle not already moving
            let needed = target_count - zone_counts[zi];
            let mut candidates: Vec<(usize, f64)> = vehicles.iter()
                .enumerate()
                .filter(|(i, v)| v.is_available() && !moves.contains_key(i) && !zone.contains(v.position))
                .map(|(i, v)| {
                    let dx = v.position.0 - zone.center.0;
                    let dy = v.position.1 - zone.center.1;
                    (i, (dx * dx + dy * dy).sqrt())
                })
                .collect();
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for (vi, _) in candidates.into_iter().take(needed) {
                moves.insert(vi, zone.center);
            }
        }
    }

    moves
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicle_creation() {
        let v = Vehicle::new(0, "car-0", (1.0, 2.0));
        assert!(v.is_available());
        assert!((v.battery_percent() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_vehicle_distance() {
        let v = Vehicle::new(0, "v", (0.0, 0.0));
        assert!((v.distance_to((3.0, 4.0)) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vehicle_move() {
        let mut v = Vehicle::new(0, "v", (0.0, 0.0)).with_speed(5.0);
        let arrived = v.move_towards((10.0, 0.0), 1.0);
        assert!(!arrived);
        assert!(v.position.0 > 0.0);
    }

    #[test]
    fn test_vehicle_move_arrive() {
        let mut v = Vehicle::new(0, "v", (0.0, 0.0)).with_speed(100.0);
        let arrived = v.move_towards((3.0, 4.0), 1.0);
        assert!(arrived);
        assert!((v.position.0 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dispatch_request() {
        let req = DispatchRequest::new(0, (0.0, 0.0), (3.0, 4.0)).with_priority(5.0);
        assert!((req.trip_distance() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fleet_manager_dispatch() {
        let mut fm = FleetManager::new();
        fm.add_vehicle(Vehicle::new(0, "v0", (0.0, 0.0)));
        fm.add_vehicle(Vehicle::new(1, "v1", (10.0, 0.0)));
        fm.add_request(DispatchRequest::new(0, (1.0, 0.0), (5.0, 0.0)));

        let assignments = fm.dispatch();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].0, 0); // v0 is nearest
    }

    #[test]
    fn test_fleet_manager_highest_battery() {
        let mut fm = FleetManager::new().with_strategy(DispatchStrategy::HighestBattery);
        fm.add_vehicle(Vehicle::new(0, "v0", (0.0, 0.0)).with_battery(30.0, 100.0));
        fm.add_vehicle(Vehicle::new(1, "v1", (0.0, 0.0)).with_battery(90.0, 100.0));
        fm.add_request(DispatchRequest::new(0, (5.0, 0.0), (10.0, 0.0)));

        let assignments = fm.dispatch();
        assert_eq!(assignments[0].0, 1); // v1 has higher battery
    }

    #[test]
    fn test_fleet_stats() {
        let mut fm = FleetManager::new();
        fm.add_vehicle(Vehicle::new(0, "v0", (0.0, 0.0)));
        fm.add_vehicle(Vehicle::new(1, "v1", (1.0, 0.0)));
        fm.add_request(DispatchRequest::new(0, (5.0, 0.0), (10.0, 0.0)));

        let stats = fm.stats();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.available, 2);
        assert_eq!(stats.pending, 1);
    }

    #[test]
    fn test_vehicles_needing_charge() {
        let mut fm = FleetManager::new().with_low_battery_threshold(25.0);
        fm.add_vehicle(Vehicle::new(0, "v0", (0.0, 0.0)).with_battery(50.0, 100.0));
        fm.add_vehicle(Vehicle::new(1, "v1", (0.0, 0.0)).with_battery(15.0, 100.0));

        let need_charge = fm.vehicles_needing_charge();
        assert_eq!(need_charge.len(), 1);
        assert_eq!(need_charge[0], 1);
    }

    #[test]
    fn test_nearest_charger() {
        let mut fm = FleetManager::new();
        fm.add_vehicle(Vehicle::new(0, "v0", (0.0, 0.0)));
        fm.add_charging_station((5.0, 0.0));
        fm.add_charging_station((1.0, 0.0));

        let nearest = fm.nearest_charger(0).unwrap();
        assert!((nearest.0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rebalance_zone() {
        let zone = RebalanceZone::new(0, (5.0, 5.0), 3.0);
        assert!(zone.contains((5.0, 5.0)));
        assert!(zone.contains((7.0, 5.0)));
        assert!(!zone.contains((9.0, 5.0)));
    }

    #[test]
    fn test_compute_rebalance() {
        let vehicles = vec![
            Vehicle::new(0, "v0", (0.0, 0.0)),
            Vehicle::new(1, "v1", (0.5, 0.0)),
            Vehicle::new(2, "v2", (1.0, 0.0)),
        ];
        let zones = vec![
            RebalanceZone::new(0, (0.0, 0.0), 2.0).with_demand(1.0),
            RebalanceZone::new(1, (10.0, 0.0), 2.0).with_demand(2.0),
        ];
        let moves = compute_rebalance(&vehicles, &zones);
        assert!(!moves.is_empty());
    }

    #[test]
    fn test_vehicle_not_available_low_battery() {
        let v = Vehicle::new(0, "v", (0.0, 0.0)).with_battery(5.0, 100.0);
        assert!(!v.is_available());
    }

    #[test]
    fn test_display_formats() {
        let v = Vehicle::new(0, "test", (1.0, 2.0));
        assert!(format!("{v}").contains("test"));

        let fm = FleetManager::new();
        assert!(format!("{fm}").contains("FleetManager"));

        let stats = FleetStats {
            total: 5, available: 3, on_task: 1, charging: 1,
            avg_battery: 75.0, total_completed: 10, pending: 2,
        };
        assert!(format!("{stats}").contains("3/5"));
    }
}
