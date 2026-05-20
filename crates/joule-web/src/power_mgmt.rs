//! Power management: power states (active/idle/sleep/deep_sleep),
//! transition rules, wake sources, power budget tracking, battery model
//! (capacity, discharge rate, remaining), and power profile optimization.

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};

// ── Types ──

/// Device power state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PowerState {
    Active,
    Idle,
    Sleep,
    DeepSleep,
    Shutdown,
}

impl PowerState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Idle => "idle",
            Self::Sleep => "sleep",
            Self::DeepSleep => "deep_sleep",
            Self::Shutdown => "shutdown",
        }
    }

    /// Typical relative power draw (1.0 = active baseline).
    pub fn power_factor(&self) -> f64 {
        match self {
            Self::Active => 1.0,
            Self::Idle => 0.3,
            Self::Sleep => 0.05,
            Self::DeepSleep => 0.005,
            Self::Shutdown => 0.0,
        }
    }
}

/// Source that can wake the device from a low-power state.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WakeSource {
    Timer,
    Interrupt(String),
    NetworkActivity,
    UserInput,
    Sensor(String),
    External,
}

impl WakeSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Timer => "timer",
            Self::Interrupt(_) => "interrupt",
            Self::NetworkActivity => "network",
            Self::UserInput => "user_input",
            Self::Sensor(_) => "sensor",
            Self::External => "external",
        }
    }
}

/// A recorded state transition.
#[derive(Debug, Clone)]
pub struct Transition {
    pub from: PowerState,
    pub to: PowerState,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
    pub wake_source: Option<WakeSource>,
}

/// Transition rule defining allowed state changes.
#[derive(Debug, Clone)]
pub struct TransitionRule {
    pub from: PowerState,
    pub to: PowerState,
    pub allowed_wake_sources: Vec<WakeSource>,
    /// Minimum time in current state before transition (seconds).
    pub min_dwell_secs: i64,
}

impl TransitionRule {
    pub fn new(from: PowerState, to: PowerState) -> Self {
        Self {
            from,
            to,
            allowed_wake_sources: Vec::new(),
            min_dwell_secs: 0,
        }
    }

    pub fn with_wake_sources(mut self, sources: Vec<WakeSource>) -> Self {
        self.allowed_wake_sources = sources;
        self
    }

    pub fn with_min_dwell(mut self, secs: i64) -> Self {
        self.min_dwell_secs = secs;
        self
    }
}

// ── Battery Model ──

/// Simulated battery.
#[derive(Debug, Clone)]
pub struct Battery {
    /// Total capacity in milliamp-hours (mAh).
    pub capacity_mah: f64,
    /// Current charge in mAh.
    pub charge_mah: f64,
    /// Nominal voltage in volts.
    pub nominal_voltage: f64,
    /// Current discharge rate in milliamps (mA).
    pub discharge_rate_ma: f64,
    /// Charge cycles count.
    pub cycle_count: u32,
    /// Health factor (0.0..1.0), degrades over cycles.
    pub health: f64,
}

impl Battery {
    pub fn new(capacity_mah: f64, voltage: f64) -> Self {
        Self {
            capacity_mah,
            charge_mah: capacity_mah,
            nominal_voltage: voltage,
            discharge_rate_ma: 0.0,
            cycle_count: 0,
            health: 1.0,
        }
    }

    /// State of charge (0.0..1.0).
    pub fn soc(&self) -> f64 {
        if self.capacity_mah <= 0.0 {
            return 0.0;
        }
        (self.charge_mah / (self.capacity_mah * self.health)).clamp(0.0, 1.0)
    }

    /// State of charge as percentage.
    pub fn soc_percent(&self) -> f64 {
        self.soc() * 100.0
    }

    /// Remaining energy in milliwatt-hours (mWh).
    pub fn remaining_energy_mwh(&self) -> f64 {
        self.charge_mah * self.nominal_voltage
    }

    /// Remaining energy in joules.
    pub fn remaining_energy_j(&self) -> f64 {
        self.remaining_energy_mwh() / 1000.0 * 3600.0
    }

    /// Estimated remaining time in hours at current discharge rate.
    pub fn remaining_hours(&self) -> Option<f64> {
        if self.discharge_rate_ma <= 0.0 {
            return None;
        }
        Some(self.charge_mah / self.discharge_rate_ma)
    }

    /// Simulate discharge over a time period.
    pub fn discharge(&mut self, duration_secs: f64) {
        if self.discharge_rate_ma <= 0.0 {
            return;
        }
        let hours = duration_secs / 3600.0;
        let consumed = self.discharge_rate_ma * hours;
        self.charge_mah = (self.charge_mah - consumed).max(0.0);
    }

    /// Simulate charging.
    pub fn charge(&mut self, charge_rate_ma: f64, duration_secs: f64) {
        let hours = duration_secs / 3600.0;
        let added = charge_rate_ma * hours;
        let max_charge = self.capacity_mah * self.health;
        self.charge_mah = (self.charge_mah + added).min(max_charge);
    }

    /// Full charge cycle (simulate).
    pub fn full_charge(&mut self) {
        self.charge_mah = self.capacity_mah * self.health;
        self.cycle_count += 1;
        // Degrade health slightly per cycle.
        self.health = (self.health - 0.0005).max(0.5);
    }

    /// Current power draw in milliwatts.
    pub fn power_draw_mw(&self) -> f64 {
        self.discharge_rate_ma * self.nominal_voltage
    }

    pub fn is_depleted(&self) -> bool {
        self.charge_mah <= 0.0
    }

    pub fn is_low(&self) -> bool {
        self.soc() < 0.2
    }

    pub fn is_critical(&self) -> bool {
        self.soc() < 0.05
    }
}

// ── Power Budget ──

/// Tracks power budget: allocations per subsystem.
#[derive(Debug, Clone)]
pub struct PowerBudget {
    /// Total power budget in milliwatts.
    pub total_mw: f64,
    /// Allocations per subsystem.
    allocations: HashMap<String, f64>,
}

impl PowerBudget {
    pub fn new(total_mw: f64) -> Self {
        Self {
            total_mw,
            allocations: HashMap::new(),
        }
    }

    /// Allocate power to a subsystem.
    pub fn allocate(&mut self, subsystem: &str, power_mw: f64) -> Result<(), String> {
        let current = self.allocated_mw();
        if current + power_mw > self.total_mw {
            return Err(format!(
                "budget exceeded: {:.1}mW allocated + {:.1}mW requested > {:.1}mW total",
                current, power_mw, self.total_mw
            ));
        }
        *self.allocations.entry(subsystem.to_string()).or_insert(0.0) += power_mw;
        Ok(())
    }

    /// Release power allocation for a subsystem.
    pub fn release(&mut self, subsystem: &str) -> f64 {
        self.allocations.remove(subsystem).unwrap_or(0.0)
    }

    /// Total allocated power.
    pub fn allocated_mw(&self) -> f64 {
        self.allocations.values().sum()
    }

    /// Remaining budget.
    pub fn remaining_mw(&self) -> f64 {
        (self.total_mw - self.allocated_mw()).max(0.0)
    }

    /// Utilization as a fraction (0.0..1.0).
    pub fn utilization(&self) -> f64 {
        if self.total_mw <= 0.0 {
            return 0.0;
        }
        (self.allocated_mw() / self.total_mw).min(1.0)
    }

    pub fn subsystem_count(&self) -> usize {
        self.allocations.len()
    }

    /// Get power allocation for a specific subsystem.
    pub fn get_allocation(&self, subsystem: &str) -> f64 {
        self.allocations.get(subsystem).copied().unwrap_or(0.0)
    }
}

// ── Power Profile ──

/// Named power profile with per-state configurations.
#[derive(Debug, Clone)]
pub struct PowerProfile {
    pub name: String,
    /// Base power draw in mW for each state.
    pub state_power: HashMap<PowerState, f64>,
    /// Target idle timeout in seconds before transitioning to sleep.
    pub idle_timeout_secs: i64,
    /// Target sleep timeout in seconds before transitioning to deep sleep.
    pub sleep_timeout_secs: i64,
}

impl PowerProfile {
    pub fn performance() -> Self {
        let mut state_power = HashMap::new();
        state_power.insert(PowerState::Active, 500.0);
        state_power.insert(PowerState::Idle, 200.0);
        state_power.insert(PowerState::Sleep, 50.0);
        state_power.insert(PowerState::DeepSleep, 5.0);
        Self {
            name: "performance".to_string(),
            state_power,
            idle_timeout_secs: 300,
            sleep_timeout_secs: 600,
        }
    }

    pub fn balanced() -> Self {
        let mut state_power = HashMap::new();
        state_power.insert(PowerState::Active, 300.0);
        state_power.insert(PowerState::Idle, 100.0);
        state_power.insert(PowerState::Sleep, 20.0);
        state_power.insert(PowerState::DeepSleep, 2.0);
        Self {
            name: "balanced".to_string(),
            state_power,
            idle_timeout_secs: 120,
            sleep_timeout_secs: 300,
        }
    }

    pub fn power_saver() -> Self {
        let mut state_power = HashMap::new();
        state_power.insert(PowerState::Active, 150.0);
        state_power.insert(PowerState::Idle, 50.0);
        state_power.insert(PowerState::Sleep, 10.0);
        state_power.insert(PowerState::DeepSleep, 1.0);
        Self {
            name: "power_saver".to_string(),
            state_power,
            idle_timeout_secs: 30,
            sleep_timeout_secs: 60,
        }
    }

    /// Get power draw for a given state under this profile.
    pub fn power_for_state(&self, state: PowerState) -> f64 {
        self.state_power.get(&state).copied().unwrap_or(0.0)
    }

    /// Estimated battery life in hours for a given battery and usage pattern.
    /// `active_fraction` is the fraction of time spent in active state.
    pub fn estimated_battery_life(&self, battery: &Battery, active_fraction: f64) -> Option<f64> {
        let active_fraction = active_fraction.clamp(0.0, 1.0);
        let idle_fraction = 1.0 - active_fraction;
        let active_power = self.power_for_state(PowerState::Active);
        let idle_power = self.power_for_state(PowerState::Idle);
        let avg_power_mw = active_fraction * active_power + idle_fraction * idle_power;
        if avg_power_mw <= 0.0 {
            return None;
        }
        let avg_current_ma = avg_power_mw / battery.nominal_voltage;
        Some(battery.charge_mah / avg_current_ma)
    }
}

// ── Power Manager ──

/// Power state manager with transition rules, battery tracking, and profiles.
pub struct PowerManager {
    state: PowerState,
    state_entered_at: DateTime<Utc>,
    battery: Option<Battery>,
    budget: PowerBudget,
    profile: PowerProfile,
    transition_rules: Vec<TransitionRule>,
    history: Vec<Transition>,
    max_history: usize,
    registered_wake_sources: Vec<WakeSource>,
}

impl PowerManager {
    pub fn new(budget_mw: f64) -> Self {
        Self {
            state: PowerState::Active,
            state_entered_at: Utc::now(),
            battery: None,
            budget: PowerBudget::new(budget_mw),
            profile: PowerProfile::balanced(),
            transition_rules: Vec::new(),
            history: Vec::new(),
            max_history: 500,
            registered_wake_sources: Vec::new(),
        }
    }

    pub fn set_battery(&mut self, mut battery: Battery) {
        // Set the initial discharge rate based on the current state and profile.
        let power_mw = self.profile.power_for_state(self.state);
        if battery.nominal_voltage > 0.0 {
            battery.discharge_rate_ma = power_mw / battery.nominal_voltage;
        }
        self.battery = Some(battery);
    }

    pub fn battery(&self) -> Option<&Battery> {
        self.battery.as_ref()
    }

    pub fn battery_mut(&mut self) -> Option<&mut Battery> {
        self.battery.as_mut()
    }

    pub fn set_profile(&mut self, profile: PowerProfile) {
        self.profile = profile;
    }

    pub fn profile(&self) -> &PowerProfile {
        &self.profile
    }

    pub fn add_transition_rule(&mut self, rule: TransitionRule) {
        self.transition_rules.push(rule);
    }

    pub fn register_wake_source(&mut self, source: WakeSource) {
        if !self.registered_wake_sources.contains(&source) {
            self.registered_wake_sources.push(source);
        }
    }

    pub fn state(&self) -> PowerState {
        self.state
    }

    pub fn budget(&self) -> &PowerBudget {
        &self.budget
    }

    pub fn budget_mut(&mut self) -> &mut PowerBudget {
        &mut self.budget
    }

    /// Request a state transition.
    pub fn transition(&mut self, to: PowerState, reason: &str, now: DateTime<Utc>) -> Result<(), String> {
        if self.state == to {
            return Ok(());
        }

        // Check if transition is allowed by rules.
        let dwell = now.signed_duration_since(self.state_entered_at).num_seconds();
        let from = self.state;

        // Find applicable rule.
        let rule = self.transition_rules.iter().find(|r| r.from == from && r.to == to);
        if let Some(rule) = rule {
            if dwell < rule.min_dwell_secs {
                return Err(format!(
                    "minimum dwell time not met: {}s < {}s",
                    dwell, rule.min_dwell_secs
                ));
            }
        }

        // Cannot wake from shutdown (only external can restart).
        if self.state == PowerState::Shutdown && to != PowerState::Active {
            return Err("can only transition from shutdown to active".to_string());
        }

        // Update battery discharge rate based on new state.
        if let Some(battery) = &mut self.battery {
            // Discharge for time spent in current state.
            battery.discharge(dwell as f64);
            // Update discharge rate for new state.
            let power_mw = self.profile.power_for_state(to);
            battery.discharge_rate_ma = power_mw / battery.nominal_voltage;
        }

        self.history.push(Transition {
            from,
            to,
            reason: reason.to_string(),
            timestamp: now,
            wake_source: None,
        });

        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        self.state = to;
        self.state_entered_at = now;

        Ok(())
    }

    /// Wake the device from a low-power state.
    pub fn wake(&mut self, source: WakeSource, now: DateTime<Utc>) -> Result<(), String> {
        let from = self.state;

        if from == PowerState::Active {
            return Ok(()); // Already awake.
        }

        if from == PowerState::Shutdown {
            return Err("cannot wake from shutdown".to_string());
        }

        // Check if wake source is allowed for current state transition.
        let applicable_rule = self.transition_rules.iter().find(|r| {
            r.from == from && r.to == PowerState::Active
        });

        if let Some(rule) = applicable_rule {
            if !rule.allowed_wake_sources.is_empty() {
                let allowed = rule.allowed_wake_sources.iter().any(|ws| {
                    std::mem::discriminant(ws) == std::mem::discriminant(&source)
                });
                if !allowed {
                    return Err(format!("wake source {} not allowed from {}", source.as_str(), from.as_str()));
                }
            }
        }

        let dwell = now.signed_duration_since(self.state_entered_at).num_seconds();
        if let Some(battery) = &mut self.battery {
            battery.discharge(dwell as f64);
            let power_mw = self.profile.power_for_state(PowerState::Active);
            battery.discharge_rate_ma = power_mw / battery.nominal_voltage;
        }

        self.history.push(Transition {
            from,
            to: PowerState::Active,
            reason: format!("wake: {}", source.as_str()),
            timestamp: now,
            wake_source: Some(source),
        });

        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        self.state = PowerState::Active;
        self.state_entered_at = now;

        Ok(())
    }

    /// Check if the device should auto-transition based on idle/sleep timeouts.
    pub fn check_timeouts(&mut self, now: DateTime<Utc>) -> Option<PowerState> {
        let dwell = now.signed_duration_since(self.state_entered_at).num_seconds();

        match self.state {
            PowerState::Active => {
                // Not handled here — application decides when idle.
                None
            }
            PowerState::Idle => {
                if dwell >= self.profile.idle_timeout_secs {
                    let _ = self.transition(PowerState::Sleep, "idle timeout", now);
                    Some(PowerState::Sleep)
                } else {
                    None
                }
            }
            PowerState::Sleep => {
                if dwell >= self.profile.sleep_timeout_secs {
                    let _ = self.transition(PowerState::DeepSleep, "sleep timeout", now);
                    Some(PowerState::DeepSleep)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Current power draw in mW based on state and profile.
    pub fn current_power_draw_mw(&self) -> f64 {
        self.profile.power_for_state(self.state)
    }

    pub fn history(&self) -> &[Transition] {
        &self.history
    }

    pub fn transition_count(&self) -> usize {
        self.history.len()
    }

    /// Time spent in current state.
    pub fn dwell_time(&self, now: DateTime<Utc>) -> Duration {
        now.signed_duration_since(self.state_entered_at)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_state_factor() {
        assert!((PowerState::Active.power_factor() - 1.0).abs() < 1e-10);
        assert!(PowerState::Sleep.power_factor() < PowerState::Idle.power_factor());
        assert!((PowerState::Shutdown.power_factor() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn battery_soc() {
        let bat = Battery::new(3000.0, 3.7);
        assert!((bat.soc() - 1.0).abs() < 1e-10);
        assert!((bat.soc_percent() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn battery_discharge() {
        let mut bat = Battery::new(3000.0, 3.7);
        bat.discharge_rate_ma = 300.0;
        bat.discharge(3600.0); // 1 hour
        assert!((bat.charge_mah - 2700.0).abs() < 1e-6);
    }

    #[test]
    fn battery_charge() {
        let mut bat = Battery::new(3000.0, 3.7);
        bat.charge_mah = 1000.0;
        bat.charge(500.0, 3600.0); // 500mA for 1 hour
        assert!((bat.charge_mah - 1500.0).abs() < 1e-6);
    }

    #[test]
    fn battery_full_charge_degrades() {
        let mut bat = Battery::new(3000.0, 3.7);
        bat.charge_mah = 0.0;
        bat.full_charge();
        assert_eq!(bat.cycle_count, 1);
        assert!(bat.health < 1.0);
    }

    #[test]
    fn battery_remaining_hours() {
        let mut bat = Battery::new(3000.0, 3.7);
        bat.discharge_rate_ma = 300.0;
        let hours = bat.remaining_hours().unwrap();
        assert!((hours - 10.0).abs() < 1e-6);
    }

    #[test]
    fn battery_remaining_energy() {
        let bat = Battery::new(3000.0, 3.7);
        let mwh = bat.remaining_energy_mwh();
        assert!((mwh - 11100.0).abs() < 1e-6);
        let j = bat.remaining_energy_j();
        assert!(j > 0.0);
    }

    #[test]
    fn battery_low_critical() {
        let mut bat = Battery::new(3000.0, 3.7);
        bat.charge_mah = 500.0;
        assert!(bat.is_low());
        assert!(!bat.is_critical());
        bat.charge_mah = 100.0;
        assert!(bat.is_critical());
    }

    #[test]
    fn power_budget_allocation() {
        let mut budget = PowerBudget::new(1000.0);
        budget.allocate("radio", 300.0).unwrap();
        budget.allocate("cpu", 400.0).unwrap();
        assert!((budget.allocated_mw() - 700.0).abs() < 1e-10);
        assert!((budget.remaining_mw() - 300.0).abs() < 1e-10);
    }

    #[test]
    fn power_budget_exceeded() {
        let mut budget = PowerBudget::new(500.0);
        budget.allocate("cpu", 400.0).unwrap();
        assert!(budget.allocate("gpu", 200.0).is_err());
    }

    #[test]
    fn power_budget_release() {
        let mut budget = PowerBudget::new(1000.0);
        budget.allocate("radio", 300.0).unwrap();
        let released = budget.release("radio");
        assert!((released - 300.0).abs() < 1e-10);
        assert!((budget.allocated_mw() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn power_budget_utilization() {
        let mut budget = PowerBudget::new(1000.0);
        budget.allocate("cpu", 500.0).unwrap();
        assert!((budget.utilization() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn power_profile_performance() {
        let p = PowerProfile::performance();
        assert_eq!(p.name, "performance");
        assert!(p.power_for_state(PowerState::Active) > p.power_for_state(PowerState::Idle));
    }

    #[test]
    fn power_profile_battery_life() {
        let p = PowerProfile::balanced();
        let bat = Battery::new(3000.0, 3.7);
        let hours = p.estimated_battery_life(&bat, 0.5).unwrap();
        assert!(hours > 0.0);
    }

    #[test]
    fn power_manager_transition() {
        let mut pm = PowerManager::new(1000.0);
        let now = Utc::now();
        pm.transition(PowerState::Idle, "user inactive", now).unwrap();
        assert_eq!(pm.state(), PowerState::Idle);
        assert_eq!(pm.transition_count(), 1);
    }

    #[test]
    fn power_manager_same_state_noop() {
        let mut pm = PowerManager::new(1000.0);
        let now = Utc::now();
        pm.transition(PowerState::Active, "noop", now).unwrap();
        assert_eq!(pm.transition_count(), 0);
    }

    #[test]
    fn power_manager_dwell_check() {
        let mut pm = PowerManager::new(1000.0);
        pm.add_transition_rule(TransitionRule::new(PowerState::Active, PowerState::Sleep).with_min_dwell(60));
        let now = Utc::now();
        // Should fail — haven't dwelled long enough.
        let result = pm.transition(PowerState::Sleep, "too early", now);
        assert!(result.is_err());
    }

    #[test]
    fn power_manager_wake() {
        let mut pm = PowerManager::new(1000.0);
        let now = Utc::now();
        pm.transition(PowerState::Sleep, "idle", now).unwrap();
        let later = now + Duration::seconds(10);
        pm.wake(WakeSource::Timer, later).unwrap();
        assert_eq!(pm.state(), PowerState::Active);
    }

    #[test]
    fn power_manager_timeout_idle_to_sleep() {
        let mut pm = PowerManager::new(1000.0);
        pm.set_profile(PowerProfile::power_saver()); // idle_timeout = 30s
        let now = Utc::now();
        pm.transition(PowerState::Idle, "idle", now).unwrap();
        let later = now + Duration::seconds(60);
        let new_state = pm.check_timeouts(later);
        assert_eq!(new_state, Some(PowerState::Sleep));
    }

    #[test]
    fn power_manager_battery_discharge_on_transition() {
        let mut pm = PowerManager::new(1000.0);
        let bat = Battery::new(3000.0, 3.7);
        pm.set_battery(bat);
        let now = Utc::now();
        let later = now + Duration::seconds(3600);
        pm.transition(PowerState::Idle, "idle", later).unwrap();
        let charge = pm.battery().unwrap().charge_mah;
        assert!(charge < 3000.0);
    }

    #[test]
    fn power_manager_current_draw() {
        let pm = PowerManager::new(1000.0);
        let draw = pm.current_power_draw_mw();
        assert!(draw > 0.0);
    }

    #[test]
    fn wake_source_str() {
        assert_eq!(WakeSource::Timer.as_str(), "timer");
        assert_eq!(WakeSource::NetworkActivity.as_str(), "network");
    }

    #[test]
    fn battery_depleted() {
        let mut bat = Battery::new(3000.0, 3.7);
        bat.charge_mah = 0.0;
        assert!(bat.is_depleted());
    }
}
