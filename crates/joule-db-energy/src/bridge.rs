//! Bridge to invisible-os `EnergyAccountant` trait.
//!
//! Wraps joule-db-energy's `EnergyMonitor` and `EnergySnapshot` into the
//! shared `EnergyAccountant` trait from `inv-energy-core`, enabling joule-db
//! to participate in the system-wide energy chain:
//!
//! **kernel measures → mesh aggregates → database indexes → applications display**

use crate::monitor::{EnergySnapshot, ThermalState as JdbThermalState};
use chrono::Utc;
use inv_energy_core::{
    EnergyAccountant, EnergyError, EnergyReceipt, EnergySnapshot as CoreSnapshot, EnergySource,
    OperationId, ThermalState as CoreThermalState,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Converts joule-db's `ThermalState` to inv-energy-core's `ThermalState`.
fn map_thermal_state(state: JdbThermalState) -> CoreThermalState {
    match state {
        JdbThermalState::Nominal => CoreThermalState::Nominal,
        JdbThermalState::Fair => CoreThermalState::Fair,
        JdbThermalState::Serious => CoreThermalState::Serious,
        JdbThermalState::Critical => CoreThermalState::Critical,
    }
}

/// Determines the energy source from the joule-db snapshot.
fn detect_source(snapshot: &EnergySnapshot) -> EnergySource {
    match (snapshot.battery_percent, snapshot.battery_charging) {
        (Some(_), true) => EnergySource::Grid, // Charging = plugged in
        (Some(_), false) => EnergySource::Battery, // On battery
        (None, _) => EnergySource::Grid,       // Desktop / server
    }
}

/// An [`EnergyAccountant`] implementation that wraps joule-db-energy's
/// [`EnergyMonitor`] (via its shared `Arc<RwLock<EnergySnapshot>>`).
///
/// # Usage
///
/// ```ignore
/// use joule_db_energy::{EnergyConfig, EnergyMonitor};
/// use joule_db_energy::bridge::JouleDbAccountant;
///
/// let monitor = EnergyMonitor::new(EnergyConfig::default());
/// let (handle, _thread) = monitor.start_background();
/// let accountant = JouleDbAccountant::new(handle);
///
/// // Now usable as an EnergyAccountant
/// let op = accountant.begin_operation("SELECT * FROM users");
/// // ... execute query ...
/// let receipt = accountant.end_operation(op).unwrap();
/// println!("Query consumed {:.6} J", receipt.joules);
/// ```
pub struct JouleDbAccountant {
    /// Shared handle to the latest hardware snapshot from EnergyMonitor.
    snapshot_handle: Arc<RwLock<EnergySnapshot>>,
    /// In-flight operations: id → (label, start_time, snapshot_at_start).
    operations: RwLock<HashMap<u64, (String, Instant, f64)>>,
    /// Monotonic operation ID counter.
    next_id: AtomicU64,
}

impl JouleDbAccountant {
    /// Create a new accountant from an EnergyMonitor's shared snapshot handle.
    ///
    /// Obtain the handle via `EnergyMonitor::snapshot_handle()` or
    /// `EnergyMonitor::start_background()`.
    pub fn new(snapshot_handle: Arc<RwLock<EnergySnapshot>>) -> Self {
        Self {
            snapshot_handle,
            operations: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Read the current snapshot from the shared handle.
    fn current_snapshot(&self) -> EnergySnapshot {
        self.snapshot_handle
            .read()
            .map(|s| s.clone())
            .unwrap_or_default()
    }
}

impl EnergyAccountant for JouleDbAccountant {
    fn begin_operation(&self, label: &str) -> OperationId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let snap = self.current_snapshot();
        self.operations
            .write()
            .unwrap()
            .insert(id, (label.to_string(), Instant::now(), snap.power_watts));
        OperationId(id)
    }

    fn end_operation(&self, id: OperationId) -> Result<EnergyReceipt, EnergyError> {
        let (label, start, power_at_start) = self
            .operations
            .write()
            .unwrap()
            .remove(&id.0)
            .ok_or(EnergyError::UnknownOperation(id))?;

        let duration = start.elapsed();
        let snap = self.current_snapshot();

        // Average the power at start and end for a better estimate.
        let avg_watts = (power_at_start + snap.power_watts) / 2.0;
        let joules = avg_watts * duration.as_secs_f64();

        Ok(EnergyReceipt {
            operation_id: id,
            label,
            joules,
            duration,
            avg_watts,
            thermal_state: map_thermal_state(snap.thermal_state),
            source: detect_source(&snap),
            completed_at: Utc::now(),
        })
    }

    fn snapshot(&self) -> CoreSnapshot {
        let snap = self.current_snapshot();
        CoreSnapshot {
            current_watts: snap.power_watts,
            thermal_state: map_thermal_state(snap.thermal_state),
            source: detect_source(&snap),
            cumulative_joules: snap.cumulative_joules,
            budget_remaining_joules: None,
            timestamp: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::ThermalState;

    fn make_handle(snapshot: EnergySnapshot) -> Arc<RwLock<EnergySnapshot>> {
        Arc::new(RwLock::new(snapshot))
    }

    #[test]
    fn bridge_begin_end_operation() {
        let snap = EnergySnapshot {
            power_watts: 25.0,
            thermal_state: ThermalState::Nominal,
            cumulative_joules: 100.0,
            ..Default::default()
        };
        let accountant = JouleDbAccountant::new(make_handle(snap));

        let op = accountant.begin_operation("test_query");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let receipt = accountant.end_operation(op).unwrap();

        assert_eq!(receipt.label, "test_query");
        assert!(receipt.joules > 0.0);
        assert!(receipt.duration.as_millis() >= 10);
        assert_eq!(receipt.thermal_state, CoreThermalState::Nominal);
    }

    #[test]
    fn bridge_unknown_operation() {
        let handle = make_handle(EnergySnapshot::default());
        let accountant = JouleDbAccountant::new(handle);
        let result = accountant.end_operation(OperationId(999));
        assert!(matches!(result, Err(EnergyError::UnknownOperation(_))));
    }

    #[test]
    fn bridge_snapshot() {
        let snap = EnergySnapshot {
            power_watts: 30.0,
            thermal_state: ThermalState::Fair,
            cumulative_joules: 42.5,
            battery_percent: Some(80.0),
            battery_charging: false,
            ..Default::default()
        };
        let accountant = JouleDbAccountant::new(make_handle(snap));
        let core_snap = accountant.snapshot();

        assert!((core_snap.current_watts - 30.0).abs() < 0.01);
        assert_eq!(core_snap.thermal_state, CoreThermalState::Fair);
        assert!((core_snap.cumulative_joules - 42.5).abs() < 0.01);
        assert_eq!(core_snap.source, EnergySource::Battery);
    }

    #[test]
    fn bridge_thermal_state_mapping() {
        assert_eq!(
            map_thermal_state(ThermalState::Nominal),
            CoreThermalState::Nominal
        );
        assert_eq!(
            map_thermal_state(ThermalState::Fair),
            CoreThermalState::Fair
        );
        assert_eq!(
            map_thermal_state(ThermalState::Serious),
            CoreThermalState::Serious
        );
        assert_eq!(
            map_thermal_state(ThermalState::Critical),
            CoreThermalState::Critical
        );
    }

    #[test]
    fn bridge_source_detection() {
        let mut snap = EnergySnapshot::default();

        // No battery = grid
        snap.battery_percent = None;
        assert_eq!(detect_source(&snap), EnergySource::Grid);

        // Battery + charging = grid
        snap.battery_percent = Some(80.0);
        snap.battery_charging = true;
        assert_eq!(detect_source(&snap), EnergySource::Grid);

        // Battery + not charging = battery
        snap.battery_charging = false;
        assert_eq!(detect_source(&snap), EnergySource::Battery);
    }
}
