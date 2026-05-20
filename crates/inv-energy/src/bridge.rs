//! Bridge to invisible-os `EnergyAccountant` trait.
//!
//! Wraps inv-energy's `EnergyMeter` (typically `CompositeMeter`) into the
//! shared `EnergyAccountant` trait from `inv-energy-core`, enabling the
//! invisible-infrastructure mesh to participate in the system-wide energy chain:
//!
//! **kernel measures → mesh aggregates → database indexes → applications display**

use crate::meter::EnergyMeter;
use chrono::Utc;
use inv_core::energy::{EnergySource as InfraSource, ThermalState as InfraThermalState};
use inv_energy_core::{
    EnergyAccountant, EnergyError, EnergyReceipt, EnergySnapshot as CoreSnapshot, EnergySource,
    OperationId, ThermalState as CoreThermalState,
};
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Converts infrastructure's `ThermalState` to inv-energy-core's `ThermalState`.
fn map_thermal_state(state: InfraThermalState) -> CoreThermalState {
    match state {
        InfraThermalState::Normal => CoreThermalState::Nominal,
        InfraThermalState::Warm => CoreThermalState::Fair,
        InfraThermalState::Throttled => CoreThermalState::Serious,
        InfraThermalState::Critical => CoreThermalState::Critical,
        InfraThermalState::OrbitalVacuum => CoreThermalState::Nominal,
    }
}

/// Converts infrastructure's `EnergySource` to inv-energy-core's `EnergySource`.
fn map_source(source: InfraSource) -> EnergySource {
    match source {
        InfraSource::WallPower => EnergySource::Grid,
        InfraSource::Battery => EnergySource::Battery,
        InfraSource::Solar => EnergySource::Solar,
        InfraSource::Unknown => EnergySource::Unknown,
        InfraSource::OrbitalSolar => EnergySource::Solar,
        InfraSource::OrbitalBattery => EnergySource::Battery,
    }
}

/// An [`EnergyAccountant`] implementation that wraps any `EnergyMeter`
/// from invisible-infrastructure.
///
/// # Usage
///
/// ```ignore
/// use inv_energy::bridge::MeshAccountant;
/// use inv_energy::meter::detect_meter;
///
/// let meter = detect_meter();
/// let accountant = MeshAccountant::new(meter);
///
/// // Now usable as an EnergyAccountant
/// let op = accountant.begin_operation("workload_schedule");
/// // ... run workload ...
/// let receipt = accountant.end_operation(op).unwrap();
/// println!("Workload consumed {:.6} J", receipt.joules);
/// ```
pub struct MeshAccountant<M: EnergyMeter> {
    /// The underlying energy meter.
    meter: M,
    /// In-flight operations: id → (label, start_time, joules_at_start, watts_at_start).
    operations: RwLock<HashMap<u64, (String, Instant, f64, f64)>>,
    /// Monotonic operation ID counter.
    next_id: AtomicU64,
}

impl<M: EnergyMeter> MeshAccountant<M> {
    /// Create a new mesh accountant wrapping an `EnergyMeter`.
    pub fn new(meter: M) -> Self {
        Self {
            meter,
            operations: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Get a reference to the underlying meter.
    pub fn meter(&self) -> &M {
        &self.meter
    }
}

impl<M: EnergyMeter> EnergyAccountant for MeshAccountant<M> {
    fn begin_operation(&self, label: &str) -> OperationId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (joules, watts) = self
            .meter
            .read()
            .map(|r| (r.joules.as_f64(), r.watts_current.as_f64()))
            .unwrap_or((0.0, 0.0));

        self.operations
            .write()
            .unwrap()
            .insert(id, (label.to_string(), Instant::now(), joules, watts));
        OperationId(id)
    }

    fn end_operation(&self, id: OperationId) -> Result<EnergyReceipt, EnergyError> {
        let (label, start, joules_at_start, watts_at_start) = self
            .operations
            .write()
            .unwrap()
            .remove(&id.0)
            .ok_or(EnergyError::UnknownOperation(id))?;

        let duration = start.elapsed();

        // Try to read current meter state for accurate delta.
        let (joules, avg_watts, thermal, source) = match self.meter.read() {
            Ok(reading) => {
                let delta = reading.joules.as_f64() - joules_at_start;
                let avg_w = (watts_at_start + reading.watts_current.as_f64()) / 2.0;
                let thermal = map_thermal_state(self.meter.thermal_state());
                let source = map_source(self.meter.energy_source());
                // Use the meter delta if positive, otherwise estimate from power × time.
                let joules = if delta > 0.0 {
                    delta
                } else {
                    avg_w * duration.as_secs_f64()
                };
                (joules, avg_w, thermal, source)
            }
            Err(_) => {
                // Fallback: estimate from power at start.
                let joules = watts_at_start * duration.as_secs_f64();
                (
                    joules,
                    watts_at_start,
                    CoreThermalState::Nominal,
                    EnergySource::Unknown,
                )
            }
        };

        Ok(EnergyReceipt {
            operation_id: id,
            label,
            joules,
            duration,
            avg_watts,
            thermal_state: thermal,
            source,
            completed_at: Utc::now(),
        })
    }

    fn snapshot(&self) -> CoreSnapshot {
        match self.meter.read() {
            Ok(reading) => CoreSnapshot {
                current_watts: reading.watts_current.as_f64(),
                thermal_state: map_thermal_state(self.meter.thermal_state()),
                source: map_source(self.meter.energy_source()),
                cumulative_joules: reading.joules.as_f64(),
                budget_remaining_joules: None,
                timestamp: Utc::now(),
            },
            Err(_) => CoreSnapshot {
                current_watts: 0.0,
                thermal_state: CoreThermalState::Nominal,
                source: EnergySource::Unknown,
                cumulative_joules: 0.0,
                budget_remaining_joules: None,
                timestamp: Utc::now(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meter::EstimationMeter;

    #[test]
    fn mesh_bridge_begin_end() {
        let meter = EstimationMeter::new(30.0);
        let accountant = MeshAccountant::new(meter);

        let op = accountant.begin_operation("test_workload");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let receipt = accountant.end_operation(op).unwrap();

        assert_eq!(receipt.label, "test_workload");
        assert!(receipt.joules >= 0.0);
        assert!(receipt.duration.as_millis() >= 10);
    }

    #[test]
    fn mesh_bridge_unknown_operation() {
        let meter = EstimationMeter::new(30.0);
        let accountant = MeshAccountant::new(meter);
        let result = accountant.end_operation(OperationId(999));
        assert!(matches!(result, Err(EnergyError::UnknownOperation(_))));
    }

    #[test]
    fn mesh_bridge_snapshot() {
        let meter = EstimationMeter::new(30.0);
        let accountant = MeshAccountant::new(meter);
        let snap = accountant.snapshot();

        assert!(snap.current_watts > 0.0);
        assert_eq!(snap.source, EnergySource::Unknown); // EstimationMeter → Unknown
    }

    #[test]
    fn thermal_state_mapping() {
        assert_eq!(
            map_thermal_state(InfraThermalState::Normal),
            CoreThermalState::Nominal
        );
        assert_eq!(
            map_thermal_state(InfraThermalState::Warm),
            CoreThermalState::Fair
        );
        assert_eq!(
            map_thermal_state(InfraThermalState::Throttled),
            CoreThermalState::Serious
        );
        assert_eq!(
            map_thermal_state(InfraThermalState::Critical),
            CoreThermalState::Critical
        );
    }

    #[test]
    fn source_mapping() {
        assert_eq!(map_source(InfraSource::WallPower), EnergySource::Grid);
        assert_eq!(map_source(InfraSource::Battery), EnergySource::Battery);
        assert_eq!(map_source(InfraSource::Solar), EnergySource::Solar);
        assert_eq!(map_source(InfraSource::Unknown), EnergySource::Unknown);
    }
}
