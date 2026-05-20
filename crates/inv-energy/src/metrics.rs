use inv_telemetry::histogram::energy_buckets;
use inv_telemetry::{Counter, Gauge, Histogram, MetricsRegistry};

/// Metrics for the energy measurement subsystem.
///
/// Tracks energy readings, power draw, cumulative consumption,
/// per-invocation energy, budget utilization, carbon intensity,
/// battery state, and thermal conditions.
#[derive(Debug, Clone)]
pub struct EnergyMetrics {
    pub readings_total: Counter,
    pub current_watts: Gauge,
    pub cumulative_joules: Gauge,
    pub joules_per_invocation: Histogram,
    pub budget_utilization: Gauge,
    pub carbon_intensity: Gauge,
    pub battery_percent: Gauge,
    pub thermal_throttle_events: Counter,
    pub meter_errors: Counter,
}

impl EnergyMetrics {
    /// Register all energy metrics in the given registry.
    pub fn new(registry: &MetricsRegistry) -> Self {
        Self {
            readings_total: registry
                .counter("energy_readings_total", "Total energy meter readings taken"),
            current_watts: registry.gauge("energy_current_watts", "Current power draw in watts"),
            cumulative_joules: registry.gauge(
                "energy_cumulative_joules",
                "Total energy consumed in joules",
            ),
            joules_per_invocation: registry.histogram_with_buckets(
                "energy_joules_per_invocation",
                "Energy consumed per invocation in microjoules",
                &energy_buckets(),
            ),
            budget_utilization: registry.gauge(
                "energy_budget_utilization",
                "Fraction of energy budget consumed (0.0-1.0)",
            ),
            carbon_intensity: registry.gauge(
                "energy_carbon_intensity_gco2_kwh",
                "Current carbon intensity in gCO2/kWh",
            ),
            battery_percent: registry.gauge(
                "energy_battery_percent",
                "Battery charge percentage (0-100)",
            ),
            thermal_throttle_events: registry.counter(
                "energy_thermal_throttle_events_total",
                "Thermal throttling events detected",
            ),
            meter_errors: registry.counter("energy_meter_errors_total", "Energy meter read errors"),
        }
    }

    /// Record an energy reading.
    pub fn record_reading(&self, watts: f64) {
        self.readings_total.inc();
        self.current_watts.set(watts);
    }

    /// Update cumulative energy consumption.
    pub fn set_cumulative_joules(&self, joules: f64) {
        self.cumulative_joules.set(joules);
    }

    /// Record per-invocation energy usage.
    pub fn record_invocation_energy(&self, microjoules: f64) {
        self.joules_per_invocation.observe(microjoules);
    }

    /// Update budget utilization (0.0 to 1.0).
    pub fn set_budget_utilization(&self, fraction: f64) {
        self.budget_utilization.set(fraction);
    }

    /// Update carbon intensity.
    pub fn set_carbon_intensity(&self, gco2_per_kwh: f64) {
        self.carbon_intensity.set(gco2_per_kwh);
    }

    /// Update battery percentage.
    pub fn set_battery_percent(&self, percent: f64) {
        self.battery_percent.set(percent);
    }

    /// Record a thermal throttle event.
    pub fn record_thermal_throttle(&self) {
        self.thermal_throttle_events.inc();
    }

    /// Record a meter read error.
    pub fn record_meter_error(&self) {
        self.meter_errors.inc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics() -> EnergyMetrics {
        let reg = MetricsRegistry::new();
        EnergyMetrics::new(&reg)
    }

    #[test]
    fn initial_values_zero() {
        let m = metrics();
        assert_eq!(m.readings_total.get(), 0);
        assert!((m.current_watts.get() - 0.0).abs() < 0.01);
        assert!((m.cumulative_joules.get() - 0.0).abs() < 0.01);
    }

    #[test]
    fn record_reading() {
        let m = metrics();
        m.record_reading(42.5);
        assert_eq!(m.readings_total.get(), 1);
        assert!((m.current_watts.get() - 42.5).abs() < 0.01);
    }

    #[test]
    fn cumulative_joules() {
        let m = metrics();
        m.set_cumulative_joules(1000.0);
        assert!((m.cumulative_joules.get() - 1000.0).abs() < 0.01);
    }

    #[test]
    fn invocation_energy() {
        let m = metrics();
        m.record_invocation_energy(500.0);
        m.record_invocation_energy(1500.0);
        assert_eq!(m.joules_per_invocation.count(), 2);
    }

    #[test]
    fn budget_utilization() {
        let m = metrics();
        m.set_budget_utilization(0.75);
        assert!((m.budget_utilization.get() - 0.75).abs() < 0.01);
    }

    #[test]
    fn carbon_intensity() {
        let m = metrics();
        m.set_carbon_intensity(200.0);
        assert!((m.carbon_intensity.get() - 200.0).abs() < 0.01);
    }

    #[test]
    fn battery_percent() {
        let m = metrics();
        m.set_battery_percent(85.0);
        assert!((m.battery_percent.get() - 85.0).abs() < 0.01);
    }

    #[test]
    fn thermal_throttle() {
        let m = metrics();
        m.record_thermal_throttle();
        m.record_thermal_throttle();
        assert_eq!(m.thermal_throttle_events.get(), 2);
    }

    #[test]
    fn meter_errors() {
        let m = metrics();
        m.record_meter_error();
        assert_eq!(m.meter_errors.get(), 1);
    }

    #[test]
    fn metrics_registered_in_registry() {
        let reg = MetricsRegistry::new();
        let _m = EnergyMetrics::new(&reg);
        // 3 counters + 5 gauges + 1 histogram = 9
        assert_eq!(reg.counter_count(), 3);
        assert_eq!(reg.gauge_count(), 5);
        assert_eq!(reg.histogram_count(), 1);
    }
}
