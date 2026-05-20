use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for joules -> kWh -> kgCO2e conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonConfig {
    /// Grid carbon intensity in kgCO2e per kWh (default: 0.4 world average).
    pub grid_factor_kg_co2e_per_kwh: f64,
    /// Grid region identifier (e.g., "US-CAL-CISO", "DE", "FR").
    pub grid_region: String,
    /// Source of the carbon factor data (e.g., "electricity-maps-2025").
    pub grid_factor_source: String,
}

impl Default for CarbonConfig {
    fn default() -> Self {
        Self {
            grid_factor_kg_co2e_per_kwh: 0.4,
            grid_region: "UNKNOWN".to_string(),
            grid_factor_source: "default-world-average-2025".to_string(),
        }
    }
}

/// Convert joules to kilowatt-hours.
pub fn joules_to_kwh(joules: f64) -> f64 {
    joules / 3_600_000.0
}

/// Convert joules to kgCO2e using the configured grid factor.
pub fn joules_to_kg_co2e(joules: f64, config: &CarbonConfig) -> f64 {
    joules_to_kwh(joules) * config.grid_factor_kg_co2e_per_kwh
}

/// Convert joules to kgCO2e using a dynamic carbon intensity value.
///
/// Use this when the intensity is fetched from a live data source via
/// `CarbonIntensityCache` rather than a static `CarbonConfig`.
pub fn joules_to_kg_co2e_dynamic(joules: f64, carbon_intensity_kg_co2e_per_kwh: f64) -> f64 {
    joules_to_kwh(joules) * carbon_intensity_kg_co2e_per_kwh
}

// ---------------------------------------------------------------------------
// Pluggable carbon data source
// ---------------------------------------------------------------------------

/// Trait for pluggable carbon intensity data sources.
///
/// Implementations can fetch live grid intensity from external APIs
/// (e.g., Electricity Maps, WattTime, RTE France) or return a fixed value.
pub trait CarbonDataSource: Send + Sync {
    /// Human-readable name of this data source.
    fn name(&self) -> &str;

    /// Fetch the current carbon intensity (kgCO2e/kWh) for a grid region.
    ///
    /// Returns `None` if the source cannot provide data for this region.
    fn get_carbon_intensity(
        &self,
        grid_region: &str,
    ) -> impl std::future::Future<Output = Option<f64>> + Send;

    /// Attribution string for the data source.
    fn attribution(&self) -> &str;
}

/// Static carbon data source that returns a fixed value regardless of region.
pub struct StaticCarbonSource {
    intensity: f64,
    source_name: String,
    attribution_text: String,
}

impl StaticCarbonSource {
    /// Create a static source with a custom intensity value.
    pub fn new(intensity_kg_co2e_per_kwh: f64, name: &str, attribution: &str) -> Self {
        Self {
            intensity: intensity_kg_co2e_per_kwh,
            source_name: name.to_string(),
            attribution_text: attribution.to_string(),
        }
    }

    /// World-average static source (0.4 kgCO2e/kWh, IEA 2024 estimate).
    pub fn world_average() -> Self {
        Self {
            intensity: 0.4,
            source_name: "static-world-average".to_string(),
            attribution_text: "IEA World Energy Outlook 2024 global average".to_string(),
        }
    }
}

impl CarbonDataSource for StaticCarbonSource {
    fn name(&self) -> &str {
        &self.source_name
    }

    async fn get_carbon_intensity(&self, _grid_region: &str) -> Option<f64> {
        Some(self.intensity)
    }

    fn attribution(&self) -> &str {
        &self.attribution_text
    }
}

// ---------------------------------------------------------------------------
// Carbon intensity cache
// ---------------------------------------------------------------------------

/// Thread-safe cache for a live carbon intensity value.
///
/// The collector reads from this synchronously on every receipt. A background
/// task periodically calls `update()` with fresh data from a `CarbonDataSource`.
#[derive(Clone)]
pub struct CarbonIntensityCache {
    value: Arc<RwLock<f64>>,
}

impl CarbonIntensityCache {
    /// Create a new cache with the given initial intensity.
    pub fn new(initial_intensity: f64) -> Self {
        Self {
            value: Arc::new(RwLock::new(initial_intensity)),
        }
    }

    /// Get the current cached intensity (kgCO2e/kWh).
    pub async fn get(&self) -> f64 {
        *self.value.read().await
    }

    /// Update the cached intensity value.
    pub async fn update(&self, new_intensity: f64) {
        *self.value.write().await = new_intensity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kwh_conversion() {
        // 1 kWh = 3,600,000 J
        let kwh = joules_to_kwh(3_600_000.0);
        assert!((kwh - 1.0).abs() < 1e-10);
    }

    #[test]
    fn co2e_default_factor() {
        let config = CarbonConfig::default();
        // 1 kWh at 0.4 kgCO2e/kWh = 0.4 kgCO2e
        let co2 = joules_to_kg_co2e(3_600_000.0, &config);
        assert!((co2 - 0.4).abs() < 1e-10);
    }

    #[test]
    fn zero_joules() {
        let config = CarbonConfig::default();
        assert_eq!(joules_to_kwh(0.0), 0.0);
        assert_eq!(joules_to_kg_co2e(0.0, &config), 0.0);
    }

    #[test]
    fn custom_grid_factor() {
        let config = CarbonConfig {
            grid_factor_kg_co2e_per_kwh: 0.05, // France (nuclear-heavy)
            grid_region: "FR".to_string(),
            grid_factor_source: "rte-france-2025".to_string(),
        };
        let co2 = joules_to_kg_co2e(3_600_000.0, &config);
        assert!((co2 - 0.05).abs() < 1e-10);
    }

    #[tokio::test]
    async fn static_source_returns_value() {
        let source = StaticCarbonSource::new(0.25, "test", "test source");
        assert_eq!(source.get_carbon_intensity("US-CAL").await, Some(0.25));
        assert_eq!(source.get_carbon_intensity("DE").await, Some(0.25));
        assert_eq!(source.name(), "test");
        assert_eq!(source.attribution(), "test source");
    }

    #[tokio::test]
    async fn world_average_is_0_4() {
        let source = StaticCarbonSource::world_average();
        let intensity = source.get_carbon_intensity("any").await.unwrap();
        assert!((intensity - 0.4).abs() < 1e-10);
    }

    #[test]
    fn dynamic_conversion_matches_static() {
        let config = CarbonConfig {
            grid_factor_kg_co2e_per_kwh: 0.3,
            ..Default::default()
        };
        let static_result = joules_to_kg_co2e(7_200_000.0, &config);
        let dynamic_result = joules_to_kg_co2e_dynamic(7_200_000.0, 0.3);
        assert!((static_result - dynamic_result).abs() < 1e-15);
    }

    #[tokio::test]
    async fn cache_update_reflected() {
        let cache = CarbonIntensityCache::new(0.4);
        assert!((cache.get().await - 0.4).abs() < 1e-10);

        cache.update(0.15).await;
        assert!((cache.get().await - 0.15).abs() < 1e-10);
    }
}
