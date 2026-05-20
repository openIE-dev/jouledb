//! Carbon intensity fusion engine.
//!
//! Instead of treating multiple data sources as a fallback chain (try A, if
//! fail try B), the fusion engine treats every data source as a **sensor**
//! producing observations with different precision, recency, and geographic
//! scope. All available observations contribute to the estimate, weighted by
//! quality.
//!
//! **Key capabilities:**
//! - **Weighted fusion**: every observation contributes proportional to
//!   freshness × geographic precision × source reliability.
//! - **Temporal patterns**: extracts diurnal (hour-of-day) and weekly
//!   (day-of-week) intensity patterns from observation history.
//! - **Spatial interpolation**: borrows real-time data from electrically
//!   interconnected neighbor zones to improve estimates for zones with
//!   only static or low-frequency data.
//! - **Forecasting**: combines current fused intensity with temporal patterns
//!   to predict carbon intensity windows over the next 24 hours.
//! - **Confidence bounds**: every estimate carries a 0–1 confidence score
//!   reflecting source diversity, freshness, and agreement.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use chrono::{DateTime, Datelike, Duration as ChronoDuration, Timelike, Utc};
use serde::{Deserialize, Serialize};

use crate::carbon::CarbonIntensity;
use crate::carbon_aggregator::{
    AggregatorConfig, CarbonBackend, build_backends, country_grid_mix, zone_to_country,
};
use crate::carbon_api::{CarbonApiError, cloud_zone_mapping};

// ---------------------------------------------------------------------------
// Observation model
// ---------------------------------------------------------------------------

/// Which backend produced this observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceKind {
    ElectricityMaps,
    UkCarbon,
    Eia,
    Entsoe,
    OpenNem,
    Ember,
    Epias,
    Cen,
    Ons,
    Cammesa,
    Niggrid,
    Csep,
    IpccStatic,
}

impl SourceKind {
    /// Intrinsic reliability score (0–1) reflecting data quality.
    pub fn reliability(self) -> f64 {
        match self {
            Self::ElectricityMaps => 0.95,
            Self::UkCarbon => 0.95,
            Self::Eia => 0.90,
            Self::Entsoe => 0.90,
            Self::OpenNem => 0.90,
            Self::Epias => 0.90,
            Self::Cen => 0.90,
            Self::Ons => 0.75,
            Self::Cammesa => 0.75,
            Self::Niggrid => 0.75,
            Self::Csep => 0.80,
            Self::Ember => 0.70,
            Self::IpccStatic => 0.50,
        }
    }

    /// Map backend name (from CarbonBackend::name()) to SourceKind.
    pub fn from_backend_name(name: &str) -> Self {
        match name {
            "electricity-maps" => Self::ElectricityMaps,
            "uk-carbon" | "uk-carbon-intensity" => Self::UkCarbon,
            "eia" => Self::Eia,
            "entsoe" => Self::Entsoe,
            "opennem" => Self::OpenNem,
            "ember" => Self::Ember,
            "epias" => Self::Epias,
            "cen" => Self::Cen,
            "ons" => Self::Ons,
            "cammesa" => Self::Cammesa,
            "niggrid" => Self::Niggrid,
            "csep" => Self::Csep,
            _ => Self::IpccStatic,
        }
    }

    /// Typical measurement granularity for this source.
    pub fn granularity_secs(self) -> u64 {
        match self {
            Self::ElectricityMaps => 3600,   // hourly
            Self::UkCarbon => 1800,          // 30 min
            Self::Eia => 3600,               // hourly
            Self::Entsoe => 900,             // 15 min
            Self::OpenNem => 300,            // 5 min
            Self::Epias => 3600,             // hourly
            Self::Cen => 3600,               // hourly
            Self::Ons => 600,                // ~10 min
            Self::Cammesa => 600,            // ~10 min
            Self::Niggrid => 600,            // ~10 min
            Self::Csep => 300,               // ~5 min
            Self::Ember => 30 * 86400,       // monthly
            Self::IpccStatic => 365 * 86400, // yearly/static
        }
    }
}

/// Geographic scope of the observation relative to the target zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeoScope {
    /// Direct match for the exact zone (e.g., UK Carbon for GB).
    ExactZone,
    /// Country-level average (e.g., Ember monthly for France).
    Country,
    /// Global average fallback (e.g., IPCC world average).
    Global,
}

impl GeoScope {
    /// Precision weight factor.
    pub fn precision(self) -> f64 {
        match self {
            Self::ExactZone => 1.0,
            Self::Country => 0.3,
            Self::Global => 0.05,
        }
    }
}

/// A single observation from a carbon data source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub zone: String,
    pub intensity_gco2_kwh: f64,
    pub renewable_pct: f64,
    pub fossil_pct: f64,
    pub observed_at: DateTime<Utc>,
    pub source: SourceKind,
    pub granularity_secs: u64,
    pub geo_scope: GeoScope,
}

// ---------------------------------------------------------------------------
// Observation store (time-series ring buffer per zone)
// ---------------------------------------------------------------------------

/// Time-series storage of observations per zone.
pub struct ObservationStore {
    zones: HashMap<String, VecDeque<Observation>>,
    max_age: ChronoDuration,
}

impl ObservationStore {
    pub fn new(max_age: ChronoDuration) -> Self {
        Self {
            zones: HashMap::new(),
            max_age,
        }
    }

    /// Record a new observation, evicting any that exceed max_age.
    pub fn record(&mut self, obs: Observation) {
        let cutoff = Utc::now() - self.max_age;
        let buf = self.zones.entry(obs.zone.clone()).or_default();
        buf.push_front(obs);
        // Evict old entries from the back.
        while let Some(oldest) = buf.back() {
            if oldest.observed_at < cutoff {
                buf.pop_back();
            } else {
                break;
            }
        }
    }

    /// All observations for a zone within the given duration from now.
    pub fn recent(&self, zone: &str, within: ChronoDuration) -> Vec<&Observation> {
        let cutoff = Utc::now() - within;
        self.zones
            .get(zone)
            .map(|buf| buf.iter().filter(|o| o.observed_at >= cutoff).collect())
            .unwrap_or_default()
    }

    /// Full observation history for a zone (for pattern extraction).
    pub fn history(&self, zone: &str) -> Option<&VecDeque<Observation>> {
        self.zones.get(zone)
    }

    /// All zone keys with at least one observation.
    pub fn all_zones(&self) -> Vec<&str> {
        self.zones.keys().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Weight calculation
// ---------------------------------------------------------------------------

/// Tuning parameters for observation weighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightConfig {
    /// Half-life for temporal freshness decay in seconds.
    /// At this age, an observation's freshness factor is 0.5.
    pub freshness_halflife_secs: f64,
}

impl Default for WeightConfig {
    fn default() -> Self {
        Self {
            freshness_halflife_secs: 1800.0, // 30 minutes
        }
    }
}

/// Compute the composite weight of an observation.
///
/// `w = freshness × precision × reliability`
///
/// Freshness uses exponential decay: `0.5 ^ (age_secs / halflife)`.
pub fn observation_weight(obs: &Observation, now: DateTime<Utc>, config: &WeightConfig) -> f64 {
    let age_secs = (now - obs.observed_at).num_seconds().max(0) as f64;

    // Exponential decay: half-life model.
    let freshness = (0.5_f64).powf(age_secs / config.freshness_halflife_secs);

    let precision = obs.geo_scope.precision();
    let reliability = obs.source.reliability();

    freshness * precision * reliability
}

// ---------------------------------------------------------------------------
// Fused estimate
// ---------------------------------------------------------------------------

/// Result of fusing all available observations for a zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedEstimate {
    pub zone: String,
    /// Weighted mean carbon intensity (gCO2eq/kWh).
    pub intensity_gco2_kwh: f64,
    /// Weighted mean renewable percentage (0–100).
    pub renewable_pct: f64,
    /// Weighted mean fossil percentage (0–100).
    pub fossil_pct: f64,
    /// Confidence score (0.0–1.0).
    pub confidence: f64,
    /// Number of observations that contributed.
    pub observation_count: usize,
    /// Source kind of the highest-weight observation.
    pub freshest_source: SourceKind,
    pub timestamp: DateTime<Utc>,
}

/// Fuse a set of observations into a single weighted estimate.
///
/// Each observation contributes proportional to its weight. The confidence
/// score reflects source diversity, total weight, and agreement between
/// observations.
pub fn fuse(zone: &str, observations: &[(&Observation, f64)]) -> FusedEstimate {
    let now = Utc::now();

    if observations.is_empty() {
        // No observations at all — return IPCC global fallback.
        let mix = country_grid_mix(zone_to_country(zone));
        return FusedEstimate {
            zone: zone.to_string(),
            intensity_gco2_kwh: mix.carbon_intensity(),
            renewable_pct: mix.renewable_pct(),
            fossil_pct: mix.fossil_pct(),
            confidence: 0.1,
            observation_count: 0,
            freshest_source: SourceKind::IpccStatic,
            timestamp: now,
        };
    }

    let total_weight: f64 = observations.iter().map(|(_, w)| w).sum();

    if total_weight <= 0.0 {
        let mix = country_grid_mix(zone_to_country(zone));
        return FusedEstimate {
            zone: zone.to_string(),
            intensity_gco2_kwh: mix.carbon_intensity(),
            renewable_pct: mix.renewable_pct(),
            fossil_pct: mix.fossil_pct(),
            confidence: 0.1,
            observation_count: observations.len(),
            freshest_source: SourceKind::IpccStatic,
            timestamp: now,
        };
    }

    // Weighted means.
    let intensity = observations
        .iter()
        .map(|(o, w)| o.intensity_gco2_kwh * w)
        .sum::<f64>()
        / total_weight;

    let renewable = observations
        .iter()
        .map(|(o, w)| o.renewable_pct * w)
        .sum::<f64>()
        / total_weight;

    let fossil = observations
        .iter()
        .map(|(o, w)| o.fossil_pct * w)
        .sum::<f64>()
        / total_weight;

    // Freshest source = highest weight.
    let freshest_source = observations
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(o, _)| o.source)
        .unwrap_or(SourceKind::IpccStatic);

    // Confidence: combines source diversity, total weight, and agreement.
    let confidence = compute_confidence(observations, intensity, total_weight);

    FusedEstimate {
        zone: zone.to_string(),
        intensity_gco2_kwh: intensity,
        renewable_pct: renewable,
        fossil_pct: fossil,
        confidence,
        observation_count: observations.len(),
        freshest_source,
        timestamp: now,
    }
}

/// Compute confidence score from observation quality.
///
/// Factors:
/// 1. Total weight (capped contribution at 0.4): higher total weight → more data.
/// 2. Source diversity (up to 0.3): more distinct sources → more corroboration.
/// 3. Agreement (up to 0.3): low variance among observations → higher confidence.
fn compute_confidence(
    observations: &[(&Observation, f64)],
    mean_intensity: f64,
    total_weight: f64,
) -> f64 {
    // 1. Weight factor: sigmoid-like, saturates at ~0.4 for total_weight ≥ 1.0.
    let weight_factor = 0.4 * (1.0 - (-total_weight * 2.0).exp());

    // 2. Diversity: count distinct source kinds.
    let mut sources = std::collections::HashSet::new();
    for (obs, _) in observations {
        sources.insert(obs.source);
    }
    let diversity = (sources.len() as f64 / 13.0).min(1.0) * 0.3;

    // 3. Agreement: weighted coefficient of variation (lower CV → higher agreement).
    let agreement = if mean_intensity > 0.0 && observations.len() > 1 {
        let weighted_var: f64 = observations
            .iter()
            .map(|(o, w)| w * (o.intensity_gco2_kwh - mean_intensity).powi(2))
            .sum::<f64>()
            / total_weight;
        let cv = weighted_var.sqrt() / mean_intensity;
        // CV of 0 → agreement 0.3, CV of 0.5+ → agreement ~0.
        0.3 * (1.0 - (cv * 2.0).min(1.0))
    } else {
        0.15 // neutral when we can't compute
    };

    (weight_factor + diversity + agreement).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Temporal model (diurnal + weekly patterns)
// ---------------------------------------------------------------------------

/// Extracted intensity patterns from observation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalProfile {
    /// Intensity ratio per hour of day (0–23) relative to daily mean.
    /// e.g., `diurnal[14] = 0.85` means 15% below average at 2 PM.
    pub diurnal: [f64; 24],
    /// Intensity ratio per day of week (0=Mon..6=Sun).
    pub weekly: [f64; 7],
    /// Number of observations used to build this profile.
    pub sample_count: usize,
}

impl Default for TemporalProfile {
    fn default() -> Self {
        Self {
            diurnal: [1.0; 24],
            weekly: [1.0; 7],
            sample_count: 0,
        }
    }
}

impl TemporalProfile {
    /// Build a temporal profile from observation history.
    ///
    /// Groups observations by hour-of-day and day-of-week, computing the
    /// average intensity for each bucket as a ratio to the overall average.
    pub fn from_history(observations: &VecDeque<Observation>) -> Self {
        if observations.is_empty() {
            return Self::default();
        }

        // Accumulate intensity per hour and per day-of-week.
        let mut hour_sum = [0.0_f64; 24];
        let mut hour_count = [0u32; 24];
        let mut dow_sum = [0.0_f64; 7];
        let mut dow_count = [0u32; 7];
        let mut overall_sum = 0.0_f64;
        let mut overall_count = 0u32;

        for obs in observations {
            let h = obs.observed_at.hour() as usize;
            let d = obs.observed_at.weekday().num_days_from_monday() as usize;
            let v = obs.intensity_gco2_kwh;

            hour_sum[h] += v;
            hour_count[h] += 1;
            dow_sum[d] += v;
            dow_count[d] += 1;
            overall_sum += v;
            overall_count += 1;
        }

        let overall_avg = if overall_count > 0 {
            overall_sum / overall_count as f64
        } else {
            1.0
        };

        let mut diurnal = [1.0_f64; 24];
        for h in 0..24 {
            if hour_count[h] > 0 && overall_avg > 0.0 {
                diurnal[h] = (hour_sum[h] / hour_count[h] as f64) / overall_avg;
            }
        }

        let mut weekly = [1.0_f64; 7];
        for d in 0..7 {
            if dow_count[d] > 0 && overall_avg > 0.0 {
                weekly[d] = (dow_sum[d] / dow_count[d] as f64) / overall_avg;
            }
        }

        Self {
            diurnal,
            weekly,
            sample_count: overall_count as usize,
        }
    }

    /// Predict intensity at a future time given a current base intensity.
    ///
    /// `predicted = base_intensity × diurnal[hour] × weekly[day]`
    /// (Normalized so factors cancel at the current time.)
    pub fn predict(&self, base_intensity: f64, current: DateTime<Utc>, at: DateTime<Utc>) -> f64 {
        let current_factor = self.diurnal[current.hour() as usize]
            * self.weekly[current.weekday().num_days_from_monday() as usize];

        let target_factor = self.diurnal[at.hour() as usize]
            * self.weekly[at.weekday().num_days_from_monday() as usize];

        if current_factor > 0.0 {
            base_intensity * (target_factor / current_factor)
        } else {
            base_intensity * target_factor
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial model (grid interconnection topology)
// ---------------------------------------------------------------------------

/// Electrical grid interconnection graph for spatial interpolation.
pub struct SpatialModel {
    /// zone → [(neighbor_zone, coupling_weight)]
    neighbors: HashMap<&'static str, Vec<(&'static str, f64)>>,
}

impl SpatialModel {
    /// Build the default grid topology (European interconnections + US RTO
    /// adjacency + Australian NEM).
    pub fn default_grid() -> Self {
        let mut neighbors: HashMap<&'static str, Vec<(&'static str, f64)>> = HashMap::new();

        // Helper: add a bidirectional edge.
        let mut link = |a: &'static str, b: &'static str, w: f64| {
            neighbors.entry(a).or_default().push((b, w));
            neighbors.entry(b).or_default().push((a, w));
        };

        // European interconnections (coupling weight ≈ transfer capacity / peak demand).
        link("DE", "FR", 0.6);
        link("DE", "NL", 0.7);
        link("DE", "PL", 0.5);
        link("DE", "AT", 0.8);
        link("DE", "CH", 0.6);
        link("DE", "DK", 0.5);
        link("DE", "CZ", 0.5);
        link("DE", "BE", 0.4);
        link("FR", "ES", 0.5);
        link("FR", "BE", 0.6);
        link("FR", "IT-NO", 0.4);
        link("FR", "GB", 0.3); // HVDC IFA/IFA2
        link("FR", "CH", 0.6);
        link("GB", "IE", 0.4); // EWIC/Moyle
        link("GB", "NL", 0.3); // BritNed HVDC
        link("GB", "NO", 0.3); // North Sea Link HVDC
        link("GB", "BE", 0.3); // Nemo Link HVDC
        link("NO", "SE", 0.8);
        link("NO", "DK", 0.6);
        link("SE", "FI", 0.6);
        link("SE", "DK", 0.5);
        link("SE", "PL", 0.3); // SwePol
        link("DK", "NL", 0.3); // COBRAcable
        link("NL", "BE", 0.5);
        link("BE", "LU", 0.5);
        link("AT", "CH", 0.5);
        link("AT", "IT-NO", 0.4);
        link("AT", "HU", 0.4);
        link("AT", "CZ", 0.5);
        link("AT", "SI", 0.4);
        link("IT-NO", "SI", 0.3);
        link("PL", "CZ", 0.4);
        link("PL", "SK", 0.3);
        link("PL", "LT", 0.3); // LitPol
        link("CZ", "SK", 0.5);
        link("SK", "HU", 0.4);
        link("HU", "RO", 0.3);
        link("HU", "HR", 0.3);
        link("RO", "BG", 0.4);
        link("BG", "GR", 0.3);
        link("HR", "SI", 0.4);
        link("EE", "LV", 0.6);
        link("LV", "LT", 0.6);
        link("EE", "FI", 0.4); // EstLink
        link("LT", "SE", 0.3); // NordBalt
        link("ES", "PT", 0.6);
        link("GR", "IT-NO", 0.2);

        // US RTO adjacency.
        link("US-MIDA-PJM", "US-NY-NYIS", 0.7);
        link("US-MIDA-PJM", "US-SE-SOCO", 0.5);
        link("US-MIDA-PJM", "US-NE-ISNE", 0.5);
        link("US-MIDA-PJM", "US-MIDW-MISO", 0.5);
        link("US-MIDA-PJM", "US-TEN-TVA", 0.4);
        link("US-NY-NYIS", "US-NE-ISNE", 0.6);
        link("US-MIDW-MISO", "US-TEX-ERCO", 0.3);
        link("US-MIDW-MISO", "US-SW-SRP", 0.3);
        link("US-CAL-CISO", "US-NW-BPAT", 0.5);
        link("US-CAL-CISO", "US-SW-AZPS", 0.6);
        link("US-CAL-CISO", "US-SW-SRP", 0.4);
        link("US-NW-BPAT", "US-NW-PACW", 0.6);
        link("US-SW-AZPS", "US-SW-SRP", 0.5);
        link("US-SW-AZPS", "US-SW-EPE", 0.4);
        link("US-SE-SOCO", "US-FLA-FPL", 0.4);
        link("US-SE-SOCO", "US-TEN-TVA", 0.5);
        link("US-SE-SOCO", "US-SE-AEC", 0.5);

        // Australian NEM.
        link("AU-NSW", "AU-VIC", 0.7);
        link("AU-NSW", "AU-QLD", 0.6);
        link("AU-VIC", "AU-SA", 0.5);
        link("AU-VIC", "AU-TAS", 0.4); // Basslink

        // --- Africa ---
        link("MA", "ES", 0.4); // Morocco-Spain HVDC
        link("MA", "DZ", 0.3);
        link("DZ", "TN", 0.5);
        link("EG", "JO", 0.3); // Egypt-Jordan interconnector
        link("KE", "ET", 0.4); // Eastern Africa Power Pool
        link("KE", "TZ", 0.4);
        link("NG", "GH", 0.3); // West African Power Pool

        // --- South America ---
        link("AR", "UY", 0.6);
        link("AR", "CL", 0.4);
        link("AR", "PY", 0.5); // Yaciretá
        link("AR", "BO", 0.3);
        link("BR-S", "AR", 0.4);
        link("BR-S", "PY", 0.5); // Itaipú
        link("BR-S", "UY", 0.4);
        link("CO", "EC", 0.4);
        link("CO", "VE", 0.3);
        link("PE", "EC", 0.3);
        link("PE", "BO", 0.2);
        link("PE", "CL", 0.2);
        link("MX", "US-TEX-ERCO", 0.2); // Mexico-US HVDC ties

        // --- Middle East (GCC interconnector + neighbors) ---
        link("SA", "AE", 0.5); // GCC interconnector
        link("SA", "KW", 0.4);
        link("SA", "QA", 0.3);
        link("SA", "JO", 0.3);
        link("SA", "BH", 0.4);
        link("AE", "OM", 0.5);
        link("KW", "IQ", 0.3);
        link("TR", "GR", 0.4); // Turkey-Greece interconnector
        link("TR", "BG", 0.3);
        link("TR", "IQ", 0.2);
        link("TR", "IR", 0.2);
        link("JO", "IQ", 0.2);

        // --- Southeast Asia ---
        link("SG", "MY", 0.6); // Singapore-Malaysia interconnector
        link("MY", "TH", 0.5); // HVDC Malaysia-Thailand
        link("TH", "MM", 0.3);
        link("ID", "MY", 0.3); // Borneo interconnector
        link("PK", "IN-WE", 0.2);
        link("BD", "IN-WE", 0.3); // India-Bangladesh

        // --- South Asia to existing ---
        link("IN-WE", "IN-SO", 0.7); // internal India interconnection (strengthen existing)

        Self { neighbors }
    }

    /// Get neighbor observations from the store, each attenuated by the
    /// coupling weight. Returns (effective_weight_multiplier, observation).
    pub fn neighbor_observations<'a>(
        &self,
        zone: &str,
        store: &'a ObservationStore,
        within: ChronoDuration,
    ) -> Vec<(f64, &'a Observation)> {
        let mut result = Vec::new();
        if let Some(neighbors) = self.neighbors.get(zone) {
            for &(neighbor_zone, coupling) in neighbors {
                for obs in store.recent(neighbor_zone, within) {
                    // Neighbor data: downgrade precision to Country-equivalent
                    // and attenuate by coupling weight.
                    result.push((coupling * GeoScope::Country.precision(), obs));
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Forecast
// ---------------------------------------------------------------------------

/// A predicted carbon intensity window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastWindow {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub intensity_gco2_kwh: f64,
    pub confidence: f64,
}

/// Forecast of carbon intensity over a future time horizon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonForecast {
    pub zone: String,
    pub generated_at: DateTime<Utc>,
    pub windows: Vec<ForecastWindow>,
}

impl CarbonForecast {
    /// Find the window with the lowest predicted intensity.
    pub fn lowest_intensity_window(&self) -> Option<&ForecastWindow> {
        self.windows.iter().min_by(|a, b| {
            a.intensity_gco2_kwh
                .partial_cmp(&b.intensity_gco2_kwh)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ---------------------------------------------------------------------------
// Fusion config
// ---------------------------------------------------------------------------

/// Configuration for the carbon fusion engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    pub weight_config: WeightConfig,
    /// How long to retain observations (default 7 days).
    pub observation_retention_secs: u64,
    /// Forecast horizon (default 24 hours).
    pub forecast_horizon_secs: u64,
    /// Forecast window size (default 30 minutes).
    pub forecast_resolution_secs: u64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            weight_config: WeightConfig::default(),
            observation_retention_secs: 7 * 86400,
            forecast_horizon_secs: 24 * 3600,
            forecast_resolution_secs: 1800,
        }
    }
}

// ---------------------------------------------------------------------------
// CarbonFusionEngine
// ---------------------------------------------------------------------------

/// Sensor fusion engine for carbon intensity estimation.
///
/// Collects observations from all available backends, stores them in a
/// time-series ring buffer, and produces fused estimates using weighted
/// combination with temporal pattern overlays and spatial interpolation.
pub struct CarbonFusionEngine {
    backends: Vec<Box<dyn CarbonBackend>>,
    store: Mutex<ObservationStore>,
    spatial: SpatialModel,
    config: FusionConfig,
}

impl std::fmt::Debug for CarbonFusionEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonFusionEngine")
            .field("backends", &self.backends.len())
            .field("config", &self.config)
            .finish()
    }
}

impl CarbonFusionEngine {
    /// Create a fusion engine from aggregator config (builds the same backends)
    /// and fusion-specific config.
    pub fn new(agg_config: AggregatorConfig, fusion_config: FusionConfig) -> Self {
        let backends = build_backends(&agg_config);
        let retention = ChronoDuration::seconds(fusion_config.observation_retention_secs as i64);
        Self {
            backends,
            store: Mutex::new(ObservationStore::new(retention)),
            spatial: SpatialModel::default_grid(),
            config: fusion_config,
        }
    }

    /// Create a minimal engine (IPCC-only) for testing or zero-config setups.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(AggregatorConfig::default(), FusionConfig::default())
    }

    /// Poll **all** backends that support this zone and record observations.
    ///
    /// Unlike the fallback aggregator, this queries every backend in parallel
    /// and records all successful responses. Failures are logged and tolerated.
    pub async fn collect(&self, zone: &str) {
        for backend in &self.backends {
            if !backend.supports_zone(zone) {
                continue;
            }
            // We can't easily spawn because CarbonBackend isn't 'static,
            // so we query sequentially per zone. In practice, collect_all()
            // parallelizes across zones.
            let name = backend.name().to_string();
            match backend.carbon_intensity(zone).await {
                Ok(resp) => {
                    let source = SourceKind::from_backend_name(&name);
                    let geo_scope = classify_geo_scope(&name, zone);
                    let obs = Observation {
                        zone: zone.to_string(),
                        intensity_gco2_kwh: resp.carbon_intensity_gco2_kwh,
                        renewable_pct: 100.0 - resp.fossil_fuel_percentage,
                        fossil_pct: resp.fossil_fuel_percentage,
                        observed_at: resp.datetime,
                        source,
                        granularity_secs: source.granularity_secs(),
                        geo_scope,
                    };
                    let mut store = self.store.lock().unwrap();
                    store.record(obs);
                }
                Err(e) => {
                    tracing::debug!(
                        backend = name,
                        zone,
                        error = %e,
                        "fusion: backend query failed, continuing"
                    );
                }
            }

            // Also collect power breakdown for renewable/fossil detail.
            match backend.power_breakdown(zone).await {
                Ok(resp) => {
                    let source = SourceKind::from_backend_name(&name);
                    let geo_scope = classify_geo_scope(&name, zone);
                    // If we already recorded carbon intensity, update with more
                    // precise renewable/fossil data from breakdown.
                    let obs = Observation {
                        zone: zone.to_string(),
                        // Use the IPCC-weighted intensity from grid mix as a
                        // cross-check; the carbon_intensity call is primary.
                        intensity_gco2_kwh: estimate_intensity_from_breakdown(&resp),
                        renewable_pct: resp.renewable_percentage,
                        fossil_pct: resp.fossil_percentage,
                        observed_at: resp.datetime,
                        source,
                        granularity_secs: source.granularity_secs(),
                        geo_scope,
                    };
                    let mut store = self.store.lock().unwrap();
                    store.record(obs);
                }
                Err(_) => {
                    // Power breakdown is supplementary; failure is fine.
                }
            }
        }
    }

    /// Produce a fused point estimate for a zone.
    ///
    /// Combines direct observations, spatial neighbor observations, and
    /// IPCC static baseline, weighted by freshness × precision × reliability.
    pub fn estimate(&self, zone: &str) -> FusedEstimate {
        let store = self.store.lock().unwrap();
        let now = Utc::now();
        let config = &self.config.weight_config;

        // 1. Direct observations (all time, weighted by freshness).
        let retention = ChronoDuration::seconds(self.config.observation_retention_secs as i64);
        let direct: Vec<(&Observation, f64)> = store
            .recent(zone, retention)
            .into_iter()
            .map(|obs| {
                let w = observation_weight(obs, now, config);
                (obs, w)
            })
            .collect();

        // 2. Spatial neighbor observations (last 2 hours).
        let neighbor_horizon = ChronoDuration::hours(2);
        let spatial_obs = self
            .spatial
            .neighbor_observations(zone, &store, neighbor_horizon);
        let spatial: Vec<(&Observation, f64)> = spatial_obs
            .into_iter()
            .map(|(coupling_discount, obs)| {
                let base_weight = observation_weight(obs, now, config);
                // Coupling discount replaces the geo_scope precision factor
                // since we already computed the base weight with the obs's own
                // geo_scope. Multiply by coupling to attenuate.
                (
                    obs,
                    base_weight * coupling_discount / obs.geo_scope.precision(),
                )
            })
            .collect();

        // 3. Always include IPCC static baseline (low weight, always present).
        let ipcc_mix = country_grid_mix(zone_to_country(zone));
        let ipcc_obs = Observation {
            zone: zone.to_string(),
            intensity_gco2_kwh: ipcc_mix.carbon_intensity(),
            renewable_pct: ipcc_mix.renewable_pct(),
            fossil_pct: ipcc_mix.fossil_pct(),
            observed_at: now,
            source: SourceKind::IpccStatic,
            granularity_secs: 365 * 86400,
            geo_scope: GeoScope::Country,
        };
        let ipcc_weight = GeoScope::Country.precision() * SourceKind::IpccStatic.reliability();

        // Combine all observations.
        let mut all: Vec<(&Observation, f64)> = Vec::new();
        all.extend(direct);
        all.extend(spatial);
        // We can't borrow ipcc_obs through the combined vec easily, so just
        // inline its contribution into the fuse call.
        // Instead, we'll add it as a temporary.
        let all_with_ipcc: Vec<(&Observation, f64)> = {
            let mut v = all;
            v.push((&ipcc_obs, ipcc_weight));
            v
        };

        fuse(zone, &all_with_ipcc)
    }

    /// Produce a 24-hour forecast for a zone.
    ///
    /// Uses the current fused estimate as the base, overlays the temporal
    /// profile (diurnal/weekly patterns) extracted from observation history,
    /// and applies decaying confidence with forecast horizon.
    pub fn forecast(&self, zone: &str) -> CarbonForecast {
        let now = Utc::now();
        let current = self.estimate(zone);

        // Build temporal profile from history.
        let store = self.store.lock().unwrap();
        let profile = store
            .history(zone)
            .map(TemporalProfile::from_history)
            .unwrap_or_default();
        drop(store);

        let horizon_secs = self.config.forecast_horizon_secs as i64;
        let resolution_secs = self.config.forecast_resolution_secs as i64;
        let mut windows = Vec::new();

        let mut t = now;
        let end_time = now + ChronoDuration::seconds(horizon_secs);

        while t < end_time {
            let window_end = t + ChronoDuration::seconds(resolution_secs);
            let mid = t + ChronoDuration::seconds(resolution_secs / 2);

            let predicted = profile.predict(current.intensity_gco2_kwh, now, mid);

            // Confidence decays exponentially with forecast horizon.
            let horizon_hours = (mid - now).num_seconds() as f64 / 3600.0;
            let conf = current.confidence * (-horizon_hours / 12.0).exp();

            windows.push(ForecastWindow {
                start: t,
                end: window_end.min(end_time),
                intensity_gco2_kwh: predicted,
                confidence: conf.clamp(0.0, 1.0),
            });

            t = window_end;
        }

        CarbonForecast {
            zone: zone.to_string(),
            generated_at: now,
            windows,
        }
    }

    /// Convert the fused estimate to the existing `CarbonIntensity` type
    /// used by SCI calculations and the scheduler.
    pub fn to_carbon_intensity(&self, zone: &str) -> CarbonIntensity {
        let est = self.estimate(zone);
        CarbonIntensity {
            region: zone.to_string(),
            gco2_per_kwh: est.intensity_gco2_kwh,
            timestamp: est.timestamp.timestamp() as u64,
            source: format!("fusion/{:?}", est.freshest_source),
        }
    }

    /// Fused estimate for a cloud provider region (e.g., "us-east-1").
    pub fn estimate_for_cloud_region(&self, region: &str) -> Result<FusedEstimate, CarbonApiError> {
        let zone = cloud_zone_mapping(region)
            .ok_or_else(|| CarbonApiError::UnknownZone(region.to_string()))?;
        Ok(self.estimate(zone))
    }

    /// Global action map: fused estimates for all zones with observations.
    pub fn global_map(&self) -> Vec<FusedEstimate> {
        let store = self.store.lock().unwrap();
        let zones: Vec<String> = store.all_zones().iter().map(|s| s.to_string()).collect();
        drop(store);

        zones.iter().map(|z| self.estimate(z)).collect()
    }

    /// Inject an observation directly (useful for testing or external feeds).
    pub fn inject(&self, obs: Observation) {
        let mut store = self.store.lock().unwrap();
        store.record(obs);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Classify the geographic scope of a backend's response for a given zone.
fn classify_geo_scope(backend_name: &str, _zone: &str) -> GeoScope {
    match backend_name {
        "uk-carbon"
        | "uk-carbon-intensity"
        | "eia"
        | "entsoe"
        | "opennem"
        | "electricity-maps"
        | "epias"
        | "cen"
        | "ons"
        | "cammesa"
        | "niggrid"
        | "csep" => GeoScope::ExactZone,
        "ember" => GeoScope::Country,
        _ => GeoScope::Country, // IPCC static = country-level
    }
}

/// Estimate carbon intensity from a power breakdown using IPCC factors.
fn estimate_intensity_from_breakdown(resp: &crate::carbon_api::RenewableEnergyResponse) -> f64 {
    // Approximate: fossil = coal + gas + oil, renewable = wind + solar + hydro.
    // Use the reported percentages with rough IPCC weights.
    let fossil_intensity = 550.0; // weighted avg of coal (820) + gas (490)
    let nuclear_intensity = 12.0;
    let renewable_intensity = 25.0; // weighted avg of wind + solar + hydro

    let fossil_frac = resp.fossil_percentage / 100.0;
    let nuclear_frac = resp.nuclear_percentage / 100.0;
    let renewable_frac = resp.renewable_percentage / 100.0;

    fossil_frac * fossil_intensity
        + nuclear_frac * nuclear_intensity
        + renewable_frac * renewable_intensity
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn make_obs(
        zone: &str,
        intensity: f64,
        renewable: f64,
        fossil: f64,
        source: SourceKind,
        geo: GeoScope,
        age_secs: i64,
    ) -> Observation {
        Observation {
            zone: zone.to_string(),
            intensity_gco2_kwh: intensity,
            renewable_pct: renewable,
            fossil_pct: fossil,
            observed_at: Utc::now() - ChronoDuration::seconds(age_secs),
            source,
            granularity_secs: source.granularity_secs(),
            geo_scope: geo,
        }
    }

    // -- Fusion core tests --

    #[test]
    fn fuse_single_observation() {
        let obs = make_obs(
            "GB",
            200.0,
            50.0,
            34.0,
            SourceKind::UkCarbon,
            GeoScope::ExactZone,
            60,
        );
        let config = WeightConfig::default();
        let w = observation_weight(&obs, Utc::now(), &config);
        let result = fuse("GB", &[(&obs, w)]);

        assert!((result.intensity_gco2_kwh - 200.0).abs() < 0.1);
        assert!((result.renewable_pct - 50.0).abs() < 0.1);
        assert_eq!(result.observation_count, 1);
        assert_eq!(result.freshest_source, SourceKind::UkCarbon);
    }

    #[test]
    fn fuse_multiple_same_zone_weighted_toward_fresh() {
        // Fresh EIA (5 min old, 350 gCO2), stale Ember (12h old, 400 gCO2),
        // IPCC static (200 gCO2). Fresh EIA should dominate.
        let eia = make_obs(
            "US-MIDA-PJM",
            350.0,
            30.0,
            55.0,
            SourceKind::Eia,
            GeoScope::ExactZone,
            300,
        );
        let ember = make_obs(
            "US-MIDA-PJM",
            400.0,
            25.0,
            60.0,
            SourceKind::Ember,
            GeoScope::Country,
            43200,
        );
        let ipcc = make_obs(
            "US-MIDA-PJM",
            320.0,
            38.0,
            62.0,
            SourceKind::IpccStatic,
            GeoScope::Country,
            0,
        );

        let config = WeightConfig::default();
        let now = Utc::now();
        let w_eia = observation_weight(&eia, now, &config);
        let w_ember = observation_weight(&ember, now, &config);
        let w_ipcc = observation_weight(&ipcc, now, &config);

        // EIA should have highest weight (fresh + exact zone + high reliability).
        assert!(w_eia > w_ember);
        assert!(w_eia > w_ipcc);

        let result = fuse(
            "US-MIDA-PJM",
            &[(&eia, w_eia), (&ember, w_ember), (&ipcc, w_ipcc)],
        );

        // Fused intensity should be closer to EIA (350) than Ember (400).
        assert!(result.intensity_gco2_kwh > 320.0);
        assert!(result.intensity_gco2_kwh < 400.0);
        assert!((result.intensity_gco2_kwh - 350.0).abs() < 30.0);
        assert_eq!(result.observation_count, 3);
    }

    #[test]
    fn fuse_weights_prefer_exact_zone() {
        // Same freshness, but different geo scopes.
        let exact = make_obs(
            "DE",
            300.0,
            50.0,
            38.0,
            SourceKind::Entsoe,
            GeoScope::ExactZone,
            60,
        );
        let country = make_obs(
            "DE",
            350.0,
            45.0,
            42.0,
            SourceKind::Ember,
            GeoScope::Country,
            60,
        );

        let config = WeightConfig::default();
        let now = Utc::now();
        let w_exact = observation_weight(&exact, now, &config);
        let w_country = observation_weight(&country, now, &config);

        // Exact zone should have higher weight.
        assert!(w_exact > w_country * 2.0);

        let result = fuse("DE", &[(&exact, w_exact), (&country, w_country)]);
        // Should be closer to the exact zone observation.
        assert!((result.intensity_gco2_kwh - 300.0).abs() < 20.0);
    }

    #[test]
    fn fuse_empty_returns_ipcc_fallback() {
        let result = fuse("DE", &[]);
        // Should return IPCC static for Germany.
        assert!(result.intensity_gco2_kwh > 0.0);
        assert!(result.confidence < 0.2);
        assert_eq!(result.observation_count, 0);
    }

    // -- Weight tests --

    #[test]
    fn weight_decays_with_age() {
        let config = WeightConfig::default(); // halflife = 1800s
        let fresh = make_obs(
            "GB",
            200.0,
            50.0,
            34.0,
            SourceKind::UkCarbon,
            GeoScope::ExactZone,
            60,
        );
        let old = make_obs(
            "GB",
            200.0,
            50.0,
            34.0,
            SourceKind::UkCarbon,
            GeoScope::ExactZone,
            7200,
        );

        let now = Utc::now();
        let w_fresh = observation_weight(&fresh, now, &config);
        let w_old = observation_weight(&old, now, &config);

        assert!(w_fresh > w_old * 3.0);
    }

    // -- Observation store tests --

    #[test]
    fn observation_store_records_and_retrieves() {
        let mut store = ObservationStore::new(ChronoDuration::days(7));
        let obs = make_obs(
            "GB",
            200.0,
            50.0,
            34.0,
            SourceKind::UkCarbon,
            GeoScope::ExactZone,
            60,
        );
        store.record(obs);

        let recent = store.recent("GB", ChronoDuration::hours(1));
        assert_eq!(recent.len(), 1);
        assert!((recent[0].intensity_gco2_kwh - 200.0).abs() < 0.01);
    }

    #[test]
    fn observation_store_evicts_old() {
        let mut store = ObservationStore::new(ChronoDuration::hours(1));
        // Insert an observation 2 hours old.
        let old = make_obs(
            "GB",
            200.0,
            50.0,
            34.0,
            SourceKind::UkCarbon,
            GeoScope::ExactZone,
            7200,
        );
        store.record(old);

        // Insert a fresh one — the old one should be evicted.
        let fresh = make_obs(
            "GB",
            210.0,
            51.0,
            33.0,
            SourceKind::UkCarbon,
            GeoScope::ExactZone,
            30,
        );
        store.record(fresh);

        let all = store.recent("GB", ChronoDuration::days(1));
        assert_eq!(all.len(), 1);
        assert!((all[0].intensity_gco2_kwh - 210.0).abs() < 0.01);
    }

    // -- Temporal model tests --

    #[test]
    fn temporal_profile_from_synthetic_data() {
        let mut history = VecDeque::new();

        // Fixed base at noon UTC so that `hour` maps 1:1 to UTC hour
        // (hour 0 → 00:00 UTC, hour 14 → 14:00 UTC, etc.).
        // Using Utc::now() made this test flaky: the hour offset shifted
        // which UTC hours received "day" vs "night" intensity depending
        // on when the test ran.
        let base_time = Utc::now()
            .date_naive()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc();

        // Create 3 days of observations: higher intensity during day (8–20),
        // lower at night.
        for day_offset in 0..3 {
            for hour in 0..24 {
                let base = if (8..20).contains(&hour) {
                    400.0
                } else {
                    200.0
                };
                let dt =
                    base_time - ChronoDuration::days(day_offset) + ChronoDuration::hours(hour - 12); // hour 0→00:00, hour 12→12:00, etc.
                history.push_back(Observation {
                    zone: "DE".to_string(),
                    intensity_gco2_kwh: base,
                    renewable_pct: if base > 300.0 { 30.0 } else { 60.0 },
                    fossil_pct: if base > 300.0 { 55.0 } else { 25.0 },
                    observed_at: dt,
                    source: SourceKind::Entsoe,
                    granularity_secs: 3600,
                    geo_scope: GeoScope::ExactZone,
                });
            }
        }

        let profile = TemporalProfile::from_history(&history);
        assert!(profile.sample_count > 0);

        // Daytime hours should have higher ratio than nighttime.
        let day_ratio = profile.diurnal[14]; // 2 PM — always in day range (8..20)
        let night_ratio = profile.diurnal[3]; // 3 AM — always in night range
        assert!(
            day_ratio > night_ratio,
            "expected diurnal[14]={day_ratio} > diurnal[3]={night_ratio}"
        );
    }

    #[test]
    fn temporal_profile_prediction() {
        let mut profile = TemporalProfile::default();
        // Set hour 14 to 1.2× average, hour 3 to 0.7× average.
        profile.diurnal[14] = 1.2;
        profile.diurnal[3] = 0.7;
        profile.sample_count = 100;

        let now = Utc::now();
        let base = 300.0;

        // Predict at two different hours relative to "now".
        let predicted_14 = profile.predict(
            base,
            now,
            now.date_naive().and_hms_opt(14, 0, 0).unwrap().and_utc(),
        );
        let predicted_3 = profile.predict(
            base,
            now,
            now.date_naive().and_hms_opt(3, 0, 0).unwrap().and_utc(),
        );

        // Hour 14 prediction should be higher than hour 3.
        assert!(predicted_14 > predicted_3);
    }

    // -- Spatial model tests --

    #[test]
    fn spatial_neighbors_exist() {
        let spatial = SpatialModel::default_grid();
        assert!(spatial.neighbors.contains_key("DE"));
        assert!(spatial.neighbors.contains_key("US-MIDA-PJM"));
        assert!(spatial.neighbors.contains_key("AU-NSW"));
    }

    #[test]
    fn spatial_neighbor_contributes() {
        let spatial = SpatialModel::default_grid();
        let mut store = ObservationStore::new(ChronoDuration::days(7));

        // Add a fresh observation for DE (neighbor of AT).
        let de_obs = make_obs(
            "DE",
            350.0,
            40.0,
            38.0,
            SourceKind::Entsoe,
            GeoScope::ExactZone,
            300,
        );
        store.record(de_obs);

        // Get neighbor observations for AT.
        let neighbors = spatial.neighbor_observations("AT", &store, ChronoDuration::hours(2));
        assert!(!neighbors.is_empty());
        // The DE observation should appear with a coupling discount.
        let (weight, obs) = &neighbors[0];
        assert!(*weight > 0.0);
        assert!((obs.intensity_gco2_kwh - 350.0).abs() < 0.01);
    }

    #[test]
    fn spatial_no_neighbors_returns_empty() {
        let spatial = SpatialModel::default_grid();
        let store = ObservationStore::new(ChronoDuration::days(7));

        // Zone with no data at all.
        let neighbors =
            spatial.neighbor_observations("XX-UNKNOWN", &store, ChronoDuration::hours(2));
        assert!(neighbors.is_empty());
    }

    // -- Forecast tests --

    #[test]
    fn forecast_produces_windows() {
        let engine = CarbonFusionEngine::default();

        // Inject some observations.
        engine.inject(make_obs(
            "DE",
            350.0,
            40.0,
            38.0,
            SourceKind::Entsoe,
            GeoScope::ExactZone,
            300,
        ));

        let fc = engine.forecast("DE");
        assert!(!fc.windows.is_empty());
        assert_eq!(fc.zone, "DE");

        // 24h / 30min = 48 windows.
        assert_eq!(fc.windows.len(), 48);
    }

    #[test]
    fn forecast_confidence_decays() {
        let engine = CarbonFusionEngine::default();
        engine.inject(make_obs(
            "DE",
            350.0,
            40.0,
            38.0,
            SourceKind::Entsoe,
            GeoScope::ExactZone,
            60,
        ));

        let fc = engine.forecast("DE");
        // First window should have higher confidence than last.
        let first = &fc.windows[0];
        let last = fc.windows.last().unwrap();
        assert!(first.confidence > last.confidence);
    }

    #[test]
    fn forecast_lowest_window() {
        let engine = CarbonFusionEngine::default();

        // Inject observations with a diurnal pattern.
        for hour in 0..24 {
            let intensity = if (8..20).contains(&hour) {
                400.0
            } else {
                200.0
            };
            let dt = Utc::now() - ChronoDuration::days(1) + ChronoDuration::hours(hour as i64 - 12);
            engine.inject(Observation {
                zone: "DE".to_string(),
                intensity_gco2_kwh: intensity,
                renewable_pct: if intensity > 300.0 { 30.0 } else { 60.0 },
                fossil_pct: if intensity > 300.0 { 55.0 } else { 25.0 },
                observed_at: dt,
                source: SourceKind::Entsoe,
                granularity_secs: 3600,
                geo_scope: GeoScope::ExactZone,
            });
        }

        let fc = engine.forecast("DE");
        let lowest = fc.lowest_intensity_window();
        assert!(lowest.is_some());
    }

    // -- Engine integration tests --

    #[test]
    fn engine_estimate_with_injected_data() {
        let engine = CarbonFusionEngine::default();

        // Inject a fresh EIA observation + IPCC will be added automatically.
        engine.inject(make_obs(
            "US-MIDA-PJM",
            380.0,
            28.0,
            62.0,
            SourceKind::Eia,
            GeoScope::ExactZone,
            120,
        ));

        let est = engine.estimate("US-MIDA-PJM");
        // Should be dominated by the fresh EIA observation.
        assert!((est.intensity_gco2_kwh - 380.0).abs() < 50.0);
        assert!(est.confidence > 0.0);
        assert!(est.observation_count >= 1);
    }

    #[test]
    fn engine_estimate_no_data_returns_ipcc() {
        let engine = CarbonFusionEngine::default();

        let est = engine.estimate("FR");
        // No injected data → IPCC static for France.
        // France is ~65% nuclear → should be low carbon (~70-80 gCO2/kWh).
        assert!(est.intensity_gco2_kwh > 30.0);
        assert!(est.intensity_gco2_kwh < 200.0);
        assert!(est.confidence < 0.5);
    }

    #[test]
    fn engine_confidence_increases_with_sources() {
        let engine = CarbonFusionEngine::default();

        // Estimate with zero injected data.
        let est_none = engine.estimate("GB");

        // Now inject from two different sources.
        engine.inject(make_obs(
            "GB",
            200.0,
            50.0,
            34.0,
            SourceKind::UkCarbon,
            GeoScope::ExactZone,
            60,
        ));
        engine.inject(make_obs(
            "GB",
            210.0,
            48.0,
            36.0,
            SourceKind::Ember,
            GeoScope::Country,
            3600,
        ));

        let est_multi = engine.estimate("GB");

        // More sources → higher confidence.
        assert!(est_multi.confidence > est_none.confidence);
    }

    #[test]
    fn engine_to_carbon_intensity_compat() {
        let engine = CarbonFusionEngine::default();
        engine.inject(make_obs(
            "GB",
            200.0,
            50.0,
            34.0,
            SourceKind::UkCarbon,
            GeoScope::ExactZone,
            60,
        ));

        let ci = engine.to_carbon_intensity("GB");
        assert_eq!(ci.region, "GB");
        assert!(ci.gco2_per_kwh > 0.0);
        assert!(ci.source.starts_with("fusion/"));
    }

    #[test]
    fn engine_estimate_for_cloud_region() {
        let engine = CarbonFusionEngine::default();
        engine.inject(make_obs(
            "US-CAL-CISO",
            180.0,
            45.0,
            40.0,
            SourceKind::Eia,
            GeoScope::ExactZone,
            60,
        ));

        let est = engine.estimate_for_cloud_region("us-west-2");
        // us-west-2 → US-NW-BPAT (not CISO), but neighbor data from CISO
        // should contribute via spatial model.
        assert!(est.is_ok());
    }

    #[test]
    fn engine_global_map() {
        let engine = CarbonFusionEngine::default();
        engine.inject(make_obs(
            "GB",
            200.0,
            50.0,
            34.0,
            SourceKind::UkCarbon,
            GeoScope::ExactZone,
            60,
        ));
        engine.inject(make_obs(
            "DE",
            350.0,
            40.0,
            38.0,
            SourceKind::Entsoe,
            GeoScope::ExactZone,
            60,
        ));

        let map = engine.global_map();
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn source_kind_reliability_ordering() {
        // Real-time sources should have higher reliability than static.
        assert!(SourceKind::UkCarbon.reliability() > SourceKind::Ember.reliability());
        assert!(SourceKind::Ember.reliability() > SourceKind::IpccStatic.reliability());
    }

    #[test]
    fn geo_scope_precision_ordering() {
        assert!(GeoScope::ExactZone.precision() > GeoScope::Country.precision());
        assert!(GeoScope::Country.precision() > GeoScope::Global.precision());
    }

    // -- Phase 1F: Tier 3 spatial expansion tests --

    #[test]
    fn spatial_new_region_neighbors() {
        let spatial = SpatialModel::default_grid();

        // Africa
        assert!(
            spatial.neighbors.contains_key("MA"),
            "Morocco should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("KE"),
            "Kenya should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("NG"),
            "Nigeria should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("EG"),
            "Egypt should have neighbors"
        );

        // South America
        assert!(
            spatial.neighbors.contains_key("AR"),
            "Argentina should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("CL"),
            "Chile should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("CO"),
            "Colombia should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("PE"),
            "Peru should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("UY"),
            "Uruguay should have neighbors"
        );

        // Middle East
        assert!(
            spatial.neighbors.contains_key("SA"),
            "Saudi Arabia should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("AE"),
            "UAE should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("TR"),
            "Turkey should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("KW"),
            "Kuwait should have neighbors"
        );

        // Southeast Asia
        assert!(
            spatial.neighbors.contains_key("MY"),
            "Malaysia should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("TH"),
            "Thailand should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("ID"),
            "Indonesia should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("PK"),
            "Pakistan should have neighbors"
        );
        assert!(
            spatial.neighbors.contains_key("BD"),
            "Bangladesh should have neighbors"
        );
    }

    #[test]
    fn spatial_cross_continent_links() {
        let spatial = SpatialModel::default_grid();

        // Morocco ↔ Spain (Africa-Europe HVDC link)
        let ma_neighbors = spatial.neighbors.get("MA").unwrap();
        assert!(
            ma_neighbors.iter().any(|(z, _)| *z == "ES"),
            "Morocco should be connected to Spain"
        );

        // Turkey ↔ Greece (Europe-Middle East link)
        let tr_neighbors = spatial.neighbors.get("TR").unwrap();
        assert!(
            tr_neighbors.iter().any(|(z, _)| *z == "GR"),
            "Turkey should be connected to Greece"
        );

        // Mexico ↔ Texas
        let mx_neighbors = spatial.neighbors.get("MX").unwrap();
        assert!(
            mx_neighbors.iter().any(|(z, _)| *z == "US-TEX-ERCO"),
            "Mexico should be connected to Texas"
        );
    }
}
