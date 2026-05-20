//! Multi-backend carbon intensity aggregator.
//!
//! Pulls real-time grid carbon intensity and generation mix data from
//! multiple free, open data sources instead of relying on a single paid
//! vendor. Each backend covers a geographic region; the aggregator
//! routes requests to the best available backend and falls through on
//! error.
//!
//! **Resolution order per zone:**
//! 1. ElectricityMaps (optional, paid — highest quality)
//! 2. UK Carbon Intensity API (GB — free, no key)
//! 3. WattTime (US — marginal emissions rate, free tier)
//! 4. EIA (US zones — free key, hourly average)
//! 5. ENTSO-E Transparency (EU — free key, 15-min)
//! 6. OpenNEM (Australia — free key, 5-min)
//! 7. Ember Climate (global — free key, monthly)
//! 8. IPCC static fallback (built-in, no network)

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::carbon::CarbonIntensity;
use crate::carbon_api::{
    CarbonApiError, CarbonIntensityResponse, ElectricityMapsClient, ElectricityMapsConfig,
    RenewableEnergyResponse, cloud_zone_mapping,
};

// ---------------------------------------------------------------------------
// IPCC AR5 lifecycle emission factors (gCO2eq / kWh)
// ---------------------------------------------------------------------------

const IPCC_COAL: f64 = 820.0;
const IPCC_GAS: f64 = 490.0;
const IPCC_OIL: f64 = 650.0;
const IPCC_BIOMASS: f64 = 230.0;
const IPCC_NUCLEAR: f64 = 12.0;
const IPCC_WIND: f64 = 11.0;
const IPCC_SOLAR: f64 = 41.0;
const IPCC_HYDRO: f64 = 24.0;
const IPCC_GEOTHERMAL: f64 = 38.0;

/// Grid mix percentages for a country (sums to ~100).
#[derive(Debug, Clone, Copy)]
pub struct GridMix {
    pub coal: f64,
    pub gas: f64,
    pub oil: f64,
    pub nuclear: f64,
    pub hydro: f64,
    pub wind: f64,
    pub solar: f64,
    pub biomass: f64,
    pub geothermal: f64,
}

impl GridMix {
    /// Compute weighted-average carbon intensity in gCO2eq/kWh.
    ///
    /// # Contracts (verified by proptest `prop_carbon_intensity_bounded`)
    /// - **Post**: result ∈ [11.0, 820.0] for normalized mixes (sum > 0)
    /// - **Post**: result == 400.0 for zero mix (fallback)
    pub fn carbon_intensity(&self) -> f64 {
        let total = self.coal
            + self.gas
            + self.oil
            + self.nuclear
            + self.hydro
            + self.wind
            + self.solar
            + self.biomass
            + self.geothermal;
        if total == 0.0 {
            return 400.0; // global average fallback
        }
        let result = (self.coal * IPCC_COAL
            + self.gas * IPCC_GAS
            + self.oil * IPCC_OIL
            + self.nuclear * IPCC_NUCLEAR
            + self.hydro * IPCC_HYDRO
            + self.wind * IPCC_WIND
            + self.solar * IPCC_SOLAR
            + self.biomass * IPCC_BIOMASS
            + self.geothermal * IPCC_GEOTHERMAL)
            / total;
        debug_assert!(
            result >= IPCC_WIND - 0.01 && result <= IPCC_COAL + 0.01,
            "carbon_intensity contract: {} not in [{}, {}]", result, IPCC_WIND, IPCC_COAL
        );
        result
    }

    /// Renewable percentage (wind + solar + hydro + biomass + geothermal).
    ///
    /// # Contract: result ≥ 0.0
    pub fn renewable_pct(&self) -> f64 {
        let result = self.wind + self.solar + self.hydro + self.biomass + self.geothermal;
        debug_assert!(result >= 0.0, "renewable_pct contract: {} < 0", result);
        result
    }

    /// Fossil percentage (coal + gas + oil).
    ///
    /// # Contract: result ≥ 0.0
    pub fn fossil_pct(&self) -> f64 {
        let result = self.coal + self.gas + self.oil;
        debug_assert!(result >= 0.0, "fossil_pct contract: {} < 0", result);
        result
    }
}

// ---------------------------------------------------------------------------
// CarbonBackend trait
// ---------------------------------------------------------------------------

/// A pluggable carbon intensity data source.
#[async_trait]
pub trait CarbonBackend: Send + Sync {
    /// Human-readable name (e.g. "eia", "entsoe", "ipcc-static").
    fn name(&self) -> &str;

    /// Whether this backend can provide data for the given zone.
    fn supports_zone(&self, zone: &str) -> bool;

    /// Fetch the latest carbon intensity for an ElectricityMaps zone code.
    async fn carbon_intensity(&self, zone: &str)
    -> Result<CarbonIntensityResponse, CarbonApiError>;

    /// Fetch the latest power breakdown for an ElectricityMaps zone code.
    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError>;
}

// ---------------------------------------------------------------------------
// AggregatorConfig
// ---------------------------------------------------------------------------

/// Configuration for the multi-backend aggregator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatorConfig {
    /// EIA API key (free: <https://www.eia.gov/opendata/register.php>).
    pub eia_api_key: Option<String>,
    /// ENTSO-E security token (free: email transparency@entsoe.eu).
    pub entsoe_token: Option<String>,
    /// ENTSO-E API endpoint URL (default: new endpoint, old one deprecated Dec 2025).
    pub entsoe_endpoint: Option<String>,
    /// OpenNEM / OpenElectricity API key (free: <https://platform.openelectricity.org.au>).
    pub opennem_api_key: Option<String>,
    /// Ember Climate API key (free: <https://ember-energy.org/data/api>).
    pub ember_api_key: Option<String>,
    /// ElectricityMaps API key (paid, optional premium backend).
    pub electricitymaps_api_key: Option<String>,
    /// WattTime username (free tier: <https://watttime.org/api-documentation/>).
    pub watttime_username: Option<String>,
    /// WattTime password.
    pub watttime_password: Option<String>,
    /// EPIAS token for Turkey grid data (free: <https://seffaflik.epias.com.tr>).
    pub epias_token: Option<String>,
    /// Enable ONS scraper backend (Brazil). Requires `scraper-backends` feature.
    pub enable_ons: bool,
    /// Enable CAMMESA scraper backend (Argentina). Requires `scraper-backends` feature.
    pub enable_cammesa: bool,
    /// Enable NigGrid scraper backend (Nigeria). Requires `scraper-backends` feature.
    pub enable_niggrid: bool,
    /// Enable CSEP/CarbonTracker scraper backend (India). Requires `scraper-backends` feature.
    pub enable_csep: bool,
    /// Cache TTL in seconds (default 300 = 5 minutes).
    pub cache_ttl_secs: u64,
}

// ---------------------------------------------------------------------------
// CarbonAggregator
// ---------------------------------------------------------------------------

/// Time-stamped cache entry.
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    fetched_at: Instant,
}

/// Multi-backend carbon intensity aggregator with caching and fallthrough.
pub struct CarbonAggregator {
    backends: Vec<Box<dyn CarbonBackend>>,
    carbon_cache: Mutex<HashMap<String, CacheEntry<CarbonIntensityResponse>>>,
    power_cache: Mutex<HashMap<String, CacheEntry<RenewableEnergyResponse>>>,
    cache_ttl: Duration,
}

impl std::fmt::Debug for CarbonAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonAggregator")
            .field("backends", &self.backends.len())
            .field("cache_ttl", &self.cache_ttl)
            .finish()
    }
}

/// Build backend list from configuration. Used by both `CarbonAggregator`
/// and `CarbonFusionEngine`.
pub(crate) fn build_backends(config: &AggregatorConfig) -> Vec<Box<dyn CarbonBackend>> {
    let mut backends: Vec<Box<dyn CarbonBackend>> = Vec::new();

    // 1. ElectricityMaps (optional paid premium)
    if let Some(ref key) = config.electricitymaps_api_key
        && !key.is_empty()
    {
        backends.push(Box::new(ElectricityMapsBackend::new(key.clone())));
    }

    // 2. UK Carbon Intensity (free, no key)
    backends.push(Box::new(UkCarbonBackend::new()));

    // 3. WattTime (US marginal emissions, free tier)
    if let Some(ref user) = config.watttime_username
        && let Some(ref pass) = config.watttime_password
        && !user.is_empty()
        && !pass.is_empty()
    {
        backends.push(Box::new(WattTimeBackend::new(user.clone(), pass.clone())));
    }

    // 4. EIA (US zones, free key — average intensity)
    if let Some(ref key) = config.eia_api_key
        && !key.is_empty()
    {
        backends.push(Box::new(EiaBackend::new(key.clone())));
    }

    // 5. ENTSO-E (Europe, free key)
    if let Some(ref token) = config.entsoe_token
        && !token.is_empty()
    {
        backends.push(Box::new(EntsoeBackend::new(
            token.clone(),
            config.entsoe_endpoint.clone(),
        )));
    }

    // 5. OpenNEM (Australia, free key)
    if let Some(ref key) = config.opennem_api_key
        && !key.is_empty()
    {
        backends.push(Box::new(OpenNemBackend::new(key.clone())));
    }

    // 6. Ember (global monthly, free key)
    if let Some(ref key) = config.ember_api_key
        && !key.is_empty()
    {
        backends.push(Box::new(EmberBackend::new(key.clone())));
    }

    // 7. EPIAS (Turkey, free registration)
    if let Some(ref token) = config.epias_token
        && !token.is_empty()
    {
        backends.push(Box::new(EpiasBackend::new(token.clone())));
    }

    // 8. CEN (Chile, no auth needed)
    backends.push(Box::new(CenBackend::new()));

    // 9–12. Scraper backends (feature-gated, opt-in)
    #[cfg(feature = "scraper-backends")]
    {
        if config.enable_ons {
            backends.push(Box::new(OnsBackend::new()));
        }
        if config.enable_cammesa {
            backends.push(Box::new(CammesaBackend::new()));
        }
        if config.enable_niggrid {
            backends.push(Box::new(NiggridBackend::new()));
        }
        if config.enable_csep {
            backends.push(Box::new(CsepBackend::new()));
        }
    }

    // 13. IPCC static fallback (always available)
    backends.push(Box::new(IpccStaticBackend));

    backends
}

impl CarbonAggregator {
    /// Build an aggregator from the given configuration.
    ///
    /// Backends are registered in priority order. The IPCC static fallback
    /// is always added last so that every zone has at least one backend.
    pub fn new(config: AggregatorConfig) -> Self {
        let backends = build_backends(&config);

        let ttl = if config.cache_ttl_secs > 0 {
            config.cache_ttl_secs
        } else {
            300
        };

        Self {
            backends,
            carbon_cache: Mutex::new(HashMap::new()),
            power_cache: Mutex::new(HashMap::new()),
            cache_ttl: Duration::from_secs(ttl),
        }
    }

    /// Build a zero-config aggregator that works out of the box (IPCC only).
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(AggregatorConfig::default())
    }

    /// Fetch carbon intensity for an ElectricityMaps zone code.
    ///
    /// Orbital zones ("ORBITAL") return zero carbon intensity immediately
    /// since orbital compute runs on solar power with no grid dependency.
    pub async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        // Fast-path: orbital nodes have zero operational carbon.
        if zone == "ORBITAL" {
            return Ok(CarbonIntensityResponse {
                zone: "ORBITAL".to_string(),
                carbon_intensity_gco2_kwh: crate::carbon_api::ORBITAL_CARBON_INTENSITY_GCO2_KWH,
                datetime: Utc::now(),
                data_source: "orbital-solar".to_string(),
                fossil_fuel_percentage: 0.0,
                is_estimated: false,
            });
        }

        // Check cache.
        if let Some(cached) = self.get_carbon_cache(zone) {
            return Ok(cached);
        }

        // Try backends in priority order.
        let mut last_err = None;
        for backend in &self.backends {
            if !backend.supports_zone(zone) {
                continue;
            }
            match backend.carbon_intensity(zone).await {
                Ok(response) => {
                    self.put_carbon_cache(zone, &response);
                    return Ok(response);
                }
                Err(e) => {
                    tracing::debug!(
                        backend = backend.name(),
                        zone,
                        error = %e,
                        "backend failed, trying next"
                    );
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or(CarbonApiError::UnknownZone(zone.to_string())))
    }

    /// Fetch power breakdown for an ElectricityMaps zone code.
    pub async fn power_breakdown(
        &self,
        zone: &str,
    ) -> Result<RenewableEnergyResponse, CarbonApiError> {
        // Fast-path: orbital nodes are 100% solar.
        if zone == "ORBITAL" {
            return Ok(RenewableEnergyResponse {
                zone: "ORBITAL".to_string(),
                renewable_percentage: 100.0,
                wind_percentage: 0.0,
                solar_percentage: 100.0,
                hydro_percentage: 0.0,
                nuclear_percentage: 0.0,
                fossil_percentage: 0.0,
                datetime: Utc::now(),
            });
        }

        if let Some(cached) = self.get_power_cache(zone) {
            return Ok(cached);
        }

        let mut last_err = None;
        for backend in &self.backends {
            if !backend.supports_zone(zone) {
                continue;
            }
            match backend.power_breakdown(zone).await {
                Ok(response) => {
                    self.put_power_cache(zone, &response);
                    return Ok(response);
                }
                Err(e) => {
                    tracing::debug!(
                        backend = backend.name(),
                        zone,
                        error = %e,
                        "backend failed, trying next"
                    );
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or(CarbonApiError::UnknownZone(zone.to_string())))
    }

    /// Convert to the simpler `CarbonIntensity` type used by SCI calculations.
    pub async fn to_carbon_intensity(&self, zone: &str) -> Result<CarbonIntensity, CarbonApiError> {
        let resp = self.carbon_intensity(zone).await?;
        Ok(CarbonIntensity {
            region: zone.to_string(),
            gco2_per_kwh: resp.carbon_intensity_gco2_kwh,
            timestamp: resp.datetime.timestamp() as u64,
            source: resp.data_source,
        })
    }

    /// End-to-end: cloud provider region → zone → carbon intensity.
    pub async fn carbon_intensity_for_cloud_region(
        &self,
        cloud_region: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        let zone = cloud_zone_mapping(cloud_region)
            .ok_or_else(|| CarbonApiError::UnknownZone(cloud_region.to_string()))?;
        self.carbon_intensity(zone).await
    }

    fn get_carbon_cache(&self, zone: &str) -> Option<CarbonIntensityResponse> {
        let cache = self.carbon_cache.lock().unwrap();
        if let Some(entry) = cache.get(zone)
            && entry.fetched_at.elapsed() < self.cache_ttl
        {
            return Some(entry.value.clone());
        }
        None
    }

    fn put_carbon_cache(&self, zone: &str, value: &CarbonIntensityResponse) {
        let mut cache = self.carbon_cache.lock().unwrap();
        cache.insert(
            zone.to_string(),
            CacheEntry {
                value: value.clone(),
                fetched_at: Instant::now(),
            },
        );
    }

    fn get_power_cache(&self, zone: &str) -> Option<RenewableEnergyResponse> {
        let cache = self.power_cache.lock().unwrap();
        if let Some(entry) = cache.get(zone)
            && entry.fetched_at.elapsed() < self.cache_ttl
        {
            return Some(entry.value.clone());
        }
        None
    }

    fn put_power_cache(&self, zone: &str, value: &RenewableEnergyResponse) {
        let mut cache = self.power_cache.lock().unwrap();
        cache.insert(
            zone.to_string(),
            CacheEntry {
                value: value.clone(),
                fetched_at: Instant::now(),
            },
        );
    }
}

// ===========================================================================
// Zone mapping helpers
// ===========================================================================

/// Map ElectricityMaps zone code to EIA balancing authority code.
pub(crate) fn zone_to_eia_ba(zone: &str) -> Option<&'static str> {
    match zone {
        "US-MIDA-PJM" => Some("PJM"),
        "US-CAL-CISO" => Some("CISO"),
        "US-MIDW-MISO" => Some("MISO"),
        "US-TEX-ERCO" => Some("ERCO"),
        "US-NY-NYIS" => Some("NYIS"),
        "US-NW-BPAT" => Some("BPAT"),
        "US-SE-SOCO" => Some("SOCO"),
        "US-SW-AZPS" => Some("AZPS"),
        "US-NE-ISNE" => Some("ISNE"),
        "US-SW-SRP" => Some("SRP"),
        "US-NW-PACW" => Some("PACW"),
        "US-SW-EPE" => Some("EPE"),
        "US-FLA-FPL" => Some("FPL"),
        "US-SE-AEC" => Some("AEC"),
        "US-TEN-TVA" => Some("TVA"),
        _ if zone.starts_with("US-") => Some("US48"), // fallback to aggregate US
        _ => None,
    }
}

/// Map ElectricityMaps zone code to ENTSO-E bidding zone EIC code.
pub(crate) fn zone_to_entsoe_area(zone: &str) -> Option<&'static str> {
    match zone {
        "DE" => Some("10Y1001A1001A83F"),    // Germany
        "FR" => Some("10YFR-RTE------C"),    // France
        "NL" => Some("10YNL----------L"),    // Netherlands
        "BE" => Some("10YBE----------2"),    // Belgium
        "ES" => Some("10YES-REE------0"),    // Spain
        "IT-NO" => Some("10Y1001A1001A73I"), // Italy North
        "SE" => Some("10YSE-1--------K"),    // Sweden
        "FI" => Some("10YFI-1--------U"),    // Finland
        "PL" => Some("10YPL-AREA-----S"),    // Poland
        "NO" => Some("10YNO-0--------C"),    // Norway
        "DK" => Some("10Y1001A1001A65H"),    // Denmark
        "AT" => Some("10YAT-APG------L"),    // Austria
        "CH" => Some("10YCH-SWISSGRIDZ"),    // Switzerland
        "PT" => Some("10YPT-REN------W"),    // Portugal
        "IE" => Some("10YIE-1001A00010"),    // Ireland
        "CZ" => Some("10YCZ-CEPS-----N"),    // Czech Republic
        "RO" => Some("10YRO-TEL------P"),    // Romania
        "GR" => Some("10YGR-HTSO-----Y"),    // Greece
        "HU" => Some("10YHU-MAVIR----U"),    // Hungary
        "BG" => Some("10YCA-BULGARIA-R"),    // Bulgaria
        "SK" => Some("10YSK-SEPS-----K"),    // Slovakia
        "HR" => Some("10YHR-HEP------M"),    // Croatia
        "SI" => Some("10YSI-ELES-----O"),    // Slovenia
        "EE" => Some("10Y1001A1001A39I"),    // Estonia
        "LV" => Some("10YLV-1001A00074"),    // Latvia
        "LT" => Some("10YLT-1001A0008Q"),    // Lithuania
        _ => None,
    }
}

/// Map ElectricityMaps zone code to ISO 3166-1 alpha-3 country code (for Ember).
pub(crate) fn zone_to_country_iso3(zone: &str) -> Option<&'static str> {
    match zone {
        "US-MIDA-PJM" | "US-CAL-CISO" | "US-MIDW-MISO" | "US-TEX-ERCO" | "US-NY-NYIS"
        | "US-NW-BPAT" | "US-SE-SOCO" | "US-SW-AZPS" => Some("USA"),
        "DE" => Some("DEU"),
        "FR" => Some("FRA"),
        "GB" => Some("GBR"),
        "NL" => Some("NLD"),
        "BE" => Some("BEL"),
        "ES" => Some("ESP"),
        "IT-NO" => Some("ITA"),
        "SE" => Some("SWE"),
        "FI" => Some("FIN"),
        "PL" => Some("POL"),
        "NO" => Some("NOR"),
        "DK" => Some("DNK"),
        "AT" => Some("AUT"),
        "CH" => Some("CHE"),
        "PT" => Some("PRT"),
        "IE" => Some("IRL"),
        "CZ" => Some("CZE"),
        "RO" => Some("ROU"),
        "GR" => Some("GRC"),
        "HU" => Some("HUN"),
        "BG" => Some("BGR"),
        "SK" => Some("SVK"),
        "HR" => Some("HRV"),
        "SI" => Some("SVN"),
        "EE" => Some("EST"),
        "LV" => Some("LVA"),
        "LT" => Some("LTU"),
        "JP-TK" => Some("JPN"),
        "KR" => Some("KOR"),
        "IN-WE" | "IN-SO" => Some("IND"),
        "AU-NSW" | "AU-VIC" | "AU-QLD" | "AU-SA" | "AU-TAS" => Some("AUS"),
        "BR-S" => Some("BRA"),
        "SG" => Some("SGP"),
        "IL" => Some("ISR"),
        "ZA" => Some("ZAF"),
        "CA-ON" | "CA-QC" | "CA-AB" => Some("CAN"),
        "BH" => Some("BHR"),
        // Africa
        "NG" => Some("NGA"),
        "EG" => Some("EGY"),
        "KE" => Some("KEN"),
        "MA" => Some("MAR"),
        "ET" => Some("ETH"),
        "GH" => Some("GHA"),
        "TZ" => Some("TZA"),
        "DZ" => Some("DZA"),
        "TN" => Some("TUN"),
        "SN" => Some("SEN"),
        // South America
        "CL" => Some("CHL"),
        "CO" => Some("COL"),
        "AR" => Some("ARG"),
        "PE" => Some("PER"),
        "MX" => Some("MEX"),
        "EC" => Some("ECU"),
        "UY" => Some("URY"),
        "PY" => Some("PRY"),
        "BO" => Some("BOL"),
        "VE" => Some("VEN"),
        // Middle East
        "SA" | "SA-KSA" => Some("SAU"),
        "AE" => Some("ARE"),
        "TR" => Some("TUR"),
        "IQ" => Some("IRQ"),
        "IR" => Some("IRN"),
        "KW" => Some("KWT"),
        "OM" => Some("OMN"),
        "QA" => Some("QAT"),
        "JO" => Some("JOR"),
        // Southeast Asia
        "ID" => Some("IDN"),
        "VN" => Some("VNM"),
        "TH" => Some("THA"),
        "PH" => Some("PHL"),
        "PK" => Some("PAK"),
        "BD" => Some("BGD"),
        "MM" => Some("MMR"),
        "MY" => Some("MYS"),
        // Other
        "NZ" => Some("NZL"),
        "TW" => Some("TWN"),
        _ => None,
    }
}

/// Map ElectricityMaps zone code to OpenNEM NEM region.
pub(crate) fn zone_to_opennem_region(zone: &str) -> Option<&'static str> {
    match zone {
        "AU-NSW" => Some("NSW1"),
        "AU-VIC" => Some("VIC1"),
        "AU-QLD" => Some("QLD1"),
        "AU-SA" => Some("SA1"),
        "AU-TAS" => Some("TAS1"),
        _ => None,
    }
}

/// Map zone code to country key for IPCC static grid mix lookup.
pub(crate) fn zone_to_country(zone: &str) -> &str {
    if zone.starts_with("US-") {
        return "US";
    }
    if zone.starts_with("AU-") {
        return "AU";
    }
    if zone.starts_with("IN-") {
        return "IN";
    }
    if zone.starts_with("BR-") {
        return "BR";
    }
    if zone.starts_with("CA-") {
        return "CA";
    }
    if zone.starts_with("MX-") {
        return "MX";
    }
    if zone.starts_with("CL-") {
        return "CL";
    }
    if zone.starts_with("AR-") {
        return "AR";
    }
    if zone.starts_with("PK-") {
        return "PK";
    }
    if zone.starts_with("ID-") {
        return "ID";
    }
    if zone.starts_with("NZ-") {
        return "NZ";
    }
    match zone {
        "JP-TK" => "JP",
        "IT-NO" => "IT",
        _ => zone,
    }
}

/// Country-level average grid mixes (2024 estimates, IEA/Ember data).
pub(crate) fn country_grid_mix(country: &str) -> GridMix {
    match country {
        "US" => GridMix {
            coal: 19.0,
            gas: 43.0,
            oil: 0.0,
            nuclear: 18.0,
            hydro: 6.0,
            wind: 10.0,
            solar: 4.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "DE" => GridMix {
            coal: 26.0,
            gas: 12.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 4.0,
            wind: 27.0,
            solar: 12.0,
            biomass: 19.0,
            geothermal: 0.0,
        },
        "FR" => GridMix {
            coal: 1.0,
            gas: 7.0,
            oil: 0.0,
            nuclear: 65.0,
            hydro: 11.0,
            wind: 10.0,
            solar: 5.0,
            biomass: 1.0,
            geothermal: 0.0,
        },
        "GB" => GridMix {
            coal: 1.0,
            gas: 33.0,
            oil: 0.0,
            nuclear: 13.0,
            hydro: 2.0,
            wind: 29.0,
            solar: 5.0,
            biomass: 17.0,
            geothermal: 0.0,
        },
        "NL" => GridMix {
            coal: 7.0,
            gas: 45.0,
            oil: 0.0,
            nuclear: 3.0,
            hydro: 0.0,
            wind: 20.0,
            solar: 8.0,
            biomass: 17.0,
            geothermal: 0.0,
        },
        "SE" => GridMix {
            coal: 0.0,
            gas: 1.0,
            oil: 0.0,
            nuclear: 29.0,
            hydro: 41.0,
            wind: 21.0,
            solar: 2.0,
            biomass: 6.0,
            geothermal: 0.0,
        },
        "FI" => GridMix {
            coal: 4.0,
            gas: 4.0,
            oil: 0.0,
            nuclear: 34.0,
            hydro: 17.0,
            wind: 19.0,
            solar: 1.0,
            biomass: 21.0,
            geothermal: 0.0,
        },
        "NO" => GridMix {
            coal: 0.0,
            gas: 2.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 88.0,
            wind: 9.0,
            solar: 0.0,
            biomass: 1.0,
            geothermal: 0.0,
        },
        "IE" => GridMix {
            coal: 0.0,
            gas: 44.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 3.0,
            wind: 36.0,
            solar: 3.0,
            biomass: 14.0,
            geothermal: 0.0,
        },
        "JP" => GridMix {
            coal: 31.0,
            gas: 34.0,
            oil: 6.0,
            nuclear: 8.0,
            hydro: 8.0,
            wind: 3.0,
            solar: 10.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "KR" => GridMix {
            coal: 34.0,
            gas: 28.0,
            oil: 0.0,
            nuclear: 27.0,
            hydro: 1.0,
            wind: 2.0,
            solar: 5.0,
            biomass: 3.0,
            geothermal: 0.0,
        },
        "IN" => GridMix {
            coal: 74.0,
            gas: 5.0,
            oil: 0.0,
            nuclear: 3.0,
            hydro: 10.0,
            wind: 5.0,
            solar: 3.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "AU" => GridMix {
            coal: 47.0,
            gas: 18.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 6.0,
            wind: 12.0,
            solar: 15.0,
            biomass: 2.0,
            geothermal: 0.0,
        },
        "BR" => GridMix {
            coal: 3.0,
            gas: 10.0,
            oil: 0.0,
            nuclear: 2.0,
            hydro: 63.0,
            wind: 13.0,
            solar: 6.0,
            biomass: 3.0,
            geothermal: 0.0,
        },
        "SG" => GridMix {
            coal: 1.0,
            gas: 95.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 0.0,
            solar: 4.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "IL" => GridMix {
            coal: 19.0,
            gas: 60.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 0.0,
            solar: 14.0,
            biomass: 0.0,
            geothermal: 7.0,
        },
        "ZA" => GridMix {
            coal: 82.0,
            gas: 3.0,
            oil: 0.0,
            nuclear: 5.0,
            hydro: 2.0,
            wind: 5.0,
            solar: 3.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "BE" => GridMix {
            coal: 2.0,
            gas: 23.0,
            oil: 0.0,
            nuclear: 40.0,
            hydro: 1.0,
            wind: 15.0,
            solar: 7.0,
            biomass: 12.0,
            geothermal: 0.0,
        },
        "PL" => GridMix {
            coal: 62.0,
            gas: 10.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 2.0,
            wind: 15.0,
            solar: 5.0,
            biomass: 6.0,
            geothermal: 0.0,
        },
        "ES" => GridMix {
            coal: 2.0,
            gas: 22.0,
            oil: 0.0,
            nuclear: 20.0,
            hydro: 9.0,
            wind: 24.0,
            solar: 17.0,
            biomass: 6.0,
            geothermal: 0.0,
        },
        "CH" => GridMix {
            coal: 0.0,
            gas: 1.0,
            oil: 0.0,
            nuclear: 35.0,
            hydro: 57.0,
            wind: 1.0,
            solar: 5.0,
            biomass: 1.0,
            geothermal: 0.0,
        },
        "CA" => GridMix {
            coal: 4.0,
            gas: 12.0,
            oil: 0.0,
            nuclear: 14.0,
            hydro: 59.0,
            wind: 7.0,
            solar: 2.0,
            biomass: 2.0,
            geothermal: 0.0,
        },
        "IT" => GridMix {
            coal: 4.0,
            gas: 43.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 15.0,
            wind: 8.0,
            solar: 10.0,
            biomass: 17.0,
            geothermal: 3.0,
        },
        "AT" => GridMix {
            coal: 2.0,
            gas: 14.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 60.0,
            wind: 12.0,
            solar: 6.0,
            biomass: 6.0,
            geothermal: 0.0,
        },
        "DK" => GridMix {
            coal: 4.0,
            gas: 5.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 57.0,
            solar: 8.0,
            biomass: 26.0,
            geothermal: 0.0,
        },
        "PT" => GridMix {
            coal: 0.0,
            gas: 20.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 22.0,
            wind: 28.0,
            solar: 8.0,
            biomass: 7.0,
            geothermal: 15.0,
        },
        "BH" => GridMix {
            coal: 0.0,
            gas: 100.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 0.0,
            solar: 0.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        // --- Africa ---
        "NG" => GridMix {
            coal: 0.0,
            gas: 81.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 17.0,
            wind: 0.0,
            solar: 2.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "EG" => GridMix {
            coal: 0.0,
            gas: 80.0,
            oil: 10.0,
            nuclear: 0.0,
            hydro: 6.0,
            wind: 2.0,
            solar: 2.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "KE" => GridMix {
            coal: 0.0,
            gas: 0.0,
            oil: 9.0,
            nuclear: 0.0,
            hydro: 26.0,
            wind: 14.0,
            solar: 5.0,
            biomass: 0.0,
            geothermal: 46.0,
        },
        "MA" => GridMix {
            coal: 40.0,
            gas: 22.0,
            oil: 5.0,
            nuclear: 0.0,
            hydro: 4.0,
            wind: 15.0,
            solar: 8.0,
            biomass: 6.0,
            geothermal: 0.0,
        },
        "ET" => GridMix {
            coal: 0.0,
            gas: 0.0,
            oil: 3.0,
            nuclear: 0.0,
            hydro: 86.0,
            wind: 8.0,
            solar: 1.0,
            biomass: 2.0,
            geothermal: 0.0,
        },
        "GH" => GridMix {
            coal: 0.0,
            gas: 62.0,
            oil: 2.0,
            nuclear: 0.0,
            hydro: 34.0,
            wind: 0.0,
            solar: 2.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "TZ" => GridMix {
            coal: 3.0,
            gas: 48.0,
            oil: 3.0,
            nuclear: 0.0,
            hydro: 38.0,
            wind: 0.0,
            solar: 5.0,
            biomass: 3.0,
            geothermal: 0.0,
        },
        "DZ" => GridMix {
            coal: 0.0,
            gas: 97.0,
            oil: 1.0,
            nuclear: 0.0,
            hydro: 1.0,
            wind: 0.0,
            solar: 1.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "TN" => GridMix {
            coal: 0.0,
            gas: 85.0,
            oil: 6.0,
            nuclear: 0.0,
            hydro: 1.0,
            wind: 5.0,
            solar: 3.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "SN" => GridMix {
            coal: 11.0,
            gas: 40.0,
            oil: 32.0,
            nuclear: 0.0,
            hydro: 4.0,
            wind: 4.0,
            solar: 9.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        // --- South America ---
        "CL" => GridMix {
            coal: 17.0,
            gas: 15.0,
            oil: 3.0,
            nuclear: 0.0,
            hydro: 23.0,
            wind: 17.0,
            solar: 22.0,
            biomass: 3.0,
            geothermal: 0.0,
        },
        "CO" => GridMix {
            coal: 8.0,
            gas: 17.0,
            oil: 1.0,
            nuclear: 0.0,
            hydro: 65.0,
            wind: 4.0,
            solar: 4.0,
            biomass: 1.0,
            geothermal: 0.0,
        },
        "AR" => GridMix {
            coal: 1.0,
            gas: 56.0,
            oil: 3.0,
            nuclear: 7.0,
            hydro: 22.0,
            wind: 8.0,
            solar: 3.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "PE" => GridMix {
            coal: 3.0,
            gas: 38.0,
            oil: 5.0,
            nuclear: 0.0,
            hydro: 48.0,
            wind: 3.0,
            solar: 3.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "MX" => GridMix {
            coal: 5.0,
            gas: 55.0,
            oil: 8.0,
            nuclear: 4.0,
            hydro: 10.0,
            wind: 8.0,
            solar: 7.0,
            biomass: 3.0,
            geothermal: 0.0,
        },
        "EC" => GridMix {
            coal: 0.0,
            gas: 10.0,
            oil: 14.0,
            nuclear: 0.0,
            hydro: 70.0,
            wind: 2.0,
            solar: 2.0,
            biomass: 2.0,
            geothermal: 0.0,
        },
        "UY" => GridMix {
            coal: 0.0,
            gas: 3.0,
            oil: 4.0,
            nuclear: 0.0,
            hydro: 36.0,
            wind: 42.0,
            solar: 8.0,
            biomass: 7.0,
            geothermal: 0.0,
        },
        "PY" => GridMix {
            coal: 0.0,
            gas: 0.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 100.0,
            wind: 0.0,
            solar: 0.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "BO" => GridMix {
            coal: 0.0,
            gas: 65.0,
            oil: 5.0,
            nuclear: 0.0,
            hydro: 22.0,
            wind: 3.0,
            solar: 5.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "VE" => GridMix {
            coal: 0.0,
            gas: 20.0,
            oil: 12.0,
            nuclear: 0.0,
            hydro: 66.0,
            wind: 1.0,
            solar: 1.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        // --- Middle East ---
        "SA" => GridMix {
            coal: 0.0,
            gas: 57.0,
            oil: 38.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 1.0,
            solar: 4.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "AE" => GridMix {
            coal: 0.0,
            gas: 86.0,
            oil: 1.0,
            nuclear: 10.0,
            hydro: 0.0,
            wind: 0.0,
            solar: 3.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "TR" => GridMix {
            coal: 31.0,
            gas: 23.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 22.0,
            wind: 11.0,
            solar: 8.0,
            biomass: 2.0,
            geothermal: 3.0,
        },
        "IQ" => GridMix {
            coal: 0.0,
            gas: 70.0,
            oil: 28.0,
            nuclear: 0.0,
            hydro: 2.0,
            wind: 0.0,
            solar: 0.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "IR" => GridMix {
            coal: 0.0,
            gas: 80.0,
            oil: 12.0,
            nuclear: 2.0,
            hydro: 5.0,
            wind: 0.0,
            solar: 1.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "KW" => GridMix {
            coal: 0.0,
            gas: 61.0,
            oil: 38.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 0.0,
            solar: 1.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "OM" => GridMix {
            coal: 0.0,
            gas: 97.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 0.0,
            solar: 3.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "QA" => GridMix {
            coal: 0.0,
            gas: 96.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 0.0,
            solar: 4.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "JO" => GridMix {
            coal: 0.0,
            gas: 72.0,
            oil: 6.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 10.0,
            solar: 12.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        // --- Southeast Asia ---
        "ID" => GridMix {
            coal: 62.0,
            gas: 17.0,
            oil: 2.0,
            nuclear: 0.0,
            hydro: 7.0,
            wind: 0.0,
            solar: 1.0,
            biomass: 5.0,
            geothermal: 6.0,
        },
        "VN" => GridMix {
            coal: 46.0,
            gas: 13.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 28.0,
            wind: 5.0,
            solar: 7.0,
            biomass: 1.0,
            geothermal: 0.0,
        },
        "TH" => GridMix {
            coal: 17.0,
            gas: 55.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 5.0,
            wind: 3.0,
            solar: 6.0,
            biomass: 14.0,
            geothermal: 0.0,
        },
        "PH" => GridMix {
            coal: 47.0,
            gas: 22.0,
            oil: 2.0,
            nuclear: 0.0,
            hydro: 10.0,
            wind: 3.0,
            solar: 5.0,
            biomass: 2.0,
            geothermal: 9.0,
        },
        "PK" => GridMix {
            coal: 18.0,
            gas: 30.0,
            oil: 7.0,
            nuclear: 11.0,
            hydro: 27.0,
            wind: 4.0,
            solar: 3.0,
            biomass: 0.0,
            geothermal: 0.0,
        },
        "BD" => GridMix {
            coal: 5.0,
            gas: 82.0,
            oil: 4.0,
            nuclear: 0.0,
            hydro: 2.0,
            wind: 0.0,
            solar: 3.0,
            biomass: 4.0,
            geothermal: 0.0,
        },
        "MM" => GridMix {
            coal: 3.0,
            gas: 45.0,
            oil: 3.0,
            nuclear: 0.0,
            hydro: 46.0,
            wind: 0.0,
            solar: 1.0,
            biomass: 2.0,
            geothermal: 0.0,
        },
        "MY" => GridMix {
            coal: 38.0,
            gas: 40.0,
            oil: 1.0,
            nuclear: 0.0,
            hydro: 14.0,
            wind: 0.0,
            solar: 4.0,
            biomass: 3.0,
            geothermal: 0.0,
        },
        // --- Other ---
        "NZ" => GridMix {
            coal: 3.0,
            gas: 11.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 56.0,
            wind: 8.0,
            solar: 3.0,
            biomass: 2.0,
            geothermal: 17.0,
        },
        "TW" => GridMix {
            coal: 33.0,
            gas: 39.0,
            oil: 3.0,
            nuclear: 8.0,
            hydro: 2.0,
            wind: 4.0,
            solar: 10.0,
            biomass: 1.0,
            geothermal: 0.0,
        },
        _ => GridMix {
            coal: 36.0,
            gas: 23.0,
            oil: 3.0,
            nuclear: 10.0,
            hydro: 15.0,
            wind: 7.0,
            solar: 4.0,
            biomass: 2.0,
            geothermal: 0.0,
        }, // global average
    }
}

// ===========================================================================
// Backend 1: ElectricityMaps (optional paid)
// ===========================================================================

struct ElectricityMapsBackend {
    client: ElectricityMapsClient,
}

impl ElectricityMapsBackend {
    fn new(api_key: String) -> Self {
        Self {
            client: ElectricityMapsClient::new(ElectricityMapsConfig {
                api_key,
                ..ElectricityMapsConfig::default()
            }),
        }
    }
}

#[async_trait]
impl CarbonBackend for ElectricityMapsBackend {
    fn name(&self) -> &str {
        "electricitymaps"
    }

    fn supports_zone(&self, _zone: &str) -> bool {
        true // ElectricityMaps supports all zones
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        self.client.carbon_intensity(zone).await
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        self.client.power_breakdown(zone).await
    }
}

// ===========================================================================
// Backend 2: UK Carbon Intensity (free, no API key)
// ===========================================================================

struct UkCarbonBackend {
    http: reqwest::Client,
}

impl UkCarbonBackend {
    fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("failed to build reqwest client"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct UkIntensityEnvelope {
    data: Vec<UkIntensityData>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct UkIntensityData {
    #[serde(default)]
    from: Option<String>,
    intensity: UkIntensityValue,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct UkIntensityValue {
    #[serde(default)]
    forecast: Option<f64>,
    #[serde(default)]
    actual: Option<f64>,
    #[serde(default)]
    index: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UkGenerationEnvelope {
    data: UkGenerationData,
}

#[derive(Debug, Deserialize)]
struct UkGenerationData {
    generationmix: Vec<UkGenerationFuel>,
}

#[derive(Debug, Deserialize)]
struct UkGenerationFuel {
    fuel: String,
    perc: f64,
}

#[async_trait]
impl CarbonBackend for UkCarbonBackend {
    fn name(&self) -> &str {
        "uk-carbon-intensity"
    }

    fn supports_zone(&self, zone: &str) -> bool {
        zone == "GB"
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        if zone != "GB" {
            return Err(CarbonApiError::UnknownZone(zone.to_string()));
        }

        let resp = self
            .http
            .get("https://api.carbonintensity.org.uk/intensity")
            .header("Accept", "application/json")
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let envelope: UkIntensityEnvelope = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("UK CI parse error: {e}")))?;

        let data = envelope
            .data
            .first()
            .ok_or_else(|| CarbonApiError::Parse("empty UK CI response".into()))?;

        let intensity = data
            .intensity
            .actual
            .or(data.intensity.forecast)
            .unwrap_or(200.0);

        Ok(CarbonIntensityResponse {
            zone: "GB".to_string(),
            carbon_intensity_gco2_kwh: intensity,
            fossil_fuel_percentage: 0.0, // will be filled by power_breakdown
            datetime: Utc::now(),
            data_source: "carbonintensity.org.uk".to_string(),
            is_estimated: data.intensity.actual.is_none(),
        })
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        if zone != "GB" {
            return Err(CarbonApiError::UnknownZone(zone.to_string()));
        }

        let resp = self
            .http
            .get("https://api.carbonintensity.org.uk/generation")
            .header("Accept", "application/json")
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let envelope: UkGenerationEnvelope = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("UK generation parse error: {e}")))?;

        let mut wind = 0.0;
        let mut solar = 0.0;
        let mut hydro = 0.0;
        let mut nuclear = 0.0;
        let mut fossil = 0.0;

        for fuel in &envelope.data.generationmix {
            match fuel.fuel.as_str() {
                "wind" => wind = fuel.perc,
                "solar" => solar = fuel.perc,
                "hydro" => hydro = fuel.perc,
                "nuclear" => nuclear = fuel.perc,
                "gas" | "coal" | "oil" => fossil += fuel.perc,
                _ => {} // biomass, imports, other
            }
        }

        Ok(RenewableEnergyResponse {
            zone: "GB".to_string(),
            renewable_percentage: wind + solar + hydro,
            wind_percentage: wind,
            solar_percentage: solar,
            hydro_percentage: hydro,
            nuclear_percentage: nuclear,
            fossil_percentage: fossil,
            datetime: Utc::now(),
        })
    }
}

// ===========================================================================
// Backend 3: EIA (US zones — hourly, free)
// ===========================================================================

struct EiaBackend {
    api_key: String,
    http: reqwest::Client,
}

impl EiaBackend {
    fn new(api_key: String) -> Self {
        Self {
            api_key,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("failed to build reqwest client"),
        }
    }

    fn fuel_emission_factor(fuel: &str) -> f64 {
        match fuel {
            "COL" => IPCC_COAL,
            "NG" => IPCC_GAS,
            "OIL" | "PET" | "PC" => IPCC_OIL,
            "NUC" => IPCC_NUCLEAR,
            "SUN" => IPCC_SOLAR,
            "WND" => IPCC_WIND,
            "WAT" => IPCC_HYDRO,
            "GEO" => IPCC_GEOTHERMAL,
            "OTH" | "WAS" => IPCC_BIOMASS,
            _ => 400.0, // unknown → global average
        }
    }
}

#[derive(Debug, Deserialize)]
struct EiaResponse {
    response: EiaResponseBody,
}

#[derive(Debug, Deserialize)]
struct EiaResponseBody {
    data: Vec<EiaDataPoint>,
}

#[derive(Debug, Deserialize)]
struct EiaDataPoint {
    #[serde(default)]
    fueltype: Option<String>,
    #[serde(default)]
    value: Option<f64>,
}

#[async_trait]
impl CarbonBackend for EiaBackend {
    fn name(&self) -> &str {
        "eia"
    }

    fn supports_zone(&self, zone: &str) -> bool {
        zone_to_eia_ba(zone).is_some()
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        let ba =
            zone_to_eia_ba(zone).ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        let url = format!(
            "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={}&frequency=hourly&data[0]=value&facets[respondent][]={}&sort[0][column]=period&sort[0][direction]=desc&length=20",
            self.api_key, ba
        );

        let resp = self.http.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let eia: EiaResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("EIA parse error: {e}")))?;

        let mut total_mw = 0.0;
        let mut weighted_co2 = 0.0;
        let mut fossil_mw = 0.0;

        for point in &eia.response.data {
            let mw = point.value.unwrap_or(0.0).max(0.0);
            let fuel = point.fueltype.as_deref().unwrap_or("OTH");
            let factor = Self::fuel_emission_factor(fuel);
            weighted_co2 += mw * factor;
            total_mw += mw;
            if matches!(fuel, "COL" | "NG" | "OIL" | "PET" | "PC") {
                fossil_mw += mw;
            }
        }

        let intensity = if total_mw > 0.0 {
            weighted_co2 / total_mw
        } else {
            400.0
        };

        let fossil_pct = if total_mw > 0.0 {
            (fossil_mw / total_mw) * 100.0
        } else {
            0.0
        };

        Ok(CarbonIntensityResponse {
            zone: zone.to_string(),
            carbon_intensity_gco2_kwh: intensity,
            fossil_fuel_percentage: fossil_pct,
            datetime: Utc::now(),
            data_source: "eia.gov".to_string(),
            is_estimated: false,
        })
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        let ba =
            zone_to_eia_ba(zone).ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        let url = format!(
            "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={}&frequency=hourly&data[0]=value&facets[respondent][]={}&sort[0][column]=period&sort[0][direction]=desc&length=20",
            self.api_key, ba
        );

        let resp = self.http.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let eia: EiaResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("EIA parse error: {e}")))?;

        let mut total_mw = 0.0;
        let mut wind_mw = 0.0;
        let mut solar_mw = 0.0;
        let mut hydro_mw = 0.0;
        let mut nuclear_mw = 0.0;
        let mut fossil_mw = 0.0;

        for point in &eia.response.data {
            let mw = point.value.unwrap_or(0.0).max(0.0);
            let fuel = point.fueltype.as_deref().unwrap_or("OTH");
            total_mw += mw;
            match fuel {
                "WND" => wind_mw += mw,
                "SUN" => solar_mw += mw,
                "WAT" => hydro_mw += mw,
                "NUC" => nuclear_mw += mw,
                "COL" | "NG" | "OIL" | "PET" | "PC" => fossil_mw += mw,
                _ => {}
            }
        }

        let pct = |mw: f64| {
            if total_mw > 0.0 {
                (mw / total_mw) * 100.0
            } else {
                0.0
            }
        };

        Ok(RenewableEnergyResponse {
            zone: zone.to_string(),
            renewable_percentage: pct(wind_mw + solar_mw + hydro_mw),
            wind_percentage: pct(wind_mw),
            solar_percentage: pct(solar_mw),
            hydro_percentage: pct(hydro_mw),
            nuclear_percentage: pct(nuclear_mw),
            fossil_percentage: pct(fossil_mw),
            datetime: Utc::now(),
        })
    }
}

// ===========================================================================
// Backend 4: ENTSO-E (Europe — 15-min, free)
// ===========================================================================

/// Default ENTSO-E API endpoint (new endpoint, old `web-api.tp.entsoe.eu` deprecated Dec 2025).
const ENTSOE_DEFAULT_ENDPOINT: &str = "https://external-api.tp.entsoe.eu/api";

struct EntsoeBackend {
    token: String,
    endpoint: String,
    http: reqwest::Client,
}

impl EntsoeBackend {
    fn new(token: String, endpoint: Option<String>) -> Self {
        Self {
            token,
            endpoint: endpoint.unwrap_or_else(|| ENTSOE_DEFAULT_ENDPOINT.to_string()),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("failed to build reqwest client"),
        }
    }

    /// Map ENTSO-E PsrType B-codes to IPCC emission factors.
    fn psr_emission_factor(psr_type: &str) -> f64 {
        match psr_type {
            "B01" => IPCC_BIOMASS,               // biomass
            "B02" | "B05" => IPCC_COAL,          // brown coal, hard coal
            "B04" => IPCC_GAS,                   // fossil gas
            "B06" => IPCC_OIL,                   // fossil oil
            "B09" => IPCC_GEOTHERMAL,            // geothermal
            "B10" | "B11" | "B12" => IPCC_HYDRO, // hydro variants
            "B14" => IPCC_NUCLEAR,               // nuclear
            "B15" => 50.0,                       // other renewable
            "B16" => IPCC_SOLAR,                 // solar
            "B17" => IPCC_SOLAR,                 // solar thermal (CSP)
            "B18" | "B19" => IPCC_WIND,          // wind offshore/onshore
            _ => 400.0,                          // unknown
        }
    }

    fn psr_is_fossil(psr_type: &str) -> bool {
        matches!(psr_type, "B02" | "B04" | "B05" | "B06")
    }

    fn psr_is_wind(psr_type: &str) -> bool {
        matches!(psr_type, "B18" | "B19")
    }

    fn psr_is_solar(psr_type: &str) -> bool {
        matches!(psr_type, "B16" | "B17")
    }

    fn psr_is_hydro(psr_type: &str) -> bool {
        matches!(psr_type, "B10" | "B11" | "B12")
    }

    /// Parse the XML response to extract (psr_type, quantity_mw) pairs.
    fn parse_generation_xml(xml: &str) -> Vec<(String, f64)> {
        let mut results = Vec::new();
        let mut current_psr = String::new();

        // Simple stateful XML parse for ENTSO-E generation output.
        // Structure: <TimeSeries>...<MktPSRType><psrType>B04</psrType>...
        //            <Point><position>1</position><quantity>1234</quantity></Point>
        let reader = quick_xml::Reader::from_str(xml);
        let mut buf = Vec::new();
        let mut in_psr_type = false;
        let mut in_quantity = false;
        let mut reader = reader;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(quick_xml::events::Event::Start(ref e)) => {
                    let local = String::from_utf8_lossy(e.local_name().as_ref()).to_string();
                    if local == "psrType" {
                        in_psr_type = true;
                    } else if local == "quantity" {
                        in_quantity = true;
                    }
                }
                Ok(quick_xml::events::Event::Text(ref e)) => {
                    if in_psr_type {
                        current_psr = e.xml_content().unwrap_or_default().to_string();
                        in_psr_type = false;
                    } else if in_quantity {
                        if let Ok(val) = e.xml_content().unwrap_or_default().parse::<f64>()
                            && !current_psr.is_empty()
                        {
                            results.push((current_psr.clone(), val));
                        }
                        in_quantity = false;
                    }
                }
                Ok(quick_xml::events::Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
            buf.clear();
        }

        results
    }
}

#[async_trait]
impl CarbonBackend for EntsoeBackend {
    fn name(&self) -> &str {
        "entsoe"
    }

    fn supports_zone(&self, zone: &str) -> bool {
        zone_to_entsoe_area(zone).is_some()
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        let area = zone_to_entsoe_area(zone)
            .ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        let now = Utc::now();
        let period_start = (now - chrono::Duration::hours(2))
            .format("%Y%m%d%H00")
            .to_string();
        let period_end = now.format("%Y%m%d%H00").to_string();

        let url = format!(
            "{}?securityToken={}&documentType=A75&processType=A16&in_Domain={}&periodStart={}&periodEnd={}",
            self.endpoint, self.token, area, period_start, period_end
        );

        let resp = self.http.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let xml = resp
            .text()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("ENTSO-E body read error: {e}")))?;

        let gen_data = Self::parse_generation_xml(&xml);
        if gen_data.is_empty() {
            return Err(CarbonApiError::Parse(
                "ENTSO-E returned no generation data".into(),
            ));
        }

        let mut total_mw = 0.0;
        let mut weighted_co2 = 0.0;
        let mut fossil_mw = 0.0;

        for (psr, mw) in &gen_data {
            let factor = Self::psr_emission_factor(psr);
            weighted_co2 += mw * factor;
            total_mw += mw;
            if Self::psr_is_fossil(psr) {
                fossil_mw += mw;
            }
        }

        let intensity = if total_mw > 0.0 {
            weighted_co2 / total_mw
        } else {
            400.0
        };

        Ok(CarbonIntensityResponse {
            zone: zone.to_string(),
            carbon_intensity_gco2_kwh: intensity,
            fossil_fuel_percentage: if total_mw > 0.0 {
                (fossil_mw / total_mw) * 100.0
            } else {
                0.0
            },
            datetime: now,
            data_source: "entsoe.eu".to_string(),
            is_estimated: false,
        })
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        let area = zone_to_entsoe_area(zone)
            .ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        let now = Utc::now();
        let period_start = (now - chrono::Duration::hours(2))
            .format("%Y%m%d%H00")
            .to_string();
        let period_end = now.format("%Y%m%d%H00").to_string();

        let url = format!(
            "{}?securityToken={}&documentType=A75&processType=A16&in_Domain={}&periodStart={}&periodEnd={}",
            self.endpoint, self.token, area, period_start, period_end
        );

        let resp = self.http.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let xml = resp
            .text()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("ENTSO-E body read error: {e}")))?;

        let gen_data = Self::parse_generation_xml(&xml);

        let mut total_mw = 0.0;
        let mut wind_mw = 0.0;
        let mut solar_mw = 0.0;
        let mut hydro_mw = 0.0;
        let mut nuclear_mw = 0.0;
        let mut fossil_mw = 0.0;

        for (psr, mw) in &gen_data {
            total_mw += mw;
            if Self::psr_is_wind(psr) {
                wind_mw += mw;
            }
            if Self::psr_is_solar(psr) {
                solar_mw += mw;
            }
            if Self::psr_is_hydro(psr) {
                hydro_mw += mw;
            }
            if psr == "B14" {
                nuclear_mw += mw;
            }
            if Self::psr_is_fossil(psr) {
                fossil_mw += mw;
            }
        }

        let pct = |mw: f64| {
            if total_mw > 0.0 {
                (mw / total_mw) * 100.0
            } else {
                0.0
            }
        };

        Ok(RenewableEnergyResponse {
            zone: zone.to_string(),
            renewable_percentage: pct(wind_mw + solar_mw + hydro_mw),
            wind_percentage: pct(wind_mw),
            solar_percentage: pct(solar_mw),
            hydro_percentage: pct(hydro_mw),
            nuclear_percentage: pct(nuclear_mw),
            fossil_percentage: pct(fossil_mw),
            datetime: Utc::now(),
        })
    }
}

// ===========================================================================
// Backend 5: OpenNEM / OpenElectricity (Australia — 5-min, free)
// ===========================================================================

#[allow(dead_code)] // Fields reserved for real API integration
struct OpenNemBackend {
    api_key: String,
    http: reqwest::Client,
}

impl OpenNemBackend {
    fn new(api_key: String) -> Self {
        Self {
            api_key,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("failed to build reqwest client"),
        }
    }
}

#[async_trait]
impl CarbonBackend for OpenNemBackend {
    fn name(&self) -> &str {
        "opennem"
    }

    fn supports_zone(&self, zone: &str) -> bool {
        zone_to_opennem_region(zone).is_some()
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        // OpenNEM API requires specific endpoint discovery; fall through
        // to IPCC static for now if the API shape changes.
        let _region = zone_to_opennem_region(zone)
            .ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        // Use IPCC static data for Australia as a reliable baseline.
        // Real OpenNEM integration can be layered on once API key is obtained.
        let mix = country_grid_mix("AU");

        Ok(CarbonIntensityResponse {
            zone: zone.to_string(),
            carbon_intensity_gco2_kwh: mix.carbon_intensity(),
            fossil_fuel_percentage: mix.fossil_pct(),
            datetime: Utc::now(),
            data_source: "opennem/ipcc-static".to_string(),
            is_estimated: true,
        })
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        let _region = zone_to_opennem_region(zone)
            .ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        let mix = country_grid_mix("AU");

        Ok(RenewableEnergyResponse {
            zone: zone.to_string(),
            renewable_percentage: mix.renewable_pct(),
            wind_percentage: mix.wind,
            solar_percentage: mix.solar,
            hydro_percentage: mix.hydro,
            nuclear_percentage: mix.nuclear,
            fossil_percentage: mix.fossil_pct(),
            datetime: Utc::now(),
        })
    }
}

// ===========================================================================
// Backend 6: Ember Climate (global monthly, free)
// ===========================================================================

struct EmberBackend {
    api_key: String,
    http: reqwest::Client,
}

impl EmberBackend {
    fn new(api_key: String) -> Self {
        Self {
            api_key,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("failed to build reqwest client"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct EmberResponse {
    #[serde(default)]
    data: Vec<EmberDataPoint>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct EmberDataPoint {
    #[serde(default)]
    carbon_intensity: Option<f64>,
    #[serde(default)]
    generation_from_fossil: Option<f64>,
    #[serde(default)]
    generation_from_renewables: Option<f64>,
}

#[async_trait]
impl CarbonBackend for EmberBackend {
    fn name(&self) -> &str {
        "ember"
    }

    fn supports_zone(&self, zone: &str) -> bool {
        zone_to_country_iso3(zone).is_some()
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        let country = zone_to_country_iso3(zone)
            .ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        let url = format!(
            "https://api.ember-energy.org/v1/carbon-intensity/yearly?entity_code={}&api_key={}&sort=-date&limit=1",
            country, self.api_key
        );

        let resp = self.http.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let ember: EmberResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("Ember parse error: {e}")))?;

        let point = ember.data.first().ok_or_else(|| {
            CarbonApiError::Parse(format!("Ember returned no data for {country}"))
        })?;

        Ok(CarbonIntensityResponse {
            zone: zone.to_string(),
            carbon_intensity_gco2_kwh: point.carbon_intensity.unwrap_or(400.0),
            fossil_fuel_percentage: point.generation_from_fossil.unwrap_or(0.0),
            datetime: Utc::now(),
            data_source: "ember-energy.org".to_string(),
            is_estimated: false,
        })
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        let _country = zone_to_country_iso3(zone)
            .ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        // Fall back to IPCC static for detailed breakdown since Ember
        // yearly endpoint doesn't split wind/solar/hydro individually.
        let country_key = zone_to_country(zone);
        let mix = country_grid_mix(country_key);

        Ok(RenewableEnergyResponse {
            zone: zone.to_string(),
            renewable_percentage: mix.renewable_pct(),
            wind_percentage: mix.wind,
            solar_percentage: mix.solar,
            hydro_percentage: mix.hydro,
            nuclear_percentage: mix.nuclear,
            fossil_percentage: mix.fossil_pct(),
            datetime: Utc::now(),
        })
    }
}

// ===========================================================================
// Backend 7: EPIAS (Turkey — hourly, free registration)
// ===========================================================================

struct EpiasBackend {
    token: String,
    http: reqwest::Client,
}

impl EpiasBackend {
    fn new(token: String) -> Self {
        Self {
            token,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("failed to build reqwest client"),
        }
    }

    fn fuel_emission_factor(fuel: &str) -> f64 {
        match fuel {
            "LIGNITE" | "HARD_COAL" | "IMPORTED_COAL" | "ASPHALTITE_COAL" => IPCC_COAL,
            "NATURAL_GAS" | "LNG" => IPCC_GAS,
            "FUEL_OIL" | "NAPHTHA" => IPCC_OIL,
            "WIND" => IPCC_WIND,
            "SUN" | "SOLAR" => IPCC_SOLAR,
            "HYDRAULIC" | "RIVER" | "DAM" => IPCC_HYDRO,
            "GEOTHERMAL" => IPCC_GEOTHERMAL,
            "BIOMASS" | "BIOGAS" | "WASTE" => IPCC_BIOMASS,
            "NUCLEAR" => IPCC_NUCLEAR,
            _ => 400.0,
        }
    }
}

#[derive(Debug, Deserialize)]
struct EpiasGenerationItem {
    #[serde(default, alias = "fuelType", alias = "fuel_type")]
    fuel_type: Option<String>,
    #[serde(default, alias = "generation")]
    value: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct EpiasResponseBody {
    #[serde(default, alias = "items")]
    items: Vec<EpiasGenerationItem>,
}

#[async_trait]
impl CarbonBackend for EpiasBackend {
    fn name(&self) -> &str {
        "epias"
    }

    fn supports_zone(&self, zone: &str) -> bool {
        zone == "TR"
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        if zone != "TR" {
            return Err(CarbonApiError::UnknownZone(zone.to_string()));
        }

        let now = Utc::now();
        let start = (now - chrono::Duration::hours(1)).format("%Y-%m-%dT%H:%M:%S+03:00");
        let end = now.format("%Y-%m-%dT%H:%M:%S+03:00");

        let url = format!(
            "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/realtime-generation?startDate={}&endDate={}",
            start, end
        );

        let resp = self
            .http
            .get(&url)
            .header("TGT", &self.token)
            .header("Accept", "application/json")
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let body: EpiasResponseBody = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("EPIAS parse error: {e}")))?;

        let mut total_mw = 0.0;
        let mut weighted_co2 = 0.0;
        let mut fossil_mw = 0.0;

        for item in &body.items {
            let mw = item.value.unwrap_or(0.0).max(0.0);
            let fuel = item.fuel_type.as_deref().unwrap_or("OTHER");
            let factor = Self::fuel_emission_factor(fuel);
            weighted_co2 += mw * factor;
            total_mw += mw;
            if matches!(
                fuel,
                "LIGNITE"
                    | "HARD_COAL"
                    | "IMPORTED_COAL"
                    | "ASPHALTITE_COAL"
                    | "NATURAL_GAS"
                    | "LNG"
                    | "FUEL_OIL"
                    | "NAPHTHA"
            ) {
                fossil_mw += mw;
            }
        }

        let intensity = if total_mw > 0.0 {
            weighted_co2 / total_mw
        } else {
            400.0
        };
        let fossil_pct = if total_mw > 0.0 {
            (fossil_mw / total_mw) * 100.0
        } else {
            0.0
        };

        Ok(CarbonIntensityResponse {
            zone: "TR".to_string(),
            carbon_intensity_gco2_kwh: intensity,
            fossil_fuel_percentage: fossil_pct,
            datetime: now,
            data_source: "epias".to_string(),
            is_estimated: false,
        })
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        // Reuse the same generation endpoint to extract breakdown.
        if zone != "TR" {
            return Err(CarbonApiError::UnknownZone(zone.to_string()));
        }

        let now = Utc::now();
        let start = (now - chrono::Duration::hours(1)).format("%Y-%m-%dT%H:%M:%S+03:00");
        let end = now.format("%Y-%m-%dT%H:%M:%S+03:00");

        let url = format!(
            "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/realtime-generation?startDate={}&endDate={}",
            start, end
        );

        let resp = self
            .http
            .get(&url)
            .header("TGT", &self.token)
            .header("Accept", "application/json")
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let body: EpiasResponseBody = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("EPIAS parse error: {e}")))?;

        let mut total = 0.0;
        let mut wind = 0.0;
        let mut solar = 0.0;
        let mut hydro = 0.0;
        let mut nuclear = 0.0;
        let mut fossil = 0.0;

        for item in &body.items {
            let mw = item.value.unwrap_or(0.0).max(0.0);
            let fuel = item.fuel_type.as_deref().unwrap_or("OTHER");
            total += mw;
            match fuel {
                "WIND" => wind += mw,
                "SUN" | "SOLAR" => solar += mw,
                "HYDRAULIC" | "RIVER" | "DAM" => hydro += mw,
                "NUCLEAR" => nuclear += mw,
                "LIGNITE" | "HARD_COAL" | "IMPORTED_COAL" | "ASPHALTITE_COAL" | "NATURAL_GAS"
                | "LNG" | "FUEL_OIL" | "NAPHTHA" => fossil += mw,
                _ => {}
            }
        }

        let pct = |v: f64| {
            if total > 0.0 {
                (v / total) * 100.0
            } else {
                0.0
            }
        };

        Ok(RenewableEnergyResponse {
            zone: "TR".to_string(),
            renewable_percentage: pct(wind + solar + hydro),
            wind_percentage: pct(wind),
            solar_percentage: pct(solar),
            hydro_percentage: pct(hydro),
            nuclear_percentage: pct(nuclear),
            fossil_percentage: pct(fossil),
            datetime: now,
        })
    }
}

// ===========================================================================
// Backend 8: CEN (Chile — hourly, no auth needed)
// ===========================================================================

struct CenBackend {
    http: reqwest::Client,
}

impl CenBackend {
    fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("failed to build reqwest client"),
        }
    }

    fn fuel_emission_factor(fuel: &str) -> f64 {
        let fuel_lower = fuel.to_lowercase();
        if fuel_lower.contains("carbon") || fuel_lower.contains("coal") {
            IPCC_COAL
        } else if fuel_lower.contains("gas") || fuel_lower.contains("gnl") {
            IPCC_GAS
        } else if fuel_lower.contains("diesel") || fuel_lower.contains("petro") {
            IPCC_OIL
        } else if fuel_lower.contains("eolic") || fuel_lower.contains("wind") {
            IPCC_WIND
        } else if fuel_lower.contains("solar") || fuel_lower.contains("fotovol") {
            IPCC_SOLAR
        } else if fuel_lower.contains("hidr") || fuel_lower.contains("hidra") {
            IPCC_HYDRO
        } else if fuel_lower.contains("geoter") {
            IPCC_GEOTHERMAL
        } else if fuel_lower.contains("biom") || fuel_lower.contains("biog") {
            IPCC_BIOMASS
        } else {
            400.0
        }
    }
}

#[derive(Debug, Deserialize)]
struct CenGenerationItem {
    #[serde(default, alias = "tecnologia", alias = "technology")]
    technology: Option<String>,
    #[serde(
        default,
        alias = "generacion_mwh",
        alias = "generation_mwh",
        alias = "value"
    )]
    value: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct CenResponse {
    #[serde(default, alias = "data")]
    data: Vec<CenGenerationItem>,
}

#[async_trait]
impl CarbonBackend for CenBackend {
    fn name(&self) -> &str {
        "cen"
    }

    fn supports_zone(&self, zone: &str) -> bool {
        zone == "CL" || zone.starts_with("CL-")
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        if !self.supports_zone(zone) {
            return Err(CarbonApiError::UnknownZone(zone.to_string()));
        }

        let now = Utc::now();
        let date = now.format("%Y-%m-%d");

        let url = format!(
            "https://sipub.coordinador.cl/api/v1/recursos/generacion_bruta_real?fecha={}&tipo=horario",
            date
        );

        let resp = self
            .http
            .get(&url)
            .header("Accept", "application/json")
            .header("User-Agent", "invisible-infrastructure/0.1")
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let body: CenResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("CEN parse error: {e}")))?;

        let mut total_mwh = 0.0;
        let mut weighted_co2 = 0.0;
        let mut fossil_mwh = 0.0;

        for item in &body.data {
            let mwh = item.value.unwrap_or(0.0).max(0.0);
            let tech = item.technology.as_deref().unwrap_or("Otro");
            let factor = Self::fuel_emission_factor(tech);
            weighted_co2 += mwh * factor;
            total_mwh += mwh;
            if factor >= IPCC_GAS {
                fossil_mwh += mwh;
            }
        }

        let intensity = if total_mwh > 0.0 {
            weighted_co2 / total_mwh
        } else {
            300.0
        };
        let fossil_pct = if total_mwh > 0.0 {
            (fossil_mwh / total_mwh) * 100.0
        } else {
            0.0
        };

        Ok(CarbonIntensityResponse {
            zone: zone.to_string(),
            carbon_intensity_gco2_kwh: intensity,
            fossil_fuel_percentage: fossil_pct,
            datetime: now,
            data_source: "cen".to_string(),
            is_estimated: false,
        })
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        if !self.supports_zone(zone) {
            return Err(CarbonApiError::UnknownZone(zone.to_string()));
        }

        let now = Utc::now();
        let date = now.format("%Y-%m-%d");

        let url = format!(
            "https://sipub.coordinador.cl/api/v1/recursos/generacion_bruta_real?fecha={}&tipo=horario",
            date
        );

        let resp = self
            .http
            .get(&url)
            .header("Accept", "application/json")
            .header("User-Agent", "invisible-infrastructure/0.1")
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let body: CenResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("CEN parse error: {e}")))?;

        let mut total = 0.0;
        let mut wind = 0.0;
        let mut solar = 0.0;
        let mut hydro = 0.0;
        let mut fossil = 0.0;

        for item in &body.data {
            let mwh = item.value.unwrap_or(0.0).max(0.0);
            let tech = item.technology.as_deref().unwrap_or("Otro");
            let tech_lower = tech.to_lowercase();
            total += mwh;
            if tech_lower.contains("eolic") || tech_lower.contains("wind") {
                wind += mwh;
            } else if tech_lower.contains("solar") || tech_lower.contains("fotovol") {
                solar += mwh;
            } else if tech_lower.contains("hidr") || tech_lower.contains("hidra") {
                hydro += mwh;
            } else if tech_lower.contains("carbon")
                || tech_lower.contains("coal")
                || tech_lower.contains("gas")
                || tech_lower.contains("gnl")
                || tech_lower.contains("diesel")
                || tech_lower.contains("petro")
            {
                fossil += mwh;
            }
        }

        let pct = |v: f64| {
            if total > 0.0 {
                (v / total) * 100.0
            } else {
                0.0
            }
        };

        Ok(RenewableEnergyResponse {
            zone: zone.to_string(),
            renewable_percentage: pct(wind + solar + hydro),
            wind_percentage: pct(wind),
            solar_percentage: pct(solar),
            hydro_percentage: pct(hydro),
            nuclear_percentage: 0.0,
            fossil_percentage: pct(fossil),
            datetime: now,
        })
    }
}

// ===========================================================================
// Backends 9–12: HTML Scraper Backends (feature-gated)
// ===========================================================================

#[cfg(feature = "scraper-backends")]
mod scraper_backends {
    use super::*;
    use ::scraper::{Html, Selector};

    // ── Helper: extract gCO2/kWh from generation-by-fuel MW data ──────────

    fn weighted_intensity(fuels: &[(f64, f64)]) -> (f64, f64) {
        let mut total = 0.0;
        let mut weighted = 0.0;
        let mut fossil = 0.0;
        for &(mw, factor) in fuels {
            total += mw;
            weighted += mw * factor;
            if factor >= IPCC_GAS {
                fossil += mw;
            }
        }
        let intensity = if total > 0.0 { weighted / total } else { 400.0 };
        let fossil_pct = if total > 0.0 {
            (fossil / total) * 100.0
        } else {
            0.0
        };
        (intensity, fossil_pct)
    }

    fn renewable_breakdown(fuels: &[(&str, f64)]) -> (f64, f64, f64, f64, f64) {
        let total: f64 = fuels.iter().map(|(_, mw)| *mw).sum();
        if total <= 0.0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let mut wind = 0.0;
        let mut solar = 0.0;
        let mut hydro = 0.0;
        let mut nuclear = 0.0;
        let mut fossil = 0.0;
        for &(fuel, mw) in fuels {
            let fl = fuel.to_lowercase();
            if fl.contains("wind") || fl.contains("eóli") || fl.contains("eolic") {
                wind += mw;
            } else if fl.contains("solar") || fl.contains("fotov") {
                solar += mw;
            } else if fl.contains("hydr") || fl.contains("hidr") || fl.contains("water") {
                hydro += mw;
            } else if fl.contains("nucl") {
                nuclear += mw;
            } else if fl.contains("coal")
                || fl.contains("gas")
                || fl.contains("oil")
                || fl.contains("diesel")
                || fl.contains("thermal")
                || fl.contains("térm")
            {
                fossil += mw;
            }
        }
        let pct = |v: f64| (v / total) * 100.0;
        (pct(wind), pct(solar), pct(hydro), pct(nuclear), pct(fossil))
    }

    // ── Backend 9: ONS (Brazil) ───────────────────────────────────────────

    pub(crate) struct OnsBackend {
        http: reqwest::Client,
    }

    impl OnsBackend {
        pub(crate) fn new() -> Self {
            Self {
                http: reqwest::Client::builder()
                    .timeout(Duration::from_secs(20))
                    .build()
                    .expect("failed to build reqwest client"),
            }
        }

        /// Parse generation data from ONS HTML.
        pub fn parse_generation_html(html: &str) -> Vec<(String, f64)> {
            let doc = Html::parse_document(html);
            let row_sel = Selector::parse("tr, .generation-row, [data-fuel]").unwrap();
            let td_sel = Selector::parse("td, span, .value").unwrap();
            let mut results = Vec::new();

            for row in doc.select(&row_sel) {
                let cells: Vec<String> = row
                    .select(&td_sel)
                    .map(|c| c.text().collect::<String>().trim().to_string())
                    .collect();

                if cells.len() >= 2 {
                    let fuel = cells[0].clone();
                    if let Ok(mw) = cells[cells.len() - 1]
                        .replace('.', "")
                        .replace(',', ".")
                        .parse::<f64>()
                    {
                        if mw > 0.0 && !fuel.is_empty() {
                            results.push((fuel, mw));
                        }
                    }
                }
            }
            results
        }

        fn fuel_factor(fuel: &str) -> f64 {
            let fl = fuel.to_lowercase();
            if fl.contains("térm") || fl.contains("therm") {
                // Brazil "Térmica" is mostly gas + biomass, use average
                IPCC_GAS
            } else if fl.contains("eóli") || fl.contains("wind") {
                IPCC_WIND
            } else if fl.contains("solar") {
                IPCC_SOLAR
            } else if fl.contains("hidr") || fl.contains("hydr") {
                IPCC_HYDRO
            } else if fl.contains("nucl") {
                IPCC_NUCLEAR
            } else if fl.contains("biomass") || fl.contains("bio") {
                IPCC_BIOMASS
            } else {
                400.0
            }
        }
    }

    #[async_trait]
    impl CarbonBackend for OnsBackend {
        fn name(&self) -> &str {
            "ons"
        }

        fn supports_zone(&self, zone: &str) -> bool {
            zone == "BR" || zone.starts_with("BR-")
        }

        async fn carbon_intensity(
            &self,
            zone: &str,
        ) -> Result<CarbonIntensityResponse, CarbonApiError> {
            if !self.supports_zone(zone) {
                return Err(CarbonApiError::UnknownZone(zone.to_string()));
            }

            let resp = self
                .http
                .get("https://www.ons.org.br/paginas/energia-agora/carga-e-geracao")
                .header("User-Agent", "invisible-infrastructure/0.1")
                .send()
                .await?;

            if !resp.status().is_success() {
                return Err(CarbonApiError::ApiError {
                    status: resp.status().as_u16(),
                    message: resp.text().await.unwrap_or_default(),
                });
            }

            let html = resp
                .text()
                .await
                .map_err(|e| CarbonApiError::Parse(format!("ONS text error: {e}")))?;

            let fuels = Self::parse_generation_html(&html);
            if fuels.is_empty() {
                return Err(CarbonApiError::Parse("empty ONS generation data".into()));
            }

            let weighted: Vec<(f64, f64)> = fuels
                .iter()
                .map(|(f, mw)| (*mw, Self::fuel_factor(f)))
                .collect();
            let (intensity, fossil_pct) = weighted_intensity(&weighted);

            Ok(CarbonIntensityResponse {
                zone: zone.to_string(),
                carbon_intensity_gco2_kwh: intensity,
                fossil_fuel_percentage: fossil_pct,
                datetime: Utc::now(),
                data_source: "ons".to_string(),
                is_estimated: false,
            })
        }

        async fn power_breakdown(
            &self,
            zone: &str,
        ) -> Result<RenewableEnergyResponse, CarbonApiError> {
            if !self.supports_zone(zone) {
                return Err(CarbonApiError::UnknownZone(zone.to_string()));
            }

            let resp = self
                .http
                .get("https://www.ons.org.br/paginas/energia-agora/carga-e-geracao")
                .header("User-Agent", "invisible-infrastructure/0.1")
                .send()
                .await?;

            if !resp.status().is_success() {
                return Err(CarbonApiError::ApiError {
                    status: resp.status().as_u16(),
                    message: resp.text().await.unwrap_or_default(),
                });
            }

            let html = resp
                .text()
                .await
                .map_err(|e| CarbonApiError::Parse(format!("ONS text error: {e}")))?;

            let fuels = Self::parse_generation_html(&html);
            let fuel_refs: Vec<(&str, f64)> =
                fuels.iter().map(|(f, mw)| (f.as_str(), *mw)).collect();
            let (wind, solar, hydro, nuclear, fossil) = renewable_breakdown(&fuel_refs);

            Ok(RenewableEnergyResponse {
                zone: zone.to_string(),
                renewable_percentage: wind + solar + hydro,
                wind_percentage: wind,
                solar_percentage: solar,
                hydro_percentage: hydro,
                nuclear_percentage: nuclear,
                fossil_percentage: fossil,
                datetime: Utc::now(),
            })
        }
    }

    // ── Backend 10: CAMMESA (Argentina) ───────────────────────────────────

    pub(crate) struct CammesaBackend {
        http: reqwest::Client,
    }

    impl CammesaBackend {
        pub(crate) fn new() -> Self {
            Self {
                http: reqwest::Client::builder()
                    .timeout(Duration::from_secs(20))
                    .build()
                    .expect("failed to build reqwest client"),
            }
        }

        pub fn parse_generation_html(html: &str) -> Vec<(String, f64)> {
            let doc = Html::parse_document(html);
            let row_sel = Selector::parse("tr, .gen-row, [data-type]").unwrap();
            let td_sel = Selector::parse("td, span, .value").unwrap();
            let mut results = Vec::new();

            for row in doc.select(&row_sel) {
                let cells: Vec<String> = row
                    .select(&td_sel)
                    .map(|c| c.text().collect::<String>().trim().to_string())
                    .collect();

                if cells.len() >= 2 {
                    let fuel = cells[0].clone();
                    if let Ok(mw) = cells[cells.len() - 1]
                        .replace('.', "")
                        .replace(',', ".")
                        .parse::<f64>()
                    {
                        if mw > 0.0 && !fuel.is_empty() {
                            results.push((fuel, mw));
                        }
                    }
                }
            }
            results
        }

        fn fuel_factor(fuel: &str) -> f64 {
            let fl = fuel.to_lowercase();
            if fl.contains("térm") || fl.contains("therm") || fl.contains("gas") {
                IPCC_GAS
            } else if fl.contains("eóli") || fl.contains("wind") {
                IPCC_WIND
            } else if fl.contains("solar") {
                IPCC_SOLAR
            } else if fl.contains("hidr") || fl.contains("hydr") {
                IPCC_HYDRO
            } else if fl.contains("nucl") {
                IPCC_NUCLEAR
            } else {
                400.0
            }
        }
    }

    #[async_trait]
    impl CarbonBackend for CammesaBackend {
        fn name(&self) -> &str {
            "cammesa"
        }

        fn supports_zone(&self, zone: &str) -> bool {
            zone == "AR" || zone.starts_with("AR-")
        }

        async fn carbon_intensity(
            &self,
            zone: &str,
        ) -> Result<CarbonIntensityResponse, CarbonApiError> {
            if !self.supports_zone(zone) {
                return Err(CarbonApiError::UnknownZone(zone.to_string()));
            }

            let resp = self
                .http
                .get("https://portalweb.cammesa.com/Memnet1/Pages/Informes/Graficos/generacionTipoWN.aspx")
                .header("User-Agent", "invisible-infrastructure/0.1")
                .send()
                .await?;

            if !resp.status().is_success() {
                return Err(CarbonApiError::ApiError {
                    status: resp.status().as_u16(),
                    message: resp.text().await.unwrap_or_default(),
                });
            }

            let html = resp
                .text()
                .await
                .map_err(|e| CarbonApiError::Parse(format!("CAMMESA text error: {e}")))?;

            let fuels = Self::parse_generation_html(&html);
            if fuels.is_empty() {
                return Err(CarbonApiError::Parse(
                    "empty CAMMESA generation data".into(),
                ));
            }

            let weighted: Vec<(f64, f64)> = fuels
                .iter()
                .map(|(f, mw)| (*mw, Self::fuel_factor(f)))
                .collect();
            let (intensity, fossil_pct) = weighted_intensity(&weighted);

            Ok(CarbonIntensityResponse {
                zone: zone.to_string(),
                carbon_intensity_gco2_kwh: intensity,
                fossil_fuel_percentage: fossil_pct,
                datetime: Utc::now(),
                data_source: "cammesa".to_string(),
                is_estimated: false,
            })
        }

        async fn power_breakdown(
            &self,
            zone: &str,
        ) -> Result<RenewableEnergyResponse, CarbonApiError> {
            if !self.supports_zone(zone) {
                return Err(CarbonApiError::UnknownZone(zone.to_string()));
            }

            let resp = self
                .http
                .get("https://portalweb.cammesa.com/Memnet1/Pages/Informes/Graficos/generacionTipoWN.aspx")
                .header("User-Agent", "invisible-infrastructure/0.1")
                .send()
                .await?;

            if !resp.status().is_success() {
                return Err(CarbonApiError::ApiError {
                    status: resp.status().as_u16(),
                    message: resp.text().await.unwrap_or_default(),
                });
            }

            let html = resp
                .text()
                .await
                .map_err(|e| CarbonApiError::Parse(format!("CAMMESA text error: {e}")))?;

            let fuels = Self::parse_generation_html(&html);
            let fuel_refs: Vec<(&str, f64)> =
                fuels.iter().map(|(f, mw)| (f.as_str(), *mw)).collect();
            let (wind, solar, hydro, nuclear, fossil) = renewable_breakdown(&fuel_refs);

            Ok(RenewableEnergyResponse {
                zone: zone.to_string(),
                renewable_percentage: wind + solar + hydro,
                wind_percentage: wind,
                solar_percentage: solar,
                hydro_percentage: hydro,
                nuclear_percentage: nuclear,
                fossil_percentage: fossil,
                datetime: Utc::now(),
            })
        }
    }

    // ── Backend 11: NigGrid (Nigeria) ─────────────────────────────────────

    pub(crate) struct NiggridBackend {
        http: reqwest::Client,
    }

    impl NiggridBackend {
        pub(crate) fn new() -> Self {
            Self {
                http: reqwest::Client::builder()
                    .timeout(Duration::from_secs(20))
                    .build()
                    .expect("failed to build reqwest client"),
            }
        }

        pub fn parse_generation_html(html: &str) -> Vec<(String, f64)> {
            let doc = Html::parse_document(html);
            let row_sel = Selector::parse("tr, .gen-item, [data-plant]").unwrap();
            let td_sel = Selector::parse("td, span, .value").unwrap();
            let mut results = Vec::new();

            for row in doc.select(&row_sel) {
                let cells: Vec<String> = row
                    .select(&td_sel)
                    .map(|c| c.text().collect::<String>().trim().to_string())
                    .collect();

                if cells.len() >= 2 {
                    let fuel = cells[0].clone();
                    if let Ok(mw) = cells[cells.len() - 1]
                        .replace('.', "")
                        .replace(',', ".")
                        .parse::<f64>()
                    {
                        if mw > 0.0 && !fuel.is_empty() {
                            results.push((fuel, mw));
                        }
                    }
                }
            }
            results
        }

        fn fuel_factor(fuel: &str) -> f64 {
            let fl = fuel.to_lowercase();
            if fl.contains("gas") || fl.contains("thermal") {
                IPCC_GAS
            } else if fl.contains("hydro") {
                IPCC_HYDRO
            } else if fl.contains("solar") {
                IPCC_SOLAR
            } else if fl.contains("wind") {
                IPCC_WIND
            } else {
                IPCC_GAS // Nigeria is predominantly gas
            }
        }
    }

    #[async_trait]
    impl CarbonBackend for NiggridBackend {
        fn name(&self) -> &str {
            "niggrid"
        }

        fn supports_zone(&self, zone: &str) -> bool {
            zone == "NG"
        }

        async fn carbon_intensity(
            &self,
            zone: &str,
        ) -> Result<CarbonIntensityResponse, CarbonApiError> {
            if zone != "NG" {
                return Err(CarbonApiError::UnknownZone(zone.to_string()));
            }

            let resp = self
                .http
                .get("https://niggrid.com/GenerationProfile")
                .header("User-Agent", "invisible-infrastructure/0.1")
                .send()
                .await?;

            if !resp.status().is_success() {
                return Err(CarbonApiError::ApiError {
                    status: resp.status().as_u16(),
                    message: resp.text().await.unwrap_or_default(),
                });
            }

            let html = resp
                .text()
                .await
                .map_err(|e| CarbonApiError::Parse(format!("NigGrid text error: {e}")))?;

            let fuels = Self::parse_generation_html(&html);
            if fuels.is_empty() {
                return Err(CarbonApiError::Parse(
                    "empty NigGrid generation data".into(),
                ));
            }

            let weighted: Vec<(f64, f64)> = fuels
                .iter()
                .map(|(f, mw)| (*mw, Self::fuel_factor(f)))
                .collect();
            let (intensity, fossil_pct) = weighted_intensity(&weighted);

            Ok(CarbonIntensityResponse {
                zone: "NG".to_string(),
                carbon_intensity_gco2_kwh: intensity,
                fossil_fuel_percentage: fossil_pct,
                datetime: Utc::now(),
                data_source: "niggrid".to_string(),
                is_estimated: false,
            })
        }

        async fn power_breakdown(
            &self,
            zone: &str,
        ) -> Result<RenewableEnergyResponse, CarbonApiError> {
            if zone != "NG" {
                return Err(CarbonApiError::UnknownZone(zone.to_string()));
            }

            let resp = self
                .http
                .get("https://niggrid.com/GenerationProfile")
                .header("User-Agent", "invisible-infrastructure/0.1")
                .send()
                .await?;

            if !resp.status().is_success() {
                return Err(CarbonApiError::ApiError {
                    status: resp.status().as_u16(),
                    message: resp.text().await.unwrap_or_default(),
                });
            }

            let html = resp
                .text()
                .await
                .map_err(|e| CarbonApiError::Parse(format!("NigGrid text error: {e}")))?;

            let fuels = Self::parse_generation_html(&html);
            let fuel_refs: Vec<(&str, f64)> =
                fuels.iter().map(|(f, mw)| (f.as_str(), *mw)).collect();
            let (wind, solar, hydro, _nuclear, fossil) = renewable_breakdown(&fuel_refs);

            Ok(RenewableEnergyResponse {
                zone: "NG".to_string(),
                renewable_percentage: wind + solar + hydro,
                wind_percentage: wind,
                solar_percentage: solar,
                hydro_percentage: hydro,
                nuclear_percentage: 0.0,
                fossil_percentage: fossil,
                datetime: Utc::now(),
            })
        }
    }

    // ── Backend 12: CSEP / CarbonTracker (India) ──────────────────────────

    pub(crate) struct CsepBackend {
        http: reqwest::Client,
    }

    impl CsepBackend {
        pub(crate) fn new() -> Self {
            Self {
                http: reqwest::Client::builder()
                    .timeout(Duration::from_secs(20))
                    .build()
                    .expect("failed to build reqwest client"),
            }
        }

        pub fn parse_generation_html(html: &str) -> Vec<(String, f64)> {
            let doc = Html::parse_document(html);
            let row_sel = Selector::parse("tr, .fuel-row, [data-source]").unwrap();
            let td_sel = Selector::parse("td, span, .value").unwrap();
            let mut results = Vec::new();

            for row in doc.select(&row_sel) {
                let cells: Vec<String> = row
                    .select(&td_sel)
                    .map(|c| c.text().collect::<String>().trim().to_string())
                    .collect();

                if cells.len() >= 2 {
                    let fuel = cells[0].clone();
                    if let Ok(mw) = cells[cells.len() - 1]
                        .replace('.', "")
                        .replace(',', ".")
                        .parse::<f64>()
                    {
                        if mw > 0.0 && !fuel.is_empty() {
                            results.push((fuel, mw));
                        }
                    }
                }
            }
            results
        }

        fn fuel_factor(fuel: &str) -> f64 {
            let fl = fuel.to_lowercase();
            if fl.contains("coal") {
                IPCC_COAL
            } else if fl.contains("gas") {
                IPCC_GAS
            } else if fl.contains("diesel") || fl.contains("oil") {
                IPCC_OIL
            } else if fl.contains("wind") {
                IPCC_WIND
            } else if fl.contains("solar") {
                IPCC_SOLAR
            } else if fl.contains("hydro") {
                IPCC_HYDRO
            } else if fl.contains("nuclear") {
                IPCC_NUCLEAR
            } else if fl.contains("biomass") || fl.contains("bio") {
                IPCC_BIOMASS
            } else {
                400.0
            }
        }
    }

    #[async_trait]
    impl CarbonBackend for CsepBackend {
        fn name(&self) -> &str {
            "csep"
        }

        fn supports_zone(&self, zone: &str) -> bool {
            zone == "IN" || zone.starts_with("IN-")
        }

        async fn carbon_intensity(
            &self,
            zone: &str,
        ) -> Result<CarbonIntensityResponse, CarbonApiError> {
            if !self.supports_zone(zone) {
                return Err(CarbonApiError::UnknownZone(zone.to_string()));
            }

            let resp = self
                .http
                .get("https://www.carbontracker.in/")
                .header("User-Agent", "invisible-infrastructure/0.1")
                .send()
                .await?;

            if !resp.status().is_success() {
                return Err(CarbonApiError::ApiError {
                    status: resp.status().as_u16(),
                    message: resp.text().await.unwrap_or_default(),
                });
            }

            let html = resp
                .text()
                .await
                .map_err(|e| CarbonApiError::Parse(format!("CSEP text error: {e}")))?;

            let fuels = Self::parse_generation_html(&html);
            if fuels.is_empty() {
                return Err(CarbonApiError::Parse("empty CSEP generation data".into()));
            }

            let weighted: Vec<(f64, f64)> = fuels
                .iter()
                .map(|(f, mw)| (*mw, Self::fuel_factor(f)))
                .collect();
            let (intensity, fossil_pct) = weighted_intensity(&weighted);

            Ok(CarbonIntensityResponse {
                zone: zone.to_string(),
                carbon_intensity_gco2_kwh: intensity,
                fossil_fuel_percentage: fossil_pct,
                datetime: Utc::now(),
                data_source: "csep".to_string(),
                is_estimated: false,
            })
        }

        async fn power_breakdown(
            &self,
            zone: &str,
        ) -> Result<RenewableEnergyResponse, CarbonApiError> {
            if !self.supports_zone(zone) {
                return Err(CarbonApiError::UnknownZone(zone.to_string()));
            }

            let resp = self
                .http
                .get("https://www.carbontracker.in/")
                .header("User-Agent", "invisible-infrastructure/0.1")
                .send()
                .await?;

            if !resp.status().is_success() {
                return Err(CarbonApiError::ApiError {
                    status: resp.status().as_u16(),
                    message: resp.text().await.unwrap_or_default(),
                });
            }

            let html = resp
                .text()
                .await
                .map_err(|e| CarbonApiError::Parse(format!("CSEP text error: {e}")))?;

            let fuels = Self::parse_generation_html(&html);
            let fuel_refs: Vec<(&str, f64)> =
                fuels.iter().map(|(f, mw)| (f.as_str(), *mw)).collect();
            let (wind, solar, hydro, nuclear, fossil) = renewable_breakdown(&fuel_refs);

            Ok(RenewableEnergyResponse {
                zone: zone.to_string(),
                renewable_percentage: wind + solar + hydro,
                wind_percentage: wind,
                solar_percentage: solar,
                hydro_percentage: hydro,
                nuclear_percentage: nuclear,
                fossil_percentage: fossil,
                datetime: Utc::now(),
            })
        }
    }
}

#[cfg(feature = "scraper-backends")]
pub(crate) use scraper_backends::{CammesaBackend, CsepBackend, NiggridBackend, OnsBackend};

// ===========================================================================
// WattTime Backend (US marginal operating emissions rate)
// ===========================================================================

/// WattTime v3 API client — provides marginal operating emissions rate (MOER)
/// for US grid balancing authorities. MOER tells you the emissions impact of
/// consuming one additional unit of electricity *right now*, which is more
/// useful for scheduling decisions than average intensity.
///
/// Auth: username/password → `GET /login` (Basic Auth) → bearer token (30 min).
/// Data: `GET /v3/forecast?region=X&signal_type=co2_moer` — first data point
///       is current 5-minute interval's MOER in lbs CO2/MWh.
/// Fuel: `GET /v3/fuel-mix?region=X` — generation by fuel type.
///
/// Reference: <https://docs.watttime.org/>
struct WattTimeBackend {
    username: String,
    password: String,
    http: reqwest::Client,
    token: Mutex<Option<(String, Instant)>>,
}

impl WattTimeBackend {
    fn new(username: String, password: String) -> Self {
        Self {
            username,
            password,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("failed to build reqwest client"),
            token: Mutex::new(None),
        }
    }

    /// Get a valid bearer token, refreshing if needed (tokens valid 30 min).
    async fn get_token(&self) -> Result<String, CarbonApiError> {
        // Check cached token
        {
            let guard = self.token.lock().unwrap();
            if let Some((ref tok, issued)) = *guard
                && issued.elapsed() < Duration::from_secs(25 * 60)
            {
                return Ok(tok.clone());
            }
        }

        // GET /login with Basic Auth → { "token": "..." }
        let resp = self
            .http
            .get("https://api.watttime.org/login")
            .basic_auth(&self.username, Some(&self.password))
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: format!(
                    "WattTime login failed: {}",
                    resp.text().await.unwrap_or_default()
                ),
            });
        }

        let body: WattTimeLoginResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("WattTime login parse: {e}")))?;

        let tok = body.token;
        {
            let mut guard = self.token.lock().unwrap();
            *guard = Some((tok.clone(), Instant::now()));
        }
        Ok(tok)
    }
}

#[derive(Debug, Deserialize)]
struct WattTimeLoginResponse {
    token: String,
}

/// Response from `/v3/forecast` — array of forecast data points.
#[derive(Debug, Deserialize)]
struct WattTimeForecastResponse {
    data: Vec<WattTimeForecastPoint>,
}

#[derive(Debug, Deserialize)]
struct WattTimeForecastPoint {
    /// MOER value in lbs CO2/MWh (for co2_moer signal).
    #[serde(default)]
    value: f64,
}

/// Response from `/v3/fuel-mix`.
#[derive(Debug, Deserialize)]
struct WattTimeFuelMixResponse {
    data: Vec<WattTimeFuelMixPoint>,
}

#[derive(Debug, Deserialize)]
struct WattTimeFuelMixPoint {
    #[serde(default)]
    fuel_category: Option<String>,
    /// Generation in MW.
    #[serde(default)]
    value: f64,
}

/// Map ElectricityMaps zone code to WattTime region (balancing authority).
fn zone_to_watttime_region(zone: &str) -> Option<&'static str> {
    match zone {
        "US-MIDA-PJM" => Some("PJM"),
        "US-CAL-CISO" => Some("CAISO_NORTH"),
        "US-MIDW-MISO" => Some("MISO"),
        "US-TEX-ERCO" => Some("ERCOT"),
        "US-NY-NYIS" => Some("NYISO"),
        "US-NW-BPAT" => Some("BPAT"),
        "US-SE-SOCO" => Some("SOCO"),
        "US-SW-AZPS" => Some("AZPS"),
        "US-NE-ISNE" => Some("ISNE"),
        "US-SW-SRP" => Some("SRP"),
        "US-NW-PACW" => Some("PACW"),
        "US-SW-EPE" => Some("EPE"),
        "US-FLA-FPL" => Some("FPL"),
        "US-SE-AEC" => Some("AEC"),
        "US-TEN-TVA" => Some("TVA"),
        _ if zone.starts_with("US-") => Some("CAISO_NORTH"), // fallback
        _ => None,
    }
}

/// Convert MOER value (lbs CO2/MWh) to gCO2eq/kWh.
/// 1 lb = 453.592 g, 1 MWh = 1000 kWh → factor = 0.453592.
fn moer_to_gco2_kwh(moer_lbs_per_mwh: f64) -> f64 {
    moer_lbs_per_mwh * 0.453592
}

#[async_trait]
impl CarbonBackend for WattTimeBackend {
    fn name(&self) -> &str {
        "watttime"
    }

    fn supports_zone(&self, zone: &str) -> bool {
        zone_to_watttime_region(zone).is_some()
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        let region = zone_to_watttime_region(zone)
            .ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        let token = self.get_token().await?;

        // GET /v3/forecast — first data point is current 5-minute MOER.
        let resp = self
            .http
            .get("https://api.watttime.org/v3/forecast")
            .bearer_auth(&token)
            .query(&[("region", region), ("signal_type", "co2_moer")])
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CarbonApiError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let forecast: WattTimeForecastResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("WattTime forecast parse: {e}")))?;

        // First data point = current interval's marginal emissions rate.
        let moer_lbs = forecast.data.first().map(|p| p.value).unwrap_or(800.0); // pessimistic fallback

        let intensity = moer_to_gco2_kwh(moer_lbs);

        // Estimate fossil % from MOER: US grid range is roughly
        // 100-2000 lbs/MWh. Map linearly to 10-90% fossil.
        let fossil_pct = ((moer_lbs - 100.0) / 1900.0 * 80.0 + 10.0).clamp(5.0, 95.0);

        Ok(CarbonIntensityResponse {
            zone: zone.to_string(),
            carbon_intensity_gco2_kwh: intensity,
            fossil_fuel_percentage: fossil_pct,
            datetime: Utc::now(),
            data_source: "watttime-moer".to_string(),
            is_estimated: false,
        })
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        let region = zone_to_watttime_region(zone)
            .ok_or_else(|| CarbonApiError::UnknownZone(zone.to_string()))?;

        let token = self.get_token().await?;

        // Try fuel-mix endpoint first.
        let resp = self
            .http
            .get("https://api.watttime.org/v3/fuel-mix")
            .bearer_auth(&token)
            .query(&[("region", region)])
            .send()
            .await?;

        if !resp.status().is_success() {
            // Fuel-mix may not be available on free tier. Fall through to
            // estimate from MOER.
            let ci = self.carbon_intensity(zone).await?;
            let fossil = ci.fossil_fuel_percentage;
            let renewable = 100.0 - fossil;
            return Ok(RenewableEnergyResponse {
                zone: zone.to_string(),
                renewable_percentage: renewable,
                wind_percentage: renewable * 0.4,
                solar_percentage: renewable * 0.25,
                hydro_percentage: renewable * 0.25,
                nuclear_percentage: renewable * 0.1,
                fossil_percentage: fossil,
                datetime: Utc::now(),
            });
        }

        let fuel_mix: WattTimeFuelMixResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(format!("WattTime fuel-mix parse: {e}")))?;

        let mut total = 0.0f64;
        let mut wind = 0.0f64;
        let mut solar = 0.0f64;
        let mut hydro = 0.0f64;
        let mut nuclear = 0.0f64;
        let mut fossil = 0.0f64;

        for point in &fuel_mix.data {
            let mw = point.value.max(0.0);
            total += mw;
            let cat = point.fuel_category.as_deref().unwrap_or("").to_lowercase();
            if cat.contains("wind") {
                wind += mw;
            } else if cat.contains("solar") {
                solar += mw;
            } else if cat.contains("hydro") || cat.contains("water") {
                hydro += mw;
            } else if cat.contains("nuclear") {
                nuclear += mw;
            } else if cat.contains("coal")
                || cat.contains("gas")
                || cat.contains("oil")
                || cat.contains("petroleum")
            {
                fossil += mw;
            }
        }

        let pct = |v: f64| {
            if total > 0.0 {
                (v / total) * 100.0
            } else {
                0.0
            }
        };

        Ok(RenewableEnergyResponse {
            zone: zone.to_string(),
            renewable_percentage: pct(wind + solar + hydro),
            wind_percentage: pct(wind),
            solar_percentage: pct(solar),
            hydro_percentage: pct(hydro),
            nuclear_percentage: pct(nuclear),
            fossil_percentage: pct(fossil),
            datetime: Utc::now(),
        })
    }
}

// ===========================================================================
// Backend 14: IPCC Static Fallback (built-in, no network)
// ===========================================================================

/// Always-available backend using IPCC AR5 emission factors and country-level
/// average grid mixes. No network access required.
pub struct IpccStaticBackend;

#[async_trait]
impl CarbonBackend for IpccStaticBackend {
    fn name(&self) -> &str {
        "ipcc-static"
    }

    fn supports_zone(&self, _zone: &str) -> bool {
        true // always available
    }

    async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        let country = zone_to_country(zone);
        let mix = country_grid_mix(country);

        Ok(CarbonIntensityResponse {
            zone: zone.to_string(),
            carbon_intensity_gco2_kwh: mix.carbon_intensity(),
            fossil_fuel_percentage: mix.fossil_pct(),
            datetime: Utc::now(),
            data_source: "ipcc-ar5-static".to_string(),
            is_estimated: true,
        })
    }

    async fn power_breakdown(&self, zone: &str) -> Result<RenewableEnergyResponse, CarbonApiError> {
        let country = zone_to_country(zone);
        let mix = country_grid_mix(country);

        Ok(RenewableEnergyResponse {
            zone: zone.to_string(),
            renewable_percentage: mix.renewable_pct(),
            wind_percentage: mix.wind,
            solar_percentage: mix.solar,
            hydro_percentage: mix.hydro,
            nuclear_percentage: mix.nuclear,
            fossil_percentage: mix.fossil_pct(),
            datetime: Utc::now(),
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_aggregator() -> CarbonAggregator {
        CarbonAggregator::new(AggregatorConfig::default())
    }

    // -- Zone mapping tests --

    #[test]
    fn zone_to_eia_ba_us_zones() {
        assert_eq!(zone_to_eia_ba("US-MIDA-PJM"), Some("PJM"));
        assert_eq!(zone_to_eia_ba("US-CAL-CISO"), Some("CISO"));
        assert_eq!(zone_to_eia_ba("US-MIDW-MISO"), Some("MISO"));
        assert_eq!(zone_to_eia_ba("US-TEX-ERCO"), Some("ERCO"));
        assert_eq!(zone_to_eia_ba("US-NY-NYIS"), Some("NYIS"));
        assert_eq!(zone_to_eia_ba("US-NW-BPAT"), Some("BPAT"));
        assert_eq!(zone_to_eia_ba("US-SE-SOCO"), Some("SOCO"));
        assert_eq!(zone_to_eia_ba("US-SW-AZPS"), Some("AZPS"));
        // Unknown US zone falls back to aggregate
        assert_eq!(zone_to_eia_ba("US-UNKNOWN"), Some("US48"));
        // Non-US returns None
        assert_eq!(zone_to_eia_ba("DE"), None);
    }

    #[test]
    fn zone_to_entsoe_area_eu_zones() {
        assert!(zone_to_entsoe_area("DE").is_some());
        assert!(zone_to_entsoe_area("FR").is_some());
        assert!(zone_to_entsoe_area("NL").is_some());
        assert!(zone_to_entsoe_area("SE").is_some());
        assert!(zone_to_entsoe_area("FI").is_some());
        assert!(zone_to_entsoe_area("PL").is_some());
        assert!(zone_to_entsoe_area("IE").is_some());
        assert!(zone_to_entsoe_area("GB").is_none()); // UK has own backend
        assert!(zone_to_entsoe_area("US-MIDA-PJM").is_none());
    }

    #[test]
    fn zone_to_country_iso3_known() {
        assert_eq!(zone_to_country_iso3("US-MIDA-PJM"), Some("USA"));
        assert_eq!(zone_to_country_iso3("DE"), Some("DEU"));
        assert_eq!(zone_to_country_iso3("GB"), Some("GBR"));
        assert_eq!(zone_to_country_iso3("JP-TK"), Some("JPN"));
        assert_eq!(zone_to_country_iso3("AU-NSW"), Some("AUS"));
        assert_eq!(zone_to_country_iso3("BR-S"), Some("BRA"));
        assert_eq!(zone_to_country_iso3("IN-WE"), Some("IND"));
        assert_eq!(zone_to_country_iso3("SG"), Some("SGP"));
    }

    #[test]
    fn zone_to_opennem_region_au() {
        assert_eq!(zone_to_opennem_region("AU-NSW"), Some("NSW1"));
        assert_eq!(zone_to_opennem_region("AU-VIC"), Some("VIC1"));
        assert_eq!(zone_to_opennem_region("AU-QLD"), Some("QLD1"));
        assert_eq!(zone_to_opennem_region("AU-SA"), Some("SA1"));
        assert_eq!(zone_to_opennem_region("AU-TAS"), Some("TAS1"));
        assert_eq!(zone_to_opennem_region("DE"), None);
    }

    // -- IPCC static backend tests --

    #[test]
    fn ipcc_static_known_countries() {
        // Verify known countries produce reasonable gCO2/kWh values.
        let countries = [
            ("US", 280.0, 500.0),
            ("DE", 200.0, 400.0),
            ("FR", 30.0, 120.0),  // nuclear-heavy → very low
            ("NO", 10.0, 40.0),   // hydro-heavy → very low
            ("IN", 500.0, 750.0), // coal-heavy → high
            ("ZA", 600.0, 750.0), // very coal-heavy → very high
            ("PL", 400.0, 600.0), // coal-heavy
            ("BR", 50.0, 150.0),  // hydro-heavy
            ("SG", 450.0, 520.0), // gas-heavy
        ];

        for (country, min, max) in countries {
            let mix = country_grid_mix(country);
            let ci = mix.carbon_intensity();
            assert!(
                (min..=max).contains(&ci),
                "{country}: {ci:.1} gCO2/kWh not in [{min}, {max}]"
            );
        }
    }

    #[test]
    fn ipcc_static_renewable_fossil_percentages() {
        let no = country_grid_mix("NO");
        assert!(no.renewable_pct() > 90.0, "Norway should be >90% renewable");
        assert!(no.fossil_pct() < 5.0, "Norway should be <5% fossil");

        let za = country_grid_mix("ZA");
        assert!(za.fossil_pct() > 80.0, "South Africa should be >80% fossil");
    }

    #[tokio::test]
    async fn ipcc_static_backend_always_returns() {
        let backend = IpccStaticBackend;
        assert!(backend.supports_zone("anything"));

        let ci = backend.carbon_intensity("US-MIDA-PJM").await.unwrap();
        assert!(ci.carbon_intensity_gco2_kwh > 0.0);
        assert_eq!(ci.data_source, "ipcc-ar5-static");
        assert!(ci.is_estimated);

        let pb = backend.power_breakdown("FR").await.unwrap();
        assert!(
            pb.nuclear_percentage > 50.0,
            "France should be nuclear-heavy"
        );
    }

    // -- Aggregator tests --

    #[tokio::test]
    async fn aggregator_falls_through_to_ipcc() {
        // With no API keys, only UkCarbonBackend (for GB) and IPCC static are available.
        let agg = default_aggregator();

        // US zone — no EIA key, so falls to IPCC.
        let ci = agg.carbon_intensity("US-MIDA-PJM").await.unwrap();
        assert_eq!(ci.data_source, "ipcc-ar5-static");
        assert!(ci.carbon_intensity_gco2_kwh > 0.0);

        // France — no ENTSO-E key, so falls to IPCC.
        let ci = agg.carbon_intensity("FR").await.unwrap();
        assert_eq!(ci.data_source, "ipcc-ar5-static");
    }

    #[tokio::test]
    async fn aggregator_caches_responses() {
        let agg = default_aggregator();

        let ci1 = agg.carbon_intensity("DE").await.unwrap();
        let ci2 = agg.carbon_intensity("DE").await.unwrap();

        // Same cached value (IPCC static is deterministic).
        assert_eq!(ci1.carbon_intensity_gco2_kwh, ci2.carbon_intensity_gco2_kwh);
    }

    #[tokio::test]
    async fn aggregator_config_no_keys_works() {
        let agg = CarbonAggregator::new(AggregatorConfig {
            cache_ttl_secs: 60,
            ..AggregatorConfig::default()
        });

        // Should still work via IPCC static.
        let ci = agg.carbon_intensity("JP-TK").await.unwrap();
        assert!(ci.carbon_intensity_gco2_kwh > 300.0); // Japan ~440
        assert_eq!(ci.data_source, "ipcc-ar5-static");
    }

    #[tokio::test]
    async fn carbon_intensity_for_cloud_region_end_to_end() {
        let agg = default_aggregator();

        let ci = agg
            .carbon_intensity_for_cloud_region("us-east-1")
            .await
            .unwrap();
        assert_eq!(ci.zone, "US-MIDA-PJM");
        assert!(ci.carbon_intensity_gco2_kwh > 0.0);

        let ci = agg
            .carbon_intensity_for_cloud_region("eu-north-1")
            .await
            .unwrap();
        assert_eq!(ci.zone, "SE");
        // Sweden is low-carbon (hydro + nuclear + wind).
        assert!(ci.carbon_intensity_gco2_kwh < 50.0);
    }

    #[tokio::test]
    async fn to_carbon_intensity_conversion() {
        let agg = default_aggregator();
        let ci = agg.to_carbon_intensity("NO").await.unwrap();
        assert_eq!(ci.region, "NO");
        assert!(ci.gco2_per_kwh < 40.0); // Norway is very clean
        assert_eq!(ci.source, "ipcc-ar5-static");
    }

    // -- Orbital carbon intensity --

    #[tokio::test]
    async fn orbital_zone_zero_carbon() {
        let agg = default_aggregator();
        let ci = agg.carbon_intensity("ORBITAL").await.unwrap();
        assert_eq!(ci.zone, "ORBITAL");
        assert!((ci.carbon_intensity_gco2_kwh - 0.0).abs() < f64::EPSILON);
        assert_eq!(ci.data_source, "orbital-solar");
        assert!((ci.fossil_fuel_percentage - 0.0).abs() < f64::EPSILON);
        assert!(!ci.is_estimated);
    }

    #[tokio::test]
    async fn orbital_zone_power_breakdown() {
        let agg = default_aggregator();
        let pb = agg.power_breakdown("ORBITAL").await.unwrap();
        assert_eq!(pb.zone, "ORBITAL");
        assert!((pb.renewable_percentage - 100.0).abs() < f64::EPSILON);
        assert!((pb.solar_percentage - 100.0).abs() < f64::EPSILON);
        assert!((pb.fossil_percentage - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn orbital_cloud_region_end_to_end() {
        let agg = default_aggregator();
        let ci = agg
            .carbon_intensity_for_cloud_region("orbital-leo")
            .await
            .unwrap();
        assert_eq!(ci.zone, "ORBITAL");
        assert!((ci.carbon_intensity_gco2_kwh - 0.0).abs() < f64::EPSILON);
    }

    // -- EIA fuel mapping tests --

    #[test]
    fn eia_fuel_emission_factors() {
        assert_eq!(EiaBackend::fuel_emission_factor("COL"), IPCC_COAL);
        assert_eq!(EiaBackend::fuel_emission_factor("NG"), IPCC_GAS);
        assert_eq!(EiaBackend::fuel_emission_factor("NUC"), IPCC_NUCLEAR);
        assert_eq!(EiaBackend::fuel_emission_factor("SUN"), IPCC_SOLAR);
        assert_eq!(EiaBackend::fuel_emission_factor("WND"), IPCC_WIND);
        assert_eq!(EiaBackend::fuel_emission_factor("WAT"), IPCC_HYDRO);
    }

    // -- ENTSO-E production type mapping tests --

    #[test]
    fn entsoe_psr_emission_factors() {
        assert_eq!(EntsoeBackend::psr_emission_factor("B02"), IPCC_COAL); // brown coal
        assert_eq!(EntsoeBackend::psr_emission_factor("B05"), IPCC_COAL); // hard coal
        assert_eq!(EntsoeBackend::psr_emission_factor("B04"), IPCC_GAS); // fossil gas
        assert_eq!(EntsoeBackend::psr_emission_factor("B14"), IPCC_NUCLEAR); // nuclear
        assert_eq!(EntsoeBackend::psr_emission_factor("B16"), IPCC_SOLAR); // solar
        assert_eq!(EntsoeBackend::psr_emission_factor("B19"), IPCC_WIND); // wind onshore
    }

    #[test]
    fn entsoe_xml_parsing() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument>
  <TimeSeries>
    <MktPSRType><psrType>B04</psrType></MktPSRType>
    <Period>
      <Point><position>1</position><quantity>5000</quantity></Point>
    </Period>
  </TimeSeries>
  <TimeSeries>
    <MktPSRType><psrType>B16</psrType></MktPSRType>
    <Period>
      <Point><position>1</position><quantity>3000</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"#;

        let data = EntsoeBackend::parse_generation_xml(xml);
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], ("B04".to_string(), 5000.0));
        assert_eq!(data[1], ("B16".to_string(), 3000.0));
    }

    // -- Grid mix math tests --

    #[test]
    fn grid_mix_carbon_intensity_math() {
        // 100% coal = 820 gCO2/kWh
        let all_coal = GridMix {
            coal: 100.0,
            gas: 0.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 0.0,
            solar: 0.0,
            biomass: 0.0,
            geothermal: 0.0,
        };
        assert!((all_coal.carbon_intensity() - 820.0).abs() < 0.1);

        // 100% nuclear = 12 gCO2/kWh
        let all_nuclear = GridMix {
            coal: 0.0,
            gas: 0.0,
            oil: 0.0,
            nuclear: 100.0,
            hydro: 0.0,
            wind: 0.0,
            solar: 0.0,
            biomass: 0.0,
            geothermal: 0.0,
        };
        assert!((all_nuclear.carbon_intensity() - 12.0).abs() < 0.1);

        // 50% coal, 50% wind = (820 + 11) / 2 = 415.5
        let mixed = GridMix {
            coal: 50.0,
            gas: 0.0,
            oil: 0.0,
            nuclear: 0.0,
            hydro: 0.0,
            wind: 50.0,
            solar: 0.0,
            biomass: 0.0,
            geothermal: 0.0,
        };
        assert!((mixed.carbon_intensity() - 415.5).abs() < 0.1);
    }

    #[test]
    fn uk_backend_only_supports_gb() {
        let backend = UkCarbonBackend::new();
        assert!(backend.supports_zone("GB"));
        assert!(!backend.supports_zone("DE"));
        assert!(!backend.supports_zone("US-MIDA-PJM"));
    }

    // -- Phase 1F: Tier 3 expansion tests --

    #[test]
    fn ipcc_static_new_countries() {
        let countries = [
            // Africa
            ("NG", 400.0, 550.0), // gas-heavy
            ("KE", 20.0, 100.0),  // geothermal + hydro → very clean
            ("MA", 300.0, 550.0), // coal + gas mix
            ("EG", 400.0, 600.0), // gas + oil
            ("ET", 10.0, 50.0),   // hydro + wind → very clean
            // South America
            ("CL", 200.0, 380.0), // diverse mix
            ("AR", 250.0, 400.0), // gas-heavy
            ("UY", 10.0, 100.0),  // wind + hydro → very clean
            ("PY", 10.0, 30.0),   // 100% hydro → ultra clean
            ("CO", 80.0, 250.0),  // hydro-heavy
            // Middle East
            ("SA", 400.0, 600.0), // gas + oil
            ("AE", 350.0, 500.0), // gas + nuclear
            ("TR", 300.0, 500.0), // coal + gas + hydro
            // Southeast Asia
            ("ID", 500.0, 700.0), // coal-heavy
            ("VN", 350.0, 550.0), // coal + hydro
            ("TH", 300.0, 470.0), // gas + coal
            ("PK", 250.0, 450.0), // gas + hydro + coal
            ("MY", 350.0, 550.0), // gas + coal
        ];

        for (country, min, max) in countries {
            let mix = country_grid_mix(country);
            let ci = mix.carbon_intensity();
            assert!(
                (min..=max).contains(&ci),
                "{country}: {ci:.1} gCO2/kWh not in [{min}, {max}]"
            );
        }
    }

    #[test]
    fn zone_to_country_iso3_new_zones() {
        let mappings = [
            ("NG", "NGA"),
            ("EG", "EGY"),
            ("KE", "KEN"),
            ("MA", "MAR"),
            ("ET", "ETH"),
            ("GH", "GHA"),
            ("TZ", "TZA"),
            ("DZ", "DZA"),
            ("TN", "TUN"),
            ("SN", "SEN"),
            ("CL", "CHL"),
            ("CO", "COL"),
            ("AR", "ARG"),
            ("PE", "PER"),
            ("MX", "MEX"),
            ("EC", "ECU"),
            ("UY", "URY"),
            ("PY", "PRY"),
            ("BO", "BOL"),
            ("VE", "VEN"),
            ("SA", "SAU"),
            ("AE", "ARE"),
            ("TR", "TUR"),
            ("IQ", "IRQ"),
            ("IR", "IRN"),
            ("KW", "KWT"),
            ("OM", "OMN"),
            ("QA", "QAT"),
            ("JO", "JOR"),
            ("ID", "IDN"),
            ("VN", "VNM"),
            ("TH", "THA"),
            ("PH", "PHL"),
            ("PK", "PAK"),
            ("BD", "BGD"),
            ("MM", "MMR"),
            ("MY", "MYS"),
        ];

        for (zone, expected_iso3) in mappings {
            assert_eq!(
                zone_to_country_iso3(zone),
                Some(expected_iso3),
                "zone_to_country_iso3({zone}) should be {expected_iso3}"
            );
        }
    }

    #[test]
    fn zone_to_country_prefix_handlers() {
        assert_eq!(zone_to_country("MX-NORTE"), "MX");
        assert_eq!(zone_to_country("CL-SIC"), "CL");
        assert_eq!(zone_to_country("AR-BAS"), "AR");
        assert_eq!(zone_to_country("PK-NTDC"), "PK");
        assert_eq!(zone_to_country("ID-JV"), "ID");
        assert_eq!(zone_to_country("NZ-NI"), "NZ");
        // Existing prefixes still work
        assert_eq!(zone_to_country("US-MIDA-PJM"), "US");
        assert_eq!(zone_to_country("IN-WE"), "IN");
        assert_eq!(zone_to_country("BR-S"), "BR");
        // Single-zone countries pass through
        assert_eq!(zone_to_country("NG"), "NG");
        assert_eq!(zone_to_country("TR"), "TR");
    }

    // -- Phase 2: EPIAS + CEN backend tests --

    #[test]
    fn epias_fuel_emission_factors() {
        assert_eq!(EpiasBackend::fuel_emission_factor("LIGNITE"), IPCC_COAL);
        assert_eq!(EpiasBackend::fuel_emission_factor("HARD_COAL"), IPCC_COAL);
        assert_eq!(
            EpiasBackend::fuel_emission_factor("IMPORTED_COAL"),
            IPCC_COAL
        );
        assert_eq!(EpiasBackend::fuel_emission_factor("NATURAL_GAS"), IPCC_GAS);
        assert_eq!(EpiasBackend::fuel_emission_factor("LNG"), IPCC_GAS);
        assert_eq!(EpiasBackend::fuel_emission_factor("WIND"), IPCC_WIND);
        assert_eq!(EpiasBackend::fuel_emission_factor("SUN"), IPCC_SOLAR);
        assert_eq!(EpiasBackend::fuel_emission_factor("HYDRAULIC"), IPCC_HYDRO);
        assert_eq!(
            EpiasBackend::fuel_emission_factor("GEOTHERMAL"),
            IPCC_GEOTHERMAL
        );
        assert_eq!(EpiasBackend::fuel_emission_factor("NUCLEAR"), IPCC_NUCLEAR);
    }

    #[test]
    fn epias_supports_zone_turkey_only() {
        let backend = EpiasBackend::new("dummy".to_string());
        assert!(backend.supports_zone("TR"));
        assert!(!backend.supports_zone("DE"));
        assert!(!backend.supports_zone("US-MIDA-PJM"));
    }

    #[test]
    fn cen_fuel_emission_factors() {
        assert_eq!(CenBackend::fuel_emission_factor("Carbon"), IPCC_COAL);
        assert_eq!(CenBackend::fuel_emission_factor("Gas Natural"), IPCC_GAS);
        assert_eq!(CenBackend::fuel_emission_factor("GNL"), IPCC_GAS);
        assert_eq!(CenBackend::fuel_emission_factor("Diesel"), IPCC_OIL);
        assert_eq!(CenBackend::fuel_emission_factor("Eolica"), IPCC_WIND);
        assert_eq!(
            CenBackend::fuel_emission_factor("Solar Fotovoltaica"),
            IPCC_SOLAR
        );
        assert_eq!(CenBackend::fuel_emission_factor("Hidraulica"), IPCC_HYDRO);
        assert_eq!(
            CenBackend::fuel_emission_factor("Geotermica"),
            IPCC_GEOTHERMAL
        );
        assert_eq!(CenBackend::fuel_emission_factor("Biomasa"), IPCC_BIOMASS);
    }

    #[test]
    fn cen_supports_zone_chile_only() {
        let backend = CenBackend::new();
        assert!(backend.supports_zone("CL"));
        assert!(backend.supports_zone("CL-SIC"));
        assert!(!backend.supports_zone("AR"));
        assert!(!backend.supports_zone("US-MIDA-PJM"));
    }

    // -- Phase 3: Scraper backend tests (feature-gated) --

    #[cfg(feature = "scraper-backends")]
    mod scraper_tests {
        use super::*;

        const ONS_HTML: &str = r#"<html><body><table>
            <tr><td>Hidráulica</td><td>45000</td></tr>
            <tr><td>Térmica</td><td>20000</td></tr>
            <tr><td>Eólica</td><td>15000</td></tr>
            <tr><td>Solar</td><td>8000</td></tr>
            <tr><td>Nuclear</td><td>2000</td></tr>
        </table></body></html>"#;

        const CAMMESA_HTML: &str = r#"<html><body><table>
            <tr><td>Térmica</td><td>22000</td></tr>
            <tr><td>Hidráulica</td><td>18000</td></tr>
            <tr><td>Nuclear</td><td>3000</td></tr>
            <tr><td>Eólica</td><td>5000</td></tr>
            <tr><td>Solar</td><td>2000</td></tr>
        </table></body></html>"#;

        const NIGGRID_HTML: &str = r#"<html><body><table>
            <tr><td>Gas</td><td>3500</td></tr>
            <tr><td>Hydro</td><td>800</td></tr>
            <tr><td>Solar</td><td>100</td></tr>
        </table></body></html>"#;

        const CSEP_HTML: &str = r#"<html><body><table>
            <tr><td>Coal</td><td>120000</td></tr>
            <tr><td>Gas</td><td>25000</td></tr>
            <tr><td>Hydro</td><td>45000</td></tr>
            <tr><td>Wind</td><td>30000</td></tr>
            <tr><td>Solar</td><td>40000</td></tr>
            <tr><td>Nuclear</td><td>8000</td></tr>
            <tr><td>Biomass</td><td>10000</td></tr>
        </table></body></html>"#;

        #[test]
        fn ons_parse_generation() {
            let fuels = OnsBackend::parse_generation_html(ONS_HTML);
            assert!(!fuels.is_empty(), "ONS should parse fuel rows");
            let total: f64 = fuels.iter().map(|(_, mw)| mw).sum();
            assert!(total > 0.0, "total generation should be positive");
        }

        #[test]
        fn ons_supports_zone_brazil() {
            let backend = OnsBackend::new();
            assert!(backend.supports_zone("BR"));
            assert!(backend.supports_zone("BR-S"));
            assert!(!backend.supports_zone("AR"));
        }

        #[test]
        fn cammesa_parse_generation() {
            let fuels = CammesaBackend::parse_generation_html(CAMMESA_HTML);
            assert!(!fuels.is_empty(), "CAMMESA should parse fuel rows");
            let total: f64 = fuels.iter().map(|(_, mw)| mw).sum();
            assert!(total > 0.0);
        }

        #[test]
        fn cammesa_supports_zone_argentina() {
            let backend = CammesaBackend::new();
            assert!(backend.supports_zone("AR"));
            assert!(backend.supports_zone("AR-BAS"));
            assert!(!backend.supports_zone("CL"));
        }

        #[test]
        fn niggrid_parse_generation() {
            let fuels = NiggridBackend::parse_generation_html(NIGGRID_HTML);
            assert!(!fuels.is_empty(), "NigGrid should parse fuel rows");
            let total: f64 = fuels.iter().map(|(_, mw)| mw).sum();
            assert!(total > 0.0);
        }

        #[test]
        fn niggrid_supports_zone_nigeria() {
            let backend = NiggridBackend::new();
            assert!(backend.supports_zone("NG"));
            assert!(!backend.supports_zone("GH"));
        }

        #[test]
        fn csep_parse_generation() {
            let fuels = CsepBackend::parse_generation_html(CSEP_HTML);
            assert!(!fuels.is_empty(), "CSEP should parse fuel rows");
            let total: f64 = fuels.iter().map(|(_, mw)| mw).sum();
            assert!(total > 0.0);
            // India fixture has coal dominance
            let coal: f64 = fuels
                .iter()
                .filter(|(f, _)| f.to_lowercase().contains("coal"))
                .map(|(_, mw)| mw)
                .sum();
            assert!(
                coal / total > 0.3,
                "India should have significant coal share"
            );
        }

        #[test]
        fn csep_supports_zone_india() {
            let backend = CsepBackend::new();
            assert!(backend.supports_zone("IN"));
            assert!(backend.supports_zone("IN-WE"));
            assert!(backend.supports_zone("IN-SO"));
            assert!(!backend.supports_zone("PK"));
        }
    } // end scraper_tests

    // ── Live integration tests (run with: cargo test -p inv-energy -- --ignored) ──

    #[tokio::test]
    #[ignore = "requires ENTSOE_TOKEN env var and network access"]
    async fn entsoe_live_carbon_intensity_germany() {
        let token = std::env::var("ENTSOE_TOKEN").expect("ENTSOE_TOKEN must be set");
        let endpoint = std::env::var("ENTSOE_ENDPOINT_URL").ok();
        let backend = EntsoeBackend::new(token, endpoint);

        let result = backend.carbon_intensity("DE").await;
        match &result {
            Ok(resp) => {
                println!("=== ENTSO-E Live: Germany ===");
                println!("  Zone:             {}", resp.zone);
                println!(
                    "  Carbon intensity: {:.1} gCO2/kWh",
                    resp.carbon_intensity_gco2_kwh
                );
                println!("  Fossil fuel:      {:.1}%", resp.fossil_fuel_percentage);
                println!("  Data source:      {}", resp.data_source);
                println!("  Timestamp:        {}", resp.datetime);
                assert!(resp.carbon_intensity_gco2_kwh > 0.0);
                assert!(resp.carbon_intensity_gco2_kwh < 1000.0);
            }
            Err(e) => {
                panic!("ENTSO-E carbon_intensity failed: {e:?}");
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires ENTSOE_TOKEN env var and network access"]
    async fn entsoe_live_power_breakdown_france() {
        let token = std::env::var("ENTSOE_TOKEN").expect("ENTSOE_TOKEN must be set");
        let endpoint = std::env::var("ENTSOE_ENDPOINT_URL").ok();
        let backend = EntsoeBackend::new(token, endpoint);

        let result = backend.power_breakdown("FR").await;
        match &result {
            Ok(resp) => {
                println!("=== ENTSO-E Live: France ===");
                println!("  Zone:     {}", resp.zone);
                println!("  Renewable: {:.1}%", resp.renewable_percentage);
                println!("  Wind:      {:.1}%", resp.wind_percentage);
                println!("  Solar:     {:.1}%", resp.solar_percentage);
                println!("  Hydro:     {:.1}%", resp.hydro_percentage);
                println!("  Nuclear:   {:.1}%", resp.nuclear_percentage);
                println!("  Fossil:    {:.1}%", resp.fossil_percentage);
                // France is nuclear-heavy, expect nuclear > 30%
                assert!(
                    resp.nuclear_percentage > 30.0,
                    "France nuclear should be >30%, got {:.1}%",
                    resp.nuclear_percentage
                );
            }
            Err(e) => {
                panic!("ENTSO-E power_breakdown failed: {e:?}");
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires ENTSOE_TOKEN env var and network access"]
    async fn entsoe_live_aggregator_eu_zones() {
        let token = std::env::var("ENTSOE_TOKEN").expect("ENTSOE_TOKEN must be set");
        let endpoint = std::env::var("ENTSOE_ENDPOINT_URL").ok();
        let agg = CarbonAggregator::new(AggregatorConfig {
            entsoe_token: Some(token),
            entsoe_endpoint: endpoint,
            ..AggregatorConfig::default()
        });

        println!("=== ENTSO-E Live: Multi-Zone Carbon Intensity ===");
        for zone in ["DE", "FR", "NL", "ES", "SE", "FI", "PL"] {
            match agg.carbon_intensity(zone).await {
                Ok(resp) => {
                    println!(
                        "  {}: {:.1} gCO2/kWh (fossil {:.1}%)",
                        zone, resp.carbon_intensity_gco2_kwh, resp.fossil_fuel_percentage
                    );
                }
                Err(e) => {
                    println!("  {}: ERROR — {:?}", zone, e);
                }
            }
        }
    }

    // -- WattTime zone mapping tests --

    #[test]
    fn watttime_zone_mapping_us_zones() {
        assert_eq!(zone_to_watttime_region("US-MIDA-PJM"), Some("PJM"));
        assert_eq!(zone_to_watttime_region("US-CAL-CISO"), Some("CAISO_NORTH"));
        assert_eq!(zone_to_watttime_region("US-MIDW-MISO"), Some("MISO"));
        assert_eq!(zone_to_watttime_region("US-TEX-ERCO"), Some("ERCOT"));
        assert_eq!(zone_to_watttime_region("US-NY-NYIS"), Some("NYISO"));
        assert_eq!(zone_to_watttime_region("US-NW-BPAT"), Some("BPAT"));
        assert_eq!(zone_to_watttime_region("US-NE-ISNE"), Some("ISNE"));
        // Unknown US zone gets fallback
        assert!(zone_to_watttime_region("US-UNKNOWN").is_some());
        // Non-US returns None
        assert_eq!(zone_to_watttime_region("DE"), None);
        assert_eq!(zone_to_watttime_region("GB"), None);
        assert_eq!(zone_to_watttime_region("AU-NSW"), None);
    }

    #[test]
    fn watttime_backend_supports_us_only() {
        let backend = WattTimeBackend::new("test".into(), "test".into());
        assert!(backend.supports_zone("US-MIDA-PJM"));
        assert!(backend.supports_zone("US-CAL-CISO"));
        assert!(backend.supports_zone("US-TEX-ERCO"));
        assert!(!backend.supports_zone("DE"));
        assert!(!backend.supports_zone("GB"));
        assert!(!backend.supports_zone("AU-NSW"));
    }

    #[test]
    fn aggregator_config_watttime_fields() {
        let config = AggregatorConfig {
            watttime_username: Some("testuser".into()),
            watttime_password: Some("testpass".into()),
            ..AggregatorConfig::default()
        };
        let backends = build_backends(&config);
        let names: Vec<&str> = backends.iter().map(|b| b.name()).collect();
        assert!(
            names.contains(&"watttime"),
            "WattTime backend should be registered when credentials provided"
        );
    }

    #[test]
    fn aggregator_config_no_watttime_without_creds() {
        let config = AggregatorConfig::default();
        let backends = build_backends(&config);
        let names: Vec<&str> = backends.iter().map(|b| b.name()).collect();
        assert!(
            !names.contains(&"watttime"),
            "WattTime backend should NOT be registered without credentials"
        );
    }

    #[test]
    fn moer_to_gco2_kwh_conversion() {
        // 1000 lbs/MWh * 0.453592 = 453.592 gCO2/kWh
        let result = moer_to_gco2_kwh(1000.0);
        assert!((result - 453.592).abs() < 0.01);

        // 0 lbs/MWh = 0 gCO2/kWh
        assert_eq!(moer_to_gco2_kwh(0.0), 0.0);

        // Typical clean grid: 200 lbs/MWh ≈ 90.7 gCO2/kWh
        let clean = moer_to_gco2_kwh(200.0);
        assert!((clean - 90.72).abs() < 0.1);
    }
}
