//! Async HTTP client for the ElectricityMaps API.
//!
//! Fetches real-time carbon intensity and power breakdown data so that SCI
//! calculations can use live grid measurements instead of static fallbacks.
//!
//! The client caches responses for a configurable TTL (default 5 min) and
//! maps cloud-provider regions to ElectricityMaps zone codes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use crate::carbon::CarbonIntensity;

// ---------------------------------------------------------------------------
// ElectricityMapsConfig
// ---------------------------------------------------------------------------

/// Configuration for the ElectricityMaps API client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectricityMapsConfig {
    /// API token (e.g. "emaps-xxxxxx").
    pub api_key: String,
    /// Base URL for the API.
    pub base_url: String,
    /// HTTP request timeout in seconds.
    pub timeout_secs: u64,
    /// How long cached responses remain valid, in seconds.
    pub cache_ttl_secs: u64,
}

impl Default for ElectricityMapsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.electricitymap.org/v3".to_string(),
            timeout_secs: 10,
            cache_ttl_secs: 300,
        }
    }
}

// ---------------------------------------------------------------------------
// CarbonIntensityResponse
// ---------------------------------------------------------------------------

/// Carbon intensity data returned by the ElectricityMaps API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonIntensityResponse {
    /// ElectricityMaps zone code (e.g. "IE").
    pub zone: String,
    /// Lifecycle carbon intensity in gCO2eq/kWh.
    pub carbon_intensity_gco2_kwh: f64,
    /// Percentage of electricity generated from fossil fuels (0--100).
    pub fossil_fuel_percentage: f64,
    /// Timestamp of the measurement.
    pub datetime: DateTime<Utc>,
    /// Data source identifier.
    pub data_source: String,
    /// Whether the value is estimated rather than measured.
    pub is_estimated: bool,
}

// ---------------------------------------------------------------------------
// RenewableEnergyResponse
// ---------------------------------------------------------------------------

/// Power breakdown data returned by the ElectricityMaps API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenewableEnergyResponse {
    /// ElectricityMaps zone code.
    pub zone: String,
    /// Percentage of electricity from renewable sources (0--100).
    pub renewable_percentage: f64,
    /// Percentage from wind generation.
    pub wind_percentage: f64,
    /// Percentage from solar generation.
    pub solar_percentage: f64,
    /// Percentage from hydro generation.
    pub hydro_percentage: f64,
    /// Percentage from nuclear (low-carbon but not renewable).
    pub nuclear_percentage: f64,
    /// Percentage from fossil fuel sources.
    pub fossil_percentage: f64,
    /// Timestamp of the measurement.
    pub datetime: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// CarbonApiError
// ---------------------------------------------------------------------------

/// Errors that can occur when interacting with the ElectricityMaps API.
#[derive(Debug, thiserror::Error)]
pub enum CarbonApiError {
    /// An HTTP transport error occurred.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    /// The API returned a non-success status code.
    #[error("API returned error status {status}: {message}")]
    ApiError {
        /// HTTP status code.
        status: u16,
        /// Error message from the response body.
        message: String,
    },
    /// No API key has been configured.
    #[error("API key not configured")]
    NoApiKey,
    /// The requested zone is not recognised.
    #[error("unknown zone: {0}")]
    UnknownZone(String),
    /// Failed to parse the API response.
    #[error("parse error: {0}")]
    Parse(String),
}

// ---------------------------------------------------------------------------
// Internal serde types (raw API shapes)
// ---------------------------------------------------------------------------

/// Raw JSON shape returned by `/v3/carbon-intensity/latest`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiCarbonResponse {
    zone: String,
    carbon_intensity: f64,
    datetime: String,
    #[serde(default)]
    fossil_fuel_percentage: Option<f64>,
    #[serde(default)]
    is_estimated: Option<bool>,
}

/// Raw JSON shape returned by `/v3/power-breakdown/latest`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiPowerResponse {
    zone: String,
    datetime: String,
    #[serde(default)]
    power_consumption_breakdown: Option<PowerBreakdown>,
    #[serde(default)]
    renewable_percentage: Option<f64>,
    #[serde(default)]
    fossil_free_percentage: Option<f64>,
}

/// Nested power breakdown object inside the power-breakdown response.
#[derive(Debug, Deserialize)]
struct PowerBreakdown {
    #[serde(default)]
    wind: Option<f64>,
    #[serde(default)]
    solar: Option<f64>,
    #[serde(default)]
    hydro: Option<f64>,
    #[serde(default)]
    nuclear: Option<f64>,
    #[serde(default)]
    coal: Option<f64>,
    #[serde(default)]
    gas: Option<f64>,
    #[serde(default)]
    oil: Option<f64>,
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// A time-stamped cache entry for API responses.
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    fetched_at: Instant,
}

// ---------------------------------------------------------------------------
// ElectricityMapsClient
// ---------------------------------------------------------------------------

/// Async client for the ElectricityMaps API with in-memory caching.
pub struct ElectricityMapsClient {
    config: ElectricityMapsConfig,
    http: reqwest::Client,
    carbon_cache: Mutex<HashMap<String, CacheEntry<CarbonIntensityResponse>>>,
    power_cache: Mutex<HashMap<String, CacheEntry<RenewableEnergyResponse>>>,
}

impl std::fmt::Debug for ElectricityMapsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElectricityMapsClient")
            .field("config", &self.config)
            .finish()
    }
}

impl ElectricityMapsClient {
    /// Create a new client from the given configuration.
    pub fn new(config: ElectricityMapsConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("failed to build reqwest client");

        Self {
            config,
            http,
            carbon_cache: Mutex::new(HashMap::new()),
            power_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Check whether an API key has been configured (non-empty).
    pub fn is_configured(&self) -> bool {
        !self.config.api_key.is_empty()
    }

    /// Fetch the latest carbon intensity for an ElectricityMaps zone.
    ///
    /// Results are cached for `cache_ttl_secs`. Returns [`CarbonApiError::NoApiKey`]
    /// if no API key is configured.
    pub async fn carbon_intensity(
        &self,
        zone: &str,
    ) -> Result<CarbonIntensityResponse, CarbonApiError> {
        if !self.is_configured() {
            return Err(CarbonApiError::NoApiKey);
        }

        // Check the cache.
        if let Some(cached) = self.get_carbon_cache(zone) {
            tracing::debug!(zone, "returning cached carbon intensity");
            return Ok(cached);
        }

        let url = format!(
            "{}/carbon-intensity/latest?zone={}",
            self.config.base_url, zone
        );

        tracing::debug!(zone, url = %url, "fetching carbon intensity from ElectricityMaps");

        let resp = self
            .http
            .get(&url)
            .header("auth-token", &self.config.api_key)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let message = resp
                .text()
                .await
                .unwrap_or_else(|e| format!("(failed to read response body: {e})"));
            return Err(CarbonApiError::ApiError { status, message });
        }

        let raw: ApiCarbonResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(e.to_string()))?;

        let datetime = raw
            .datetime
            .parse::<DateTime<Utc>>()
            .map_err(|e| CarbonApiError::Parse(format!("invalid datetime: {e}")))?;

        if raw.fossil_fuel_percentage.is_none() {
            tracing::warn!(zone = %raw.zone, "ElectricityMaps: fossil_fuel_percentage missing from API response");
        }
        if raw.is_estimated.is_none() {
            tracing::warn!(zone = %raw.zone, "ElectricityMaps: is_estimated flag missing, assuming estimated");
        }

        let result = CarbonIntensityResponse {
            zone: raw.zone,
            carbon_intensity_gco2_kwh: raw.carbon_intensity,
            fossil_fuel_percentage: raw.fossil_fuel_percentage.unwrap_or(100.0),
            datetime,
            data_source: "electricitymaps.com".to_string(),
            is_estimated: raw.is_estimated.unwrap_or(true),
        };

        self.set_carbon_cache(zone, result.clone());
        Ok(result)
    }

    /// Fetch the latest power breakdown for an ElectricityMaps zone.
    ///
    /// Results are cached for `cache_ttl_secs`.
    pub async fn power_breakdown(
        &self,
        zone: &str,
    ) -> Result<RenewableEnergyResponse, CarbonApiError> {
        if !self.is_configured() {
            return Err(CarbonApiError::NoApiKey);
        }

        // Check the cache.
        if let Some(cached) = self.get_power_cache(zone) {
            tracing::debug!(zone, "returning cached power breakdown");
            return Ok(cached);
        }

        let url = format!(
            "{}/power-breakdown/latest?zone={}",
            self.config.base_url, zone
        );

        tracing::debug!(zone, url = %url, "fetching power breakdown from ElectricityMaps");

        let resp = self
            .http
            .get(&url)
            .header("auth-token", &self.config.api_key)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let message = resp
                .text()
                .await
                .unwrap_or_else(|e| format!("(failed to read response body: {e})"));
            return Err(CarbonApiError::ApiError { status, message });
        }

        let raw: ApiPowerResponse = resp
            .json()
            .await
            .map_err(|e| CarbonApiError::Parse(e.to_string()))?;

        let datetime = raw
            .datetime
            .parse::<DateTime<Utc>>()
            .map_err(|e| CarbonApiError::Parse(format!("invalid datetime: {e}")))?;

        let breakdown = match raw.power_consumption_breakdown {
            Some(b) => b,
            None => {
                return Err(CarbonApiError::Parse(format!(
                    "zone {}: power_consumption_breakdown missing from response",
                    raw.zone
                )));
            }
        };

        // Compute total generation to derive percentages.
        let wind = breakdown.wind.unwrap_or(0.0);
        let solar = breakdown.solar.unwrap_or(0.0);
        let hydro = breakdown.hydro.unwrap_or(0.0);
        let nuclear = breakdown.nuclear.unwrap_or(0.0);
        let coal = breakdown.coal.unwrap_or(0.0);
        let gas = breakdown.gas.unwrap_or(0.0);
        let oil = breakdown.oil.unwrap_or(0.0);
        let total = wind + solar + hydro + nuclear + coal + gas + oil;

        let pct = |v: f64| -> f64 {
            if total > 0.0 {
                (v / total) * 100.0
            } else {
                0.0
            }
        };

        let zone_name = raw.zone;

        let renewable_percentage = raw.renewable_percentage.unwrap_or_else(|| {
            tracing::warn!(zone = %zone_name, "ElectricityMaps: renewable_percentage missing, computing from breakdown");
            pct(wind + solar + hydro)
        });
        let fossil_percentage = raw.fossil_free_percentage.map(|ffp| 100.0 - ffp).unwrap_or_else(|| {
            tracing::warn!(zone = %zone_name, "ElectricityMaps: fossil_free_percentage missing, computing from breakdown");
            pct(coal + gas + oil)
        });

        let result = RenewableEnergyResponse {
            zone: zone_name,
            renewable_percentage,
            wind_percentage: pct(wind),
            solar_percentage: pct(solar),
            hydro_percentage: pct(hydro),
            nuclear_percentage: pct(nuclear),
            fossil_percentage,
            datetime,
        };

        self.set_power_cache(zone, result.clone());
        Ok(result)
    }

    /// Convert the live API response into the existing [`CarbonIntensity`] type
    /// used by the SCI calculation pipeline.
    pub async fn to_carbon_intensity(&self, zone: &str) -> Result<CarbonIntensity, CarbonApiError> {
        let resp = self.carbon_intensity(zone).await?;

        Ok(CarbonIntensity {
            region: resp.zone,
            gco2_per_kwh: resp.carbon_intensity_gco2_kwh,
            timestamp: resp.datetime.timestamp() as u64,
            source: resp.data_source,
        })
    }

    // -- cache helpers -------------------------------------------------------

    fn get_carbon_cache(&self, zone: &str) -> Option<CarbonIntensityResponse> {
        let cache = self.carbon_cache.lock().ok()?;
        let entry = cache.get(zone)?;
        if entry.fetched_at.elapsed().as_secs() < self.config.cache_ttl_secs {
            Some(entry.value.clone())
        } else {
            None
        }
    }

    fn set_carbon_cache(&self, zone: &str, value: CarbonIntensityResponse) {
        if let Ok(mut cache) = self.carbon_cache.lock() {
            cache.insert(
                zone.to_string(),
                CacheEntry {
                    value,
                    fetched_at: Instant::now(),
                },
            );
        }
    }

    fn get_power_cache(&self, zone: &str) -> Option<RenewableEnergyResponse> {
        let cache = self.power_cache.lock().ok()?;
        let entry = cache.get(zone)?;
        if entry.fetched_at.elapsed().as_secs() < self.config.cache_ttl_secs {
            Some(entry.value.clone())
        } else {
            None
        }
    }

    fn set_power_cache(&self, zone: &str, value: RenewableEnergyResponse) {
        if let Ok(mut cache) = self.power_cache.lock() {
            cache.insert(
                zone.to_string(),
                CacheEntry {
                    value,
                    fetched_at: Instant::now(),
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Cloud zone mapping
// ---------------------------------------------------------------------------

/// Maps cloud provider regions to ElectricityMaps zone codes.
///
/// Supports AWS, GCP, and Azure region identifiers. Returns `None` for
/// unrecognised regions.
pub fn cloud_zone_mapping(cloud_region: &str) -> Option<&'static str> {
    match cloud_region {
        // AWS US
        "us-east-1" | "us-east-2" => Some("US-MIDA-PJM"),
        "us-west-1" => Some("US-CAL-CISO"),
        "us-west-2" => Some("US-NW-BPAT"),
        // AWS EU
        "eu-west-1" => Some("IE"),
        "eu-west-2" => Some("GB"),
        "eu-west-3" => Some("FR"),
        "eu-central-1" => Some("DE"),
        "eu-north-1" => Some("SE"),
        // AWS Asia-Pacific
        "ap-northeast-1" => Some("JP-TK"),
        "ap-south-1" => Some("IN-WE"),
        "ap-southeast-1" => Some("SG"),
        "ap-southeast-2" => Some("AU-NSW"),
        // GCP
        "europe-west1" => Some("BE"),
        "europe-west4" => Some("NL"),
        "europe-north1" => Some("FI"),
        "us-central1" => Some("US-MIDW-MISO"),
        // Azure
        "westeurope" => Some("NL"),
        "northeurope" => Some("IE"),
        "westus2" => Some("US-NW-BPAT"),
        "eastus" => Some("US-MIDA-PJM"),
        // Hetzner
        "hetzner-fsn1" | "hetzner-nbg1" => Some("DE"),
        "hetzner-hel1" => Some("FI"),
        "hetzner-ash" => Some("US-MIDA-PJM"),
        "hetzner-hil" => Some("US-NW-BPAT"),
        "hetzner-sin" => Some("SG"),
        // Vultr
        "vultr-ewr" => Some("US-NY-NYIS"),
        "vultr-ord" => Some("US-MIDW-MISO"),
        "vultr-dfw" => Some("US-TEX-ERCO"),
        "vultr-atl" | "vultr-mia" => Some("US-SE-SOCO"),
        "vultr-lax" | "vultr-sjc" => Some("US-CAL-CISO"),
        "vultr-sea" => Some("US-NW-BPAT"),
        "vultr-lhr" => Some("GB"),
        "vultr-ams" => Some("NL"),
        "vultr-fra" => Some("DE"),
        "vultr-cdg" => Some("FR"),
        "vultr-sto" => Some("SE"),
        "vultr-waw" => Some("PL"),
        "vultr-mad" => Some("ES"),
        "vultr-nrt" | "vultr-itm" => Some("JP-TK"),
        "vultr-sgp" => Some("SG"),
        "vultr-bom" | "vultr-del" | "vultr-blr" => Some("IN-WE"),
        "vultr-syd" | "vultr-mel" => Some("AU-NSW"),
        "vultr-sao" => Some("BR-S"),
        "vultr-yto" => Some("CA-ON"),
        "vultr-jnb" => Some("ZA"),
        "vultr-icn" => Some("KR"),
        // Latitude.sh
        "latitude-mia" | "latitude-mia2" => Some("US-SE-SOCO"),
        "latitude-dal" => Some("US-TEX-ERCO"),
        "latitude-chi" | "latitude-chi2" => Some("US-MIDW-MISO"),
        "latitude-lax" | "latitude-lax2" => Some("US-CAL-CISO"),
        "latitude-nyc" | "latitude-nyc2" => Some("US-NY-NYIS"),
        "latitude-sao" | "latitude-sao2" => Some("BR-S"),
        "latitude-fra" => Some("DE"),
        "latitude-ams" => Some("NL"),
        "latitude-lon" => Some("GB"),
        "latitude-par" => Some("FR"),
        "latitude-mad" => Some("ES"),
        "latitude-osl" => Some("NO"),
        "latitude-syd" => Some("AU-NSW"),
        "latitude-tyo" => Some("JP-TK"),
        "latitude-sin" => Some("SG"),
        // DigitalOcean
        "do-nyc1" | "do-nyc2" | "do-nyc3" => Some("US-NY-NYIS"),
        "do-sfo2" | "do-sfo3" => Some("US-CAL-CISO"),
        "do-ams3" => Some("NL"),
        "do-sgp1" => Some("SG"),
        "do-lon1" => Some("GB"),
        "do-fra1" => Some("DE"),
        "do-tor1" => Some("CA-ON"),
        "do-blr1" => Some("IN-SO"),
        "do-syd1" => Some("AU-NSW"),
        "do-atl1" => Some("US-SE-SOCO"),
        // Fly.io
        "fly-iad" | "fly-ewr" => Some("US-MIDA-PJM"),
        "fly-ord" => Some("US-MIDW-MISO"),
        "fly-dfw" => Some("US-TEX-ERCO"),
        "fly-lax" | "fly-sjc" => Some("US-CAL-CISO"),
        "fly-sea" => Some("US-NW-BPAT"),
        "fly-atl" | "fly-mia" => Some("US-SE-SOCO"),
        "fly-yul" | "fly-yyz" => Some("CA-ON"),
        "fly-lhr" => Some("GB"),
        "fly-ams" => Some("NL"),
        "fly-fra" => Some("DE"),
        "fly-cdg" => Some("FR"),
        "fly-arn" => Some("SE"),
        "fly-waw" => Some("PL"),
        "fly-mad" => Some("ES"),
        "fly-nrt" => Some("JP-TK"),
        "fly-sin" => Some("SG"),
        "fly-syd" => Some("AU-NSW"),
        "fly-bom" => Some("IN-WE"),
        "fly-gru" => Some("BR-S"),
        "fly-jnb" => Some("ZA"),
        // Nebius
        "nebius-eu-north1" => Some("FI"),
        "nebius-eu-west1" => Some("FR"),
        "nebius-us-central1" => Some("US-MIDW-MISO"),
        "nebius-me-west1" => Some("IL"),
        // AWS (prefixed)
        "aws-us-east-1" | "aws-us-east-2" => Some("US-MIDA-PJM"),
        "aws-us-west-1" => Some("US-CAL-CISO"),
        "aws-us-west-2" => Some("US-NW-BPAT"),
        "aws-eu-west-1" => Some("IE"),
        "aws-eu-west-2" => Some("GB"),
        "aws-eu-west-3" => Some("FR"),
        "aws-eu-central-1" => Some("DE"),
        "aws-eu-north-1" => Some("SE"),
        "aws-ap-northeast-1" => Some("JP-TK"),
        "aws-ap-south-1" => Some("IN-WE"),
        "aws-ap-southeast-1" => Some("SG"),
        "aws-ap-southeast-2" => Some("AU-NSW"),
        "aws-sa-east-1" => Some("BR-S"),
        "aws-ca-central-1" => Some("CA-ON"),
        "aws-af-south-1" => Some("ZA"),
        "aws-me-south-1" => Some("BH"),
        // Azure (prefixed)
        "azure-eastus" | "azure-eastus2" => Some("US-MIDA-PJM"),
        "azure-westus2" | "azure-westus3" => Some("US-NW-BPAT"),
        "azure-centralus" => Some("US-MIDW-MISO"),
        "azure-westeurope" => Some("NL"),
        "azure-northeurope" => Some("IE"),
        "azure-uksouth" => Some("GB"),
        "azure-francecentral" => Some("FR"),
        "azure-germanywestcentral" => Some("DE"),
        "azure-swedencentral" => Some("SE"),
        "azure-norwayeast" => Some("NO"),
        "azure-japaneast" => Some("JP-TK"),
        "azure-southeastasia" => Some("SG"),
        "azure-australiaeast" => Some("AU-NSW"),
        "azure-centralindia" => Some("IN-WE"),
        "azure-brazilsouth" => Some("BR-S"),
        "azure-canadacentral" => Some("CA-ON"),
        "azure-southafricanorth" => Some("ZA"),
        // OCI
        "oci-us-ashburn-1" => Some("US-MIDA-PJM"),
        "oci-us-phoenix-1" => Some("US-SW-AZPS"),
        "oci-us-chicago-1" => Some("US-MIDW-MISO"),
        "oci-us-sanjose-1" => Some("US-CAL-CISO"),
        "oci-eu-frankfurt-1" => Some("DE"),
        "oci-eu-amsterdam-1" => Some("NL"),
        "oci-eu-zurich-1" => Some("CH"),
        "oci-eu-stockholm-1" => Some("SE"),
        "oci-eu-paris-1" | "oci-eu-marseille-1" => Some("FR"),
        "oci-eu-milan-1" => Some("IT-NO"),
        "oci-eu-madrid-1" => Some("ES"),
        "oci-uk-london-1" => Some("GB"),
        "oci-ap-tokyo-1" | "oci-ap-osaka-1" => Some("JP-TK"),
        "oci-ap-singapore-1" => Some("SG"),
        "oci-ap-sydney-1" | "oci-ap-melbourne-1" => Some("AU-NSW"),
        "oci-ap-mumbai-1" | "oci-ap-hyderabad-1" => Some("IN-WE"),
        "oci-sa-saopaulo-1" => Some("BR-S"),
        "oci-ca-montreal-1" | "oci-ca-toronto-1" => Some("CA-ON"),
        // CoreWeave
        "coreweave-us-east-04a" | "coreweave-us-east-05a" => Some("US-MIDA-PJM"),
        "coreweave-us-central-03a" => Some("US-MIDW-MISO"),
        "coreweave-gb-london-01a" => Some("GB"),
        "coreweave-eu-oslo-01a" => Some("NO"),
        "coreweave-eu-paris-01a" => Some("FR"),
        // Lambda Labs
        "lambda-us-east-1" => Some("US-MIDA-PJM"),
        "lambda-us-west-1" => Some("US-CAL-CISO"),
        "lambda-us-south-1" => Some("US-TEX-ERCO"),
        "lambda-us-west-3" => Some("US-NW-BPAT"),
        "lambda-me-west-1" => Some("IL"),
        // Scaleway
        "scaleway-fr-par-1" | "scaleway-fr-par-2" | "scaleway-fr-par-3" => Some("FR"),
        "scaleway-nl-ams-1" | "scaleway-nl-ams-2" | "scaleway-nl-ams-3" => Some("NL"),
        "scaleway-pl-waw-1" | "scaleway-pl-waw-2" | "scaleway-pl-waw-3" => Some("PL"),
        // Akamai / Linode
        "linode-us-east" => Some("US-MIDA-PJM"),
        "linode-us-central" => Some("US-TEX-ERCO"),
        "linode-us-west" => Some("US-CAL-CISO"),
        "linode-us-southeast" => Some("US-SE-SOCO"),
        "linode-eu-west" => Some("GB"),
        "linode-eu-central" => Some("DE"),
        "linode-ap-south" => Some("SG"),
        "linode-ap-northeast" => Some("JP-TK"),
        "linode-ap-west" => Some("IN-WE"),
        "linode-ap-southeast" => Some("AU-NSW"),
        "linode-ca-central" => Some("CA-ON"),
        "linode-br-gru" => Some("BR-S"),
        // ── New regions: AWS ────────────────────────────────────────
        "ap-southeast-3" | "aws-ap-southeast-3" => Some("ID"), // Jakarta
        "ap-southeast-5" | "aws-ap-southeast-5" => Some("NZ"), // Auckland (NZ)
        "me-central-1" | "aws-me-central-1" => Some("AE"),     // UAE
        "me-south-2" | "aws-me-south-2" => Some("AE"),         // UAE
        "ap-south-2" | "aws-ap-south-2" => Some("IN-WE"),      // Hyderabad
        "eu-south-1" | "aws-eu-south-1" => Some("IT-NO"),      // Milan
        "eu-south-2" | "aws-eu-south-2" => Some("ES"),         // Spain
        "eu-central-2" | "aws-eu-central-2" => Some("CH"),     // Zurich
        "ap-southeast-4" | "aws-ap-southeast-4" => Some("AU-NSW"), // Melbourne
        "il-central-1" | "aws-il-central-1" => Some("IL"),     // Tel Aviv
        // ── New regions: GCP ────────────────────────────────────────
        "asia-southeast2" => Some("ID"),            // Jakarta
        "southamerica-west1" => Some("CL"),         // Santiago
        "me-central1" => Some("QA"),                // Doha
        "me-central2" => Some("SA"),                // Dammam
        "me-west1" => Some("IL"),                   // Tel Aviv
        "europe-west8" => Some("IT-NO"),            // Milan
        "europe-west9" => Some("FR"),               // Paris
        "europe-west12" => Some("IT-NO"),           // Turin
        "europe-southwest1" => Some("ES"),          // Madrid
        "asia-south2" => Some("IN-WE"),             // Delhi
        "australia-southeast2" => Some("AU-NSW"),   // Melbourne
        "southamerica-east1" => Some("BR-S"),       // São Paulo
        "northamerica-northeast1" => Some("CA-ON"), // Montréal
        // ── New regions: Azure ──────────────────────────────────────
        "uaenorth" | "azure-uaenorth" => Some("AE"),
        "qatarcentral" | "azure-qatarcentral" => Some("QA"),
        "israelcentral" | "azure-israelcentral" => Some("IL"),
        "italynorth" | "azure-italynorth" => Some("IT-NO"),
        "spaincentral" | "azure-spaincentral" => Some("ES"),
        "switzerlandnorth" | "azure-switzerlandnorth" => Some("CH"),
        "polandcentral" | "azure-polandcentral" => Some("PL"),
        "mexicocentral" | "azure-mexicocentral" => Some("MX"),
        "southindia" | "azure-southindia" => Some("IN-SO"),
        "koreacentral" | "azure-koreacentral" => Some("KR"),
        "japanwest" | "azure-japanwest" => Some("JP-TK"),
        "eastasia" | "azure-eastasia" => Some("HK"),
        "southeastasia" => Some("SG"),
        // ── New regions: OCI ────────────────────────────────────────
        "oci-me-jeddah-1" => Some("SA"),
        "oci-me-abudhabi-1" | "oci-me-dubai-1" => Some("AE"),
        "oci-sa-santiago-1" => Some("CL"),
        "oci-af-johannesburg-1" => Some("ZA"),
        "oci-ap-seoul-1" | "oci-ap-chuncheon-1" => Some("KR"),
        // ── New regions: GCP (prefixed) ─────────────────────────────
        "gcp-asia-southeast2" => Some("ID"),
        "gcp-southamerica-west1" => Some("CL"),
        "gcp-me-central1" => Some("QA"),
        "gcp-me-central2" => Some("SA"),
        "gcp-me-west1" => Some("IL"),
        // ── Developing nations: Africa ─────────────────────────────
        // (AWS af-south-1, Vultr vultr-jnb, Fly fly-jnb, OCI oci-af-johannesburg-1,
        //  Azure azure-southafricanorth already mapped above)
        "vultr-lag" => Some("NG"), // Lagos
        // Regional / self-hosted Africa
        "af-ng-lag" | "africa-ng" => Some("NG"), // Nigeria Lagos
        "af-ng-abj" => Some("NG"),               // Nigeria Abuja
        "af-ke-nrb" | "africa-ke" => Some("KE"), // Kenya Nairobi
        "af-gh-acc" | "africa-gh" => Some("GH"), // Ghana Accra
        "af-et-add" | "africa-et" => Some("ET"), // Ethiopia Addis
        "af-rw-kgl" | "africa-rw" => Some("RW"), // Rwanda Kigali
        "af-sn-dkr" | "africa-sn" => Some("SN"), // Senegal Dakar
        "af-tz-dar" | "africa-tz" => Some("TZ"), // Tanzania Dar es Salaam
        "af-ug-kla" | "africa-ug" => Some("UG"), // Uganda Kampala
        "af-ci-abj" | "africa-ci" => Some("CI"), // Ivory Coast Abidjan
        "af-ao-lad" | "africa-ao" => Some("AO"), // Angola Luanda
        "af-mz-mpm" | "africa-mz" => Some("MZ"), // Mozambique Maputo
        "af-eg-cai" | "africa-eg" => Some("EG"), // Egypt Cairo
        "af-ma-cmn" | "africa-ma" => Some("MA"), // Morocco Casablanca
        "af-tn-tun" | "africa-tn" => Some("TN"), // Tunisia Tunis
        "af-dz-alg" | "africa-dz" => Some("DZ"), // Algeria Algiers
        // ── Developing nations: South Asia ────────────────────────
        "sa-pk-lhe" | "asia-pk" => Some("PK"), // Pakistan Lahore
        "sa-pk-khi" => Some("PK"),             // Pakistan Karachi
        "sa-pk-isb" => Some("PK"),             // Pakistan Islamabad
        "sa-bd-dac" | "asia-bd" => Some("BD"), // Bangladesh Dhaka
        "sa-lk-cmb" | "asia-lk" => Some("LK"), // Sri Lanka Colombo
        "sa-np-ktm" | "asia-np" => Some("NP"), // Nepal Kathmandu
        // ── Developing nations: Southeast Asia ────────────────────
        "ap-southeast-vn" | "asia-vn" => Some("VN"), // Vietnam
        "vultr-sgn" | "vultr-han" => Some("VN"),     // Vultr Vietnam
        "ap-southeast-th" | "asia-th" => Some("TH"), // Thailand
        "vultr-bkk" => Some("TH"),                   // Vultr Bangkok
        "ap-southeast-ph" | "asia-ph" => Some("PH"), // Philippines
        "vultr-mnl" => Some("PH"),                   // Vultr Manila
        "ap-southeast-my" | "asia-my" => Some("MY-WM"), // Malaysia
        "ap-southeast-mm" | "asia-mm" => Some("MM"), // Myanmar
        "ap-southeast-kh" | "asia-kh" => Some("KH"), // Cambodia
        "ap-southeast-la" | "asia-la" => Some("LA"), // Laos
        // ── Developing nations: Central Asia ──────────────────────
        "ca-kz-ala" | "asia-kz" => Some("KZ"), // Kazakhstan Almaty
        "ca-kz-nqz" => Some("KZ"),             // Kazakhstan Nursultan
        "ca-uz-tas" | "asia-uz" => Some("UZ"), // Uzbekistan Tashkent
        "ca-ge-tbs" | "europe-ge" => Some("GE"), // Georgia Tbilisi
        "ca-mn-uln" | "asia-mn" => Some("MN"), // Mongolia Ulaanbaatar
        // ── Developing nations: Latin America & Caribbean ─────────
        "la-co-bog" | "latam-co" => Some("CO"), // Colombia Bogotá
        "vultr-bog" => Some("CO"),              // Vultr Bogotá
        "la-pe-lim" | "latam-pe" => Some("PE"), // Peru Lima
        "la-ar-bue" | "latam-ar" => Some("AR"), // Argentina Buenos Aires
        "la-ec-uio" | "latam-ec" => Some("EC"), // Ecuador Quito
        "la-py-asu" | "latam-py" => Some("PY"), // Paraguay Asunción
        "la-uy-mvd" | "latam-uy" => Some("UY"), // Uruguay Montevideo
        "la-cr-sjo" | "latam-cr" => Some("CR"), // Costa Rica San José
        "la-gt-gua" | "latam-gt" => Some("GT"), // Guatemala
        "la-do-sdq" | "latam-do" => Some("DO"), // Dominican Republic
        "la-ht-pap" | "latam-ht" => Some("HT"), // Haiti
        "la-jm-kin" | "latam-jm" => Some("JM"), // Jamaica
        "la-tt-pos" | "latam-tt" => Some("TT"), // Trinidad
        // Vultr Latin America
        "vultr-scl" => Some("CL"), // Vultr Santiago
        "vultr-mex" => Some("MX"), // Vultr Mexico City
        "vultr-bue" => Some("AR"), // Vultr Buenos Aires
        "vultr-lim" => Some("PE"), // Vultr Lima
        // Linode Latin America
        "linode-la-south" => Some("BR-S"),
        // ── Developing nations: Pacific Islands ───────────────────
        "pac-fj-suv" | "pacific-fj" => Some("FJ"), // Fiji Suva
        "pac-pg-pom" | "pacific-pg" => Some("PG"), // Papua New Guinea
        // ── Middle East (additional) ──────────────────────────────
        "me-iq-bgw" | "asia-iq" => Some("IQ"), // Iraq Baghdad
        "me-jo-amm" | "asia-jo" => Some("JO"), // Jordan Amman
        "me-om-mct" | "asia-om" => Some("OM"), // Oman Muscat
        // ── Orbital zones ──────────────────────────────────────────
        // Orbital nodes use 100% solar power when sunlit (zero carbon).
        // Map them to a synthetic zone handled by the aggregator.
        "orbital-leo" | "orbital-meo" | "orbital-geo" | "orbital-heo" => Some("ORBITAL"),
        _ => None,
    }
}

/// Carbon intensity for orbital compute (gCO2eq/kWh).
///
/// Satellites in sunlight use photovoltaic panels — zero operational
/// carbon emissions. During eclipse they drain batteries charged from
/// solar, so lifecycle emissions are dominated by launch carbon amortized
/// over operational life. We use 0.0 for operational intensity since
/// the energy source is pure solar with no grid dependency.
pub const ORBITAL_CARBON_INTENSITY_GCO2_KWH: f64 = 0.0;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- config defaults ----------------------------------------------------

    #[test]
    fn config_defaults() {
        let cfg = ElectricityMapsConfig::default();
        assert_eq!(cfg.base_url, "https://api.electricitymap.org/v3");
        assert_eq!(cfg.timeout_secs, 10);
        assert_eq!(cfg.cache_ttl_secs, 300);
        assert!(cfg.api_key.is_empty());
    }

    // -- cloud zone mapping: AWS -------------------------------------------

    #[test]
    fn cloud_zone_mapping_aws() {
        assert_eq!(cloud_zone_mapping("us-east-1"), Some("US-MIDA-PJM"));
        assert_eq!(cloud_zone_mapping("us-east-2"), Some("US-MIDA-PJM"));
        assert_eq!(cloud_zone_mapping("us-west-1"), Some("US-CAL-CISO"));
        assert_eq!(cloud_zone_mapping("us-west-2"), Some("US-NW-BPAT"));
        assert_eq!(cloud_zone_mapping("eu-west-1"), Some("IE"));
        assert_eq!(cloud_zone_mapping("eu-west-2"), Some("GB"));
        assert_eq!(cloud_zone_mapping("eu-west-3"), Some("FR"));
        assert_eq!(cloud_zone_mapping("eu-central-1"), Some("DE"));
        assert_eq!(cloud_zone_mapping("eu-north-1"), Some("SE"));
        assert_eq!(cloud_zone_mapping("ap-northeast-1"), Some("JP-TK"));
        assert_eq!(cloud_zone_mapping("ap-south-1"), Some("IN-WE"));
        assert_eq!(cloud_zone_mapping("ap-southeast-1"), Some("SG"));
        assert_eq!(cloud_zone_mapping("ap-southeast-2"), Some("AU-NSW"));
    }

    // -- cloud zone mapping: GCP -------------------------------------------

    #[test]
    fn cloud_zone_mapping_gcp() {
        assert_eq!(cloud_zone_mapping("europe-west1"), Some("BE"));
        assert_eq!(cloud_zone_mapping("europe-west4"), Some("NL"));
        assert_eq!(cloud_zone_mapping("europe-north1"), Some("FI"));
        assert_eq!(cloud_zone_mapping("us-central1"), Some("US-MIDW-MISO"));
    }

    // -- cloud zone mapping: Azure -----------------------------------------

    #[test]
    fn cloud_zone_mapping_azure() {
        assert_eq!(cloud_zone_mapping("westeurope"), Some("NL"));
        assert_eq!(cloud_zone_mapping("northeurope"), Some("IE"));
        assert_eq!(cloud_zone_mapping("westus2"), Some("US-NW-BPAT"));
        assert_eq!(cloud_zone_mapping("eastus"), Some("US-MIDA-PJM"));
    }

    // -- cloud zone mapping: Hetzner ------------------------------------------

    #[test]
    fn cloud_zone_mapping_hetzner() {
        assert_eq!(cloud_zone_mapping("hetzner-fsn1"), Some("DE"));
        assert_eq!(cloud_zone_mapping("hetzner-nbg1"), Some("DE"));
        assert_eq!(cloud_zone_mapping("hetzner-hel1"), Some("FI"));
        assert_eq!(cloud_zone_mapping("hetzner-ash"), Some("US-MIDA-PJM"));
        assert_eq!(cloud_zone_mapping("hetzner-hil"), Some("US-NW-BPAT"));
        assert_eq!(cloud_zone_mapping("hetzner-sin"), Some("SG"));
    }

    // -- cloud zone mapping: Vultr --------------------------------------------

    #[test]
    fn cloud_zone_mapping_vultr() {
        assert_eq!(cloud_zone_mapping("vultr-ewr"), Some("US-NY-NYIS"));
        assert_eq!(cloud_zone_mapping("vultr-ord"), Some("US-MIDW-MISO"));
        assert_eq!(cloud_zone_mapping("vultr-dfw"), Some("US-TEX-ERCO"));
        assert_eq!(cloud_zone_mapping("vultr-lax"), Some("US-CAL-CISO"));
        assert_eq!(cloud_zone_mapping("vultr-sea"), Some("US-NW-BPAT"));
        assert_eq!(cloud_zone_mapping("vultr-lhr"), Some("GB"));
        assert_eq!(cloud_zone_mapping("vultr-ams"), Some("NL"));
        assert_eq!(cloud_zone_mapping("vultr-fra"), Some("DE"));
        assert_eq!(cloud_zone_mapping("vultr-cdg"), Some("FR"));
        assert_eq!(cloud_zone_mapping("vultr-sto"), Some("SE"));
        assert_eq!(cloud_zone_mapping("vultr-nrt"), Some("JP-TK"));
        assert_eq!(cloud_zone_mapping("vultr-sgp"), Some("SG"));
        assert_eq!(cloud_zone_mapping("vultr-syd"), Some("AU-NSW"));
        assert_eq!(cloud_zone_mapping("vultr-bom"), Some("IN-WE"));
        assert_eq!(cloud_zone_mapping("vultr-sao"), Some("BR-S"));
        assert_eq!(cloud_zone_mapping("vultr-yto"), Some("CA-ON"));
        assert_eq!(cloud_zone_mapping("vultr-jnb"), Some("ZA"));
    }

    // -- cloud zone mapping: Latitude.sh --------------------------------------

    #[test]
    fn cloud_zone_mapping_latitude() {
        assert_eq!(cloud_zone_mapping("latitude-mia"), Some("US-SE-SOCO"));
        assert_eq!(cloud_zone_mapping("latitude-dal"), Some("US-TEX-ERCO"));
        assert_eq!(cloud_zone_mapping("latitude-chi"), Some("US-MIDW-MISO"));
        assert_eq!(cloud_zone_mapping("latitude-lax"), Some("US-CAL-CISO"));
        assert_eq!(cloud_zone_mapping("latitude-nyc"), Some("US-NY-NYIS"));
        assert_eq!(cloud_zone_mapping("latitude-sao"), Some("BR-S"));
        assert_eq!(cloud_zone_mapping("latitude-fra"), Some("DE"));
        assert_eq!(cloud_zone_mapping("latitude-lon"), Some("GB"));
        assert_eq!(cloud_zone_mapping("latitude-par"), Some("FR"));
        assert_eq!(cloud_zone_mapping("latitude-sin"), Some("SG"));
    }

    // -- cloud zone mapping: DigitalOcean -------------------------------------

    #[test]
    fn cloud_zone_mapping_digitalocean() {
        assert_eq!(cloud_zone_mapping("do-nyc1"), Some("US-NY-NYIS"));
        assert_eq!(cloud_zone_mapping("do-sfo2"), Some("US-CAL-CISO"));
        assert_eq!(cloud_zone_mapping("do-ams3"), Some("NL"));
        assert_eq!(cloud_zone_mapping("do-sgp1"), Some("SG"));
        assert_eq!(cloud_zone_mapping("do-lon1"), Some("GB"));
        assert_eq!(cloud_zone_mapping("do-fra1"), Some("DE"));
        assert_eq!(cloud_zone_mapping("do-tor1"), Some("CA-ON"));
        assert_eq!(cloud_zone_mapping("do-blr1"), Some("IN-SO"));
        assert_eq!(cloud_zone_mapping("do-syd1"), Some("AU-NSW"));
    }

    // -- cloud zone mapping: Fly.io -------------------------------------------

    #[test]
    fn cloud_zone_mapping_fly() {
        assert_eq!(cloud_zone_mapping("fly-iad"), Some("US-MIDA-PJM"));
        assert_eq!(cloud_zone_mapping("fly-ord"), Some("US-MIDW-MISO"));
        assert_eq!(cloud_zone_mapping("fly-dfw"), Some("US-TEX-ERCO"));
        assert_eq!(cloud_zone_mapping("fly-lax"), Some("US-CAL-CISO"));
        assert_eq!(cloud_zone_mapping("fly-sea"), Some("US-NW-BPAT"));
        assert_eq!(cloud_zone_mapping("fly-lhr"), Some("GB"));
        assert_eq!(cloud_zone_mapping("fly-ams"), Some("NL"));
        assert_eq!(cloud_zone_mapping("fly-cdg"), Some("FR"));
        assert_eq!(cloud_zone_mapping("fly-arn"), Some("SE"));
        assert_eq!(cloud_zone_mapping("fly-nrt"), Some("JP-TK"));
        assert_eq!(cloud_zone_mapping("fly-sin"), Some("SG"));
        assert_eq!(cloud_zone_mapping("fly-syd"), Some("AU-NSW"));
        assert_eq!(cloud_zone_mapping("fly-bom"), Some("IN-WE"));
        assert_eq!(cloud_zone_mapping("fly-gru"), Some("BR-S"));
        assert_eq!(cloud_zone_mapping("fly-jnb"), Some("ZA"));
    }

    // -- cloud zone mapping: Nebius -------------------------------------------

    #[test]
    fn cloud_zone_mapping_nebius() {
        assert_eq!(cloud_zone_mapping("nebius-eu-north1"), Some("FI"));
        assert_eq!(cloud_zone_mapping("nebius-eu-west1"), Some("FR"));
        assert_eq!(
            cloud_zone_mapping("nebius-us-central1"),
            Some("US-MIDW-MISO")
        );
        assert_eq!(cloud_zone_mapping("nebius-me-west1"), Some("IL"));
    }

    // -- cloud zone mapping: AWS (prefixed) -----------------------------------

    #[test]
    fn cloud_zone_mapping_aws_prefixed() {
        assert_eq!(cloud_zone_mapping("aws-us-east-1"), Some("US-MIDA-PJM"));
        assert_eq!(cloud_zone_mapping("aws-us-west-2"), Some("US-NW-BPAT"));
        assert_eq!(cloud_zone_mapping("aws-eu-west-1"), Some("IE"));
        assert_eq!(cloud_zone_mapping("aws-eu-north-1"), Some("SE"));
        assert_eq!(cloud_zone_mapping("aws-ap-northeast-1"), Some("JP-TK"));
        assert_eq!(cloud_zone_mapping("aws-sa-east-1"), Some("BR-S"));
        assert_eq!(cloud_zone_mapping("aws-ca-central-1"), Some("CA-ON"));
        assert_eq!(cloud_zone_mapping("aws-af-south-1"), Some("ZA"));
    }

    // -- cloud zone mapping: Azure (prefixed) ---------------------------------

    #[test]
    fn cloud_zone_mapping_azure_prefixed() {
        assert_eq!(cloud_zone_mapping("azure-eastus"), Some("US-MIDA-PJM"));
        assert_eq!(cloud_zone_mapping("azure-westeurope"), Some("NL"));
        assert_eq!(cloud_zone_mapping("azure-northeurope"), Some("IE"));
        assert_eq!(cloud_zone_mapping("azure-uksouth"), Some("GB"));
        assert_eq!(cloud_zone_mapping("azure-francecentral"), Some("FR"));
        assert_eq!(cloud_zone_mapping("azure-swedencentral"), Some("SE"));
        assert_eq!(cloud_zone_mapping("azure-japaneast"), Some("JP-TK"));
        assert_eq!(cloud_zone_mapping("azure-canadacentral"), Some("CA-ON"));
    }

    // -- cloud zone mapping: OCI ----------------------------------------------

    #[test]
    fn cloud_zone_mapping_oci() {
        assert_eq!(cloud_zone_mapping("oci-us-ashburn-1"), Some("US-MIDA-PJM"));
        assert_eq!(cloud_zone_mapping("oci-eu-frankfurt-1"), Some("DE"));
        assert_eq!(cloud_zone_mapping("oci-eu-amsterdam-1"), Some("NL"));
        assert_eq!(cloud_zone_mapping("oci-uk-london-1"), Some("GB"));
        assert_eq!(cloud_zone_mapping("oci-ap-tokyo-1"), Some("JP-TK"));
        assert_eq!(cloud_zone_mapping("oci-ap-singapore-1"), Some("SG"));
        assert_eq!(cloud_zone_mapping("oci-sa-saopaulo-1"), Some("BR-S"));
        assert_eq!(cloud_zone_mapping("oci-eu-stockholm-1"), Some("SE"));
    }

    // -- cloud zone mapping: CoreWeave ----------------------------------------

    #[test]
    fn cloud_zone_mapping_coreweave() {
        assert_eq!(
            cloud_zone_mapping("coreweave-us-east-04a"),
            Some("US-MIDA-PJM")
        );
        assert_eq!(
            cloud_zone_mapping("coreweave-us-central-03a"),
            Some("US-MIDW-MISO")
        );
        assert_eq!(cloud_zone_mapping("coreweave-gb-london-01a"), Some("GB"));
        assert_eq!(cloud_zone_mapping("coreweave-eu-oslo-01a"), Some("NO"));
        assert_eq!(cloud_zone_mapping("coreweave-eu-paris-01a"), Some("FR"));
    }

    // -- cloud zone mapping: Lambda Labs --------------------------------------

    #[test]
    fn cloud_zone_mapping_lambda() {
        assert_eq!(cloud_zone_mapping("lambda-us-east-1"), Some("US-MIDA-PJM"));
        assert_eq!(cloud_zone_mapping("lambda-us-west-1"), Some("US-CAL-CISO"));
        assert_eq!(cloud_zone_mapping("lambda-us-south-1"), Some("US-TEX-ERCO"));
        assert_eq!(cloud_zone_mapping("lambda-us-west-3"), Some("US-NW-BPAT"));
        assert_eq!(cloud_zone_mapping("lambda-me-west-1"), Some("IL"));
    }

    // -- cloud zone mapping: Scaleway -----------------------------------------

    #[test]
    fn cloud_zone_mapping_scaleway() {
        assert_eq!(cloud_zone_mapping("scaleway-fr-par-1"), Some("FR"));
        assert_eq!(cloud_zone_mapping("scaleway-nl-ams-1"), Some("NL"));
        assert_eq!(cloud_zone_mapping("scaleway-pl-waw-1"), Some("PL"));
    }

    // -- cloud zone mapping: Linode -------------------------------------------

    #[test]
    fn cloud_zone_mapping_linode() {
        assert_eq!(cloud_zone_mapping("linode-us-east"), Some("US-MIDA-PJM"));
        assert_eq!(cloud_zone_mapping("linode-us-central"), Some("US-TEX-ERCO"));
        assert_eq!(cloud_zone_mapping("linode-us-west"), Some("US-CAL-CISO"));
        assert_eq!(cloud_zone_mapping("linode-eu-west"), Some("GB"));
        assert_eq!(cloud_zone_mapping("linode-eu-central"), Some("DE"));
        assert_eq!(cloud_zone_mapping("linode-ap-south"), Some("SG"));
        assert_eq!(cloud_zone_mapping("linode-ap-northeast"), Some("JP-TK"));
        assert_eq!(cloud_zone_mapping("linode-ca-central"), Some("CA-ON"));
        assert_eq!(cloud_zone_mapping("linode-br-gru"), Some("BR-S"));
    }

    // -- cloud zone mapping: unknown ---------------------------------------

    #[test]
    fn cloud_zone_mapping_unknown() {
        assert_eq!(cloud_zone_mapping("mars-colony-1"), None);
        assert_eq!(cloud_zone_mapping(""), None);
        assert_eq!(cloud_zone_mapping("us-south-1"), None);
    }

    // -- cloud zone mapping: orbital ----------------------------------------

    #[test]
    fn cloud_zone_mapping_orbital() {
        assert_eq!(cloud_zone_mapping("orbital-leo"), Some("ORBITAL"));
        assert_eq!(cloud_zone_mapping("orbital-meo"), Some("ORBITAL"));
        assert_eq!(cloud_zone_mapping("orbital-geo"), Some("ORBITAL"));
        assert_eq!(cloud_zone_mapping("orbital-heo"), Some("ORBITAL"));
    }

    #[test]
    fn orbital_carbon_intensity_is_zero() {
        assert!((ORBITAL_CARBON_INTENSITY_GCO2_KWH - 0.0).abs() < f64::EPSILON);
    }

    // -- serde roundtrip: CarbonIntensityResponse --------------------------

    #[test]
    fn carbon_intensity_response_serde() {
        let original = CarbonIntensityResponse {
            zone: "IE".to_string(),
            carbon_intensity_gco2_kwh: 320.5,
            fossil_fuel_percentage: 45.2,
            datetime: Utc::now(),
            data_source: "electricitymaps.com".to_string(),
            is_estimated: false,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: CarbonIntensityResponse = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.zone, original.zone);
        assert!(
            (restored.carbon_intensity_gco2_kwh - original.carbon_intensity_gco2_kwh).abs() < 1e-10
        );
        assert!((restored.fossil_fuel_percentage - original.fossil_fuel_percentage).abs() < 1e-10);
        assert_eq!(restored.data_source, original.data_source);
        assert_eq!(restored.is_estimated, original.is_estimated);
    }

    // -- serde roundtrip: RenewableEnergyResponse --------------------------

    #[test]
    fn renewable_energy_response_serde() {
        let original = RenewableEnergyResponse {
            zone: "DE".to_string(),
            renewable_percentage: 55.3,
            wind_percentage: 30.0,
            solar_percentage: 10.0,
            hydro_percentage: 15.3,
            nuclear_percentage: 12.0,
            fossil_percentage: 32.7,
            datetime: Utc::now(),
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: RenewableEnergyResponse = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.zone, original.zone);
        assert!((restored.renewable_percentage - original.renewable_percentage).abs() < 1e-10);
        assert!((restored.wind_percentage - original.wind_percentage).abs() < 1e-10);
        assert!((restored.solar_percentage - original.solar_percentage).abs() < 1e-10);
        assert!((restored.hydro_percentage - original.hydro_percentage).abs() < 1e-10);
        assert!((restored.nuclear_percentage - original.nuclear_percentage).abs() < 1e-10);
        assert!((restored.fossil_percentage - original.fossil_percentage).abs() < 1e-10);
    }

    // -- client not configured ---------------------------------------------

    #[test]
    fn client_not_configured() {
        let client = ElectricityMapsClient::new(ElectricityMapsConfig::default());
        assert!(!client.is_configured());

        let client_with_key = ElectricityMapsClient::new(ElectricityMapsConfig {
            api_key: "emaps-test-key".to_string(),
            ..Default::default()
        });
        assert!(client_with_key.is_configured());
    }

    // -- error display -----------------------------------------------------

    #[test]
    fn error_display() {
        let err = CarbonApiError::NoApiKey;
        assert_eq!(err.to_string(), "API key not configured");

        let err = CarbonApiError::ApiError {
            status: 403,
            message: "Forbidden".to_string(),
        };
        assert_eq!(err.to_string(), "API returned error status 403: Forbidden");

        let err = CarbonApiError::UnknownZone("XY".to_string());
        assert_eq!(err.to_string(), "unknown zone: XY");

        let err = CarbonApiError::Parse("bad json".to_string());
        assert_eq!(err.to_string(), "parse error: bad json");
    }

    // -- Phase 1F: new cloud region mappings --

    #[test]
    fn cloud_zone_mapping_new_regions() {
        // AWS new regions
        assert_eq!(cloud_zone_mapping("ap-southeast-3"), Some("ID"));
        assert_eq!(cloud_zone_mapping("me-central-1"), Some("AE"));
        assert_eq!(cloud_zone_mapping("il-central-1"), Some("IL"));
        // GCP new regions
        assert_eq!(cloud_zone_mapping("asia-southeast2"), Some("ID"));
        assert_eq!(cloud_zone_mapping("southamerica-west1"), Some("CL"));
        assert_eq!(cloud_zone_mapping("me-central1"), Some("QA"));
        assert_eq!(cloud_zone_mapping("me-central2"), Some("SA"));
        assert_eq!(cloud_zone_mapping("southamerica-east1"), Some("BR-S"));
        // Azure new regions
        assert_eq!(cloud_zone_mapping("uaenorth"), Some("AE"));
        assert_eq!(cloud_zone_mapping("qatarcentral"), Some("QA"));
        assert_eq!(cloud_zone_mapping("israelcentral"), Some("IL"));
        assert_eq!(cloud_zone_mapping("mexicocentral"), Some("MX"));
        assert_eq!(cloud_zone_mapping("koreacentral"), Some("KR"));
        // OCI new regions
        assert_eq!(cloud_zone_mapping("oci-me-jeddah-1"), Some("SA"));
        assert_eq!(cloud_zone_mapping("oci-sa-santiago-1"), Some("CL"));
        // Prefixed new regions
        assert_eq!(cloud_zone_mapping("aws-ap-southeast-3"), Some("ID"));
        assert_eq!(cloud_zone_mapping("azure-uaenorth"), Some("AE"));
        assert_eq!(cloud_zone_mapping("gcp-me-central1"), Some("QA"));
    }

    // -- cloud zone mapping: developing nations --------------------------------

    #[test]
    fn cloud_zone_mapping_africa() {
        // Nigeria
        assert_eq!(cloud_zone_mapping("af-ng-lag"), Some("NG"));
        assert_eq!(cloud_zone_mapping("africa-ng"), Some("NG"));
        assert_eq!(cloud_zone_mapping("vultr-lag"), Some("NG"));
        // Kenya
        assert_eq!(cloud_zone_mapping("af-ke-nrb"), Some("KE"));
        assert_eq!(cloud_zone_mapping("africa-ke"), Some("KE"));
        // Ethiopia
        assert_eq!(cloud_zone_mapping("af-et-add"), Some("ET"));
        // Egypt
        assert_eq!(cloud_zone_mapping("af-eg-cai"), Some("EG"));
        // Morocco
        assert_eq!(cloud_zone_mapping("af-ma-cmn"), Some("MA"));
        // Ghana
        assert_eq!(cloud_zone_mapping("af-gh-acc"), Some("GH"));
    }

    #[test]
    fn cloud_zone_mapping_south_asia() {
        assert_eq!(cloud_zone_mapping("sa-pk-lhe"), Some("PK"));
        assert_eq!(cloud_zone_mapping("sa-bd-dac"), Some("BD"));
        assert_eq!(cloud_zone_mapping("sa-lk-cmb"), Some("LK"));
        assert_eq!(cloud_zone_mapping("sa-np-ktm"), Some("NP"));
        assert_eq!(cloud_zone_mapping("asia-pk"), Some("PK"));
    }

    #[test]
    fn cloud_zone_mapping_southeast_asia() {
        assert_eq!(cloud_zone_mapping("ap-southeast-vn"), Some("VN"));
        assert_eq!(cloud_zone_mapping("ap-southeast-th"), Some("TH"));
        assert_eq!(cloud_zone_mapping("ap-southeast-ph"), Some("PH"));
        assert_eq!(cloud_zone_mapping("ap-southeast-my"), Some("MY-WM"));
        assert_eq!(cloud_zone_mapping("ap-southeast-mm"), Some("MM"));
        assert_eq!(cloud_zone_mapping("ap-southeast-kh"), Some("KH"));
    }

    #[test]
    fn cloud_zone_mapping_central_asia() {
        assert_eq!(cloud_zone_mapping("ca-kz-ala"), Some("KZ"));
        assert_eq!(cloud_zone_mapping("ca-uz-tas"), Some("UZ"));
        assert_eq!(cloud_zone_mapping("ca-ge-tbs"), Some("GE"));
        assert_eq!(cloud_zone_mapping("ca-mn-uln"), Some("MN"));
    }

    #[test]
    fn cloud_zone_mapping_latin_america() {
        assert_eq!(cloud_zone_mapping("la-co-bog"), Some("CO"));
        assert_eq!(cloud_zone_mapping("la-pe-lim"), Some("PE"));
        assert_eq!(cloud_zone_mapping("la-ar-bue"), Some("AR"));
        assert_eq!(cloud_zone_mapping("la-cr-sjo"), Some("CR"));
        assert_eq!(cloud_zone_mapping("la-ht-pap"), Some("HT"));
        assert_eq!(cloud_zone_mapping("la-jm-kin"), Some("JM"));
        assert_eq!(cloud_zone_mapping("la-py-asu"), Some("PY"));
    }

    #[test]
    fn cloud_zone_mapping_pacific() {
        assert_eq!(cloud_zone_mapping("pac-fj-suv"), Some("FJ"));
        assert_eq!(cloud_zone_mapping("pac-pg-pom"), Some("PG"));
    }

    #[test]
    fn cloud_zone_mapping_middle_east_expanded() {
        assert_eq!(cloud_zone_mapping("me-iq-bgw"), Some("IQ"));
        assert_eq!(cloud_zone_mapping("me-jo-amm"), Some("JO"));
        assert_eq!(cloud_zone_mapping("me-om-mct"), Some("OM"));
    }
}
