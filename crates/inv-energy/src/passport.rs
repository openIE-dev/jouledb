//! AI Service Passport types for machine-readable energy, carbon, and
//! compliance records.
//!
//! An [`AiServicePassport`] bundles every metric an API consumer needs to
//! compare inference endpoints on energy efficiency, carbon footprint, and
//! regulatory compliance (EU AI Act, CSRD, EED, ESPR DPP).

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::energy_label::EnergyLabel;

// ---------------------------------------------------------------------------
// MeasurementMethod
// ---------------------------------------------------------------------------

/// How the energy profile was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasurementMethod {
    /// Intel / AMD RAPL counters.
    Rapl,
    /// Apple Silicon IOReport.
    IoReport,
    /// NVIDIA Management Library.
    Nvml,
    /// Combined GPU (NVML) + CPU (RAPL) measurement.
    NvmlRapl,
    /// Derived from billing data or hardware specs, not measured.
    Estimated,
}

impl fmt::Display for MeasurementMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rapl => write!(f, "RAPL"),
            Self::IoReport => write!(f, "IOReport"),
            Self::Nvml => write!(f, "NVML"),
            Self::NvmlRapl => write!(f, "NVML+RAPL"),
            Self::Estimated => write!(f, "Estimated"),
        }
    }
}

// ---------------------------------------------------------------------------
// ModelProfile
// ---------------------------------------------------------------------------

/// Identity and configuration of the served model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Model identifier (e.g. "meta-llama/Llama-3.3-70B-Instruct").
    pub id: String,
    /// Model family (e.g. "Llama 3.3").
    pub family: String,
    /// Total parameter count.
    pub parameters: u64,
    /// Quantization format: "FP8", "FP16", "INT4", etc.
    pub quantization: String,
    /// Model version tag.
    pub version: String,
}

// ---------------------------------------------------------------------------
// HardwareProfile
// ---------------------------------------------------------------------------

/// Hardware running the inference service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Accelerator model (e.g. "NVIDIA H100 SXM").
    pub accelerator: String,
    /// CPU model.
    pub cpu: String,
    /// Total memory in GiB.
    pub memory_gb: u32,
    /// Interconnect type (e.g. "NVLink 4.0", "PCIe 5.0").
    pub interconnect: String,
}

// ---------------------------------------------------------------------------
// EnergyProfile
// ---------------------------------------------------------------------------

/// Measured energy characteristics of the inference service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyProfile {
    /// Energy consumed per input token (joules).
    pub joules_per_input_token: f64,
    /// Energy consumed per output token (joules).
    pub joules_per_output_token: f64,
    /// How the measurements were obtained.
    pub measurement_method: MeasurementMethod,
    /// Number of inference requests in the measurement sample.
    pub sample_size: u64,
    /// Date the benchmark was performed.
    pub benchmark_date: DateTime<Utc>,
    /// Idle power draw (watts).
    pub idle_power_watts: f64,
    /// Peak observed power draw (watts).
    pub peak_power_watts: f64,
}

// ---------------------------------------------------------------------------
// LocationInfo
// ---------------------------------------------------------------------------

/// Geographic location of the inference service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationInfo {
    /// Cloud region or data-centre identifier (e.g. "eu-west-1").
    pub region: String,
    /// ISO 3166-1 alpha-2 country code.
    pub country: String,
    /// Electricity grid zone identifier.
    pub grid_zone: String,
}

// ---------------------------------------------------------------------------
// CarbonProfile
// ---------------------------------------------------------------------------

/// Carbon footprint of the inference service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonProfile {
    /// SCI score: grams CO2 equivalent per request.
    pub sci_score_gco2_per_request: f64,
    /// Grid carbon intensity (gCO2 / kWh).
    pub grid_carbon_intensity_gco2_kwh: f64,
    /// Source of the carbon-intensity data (e.g. "electricitymaps.com").
    pub carbon_intensity_source: String,
    /// Percentage of energy from renewable sources (0.0 -- 100.0).
    pub renewable_percentage: f64,
    /// Amortised embodied carbon per request (gCO2).
    pub embodied_carbon_gco2_per_request: f64,
    /// Where the service is located.
    pub location: LocationInfo,
}

// ---------------------------------------------------------------------------
// PassportLabel
// ---------------------------------------------------------------------------

/// Energy-efficiency label derived from the existing A-G rating scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassportLabel {
    /// EU-style A-G efficiency rating.
    pub rating: EnergyLabel,
    /// Labelling methodology description.
    pub methodology: String,
    /// Model used as the efficiency baseline.
    pub baseline_model: String,
    /// Ratio of measured energy to baseline (lower is better).
    pub efficiency_ratio: f64,
}

// ---------------------------------------------------------------------------
// AiActCompliance
// ---------------------------------------------------------------------------

/// EU AI Act energy and emissions disclosure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiActCompliance {
    /// Whether training energy has been publicly disclosed.
    pub training_energy_disclosed: bool,
    /// Total training energy consumption (kWh), if known.
    pub training_energy_kwh: Option<f64>,
    /// Total training emissions (tonnes CO2), if known.
    pub training_emissions_tco2: Option<f64>,
    /// Whether inference energy is actively measured (not estimated).
    pub inference_energy_measured: bool,
}

// ---------------------------------------------------------------------------
// CsrdCompliance
// ---------------------------------------------------------------------------

/// Corporate Sustainability Reporting Directive scope-3 compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrdCompliance {
    /// Whether the figures have been independently verified.
    pub verified: bool,
    /// Methodology used for the scope-3 calculation.
    pub methodology: String,
    /// Reference to the full audit trail.
    pub audit_trail: String,
}

// ---------------------------------------------------------------------------
// EedCompliance
// ---------------------------------------------------------------------------

/// Energy Efficiency Directive data-centre metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EedCompliance {
    /// Power Usage Effectiveness.
    pub facility_pue: f64,
    /// Water Usage Effectiveness.
    pub facility_wue: f64,
    /// Percentage of waste heat recovered.
    pub waste_heat_recovery_pct: f64,
    /// Renewable energy factor (0.0 -- 1.0).
    pub renewable_energy_factor: f64,
}

// ---------------------------------------------------------------------------
// EsprCompliance
// ---------------------------------------------------------------------------

/// Ecodesign for Sustainable Products Regulation -- Digital Product Passport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsprCompliance {
    /// URL of the hosted digital product passport.
    pub passport_url: String,
    /// Whether the passport has been submitted to the EU registry.
    pub registry_submitted: bool,
}

// ---------------------------------------------------------------------------
// ComplianceRecord
// ---------------------------------------------------------------------------

/// Aggregate compliance status across EU regulatory frameworks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRecord {
    /// EU AI Act compliance.
    pub eu_ai_act: AiActCompliance,
    /// CSRD scope-3 compliance.
    pub csrd_scope_3: CsrdCompliance,
    /// Energy Efficiency Directive compliance.
    pub eed: EedCompliance,
    /// ESPR Digital Product Passport compliance.
    pub espr_dpp: EsprCompliance,
}

// ---------------------------------------------------------------------------
// PricingInfo
// ---------------------------------------------------------------------------

/// Token and energy pricing with on-chain settlement details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingInfo {
    /// Price per input token (USDC).
    pub usdc_per_input_token: f64,
    /// Price per output token (USDC).
    pub usdc_per_output_token: f64,
    /// Price per joule of energy consumed (USDC).
    pub usdc_per_joule: f64,
    /// CAIP-2 chain identifier for settlement (e.g. "eip155:8453").
    pub settlement_network: String,
    /// HTTP endpoint for x402 payment negotiation.
    pub x402_endpoint: String,
    /// Free-tier energy budget per day (joules).
    pub free_tier_joules_per_day: f64,
}

impl Default for PricingInfo {
    fn default() -> Self {
        Self {
            usdc_per_input_token: 0.0,
            usdc_per_output_token: 0.0,
            usdc_per_joule: 0.0,
            settlement_network: "eip155:8453".to_string(),
            x402_endpoint: String::new(),
            free_tier_joules_per_day: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// AiServicePassport
// ---------------------------------------------------------------------------

/// Complete AI Service Passport -- energy, carbon, and compliance record for
/// an AI inference service endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiServicePassport {
    /// Passport schema version (e.g. "1.0").
    pub passport_version: String,
    /// Unique service identifier (e.g. "invisible:llama-3.3-70b:base-eu-west").
    pub service_id: String,
    /// Infrastructure provider (e.g. "invisible.dev").
    pub provider: String,
    /// When the passport was issued.
    pub issued_at: DateTime<Utc>,
    /// When the passport expires.
    pub valid_until: DateTime<Utc>,
    /// Served model description.
    pub model: ModelProfile,
    /// Hardware running the service.
    pub hardware: HardwareProfile,
    /// Energy measurements.
    pub energy_profile: EnergyProfile,
    /// Carbon footprint.
    pub carbon_profile: CarbonProfile,
    /// EU-style energy label.
    pub energy_label: PassportLabel,
    /// Regulatory compliance status.
    pub compliance: ComplianceRecord,
    /// Pricing and settlement information.
    pub pricing: PricingInfo,
}

impl AiServicePassport {
    /// Returns a builder pre-populated with the given `service_id`.
    pub fn builder(service_id: &str) -> PassportBuilder {
        PassportBuilder {
            service_id: service_id.to_string(),
            passport_version: "1.0".to_string(),
            provider: String::new(),
            issued_at: None,
            valid_until: None,
            model: None,
            hardware: None,
            energy_profile: None,
            carbon_profile: None,
            energy_label: None,
            compliance: None,
            pricing: None,
        }
    }
}

// ---------------------------------------------------------------------------
// PassportBuilder
// ---------------------------------------------------------------------------

/// Incremental builder for [`AiServicePassport`].
#[derive(Debug)]
pub struct PassportBuilder {
    service_id: String,
    passport_version: String,
    provider: String,
    issued_at: Option<DateTime<Utc>>,
    valid_until: Option<DateTime<Utc>>,
    model: Option<ModelProfile>,
    hardware: Option<HardwareProfile>,
    energy_profile: Option<EnergyProfile>,
    carbon_profile: Option<CarbonProfile>,
    energy_label: Option<PassportLabel>,
    compliance: Option<ComplianceRecord>,
    pricing: Option<PricingInfo>,
}

impl PassportBuilder {
    /// Sets the passport schema version (default "1.0").
    pub fn passport_version(mut self, v: &str) -> Self {
        self.passport_version = v.to_string();
        self
    }

    /// Sets the provider name.
    pub fn provider(mut self, p: &str) -> Self {
        self.provider = p.to_string();
        self
    }

    /// Sets the issuance timestamp.
    pub fn issued_at(mut self, t: DateTime<Utc>) -> Self {
        self.issued_at = Some(t);
        self
    }

    /// Sets the expiry timestamp.
    pub fn valid_until(mut self, t: DateTime<Utc>) -> Self {
        self.valid_until = Some(t);
        self
    }

    /// Sets the model profile.
    pub fn model(mut self, m: ModelProfile) -> Self {
        self.model = Some(m);
        self
    }

    /// Sets the hardware profile.
    pub fn hardware(mut self, h: HardwareProfile) -> Self {
        self.hardware = Some(h);
        self
    }

    /// Sets the energy profile.
    pub fn energy_profile(mut self, e: EnergyProfile) -> Self {
        self.energy_profile = Some(e);
        self
    }

    /// Sets the carbon profile.
    pub fn carbon_profile(mut self, c: CarbonProfile) -> Self {
        self.carbon_profile = Some(c);
        self
    }

    /// Sets the energy label.
    pub fn energy_label(mut self, l: PassportLabel) -> Self {
        self.energy_label = Some(l);
        self
    }

    /// Sets the compliance record.
    pub fn compliance(mut self, c: ComplianceRecord) -> Self {
        self.compliance = Some(c);
        self
    }

    /// Sets the pricing information.
    pub fn pricing(mut self, p: PricingInfo) -> Self {
        self.pricing = Some(p);
        self
    }

    /// Consumes the builder and returns a completed [`AiServicePassport`].
    ///
    /// # Panics
    ///
    /// Panics if any required field has not been set.
    pub fn build(self) -> AiServicePassport {
        AiServicePassport {
            passport_version: self.passport_version,
            service_id: self.service_id,
            provider: self.provider,
            issued_at: self.issued_at.expect("issued_at is required"),
            valid_until: self.valid_until.expect("valid_until is required"),
            model: self.model.expect("model is required"),
            hardware: self.hardware.expect("hardware is required"),
            energy_profile: self.energy_profile.expect("energy_profile is required"),
            carbon_profile: self.carbon_profile.expect("carbon_profile is required"),
            energy_label: self.energy_label.expect("energy_label is required"),
            compliance: self.compliance.expect("compliance is required"),
            pricing: self.pricing.unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------------
// PassportCatalog
// ---------------------------------------------------------------------------

/// Collection of available service passports for discovery and comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PassportCatalog {
    /// All registered passports.
    pub passports: Vec<AiServicePassport>,
}

impl PassportCatalog {
    /// Creates an empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a passport to the catalog.
    pub fn add(&mut self, passport: AiServicePassport) {
        self.passports.push(passport);
    }

    /// Finds a passport by exact service identifier.
    pub fn find_by_service_id(&self, id: &str) -> Option<&AiServicePassport> {
        self.passports.iter().find(|p| p.service_id == id)
    }

    /// Returns all passports serving a given model identifier.
    pub fn find_by_model(&self, model_id: &str) -> Vec<&AiServicePassport> {
        self.passports
            .iter()
            .filter(|p| p.model.id == model_id)
            .collect()
    }

    /// Returns all passports with an energy rating at least as good as
    /// `min_rating` (where A is the best).
    pub fn find_by_rating(&self, min_rating: EnergyLabel) -> Vec<&AiServicePassport> {
        self.passports
            .iter()
            .filter(|p| p.energy_label.rating <= min_rating)
            .collect()
    }

    /// Returns passports sorted by efficiency ratio (most efficient first).
    pub fn leaderboard(&self) -> Vec<&AiServicePassport> {
        let mut sorted: Vec<&AiServicePassport> = self.passports.iter().collect();
        sorted.sort_by(|a, b| {
            a.energy_label
                .efficiency_ratio
                .partial_cmp(&b.energy_label.efficiency_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    // -- helpers ----------------------------------------------------------

    fn sample_model() -> ModelProfile {
        ModelProfile {
            id: "meta-llama/Llama-3.3-70B-Instruct".to_string(),
            family: "Llama 3.3".to_string(),
            parameters: 70_000_000_000,
            quantization: "FP8".to_string(),
            version: "1.0".to_string(),
        }
    }

    fn sample_hardware() -> HardwareProfile {
        HardwareProfile {
            accelerator: "NVIDIA H100 SXM".to_string(),
            cpu: "AMD EPYC 9654".to_string(),
            memory_gb: 80,
            interconnect: "NVLink 4.0".to_string(),
        }
    }

    fn sample_energy_profile() -> EnergyProfile {
        EnergyProfile {
            joules_per_input_token: 0.001,
            joules_per_output_token: 0.003,
            measurement_method: MeasurementMethod::NvmlRapl,
            sample_size: 10_000,
            benchmark_date: Utc::now(),
            idle_power_watts: 75.0,
            peak_power_watts: 700.0,
        }
    }

    fn sample_carbon_profile() -> CarbonProfile {
        CarbonProfile {
            sci_score_gco2_per_request: 0.42,
            grid_carbon_intensity_gco2_kwh: 45.0,
            carbon_intensity_source: "electricitymaps.com".to_string(),
            renewable_percentage: 92.0,
            embodied_carbon_gco2_per_request: 0.005,
            location: LocationInfo {
                region: "eu-west-1".to_string(),
                country: "IE".to_string(),
                grid_zone: "IE-SEM".to_string(),
            },
        }
    }

    fn sample_label() -> PassportLabel {
        PassportLabel {
            rating: EnergyLabel::A,
            methodology: "inv-energy v1.0".to_string(),
            baseline_model: "meta-llama/Llama-3.3-70B-Instruct-FP16".to_string(),
            efficiency_ratio: 0.35,
        }
    }

    fn sample_compliance() -> ComplianceRecord {
        ComplianceRecord {
            eu_ai_act: AiActCompliance {
                training_energy_disclosed: true,
                training_energy_kwh: Some(6_500_000.0),
                training_emissions_tco2: Some(2_290.0),
                inference_energy_measured: true,
            },
            csrd_scope_3: CsrdCompliance {
                verified: true,
                methodology: "GHG Protocol".to_string(),
                audit_trail: "https://invisible.dev/audit/2025-Q4".to_string(),
            },
            eed: EedCompliance {
                facility_pue: 1.08,
                facility_wue: 0.3,
                waste_heat_recovery_pct: 45.0,
                renewable_energy_factor: 0.92,
            },
            espr_dpp: EsprCompliance {
                passport_url: "https://invisible.dev/dpp/llama-3.3-70b".to_string(),
                registry_submitted: true,
            },
        }
    }

    fn sample_pricing() -> PricingInfo {
        PricingInfo {
            usdc_per_input_token: 0.0000015,
            usdc_per_output_token: 0.000006,
            usdc_per_joule: 0.0001,
            settlement_network: "eip155:8453".to_string(),
            x402_endpoint: "https://invisible.dev/x402".to_string(),
            free_tier_joules_per_day: 1000.0,
        }
    }

    fn build_sample_passport(
        service_id: &str,
        ratio: f64,
        rating: EnergyLabel,
    ) -> AiServicePassport {
        let now = Utc::now();
        let label = PassportLabel {
            rating,
            methodology: "inv-energy v1.0".to_string(),
            baseline_model: "baseline".to_string(),
            efficiency_ratio: ratio,
        };
        AiServicePassport::builder(service_id)
            .provider("invisible.dev")
            .issued_at(now)
            .valid_until(now + Duration::days(90))
            .model(sample_model())
            .hardware(sample_hardware())
            .energy_profile(sample_energy_profile())
            .carbon_profile(sample_carbon_profile())
            .energy_label(label)
            .compliance(sample_compliance())
            .pricing(sample_pricing())
            .build()
    }

    // -- tests ------------------------------------------------------------

    #[test]
    fn build_passport_via_builder() {
        let now = Utc::now();
        let passport = AiServicePassport::builder("invisible:llama-3.3-70b:base-eu-west")
            .passport_version("1.0")
            .provider("invisible.dev")
            .issued_at(now)
            .valid_until(now + Duration::days(90))
            .model(sample_model())
            .hardware(sample_hardware())
            .energy_profile(sample_energy_profile())
            .carbon_profile(sample_carbon_profile())
            .energy_label(sample_label())
            .compliance(sample_compliance())
            .pricing(sample_pricing())
            .build();

        assert_eq!(passport.passport_version, "1.0");
        assert_eq!(passport.service_id, "invisible:llama-3.3-70b:base-eu-west");
        assert_eq!(passport.provider, "invisible.dev");
        assert_eq!(passport.model.parameters, 70_000_000_000);
        assert_eq!(passport.hardware.accelerator, "NVIDIA H100 SXM");
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let passport = build_sample_passport("svc-roundtrip", 0.35, EnergyLabel::A);

        let json = serde_json::to_string(&passport).expect("serialize passport");
        let deserialized: AiServicePassport =
            serde_json::from_str(&json).expect("deserialize passport");

        assert_eq!(deserialized.service_id, "svc-roundtrip");
        assert_eq!(deserialized.model.id, passport.model.id);
        assert_eq!(deserialized.energy_label.rating, EnergyLabel::A);
        assert!((deserialized.energy_label.efficiency_ratio - 0.35).abs() < 1e-10);
        assert!(
            (deserialized.pricing.usdc_per_input_token - passport.pricing.usdc_per_input_token)
                .abs()
                < 1e-15
        );
    }

    #[test]
    fn measurement_method_display_and_serde() {
        let methods = [
            (MeasurementMethod::Rapl, "RAPL"),
            (MeasurementMethod::IoReport, "IOReport"),
            (MeasurementMethod::Nvml, "NVML"),
            (MeasurementMethod::NvmlRapl, "NVML+RAPL"),
            (MeasurementMethod::Estimated, "Estimated"),
        ];

        for (method, expected_display) in &methods {
            assert_eq!(format!("{}", method), *expected_display);

            let json = serde_json::to_string(method).expect("serialize method");
            let deserialized: MeasurementMethod =
                serde_json::from_str(&json).expect("deserialize method");
            assert_eq!(*method, deserialized);
        }
    }

    #[test]
    fn catalog_add_and_find_by_service_id() {
        let mut catalog = PassportCatalog::new();
        assert!(catalog.passports.is_empty());

        let p1 = build_sample_passport("svc-alpha", 0.30, EnergyLabel::A);
        let p2 = build_sample_passport("svc-beta", 0.60, EnergyLabel::B);
        catalog.add(p1);
        catalog.add(p2);

        assert_eq!(catalog.passports.len(), 2);

        let found = catalog.find_by_service_id("svc-alpha");
        assert!(found.is_some());
        assert_eq!(found.unwrap().service_id, "svc-alpha");

        assert!(catalog.find_by_service_id("svc-missing").is_none());
    }

    #[test]
    fn catalog_find_by_model() {
        let mut catalog = PassportCatalog::new();
        catalog.add(build_sample_passport("svc-1", 0.30, EnergyLabel::A));
        catalog.add(build_sample_passport("svc-2", 0.60, EnergyLabel::B));

        let results = catalog.find_by_model("meta-llama/Llama-3.3-70B-Instruct");
        assert_eq!(results.len(), 2);

        let empty = catalog.find_by_model("nonexistent-model");
        assert!(empty.is_empty());
    }

    #[test]
    fn catalog_find_by_rating() {
        let mut catalog = PassportCatalog::new();
        catalog.add(build_sample_passport("svc-a", 0.30, EnergyLabel::A));
        catalog.add(build_sample_passport("svc-b", 0.60, EnergyLabel::B));
        catalog.add(build_sample_passport("svc-d", 1.00, EnergyLabel::D));
        catalog.add(build_sample_passport("svc-g", 2.00, EnergyLabel::G));

        // min_rating = C means A, B, C are acceptable (rating <= C)
        let efficient = catalog.find_by_rating(EnergyLabel::C);
        assert_eq!(efficient.len(), 2); // A and B

        let all = catalog.find_by_rating(EnergyLabel::G);
        assert_eq!(all.len(), 4);

        let best_only = catalog.find_by_rating(EnergyLabel::A);
        assert_eq!(best_only.len(), 1);
        assert_eq!(best_only[0].service_id, "svc-a");
    }

    #[test]
    fn catalog_leaderboard_ordering() {
        let mut catalog = PassportCatalog::new();
        catalog.add(build_sample_passport("svc-worst", 1.80, EnergyLabel::G));
        catalog.add(build_sample_passport("svc-best", 0.20, EnergyLabel::A));
        catalog.add(build_sample_passport("svc-mid", 0.85, EnergyLabel::C));

        let board = catalog.leaderboard();
        assert_eq!(board.len(), 3);
        assert_eq!(board[0].service_id, "svc-best");
        assert_eq!(board[1].service_id, "svc-mid");
        assert_eq!(board[2].service_id, "svc-worst");
    }

    #[test]
    fn passport_label_references_energy_label() {
        let label = PassportLabel {
            rating: EnergyLabel::B,
            methodology: "test".to_string(),
            baseline_model: "baseline".to_string(),
            efficiency_ratio: 0.65,
        };

        assert_eq!(label.rating, EnergyLabel::B);
        assert_eq!(format!("{}", label.rating), "B");
        assert_eq!(label.rating.description(), "Very efficient");
        assert!(EnergyLabel::A < label.rating);
        assert!(label.rating < EnergyLabel::C);
    }

    #[test]
    fn compliance_record_serialization() {
        let record = sample_compliance();
        let json = serde_json::to_string(&record).expect("serialize compliance");
        let deserialized: ComplianceRecord =
            serde_json::from_str(&json).expect("deserialize compliance");

        assert!(deserialized.eu_ai_act.training_energy_disclosed);
        assert_eq!(
            deserialized.eu_ai_act.training_energy_kwh,
            Some(6_500_000.0)
        );
        assert_eq!(
            deserialized.eu_ai_act.training_emissions_tco2,
            Some(2_290.0)
        );
        assert!(deserialized.eu_ai_act.inference_energy_measured);

        assert!(deserialized.csrd_scope_3.verified);
        assert_eq!(deserialized.csrd_scope_3.methodology, "GHG Protocol");

        assert!((deserialized.eed.facility_pue - 1.08).abs() < 1e-10);
        assert!((deserialized.eed.facility_wue - 0.3).abs() < 1e-10);

        assert!(deserialized.espr_dpp.registry_submitted);
    }

    #[test]
    fn default_pricing_info() {
        let pricing = PricingInfo::default();

        assert!((pricing.usdc_per_input_token - 0.0).abs() < 1e-15);
        assert!((pricing.usdc_per_output_token - 0.0).abs() < 1e-15);
        assert!((pricing.usdc_per_joule - 0.0).abs() < 1e-15);
        assert_eq!(pricing.settlement_network, "eip155:8453");
        assert!(pricing.x402_endpoint.is_empty());
        assert!((pricing.free_tier_joules_per_day - 0.0).abs() < 1e-15);
    }
}
