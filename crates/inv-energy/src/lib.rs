pub mod apple;
pub mod baselines;
pub mod battery;
pub mod bridge;
pub mod budget;
pub mod carbon;
pub mod carbon_aggregator;
pub mod carbon_api;
pub mod carbon_fusion;
pub mod cfe_matching;
pub mod compliance_report;
pub mod daily_report;
pub mod energy_label;
pub mod eu_compliance;
pub mod global_grid;
pub mod hf_energy_score;
pub mod hwdetect;
pub mod intermittent;
pub mod joule_meter;
pub mod ledger;
pub mod meter;
pub mod metrics;
pub mod nvml;
pub mod passport;
pub mod product_budget;
pub mod rapl;
pub mod rate_provider;
pub mod receipt;
pub mod sci;
pub mod ternary_energy;
pub mod wasm_energy;

#[cfg(test)]
mod proptest_carbon;
#[cfg(test)]
mod proptest_ledger;

pub use baselines::{
    EnergyBaseline, KnownModelEnergy, ModelSizeClass, baseline_for_class, classify_model_size,
    compute_energy_rating, known_model_energy,
};
pub use battery::BatteryState;
pub use bridge::MeshAccountant;
pub use budget::{BudgetEnforcer, BudgetError};
pub use carbon::{CarbonIntensity, ConnectionType, TransferCost};
pub use carbon_aggregator::{
    AggregatorConfig, CarbonAggregator, CarbonBackend, GridMix, IpccStaticBackend,
};
pub use carbon_api::{
    CarbonApiError, CarbonIntensityResponse, ElectricityMapsClient, ElectricityMapsConfig,
    RenewableEnergyResponse, cloud_zone_mapping,
};
pub use carbon_fusion::{
    CarbonForecast, CarbonFusionEngine, ForecastWindow, FusedEstimate, FusionConfig, GeoScope,
    Observation, ObservationStore, SourceKind, TemporalProfile, WeightConfig,
};
pub use cfe_matching::{CfeHour, CfeSchedule, CfeSource};
pub use compliance_report::{
    AiModelInfo, ComplianceFramework, ComplianceIssue, ComplianceReportGenerator, EntityInfo,
    FacilityInfo, IssueSeverity, ProductEnergyData, ProductInfo, UnifiedComplianceReport,
};
pub use daily_report::{DailyEnergyReport, ServiceEnergyRow, format_report, generate_daily_report};
pub use energy_label::{EnergyLabel, LabelReport, LabelThresholds};
pub use eu_compliance::{AiActReport, CsrdScope3Report, EedReport, EsprReport};
pub use global_grid::{
    CountryEnergyProfile, PROFILES as GLOBAL_ENERGY_PROFILES, cheap_energy_countries,
    composite_score, green_energy_countries, profile_by_country, profile_by_emaps_zone,
    profiles_by_carbon, profiles_by_cost,
};
pub use hf_energy_score::{
    HfBaselineComparison, HfEnergyScore, HfHardwareInfo, HfMethodology, HfModelInfo, all_hf_scores,
    hf_baseline_comparison, hf_score_for_known_model, hf_score_from_known_energy,
    hf_score_from_measurement, hf_score_from_sci,
};
pub use hwdetect::detect_hardware;
pub use intermittent::{
    Checkpoint, HarvestProfile, HarvestSource, HarvestWindow, IntermittentScheduler,
};
pub use joule_meter::{JouleMeter, OperationProfile};
pub use ledger::{
    CategoryBreakdown, EnergyLedger, LayerSnapshot, LedgerSnapshot, OperationalLayer,
};
pub use meter::{CompositeMeter, EnergyMeter, EnergyMeterError, EstimationMeter, detect_meter};
pub use passport::{
    AiActCompliance, AiServicePassport, CarbonProfile, ComplianceRecord, CsrdCompliance,
    EedCompliance, EnergyProfile, EsprCompliance, HardwareProfile, LocationInfo, MeasurementMethod,
    ModelProfile, PassportBuilder, PassportCatalog, PassportLabel, PricingInfo,
};
pub use product_budget::{
    AlertChannel, AlertThreshold, BudgetAlert, BudgetPeriod, BudgetStatus, ProductBudget,
    ProductBudgetManager,
};
pub use rate_provider::{EnergyZone, NodeEnergyRate};
pub use receipt::{EnergyReceipt, MeasurementSource, MemoryTier, SiliconType};
pub use sci::{SciConfig, SciError, SciMeasurements, SciScore};
pub use ternary_energy::{TernaryBackend, TernaryEnergyProfile};
pub use wasm_energy::{BudgetViolation, FunctionProfile, WasmEnergyBudget, WasmEnergyTracker};
