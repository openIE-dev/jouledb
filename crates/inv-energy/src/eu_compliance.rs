//! EU regulatory compliance reporting for energy and sustainability.
//!
//! Generates reports for EU Energy Efficiency Directive (EED),
//! CSRD Scope 3 emissions, AI Act energy disclosures, and
//! Ecodesign for Sustainable Products Regulation (ESPR).

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// EED Report
// ---------------------------------------------------------------------------

/// Report compliant with the EU Energy Efficiency Directive (EED).
///
/// Covers data-centre energy metrics including PUE, renewable energy share,
/// waste-heat recovery, and water usage for a given reporting period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EedReport {
    /// Reporting period identifier (e.g. "2025-Q1").
    pub reporting_period: String,
    /// Power Usage Effectiveness (total facility energy / IT equipment energy).
    pub pue: f64,
    /// Total energy consumed by the entire facility (kWh).
    pub total_energy_kwh: f64,
    /// Energy consumed by IT equipment only (kWh).
    pub it_energy_kwh: f64,
    /// Percentage of energy sourced from renewables (0 - 100).
    pub renewable_pct: f64,
    /// Percentage of waste heat that is recovered (0 - 100).
    pub waste_heat_recovery_pct: f64,
    /// Total water usage for cooling (litres).
    pub water_usage_liters: f64,
}

impl EedReport {
    /// Returns `true` when the report meets the EED compliance thresholds:
    /// PUE < 1.5 **and** renewable energy share >= 50%.
    pub fn is_compliant(&self) -> bool {
        self.pue < 1.5 && self.renewable_pct >= 50.0
    }
}

// ---------------------------------------------------------------------------
// CSRD Scope 3
// ---------------------------------------------------------------------------

/// Per-customer energy and emissions allocation for CSRD Scope 3 reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerAllocation {
    /// Opaque customer identifier.
    pub customer_id: String,
    /// Energy attributed to this customer (kWh).
    pub energy_kwh: f64,
    /// Emissions attributed to this customer (tonnes CO2).
    pub emissions_tco2: f64,
    /// Number of workloads run on behalf of this customer.
    pub workload_count: u32,
}

/// CSRD Scope 3 emissions report for an organisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrdScope3Report {
    /// Organisation name.
    pub organization: String,
    /// Reporting period (e.g. "2025-H1").
    pub reporting_period: String,
    /// Total downstream Scope 3 emissions (tonnes CO2).
    pub total_emissions_tco2: f64,
    /// Breakdown by customer.
    pub per_customer_allocations: Vec<CustomerAllocation>,
    /// Description of the allocation methodology used.
    pub methodology: String,
}

// ---------------------------------------------------------------------------
// AI Act
// ---------------------------------------------------------------------------

/// Energy and emissions disclosure for an AI model under the EU AI Act.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiActReport {
    /// Model name or identifier.
    pub model_name: String,
    /// Total energy consumed during model training (kWh).
    pub training_energy_kwh: f64,
    /// Total emissions from training (tonnes CO2).
    pub training_emissions_tco2: f64,
    /// Energy per single inference request (kWh).
    pub inference_energy_kwh_per_request: f64,
    /// Total number of inference requests served.
    pub total_inference_requests: u64,
    /// Free-text description of the hardware used.
    pub hardware_description: String,
}

impl AiActReport {
    /// Computes total inference energy (kWh) from per-request cost and
    /// request count.
    pub fn total_inference_energy_kwh(&self) -> f64 {
        self.inference_energy_kwh_per_request * self.total_inference_requests as f64
    }
}

// ---------------------------------------------------------------------------
// ESPR
// ---------------------------------------------------------------------------

/// Report compliant with the EU Ecodesign for Sustainable Products Regulation
/// (ESPR).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsprReport {
    /// Product name or model identifier.
    pub product_name: String,
    /// EU energy label grade (A through G).
    pub energy_label: String,
    /// Repairability score on a 0.0 to 10.0 scale.
    pub repairability_score: f64,
    /// Expected product lifetime in years.
    pub expected_lifetime_years: f64,
    /// Percentage of materials that are recyclable (0 - 100).
    pub recyclable_materials_pct: f64,
}

// ---------------------------------------------------------------------------
// Report generators
// ---------------------------------------------------------------------------

/// Builds an [`EedReport`] from raw facility metrics.
pub fn generate_eed_report(
    period: &str,
    total_kwh: f64,
    it_kwh: f64,
    renewable_pct: f64,
    waste_heat_pct: f64,
    water_liters: f64,
) -> EedReport {
    let pue = if it_kwh > 0.0 {
        total_kwh / it_kwh
    } else {
        0.0
    };

    EedReport {
        reporting_period: period.to_string(),
        pue,
        total_energy_kwh: total_kwh,
        it_energy_kwh: it_kwh,
        renewable_pct,
        waste_heat_recovery_pct: waste_heat_pct,
        water_usage_liters: water_liters,
    }
}

/// Builds a [`CsrdScope3Report`] from per-customer allocations.
///
/// `total_emissions_tco2` is derived by summing the allocations.
pub fn generate_csrd_scope3(
    org: &str,
    period: &str,
    allocations: Vec<CustomerAllocation>,
    methodology: &str,
) -> CsrdScope3Report {
    let total_emissions_tco2: f64 = allocations.iter().map(|a| a.emissions_tco2).sum();

    CsrdScope3Report {
        organization: org.to_string(),
        reporting_period: period.to_string(),
        total_emissions_tco2,
        per_customer_allocations: allocations,
        methodology: methodology.to_string(),
    }
}

/// Builds an [`AiActReport`] from training and inference metrics.
pub fn generate_ai_act_report(
    model: &str,
    training_kwh: f64,
    training_tco2: f64,
    inference_kwh_per_req: f64,
    total_requests: u64,
    hardware: &str,
) -> AiActReport {
    AiActReport {
        model_name: model.to_string(),
        training_energy_kwh: training_kwh,
        training_emissions_tco2: training_tco2,
        inference_energy_kwh_per_request: inference_kwh_per_req,
        total_inference_requests: total_requests,
        hardware_description: hardware.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- EED ----------------------------------------------------------------

    #[test]
    fn eed_report_compliant() {
        let report = EedReport {
            reporting_period: "2025-Q1".to_string(),
            pue: 1.2,
            total_energy_kwh: 120_000.0,
            it_energy_kwh: 100_000.0,
            renewable_pct: 80.0,
            waste_heat_recovery_pct: 30.0,
            water_usage_liters: 500_000.0,
        };
        assert!(report.is_compliant());
    }

    #[test]
    fn eed_report_non_compliant_pue() {
        let report = EedReport {
            reporting_period: "2025-Q2".to_string(),
            pue: 1.8,
            total_energy_kwh: 180_000.0,
            it_energy_kwh: 100_000.0,
            renewable_pct: 60.0,
            waste_heat_recovery_pct: 10.0,
            water_usage_liters: 800_000.0,
        };
        assert!(!report.is_compliant());
    }

    #[test]
    fn eed_report_non_compliant_renewable() {
        let report = EedReport {
            reporting_period: "2025-Q3".to_string(),
            pue: 1.3,
            total_energy_kwh: 130_000.0,
            it_energy_kwh: 100_000.0,
            renewable_pct: 40.0,
            waste_heat_recovery_pct: 20.0,
            water_usage_liters: 600_000.0,
        };
        assert!(!report.is_compliant());
    }

    // -- CSRD ---------------------------------------------------------------

    #[test]
    fn csrd_total_emissions() {
        let allocations = vec![
            CustomerAllocation {
                customer_id: "cust-1".to_string(),
                energy_kwh: 1000.0,
                emissions_tco2: 0.5,
                workload_count: 10,
            },
            CustomerAllocation {
                customer_id: "cust-2".to_string(),
                energy_kwh: 2000.0,
                emissions_tco2: 1.0,
                workload_count: 20,
            },
        ];
        let report = generate_csrd_scope3("AcmeCorp", "2025-H1", allocations, "energy-based");
        assert!((report.total_emissions_tco2 - 1.5).abs() < 1e-10);
    }

    #[test]
    fn csrd_empty_allocations() {
        let report = generate_csrd_scope3("AcmeCorp", "2025-H1", Vec::new(), "energy-based");
        assert!((report.total_emissions_tco2).abs() < 1e-10);
        assert!(report.per_customer_allocations.is_empty());
    }

    #[test]
    fn csrd_multiple_customers() {
        let allocations = vec![
            CustomerAllocation {
                customer_id: "a".to_string(),
                energy_kwh: 100.0,
                emissions_tco2: 0.1,
                workload_count: 5,
            },
            CustomerAllocation {
                customer_id: "b".to_string(),
                energy_kwh: 200.0,
                emissions_tco2: 0.2,
                workload_count: 8,
            },
            CustomerAllocation {
                customer_id: "c".to_string(),
                energy_kwh: 300.0,
                emissions_tco2: 0.3,
                workload_count: 12,
            },
        ];
        let report = generate_csrd_scope3("Org", "2025-Q4", allocations, "workload-based");
        assert_eq!(report.per_customer_allocations.len(), 3);
        assert!((report.total_emissions_tco2 - 0.6).abs() < 1e-10);
    }

    // -- AI Act -------------------------------------------------------------

    #[test]
    fn ai_act_total_inference_energy() {
        let report = AiActReport {
            model_name: "llm-v1".to_string(),
            training_energy_kwh: 50_000.0,
            training_emissions_tco2: 25.0,
            inference_energy_kwh_per_request: 0.001,
            total_inference_requests: 1_000_000,
            hardware_description: "8xA100".to_string(),
        };
        // 0.001 * 1_000_000 = 1000.0
        assert!((report.total_inference_energy_kwh() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn ai_act_zero_requests() {
        let report = AiActReport {
            model_name: "llm-v2".to_string(),
            training_energy_kwh: 10_000.0,
            training_emissions_tco2: 5.0,
            inference_energy_kwh_per_request: 0.002,
            total_inference_requests: 0,
            hardware_description: "4xH100".to_string(),
        };
        assert!((report.total_inference_energy_kwh()).abs() < 1e-10);
    }

    // -- ESPR ---------------------------------------------------------------

    #[test]
    fn espr_report_creation() {
        let report = EsprReport {
            product_name: "InvNode-1".to_string(),
            energy_label: "B".to_string(),
            repairability_score: 7.5,
            expected_lifetime_years: 6.0,
            recyclable_materials_pct: 85.0,
        };
        assert_eq!(report.product_name, "InvNode-1");
        assert_eq!(report.energy_label, "B");
        assert!((report.repairability_score - 7.5).abs() < 1e-10);
        assert!((report.expected_lifetime_years - 6.0).abs() < 1e-10);
        assert!((report.recyclable_materials_pct - 85.0).abs() < 1e-10);
    }

    // -- Serialization round-trips ------------------------------------------

    #[test]
    fn eed_serialization() {
        let report = EedReport {
            reporting_period: "2025-Q1".to_string(),
            pue: 1.3,
            total_energy_kwh: 130_000.0,
            it_energy_kwh: 100_000.0,
            renewable_pct: 70.0,
            waste_heat_recovery_pct: 25.0,
            water_usage_liters: 400_000.0,
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let d: EedReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.reporting_period, "2025-Q1");
        assert!((d.pue - 1.3).abs() < 1e-10);
    }

    #[test]
    fn csrd_serialization() {
        let report = CsrdScope3Report {
            organization: "Org".to_string(),
            reporting_period: "2025-H2".to_string(),
            total_emissions_tco2: 3.5,
            per_customer_allocations: vec![CustomerAllocation {
                customer_id: "c1".to_string(),
                energy_kwh: 500.0,
                emissions_tco2: 3.5,
                workload_count: 15,
            }],
            methodology: "energy-based".to_string(),
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let d: CsrdScope3Report = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.organization, "Org");
        assert_eq!(d.per_customer_allocations.len(), 1);
    }

    #[test]
    fn ai_act_serialization() {
        let report = generate_ai_act_report("model-x", 1000.0, 0.5, 0.0001, 500_000, "TPUv5");
        let json = serde_json::to_string(&report).expect("serialize");
        let d: AiActReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.model_name, "model-x");
        assert_eq!(d.total_inference_requests, 500_000);
    }

    #[test]
    fn espr_serialization() {
        let report = EsprReport {
            product_name: "Node-2".to_string(),
            energy_label: "A".to_string(),
            repairability_score: 9.0,
            expected_lifetime_years: 8.0,
            recyclable_materials_pct: 92.0,
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let d: EsprReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.product_name, "Node-2");
        assert_eq!(d.energy_label, "A");
    }

    // -- Generator functions ------------------------------------------------

    #[test]
    fn generate_eed_report_fn() {
        let report = generate_eed_report("2025-Q4", 150_000.0, 100_000.0, 75.0, 20.0, 300_000.0);
        assert_eq!(report.reporting_period, "2025-Q4");
        // PUE = 150_000 / 100_000 = 1.5
        assert!((report.pue - 1.5).abs() < 1e-10);
        assert!(report.renewable_pct >= 50.0);
        // pue is NOT < 1.5 (it equals 1.5), so not compliant
        assert!(!report.is_compliant());
    }

    #[test]
    fn generate_csrd_scope3_fn() {
        let allocations = vec![CustomerAllocation {
            customer_id: "cust-x".to_string(),
            energy_kwh: 750.0,
            emissions_tco2: 0.35,
            workload_count: 7,
        }];
        let report = generate_csrd_scope3("TestOrg", "2025-FY", allocations, "proportional");
        assert_eq!(report.organization, "TestOrg");
        assert_eq!(report.methodology, "proportional");
        assert!((report.total_emissions_tco2 - 0.35).abs() < 1e-10);
    }

    #[test]
    fn generate_ai_act_report_fn() {
        let report =
            generate_ai_act_report("gpt-test", 20_000.0, 10.0, 0.005, 100_000, "8xA100 SXM");
        assert_eq!(report.model_name, "gpt-test");
        assert!((report.training_energy_kwh - 20_000.0).abs() < 1e-10);
        // total inference = 0.005 * 100_000 = 500.0
        assert!((report.total_inference_energy_kwh() - 500.0).abs() < 1e-10);
    }
}
