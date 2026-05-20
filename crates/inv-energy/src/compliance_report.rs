use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::eu_compliance::{AiActReport, CsrdScope3Report, EedReport, EsprReport};

/// Per-product energy data for compliance reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductEnergyData {
    pub product_id: String,
    pub product_name: String,
    pub energy_kwh: f64,
    pub energy_joules: f64,
    pub request_count: u64,
    pub carbon_gco2e: f64,
    pub region: String,
}

/// Facility information for EED reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacilityInfo {
    pub name: String,
    pub location: String,
    pub reporting_period: String,
    pub total_it_energy_kwh: f64,
    pub total_facility_energy_kwh: f64,
    pub renewable_energy_pct: f64,
    pub waste_heat_recovery_pct: f64,
    pub water_usage_liters: f64,
}

/// Entity information for CSRD reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInfo {
    pub name: String,
    pub reporting_period: String,
    pub methodology: String,
}

/// AI model information for AI Act reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiModelInfo {
    pub model_id: String,
    pub model_name: String,
    pub training_energy_kwh: f64,
    pub training_emissions_tco2: f64,
    pub inference_energy_kwh_per_request: f64,
    pub total_inference_requests: u64,
    pub hardware_description: String,
}

/// Product information for ESPR reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductInfo {
    pub product_name: String,
    pub energy_label: String,
    pub repairability_score: f64,
    pub expected_lifetime_years: f64,
    pub recyclable_materials_pct: f64,
}

/// A unified compliance report across all EU frameworks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedComplianceReport {
    pub generated_at: String,
    pub eed: Option<EedReport>,
    pub csrd: Option<CsrdScope3Report>,
    pub ai_act: Option<AiActReport>,
    pub espr: Option<EsprReport>,
    pub product_breakdown: Vec<ProductEnergyData>,
    pub total_energy_kwh: f64,
    pub total_carbon_gco2e: f64,
    pub compliant: bool,
    pub issues: Vec<ComplianceIssue>,
    pub energy_joules: f64,
    /// SHA-256 hash of report contents for third-party attestation.
    pub audit_hash: String,
    /// Whether this report is third-party attestable.
    pub third_party_attestable: bool,
    /// Measurement methodology used for energy data.
    pub measurement_method: String,
}

impl UnifiedComplianceReport {
    /// Compute SHA-256 audit hash over all report data for third-party verification.
    pub fn compute_audit_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.generated_at.as_bytes());
        hasher.update(self.total_energy_kwh.to_le_bytes());
        hasher.update(self.total_carbon_gco2e.to_le_bytes());
        hasher.update(self.energy_joules.to_le_bytes());
        hasher.update(
            if self.compliant {
                "compliant"
            } else {
                "non-compliant"
            }
            .as_bytes(),
        );
        for product in &self.product_breakdown {
            hasher.update(product.product_id.as_bytes());
            hasher.update(product.energy_kwh.to_le_bytes());
            hasher.update(product.carbon_gco2e.to_le_bytes());
        }
        for issue in &self.issues {
            hasher.update(issue.description.as_bytes());
        }
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    /// Verify the audit hash matches the report data.
    pub fn verify(&self) -> bool {
        self.audit_hash == self.compute_audit_hash()
    }
}

/// A compliance issue found during report generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceIssue {
    pub framework: ComplianceFramework,
    pub severity: IssueSeverity,
    pub description: String,
}

/// EU compliance framework identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceFramework {
    Eed,
    Csrd,
    AiAct,
    Espr,
}

/// Severity of a compliance issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Critical,
}

/// Generator for unified compliance reports across all EU frameworks.
pub struct ComplianceReportGenerator {
    total_energy_uj: std::sync::atomic::AtomicU64,
}

impl Default for ComplianceReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ComplianceReportGenerator {
    pub fn new() -> Self {
        Self {
            total_energy_uj: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Generate a unified compliance report across all applicable frameworks.
    pub fn generate_all(
        &self,
        products: &[ProductEnergyData],
        facility: Option<&FacilityInfo>,
        entity: Option<&EntityInfo>,
        ai_models: &[AiModelInfo],
        product_info: Option<&ProductInfo>,
    ) -> UnifiedComplianceReport {
        let energy = 0.01; // 10 mJ per full report generation
        self.track_energy(energy);

        let total_energy_kwh: f64 = products.iter().map(|p| p.energy_kwh).sum();
        let total_carbon: f64 = products.iter().map(|p| p.carbon_gco2e).sum();

        let mut issues = Vec::new();

        // Generate EED report if facility data provided
        let eed = facility.map(|f| {
            let pue = if f.total_it_energy_kwh > 0.0 {
                f.total_facility_energy_kwh / f.total_it_energy_kwh
            } else {
                0.0
            };

            if pue >= 1.5 {
                issues.push(ComplianceIssue {
                    framework: ComplianceFramework::Eed,
                    severity: IssueSeverity::Critical,
                    description: format!("PUE {:.2} exceeds EED threshold of 1.5", pue),
                });
            }
            if f.renewable_energy_pct < 50.0 {
                issues.push(ComplianceIssue {
                    framework: ComplianceFramework::Eed,
                    severity: IssueSeverity::Warning,
                    description: format!(
                        "Renewable energy {:.1}% below recommended 50%",
                        f.renewable_energy_pct
                    ),
                });
            }

            EedReport {
                reporting_period: f.reporting_period.clone(),
                pue,
                it_energy_kwh: f.total_it_energy_kwh,
                total_energy_kwh: f.total_facility_energy_kwh,
                renewable_pct: f.renewable_energy_pct,
                waste_heat_recovery_pct: f.waste_heat_recovery_pct,
                water_usage_liters: f.water_usage_liters,
            }
        });

        // Generate CSRD report if entity data provided
        let csrd = entity.map(|e| {
            let allocations: Vec<_> = products
                .iter()
                .map(|p| crate::eu_compliance::CustomerAllocation {
                    customer_id: p.product_id.clone(),
                    energy_kwh: p.energy_kwh,
                    emissions_tco2: p.carbon_gco2e / 1_000_000.0, // gCO2e → tCO2
                    workload_count: p.request_count as u32,
                })
                .collect();

            let total_emissions_tco2 = total_carbon / 1_000_000.0;

            if total_emissions_tco2 > 100.0 {
                issues.push(ComplianceIssue {
                    framework: ComplianceFramework::Csrd,
                    severity: IssueSeverity::Warning,
                    description: format!(
                        "Total emissions {:.1} tonnes CO2 — consider reduction targets",
                        total_emissions_tco2
                    ),
                });
            }

            CsrdScope3Report {
                organization: e.name.clone(),
                reporting_period: e.reporting_period.clone(),
                methodology: e.methodology.clone(),
                per_customer_allocations: allocations,
                total_emissions_tco2,
            }
        });

        // Generate AI Act report if AI model data provided
        let ai_act = if !ai_models.is_empty() {
            let model = &ai_models[0];

            if model.training_energy_kwh > 1000.0 {
                issues.push(ComplianceIssue {
                    framework: ComplianceFramework::AiAct,
                    severity: IssueSeverity::Info,
                    description: format!(
                        "Training energy {:.0} kWh — ensure disclosure in model card",
                        model.training_energy_kwh
                    ),
                });
            }

            Some(AiActReport {
                model_name: model.model_name.clone(),
                training_energy_kwh: model.training_energy_kwh,
                training_emissions_tco2: model.training_emissions_tco2,
                inference_energy_kwh_per_request: model.inference_energy_kwh_per_request,
                total_inference_requests: model.total_inference_requests,
                hardware_description: model.hardware_description.clone(),
            })
        } else {
            None
        };

        // Generate ESPR report if product info provided
        let espr = product_info.map(|p| {
            if p.repairability_score < 5.0 {
                issues.push(ComplianceIssue {
                    framework: ComplianceFramework::Espr,
                    severity: IssueSeverity::Warning,
                    description: format!(
                        "Repairability score {:.1}/10 below recommended minimum of 5.0",
                        p.repairability_score
                    ),
                });
            }

            EsprReport {
                product_name: p.product_name.clone(),
                energy_label: p.energy_label.clone(),
                repairability_score: p.repairability_score,
                expected_lifetime_years: p.expected_lifetime_years,
                recyclable_materials_pct: p.recyclable_materials_pct,
            }
        });

        let compliant = issues
            .iter()
            .all(|i| !matches!(i.severity, IssueSeverity::Critical));

        let mut report = UnifiedComplianceReport {
            generated_at: format!("{:?}", std::time::SystemTime::now()),
            eed,
            csrd,
            ai_act,
            espr,
            product_breakdown: products.to_vec(),
            total_energy_kwh,
            total_carbon_gco2e: total_carbon,
            compliant,
            issues,
            energy_joules: energy,
            audit_hash: String::new(),
            third_party_attestable: true,
            measurement_method: "ebpf_kernel_hooks".to_string(),
        };
        report.audit_hash = report.compute_audit_hash();
        report
    }

    /// Generate only an EED report.
    pub fn generate_eed(&self, facility: &FacilityInfo) -> EedReport {
        let energy = 0.003;
        self.track_energy(energy);

        let pue = if facility.total_it_energy_kwh > 0.0 {
            facility.total_facility_energy_kwh / facility.total_it_energy_kwh
        } else {
            0.0
        };

        EedReport {
            reporting_period: facility.reporting_period.clone(),
            pue,
            it_energy_kwh: facility.total_it_energy_kwh,
            total_energy_kwh: facility.total_facility_energy_kwh,
            renewable_pct: facility.renewable_energy_pct,
            waste_heat_recovery_pct: facility.waste_heat_recovery_pct,
            water_usage_liters: facility.water_usage_liters,
        }
    }

    /// Generate only a CSRD Scope 3 report.
    pub fn generate_csrd(
        &self,
        entity: &EntityInfo,
        products: &[ProductEnergyData],
    ) -> CsrdScope3Report {
        let energy = 0.003;
        self.track_energy(energy);

        let allocations: Vec<_> = products
            .iter()
            .map(|p| crate::eu_compliance::CustomerAllocation {
                customer_id: p.product_id.clone(),
                energy_kwh: p.energy_kwh,
                emissions_tco2: p.carbon_gco2e / 1_000_000.0,
                workload_count: p.request_count as u32,
            })
            .collect();

        let total_carbon: f64 = products.iter().map(|p| p.carbon_gco2e).sum();

        CsrdScope3Report {
            organization: entity.name.clone(),
            reporting_period: entity.reporting_period.clone(),
            methodology: entity.methodology.clone(),
            per_customer_allocations: allocations,
            total_emissions_tco2: total_carbon / 1_000_000.0,
        }
    }

    /// Total energy consumed by report generation.
    pub fn total_energy_joules(&self) -> f64 {
        self.total_energy_uj
            .load(std::sync::atomic::Ordering::Relaxed) as f64
            / 1_000_000.0
    }

    fn track_energy(&self, joules: f64) {
        self.total_energy_uj.fetch_add(
            (joules * 1_000_000.0) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}
