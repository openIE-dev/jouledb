use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::policy::DataPolicy;

/// Supported compliance frameworks.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceFramework {
    Gdpr,
    Hipaa,
    Sox,
    PciDss,
    Iso27001,
    Soc2,
    FedRamp,
    Custom(String),
}

impl std::fmt::Display for ComplianceFramework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gdpr => write!(f, "GDPR"),
            Self::Hipaa => write!(f, "HIPAA"),
            Self::Sox => write!(f, "SOX"),
            Self::PciDss => write!(f, "PCI-DSS"),
            Self::Iso27001 => write!(f, "ISO 27001"),
            Self::Soc2 => write!(f, "SOC 2"),
            Self::FedRamp => write!(f, "FedRAMP"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

/// Severity of a compliance control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ControlSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// A single compliance control (requirement).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceControl {
    pub id: String,
    pub framework: ComplianceFramework,
    pub name: String,
    pub description: String,
    pub severity: ControlSeverity,
}

/// Result status of a compliance check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Passing,
    Failing,
    NotApplicable,
    NeedsReview,
}

/// Result of evaluating a single compliance control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheckResult {
    pub control_id: String,
    pub framework: ComplianceFramework,
    pub status: ComplianceStatus,
    pub evidence: String,
    pub checked_at: u64,
}

/// A full compliance report for an organization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub org_id: String,
    pub generated_at: u64,
    pub frameworks: Vec<ComplianceFramework>,
    pub total_controls: usize,
    pub passing: usize,
    pub failing: usize,
    pub needs_review: usize,
    pub not_applicable: usize,
    pub results: Vec<ComplianceCheckResult>,
}

/// Evaluates policies against compliance frameworks.
pub struct ComplianceChecker {
    controls: Vec<ComplianceControl>,
}

impl ComplianceChecker {
    /// Create a new checker with built-in GDPR and HIPAA controls registered.
    pub fn new() -> Self {
        let mut checker = Self {
            controls: Vec::new(),
        };

        // GDPR controls
        checker.register_control(ComplianceControl {
            id: "gdpr-encrypt-rest".into(),
            framework: ComplianceFramework::Gdpr,
            name: "Encryption at Rest".into(),
            description: "All data must be encrypted at rest".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "gdpr-encrypt-transit".into(),
            framework: ComplianceFramework::Gdpr,
            name: "Encryption in Transit".into(),
            description: "All data must be encrypted in transit".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "gdpr-right-to-delete".into(),
            framework: ComplianceFramework::Gdpr,
            name: "Right to Erasure".into(),
            description: "All data must support right-to-delete requests".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "gdpr-data-residency".into(),
            framework: ComplianceFramework::Gdpr,
            name: "Data Residency Controls".into(),
            description: "All policies must specify allowed regions for data residency".into(),
            severity: ControlSeverity::High,
        });
        checker.register_control(ComplianceControl {
            id: "gdpr-audit-logging".into(),
            framework: ComplianceFramework::Gdpr,
            name: "Audit Logging".into(),
            description: "Audit logging must be enabled".into(),
            severity: ControlSeverity::High,
        });

        // HIPAA controls
        checker.register_control(ComplianceControl {
            id: "hipaa-encrypt-rest".into(),
            framework: ComplianceFramework::Hipaa,
            name: "PHI Encryption at Rest".into(),
            description: "All PHI must be encrypted at rest".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "hipaa-encrypt-transit".into(),
            framework: ComplianceFramework::Hipaa,
            name: "PHI Encryption in Transit".into(),
            description: "All PHI must be encrypted in transit".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "hipaa-audit-trail".into(),
            framework: ComplianceFramework::Hipaa,
            name: "Audit Trail".into(),
            description: "A complete audit trail must be maintained".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "hipaa-access-controls".into(),
            framework: ComplianceFramework::Hipaa,
            name: "Access Controls".into(),
            description: "All data access must be audited and controlled".into(),
            severity: ControlSeverity::High,
        });
        checker.register_control(ComplianceControl {
            id: "hipaa-retention".into(),
            framework: ComplianceFramework::Hipaa,
            name: "Data Retention".into(),
            description: "All policies must have a defined retention period".into(),
            severity: ControlSeverity::Medium,
        });

        // SOC 2 Type II controls
        checker.register_control(ComplianceControl {
            id: "soc2-encrypt-rest".into(),
            framework: ComplianceFramework::Soc2,
            name: "Data Encryption at Rest".into(),
            description: "All data must be encrypted at rest per SOC 2 CC6.1".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "soc2-encrypt-transit".into(),
            framework: ComplianceFramework::Soc2,
            name: "Data Encryption in Transit".into(),
            description: "All data must be encrypted in transit per SOC 2 CC6.1".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "soc2-access-controls".into(),
            framework: ComplianceFramework::Soc2,
            name: "Logical Access Controls".into(),
            description: "Role-based access controls must be enforced per SOC 2 CC6.1".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "soc2-audit-trail".into(),
            framework: ComplianceFramework::Soc2,
            name: "Audit Trail".into(),
            description: "Complete audit trail for all changes per SOC 2 CC7.2".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "soc2-change-management".into(),
            framework: ComplianceFramework::Soc2,
            name: "Change Management".into(),
            description: "Change management procedures with retention per SOC 2 CC8.1".into(),
            severity: ControlSeverity::High,
        });

        // FedRAMP controls
        checker.register_control(ComplianceControl {
            id: "fedramp-encrypt-rest".into(),
            framework: ComplianceFramework::FedRamp,
            name: "FIPS 140-3 Encryption at Rest".into(),
            description: "All data must be encrypted at rest with FIPS 140-3 modules".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "fedramp-encrypt-transit".into(),
            framework: ComplianceFramework::FedRamp,
            name: "FIPS 140-3 Encryption in Transit".into(),
            description: "All data must be encrypted in transit with FIPS 140-3 modules".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "fedramp-audit-trail".into(),
            framework: ComplianceFramework::FedRamp,
            name: "Continuous Monitoring Audit Trail".into(),
            description: "Continuous monitoring audit trail per FedRAMP AU-2".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "fedramp-access-controls".into(),
            framework: ComplianceFramework::FedRamp,
            name: "Access Control Enforcement".into(),
            description: "Access control enforcement per FedRAMP AC-2".into(),
            severity: ControlSeverity::Critical,
        });
        checker.register_control(ComplianceControl {
            id: "fedramp-data-residency".into(),
            framework: ComplianceFramework::FedRamp,
            name: "US Data Residency".into(),
            description: "All data must reside within US boundaries per FedRAMP requirements"
                .into(),
            severity: ControlSeverity::High,
        });

        checker
    }

    /// Add a custom control.
    pub fn register_control(&mut self, control: ComplianceControl) {
        self.controls.push(control);
    }

    /// Filter controls by framework.
    pub fn controls_for_framework(&self, fw: &ComplianceFramework) -> Vec<&ComplianceControl> {
        self.controls
            .iter()
            .filter(|c| &c.framework == fw)
            .collect()
    }

    /// Return all controls.
    pub fn all_controls(&self) -> &[ComplianceControl] {
        &self.controls
    }

    /// Run all controls for the given framework and generate a report.
    pub fn run_checks(
        &self,
        fw: &ComplianceFramework,
        org_id: &str,
        policies: &[DataPolicy],
        has_audit: bool,
    ) -> ComplianceReport {
        let fw_controls = self.controls_for_framework(fw);
        let total_controls = fw_controls.len();

        let results: Vec<ComplianceCheckResult> = fw_controls
            .iter()
            .map(|control| self.check_control(control, policies, has_audit))
            .collect();

        let passing = results
            .iter()
            .filter(|r| r.status == ComplianceStatus::Passing)
            .count();
        let failing = results
            .iter()
            .filter(|r| r.status == ComplianceStatus::Failing)
            .count();
        let needs_review = results
            .iter()
            .filter(|r| r.status == ComplianceStatus::NeedsReview)
            .count();
        let not_applicable = results
            .iter()
            .filter(|r| r.status == ComplianceStatus::NotApplicable)
            .count();

        let generated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        ComplianceReport {
            org_id: org_id.into(),
            generated_at,
            frameworks: vec![fw.clone()],
            total_controls,
            passing,
            failing,
            needs_review,
            not_applicable,
            results,
        }
    }

    /// Evaluate a single control against the given policies and audit state.
    fn check_control(
        &self,
        control: &ComplianceControl,
        policies: &[DataPolicy],
        has_audit: bool,
    ) -> ComplianceCheckResult {
        let checked_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let id = &control.id;

        let (status, evidence) = if id.contains("encrypt-rest") {
            check_all_policies_bool(policies, |p| p.encrypt_at_rest, "encrypt_at_rest")
        } else if id.contains("encrypt-transit") {
            check_all_policies_bool(policies, |p| p.encrypt_in_transit, "encrypt_in_transit")
        } else if id.contains("right-to-delete") {
            check_all_policies_bool(policies, |p| p.right_to_delete, "right_to_delete")
        } else if id.contains("data-residency") {
            check_all_policies_bool(
                policies,
                |p| !p.allowed_regions.is_empty(),
                "allowed_regions non-empty",
            )
        } else if id.contains("audit") {
            if has_audit {
                (ComplianceStatus::Passing, "Audit logging is enabled".into())
            } else {
                (
                    ComplianceStatus::Failing,
                    "Audit logging is not enabled".into(),
                )
            }
        } else if id.contains("access-controls") {
            check_all_policies_bool(policies, |p| p.audit_access, "audit_access")
        } else if id.contains("retention") || id.contains("change-management") {
            check_all_policies_bool(
                policies,
                |p| p.retention_days.is_some(),
                "retention_days set",
            )
        } else {
            (
                ComplianceStatus::NeedsReview,
                "No automated check available for this control".into(),
            )
        };

        ComplianceCheckResult {
            control_id: control.id.clone(),
            framework: control.framework.clone(),
            status,
            evidence,
            checked_at,
        }
    }
}

impl Default for ComplianceChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper: check that a boolean predicate holds for all policies.
/// Returns `NeedsReview` if there are no policies.
fn check_all_policies_bool(
    policies: &[DataPolicy],
    predicate: impl Fn(&DataPolicy) -> bool,
    field_name: &str,
) -> (ComplianceStatus, String) {
    if policies.is_empty() {
        return (
            ComplianceStatus::NeedsReview,
            "No policies configured".into(),
        );
    }

    if let Some(failing_policy) = policies.iter().find(|p| !predicate(p)) {
        (
            ComplianceStatus::Failing,
            format!(
                "Policy '{}' does not satisfy {field_name}",
                failing_policy.name
            ),
        )
    } else {
        (
            ComplianceStatus::Passing,
            format!("All {} policies satisfy {field_name}", policies.len()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::NodeClass;
    use crate::identity::RegionId;

    fn test_policy(
        name: &str,
        encrypt_at_rest: bool,
        encrypt_in_transit: bool,
        right_to_delete: bool,
    ) -> DataPolicy {
        DataPolicy {
            name: name.into(),
            encrypt_at_rest,
            encrypt_in_transit,
            right_to_delete,
            allowed_regions: vec![RegionId::new("eu-west")],
            allowed_node_classes: vec![NodeClass::Cloud],
            audit_access: true,
            retention_days: Some(90),
            ..Default::default()
        }
    }

    #[test]
    fn new_checker_has_controls() {
        let checker = ComplianceChecker::new();
        let controls = checker.all_controls();
        // Should have GDPR and HIPAA controls.
        assert!(
            controls
                .iter()
                .any(|c| c.framework == ComplianceFramework::Gdpr)
        );
        assert!(
            controls
                .iter()
                .any(|c| c.framework == ComplianceFramework::Hipaa)
        );
        assert_eq!(controls.len(), 20);
    }

    #[test]
    fn controls_for_gdpr() {
        let checker = ComplianceChecker::new();
        let gdpr = checker.controls_for_framework(&ComplianceFramework::Gdpr);
        assert_eq!(gdpr.len(), 5);
    }

    #[test]
    fn controls_for_hipaa() {
        let checker = ComplianceChecker::new();
        let hipaa = checker.controls_for_framework(&ComplianceFramework::Hipaa);
        assert_eq!(hipaa.len(), 5);
    }

    #[test]
    fn controls_for_soc2() {
        let checker = ComplianceChecker::new();
        let soc2 = checker.controls_for_framework(&ComplianceFramework::Soc2);
        assert_eq!(soc2.len(), 5);
    }

    #[test]
    fn controls_for_fedramp() {
        let checker = ComplianceChecker::new();
        let fedramp = checker.controls_for_framework(&ComplianceFramework::FedRamp);
        assert_eq!(fedramp.len(), 5);
    }

    #[test]
    fn run_soc2_all_passing() {
        let checker = ComplianceChecker::new();
        let policies = vec![
            test_policy("p1", true, true, true),
            test_policy("p2", true, true, true),
        ];
        let report = checker.run_checks(&ComplianceFramework::Soc2, "org-1", &policies, true);
        assert_eq!(report.total_controls, 5);
        assert_eq!(report.passing, 5);
        assert_eq!(report.failing, 0);
    }

    #[test]
    fn run_fedramp_all_passing() {
        let checker = ComplianceChecker::new();
        let policies = vec![
            test_policy("p1", true, true, true),
            test_policy("p2", true, true, true),
        ];
        let report = checker.run_checks(&ComplianceFramework::FedRamp, "org-1", &policies, true);
        assert_eq!(report.total_controls, 5);
        assert_eq!(report.passing, 5);
        assert_eq!(report.failing, 0);
    }

    #[test]
    fn register_custom_control() {
        let mut checker = ComplianceChecker::new();
        let before = checker.all_controls().len();
        checker.register_control(ComplianceControl {
            id: "custom-check".into(),
            framework: ComplianceFramework::Custom("MyFramework".into()),
            name: "Custom Check".into(),
            description: "A custom compliance check".into(),
            severity: ControlSeverity::Low,
        });
        assert_eq!(checker.all_controls().len(), before + 1);
    }

    #[test]
    fn check_encrypt_at_rest_passing() {
        let checker = ComplianceChecker::new();
        let policies = vec![
            test_policy("p1", true, true, true),
            test_policy("p2", true, true, true),
        ];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-1", &policies, true);
        let result = report
            .results
            .iter()
            .find(|r| r.control_id == "gdpr-encrypt-rest")
            .unwrap();
        assert_eq!(result.status, ComplianceStatus::Passing);
    }

    #[test]
    fn check_encrypt_at_rest_failing() {
        let checker = ComplianceChecker::new();
        let policies = vec![
            test_policy("p1", true, true, true),
            test_policy("p2", false, true, true),
        ];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-1", &policies, true);
        let result = report
            .results
            .iter()
            .find(|r| r.control_id == "gdpr-encrypt-rest")
            .unwrap();
        assert_eq!(result.status, ComplianceStatus::Failing);
        assert!(result.evidence.contains("p2"));
    }

    #[test]
    fn check_encrypt_in_transit_passing() {
        let checker = ComplianceChecker::new();
        let policies = vec![test_policy("p1", true, true, true)];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-1", &policies, true);
        let result = report
            .results
            .iter()
            .find(|r| r.control_id == "gdpr-encrypt-transit")
            .unwrap();
        assert_eq!(result.status, ComplianceStatus::Passing);
    }

    #[test]
    fn check_right_to_delete_passing() {
        let checker = ComplianceChecker::new();
        let policies = vec![test_policy("p1", true, true, true)];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-1", &policies, true);
        let result = report
            .results
            .iter()
            .find(|r| r.control_id == "gdpr-right-to-delete")
            .unwrap();
        assert_eq!(result.status, ComplianceStatus::Passing);
    }

    #[test]
    fn check_right_to_delete_failing() {
        let checker = ComplianceChecker::new();
        let policies = vec![test_policy("p1", true, true, false)];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-1", &policies, true);
        let result = report
            .results
            .iter()
            .find(|r| r.control_id == "gdpr-right-to-delete")
            .unwrap();
        assert_eq!(result.status, ComplianceStatus::Failing);
    }

    #[test]
    fn check_data_residency_passing() {
        let checker = ComplianceChecker::new();
        let policies = vec![test_policy("p1", true, true, true)];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-1", &policies, true);
        let result = report
            .results
            .iter()
            .find(|r| r.control_id == "gdpr-data-residency")
            .unwrap();
        assert_eq!(result.status, ComplianceStatus::Passing);
    }

    #[test]
    fn check_audit_logging_passing() {
        let checker = ComplianceChecker::new();
        let policies = vec![test_policy("p1", true, true, true)];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-1", &policies, true);
        let result = report
            .results
            .iter()
            .find(|r| r.control_id == "gdpr-audit-logging")
            .unwrap();
        assert_eq!(result.status, ComplianceStatus::Passing);
    }

    #[test]
    fn check_audit_logging_failing() {
        let checker = ComplianceChecker::new();
        let policies = vec![test_policy("p1", true, true, true)];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-1", &policies, false);
        let result = report
            .results
            .iter()
            .find(|r| r.control_id == "gdpr-audit-logging")
            .unwrap();
        assert_eq!(result.status, ComplianceStatus::Failing);
    }

    #[test]
    fn run_gdpr_all_passing() {
        let checker = ComplianceChecker::new();
        let policies = vec![
            test_policy("p1", true, true, true),
            test_policy("p2", true, true, true),
        ];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-1", &policies, true);
        assert_eq!(report.total_controls, 5);
        assert_eq!(report.passing, 5);
        assert_eq!(report.failing, 0);
        assert_eq!(report.needs_review, 0);
        assert_eq!(report.not_applicable, 0);
        assert_eq!(report.org_id, "org-1");
        assert_eq!(report.frameworks, vec![ComplianceFramework::Gdpr]);
    }

    #[test]
    fn run_hipaa_mixed_results() {
        let checker = ComplianceChecker::new();
        // Policy with audit_access=false and no retention_days to cause failures.
        let mut policy = test_policy("mixed", true, true, true);
        policy.audit_access = false;
        policy.retention_days = None;
        let policies = vec![policy];
        let report = checker.run_checks(&ComplianceFramework::Hipaa, "org-2", &policies, true);
        assert_eq!(report.total_controls, 5);
        // encrypt-rest: Passing, encrypt-transit: Passing, audit-trail: Passing (has_audit=true)
        // access-controls: Failing (audit_access=false), retention: Failing (retention_days=None)
        assert_eq!(report.passing, 3);
        assert_eq!(report.failing, 2);
    }

    #[test]
    fn report_counts() {
        let checker = ComplianceChecker::new();
        // One policy that fails right-to-delete but passes everything else.
        let policies = vec![test_policy("partial", true, true, false)];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-3", &policies, false);
        // gdpr-encrypt-rest: Passing
        // gdpr-encrypt-transit: Passing
        // gdpr-right-to-delete: Failing
        // gdpr-data-residency: Passing
        // gdpr-audit-logging: Failing (has_audit=false)
        assert_eq!(report.passing, 3);
        assert_eq!(report.failing, 2);
        assert_eq!(report.needs_review, 0);
        assert_eq!(report.not_applicable, 0);
        assert_eq!(
            report.passing + report.failing + report.needs_review + report.not_applicable,
            report.total_controls
        );
    }

    #[test]
    fn empty_policies_needs_review() {
        let checker = ComplianceChecker::new();
        let policies: Vec<DataPolicy> = vec![];
        let report = checker.run_checks(&ComplianceFramework::Gdpr, "org-4", &policies, true);
        // Policy-based checks should be NeedsReview; audit check should pass.
        for result in &report.results {
            if result.control_id.contains("audit") {
                assert_eq!(result.status, ComplianceStatus::Passing);
            } else {
                assert_eq!(result.status, ComplianceStatus::NeedsReview);
                assert_eq!(result.evidence, "No policies configured");
            }
        }
        // 4 NeedsReview (policy-based) + 1 Passing (audit)
        assert_eq!(report.needs_review, 4);
        assert_eq!(report.passing, 1);
    }
}
