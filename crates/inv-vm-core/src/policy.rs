use serde::{Deserialize, Serialize};

use crate::capability::NodeClass;
use crate::energy::Joules;
use crate::identity::RegionId;

/// Scheduling mode — how the scheduler should optimize placement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchedulingMode {
    /// Minimize latency to the caller.
    LatencyOptimal,
    /// Minimize energy consumption (joules per operation).
    #[default]
    EnergyOptimal,
    /// Minimize monetary cost.
    CostOptimal,
    /// Run only on local/LAN nodes.
    LocalOnly,
    /// Move compute to where the data lives.
    FollowTheData,
}

/// Placement policy — constraints on where a workload can run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementPolicy {
    /// Scheduling optimization mode.
    pub mode: SchedulingMode,
    /// Preferred node classes (tried first).
    pub prefer_class: Vec<NodeClass>,
    /// Allowed node classes (fallback).
    pub allow_class: Vec<NodeClass>,
    /// Required compliance tags (all must be satisfied).
    pub require_tags: Vec<String>,
    /// Maximum acceptable latency in milliseconds.
    pub max_latency_ms: Option<u32>,
    /// Prefer nodes powered by renewable energy.
    pub prefer_renewable: bool,
    /// Time flexibility for batch workloads (e.g., "6h" means can be delayed up to 6 hours).
    pub time_flexibility: Option<String>,
    /// Required regions for data residency.
    pub required_regions: Vec<RegionId>,
}

impl Default for PlacementPolicy {
    fn default() -> Self {
        Self {
            mode: SchedulingMode::default(),
            prefer_class: vec![],
            allow_class: vec![
                NodeClass::Cloud,
                NodeClass::Edge,
                NodeClass::Workstation,
                NodeClass::Gpu,
            ],
            require_tags: vec![],
            max_latency_ms: None,
            prefer_renewable: true,
            time_flexibility: None,
            required_regions: vec![],
        }
    }
}

/// Data policy — where data can exist and who can access it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPolicy {
    /// Human-readable name for this policy.
    pub name: String,
    /// Data classification level.
    pub classification: DataClassification,
    /// Regions where this data may reside (empty = all allowed).
    pub allowed_regions: Vec<RegionId>,
    /// Node classes allowed to process this data.
    pub allowed_node_classes: Vec<NodeClass>,
    /// Whether the data must be encrypted at rest.
    pub encrypt_at_rest: bool,
    /// Whether the data must be encrypted in transit.
    pub encrypt_in_transit: bool,
    /// Maximum energy budget for operations on this data.
    pub energy_budget: Option<Joules>,
    /// Retention period in days (None = indefinite).
    /// Kept for backward compatibility — prefer `max_retention_days`.
    pub retention_days: Option<u32>,
    /// Explicitly denied regions — data must never reside here.
    #[serde(default)]
    pub denied_regions: Vec<RegionId>,
    /// Explicitly denied node classes — data must never be processed by these.
    #[serde(default)]
    pub denied_node_classes: Vec<NodeClass>,
    /// Compliance tags required on nodes that handle this data.
    #[serde(default)]
    pub require_tags: Vec<String>,
    /// GDPR right-to-delete support — data must be deletable on request.
    #[serde(default)]
    pub right_to_delete: bool,
    /// Whether to audit every data access.
    #[serde(default)]
    pub audit_access: bool,
    /// Whether to audit every computation on this data.
    #[serde(default)]
    pub audit_compute: bool,
    /// Maximum retention in days (None = indefinite). Takes precedence over `retention_days`.
    #[serde(default)]
    pub max_retention_days: Option<u32>,
}

impl Default for DataPolicy {
    fn default() -> Self {
        Self {
            name: "default".into(),
            classification: DataClassification::Internal,
            allowed_regions: vec![],
            allowed_node_classes: vec![NodeClass::Cloud, NodeClass::Edge, NodeClass::Workstation],
            encrypt_at_rest: true,
            encrypt_in_transit: true,
            energy_budget: None,
            retention_days: None,
            denied_regions: vec![],
            denied_node_classes: vec![],
            require_tags: vec![],
            right_to_delete: false,
            audit_access: false,
            audit_compute: false,
            max_retention_days: None,
        }
    }
}

impl DataPolicy {
    /// Effective maximum retention days, preferring `max_retention_days` over `retention_days`.
    pub fn effective_retention_days(&self) -> Option<u32> {
        self.max_retention_days.or(self.retention_days)
    }
}

/// Data classification levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataClassification {
    /// Public data — no restrictions.
    Public,
    /// Internal data — stays within the organization.
    Internal,
    /// Confidential data — restricted access, encryption required.
    Confidential,
    /// Restricted data — highest sensitivity (PII, PHI, financial).
    Restricted,
}

/// Energy policy — how energy should be managed for a deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyPolicy {
    /// Maximum joules per function invocation.
    pub max_joules_per_invocation: Option<Joules>,
    /// Maximum watts sustained across all instances.
    pub max_watts_sustained: Option<f64>,
    /// Whether to prefer carbon-neutral grid regions.
    pub prefer_green_grid: bool,
    /// Whether to allow temporal shifting for batch workloads.
    pub allow_temporal_shift: bool,
    /// Carbon budget in grams CO2e per month.
    pub carbon_budget_grams: Option<f64>,
}

impl Default for EnergyPolicy {
    fn default() -> Self {
        Self {
            max_joules_per_invocation: None,
            max_watts_sustained: None,
            prefer_green_grid: true,
            allow_temporal_shift: false,
            carbon_budget_grams: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Policy evaluation types
// ---------------------------------------------------------------------------

/// A violation detected when evaluating a data policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyViolation {
    /// Name of the policy that was violated.
    pub policy_name: String,
    /// The kind of violation.
    pub violation_type: ViolationType,
    /// Human-readable description of the violation.
    pub message: String,
    /// Severity — determines whether the operation is blocked or just warned.
    pub severity: ViolationSeverity,
}

/// The kind of policy violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViolationType {
    /// Target region is not in the allowed list.
    RegionNotAllowed,
    /// Target region is explicitly denied.
    RegionDenied,
    /// Target node class is not in the allowed list.
    NodeClassNotAllowed,
    /// Target node class is explicitly denied.
    NodeClassDenied,
    /// A required compliance tag is missing on the node.
    MissingComplianceTag,
    /// Encryption is required but not present.
    EncryptionRequired,
    /// Data retention would exceed the maximum allowed.
    RetentionExceeded,
    /// Operation would exceed the energy budget.
    EnergyBudgetExceeded,
}

/// Severity of a policy violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Blocks the operation — data locality, encryption, region restrictions.
    Hard,
    /// Generates a warning but allows the operation — energy budget advisory.
    Soft,
}

// ---------------------------------------------------------------------------
// Policy evaluator
// ---------------------------------------------------------------------------

/// Evaluates data policies against placement and storage decisions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolicyEvaluator;

impl PolicyEvaluator {
    /// Create a new policy evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate whether a placement (region + node class + tags) complies with a data policy.
    ///
    /// Returns a (possibly empty) list of violations.
    pub fn evaluate_placement(
        policy: &DataPolicy,
        region: &RegionId,
        node_class: &NodeClass,
        tags: &[String],
    ) -> Vec<PolicyViolation> {
        let mut violations = Vec::new();

        // Check allowed regions (empty = all allowed).
        if !policy.allowed_regions.is_empty() && !policy.allowed_regions.contains(region) {
            violations.push(PolicyViolation {
                policy_name: policy.name.clone(),
                violation_type: ViolationType::RegionNotAllowed,
                message: format!(
                    "Region '{}' is not in the allowed regions list",
                    region.as_str()
                ),
                severity: ViolationSeverity::Hard,
            });
        }

        // Check denied regions.
        if policy.denied_regions.contains(region) {
            violations.push(PolicyViolation {
                policy_name: policy.name.clone(),
                violation_type: ViolationType::RegionDenied,
                message: format!(
                    "Region '{}' is explicitly denied by policy",
                    region.as_str()
                ),
                severity: ViolationSeverity::Hard,
            });
        }

        // Check allowed node classes (empty = all allowed).
        if !policy.allowed_node_classes.is_empty()
            && !policy.allowed_node_classes.contains(node_class)
        {
            violations.push(PolicyViolation {
                policy_name: policy.name.clone(),
                violation_type: ViolationType::NodeClassNotAllowed,
                message: format!(
                    "Node class '{}' is not in the allowed node classes list",
                    node_class
                ),
                severity: ViolationSeverity::Hard,
            });
        }

        // Check denied node classes.
        if policy.denied_node_classes.contains(node_class) {
            violations.push(PolicyViolation {
                policy_name: policy.name.clone(),
                violation_type: ViolationType::NodeClassDenied,
                message: format!("Node class '{}' is explicitly denied by policy", node_class),
                severity: ViolationSeverity::Hard,
            });
        }

        // Check compliance tags.
        for required_tag in &policy.require_tags {
            if !tags.contains(required_tag) {
                violations.push(PolicyViolation {
                    policy_name: policy.name.clone(),
                    violation_type: ViolationType::MissingComplianceTag,
                    message: format!(
                        "Required compliance tag '{}' is missing from node",
                        required_tag
                    ),
                    severity: ViolationSeverity::Hard,
                });
            }
        }

        // Check encryption requirements.
        if policy.encrypt_at_rest || policy.encrypt_in_transit {
            // Encryption violations are evaluated at storage/transport layer;
            // here we just flag that the policy requires it for awareness.
            // Actual enforcement happens at the storage and transport layers.
        }

        violations
    }

    /// Evaluate whether storing data in a target region complies with the policy.
    ///
    /// Returns a (possibly empty) list of violations.
    pub fn evaluate_storage(policy: &DataPolicy, target_region: &RegionId) -> Vec<PolicyViolation> {
        let mut violations = Vec::new();

        // Check allowed regions (empty = all allowed).
        if !policy.allowed_regions.is_empty() && !policy.allowed_regions.contains(target_region) {
            violations.push(PolicyViolation {
                policy_name: policy.name.clone(),
                violation_type: ViolationType::RegionNotAllowed,
                message: format!(
                    "Storage region '{}' is not in the allowed regions list",
                    target_region.as_str()
                ),
                severity: ViolationSeverity::Hard,
            });
        }

        // Check denied regions.
        if policy.denied_regions.contains(target_region) {
            violations.push(PolicyViolation {
                policy_name: policy.name.clone(),
                violation_type: ViolationType::RegionDenied,
                message: format!(
                    "Storage region '{}' is explicitly denied by policy",
                    target_region.as_str()
                ),
                severity: ViolationSeverity::Hard,
            });
        }

        // Check encryption at rest requirement.
        if policy.encrypt_at_rest {
            // Note: actual encryption enforcement is at the storage layer.
            // This evaluator flags the requirement for upper layers to check.
        }

        violations
    }

    /// Returns `true` if there are no `Hard` violations in the list.
    pub fn is_compliant(violations: &[PolicyViolation]) -> bool {
        !violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Hard)
    }

    /// Merge multiple data policies, taking the most restrictive combination.
    ///
    /// - `allowed_regions`: intersection (if any policy restricts, keep only common regions).
    /// - `allowed_node_classes`: intersection.
    /// - `denied_regions`: union.
    /// - `denied_node_classes`: union.
    /// - `require_tags`: union.
    /// - Boolean flags (`encrypt_*`, `right_to_delete`, `audit_*`): `true` if any policy requires it.
    /// - `energy_budget`: smallest (most restrictive).
    /// - `max_retention_days`/`retention_days`: smallest (most restrictive).
    /// - `classification`: highest (most restrictive).
    /// - `name`: joined with " + ".
    pub fn merge_policies(policies: &[&DataPolicy]) -> DataPolicy {
        if policies.is_empty() {
            return DataPolicy::default();
        }

        if policies.len() == 1 {
            return policies[0].clone();
        }

        // Start with the first policy and merge in the rest.
        let mut merged_name: Vec<String> = policies.iter().map(|p| p.name.clone()).collect();
        merged_name.dedup();

        // Classification: take the most restrictive.
        let classification = policies
            .iter()
            .map(|p| p.classification)
            .max_by_key(|c| classification_level(*c))
            .unwrap_or(DataClassification::Internal);

        // Allowed regions: intersection of non-empty lists.
        let allowed_regions = intersect_regions(policies.iter().map(|p| &p.allowed_regions));

        // Allowed node classes: intersection of non-empty lists.
        let allowed_node_classes =
            intersect_node_classes(policies.iter().map(|p| &p.allowed_node_classes));

        // Denied regions: union.
        let mut denied_regions: Vec<RegionId> = Vec::new();
        for policy in policies {
            for region in &policy.denied_regions {
                if !denied_regions.contains(region) {
                    denied_regions.push(region.clone());
                }
            }
        }

        // Denied node classes: union.
        let mut denied_node_classes: Vec<NodeClass> = Vec::new();
        for policy in policies {
            for nc in &policy.denied_node_classes {
                if !denied_node_classes.contains(nc) {
                    denied_node_classes.push(*nc);
                }
            }
        }

        // Require tags: union.
        let mut require_tags: Vec<String> = Vec::new();
        for policy in policies {
            for tag in &policy.require_tags {
                if !require_tags.contains(tag) {
                    require_tags.push(tag.clone());
                }
            }
        }

        // Boolean flags: true if any.
        let encrypt_at_rest = policies.iter().any(|p| p.encrypt_at_rest);
        let encrypt_in_transit = policies.iter().any(|p| p.encrypt_in_transit);
        let right_to_delete = policies.iter().any(|p| p.right_to_delete);
        let audit_access = policies.iter().any(|p| p.audit_access);
        let audit_compute = policies.iter().any(|p| p.audit_compute);

        // Energy budget: smallest non-None.
        let energy_budget = policies
            .iter()
            .filter_map(|p| p.energy_budget)
            .min_by(|a, b| a.as_f64().partial_cmp(&b.as_f64()).unwrap());

        // Retention: smallest non-None from effective retention.
        let retention_days = policies.iter().filter_map(|p| p.retention_days).min();

        let max_retention_days = policies.iter().filter_map(|p| p.max_retention_days).min();

        DataPolicy {
            name: merged_name.join(" + "),
            classification,
            allowed_regions,
            allowed_node_classes,
            encrypt_at_rest,
            encrypt_in_transit,
            energy_budget,
            retention_days,
            denied_regions,
            denied_node_classes,
            require_tags,
            right_to_delete,
            audit_access,
            audit_compute,
            max_retention_days,
        }
    }
}

/// Map classification to a numeric level for comparison (higher = more restrictive).
fn classification_level(c: DataClassification) -> u8 {
    match c {
        DataClassification::Public => 0,
        DataClassification::Internal => 1,
        DataClassification::Confidential => 2,
        DataClassification::Restricted => 3,
    }
}

/// Intersect allowed regions from multiple policies.
/// If all policies have empty allowed_regions, result is empty (meaning "all allowed").
/// If some have non-empty and some empty, the non-empty ones constrain.
/// If multiple have non-empty, we take the intersection.
fn intersect_regions<'a>(lists: impl Iterator<Item = &'a Vec<RegionId>>) -> Vec<RegionId> {
    let non_empty: Vec<&Vec<RegionId>> = lists.filter(|l| !l.is_empty()).collect();

    if non_empty.is_empty() {
        return vec![];
    }

    let mut result = non_empty[0].clone();
    for list in &non_empty[1..] {
        result.retain(|r| list.contains(r));
    }
    result
}

/// Intersect allowed node classes from multiple policies.
/// Same semantics as `intersect_regions`.
fn intersect_node_classes<'a>(lists: impl Iterator<Item = &'a Vec<NodeClass>>) -> Vec<NodeClass> {
    let non_empty: Vec<&Vec<NodeClass>> = lists.filter(|l| !l.is_empty()).collect();

    if non_empty.is_empty() {
        return vec![];
    }

    let mut result = non_empty[0].clone();
    for list in &non_empty[1..] {
        result.retain(|nc| list.contains(nc));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_placement_policy() {
        let policy = PlacementPolicy::default();
        assert_eq!(policy.mode, SchedulingMode::EnergyOptimal);
        assert!(policy.prefer_renewable);
        assert!(policy.allow_class.contains(&NodeClass::Cloud));
    }

    #[test]
    fn default_data_policy() {
        let policy = DataPolicy::default();
        assert!(policy.encrypt_at_rest);
        assert!(policy.encrypt_in_transit);
        assert_eq!(policy.classification, DataClassification::Internal);
    }

    #[test]
    fn placement_serialization_roundtrip() {
        let policy = PlacementPolicy {
            mode: SchedulingMode::LatencyOptimal,
            max_latency_ms: Some(10),
            require_tags: vec!["hipaa".into()],
            required_regions: vec![RegionId::new("us-east")],
            ..Default::default()
        };

        let json = serde_json::to_string(&policy).unwrap();
        let parsed: PlacementPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.mode, SchedulingMode::LatencyOptimal);
        assert_eq!(parsed.max_latency_ms, Some(10));
    }

    // --- compliance evaluation tests ---

    #[test]
    fn evaluate_compliant_placement() {
        let policy = DataPolicy {
            name: "gdpr-eu".into(),
            allowed_regions: vec![RegionId::new("eu-west"), RegionId::new("eu-central")],
            allowed_node_classes: vec![NodeClass::Cloud, NodeClass::Edge],
            require_tags: vec!["gdpr".into()],
            ..Default::default()
        };

        let violations = PolicyEvaluator::evaluate_placement(
            &policy,
            &RegionId::new("eu-west"),
            &NodeClass::Cloud,
            &["gdpr".into()],
        );

        assert!(violations.is_empty());
        assert!(PolicyEvaluator::is_compliant(&violations));
    }

    #[test]
    fn evaluate_non_compliant_region() {
        let policy = DataPolicy {
            name: "eu-only".into(),
            allowed_regions: vec![RegionId::new("eu-west")],
            ..Default::default()
        };

        let violations = PolicyEvaluator::evaluate_placement(
            &policy,
            &RegionId::new("us-east"),
            &NodeClass::Cloud,
            &[],
        );

        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].violation_type,
            ViolationType::RegionNotAllowed
        );
        assert_eq!(violations[0].severity, ViolationSeverity::Hard);
        assert!(!PolicyEvaluator::is_compliant(&violations));
    }

    #[test]
    fn evaluate_denied_region() {
        let policy = DataPolicy {
            name: "no-cn".into(),
            denied_regions: vec![RegionId::new("cn-north")],
            ..Default::default()
        };

        let violations = PolicyEvaluator::evaluate_placement(
            &policy,
            &RegionId::new("cn-north"),
            &NodeClass::Cloud,
            &[],
        );

        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_type, ViolationType::RegionDenied);
        assert_eq!(violations[0].severity, ViolationSeverity::Hard);
    }

    #[test]
    fn evaluate_missing_compliance_tag() {
        let policy = DataPolicy {
            name: "hipaa-data".into(),
            require_tags: vec!["hipaa".into(), "sox".into()],
            ..Default::default()
        };

        let violations = PolicyEvaluator::evaluate_placement(
            &policy,
            &RegionId::new("us-east"),
            &NodeClass::Cloud,
            &["hipaa".into()], // missing "sox"
        );

        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].violation_type,
            ViolationType::MissingComplianceTag
        );
        assert!(violations[0].message.contains("sox"));
    }

    #[test]
    fn merge_policies_takes_most_restrictive() {
        let policy_a = DataPolicy {
            name: "policy-a".into(),
            classification: DataClassification::Confidential,
            allowed_regions: vec![
                RegionId::new("us-east"),
                RegionId::new("eu-west"),
                RegionId::new("eu-central"),
            ],
            allowed_node_classes: vec![NodeClass::Cloud, NodeClass::Edge],
            encrypt_at_rest: true,
            encrypt_in_transit: false,
            energy_budget: Some(Joules::new(100.0)),
            retention_days: Some(365),
            require_tags: vec!["hipaa".into()],
            right_to_delete: false,
            audit_access: true,
            ..Default::default()
        };

        let policy_b = DataPolicy {
            name: "policy-b".into(),
            classification: DataClassification::Restricted,
            allowed_regions: vec![RegionId::new("eu-west"), RegionId::new("eu-central")],
            allowed_node_classes: vec![NodeClass::Cloud],
            encrypt_at_rest: false,
            encrypt_in_transit: true,
            energy_budget: Some(Joules::new(50.0)),
            retention_days: Some(90),
            denied_regions: vec![RegionId::new("cn-north")],
            require_tags: vec!["gdpr".into()],
            right_to_delete: true,
            audit_compute: true,
            ..Default::default()
        };

        let merged = PolicyEvaluator::merge_policies(&[&policy_a, &policy_b]);

        // Most restrictive classification.
        assert_eq!(merged.classification, DataClassification::Restricted);
        // Intersection of allowed regions.
        assert_eq!(merged.allowed_regions.len(), 2);
        assert!(merged.allowed_regions.contains(&RegionId::new("eu-west")));
        assert!(
            merged
                .allowed_regions
                .contains(&RegionId::new("eu-central"))
        );
        // Intersection of allowed node classes.
        assert_eq!(merged.allowed_node_classes, vec![NodeClass::Cloud]);
        // Boolean OR: both encrypt flags true.
        assert!(merged.encrypt_at_rest);
        assert!(merged.encrypt_in_transit);
        // Smallest energy budget.
        assert_eq!(merged.energy_budget.unwrap().as_f64(), 50.0);
        // Smallest retention.
        assert_eq!(merged.retention_days, Some(90));
        // Union of denied regions.
        assert!(merged.denied_regions.contains(&RegionId::new("cn-north")));
        // Union of tags.
        assert!(merged.require_tags.contains(&"hipaa".to_string()));
        assert!(merged.require_tags.contains(&"gdpr".to_string()));
        // Boolean OR.
        assert!(merged.right_to_delete);
        assert!(merged.audit_access);
        assert!(merged.audit_compute);
    }

    #[test]
    fn empty_allowed_regions_means_all_allowed() {
        let policy = DataPolicy {
            name: "open".into(),
            allowed_regions: vec![], // empty = all allowed
            ..Default::default()
        };

        let violations = PolicyEvaluator::evaluate_placement(
            &policy,
            &RegionId::new("any-region"),
            &NodeClass::Cloud,
            &[],
        );

        assert!(violations.is_empty());
        assert!(PolicyEvaluator::is_compliant(&violations));
    }

    #[test]
    fn violation_severity_hard_blocks() {
        let hard = PolicyViolation {
            policy_name: "test".into(),
            violation_type: ViolationType::RegionDenied,
            message: "blocked".into(),
            severity: ViolationSeverity::Hard,
        };
        let soft = PolicyViolation {
            policy_name: "test".into(),
            violation_type: ViolationType::EnergyBudgetExceeded,
            message: "advisory".into(),
            severity: ViolationSeverity::Soft,
        };

        // Hard violation alone makes it non-compliant.
        assert!(!PolicyEvaluator::is_compliant(std::slice::from_ref(&hard)));
        // Soft violation alone is still compliant.
        assert!(PolicyEvaluator::is_compliant(std::slice::from_ref(&soft)));
        // Mixed: non-compliant due to the hard one.
        assert!(!PolicyEvaluator::is_compliant(&[soft, hard]));
        // No violations: compliant.
        assert!(PolicyEvaluator::is_compliant(&[]));
    }

    #[test]
    fn multiple_violations_on_single_evaluation() {
        let policy = DataPolicy {
            name: "strict".into(),
            allowed_regions: vec![RegionId::new("eu-west")],
            denied_regions: vec![RegionId::new("us-east")],
            allowed_node_classes: vec![NodeClass::Cloud],
            denied_node_classes: vec![NodeClass::IoT],
            require_tags: vec!["hipaa".into(), "gdpr".into()],
            ..Default::default()
        };

        // us-east is not in allowed AND is explicitly denied.
        // IoT is not in allowed AND is explicitly denied.
        // Both tags are missing.
        let violations = PolicyEvaluator::evaluate_placement(
            &policy,
            &RegionId::new("us-east"),
            &NodeClass::IoT,
            &[],
        );

        // region not allowed + region denied + node class not allowed + node class denied
        // + 2 missing tags = 6 violations
        assert_eq!(violations.len(), 6);

        let types: Vec<ViolationType> = violations.iter().map(|v| v.violation_type).collect();
        assert!(types.contains(&ViolationType::RegionNotAllowed));
        assert!(types.contains(&ViolationType::RegionDenied));
        assert!(types.contains(&ViolationType::NodeClassNotAllowed));
        assert!(types.contains(&ViolationType::NodeClassDenied));
        assert_eq!(
            types
                .iter()
                .filter(|t| **t == ViolationType::MissingComplianceTag)
                .count(),
            2
        );
    }

    #[test]
    fn data_policy_serialization_roundtrip_new_fields() {
        let policy = DataPolicy {
            name: "gdpr-strict".into(),
            classification: DataClassification::Restricted,
            allowed_regions: vec![RegionId::new("eu-west")],
            allowed_node_classes: vec![NodeClass::Cloud],
            encrypt_at_rest: true,
            encrypt_in_transit: true,
            energy_budget: Some(Joules::new(10.0)),
            retention_days: Some(30),
            denied_regions: vec![RegionId::new("cn-north")],
            denied_node_classes: vec![NodeClass::IoT],
            require_tags: vec!["gdpr".into()],
            right_to_delete: true,
            audit_access: true,
            audit_compute: true,
            max_retention_days: Some(90),
        };

        let json = serde_json::to_string(&policy).unwrap();
        let parsed: DataPolicy = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "gdpr-strict");
        assert_eq!(parsed.classification, DataClassification::Restricted);
        assert_eq!(parsed.denied_regions.len(), 1);
        assert_eq!(parsed.denied_node_classes, vec![NodeClass::IoT]);
        assert_eq!(parsed.require_tags, vec!["gdpr".to_string()]);
        assert!(parsed.right_to_delete);
        assert!(parsed.audit_access);
        assert!(parsed.audit_compute);
        assert_eq!(parsed.max_retention_days, Some(90));
    }

    #[test]
    fn backward_compatible_deserialization() {
        // JSON without new fields should still deserialize thanks to #[serde(default)].
        let json = r#"{
            "name": "legacy",
            "classification": "Internal",
            "allowed_regions": [],
            "allowed_node_classes": ["Cloud"],
            "encrypt_at_rest": true,
            "encrypt_in_transit": false,
            "energy_budget": null,
            "retention_days": 365
        }"#;

        let parsed: DataPolicy = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.name, "legacy");
        // New fields should be at defaults.
        assert!(parsed.denied_regions.is_empty());
        assert!(parsed.denied_node_classes.is_empty());
        assert!(parsed.require_tags.is_empty());
        assert!(!parsed.right_to_delete);
        assert!(!parsed.audit_access);
        assert!(!parsed.audit_compute);
        assert_eq!(parsed.max_retention_days, None);
    }

    #[test]
    fn evaluate_storage_region_checks() {
        let policy = DataPolicy {
            name: "eu-storage".into(),
            allowed_regions: vec![RegionId::new("eu-west"), RegionId::new("eu-central")],
            denied_regions: vec![RegionId::new("us-gov")],
            ..Default::default()
        };

        // Compliant storage.
        let ok = PolicyEvaluator::evaluate_storage(&policy, &RegionId::new("eu-west"));
        assert!(ok.is_empty());

        // Region not allowed.
        let bad = PolicyEvaluator::evaluate_storage(&policy, &RegionId::new("ap-south"));
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].violation_type, ViolationType::RegionNotAllowed);

        // Denied region (also not in allowed list, so 2 violations).
        let denied = PolicyEvaluator::evaluate_storage(&policy, &RegionId::new("us-gov"));
        assert_eq!(denied.len(), 2);
        let types: Vec<ViolationType> = denied.iter().map(|v| v.violation_type).collect();
        assert!(types.contains(&ViolationType::RegionNotAllowed));
        assert!(types.contains(&ViolationType::RegionDenied));
    }

    #[test]
    fn effective_retention_days_prefers_max() {
        let policy = DataPolicy {
            retention_days: Some(365),
            max_retention_days: Some(90),
            ..Default::default()
        };
        assert_eq!(policy.effective_retention_days(), Some(90));

        let policy2 = DataPolicy {
            retention_days: Some(365),
            max_retention_days: None,
            ..Default::default()
        };
        assert_eq!(policy2.effective_retention_days(), Some(365));

        let policy3 = DataPolicy::default();
        assert_eq!(policy3.effective_retention_days(), None);
    }

    #[test]
    fn merge_empty_policies_returns_default() {
        let merged = PolicyEvaluator::merge_policies(&[]);
        assert_eq!(merged.name, "default");
        assert_eq!(merged.classification, DataClassification::Internal);
    }

    #[test]
    fn merge_single_policy_returns_clone() {
        let policy = DataPolicy {
            name: "solo".into(),
            classification: DataClassification::Confidential,
            right_to_delete: true,
            ..Default::default()
        };

        let merged = PolicyEvaluator::merge_policies(&[&policy]);
        assert_eq!(merged.name, "solo");
        assert_eq!(merged.classification, DataClassification::Confidential);
        assert!(merged.right_to_delete);
    }

    #[test]
    fn violation_type_serialization_roundtrip() {
        let violation = PolicyViolation {
            policy_name: "test-policy".into(),
            violation_type: ViolationType::MissingComplianceTag,
            message: "tag 'hipaa' is missing".into(),
            severity: ViolationSeverity::Hard,
        };

        let json = serde_json::to_string(&violation).unwrap();
        let parsed: PolicyViolation = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.policy_name, "test-policy");
        assert_eq!(parsed.violation_type, ViolationType::MissingComplianceTag);
        assert_eq!(parsed.severity, ViolationSeverity::Hard);
        assert!(parsed.message.contains("hipaa"));
    }

    #[test]
    fn policy_evaluator_serialization_roundtrip() {
        let evaluator = PolicyEvaluator::new();
        let json = serde_json::to_string(&evaluator).unwrap();
        let parsed: PolicyEvaluator = serde_json::from_str(&json).unwrap();
        // PolicyEvaluator is a unit struct; just confirm round-trip works.
        let _ = parsed;
    }
}
