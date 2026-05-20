//! HDC-powered Insurance and Actuarial Analysis module
//!
//! Provides holographic encoding for:
//! - Claims similarity and fraud detection
//! - Risk assessment and underwriting
//! - Policy matching and recommendations
//! - Actuarial pattern recognition

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PolicyType {
    Auto,
    Home,
    Life,
    Health,
    Commercial,
    Liability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClaimStatus {
    Open,
    UnderReview,
    Approved,
    Denied,
    Paid,
    Appealed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskCategory {
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub id: String,
    pub policy_type: PolicyType,
    pub premium: f64,
    pub coverage_amount: f64,
    pub deductible: f64,
    pub start_date: u64,
    pub end_date: u64,
    pub holder_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub id: String,
    pub policy_id: String,
    pub claim_type: String,
    pub amount: f64,
    pub filed_date: u64,
    pub status: ClaimStatus,
    pub description: String,
    pub location: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskProfile {
    pub holder_id: String,
    pub age: u8,
    pub credit_score: u16,
    pub claims_history: Vec<String>,
    pub risk_factors: Vec<String>,
    pub risk_category: RiskCategory,
}

// ============================================================================
// Insurance Encoder
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// HDC encoder for insurance domain data
    pub struct InsuranceLink {
        seed: 0x1050_0001,
        dimension: 10000,
        fields: ["policy", "claim", "amount", "premium", "coverage", "deductible", "holder", "type", "risk"],
        scalars: ["amount", "premium", "age", "credit", "count"],
        enums: {
            policy_type_vectors: PolicyType => [PolicyType::Auto, PolicyType::Home, PolicyType::Life, PolicyType::Health, PolicyType::Commercial, PolicyType::Liability],
            status_vectors: ClaimStatus => [ClaimStatus::Open, ClaimStatus::UnderReview, ClaimStatus::Approved, ClaimStatus::Denied, ClaimStatus::Paid, ClaimStatus::Appealed],
            risk_vectors: RiskCategory => [RiskCategory::Low, RiskCategory::Medium, RiskCategory::High, RiskCategory::VeryHigh]
        },
        dynamic: {
            claim_type_vectors: "claim_type"
        },
    }
}

impl InsuranceLink {
    pub fn encode_policy(&self, policy: &Policy) -> BinaryHV {
        let type_hv =
            self.field_vectors["type"].bind(&self.policy_type_vectors[&policy.policy_type]);
        let premium_hv = self.field_vectors["premium"].bind(&self.encode_scalar(
            "premium",
            policy.premium as u32,
            50000,
        ));
        let coverage_hv = self.field_vectors["coverage"].bind(&self.encode_scalar(
            "amount",
            (policy.coverage_amount / 1000.0) as u32,
            10000,
        ));
        self.bundle(&[type_hv, premium_hv, coverage_hv])
    }

    pub fn encode_claim(&mut self, claim: &Claim) -> BinaryHV {
        let type_vec = self.claim_type_vectors(&claim.claim_type);
        let type_hv = self.field_vectors["type"].bind(&type_vec);
        let amount_hv = self.field_vectors["amount"].bind(&self.encode_scalar(
            "amount",
            claim.amount as u32,
            100000,
        ));
        let status_hv = self.field_vectors["claim"].bind(&self.status_vectors[&claim.status]);
        let desc_hv = BinaryHV::from_hash(claim.description.as_bytes(), DIMENSION);
        self.bundle(&[type_hv, amount_hv, status_hv, desc_hv])
    }

    pub fn encode_risk_profile(&self, profile: &RiskProfile) -> BinaryHV {
        let risk_hv = self.field_vectors["risk"].bind(&self.risk_vectors[&profile.risk_category]);
        let age_hv = self.encode_scalar("age", profile.age as u32, 100);
        let credit_hv = self.encode_scalar("credit", profile.credit_score as u32, 850);
        let history_hv = self.encode_scalar("count", profile.claims_history.len() as u32, 20);
        self.bundle(&[risk_hv, age_hv, credit_hv, history_hv])
    }
}

// ============================================================================
// Claims Database
// ============================================================================

pub struct ClaimsDb {
    encoder: InsuranceLink,
    claims_hologram: BundleAccumulator,
    claim_vectors: HashMap<String, BinaryHV>,
    claims: HashMap<String, Claim>,
}

impl ClaimsDb {
    pub fn new() -> Self {
        Self {
            encoder: InsuranceLink::new(),
            claims_hologram: BundleAccumulator::new(DIMENSION),
            claim_vectors: HashMap::new(),
            claims: HashMap::new(),
        }
    }

    pub fn add_claim(&mut self, claim: Claim) {
        let hv = self.encoder.encode_claim(&claim);
        self.claims_hologram.add(&hv);
        self.claim_vectors.insert(claim.id.clone(), hv);
        self.claims.insert(claim.id.clone(), claim);
    }

    pub fn find_similar(&self, claim_id: &str, min_sim: f32, limit: usize) -> Vec<(String, f32)> {
        let query = match self.claim_vectors.get(claim_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .claim_vectors
            .iter()
            .filter(|(id, _)| *id != claim_id)
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .filter(|(_, s)| *s >= min_sim)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }
}

impl Default for ClaimsDb {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Fraud Detector
// ============================================================================

pub struct FraudDetector {
    encoder: InsuranceLink,
    fraud_patterns: BundleAccumulator,
    legitimate_patterns: BundleAccumulator,
    threshold: f32,
}

#[derive(Debug, Clone)]
pub struct FraudAlert {
    pub claim_id: String,
    pub fraud_score: f32,
    pub risk_indicators: Vec<String>,
}

impl FraudDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: InsuranceLink::new(),
            fraud_patterns: BundleAccumulator::new(DIMENSION),
            legitimate_patterns: BundleAccumulator::new(DIMENSION),
            threshold,
        }
    }

    pub fn learn_fraud(&mut self, claim: &Claim) {
        self.fraud_patterns.add(&self.encoder.encode_claim(claim));
    }
    pub fn learn_legitimate(&mut self, claim: &Claim) {
        self.legitimate_patterns
            .add(&self.encoder.encode_claim(claim));
    }

    pub fn detect(&mut self, claim: &Claim) -> Option<FraudAlert> {
        let hv = self.encoder.encode_claim(claim);
        let fraud_sim = hv.similarity(&self.fraud_patterns.threshold());
        let legit_sim = hv.similarity(&self.legitimate_patterns.threshold());
        let score = fraud_sim - legit_sim;

        if score > self.threshold {
            Some(FraudAlert {
                claim_id: claim.id.clone(),
                fraud_score: score,
                risk_indicators: vec!["pattern_match".to_string()],
            })
        } else {
            None
        }
    }
}

impl Default for FraudDetector {
    fn default() -> Self {
        Self::new(0.3)
    }
}

// ============================================================================
// Risk Assessor
// ============================================================================

pub struct RiskAssessor {
    encoder: InsuranceLink,
    risk_patterns: HashMap<RiskCategory, BundleAccumulator>,
}

impl RiskAssessor {
    pub fn new() -> Self {
        let mut risk_patterns = HashMap::new();
        for cat in [
            RiskCategory::Low,
            RiskCategory::Medium,
            RiskCategory::High,
            RiskCategory::VeryHigh,
        ] {
            risk_patterns.insert(cat, BundleAccumulator::new(DIMENSION));
        }
        Self {
            encoder: InsuranceLink::new(),
            risk_patterns,
        }
    }

    pub fn train(&mut self, profile: &RiskProfile) {
        let hv = self.encoder.encode_risk_profile(profile);
        self.risk_patterns
            .get_mut(&profile.risk_category)
            .unwrap()
            .add(&hv);
    }

    pub fn assess(&self, profile: &RiskProfile) -> (RiskCategory, f32) {
        let hv = self.encoder.encode_risk_profile(profile);
        let mut best = (RiskCategory::Medium, 0.0f32);
        for (cat, pattern) in &self.risk_patterns {
            let sim = hv.similarity(&pattern.threshold());
            if sim > best.1 {
                best = (*cat, sim);
            }
        }
        best
    }
}

impl Default for RiskAssessor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_encoding() {
        let encoder = InsuranceLink::new();
        let policy = Policy {
            id: "P001".to_string(),
            policy_type: PolicyType::Auto,
            premium: 1200.0,
            coverage_amount: 50000.0,
            deductible: 500.0,
            start_date: 0,
            end_date: 365,
            holder_id: "H001".to_string(),
        };
        let hv = encoder.encode_policy(&policy);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_claim_encoding() {
        let mut encoder = InsuranceLink::new();
        let claim = Claim {
            id: "C001".to_string(),
            policy_id: "P001".to_string(),
            claim_type: "collision".to_string(),
            amount: 5000.0,
            filed_date: 100,
            status: ClaimStatus::Open,
            description: "Car accident".to_string(),
            location: None,
        };
        let hv = encoder.encode_claim(&claim);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_claims_db() {
        let mut db = ClaimsDb::new();
        db.add_claim(Claim {
            id: "C001".to_string(),
            policy_id: "P001".to_string(),
            claim_type: "collision".to_string(),
            amount: 5000.0,
            filed_date: 100,
            status: ClaimStatus::Open,
            description: "Car accident".to_string(),
            location: None,
        });
        assert_eq!(db.claim_count(), 1);
    }

    #[test]
    fn test_fraud_detection() {
        let mut detector = FraudDetector::new(0.3);
        let fraud_claim = Claim {
            id: "F001".to_string(),
            policy_id: "P001".to_string(),
            claim_type: "theft".to_string(),
            amount: 50000.0,
            filed_date: 100,
            status: ClaimStatus::Open,
            description: "Suspicious theft".to_string(),
            location: None,
        };
        detector.learn_fraud(&fraud_claim);
        // Test detection
        let result = detector.detect(&fraud_claim);
        assert!(result.is_some());
    }

    #[test]
    fn test_risk_assessment() {
        let mut assessor = RiskAssessor::new();
        let profile = RiskProfile {
            holder_id: "H001".to_string(),
            age: 35,
            credit_score: 720,
            claims_history: vec![],
            risk_factors: vec![],
            risk_category: RiskCategory::Low,
        };
        assessor.train(&profile);
        let (cat, _) = assessor.assess(&profile);
        assert_eq!(cat, RiskCategory::Low);
    }
}
