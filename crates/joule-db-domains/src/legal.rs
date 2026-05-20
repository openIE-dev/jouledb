//! HDC-powered Legal and Compliance module
//!
//! Provides holographic encoding for:
//! - Contract similarity and clause extraction
//! - Regulatory compliance pattern matching
//! - Case law similarity search
//! - Risk assessment for legal documents

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentType {
    Contract,
    Agreement,
    Policy,
    Regulation,
    CaseLaw,
    Filing,
    Patent,
    Trademark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Jurisdiction {
    Federal,
    State,
    Local,
    International,
    EU,
    UK,
    APAC,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    Minimal,
    Low,
    Moderate,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClauseType {
    Indemnification,
    Limitation,
    Termination,
    Confidentiality,
    IP,
    Arbitration,
    ForceMAjeure,
    Warranty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalDocument {
    pub id: String,
    pub doc_type: DocumentType,
    pub title: String,
    pub content: String,
    pub jurisdiction: Jurisdiction,
    pub effective_date: u64,
    pub parties: Vec<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clause {
    pub id: String,
    pub doc_id: String,
    pub clause_type: ClauseType,
    pub text: String,
    pub position: usize,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regulation {
    pub id: String,
    pub name: String,
    pub jurisdiction: Jurisdiction,
    pub effective_date: u64,
    pub requirements: Vec<String>,
    pub penalties: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseLaw {
    pub id: String,
    pub citation: String,
    pub court: String,
    pub jurisdiction: Jurisdiction,
    pub date: u64,
    pub summary: String,
    pub holding: String,
    pub precedents: Vec<String>,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for legal domain data
    pub struct LegalLink {
        seed: 0x1E6A_0001,
        dimension: 10000,
        fields: ["document", "clause", "regulation", "case", "party", "tag", "content"],
        scalars: ["date", "position", "count", "value"],
        enums: {
            doc_type_vectors: DocumentType => [DocumentType::Contract, DocumentType::Agreement, DocumentType::Policy, DocumentType::Regulation, DocumentType::CaseLaw, DocumentType::Filing, DocumentType::Patent, DocumentType::Trademark],
            jurisdiction_vectors: Jurisdiction => [Jurisdiction::Federal, Jurisdiction::State, Jurisdiction::Local, Jurisdiction::International, Jurisdiction::EU, Jurisdiction::UK, Jurisdiction::APAC],
            risk_vectors: RiskLevel => [RiskLevel::Minimal, RiskLevel::Low, RiskLevel::Moderate, RiskLevel::High, RiskLevel::Critical],
            clause_type_vectors: ClauseType => [ClauseType::Indemnification, ClauseType::Limitation, ClauseType::Termination, ClauseType::Confidentiality, ClauseType::IP, ClauseType::Arbitration, ClauseType::ForceMAjeure, ClauseType::Warranty]
        },
    }
}

impl LegalLink {
    pub fn encode_document(&self, doc: &LegalDocument) -> BinaryHV {
        let type_hv = self.field_vectors["document"].bind(&self.doc_type_vectors[&doc.doc_type]);
        let jurisdiction_hv =
            self.field_vectors["document"].bind(&self.jurisdiction_vectors[&doc.jurisdiction]);
        let content_hv = BinaryHV::from_hash(doc.content.as_bytes(), DIMENSION);
        let title_hv = BinaryHV::from_hash(doc.title.as_bytes(), DIMENSION);
        let mut components = vec![type_hv, jurisdiction_hv, content_hv, title_hv];
        for party in &doc.parties {
            components.push(
                self.field_vectors["party"].bind(&BinaryHV::from_hash(party.as_bytes(), DIMENSION)),
            );
        }
        for tag in &doc.tags {
            components.push(
                self.field_vectors["tag"].bind(&BinaryHV::from_hash(tag.as_bytes(), DIMENSION)),
            );
        }
        self.bundle(&components)
    }

    pub fn encode_clause(&self, clause: &Clause) -> BinaryHV {
        let type_hv =
            self.field_vectors["clause"].bind(&self.clause_type_vectors[&clause.clause_type]);
        let risk_hv = self.risk_vectors[&clause.risk_level].clone();
        let text_hv = BinaryHV::from_hash(clause.text.as_bytes(), DIMENSION);
        let position_hv = self.encode_scalar("position", clause.position as u32, 1000);
        self.bundle(&[type_hv, risk_hv, text_hv, position_hv])
    }

    pub fn encode_regulation(&self, reg: &Regulation) -> BinaryHV {
        let jurisdiction_hv =
            self.field_vectors["regulation"].bind(&self.jurisdiction_vectors[&reg.jurisdiction]);
        let name_hv = BinaryHV::from_hash(reg.name.as_bytes(), DIMENSION);
        let mut components = vec![jurisdiction_hv, name_hv];
        for req in &reg.requirements {
            components.push(BinaryHV::from_hash(req.as_bytes(), DIMENSION));
        }
        self.bundle(&components)
    }

    pub fn encode_case_law(&self, case: &CaseLaw) -> BinaryHV {
        let jurisdiction_hv =
            self.field_vectors["case"].bind(&self.jurisdiction_vectors[&case.jurisdiction]);
        let summary_hv = BinaryHV::from_hash(case.summary.as_bytes(), DIMENSION);
        let holding_hv = BinaryHV::from_hash(case.holding.as_bytes(), DIMENSION);
        self.bundle(&[jurisdiction_hv, summary_hv, holding_hv])
    }
}

pub struct ContractDb {
    encoder: LegalLink,
    documents_hologram: BundleAccumulator,
    document_vectors: HashMap<String, BinaryHV>,
    documents: HashMap<String, LegalDocument>,
    clause_vectors: HashMap<String, BinaryHV>,
}

impl ContractDb {
    pub fn new() -> Self {
        Self {
            encoder: LegalLink::new(),
            documents_hologram: BundleAccumulator::new(DIMENSION),
            document_vectors: HashMap::new(),
            documents: HashMap::new(),
            clause_vectors: HashMap::new(),
        }
    }

    pub fn add_document(&mut self, doc: LegalDocument) {
        let hv = self.encoder.encode_document(&doc);
        self.documents_hologram.add(&hv);
        self.document_vectors.insert(doc.id.clone(), hv);
        self.documents.insert(doc.id.clone(), doc);
    }

    pub fn add_clause(&mut self, clause: &Clause) {
        let hv = self.encoder.encode_clause(clause);
        self.clause_vectors.insert(clause.id.clone(), hv);
    }

    pub fn find_similar_documents(
        &self,
        doc_id: &str,
        min_sim: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let query = match self.document_vectors.get(doc_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .document_vectors
            .iter()
            .filter(|(id, _)| *id != doc_id)
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .filter(|(_, s)| *s >= min_sim)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn find_similar_clauses(
        &self,
        clause_id: &str,
        min_sim: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let query = match self.clause_vectors.get(clause_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .clause_vectors
            .iter()
            .filter(|(id, _)| *id != clause_id)
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .filter(|(_, s)| *s >= min_sim)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }
}

impl Default for ContractDb {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ComplianceChecker {
    encoder: LegalLink,
    regulation_vectors: HashMap<String, BinaryHV>,
    regulations: HashMap<String, Regulation>,
    compliance_threshold: f32,
}

impl ComplianceChecker {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: LegalLink::new(),
            regulation_vectors: HashMap::new(),
            regulations: HashMap::new(),
            compliance_threshold: threshold,
        }
    }

    pub fn add_regulation(&mut self, reg: Regulation) {
        let hv = self.encoder.encode_regulation(&reg);
        self.regulation_vectors.insert(reg.id.clone(), hv);
        self.regulations.insert(reg.id.clone(), reg);
    }

    pub fn check_compliance(&self, doc: &LegalDocument) -> Vec<(String, f32, bool)> {
        let doc_hv = self.encoder.encode_document(doc);
        self.regulation_vectors
            .iter()
            .map(|(reg_id, reg_hv)| {
                let sim = doc_hv.similarity(reg_hv);
                let compliant = sim >= self.compliance_threshold;
                (reg_id.clone(), sim, compliant)
            })
            .collect()
    }

    pub fn regulation_count(&self) -> usize {
        self.regulations.len()
    }
}

impl Default for ComplianceChecker {
    fn default() -> Self {
        Self::new(0.7)
    }
}

pub struct CaseLawSearcher {
    encoder: LegalLink,
    case_vectors: HashMap<String, BinaryHV>,
    cases: HashMap<String, CaseLaw>,
}

impl CaseLawSearcher {
    pub fn new() -> Self {
        Self {
            encoder: LegalLink::new(),
            case_vectors: HashMap::new(),
            cases: HashMap::new(),
        }
    }

    pub fn add_case(&mut self, case: CaseLaw) {
        let hv = self.encoder.encode_case_law(&case);
        self.case_vectors.insert(case.id.clone(), hv);
        self.cases.insert(case.id.clone(), case);
    }

    pub fn find_precedents(&self, case: &CaseLaw, limit: usize) -> Vec<(String, f32)> {
        let query_hv = self.encoder.encode_case_law(case);
        let mut results: Vec<_> = self
            .case_vectors
            .iter()
            .filter(|(id, _)| *id != &case.id)
            .map(|(id, hv)| (id.clone(), query_hv.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn case_count(&self) -> usize {
        self.cases.len()
    }
}

impl Default for CaseLawSearcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_encoding() {
        let encoder = LegalLink::new();
        let doc = LegalDocument {
            id: "D1".to_string(),
            doc_type: DocumentType::Contract,
            title: "Service Agreement".to_string(),
            content: "This agreement governs...".to_string(),
            jurisdiction: Jurisdiction::State,
            effective_date: 0,
            parties: vec!["Party A".to_string()],
            tags: vec!["services".to_string()],
        };
        assert_eq!(encoder.encode_document(&doc).dimension(), DIMENSION);
    }

    #[test]
    fn test_clause_encoding() {
        let encoder = LegalLink::new();
        let clause = Clause {
            id: "CL1".to_string(),
            doc_id: "D1".to_string(),
            clause_type: ClauseType::Indemnification,
            text: "Party shall indemnify...".to_string(),
            position: 5,
            risk_level: RiskLevel::High,
        };
        assert_eq!(encoder.encode_clause(&clause).dimension(), DIMENSION);
    }

    #[test]
    fn test_contract_db() {
        let mut db = ContractDb::new();
        db.add_document(LegalDocument {
            id: "D1".to_string(),
            doc_type: DocumentType::Contract,
            title: "NDA".to_string(),
            content: "Confidentiality agreement".to_string(),
            jurisdiction: Jurisdiction::Federal,
            effective_date: 0,
            parties: vec![],
            tags: vec![],
        });
        assert_eq!(db.document_count(), 1);
    }

    #[test]
    fn test_compliance_checker() {
        let mut checker = ComplianceChecker::new(0.5);
        checker.add_regulation(Regulation {
            id: "R1".to_string(),
            name: "GDPR".to_string(),
            jurisdiction: Jurisdiction::EU,
            effective_date: 0,
            requirements: vec!["data protection".to_string()],
            penalties: vec![],
        });
        assert_eq!(checker.regulation_count(), 1);
    }

    #[test]
    fn test_case_law_searcher() {
        let mut searcher = CaseLawSearcher::new();
        searcher.add_case(CaseLaw {
            id: "C1".to_string(),
            citation: "123 F.3d 456".to_string(),
            court: "9th Cir".to_string(),
            jurisdiction: Jurisdiction::Federal,
            date: 0,
            summary: "Contract dispute".to_string(),
            holding: "Affirmed".to_string(),
            precedents: vec![],
        });
        assert_eq!(searcher.case_count(), 1);
    }
}
