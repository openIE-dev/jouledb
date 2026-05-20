//! KYC Verification — identity matching (fuzzy name, DOB), sanctions list
//! screening (binary search), PEP detection, risk scoring, document expiry
//! tracking, and KycConfig builder.
//!
//! Pure-Rust KYC pipeline that evaluates customer identity data against
//! configurable sanctions lists, PEP databases, and risk-scoring heuristics
//! without any external crates.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum KycError {
    InvalidInput(String),
    SanctionsHit(String),
    DocumentExpired(String),
    HighRisk(String),
    ConfigError(String),
}

impl fmt::Display for KycError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::SanctionsHit(s) => write!(f, "sanctions hit: {s}"),
            Self::DocumentExpired(s) => write!(f, "document expired: {s}"),
            Self::HighRisk(s) => write!(f, "high risk: {s}"),
            Self::ConfigError(s) => write!(f, "config error: {s}"),
        }
    }
}

impl std::error::Error for KycError {}

// ── Risk level ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Prohibited,
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Prohibited => write!(f, "PROHIBITED"),
        }
    }
}

// ── Document type ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentType {
    Passport,
    DriverLicense,
    NationalId,
    UtilityBill,
    BankStatement,
}

impl fmt::Display for DocumentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Passport => write!(f, "PASSPORT"),
            Self::DriverLicense => write!(f, "DRIVER_LICENSE"),
            Self::NationalId => write!(f, "NATIONAL_ID"),
            Self::UtilityBill => write!(f, "UTILITY_BILL"),
            Self::BankStatement => write!(f, "BANK_STATEMENT"),
        }
    }
}

// ── Identity record ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IdentityRecord {
    pub first_name: String,
    pub last_name: String,
    pub dob: (u16, u8, u8),
    pub country: String,
    pub document_type: DocumentType,
    pub document_number: String,
    /// Expiry as days since epoch (simplification for std-only).
    pub document_expiry_days: u64,
    pub is_pep: bool,
}

impl fmt::Display for IdentityRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} (DOB {}-{:02}-{:02}, {} {})",
            self.first_name, self.last_name,
            self.dob.0, self.dob.1, self.dob.2,
            self.document_type, self.country,
        )
    }
}

// ── Sanctions entry ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SanctionsEntry {
    /// Normalised name key for binary search (uppercase, no spaces).
    pub name_key: String,
    pub program: String,
    pub entity_type: String,
}

impl fmt::Display for SanctionsEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}] ({})", self.name_key, self.program, self.entity_type)
    }
}

// ── KYC result ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KycResult {
    pub identity: String,
    pub risk_level: RiskLevel,
    pub risk_score: f64,
    pub sanctions_match: bool,
    pub pep_flag: bool,
    pub document_valid: bool,
    pub name_match_score: f64,
    pub notes: Vec<String>,
}

impl fmt::Display for KycResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KYC[{}] risk={} score={:.2} sanctions={} pep={} doc_valid={}",
            self.identity, self.risk_level, self.risk_score,
            self.sanctions_match, self.pep_flag, self.document_valid,
        )
    }
}

// ── KycConfig builder ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KycConfig {
    pub fuzzy_threshold: f64,
    pub high_risk_score: f64,
    pub medium_risk_score: f64,
    pub pep_risk_weight: f64,
    pub sanctions_risk_weight: f64,
    pub expired_doc_risk_weight: f64,
    pub high_risk_countries: Vec<String>,
    pub current_day: u64,
}

impl Default for KycConfig {
    fn default() -> Self {
        Self {
            fuzzy_threshold: 0.80,
            high_risk_score: 75.0,
            medium_risk_score: 40.0,
            pep_risk_weight: 25.0,
            sanctions_risk_weight: 100.0,
            expired_doc_risk_weight: 15.0,
            high_risk_countries: Vec::new(),
            current_day: 20000,
        }
    }
}

impl KycConfig {
    pub fn new() -> Self { Self::default() }

    pub fn with_fuzzy_threshold(mut self, t: f64) -> Self { self.fuzzy_threshold = t; self }
    pub fn with_high_risk_score(mut self, s: f64) -> Self { self.high_risk_score = s; self }
    pub fn with_medium_risk_score(mut self, s: f64) -> Self { self.medium_risk_score = s; self }
    pub fn with_pep_risk_weight(mut self, w: f64) -> Self { self.pep_risk_weight = w; self }
    pub fn with_sanctions_risk_weight(mut self, w: f64) -> Self { self.sanctions_risk_weight = w; self }
    pub fn with_expired_doc_risk_weight(mut self, w: f64) -> Self { self.expired_doc_risk_weight = w; self }
    pub fn with_high_risk_countries(mut self, c: Vec<String>) -> Self { self.high_risk_countries = c; self }
    pub fn with_current_day(mut self, d: u64) -> Self { self.current_day = d; self }
}

impl fmt::Display for KycConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KycConfig(fuzzy={:.2} high={:.0} med={:.0} countries={})",
            self.fuzzy_threshold, self.high_risk_score,
            self.medium_risk_score, self.high_risk_countries.len(),
        )
    }
}

// ── Fuzzy name matching ─────────────────────────────────────────

fn normalise_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// Jaro similarity between two strings, returned in [0.0, 1.0].
fn jaro_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let match_dist = (a_chars.len().max(b_chars.len()) / 2).saturating_sub(1);
    let mut a_matched = vec![false; a_chars.len()];
    let mut b_matched = vec![false; b_chars.len()];
    let mut matches = 0usize;
    for (i, &ac) in a_chars.iter().enumerate() {
        let lo = i.saturating_sub(match_dist);
        let hi = (i + match_dist + 1).min(b_chars.len());
        for j in lo..hi {
            if !b_matched[j] && b_chars[j] == ac {
                a_matched[i] = true;
                b_matched[j] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    let mut transpositions = 0usize;
    let mut k = 0;
    for (i, _) in a_chars.iter().enumerate().filter(|(i, _)| a_matched[*i]) {
        while !b_matched[k] { k += 1; }
        if a_chars[i] != b_chars[k] { transpositions += 1; }
        k += 1;
    }
    let m = matches as f64;
    let t = transpositions as f64 / 2.0;
    (m / a_chars.len() as f64 + m / b_chars.len() as f64 + (m - t) / m) / 3.0
}

/// Combined fuzzy match score for first+last name, in [0.0, 1.0].
pub fn fuzzy_name_score(first_a: &str, last_a: &str, first_b: &str, last_b: &str) -> f64 {
    let fa = normalise_name(first_a);
    let la = normalise_name(last_a);
    let fb = normalise_name(first_b);
    let lb = normalise_name(last_b);
    let first_score = jaro_similarity(&fa, &fb);
    let last_score = jaro_similarity(&la, &lb);
    first_score * 0.4 + last_score * 0.6
}

// ── Sanctions screening (binary search) ─────────────────────────

pub fn screen_sanctions(
    list: &[SanctionsEntry],
    first_name: &str,
    last_name: &str,
    threshold: f64,
) -> Vec<(usize, f64)> {
    let query = normalise_name(&format!("{first_name}{last_name}"));
    let mut hits = Vec::new();
    // Fast path: binary search for exact match.
    if let Ok(idx) = list.binary_search_by(|e| e.name_key.cmp(&query)) {
        hits.push((idx, 1.0));
        return hits;
    }
    // Slow path: linear scan with fuzzy matching.
    for (i, entry) in list.iter().enumerate() {
        let score = jaro_similarity(&entry.name_key, &query);
        if score >= threshold {
            hits.push((i, score));
        }
    }
    hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

// ── DOB matching ────────────────────────────────────────────────

pub fn dob_match(a: (u16, u8, u8), b: (u16, u8, u8)) -> bool {
    a.0 == b.0 && a.1 == b.1 && a.2 == b.2
}

// ── Document expiry ─────────────────────────────────────────────

pub fn is_document_expired(expiry_day: u64, current_day: u64) -> bool {
    expiry_day < current_day
}

pub fn days_until_expiry(expiry_day: u64, current_day: u64) -> i64 {
    expiry_day as i64 - current_day as i64
}

// ── KYC engine ──────────────────────────────────────────────────

pub struct KycEngine {
    config: KycConfig,
    sanctions_list: Vec<SanctionsEntry>,
}

impl KycEngine {
    pub fn new(config: KycConfig, mut sanctions_list: Vec<SanctionsEntry>) -> Self {
        sanctions_list.sort_by(|a, b| a.name_key.cmp(&b.name_key));
        Self { config, sanctions_list }
    }

    pub fn config(&self) -> &KycConfig { &self.config }
    pub fn sanctions_count(&self) -> usize { self.sanctions_list.len() }

    pub fn check(&self, record: &IdentityRecord) -> KycResult {
        let mut score: f64 = 0.0;
        let mut notes = Vec::new();

        // Sanctions screening.
        let hits = screen_sanctions(
            &self.sanctions_list,
            &record.first_name,
            &record.last_name,
            self.config.fuzzy_threshold,
        );
        let sanctions_match = !hits.is_empty();
        let best_name_score = hits.first().map(|h| h.1).unwrap_or(0.0);
        if sanctions_match {
            score += self.config.sanctions_risk_weight;
            notes.push(format!("sanctions hit (score={best_name_score:.2})"));
        }

        // PEP flag.
        if record.is_pep {
            score += self.config.pep_risk_weight;
            notes.push("PEP flagged".into());
        }

        // Document expiry.
        let doc_valid = !is_document_expired(record.document_expiry_days, self.config.current_day);
        if !doc_valid {
            score += self.config.expired_doc_risk_weight;
            notes.push("document expired".into());
        }

        // High-risk country.
        let country_upper = record.country.to_uppercase();
        if self.config.high_risk_countries.iter().any(|c| c.to_uppercase() == country_upper) {
            score += 20.0;
            notes.push(format!("high-risk country: {}", record.country));
        }

        let risk_level = if score >= self.config.sanctions_risk_weight {
            RiskLevel::Prohibited
        } else if score >= self.config.high_risk_score {
            RiskLevel::High
        } else if score >= self.config.medium_risk_score {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        KycResult {
            identity: format!("{} {}", record.first_name, record.last_name),
            risk_level,
            risk_score: score,
            sanctions_match,
            pep_flag: record.is_pep,
            document_valid: doc_valid,
            name_match_score: best_name_score,
            notes,
        }
    }

    /// Batch check multiple records.
    pub fn batch_check(&self, records: &[IdentityRecord]) -> Vec<KycResult> {
        records.iter().map(|r| self.check(r)).collect()
    }

    /// Returns utilization report: count by risk level.
    pub fn risk_distribution(&self, results: &[KycResult]) -> [(RiskLevel, usize); 4] {
        let mut counts = [0usize; 4];
        for r in results {
            match r.risk_level {
                RiskLevel::Low => counts[0] += 1,
                RiskLevel::Medium => counts[1] += 1,
                RiskLevel::High => counts[2] += 1,
                RiskLevel::Prohibited => counts[3] += 1,
            }
        }
        [
            (RiskLevel::Low, counts[0]),
            (RiskLevel::Medium, counts[1]),
            (RiskLevel::High, counts[2]),
            (RiskLevel::Prohibited, counts[3]),
        ]
    }
}

impl fmt::Display for KycEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KycEngine(sanctions={} {})", self.sanctions_list.len(), self.config)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> KycConfig {
        KycConfig::new()
            .with_high_risk_countries(vec!["XX".into(), "YY".into()])
            .with_current_day(20000)
    }

    fn sample_sanctions() -> Vec<SanctionsEntry> {
        vec![
            SanctionsEntry { name_key: "JOHNDOE".into(), program: "SDN".into(), entity_type: "Individual".into() },
            SanctionsEntry { name_key: "JANESMITH".into(), program: "EU".into(), entity_type: "Individual".into() },
        ]
    }

    fn sample_record() -> IdentityRecord {
        IdentityRecord {
            first_name: "Alice".into(),
            last_name: "Wonder".into(),
            dob: (1990, 5, 15),
            country: "US".into(),
            document_type: DocumentType::Passport,
            document_number: "X123456".into(),
            document_expiry_days: 21000,
            is_pep: false,
        }
    }

    #[test]
    fn test_normalise_name() {
        assert_eq!(normalise_name("John  Doe"), "JOHNDOE");
        assert_eq!(normalise_name("  "), "");
    }

    #[test]
    fn test_jaro_exact() {
        assert!((jaro_similarity("HELLO", "HELLO") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaro_empty() {
        assert!((jaro_similarity("", "") - 1.0).abs() < 1e-9);
        assert!((jaro_similarity("A", "")).abs() < 1e-9);
    }

    #[test]
    fn test_jaro_similar() {
        let s = jaro_similarity("MARTHA", "MARHTA");
        assert!(s > 0.9);
    }

    #[test]
    fn test_fuzzy_name_score_exact() {
        let s = fuzzy_name_score("John", "Doe", "John", "Doe");
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fuzzy_name_score_different() {
        let s = fuzzy_name_score("Alice", "Wonder", "Bob", "Builder");
        assert!(s < 0.7);
    }

    #[test]
    fn test_dob_match() {
        assert!(dob_match((1990, 5, 15), (1990, 5, 15)));
        assert!(!dob_match((1990, 5, 15), (1990, 5, 16)));
    }

    #[test]
    fn test_document_expired() {
        assert!(is_document_expired(19999, 20000));
        assert!(!is_document_expired(20001, 20000));
    }

    #[test]
    fn test_days_until_expiry() {
        assert_eq!(days_until_expiry(20100, 20000), 100);
        assert_eq!(days_until_expiry(19900, 20000), -100);
    }

    #[test]
    fn test_sanctions_exact_hit() {
        let list = sample_sanctions();
        let mut sorted = list.clone();
        sorted.sort();
        let hits = screen_sanctions(&sorted, "John", "Doe", 0.80);
        assert!(!hits.is_empty());
        assert!((hits[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sanctions_no_hit() {
        let list = sample_sanctions();
        let mut sorted = list.clone();
        sorted.sort();
        let hits = screen_sanctions(&sorted, "Alice", "Wonder", 0.95);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_kyc_clean_customer() {
        let engine = KycEngine::new(sample_config(), sample_sanctions());
        let result = engine.check(&sample_record());
        assert_eq!(result.risk_level, RiskLevel::Low);
        assert!(!result.sanctions_match);
        assert!(result.document_valid);
    }

    #[test]
    fn test_kyc_sanctions_hit() {
        let engine = KycEngine::new(sample_config(), sample_sanctions());
        let mut rec = sample_record();
        rec.first_name = "John".into();
        rec.last_name = "Doe".into();
        let result = engine.check(&rec);
        assert_eq!(result.risk_level, RiskLevel::Prohibited);
        assert!(result.sanctions_match);
    }

    #[test]
    fn test_kyc_pep_flag() {
        let engine = KycEngine::new(sample_config(), sample_sanctions());
        let mut rec = sample_record();
        rec.is_pep = true;
        let result = engine.check(&rec);
        assert!(result.pep_flag);
        assert!(result.risk_score >= 25.0);
    }

    #[test]
    fn test_kyc_expired_document() {
        let engine = KycEngine::new(sample_config(), sample_sanctions());
        let mut rec = sample_record();
        rec.document_expiry_days = 19000;
        let result = engine.check(&rec);
        assert!(!result.document_valid);
    }

    #[test]
    fn test_kyc_high_risk_country() {
        let engine = KycEngine::new(sample_config(), sample_sanctions());
        let mut rec = sample_record();
        rec.country = "XX".into();
        let result = engine.check(&rec);
        assert!(result.risk_score >= 20.0);
    }

    #[test]
    fn test_batch_check() {
        let engine = KycEngine::new(sample_config(), sample_sanctions());
        let records = vec![sample_record(), sample_record()];
        let results = engine.batch_check(&records);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_risk_distribution() {
        let engine = KycEngine::new(sample_config(), sample_sanctions());
        let results = engine.batch_check(&[sample_record()]);
        let dist = engine.risk_distribution(&results);
        assert_eq!(dist[0].1, 1); // 1 Low
    }

    #[test]
    fn test_config_builder() {
        let cfg = KycConfig::new()
            .with_fuzzy_threshold(0.90)
            .with_pep_risk_weight(50.0);
        assert!((cfg.fuzzy_threshold - 0.90).abs() < 1e-9);
        assert!((cfg.pep_risk_weight - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_impls() {
        let rec = sample_record();
        let s = format!("{rec}");
        assert!(s.contains("Alice"));

        let entry = SanctionsEntry { name_key: "FOO".into(), program: "X".into(), entity_type: "Y".into() };
        assert!(format!("{entry}").contains("FOO"));

        let engine = KycEngine::new(sample_config(), sample_sanctions());
        assert!(format!("{engine}").contains("KycEngine"));
    }
}
