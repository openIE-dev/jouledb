//! Position Limits — per-instrument limits, account-level limits, firm-level
//! aggregation, limit breach detection, exemption handling, and limit
//! utilisation reporting.
//!
//! Pure-Rust position-limit engine that tracks holdings across accounts and
//! instruments, enforces configurable limits, and reports utilisation with
//! exemption support.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum LimitError {
    InstrumentNotFound(String),
    AccountNotFound(String),
    LimitBreached(String),
    ExemptionExpired(String),
    InvalidLimit(String),
}

impl fmt::Display for LimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InstrumentNotFound(s) => write!(f, "instrument not found: {s}"),
            Self::AccountNotFound(s) => write!(f, "account not found: {s}"),
            Self::LimitBreached(s) => write!(f, "limit breached: {s}"),
            Self::ExemptionExpired(s) => write!(f, "exemption expired: {s}"),
            Self::InvalidLimit(s) => write!(f, "invalid limit: {s}"),
        }
    }
}

impl std::error::Error for LimitError {}

// ── Limit kind ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitKind {
    Instrument,
    Account,
    Firm,
}

impl fmt::Display for LimitKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Instrument => write!(f, "INSTRUMENT"),
            Self::Account => write!(f, "ACCOUNT"),
            Self::Firm => write!(f, "FIRM"),
        }
    }
}

// ── Breach severity ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BreachSeverity {
    Warning,
    SoftBreach,
    HardBreach,
}

impl fmt::Display for BreachSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Warning => write!(f, "WARNING"),
            Self::SoftBreach => write!(f, "SOFT_BREACH"),
            Self::HardBreach => write!(f, "HARD_BREACH"),
        }
    }
}

// ── Limit definition ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LimitDef {
    pub kind: LimitKind,
    pub key: String,
    pub max_long: f64,
    pub max_short: f64,
    pub warning_pct: f64,
    pub soft_breach_pct: f64,
}

impl LimitDef {
    pub fn new(kind: LimitKind, key: &str, max_long: f64, max_short: f64) -> Self {
        Self {
            kind,
            key: key.into(),
            max_long,
            max_short,
            warning_pct: 0.80,
            soft_breach_pct: 0.95,
        }
    }

    pub fn with_warning_pct(mut self, pct: f64) -> Self { self.warning_pct = pct; self }
    pub fn with_soft_breach_pct(mut self, pct: f64) -> Self { self.soft_breach_pct = pct; self }
}

impl fmt::Display for LimitDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Limit({} {} long={:.0} short={:.0})",
            self.kind, self.key, self.max_long, self.max_short,
        )
    }
}

// ── Exemption ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Exemption {
    pub limit_key: String,
    pub account_id: String,
    pub multiplier: f64,
    pub expiry_day: u64,
    pub reason: String,
}

impl Exemption {
    pub fn new(limit_key: &str, account_id: &str, multiplier: f64, expiry_day: u64, reason: &str) -> Self {
        Self {
            limit_key: limit_key.into(),
            account_id: account_id.into(),
            multiplier,
            expiry_day,
            reason: reason.into(),
        }
    }

    pub fn is_expired(&self, current_day: u64) -> bool {
        self.expiry_day < current_day
    }
}

impl fmt::Display for Exemption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Exemption({} acct={} mult={:.2} exp={})",
            self.limit_key, self.account_id, self.multiplier, self.expiry_day,
        )
    }
}

// ── Position ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Position {
    pub account_id: String,
    pub instrument: String,
    pub quantity: f64,
}

impl Position {
    pub fn new(account_id: &str, instrument: &str, quantity: f64) -> Self {
        Self { account_id: account_id.into(), instrument: instrument.into(), quantity }
    }

    pub fn is_long(&self) -> bool { self.quantity > 0.0 }
    pub fn is_short(&self) -> bool { self.quantity < 0.0 }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pos({} {} {:.2})", self.account_id, self.instrument, self.quantity)
    }
}

// ── Breach record ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Breach {
    pub limit_key: String,
    pub kind: LimitKind,
    pub severity: BreachSeverity,
    pub current_position: f64,
    pub limit_value: f64,
    pub utilisation_pct: f64,
    pub account_id: String,
}

impl fmt::Display for Breach {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Breach({} {} {} pos={:.0} lim={:.0} util={:.1}%)",
            self.limit_key, self.kind, self.severity,
            self.current_position, self.limit_value, self.utilisation_pct * 100.0,
        )
    }
}

// ── Utilisation report ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UtilisationReport {
    pub limit_key: String,
    pub kind: LimitKind,
    pub max_value: f64,
    pub current_value: f64,
    pub utilisation_pct: f64,
}

impl fmt::Display for UtilisationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Util({} {} cur={:.0}/{:.0} {:.1}%)",
            self.limit_key, self.kind, self.current_value,
            self.max_value, self.utilisation_pct * 100.0,
        )
    }
}

// ── Limit engine ────────────────────────────────────────────────

pub struct LimitEngine {
    limits: Vec<LimitDef>,
    exemptions: Vec<Exemption>,
    current_day: u64,
}

impl LimitEngine {
    pub fn new(current_day: u64) -> Self {
        Self { limits: Vec::new(), exemptions: Vec::new(), current_day }
    }

    pub fn add_limit(&mut self, limit: LimitDef) { self.limits.push(limit); }
    pub fn add_exemption(&mut self, ex: Exemption) { self.exemptions.push(ex); }
    pub fn limits(&self) -> &[LimitDef] { &self.limits }

    fn effective_limit(&self, limit: &LimitDef, account_id: &str) -> (f64, f64) {
        let mut long = limit.max_long;
        let mut short = limit.max_short;
        for ex in &self.exemptions {
            if ex.limit_key == limit.key
                && ex.account_id == account_id
                && !ex.is_expired(self.current_day)
            {
                long *= ex.multiplier;
                short *= ex.multiplier;
            }
        }
        (long, short)
    }

    /// Aggregate position per instrument across all accounts.
    fn aggregate_firm(&self, positions: &[Position]) -> HashMap<String, f64> {
        let mut agg: HashMap<String, f64> = HashMap::new();
        for p in positions {
            *agg.entry(p.instrument.clone()).or_default() += p.quantity;
        }
        agg
    }

    /// Aggregate position per account across all instruments.
    fn aggregate_account(&self, positions: &[Position]) -> HashMap<String, f64> {
        let mut agg: HashMap<String, f64> = HashMap::new();
        for p in positions {
            *agg.entry(p.account_id.clone()).or_default() += p.quantity;
        }
        agg
    }

    /// Check all limits against positions. Returns breaches.
    pub fn check_all(&self, positions: &[Position]) -> Vec<Breach> {
        let mut breaches = Vec::new();
        let firm_agg = self.aggregate_firm(positions);
        let acct_agg = self.aggregate_account(positions);

        for limit in &self.limits {
            match limit.kind {
                LimitKind::Instrument => {
                    for p in positions {
                        if p.instrument != limit.key { continue; }
                        let (max_l, max_s) = self.effective_limit(limit, &p.account_id);
                        self.check_one(p.quantity, max_l, max_s, limit, &p.account_id, &mut breaches);
                    }
                }
                LimitKind::Account => {
                    if let Some(&pos) = acct_agg.get(&limit.key) {
                        let (max_l, max_s) = self.effective_limit(limit, &limit.key);
                        self.check_one(pos, max_l, max_s, limit, &limit.key, &mut breaches);
                    }
                }
                LimitKind::Firm => {
                    if let Some(&pos) = firm_agg.get(&limit.key) {
                        let (max_l, max_s) = self.effective_limit(limit, "FIRM");
                        self.check_one(pos, max_l, max_s, limit, "FIRM", &mut breaches);
                    }
                }
            }
        }
        breaches
    }

    fn check_one(
        &self, pos: f64, max_long: f64, max_short: f64,
        limit: &LimitDef, account_id: &str, out: &mut Vec<Breach>,
    ) {
        let (effective, limit_val) = if pos >= 0.0 {
            (pos, max_long)
        } else {
            (pos.abs(), max_short)
        };
        if limit_val <= 0.0 { return; }
        let util = effective / limit_val;
        let severity = if util >= 1.0 {
            Some(BreachSeverity::HardBreach)
        } else if util >= limit.soft_breach_pct {
            Some(BreachSeverity::SoftBreach)
        } else if util >= limit.warning_pct {
            Some(BreachSeverity::Warning)
        } else {
            None
        };
        if let Some(sev) = severity {
            out.push(Breach {
                limit_key: limit.key.clone(),
                kind: limit.kind,
                severity: sev,
                current_position: pos,
                limit_value: limit_val,
                utilisation_pct: util,
                account_id: account_id.into(),
            });
        }
    }

    /// Generate utilisation report for all limits.
    pub fn utilisation_report(&self, positions: &[Position]) -> Vec<UtilisationReport> {
        let firm_agg = self.aggregate_firm(positions);
        let acct_agg = self.aggregate_account(positions);
        let mut reports = Vec::new();
        for limit in &self.limits {
            let current = match limit.kind {
                LimitKind::Instrument => {
                    positions.iter()
                        .filter(|p| p.instrument == limit.key)
                        .map(|p| p.quantity)
                        .sum::<f64>()
                }
                LimitKind::Account => *acct_agg.get(&limit.key).unwrap_or(&0.0),
                LimitKind::Firm => *firm_agg.get(&limit.key).unwrap_or(&0.0),
            };
            let max_val = if current >= 0.0 { limit.max_long } else { limit.max_short };
            let util = if max_val > 0.0 { current.abs() / max_val } else { 0.0 };
            reports.push(UtilisationReport {
                limit_key: limit.key.clone(),
                kind: limit.kind,
                max_value: max_val,
                current_value: current,
                utilisation_pct: util,
            });
        }
        reports
    }
}

impl fmt::Display for LimitEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "LimitEngine(limits={} exemptions={} day={})",
            self.limits.len(), self.exemptions.len(), self.current_day,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_with_limits() -> LimitEngine {
        let mut e = LimitEngine::new(100);
        e.add_limit(LimitDef::new(LimitKind::Instrument, "AAPL", 1000.0, 500.0));
        e.add_limit(LimitDef::new(LimitKind::Account, "ACCT1", 5000.0, 2000.0));
        e.add_limit(LimitDef::new(LimitKind::Firm, "AAPL", 10000.0, 5000.0));
        e
    }

    #[test]
    fn test_no_breach() {
        let e = engine_with_limits();
        let pos = vec![Position::new("ACCT1", "AAPL", 100.0)];
        let breaches = e.check_all(&pos);
        assert!(breaches.is_empty());
    }

    #[test]
    fn test_hard_breach_instrument() {
        let e = engine_with_limits();
        let pos = vec![Position::new("ACCT1", "AAPL", 1200.0)];
        let breaches = e.check_all(&pos);
        let inst_breach = breaches.iter().find(|b| b.kind == LimitKind::Instrument);
        assert!(inst_breach.is_some());
        assert_eq!(inst_breach.unwrap().severity, BreachSeverity::HardBreach);
    }

    #[test]
    fn test_warning_threshold() {
        let e = engine_with_limits();
        let pos = vec![Position::new("ACCT1", "AAPL", 850.0)];
        let breaches = e.check_all(&pos);
        let inst_breach = breaches.iter().find(|b| b.kind == LimitKind::Instrument);
        assert!(inst_breach.is_some());
        assert_eq!(inst_breach.unwrap().severity, BreachSeverity::Warning);
    }

    #[test]
    fn test_soft_breach() {
        let e = engine_with_limits();
        let pos = vec![Position::new("ACCT1", "AAPL", 960.0)];
        let breaches = e.check_all(&pos);
        let inst_breach = breaches.iter().find(|b| b.kind == LimitKind::Instrument);
        assert!(inst_breach.is_some());
        assert_eq!(inst_breach.unwrap().severity, BreachSeverity::SoftBreach);
    }

    #[test]
    fn test_short_breach() {
        let e = engine_with_limits();
        let pos = vec![Position::new("ACCT1", "AAPL", -600.0)];
        let breaches = e.check_all(&pos);
        assert!(!breaches.is_empty());
    }

    #[test]
    fn test_exemption_raises_limit() {
        let mut e = engine_with_limits();
        e.add_exemption(Exemption::new("AAPL", "ACCT1", 2.0, 200, "hedging"));
        let pos = vec![Position::new("ACCT1", "AAPL", 1200.0)];
        let breaches = e.check_all(&pos);
        let inst_breach = breaches.iter().find(|b| b.kind == LimitKind::Instrument);
        assert!(inst_breach.is_none());
    }

    #[test]
    fn test_expired_exemption() {
        let mut e = engine_with_limits();
        e.add_exemption(Exemption::new("AAPL", "ACCT1", 2.0, 50, "old"));
        let pos = vec![Position::new("ACCT1", "AAPL", 1200.0)];
        let breaches = e.check_all(&pos);
        assert!(!breaches.is_empty());
    }

    #[test]
    fn test_firm_level_aggregation() {
        let e = engine_with_limits();
        let pos = vec![
            Position::new("ACCT1", "AAPL", 4000.0),
            Position::new("ACCT2", "AAPL", 7000.0),
        ];
        let breaches = e.check_all(&pos);
        let firm_breach = breaches.iter().find(|b| b.kind == LimitKind::Firm);
        assert!(firm_breach.is_some());
        assert_eq!(firm_breach.unwrap().severity, BreachSeverity::HardBreach);
    }

    #[test]
    fn test_utilisation_report() {
        let e = engine_with_limits();
        let pos = vec![Position::new("ACCT1", "AAPL", 500.0)];
        let reports = e.utilisation_report(&pos);
        assert!(!reports.is_empty());
        let inst_r = reports.iter().find(|r| r.kind == LimitKind::Instrument).unwrap();
        assert!((inst_r.utilisation_pct - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_position_is_long_short() {
        assert!(Position::new("A", "X", 100.0).is_long());
        assert!(Position::new("A", "X", -100.0).is_short());
        assert!(!Position::new("A", "X", 0.0).is_long());
    }

    #[test]
    fn test_limit_def_builder() {
        let ld = LimitDef::new(LimitKind::Instrument, "AAPL", 1000.0, 500.0)
            .with_warning_pct(0.70)
            .with_soft_breach_pct(0.90);
        assert!((ld.warning_pct - 0.70).abs() < 1e-9);
        assert!((ld.soft_breach_pct - 0.90).abs() < 1e-9);
    }

    #[test]
    fn test_display_limit_def() {
        let ld = LimitDef::new(LimitKind::Instrument, "AAPL", 1000.0, 500.0);
        assert!(format!("{ld}").contains("AAPL"));
    }

    #[test]
    fn test_display_breach() {
        let b = Breach {
            limit_key: "AAPL".into(), kind: LimitKind::Instrument,
            severity: BreachSeverity::HardBreach, current_position: 1100.0,
            limit_value: 1000.0, utilisation_pct: 1.1, account_id: "A".into(),
        };
        assert!(format!("{b}").contains("HARD_BREACH"));
    }

    #[test]
    fn test_display_position() {
        let p = Position::new("ACCT1", "AAPL", 500.0);
        assert!(format!("{p}").contains("ACCT1"));
    }

    #[test]
    fn test_display_exemption() {
        let ex = Exemption::new("AAPL", "A", 2.0, 200, "hedge");
        assert!(format!("{ex}").contains("Exemption"));
    }

    #[test]
    fn test_display_engine() {
        let e = engine_with_limits();
        assert!(format!("{e}").contains("limits=3"));
    }

    #[test]
    fn test_display_utilisation() {
        let u = UtilisationReport {
            limit_key: "AAPL".into(), kind: LimitKind::Instrument,
            max_value: 1000.0, current_value: 500.0, utilisation_pct: 0.5,
        };
        assert!(format!("{u}").contains("50.0%"));
    }

    #[test]
    fn test_account_level_breach() {
        let e = engine_with_limits();
        let pos = vec![
            Position::new("ACCT1", "AAPL", 3000.0),
            Position::new("ACCT1", "GOOG", 3000.0),
        ];
        let breaches = e.check_all(&pos);
        let acct_breach = breaches.iter().find(|b| b.kind == LimitKind::Account);
        assert!(acct_breach.is_some());
    }

    #[test]
    fn test_exemption_is_expired() {
        let ex = Exemption::new("X", "A", 1.5, 50, "test");
        assert!(ex.is_expired(100));
        assert!(!ex.is_expired(40));
    }
}
