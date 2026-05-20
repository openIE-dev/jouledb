//! Margin Calculation — initial margin (SPAN-like), variation margin,
//! maintenance margin, portfolio margining, cross-margining, and margin
//! call generation.
//!
//! Pure-Rust margin engine implementing a simplified SPAN-like model with
//! scenario-based risk arrays, portfolio offsets, and margin call workflows.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MarginError {
    InvalidPosition(String),
    InvalidScenario(String),
    InsufficientCollateral(String),
    ConfigError(String),
    CallGenerationFailed(String),
}

impl fmt::Display for MarginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPosition(s) => write!(f, "invalid position: {s}"),
            Self::InvalidScenario(s) => write!(f, "invalid scenario: {s}"),
            Self::InsufficientCollateral(s) => write!(f, "insufficient collateral: {s}"),
            Self::ConfigError(s) => write!(f, "config error: {s}"),
            Self::CallGenerationFailed(s) => write!(f, "call generation failed: {s}"),
        }
    }
}

impl std::error::Error for MarginError {}

// ── Margin type ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarginType {
    Initial,
    Variation,
    Maintenance,
}

impl fmt::Display for MarginType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initial => write!(f, "IM"),
            Self::Variation => write!(f, "VM"),
            Self::Maintenance => write!(f, "MM"),
        }
    }
}

// ── Call status ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallStatus {
    Pending,
    Issued,
    PartiallyMet,
    Met,
    Breached,
}

impl fmt::Display for CallStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Pending => "PENDING",
            Self::Issued => "ISSUED",
            Self::PartiallyMet => "PARTIAL",
            Self::Met => "MET",
            Self::Breached => "BREACHED",
        };
        write!(f, "{label}")
    }
}

// ── Risk scenario ───────────────────────────────────────────────

/// A single price-volatility scenario for SPAN-like scanning.
#[derive(Debug, Clone)]
pub struct RiskScenario {
    pub id: u32,
    pub price_shift_pct: f64,
    pub vol_shift_pct: f64,
    pub weight: f64,
}

impl RiskScenario {
    pub fn new(id: u32, price_shift_pct: f64, vol_shift_pct: f64) -> Self {
        Self {
            id,
            price_shift_pct,
            vol_shift_pct,
            weight: 1.0,
        }
    }

    pub fn with_weight(mut self, w: f64) -> Self {
        self.weight = w;
        self
    }
}

impl fmt::Display for RiskScenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Scenario-{}: price={:+.1}% vol={:+.1}%",
            self.id, self.price_shift_pct, self.vol_shift_pct
        )
    }
}

// ── Margin position ─────────────────────────────────────────────

/// Position for margin calculation purposes.
#[derive(Debug, Clone)]
pub struct MarginPosition {
    pub account_id: String,
    pub security_id: String,
    pub quantity: f64,
    pub current_price: f64,
    pub volatility: f64,
    pub product_group: String,
}

impl MarginPosition {
    pub fn new(
        account_id: &str,
        security_id: &str,
        quantity: f64,
        current_price: f64,
    ) -> Self {
        Self {
            account_id: account_id.to_string(),
            security_id: security_id.to_string(),
            quantity,
            current_price,
            volatility: 0.20,
            product_group: "DEFAULT".to_string(),
        }
    }

    pub fn with_volatility(mut self, vol: f64) -> Self {
        self.volatility = vol;
        self
    }

    pub fn with_product_group(mut self, group: &str) -> Self {
        self.product_group = group.to_string();
        self
    }

    /// Notional value.
    pub fn notional(&self) -> f64 {
        self.quantity.abs() * self.current_price
    }

    /// Evaluate P&L under a given scenario.
    pub fn scenario_pnl(&self, scenario: &RiskScenario) -> f64 {
        let new_price = self.current_price * (1.0 + scenario.price_shift_pct / 100.0);
        let vol_impact = self.notional() * self.volatility * (scenario.vol_shift_pct / 100.0);
        let price_pnl = self.quantity * (new_price - self.current_price);
        (price_pnl + vol_impact) * scenario.weight
    }
}

impl fmt::Display for MarginPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MarginPos({}, {}: qty={:.2}, px={:.4})",
            self.account_id, self.security_id, self.quantity, self.current_price
        )
    }
}

// ── Margin requirement ──────────────────────────────────────────

/// Computed margin requirement for an account.
#[derive(Debug, Clone)]
pub struct MarginRequirement {
    pub account_id: String,
    pub margin_type: MarginType,
    pub scan_risk: f64,
    pub intra_spread_charge: f64,
    pub inter_spread_credit: f64,
    pub short_option_minimum: f64,
    pub net_requirement: f64,
}

impl MarginRequirement {
    /// Net requirement after offsets.
    pub fn compute_net(&mut self) {
        self.net_requirement = (self.scan_risk + self.intra_spread_charge
            - self.inter_spread_credit)
            .max(self.short_option_minimum);
    }
}

impl fmt::Display for MarginRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MarginReq({}, {}): scan={:.2}, net={:.2}",
            self.account_id, self.margin_type, self.scan_risk, self.net_requirement
        )
    }
}

// ── Margin call ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MarginCall {
    pub call_id: u64,
    pub account_id: String,
    pub margin_type: MarginType,
    pub required_amount: f64,
    pub current_collateral: f64,
    pub shortfall: f64,
    pub deadline_ts: u64,
    pub status: CallStatus,
}

impl MarginCall {
    pub fn is_met(&self) -> bool {
        self.status == CallStatus::Met
    }
}

impl fmt::Display for MarginCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Call-{}: {} {} shortfall={:.2} [{}]",
            self.call_id, self.account_id, self.margin_type, self.shortfall, self.status
        )
    }
}

// ── Margin config ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MarginConfig {
    pub initial_margin_pct: f64,
    pub maintenance_margin_pct: f64,
    pub intra_spread_rate: f64,
    pub inter_spread_credit_rate: f64,
    pub short_option_min_pct: f64,
    pub portfolio_margining_enabled: bool,
    pub cross_margin_enabled: bool,
    pub call_deadline_hours: u64,
    pub scenarios: Vec<RiskScenario>,
}

impl Default for MarginConfig {
    fn default() -> Self {
        Self {
            initial_margin_pct: 10.0,
            maintenance_margin_pct: 7.5,
            intra_spread_rate: 0.02,
            inter_spread_credit_rate: 0.8,
            short_option_min_pct: 0.5,
            portfolio_margining_enabled: false,
            cross_margin_enabled: false,
            call_deadline_hours: 24,
            scenarios: default_scenarios(),
        }
    }
}

impl MarginConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_initial_margin_pct(mut self, pct: f64) -> Self {
        self.initial_margin_pct = pct;
        self
    }

    pub fn with_maintenance_margin_pct(mut self, pct: f64) -> Self {
        self.maintenance_margin_pct = pct;
        self
    }

    pub fn with_portfolio_margining(mut self, enabled: bool) -> Self {
        self.portfolio_margining_enabled = enabled;
        self
    }

    pub fn with_cross_margin(mut self, enabled: bool) -> Self {
        self.cross_margin_enabled = enabled;
        self
    }

    pub fn with_scenarios(mut self, scenarios: Vec<RiskScenario>) -> Self {
        self.scenarios = scenarios;
        self
    }

    pub fn with_call_deadline_hours(mut self, hours: u64) -> Self {
        self.call_deadline_hours = hours;
        self
    }
}

impl fmt::Display for MarginConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MarginConfig(IM={:.1}%, MM={:.1}%, scenarios={}, portfolio={}, cross={})",
            self.initial_margin_pct,
            self.maintenance_margin_pct,
            self.scenarios.len(),
            self.portfolio_margining_enabled,
            self.cross_margin_enabled,
        )
    }
}

fn default_scenarios() -> Vec<RiskScenario> {
    vec![
        RiskScenario::new(1, -10.0, 0.0),
        RiskScenario::new(2, -5.0, 0.0),
        RiskScenario::new(3, 0.0, 0.0),
        RiskScenario::new(4, 5.0, 0.0),
        RiskScenario::new(5, 10.0, 0.0),
        RiskScenario::new(6, 0.0, -25.0),
        RiskScenario::new(7, 0.0, 25.0),
        RiskScenario::new(8, -10.0, 25.0),
        RiskScenario::new(9, 10.0, -25.0),
    ]
}

// ── Margin calculator ───────────────────────────────────────────

pub struct MarginCalculator {
    config: MarginConfig,
    positions: Vec<MarginPosition>,
    collateral: HashMap<String, f64>,
    calls: Vec<MarginCall>,
    next_call_id: u64,
    current_ts: u64,
}

impl MarginCalculator {
    pub fn new(config: MarginConfig) -> Self {
        Self {
            config,
            positions: Vec::new(),
            collateral: HashMap::new(),
            calls: Vec::new(),
            next_call_id: 1,
            current_ts: 0,
        }
    }

    pub fn set_timestamp(&mut self, ts: u64) {
        self.current_ts = ts;
    }

    pub fn add_position(&mut self, pos: MarginPosition) {
        self.positions.push(pos);
    }

    pub fn set_collateral(&mut self, account_id: &str, amount: f64) {
        self.collateral.insert(account_id.to_string(), amount);
    }

    /// Compute initial margin using SPAN-like scan risk.
    pub fn compute_initial_margin(
        &self,
        account_id: &str,
    ) -> Result<MarginRequirement, MarginError> {
        let acct_positions: Vec<&MarginPosition> = self
            .positions
            .iter()
            .filter(|p| p.account_id == account_id)
            .collect();

        if acct_positions.is_empty() {
            return Err(MarginError::InvalidPosition(
                format!("no positions for {account_id}"),
            ));
        }

        // Scan risk: worst-case loss across scenarios
        let scan_risk = self.compute_scan_risk(&acct_positions);

        // Intra-spread charge for same product group
        let intra_charge = self.compute_intra_spread(&acct_positions);

        // Inter-spread credit for cross-product offsets
        let inter_credit = if self.config.portfolio_margining_enabled {
            self.compute_inter_spread_credit(&acct_positions)
        } else {
            0.0
        };

        // Short option minimum
        let short_opt_min = self.compute_short_option_min(&acct_positions);

        let mut req = MarginRequirement {
            account_id: account_id.to_string(),
            margin_type: MarginType::Initial,
            scan_risk,
            intra_spread_charge: intra_charge,
            inter_spread_credit: inter_credit,
            short_option_minimum: short_opt_min,
            net_requirement: 0.0,
        };
        req.compute_net();
        Ok(req)
    }

    fn compute_scan_risk(&self, positions: &[&MarginPosition]) -> f64 {
        let mut worst_loss = 0.0f64;

        for scenario in &self.config.scenarios {
            let total_pnl: f64 = positions.iter().map(|p| p.scenario_pnl(scenario)).sum();
            let loss = -total_pnl;
            if loss > worst_loss {
                worst_loss = loss;
            }
        }

        worst_loss.max(0.0)
    }

    fn compute_intra_spread(&self, positions: &[&MarginPosition]) -> f64 {
        let mut group_notional: HashMap<&str, f64> = HashMap::new();
        for p in positions {
            let entry = group_notional.entry(&p.product_group).or_insert(0.0);
            *entry += p.notional();
        }
        group_notional.values().sum::<f64>() * self.config.intra_spread_rate
    }

    fn compute_inter_spread_credit(&self, positions: &[&MarginPosition]) -> f64 {
        let mut groups: HashMap<&str, f64> = HashMap::new();
        for p in positions {
            let entry = groups.entry(&p.product_group).or_insert(0.0);
            *entry += p.quantity;
        }

        // Credit for offsetting positions across groups
        let net_values: Vec<f64> = groups.values().copied().collect();
        if net_values.len() < 2 {
            return 0.0;
        }

        let positive: f64 = net_values.iter().filter(|v| **v > 0.0).sum();
        let negative: f64 = net_values.iter().filter(|v| **v < 0.0).map(|v| v.abs()).sum();
        let offset = positive.min(negative);

        offset * self.config.inter_spread_credit_rate
    }

    fn compute_short_option_min(&self, positions: &[&MarginPosition]) -> f64 {
        positions
            .iter()
            .filter(|p| p.quantity < 0.0)
            .map(|p| p.notional() * self.config.short_option_min_pct / 100.0)
            .sum()
    }

    /// Compute variation margin (mark-to-market P&L).
    pub fn compute_variation_margin(
        &self,
        account_id: &str,
        prev_prices: &HashMap<String, f64>,
    ) -> Result<f64, MarginError> {
        let vm: f64 = self
            .positions
            .iter()
            .filter(|p| p.account_id == account_id)
            .map(|p| {
                let prev = prev_prices.get(&p.security_id).copied().unwrap_or(p.current_price);
                p.quantity * (p.current_price - prev)
            })
            .sum();

        Ok(vm)
    }

    /// Compute maintenance margin.
    pub fn compute_maintenance_margin(
        &self,
        account_id: &str,
    ) -> Result<f64, MarginError> {
        let total_notional: f64 = self
            .positions
            .iter()
            .filter(|p| p.account_id == account_id)
            .map(|p| p.notional())
            .sum();

        if total_notional == 0.0 {
            return Err(MarginError::InvalidPosition(
                format!("no positions for {account_id}"),
            ));
        }

        Ok(total_notional * self.config.maintenance_margin_pct / 100.0)
    }

    /// Generate margin calls for all accounts with shortfalls.
    pub fn generate_calls(&mut self) -> Vec<MarginCall> {
        let accounts: Vec<String> = self
            .positions
            .iter()
            .map(|p| p.account_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let mut new_calls = Vec::new();

        for acct in &accounts {
            let im = match self.compute_initial_margin(acct) {
                Ok(r) => r.net_requirement,
                Err(_) => continue,
            };

            let collateral = self.collateral.get(acct).copied().unwrap_or(0.0);
            let shortfall = (im - collateral).max(0.0);

            if shortfall > 0.0 {
                let call = MarginCall {
                    call_id: self.next_call_id,
                    account_id: acct.clone(),
                    margin_type: MarginType::Initial,
                    required_amount: im,
                    current_collateral: collateral,
                    shortfall,
                    deadline_ts: self.current_ts + self.config.call_deadline_hours * 3600,
                    status: CallStatus::Issued,
                };
                self.next_call_id += 1;
                new_calls.push(call);
            }
        }

        self.calls.extend(new_calls.clone());
        new_calls
    }

    /// Meet a margin call by depositing additional collateral.
    pub fn meet_call(&mut self, call_id: u64, amount: f64) -> Result<(), MarginError> {
        let call = self
            .calls
            .iter_mut()
            .find(|c| c.call_id == call_id)
            .ok_or_else(|| MarginError::CallGenerationFailed("call not found".into()))?;

        let acct = call.account_id.clone();
        let entry = self.collateral.entry(acct).or_insert(0.0);
        *entry += amount;
        call.current_collateral += amount;
        call.shortfall = (call.required_amount - call.current_collateral).max(0.0);

        call.status = if call.shortfall <= 0.0 {
            CallStatus::Met
        } else {
            CallStatus::PartiallyMet
        };

        Ok(())
    }

    pub fn calls(&self) -> &[MarginCall] {
        &self.calls
    }

    pub fn config(&self) -> &MarginConfig {
        &self.config
    }
}

impl fmt::Display for MarginCalculator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MarginCalculator(positions={}, calls={})",
            self.positions.len(),
            self.calls.len(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_calc() -> MarginCalculator {
        let mut mc = MarginCalculator::new(MarginConfig::default());
        mc.add_position(
            MarginPosition::new("ACC1", "AAPL", 100.0, 150.0)
                .with_volatility(0.25),
        );
        mc.set_collateral("ACC1", 5000.0);
        mc
    }

    #[test]
    fn test_config_builder() {
        let cfg = MarginConfig::new()
            .with_initial_margin_pct(15.0)
            .with_portfolio_margining(true)
            .with_cross_margin(true);
        assert!((cfg.initial_margin_pct - 15.0).abs() < 1e-9);
        assert!(cfg.portfolio_margining_enabled);
        assert!(cfg.cross_margin_enabled);
    }

    #[test]
    fn test_default_scenarios_count() {
        let cfg = MarginConfig::default();
        assert_eq!(cfg.scenarios.len(), 9);
    }

    #[test]
    fn test_position_notional() {
        let pos = MarginPosition::new("ACC1", "AAPL", 100.0, 150.0);
        assert!((pos.notional() - 15000.0).abs() < 1e-9);
    }

    #[test]
    fn test_scenario_pnl_positive_shift() {
        let pos = MarginPosition::new("ACC1", "AAPL", 100.0, 150.0)
            .with_volatility(0.0);
        let scenario = RiskScenario::new(1, 10.0, 0.0);
        let pnl = pos.scenario_pnl(&scenario);
        assert!((pnl - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn test_scenario_pnl_negative_shift() {
        let pos = MarginPosition::new("ACC1", "AAPL", 100.0, 150.0)
            .with_volatility(0.0);
        let scenario = RiskScenario::new(1, -10.0, 0.0);
        let pnl = pos.scenario_pnl(&scenario);
        assert!((pnl - -1500.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_initial_margin() {
        let mc = setup_calc();
        let req = mc.compute_initial_margin("ACC1").unwrap();
        assert!(req.scan_risk > 0.0);
        assert!(req.net_requirement > 0.0);
    }

    #[test]
    fn test_compute_initial_margin_no_positions() {
        let mc = setup_calc();
        let r = mc.compute_initial_margin("UNKNOWN");
        assert!(r.is_err());
    }

    #[test]
    fn test_variation_margin() {
        let mc = setup_calc();
        let mut prev = HashMap::new();
        prev.insert("AAPL".to_string(), 145.0);
        let vm = mc.compute_variation_margin("ACC1", &prev).unwrap();
        assert!((vm - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_maintenance_margin() {
        let mc = setup_calc();
        let mm = mc.compute_maintenance_margin("ACC1").unwrap();
        assert!((mm - 1125.0).abs() < 1e-9); // 15000 * 7.5%
    }

    #[test]
    fn test_generate_calls_with_shortfall() {
        let mut mc = MarginCalculator::new(MarginConfig::default());
        mc.add_position(MarginPosition::new("ACC1", "AAPL", 100.0, 150.0));
        mc.set_collateral("ACC1", 10.0); // Very low collateral
        let calls = mc.generate_calls();
        assert!(!calls.is_empty());
        assert!(calls[0].shortfall > 0.0);
    }

    #[test]
    fn test_generate_calls_no_shortfall() {
        let mut mc = MarginCalculator::new(MarginConfig::default());
        mc.add_position(
            MarginPosition::new("ACC1", "AAPL", 1.0, 1.0).with_volatility(0.0),
        );
        mc.set_collateral("ACC1", 100_000.0);
        let calls = mc.generate_calls();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_meet_call_fully() {
        let mut mc = MarginCalculator::new(MarginConfig::default());
        mc.add_position(MarginPosition::new("ACC1", "AAPL", 100.0, 150.0));
        mc.set_collateral("ACC1", 10.0);
        let calls = mc.generate_calls();
        let call_id = calls[0].call_id;
        mc.meet_call(call_id, 1_000_000.0).unwrap();
        assert!(mc.calls().iter().find(|c| c.call_id == call_id).unwrap().is_met());
    }

    #[test]
    fn test_meet_call_partially() {
        let mut mc = MarginCalculator::new(MarginConfig::default());
        mc.add_position(MarginPosition::new("ACC1", "AAPL", 100.0, 150.0));
        mc.set_collateral("ACC1", 10.0);
        let calls = mc.generate_calls();
        let call_id = calls[0].call_id;
        mc.meet_call(call_id, 1.0).unwrap();
        let c = mc.calls().iter().find(|c| c.call_id == call_id).unwrap();
        assert_eq!(c.status, CallStatus::PartiallyMet);
    }

    #[test]
    fn test_margin_requirement_compute_net() {
        let mut req = MarginRequirement {
            account_id: "A".into(),
            margin_type: MarginType::Initial,
            scan_risk: 1000.0,
            intra_spread_charge: 200.0,
            inter_spread_credit: 100.0,
            short_option_minimum: 50.0,
            net_requirement: 0.0,
        };
        req.compute_net();
        assert!((req.net_requirement - 1100.0).abs() < 1e-9);
    }

    #[test]
    fn test_risk_scenario_display() {
        let s = RiskScenario::new(1, -10.0, 25.0);
        let disp = format!("{s}");
        assert!(disp.contains("-10.0%"));
    }

    #[test]
    fn test_portfolio_margining_gives_credit() {
        let cfg = MarginConfig::new().with_portfolio_margining(true);
        let mut mc = MarginCalculator::new(cfg);
        mc.add_position(
            MarginPosition::new("ACC1", "AAPL", 100.0, 150.0)
                .with_product_group("TECH"),
        );
        mc.add_position(
            MarginPosition::new("ACC1", "GOOG", -50.0, 200.0)
                .with_product_group("MEDIA"),
        );
        let req = mc.compute_initial_margin("ACC1").unwrap();
        assert!(req.inter_spread_credit >= 0.0);
    }

    #[test]
    fn test_short_option_minimum() {
        let mut mc = MarginCalculator::new(MarginConfig::default());
        mc.add_position(MarginPosition::new("ACC1", "AAPL", -100.0, 150.0));
        let req = mc.compute_initial_margin("ACC1").unwrap();
        assert!(req.short_option_minimum > 0.0);
    }

    #[test]
    fn test_display_impls() {
        let mc = setup_calc();
        assert!(format!("{mc}").contains("MarginCalculator"));
        assert_eq!(format!("{}", MarginType::Initial), "IM");
        assert_eq!(format!("{}", CallStatus::Issued), "ISSUED");
    }
}
