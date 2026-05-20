//! Margin Optimization
//!
//! Portfolio margin calculation with spread recognition, correlation netting,
//! and hedge discovery for optimal capital efficiency.

use super::correlation::CorrelationMatrix;
use super::risk::RiskAggregator;
use super::{Greeks, OptionType, Side};
use std::collections::HashMap;

/// Margin calculation rules
#[derive(Debug, Clone)]
pub struct MarginRules {
    /// Base margin per contract (naked)
    pub base_margin_per_contract: f64,

    /// Spread margin reduction factor (e.g., 0.3 = 70% reduction for spreads)
    pub spread_reduction: f64,

    /// Correlation threshold for netting (e.g., 0.8)
    pub correlation_netting_threshold: f32,

    /// Correlation netting factor (e.g., 0.5 = 50% reduction for correlated hedges)
    pub correlation_netting_factor: f64,

    /// VaR confidence level
    pub var_confidence: f64,

    /// VaR floor factor (minimum margin = VaR * factor)
    pub var_floor_factor: f64,
}

impl Default for MarginRules {
    fn default() -> Self {
        Self {
            base_margin_per_contract: 100.0,
            spread_reduction: 0.3,
            correlation_netting_threshold: 0.8,
            correlation_netting_factor: 0.5,
            var_confidence: 0.99,
            var_floor_factor: 1.5,
        }
    }
}

/// A position leg in the margin system
#[derive(Debug, Clone)]
pub struct PositionLeg {
    /// Unique position ID
    pub id: String,
    /// Underlying symbol
    pub symbol: String,
    /// Strike price
    pub strike: f64,
    /// Expiration epoch (days)
    pub expiration: u64,
    /// Option type
    pub option_type: OptionType,
    /// Position direction
    pub side: Side,
    /// Quantity (contracts)
    pub quantity: f64,
    /// Greeks
    pub greeks: Greeks,
    /// Notional value
    pub notional: f64,
}

impl PositionLeg {
    /// Check if this is a long position
    pub fn is_long(&self) -> bool {
        self.side == Side::Buy
    }

    /// Check if this is a short position
    pub fn is_short(&self) -> bool {
        self.side == Side::Sell
    }

    /// Net delta exposure
    pub fn net_delta(&self) -> f64 {
        let direction = if self.is_long() { 1.0 } else { -1.0 };
        self.greeks.delta * self.quantity * direction * 100.0
    }
}

/// A proposed hedge to reduce margin
#[derive(Debug, Clone)]
pub struct HedgeProposal {
    /// Target symbol
    pub symbol: String,
    /// Proposed strike
    pub strike: f64,
    /// Proposed expiration
    pub expiration: u64,
    /// Proposed option type
    pub option_type: OptionType,
    /// Proposed side
    pub side: Side,
    /// Proposed quantity
    pub quantity: f64,
    /// Expected margin reduction
    pub margin_reduction: f64,
    /// Reason for proposal
    pub reason: String,
}

/// Margin calculation result
#[derive(Debug, Clone)]
pub struct MarginResult {
    /// Gross margin (before netting)
    pub gross_margin: f64,
    /// Spread margin reduction
    pub spread_reduction: f64,
    /// Correlation netting reduction
    pub correlation_reduction: f64,
    /// Net margin requirement
    pub net_margin: f64,
    /// VaR floor applied?
    pub var_floor_applied: bool,
    /// Breakdown by symbol
    pub by_symbol: HashMap<String, f64>,
}

/// Margin Optimizer
///
/// Calculates portfolio margin with spread recognition and correlation netting.
pub struct MarginOptimizer {
    /// Positions by (symbol, expiration)
    positions: HashMap<(String, u64), Vec<PositionLeg>>,

    /// Risk aggregator reference
    risk_agg: RiskAggregator,

    /// Correlation matrix reference
    correlations: CorrelationMatrix,

    /// Margin calculation rules
    rules: MarginRules,

    /// Position index by ID
    position_index: HashMap<String, (String, u64)>,
}

impl MarginOptimizer {
    /// Create a new margin optimizer
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            risk_agg: RiskAggregator::new(),
            correlations: CorrelationMatrix::new(),
            rules: MarginRules::default(),
            position_index: HashMap::new(),
        }
    }

    /// Create with custom rules
    pub fn with_rules(rules: MarginRules) -> Self {
        Self {
            positions: HashMap::new(),
            risk_agg: RiskAggregator::new(),
            correlations: CorrelationMatrix::new(),
            rules,
            position_index: HashMap::new(),
        }
    }

    /// Add a position leg
    pub fn add_position(&mut self, leg: PositionLeg) {
        let key = (leg.symbol.clone(), leg.expiration);

        // Update risk aggregator
        self.risk_agg
            .add_position(&leg.symbol, &leg.greeks, leg.notional, leg.expiration);

        // Store position
        self.position_index.insert(leg.id.clone(), key.clone());
        self.positions.entry(key).or_default().push(leg);
    }

    /// Remove a position leg
    pub fn remove_position(&mut self, position_id: &str) -> Option<PositionLeg> {
        let key = self.position_index.remove(position_id)?;
        let legs = self.positions.get_mut(&key)?;

        let idx = legs.iter().position(|l| l.id == position_id)?;
        let leg = legs.remove(idx);

        // Update risk aggregator
        self.risk_agg
            .remove_position(&leg.symbol, &leg.greeks, leg.notional, leg.expiration);

        // Clean up empty vectors
        if legs.is_empty() {
            self.positions.remove(&key);
        }

        Some(leg)
    }

    /// Calculate total margin requirement
    pub fn calculate_total_margin(&self) -> MarginResult {
        let mut gross_margin = 0.0;
        let mut spread_reduction_total = 0.0;
        let mut by_symbol: HashMap<String, f64> = HashMap::new();

        // Calculate per-symbol margin with spread recognition
        let mut symbols: Vec<&String> = self.positions.keys().map(|(s, _)| s).collect();
        symbols.sort();
        symbols.dedup();

        for symbol in &symbols {
            let symbol_positions: Vec<&PositionLeg> = self
                .positions
                .iter()
                .filter(|((s, _), _)| s == *symbol)
                .flat_map(|(_, legs)| legs.iter())
                .collect();

            let (gross, spread_red) = self.calculate_symbol_margin(&symbol_positions);
            gross_margin += gross;
            spread_reduction_total += spread_red;
            by_symbol.insert((*symbol).clone(), gross - spread_red);
        }

        // Calculate correlation netting
        let correlation_reduction = self.calculate_correlation_netting(&symbols);

        // Calculate net margin
        let mut net_margin = gross_margin - spread_reduction_total - correlation_reduction;

        // Apply VaR floor
        let var_estimate = self.risk_agg.estimate_var(self.rules.var_confidence, 1);
        let var_floor = var_estimate * self.rules.var_floor_factor;
        let var_floor_applied = net_margin < var_floor;

        if var_floor_applied {
            net_margin = var_floor;
        }

        MarginResult {
            gross_margin,
            spread_reduction: spread_reduction_total,
            correlation_reduction,
            net_margin,
            var_floor_applied,
            by_symbol,
        }
    }

    /// Calculate margin for a single symbol's positions
    fn calculate_symbol_margin(&self, positions: &[&PositionLeg]) -> (f64, f64) {
        if positions.is_empty() {
            return (0.0, 0.0);
        }

        // Group by expiration
        let mut by_expiry: HashMap<u64, Vec<&PositionLeg>> = HashMap::new();
        for pos in positions {
            by_expiry.entry(pos.expiration).or_default().push(*pos);
        }

        let mut gross = 0.0;
        let mut spread_reduction = 0.0;

        for (_expiry, legs) in &by_expiry {
            let (exp_gross, exp_spread) = self.calculate_expiry_margin(legs);
            gross += exp_gross;
            spread_reduction += exp_spread;
        }

        (gross, spread_reduction)
    }

    /// Calculate margin for positions at a single expiration
    fn calculate_expiry_margin(&self, legs: &[&PositionLeg]) -> (f64, f64) {
        // Count long/short calls and puts
        let mut long_calls: Vec<&PositionLeg> = Vec::new();
        let mut short_calls: Vec<&PositionLeg> = Vec::new();
        let mut long_puts: Vec<&PositionLeg> = Vec::new();
        let mut short_puts: Vec<&PositionLeg> = Vec::new();

        for leg in legs {
            match (&leg.option_type, leg.is_long()) {
                (OptionType::Call, true) => long_calls.push(*leg),
                (OptionType::Call, false) => short_calls.push(*leg),
                (OptionType::Put, true) => long_puts.push(*leg),
                (OptionType::Put, false) => short_puts.push(*leg),
            }
        }

        // Sort by strike
        long_calls.sort_by(|a, b| {
            a.strike
                .partial_cmp(&b.strike)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        short_calls.sort_by(|a, b| {
            a.strike
                .partial_cmp(&b.strike)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        long_puts.sort_by(|a, b| {
            a.strike
                .partial_cmp(&b.strike)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        short_puts.sort_by(|a, b| {
            a.strike
                .partial_cmp(&b.strike)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Calculate gross margin (short positions only)
        let short_qty: f64 = short_calls.iter().map(|p| p.quantity).sum::<f64>()
            + short_puts.iter().map(|p| p.quantity).sum::<f64>();

        let gross = short_qty * self.rules.base_margin_per_contract;

        // Detect spreads and calculate reduction
        let mut spread_reduction = 0.0;

        // Call spreads: short call hedged by long call at lower strike
        let call_spreads = self.count_spreads(&short_calls, &long_calls);
        spread_reduction += call_spreads
            * self.rules.base_margin_per_contract
            * (1.0 - self.rules.spread_reduction);

        // Put spreads: short put hedged by long put at higher strike
        let put_spreads = self.count_spreads(&short_puts, &long_puts);
        spread_reduction +=
            put_spreads * self.rules.base_margin_per_contract * (1.0 - self.rules.spread_reduction);

        (gross, spread_reduction)
    }

    /// Count spreads (matching short with long)
    fn count_spreads(&self, shorts: &[&PositionLeg], longs: &[&PositionLeg]) -> f64 {
        if shorts.is_empty() || longs.is_empty() {
            return 0.0;
        }

        let short_qty: f64 = shorts.iter().map(|p| p.quantity).sum();
        let long_qty: f64 = longs.iter().map(|p| p.quantity).sum();

        // Spread count = min of long and short quantities
        short_qty.min(long_qty)
    }

    /// Calculate correlation netting across symbols
    fn calculate_correlation_netting(&self, symbols: &[&String]) -> f64 {
        if symbols.len() < 2 {
            return 0.0;
        }

        let mut total_reduction = 0.0;

        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let sym1 = symbols[i];
                let sym2 = symbols[j];

                // Check correlation
                if let Some(corr) = self.correlations.query_correlation(sym1, sym2) {
                    if corr >= self.rules.correlation_netting_threshold {
                        // Calculate hedge benefit
                        let delta1 = self.risk_agg.delta_by_symbol(sym1);
                        let delta2 = self.risk_agg.delta_by_symbol(sym2);

                        // Opposite deltas = natural hedge
                        if delta1 * delta2 < 0.0 {
                            let hedge_benefit = delta1.abs().min(delta2.abs());
                            let reduction = hedge_benefit * self.rules.correlation_netting_factor;
                            total_reduction += reduction;
                        }
                    }
                }
            }
        }

        total_reduction
    }

    /// Find optimal hedges to reduce margin
    pub fn find_optimal_hedges(&self, max_proposals: usize) -> Vec<HedgeProposal> {
        let mut proposals = Vec::new();

        // Strategy 1: Delta reduction within same symbol
        for ((symbol, expiry), legs) in &self.positions {
            let net_delta: f64 = legs.iter().map(|l| l.net_delta()).sum();

            if net_delta.abs() > 50.0 {
                // Significant delta exposure
                let hedge_type = if net_delta > 0.0 {
                    (OptionType::Put, Side::Buy) // Long put to reduce long delta
                } else {
                    (OptionType::Call, Side::Buy) // Long call to reduce short delta
                };

                // Estimate strike (ATM)
                let avg_strike = if !legs.is_empty() {
                    legs.iter().map(|l| l.strike).sum::<f64>() / legs.len() as f64
                } else {
                    100.0
                };

                let qty = (net_delta.abs() / 50.0).ceil(); // ~50 delta per ATM option

                proposals.push(HedgeProposal {
                    symbol: symbol.clone(),
                    strike: avg_strike,
                    expiration: *expiry,
                    option_type: hedge_type.0,
                    side: hedge_type.1,
                    quantity: qty,
                    margin_reduction: qty
                        * self.rules.base_margin_per_contract
                        * (1.0 - self.rules.spread_reduction),
                    reason: format!("Reduce {} delta exposure of {:.0}", symbol, net_delta),
                });
            }

            if proposals.len() >= max_proposals {
                break;
            }
        }

        // Strategy 2: Spread completion
        for ((symbol, expiry), legs) in &self.positions {
            // Find naked shorts
            let short_calls: Vec<&PositionLeg> = legs
                .iter()
                .filter(|l| l.option_type == OptionType::Call && l.is_short())
                .collect();

            let long_calls: Vec<&PositionLeg> = legs
                .iter()
                .filter(|l| l.option_type == OptionType::Call && l.is_long())
                .collect();

            let short_qty: f64 = short_calls.iter().map(|l| l.quantity).sum();
            let long_qty: f64 = long_calls.iter().map(|l| l.quantity).sum();

            if short_qty > long_qty && proposals.len() < max_proposals {
                let naked_qty = short_qty - long_qty;
                let lowest_short_strike = short_calls
                    .iter()
                    .map(|l| l.strike)
                    .fold(f64::INFINITY, f64::min);

                proposals.push(HedgeProposal {
                    symbol: symbol.clone(),
                    strike: lowest_short_strike - 5.0, // Lower strike
                    expiration: *expiry,
                    option_type: OptionType::Call,
                    side: Side::Buy,
                    quantity: naked_qty,
                    margin_reduction: naked_qty
                        * self.rules.base_margin_per_contract
                        * (1.0 - self.rules.spread_reduction),
                    reason: format!("Complete call spread on {} ({} naked)", symbol, naked_qty),
                });
            }

            if proposals.len() >= max_proposals {
                break;
            }
        }

        proposals
    }

    /// Update correlation data (pass-through to internal matrix)
    pub fn observe_price_moves(&mut self, moves: &[(&str, f64, f64)]) {
        let symbols: Vec<&str> = moves.iter().map(|(s, _, _)| *s).collect();

        for (symbol, prev, curr) in moves {
            self.correlations.observe_price_move(symbol, *prev, *curr);
        }

        self.correlations.update_correlations(&symbols);
    }

    /// Get risk summary
    pub fn risk_summary(&self) -> super::risk::PortfolioRiskSummary {
        self.risk_agg.portfolio_summary()
    }

    /// Get position count
    pub fn position_count(&self) -> usize {
        self.positions.values().map(|v| v.len()).sum()
    }

    /// Get margin rules
    pub fn rules(&self) -> &MarginRules {
        &self.rules
    }

    /// Update margin rules
    pub fn set_rules(&mut self, rules: MarginRules) {
        self.rules = rules;
    }
}

impl Default for MarginOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_leg(
        id: &str,
        symbol: &str,
        strike: f64,
        expiry: u64,
        opt_type: OptionType,
        side: Side,
        qty: f64,
    ) -> PositionLeg {
        let delta = match (&opt_type, &side) {
            (OptionType::Call, Side::Buy) => 0.5,
            (OptionType::Call, Side::Sell) => 0.5,
            (OptionType::Put, Side::Buy) => -0.5,
            (OptionType::Put, Side::Sell) => -0.5,
        };

        PositionLeg {
            id: id.to_string(),
            symbol: symbol.to_string(),
            strike,
            expiration: expiry,
            option_type: opt_type,
            side,
            quantity: qty,
            greeks: Greeks::new(delta, 0.02, -0.05, 0.25),
            notional: strike * qty * 100.0,
        }
    }

    #[test]
    fn test_naked_short_margin() {
        let mut optimizer = MarginOptimizer::new();

        // Add naked short call
        let leg = make_leg(
            "1",
            "AAPL",
            150.0,
            19800,
            OptionType::Call,
            Side::Sell,
            10.0,
        );
        optimizer.add_position(leg);

        let result = optimizer.calculate_total_margin();

        // 10 contracts * 100 base = 1000 gross
        assert_eq!(result.gross_margin, 1000.0);
        assert_eq!(result.spread_reduction, 0.0); // No spread
    }

    #[test]
    fn test_spread_margin_reduction() {
        let mut optimizer = MarginOptimizer::new();

        // Bull call spread: short 155 call, long 150 call
        optimizer.add_position(make_leg(
            "1",
            "AAPL",
            155.0,
            19800,
            OptionType::Call,
            Side::Sell,
            10.0,
        ));
        optimizer.add_position(make_leg(
            "2",
            "AAPL",
            150.0,
            19800,
            OptionType::Call,
            Side::Buy,
            10.0,
        ));

        let result = optimizer.calculate_total_margin();

        println!("Gross: {}", result.gross_margin);
        println!("Spread reduction: {}", result.spread_reduction);
        println!("Net: {}", result.net_margin);
        println!("VaR floor applied: {}", result.var_floor_applied);

        // Should have spread reduction recognized
        assert!(result.spread_reduction > 0.0);

        // Spread reduction should be significant (70% of gross for full spread)
        // 10 contracts * $100 base * 0.7 reduction = $700
        assert!(result.spread_reduction >= 600.0);

        // If VaR floor not applied, net should be less than gross
        // Otherwise VaR floor may override (which is correct behavior)
        if !result.var_floor_applied {
            assert!(result.net_margin < result.gross_margin);
        }
    }

    #[test]
    fn test_hedge_proposals() {
        let mut optimizer = MarginOptimizer::new();

        // Large long delta position
        let mut leg = make_leg("1", "SPY", 400.0, 19800, OptionType::Call, Side::Buy, 20.0);
        leg.greeks.delta = 0.7; // Deep ITM
        optimizer.add_position(leg);

        let proposals = optimizer.find_optimal_hedges(5);

        assert!(!proposals.is_empty());
        println!("Proposal: {:?}", proposals[0]);

        // Should suggest put to hedge long delta
        assert!(proposals.iter().any(|p| p.option_type == OptionType::Put));
    }

    #[test]
    fn test_position_add_remove() {
        let mut optimizer = MarginOptimizer::new();

        let leg = make_leg(
            "test_1",
            "GOOG",
            100.0,
            19800,
            OptionType::Put,
            Side::Sell,
            5.0,
        );

        optimizer.add_position(leg.clone());
        assert_eq!(optimizer.position_count(), 1);

        let removed = optimizer.remove_position("test_1");
        assert!(removed.is_some());
        assert_eq!(optimizer.position_count(), 0);
    }

    #[test]
    fn test_margin_rules() {
        let rules = MarginRules {
            base_margin_per_contract: 200.0,
            spread_reduction: 0.2,
            ..Default::default()
        };

        let optimizer = MarginOptimizer::with_rules(rules);
        assert_eq!(optimizer.rules().base_margin_per_contract, 200.0);
        assert_eq!(optimizer.rules().spread_reduction, 0.2);
    }
}
