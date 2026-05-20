//! Real-Time Risk Aggregation Module
//!
//! Provides portfolio-level Greek aggregation, VaR estimation, and margin calculations.
//! Uses a 6-layer architecture combining arithmetic sums with holographic pattern queries.

use super::{DIMENSION, Greeks};
use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// Risk state for a single symbol
#[derive(Debug, Clone)]
pub struct SymbolRisk {
    pub symbol: String,
    pub total_delta: f64,
    pub total_gamma: f64,
    pub total_theta: f64,
    pub total_vega: f64,
    pub position_count: usize,
    pub notional_value: f64,
}

impl SymbolRisk {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            total_delta: 0.0,
            total_gamma: 0.0,
            total_theta: 0.0,
            total_vega: 0.0,
            position_count: 0,
            notional_value: 0.0,
        }
    }

    pub fn add_position(&mut self, greeks: &Greeks, notional: f64) {
        self.total_delta += greeks.delta;
        self.total_gamma += greeks.gamma;
        self.total_theta += greeks.theta;
        self.total_vega += greeks.vega;
        self.position_count += 1;
        self.notional_value += notional;
    }

    pub fn remove_position(&mut self, greeks: &Greeks, notional: f64) {
        self.total_delta -= greeks.delta;
        self.total_gamma -= greeks.gamma;
        self.total_theta -= greeks.theta;
        self.total_vega -= greeks.vega;
        self.position_count = self.position_count.saturating_sub(1);
        self.notional_value -= notional;
    }
}

/// Risk state for positions at a specific expiration
#[derive(Debug, Clone)]
pub struct ExpirationRisk {
    pub symbol: String,
    pub expiration_epoch: u64,
    pub total_delta: f64,
    pub total_gamma: f64,
    pub total_theta: f64,
    pub total_vega: f64,
    pub position_count: usize,
}

impl ExpirationRisk {
    pub fn new(symbol: &str, expiration_epoch: u64) -> Self {
        Self {
            symbol: symbol.to_string(),
            expiration_epoch,
            total_delta: 0.0,
            total_gamma: 0.0,
            total_theta: 0.0,
            total_vega: 0.0,
            position_count: 0,
        }
    }

    pub fn add_position(&mut self, greeks: &Greeks) {
        self.total_delta += greeks.delta;
        self.total_gamma += greeks.gamma;
        self.total_theta += greeks.theta;
        self.total_vega += greeks.vega;
        self.position_count += 1;
    }
}

/// P&L tracking snapshot
#[derive(Debug, Clone)]
pub struct PnLSnapshot {
    pub timestamp: u64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub portfolio_value: f64,
}

/// Historical risk snapshot for VaR calculation
#[derive(Debug, Clone)]
pub struct RiskSnapshot {
    pub timestamp: u64,
    pub portfolio_delta: f64,
    pub portfolio_gamma: f64,
    pub portfolio_theta: f64,
    pub portfolio_vega: f64,
    pub total_notional: f64,
    pub var_95: f64,
    pub margin_requirement: f64,
}

/// Real-Time Risk Aggregator
///
/// 6-Layer Architecture:
/// 1. Running sums (f64) - Portfolio-level Greeks
/// 2. BundleAccumulator - Holographic pattern queries
/// 3. HashMap<String, SymbolRisk> - Per-symbol aggregation
/// 4. HashMap<(String, u64), ExpirationRisk> - Per-expiration aggregation
/// 5. VecDeque<PnLSnapshot> - P&L time series
/// 6. VecDeque<RiskSnapshot> - Historical for VaR
pub struct RiskAggregator {
    // Layer 1: Running portfolio sums
    pub portfolio_delta: f64,
    pub portfolio_gamma: f64,
    pub portfolio_theta: f64,
    pub portfolio_vega: f64,
    pub total_notional: f64,
    pub total_positions: usize,

    // Layer 2: Holographic risk surface
    risk_surface: BundleAccumulator,
    field_vectors: HashMap<String, BinaryHV>,
    scalar_bases: HashMap<String, BinaryHV>,

    // Layer 3: Per-symbol aggregation
    symbol_risks: Arc<RwLock<HashMap<String, SymbolRisk>>>,

    // Layer 4: Per-expiration aggregation
    expiration_risks: Arc<RwLock<HashMap<(String, u64), ExpirationRisk>>>,

    // Layer 5: P&L tracking
    pnl_snapshots: Arc<RwLock<VecDeque<PnLSnapshot>>>,

    // Layer 6: Historical snapshots for VaR
    risk_history: Arc<RwLock<VecDeque<RiskSnapshot>>>,

    // Risk parameters (report only, no enforcement)
    pub max_history_size: usize,
    pub assumed_daily_vol: f64, // For VaR estimation
}

impl RiskAggregator {
    pub fn new() -> Self {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let mut field_vectors = HashMap::new();
        let mut scalar_bases = HashMap::new();

        // Initialize field vectors for Greeks encoding
        field_vectors.insert(
            "delta".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        field_vectors.insert(
            "gamma".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        field_vectors.insert(
            "theta".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        field_vectors.insert(
            "vega".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );

        // Scalar bases for permutation encoding
        scalar_bases.insert(
            "delta".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        scalar_bases.insert(
            "gamma".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        scalar_bases.insert(
            "theta".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );
        scalar_bases.insert(
            "vega".to_string(),
            BinaryHV::random(DIMENSION, rng.random()),
        );

        Self {
            portfolio_delta: 0.0,
            portfolio_gamma: 0.0,
            portfolio_theta: 0.0,
            portfolio_vega: 0.0,
            total_notional: 0.0,
            total_positions: 0,

            risk_surface: BundleAccumulator::new(DIMENSION),
            field_vectors,
            scalar_bases,

            symbol_risks: Arc::new(RwLock::new(HashMap::new())),
            expiration_risks: Arc::new(RwLock::new(HashMap::new())),
            pnl_snapshots: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            risk_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),

            max_history_size: 1000,
            assumed_daily_vol: 0.02, // 2% daily volatility assumption
        }
    }

    /// Update risk when a position is added
    pub fn add_position(
        &mut self,
        symbol: &str,
        greeks: &Greeks,
        notional: f64,
        expiration_epoch: u64,
    ) {
        // Layer 1: Update portfolio sums
        self.portfolio_delta += greeks.delta;
        self.portfolio_gamma += greeks.gamma;
        self.portfolio_theta += greeks.theta;
        self.portfolio_vega += greeks.vega;
        self.total_notional += notional;
        self.total_positions += 1;

        // Layer 2: Add to holographic risk surface
        let greek_hv = self.encode_greeks(greeks);
        self.risk_surface.add(&greek_hv);

        // Layer 3: Update symbol-level aggregation
        {
            let mut symbol_map = self.symbol_risks.write().unwrap();
            symbol_map
                .entry(symbol.to_string())
                .or_insert_with(|| SymbolRisk::new(symbol))
                .add_position(greeks, notional);
        }

        // Layer 4: Update expiration-level aggregation
        {
            let mut exp_map = self.expiration_risks.write().unwrap();
            let key = (symbol.to_string(), expiration_epoch);
            exp_map
                .entry(key)
                .or_insert_with(|| ExpirationRisk::new(symbol, expiration_epoch))
                .add_position(greeks);
        }
    }

    /// Update risk when a position is removed
    pub fn remove_position(
        &mut self,
        symbol: &str,
        greeks: &Greeks,
        notional: f64,
        expiration_epoch: u64,
    ) {
        // Layer 1: Update portfolio sums
        self.portfolio_delta -= greeks.delta;
        self.portfolio_gamma -= greeks.gamma;
        self.portfolio_theta -= greeks.theta;
        self.portfolio_vega -= greeks.vega;
        self.total_notional -= notional;
        self.total_positions = self.total_positions.saturating_sub(1);

        // Layer 2: Remove from holographic risk surface
        let greek_hv = self.encode_greeks(greeks);
        self.risk_surface.subtract(&greek_hv);

        // Layer 3: Update symbol-level aggregation
        {
            let mut symbol_map = self.symbol_risks.write().unwrap();
            if let Some(risk) = symbol_map.get_mut(symbol) {
                risk.remove_position(greeks, notional);
            }
        }

        // Layer 4: Update expiration-level aggregation
        {
            let mut exp_map = self.expiration_risks.write().unwrap();
            let key = (symbol.to_string(), expiration_epoch);
            if let Some(risk) = exp_map.get_mut(&key) {
                risk.total_delta -= greeks.delta;
                risk.total_gamma -= greeks.gamma;
                risk.total_theta -= greeks.theta;
                risk.total_vega -= greeks.vega;
                risk.position_count = risk.position_count.saturating_sub(1);
            }
        }
    }

    // ==================== Query Methods ====================

    /// "Total delta exposure on AAPL"
    pub fn delta_by_symbol(&self, symbol: &str) -> f64 {
        self.symbol_risks
            .read()
            .unwrap()
            .get(symbol)
            .map(|r| r.total_delta)
            .unwrap_or(0.0)
    }

    /// "Total gamma exposure on AAPL"
    pub fn gamma_by_symbol(&self, symbol: &str) -> f64 {
        self.symbol_risks
            .read()
            .unwrap()
            .get(symbol)
            .map(|r| r.total_gamma)
            .unwrap_or(0.0)
    }

    /// "Net gamma across all positions expiring at epoch"
    pub fn gamma_by_expiration(&self, exp_epoch: u64) -> f64 {
        self.expiration_risks
            .read()
            .unwrap()
            .iter()
            .filter(|((_, exp), _)| *exp == exp_epoch)
            .map(|(_, r)| r.total_gamma)
            .sum()
    }

    /// "Portfolio theta decay per day"
    pub fn portfolio_theta(&self) -> f64 {
        self.portfolio_theta
    }

    /// "Portfolio vega (volatility exposure)"
    pub fn portfolio_vega(&self) -> f64 {
        self.portfolio_vega
    }

    /// Get all Greeks for a specific symbol
    pub fn greeks_by_symbol(&self, symbol: &str) -> Option<Greeks> {
        self.symbol_risks
            .read()
            .unwrap()
            .get(symbol)
            .map(|r| Greeks {
                delta: r.total_delta,
                gamma: r.total_gamma,
                theta: r.total_theta,
                vega: r.total_vega,
            })
    }

    /// Get aggregated Greeks for all positions expiring at a specific date
    pub fn greeks_by_expiration(&self, symbol: &str, exp_epoch: u64) -> Option<Greeks> {
        self.expiration_risks
            .read()
            .unwrap()
            .get(&(symbol.to_string(), exp_epoch))
            .map(|r| Greeks {
                delta: r.total_delta,
                gamma: r.total_gamma,
                theta: r.total_theta,
                vega: r.total_vega,
            })
    }

    /// Estimate VaR at confidence level (e.g., 0.95 for 95%)
    /// Uses parametric approach with delta-normal approximation
    pub fn estimate_var(&self, confidence: f64, holding_period_days: u32) -> f64 {
        // Z-score for confidence level
        let z_score = z_at_confidence(confidence);

        // Volatility over holding period
        let horizon_vol = self.assumed_daily_vol * (holding_period_days as f64).sqrt();

        // Delta-normal VaR: exposure * volatility * z-score
        let exposure = self.portfolio_delta.abs() * self.total_notional;

        exposure * horizon_vol * z_score
    }

    /// Estimate margin requirement (simplified)
    /// Reports value without enforcement
    pub fn margin_requirement(&self) -> f64 {
        let var_95 = self.estimate_var(0.95, 1);
        let gamma_adjustment = self.portfolio_gamma.abs() * self.total_notional * 0.01;
        let vega_adjustment =
            self.portfolio_vega.abs() * self.assumed_daily_vol * self.total_notional;

        // Base margin = VaR + gamma risk + vega risk
        var_95 + gamma_adjustment + vega_adjustment
    }

    /// Take a snapshot of current risk state
    pub fn take_snapshot(&mut self, timestamp: u64) {
        let snapshot = RiskSnapshot {
            timestamp,
            portfolio_delta: self.portfolio_delta,
            portfolio_gamma: self.portfolio_gamma,
            portfolio_theta: self.portfolio_theta,
            portfolio_vega: self.portfolio_vega,
            total_notional: self.total_notional,
            var_95: self.estimate_var(0.95, 1),
            margin_requirement: self.margin_requirement(),
        };

        let mut history = self.risk_history.write().unwrap();
        if history.len() >= self.max_history_size {
            history.pop_front();
        }
        history.push_back(snapshot);
    }

    /// Record P&L snapshot
    pub fn record_pnl(&self, timestamp: u64, realized: f64, unrealized: f64) {
        let snapshot = PnLSnapshot {
            timestamp,
            realized_pnl: realized,
            unrealized_pnl: unrealized,
            portfolio_value: self.total_notional + unrealized,
        };

        let mut snapshots = self.pnl_snapshots.write().unwrap();
        if snapshots.len() >= self.max_history_size {
            snapshots.pop_front();
        }
        snapshots.push_back(snapshot);
    }

    /// Get list of all symbols with positions
    pub fn symbols_with_positions(&self) -> Vec<String> {
        self.symbol_risks.read().unwrap().keys().cloned().collect()
    }

    /// Get summary of portfolio risk
    pub fn portfolio_summary(&self) -> PortfolioRiskSummary {
        PortfolioRiskSummary {
            total_positions: self.total_positions,
            total_notional: self.total_notional,
            portfolio_delta: self.portfolio_delta,
            portfolio_gamma: self.portfolio_gamma,
            portfolio_theta: self.portfolio_theta,
            portfolio_vega: self.portfolio_vega,
            var_95: self.estimate_var(0.95, 1),
            margin_requirement: self.margin_requirement(),
        }
    }

    // ==================== Internal Methods ====================

    /// Encode Greeks into a holographic vector for pattern queries
    fn encode_greeks(&self, greeks: &Greeks) -> BinaryHV {
        let mut accumulator = BundleAccumulator::new(DIMENSION);

        // Encode delta: scale [-1, 1] to [0, 2000]
        let delta_shift = ((greeks.delta + 1.0) * 1000.0).clamp(0.0, 2000.0) as usize % 157;
        let delta_hv = self.scalar_bases["delta"].permute_words(delta_shift);
        let bound_delta = self.field_vectors["delta"].bind(&delta_hv);
        accumulator.add(&bound_delta);

        // Encode gamma: log scale
        let gamma_shift = if greeks.gamma < 0.0001 {
            0
        } else {
            ((greeks.gamma.log10() + 4.0) * 75.0).clamp(0.0, 300.0) as usize % 157
        };
        let gamma_hv = self.scalar_bases["gamma"].permute_words(gamma_shift);
        let bound_gamma = self.field_vectors["gamma"].bind(&gamma_hv);
        accumulator.add(&bound_gamma);

        // Encode theta: absolute bucketed
        let theta_shift = if greeks.theta.abs() < 0.01 {
            0
        } else if greeks.theta.abs() < 0.1 {
            50
        } else {
            100
        } % 157;
        let theta_hv = self.scalar_bases["theta"].permute_words(theta_shift);
        let bound_theta = self.field_vectors["theta"].bind(&theta_hv);
        accumulator.add(&bound_theta);

        // Encode vega: linear scale
        let vega_shift = (greeks.vega * 100.0).clamp(0.0, 100.0) as usize % 157;
        let vega_hv = self.scalar_bases["vega"].permute_words(vega_shift);
        let bound_vega = self.field_vectors["vega"].bind(&vega_hv);
        accumulator.add(&bound_vega);

        accumulator.threshold()
    }
}

impl Default for RiskAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Portfolio risk summary struct
#[derive(Debug, Clone)]
pub struct PortfolioRiskSummary {
    pub total_positions: usize,
    pub total_notional: f64,
    pub portfolio_delta: f64,
    pub portfolio_gamma: f64,
    pub portfolio_theta: f64,
    pub portfolio_vega: f64,
    pub var_95: f64,
    pub margin_requirement: f64,
}

/// Z-score lookup for VaR confidence levels
fn z_at_confidence(confidence: f64) -> f64 {
    if confidence >= 0.99 {
        2.326
    } else if confidence >= 0.975 {
        1.96
    } else if confidence >= 0.95 {
        1.645
    } else if confidence >= 0.90 {
        1.282
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_aggregator_basic() {
        let mut aggregator = RiskAggregator::new();

        let greeks = Greeks::new(0.5, 0.02, -0.05, 0.3);
        aggregator.add_position("AAPL", &greeks, 10000.0, 20000);

        assert!((aggregator.portfolio_delta - 0.5).abs() < 0.001);
        assert!((aggregator.delta_by_symbol("AAPL") - 0.5).abs() < 0.001);
        assert_eq!(aggregator.total_positions, 1);
    }

    #[test]
    fn test_risk_aggregator_multiple_positions() {
        let mut aggregator = RiskAggregator::new();

        aggregator.add_position("AAPL", &Greeks::new(0.5, 0.02, -0.05, 0.3), 10000.0, 20000);
        aggregator.add_position("AAPL", &Greeks::new(-0.3, 0.01, -0.03, 0.2), 5000.0, 20000);
        aggregator.add_position("TSLA", &Greeks::new(0.7, 0.03, -0.08, 0.4), 15000.0, 20100);

        // AAPL delta: 0.5 + (-0.3) = 0.2
        assert!((aggregator.delta_by_symbol("AAPL") - 0.2).abs() < 0.001);

        // TSLA delta: 0.7
        assert!((aggregator.delta_by_symbol("TSLA") - 0.7).abs() < 0.001);

        // Portfolio delta: 0.5 - 0.3 + 0.7 = 0.9
        assert!((aggregator.portfolio_delta - 0.9).abs() < 0.001);

        assert_eq!(aggregator.total_positions, 3);
    }

    #[test]
    fn test_var_estimation() {
        let mut aggregator = RiskAggregator::new();

        aggregator.add_position("AAPL", &Greeks::new(0.5, 0.02, -0.05, 0.3), 100000.0, 20000);

        let var_95 = aggregator.estimate_var(0.95, 1);
        assert!(var_95 > 0.0);
        println!("VaR 95%: {}", var_95);
    }

    #[test]
    fn test_snapshot() {
        let mut aggregator = RiskAggregator::new();

        aggregator.add_position("AAPL", &Greeks::new(0.5, 0.02, -0.05, 0.3), 10000.0, 20000);
        aggregator.take_snapshot(1000);

        let history = aggregator.risk_history.read().unwrap();
        assert_eq!(history.len(), 1);
        assert!((history[0].portfolio_delta - 0.5).abs() < 0.001);
    }
}
