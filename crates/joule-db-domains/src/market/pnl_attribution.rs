//! P&L Attribution Engine
//!
//! Decomposes profit and loss into Greek-based components:
//! - Delta P&L: Price movement
//! - Gamma P&L: Convexity
//! - Theta P&L: Time decay
//! - Vega P&L: Volatility changes

use super::Greeks;
use std::collections::{HashMap, VecDeque};

/// P&L decomposed by Greek
#[derive(Debug, Clone, Default)]
pub struct GreekPnL {
    /// P&L from delta (price sensitivity)
    pub delta_pnl: f64,
    /// P&L from gamma (convexity)
    pub gamma_pnl: f64,
    /// P&L from theta (time decay)
    pub theta_pnl: f64,
    /// P&L from vega (volatility)
    pub vega_pnl: f64,
    /// Unexplained P&L (higher order effects)
    pub residual_pnl: f64,
    /// Whether this P&L has been realized (position closed)
    pub realized: bool,
}

impl GreekPnL {
    /// Total P&L across all Greeks
    pub fn total(&self) -> f64 {
        self.delta_pnl + self.gamma_pnl + self.theta_pnl + self.vega_pnl + self.residual_pnl
    }

    /// Create from individual components
    pub fn new(delta: f64, gamma: f64, theta: f64, vega: f64, residual: f64) -> Self {
        Self {
            delta_pnl: delta,
            gamma_pnl: gamma,
            theta_pnl: theta,
            vega_pnl: vega,
            residual_pnl: residual,
            realized: false,
        }
    }

    /// Combine two GreekPnL instances
    pub fn combine(&self, other: &GreekPnL) -> Self {
        Self {
            delta_pnl: self.delta_pnl + other.delta_pnl,
            gamma_pnl: self.gamma_pnl + other.gamma_pnl,
            theta_pnl: self.theta_pnl + other.theta_pnl,
            vega_pnl: self.vega_pnl + other.vega_pnl,
            residual_pnl: self.residual_pnl + other.residual_pnl,
            realized: self.realized && other.realized,
        }
    }
}

/// Mark-to-market snapshot for a position
#[derive(Debug, Clone)]
pub struct MtmSnapshot {
    /// Underlying price at snapshot
    pub spot_price: f64,
    /// Implied volatility at snapshot
    pub implied_vol: f64,
    /// Greeks at snapshot
    pub greeks: Greeks,
    /// Option/position price at snapshot
    pub position_price: f64,
    /// Timestamp (epoch seconds)
    pub timestamp: u64,
}

/// Historical MTM data for a position
#[derive(Debug, Clone)]
pub struct MarkToMarketHistory {
    /// Position identifier (symbol + strike + expiry)
    pub position_id: String,
    /// Entry price
    pub entry_price: f64,
    /// Current position size
    pub quantity: f64,
    /// Is this a long position?
    pub is_long: bool,
    /// MTM snapshots over time
    pub snapshots: VecDeque<MtmSnapshot>,
    /// Accumulated Greek P&L
    pub accumulated_pnl: GreekPnL,
    /// Entry timestamp
    pub entry_timestamp: u64,
}

impl MarkToMarketHistory {
    /// Create new position history
    pub fn new(position_id: &str, entry_price: f64, quantity: f64, is_long: bool) -> Self {
        Self {
            position_id: position_id.to_string(),
            entry_price,
            quantity,
            is_long,
            snapshots: VecDeque::new(),
            accumulated_pnl: GreekPnL::default(),
            entry_timestamp: now_epoch(),
        }
    }

    /// Record an MTM snapshot
    pub fn add_snapshot(&mut self, snapshot: MtmSnapshot) {
        self.snapshots.push_back(snapshot);

        // Limit history to 1000 snapshots
        while self.snapshots.len() > 1000 {
            self.snapshots.pop_front();
        }
    }

    /// Get latest snapshot
    pub fn latest(&self) -> Option<&MtmSnapshot> {
        self.snapshots.back()
    }

    /// Get previous snapshot (for delta calculation)
    pub fn previous(&self) -> Option<&MtmSnapshot> {
        if self.snapshots.len() >= 2 {
            self.snapshots.get(self.snapshots.len() - 2)
        } else {
            None
        }
    }
}

/// Portfolio-level P&L snapshot
#[derive(Debug, Clone)]
pub struct PortfolioPnLSnapshot {
    /// Timestamp
    pub timestamp: u64,
    /// Total P&L by Greek
    pub total_greek_pnl: GreekPnL,
    /// P&L by symbol
    pub by_symbol: HashMap<String, GreekPnL>,
    /// Realized P&L from closed positions
    pub realized_pnl: f64,
    /// Unrealized P&L from open positions
    pub unrealized_pnl: f64,
}

/// P&L Attribution Engine
///
/// Tracks positions and decomposes P&L changes into Greek components.
pub struct PnLAttributionEngine {
    /// Active positions with MTM history
    positions: HashMap<String, MarkToMarketHistory>,

    /// Historical portfolio P&L snapshots
    pnl_history: VecDeque<PortfolioPnLSnapshot>,

    /// Entry prices for closed position tracking
    entry_prices: HashMap<String, (f64, f64)>, // position_id -> (entry_price, quantity)

    /// Cumulative realized P&L
    total_realized_pnl: f64,
}

impl PnLAttributionEngine {
    /// Create a new P&L attribution engine
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            pnl_history: VecDeque::new(),
            entry_prices: HashMap::new(),
            total_realized_pnl: 0.0,
        }
    }

    /// Record a new position entry
    ///
    /// # Arguments
    /// * `position_id` - Unique identifier (e.g., "AAPL_150C_19800")
    /// * `entry_price` - Entry price per unit
    /// * `quantity` - Position size (positive)
    /// * `is_long` - True for long, false for short
    /// * `spot_price` - Underlying price at entry
    /// * `implied_vol` - IV at entry
    /// * `greeks` - Greeks at entry
    pub fn record_entry(
        &mut self,
        position_id: &str,
        entry_price: f64,
        quantity: f64,
        is_long: bool,
        spot_price: f64,
        implied_vol: f64,
        greeks: Greeks,
    ) {
        let mut history = MarkToMarketHistory::new(position_id, entry_price, quantity, is_long);

        // Add initial MTM snapshot
        let snapshot = MtmSnapshot {
            spot_price,
            implied_vol,
            greeks,
            position_price: entry_price,
            timestamp: now_epoch(),
        };
        history.add_snapshot(snapshot);

        self.positions.insert(position_id.to_string(), history);
        self.entry_prices
            .insert(position_id.to_string(), (entry_price, quantity));
    }

    /// Update mark-to-market for a position
    ///
    /// Calculates Greek P&L attribution based on changes since last snapshot.
    pub fn update_mtm(
        &mut self,
        position_id: &str,
        spot_price: f64,
        implied_vol: f64,
        greeks: Greeks,
        position_price: f64,
    ) -> Option<GreekPnL> {
        let history = self.positions.get_mut(position_id)?;

        // Get previous snapshot for comparison
        let prev = history.latest()?.clone();

        // Calculate price moves
        let spot_move = spot_price - prev.spot_price;
        let iv_move = implied_vol - prev.implied_vol;
        let time_elapsed_days = (now_epoch() - prev.timestamp) as f64 / 86400.0;

        // Position direction multiplier
        let direction = if history.is_long { 1.0 } else { -1.0 };
        let qty = history.quantity;

        // Calculate Greek P&L components
        // Delta P&L = delta * spot_move * qty
        let delta_pnl = prev.greeks.delta * spot_move * qty * direction * 100.0; // 100 shares per contract

        // Gamma P&L = 0.5 * gamma * spot_move^2 * qty
        let gamma_pnl = 0.5 * prev.greeks.gamma * spot_move * spot_move * qty * direction * 100.0;

        // Theta P&L = theta * days_elapsed * qty
        let theta_pnl = prev.greeks.theta * time_elapsed_days * qty * direction * 100.0;

        // Vega P&L = vega * iv_change * qty (vega is per 1% IV move)
        let vega_pnl = prev.greeks.vega * (iv_move * 100.0) * qty * direction * 100.0;

        // Actual P&L
        let actual_pnl = (position_price - prev.position_price) * qty * direction * 100.0;

        // Residual = actual - (delta + gamma + theta + vega)
        let explained_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl;
        let residual_pnl = actual_pnl - explained_pnl;

        let greek_pnl = GreekPnL::new(delta_pnl, gamma_pnl, theta_pnl, vega_pnl, residual_pnl);

        // Accumulate P&L
        history.accumulated_pnl = history.accumulated_pnl.combine(&greek_pnl);

        // Add new snapshot
        let snapshot = MtmSnapshot {
            spot_price,
            implied_vol,
            greeks,
            position_price,
            timestamp: now_epoch(),
        };
        history.add_snapshot(snapshot);

        Some(greek_pnl)
    }

    /// Record position exit and realize P&L
    pub fn record_exit(&mut self, position_id: &str, exit_price: f64) -> Option<GreekPnL> {
        let history = self.positions.remove(position_id)?;

        // Calculate total realized P&L
        let direction = if history.is_long { 1.0 } else { -1.0 };
        let total_pnl = (exit_price - history.entry_price) * history.quantity * direction * 100.0;

        self.total_realized_pnl += total_pnl;

        let mut final_pnl = history.accumulated_pnl.clone();
        final_pnl.realized = true;

        // Adjust residual to match actual realized
        let explained =
            final_pnl.delta_pnl + final_pnl.gamma_pnl + final_pnl.theta_pnl + final_pnl.vega_pnl;
        final_pnl.residual_pnl = total_pnl - explained;

        self.entry_prices.remove(position_id);

        Some(final_pnl)
    }

    /// Create a portfolio-level P&L snapshot
    pub fn create_portfolio_snapshot(&self) -> PortfolioPnLSnapshot {
        let mut total_greek_pnl = GreekPnL::default();
        let mut by_symbol: HashMap<String, GreekPnL> = HashMap::new();
        let mut unrealized_pnl = 0.0;

        for (position_id, history) in &self.positions {
            // Extract symbol from position_id (assume format: "SYMBOL_STRIKE_EXPIRY")
            let symbol = position_id.split('_').next().unwrap_or(position_id);

            total_greek_pnl = total_greek_pnl.combine(&history.accumulated_pnl);

            by_symbol
                .entry(symbol.to_string())
                .and_modify(|pnl| *pnl = pnl.combine(&history.accumulated_pnl))
                .or_insert_with(|| history.accumulated_pnl.clone());

            unrealized_pnl += history.accumulated_pnl.total();
        }

        PortfolioPnLSnapshot {
            timestamp: now_epoch(),
            total_greek_pnl,
            by_symbol,
            realized_pnl: self.total_realized_pnl,
            unrealized_pnl,
        }
    }

    /// Get P&L attribution for a specific position
    pub fn get_position_pnl(&self, position_id: &str) -> Option<&GreekPnL> {
        self.positions.get(position_id).map(|h| &h.accumulated_pnl)
    }

    /// Get all active positions
    pub fn active_positions(&self) -> impl Iterator<Item = &str> {
        self.positions.keys().map(|s| s.as_str())
    }

    /// Get total realized P&L
    pub fn total_realized(&self) -> f64 {
        self.total_realized_pnl
    }

    /// Get total unrealized P&L
    pub fn total_unrealized(&self) -> f64 {
        self.positions
            .values()
            .map(|h| h.accumulated_pnl.total())
            .sum()
    }

    /// Get P&L history depth
    pub fn history_depth(&self) -> usize {
        self.pnl_history.len()
    }

    /// Store current snapshot in history
    pub fn checkpoint(&mut self) {
        let snapshot = self.create_portfolio_snapshot();
        self.pnl_history.push_back(snapshot);

        // Keep last 1000 snapshots
        while self.pnl_history.len() > 1000 {
            self.pnl_history.pop_front();
        }
    }

    /// Get position count
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }
}

impl Default for PnLAttributionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current epoch timestamp in seconds
fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnl_entry_exit() {
        let mut engine = PnLAttributionEngine::new();

        // Enter a long call position
        let greeks = Greeks::new(0.5, 0.03, -0.05, 0.25);
        engine.record_entry("AAPL_150C_19800", 5.0, 10.0, true, 150.0, 0.25, greeks);

        assert_eq!(engine.position_count(), 1);
        assert!(engine.active_positions().any(|p| p == "AAPL_150C_19800"));

        // Exit the position with profit
        let final_pnl = engine.record_exit("AAPL_150C_19800", 7.0);
        assert!(final_pnl.is_some());

        let pnl = final_pnl.unwrap();
        assert!(pnl.realized);

        // Position closed
        assert_eq!(engine.position_count(), 0);

        // Realized P&L = (7.0 - 5.0) * 10 * 100 = 2000
        assert!((engine.total_realized() - 2000.0).abs() < 0.01);
    }

    #[test]
    fn test_pnl_attribution_delta() {
        let mut engine = PnLAttributionEngine::new();

        // ATM call with delta = 0.5
        let greeks = Greeks::new(0.5, 0.02, -0.03, 0.20);
        engine.record_entry("SPY_400C_19800", 10.0, 1.0, true, 400.0, 0.20, greeks);

        // Simulate spot move from 400 to 405 (+5)
        // Expected delta P&L = 0.5 * 5 * 1 * 100 = 250
        let new_greeks = Greeks::new(0.55, 0.02, -0.03, 0.20);

        // Note: actual P&L depends on time elapsed. For instant update, mostly delta.
        let pnl = engine.update_mtm("SPY_400C_19800", 405.0, 0.20, new_greeks, 12.5);

        assert!(pnl.is_some());
        let pnl = pnl.unwrap();

        // Delta should dominate for spot-only move
        println!("Delta P&L: {}", pnl.delta_pnl);
        println!("Gamma P&L: {}", pnl.gamma_pnl);
        println!("Total: {}", pnl.total());
    }

    #[test]
    fn test_portfolio_snapshot() {
        let mut engine = PnLAttributionEngine::new();

        // Multiple positions
        engine.record_entry(
            "AAPL_150C_19800",
            5.0,
            10.0,
            true,
            150.0,
            0.25,
            Greeks::new(0.5, 0.03, -0.05, 0.25),
        );
        engine.record_entry(
            "GOOG_2800P_19800",
            15.0,
            5.0,
            true,
            2800.0,
            0.30,
            Greeks::new(-0.45, 0.02, -0.08, 0.40),
        );

        let snapshot = engine.create_portfolio_snapshot();

        assert_eq!(snapshot.by_symbol.len(), 2);
        assert!(snapshot.by_symbol.contains_key("AAPL"));
        assert!(snapshot.by_symbol.contains_key("GOOG"));
    }

    #[test]
    fn test_greek_pnl_combine() {
        let pnl1 = GreekPnL::new(100.0, 10.0, -5.0, 20.0, 2.0);
        let pnl2 = GreekPnL::new(50.0, 5.0, -3.0, 10.0, 1.0);

        let combined = pnl1.combine(&pnl2);

        assert_eq!(combined.delta_pnl, 150.0);
        assert_eq!(combined.gamma_pnl, 15.0);
        assert_eq!(combined.theta_pnl, -8.0);
        assert_eq!(combined.vega_pnl, 30.0);
        assert_eq!(combined.residual_pnl, 3.0);
        assert_eq!(combined.total(), 190.0);
    }
}
