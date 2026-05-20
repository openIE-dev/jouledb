//! Multi-Leg Option Strategy Module
//!
//! Supports complex option strategies like iron condors, butterflies, and spreads.
//! Uses a hybrid architecture with holographic storage for pattern queries
//! and metadata layer for partial fill tracking.

use super::{DIMENSION, Greeks, MarketLink, OptionTrade, OptionType, Side};
use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Types of multi-leg option strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StrategyType {
    /// Bull call spread: Buy lower strike call, sell higher strike call
    BullCallSpread,
    /// Bear call spread: Sell lower strike call, buy higher strike call
    BearCallSpread,
    /// Bull put spread: Sell higher strike put, buy lower strike put
    BullPutSpread,
    /// Bear put spread: Buy higher strike put, sell lower strike put
    BearPutSpread,
    /// Iron condor: 4 legs - sell OTM call/put, buy further OTM call/put
    IronCondor,
    /// Iron butterfly: 4 legs - sell ATM call/put, buy OTM call/put
    IronButterfly,
    /// Butterfly: 3 legs - buy lower, sell 2x middle, buy upper
    Butterfly,
    /// Long straddle: Buy call and put at same strike
    LongStraddle,
    /// Short straddle: Sell call and put at same strike
    ShortStraddle,
    /// Long strangle: Buy OTM call and put
    LongStrangle,
    /// Short strangle: Sell OTM call and put
    ShortStrangle,
    /// Calendar spread: Same strike, different expirations
    CalendarSpread,
    /// Custom multi-leg strategy
    Custom,
}

impl StrategyType {
    /// Expected number of legs for this strategy type
    pub fn expected_legs(&self) -> usize {
        match self {
            StrategyType::BullCallSpread => 2,
            StrategyType::BearCallSpread => 2,
            StrategyType::BullPutSpread => 2,
            StrategyType::BearPutSpread => 2,
            StrategyType::IronCondor => 4,
            StrategyType::IronButterfly => 4,
            StrategyType::Butterfly => 3,
            StrategyType::LongStraddle => 2,
            StrategyType::ShortStraddle => 2,
            StrategyType::LongStrangle => 2,
            StrategyType::ShortStrangle => 2,
            StrategyType::CalendarSpread => 2,
            StrategyType::Custom => 0, // Variable
        }
    }
}

/// A multi-leg option strategy
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiLegStrategy {
    /// Unique strategy identifier (UUID v4)
    pub id: u128,
    /// Type of strategy
    pub strategy_type: StrategyType,
    /// Underlying symbol
    pub symbol: String,
    /// Individual legs of the strategy
    pub legs: Vec<OptionTrade>,
    /// Primary expiration (for single-expiration strategies)
    pub expiration_epoch: u64,
    /// Creation timestamp
    pub created_at: u64,
}

impl MultiLegStrategy {
    /// Create a new strategy with a generated UUID
    pub fn new(
        strategy_type: StrategyType,
        symbol: &str,
        legs: Vec<OptionTrade>,
        expiration_epoch: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().as_u128(),
            strategy_type,
            symbol: symbol.to_string(),
            legs,
            expiration_epoch,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Calculate aggregate Greeks for the strategy
    pub fn aggregate_greeks(&self) -> Greeks {
        let mut total = Greeks::zero();
        for leg in &self.legs {
            // Side determines sign of position
            let multiplier = match leg.side {
                Side::Buy => leg.quantity,
                Side::Sell => -leg.quantity,
            };
            total.delta += leg.greeks.delta * multiplier;
            total.gamma += leg.greeks.gamma * multiplier.abs();
            total.theta += leg.greeks.theta * multiplier;
            total.vega += leg.greeks.vega * multiplier.abs();
        }
        total
    }

    /// Calculate total premium (cost/credit) of the strategy
    pub fn total_premium(&self) -> f64 {
        self.legs
            .iter()
            .map(|leg| {
                match leg.side {
                    Side::Buy => -leg.price * leg.quantity, // Pay premium
                    Side::Sell => leg.price * leg.quantity, // Receive premium
                }
            })
            .sum()
    }

    /// Get maximum profit potential (if calculable)
    pub fn max_profit(&self) -> Option<f64> {
        match self.strategy_type {
            StrategyType::IronCondor
            | StrategyType::IronButterfly
            | StrategyType::ShortStraddle
            | StrategyType::ShortStrangle => {
                // Credit strategies: max profit = net credit received
                let credit = self.total_premium();
                if credit > 0.0 { Some(credit) } else { None }
            }
            _ => None,
        }
    }

    /// Get maximum loss potential (if calculable)
    pub fn max_loss(&self) -> Option<f64> {
        match self.strategy_type {
            StrategyType::BullCallSpread | StrategyType::BearPutSpread => {
                // Debit spreads: max loss = net debit paid
                let debit = -self.total_premium();
                if debit > 0.0 { Some(debit) } else { None }
            }
            StrategyType::IronCondor => {
                // Iron condor: max loss = width of spread - credit
                if self.legs.len() == 4 {
                    let strikes: Vec<f64> = self.legs.iter().map(|l| l.strike).collect();
                    let width = (strikes.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                        - strikes.iter().cloned().fold(f64::INFINITY, f64::min))
                        / 2.0;
                    let credit = self.total_premium();
                    Some(width - credit)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Fill status for a strategy's legs
#[derive(Debug, Clone)]
pub struct StrategyFillStatus {
    /// Which legs have been filled
    pub leg_fills: Vec<bool>,
    /// Partial fill quantities for each leg
    pub partial_quantities: Vec<f64>,
}

impl StrategyFillStatus {
    pub fn new(leg_count: usize) -> Self {
        Self {
            leg_fills: vec![false; leg_count],
            partial_quantities: vec![0.0; leg_count],
        }
    }

    pub fn is_fully_filled(&self) -> bool {
        self.leg_fills.iter().all(|&f| f)
    }

    pub fn filled_leg_count(&self) -> usize {
        self.leg_fills.iter().filter(|&&f| f).count()
    }

    pub fn mark_filled(&mut self, leg_index: usize) {
        if leg_index < self.leg_fills.len() {
            self.leg_fills[leg_index] = true;
        }
    }

    pub fn update_partial(&mut self, leg_index: usize, filled_qty: f64) {
        if leg_index < self.partial_quantities.len() {
            self.partial_quantities[leg_index] = filled_qty;
        }
    }
}

/// Holographic Strategy Book
///
/// Hybrid architecture:
/// - Holographic layer: BundleAccumulator per (symbol, expiration) for fast pattern queries
/// - Metadata layer: HashMap for partial fill tracking and enumeration
pub struct HolographicStrategyBook {
    /// Holographic storage per (symbol, expiration)
    strategy_books: HashMap<(String, u64), BundleAccumulator>,

    /// Strategy metadata with fill status
    strategies: Arc<RwLock<HashMap<u128, (MultiLegStrategy, StrategyFillStatus)>>>,

    /// Index: symbol -> strategy IDs
    symbol_index: Arc<RwLock<HashMap<String, Vec<u128>>>>,

    /// Index: strategy type -> strategy IDs
    type_index: Arc<RwLock<HashMap<StrategyType, Vec<u128>>>>,

    /// Market link encoder
    link: MarketLink,

    /// Pre-computed strategy type vectors
    strategy_type_vectors: HashMap<StrategyType, BinaryHV>,

    /// Position encoding vectors
    position_vectors: Vec<BinaryHV>,
}

impl HolographicStrategyBook {
    pub fn new() -> Self {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);

        // Pre-compute strategy type vectors
        let mut strategy_type_vectors = HashMap::new();
        strategy_type_vectors.insert(
            StrategyType::BullCallSpread,
            BinaryHV::from_hash(b"BULL_CALL_SPREAD", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::BearCallSpread,
            BinaryHV::from_hash(b"BEAR_CALL_SPREAD", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::BullPutSpread,
            BinaryHV::from_hash(b"BULL_PUT_SPREAD", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::BearPutSpread,
            BinaryHV::from_hash(b"BEAR_PUT_SPREAD", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::IronCondor,
            BinaryHV::from_hash(b"IRON_CONDOR", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::IronButterfly,
            BinaryHV::from_hash(b"IRON_BUTTERFLY", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::Butterfly,
            BinaryHV::from_hash(b"BUTTERFLY", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::LongStraddle,
            BinaryHV::from_hash(b"LONG_STRADDLE", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::ShortStraddle,
            BinaryHV::from_hash(b"SHORT_STRADDLE", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::LongStrangle,
            BinaryHV::from_hash(b"LONG_STRANGLE", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::ShortStrangle,
            BinaryHV::from_hash(b"SHORT_STRANGLE", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::CalendarSpread,
            BinaryHV::from_hash(b"CALENDAR_SPREAD", DIMENSION),
        );
        strategy_type_vectors.insert(
            StrategyType::Custom,
            BinaryHV::from_hash(b"CUSTOM", DIMENSION),
        );

        // Pre-compute position vectors (for leg binding)
        let position_vectors: Vec<BinaryHV> = (0..8)
            .map(|i| BinaryHV::random(DIMENSION, rng.random()))
            .collect();

        Self {
            strategy_books: HashMap::new(),
            strategies: Arc::new(RwLock::new(HashMap::new())),
            symbol_index: Arc::new(RwLock::new(HashMap::new())),
            type_index: Arc::new(RwLock::new(HashMap::new())),
            link: MarketLink::new(),
            strategy_type_vectors,
            position_vectors,
        }
    }

    /// Add a strategy to the book
    pub fn add_strategy(&mut self, strategy: MultiLegStrategy) -> u128 {
        let strategy_id = strategy.id;
        let key = (strategy.symbol.clone(), strategy.expiration_epoch);

        // Encode strategy as holographic vector
        let strategy_hv = self.encode_strategy(&strategy);

        // Add to holographic book
        self.strategy_books
            .entry(key)
            .or_insert_with(|| BundleAccumulator::new(DIMENSION))
            .add(&strategy_hv);

        // Add to metadata
        let fill_status = StrategyFillStatus::new(strategy.legs.len());
        {
            let mut strategies = self.strategies.write().unwrap();
            strategies.insert(strategy_id, (strategy.clone(), fill_status));
        }

        // Update symbol index
        {
            let mut symbol_idx = self.symbol_index.write().unwrap();
            symbol_idx
                .entry(strategy.symbol.clone())
                .or_insert_with(Vec::new)
                .push(strategy_id);
        }

        // Update type index
        {
            let mut type_idx = self.type_index.write().unwrap();
            type_idx
                .entry(strategy.strategy_type)
                .or_insert_with(Vec::new)
                .push(strategy_id);
        }

        strategy_id
    }

    /// Remove a strategy from the book
    pub fn remove_strategy(&mut self, strategy_id: u128) -> Option<MultiLegStrategy> {
        let (strategy, _) = {
            let strategies = self.strategies.read().unwrap();
            strategies.get(&strategy_id)?.clone()
        };

        let key = (strategy.symbol.clone(), strategy.expiration_epoch);

        // Encode first to avoid borrow conflict
        let strategy_hv = self.encode_strategy(&strategy);

        // Remove from holographic book
        if let Some(book) = self.strategy_books.get_mut(&key) {
            book.subtract(&strategy_hv);
        }

        // Remove from metadata
        {
            let mut strategies = self.strategies.write().unwrap();
            strategies.remove(&strategy_id);
        }

        // Remove from indices
        {
            let mut symbol_idx = self.symbol_index.write().unwrap();
            if let Some(ids) = symbol_idx.get_mut(&strategy.symbol) {
                ids.retain(|&id| id != strategy_id);
            }
        }

        {
            let mut type_idx = self.type_index.write().unwrap();
            if let Some(ids) = type_idx.get_mut(&strategy.strategy_type) {
                ids.retain(|&id| id != strategy_id);
            }
        }

        Some(strategy)
    }

    /// Handle leg fill event
    pub fn on_leg_fill(&mut self, strategy_id: u128, leg_index: usize) -> bool {
        // Clone strategy and fill status to avoid borrow conflicts
        let (strategy, mut fill_status, key) = {
            let strategies = self.strategies.read().unwrap();
            if let Some((s, fs)) = strategies.get(&strategy_id) {
                let key = (s.symbol.clone(), s.expiration_epoch);
                (s.clone(), fs.clone(), key)
            } else {
                return false;
            }
        };

        // Compute old HV before marking filled
        let old_hv = self.encode_strategy_with_fills(&strategy, &fill_status);

        // Mark leg as filled
        fill_status.mark_filled(leg_index);

        // Compute new HV after marking filled
        let new_hv = self.encode_strategy_with_fills(&strategy, &fill_status);

        let is_fully_filled = fill_status.is_fully_filled();

        // Update the metadata
        {
            let mut strategies = self.strategies.write().unwrap();
            if let Some((_, fs)) = strategies.get_mut(&strategy_id) {
                fs.mark_filled(leg_index);
            }
        }

        // Update the holographic book
        if let Some(book) = self.strategy_books.get_mut(&key) {
            book.subtract(&old_hv);
            book.add(&new_hv);
        }

        is_fully_filled
    }

    /// Get strategy by ID
    pub fn get_strategy(
        &self,
        strategy_id: u128,
    ) -> Option<(MultiLegStrategy, StrategyFillStatus)> {
        let strategies = self.strategies.read().unwrap();
        strategies.get(&strategy_id).cloned()
    }

    /// Query strategies by symbol and expiration
    pub fn find_by_symbol_expiration(&self, symbol: &str, expiration: u64) -> Vec<u128> {
        let strategies = self.strategies.read().unwrap();
        strategies
            .iter()
            .filter(|(_, (s, _))| s.symbol == symbol && s.expiration_epoch == expiration)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Query strategies by type
    pub fn find_by_type(&self, strategy_type: StrategyType) -> Vec<u128> {
        let type_idx = self.type_index.read().unwrap();
        type_idx.get(&strategy_type).cloned().unwrap_or_default()
    }

    /// Query strategies by symbol
    pub fn find_by_symbol(&self, symbol: &str) -> Vec<u128> {
        let symbol_idx = self.symbol_index.read().unwrap();
        symbol_idx.get(symbol).cloned().unwrap_or_default()
    }

    /// Find all iron condors on a symbol
    pub fn find_iron_condors(&self, symbol: &str) -> Vec<u128> {
        let strategies = self.strategies.read().unwrap();
        strategies
            .iter()
            .filter(|(_, (s, _))| s.symbol == symbol && s.strategy_type == StrategyType::IronCondor)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all pending (not fully filled) strategies
    pub fn find_pending_strategies(&self) -> Vec<u128> {
        let strategies = self.strategies.read().unwrap();
        strategies
            .iter()
            .filter(|(_, (_, fill_status))| !fill_status.is_fully_filled())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get count of strategies
    pub fn strategy_count(&self) -> usize {
        self.strategies.read().unwrap().len()
    }

    // ==================== Internal Encoding ====================

    /// Encode a strategy into a holographic vector
    fn encode_strategy(&mut self, strategy: &MultiLegStrategy) -> BinaryHV {
        let mut accumulator = BundleAccumulator::new(DIMENSION);

        // Encode strategy ID
        let id_hv = BinaryHV::from_hash(&strategy.id.to_le_bytes(), DIMENSION);
        accumulator.add(&id_hv);

        // Encode strategy type
        if let Some(type_hv) = self.strategy_type_vectors.get(&strategy.strategy_type) {
            accumulator.add(type_hv);
        }

        // Encode each leg with position binding
        for (pos, leg) in strategy.legs.iter().enumerate() {
            let leg_hv = self.link.encode_option_trade(leg);
            if pos < self.position_vectors.len() {
                let bound_leg = leg_hv.bind(&self.position_vectors[pos]);
                accumulator.add(&bound_leg);
            } else {
                accumulator.add(&leg_hv);
            }
        }

        accumulator.threshold()
    }

    /// Encode strategy with fill status markers
    fn encode_strategy_with_fills(
        &mut self,
        strategy: &MultiLegStrategy,
        fill_status: &StrategyFillStatus,
    ) -> BinaryHV {
        let mut accumulator = BundleAccumulator::new(DIMENSION);

        // Encode strategy ID
        let id_hv = BinaryHV::from_hash(&strategy.id.to_le_bytes(), DIMENSION);
        accumulator.add(&id_hv);

        // Encode strategy type
        if let Some(type_hv) = self.strategy_type_vectors.get(&strategy.strategy_type) {
            accumulator.add(type_hv);
        }

        // Encode fill status (number of filled legs)
        let fill_count_hv =
            BinaryHV::from_hash(&fill_status.filled_leg_count().to_le_bytes(), DIMENSION);
        accumulator.add(&fill_count_hv);

        // Encode each leg with position binding
        for (pos, leg) in strategy.legs.iter().enumerate() {
            let leg_hv = self.link.encode_option_trade(leg);
            if pos < self.position_vectors.len() {
                let bound_leg = leg_hv.bind(&self.position_vectors[pos]);
                accumulator.add(&bound_leg);
            }
        }

        accumulator.threshold()
    }
}

impl Default for HolographicStrategyBook {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Strategy Builders ====================

/// Builder for creating common option strategies
pub struct StrategyBuilder;

impl StrategyBuilder {
    /// Create a bull call spread
    pub fn bull_call_spread(
        symbol: &str,
        lower_strike: f64,
        upper_strike: f64,
        expiration: u64,
        quantity: f64,
        lower_premium: f64,
        upper_premium: f64,
        lower_greeks: Greeks,
        upper_greeks: Greeks,
    ) -> MultiLegStrategy {
        let legs = vec![
            OptionTrade {
                symbol: symbol.to_string(),
                price: lower_premium,
                quantity,
                side: Side::Buy,
                strike: lower_strike,
                option_type: OptionType::Call,
                expiration_epoch: expiration,
                greeks: lower_greeks,
            },
            OptionTrade {
                symbol: symbol.to_string(),
                price: upper_premium,
                quantity,
                side: Side::Sell,
                strike: upper_strike,
                option_type: OptionType::Call,
                expiration_epoch: expiration,
                greeks: upper_greeks,
            },
        ];

        MultiLegStrategy::new(StrategyType::BullCallSpread, symbol, legs, expiration)
    }

    /// Create an iron condor
    pub fn iron_condor(
        symbol: &str,
        put_buy_strike: f64,
        put_sell_strike: f64,
        call_sell_strike: f64,
        call_buy_strike: f64,
        expiration: u64,
        quantity: f64,
        premiums: [f64; 4],
        greeks: [Greeks; 4],
    ) -> MultiLegStrategy {
        let legs = vec![
            // Long put (protection)
            OptionTrade {
                symbol: symbol.to_string(),
                price: premiums[0],
                quantity,
                side: Side::Buy,
                strike: put_buy_strike,
                option_type: OptionType::Put,
                expiration_epoch: expiration,
                greeks: greeks[0],
            },
            // Short put (credit)
            OptionTrade {
                symbol: symbol.to_string(),
                price: premiums[1],
                quantity,
                side: Side::Sell,
                strike: put_sell_strike,
                option_type: OptionType::Put,
                expiration_epoch: expiration,
                greeks: greeks[1],
            },
            // Short call (credit)
            OptionTrade {
                symbol: symbol.to_string(),
                price: premiums[2],
                quantity,
                side: Side::Sell,
                strike: call_sell_strike,
                option_type: OptionType::Call,
                expiration_epoch: expiration,
                greeks: greeks[2],
            },
            // Long call (protection)
            OptionTrade {
                symbol: symbol.to_string(),
                price: premiums[3],
                quantity,
                side: Side::Buy,
                strike: call_buy_strike,
                option_type: OptionType::Call,
                expiration_epoch: expiration,
                greeks: greeks[3],
            },
        ];

        MultiLegStrategy::new(StrategyType::IronCondor, symbol, legs, expiration)
    }

    /// Create a long straddle
    pub fn long_straddle(
        symbol: &str,
        strike: f64,
        expiration: u64,
        quantity: f64,
        call_premium: f64,
        put_premium: f64,
        call_greeks: Greeks,
        put_greeks: Greeks,
    ) -> MultiLegStrategy {
        let legs = vec![
            OptionTrade {
                symbol: symbol.to_string(),
                price: call_premium,
                quantity,
                side: Side::Buy,
                strike,
                option_type: OptionType::Call,
                expiration_epoch: expiration,
                greeks: call_greeks,
            },
            OptionTrade {
                symbol: symbol.to_string(),
                price: put_premium,
                quantity,
                side: Side::Buy,
                strike,
                option_type: OptionType::Put,
                expiration_epoch: expiration,
                greeks: put_greeks,
            },
        ];

        MultiLegStrategy::new(StrategyType::LongStraddle, symbol, legs, expiration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_creation() {
        let strategy = StrategyBuilder::bull_call_spread(
            "AAPL",
            150.0,
            160.0,
            20000,
            10.0,
            5.0,
            2.0,
            Greeks::new(0.6, 0.02, -0.05, 0.3),
            Greeks::new(0.4, 0.015, -0.04, 0.25),
        );

        assert_eq!(strategy.strategy_type, StrategyType::BullCallSpread);
        assert_eq!(strategy.legs.len(), 2);
        assert_eq!(strategy.symbol, "AAPL");
    }

    #[test]
    fn test_iron_condor() {
        let strategy = StrategyBuilder::iron_condor(
            "SPY",
            430.0, // long put
            440.0, // short put
            460.0, // short call
            470.0, // long call
            20000,
            5.0,
            [1.5, 3.0, 3.5, 1.0],
            [
                Greeks::new(-0.15, 0.01, -0.02, 0.1),
                Greeks::new(-0.30, 0.02, -0.04, 0.2),
                Greeks::new(0.35, 0.02, -0.04, 0.2),
                Greeks::new(0.20, 0.01, -0.02, 0.1),
            ],
        );

        assert_eq!(strategy.strategy_type, StrategyType::IronCondor);
        assert_eq!(strategy.legs.len(), 4);

        // Net credit should be positive (short premiums > long premiums)
        let credit = strategy.total_premium();
        println!("Iron condor net credit: {}", credit);
        assert!(credit > 0.0);
    }

    #[test]
    fn test_strategy_book() {
        let mut book = HolographicStrategyBook::new();

        let strategy = StrategyBuilder::bull_call_spread(
            "AAPL",
            150.0,
            160.0,
            20000,
            10.0,
            5.0,
            2.0,
            Greeks::new(0.6, 0.02, -0.05, 0.3),
            Greeks::new(0.4, 0.015, -0.04, 0.25),
        );

        let id = book.add_strategy(strategy);
        assert_eq!(book.strategy_count(), 1);

        // Test retrieval
        let (retrieved, _) = book.get_strategy(id).unwrap();
        assert_eq!(retrieved.symbol, "AAPL");

        // Test query by symbol
        let found = book.find_by_symbol("AAPL");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], id);
    }

    #[test]
    fn test_partial_fills() {
        let mut book = HolographicStrategyBook::new();

        let strategy = StrategyBuilder::iron_condor(
            "SPY",
            430.0,
            440.0,
            460.0,
            470.0,
            20000,
            5.0,
            [1.5, 3.0, 3.5, 1.0],
            [
                Greeks::new(-0.15, 0.01, -0.02, 0.1),
                Greeks::new(-0.30, 0.02, -0.04, 0.2),
                Greeks::new(0.35, 0.02, -0.04, 0.2),
                Greeks::new(0.20, 0.01, -0.02, 0.1),
            ],
        );

        let id = book.add_strategy(strategy);

        // Fill legs one by one
        assert!(!book.on_leg_fill(id, 0)); // Not fully filled yet
        assert!(!book.on_leg_fill(id, 1));
        assert!(!book.on_leg_fill(id, 2));
        assert!(book.on_leg_fill(id, 3)); // Now fully filled

        // Verify pending strategies
        let pending = book.find_pending_strategies();
        assert!(pending.is_empty());
    }
}
