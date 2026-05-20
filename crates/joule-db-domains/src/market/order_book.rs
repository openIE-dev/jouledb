use super::{DIMENSION, MarketLink, Side, Trade};
use joule_db_hdc::BundleAccumulator;

/// A Constant-Time Holographic Order Book
///
/// Stores thousands of active orders in constant memory.
/// Operations (Add, Cancel) are O(1).
pub struct HolographicOrderBook {
    /// Superposition of all active Bids
    pub bids: BundleAccumulator,
    /// Superposition of all active Asks
    pub asks: BundleAccumulator,
    /// Helper to encode prices for queries
    pub link: MarketLink,
}

impl HolographicOrderBook {
    pub fn new() -> Self {
        Self {
            bids: BundleAccumulator::new(DIMENSION),
            asks: BundleAccumulator::new(DIMENSION),
            link: MarketLink::new(),
        }
    }

    /// Add a trade (Limit Order) to the book O(1)
    pub fn add(&mut self, trade: &Trade) {
        let order_hv = self.link.encode_trade(trade);

        match trade.side {
            Side::Buy => self.bids.add(&order_hv),
            Side::Sell => self.asks.add(&order_hv),
        }
    }

    /// Cancel/Fill an order (Remove from book) O(1)
    pub fn remove(&mut self, trade: &Trade) {
        let order_hv = self.link.encode_trade(trade);

        match trade.side {
            Side::Buy => self.bids.subtract(&order_hv),
            Side::Sell => self.asks.subtract(&order_hv),
        }
    }

    /// Query Liquidity at a specific price level O(1)
    ///
    /// Returns "Liquidity Density" (0.0 - 1.0) indicating presence of orders at this price.
    /// 1.0 = High confidence/volume at this price.
    /// 0.0 = No orders (noise floor).
    pub fn query_liquidity_at(&self, price: f64, side: Side) -> f32 {
        // Construct a probe vector representing "Price P"
        let probe = self.link.encode_price_term(price);

        // Get the consensus state of the book side
        let book_hologram = match side {
            Side::Buy => self.bids.threshold(),
            Side::Sell => self.asks.threshold(),
        };

        // Measure similarity.
        // If many orders have this Price Term, the book hologram will correlate strongly.
        // Result is 0.5 (random) to 1.0 (perfect match).
        // We normalize to 0.0 - 1.0 range (where 0.0 is random noise).
        let raw_sim = book_hologram.similarity(&probe);

        // Normalize: (Sim - 0.5) * 2.0. If Sim < 0.5, clamp to 0.
        ((raw_sim - 0.5) * 2.0).max(0.0)
    }
}
