//! # Auction Market
//!
//! Implements opening and closing auction mechanisms with indicative price
//! calculation, order imbalance tracking, price determination via uncrossing,
//! extension rules, and collar logic. Supports call auctions with multiple
//! phases: order entry, price determination, and allocation.

use std::fmt;
use std::collections::VecDeque;

// ── Core Types ──

/// Side of an auction order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AuctionSide {
    Buy,
    Sell,
}

impl fmt::Display for AuctionSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuctionSide::Buy => write!(f, "BUY"),
            AuctionSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Type of auction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuctionType {
    Opening,
    Closing,
    Intraday,
    Volatility,
}

impl fmt::Display for AuctionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AuctionType::Opening => "OPENING",
            AuctionType::Closing => "CLOSING",
            AuctionType::Intraday => "INTRADAY",
            AuctionType::Volatility => "VOLATILITY",
        };
        write!(f, "{s}")
    }
}

/// Phase of the auction lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuctionPhase {
    /// Accepting new orders and modifications.
    OrderEntry,
    /// Frozen — calculating price.
    PriceDetermination,
    /// Allocation in progress.
    Allocation,
    /// Auction complete.
    Complete,
    /// Extended due to price collar or imbalance.
    Extended,
}

impl fmt::Display for AuctionPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AuctionPhase::OrderEntry => "ENTRY",
            AuctionPhase::PriceDetermination => "PRICE_CALC",
            AuctionPhase::Allocation => "ALLOCATION",
            AuctionPhase::Complete => "COMPLETE",
            AuctionPhase::Extended => "EXTENDED",
        };
        write!(f, "{s}")
    }
}

// ── Auction Order ──

/// An order submitted to the auction.
#[derive(Clone, Debug)]
pub struct AuctionOrder {
    pub id: u64,
    pub symbol: String,
    pub side: AuctionSide,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub limit_price: f64,
    pub is_market: bool,
    pub timestamp_ns: u64,
    pub is_active: bool,
}

impl AuctionOrder {
    pub fn market(id: u64, symbol: &str, side: AuctionSide, quantity: f64) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            side,
            quantity,
            filled_quantity: 0.0,
            limit_price: 0.0,
            is_market: true,
            timestamp_ns: 0,
            is_active: true,
        }
    }

    pub fn limit(id: u64, symbol: &str, side: AuctionSide, quantity: f64, price: f64) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            side,
            quantity,
            filled_quantity: 0.0,
            limit_price: price,
            is_market: false,
            timestamp_ns: 0,
            is_active: true,
        }
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp_ns = ts;
        self
    }

    pub fn remaining(&self) -> f64 {
        (self.quantity - self.filled_quantity).max(0.0)
    }

    /// Whether this order is eligible at a given auction price.
    pub fn eligible_at(&self, price: f64) -> bool {
        if !self.is_active || self.remaining() < 1e-12 { return false; }
        if self.is_market { return true; }
        match self.side {
            AuctionSide::Buy => price <= self.limit_price,
            AuctionSide::Sell => price >= self.limit_price,
        }
    }
}

impl fmt::Display for AuctionOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ptype = if self.is_market { "MKT" } else { "LMT" };
        write!(
            f,
            "AuctionOrder(#{} {} {} {} qty={:.2} limit={:.2})",
            self.id, ptype, self.side, self.symbol,
            self.quantity, self.limit_price,
        )
    }
}

// ── Indicative Data ──

/// Indicative auction data published during order entry.
#[derive(Clone, Debug)]
pub struct IndicativeData {
    pub indicative_price: f64,
    pub indicative_volume: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub imbalance: f64,
    pub imbalance_side: Option<AuctionSide>,
    pub far_price: f64,
    pub near_price: f64,
}

impl IndicativeData {
    pub fn empty() -> Self {
        Self {
            indicative_price: 0.0,
            indicative_volume: 0.0,
            buy_volume: 0.0,
            sell_volume: 0.0,
            imbalance: 0.0,
            imbalance_side: None,
            far_price: 0.0,
            near_price: 0.0,
        }
    }

    pub fn imbalance_pct(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total > 1e-12 {
            (self.imbalance.abs() / total) * 100.0
        } else {
            0.0
        }
    }

    pub fn paired_volume(&self) -> f64 {
        self.buy_volume.min(self.sell_volume)
    }
}

impl fmt::Display for IndicativeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Indicative(price={:.4} vol={:.2} imbal={:.2} imbal%={:.1}%)",
            self.indicative_price, self.indicative_volume,
            self.imbalance, self.imbalance_pct(),
        )
    }
}

// ── Auction Fill ──

/// A fill resulting from the auction.
#[derive(Clone, Debug)]
pub struct AuctionFill {
    pub order_id: u64,
    pub side: AuctionSide,
    pub price: f64,
    pub quantity: f64,
}

impl AuctionFill {
    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }
}

impl fmt::Display for AuctionFill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AuctionFill(#{} {} {:.4}@{:.4})",
            self.order_id, self.side, self.quantity, self.price,
        )
    }
}

// ── Auction Result ──

/// Complete result of an auction.
#[derive(Clone, Debug)]
pub struct AuctionResult {
    pub auction_type: AuctionType,
    pub auction_price: f64,
    pub total_volume: f64,
    pub fills: Vec<AuctionFill>,
    pub was_extended: bool,
    pub extensions: u32,
    pub buy_surplus: f64,
    pub sell_surplus: f64,
    pub timestamp_ns: u64,
}

impl AuctionResult {
    pub fn fill_count(&self) -> usize {
        self.fills.len()
    }

    pub fn total_notional(&self) -> f64 {
        self.auction_price * self.total_volume
    }

    pub fn imbalance(&self) -> f64 {
        self.buy_surplus - self.sell_surplus
    }
}

impl fmt::Display for AuctionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AuctionResult({} price={:.4} vol={:.2} fills={} extended={})",
            self.auction_type, self.auction_price, self.total_volume,
            self.fills.len(), self.was_extended,
        )
    }
}

// ── Collar Configuration ──

/// Price collar configuration for extension rules.
#[derive(Clone, Debug)]
pub struct CollarConfig {
    pub reference_price: f64,
    pub upper_pct: f64,
    pub lower_pct: f64,
    pub max_extensions: u32,
    pub extension_duration_ns: u64,
}

impl CollarConfig {
    pub fn new(reference: f64, band_pct: f64) -> Self {
        Self {
            reference_price: reference,
            upper_pct: band_pct,
            lower_pct: band_pct,
            max_extensions: 3,
            extension_duration_ns: 300_000_000_000, // 5 min.
        }
    }

    pub fn with_asymmetric(mut self, upper: f64, lower: f64) -> Self {
        self.upper_pct = upper;
        self.lower_pct = lower;
        self
    }

    pub fn with_max_extensions(mut self, n: u32) -> Self {
        self.max_extensions = n;
        self
    }

    pub fn upper_limit(&self) -> f64 {
        self.reference_price * (1.0 + self.upper_pct / 100.0)
    }

    pub fn lower_limit(&self) -> f64 {
        self.reference_price * (1.0 - self.lower_pct / 100.0)
    }

    pub fn within_collar(&self, price: f64) -> bool {
        price >= self.lower_limit() && price <= self.upper_limit()
    }
}

impl fmt::Display for CollarConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Collar(ref={:.2} [{:.2},{:.2}] max_ext={})",
            self.reference_price, self.lower_limit(),
            self.upper_limit(), self.max_extensions,
        )
    }
}

// ── Auction Market ──

/// The auction market engine.
#[derive(Clone, Debug)]
pub struct AuctionMarket {
    pub symbol: String,
    pub auction_type: AuctionType,
    pub phase: AuctionPhase,
    buys: VecDeque<AuctionOrder>,
    sells: VecDeque<AuctionOrder>,
    pub collar: Option<CollarConfig>,
    pub extensions: u32,
    pub results: Vec<AuctionResult>,
}

impl AuctionMarket {
    pub fn new(symbol: &str, auction_type: AuctionType) -> Self {
        Self {
            symbol: symbol.to_string(),
            auction_type,
            phase: AuctionPhase::OrderEntry,
            buys: VecDeque::new(),
            sells: VecDeque::new(),
            collar: None,
            extensions: 0,
            results: Vec::new(),
        }
    }

    pub fn with_collar(mut self, collar: CollarConfig) -> Self {
        self.collar = Some(collar);
        self
    }

    /// Submit an order during the entry phase.
    pub fn submit(&mut self, order: AuctionOrder) -> bool {
        if self.phase != AuctionPhase::OrderEntry && self.phase != AuctionPhase::Extended {
            return false;
        }
        match order.side {
            AuctionSide::Buy => self.buys.push_back(order),
            AuctionSide::Sell => self.sells.push_back(order),
        }
        true
    }

    /// Cancel an order.
    pub fn cancel(&mut self, order_id: u64) -> bool {
        if self.phase != AuctionPhase::OrderEntry && self.phase != AuctionPhase::Extended {
            return false;
        }
        for o in self.buys.iter_mut().chain(self.sells.iter_mut()) {
            if o.id == order_id && o.is_active {
                o.is_active = false;
                return true;
            }
        }
        false
    }

    /// Calculate indicative data.
    pub fn indicative(&self) -> IndicativeData {
        let prices = self.candidate_prices();
        if prices.is_empty() {
            return IndicativeData::empty();
        }

        let mut best_price = 0.0;
        let mut best_volume = 0.0_f64;
        let mut best_imbalance = f64::MAX;

        for &price in &prices {
            let buy_vol = self.eligible_volume(AuctionSide::Buy, price);
            let sell_vol = self.eligible_volume(AuctionSide::Sell, price);
            let matched = buy_vol.min(sell_vol);
            let imbal = (buy_vol - sell_vol).abs();

            if matched > best_volume || (matched == best_volume && imbal < best_imbalance) {
                best_volume = matched;
                best_price = price;
                best_imbalance = imbal;
            }
        }

        let buy_vol = self.eligible_volume(AuctionSide::Buy, best_price);
        let sell_vol = self.eligible_volume(AuctionSide::Sell, best_price);
        let imbal = buy_vol - sell_vol;

        IndicativeData {
            indicative_price: best_price,
            indicative_volume: best_volume,
            buy_volume: buy_vol,
            sell_volume: sell_vol,
            imbalance: imbal,
            imbalance_side: if imbal > 1e-12 {
                Some(AuctionSide::Buy)
            } else if imbal < -1e-12 {
                Some(AuctionSide::Sell)
            } else {
                None
            },
            far_price: *prices.last().unwrap_or(&0.0),
            near_price: *prices.first().unwrap_or(&0.0),
        }
    }

    /// Execute the auction.
    pub fn execute(&mut self, timestamp_ns: u64) -> Option<AuctionResult> {
        self.phase = AuctionPhase::PriceDetermination;

        let ind = self.indicative();
        if ind.indicative_volume < 1e-12 {
            self.phase = AuctionPhase::Complete;
            return None;
        }

        let price = ind.indicative_price;

        // Check collar.
        if let Some(collar) = &self.collar {
            if !collar.within_collar(price) && self.extensions < collar.max_extensions {
                self.extensions += 1;
                self.phase = AuctionPhase::Extended;
                return None;
            }
        }

        self.phase = AuctionPhase::Allocation;

        let matched_volume = ind.indicative_volume;
        let mut fills = Vec::new();

        // Capture eligible volume before allocation modifies remaining quantities.
        let buy_total = self.eligible_volume(AuctionSide::Buy, price);
        let sell_total = self.eligible_volume(AuctionSide::Sell, price);

        // Allocate buys.
        self.allocate_side(AuctionSide::Buy, price, matched_volume, &mut fills);
        // Allocate sells.
        self.allocate_side(AuctionSide::Sell, price, matched_volume, &mut fills);

        let result = AuctionResult {
            auction_type: self.auction_type,
            auction_price: price,
            total_volume: matched_volume,
            fills,
            was_extended: self.extensions > 0,
            extensions: self.extensions,
            buy_surplus: (buy_total - matched_volume).max(0.0),
            sell_surplus: (sell_total - matched_volume).max(0.0),
            timestamp_ns,
        };

        self.phase = AuctionPhase::Complete;
        self.results.push(result.clone());
        Some(result)
    }

    fn allocate_side(
        &mut self,
        side: AuctionSide,
        price: f64,
        total: f64,
        fills: &mut Vec<AuctionFill>,
    ) {
        let orders = match side {
            AuctionSide::Buy => &mut self.buys,
            AuctionSide::Sell => &mut self.sells,
        };

        let eligible_qty: f64 = orders.iter()
            .filter(|o| o.eligible_at(price))
            .map(|o| o.remaining())
            .sum();

        if eligible_qty < 1e-12 { return; }
        let mut left = total;

        for order in orders.iter_mut() {
            if !order.eligible_at(price) || left < 1e-12 { continue; }
            let share = order.remaining() / eligible_qty;
            let alloc = (total * share).min(order.remaining()).min(left);
            if alloc < 1e-12 { continue; }
            order.filled_quantity += alloc;
            left -= alloc;
            fills.push(AuctionFill {
                order_id: order.id,
                side: order.side,
                price,
                quantity: alloc,
            });
        }
    }

    fn eligible_volume(&self, side: AuctionSide, price: f64) -> f64 {
        let orders = match side {
            AuctionSide::Buy => &self.buys,
            AuctionSide::Sell => &self.sells,
        };
        orders.iter()
            .filter(|o| o.eligible_at(price))
            .map(|o| o.remaining())
            .sum()
    }

    fn candidate_prices(&self) -> Vec<f64> {
        let mut prices: Vec<f64> = Vec::new();
        for o in self.buys.iter().chain(self.sells.iter()) {
            if !o.is_market && o.is_active && o.limit_price > 1e-12 {
                prices.push(o.limit_price);
            }
        }
        prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        prices.dedup();
        prices
    }

    pub fn buy_interest(&self) -> f64 {
        self.buys.iter().filter(|o| o.is_active).map(|o| o.remaining()).sum()
    }

    pub fn sell_interest(&self) -> f64 {
        self.sells.iter().filter(|o| o.is_active).map(|o| o.remaining()).sum()
    }

    pub fn order_count(&self) -> usize {
        self.buys.iter().chain(self.sells.iter()).filter(|o| o.is_active).count()
    }

    /// Reset for a new auction session.
    pub fn reset(&mut self) {
        self.buys.clear();
        self.sells.clear();
        self.phase = AuctionPhase::OrderEntry;
        self.extensions = 0;
    }
}

impl fmt::Display for AuctionMarket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AuctionMarket({} {} phase={} buys={:.0} sells={:.0} ext={})",
            self.symbol, self.auction_type, self.phase,
            self.buy_interest(), self.sell_interest(), self.extensions,
        )
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn market() -> AuctionMarket {
        AuctionMarket::new("AAPL", AuctionType::Opening)
    }

    #[test]
    fn test_submit_order() {
        let mut m = market();
        assert!(m.submit(AuctionOrder::market(1, "AAPL", AuctionSide::Buy, 100.0)));
        assert_eq!(m.order_count(), 1);
    }

    #[test]
    fn test_simple_auction() {
        let mut m = market();
        m.submit(AuctionOrder::limit(1, "AAPL", AuctionSide::Buy, 100.0, 150.0));
        m.submit(AuctionOrder::limit(2, "AAPL", AuctionSide::Sell, 100.0, 150.0));
        let result = m.execute(0).unwrap();
        assert!((result.auction_price - 150.0).abs() < 1e-9);
        assert!((result.total_volume - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_price_maximises_volume() {
        let mut m = market();
        m.submit(AuctionOrder::limit(1, "AAPL", AuctionSide::Buy, 100.0, 152.0));
        m.submit(AuctionOrder::limit(2, "AAPL", AuctionSide::Buy, 50.0, 150.0));
        m.submit(AuctionOrder::limit(3, "AAPL", AuctionSide::Sell, 120.0, 150.0));
        m.submit(AuctionOrder::limit(4, "AAPL", AuctionSide::Sell, 30.0, 152.0));
        let result = m.execute(0).unwrap();
        // At 150: buy=150, sell=150 → matched 150.
        // At 152: buy=100, sell=150 → matched 100.
        // Price 150 should win.
        assert!((result.auction_price - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_market_orders_always_eligible() {
        let mut m = market();
        m.submit(AuctionOrder::market(1, "AAPL", AuctionSide::Buy, 100.0));
        m.submit(AuctionOrder::limit(2, "AAPL", AuctionSide::Sell, 100.0, 155.0));
        let result = m.execute(0).unwrap();
        assert!((result.auction_price - 155.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalanced_auction() {
        let mut m = market();
        m.submit(AuctionOrder::limit(1, "AAPL", AuctionSide::Buy, 200.0, 150.0));
        m.submit(AuctionOrder::limit(2, "AAPL", AuctionSide::Sell, 100.0, 150.0));
        let result = m.execute(0).unwrap();
        assert!((result.total_volume - 100.0).abs() < 1e-9);
        assert!((result.buy_surplus - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_indicative_data() {
        let mut m = market();
        m.submit(AuctionOrder::limit(1, "AAPL", AuctionSide::Buy, 100.0, 150.0));
        m.submit(AuctionOrder::limit(2, "AAPL", AuctionSide::Sell, 80.0, 150.0));
        let ind = m.indicative();
        assert!((ind.indicative_price - 150.0).abs() < 1e-9);
        assert!((ind.indicative_volume - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_indicative_imbalance() {
        let mut m = market();
        m.submit(AuctionOrder::limit(1, "AAPL", AuctionSide::Buy, 200.0, 150.0));
        m.submit(AuctionOrder::limit(2, "AAPL", AuctionSide::Sell, 100.0, 150.0));
        let ind = m.indicative();
        assert!((ind.imbalance - 100.0).abs() < 1e-9);
        assert_eq!(ind.imbalance_side, Some(AuctionSide::Buy));
    }

    #[test]
    fn test_collar_extension() {
        let collar = CollarConfig::new(100.0, 5.0);
        let mut m = AuctionMarket::new("AAPL", AuctionType::Opening).with_collar(collar);
        // These prices are outside the 95-105 collar.
        m.submit(AuctionOrder::limit(1, "AAPL", AuctionSide::Buy, 100.0, 120.0));
        m.submit(AuctionOrder::limit(2, "AAPL", AuctionSide::Sell, 100.0, 120.0));
        let result = m.execute(0);
        assert!(result.is_none());
        assert_eq!(m.phase, AuctionPhase::Extended);
        assert_eq!(m.extensions, 1);
    }

    #[test]
    fn test_collar_within_band() {
        let collar = CollarConfig::new(150.0, 10.0);
        assert!(collar.within_collar(150.0));
        assert!(collar.within_collar(155.0));
        assert!(!collar.within_collar(170.0));
    }

    #[test]
    fn test_collar_limits() {
        let collar = CollarConfig::new(100.0, 5.0);
        assert!((collar.upper_limit() - 105.0).abs() < 1e-9);
        assert!((collar.lower_limit() - 95.0).abs() < 1e-9);
    }

    #[test]
    fn test_cancel() {
        let mut m = market();
        m.submit(AuctionOrder::limit(1, "AAPL", AuctionSide::Buy, 100.0, 150.0));
        assert!(m.cancel(1));
        assert_eq!(m.order_count(), 0);
    }

    #[test]
    fn test_no_match_empty() {
        let mut m = market();
        let result = m.execute(0);
        assert!(result.is_none());
    }

    #[test]
    fn test_reset() {
        let mut m = market();
        m.submit(AuctionOrder::limit(1, "AAPL", AuctionSide::Buy, 100.0, 150.0));
        m.execute(0);
        m.reset();
        assert_eq!(m.phase, AuctionPhase::OrderEntry);
        assert_eq!(m.order_count(), 0);
    }

    #[test]
    fn test_auction_result_display() {
        let r = AuctionResult {
            auction_type: AuctionType::Opening,
            auction_price: 150.0,
            total_volume: 100.0,
            fills: Vec::new(),
            was_extended: false,
            extensions: 0,
            buy_surplus: 0.0,
            sell_surplus: 0.0,
            timestamp_ns: 0,
        };
        let s = format!("{r}");
        assert!(s.contains("OPENING"));
        assert!(s.contains("150.0"));
    }

    #[test]
    fn test_indicative_display() {
        let ind = IndicativeData::empty();
        let s = format!("{ind}");
        assert!(s.contains("Indicative"));
    }

    #[test]
    fn test_market_display() {
        let m = market();
        let s = format!("{m}");
        assert!(s.contains("AuctionMarket"));
        assert!(s.contains("AAPL"));
    }

    #[test]
    fn test_collar_display() {
        let c = CollarConfig::new(100.0, 5.0);
        let s = format!("{c}");
        assert!(s.contains("Collar"));
    }

    #[test]
    fn test_auction_fill_notional() {
        let f = AuctionFill {
            order_id: 1, side: AuctionSide::Buy, price: 150.0, quantity: 100.0,
        };
        assert!((f.notional() - 15000.0).abs() < 1e-9);
    }

    #[test]
    fn test_paired_volume() {
        let ind = IndicativeData {
            indicative_price: 100.0,
            indicative_volume: 80.0,
            buy_volume: 120.0,
            sell_volume: 80.0,
            imbalance: 40.0,
            imbalance_side: Some(AuctionSide::Buy),
            far_price: 0.0,
            near_price: 0.0,
        };
        assert!((ind.paired_volume() - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_closing_auction_type() {
        let m = AuctionMarket::new("AAPL", AuctionType::Closing);
        assert_eq!(m.auction_type, AuctionType::Closing);
    }

    #[test]
    fn test_result_imbalance() {
        let r = AuctionResult {
            auction_type: AuctionType::Opening,
            auction_price: 100.0,
            total_volume: 100.0,
            fills: Vec::new(),
            was_extended: false,
            extensions: 0,
            buy_surplus: 50.0,
            sell_surplus: 10.0,
            timestamp_ns: 0,
        };
        assert!((r.imbalance() - 40.0).abs() < 1e-9);
    }
}
