//! Book Depth — market depth calculation, L2/L3 data construction,
//! depth visualization data, and cumulative volume analysis.
//!
//! Pure-Rust market depth engine producing L2 (price-aggregated) and
//! L3 (order-level) depth snapshots, cumulative volume curves, and
//! depth imbalance metrics for order book visualization.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum DepthError {
    InvalidDepth(String),
    EmptySide(String),
    InvalidPrice(String),
}

impl fmt::Display for DepthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDepth(s) => write!(f, "invalid depth: {s}"),
            Self::EmptySide(s) => write!(f, "empty side: {s}"),
            Self::InvalidPrice(s) => write!(f, "invalid price: {s}"),
        }
    }
}

impl std::error::Error for DepthError {}

// ── DepthSide ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthSide {
    Bid,
    Ask,
}

impl fmt::Display for DepthSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bid => write!(f, "BID"),
            Self::Ask => write!(f, "ASK"),
        }
    }
}

// ── L2Entry ─────────────────────────────────────────────────────

/// Level-2 depth entry: aggregated volume at a price.
#[derive(Debug, Clone, PartialEq)]
pub struct L2Entry {
    pub price: f64,
    pub volume: f64,
    pub order_count: usize,
    pub cumulative_volume: f64,
    pub cumulative_notional: f64,
}

impl L2Entry {
    pub fn new(price: f64, volume: f64, order_count: usize) -> Self {
        Self { price, volume, order_count, cumulative_volume: 0.0, cumulative_notional: 0.0 }
    }

    pub fn notional(&self) -> f64 { self.price * self.volume }
}

impl fmt::Display for L2Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L2({:.2}: vol={:.4}, orders={}, cum={:.4})",
            self.price, self.volume, self.order_count, self.cumulative_volume)
    }
}

// ── L3Entry ─────────────────────────────────────────────────────

/// Level-3 depth entry: individual order detail.
#[derive(Debug, Clone, PartialEq)]
pub struct L3Entry {
    pub price: f64,
    pub order_id: u64,
    pub quantity: f64,
    pub timestamp_ns: u64,
    pub is_hidden: bool,
}

impl L3Entry {
    pub fn new(price: f64, order_id: u64, quantity: f64, timestamp_ns: u64) -> Self {
        Self { price, order_id, quantity, timestamp_ns, is_hidden: false }
    }

    pub fn with_hidden(mut self, hidden: bool) -> Self {
        self.is_hidden = hidden;
        self
    }
}

impl fmt::Display for L3Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let h = if self.is_hidden { " [H]" } else { "" };
        write!(f, "L3({:.2}: id={}, qty={:.4}{h})", self.price, self.order_id, self.quantity)
    }
}

// ── DepthSnapshot ───────────────────────────────────────────────

/// A complete depth snapshot with bid/ask L2 data.
#[derive(Debug, Clone)]
pub struct DepthSnapshot {
    pub bid_levels: Vec<L2Entry>,
    pub ask_levels: Vec<L2Entry>,
    pub timestamp_ns: u64,
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
    pub total_bid_notional: f64,
    pub total_ask_notional: f64,
}

impl DepthSnapshot {
    /// Bid/ask volume imbalance ratio: (bid - ask) / (bid + ask), range [-1, 1].
    pub fn imbalance(&self) -> f64 {
        let total = self.total_bid_volume + self.total_ask_volume;
        if total < 1e-12 { 0.0 }
        else { (self.total_bid_volume - self.total_ask_volume) / total }
    }

    /// Bid depth (number of price levels).
    pub fn bid_depth(&self) -> usize { self.bid_levels.len() }

    /// Ask depth (number of price levels).
    pub fn ask_depth(&self) -> usize { self.ask_levels.len() }

    /// Best bid price.
    pub fn best_bid(&self) -> Option<f64> { self.bid_levels.first().map(|e| e.price) }

    /// Best ask price.
    pub fn best_ask(&self) -> Option<f64> { self.ask_levels.first().map(|e| e.price) }

    /// Spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        }
    }
}

impl fmt::Display for DepthSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DepthSnap(bids={}, asks={}, imbalance={:.3}, bid_vol={:.2}, ask_vol={:.2})",
            self.bid_depth(), self.ask_depth(), self.imbalance(),
            self.total_bid_volume, self.total_ask_volume)
    }
}

// ── DepthBar ────────────────────────────────────────────────────

/// A bar for depth chart visualization.
#[derive(Debug, Clone)]
pub struct DepthBar {
    pub price: f64,
    pub cumulative_volume: f64,
    pub side: DepthSide,
    pub bar_width: f64,
}

impl fmt::Display for DepthBar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bar({} {:.2}: cum={:.4}, w={:.3})",
            self.side, self.price, self.cumulative_volume, self.bar_width)
    }
}

// ── BookDepth ───────────────────────────────────────────────────

/// Market depth calculator that builds L2, L3, and visualization data.
#[derive(Debug, Clone)]
pub struct BookDepth {
    bid_l2: Vec<L2Entry>,
    ask_l2: Vec<L2Entry>,
    bid_l3: Vec<L3Entry>,
    ask_l3: Vec<L3Entry>,
    max_levels: usize,
    timestamp_ns: u64,
}

impl BookDepth {
    pub fn new() -> Self {
        Self {
            bid_l2: Vec::new(),
            ask_l2: Vec::new(),
            bid_l3: Vec::new(),
            ask_l3: Vec::new(),
            max_levels: 20,
            timestamp_ns: 0,
        }
    }

    pub fn with_max_levels(mut self, n: usize) -> Self {
        self.max_levels = n;
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp_ns = ts;
        self
    }

    /// Set bid-side L2 data (sorted best to worst: descending price).
    pub fn set_bids_l2(&mut self, entries: Vec<L2Entry>) {
        self.bid_l2 = entries;
        self.bid_l2.truncate(self.max_levels);
        self.compute_cumulative_bids();
    }

    /// Set ask-side L2 data (sorted best to worst: ascending price).
    pub fn set_asks_l2(&mut self, entries: Vec<L2Entry>) {
        self.ask_l2 = entries;
        self.ask_l2.truncate(self.max_levels);
        self.compute_cumulative_asks();
    }

    /// Set bid-side L3 data.
    pub fn set_bids_l3(&mut self, entries: Vec<L3Entry>) {
        self.bid_l3 = entries;
    }

    /// Set ask-side L3 data.
    pub fn set_asks_l3(&mut self, entries: Vec<L3Entry>) {
        self.ask_l3 = entries;
    }

    fn compute_cumulative_bids(&mut self) {
        let mut cum_vol = 0.0;
        let mut cum_not = 0.0;
        for e in self.bid_l2.iter_mut() {
            cum_vol += e.volume;
            cum_not += e.notional();
            e.cumulative_volume = cum_vol;
            e.cumulative_notional = cum_not;
        }
    }

    fn compute_cumulative_asks(&mut self) {
        let mut cum_vol = 0.0;
        let mut cum_not = 0.0;
        for e in self.ask_l2.iter_mut() {
            cum_vol += e.volume;
            cum_not += e.notional();
            e.cumulative_volume = cum_vol;
            e.cumulative_notional = cum_not;
        }
    }

    /// Build a depth snapshot.
    pub fn snapshot(&self) -> DepthSnapshot {
        let total_bid_vol: f64 = self.bid_l2.iter().map(|e| e.volume).sum();
        let total_ask_vol: f64 = self.ask_l2.iter().map(|e| e.volume).sum();
        let total_bid_not: f64 = self.bid_l2.iter().map(|e| e.notional()).sum();
        let total_ask_not: f64 = self.ask_l2.iter().map(|e| e.notional()).sum();
        DepthSnapshot {
            bid_levels: self.bid_l2.clone(),
            ask_levels: self.ask_l2.clone(),
            timestamp_ns: self.timestamp_ns,
            total_bid_volume: total_bid_vol,
            total_ask_volume: total_ask_vol,
            total_bid_notional: total_bid_not,
            total_ask_notional: total_ask_not,
        }
    }

    /// Generate depth chart bars for visualization.
    pub fn depth_bars(&self) -> Vec<DepthBar> {
        let max_cum = {
            let bid_max = self.bid_l2.last().map_or(0.0, |e| e.cumulative_volume);
            let ask_max = self.ask_l2.last().map_or(0.0, |e| e.cumulative_volume);
            bid_max.max(ask_max)
        };
        let mut bars = Vec::new();
        for e in self.bid_l2.iter().rev() {
            bars.push(DepthBar {
                price: e.price,
                cumulative_volume: e.cumulative_volume,
                side: DepthSide::Bid,
                bar_width: if max_cum > 1e-12 { e.cumulative_volume / max_cum } else { 0.0 },
            });
        }
        for e in &self.ask_l2 {
            bars.push(DepthBar {
                price: e.price,
                cumulative_volume: e.cumulative_volume,
                side: DepthSide::Ask,
                bar_width: if max_cum > 1e-12 { e.cumulative_volume / max_cum } else { 0.0 },
            });
        }
        bars
    }

    /// Volume at a specific price on a given side.
    pub fn volume_at_price(&self, price: f64, side: DepthSide) -> f64 {
        let levels = match side {
            DepthSide::Bid => &self.bid_l2,
            DepthSide::Ask => &self.ask_l2,
        };
        levels.iter()
            .find(|e| (e.price - price).abs() < 1e-8)
            .map_or(0.0, |e| e.volume)
    }

    /// Cumulative volume up to N levels.
    pub fn cumulative_volume(&self, side: DepthSide, levels: usize) -> f64 {
        let data = match side {
            DepthSide::Bid => &self.bid_l2,
            DepthSide::Ask => &self.ask_l2,
        };
        data.iter().take(levels).map(|e| e.volume).sum()
    }

    /// Imbalance at top N levels.
    pub fn top_n_imbalance(&self, n: usize) -> f64 {
        let bid_vol = self.cumulative_volume(DepthSide::Bid, n);
        let ask_vol = self.cumulative_volume(DepthSide::Ask, n);
        let total = bid_vol + ask_vol;
        if total < 1e-12 { 0.0 } else { (bid_vol - ask_vol) / total }
    }

    /// VWAP across top N levels.
    pub fn vwap(&self, side: DepthSide, levels: usize) -> Option<f64> {
        let data = match side {
            DepthSide::Bid => &self.bid_l2,
            DepthSide::Ask => &self.ask_l2,
        };
        let mut total_not = 0.0;
        let mut total_vol = 0.0;
        for e in data.iter().take(levels) {
            total_not += e.notional();
            total_vol += e.volume;
        }
        if total_vol > 1e-12 { Some(total_not / total_vol) } else { None }
    }

    /// L3 bid data accessor.
    pub fn bids_l3(&self) -> &[L3Entry] { &self.bid_l3 }

    /// L3 ask data accessor.
    pub fn asks_l3(&self) -> &[L3Entry] { &self.ask_l3 }

    /// L2 bid data accessor.
    pub fn bids_l2(&self) -> &[L2Entry] { &self.bid_l2 }

    /// L2 ask data accessor.
    pub fn asks_l2(&self) -> &[L2Entry] { &self.ask_l2 }
}

impl fmt::Display for BookDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BookDepth(bid_levels={}, ask_levels={}, max={})",
            self.bid_l2.len(), self.ask_l2.len(), self.max_levels)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_depth() -> BookDepth {
        let mut d = BookDepth::new().with_max_levels(10).with_timestamp(1000);
        d.set_bids_l2(vec![
            L2Entry::new(100.0, 50.0, 3),
            L2Entry::new(99.0, 80.0, 5),
            L2Entry::new(98.0, 30.0, 2),
        ]);
        d.set_asks_l2(vec![
            L2Entry::new(101.0, 40.0, 2),
            L2Entry::new(102.0, 60.0, 4),
            L2Entry::new(103.0, 20.0, 1),
        ]);
        d
    }

    #[test]
    fn test_new_depth() {
        let d = BookDepth::new();
        assert_eq!(d.bids_l2().len(), 0);
        assert_eq!(d.asks_l2().len(), 0);
    }

    #[test]
    fn test_set_bids() {
        let d = sample_depth();
        assert_eq!(d.bids_l2().len(), 3);
        assert!((d.bids_l2()[0].price - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_cumulative_volume_bids() {
        let d = sample_depth();
        // First: 50, second: 50+80=130, third: 130+30=160
        assert!((d.bids_l2()[0].cumulative_volume - 50.0).abs() < 1e-6);
        assert!((d.bids_l2()[1].cumulative_volume - 130.0).abs() < 1e-6);
        assert!((d.bids_l2()[2].cumulative_volume - 160.0).abs() < 1e-6);
    }

    #[test]
    fn test_cumulative_volume_asks() {
        let d = sample_depth();
        assert!((d.asks_l2()[0].cumulative_volume - 40.0).abs() < 1e-6);
        assert!((d.asks_l2()[1].cumulative_volume - 100.0).abs() < 1e-6);
        assert!((d.asks_l2()[2].cumulative_volume - 120.0).abs() < 1e-6);
    }

    #[test]
    fn test_snapshot() {
        let d = sample_depth();
        let snap = d.snapshot();
        assert!((snap.total_bid_volume - 160.0).abs() < 1e-6);
        assert!((snap.total_ask_volume - 120.0).abs() < 1e-6);
        assert_eq!(snap.bid_depth(), 3);
        assert_eq!(snap.ask_depth(), 3);
    }

    #[test]
    fn test_snapshot_imbalance() {
        let d = sample_depth();
        let snap = d.snapshot();
        let imb = snap.imbalance();
        // (160 - 120) / (160 + 120) = 40/280 = 0.14286
        assert!((imb - 40.0 / 280.0).abs() < 1e-4);
    }

    #[test]
    fn test_snapshot_spread() {
        let d = sample_depth();
        let snap = d.snapshot();
        assert!((snap.spread().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_volume_at_price() {
        let d = sample_depth();
        assert!((d.volume_at_price(100.0, DepthSide::Bid) - 50.0).abs() < 1e-6);
        assert!((d.volume_at_price(101.0, DepthSide::Ask) - 40.0).abs() < 1e-6);
        assert!((d.volume_at_price(999.0, DepthSide::Bid) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cumulative_volume_fn() {
        let d = sample_depth();
        assert!((d.cumulative_volume(DepthSide::Bid, 2) - 130.0).abs() < 1e-6);
        assert!((d.cumulative_volume(DepthSide::Ask, 1) - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_n_imbalance() {
        let d = sample_depth();
        let imb = d.top_n_imbalance(1);
        // bid=50, ask=40 => (50-40)/(50+40) = 10/90
        assert!((imb - 10.0 / 90.0).abs() < 1e-4);
    }

    #[test]
    fn test_vwap() {
        let d = sample_depth();
        let vwap = d.vwap(DepthSide::Bid, 2).unwrap();
        // (100*50 + 99*80) / (50+80) = (5000+7920)/130 = 12920/130 = 99.3846
        assert!((vwap - 12920.0 / 130.0).abs() < 1e-4);
    }

    #[test]
    fn test_depth_bars() {
        let d = sample_depth();
        let bars = d.depth_bars();
        assert_eq!(bars.len(), 6); // 3 bid + 3 ask
        assert_eq!(bars[0].side, DepthSide::Bid);
        assert_eq!(bars[3].side, DepthSide::Ask);
        // All bar_widths in [0,1]
        for b in &bars {
            assert!(b.bar_width >= 0.0 && b.bar_width <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_l3_data() {
        let mut d = BookDepth::new();
        d.set_bids_l3(vec![
            L3Entry::new(100.0, 1, 30.0, 1000),
            L3Entry::new(100.0, 2, 20.0, 2000),
        ]);
        assert_eq!(d.bids_l3().len(), 2);
    }

    #[test]
    fn test_l3_hidden() {
        let e = L3Entry::new(100.0, 1, 50.0, 1000).with_hidden(true);
        assert!(e.is_hidden);
    }

    #[test]
    fn test_max_levels_truncation() {
        let mut d = BookDepth::new().with_max_levels(2);
        d.set_bids_l2(vec![
            L2Entry::new(100.0, 50.0, 3),
            L2Entry::new(99.0, 80.0, 5),
            L2Entry::new(98.0, 30.0, 2),
        ]);
        assert_eq!(d.bids_l2().len(), 2);
    }

    #[test]
    fn test_display() {
        let d = sample_depth();
        let s = format!("{d}");
        assert!(s.contains("bid_levels=3"));
        assert!(s.contains("ask_levels=3"));
    }

    #[test]
    fn test_l2_display() {
        let e = L2Entry::new(100.0, 50.0, 3);
        let s = format!("{e}");
        assert!(s.contains("100.00"));
    }

    #[test]
    fn test_snapshot_display() {
        let d = sample_depth();
        let snap = d.snapshot();
        let s = format!("{snap}");
        assert!(s.contains("bids=3"));
    }

    #[test]
    fn test_depth_bar_display() {
        let b = DepthBar { price: 100.0, cumulative_volume: 50.0, side: DepthSide::Bid, bar_width: 0.5 };
        let s = format!("{b}");
        assert!(s.contains("BID"));
    }
}
