//! # Execution Algorithm Framework
//!
//! Provides a framework for building algorithmic execution strategies.
//! Includes benchmark tracking (arrival price, VWAP, close), participation
//! rate controls, order slicing, and execution quality analytics with
//! implementation shortfall measurement.

use std::fmt;
use std::collections::VecDeque;

// ── Core Types ──

/// Benchmark against which execution quality is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Benchmark {
    ArrivalPrice,
    Vwap,
    Twap,
    Close,
    Open,
    MidPoint,
}

impl fmt::Display for Benchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Benchmark::ArrivalPrice => "ARRIVAL",
            Benchmark::Vwap => "VWAP",
            Benchmark::Twap => "TWAP",
            Benchmark::Close => "CLOSE",
            Benchmark::Open => "OPEN",
            Benchmark::MidPoint => "MIDPOINT",
        };
        write!(f, "{s}")
    }
}

/// Execution side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecSide {
    Buy,
    Sell,
}

impl fmt::Display for ExecSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecSide::Buy => write!(f, "BUY"),
            ExecSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Urgency level controlling aggressiveness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Urgency {
    Passive,
    Normal,
    Aggressive,
    VeryAggressive,
}

impl Urgency {
    /// Returns a participation factor multiplier.
    pub fn factor(&self) -> f64 {
        match self {
            Urgency::Passive => 0.5,
            Urgency::Normal => 1.0,
            Urgency::Aggressive => 1.5,
            Urgency::VeryAggressive => 2.0,
        }
    }
}

impl fmt::Display for Urgency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Urgency::Passive => "PASSIVE",
            Urgency::Normal => "NORMAL",
            Urgency::Aggressive => "AGGRESSIVE",
            Urgency::VeryAggressive => "VERY_AGGRESSIVE",
        };
        write!(f, "{s}")
    }
}

// ── Fill Record ──

/// Record of a single execution fill.
#[derive(Clone, Debug)]
pub struct Fill {
    pub price: f64,
    pub quantity: f64,
    pub timestamp_ns: u64,
    pub venue: String,
    pub is_aggressive: bool,
}

impl Fill {
    pub fn new(price: f64, quantity: f64, timestamp_ns: u64) -> Self {
        Self {
            price,
            quantity,
            timestamp_ns,
            venue: String::new(),
            is_aggressive: false,
        }
    }

    pub fn with_venue(mut self, venue: &str) -> Self {
        self.venue = venue.to_string();
        self
    }

    pub fn with_aggressive(mut self, agg: bool) -> Self {
        self.is_aggressive = agg;
        self
    }

    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }
}

impl fmt::Display for Fill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fill({:.4}@{:.2} venue={} agg={})",
            self.quantity, self.price, self.venue, self.is_aggressive,
        )
    }
}

// ── Child Order Slice ──

/// A child order slice produced by the execution algorithm.
#[derive(Clone, Debug)]
pub struct ChildSlice {
    pub target_quantity: f64,
    pub limit_price: f64,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub filled_quantity: f64,
    pub avg_fill_price: f64,
}

impl ChildSlice {
    pub fn new(qty: f64, limit: f64, start: u64, end: u64) -> Self {
        Self {
            target_quantity: qty,
            limit_price: limit,
            start_time_ns: start,
            end_time_ns: end,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
        }
    }

    pub fn remaining(&self) -> f64 {
        (self.target_quantity - self.filled_quantity).max(0.0)
    }

    pub fn fill_rate(&self) -> f64 {
        if self.target_quantity > 1e-12 {
            self.filled_quantity / self.target_quantity
        } else {
            0.0
        }
    }

    pub fn record_fill(&mut self, price: f64, qty: f64) {
        let total_notional = self.avg_fill_price * self.filled_quantity + price * qty;
        self.filled_quantity += qty;
        if self.filled_quantity > 1e-12 {
            self.avg_fill_price = total_notional / self.filled_quantity;
        }
    }
}

impl fmt::Display for ChildSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Slice(target={:.4} limit={:.2} filled={:.4} avg={:.2})",
            self.target_quantity, self.limit_price,
            self.filled_quantity, self.avg_fill_price,
        )
    }
}

// ── Participation Controller ──

/// Controls participation rate relative to market volume.
#[derive(Clone, Debug)]
pub struct ParticipationController {
    pub target_rate: f64,
    pub max_rate: f64,
    pub market_volume: f64,
    pub algo_volume: f64,
    pub window_volumes: VecDeque<(u64, f64, f64)>,
    pub window_duration_ns: u64,
}

impl ParticipationController {
    pub fn new(target_rate: f64) -> Self {
        Self {
            target_rate: target_rate.clamp(0.0, 1.0),
            max_rate: (target_rate * 2.0).min(1.0),
            market_volume: 0.0,
            algo_volume: 0.0,
            window_volumes: VecDeque::new(),
            window_duration_ns: 60_000_000_000, // 1 minute
        }
    }

    pub fn with_max_rate(mut self, rate: f64) -> Self {
        self.max_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_window(mut self, duration_ns: u64) -> Self {
        self.window_duration_ns = duration_ns;
        self
    }

    /// Record observed market volume.
    pub fn record_market_volume(&mut self, qty: f64, timestamp_ns: u64) {
        self.market_volume += qty;
        self.window_volumes.push_back((timestamp_ns, qty, 0.0));
        self.prune_window(timestamp_ns);
    }

    /// Record algo fill volume.
    pub fn record_algo_fill(&mut self, qty: f64, timestamp_ns: u64) {
        self.algo_volume += qty;
        self.window_volumes.push_back((timestamp_ns, 0.0, qty));
        self.prune_window(timestamp_ns);
    }

    /// Current participation rate.
    pub fn current_rate(&self) -> f64 {
        if self.market_volume > 1e-12 {
            self.algo_volume / self.market_volume
        } else {
            0.0
        }
    }

    /// Participation rate within the rolling window.
    pub fn window_rate(&self) -> f64 {
        let (mkt, algo): (f64, f64) = self.window_volumes.iter()
            .fold((0.0, 0.0), |(m, a), &(_, mv, av)| (m + mv, a + av));
        if mkt > 1e-12 { algo / mkt } else { 0.0 }
    }

    /// Maximum allowable quantity for next slice.
    pub fn allowed_quantity(&self, market_qty: f64) -> f64 {
        let target_qty = market_qty * self.target_rate;
        let max_qty = market_qty * self.max_rate;
        let rate = self.current_rate();
        if rate > self.max_rate {
            0.0
        } else if rate > self.target_rate {
            target_qty * 0.5
        } else {
            max_qty.min(target_qty * 1.5)
        }
    }

    fn prune_window(&mut self, now: u64) {
        while let Some(&(ts, _, _)) = self.window_volumes.front() {
            if now.saturating_sub(ts) > self.window_duration_ns {
                self.window_volumes.pop_front();
            } else {
                break;
            }
        }
    }
}

impl fmt::Display for ParticipationController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Participation(target={:.1}% current={:.1}% mkt={:.0} algo={:.0})",
            self.target_rate * 100.0, self.current_rate() * 100.0,
            self.market_volume, self.algo_volume,
        )
    }
}

// ── Execution Quality ──

/// Measures execution quality against a benchmark.
#[derive(Clone, Debug)]
pub struct ExecutionQuality {
    pub benchmark: Benchmark,
    pub benchmark_price: f64,
    pub side: ExecSide,
    pub fills: Vec<Fill>,
    pub total_quantity: f64,
    pub total_notional: f64,
}

impl ExecutionQuality {
    pub fn new(benchmark: Benchmark, benchmark_price: f64, side: ExecSide) -> Self {
        Self {
            benchmark,
            benchmark_price,
            side,
            fills: Vec::new(),
            total_quantity: 0.0,
            total_notional: 0.0,
        }
    }

    pub fn record_fill(&mut self, fill: Fill) {
        self.total_quantity += fill.quantity;
        self.total_notional += fill.notional();
        self.fills.push(fill);
    }

    /// Average execution price.
    pub fn avg_price(&self) -> f64 {
        if self.total_quantity > 1e-12 {
            self.total_notional / self.total_quantity
        } else {
            0.0
        }
    }

    /// Implementation shortfall in basis points.
    pub fn shortfall_bps(&self) -> f64 {
        if self.benchmark_price < 1e-12 { return 0.0; }
        let avg = self.avg_price();
        let diff = match self.side {
            ExecSide::Buy => avg - self.benchmark_price,
            ExecSide::Sell => self.benchmark_price - avg,
        };
        (diff / self.benchmark_price) * 10_000.0
    }

    /// Standard deviation of fill prices.
    pub fn price_std_dev(&self) -> f64 {
        if self.fills.len() < 2 { return 0.0; }
        let avg = self.avg_price();
        let var: f64 = self.fills.iter()
            .map(|f| {
                let d = f.price - avg;
                d * d * f.quantity
            })
            .sum::<f64>() / self.total_quantity;
        var.sqrt()
    }

    /// Fraction of fills that were aggressive.
    pub fn aggression_rate(&self) -> f64 {
        if self.fills.is_empty() { return 0.0; }
        let agg_qty: f64 = self.fills.iter()
            .filter(|f| f.is_aggressive)
            .map(|f| f.quantity)
            .sum();
        agg_qty / self.total_quantity
    }

    pub fn fill_count(&self) -> usize {
        self.fills.len()
    }
}

impl fmt::Display for ExecutionQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ExecQuality({} bench={:.2} avg={:.2} shortfall={:.1}bps fills={})",
            self.benchmark, self.benchmark_price, self.avg_price(),
            self.shortfall_bps(), self.fills.len(),
        )
    }
}

// ── Execution Strategy ──

/// A complete execution strategy combining slicing and participation.
#[derive(Clone, Debug)]
pub struct ExecutionStrategy {
    pub symbol: String,
    pub side: ExecSide,
    pub total_quantity: f64,
    pub urgency: Urgency,
    pub benchmark: Benchmark,
    pub arrival_price: f64,
    pub slices: Vec<ChildSlice>,
    pub participation: ParticipationController,
    pub quality: ExecutionQuality,
}

impl ExecutionStrategy {
    pub fn new(symbol: &str, side: ExecSide, quantity: f64, arrival_price: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side,
            total_quantity: quantity,
            urgency: Urgency::Normal,
            benchmark: Benchmark::ArrivalPrice,
            arrival_price,
            slices: Vec::new(),
            participation: ParticipationController::new(0.10),
            quality: ExecutionQuality::new(Benchmark::ArrivalPrice, arrival_price, side),
        }
    }

    pub fn with_urgency(mut self, u: Urgency) -> Self {
        self.urgency = u;
        self
    }

    pub fn with_benchmark(mut self, b: Benchmark, price: f64) -> Self {
        self.benchmark = b;
        self.quality = ExecutionQuality::new(b, price, self.side);
        self
    }

    pub fn with_participation(mut self, rate: f64) -> Self {
        self.participation = ParticipationController::new(rate);
        self
    }

    /// Generate evenly-spaced time slices.
    pub fn generate_slices(&mut self, num_slices: usize, limit_price: f64,
                           start_ns: u64, end_ns: u64) {
        if num_slices == 0 { return; }
        let qty_per = self.total_quantity / num_slices as f64;
        let duration = end_ns.saturating_sub(start_ns);
        let interval = if num_slices > 1 { duration / num_slices as u64 } else { duration };

        self.slices.clear();
        for i in 0..num_slices {
            let s = start_ns + interval * i as u64;
            let e = if i + 1 < num_slices { s + interval } else { end_ns };
            self.slices.push(ChildSlice::new(qty_per, limit_price, s, e));
        }
    }

    /// Total filled quantity across all slices.
    pub fn filled_quantity(&self) -> f64 {
        self.quality.total_quantity
    }

    /// Completion percentage.
    pub fn completion_pct(&self) -> f64 {
        if self.total_quantity > 1e-12 {
            (self.filled_quantity() / self.total_quantity) * 100.0
        } else {
            0.0
        }
    }

    /// Record a fill on the current active slice.
    pub fn record_fill(&mut self, fill: Fill) {
        for slice in &mut self.slices {
            if slice.remaining() > 1e-12 {
                let apply = fill.quantity.min(slice.remaining());
                slice.record_fill(fill.price, apply);
                break;
            }
        }
        self.quality.record_fill(fill);
    }
}

impl fmt::Display for ExecutionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Strategy({} {} {:.0} urgency={} complete={:.1}% shortfall={:.1}bps)",
            self.symbol, self.side, self.total_quantity, self.urgency,
            self.completion_pct(), self.quality.shortfall_bps(),
        )
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_notional() {
        let f = Fill::new(100.0, 10.0, 0);
        assert!((f.notional() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_fill_display() {
        let f = Fill::new(50.0, 5.0, 0).with_venue("XNAS");
        let s = format!("{f}");
        assert!(s.contains("XNAS"));
    }

    #[test]
    fn test_child_slice_fill() {
        let mut cs = ChildSlice::new(100.0, 50.0, 0, 1000);
        cs.record_fill(50.0, 40.0);
        cs.record_fill(51.0, 30.0);
        assert!((cs.filled_quantity - 70.0).abs() < 1e-9);
        assert!((cs.remaining() - 30.0).abs() < 1e-9);
        let expected_avg = (50.0 * 40.0 + 51.0 * 30.0) / 70.0;
        assert!((cs.avg_fill_price - expected_avg).abs() < 1e-6);
    }

    #[test]
    fn test_child_slice_fill_rate() {
        let mut cs = ChildSlice::new(100.0, 50.0, 0, 1000);
        cs.record_fill(50.0, 75.0);
        assert!((cs.fill_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_participation_rate() {
        let mut pc = ParticipationController::new(0.10);
        pc.record_market_volume(1000.0, 100);
        pc.record_algo_fill(80.0, 100);
        assert!((pc.current_rate() - 0.08).abs() < 1e-9);
    }

    #[test]
    fn test_participation_allowed_when_below_target() {
        let pc = ParticipationController::new(0.10);
        let allowed = pc.allowed_quantity(100.0);
        assert!(allowed > 0.0);
    }

    #[test]
    fn test_participation_throttled_above_max() {
        let mut pc = ParticipationController::new(0.10).with_max_rate(0.20);
        pc.market_volume = 100.0;
        pc.algo_volume = 25.0; // 25% > max 20%
        let allowed = pc.allowed_quantity(100.0);
        assert!((allowed - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_exec_quality_shortfall_buy() {
        let mut eq = ExecutionQuality::new(Benchmark::ArrivalPrice, 100.0, ExecSide::Buy);
        eq.record_fill(Fill::new(100.50, 100.0, 0));
        // Bought higher than arrival → positive shortfall.
        assert!(eq.shortfall_bps() > 0.0);
        assert!((eq.shortfall_bps() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_exec_quality_shortfall_sell() {
        let mut eq = ExecutionQuality::new(Benchmark::ArrivalPrice, 100.0, ExecSide::Sell);
        eq.record_fill(Fill::new(99.50, 100.0, 0));
        // Sold lower than arrival → positive shortfall.
        assert!(eq.shortfall_bps() > 0.0);
        assert!((eq.shortfall_bps() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_exec_quality_aggression() {
        let mut eq = ExecutionQuality::new(Benchmark::Vwap, 100.0, ExecSide::Buy);
        eq.record_fill(Fill::new(100.0, 60.0, 0).with_aggressive(true));
        eq.record_fill(Fill::new(100.0, 40.0, 0).with_aggressive(false));
        assert!((eq.aggression_rate() - 0.60).abs() < 1e-9);
    }

    #[test]
    fn test_strategy_generate_slices() {
        let mut strat = ExecutionStrategy::new("AAPL", ExecSide::Buy, 1000.0, 150.0);
        strat.generate_slices(5, 151.0, 0, 10000);
        assert_eq!(strat.slices.len(), 5);
        let total: f64 = strat.slices.iter().map(|s| s.target_quantity).sum();
        assert!((total - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_strategy_completion() {
        let mut strat = ExecutionStrategy::new("AAPL", ExecSide::Buy, 100.0, 150.0);
        strat.generate_slices(2, 151.0, 0, 1000);
        strat.record_fill(Fill::new(150.5, 50.0, 100));
        assert!((strat.completion_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_urgency_factors() {
        assert!((Urgency::Passive.factor() - 0.5).abs() < 1e-9);
        assert!((Urgency::Normal.factor() - 1.0).abs() < 1e-9);
        assert!((Urgency::Aggressive.factor() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_benchmark_display() {
        assert_eq!(format!("{}", Benchmark::ArrivalPrice), "ARRIVAL");
        assert_eq!(format!("{}", Benchmark::Vwap), "VWAP");
    }

    #[test]
    fn test_strategy_display() {
        let strat = ExecutionStrategy::new("MSFT", ExecSide::Sell, 500.0, 300.0);
        let s = format!("{strat}");
        assert!(s.contains("MSFT"));
        assert!(s.contains("SELL"));
    }

    #[test]
    fn test_participation_display() {
        let pc = ParticipationController::new(0.15);
        let s = format!("{pc}");
        assert!(s.contains("15.0%"));
    }

    #[test]
    fn test_price_std_dev() {
        let mut eq = ExecutionQuality::new(Benchmark::ArrivalPrice, 100.0, ExecSide::Buy);
        eq.record_fill(Fill::new(100.0, 50.0, 0));
        eq.record_fill(Fill::new(102.0, 50.0, 0));
        assert!(eq.price_std_dev() > 0.0);
    }

    #[test]
    fn test_window_rate() {
        let mut pc = ParticipationController::new(0.10);
        pc.record_market_volume(500.0, 100);
        pc.record_algo_fill(40.0, 150);
        assert!((pc.window_rate() - (40.0 / 500.0)).abs() < 1e-9);
    }

    #[test]
    fn test_exec_quality_display() {
        let eq = ExecutionQuality::new(Benchmark::Close, 200.0, ExecSide::Buy);
        let s = format!("{eq}");
        assert!(s.contains("CLOSE"));
    }
}
