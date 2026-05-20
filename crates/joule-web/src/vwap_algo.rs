//! # Volume-Weighted Average Price Algorithm
//!
//! Implements VWAP execution that distributes order quantity according to
//! a historical or predicted volume profile. Tracks real-time deviation
//! from the market VWAP benchmark, supports intraday volume curve fitting,
//! and provides analytics for execution quality assessment.

use std::fmt;

// ── Core Types ──

/// A single bucket in the intraday volume profile.
#[derive(Clone, Debug)]
pub struct VolumeBucket {
    pub index: usize,
    pub start_ns: u64,
    pub end_ns: u64,
    pub expected_volume_pct: f64,
    pub actual_volume: f64,
    pub target_quantity: f64,
    pub filled_quantity: f64,
    pub avg_fill_price: f64,
    pub fill_count: u32,
}

impl VolumeBucket {
    pub fn new(index: usize, start_ns: u64, end_ns: u64, volume_pct: f64) -> Self {
        Self {
            index,
            start_ns,
            end_ns,
            expected_volume_pct: volume_pct,
            actual_volume: 0.0,
            target_quantity: 0.0,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            fill_count: 0,
        }
    }

    pub fn remaining(&self) -> f64 {
        (self.target_quantity - self.filled_quantity).max(0.0)
    }

    pub fn is_complete(&self) -> bool {
        self.remaining() < 1e-12
    }

    pub fn fill_rate(&self) -> f64 {
        if self.target_quantity > 1e-12 {
            self.filled_quantity / self.target_quantity
        } else {
            0.0
        }
    }

    pub fn record_fill(&mut self, price: f64, qty: f64) {
        let total = self.avg_fill_price * self.filled_quantity + price * qty;
        self.filled_quantity += qty;
        self.fill_count += 1;
        if self.filled_quantity > 1e-12 {
            self.avg_fill_price = total / self.filled_quantity;
        }
    }

    pub fn record_market_volume(&mut self, qty: f64) {
        self.actual_volume += qty;
    }

    /// Volume prediction error for this bucket.
    pub fn volume_prediction_error(&self, total_market_volume: f64) -> f64 {
        if total_market_volume < 1e-12 { return 0.0; }
        let actual_pct = self.actual_volume / total_market_volume;
        actual_pct - self.expected_volume_pct
    }
}

impl fmt::Display for VolumeBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bucket[{}](vol={:.1}% target={:.2} filled={:.2} avg={:.4})",
            self.index, self.expected_volume_pct * 100.0,
            self.target_quantity, self.filled_quantity, self.avg_fill_price,
        )
    }
}

// ── Volume Profile ──

/// Typical intraday volume profile shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeProfileShape {
    /// Standard U-shape: high at open/close, low midday.
    UShape,
    /// Flat/uniform volume.
    Flat,
    /// Custom (weights provided externally).
    Custom,
}

impl fmt::Display for VolumeProfileShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            VolumeProfileShape::UShape => "U-SHAPE",
            VolumeProfileShape::Flat => "FLAT",
            VolumeProfileShape::Custom => "CUSTOM",
        };
        write!(f, "{s}")
    }
}

/// Intraday volume profile used to schedule VWAP execution.
#[derive(Clone, Debug)]
pub struct VolumeProfile {
    pub shape: VolumeProfileShape,
    pub bucket_count: usize,
    pub weights: Vec<f64>,
}

impl VolumeProfile {
    pub fn flat(buckets: usize) -> Self {
        Self {
            shape: VolumeProfileShape::Flat,
            bucket_count: buckets,
            weights: vec![1.0 / buckets as f64; buckets],
        }
    }

    pub fn u_shape(buckets: usize) -> Self {
        let mid = buckets as f64 / 2.0;
        let raw: Vec<f64> = (0..buckets).map(|i| {
            let dist = ((i as f64) - mid).abs() / mid;
            0.3 + 0.7 * dist * dist
        }).collect();
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        Self {
            shape: VolumeProfileShape::UShape,
            bucket_count: buckets,
            weights,
        }
    }

    pub fn custom(weights: Vec<f64>) -> Self {
        let total: f64 = weights.iter().sum();
        let normed: Vec<f64> = if total > 1e-12 {
            weights.iter().map(|w| w / total).collect()
        } else {
            let n = weights.len();
            vec![1.0 / n as f64; n]
        };
        Self {
            shape: VolumeProfileShape::Custom,
            bucket_count: normed.len(),
            weights: normed,
        }
    }

    /// Peak volume bucket index.
    pub fn peak_bucket(&self) -> usize {
        self.weights.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Cumulative weight up to (exclusive) a given bucket.
    pub fn cumulative_weight(&self, up_to: usize) -> f64 {
        self.weights.iter().take(up_to).sum()
    }
}

impl fmt::Display for VolumeProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VolumeProfile({} buckets={} peak={})",
            self.shape, self.bucket_count, self.peak_bucket())
    }
}

// ── VWAP Schedule ──

/// A complete VWAP execution schedule.
#[derive(Clone, Debug)]
pub struct VwapSchedule {
    pub symbol: String,
    pub total_quantity: f64,
    pub profile: VolumeProfile,
    pub buckets: Vec<VolumeBucket>,
    pub start_ns: u64,
    pub end_ns: u64,
    pub max_participation_rate: f64,
}

impl VwapSchedule {
    pub fn new(symbol: &str, quantity: f64, profile: VolumeProfile,
               start_ns: u64, end_ns: u64) -> Self {
        let n = profile.bucket_count;
        let duration = end_ns.saturating_sub(start_ns);
        let interval = if n > 0 { duration / n as u64 } else { duration };

        let buckets: Vec<VolumeBucket> = (0..n).map(|i| {
            let s = start_ns + interval * i as u64;
            let e = if i + 1 < n { s + interval } else { end_ns };
            let mut b = VolumeBucket::new(i, s, e, profile.weights[i]);
            b.target_quantity = quantity * profile.weights[i];
            b
        }).collect();

        Self {
            symbol: symbol.to_string(),
            total_quantity: quantity,
            profile,
            buckets,
            start_ns,
            end_ns,
            max_participation_rate: 0.25,
        }
    }

    pub fn with_max_participation(mut self, rate: f64) -> Self {
        self.max_participation_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Find the active bucket for a timestamp.
    pub fn active_bucket_index(&self, now_ns: u64) -> Option<usize> {
        self.buckets.iter().position(|b| now_ns >= b.start_ns && now_ns < b.end_ns)
    }

    /// Record a fill.
    pub fn record_fill(&mut self, price: f64, qty: f64, timestamp_ns: u64) {
        if let Some(idx) = self.active_bucket_index(timestamp_ns) {
            self.buckets[idx].record_fill(price, qty);
        } else if let Some(last) = self.buckets.last_mut() {
            last.record_fill(price, qty);
        }
    }

    /// Record market volume for the active bucket.
    pub fn record_market_volume(&mut self, qty: f64, timestamp_ns: u64) {
        if let Some(idx) = self.active_bucket_index(timestamp_ns) {
            self.buckets[idx].record_market_volume(qty);
        }
    }

    /// Total filled quantity.
    pub fn filled_quantity(&self) -> f64 {
        self.buckets.iter().map(|b| b.filled_quantity).sum()
    }

    /// Completion percentage.
    pub fn completion_pct(&self) -> f64 {
        if self.total_quantity > 1e-12 {
            (self.filled_quantity() / self.total_quantity) * 100.0
        } else {
            0.0
        }
    }

    /// Realized VWAP of algo fills.
    pub fn realized_vwap(&self) -> f64 {
        let total_qty: f64 = self.buckets.iter().map(|b| b.filled_quantity).sum();
        if total_qty < 1e-12 { return 0.0; }
        let notional: f64 = self.buckets.iter()
            .map(|b| b.avg_fill_price * b.filled_quantity)
            .sum();
        notional / total_qty
    }

    /// Number of completed buckets.
    pub fn completed_buckets(&self) -> usize {
        self.buckets.iter().filter(|b| b.is_complete()).count()
    }
}

impl fmt::Display for VwapSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VWAP({} qty={:.0} buckets={} complete={:.1}% vwap={:.4})",
            self.symbol, self.total_quantity, self.buckets.len(),
            self.completion_pct(), self.realized_vwap(),
        )
    }
}

// ── VWAP Tracker ──

/// Tracks the reference market VWAP over time.
#[derive(Clone, Debug)]
pub struct VwapTracker {
    cumulative_notional: f64,
    cumulative_volume: f64,
    sample_count: usize,
    high_price: f64,
    low_price: f64,
}

impl VwapTracker {
    pub fn new() -> Self {
        Self {
            cumulative_notional: 0.0,
            cumulative_volume: 0.0,
            sample_count: 0,
            high_price: f64::MIN,
            low_price: f64::MAX,
        }
    }

    pub fn record(&mut self, price: f64, volume: f64) {
        self.cumulative_notional += price * volume;
        self.cumulative_volume += volume;
        self.sample_count += 1;
        if price > self.high_price { self.high_price = price; }
        if price < self.low_price { self.low_price = price; }
    }

    pub fn vwap(&self) -> f64 {
        if self.cumulative_volume > 1e-12 {
            self.cumulative_notional / self.cumulative_volume
        } else {
            0.0
        }
    }

    pub fn total_volume(&self) -> f64 {
        self.cumulative_volume
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn price_range(&self) -> (f64, f64) {
        if self.sample_count == 0 { (0.0, 0.0) }
        else { (self.low_price, self.high_price) }
    }

    /// Deviation of a given price from the VWAP in basis points.
    pub fn deviation_bps(&self, price: f64) -> f64 {
        let v = self.vwap();
        if v < 1e-12 { return 0.0; }
        ((price - v) / v) * 10_000.0
    }
}

impl fmt::Display for VwapTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lo, hi) = self.price_range();
        write!(
            f,
            "VwapTracker(vwap={:.4} vol={:.0} samples={} range=[{:.4},{:.4}])",
            self.vwap(), self.cumulative_volume, self.sample_count, lo, hi,
        )
    }
}

// ── VWAP Deviation Analyzer ──

/// Analyzes algo execution deviation from the market VWAP.
#[derive(Clone, Debug)]
pub struct VwapDeviation {
    pub market_vwap: f64,
    pub algo_vwap: f64,
    pub total_quantity: f64,
    pub deviation_bps: f64,
    pub tracking_error_bps: f64,
}

impl VwapDeviation {
    pub fn compute(market_vwap: f64, algo_vwap: f64, quantity: f64) -> Self {
        let dev = if market_vwap > 1e-12 {
            ((algo_vwap - market_vwap) / market_vwap) * 10_000.0
        } else {
            0.0
        };
        Self {
            market_vwap,
            algo_vwap,
            total_quantity: quantity,
            deviation_bps: dev,
            tracking_error_bps: dev.abs(),
        }
    }

    pub fn is_favorable_buy(&self) -> bool {
        self.deviation_bps < 0.0
    }

    pub fn is_favorable_sell(&self) -> bool {
        self.deviation_bps > 0.0
    }

    pub fn notional_impact(&self) -> f64 {
        (self.algo_vwap - self.market_vwap) * self.total_quantity
    }
}

impl fmt::Display for VwapDeviation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VwapDev(market={:.4} algo={:.4} dev={:.1}bps impact={:.2})",
            self.market_vwap, self.algo_vwap, self.deviation_bps,
            self.notional_impact(),
        )
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_profile() {
        let p = VolumeProfile::flat(10);
        assert_eq!(p.bucket_count, 10);
        let total: f64 = p.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
        for w in &p.weights {
            assert!((w - 0.1).abs() < 1e-9);
        }
    }

    #[test]
    fn test_u_shape_profile() {
        let p = VolumeProfile::u_shape(10);
        let total: f64 = p.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
        // Endpoints heavier than middle.
        assert!(p.weights[0] > p.weights[5]);
        assert!(p.weights[9] > p.weights[5]);
    }

    #[test]
    fn test_custom_profile() {
        let p = VolumeProfile::custom(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        assert_eq!(p.bucket_count, 5);
        let total: f64 = p.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_peak_bucket() {
        let p = VolumeProfile::custom(vec![1.0, 5.0, 2.0]);
        assert_eq!(p.peak_bucket(), 1);
    }

    #[test]
    fn test_cumulative_weight() {
        let p = VolumeProfile::flat(4);
        assert!((p.cumulative_weight(2) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_schedule_creation() {
        let p = VolumeProfile::flat(5);
        let s = VwapSchedule::new("AAPL", 1000.0, p, 0, 5000);
        assert_eq!(s.buckets.len(), 5);
        let total: f64 = s.buckets.iter().map(|b| b.target_quantity).sum();
        assert!((total - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_fill_and_vwap() {
        let p = VolumeProfile::flat(2);
        let mut s = VwapSchedule::new("AAPL", 100.0, p, 0, 2000);
        s.record_fill(100.0, 50.0, 500);
        s.record_fill(104.0, 50.0, 1500);
        let expected = (100.0 * 50.0 + 104.0 * 50.0) / 100.0;
        assert!((s.realized_vwap() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_completion() {
        let p = VolumeProfile::flat(4);
        let mut s = VwapSchedule::new("AAPL", 200.0, p, 0, 4000);
        s.record_fill(100.0, 100.0, 500);
        assert!((s.completion_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_active_bucket() {
        let p = VolumeProfile::flat(4);
        let s = VwapSchedule::new("AAPL", 100.0, p, 0, 4000);
        assert_eq!(s.active_bucket_index(500), Some(0));
        assert_eq!(s.active_bucket_index(2500), Some(2));
    }

    #[test]
    fn test_bucket_fill() {
        let mut b = VolumeBucket::new(0, 0, 1000, 0.25);
        b.target_quantity = 50.0;
        b.record_fill(100.0, 30.0);
        assert!((b.filled_quantity - 30.0).abs() < 1e-9);
        assert!((b.remaining() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_vwap_tracker() {
        let mut t = VwapTracker::new();
        t.record(100.0, 500.0);
        t.record(102.0, 300.0);
        let expected = (100.0 * 500.0 + 102.0 * 300.0) / 800.0;
        assert!((t.vwap() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_vwap_tracker_deviation() {
        let mut t = VwapTracker::new();
        t.record(100.0, 100.0);
        let dev = t.deviation_bps(100.50);
        assert!((dev - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_vwap_tracker_range() {
        let mut t = VwapTracker::new();
        t.record(98.0, 10.0);
        t.record(105.0, 10.0);
        let (lo, hi) = t.price_range();
        assert!((lo - 98.0).abs() < 1e-9);
        assert!((hi - 105.0).abs() < 1e-9);
    }

    #[test]
    fn test_vwap_deviation_buy() {
        let d = VwapDeviation::compute(100.0, 99.50, 1000.0);
        assert!(d.is_favorable_buy()); // Bought below market vwap.
        assert!(d.deviation_bps < 0.0);
    }

    #[test]
    fn test_vwap_deviation_sell() {
        let d = VwapDeviation::compute(100.0, 100.50, 1000.0);
        assert!(d.is_favorable_sell()); // Sold above market vwap.
    }

    #[test]
    fn test_vwap_deviation_impact() {
        let d = VwapDeviation::compute(100.0, 101.0, 500.0);
        assert!((d.notional_impact() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_participation_builder() {
        let p = VolumeProfile::flat(5);
        let s = VwapSchedule::new("AAPL", 100.0, p, 0, 5000).with_max_participation(0.15);
        assert!((s.max_participation_rate - 0.15).abs() < 1e-9);
    }

    #[test]
    fn test_schedule_display() {
        let p = VolumeProfile::flat(4);
        let s = VwapSchedule::new("MSFT", 500.0, p, 0, 4000);
        let txt = format!("{s}");
        assert!(txt.contains("VWAP"));
        assert!(txt.contains("MSFT"));
    }

    #[test]
    fn test_tracker_display() {
        let t = VwapTracker::new();
        let txt = format!("{t}");
        assert!(txt.contains("VwapTracker"));
    }

    #[test]
    fn test_deviation_display() {
        let d = VwapDeviation::compute(100.0, 100.10, 1000.0);
        let txt = format!("{d}");
        assert!(txt.contains("VwapDev"));
    }

    #[test]
    fn test_volume_prediction_error() {
        let mut b = VolumeBucket::new(0, 0, 1000, 0.10);
        b.record_market_volume(150.0);
        // If total market = 1000, actual pct = 0.15, expected 0.10 → error +0.05.
        let err = b.volume_prediction_error(1000.0);
        assert!((err - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_profile_display() {
        let p = VolumeProfile::u_shape(10);
        let txt = format!("{p}");
        assert!(txt.contains("U-SHAPE"));
    }
}
