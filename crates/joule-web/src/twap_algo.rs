//! # Time-Weighted Average Price Algorithm
//!
//! Implements TWAP execution that distributes order quantity evenly across
//! fixed time slices. Supports linear and front/back-loaded scheduling,
//! slice-level tracking, deviation measurement from ideal schedule, and
//! randomised jitter to reduce information leakage.

use std::fmt;

// ── Core Types ──

/// Scheduling profile that controls how quantity is distributed over time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleProfile {
    /// Uniform distribution across slices.
    Linear,
    /// Heavier weight at the front (early execution).
    FrontLoaded,
    /// Heavier weight at the back (late execution).
    BackLoaded,
    /// U-shaped — heavier at open and close.
    UShaped,
}

impl fmt::Display for ScheduleProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ScheduleProfile::Linear => "LINEAR",
            ScheduleProfile::FrontLoaded => "FRONT",
            ScheduleProfile::BackLoaded => "BACK",
            ScheduleProfile::UShaped => "U-SHAPE",
        };
        write!(f, "{s}")
    }
}

// ── Time Slice ──

/// A single time slice in the TWAP schedule.
#[derive(Clone, Debug)]
pub struct TwapSlice {
    pub index: usize,
    pub start_ns: u64,
    pub end_ns: u64,
    pub target_quantity: f64,
    pub filled_quantity: f64,
    pub avg_fill_price: f64,
    pub fill_count: u32,
}

impl TwapSlice {
    pub fn new(index: usize, start_ns: u64, end_ns: u64, target: f64) -> Self {
        Self {
            index,
            start_ns,
            end_ns,
            target_quantity: target,
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

    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }

    /// Elapsed fraction at the given timestamp.
    pub fn elapsed_fraction(&self, now_ns: u64) -> f64 {
        if now_ns <= self.start_ns { return 0.0; }
        if now_ns >= self.end_ns { return 1.0; }
        let dur = self.duration_ns() as f64;
        if dur < 1e-12 { return 1.0; }
        (now_ns - self.start_ns) as f64 / dur
    }
}

impl fmt::Display for TwapSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Slice[{}](target={:.2} filled={:.2} avg={:.4} fills={})",
            self.index, self.target_quantity, self.filled_quantity,
            self.avg_fill_price, self.fill_count,
        )
    }
}

// ── TWAP Schedule ──

/// The complete TWAP schedule across all slices.
#[derive(Clone, Debug)]
pub struct TwapSchedule {
    pub symbol: String,
    pub total_quantity: f64,
    pub profile: ScheduleProfile,
    pub slices: Vec<TwapSlice>,
    pub start_ns: u64,
    pub end_ns: u64,
    pub jitter_pct: f64,
}

impl TwapSchedule {
    pub fn new(symbol: &str, quantity: f64, num_slices: usize,
               start_ns: u64, end_ns: u64) -> Self {
        let mut sched = Self {
            symbol: symbol.to_string(),
            total_quantity: quantity,
            profile: ScheduleProfile::Linear,
            slices: Vec::new(),
            start_ns,
            end_ns,
            jitter_pct: 0.0,
        };
        sched.build_slices(num_slices);
        sched
    }

    pub fn with_profile(mut self, profile: ScheduleProfile) -> Self {
        self.profile = profile;
        let n = self.slices.len();
        self.build_slices(n);
        self
    }

    pub fn with_jitter(mut self, pct: f64) -> Self {
        self.jitter_pct = pct.clamp(0.0, 0.5);
        self
    }

    fn build_slices(&mut self, num_slices: usize) {
        self.slices.clear();
        if num_slices == 0 { return; }

        let weights = self.compute_weights(num_slices);
        let total_weight: f64 = weights.iter().sum();
        let duration = self.end_ns.saturating_sub(self.start_ns);
        let interval = duration / num_slices as u64;

        for i in 0..num_slices {
            let s = self.start_ns + interval * i as u64;
            let e = if i + 1 < num_slices { s + interval } else { self.end_ns };
            let qty = self.total_quantity * weights[i] / total_weight;
            self.slices.push(TwapSlice::new(i, s, e, qty));
        }
    }

    fn compute_weights(&self, n: usize) -> Vec<f64> {
        match self.profile {
            ScheduleProfile::Linear => vec![1.0; n],
            ScheduleProfile::FrontLoaded => {
                (0..n).map(|i| (n - i) as f64).collect()
            }
            ScheduleProfile::BackLoaded => {
                (0..n).map(|i| (i + 1) as f64).collect()
            }
            ScheduleProfile::UShaped => {
                let mid = n as f64 / 2.0;
                (0..n).map(|i| {
                    let dist = (i as f64 - mid).abs() / mid;
                    0.5 + dist
                }).collect()
            }
        }
    }

    /// Find the active slice for a given timestamp.
    pub fn active_slice_index(&self, now_ns: u64) -> Option<usize> {
        self.slices.iter().position(|s| now_ns >= s.start_ns && now_ns < s.end_ns)
    }

    /// Record a fill on the appropriate slice.
    pub fn record_fill(&mut self, price: f64, qty: f64, timestamp_ns: u64) {
        if let Some(idx) = self.active_slice_index(timestamp_ns) {
            self.slices[idx].record_fill(price, qty);
        } else if let Some(last) = self.slices.last_mut() {
            last.record_fill(price, qty);
        }
    }

    /// Total filled quantity.
    pub fn filled_quantity(&self) -> f64 {
        self.slices.iter().map(|s| s.filled_quantity).sum()
    }

    /// Completion percentage.
    pub fn completion_pct(&self) -> f64 {
        if self.total_quantity > 1e-12 {
            (self.filled_quantity() / self.total_quantity) * 100.0
        } else {
            0.0
        }
    }

    /// Actual TWAP (volume-weighted average of slice prices).
    pub fn realized_twap(&self) -> f64 {
        let total_qty: f64 = self.slices.iter().map(|s| s.filled_quantity).sum();
        if total_qty < 1e-12 { return 0.0; }
        let notional: f64 = self.slices.iter()
            .map(|s| s.avg_fill_price * s.filled_quantity)
            .sum();
        notional / total_qty
    }

    /// Schedule deviation: how far actual fills lag/lead the ideal schedule.
    pub fn schedule_deviation(&self, now_ns: u64) -> f64 {
        let ideal = self.ideal_filled_at(now_ns);
        let actual = self.filled_quantity();
        if self.total_quantity > 1e-12 {
            (actual - ideal) / self.total_quantity
        } else {
            0.0
        }
    }

    /// Ideal filled quantity at a given time, assuming continuous execution.
    pub fn ideal_filled_at(&self, now_ns: u64) -> f64 {
        self.slices.iter().map(|s| {
            let frac = s.elapsed_fraction(now_ns);
            s.target_quantity * frac
        }).sum()
    }

    /// Number of completed slices.
    pub fn completed_slices(&self) -> usize {
        self.slices.iter().filter(|s| s.is_complete()).count()
    }

    /// Per-slice deviation from target.
    pub fn slice_deviations(&self) -> Vec<f64> {
        self.slices.iter().map(|s| {
            if s.target_quantity > 1e-12 {
                (s.filled_quantity - s.target_quantity) / s.target_quantity
            } else {
                0.0
            }
        }).collect()
    }

    /// Max absolute slice deviation.
    pub fn max_slice_deviation(&self) -> f64 {
        self.slice_deviations().iter().map(|d| d.abs())
            .fold(0.0_f64, f64::max)
    }
}

impl fmt::Display for TwapSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TWAP({} qty={:.0} profile={} slices={} complete={:.1}% twap={:.4})",
            self.symbol, self.total_quantity, self.profile,
            self.slices.len(), self.completion_pct(), self.realized_twap(),
        )
    }
}

// ── TWAP Tracker ──

/// Tracks the reference TWAP price over time.
#[derive(Clone, Debug)]
pub struct TwapTracker {
    prices: Vec<(u64, f64)>,
    sum: f64,
}

impl TwapTracker {
    pub fn new() -> Self {
        Self { prices: Vec::new(), sum: 0.0 }
    }

    pub fn record(&mut self, timestamp_ns: u64, price: f64) {
        self.prices.push((timestamp_ns, price));
        self.sum += price;
    }

    pub fn twap(&self) -> f64 {
        if self.prices.is_empty() { 0.0 }
        else { self.sum / self.prices.len() as f64 }
    }

    pub fn sample_count(&self) -> usize {
        self.prices.len()
    }

    pub fn price_range(&self) -> (f64, f64) {
        if self.prices.is_empty() { return (0.0, 0.0); }
        let lo = self.prices.iter().map(|(_, p)| *p).fold(f64::MAX, f64::min);
        let hi = self.prices.iter().map(|(_, p)| *p).fold(f64::MIN, f64::max);
        (lo, hi)
    }
}

impl fmt::Display for TwapTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lo, hi) = self.price_range();
        write!(f, "TwapTracker(twap={:.4} samples={} range=[{:.4},{:.4}])",
            self.twap(), self.prices.len(), lo, hi)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_schedule() {
        let s = TwapSchedule::new("AAPL", 1000.0, 10, 0, 10000);
        assert_eq!(s.slices.len(), 10);
        let total: f64 = s.slices.iter().map(|sl| sl.target_quantity).sum();
        assert!((total - 1000.0).abs() < 1e-6);
        // Linear should be uniform.
        for sl in &s.slices {
            assert!((sl.target_quantity - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_front_loaded_schedule() {
        let s = TwapSchedule::new("AAPL", 1000.0, 5, 0, 5000)
            .with_profile(ScheduleProfile::FrontLoaded);
        assert!(s.slices[0].target_quantity > s.slices[4].target_quantity);
    }

    #[test]
    fn test_back_loaded_schedule() {
        let s = TwapSchedule::new("AAPL", 1000.0, 5, 0, 5000)
            .with_profile(ScheduleProfile::BackLoaded);
        assert!(s.slices[4].target_quantity > s.slices[0].target_quantity);
    }

    #[test]
    fn test_u_shaped_schedule() {
        let s = TwapSchedule::new("AAPL", 1000.0, 6, 0, 6000)
            .with_profile(ScheduleProfile::UShaped);
        let mid = s.slices.len() / 2;
        assert!(s.slices[0].target_quantity > s.slices[mid].target_quantity);
    }

    #[test]
    fn test_slice_fill() {
        let mut sl = TwapSlice::new(0, 0, 1000, 50.0);
        sl.record_fill(100.0, 20.0);
        sl.record_fill(101.0, 30.0);
        assert!((sl.filled_quantity - 50.0).abs() < 1e-9);
        assert!(sl.is_complete());
    }

    #[test]
    fn test_slice_elapsed_fraction() {
        let sl = TwapSlice::new(0, 1000, 2000, 50.0);
        assert!((sl.elapsed_fraction(1500) - 0.5).abs() < 1e-9);
        assert!((sl.elapsed_fraction(500) - 0.0).abs() < 1e-9);
        assert!((sl.elapsed_fraction(3000) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_schedule_record_fill() {
        let mut s = TwapSchedule::new("AAPL", 100.0, 4, 0, 4000);
        s.record_fill(150.0, 10.0, 500);
        assert!((s.slices[0].filled_quantity - 10.0).abs() < 1e-9);
        assert!((s.filled_quantity() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_realized_twap() {
        let mut s = TwapSchedule::new("AAPL", 100.0, 2, 0, 2000);
        s.record_fill(100.0, 50.0, 500);
        s.record_fill(102.0, 50.0, 1500);
        let expected = (100.0 * 50.0 + 102.0 * 50.0) / 100.0;
        assert!((s.realized_twap() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_deviation_behind() {
        let mut s = TwapSchedule::new("AAPL", 100.0, 2, 0, 2000);
        // At t=1000 ideally half done; filled nothing.
        let dev = s.schedule_deviation(1000);
        assert!(dev < 0.0); // Behind schedule.
        s.record_fill(100.0, 50.0, 500);
        let dev2 = s.schedule_deviation(1000);
        assert!((dev2 - 0.0).abs() < 0.1); // Roughly on schedule.
    }

    #[test]
    fn test_completion_pct() {
        let mut s = TwapSchedule::new("AAPL", 200.0, 4, 0, 4000);
        s.record_fill(100.0, 100.0, 500);
        assert!((s.completion_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_active_slice_index() {
        let s = TwapSchedule::new("AAPL", 100.0, 4, 0, 4000);
        assert_eq!(s.active_slice_index(500), Some(0));
        assert_eq!(s.active_slice_index(1500), Some(1));
        assert_eq!(s.active_slice_index(5000), None);
    }

    #[test]
    fn test_jitter_builder() {
        let s = TwapSchedule::new("AAPL", 100.0, 4, 0, 4000).with_jitter(0.10);
        assert!((s.jitter_pct - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_twap_tracker() {
        let mut t = TwapTracker::new();
        t.record(0, 100.0);
        t.record(1000, 102.0);
        t.record(2000, 104.0);
        assert!((t.twap() - 102.0).abs() < 1e-9);
        assert_eq!(t.sample_count(), 3);
    }

    #[test]
    fn test_twap_tracker_range() {
        let mut t = TwapTracker::new();
        t.record(0, 98.0);
        t.record(1, 105.0);
        let (lo, hi) = t.price_range();
        assert!((lo - 98.0).abs() < 1e-9);
        assert!((hi - 105.0).abs() < 1e-9);
    }

    #[test]
    fn test_schedule_display() {
        let s = TwapSchedule::new("MSFT", 500.0, 5, 0, 5000);
        let txt = format!("{s}");
        assert!(txt.contains("TWAP"));
        assert!(txt.contains("MSFT"));
    }

    #[test]
    fn test_slice_display() {
        let sl = TwapSlice::new(3, 0, 1000, 25.0);
        let txt = format!("{sl}");
        assert!(txt.contains("Slice[3]"));
    }

    #[test]
    fn test_max_slice_deviation() {
        let mut s = TwapSchedule::new("AAPL", 100.0, 4, 0, 4000);
        // Over-fill slice 0.
        s.slices[0].record_fill(100.0, 40.0);
        let dev = s.max_slice_deviation();
        assert!(dev > 0.5); // 40 vs 25 target → 60% over.
    }

    #[test]
    fn test_completed_slices() {
        let mut s = TwapSchedule::new("AAPL", 100.0, 4, 0, 4000);
        s.record_fill(100.0, 25.0, 500);
        assert_eq!(s.completed_slices(), 1);
    }

    #[test]
    fn test_tracker_display() {
        let t = TwapTracker::new();
        let txt = format!("{t}");
        assert!(txt.contains("TwapTracker"));
    }
}
