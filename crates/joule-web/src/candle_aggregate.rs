//! Candlestick aggregation — OHLCV builder from ticks, time-based candles
//! (1m/5m/1h/1d), volume-based candles, Renko bricks, Heikin-Ashi transform,
//! gap detection.
//!
//! Pure-Rust candlestick construction from raw tick data with multiple
//! aggregation modes:
//!
//! - [`OhlcvCandle`] — canonical Open/High/Low/Close/Volume candle
//! - [`CandleBuilder`] — time-based candle aggregation from trade ticks
//! - [`VolumeCandleBuilder`] — volume-bucket candles
//! - [`RenkoBrickBuilder`] — constant-range Renko bricks
//! - [`HeikinAshi`] — smoothed Heikin-Ashi transform
//! - [`GapDetector`] — inter-candle gap detection

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum CandleError {
    InvalidInterval(String),
    InvalidBrickSize(String),
    EmptyInput(String),
    InvalidPrice(String),
}

impl fmt::Display for CandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInterval(s) => write!(f, "invalid interval: {s}"),
            Self::InvalidBrickSize(s) => write!(f, "invalid brick size: {s}"),
            Self::EmptyInput(s) => write!(f, "empty input: {s}"),
            Self::InvalidPrice(s) => write!(f, "invalid price: {s}"),
        }
    }
}

impl std::error::Error for CandleError {}

// ── Candle Interval ─────────────────────────────────────────────

/// Standard time intervals for candle aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandleInterval {
    Minute1,
    Minute5,
    Minute15,
    Minute30,
    Hour1,
    Hour4,
    Day1,
    Week1,
}

impl CandleInterval {
    /// Duration in microseconds.
    pub fn duration_us(self) -> u64 {
        match self {
            Self::Minute1  =>        60_000_000,
            Self::Minute5  =>       300_000_000,
            Self::Minute15 =>       900_000_000,
            Self::Minute30 =>     1_800_000_000,
            Self::Hour1    =>     3_600_000_000,
            Self::Hour4    =>    14_400_000_000,
            Self::Day1     =>    86_400_000_000,
            Self::Week1    =>   604_800_000_000,
        }
    }

    /// Align a timestamp to the start of its interval bucket.
    pub fn bucket_start(self, timestamp_us: u64) -> u64 {
        let d = self.duration_us();
        (timestamp_us / d) * d
    }
}

impl fmt::Display for CandleInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minute1  => write!(f, "1m"),
            Self::Minute5  => write!(f, "5m"),
            Self::Minute15 => write!(f, "15m"),
            Self::Minute30 => write!(f, "30m"),
            Self::Hour1    => write!(f, "1h"),
            Self::Hour4    => write!(f, "4h"),
            Self::Day1     => write!(f, "1d"),
            Self::Week1    => write!(f, "1w"),
        }
    }
}

// ── OHLCV Candle ────────────────────────────────────────────────

/// Open-High-Low-Close-Volume candle.
#[derive(Debug, Clone, PartialEq)]
pub struct OhlcvCandle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub trade_count: u64,
    pub start_us: u64,
    pub end_us: u64,
    pub vwap: f64,
}

impl OhlcvCandle {
    pub fn new(open: f64, start_us: u64) -> Self {
        Self {
            open, high: open, low: open, close: open,
            volume: 0.0, trade_count: 0, start_us, end_us: start_us,
            vwap: 0.0,
        }
    }

    /// Update the candle with a new trade tick.
    pub fn update(&mut self, price: f64, size: f64, timestamp_us: u64) {
        if price > self.high { self.high = price; }
        if price < self.low { self.low = price; }
        self.close = price;
        let prev_notional = self.vwap * self.volume;
        self.volume += size;
        if self.volume > 0.0 {
            self.vwap = (prev_notional + price * size) / self.volume;
        }
        self.trade_count += 1;
        self.end_us = timestamp_us;
    }

    pub fn range(&self) -> f64 { self.high - self.low }

    pub fn body(&self) -> f64 { (self.close - self.open).abs() }

    pub fn is_bullish(&self) -> bool { self.close >= self.open }

    pub fn is_bearish(&self) -> bool { self.close < self.open }

    /// Upper shadow length.
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Lower shadow length.
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Body-to-range ratio (0.0 = doji, 1.0 = marubozu).
    pub fn body_ratio(&self) -> f64 {
        let r = self.range();
        if r == 0.0 { return 0.0; }
        self.body() / r
    }
}

impl fmt::Display for OhlcvCandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OHLCV[O={:.4} H={:.4} L={:.4} C={:.4} V={:.0} n={}]",
               self.open, self.high, self.low, self.close, self.volume, self.trade_count)
    }
}

// ── Time-Based Candle Builder ───────────────────────────────────

/// Aggregates trade ticks into time-bucketed candles.
pub struct CandleBuilder {
    interval: CandleInterval,
    current: Option<OhlcvCandle>,
    current_bucket: u64,
    completed: Vec<OhlcvCandle>,
}

impl CandleBuilder {
    pub fn new(interval: CandleInterval) -> Self {
        Self { interval, current: None, current_bucket: 0, completed: Vec::new() }
    }

    pub fn with_interval(mut self, interval: CandleInterval) -> Self {
        self.interval = interval;
        self
    }

    /// Feed a trade tick. Returns `Some` if a candle was completed.
    pub fn on_trade(&mut self, price: f64, size: f64, timestamp_us: u64) -> Option<&OhlcvCandle> {
        let bucket = self.interval.bucket_start(timestamp_us);

        if self.current.is_none() || bucket != self.current_bucket {
            // Rotate out the old candle
            if let Some(candle) = self.current.take() {
                self.completed.push(candle);
            }
            self.current = Some(OhlcvCandle::new(price, timestamp_us));
            self.current_bucket = bucket;
        }

        if let Some(ref mut c) = self.current {
            c.update(price, size, timestamp_us);
        }

        // Return last completed candle if we just rotated
        if self.completed.last().map_or(false, |c| c.end_us < timestamp_us) {
            self.completed.last()
        } else {
            None
        }
    }

    /// Force-close the current candle.
    pub fn flush(&mut self) -> Option<OhlcvCandle> {
        self.current.take().inspect(|c| {
            self.completed.push(c.clone());
        })
    }

    pub fn completed_candles(&self) -> &[OhlcvCandle] { &self.completed }

    pub fn current_candle(&self) -> Option<&OhlcvCandle> { self.current.as_ref() }
}

impl fmt::Display for CandleBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CandleBuilder[{} completed={} active={}]",
               self.interval, self.completed.len(), self.current.is_some())
    }
}

// ── Volume-Based Candle Builder ─────────────────────────────────

/// Aggregates ticks into candles that each contain a target volume.
pub struct VolumeCandleBuilder {
    target_volume: f64,
    current: Option<OhlcvCandle>,
    completed: Vec<OhlcvCandle>,
}

impl VolumeCandleBuilder {
    pub fn new(target_volume: f64) -> Self {
        Self { target_volume, current: None, completed: Vec::new() }
    }

    pub fn with_target_volume(mut self, v: f64) -> Self {
        self.target_volume = v;
        self
    }

    pub fn on_trade(&mut self, price: f64, size: f64, timestamp_us: u64) -> Option<&OhlcvCandle> {
        if self.current.is_none() {
            self.current = Some(OhlcvCandle::new(price, timestamp_us));
        }

        if let Some(ref mut c) = self.current {
            c.update(price, size, timestamp_us);

            if c.volume >= self.target_volume {
                let done = self.current.take().unwrap();
                self.completed.push(done);
                return self.completed.last();
            }
        }
        None
    }

    pub fn completed_candles(&self) -> &[OhlcvCandle] { &self.completed }

    pub fn flush(&mut self) -> Option<OhlcvCandle> { self.current.take() }
}

impl fmt::Display for VolumeCandleBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VolumeCandleBuilder[target={:.0} completed={}]",
               self.target_volume, self.completed.len())
    }
}

// ── Renko Brick Builder ────────────────────────────────────────

/// Direction of a Renko brick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickDirection {
    Up,
    Down,
}

impl fmt::Display for BrickDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Up => write!(f, "Up"),
            Self::Down => write!(f, "Down"),
        }
    }
}

/// A single Renko brick.
#[derive(Debug, Clone, PartialEq)]
pub struct RenkoBrick {
    pub open: f64,
    pub close: f64,
    pub direction: BrickDirection,
    pub timestamp_us: u64,
}

impl fmt::Display for RenkoBrick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Renko[{} {:.4}->{:.4}]", self.direction, self.open, self.close)
    }
}

/// Builds Renko bricks from price ticks using a fixed brick size.
pub struct RenkoBrickBuilder {
    brick_size: f64,
    last_close: Option<f64>,
    bricks: Vec<RenkoBrick>,
}

impl RenkoBrickBuilder {
    pub fn new(brick_size: f64) -> Result<Self, CandleError> {
        if brick_size <= 0.0 {
            return Err(CandleError::InvalidBrickSize(
                format!("brick size must be positive, got {brick_size}")));
        }
        Ok(Self { brick_size, last_close: None, bricks: Vec::new() })
    }

    pub fn on_price(&mut self, price: f64, timestamp_us: u64) -> Vec<RenkoBrick> {
        let mut new_bricks = Vec::new();
        let base = match self.last_close {
            Some(lc) => lc,
            None => {
                self.last_close = Some(price);
                return new_bricks;
            }
        };

        let diff = price - base;
        let count = (diff.abs() / self.brick_size).floor() as usize;

        if diff > 0.0 {
            for i in 0..count {
                let open = base + (i as f64) * self.brick_size;
                let close = open + self.brick_size;
                let brick = RenkoBrick {
                    open, close, direction: BrickDirection::Up, timestamp_us,
                };
                new_bricks.push(brick);
            }
        } else {
            for i in 0..count {
                let open = base - (i as f64) * self.brick_size;
                let close = open - self.brick_size;
                let brick = RenkoBrick {
                    open, close, direction: BrickDirection::Down, timestamp_us,
                };
                new_bricks.push(brick);
            }
        }

        if let Some(last) = new_bricks.last() {
            self.last_close = Some(last.close);
        }
        self.bricks.extend(new_bricks.clone());
        new_bricks
    }

    pub fn bricks(&self) -> &[RenkoBrick] { &self.bricks }
}

impl fmt::Display for RenkoBrickBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RenkoBrickBuilder[size={:.4} bricks={}]",
               self.brick_size, self.bricks.len())
    }
}

// ── Heikin-Ashi Transform ───────────────────────────────────────

/// Transforms standard OHLCV candles into smoothed Heikin-Ashi candles.
pub struct HeikinAshi {
    prev_ha_open: Option<f64>,
    prev_ha_close: Option<f64>,
}

impl HeikinAshi {
    pub fn new() -> Self { Self { prev_ha_open: None, prev_ha_close: None } }

    pub fn transform(&mut self, candle: &OhlcvCandle) -> OhlcvCandle {
        let ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0;
        let ha_open = match (self.prev_ha_open, self.prev_ha_close) {
            (Some(po), Some(pc)) => (po + pc) / 2.0,
            _ => (candle.open + candle.close) / 2.0,
        };
        let ha_high = candle.high.max(ha_open).max(ha_close);
        let ha_low = candle.low.min(ha_open).min(ha_close);

        self.prev_ha_open = Some(ha_open);
        self.prev_ha_close = Some(ha_close);

        OhlcvCandle {
            open: ha_open,
            high: ha_high,
            low: ha_low,
            close: ha_close,
            volume: candle.volume,
            trade_count: candle.trade_count,
            start_us: candle.start_us,
            end_us: candle.end_us,
            vwap: candle.vwap,
        }
    }

    pub fn transform_series(&mut self, candles: &[OhlcvCandle]) -> Vec<OhlcvCandle> {
        candles.iter().map(|c| self.transform(c)).collect()
    }

    pub fn reset(&mut self) {
        self.prev_ha_open = None;
        self.prev_ha_close = None;
    }
}

impl fmt::Display for HeikinAshi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HeikinAshi[initialized={}]", self.prev_ha_open.is_some())
    }
}

// ── Gap Detector ────────────────────────────────────────────────

/// Type of price gap between consecutive candles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapType {
    GapUp,
    GapDown,
}

impl fmt::Display for GapType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GapUp => write!(f, "GapUp"),
            Self::GapDown => write!(f, "GapDown"),
        }
    }
}

/// A detected gap between two consecutive candles.
#[derive(Debug, Clone, PartialEq)]
pub struct Gap {
    pub gap_type: GapType,
    pub gap_size: f64,
    pub gap_pct: f64,
    pub prev_close: f64,
    pub next_open: f64,
    pub index: usize,
}

impl fmt::Display for Gap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gap[{} {:.4} ({:.2}%) at {}]",
               self.gap_type, self.gap_size, self.gap_pct * 100.0, self.index)
    }
}

/// Detects gaps between consecutive candles.
pub struct GapDetector {
    min_gap_pct: f64,
}

impl GapDetector {
    pub fn new(min_gap_pct: f64) -> Self { Self { min_gap_pct } }

    pub fn with_min_gap_pct(mut self, pct: f64) -> Self {
        self.min_gap_pct = pct;
        self
    }

    pub fn detect(&self, candles: &[OhlcvCandle]) -> Vec<Gap> {
        let mut gaps = Vec::new();
        for i in 1..candles.len() {
            let prev_close = candles[i - 1].close;
            let next_open = candles[i].open;
            if prev_close == 0.0 { continue; }
            let gap_size = next_open - prev_close;
            let gap_pct = gap_size / prev_close;
            if gap_pct.abs() >= self.min_gap_pct {
                let gap_type = if gap_pct > 0.0 { GapType::GapUp } else { GapType::GapDown };
                gaps.push(Gap { gap_type, gap_size, gap_pct, prev_close, next_open, index: i });
            }
        }
        gaps
    }
}

impl fmt::Display for GapDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapDetector[min_pct={:.4}]", self.min_gap_pct)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(o: f64, h: f64, l: f64, c: f64, v: f64, start: u64, end: u64) -> OhlcvCandle {
        OhlcvCandle {
            open: o, high: h, low: l, close: c, volume: v,
            trade_count: 1, start_us: start, end_us: end, vwap: c,
        }
    }

    #[test]
    fn candle_update() {
        let mut c = OhlcvCandle::new(100.0, 0);
        c.update(105.0, 10.0, 1);
        c.update(98.0, 5.0, 2);
        c.update(102.0, 8.0, 3);
        assert!((c.high - 105.0).abs() < 1e-9);
        assert!((c.low - 98.0).abs() < 1e-9);
        assert!((c.close - 102.0).abs() < 1e-9);
        assert!((c.volume - 23.0).abs() < 1e-9);
        assert_eq!(c.trade_count, 3);
    }

    #[test]
    fn candle_bullish_bearish() {
        let bull = make_candle(100.0, 110.0, 99.0, 108.0, 100.0, 0, 1);
        assert!(bull.is_bullish());
        let bear = make_candle(100.0, 101.0, 95.0, 96.0, 100.0, 0, 1);
        assert!(bear.is_bearish());
    }

    #[test]
    fn candle_shadows() {
        let c = make_candle(100.0, 110.0, 90.0, 105.0, 100.0, 0, 1);
        assert!((c.upper_shadow() - 5.0).abs() < 1e-9);
        assert!((c.lower_shadow() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn candle_body_ratio() {
        let doji = make_candle(100.0, 105.0, 95.0, 100.0, 100.0, 0, 1);
        assert!((doji.body_ratio()).abs() < 1e-9);
    }

    #[test]
    fn interval_bucket() {
        let ts = 90_000_000u64; // 1.5 minutes in us
        let bucket = CandleInterval::Minute1.bucket_start(ts);
        assert_eq!(bucket, 60_000_000);
    }

    #[test]
    fn candle_builder_rotates() {
        let mut builder = CandleBuilder::new(CandleInterval::Minute1);
        // First minute
        builder.on_trade(100.0, 10.0, 10_000_000);
        builder.on_trade(101.0, 5.0, 30_000_000);
        // Second minute triggers rotation
        builder.on_trade(102.0, 8.0, 70_000_000);
        assert_eq!(builder.completed_candles().len(), 1);
        let c = &builder.completed_candles()[0];
        assert!((c.open - 100.0).abs() < 1e-9);
        assert!((c.close - 101.0).abs() < 1e-9);
    }

    #[test]
    fn candle_builder_flush() {
        let mut builder = CandleBuilder::new(CandleInterval::Minute1);
        builder.on_trade(100.0, 10.0, 10_000_000);
        let flushed = builder.flush();
        assert!(flushed.is_some());
        assert!(builder.current_candle().is_none());
    }

    #[test]
    fn volume_candle_builder() {
        let mut vb = VolumeCandleBuilder::new(100.0);
        assert!(vb.on_trade(50.0, 40.0, 0).is_none());
        assert!(vb.on_trade(51.0, 30.0, 1).is_none());
        let result = vb.on_trade(52.0, 40.0, 2);
        assert!(result.is_some());
        assert!((result.unwrap().volume - 110.0).abs() < 1e-9);
    }

    #[test]
    fn renko_basic() {
        let mut rb = RenkoBrickBuilder::new(1.0).unwrap();
        let bricks = rb.on_price(100.0, 0);
        assert!(bricks.is_empty()); // sets baseline
        let bricks = rb.on_price(102.5, 1);
        assert_eq!(bricks.len(), 2);
        assert_eq!(bricks[0].direction, BrickDirection::Up);
    }

    #[test]
    fn renko_down() {
        let mut rb = RenkoBrickBuilder::new(1.0).unwrap();
        rb.on_price(100.0, 0);
        let bricks = rb.on_price(97.5, 1);
        assert_eq!(bricks.len(), 2);
        assert_eq!(bricks[0].direction, BrickDirection::Down);
    }

    #[test]
    fn renko_no_brick() {
        let mut rb = RenkoBrickBuilder::new(1.0).unwrap();
        rb.on_price(100.0, 0);
        let bricks = rb.on_price(100.5, 1);
        assert!(bricks.is_empty());
    }

    #[test]
    fn renko_invalid_size() {
        assert!(RenkoBrickBuilder::new(0.0).is_err());
        assert!(RenkoBrickBuilder::new(-1.0).is_err());
    }

    #[test]
    fn heikin_ashi_first() {
        let mut ha = HeikinAshi::new();
        let c = make_candle(100.0, 110.0, 95.0, 105.0, 1000.0, 0, 1);
        let hac = ha.transform(&c);
        assert!((hac.close - 102.5).abs() < 1e-9); // (100+110+95+105)/4
        assert!((hac.open - 102.5).abs() < 1e-9);  // (100+105)/2
    }

    #[test]
    fn heikin_ashi_series() {
        let mut ha = HeikinAshi::new();
        let candles = vec![
            make_candle(100.0, 110.0, 95.0, 105.0, 100.0, 0, 1),
            make_candle(106.0, 112.0, 104.0, 110.0, 100.0, 2, 3),
        ];
        let results = ha.transform_series(&candles);
        assert_eq!(results.len(), 2);
        assert!(results[1].close > results[0].close);
    }

    #[test]
    fn gap_detector_up() {
        let det = GapDetector::new(0.01);
        let candles = vec![
            make_candle(100.0, 102.0, 99.0, 100.0, 100.0, 0, 1),
            make_candle(102.0, 104.0, 101.0, 103.0, 100.0, 2, 3),
        ];
        let gaps = det.detect(&candles);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].gap_type, GapType::GapUp);
    }

    #[test]
    fn gap_detector_no_gap() {
        let det = GapDetector::new(0.01);
        let candles = vec![
            make_candle(100.0, 102.0, 99.0, 100.0, 100.0, 0, 1),
            make_candle(100.5, 102.0, 99.0, 101.0, 100.0, 2, 3),
        ];
        let gaps = det.detect(&candles);
        assert!(gaps.is_empty());
    }

    #[test]
    fn gap_detector_down() {
        let det = GapDetector::new(0.01);
        let candles = vec![
            make_candle(100.0, 102.0, 99.0, 100.0, 100.0, 0, 1),
            make_candle(98.0, 99.0, 97.0, 98.5, 100.0, 2, 3),
        ];
        let gaps = det.detect(&candles);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].gap_type, GapType::GapDown);
    }

    #[test]
    fn display_impls() {
        let c = make_candle(100.0, 110.0, 90.0, 105.0, 1000.0, 0, 1);
        assert!(format!("{c}").contains("OHLCV"));
        let builder = CandleBuilder::new(CandleInterval::Hour1);
        assert!(format!("{builder}").contains("1h"));
        let rb = RenkoBrickBuilder::new(1.0).unwrap();
        assert!(format!("{rb}").contains("Renko"));
    }
}
