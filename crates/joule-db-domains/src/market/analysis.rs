//! Market Analysis Tools
//!
//! Real-time streaming technical indicators.
//! These are calculated incrementally O(1) per update.

use std::collections::HashMap;

/// Trait for incremental technical indicators
pub trait Indicator {
    /// Update the indicator with a new value (price, volume, etc.)
    fn update(&mut self, value: f64);
    /// Get the current value of the indicator
    fn value(&self) -> f64;
}

/// Relative Strength Index (RSI)
/// Standard momentum oscillator.
pub struct RSI {
    period: usize,
    prev_price: Option<f64>,
    avg_gain: f64,
    avg_loss: f64,
    current_rsi: f64,
    steps: usize,
}

impl RSI {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prev_price: None,
            avg_gain: 0.0,
            avg_loss: 0.0,
            current_rsi: 50.0, // Start neutral
            steps: 0,
        }
    }
}

impl Indicator for RSI {
    fn update(&mut self, price: f64) {
        if let Some(prev) = self.prev_price {
            let change = price - prev;
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };

            if self.steps < self.period {
                // Initial accumulation (SMA)
                self.avg_gain += gain;
                self.avg_loss += loss;
                self.steps += 1;

                if self.steps == self.period {
                    self.avg_gain /= self.period as f64;
                    self.avg_loss /= self.period as f64;
                }
            } else {
                // Wilder's Smoothing (EMA-like)
                let alpha = 1.0 / self.period as f64;
                self.avg_gain = (self.avg_gain * (1.0 - alpha)) + (gain * alpha);
                self.avg_loss = (self.avg_loss * (1.0 - alpha)) + (loss * alpha);
            }

            if self.avg_loss == 0.0 {
                self.current_rsi = 100.0;
            } else {
                let rs = self.avg_gain / self.avg_loss;
                self.current_rsi = 100.0 - (100.0 / (1.0 + rs));
            }
        }
        self.prev_price = Some(price);
    }

    fn value(&self) -> f64 {
        self.current_rsi
    }
}

/// Volume Weighted Average Price (VWAP)
/// Benchmark used by institutions.
pub struct VWAP {
    cumulative_pv: f64, // Price * Volume
    cumulative_vol: f64,
    current_vwap: f64,
}

impl VWAP {
    pub fn new() -> Self {
        Self {
            cumulative_pv: 0.0,
            cumulative_vol: 0.0,
            current_vwap: 0.0,
        }
    }

    pub fn update_pv(&mut self, price: f64, volume: f64) {
        self.cumulative_pv += price * volume;
        self.cumulative_vol += volume;
        if self.cumulative_vol > 0.0 {
            self.current_vwap = self.cumulative_pv / self.cumulative_vol;
        }
    }
}

impl Indicator for VWAP {
    fn update(&mut self, _price: f64) {
        // VWAP needs volume, use update_pv instead
    }

    fn value(&self) -> f64 {
        self.current_vwap
    }
}

/// Container for analysis state per symbol
pub struct MarketAnalyzer {
    rsi_registry: HashMap<String, RSI>,
    vwap_registry: HashMap<String, VWAP>,
}

impl MarketAnalyzer {
    pub fn new() -> Self {
        Self {
            rsi_registry: HashMap::new(),
            vwap_registry: HashMap::new(),
        }
    }

    pub fn update_trade(&mut self, symbol: &str, price: f64, qty: f64) {
        // Update RSI
        let rsi = self
            .rsi_registry
            .entry(symbol.to_string())
            .or_insert(RSI::new(14));
        rsi.update(price);

        // Update VWAP
        let vwap = self
            .vwap_registry
            .entry(symbol.to_string())
            .or_insert(VWAP::new());
        vwap.update_pv(price, qty);
    }

    pub fn get_rsi(&self, symbol: &str) -> Option<f64> {
        self.rsi_registry.get(symbol).map(|i| i.value())
    }

    pub fn get_vwap(&self, symbol: &str) -> Option<f64> {
        self.vwap_registry.get(symbol).map(|i| i.value())
    }
}
