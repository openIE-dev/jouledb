//! Financial candlestick / OHLC charts with technical indicators: SMA, EMA,
//! Bollinger Bands, RSI, MACD.  Pure Rust, SVG output.

// ── Candle ───────────────────────────────────────────────────────

/// A single OHLCV candlestick.
#[derive(Debug, Clone, Copy)]
pub struct Candle {
    /// Unix timestamp (seconds).
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    /// True when the close is above or equal to the open.
    pub fn is_bullish(&self) -> bool {
        self.close >= self.open
    }

    /// True when the close is below the open.
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Body height (absolute).
    pub fn body_height(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Full range (high − low).
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
}

// ── Moving averages ──────────────────────────────────────────────

/// Simple Moving Average over `period` values.
pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || values.len() < period {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(values.len() - period + 1);
    let mut sum: f64 = values[..period].iter().sum();
    result.push(sum / period as f64);
    for i in period..values.len() {
        sum += values[i] - values[i - period];
        result.push(sum / period as f64);
    }
    result
}

/// Exponential Moving Average.  `smoothing` is typically 2.0.
pub fn ema(values: &[f64], period: usize, smoothing: f64) -> Vec<f64> {
    if period == 0 || values.is_empty() {
        return Vec::new();
    }
    let k = smoothing / (period as f64 + 1.0);
    let mut result = Vec::with_capacity(values.len());
    result.push(values[0]);
    for i in 1..values.len() {
        let prev = result[i - 1];
        result.push(values[i] * k + prev * (1.0 - k));
    }
    result
}

// ── Bollinger Bands ──────────────────────────────────────────────

/// Bollinger band values for one point.
#[derive(Debug, Clone, Copy)]
pub struct BollingerBand {
    pub middle: f64,
    pub upper: f64,
    pub lower: f64,
}

/// Compute Bollinger Bands (SMA ± `num_std` standard deviations).
pub fn bollinger_bands(values: &[f64], period: usize, num_std: f64) -> Vec<BollingerBand> {
    if period == 0 || values.len() < period {
        return Vec::new();
    }
    let means = sma(values, period);
    means
        .iter()
        .enumerate()
        .map(|(i, &mean)| {
            let window = &values[i..i + period];
            let variance =
                window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / period as f64;
            let std_dev = variance.sqrt();
            BollingerBand {
                middle: mean,
                upper: mean + num_std * std_dev,
                lower: mean - num_std * std_dev,
            }
        })
        .collect()
}

// ── RSI ──────────────────────────────────────────────────────────

/// Relative Strength Index (Wilder's smoothing method).
pub fn rsi(closes: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || closes.len() < period + 1 {
        return Vec::new();
    }

    let changes: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();

    // Seed averages from first `period` changes
    let (mut avg_gain, mut avg_loss) = {
        let (g, l) = changes[..period].iter().fold((0.0, 0.0), |(g, l), &c| {
            if c > 0.0 {
                (g + c, l)
            } else {
                (g, l + c.abs())
            }
        });
        (g / period as f64, l / period as f64)
    };

    let mut result = Vec::with_capacity(changes.len() - period + 1);
    let rs = if avg_loss.abs() < f64::EPSILON {
        100.0
    } else {
        avg_gain / avg_loss
    };
    result.push(100.0 - 100.0 / (1.0 + rs));

    for &change in &changes[period..] {
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };
        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
        let rs = if avg_loss.abs() < f64::EPSILON {
            100.0
        } else {
            avg_gain / avg_loss
        };
        result.push(100.0 - 100.0 / (1.0 + rs));
    }

    result
}

// ── MACD ─────────────────────────────────────────────────────────

/// MACD output for one point.
#[derive(Debug, Clone, Copy)]
pub struct MacdPoint {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
}

/// Compute MACD (default 12/26/9) with configurable periods.
pub fn macd(
    closes: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Vec<MacdPoint> {
    if closes.is_empty() {
        return Vec::new();
    }
    let fast_ema = ema(closes, fast_period, 2.0);
    let slow_ema = ema(closes, slow_period, 2.0);

    let macd_line: Vec<f64> = fast_ema
        .iter()
        .zip(slow_ema.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal = ema(&macd_line, signal_period, 2.0);

    macd_line
        .iter()
        .zip(signal.iter())
        .map(|(m, s)| MacdPoint {
            macd_line: *m,
            signal_line: *s,
            histogram: m - s,
        })
        .collect()
}

// ── SVG rendering ────────────────────────────────────────────────

/// Configuration for rendering a candlestick chart.
#[derive(Debug, Clone)]
pub struct CandlestickConfig {
    pub width: f64,
    pub height: f64,
    pub volume_height: f64,
    pub bullish_color: String,
    pub bearish_color: String,
    pub padding: f64,
    pub font_size: f64,
}

impl Default for CandlestickConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 400.0,
            volume_height: 80.0,
            bullish_color: "#26a69a".into(),
            bearish_color: "#ef5350".into(),
            padding: 40.0,
            font_size: 11.0,
        }
    }
}

/// Render candlesticks + volume bars as SVG.
pub fn render_candlestick_chart(candles: &[Candle], cfg: &CandlestickConfig) -> String {
    if candles.is_empty() {
        return format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}"></svg>"#,
            cfg.width, cfg.height
        );
    }

    let n = candles.len();
    let plot_w = cfg.width - 2.0 * cfg.padding;
    let candle_h = cfg.height - cfg.volume_height - 2.0 * cfg.padding;
    let candle_w = (plot_w / n as f64).max(1.0);
    let body_w = candle_w * 0.6;

    let price_min = candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    let price_max = candles
        .iter()
        .map(|c| c.high)
        .fold(f64::NEG_INFINITY, f64::max);
    let price_range = (price_max - price_min).max(f64::EPSILON);

    let vol_max = candles
        .iter()
        .map(|c| c.volume)
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);

    let map_price = |p: f64| -> f64 {
        let t = (p - price_min) / price_range;
        cfg.padding + candle_h * (1.0 - t)
    };

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    for (i, c) in candles.iter().enumerate() {
        let x_center = cfg.padding + (i as f64 + 0.5) * candle_w;
        let color = if c.is_bullish() {
            &cfg.bullish_color
        } else {
            &cfg.bearish_color
        };

        // Wick
        let wick_y1 = map_price(c.high);
        let wick_y2 = map_price(c.low);
        svg.push_str(&format!(
            "<line x1=\"{x_center}\" y1=\"{wick_y1}\" x2=\"{x_center}\" y2=\"{wick_y2}\" stroke=\"{color}\" />"
        ));

        // Body
        let body_top = map_price(c.open.max(c.close));
        let body_bot = map_price(c.open.min(c.close));
        let bh = (body_bot - body_top).max(1.0);
        let bx = x_center - body_w / 2.0;
        svg.push_str(&format!(
            "<rect x=\"{bx}\" y=\"{body_top}\" width=\"{body_w}\" height=\"{bh}\" fill=\"{color}\" />"
        ));

        // Volume bar
        let vol_h = (c.volume / vol_max) * cfg.volume_height;
        let vol_y = cfg.height - cfg.padding - vol_h;
        let vol_x = x_center - body_w / 2.0;
        svg.push_str(&format!(
            "<rect x=\"{vol_x}\" y=\"{vol_y}\" width=\"{body_w}\" height=\"{vol_h}\" fill=\"{color}\" fill-opacity=\"0.4\" />"
        ));
    }

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candles() -> Vec<Candle> {
        vec![
            Candle { timestamp: 1, open: 100.0, high: 110.0, low: 95.0, close: 105.0, volume: 1000.0 },
            Candle { timestamp: 2, open: 105.0, high: 115.0, low: 100.0, close: 102.0, volume: 1500.0 },
            Candle { timestamp: 3, open: 102.0, high: 108.0, low: 98.0, close: 107.0, volume: 1200.0 },
            Candle { timestamp: 4, open: 107.0, high: 120.0, low: 105.0, close: 118.0, volume: 2000.0 },
            Candle { timestamp: 5, open: 118.0, high: 125.0, low: 112.0, close: 115.0, volume: 1800.0 },
        ]
    }

    #[test]
    fn bullish_bearish() {
        let c = Candle { timestamp: 0, open: 100.0, high: 110.0, low: 90.0, close: 105.0, volume: 0.0 };
        assert!(c.is_bullish());
        assert!(!c.is_bearish());

        let c2 = Candle { timestamp: 0, open: 105.0, high: 110.0, low: 90.0, close: 100.0, volume: 0.0 };
        assert!(!c2.is_bullish());
        assert!(c2.is_bearish());
    }

    #[test]
    fn candle_body_height() {
        let c = Candle { timestamp: 0, open: 100.0, high: 110.0, low: 90.0, close: 108.0, volume: 0.0 };
        assert!((c.body_height() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn candle_range() {
        let c = Candle { timestamp: 0, open: 100.0, high: 110.0, low: 90.0, close: 105.0, volume: 0.0 };
        assert!((c.range() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn sma_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&vals, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-9);
        assert!((result[1] - 3.0).abs() < 1e-9);
        assert!((result[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn sma_empty_when_period_too_large() {
        let result = sma(&[1.0, 2.0], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn ema_first_value_equals_input() {
        let vals = vec![10.0, 20.0, 30.0];
        let result = ema(&vals, 3, 2.0);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn ema_smooths_towards_latest() {
        let vals = vec![10.0, 10.0, 10.0, 100.0];
        let result = ema(&vals, 3, 2.0);
        // After a spike, EMA should move towards 100 but not reach it
        assert!(result[3] > 10.0);
        assert!(result[3] < 100.0);
    }

    #[test]
    fn bollinger_bands_symmetry() {
        let vals = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let bands = bollinger_bands(&vals, 3, 2.0);
        // Constant values → stddev = 0 → upper = lower = middle
        for b in &bands {
            assert!((b.middle - 10.0).abs() < 1e-9);
            assert!((b.upper - 10.0).abs() < 1e-9);
            assert!((b.lower - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn bollinger_bands_spread() {
        let vals = vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0];
        let bands = bollinger_bands(&vals, 2, 2.0);
        assert!(!bands.is_empty());
        for b in &bands {
            assert!(b.upper >= b.middle);
            assert!(b.lower <= b.middle);
        }
    }

    #[test]
    fn rsi_range() {
        let closes: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let result = rsi(&closes, 14);
        assert!(!result.is_empty());
        for &v in &result {
            assert!(v >= 0.0 && v <= 100.0, "RSI out of range: {v}");
        }
    }

    #[test]
    fn rsi_all_gains() {
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let result = rsi(&closes, 14);
        assert!(!result.is_empty());
        // All gains → RSI should be 100 (or close)
        assert!(result[0] > 99.0);
    }

    #[test]
    fn macd_output_length() {
        let closes: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let result = macd(&closes, 12, 26, 9);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn macd_histogram_sign() {
        // Steadily increasing → fast EMA above slow EMA → MACD line positive
        let closes: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let result = macd(&closes, 12, 26, 9);
        // After warmup, MACD line should be positive
        assert!(result.last().unwrap().macd_line > 0.0);
    }

    #[test]
    fn render_candlestick_svg() {
        let candles = sample_candles();
        let cfg = CandlestickConfig::default();
        let svg = render_candlestick_chart(&candles, &cfg);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<line"));
    }

    #[test]
    fn render_candlestick_empty() {
        let svg = render_candlestick_chart(&[], &CandlestickConfig::default());
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(!svg.contains("<rect"));
    }

    #[test]
    fn render_candlestick_colors() {
        let candles = sample_candles();
        let cfg = CandlestickConfig::default();
        let svg = render_candlestick_chart(&candles, &cfg);
        assert!(svg.contains(&cfg.bullish_color));
        assert!(svg.contains(&cfg.bearish_color));
    }
}
