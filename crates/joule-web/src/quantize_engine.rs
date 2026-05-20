//! Post-training quantization engine: INT8, INT4, dynamic quantization,
//! calibration, and scale/zero-point computation.
//!
//! Reduces model size and speeds up inference by mapping floating-point
//! weights and activations to lower-precision integer representations.
//! Supports symmetric and asymmetric quantization with per-tensor or
//! per-channel granularity.

use std::collections::HashMap;
use std::fmt;

// ── Precision ──────────────────────────────────────────────────

/// Target quantization bit-width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Int4,
    Int8,
    Int16,
    Float16,
}

impl Precision {
    /// Number of bits.
    pub fn bits(&self) -> u32 {
        match self {
            Precision::Int4 => 4,
            Precision::Int8 => 8,
            Precision::Int16 => 16,
            Precision::Float16 => 16,
        }
    }

    /// Representable range for integer precisions.
    pub fn range(&self) -> (f64, f64) {
        match self {
            Precision::Int4 => (-8.0, 7.0),
            Precision::Int8 => (-128.0, 127.0),
            Precision::Int16 => (-32768.0, 32767.0),
            Precision::Float16 => (f64::MIN, f64::MAX), // placeholder
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::Int4 => write!(f, "INT4"),
            Precision::Int8 => write!(f, "INT8"),
            Precision::Int16 => write!(f, "INT16"),
            Precision::Float16 => write!(f, "FP16"),
        }
    }
}

// ── Quantization Mode ──────────────────────────────────────────

/// Symmetric vs asymmetric quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Zero-point is always zero; range is symmetric around zero.
    Symmetric,
    /// Zero-point can be non-zero; maps [min, max] to [qmin, qmax].
    Asymmetric,
}

impl fmt::Display for QuantMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantMode::Symmetric => write!(f, "symmetric"),
            QuantMode::Asymmetric => write!(f, "asymmetric"),
        }
    }
}

// ── Granularity ────────────────────────────────────────────────

/// Quantization granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// One scale/zero-point per entire tensor.
    PerTensor,
    /// One scale/zero-point per output channel.
    PerChannel,
}

// ── QuantParams ────────────────────────────────────────────────

/// Scale and zero-point parameters for a quantized tensor.
#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: Vec<f64>,
    pub zero_point: Vec<i64>,
    pub precision: Precision,
    pub mode: QuantMode,
}

impl QuantParams {
    /// Quantize a float value to integer.
    pub fn quantize_value(&self, val: f64, channel: usize) -> i64 {
        let idx = channel.min(self.scale.len() - 1);
        let s = self.scale[idx];
        let z = self.zero_point[idx];
        let (qmin, qmax) = self.precision.range();
        let q = (val / s + z as f64).round();
        q.clamp(qmin, qmax) as i64
    }

    /// Dequantize an integer back to float.
    pub fn dequantize_value(&self, val: i64, channel: usize) -> f64 {
        let idx = channel.min(self.scale.len() - 1);
        self.scale[idx] * (val - self.zero_point[idx]) as f64
    }
}

impl fmt::Display for QuantParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantParams(precision={}, mode={}, channels={})",
            self.precision,
            self.mode,
            self.scale.len()
        )
    }
}

// ── Calibration ────────────────────────────────────────────────

/// Calibration method for determining quantization ranges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationMethod {
    /// Use min/max of observed values.
    MinMax,
    /// Use percentile clipping (e.g. 99.99th percentile).
    Percentile(f64),
    /// Entropy-based calibration (KL divergence).
    Entropy,
}

/// Collects statistics from calibration data to compute quantization parameters.
#[derive(Debug)]
pub struct CalibrationCollector {
    method: CalibrationMethod,
    /// Per-channel tracked values: (min, max, count, sum, sum_sq).
    channel_stats: Vec<ChannelStat>,
    /// Histogram for entropy method.
    histograms: Vec<Vec<u64>>,
    num_bins: usize,
}

#[derive(Debug, Clone)]
struct ChannelStat {
    min_val: f64,
    max_val: f64,
    count: u64,
    sum: f64,
    sum_sq: f64,
    /// Sorted sample for percentile (capped).
    samples: Vec<f64>,
}

impl ChannelStat {
    fn new() -> Self {
        Self {
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            samples: Vec::new(),
        }
    }

    fn observe(&mut self, val: f64) {
        if val < self.min_val {
            self.min_val = val;
        }
        if val > self.max_val {
            self.max_val = val;
        }
        self.count += 1;
        self.sum += val;
        self.sum_sq += val * val;
        if self.samples.len() < 10_000 {
            self.samples.push(val);
        }
    }

    fn mean(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.count as f64 }
    }

    fn std_dev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let variance = (self.sum_sq / n) - (self.mean() * self.mean());
        if variance < 0.0 { 0.0 } else { variance.sqrt() }
    }
}

impl CalibrationCollector {
    pub fn new(method: CalibrationMethod, num_channels: usize) -> Self {
        Self {
            method,
            channel_stats: (0..num_channels).map(|_| ChannelStat::new()).collect(),
            histograms: (0..num_channels).map(|_| vec![0u64; 2048]).collect(),
            num_bins: 2048,
        }
    }

    /// Observe a batch of values for a given channel.
    pub fn observe(&mut self, channel: usize, values: &[f64]) {
        if channel >= self.channel_stats.len() {
            return;
        }
        for &v in values {
            self.channel_stats[channel].observe(v);
        }
    }

    /// Compute quantization parameters after calibration.
    pub fn compute_params(&mut self, precision: Precision, mode: QuantMode) -> QuantParams {
        let (qmin, qmax) = precision.range();
        let qrange = qmax - qmin;
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();

        for stat in &mut self.channel_stats {
            let (rmin, rmax) = match self.method {
                CalibrationMethod::MinMax => (stat.min_val, stat.max_val),
                CalibrationMethod::Percentile(pct) => {
                    stat.samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let n = stat.samples.len();
                    if n == 0 {
                        (0.0, 0.0)
                    } else {
                        let lo = ((1.0 - pct / 100.0) * n as f64) as usize;
                        let hi = ((pct / 100.0) * n as f64).min(n as f64 - 1.0) as usize;
                        (stat.samples[lo], stat.samples[hi])
                    }
                }
                CalibrationMethod::Entropy => {
                    // Approximate: use 3-sigma clipping
                    let mu = stat.mean();
                    let sigma = stat.std_dev();
                    (mu - 3.0 * sigma, mu + 3.0 * sigma)
                }
            };

            let (scale, zp) = match mode {
                QuantMode::Symmetric => {
                    let abs_max = rmin.abs().max(rmax.abs());
                    let s = if abs_max == 0.0 { 1.0 } else { abs_max / (qmax) };
                    (s, 0i64)
                }
                QuantMode::Asymmetric => {
                    let range = rmax - rmin;
                    let s = if range == 0.0 { 1.0 } else { range / qrange };
                    let z = (qmin - rmin / s).round() as i64;
                    (s, z)
                }
            };

            scales.push(scale);
            zero_points.push(zp);
        }

        QuantParams { scale: scales, zero_point: zero_points, precision, mode }
    }
}

// ── Quantized Tensor ───────────────────────────────────────────

/// A tensor stored in quantized integer format.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub shape: Vec<usize>,
    pub data: Vec<i64>,
    pub params: QuantParams,
}

impl QuantizedTensor {
    /// Quantize a float tensor.
    pub fn from_float(shape: &[usize], values: &[f64], params: &QuantParams) -> Self {
        let data: Vec<i64> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let ch = if params.scale.len() > 1 && shape.len() >= 2 {
                    i / (values.len() / shape[0])
                } else {
                    0
                };
                params.quantize_value(v, ch)
            })
            .collect();
        Self { shape: shape.to_vec(), data, params: params.clone() }
    }

    /// Dequantize back to float.
    pub fn to_float(&self) -> Vec<f64> {
        let total = self.data.len();
        self.data
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let ch = if self.params.scale.len() > 1 && self.shape.len() >= 2 {
                    i / (total / self.shape[0])
                } else {
                    0
                };
                self.params.dequantize_value(v, ch)
            })
            .collect()
    }

    /// Compression ratio vs float64.
    pub fn compression_ratio(&self) -> f64 {
        64.0 / self.params.precision.bits() as f64
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Display for QuantizedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantizedTensor(shape={:?}, precision={}, ratio={:.1}x)",
            self.shape,
            self.params.precision,
            self.compression_ratio()
        )
    }
}

// ── Quantize Config ────────────────────────────────────────────

/// Configuration builder for quantization.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    pub precision: Precision,
    pub mode: QuantMode,
    pub granularity: Granularity,
    pub calibration: CalibrationMethod,
    pub per_layer_config: HashMap<String, Precision>,
}

impl QuantizeConfig {
    pub fn new() -> Self {
        Self {
            precision: Precision::Int8,
            mode: QuantMode::Symmetric,
            granularity: Granularity::PerTensor,
            calibration: CalibrationMethod::MinMax,
            per_layer_config: HashMap::new(),
        }
    }

    pub fn with_precision(mut self, p: Precision) -> Self {
        self.precision = p;
        self
    }

    pub fn with_mode(mut self, m: QuantMode) -> Self {
        self.mode = m;
        self
    }

    pub fn with_granularity(mut self, g: Granularity) -> Self {
        self.granularity = g;
        self
    }

    pub fn with_calibration(mut self, c: CalibrationMethod) -> Self {
        self.calibration = c;
        self
    }

    pub fn with_layer_precision(mut self, layer: impl Into<String>, p: Precision) -> Self {
        self.per_layer_config.insert(layer.into(), p);
        self
    }
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for QuantizeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantizeConfig(precision={}, mode={}, granularity={:?})",
            self.precision, self.mode, self.granularity
        )
    }
}

// ── Dynamic Quantization ───────────────────────────────────────

/// Dynamically quantize activations at runtime based on observed range.
pub fn dynamic_quantize(values: &[f64], precision: Precision) -> QuantizedTensor {
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (qmin, qmax) = precision.range();

    let abs_max = min_val.abs().max(max_val.abs());
    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / qmax };

    let params = QuantParams {
        scale: vec![scale],
        zero_point: vec![0],
        precision,
        mode: QuantMode::Symmetric,
    };

    let data: Vec<i64> = values
        .iter()
        .map(|v| {
            let q = (v / scale).round();
            q.clamp(qmin, qmax) as i64
        })
        .collect();

    QuantizedTensor { shape: vec![values.len()], data, params }
}

/// Compute mean-squared quantization error.
pub fn quantization_error(original: &[f64], quantized: &QuantizedTensor) -> f64 {
    let deq = quantized.to_float();
    let n = original.len().min(deq.len());
    if n == 0 {
        return 0.0;
    }
    let mse: f64 = original.iter().zip(&deq).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / n as f64;
    mse
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_bits() {
        assert_eq!(Precision::Int4.bits(), 4);
        assert_eq!(Precision::Int8.bits(), 8);
        assert_eq!(Precision::Int16.bits(), 16);
    }

    #[test]
    fn test_precision_range() {
        let (lo, hi) = Precision::Int8.range();
        assert_eq!(lo, -128.0);
        assert_eq!(hi, 127.0);
    }

    #[test]
    fn test_symmetric_quant_zero() {
        let params = QuantParams {
            scale: vec![0.01],
            zero_point: vec![0],
            precision: Precision::Int8,
            mode: QuantMode::Symmetric,
        };
        assert_eq!(params.quantize_value(0.0, 0), 0);
        assert!((params.dequantize_value(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_roundtrip() {
        let params = QuantParams {
            scale: vec![0.1],
            zero_point: vec![0],
            precision: Precision::Int8,
            mode: QuantMode::Symmetric,
        };
        let val = 1.5;
        let q = params.quantize_value(val, 0);
        let dq = params.dequantize_value(q, 0);
        assert!((dq - val).abs() < 0.1);
    }

    #[test]
    fn test_asymmetric_quant() {
        let params = QuantParams {
            scale: vec![0.02],
            zero_point: vec![10],
            precision: Precision::Int8,
            mode: QuantMode::Asymmetric,
        };
        let q = params.quantize_value(0.0, 0);
        assert_eq!(q, 10); // 0/0.02 + 10 = 10
    }

    #[test]
    fn test_clamp_to_range() {
        let params = QuantParams {
            scale: vec![0.001],
            zero_point: vec![0],
            precision: Precision::Int8,
            mode: QuantMode::Symmetric,
        };
        let q = params.quantize_value(1000.0, 0);
        assert_eq!(q, 127); // clamped
    }

    #[test]
    fn test_quantized_tensor_from_float() {
        let params = QuantParams {
            scale: vec![0.5],
            zero_point: vec![0],
            precision: Precision::Int8,
            mode: QuantMode::Symmetric,
        };
        let qt = QuantizedTensor::from_float(&[4], &[0.0, 0.5, 1.0, -1.0], &params);
        assert_eq!(qt.data[0], 0);
        assert_eq!(qt.data[1], 1);
        assert_eq!(qt.data[2], 2);
        assert_eq!(qt.data[3], -2);
    }

    #[test]
    fn test_quantized_tensor_roundtrip() {
        let params = QuantParams {
            scale: vec![0.1],
            zero_point: vec![0],
            precision: Precision::Int8,
            mode: QuantMode::Symmetric,
        };
        let original = vec![0.3, -0.7, 1.2, 0.0];
        let qt = QuantizedTensor::from_float(&[4], &original, &params);
        let deq = qt.to_float();
        for (o, d) in original.iter().zip(&deq) {
            assert!((o - d).abs() < 0.1);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let params = QuantParams {
            scale: vec![1.0],
            zero_point: vec![0],
            precision: Precision::Int8,
            mode: QuantMode::Symmetric,
        };
        let qt = QuantizedTensor::from_float(&[2], &[1.0, 2.0], &params);
        assert!((qt.compression_ratio() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_int4_compression() {
        let params = QuantParams {
            scale: vec![1.0],
            zero_point: vec![0],
            precision: Precision::Int4,
            mode: QuantMode::Symmetric,
        };
        let qt = QuantizedTensor::from_float(&[2], &[1.0, 2.0], &params);
        assert!((qt.compression_ratio() - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_quantize() {
        let values = vec![0.0, 1.0, -1.0, 0.5, -0.5];
        let qt = dynamic_quantize(&values, Precision::Int8);
        assert_eq!(qt.numel(), 5);
        // Should be near-lossless for this range
        let deq = qt.to_float();
        for (o, d) in values.iter().zip(&deq) {
            assert!((o - d).abs() < 0.02);
        }
    }

    #[test]
    fn test_quantization_error() {
        let original = vec![0.0, 1.0, -1.0, 0.5];
        let qt = dynamic_quantize(&original, Precision::Int8);
        let err = quantization_error(&original, &qt);
        assert!(err < 0.001);
    }

    #[test]
    fn test_calibration_minmax() {
        let mut coll = CalibrationCollector::new(CalibrationMethod::MinMax, 1);
        coll.observe(0, &[0.0, 1.0, -1.0, 0.5, -0.5]);
        let params = coll.compute_params(Precision::Int8, QuantMode::Symmetric);
        assert_eq!(params.zero_point[0], 0);
        assert!(params.scale[0] > 0.0);
    }

    #[test]
    fn test_calibration_percentile() {
        let mut coll = CalibrationCollector::new(CalibrationMethod::Percentile(99.0), 1);
        let vals: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) * 0.01).collect();
        coll.observe(0, &vals);
        let params = coll.compute_params(Precision::Int8, QuantMode::Symmetric);
        assert!(params.scale[0] > 0.0);
    }

    #[test]
    fn test_calibration_entropy() {
        let mut coll = CalibrationCollector::new(CalibrationMethod::Entropy, 1);
        coll.observe(0, &[0.0, 1.0, -1.0, 2.0, -2.0]);
        let params = coll.compute_params(Precision::Int8, QuantMode::Symmetric);
        assert!(params.scale[0] > 0.0);
    }

    #[test]
    fn test_config_builder() {
        let cfg = QuantizeConfig::new()
            .with_precision(Precision::Int4)
            .with_mode(QuantMode::Asymmetric)
            .with_granularity(Granularity::PerChannel)
            .with_calibration(CalibrationMethod::Percentile(99.9))
            .with_layer_precision("fc1", Precision::Int8);
        assert_eq!(cfg.precision, Precision::Int4);
        assert_eq!(cfg.mode, QuantMode::Asymmetric);
        assert_eq!(cfg.per_layer_config.get("fc1"), Some(&Precision::Int8));
    }

    #[test]
    fn test_config_default() {
        let cfg = QuantizeConfig::default();
        assert_eq!(cfg.precision, Precision::Int8);
    }

    #[test]
    fn test_display_impls() {
        assert!(format!("{}", Precision::Int8).contains("INT8"));
        assert!(format!("{}", QuantMode::Symmetric).contains("symmetric"));

        let cfg = QuantizeConfig::new();
        assert!(format!("{cfg}").contains("QuantizeConfig"));
    }

    #[test]
    fn test_per_channel_quantize() {
        let params = QuantParams {
            scale: vec![0.1, 0.2],
            zero_point: vec![0, 0],
            precision: Precision::Int8,
            mode: QuantMode::Symmetric,
        };
        let q0 = params.quantize_value(1.0, 0); // 1.0 / 0.1 = 10
        let q1 = params.quantize_value(1.0, 1); // 1.0 / 0.2 = 5
        assert_eq!(q0, 10);
        assert_eq!(q1, 5);
    }
}
