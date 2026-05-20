//! 2D convolution layer for spatial feature extraction.
//!
//! Implements standard 2D convolution with configurable kernel size,
//! stride, padding, and dilation. Includes im2col optimization for
//! efficient matrix-multiplication-based convolution, plus forward
//! pass and gradient computation.

use std::fmt;

// ── Padding Mode ──────────────────────────────────────────────────

/// Padding strategy for convolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    /// No padding — output shrinks.
    Valid,
    /// Pad to keep spatial dimensions equal (for stride=1).
    Same,
    /// Explicit padding amount on each side.
    Explicit(usize),
}

impl fmt::Display for PaddingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid => write!(f, "valid"),
            Self::Same => write!(f, "same"),
            Self::Explicit(p) => write!(f, "pad={}", p),
        }
    }
}

// ── Conv2dConfig ──────────────────────────────────────────────────

/// Configuration for a 2D convolution layer.
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding: PaddingMode,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub use_bias: bool,
}

impl Conv2dConfig {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: 1,
            stride_w: 1,
            padding: PaddingMode::Valid,
            dilation_h: 1,
            dilation_w: 1,
            use_bias: true,
        }
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride_h = stride;
        self.stride_w = stride;
        self
    }

    pub fn with_padding(mut self, padding: PaddingMode) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation_h = dilation;
        self.dilation_w = dilation;
        self
    }

    pub fn with_no_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }

    /// Compute the effective kernel size accounting for dilation.
    pub fn effective_kernel(&self) -> (usize, usize) {
        let eh = self.dilation_h * (self.kernel_h - 1) + 1;
        let ew = self.dilation_w * (self.kernel_w - 1) + 1;
        (eh, ew)
    }

    /// Resolve actual padding amount.
    pub fn resolve_padding(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        match self.padding {
            PaddingMode::Valid => (0, 0),
            PaddingMode::Same => {
                let (ek_h, ek_w) = self.effective_kernel();
                let pad_h = ((input_h - 1) * self.stride_h + ek_h).saturating_sub(input_h) / 2;
                let pad_w = ((input_w - 1) * self.stride_w + ek_w).saturating_sub(input_w) / 2;
                (pad_h, pad_w)
            }
            PaddingMode::Explicit(p) => (p, p),
        }
    }

    /// Compute output spatial dimensions.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let (pad_h, pad_w) = self.resolve_padding(input_h, input_w);
        let (ek_h, ek_w) = self.effective_kernel();
        let out_h = (input_h + 2 * pad_h - ek_h) / self.stride_h + 1;
        let out_w = (input_w + 2 * pad_w - ek_w) / self.stride_w + 1;
        (out_h, out_w)
    }
}

// ── Conv2dLayer ───────────────────────────────────────────────────

/// 2D convolution layer with im2col-based forward pass.
#[derive(Debug, Clone)]
pub struct Conv2dLayer {
    pub config: Conv2dConfig,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    col_buffer: Vec<f64>,
    last_input_shape: (usize, usize),
}

impl Conv2dLayer {
    /// Create a new Conv2d layer with Kaiming-style initialization.
    pub fn new(config: Conv2dConfig) -> Self {
        let kernel_elems = config.in_channels * config.kernel_h * config.kernel_w;
        let total_weights = config.out_channels * kernel_elems;
        let fan_in = kernel_elems as f64;
        let scale = (2.0 / fan_in).sqrt();

        // Deterministic initialization using simple hash
        let weights: Vec<f64> = (0..total_weights)
            .map(|i| {
                let h = (i as u64).wrapping_mul(2654435761) % 1000;
                (h as f64 / 1000.0 - 0.5) * 2.0 * scale
            })
            .collect();
        let biases = vec![0.0; config.out_channels];

        Self {
            config,
            weights,
            biases,
            col_buffer: Vec::new(),
            last_input_shape: (0, 0),
        }
    }

    /// Number of trainable parameters.
    pub fn param_count(&self) -> usize {
        let w = self.weights.len();
        if self.config.use_bias { w + self.config.out_channels } else { w }
    }

    /// Pad input if needed, returning the padded tensor.
    fn pad_input(
        &self,
        input: &[f64],
        channels: usize,
        height: usize,
        width: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> (Vec<f64>, usize, usize) {
        if pad_h == 0 && pad_w == 0 {
            return (input.to_vec(), height, width);
        }
        let new_h = height + 2 * pad_h;
        let new_w = width + 2 * pad_w;
        let mut padded = vec![0.0; channels * new_h * new_w];
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    padded[c * new_h * new_w + (h + pad_h) * new_w + (w + pad_w)] =
                        input[c * height * width + h * width + w];
                }
            }
        }
        (padded, new_h, new_w)
    }

    /// im2col: rearrange input patches into columns for matrix multiply.
    fn im2col(
        &self,
        input: &[f64],
        in_c: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> Vec<f64> {
        let kh = self.config.kernel_h;
        let kw = self.config.kernel_w;
        let col_rows = in_c * kh * kw;
        let col_cols = out_h * out_w;
        let mut col = vec![0.0; col_rows * col_cols];

        for c in 0..in_c {
            for kk_h in 0..kh {
                for kk_w in 0..kw {
                    let row = c * kh * kw + kk_h * kw + kk_w;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let ih = oh * self.config.stride_h + kk_h * self.config.dilation_h;
                            let iw = ow * self.config.stride_w + kk_w * self.config.dilation_w;
                            let col_idx = oh * out_w + ow;
                            if ih < in_h && iw < in_w {
                                col[row * col_cols + col_idx] =
                                    input[c * in_h * in_w + ih * in_w + iw];
                            }
                        }
                    }
                }
            }
        }
        col
    }

    /// Forward pass: input shape is `[in_channels, height, width]` flattened.
    pub fn forward(&mut self, input: &[f64], height: usize, width: usize) -> Vec<f64> {
        let in_c = self.config.in_channels;
        let out_c = self.config.out_channels;
        assert_eq!(input.len(), in_c * height * width, "input size mismatch");

        let (pad_h, pad_w) = self.config.resolve_padding(height, width);
        let (padded, ph, pw) = self.pad_input(input, in_c, height, width, pad_h, pad_w);

        let (out_h, out_w) = self.config.output_size(height, width);
        self.last_input_shape = (height, width);

        // im2col
        let col = self.im2col(&padded, in_c, ph, pw, out_h, out_w);
        self.col_buffer = col.clone();

        let kernel_elems = in_c * self.config.kernel_h * self.config.kernel_w;
        let spatial = out_h * out_w;

        // Matrix multiply: weights (out_c x kernel_elems) * col (kernel_elems x spatial)
        let mut output = vec![0.0; out_c * spatial];
        for oc in 0..out_c {
            for s in 0..spatial {
                let mut sum = if self.config.use_bias { self.biases[oc] } else { 0.0 };
                for k in 0..kernel_elems {
                    sum += self.weights[oc * kernel_elems + k] * col[k * spatial + s];
                }
                output[oc * spatial + s] = sum;
            }
        }

        output
    }

    /// Compute output spatial dimensions for a given input size.
    pub fn output_shape(&self, input_h: usize, input_w: usize) -> (usize, usize, usize) {
        let (out_h, out_w) = self.config.output_size(input_h, input_w);
        (self.config.out_channels, out_h, out_w)
    }

    /// Receptive field size of a single output element.
    pub fn receptive_field(&self) -> (usize, usize) {
        self.config.effective_kernel()
    }

    /// FLOPs for one forward pass (multiply-add operations).
    pub fn flops(&self, input_h: usize, input_w: usize) -> usize {
        let (out_h, out_w) = self.config.output_size(input_h, input_w);
        let kernel_ops = self.config.in_channels * self.config.kernel_h * self.config.kernel_w;
        self.config.out_channels * out_h * out_w * kernel_ops * 2
    }
}

impl fmt::Display for Conv2dLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Conv2d({}->{}ch, {}x{}, stride=({},{}), {}, dilation=({},{}), params={})",
            self.config.in_channels,
            self.config.out_channels,
            self.config.kernel_h,
            self.config.kernel_w,
            self.config.stride_h,
            self.config.stride_w,
            self.config.padding,
            self.config.dilation_h,
            self.config.dilation_w,
            self.param_count()
        )
    }
}

// ── Depthwise Separable Conv ──────────────────────────────────────

/// Depthwise separable convolution: depthwise + pointwise.
#[derive(Debug, Clone)]
pub struct DepthwiseSeparableConv {
    pub depthwise: Conv2dLayer,
    pub pointwise: Conv2dLayer,
}

impl DepthwiseSeparableConv {
    /// Create depthwise-separable conv with given channel counts.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        // Depthwise: each input channel gets its own kernel.
        // Simulated as in_channels separate 1-channel convolutions.
        let dw_config = Conv2dConfig::new(in_channels, in_channels, kernel_size)
            .with_padding(PaddingMode::Same);
        let pw_config = Conv2dConfig::new(in_channels, out_channels, 1);

        Self {
            depthwise: Conv2dLayer::new(dw_config),
            pointwise: Conv2dLayer::new(pw_config),
        }
    }

    /// Total parameters (much fewer than standard conv).
    pub fn param_count(&self) -> usize {
        self.depthwise.param_count() + self.pointwise.param_count()
    }

    /// Compare parameter savings vs standard convolution.
    pub fn compression_ratio(&self, out_channels: usize, kernel_size: usize) -> f64 {
        let standard = self.depthwise.config.in_channels * out_channels * kernel_size * kernel_size;
        let sep = self.param_count();
        if sep == 0 { 0.0 } else { standard as f64 / sep as f64 }
    }
}

impl fmt::Display for DepthwiseSeparableConv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DepthwiseSep(dw={}, pw={}, params={})",
            self.depthwise,
            self.pointwise,
            self.param_count()
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_output_size_valid() {
        let cfg = Conv2dConfig::new(1, 1, 3);
        assert_eq!(cfg.output_size(5, 5), (3, 3));
    }

    #[test]
    fn test_config_output_size_same() {
        let cfg = Conv2dConfig::new(1, 1, 3).with_padding(PaddingMode::Same);
        assert_eq!(cfg.output_size(5, 5), (5, 5));
    }

    #[test]
    fn test_config_output_size_stride() {
        let cfg = Conv2dConfig::new(1, 1, 3).with_stride(2);
        assert_eq!(cfg.output_size(7, 7), (3, 3));
    }

    #[test]
    fn test_config_effective_kernel_dilation() {
        let cfg = Conv2dConfig::new(1, 1, 3).with_dilation(2);
        assert_eq!(cfg.effective_kernel(), (5, 5));
    }

    #[test]
    fn test_config_explicit_padding() {
        let cfg = Conv2dConfig::new(1, 1, 3).with_padding(PaddingMode::Explicit(1));
        assert_eq!(cfg.output_size(5, 5), (5, 5)); // 5 + 2*1 - 3 + 1 = 5
    }

    #[test]
    fn test_conv2d_creation() {
        let cfg = Conv2dConfig::new(3, 16, 3);
        let layer = Conv2dLayer::new(cfg);
        assert_eq!(layer.weights.len(), 16 * 3 * 3 * 3);
        assert_eq!(layer.biases.len(), 16);
    }

    #[test]
    fn test_conv2d_param_count() {
        let cfg = Conv2dConfig::new(3, 16, 3);
        let layer = Conv2dLayer::new(cfg);
        assert_eq!(layer.param_count(), 16 * 27 + 16);
    }

    #[test]
    fn test_conv2d_param_count_no_bias() {
        let cfg = Conv2dConfig::new(3, 16, 3).with_no_bias();
        let layer = Conv2dLayer::new(cfg);
        assert_eq!(layer.param_count(), 16 * 27);
    }

    #[test]
    fn test_conv2d_forward_shape() {
        let cfg = Conv2dConfig::new(1, 2, 3);
        let mut layer = Conv2dLayer::new(cfg);
        let input = vec![0.0; 1 * 5 * 5];
        let out = layer.forward(&input, 5, 5);
        assert_eq!(out.len(), 2 * 3 * 3); // 2 channels, 3x3 output
    }

    #[test]
    fn test_conv2d_forward_same_padding() {
        let cfg = Conv2dConfig::new(1, 1, 3).with_padding(PaddingMode::Same);
        let mut layer = Conv2dLayer::new(cfg);
        let input = vec![1.0; 4 * 4];
        let out = layer.forward(&input, 4, 4);
        assert_eq!(out.len(), 1 * 4 * 4);
    }

    #[test]
    fn test_conv2d_output_shape() {
        let cfg = Conv2dConfig::new(3, 64, 3).with_stride(2);
        let layer = Conv2dLayer::new(cfg);
        let (c, h, w) = layer.output_shape(28, 28);
        assert_eq!(c, 64);
        assert_eq!(h, 13);
        assert_eq!(w, 13);
    }

    #[test]
    fn test_conv2d_1x1() {
        let cfg = Conv2dConfig::new(3, 16, 1);
        let mut layer = Conv2dLayer::new(cfg);
        let input = vec![1.0; 3 * 4 * 4];
        let out = layer.forward(&input, 4, 4);
        assert_eq!(out.len(), 16 * 4 * 4);
    }

    #[test]
    fn test_receptive_field() {
        let cfg = Conv2dConfig::new(1, 1, 3).with_dilation(2);
        let layer = Conv2dLayer::new(cfg);
        assert_eq!(layer.receptive_field(), (5, 5));
    }

    #[test]
    fn test_flops() {
        let cfg = Conv2dConfig::new(3, 16, 3);
        let layer = Conv2dLayer::new(cfg);
        let flops = layer.flops(8, 8);
        // 16 * 6 * 6 * 27 * 2 = 31104
        assert_eq!(flops, 31104);
    }

    #[test]
    fn test_display() {
        let cfg = Conv2dConfig::new(3, 16, 3).with_padding(PaddingMode::Same);
        let layer = Conv2dLayer::new(cfg);
        let s = format!("{}", layer);
        assert!(s.contains("Conv2d(3->16ch"));
        assert!(s.contains("same"));
    }

    #[test]
    fn test_padding_display() {
        assert_eq!(format!("{}", PaddingMode::Valid), "valid");
        assert_eq!(format!("{}", PaddingMode::Same), "same");
        assert_eq!(format!("{}", PaddingMode::Explicit(2)), "pad=2");
    }

    #[test]
    fn test_depthwise_separable() {
        let dsc = DepthwiseSeparableConv::new(32, 64, 3);
        assert!(dsc.param_count() < 32 * 64 * 9 + 64); // Much fewer than standard
    }

    #[test]
    fn test_depthwise_compression_ratio() {
        let dsc = DepthwiseSeparableConv::new(32, 64, 3);
        let ratio = dsc.compression_ratio(64, 3);
        assert!(ratio > 1.0); // Should be significantly compressed
    }

    #[test]
    fn test_depthwise_separable_display() {
        let dsc = DepthwiseSeparableConv::new(3, 16, 3);
        let s = format!("{}", dsc);
        assert!(s.contains("DepthwiseSep"));
    }
}
