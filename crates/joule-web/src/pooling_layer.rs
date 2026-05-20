//! Pooling layers for spatial downsampling in neural networks.
//!
//! Implements max pooling, average pooling, global pooling, and
//! adaptive pooling. Each records argmax indices (for max pool)
//! to enable gradient routing during backpropagation.

use std::fmt;

// ── Pooling Type ──────────────────────────────────────────────────

/// The type of pooling operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolType {
    Max,
    Average,
    /// L2-norm pooling (rarely used but well-defined).
    L2Norm,
}

impl fmt::Display for PoolType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Max => write!(f, "max"),
            Self::Average => write!(f, "avg"),
            Self::L2Norm => write!(f, "l2"),
        }
    }
}

// ── PoolingConfig ─────────────────────────────────────────────────

/// Configuration for a pooling layer.
#[derive(Debug, Clone)]
pub struct PoolingConfig {
    pub pool_type: PoolType,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub ceil_mode: bool,
}

impl PoolingConfig {
    pub fn new(pool_type: PoolType, kernel_size: usize) -> Self {
        Self {
            pool_type,
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: kernel_size,
            stride_w: kernel_size,
            padding_h: 0,
            padding_w: 0,
            ceil_mode: false,
        }
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride_h = stride;
        self.stride_w = stride;
        self
    }

    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding_h = padding;
        self.padding_w = padding;
        self
    }

    pub fn with_ceil_mode(mut self) -> Self {
        self.ceil_mode = true;
        self
    }

    /// Compute output spatial dimensions.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let h_padded = input_h + 2 * self.padding_h;
        let w_padded = input_w + 2 * self.padding_w;

        let out_h = if self.ceil_mode {
            (h_padded - self.kernel_h + self.stride_h) / self.stride_h
        } else {
            (h_padded - self.kernel_h) / self.stride_h + 1
        };

        let out_w = if self.ceil_mode {
            (w_padded - self.kernel_w + self.stride_w) / self.stride_w
        } else {
            (w_padded - self.kernel_w) / self.stride_w + 1
        };

        (out_h, out_w)
    }
}

// ── PoolingLayer ──────────────────────────────────────────────────

/// Spatial pooling layer supporting max, average, and L2 pooling.
#[derive(Debug, Clone)]
pub struct PoolingLayer {
    pub config: PoolingConfig,
    /// Argmax indices for max pooling backward pass.
    max_indices: Vec<usize>,
    last_output_shape: (usize, usize, usize),
}

impl PoolingLayer {
    pub fn new(config: PoolingConfig) -> Self {
        Self {
            config,
            max_indices: Vec::new(),
            last_output_shape: (0, 0, 0),
        }
    }

    /// Convenience constructors.
    pub fn max_pool(kernel_size: usize) -> Self {
        Self::new(PoolingConfig::new(PoolType::Max, kernel_size))
    }

    pub fn avg_pool(kernel_size: usize) -> Self {
        Self::new(PoolingConfig::new(PoolType::Average, kernel_size))
    }

    /// Forward pass. Input is `[channels, height, width]` flattened.
    pub fn forward(
        &mut self,
        input: &[f64],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Vec<f64> {
        let (out_h, out_w) = self.config.output_size(height, width);
        let spatial_out = out_h * out_w;
        let mut output = vec![0.0; channels * spatial_out];
        self.max_indices = vec![0; channels * spatial_out];
        self.last_output_shape = (channels, out_h, out_w);

        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx = c * spatial_out + oh * out_w + ow;
                    let ih_start = oh * self.config.stride_h;
                    let iw_start = ow * self.config.stride_w;

                    match self.config.pool_type {
                        PoolType::Max => {
                            let mut max_val = f64::NEG_INFINITY;
                            let mut max_idx = 0;
                            for kh in 0..self.config.kernel_h {
                                for kw in 0..self.config.kernel_w {
                                    let ih = ih_start + kh;
                                    let iw = iw_start + kw;
                                    if ih < height + self.config.padding_h
                                        && iw < width + self.config.padding_w
                                    {
                                        let real_ih = if ih >= self.config.padding_h {
                                            ih - self.config.padding_h
                                        } else {
                                            continue;
                                        };
                                        let real_iw = if iw >= self.config.padding_w {
                                            iw - self.config.padding_w
                                        } else {
                                            continue;
                                        };
                                        if real_ih < height && real_iw < width {
                                            let idx = c * height * width + real_ih * width + real_iw;
                                            if input[idx] > max_val {
                                                max_val = input[idx];
                                                max_idx = idx;
                                            }
                                        }
                                    }
                                }
                            }
                            output[out_idx] = max_val;
                            self.max_indices[out_idx] = max_idx;
                        }
                        PoolType::Average => {
                            let mut sum = 0.0;
                            let mut count = 0;
                            for kh in 0..self.config.kernel_h {
                                for kw in 0..self.config.kernel_w {
                                    let ih = ih_start + kh;
                                    let iw = iw_start + kw;
                                    if ih >= self.config.padding_h && iw >= self.config.padding_w {
                                        let real_ih = ih - self.config.padding_h;
                                        let real_iw = iw - self.config.padding_w;
                                        if real_ih < height && real_iw < width {
                                            sum += input[c * height * width + real_ih * width + real_iw];
                                            count += 1;
                                        }
                                    }
                                }
                            }
                            output[out_idx] = if count > 0 { sum / count as f64 } else { 0.0 };
                        }
                        PoolType::L2Norm => {
                            let mut sum_sq = 0.0;
                            for kh in 0..self.config.kernel_h {
                                for kw in 0..self.config.kernel_w {
                                    let ih = ih_start + kh;
                                    let iw = iw_start + kw;
                                    if ih >= self.config.padding_h && iw >= self.config.padding_w {
                                        let real_ih = ih - self.config.padding_h;
                                        let real_iw = iw - self.config.padding_w;
                                        if real_ih < height && real_iw < width {
                                            let v = input[c * height * width + real_ih * width + real_iw];
                                            sum_sq += v * v;
                                        }
                                    }
                                }
                            }
                            output[out_idx] = sum_sq.sqrt();
                        }
                    }
                }
            }
        }

        output
    }

    /// Backward pass for max pooling — routes gradient to argmax positions.
    pub fn backward_max(
        &self,
        grad_output: &[f64],
        input_size: usize,
    ) -> Vec<f64> {
        let mut grad_input = vec![0.0; input_size];
        for (i, &idx) in self.max_indices.iter().enumerate() {
            if idx < input_size {
                grad_input[idx] += grad_output[i];
            }
        }
        grad_input
    }

    /// Retrieve the output shape from the last forward pass.
    pub fn last_output_shape(&self) -> (usize, usize, usize) {
        self.last_output_shape
    }

    /// Downsampling factor (assumes square kernel and stride = kernel).
    pub fn downsample_factor(&self) -> (usize, usize) {
        (self.config.stride_h, self.config.stride_w)
    }
}

impl fmt::Display for PoolingLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pool2d({}, {}x{}, stride=({},{}))",
            self.config.pool_type,
            self.config.kernel_h,
            self.config.kernel_w,
            self.config.stride_h,
            self.config.stride_w
        )
    }
}

// ── Global Pooling ────────────────────────────────────────────────

/// Global pooling reduces each channel to a single value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GlobalPoolType {
    Max,
    Average,
}

/// Global pooling layer — reduces spatial dims to 1x1 per channel.
#[derive(Debug, Clone)]
pub struct GlobalPoolingLayer {
    pub pool_type: GlobalPoolType,
}

impl GlobalPoolingLayer {
    pub fn new(pool_type: GlobalPoolType) -> Self {
        Self { pool_type }
    }

    /// Forward: input `[channels, height, width]`, output `[channels]`.
    pub fn forward(&self, input: &[f64], channels: usize, height: usize, width: usize) -> Vec<f64> {
        let spatial = height * width;
        assert_eq!(input.len(), channels * spatial);

        (0..channels)
            .map(|c| {
                let slice = &input[c * spatial..(c + 1) * spatial];
                match self.pool_type {
                    GlobalPoolType::Max => {
                        slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    }
                    GlobalPoolType::Average => {
                        slice.iter().sum::<f64>() / spatial as f64
                    }
                }
            })
            .collect()
    }
}

impl fmt::Display for GlobalPoolingLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pool_type {
            GlobalPoolType::Max => write!(f, "GlobalMaxPool"),
            GlobalPoolType::Average => write!(f, "GlobalAvgPool"),
        }
    }
}

// ── Adaptive Pooling ──────────────────────────────────────────────

/// Adaptive pooling that targets a specific output spatial size.
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool {
    pub target_h: usize,
    pub target_w: usize,
}

impl AdaptiveAvgPool {
    pub fn new(target_h: usize, target_w: usize) -> Self {
        Self { target_h, target_w }
    }

    /// Forward pass using floor-based bin boundaries.
    pub fn forward(
        &self,
        input: &[f64],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Vec<f64> {
        let out_spatial = self.target_h * self.target_w;
        let mut output = vec![0.0; channels * out_spatial];

        for c in 0..channels {
            for oh in 0..self.target_h {
                let h_start = (oh * height) / self.target_h;
                let h_end = ((oh + 1) * height) / self.target_h;
                for ow in 0..self.target_w {
                    let w_start = (ow * width) / self.target_w;
                    let w_end = ((ow + 1) * width) / self.target_w;

                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            sum += input[c * height * width + h * width + w];
                            count += 1;
                        }
                    }
                    let out_idx = c * out_spatial + oh * self.target_w + ow;
                    output[out_idx] = if count > 0 { sum / count as f64 } else { 0.0 };
                }
            }
        }

        output
    }
}

impl fmt::Display for AdaptiveAvgPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AdaptiveAvgPool({}x{})", self.target_h, self.target_w)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_output_size() {
        let cfg = PoolingConfig::new(PoolType::Max, 2);
        assert_eq!(cfg.output_size(4, 4), (2, 2));
    }

    #[test]
    fn test_config_output_size_stride() {
        let cfg = PoolingConfig::new(PoolType::Max, 3).with_stride(1);
        assert_eq!(cfg.output_size(5, 5), (3, 3));
    }

    #[test]
    fn test_config_ceil_mode() {
        let cfg = PoolingConfig::new(PoolType::Max, 3).with_stride(2).with_ceil_mode();
        let (h, w) = cfg.output_size(5, 5);
        assert!(h >= 2);
        assert!(w >= 2);
    }

    #[test]
    fn test_max_pool_2x2() {
        let mut pool = PoolingLayer::max_pool(2);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let out = pool.forward(&input, 1, 4, 4);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 6.0).abs() < 1e-10);
        assert!((out[1] - 8.0).abs() < 1e-10);
        assert!((out[2] - 14.0).abs() < 1e-10);
        assert!((out[3] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_avg_pool_2x2() {
        let mut pool = PoolingLayer::avg_pool(2);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let out = pool.forward(&input, 1, 4, 4);
        assert_eq!(out.len(), 4);
        // top-left: (1+2+5+6)/4 = 3.5
        assert!((out[0] - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_l2_pool_2x2() {
        let mut pool = PoolingLayer::new(PoolingConfig::new(PoolType::L2Norm, 2));
        let input = vec![3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = pool.forward(&input, 1, 4, 4);
        // top-left: sqrt(9 + 0 + 16 + 0) = 5.0
        assert!((out[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_channel_pool() {
        let mut pool = PoolingLayer::max_pool(2);
        // 2 channels of 2x2
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = pool.forward(&input, 2, 2, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 4.0).abs() < 1e-10);
        assert!((out[1] - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_backward_max_routes_gradient() {
        let mut pool = PoolingLayer::max_pool(2);
        let input = vec![1.0, 3.0, 2.0, 4.0];
        pool.forward(&input, 1, 2, 2);
        let grad_out = vec![1.0];
        let grad_in = pool.backward_max(&grad_out, 4);
        // max was 4.0 at index 3
        assert!((grad_in[3] - 1.0).abs() < 1e-10);
        assert!((grad_in[0]).abs() < 1e-10);
    }

    #[test]
    fn test_global_max_pool() {
        let gp = GlobalPoolingLayer::new(GlobalPoolType::Max);
        let input = vec![1.0, 5.0, 3.0, 2.0, 10.0, 7.0, 4.0, 8.0, 6.0];
        let out = gp.forward(&input, 1, 3, 3);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_avg_pool() {
        let gp = GlobalPoolingLayer::new(GlobalPoolType::Average);
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let out = gp.forward(&input, 1, 2, 2);
        assert!((out[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_pool_multi_channel() {
        let gp = GlobalPoolingLayer::new(GlobalPoolType::Average);
        let input = vec![1.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 40.0];
        let out = gp.forward(&input, 2, 2, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 4.0).abs() < 1e-10);
        assert!((out[1] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_avg_pool() {
        let aap = AdaptiveAvgPool::new(1, 1);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let out = aap.forward(&input, 1, 3, 3);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_avg_pool_2x2() {
        let aap = AdaptiveAvgPool::new(2, 2);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let out = aap.forward(&input, 1, 4, 4);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_pool_display() {
        let pool = PoolingLayer::max_pool(2);
        let s = format!("{}", pool);
        assert!(s.contains("max"));
        assert!(s.contains("2x2"));
    }

    #[test]
    fn test_global_pool_display() {
        let gp = GlobalPoolingLayer::new(GlobalPoolType::Max);
        assert_eq!(format!("{}", gp), "GlobalMaxPool");
    }

    #[test]
    fn test_adaptive_pool_display() {
        let aap = AdaptiveAvgPool::new(7, 7);
        assert_eq!(format!("{}", aap), "AdaptiveAvgPool(7x7)");
    }

    #[test]
    fn test_pool_type_display() {
        assert_eq!(format!("{}", PoolType::Max), "max");
        assert_eq!(format!("{}", PoolType::Average), "avg");
        assert_eq!(format!("{}", PoolType::L2Norm), "l2");
    }

    #[test]
    fn test_downsample_factor() {
        let pool = PoolingLayer::max_pool(2);
        assert_eq!(pool.downsample_factor(), (2, 2));
    }

    #[test]
    fn test_last_output_shape() {
        let mut pool = PoolingLayer::max_pool(2);
        let input = vec![0.0; 3 * 8 * 8];
        pool.forward(&input, 3, 8, 8);
        assert_eq!(pool.last_output_shape(), (3, 4, 4));
    }
}
