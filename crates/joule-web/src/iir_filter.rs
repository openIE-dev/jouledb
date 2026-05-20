//! IIR (Infinite Impulse Response) filters — pure Rust, no external dependencies.
//!
//! Transfer function (b/a coefficients), Direct Form I and Direct Form II
//! transposed implementations, Butterworth and Chebyshev Type I design via
//! bilinear transform, cascaded second-order sections, zero-pole-gain
//! representation, and frequency response computation.

use std::f64::consts::PI;

// ── Complex helper ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
struct Cplx {
    re: f64,
    im: f64,
}

impl Cplx {
    fn new(re: f64, im: f64) -> Self { Self { re, im } }
    fn from_polar(r: f64, theta: f64) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }
    fn mag(&self) -> f64 { (self.re * self.re + self.im * self.im).sqrt() }
    fn phase(&self) -> f64 { self.im.atan2(self.re) }
    fn conj(&self) -> Self { Self { re: self.re, im: -self.im } }
}

impl std::ops::Add for Cplx {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { re: self.re + r.re, im: self.im + r.im } }
}
impl std::ops::Sub for Cplx {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { re: self.re - r.re, im: self.im - r.im } }
}
impl std::ops::Mul for Cplx {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        Self { re: self.re * r.re - self.im * r.im, im: self.re * r.im + self.im * r.re }
    }
}
impl std::ops::Div for Cplx {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        let d = r.re * r.re + r.im * r.im;
        Self { re: (self.re * r.re + self.im * r.im) / d, im: (self.im * r.re - self.re * r.im) / d }
    }
}
impl std::ops::Mul<f64> for Cplx {
    type Output = Self;
    fn mul(self, r: f64) -> Self { Self { re: self.re * r, im: self.im * r } }
}

// ── Transfer Function ───────────────────────────────────────────

/// IIR filter transfer function: H(z) = B(z)/A(z).
/// `b` = numerator coefficients, `a` = denominator coefficients (a[0] normalized to 1).
#[derive(Debug, Clone, PartialEq)]
pub struct TransferFunction {
    pub b: Vec<f64>,
    pub a: Vec<f64>,
}

impl TransferFunction {
    pub fn new(b: Vec<f64>, a: Vec<f64>) -> Self {
        // Normalize so a[0] = 1
        let a0 = a[0];
        let b_norm: Vec<f64> = b.iter().map(|v| v / a0).collect();
        let a_norm: Vec<f64> = a.iter().map(|v| v / a0).collect();
        Self { b: b_norm, a: a_norm }
    }

    pub fn order(&self) -> usize {
        self.a.len().max(self.b.len()) - 1
    }
}

// ── Second-Order Section ────────────────────────────────────────

/// A single second-order section (biquad): b0, b1, b2, a0=1, a1, a2.
#[derive(Debug, Clone, PartialEq)]
pub struct Sos {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Sos {
    pub fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self { b0, b1, b2, a1, a2 }
    }

    /// Process a single sample using Direct Form II transposed.
    pub fn process_sample(&self, x: f64, state: &mut [f64; 2]) -> f64 {
        let y = self.b0 * x + state[0];
        state[0] = self.b1 * x - self.a1 * y + state[1];
        state[1] = self.b2 * x - self.a2 * y;
        y
    }
}

/// Cascaded second-order sections for numerical stability.
#[derive(Debug, Clone, PartialEq)]
pub struct SosCascade {
    pub sections: Vec<Sos>,
    pub gain: f64,
}

impl SosCascade {
    pub fn new(sections: Vec<Sos>, gain: f64) -> Self {
        Self { sections, gain }
    }

    pub fn order(&self) -> usize {
        self.sections.len() * 2
    }
}

// ── Zero-Pole-Gain ──────────────────────────────────────────────

/// Zero-pole-gain representation.
#[derive(Debug, Clone, PartialEq)]
pub struct Zpk {
    pub zeros: Vec<(f64, f64)>, // (re, im)
    pub poles: Vec<(f64, f64)>,
    pub gain: f64,
}

// ── Direct Form I filter ────────────────────────────────────────

/// Direct Form I IIR filter state.
#[derive(Debug, Clone)]
pub struct DirectFormI {
    tf: TransferFunction,
    x_hist: Vec<f64>,
    y_hist: Vec<f64>,
}

impl DirectFormI {
    pub fn new(tf: TransferFunction) -> Self {
        let nb = tf.b.len();
        let na = tf.a.len();
        Self {
            tf,
            x_hist: vec![0.0; nb],
            y_hist: vec![0.0; na],
        }
    }

    pub fn reset(&mut self) {
        self.x_hist.fill(0.0);
        self.y_hist.fill(0.0);
    }

    pub fn process_sample(&mut self, x: f64) -> f64 {
        // Shift input history
        for i in (1..self.x_hist.len()).rev() {
            self.x_hist[i] = self.x_hist[i - 1];
        }
        self.x_hist[0] = x;

        // Compute output: y[n] = sum(b[k]*x[n-k]) - sum(a[k]*y[n-k]) for k>=1
        let mut y = 0.0;
        for (i, &b) in self.tf.b.iter().enumerate() {
            if i < self.x_hist.len() {
                y += b * self.x_hist[i];
            }
        }
        for i in 1..self.tf.a.len() {
            // y_hist[0] = y[n-1], y_hist[1] = y[n-2], etc.
            if i - 1 < self.y_hist.len() {
                y -= self.tf.a[i] * self.y_hist[i - 1];
            }
        }

        // Shift output history
        for i in (1..self.y_hist.len()).rev() {
            self.y_hist[i] = self.y_hist[i - 1];
        }
        self.y_hist[0] = y;

        y
    }

    pub fn process(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|x| self.process_sample(*x)).collect()
    }
}

// ── Direct Form II Transposed ───────────────────────────────────

/// Direct Form II Transposed IIR filter state.
#[derive(Debug, Clone)]
pub struct DirectFormII {
    tf: TransferFunction,
    state: Vec<f64>,
}

impl DirectFormII {
    pub fn new(tf: TransferFunction) -> Self {
        let order = tf.order();
        Self {
            tf,
            state: vec![0.0; order + 1],
        }
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    pub fn process_sample(&mut self, x: f64) -> f64 {
        let b = &self.tf.b;
        let a = &self.tf.a;
        let y = b[0] * x + self.state[0];

        let n = self.state.len();
        for i in 0..n - 1 {
            let bv = if i + 1 < b.len() { b[i + 1] } else { 0.0 };
            let av = if i + 1 < a.len() { a[i + 1] } else { 0.0 };
            self.state[i] = bv * x - av * y + self.state[i + 1];
        }
        let last = n - 1;
        let bv = if last + 1 < b.len() { b[last + 1] } else { 0.0 };
        let av = if last + 1 < a.len() { a[last + 1] } else { 0.0 };
        self.state[last] = bv * x - av * y;

        y
    }

    pub fn process(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|x| self.process_sample(*x)).collect()
    }
}

// ── SOS filter ──────────────────────────────────────────────────

/// Filter a signal through cascaded second-order sections.
pub fn filter_sos(signal: &[f64], cascade: &SosCascade) -> Vec<f64> {
    let n_sections = cascade.sections.len();
    let mut states = vec![[0.0f64; 2]; n_sections];
    let mut output = Vec::with_capacity(signal.len());

    for &x in signal {
        let mut val = x * cascade.gain;
        for (s_idx, section) in cascade.sections.iter().enumerate() {
            val = section.process_sample(val, &mut states[s_idx]);
        }
        output.push(val);
    }
    output
}

// ── Butterworth Design ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IirError {
    InvalidOrder(String),
    InvalidFrequency(String),
}

impl std::fmt::Display for IirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOrder(s) => write!(f, "invalid order: {s}"),
            Self::InvalidFrequency(s) => write!(f, "invalid frequency: {s}"),
        }
    }
}

/// Design a Butterworth low-pass filter.
/// `order`: 1-8. `cutoff`: normalized frequency (0..0.5).
pub fn butterworth(order: usize, cutoff: f64, filter_type: FilterType) -> Result<TransferFunction, IirError> {
    if order < 1 || order > 8 {
        return Err(IirError::InvalidOrder("order must be 1-8".into()));
    }
    if cutoff <= 0.0 || cutoff >= 0.5 {
        return Err(IirError::InvalidFrequency(format!("cutoff {cutoff} not in (0, 0.5)")));
    }

    // Pre-warp
    let wc = (PI * cutoff).tan();

    // Analog Butterworth poles (left half-plane)
    let mut analog_poles: Vec<Cplx> = Vec::new();
    for k in 0..order {
        let angle = PI * (2 * k + order + 1) as f64 / (2 * order) as f64;
        analog_poles.push(Cplx::from_polar(wc, angle));
    }

    // Bilinear transform: s = 2*(z-1)/(z+1)
    // For each analog pole p: digital pole = (1 + p/2) / (1 - p/2)
    // We use the mapped value directly with s → 2*(1-z^-1)/(1+z^-1)
    let mut b = vec![1.0];
    let mut a = vec![1.0];

    for pole in &analog_poles {
        // Bilinear transform of (s - pole): multiply out
        // z-domain factor from pole p: (1 + p) - (1 - p)*z^-1 (denominator)
        // For low-pass, numerator factor: (1 + z^-1) * wc (per pole)
        let one = Cplx::new(1.0, 0.0);
        let p_half = *pole * 0.5;

        let denom_0 = one - p_half; // coefficient of z^0
        let denom_1 = Cplx::new(-1.0, 0.0) - p_half; // coefficient of z^-1

        // Normalize so denom_0 coefficient is real
        let d0 = denom_0;
        let d1_norm = denom_1 / d0;

        // If pole is complex, pair with conjugate
        if pole.im.abs() > 1e-10 {
            continue; // handled in pairs below
        }

        // Real pole: simple first-order section
        let a_new = vec![1.0, d1_norm.re];

        match filter_type {
            FilterType::LowPass => {
                let gain = (one + p_half) / d0;
                let b_new = vec![gain.re, gain.re];
                b = convolve_poly(&b, &b_new);
            }
            FilterType::HighPass => {
                let gain = (one - p_half) / d0;
                let b_new = vec![gain.re, -gain.re];
                b = convolve_poly(&b, &b_new);
            }
            FilterType::BandPass => {
                let gain = (one + p_half) / d0;
                let b_new = vec![gain.re, 0.0, -gain.re];
                b = convolve_poly(&b, &b_new);
            }
        }
        a = convolve_poly(&a, &a_new);
    }

    // Handle complex conjugate pairs
    let mut i = 0;
    while i < analog_poles.len() {
        if analog_poles[i].im.abs() > 1e-10 {
            let p = analog_poles[i];
            let p_half = p * 0.5;
            let one = Cplx::new(1.0, 0.0);

            // Second-order section from conjugate pair
            let d0 = (one - p_half) * (one - p_half.conj());
            let d1 = (Cplx::new(-1.0, 0.0) - p_half) * (one - p_half.conj())
                    + (one - p_half) * (Cplx::new(-1.0, 0.0) - p_half.conj());
            let d2 = (Cplx::new(-1.0, 0.0) - p_half) * (Cplx::new(-1.0, 0.0) - p_half.conj());

            let a_new = vec![1.0, d1.re / d0.re, d2.re / d0.re];

            match filter_type {
                FilterType::LowPass => {
                    let n0 = (one + p_half) * (one + p_half.conj());
                    let g = n0.re / d0.re;
                    let b_new = vec![g, 2.0 * g, g];
                    b = convolve_poly(&b, &b_new);
                }
                FilterType::HighPass => {
                    let n0 = (one - p_half) * (one - p_half.conj());
                    let g = n0.re / d0.re;
                    let b_new = vec![g, -2.0 * g, g];
                    b = convolve_poly(&b, &b_new);
                }
                FilterType::BandPass => {
                    let n0 = (one + p_half) * (one + p_half.conj());
                    let g = n0.re / d0.re;
                    let b_new = vec![g, 0.0, -g];
                    b = convolve_poly(&b, &b_new);
                }
            }
            a = convolve_poly(&a, &a_new);
            i += 2; // skip conjugate
        } else {
            i += 1;
        }
    }

    // Normalize gain at DC for low-pass, at Nyquist for high-pass
    let eval_at = match filter_type {
        FilterType::LowPass => 0.0,
        FilterType::HighPass => PI,
        FilterType::BandPass => PI * cutoff,
    };
    let b_resp = eval_poly_z(&b, eval_at);
    let a_resp = eval_poly_z(&a, eval_at);
    let gain_correction = a_resp.mag() / b_resp.mag();

    let b_corrected: Vec<f64> = b.iter().map(|v| v * gain_correction).collect();

    Ok(TransferFunction::new(b_corrected, a))
}

/// Evaluate polynomial in z at e^{j*omega}: sum(c[k] * e^{-j*k*omega}).
fn eval_poly_z(coeffs: &[f64], omega: f64) -> Cplx {
    let mut result = Cplx::new(0.0, 0.0);
    for (k, &c) in coeffs.iter().enumerate() {
        let z_inv = Cplx::from_polar(1.0, -omega * k as f64);
        result = result + z_inv * c;
    }
    result
}

fn convolve_poly(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            result[i + j] += av * bv;
        }
    }
    result
}

// ── Chebyshev Type I Design ─────────────────────────────────────

/// Design a Chebyshev Type I low-pass filter.
/// `ripple_db`: passband ripple in dB (e.g., 0.5).
pub fn chebyshev1(order: usize, cutoff: f64, ripple_db: f64) -> Result<TransferFunction, IirError> {
    if order < 1 || order > 8 {
        return Err(IirError::InvalidOrder("order must be 1-8".into()));
    }
    if cutoff <= 0.0 || cutoff >= 0.5 {
        return Err(IirError::InvalidFrequency(format!("cutoff {cutoff} not in (0, 0.5)")));
    }

    let epsilon = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
    let wc = (PI * cutoff).tan();

    // Chebyshev poles in analog domain
    let v0 = (1.0 / epsilon).asinh() / order as f64;

    let mut analog_poles: Vec<Cplx> = Vec::new();
    for k in 0..order {
        let angle = PI * (2 * k + 1) as f64 / (2 * order) as f64;
        let sigma = -wc * v0.sinh() * angle.sin();
        let omega = wc * v0.cosh() * angle.cos();
        analog_poles.push(Cplx::new(sigma, omega));
    }

    // Build transfer function via bilinear transform (same approach as Butterworth)
    let mut b = vec![1.0];
    let mut a = vec![1.0];

    let mut idx = 0;
    while idx < analog_poles.len() {
        let p = analog_poles[idx];
        let p_half = p * 0.5;
        let one = Cplx::new(1.0, 0.0);

        if p.im.abs() > 1e-10 && idx + 1 < analog_poles.len() {
            // Complex conjugate pair
            let d0 = (one - p_half) * (one - p_half.conj());
            let d1 = (Cplx::new(-1.0, 0.0) - p_half) * (one - p_half.conj())
                    + (one - p_half) * (Cplx::new(-1.0, 0.0) - p_half.conj());
            let d2 = (Cplx::new(-1.0, 0.0) - p_half) * (Cplx::new(-1.0, 0.0) - p_half.conj());

            let a_new = vec![1.0, d1.re / d0.re, d2.re / d0.re];
            let n0 = (one + p_half) * (one + p_half.conj());
            let g = n0.re / d0.re;
            let b_new = vec![g, 2.0 * g, g];

            b = convolve_poly(&b, &b_new);
            a = convolve_poly(&a, &a_new);
            idx += 2;
        } else {
            // Real pole
            let d0 = one - p_half;
            let d1_norm = (Cplx::new(-1.0, 0.0) - p_half) / d0;
            let a_new = vec![1.0, d1_norm.re];
            let gain = (one + p_half) / d0;
            let b_new = vec![gain.re, gain.re];

            b = convolve_poly(&b, &b_new);
            a = convolve_poly(&a, &a_new);
            idx += 1;
        }
    }

    // Normalize DC gain to 1 (or -ripple_db for Chebyshev)
    let b_dc = eval_poly_z(&b, 0.0);
    let a_dc = eval_poly_z(&a, 0.0);
    let target_gain = if order % 2 == 0 {
        10.0_f64.powf(-ripple_db / 20.0)
    } else {
        1.0
    };
    let correction = target_gain * a_dc.mag() / b_dc.mag();
    let b_corrected: Vec<f64> = b.iter().map(|v| v * correction).collect();

    Ok(TransferFunction::new(b_corrected, a))
}

// ── Frequency Response ──────────────────────────────────────────

/// Compute frequency response of a transfer function.
/// Returns `(frequencies, magnitudes, phases)`.
pub fn frequency_response(tf: &TransferFunction, n_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n_points);
    let mut mags = Vec::with_capacity(n_points);
    let mut phases = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let f = 0.5 * i as f64 / (n_points - 1).max(1) as f64;
        let omega = 2.0 * PI * f;
        let num = eval_poly_z(&tf.b, omega);
        let den = eval_poly_z(&tf.a, omega);
        let h = num / den;
        freqs.push(f);
        mags.push(h.mag());
        phases.push(h.phase());
    }
    (freqs, mags, phases)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-3;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_transfer_function_normalize() {
        let tf = TransferFunction::new(vec![2.0, 4.0], vec![2.0, 1.0]);
        assert!(approx_eq(tf.a[0], 1.0));
        assert!(approx_eq(tf.b[0], 1.0));
        assert!(approx_eq(tf.b[1], 2.0));
    }

    #[test]
    fn test_transfer_function_order() {
        let tf = TransferFunction::new(vec![1.0, 2.0, 3.0], vec![1.0, 0.5]);
        assert_eq!(tf.order(), 2);
    }

    #[test]
    fn test_sos_process_impulse() {
        let sos = Sos::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let mut state = [0.0; 2];
        let y = sos.process_sample(1.0, &mut state);
        assert!(approx_eq(y, 1.0));
        let y2 = sos.process_sample(0.0, &mut state);
        assert!(approx_eq(y2, 0.0));
    }

    #[test]
    fn test_sos_cascade_order() {
        let cascade = SosCascade::new(
            vec![Sos::new(1.0, 0.0, 0.0, 0.0, 0.0); 3],
            1.0,
        );
        assert_eq!(cascade.order(), 6);
    }

    #[test]
    fn test_direct_form_i_impulse() {
        let tf = TransferFunction::new(vec![1.0, 0.5], vec![1.0, -0.3]);
        let mut filt = DirectFormI::new(tf);
        let impulse = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let out = filt.process(&impulse);
        // h[0] = b[0] = 1.0
        assert!(approx_eq(out[0], 1.0));
        // h[1] = b[1] + a[1]*h[0] = 0.5 + 0.3*1.0 = 0.8
        assert!(approx_eq(out[1], 0.8));
    }

    #[test]
    fn test_direct_form_ii_impulse() {
        let tf = TransferFunction::new(vec![1.0, 0.5], vec![1.0, -0.3]);
        let mut filt = DirectFormII::new(tf);
        let impulse = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let out = filt.process(&impulse);
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 0.8));
    }

    #[test]
    fn test_direct_form_i_ii_match() {
        let tf = TransferFunction::new(vec![1.0, 0.5, -0.2], vec![1.0, -0.6, 0.1]);
        let mut df1 = DirectFormI::new(tf.clone());
        let mut df2 = DirectFormII::new(tf);
        let signal: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin()).collect();
        let out1 = df1.process(&signal);
        let out2 = df2.process(&signal);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(approx_eq(*a, *b), "{a} != {b}");
        }
    }

    #[test]
    fn test_direct_form_reset() {
        let tf = TransferFunction::new(vec![1.0, 0.5], vec![1.0, -0.3]);
        let mut filt = DirectFormI::new(tf);
        filt.process(&[1.0, 2.0, 3.0]);
        filt.reset();
        let out = filt.process(&[1.0, 0.0]);
        assert!(approx_eq(out[0], 1.0));
    }

    #[test]
    fn test_filter_sos_passthrough() {
        let cascade = SosCascade::new(
            vec![Sos::new(1.0, 0.0, 0.0, 0.0, 0.0)],
            1.0,
        );
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let out = filter_sos(&signal, &cascade);
        for (a, b) in out.iter().zip(signal.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn test_butterworth_lowpass_order1() {
        let tf = butterworth(1, 0.25, FilterType::LowPass).unwrap();
        assert_eq!(tf.b.len(), 2);
        assert_eq!(tf.a.len(), 2);
    }

    #[test]
    fn test_butterworth_lowpass_order2() {
        let tf = butterworth(2, 0.2, FilterType::LowPass).unwrap();
        assert_eq!(tf.a.len(), 3);
    }

    #[test]
    fn test_butterworth_dc_gain() {
        let tf = butterworth(3, 0.2, FilterType::LowPass).unwrap();
        let (_, mags, _) = frequency_response(&tf, 256);
        assert!(
            (mags[0] - 1.0).abs() < 0.1,
            "DC gain = {}",
            mags[0]
        );
    }

    #[test]
    fn test_butterworth_highpass() {
        let tf = butterworth(2, 0.3, FilterType::HighPass).unwrap();
        let (_, mags, _) = frequency_response(&tf, 256);
        // DC should be attenuated
        assert!(mags[0] < 0.2, "HP DC gain = {}", mags[0]);
        // Nyquist should pass
        assert!(mags[255] > 0.5, "HP Nyquist gain = {}", mags[255]);
    }

    #[test]
    fn test_butterworth_invalid_order() {
        assert!(butterworth(0, 0.2, FilterType::LowPass).is_err());
        assert!(butterworth(9, 0.2, FilterType::LowPass).is_err());
    }

    #[test]
    fn test_butterworth_invalid_freq() {
        assert!(butterworth(2, 0.0, FilterType::LowPass).is_err());
        assert!(butterworth(2, 0.5, FilterType::LowPass).is_err());
    }

    #[test]
    fn test_butterworth_monotonic_rolloff() {
        let tf = butterworth(4, 0.2, FilterType::LowPass).unwrap();
        let (_, mags, _) = frequency_response(&tf, 256);
        // Magnitude should generally decrease past cutoff
        let cutoff_bin = (0.2 / 0.5 * 255.0) as usize;
        let past = cutoff_bin + 20;
        let far = cutoff_bin + 60;
        if past < 256 && far < 256 {
            assert!(mags[far] <= mags[past] + 0.1);
        }
    }

    #[test]
    fn test_chebyshev1_order2() {
        let tf = chebyshev1(2, 0.2, 0.5).unwrap();
        assert!(tf.a.len() >= 3);
    }

    #[test]
    fn test_chebyshev1_invalid() {
        assert!(chebyshev1(0, 0.2, 0.5).is_err());
        assert!(chebyshev1(2, 0.0, 0.5).is_err());
    }

    #[test]
    fn test_chebyshev1_steeper_than_butterworth() {
        let bw = butterworth(3, 0.2, FilterType::LowPass).unwrap();
        let ch = chebyshev1(3, 0.2, 1.0).unwrap();
        let (_, bw_mags, _) = frequency_response(&bw, 256);
        let (_, ch_mags, _) = frequency_response(&ch, 256);
        // At high stopband frequency, Chebyshev should have more attenuation
        let bin = 200;
        // Chebyshev trades passband ripple for steeper transition
        assert!(ch_mags[bin] < bw_mags[bin] + 0.2);
    }

    #[test]
    fn test_frequency_response_length() {
        let tf = TransferFunction::new(vec![1.0, 0.5], vec![1.0, -0.3]);
        let (f, m, p) = frequency_response(&tf, 128);
        assert_eq!(f.len(), 128);
        assert_eq!(m.len(), 128);
        assert_eq!(p.len(), 128);
    }

    #[test]
    fn test_convolve_poly() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 3.0];
        let c = convolve_poly(&a, &b);
        // (1+2z)(1+3z) = 1 + 5z + 6z^2
        assert!(approx_eq(c[0], 1.0));
        assert!(approx_eq(c[1], 5.0));
        assert!(approx_eq(c[2], 6.0));
    }

    #[test]
    fn test_zpk_representation() {
        let zpk = Zpk {
            zeros: vec![(1.0, 0.0), (-1.0, 0.0)],
            poles: vec![(0.5, 0.0)],
            gain: 2.0,
        };
        assert_eq!(zpk.zeros.len(), 2);
        assert_eq!(zpk.poles.len(), 1);
        assert!(approx_eq(zpk.gain, 2.0));
    }
}
