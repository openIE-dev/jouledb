//! Complex number arithmetic — pure-Rust replacement for complex.js and mathjs Complex.
//!
//! Supports arithmetic, polar form, exponential, logarithm, trig, and parsing.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ── Complex ────────────────────────────────────────────────────

/// A complex number z = re + im*i.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

/// Default epsilon for floating-point comparisons.
const DEFAULT_EPS: f64 = 1e-12;

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Magnitude (absolute value) |z|.
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Squared magnitude |z|^2 (avoids sqrt).
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle (argument) in radians, in (-pi, pi].
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate: re - im*i.
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Reciprocal 1/z.
    pub fn recip(self) -> Self {
        let d = self.norm_sq();
        Self {
            re: self.re / d,
            im: -self.im / d,
        }
    }

    // ── Polar form ─────────────────────────────────────────────

    /// Create from polar form (r, theta).
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Convert to polar form (r, theta).
    pub fn to_polar(self) -> (f64, f64) {
        (self.abs(), self.arg())
    }

    // ── Powers ─────────────────────────────────────────────────

    /// Integer power z^n.
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::ONE;
        }
        if n < 0 {
            return self.recip().powi(-n);
        }
        let mut result = Self::ONE;
        let mut base = self;
        let mut exp = n as u32;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        result
    }

    /// Complex power z^w = exp(w * ln(z)).
    pub fn powc(self, w: Self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            if w.re > 0.0 {
                return Self::ZERO;
            }
            return Self::new(f64::NAN, f64::NAN);
        }
        (w * self.ln()).exp()
    }

    /// Principal square root.
    pub fn sqrt(self) -> Self {
        let r = self.abs();
        if r < 1e-30 {
            return Self::ZERO;
        }
        let t = ((r + self.re.abs()) / 2.0).sqrt();
        if self.re >= 0.0 {
            Self::new(t, self.im / (2.0 * t))
        } else {
            Self::new(
                self.im.abs() / (2.0 * t),
                if self.im >= 0.0 { t } else { -t },
            )
        }
    }

    // ── Exponential / Logarithm ────────────────────────────────

    /// e^z.
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    /// Principal natural logarithm ln(z).
    pub fn ln(self) -> Self {
        Self {
            re: self.abs().ln(),
            im: self.arg(),
        }
    }

    // ── Trigonometric ──────────────────────────────────────────

    /// Complex sine.
    pub fn sin(self) -> Self {
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    /// Complex cosine.
    pub fn cos(self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }

    // ── Equality with epsilon ──────────────────────────────────

    /// Check approximate equality within epsilon.
    pub fn approx_eq(self, other: Self, eps: f64) -> bool {
        (self.re - other.re).abs() < eps && (self.im - other.im).abs() < eps
    }

    // ── Parsing ────────────────────────────────────────────────

    /// Parse from string. Supports formats:
    /// - "a+bi", "a-bi", "a+bj", "a-bj"
    /// - "bi", "a", "-bi"
    /// - "a + bi" (with spaces)
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().replace(' ', "");
        if s.is_empty() {
            return None;
        }

        // Pure imaginary: "bi" or "i" or "-i" or "-bi"
        if s.ends_with('i') || s.ends_with('j') {
            let without_suffix = &s[..s.len() - 1];
            // Check if there's a real part (look for +/- that splits real and imaginary)
            if let Some(split_pos) = find_split(without_suffix) {
                let re_str = &s[..split_pos];
                let im_str = &s[split_pos..s.len() - 1];
                let re = re_str.parse::<f64>().ok()?;
                let im = parse_im_part(im_str)?;
                return Some(Self::new(re, im));
            }
            // Pure imaginary
            let im = if without_suffix.is_empty() || without_suffix == "+" {
                1.0
            } else if without_suffix == "-" {
                -1.0
            } else {
                without_suffix.parse::<f64>().ok()?
            };
            return Some(Self::new(0.0, im));
        }

        // Pure real
        let re = s.parse::<f64>().ok()?;
        Some(Self::new(re, 0.0))
    }
}

/// Find the position of +/- that separates real and imaginary parts.
/// Skips the first character (which might be a sign for the real part).
fn find_split(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    for i in 1..bytes.len() {
        if (bytes[i] == b'+' || bytes[i] == b'-') && bytes[i - 1] != b'e' && bytes[i - 1] != b'E'
        {
            return Some(i);
        }
    }
    None
}

fn parse_im_part(s: &str) -> Option<f64> {
    if s == "+" || s.is_empty() {
        Some(1.0)
    } else if s == "-" {
        Some(-1.0)
    } else {
        s.parse::<f64>().ok()
    }
}

// ── Operator impls ─────────────────────────────────────────────

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let d = rhs.norm_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(*other, DEFAULT_EPS)
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{}+{}i", self.re, self.im)
        } else {
            write!(f, "{}{}i", self.re, self.im)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-10;

    fn approx(a: Complex, b: Complex) -> bool {
        a.approx_eq(b, EPS)
    }

    #[test]
    fn basic_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        assert!(approx(a + b, Complex::new(4.0, 2.0)));
        assert!(approx(a - b, Complex::new(2.0, 6.0)));
        // (3+4i)(1-2i) = 3 -6i +4i -8i^2 = 11 -2i
        assert!(approx(a * b, Complex::new(11.0, -2.0)));
    }

    #[test]
    fn division() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        let q = a / b;
        assert!(approx(q * b, a));
    }

    #[test]
    fn magnitude_and_phase() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.abs() - 5.0).abs() < EPS);
        let z2 = Complex::new(-1.0, 0.0);
        assert!((z2.arg() - PI).abs() < EPS);
    }

    #[test]
    fn conjugate() {
        let z = Complex::new(3.0, 4.0);
        let c = z.conj();
        assert!(approx(c, Complex::new(3.0, -4.0)));
        let product = z * c;
        assert!((product.re - 25.0).abs() < EPS);
        assert!(product.im.abs() < EPS);
    }

    #[test]
    fn polar_round_trip() {
        let z = Complex::new(3.0, 4.0);
        let (r, theta) = z.to_polar();
        let z2 = Complex::from_polar(r, theta);
        assert!(approx(z, z2));
    }

    #[test]
    fn integer_power() {
        let z = Complex::new(1.0, 1.0);
        let z2 = z.powi(2);
        assert!(approx(z2, Complex::new(0.0, 2.0)));
        let z3 = z.powi(3);
        assert!(approx(z3, Complex::new(-2.0, 2.0)));
    }

    #[test]
    fn negative_power() {
        let z = Complex::new(2.0, 0.0);
        let inv = z.powi(-1);
        assert!(approx(inv, Complex::new(0.5, 0.0)));
    }

    #[test]
    fn sqrt_positive_real() {
        let z = Complex::new(4.0, 0.0);
        let s = z.sqrt();
        assert!(approx(s, Complex::new(2.0, 0.0)));
    }

    #[test]
    fn sqrt_negative_real() {
        let z = Complex::new(-1.0, 0.0);
        let s = z.sqrt();
        assert!(approx(s, Complex::I));
    }

    #[test]
    fn exp_and_ln() {
        let z = Complex::new(0.0, PI);
        let e = z.exp();
        assert!(approx(e, Complex::new(-1.0, 0.0)));
        let z2 = Complex::new(1.0, 2.0);
        let round_trip = z2.exp().ln();
        assert!(approx(round_trip, z2));
    }

    #[test]
    fn trig_real_axis() {
        let z = Complex::ZERO;
        assert!(approx(z.sin(), Complex::ZERO));
        assert!(approx(z.cos(), Complex::ONE));
        let w = Complex::new(1.0, 0.5);
        let s = w.sin();
        let c = w.cos();
        let sum = s * s + c * c;
        assert!(approx(sum, Complex::ONE));
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", Complex::new(3.0, 4.0)), "3+4i");
        assert_eq!(format!("{}", Complex::new(3.0, -4.0)), "3-4i");
    }

    #[test]
    fn parse_various_formats() {
        assert!(approx(Complex::parse("3+4i").unwrap(), Complex::new(3.0, 4.0)));
        assert!(approx(Complex::parse("3-4i").unwrap(), Complex::new(3.0, -4.0)));
        assert!(approx(Complex::parse("5i").unwrap(), Complex::new(0.0, 5.0)));
        assert!(approx(Complex::parse("i").unwrap(), Complex::I));
        assert!(approx(Complex::parse("-i").unwrap(), Complex::new(0.0, -1.0)));
        assert!(approx(Complex::parse("7").unwrap(), Complex::new(7.0, 0.0)));
    }

    #[test]
    fn complex_power() {
        let result = Complex::I.powc(Complex::I);
        let expected = Complex::new((-PI / 2.0).exp(), 0.0);
        assert!(approx(result, expected));
    }
}
