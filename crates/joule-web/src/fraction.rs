//! Rational number arithmetic — pure-Rust replacement for fraction.js, mathjs Fraction.
//!
//! Supports exact arithmetic, mixed number display, continued fractions, and parsing.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ── GCD / LCM ──────────────────────────────────────────────────

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a / gcd(a, b)).abs() * b.abs()
}

// ── Fraction ───────────────────────────────────────────────────

/// A rational number p/q in lowest terms, with q > 0.
#[derive(Debug, Clone, Copy)]
pub struct Fraction {
    pub numerator: i64,
    pub denominator: i64,
}

impl Fraction {
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    /// Create a new fraction, automatically reduced.
    /// Panics if denominator is zero.
    pub fn new(numerator: i64, denominator: i64) -> Self {
        assert!(denominator != 0, "Fraction denominator cannot be zero");
        let mut n = numerator;
        let mut d = denominator;
        // Normalize sign: denominator always positive
        if d < 0 {
            n = -n;
            d = -d;
        }
        let g = gcd(n.abs(), d);
        Self {
            numerator: n / g,
            denominator: d / g,
        }
    }

    /// Create from a whole number.
    pub fn from_int(n: i64) -> Self {
        Self {
            numerator: n,
            denominator: 1,
        }
    }

    /// Approximate an f64 as a fraction using continued fraction expansion.
    /// `max_denom` limits the denominator size.
    pub fn from_f64(value: f64, max_denom: i64) -> Self {
        if value.is_nan() || value.is_infinite() {
            return Self::ZERO;
        }
        let sign = if value < 0.0 { -1 } else { 1 };
        let mut val = value.abs();
        let mut h0: i64 = 0;
        let mut h1: i64 = 1;
        let mut k0: i64 = 1;
        let mut k1: i64 = 0;
        for _ in 0..64 {
            let a = val.floor() as i64;
            let h2 = a * h1 + h0;
            let k2 = a * k1 + k0;
            if k2 > max_denom {
                break;
            }
            h0 = h1;
            h1 = h2;
            k0 = k1;
            k1 = k2;
            let frac = val - a as f64;
            if frac.abs() < 1e-12 {
                break;
            }
            val = 1.0 / frac;
        }
        if k1 == 0 {
            return Self::ZERO;
        }
        Self::new(sign * h1, k1)
    }

    /// Convert to f64.
    pub fn to_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// Whether this fraction is zero.
    pub fn is_zero(self) -> bool {
        self.numerator == 0
    }

    /// Whether this fraction represents a whole number.
    pub fn is_integer(self) -> bool {
        self.denominator == 1
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        Self {
            numerator: self.numerator.abs(),
            denominator: self.denominator,
        }
    }

    /// Reciprocal (1/x).
    pub fn recip(self) -> Self {
        Self::new(self.denominator, self.numerator)
    }

    /// Mediant of two fractions: (a+c)/(b+d).
    pub fn mediant(self, other: Self) -> Self {
        Self::new(
            self.numerator + other.numerator,
            self.denominator + other.denominator,
        )
    }

    /// Continued fraction expansion [a0; a1, a2, ...].
    pub fn continued_fraction(self) -> Vec<i64> {
        let mut result = Vec::new();
        let mut n = self.numerator;
        let mut d = self.denominator;
        if d < 0 {
            n = -n;
            d = -d;
        }
        loop {
            let a = if d != 0 {
                if n >= 0 {
                    n / d
                } else {
                    (n - d + 1) / d
                }
            } else {
                break;
            };
            result.push(a);
            let rem = n - a * d;
            if rem == 0 {
                break;
            }
            n = d;
            d = rem;
        }
        result
    }

    /// Format as mixed number: "2 3/4" or "-1 1/2" or "5" or "3/4".
    pub fn to_mixed_string(self) -> String {
        if self.denominator == 1 {
            return self.numerator.to_string();
        }
        let sign = if self.numerator < 0 { "-" } else { "" };
        let abs_n = self.numerator.abs();
        let whole = abs_n / self.denominator;
        let remainder = abs_n % self.denominator;
        if whole == 0 {
            format!("{sign}{remainder}/{}", self.denominator)
        } else if remainder == 0 {
            format!("{sign}{whole}")
        } else {
            format!("{sign}{whole} {remainder}/{}", self.denominator)
        }
    }

    /// Parse from string. Supports:
    /// - "3/4"
    /// - "2 1/2" (mixed number)
    /// - "5" (whole number)
    /// - "-3/4"
    /// - "-2 1/2"
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        // Check for mixed number: "N P/Q"
        if let Some(space_pos) = s.rfind(' ') {
            let whole_str = &s[..space_pos];
            let frac_str = &s[space_pos + 1..];
            if let Some(slash_pos) = frac_str.find('/') {
                let whole: i64 = whole_str.parse().ok()?;
                let num: i64 = frac_str[..slash_pos].parse().ok()?;
                let den: i64 = frac_str[slash_pos + 1..].parse().ok()?;
                if den == 0 {
                    return None;
                }
                let sign = if whole < 0 { -1 } else { 1 };
                return Some(Self::new(sign * (whole.abs() * den + num), den));
            }
        }

        // Check for simple fraction: "P/Q"
        if let Some(slash_pos) = s.find('/') {
            let num: i64 = s[..slash_pos].parse().ok()?;
            let den: i64 = s[slash_pos + 1..].parse().ok()?;
            if den == 0 {
                return None;
            }
            return Some(Self::new(num, den));
        }

        // Whole number
        let n: i64 = s.parse().ok()?;
        Some(Self::from_int(n))
    }

    /// LCD (least common denominator) with another fraction.
    pub fn lcd(self, other: Self) -> i64 {
        lcm(self.denominator, other.denominator)
    }
}

// ── Operator impls ─────────────────────────────────────────────

impl Add for Fraction {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let d = lcm(self.denominator, rhs.denominator);
        let n = self.numerator * (d / self.denominator) + rhs.numerator * (d / rhs.denominator);
        Self::new(n, d)
    }
}

impl Sub for Fraction {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let d = lcm(self.denominator, rhs.denominator);
        let n = self.numerator * (d / self.denominator) - rhs.numerator * (d / rhs.denominator);
        Self::new(n, d)
    }
}

impl Mul for Fraction {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.numerator * rhs.numerator,
            self.denominator * rhs.denominator,
        )
    }
}

impl Div for Fraction {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::new(
            self.numerator * rhs.denominator,
            self.denominator * rhs.numerator,
        )
    }
}

impl Neg for Fraction {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl PartialEq for Fraction {
    fn eq(&self, other: &Self) -> bool {
        self.numerator == other.numerator && self.denominator == other.denominator
    }
}

impl Eq for Fraction {}

impl PartialOrd for Fraction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Fraction {
    fn cmp(&self, other: &Self) -> Ordering {
        let lhs = self.numerator * other.denominator;
        let rhs = other.numerator * self.denominator;
        lhs.cmp(&rhs)
    }
}

impl fmt::Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_reduce() {
        let f = Fraction::new(6, 8);
        assert_eq!(f.numerator, 3);
        assert_eq!(f.denominator, 4);
    }

    #[test]
    fn negative_denominator() {
        let f = Fraction::new(3, -4);
        assert_eq!(f.numerator, -3);
        assert_eq!(f.denominator, 4);
    }

    #[test]
    fn addition() {
        let a = Fraction::new(1, 3);
        let b = Fraction::new(1, 6);
        let c = a + b;
        assert_eq!(c, Fraction::new(1, 2));
    }

    #[test]
    fn subtraction() {
        let a = Fraction::new(3, 4);
        let b = Fraction::new(1, 4);
        assert_eq!(a - b, Fraction::new(1, 2));
    }

    #[test]
    fn multiplication() {
        let a = Fraction::new(2, 3);
        let b = Fraction::new(3, 5);
        assert_eq!(a * b, Fraction::new(2, 5));
    }

    #[test]
    fn division() {
        let a = Fraction::new(1, 2);
        let b = Fraction::new(3, 4);
        assert_eq!(a / b, Fraction::new(2, 3));
    }

    #[test]
    fn mixed_number_display() {
        assert_eq!(Fraction::new(11, 4).to_mixed_string(), "2 3/4");
        assert_eq!(Fraction::new(-11, 4).to_mixed_string(), "-2 3/4");
        assert_eq!(Fraction::new(3, 4).to_mixed_string(), "3/4");
        assert_eq!(Fraction::new(5, 1).to_mixed_string(), "5");
    }

    #[test]
    fn to_from_f64() {
        let f = Fraction::new(1, 3);
        let v = f.to_f64();
        assert!((v - 1.0 / 3.0).abs() < 1e-10);
        let back = Fraction::from_f64(v, 1000);
        assert_eq!(back, Fraction::new(1, 3));
    }

    #[test]
    fn from_f64_pi() {
        let approx_pi = Fraction::from_f64(std::f64::consts::PI, 1000);
        assert_eq!(approx_pi, Fraction::new(355, 113));
    }

    #[test]
    fn comparison() {
        let a = Fraction::new(1, 3);
        let b = Fraction::new(1, 2);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(Fraction::new(2, 4), Fraction::new(1, 2));
    }

    #[test]
    fn mediant() {
        let a = Fraction::new(1, 3);
        let b = Fraction::new(1, 2);
        let m = a.mediant(b);
        assert_eq!(m, Fraction::new(2, 5));
    }

    #[test]
    fn continued_fraction_expansion() {
        let f = Fraction::new(355, 113);
        let cf = f.continued_fraction();
        assert_eq!(cf, vec![3, 7, 16]);
    }

    #[test]
    fn parse_simple() {
        assert_eq!(Fraction::parse("3/4").unwrap(), Fraction::new(3, 4));
        assert_eq!(Fraction::parse("-3/4").unwrap(), Fraction::new(-3, 4));
        assert_eq!(Fraction::parse("5").unwrap(), Fraction::new(5, 1));
    }

    #[test]
    fn parse_mixed() {
        assert_eq!(Fraction::parse("2 1/2").unwrap(), Fraction::new(5, 2));
        assert_eq!(Fraction::parse("-2 1/2").unwrap(), Fraction::new(-5, 2));
    }

    #[test]
    fn lcd() {
        let a = Fraction::new(1, 4);
        let b = Fraction::new(1, 6);
        assert_eq!(a.lcd(b), 12);
    }

    #[test]
    fn abs_and_recip() {
        let f = Fraction::new(-3, 4);
        assert_eq!(f.abs(), Fraction::new(3, 4));
        assert_eq!(f.recip(), Fraction::new(-4, 3));
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", Fraction::new(3, 4)), "3/4");
        assert_eq!(format!("{}", Fraction::new(5, 1)), "5");
        assert_eq!(format!("{}", Fraction::new(-1, 2)), "-1/2");
    }
}
