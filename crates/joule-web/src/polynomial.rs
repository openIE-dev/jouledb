//! Polynomial arithmetic — pure-Rust replacement for polynomial.js, mathjs polynomial ops.
//!
//! Supports add, subtract, multiply, divide, evaluate, derivative, integral,
//! root finding (Newton's method), GCD, and Lagrange interpolation.

use std::fmt;

// ── Polynomial ────────────────────────────────────────────────

/// A polynomial represented by its coefficients, where `coeffs[i]` is the
/// coefficient of x^i. The zero polynomial has an empty coefficients list.
#[derive(Debug, Clone)]
pub struct Polynomial {
    /// Coefficients in ascending power order: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
    pub coeffs: Vec<f64>,
}

const EPS: f64 = 1e-12;

impl Polynomial {
    /// Create a new polynomial from coefficients in ascending power order.
    pub fn new(coeffs: Vec<f64>) -> Self {
        let mut p = Self { coeffs };
        p.trim();
        p
    }

    /// The zero polynomial.
    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    /// A constant polynomial.
    pub fn constant(c: f64) -> Self {
        Self::new(vec![c])
    }

    /// Create from roots: (x - r1)(x - r2)...
    pub fn from_roots(roots: &[f64]) -> Self {
        let mut p = Self::constant(1.0);
        for &r in roots {
            p = p.multiply(&Self::new(vec![-r, 1.0]));
        }
        p
    }

    /// Remove trailing zero coefficients.
    fn trim(&mut self) {
        while self.coeffs.last().is_some_and(|c| c.abs() < EPS) {
            self.coeffs.pop();
        }
    }

    /// Degree of the polynomial. Returns None for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Whether this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Leading coefficient.
    pub fn leading_coeff(&self) -> f64 {
        self.coeffs.last().copied().unwrap_or(0.0)
    }

    /// Evaluate the polynomial at a given x using Horner's method.
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        for &c in self.coeffs.iter().rev() {
            result = result * x + c;
        }
        result
    }

    /// Add two polynomials.
    pub fn add(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = vec![0.0; len];
        for (i, &c) in self.coeffs.iter().enumerate() {
            coeffs[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            coeffs[i] += c;
        }
        Self::new(coeffs)
    }

    /// Subtract other from self.
    pub fn subtract(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = vec![0.0; len];
        for (i, &c) in self.coeffs.iter().enumerate() {
            coeffs[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            coeffs[i] -= c;
        }
        Self::new(coeffs)
    }

    /// Multiply two polynomials.
    pub fn multiply(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![0.0; len];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                coeffs[i + j] += a * b;
            }
        }
        Self::new(coeffs)
    }

    /// Scale by a constant.
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.coeffs.iter().map(|c| c * s).collect())
    }

    /// Polynomial long division. Returns (quotient, remainder).
    pub fn divide(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "Cannot divide by zero polynomial");

        if self.is_zero() {
            return (Self::zero(), Self::zero());
        }

        let self_deg = match self.degree() {
            Some(d) => d,
            None => return (Self::zero(), Self::zero()),
        };
        let div_deg = match divisor.degree() {
            Some(d) => d,
            None => panic!("Cannot divide by zero polynomial"),
        };

        if self_deg < div_deg {
            return (Self::zero(), self.clone());
        }

        let mut remainder = self.coeffs.clone();
        let lead = divisor.leading_coeff();
        let quot_len = self_deg - div_deg + 1;
        let mut quotient = vec![0.0; quot_len];

        for i in (0..quot_len).rev() {
            let idx = i + div_deg;
            let coeff = remainder[idx] / lead;
            quotient[i] = coeff;
            for (j, &d) in divisor.coeffs.iter().enumerate() {
                remainder[i + j] -= coeff * d;
            }
        }

        (Self::new(quotient), Self::new(remainder))
    }

    /// Formal derivative.
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let coeffs: Vec<f64> = self.coeffs.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64)
            .collect();
        Self::new(coeffs)
    }

    /// Formal integral (constant of integration = 0).
    pub fn integral(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![0.0];
        for (i, &c) in self.coeffs.iter().enumerate() {
            coeffs.push(c / (i as f64 + 1.0));
        }
        Self::new(coeffs)
    }

    /// Find a root near `guess` using Newton's method.
    /// Returns None if it doesn't converge in `max_iter` iterations.
    pub fn find_root_newton(&self, guess: f64, max_iter: usize, tol: f64) -> Option<f64> {
        let deriv = self.derivative();
        let mut x = guess;
        for _ in 0..max_iter {
            let fx = self.evaluate(x);
            if fx.abs() < tol {
                return Some(x);
            }
            let dfx = deriv.evaluate(x);
            if dfx.abs() < EPS {
                return None; // derivative too small
            }
            x -= fx / dfx;
        }
        if self.evaluate(x).abs() < tol * 100.0 {
            Some(x)
        } else {
            None
        }
    }

    /// GCD of two polynomials (Euclidean algorithm), normalized so leading coeff = 1.
    pub fn gcd(a: &Self, b: &Self) -> Self {
        let mut p = a.clone();
        let mut q = b.clone();
        while !q.is_zero() {
            let (_, r) = p.divide(&q);
            p = q;
            q = r;
        }
        // Normalize
        let lead = p.leading_coeff();
        if lead.abs() > EPS {
            p = p.scale(1.0 / lead);
        }
        p
    }

    /// Lagrange interpolation: given points (x_i, y_i), find the polynomial passing
    /// through all of them.
    pub fn lagrange_interpolation(points: &[(f64, f64)]) -> Self {
        let n = points.len();
        if n == 0 {
            return Self::zero();
        }

        let mut result = Self::zero();

        for i in 0..n {
            let (xi, yi) = points[i];
            // Build basis polynomial L_i(x) = product of (x - x_j)/(x_i - x_j) for j != i
            let mut basis = Self::constant(1.0);
            for j in 0..n {
                if j == i { continue; }
                let (xj, _) = points[j];
                let denom = xi - xj;
                // (x - xj) / denom
                let factor = Self::new(vec![-xj / denom, 1.0 / denom]);
                basis = basis.multiply(&factor);
            }
            result = result.add(&basis.scale(yi));
        }

        result
    }

    /// Evaluate at multiple points.
    pub fn evaluate_batch(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|x| self.evaluate(*x)).collect()
    }

    /// Compose: return self(other(x)).
    pub fn compose(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        let mut power = Self::constant(1.0);
        for &c in &self.coeffs {
            result = result.add(&power.scale(c));
            power = power.multiply(other);
        }
        result
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
        for (i, &c) in self.coeffs.iter().enumerate().rev() {
            if c.abs() < EPS { continue; }
            if !first {
                if c > 0.0 {
                    write!(f, " + ")?;
                } else {
                    write!(f, " - ")?;
                }
            } else if c < 0.0 {
                write!(f, "-")?;
            }
            let ac = c.abs();
            match i {
                0 => write!(f, "{ac}")?,
                1 => {
                    if (ac - 1.0).abs() < EPS {
                        write!(f, "x")?;
                    } else {
                        write!(f, "{ac}x")?;
                    }
                }
                _ => {
                    if (ac - 1.0).abs() < EPS {
                        write!(f, "x^{i}")?;
                    } else {
                        write!(f, "{ac}x^{i}")?;
                    }
                }
            }
            first = false;
        }
        Ok(())
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        if self.coeffs.len() != other.coeffs.len() {
            return false;
        }
        self.coeffs.iter().zip(other.coeffs.iter())
            .all(|(a, b)| (a - b).abs() < EPS)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_and_zero() {
        let z = Polynomial::zero();
        assert!(z.is_zero());
        assert_eq!(z.degree(), None);

        let c = Polynomial::constant(5.0);
        assert_eq!(c.degree(), Some(0));
        assert_eq!(c.evaluate(100.0), 5.0);
    }

    #[test]
    fn evaluate_horner() {
        // 2x^2 + 3x + 1
        let p = Polynomial::new(vec![1.0, 3.0, 2.0]);
        assert!((p.evaluate(2.0) - 15.0).abs() < EPS);
        assert!((p.evaluate(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn addition() {
        let a = Polynomial::new(vec![1.0, 2.0, 3.0]);
        let b = Polynomial::new(vec![4.0, 5.0]);
        let sum = a.add(&b);
        assert_eq!(sum.coeffs, vec![5.0, 7.0, 3.0]);
    }

    #[test]
    fn subtraction() {
        let a = Polynomial::new(vec![5.0, 3.0, 1.0]);
        let b = Polynomial::new(vec![1.0, 1.0, 1.0]);
        let diff = a.subtract(&b);
        assert_eq!(diff.coeffs, vec![4.0, 2.0]);
    }

    #[test]
    fn multiplication() {
        // (x + 1)(x + 2) = x^2 + 3x + 2
        let a = Polynomial::new(vec![1.0, 1.0]);
        let b = Polynomial::new(vec![2.0, 1.0]);
        let prod = a.multiply(&b);
        assert_eq!(prod.degree(), Some(2));
        assert!((prod.coeffs[0] - 2.0).abs() < EPS);
        assert!((prod.coeffs[1] - 3.0).abs() < EPS);
        assert!((prod.coeffs[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn division() {
        // (x^2 + 3x + 2) / (x + 1) = (x + 2), remainder 0
        let dividend = Polynomial::new(vec![2.0, 3.0, 1.0]);
        let divisor = Polynomial::new(vec![1.0, 1.0]);
        let (q, r) = dividend.divide(&divisor);
        assert_eq!(q, Polynomial::new(vec![2.0, 1.0]));
        assert!(r.is_zero());
    }

    #[test]
    fn division_with_remainder() {
        // (x^2 + 1) / (x + 1) = x - 1, remainder 2
        let dividend = Polynomial::new(vec![1.0, 0.0, 1.0]);
        let divisor = Polynomial::new(vec![1.0, 1.0]);
        let (q, r) = dividend.divide(&divisor);
        assert_eq!(q, Polynomial::new(vec![-1.0, 1.0]));
        assert!((r.coeffs[0] - 2.0).abs() < EPS);
    }

    #[test]
    fn derivative() {
        // d/dx (3x^3 + 2x^2 + x + 5) = 9x^2 + 4x + 1
        let p = Polynomial::new(vec![5.0, 1.0, 2.0, 3.0]);
        let d = p.derivative();
        assert_eq!(d.degree(), Some(2));
        assert!((d.coeffs[0] - 1.0).abs() < EPS);
        assert!((d.coeffs[1] - 4.0).abs() < EPS);
        assert!((d.coeffs[2] - 9.0).abs() < EPS);
    }

    #[test]
    fn integral() {
        // integral of 3x^2 + 2x + 1 = x^3 + x^2 + x
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        let i = p.integral();
        assert_eq!(i.degree(), Some(3));
        assert!((i.coeffs[0] - 0.0).abs() < EPS);
        assert!((i.coeffs[1] - 1.0).abs() < EPS);
        assert!((i.coeffs[2] - 1.0).abs() < EPS);
        assert!((i.coeffs[3] - 1.0).abs() < EPS);
    }

    #[test]
    fn newton_root_finding() {
        // x^2 - 4 has roots at +/- 2
        let p = Polynomial::new(vec![-4.0, 0.0, 1.0]);
        let root = p.find_root_newton(3.0, 100, 1e-10).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn newton_root_negative() {
        let p = Polynomial::new(vec![-4.0, 0.0, 1.0]);
        let root = p.find_root_newton(-3.0, 100, 1e-10).unwrap();
        assert!((root - (-2.0)).abs() < 1e-8);
    }

    #[test]
    fn gcd_polynomials() {
        // gcd of (x+1)(x+2) and (x+1)(x+3) should be (x+1) (normalized)
        let a = Polynomial::from_roots(&[-1.0, -2.0]);
        let b = Polynomial::from_roots(&[-1.0, -3.0]);
        let g = Polynomial::gcd(&a, &b);
        // Should be x + 1 (normalized leading coeff = 1)
        assert_eq!(g.degree(), Some(1));
        assert!((g.coeffs[1] - 1.0).abs() < 1e-8);
        assert!((g.coeffs[0] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn lagrange_interpolation() {
        let points = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let p = Polynomial::lagrange_interpolation(&points);
        for &(x, y) in &points {
            assert!((p.evaluate(x) - y).abs() < 1e-10);
        }
    }

    #[test]
    fn from_roots() {
        let p = Polynomial::from_roots(&[1.0, -1.0]);
        // (x - 1)(x + 1) = x^2 - 1
        assert!((p.evaluate(1.0)).abs() < EPS);
        assert!((p.evaluate(-1.0)).abs() < EPS);
        assert!((p.evaluate(0.0) - (-1.0)).abs() < EPS);
    }

    #[test]
    fn compose() {
        // f(x) = x^2, g(x) = x + 1
        // f(g(x)) = (x+1)^2 = x^2 + 2x + 1
        let f = Polynomial::new(vec![0.0, 0.0, 1.0]);
        let g = Polynomial::new(vec![1.0, 1.0]);
        let c = f.compose(&g);
        assert!((c.evaluate(2.0) - 9.0).abs() < EPS);
        assert!((c.evaluate(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn display_format() {
        let p = Polynomial::new(vec![1.0, 0.0, 3.0]);
        let s = format!("{}", p);
        assert!(s.contains("x^2"));
        assert!(s.contains("1"));
    }

    #[test]
    fn evaluate_batch() {
        let p = Polynomial::new(vec![0.0, 1.0]); // f(x) = x
        let results = p.evaluate_batch(&[1.0, 2.0, 3.0]);
        assert_eq!(results, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn scale_polynomial() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        let scaled = p.scale(2.0);
        assert_eq!(scaled.coeffs, vec![2.0, 4.0, 6.0]);
    }
}
