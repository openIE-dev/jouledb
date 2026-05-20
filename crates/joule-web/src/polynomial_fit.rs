//! Polynomial fitting, evaluation, arithmetic, and interpolation.
//!
//! Vandermonde least-squares fitting, Horner evaluation, add/sub/mul/div,
//! derivative, integral, root finding (Durand-Kerner), Lagrange and Newton
//! interpolation, Chebyshev basis, and piecewise polynomial.

use std::fmt;

// ── Polynomial ────────────────────────────────────────────────

/// A polynomial stored as coefficients `[c0, c1, ..., cn]` representing
/// c0 + c1*x + c2*x^2 + ... + cn*x^n.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    /// Coefficients from degree 0 upward.
    pub coeffs: Vec<f64>,
}

impl Polynomial {
    /// Create from coefficients (lowest degree first).
    pub fn new(coeffs: Vec<f64>) -> Self {
        let mut p = Self { coeffs };
        p.trim();
        p
    }

    /// Zero polynomial.
    pub fn zero() -> Self {
        Self { coeffs: vec![0.0] }
    }

    /// Constant polynomial.
    pub fn constant(c: f64) -> Self {
        Self { coeffs: vec![c] }
    }

    /// Monomial x^n.
    pub fn monomial(n: usize) -> Self {
        let mut c = vec![0.0; n + 1];
        c[n] = 1.0;
        Self { coeffs: c }
    }

    /// Degree of the polynomial (0 for constant/zero).
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() { 0 } else { self.coeffs.len().saturating_sub(1) }
    }

    /// Remove trailing near-zero coefficients.
    fn trim(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs.last().map_or(false, |c| c.abs() < 1e-15) {
            self.coeffs.pop();
        }
    }

    // ── Evaluation ────────────────────────────────────────────

    /// Evaluate using Horner's method.
    pub fn eval(&self, x: f64) -> f64 {
        let mut result = 0.0;
        for c in self.coeffs.iter().rev() {
            result = result * x + c;
        }
        result
    }

    /// Evaluate at multiple points.
    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|x| self.eval(*x)).collect()
    }

    // ── Arithmetic ────────────────────────────────────────────

    /// Add two polynomials.
    pub fn add(&self, other: &Polynomial) -> Polynomial {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut c = vec![0.0; n];
        for (i, v) in self.coeffs.iter().enumerate() { c[i] += v; }
        for (i, v) in other.coeffs.iter().enumerate() { c[i] += v; }
        Polynomial::new(c)
    }

    /// Subtract: self - other.
    pub fn sub(&self, other: &Polynomial) -> Polynomial {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut c = vec![0.0; n];
        for (i, v) in self.coeffs.iter().enumerate() { c[i] += v; }
        for (i, v) in other.coeffs.iter().enumerate() { c[i] -= v; }
        Polynomial::new(c)
    }

    /// Multiply two polynomials.
    pub fn mul(&self, other: &Polynomial) -> Polynomial {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Polynomial::zero();
        }
        let n = self.coeffs.len() + other.coeffs.len() - 1;
        let mut c = vec![0.0; n];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in other.coeffs.iter().enumerate() {
                c[i + j] += a * b;
            }
        }
        Polynomial::new(c)
    }

    /// Polynomial long division: self / divisor = (quotient, remainder).
    pub fn div(&self, divisor: &Polynomial) -> (Polynomial, Polynomial) {
        let dn = divisor.degree();
        if divisor.coeffs.last().map_or(true, |c| c.abs() < 1e-15) {
            // Division by zero polynomial — return self as remainder.
            return (Polynomial::zero(), self.clone());
        }
        if self.degree() < dn {
            return (Polynomial::zero(), self.clone());
        }

        let mut rem = self.coeffs.clone();
        let lead = *divisor.coeffs.last().unwrap();
        let quot_len = self.degree() - dn + 1;
        let mut quot = vec![0.0; quot_len];

        for i in (0..quot_len).rev() {
            let idx = i + dn;
            if idx >= rem.len() { continue; }
            let q = rem[idx] / lead;
            quot[i] = q;
            for (j, dc) in divisor.coeffs.iter().enumerate() {
                rem[i + j] -= q * dc;
            }
        }
        (Polynomial::new(quot), Polynomial::new(rem))
    }

    /// Scale by a constant.
    pub fn scale(&self, s: f64) -> Polynomial {
        Polynomial::new(self.coeffs.iter().map(|c| c * s).collect())
    }

    // ── Calculus ──────────────────────────────────────────────

    /// Derivative of the polynomial.
    pub fn derivative(&self) -> Polynomial {
        if self.coeffs.len() <= 1 {
            return Polynomial::zero();
        }
        let c: Vec<f64> = self.coeffs.iter().enumerate().skip(1)
            .map(|(i, c)| *c * i as f64)
            .collect();
        Polynomial::new(c)
    }

    /// Indefinite integral (constant of integration = 0).
    pub fn integral(&self) -> Polynomial {
        let mut c = vec![0.0; self.coeffs.len() + 1];
        for (i, v) in self.coeffs.iter().enumerate() {
            c[i + 1] = v / (i + 1) as f64;
        }
        Polynomial::new(c)
    }

    /// Definite integral from a to b.
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        let anti = self.integral();
        anti.eval(b) - anti.eval(a)
    }

    // ── Root finding (Durand-Kerner) ──────────────────────────

    /// Find roots using Durand-Kerner iteration.
    /// Returns (real_parts, imag_parts) for each root.
    pub fn roots(&self, max_iter: usize) -> Vec<(f64, f64)> {
        let n = self.degree();
        if n == 0 { return Vec::new(); }
        if n == 1 {
            return vec![(-self.coeffs[0] / self.coeffs[1], 0.0)];
        }

        // Normalize to monic.
        let lead = *self.coeffs.last().unwrap();
        let monic: Vec<f64> = self.coeffs.iter().map(|c| c / lead).collect();

        // Initial guesses: points on a circle.
        let mut zr = Vec::with_capacity(n);
        let mut zi = Vec::with_capacity(n);
        let radius = 1.0 + monic.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        for k in 0..n {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64 + 0.3;
            zr.push(radius * angle.cos());
            zi.push(radius * angle.sin());
        }

        for _ in 0..max_iter {
            let mut max_delta = 0.0f64;
            for i in 0..n {
                // Evaluate monic polynomial at z_i.
                let (mut pr, mut pi_val) = (0.0, 0.0);
                for (k, c) in monic.iter().enumerate() {
                    // Compute z^k.
                    let (mut pkr, mut pki) = (1.0, 0.0);
                    for _ in 0..k {
                        let nr = pkr * zr[i] - pki * zi[i];
                        let ni = pkr * zi[i] + pki * zr[i];
                        pkr = nr;
                        pki = ni;
                    }
                    pr += c * pkr;
                    pi_val += c * pki;
                }

                // Denominator: product of (z_i - z_j) for j != i.
                let (mut dr, mut di) = (1.0, 0.0);
                for j in 0..n {
                    if j == i { continue; }
                    let diffr = zr[i] - zr[j];
                    let diffi = zi[i] - zi[j];
                    let nr = dr * diffr - di * diffi;
                    let ni = dr * diffi + di * diffr;
                    dr = nr;
                    di = ni;
                }

                // z_i -= p(z_i) / prod(z_i - z_j)
                let denom = dr * dr + di * di;
                if denom < 1e-30 { continue; }
                let delta_r = (pr * dr + pi_val * di) / denom;
                let delta_i = (pi_val * dr - pr * di) / denom;
                zr[i] -= delta_r;
                zi[i] -= delta_i;
                max_delta = max_delta.max((delta_r * delta_r + delta_i * delta_i).sqrt());
            }
            if max_delta < 1e-12 { break; }
        }

        zr.into_iter().zip(zi.into_iter()).collect()
    }

    /// Real roots only (imaginary part < tolerance).
    pub fn real_roots(&self, tol: f64) -> Vec<f64> {
        self.roots(1000)
            .into_iter()
            .filter(|(_, im)| im.abs() < tol)
            .map(|(re, _)| re)
            .collect()
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.abs() < 1e-15 && self.coeffs.len() > 1 { continue; }
            if !first {
                if *c >= 0.0 { write!(f, " + ")?; } else { write!(f, " - ")?; }
            }
            let abs_c = if first { *c } else { c.abs() };
            match i {
                0 => write!(f, "{:.6}", abs_c)?,
                1 => write!(f, "{:.6}x", abs_c)?,
                _ => write!(f, "{:.6}x^{}", abs_c, i)?,
            }
            first = false;
        }
        Ok(())
    }
}

// ── Interpolation ─────────────────────────────────────────────

/// Lagrange interpolation: build polynomial passing through (xs[i], ys[i]).
pub fn lagrange_interpolate(xs: &[f64], ys: &[f64]) -> Polynomial {
    let n = xs.len();
    assert_eq!(n, ys.len());
    let mut result = Polynomial::zero();
    for i in 0..n {
        let mut basis = Polynomial::constant(ys[i]);
        for j in 0..n {
            if j == i { continue; }
            let denom = xs[i] - xs[j];
            // (x - xs[j]) / denom
            let factor = Polynomial::new(vec![-xs[j] / denom, 1.0 / denom]);
            basis = basis.mul(&factor);
        }
        result = result.add(&basis);
    }
    result
}

/// Newton's divided differences interpolation.
pub fn newton_interpolate(xs: &[f64], ys: &[f64]) -> Polynomial {
    let n = xs.len();
    assert_eq!(n, ys.len());
    // Build divided difference table.
    let mut dd = ys.to_vec();
    let mut coeffs = vec![dd[0]];
    for j in 1..n {
        let mut new_dd = Vec::with_capacity(n - j);
        for i in 0..(n - j) {
            new_dd.push((dd[i + 1] - dd[i]) / (xs[i + j] - xs[i]));
        }
        coeffs.push(new_dd[0]);
        dd = new_dd;
    }
    // Build polynomial: c0 + c1*(x-x0) + c2*(x-x0)*(x-x1) + ...
    let mut result = Polynomial::constant(coeffs[0]);
    let mut basis = Polynomial::constant(1.0);
    for (k, &ck) in coeffs.iter().enumerate().skip(1) {
        basis = basis.mul(&Polynomial::new(vec![-xs[k - 1], 1.0]));
        result = result.add(&basis.scale(ck));
    }
    result
}

// ── Chebyshev ─────────────────────────────────────────────────

/// Chebyshev polynomial of the first kind T_n(x), returned as a standard
/// polynomial in the monomial basis.
pub fn chebyshev_polynomial(n: usize) -> Polynomial {
    match n {
        0 => Polynomial::constant(1.0),
        1 => Polynomial::new(vec![0.0, 1.0]),
        _ => {
            let mut t_prev = Polynomial::constant(1.0);
            let mut t_curr = Polynomial::new(vec![0.0, 1.0]);
            for _ in 2..=n {
                let two_x = Polynomial::new(vec![0.0, 2.0]);
                let next = two_x.mul(&t_curr).sub(&t_prev);
                t_prev = t_curr;
                t_curr = next;
            }
            t_curr
        }
    }
}

/// Chebyshev nodes on [a, b] for degree n interpolation.
pub fn chebyshev_nodes(n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
            0.5 * (a + b) + 0.5 * (b - a) * theta.cos()
        })
        .collect()
}

// ── Polynomial fitting (Vandermonde least squares) ────────────

/// Fit a polynomial of degree `deg` to data points (xs, ys) via least squares.
pub fn polyfit(xs: &[f64], ys: &[f64], deg: usize) -> Polynomial {
    let m = xs.len();
    let n = deg + 1;
    assert!(m >= n, "need at least deg+1 points");

    // Build Vandermonde and solve via QR.
    let mut vand = vec![0.0; m * n];
    for i in 0..m {
        let mut xi = 1.0;
        for j in 0..n {
            vand[i * n + j] = xi;
            xi *= xs[i];
        }
    }

    // Thin QR via Householder.
    let mut r = vand.clone();
    let rows = m;
    let cols = n;
    let mut q_data = vec![0.0; rows * rows];
    for i in 0..rows { q_data[i * rows + i] = 1.0; }

    for k in 0..cols {
        let mut col: Vec<f64> = (k..rows).map(|i| r[i * cols + k]).collect();
        let norm_c: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_c < 1e-15 { continue; }
        let sign = if col[0] >= 0.0 { 1.0 } else { -1.0 };
        col[0] += sign * norm_c;
        let vsq: f64 = col.iter().map(|x| x * x).sum();
        if vsq < 1e-30 { continue; }
        for j in k..cols {
            let mut d = 0.0;
            for i in 0..col.len() { d += col[i] * r[(i + k) * cols + j]; }
            let c2 = 2.0 * d / vsq;
            for i in 0..col.len() { r[(i + k) * cols + j] -= c2 * col[i]; }
        }
        for i in 0..rows {
            let mut d = 0.0;
            for jj in 0..col.len() { d += q_data[i * rows + jj + k] * col[jj]; }
            let c2 = 2.0 * d / vsq;
            for jj in 0..col.len() { q_data[i * rows + jj + k] -= c2 * col[jj]; }
        }
    }

    // Q^T b (use first n columns of Q).
    let mut qtb = vec![0.0; n];
    for i in 0..n {
        for j in 0..rows { qtb[i] += q_data[j * rows + i] * ys[j]; }
    }
    // Back-sub.
    let mut coeffs = vec![0.0; n];
    for i in (0..n).rev() {
        let diag = r[i * cols + i];
        if diag.abs() < 1e-14 { continue; }
        let mut s = qtb[i];
        for j in (i + 1)..n { s -= r[i * cols + j] * coeffs[j]; }
        coeffs[i] = s / diag;
    }
    Polynomial::new(coeffs)
}

// ── Piecewise polynomial ──────────────────────────────────────

/// A piecewise polynomial (spline-like) defined by breakpoints and polynomials.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewisePoly {
    /// Breakpoints: xs[0] < xs[1] < ... < xs[n].
    pub breakpoints: Vec<f64>,
    /// Polynomials: polys[i] is valid on [breakpoints[i], breakpoints[i+1]).
    pub pieces: Vec<Polynomial>,
}

impl PiecewisePoly {
    /// Build from breakpoints and polynomial pieces.
    pub fn new(breakpoints: Vec<f64>, pieces: Vec<Polynomial>) -> Self {
        assert_eq!(breakpoints.len(), pieces.len() + 1);
        Self { breakpoints, pieces }
    }

    /// Evaluate the piecewise polynomial at x.
    pub fn eval(&self, x: f64) -> f64 {
        // Find the segment.
        let n = self.pieces.len();
        if n == 0 { return 0.0; }
        if x <= self.breakpoints[0] { return self.pieces[0].eval(x); }
        if x >= self.breakpoints[n] { return self.pieces[n - 1].eval(x); }
        for i in 0..n {
            if x < self.breakpoints[i + 1] {
                return self.pieces[i].eval(x);
            }
        }
        self.pieces[n - 1].eval(x)
    }

    /// Build a linear interpolating piecewise polynomial from data points.
    pub fn linear_interpolant(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        assert!(n >= 2);
        let mut pieces = Vec::with_capacity(n - 1);
        for i in 0..(n - 1) {
            let dx = xs[i + 1] - xs[i];
            if dx.abs() < 1e-30 {
                pieces.push(Polynomial::constant(ys[i]));
                continue;
            }
            let slope = (ys[i + 1] - ys[i]) / dx;
            let intercept = ys[i] - slope * xs[i];
            pieces.push(Polynomial::new(vec![intercept, slope]));
        }
        Self::new(xs.to_vec(), pieces)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    #[test]
    fn test_eval_constant() {
        let p = Polynomial::constant(5.0);
        assert!(approx_eq(p.eval(0.0), 5.0, 1e-12));
        assert!(approx_eq(p.eval(100.0), 5.0, 1e-12));
    }

    #[test]
    fn test_eval_linear() {
        let p = Polynomial::new(vec![1.0, 2.0]); // 1 + 2x
        assert!(approx_eq(p.eval(3.0), 7.0, 1e-12));
    }

    #[test]
    fn test_eval_quadratic() {
        let p = Polynomial::new(vec![1.0, 0.0, 1.0]); // 1 + x^2
        assert!(approx_eq(p.eval(3.0), 10.0, 1e-12));
    }

    #[test]
    fn test_add() {
        let a = Polynomial::new(vec![1.0, 2.0]);
        let b = Polynomial::new(vec![3.0, 4.0, 5.0]);
        let c = a.add(&b);
        assert!(approx_eq(c.eval(1.0), 15.0, 1e-12)); // (1+2) + (3+4+5) = 3 + 12 = 15
    }

    #[test]
    fn test_sub() {
        let a = Polynomial::new(vec![5.0, 3.0]);
        let b = Polynomial::new(vec![2.0, 1.0]);
        let c = a.sub(&b);
        assert!(approx_eq(c.coeffs[0], 3.0, 1e-12));
        assert!(approx_eq(c.coeffs[1], 2.0, 1e-12));
    }

    #[test]
    fn test_mul() {
        let a = Polynomial::new(vec![1.0, 1.0]); // 1+x
        let b = Polynomial::new(vec![1.0, 1.0]); // 1+x
        let c = a.mul(&b); // 1 + 2x + x^2
        assert!(approx_eq(c.eval(2.0), 9.0, 1e-12));
    }

    #[test]
    fn test_div() {
        // (x^2 - 1) / (x - 1) = (x + 1), remainder 0
        let num = Polynomial::new(vec![-1.0, 0.0, 1.0]);
        let den = Polynomial::new(vec![-1.0, 1.0]);
        let (q, r) = num.div(&den);
        assert!(approx_eq(q.coeffs[0], 1.0, 1e-10));
        assert!(approx_eq(q.coeffs[1], 1.0, 1e-10));
        assert!(r.coeffs.iter().all(|c| c.abs() < 1e-10));
    }

    #[test]
    fn test_div_with_remainder() {
        // (x^2 + 1) / (x - 1) = (x + 1) remainder 2
        let num = Polynomial::new(vec![1.0, 0.0, 1.0]);
        let den = Polynomial::new(vec![-1.0, 1.0]);
        let (q, r) = num.div(&den);
        assert!(approx_eq(q.eval(0.0), 1.0, 1e-10));
        assert!(approx_eq(r.eval(0.0), 2.0, 1e-10));
    }

    #[test]
    fn test_derivative() {
        let p = Polynomial::new(vec![1.0, 3.0, 5.0]); // 1 + 3x + 5x^2
        let dp = p.derivative(); // 3 + 10x
        assert!(approx_eq(dp.coeffs[0], 3.0, 1e-12));
        assert!(approx_eq(dp.coeffs[1], 10.0, 1e-12));
    }

    #[test]
    fn test_integral() {
        let p = Polynomial::new(vec![3.0, 2.0]); // 3 + 2x
        let ip = p.integral(); // 3x + x^2
        assert!(approx_eq(ip.coeffs[0], 0.0, 1e-12));
        assert!(approx_eq(ip.coeffs[1], 3.0, 1e-12));
        assert!(approx_eq(ip.coeffs[2], 1.0, 1e-12));
    }

    #[test]
    fn test_definite_integral() {
        let p = Polynomial::new(vec![0.0, 0.0, 1.0]); // x^2
        let val = p.integrate(0.0, 3.0); // = 9
        assert!(approx_eq(val, 9.0, 1e-12));
    }

    #[test]
    fn test_roots_linear() {
        let p = Polynomial::new(vec![-6.0, 2.0]); // 2x - 6 = 0 => x = 3
        let r = p.real_roots(1e-6);
        assert_eq!(r.len(), 1);
        assert!(approx_eq(r[0], 3.0, 1e-6));
    }

    #[test]
    fn test_roots_quadratic() {
        let p = Polynomial::new(vec![-6.0, 1.0, 1.0]); // x^2 + x - 6 = (x+3)(x-2)
        let mut r = p.real_roots(1e-4);
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(r.len(), 2);
        assert!(approx_eq(r[0], -3.0, 1e-4));
        assert!(approx_eq(r[1], 2.0, 1e-4));
    }

    #[test]
    fn test_lagrange_interpolation() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![1.0, 3.0, 9.0]; // Not a simple poly, but interpolation should match
        let p = lagrange_interpolate(&xs, &ys);
        assert!(approx_eq(p.eval(0.0), 1.0, 1e-10));
        assert!(approx_eq(p.eval(1.0), 3.0, 1e-10));
        assert!(approx_eq(p.eval(2.0), 9.0, 1e-10));
    }

    #[test]
    fn test_newton_interpolation() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![1.0, 3.0, 9.0];
        let p = newton_interpolate(&xs, &ys);
        assert!(approx_eq(p.eval(0.0), 1.0, 1e-10));
        assert!(approx_eq(p.eval(1.0), 3.0, 1e-10));
        assert!(approx_eq(p.eval(2.0), 9.0, 1e-10));
    }

    #[test]
    fn test_lagrange_equals_newton() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 2.0, 5.0, 10.0];
        let lag = lagrange_interpolate(&xs, &ys);
        let newt = newton_interpolate(&xs, &ys);
        for x in [0.5, 1.5, 2.5] {
            assert!(approx_eq(lag.eval(x), newt.eval(x), 1e-8));
        }
    }

    #[test]
    fn test_chebyshev_t0() {
        let t0 = chebyshev_polynomial(0);
        assert!(approx_eq(t0.eval(0.5), 1.0, 1e-12));
    }

    #[test]
    fn test_chebyshev_t1() {
        let t1 = chebyshev_polynomial(1);
        assert!(approx_eq(t1.eval(0.5), 0.5, 1e-12));
    }

    #[test]
    fn test_chebyshev_t2() {
        let t2 = chebyshev_polynomial(2); // 2x^2 - 1
        assert!(approx_eq(t2.eval(0.5), -0.5, 1e-12));
    }

    #[test]
    fn test_chebyshev_nodes() {
        let nodes = chebyshev_nodes(5, -1.0, 1.0);
        assert_eq!(nodes.len(), 5);
        for &x in &nodes {
            assert!(x >= -1.0 - 1e-10 && x <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_polyfit_linear() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.0, 3.0, 5.0, 7.0, 9.0]; // y = 1 + 2x
        let p = polyfit(&xs, &ys, 1);
        assert!(approx_eq(p.coeffs[0], 1.0, 1e-8));
        assert!(approx_eq(p.coeffs[1], 2.0, 1e-8));
    }

    #[test]
    fn test_polyfit_quadratic() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 0.5 * x * x).collect();
        let p = polyfit(&xs, &ys, 2);
        assert!(approx_eq(p.coeffs[0], 1.0, 1e-6));
        assert!(approx_eq(p.coeffs[2], 0.5, 1e-6));
    }

    #[test]
    fn test_piecewise_linear() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 2.0, 1.0];
        let pw = PiecewisePoly::linear_interpolant(&xs, &ys);
        assert!(approx_eq(pw.eval(0.5), 1.0, 1e-10));
        assert!(approx_eq(pw.eval(1.5), 1.5, 1e-10));
    }

    #[test]
    fn test_piecewise_endpoints() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 2.0, 1.0];
        let pw = PiecewisePoly::linear_interpolant(&xs, &ys);
        assert!(approx_eq(pw.eval(0.0), 0.0, 1e-10));
        assert!(approx_eq(pw.eval(1.0), 2.0, 1e-10));
    }

    #[test]
    fn test_monomial() {
        let p = Polynomial::monomial(3); // x^3
        assert!(approx_eq(p.eval(2.0), 8.0, 1e-12));
    }

    #[test]
    fn test_degree() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.degree(), 2);
    }

    #[test]
    fn test_scale() {
        let p = Polynomial::new(vec![1.0, 2.0]);
        let s = p.scale(3.0);
        assert!(approx_eq(s.coeffs[0], 3.0, 1e-12));
        assert!(approx_eq(s.coeffs[1], 6.0, 1e-12));
    }

    #[test]
    fn test_display() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(s.contains("1.000000"));
    }

    #[test]
    fn test_eval_many() {
        let p = Polynomial::new(vec![0.0, 1.0]); // x
        let vals = p.eval_many(&[1.0, 2.0, 3.0]);
        assert!(approx_eq(vals[0], 1.0, 1e-12));
        assert!(approx_eq(vals[1], 2.0, 1e-12));
        assert!(approx_eq(vals[2], 3.0, 1e-12));
    }
}
