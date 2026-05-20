//! Finite differences — forward/backward/central differences, gradient, Hessian,
//! Jacobian approximation, Richardson extrapolation, stencil coefficients, error order.
//!
//! Pure-Rust finite difference methods for numerical differentiation.

use std::fmt;

// ── Basic finite differences (scalar → scalar) ──────────────────

/// Forward difference approximation of f'(x).
/// f'(x) ≈ (f(x+h) - f(x)) / h
pub fn forward_diff(f: fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x)) / h
}

/// Backward difference approximation of f'(x).
/// f'(x) ≈ (f(x) - f(x-h)) / h
pub fn backward_diff(f: fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x) - f(x - h)) / h
}

/// Central difference approximation of f'(x).
/// f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
pub fn central_diff(f: fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Second derivative using central difference.
/// f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
pub fn second_central_diff(f: fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

/// Fourth-order central difference for f'(x).
/// Uses 5-point stencil: (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
pub fn central_diff_4th(f: fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
}

// ── Gradient approximation ───────────────────────────────────────

/// Approximate the gradient of f: R^n → R using central differences.
/// Returns ∇f(x) as a vector.
pub fn gradient(
    f: fn(&[f64]) -> f64,
    x: &[f64],
    h: f64,
) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for i in 0..n {
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;
        grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * h);
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    grad
}

/// Approximate gradient using forward differences (cheaper but less accurate).
pub fn gradient_forward(
    f: fn(&[f64]) -> f64,
    x: &[f64],
    h: f64,
) -> Vec<f64> {
    let n = x.len();
    let f0 = f(x);
    let mut grad = vec![0.0; n];
    let mut x_plus = x.to_vec();

    for i in 0..n {
        x_plus[i] = x[i] + h;
        grad[i] = (f(&x_plus) - f0) / h;
        x_plus[i] = x[i];
    }

    grad
}

// ── Hessian approximation ────────────────────────────────────────

/// Approximate the Hessian matrix of f: R^n → R using central differences.
/// Returns H[i][j] = ∂²f/∂x_i∂x_j.
pub fn hessian(
    f: fn(&[f64]) -> f64,
    x: &[f64],
    h: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut hess = vec![vec![0.0; n]; n];
    let mut xp = x.to_vec();

    // Diagonal: ∂²f/∂x_i² ≈ (f(x+hei) - 2f(x) + f(x-hei)) / h²
    let f0 = f(x);
    for i in 0..n {
        xp[i] = x[i] + h;
        let fph = f(&xp);
        xp[i] = x[i] - h;
        let fmh = f(&xp);
        xp[i] = x[i];
        hess[i][i] = (fph - 2.0 * f0 + fmh) / (h * h);
    }

    // Off-diagonal: ∂²f/∂x_i∂x_j ≈ (f(x+hei+hej) - f(x+hei-hej) - f(x-hei+hej) + f(x-hei-hej)) / (4h²)
    for i in 0..n {
        for j in (i + 1)..n {
            xp[i] = x[i] + h;
            xp[j] = x[j] + h;
            let fpp = f(&xp);

            xp[j] = x[j] - h;
            let fpm = f(&xp);

            xp[i] = x[i] - h;
            let fmm = f(&xp);

            xp[j] = x[j] + h;
            let fmp = f(&xp);

            xp[i] = x[i];
            xp[j] = x[j];

            let val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            hess[i][j] = val;
            hess[j][i] = val;
        }
    }

    hess
}

// ── Jacobian approximation ───────────────────────────────────────

/// Approximate the Jacobian of F: R^n → R^m using central differences.
/// Returns J[i][j] = ∂F_i/∂x_j.
pub fn jacobian(
    f: fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    h: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();

    let mut jac = vec![vec![0.0; n]; m];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for j in 0..n {
        x_plus[j] = x[j] + h;
        x_minus[j] = x[j] - h;
        let fp = f(&x_plus);
        let fm = f(&x_minus);
        for i in 0..m {
            jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
        }
        x_plus[j] = x[j];
        x_minus[j] = x[j];
    }

    jac
}

/// Approximate Jacobian using forward differences.
pub fn jacobian_forward(
    f: fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    h: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();

    let mut jac = vec![vec![0.0; n]; m];
    let mut x_plus = x.to_vec();

    for j in 0..n {
        x_plus[j] = x[j] + h;
        let fp = f(&x_plus);
        for i in 0..m {
            jac[i][j] = (fp[i] - f0[i]) / h;
        }
        x_plus[j] = x[j];
    }

    jac
}

// ── Richardson extrapolation ─────────────────────────────────────

/// Richardson extrapolation to improve a finite difference estimate.
/// Given a function that computes an approximation D(h), and assuming
/// D(h) = exact + c*h^p + ..., this combines D(h) and D(h/r) to
/// cancel the leading error term.
///
/// Returns the extrapolated value.
pub fn richardson_extrapolation(
    compute: fn(f64) -> f64,
    h: f64,
    r: f64,
    order: usize,
) -> f64 {
    let rp = r.powi(order as i32);
    let d_h = compute(h);
    let d_hr = compute(h / r);
    (rp * d_hr - d_h) / (rp - 1.0)
}

/// Richardson table: progressively refine estimates.
/// Returns a table where entry[k] is the k-th level extrapolation.
pub fn richardson_table(
    compute: fn(f64) -> f64,
    h0: f64,
    r: f64,
    levels: usize,
) -> Vec<Vec<f64>> {
    let mut table = vec![vec![0.0; levels]; levels];

    // First column: direct evaluations at h, h/r, h/r^2, ...
    for i in 0..levels {
        let h = h0 / r.powi(i as i32);
        table[i][0] = compute(h);
    }

    // Extrapolation columns
    for j in 1..levels {
        let rp = r.powi(j as i32);
        for i in j..levels {
            table[i][j] = (rp * table[i][j - 1] - table[i - 1][j - 1]) / (rp - 1.0);
        }
    }

    table
}

// ── Stencil coefficients ─────────────────────────────────────────

/// Finite difference stencil for derivative of given order.
/// Returns (offsets, coefficients) for derivative of order `deriv_order`
/// using `accuracy_order` accuracy on a uniform grid with spacing h.
///
/// Supported combinations:
/// - deriv=1, accuracy=2: central 3-point
/// - deriv=1, accuracy=4: central 5-point
/// - deriv=2, accuracy=2: central 3-point second derivative
/// - deriv=2, accuracy=4: central 5-point second derivative
#[derive(Debug, Clone)]
pub struct Stencil {
    pub offsets: Vec<i32>,
    pub coefficients: Vec<f64>,
    pub deriv_order: usize,
    pub accuracy_order: usize,
}

impl Stencil {
    /// Get a predefined stencil.
    pub fn get(deriv_order: usize, accuracy_order: usize) -> Option<Self> {
        match (deriv_order, accuracy_order) {
            (1, 2) => Some(Self {
                offsets: vec![-1, 0, 1],
                coefficients: vec![-0.5, 0.0, 0.5],
                deriv_order: 1,
                accuracy_order: 2,
            }),
            (1, 4) => Some(Self {
                offsets: vec![-2, -1, 0, 1, 2],
                coefficients: vec![1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0],
                deriv_order: 1,
                accuracy_order: 4,
            }),
            (2, 2) => Some(Self {
                offsets: vec![-1, 0, 1],
                coefficients: vec![1.0, -2.0, 1.0],
                deriv_order: 2,
                accuracy_order: 2,
            }),
            (2, 4) => Some(Self {
                offsets: vec![-2, -1, 0, 1, 2],
                coefficients: vec![
                    -1.0 / 12.0,
                    4.0 / 3.0,
                    -5.0 / 2.0,
                    4.0 / 3.0,
                    -1.0 / 12.0,
                ],
                deriv_order: 2,
                accuracy_order: 4,
            }),
            (1, 1) => Some(Self {
                offsets: vec![0, 1],
                coefficients: vec![-1.0, 1.0],
                deriv_order: 1,
                accuracy_order: 1,
            }),
            _ => None,
        }
    }

    /// Apply stencil to compute derivative of sampled data at index `idx`.
    /// `data` is uniformly sampled with spacing `h`.
    pub fn apply(&self, data: &[f64], idx: usize, h: f64) -> Option<f64> {
        let mut result = 0.0;
        for (offset, coeff) in self.offsets.iter().zip(self.coefficients.iter()) {
            let j = idx as i64 + *offset as i64;
            if j < 0 || j as usize >= data.len() {
                return None;
            }
            result += coeff * data[j as usize];
        }
        Some(result / h.powi(self.deriv_order as i32))
    }
}

impl fmt::Display for Stencil {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stencil(d{}/dx{}, O(h^{}), {} points)",
            self.deriv_order,
            self.deriv_order,
            self.accuracy_order,
            self.offsets.len()
        )
    }
}

// ── Error order analysis ─────────────────────────────────────────

/// Estimate the order of convergence of a finite difference formula.
/// Computes the approximation at h, h/2, h/4, and estimates the order
/// from the ratio of successive errors.
pub fn estimate_error_order(
    f: fn(f64) -> f64,
    exact_deriv: f64,
    x: f64,
    h: f64,
    method: DiffMethod,
) -> ErrorOrderResult {
    let d1 = apply_method(f, x, h, method);
    let d2 = apply_method(f, x, h / 2.0, method);
    let d3 = apply_method(f, x, h / 4.0, method);

    let e1 = (d1 - exact_deriv).abs();
    let e2 = (d2 - exact_deriv).abs();
    let e3 = (d3 - exact_deriv).abs();

    let order = if e2 > 1e-15 && e3 > 1e-15 {
        (e1 / e2).ln() / 2.0_f64.ln()
    } else {
        f64::NAN
    };

    let order2 = if e2 > 1e-15 && e3 > 1e-15 {
        (e2 / e3).ln() / 2.0_f64.ln()
    } else {
        f64::NAN
    };

    ErrorOrderResult {
        errors: vec![e1, e2, e3],
        step_sizes: vec![h, h / 2.0, h / 4.0],
        estimated_order: (order + order2) / 2.0,
    }
}

fn apply_method(f: fn(f64) -> f64, x: f64, h: f64, method: DiffMethod) -> f64 {
    match method {
        DiffMethod::Forward => forward_diff(f, x, h),
        DiffMethod::Backward => backward_diff(f, x, h),
        DiffMethod::Central => central_diff(f, x, h),
        DiffMethod::Central4th => central_diff_4th(f, x, h),
    }
}

/// Available finite difference methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffMethod {
    Forward,
    Backward,
    Central,
    Central4th,
}

/// Result of error order estimation.
#[derive(Debug, Clone)]
pub struct ErrorOrderResult {
    pub errors: Vec<f64>,
    pub step_sizes: Vec<f64>,
    pub estimated_order: f64,
}

impl fmt::Display for ErrorOrderResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "order~{:.2}", self.estimated_order)
    }
}

// ── Partial derivatives ──────────────────────────────────────────

/// Partial derivative ∂f/∂x_i using central differences.
pub fn partial_derivative(
    f: fn(&[f64]) -> f64,
    x: &[f64],
    i: usize,
    h: f64,
) -> f64 {
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    x_plus[i] = x[i] + h;
    x_minus[i] = x[i] - h;
    (f(&x_plus) - f(&x_minus)) / (2.0 * h)
}

/// Mixed partial derivative ∂²f/∂x_i∂x_j.
pub fn mixed_partial(
    f: fn(&[f64]) -> f64,
    x: &[f64],
    i: usize,
    j: usize,
    h: f64,
) -> f64 {
    let mut xpp = x.to_vec();
    let mut xpm = x.to_vec();
    let mut xmp = x.to_vec();
    let mut xmm = x.to_vec();

    xpp[i] = x[i] + h;
    xpp[j] = x[j] + h;
    xpm[i] = x[i] + h;
    xpm[j] = x[j] - h;
    xmp[i] = x[i] - h;
    xmp[j] = x[j] + h;
    xmm[i] = x[i] - h;
    xmm[j] = x[j] - h;

    (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h * h)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn f_x2(x: f64) -> f64 {
        x * x
    }

    fn f_sin(x: f64) -> f64 {
        x.sin()
    }

    fn f_exp(x: f64) -> f64 {
        x.exp()
    }

    fn f_multi(x: &[f64]) -> f64 {
        x[0] * x[0] + 2.0 * x[0] * x[1] + x[1] * x[1]
    }

    #[test]
    fn test_forward_diff() {
        let d = forward_diff(f_x2, 3.0, 1e-5);
        assert!(approx_eq(d, 6.0, 1e-4));
    }

    #[test]
    fn test_backward_diff() {
        let d = backward_diff(f_x2, 3.0, 1e-5);
        assert!(approx_eq(d, 6.0, 1e-4));
    }

    #[test]
    fn test_central_diff() {
        let d = central_diff(f_x2, 3.0, 1e-5);
        assert!(approx_eq(d, 6.0, 1e-8));
    }

    #[test]
    fn test_central_diff_sin() {
        // d/dx sin(x) = cos(x)
        let x = 1.0;
        let d = central_diff(f_sin, x, 1e-6);
        assert!(approx_eq(d, x.cos(), 1e-8));
    }

    #[test]
    fn test_second_central_diff() {
        // d²/dx² x² = 2
        let d = second_central_diff(f_x2, 3.0, 1e-4);
        assert!(approx_eq(d, 2.0, 1e-4));
    }

    #[test]
    fn test_central_diff_4th() {
        let x = 1.0;
        let d = central_diff_4th(f_sin, x, 1e-3);
        assert!(approx_eq(d, x.cos(), 1e-8));
    }

    #[test]
    fn test_gradient_2d() {
        // f(x,y) = x² + 2xy + y²
        // grad = (2x + 2y, 2x + 2y)
        let x = vec![1.0, 2.0];
        let g = gradient(f_multi, &x, 1e-6);
        assert!(approx_eq(g[0], 6.0, 1e-4));
        assert!(approx_eq(g[1], 6.0, 1e-4));
    }

    #[test]
    fn test_gradient_forward() {
        let x = vec![1.0, 2.0];
        let g = gradient_forward(f_multi, &x, 1e-6);
        assert!(approx_eq(g[0], 6.0, 1e-3));
        assert!(approx_eq(g[1], 6.0, 1e-3));
    }

    #[test]
    fn test_hessian_2d() {
        // f(x,y) = x² + 2xy + y²
        // H = [[2, 2], [2, 2]]
        let x = vec![1.0, 2.0];
        let h = hessian(f_multi, &x, 1e-4);
        assert!(approx_eq(h[0][0], 2.0, 1e-3));
        assert!(approx_eq(h[0][1], 2.0, 1e-3));
        assert!(approx_eq(h[1][0], 2.0, 1e-3));
        assert!(approx_eq(h[1][1], 2.0, 1e-3));
    }

    #[test]
    fn test_jacobian() {
        // F(x,y) = (x²+y, xy)
        // J = [[2x, 1], [y, x]]
        fn f_vec(x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1], x[0] * x[1]]
        }
        let x = vec![2.0, 3.0];
        let j = jacobian(f_vec, &x, 1e-6);
        assert!(approx_eq(j[0][0], 4.0, 1e-4)); // ∂F0/∂x = 2x = 4
        assert!(approx_eq(j[0][1], 1.0, 1e-4)); // ∂F0/∂y = 1
        assert!(approx_eq(j[1][0], 3.0, 1e-4)); // ∂F1/∂x = y = 3
        assert!(approx_eq(j[1][1], 2.0, 1e-4)); // ∂F1/∂y = x = 2
    }

    #[test]
    fn test_jacobian_forward() {
        fn f_vec(x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1], x[0] * x[1]]
        }
        let x = vec![2.0, 3.0];
        let j = jacobian_forward(f_vec, &x, 1e-6);
        assert!(approx_eq(j[0][0], 4.0, 1e-3));
    }

    #[test]
    fn test_richardson_extrapolation() {
        // Central diff of sin at x=1 should give cos(1)
        fn central_at_h(h: f64) -> f64 {
            central_diff(f_sin, 1.0, h)
        }
        let result = richardson_extrapolation(central_at_h, 0.1, 2.0, 2);
        assert!(approx_eq(result, 1.0_f64.cos(), 1e-6));
    }

    #[test]
    fn test_richardson_table() {
        fn forward_at_h(h: f64) -> f64 {
            forward_diff(f_exp, 0.0, h)
        }
        let table = richardson_table(forward_at_h, 0.1, 2.0, 4);
        assert_eq!(table.len(), 4);
        // Best estimate should be close to e^0 = 1
        assert!(approx_eq(table[3][3], 1.0, 1e-4));
    }

    #[test]
    fn test_stencil_first_deriv() {
        let s = Stencil::get(1, 2).unwrap();
        assert_eq!(s.offsets.len(), 3);
        // Apply to data for f(x) = x² at x=2 (sampled at 1,2,3)
        let data = vec![1.0, 4.0, 9.0]; // x² at x=1,2,3
        let d = s.apply(&data, 1, 1.0).unwrap(); // derivative at index 1 (x=2)
        assert!(approx_eq(d, 4.0, 1e-10)); // d/dx x² at x=2 = 4
    }

    #[test]
    fn test_stencil_second_deriv() {
        let s = Stencil::get(2, 2).unwrap();
        let data = vec![1.0, 4.0, 9.0]; // x² at x=1,2,3
        let d = s.apply(&data, 1, 1.0).unwrap();
        assert!(approx_eq(d, 2.0, 1e-10)); // d²/dx² x² = 2
    }

    #[test]
    fn test_stencil_out_of_bounds() {
        let s = Stencil::get(1, 2).unwrap();
        let data = vec![1.0, 4.0, 9.0];
        assert!(s.apply(&data, 0, 1.0).is_none()); // needs index -1
    }

    #[test]
    fn test_stencil_display() {
        let s = Stencil::get(1, 4).unwrap();
        let display = format!("{}", s);
        assert!(display.contains("O(h^4)"));
    }

    #[test]
    fn test_error_order_forward() {
        let result = estimate_error_order(f_sin, 1.0_f64.cos(), 1.0, 0.1, DiffMethod::Forward);
        // Forward diff is O(h), so order ~1
        assert!(approx_eq(result.estimated_order, 1.0, 0.3));
    }

    #[test]
    fn test_error_order_central() {
        let result = estimate_error_order(f_sin, 1.0_f64.cos(), 1.0, 0.1, DiffMethod::Central);
        // Central diff is O(h²), so order ~2
        assert!(approx_eq(result.estimated_order, 2.0, 0.3));
    }

    #[test]
    fn test_partial_derivative() {
        let x = vec![1.0, 2.0];
        let df_dx = partial_derivative(f_multi, &x, 0, 1e-6);
        assert!(approx_eq(df_dx, 6.0, 1e-4));
    }

    #[test]
    fn test_mixed_partial() {
        // ∂²f/∂x∂y of (x² + 2xy + y²) = 2
        let x = vec![1.0, 2.0];
        let d = mixed_partial(f_multi, &x, 0, 1, 1e-4);
        assert!(approx_eq(d, 2.0, 1e-3));
    }

    #[test]
    fn test_error_order_display() {
        let result = estimate_error_order(f_sin, 1.0_f64.cos(), 1.0, 0.1, DiffMethod::Central);
        let s = format!("{}", result);
        assert!(s.contains("order~"));
    }
}
