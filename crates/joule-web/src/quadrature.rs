//! Numerical integration (quadrature) — trapezoidal rule, Simpson's rule,
//! Gaussian quadrature, adaptive quadrature, Romberg, double integration,
//! improper integrals.
//!
//! Pure-Rust numerical integration routines.

use std::fmt;

// ── Result type ──────────────────────────────────────────────────

/// Result of a numerical integration.
#[derive(Debug, Clone)]
pub struct QuadratureResult {
    /// Computed integral value.
    pub value: f64,
    /// Estimated absolute error (if available).
    pub error_estimate: Option<f64>,
    /// Number of function evaluations.
    pub fn_evals: usize,
    /// Number of subdivisions or refinements.
    pub subdivisions: usize,
}

impl fmt::Display for QuadratureResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.error_estimate {
            Some(e) => write!(
                f,
                "integral={:.10}, error~{:.2e}, evals={}",
                self.value, e, self.fn_evals
            ),
            None => write!(
                f,
                "integral={:.10}, evals={}",
                self.value, self.fn_evals
            ),
        }
    }
}

// ── Trapezoidal rule ─────────────────────────────────────────────

/// Composite trapezoidal rule with n subintervals.
pub fn trapezoidal(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> QuadratureResult {
    assert!(n >= 1, "need at least 1 subinterval");
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    QuadratureResult {
        value: sum * h,
        error_estimate: None,
        fn_evals: n + 1,
        subdivisions: n,
    }
}

/// Trapezoidal rule with error estimate by comparing n and 2n subintervals.
pub fn trapezoidal_with_error(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> QuadratureResult {
    let r1 = trapezoidal(f, a, b, n);
    let r2 = trapezoidal(f, a, b, 2 * n);
    let err = (r2.value - r1.value).abs() / 3.0; // Richardson extrapolation O(h^2)
    QuadratureResult {
        value: r2.value,
        error_estimate: Some(err),
        fn_evals: r1.fn_evals + r2.fn_evals,
        subdivisions: 2 * n,
    }
}

// ── Simpson's rule ───────────────────────────────────────────────

/// Composite Simpson's 1/3 rule. n must be even.
pub fn simpson(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> QuadratureResult {
    let n = if n % 2 == 1 { n + 1 } else { n };
    assert!(n >= 2, "need at least 2 subintervals");
    let h = (b - a) / n as f64;

    let mut sum = f(a) + f(b);
    let mut comp = 0.0;
    for i in 1..n {
        let x = a + i as f64 * h;
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        let term = coeff * f(x) - comp;
        let new_sum = sum + term;
        comp = (new_sum - sum) - term;
        sum = new_sum;
    }

    QuadratureResult {
        value: sum * h / 3.0,
        error_estimate: None,
        fn_evals: n + 1,
        subdivisions: n,
    }
}

/// Simpson's rule with error estimate by comparing n and 2n.
pub fn simpson_with_error(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> QuadratureResult {
    let r1 = simpson(f, a, b, n);
    let r2 = simpson(f, a, b, 2 * n);
    let err = (r2.value - r1.value).abs() / 15.0; // Simpson is O(h^4)
    QuadratureResult {
        value: r2.value,
        error_estimate: Some(err),
        fn_evals: r1.fn_evals + r2.fn_evals,
        subdivisions: 2 * n,
    }
}

// ── Gaussian quadrature (Legendre) ───────────────────────────────

/// Gauss-Legendre quadrature nodes and weights for n points on [-1, 1].
pub fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => {
            let x = 1.0 / 3.0_f64.sqrt();
            (vec![-x, x], vec![1.0, 1.0])
        }
        3 => (
            vec![-(3.0 / 5.0_f64).sqrt(), 0.0, (3.0 / 5.0_f64).sqrt()],
            vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
        ),
        4 => {
            let x1 = ((3.0 - 2.0 * (6.0 / 5.0_f64).sqrt()) / 7.0).sqrt();
            let x2 = ((3.0 + 2.0 * (6.0 / 5.0_f64).sqrt()) / 7.0).sqrt();
            let w1 = (18.0 + 30.0_f64.sqrt()) / 36.0;
            let w2 = (18.0 - 30.0_f64.sqrt()) / 36.0;
            (vec![-x2, -x1, x1, x2], vec![w2, w1, w1, w2])
        }
        5 => {
            let x1 = (5.0 - 2.0 * (10.0 / 7.0_f64).sqrt()).sqrt() / 3.0;
            let x2 = (5.0 + 2.0 * (10.0 / 7.0_f64).sqrt()).sqrt() / 3.0;
            let w0 = 128.0 / 225.0;
            let w1 = (322.0 + 13.0 * 70.0_f64.sqrt()) / 900.0;
            let w2 = (322.0 - 13.0 * 70.0_f64.sqrt()) / 900.0;
            (vec![-x2, -x1, 0.0, x1, x2], vec![w2, w1, w0, w1, w2])
        }
        _ => {
            // Compute nodes/weights numerically using Newton's method on Legendre polynomials
            compute_gauss_legendre(n)
        }
    }
}

fn compute_gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    let m = (n + 1) / 2;
    for i in 0..m {
        // Initial guess
        let mut x = ((i as f64 + 0.75) / (n as f64 + 0.5) * std::f64::consts::PI).cos();

        for _ in 0..100 {
            // Evaluate Legendre polynomial and derivative
            let mut p0 = 1.0;
            let mut p1 = x;
            for j in 2..=n {
                let p2 = ((2 * j - 1) as f64 * x * p1 - (j - 1) as f64 * p0) / j as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (p0 - x * p1) / (1.0 - x * x);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }

        // Evaluate polynomial value for weight computation
        let mut p0 = 1.0;
        let mut p1 = x;
        for j in 2..=n {
            let p2 = ((2 * j - 1) as f64 * x * p1 - (j - 1) as f64 * p0) / j as f64;
            p0 = p1;
            p1 = p2;
        }
        let dp = n as f64 * (p0 - x * p1) / (1.0 - x * x);
        let w = 2.0 / ((1.0 - x * x) * dp * dp);

        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    (nodes, weights)
}

/// Gauss-Legendre quadrature on [a, b] with n nodes.
pub fn gauss_legendre(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> QuadratureResult {
    let (nodes, weights) = gauss_legendre_nodes_weights(n);

    // Transform from [-1, 1] to [a, b]
    let mid = (a + b) / 2.0;
    let half = (b - a) / 2.0;

    let mut sum = 0.0;
    for i in 0..n {
        let x = mid + half * nodes[i];
        sum += weights[i] * f(x);
    }

    QuadratureResult {
        value: sum * half,
        error_estimate: None,
        fn_evals: n,
        subdivisions: 1,
    }
}

/// Composite Gauss-Legendre: divide [a,b] into panels, apply GL to each.
pub fn gauss_legendre_composite(
    f: fn(f64) -> f64,
    a: f64,
    b: f64,
    panels: usize,
    nodes_per_panel: usize,
) -> QuadratureResult {
    let h = (b - a) / panels as f64;
    let mut total = 0.0;
    let mut evals = 0;

    let (nodes, weights) = gauss_legendre_nodes_weights(nodes_per_panel);

    for p in 0..panels {
        let pa = a + p as f64 * h;
        let pb = pa + h;
        let mid = (pa + pb) / 2.0;
        let half = (pb - pa) / 2.0;

        for i in 0..nodes_per_panel {
            let x = mid + half * nodes[i];
            total += weights[i] * f(x) * half;
            evals += 1;
        }
    }

    QuadratureResult {
        value: total,
        error_estimate: None,
        fn_evals: evals,
        subdivisions: panels,
    }
}

// ── Adaptive quadrature (recursive Simpson) ──────────────────────

/// Adaptive Simpson's quadrature. Subdivides until error < tol.
pub fn adaptive_simpson(
    f: fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> QuadratureResult {
    let mut evals = 0usize;
    let mut subdivisions = 0usize;

    let value = adaptive_simpson_recursive(f, a, b, tol, max_depth, 0, &mut evals, &mut subdivisions);

    QuadratureResult {
        value,
        error_estimate: Some(tol),
        fn_evals: evals,
        subdivisions,
    }
}

fn adaptive_simpson_recursive(
    f: fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
    depth: usize,
    evals: &mut usize,
    subdivisions: &mut usize,
) -> f64 {
    let mid = (a + b) / 2.0;
    let h = b - a;

    let fa = f(a);
    let fb = f(b);
    let fm = f(mid);
    *evals += 3;

    let whole = h / 6.0 * (fa + 4.0 * fm + fb);

    let fl = f((a + mid) / 2.0);
    let fr = f((mid + b) / 2.0);
    *evals += 2;

    let left = h / 12.0 * (fa + 4.0 * fl + fm);
    let right = h / 12.0 * (fm + 4.0 * fr + fb);
    let combined = left + right;

    let err = (combined - whole).abs() / 15.0;

    if depth >= max_depth || err < tol {
        *subdivisions += 1;
        return combined + (combined - whole) / 15.0; // Richardson extrapolation
    }

    let l = adaptive_simpson_recursive(f, a, mid, tol / 2.0, max_depth, depth + 1, evals, subdivisions);
    let r = adaptive_simpson_recursive(f, mid, b, tol / 2.0, max_depth, depth + 1, evals, subdivisions);
    l + r
}

// ── Romberg integration ──────────────────────────────────────────

/// Romberg integration. Builds a Richardson extrapolation table.
pub fn romberg(f: fn(f64) -> f64, a: f64, b: f64, max_order: usize) -> QuadratureResult {
    assert!(max_order >= 1);

    let mut table = vec![vec![0.0; max_order]; max_order];
    let mut evals = 0usize;

    // T(0,0) = trapezoidal with 1 panel
    table[0][0] = (b - a) / 2.0 * (f(a) + f(b));
    evals += 2;

    for i in 1..max_order {
        let n = 1usize << i; // 2^i panels
        let h = (b - a) / n as f64;

        // Add only NEW midpoints (odd-indexed grid points)
        let n_new = 1usize << (i - 1);
        let mut sum = 0.0;
        for k in 0..n_new {
            let x = a + (2 * k + 1) as f64 * h;
            sum += f(x);
            evals += 1;
        }
        table[i][0] = table[i - 1][0] / 2.0 + h * sum;

        // Richardson extrapolation
        for j in 1..=i {
            let factor = 4.0_f64.powi(j as i32);
            table[i][j] = (factor * table[i][j - 1] - table[i - 1][j - 1]) / (factor - 1.0);
        }
    }

    let k = max_order - 1;
    let best = table[k][k];
    let error = if k > 0 {
        Some((table[k][k] - table[k][k - 1]).abs())
    } else {
        None
    };

    QuadratureResult {
        value: best,
        error_estimate: error,
        fn_evals: evals,
        subdivisions: 1 << (max_order - 1),
    }
}

// ── Double integration ───────────────────────────────────────────

/// Double integral: ∫∫ f(x,y) dy dx over [xa,xb] x [ya,yb].
/// Uses composite Simpson in both dimensions.
pub fn double_integral(
    f: fn(f64, f64) -> f64,
    xa: f64,
    xb: f64,
    ya: f64,
    yb: f64,
    nx: usize,
    ny: usize,
) -> QuadratureResult {
    let nx = if nx % 2 == 1 { nx + 1 } else { nx };
    let ny = if ny % 2 == 1 { ny + 1 } else { ny };
    let hx = (xb - xa) / nx as f64;
    let hy = (yb - ya) / ny as f64;

    let mut evals = 0usize;

    // Outer Simpson
    let mut total = 0.0;
    for i in 0..=nx {
        let x = xa + i as f64 * hx;
        let wx = if i == 0 || i == nx {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };

        // Inner Simpson
        let mut inner_sum = 0.0;
        for j in 0..=ny {
            let y = ya + j as f64 * hy;
            let wy = if j == 0 || j == ny {
                1.0
            } else if j % 2 == 0 {
                2.0
            } else {
                4.0
            };
            inner_sum += wy * f(x, y);
            evals += 1;
        }
        inner_sum *= hy / 3.0;
        total += wx * inner_sum;
    }
    total *= hx / 3.0;

    QuadratureResult {
        value: total,
        error_estimate: None,
        fn_evals: evals,
        subdivisions: nx * ny,
    }
}

// ── Improper integral handling ───────────────────────────────────

/// Integrate f on [a, infinity) by substitution x = a + t/(1-t), t in [0,1).
/// Uses Gauss-Legendre on [0, 1-eps].
pub fn improper_to_inf(
    f: fn(f64) -> f64,
    a: f64,
    n: usize,
) -> QuadratureResult {
    // Use substitution: x = a + t/(1-t), dx = 1/(1-t)^2 dt
    // Integral becomes ∫₀¹ f(a + t/(1-t)) / (1-t)^2 dt
    let eps = 1e-8;
    let (nodes, weights) = gauss_legendre_nodes_weights(n);

    let mid = (0.0 + (1.0 - eps)) / 2.0;
    let half = (1.0 - eps) / 2.0;

    let mut sum = 0.0;
    for i in 0..n {
        let t = mid + half * nodes[i];
        if t >= 1.0 - 1e-15 {
            continue;
        }
        let one_minus_t = 1.0 - t;
        let x = a + t / one_minus_t;
        let jacobian = 1.0 / (one_minus_t * one_minus_t);
        sum += weights[i] * f(x) * jacobian;
    }

    QuadratureResult {
        value: sum * half,
        error_estimate: None,
        fn_evals: n,
        subdivisions: 1,
    }
}

/// Integrate f on [a, b] where f may have integrable singularity at a.
/// Uses x = a + (b-a)*t^2 substitution.
pub fn improper_singularity_left(
    f: fn(f64) -> f64,
    a: f64,
    b: f64,
    n: usize,
) -> QuadratureResult {
    // Substitution: x = a + (b-a)*t^2, dx = 2*(b-a)*t dt
    let (nodes, weights) = gauss_legendre_nodes_weights(n);
    let mid = 0.5;
    let half = 0.5;

    let ba = b - a;
    let mut sum = 0.0;
    for i in 0..n {
        let t = mid + half * nodes[i];
        let x = a + ba * t * t;
        let jacobian = 2.0 * ba * t;
        sum += weights[i] * f(x) * jacobian;
    }

    QuadratureResult {
        value: sum * half,
        error_estimate: None,
        fn_evals: n,
        subdivisions: 1,
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ∫₀¹ x² dx = 1/3
    fn x_squared(x: f64) -> f64 {
        x * x
    }

    // ∫₀^π sin(x) dx = 2
    fn sin_x(x: f64) -> f64 {
        x.sin()
    }

    // ∫₀¹ exp(x) dx = e - 1
    fn exp_x(x: f64) -> f64 {
        x.exp()
    }

    // ∫₀¹ 4/(1+x²) dx = π
    fn pi_integrand(x: f64) -> f64 {
        4.0 / (1.0 + x * x)
    }

    #[test]
    fn test_trapezoidal_x_squared() {
        let res = trapezoidal(x_squared, 0.0, 1.0, 1000);
        assert!(approx_eq(res.value, 1.0 / 3.0, 1e-6));
    }

    #[test]
    fn test_trapezoidal_sin() {
        let res = trapezoidal(sin_x, 0.0, std::f64::consts::PI, 1000);
        assert!(approx_eq(res.value, 2.0, 1e-5));
    }

    #[test]
    fn test_trapezoidal_with_error() {
        let res = trapezoidal_with_error(x_squared, 0.0, 1.0, 100);
        assert!(approx_eq(res.value, 1.0 / 3.0, 1e-4));
        assert!(res.error_estimate.unwrap() < 1e-4);
    }

    #[test]
    fn test_simpson_x_squared() {
        let res = simpson(x_squared, 0.0, 1.0, 10);
        // Simpson is exact for polynomials up to degree 3
        assert!(approx_eq(res.value, 1.0 / 3.0, 1e-12));
    }

    #[test]
    fn test_simpson_sin() {
        let res = simpson(sin_x, 0.0, std::f64::consts::PI, 100);
        assert!(approx_eq(res.value, 2.0, 1e-7));
    }

    #[test]
    fn test_simpson_with_error() {
        let res = simpson_with_error(exp_x, 0.0, 1.0, 10);
        assert!(approx_eq(res.value, std::f64::consts::E - 1.0, 1e-7));
    }

    #[test]
    fn test_gauss_legendre_1_node() {
        let (nodes, weights) = gauss_legendre_nodes_weights(1);
        assert_eq!(nodes.len(), 1);
        assert!(approx_eq(nodes[0], 0.0, 1e-12));
        assert!(approx_eq(weights[0], 2.0, 1e-12));
    }

    #[test]
    fn test_gauss_legendre_exact_for_poly() {
        // GL with 3 nodes is exact for polynomials up to degree 5
        let res = gauss_legendre(x_squared, 0.0, 1.0, 3);
        assert!(approx_eq(res.value, 1.0 / 3.0, 1e-12));
    }

    #[test]
    fn test_gauss_legendre_pi() {
        let res = gauss_legendre(pi_integrand, 0.0, 1.0, 5);
        assert!(approx_eq(res.value, std::f64::consts::PI, 1e-6));
    }

    #[test]
    fn test_gauss_legendre_composite() {
        let res = gauss_legendre_composite(sin_x, 0.0, std::f64::consts::PI, 10, 3);
        assert!(approx_eq(res.value, 2.0, 1e-8));
    }

    #[test]
    fn test_adaptive_simpson() {
        let res = adaptive_simpson(sin_x, 0.0, std::f64::consts::PI, 1e-10, 20);
        assert!(approx_eq(res.value, 2.0, 1e-8));
    }

    #[test]
    fn test_romberg_basic() {
        let res = romberg(x_squared, 0.0, 1.0, 6);
        assert!(approx_eq(res.value, 1.0 / 3.0, 1e-10));
    }

    #[test]
    fn test_romberg_sin() {
        let res = romberg(sin_x, 0.0, std::f64::consts::PI, 8);
        assert!(approx_eq(res.value, 2.0, 1e-8));
    }

    #[test]
    fn test_romberg_error_estimate() {
        let res = romberg(exp_x, 0.0, 1.0, 6);
        assert!(approx_eq(res.value, std::f64::consts::E - 1.0, 1e-8));
        assert!(res.error_estimate.is_some());
    }

    #[test]
    fn test_double_integral() {
        // ∫₀¹ ∫₀¹ (x + y) dy dx = 1
        fn f_xy(x: f64, y: f64) -> f64 {
            x + y
        }
        let res = double_integral(f_xy, 0.0, 1.0, 0.0, 1.0, 20, 20);
        assert!(approx_eq(res.value, 1.0, 1e-8));
    }

    #[test]
    fn test_double_integral_product() {
        // ∫₀¹ ∫₀¹ x*y dy dx = 1/4
        fn f_xy(x: f64, y: f64) -> f64 {
            x * y
        }
        let res = double_integral(f_xy, 0.0, 1.0, 0.0, 1.0, 10, 10);
        assert!(approx_eq(res.value, 0.25, 1e-8));
    }

    #[test]
    fn test_improper_to_inf() {
        // ∫₁^∞ 1/x² dx = 1
        fn inv_x2(x: f64) -> f64 {
            1.0 / (x * x)
        }
        let res = improper_to_inf(inv_x2, 1.0, 20);
        assert!(approx_eq(res.value, 1.0, 0.05));
    }

    #[test]
    fn test_improper_singularity() {
        // ∫₀¹ 1/sqrt(x) dx = 2
        fn inv_sqrt(x: f64) -> f64 {
            if x < 1e-15 {
                return 0.0;
            }
            1.0 / x.sqrt()
        }
        let res = improper_singularity_left(inv_sqrt, 0.0, 1.0, 20);
        assert!(approx_eq(res.value, 2.0, 0.1));
    }

    #[test]
    fn test_gauss_legendre_many_nodes() {
        let (nodes, weights) = gauss_legendre_nodes_weights(10);
        assert_eq!(nodes.len(), 10);
        assert_eq!(weights.len(), 10);
        // All weights should be positive
        for w in &weights {
            assert!(*w > 0.0);
        }
        // Sum of weights should be 2 (integral of 1 on [-1,1])
        let sum: f64 = weights.iter().sum();
        assert!(approx_eq(sum, 2.0, 1e-12));
    }

    #[test]
    fn test_quadrature_result_display() {
        let res = simpson(x_squared, 0.0, 1.0, 10);
        let s = format!("{}", res);
        assert!(s.contains("integral="));
    }
}
