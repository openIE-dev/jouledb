//! Interpolation — linear, cubic spline, Lagrange, Newton divided differences,
//! bilinear 2D, Hermite, nearest neighbor, extrapolation handling.
//!
//! Pure-Rust replacement for D3-interpolate, scipy.interpolate, and similar libraries.

use std::fmt;

// ── Extrapolation handling ───────────────────────────────────────

/// How to handle queries outside the data range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extrapolation {
    /// Clamp to the nearest boundary value.
    Clamp,
    /// Linearly extrapolate from the boundary.
    Linear,
    /// Return NaN for out-of-range queries.
    Nan,
}

// ── Linear interpolation ─────────────────────────────────────────

/// Linear interpolation between two values at parameter t in [0,1].
pub fn lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Inverse lerp: given value v in [a,b], returns t in [0,1].
pub fn inverse_lerp(a: f64, b: f64, v: f64) -> f64 {
    if (b - a).abs() < 1e-15 { 0.0 } else { (v - a) / (b - a) }
}

/// Piecewise linear interpolation on sorted (xs, ys) data.
pub fn linear_interp(xs: &[f64], ys: &[f64], x: f64, extrap: Extrapolation) -> f64 {
    assert_eq!(xs.len(), ys.len());
    assert!(xs.len() >= 2);

    if x < xs[0] {
        return match extrap {
            Extrapolation::Clamp => ys[0],
            Extrapolation::Linear => ys[0] + (ys[1]-ys[0])/(xs[1]-xs[0]) * (x-xs[0]),
            Extrapolation::Nan => f64::NAN,
        };
    }
    let n = xs.len();
    if x > xs[n - 1] {
        return match extrap {
            Extrapolation::Clamp => ys[n-1],
            Extrapolation::Linear => ys[n-1] + (ys[n-1]-ys[n-2])/(xs[n-1]-xs[n-2]) * (x-xs[n-1]),
            Extrapolation::Nan => f64::NAN,
        };
    }

    let i = find_interval(xs, x);
    let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
    lerp(ys[i], ys[i + 1], t)
}

/// Find interval index i such that xs[i] <= x < xs[i+1].
fn find_interval(xs: &[f64], x: f64) -> usize {
    let n = xs.len();
    if x >= xs[n - 1] { return n - 2; }
    match xs.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
        Ok(i) => if i >= n - 1 { n - 2 } else { i },
        Err(i) => if i == 0 { 0 } else { i - 1 },
    }
}

// ── Cubic spline (natural) ───────────────────────────────────────

/// Natural cubic spline interpolant.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    xs: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl CubicSpline {
    /// Build a natural cubic spline from sorted data.
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        assert_eq!(n, ys.len());
        assert!(n >= 2);

        if n == 2 {
            let slope = (ys[1] - ys[0]) / (xs[1] - xs[0]);
            return Self { xs: xs.to_vec(), a: vec![ys[0]], b: vec![slope], c: vec![0.0], d: vec![0.0] };
        }

        let segs = n - 1;
        let mut h = vec![0.0; segs];
        for i in 0..segs { h[i] = xs[i + 1] - xs[i]; }

        let mut alpha = vec![0.0; n];
        for i in 1..segs {
            alpha[i] = 3.0/h[i]*(ys[i+1]-ys[i]) - 3.0/h[i-1]*(ys[i]-ys[i-1]);
        }

        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];
        for i in 1..segs {
            l[i] = 2.0*(xs[i+1]-xs[i-1]) - h[i-1]*mu[i-1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i];
        }

        let mut c_coeff = vec![0.0; n];
        let mut b_coeff = vec![0.0; segs];
        let mut d_coeff = vec![0.0; segs];
        for j in (0..segs).rev() {
            c_coeff[j] = z[j] - mu[j]*c_coeff[j+1];
            b_coeff[j] = (ys[j+1]-ys[j])/h[j] - h[j]*(c_coeff[j+1]+2.0*c_coeff[j])/3.0;
            d_coeff[j] = (c_coeff[j+1]-c_coeff[j])/(3.0*h[j]);
        }

        Self {
            xs: xs.to_vec(), a: ys[..segs].to_vec(),
            b: b_coeff, c: c_coeff[..segs].to_vec(), d: d_coeff,
        }
    }

    /// Evaluate the spline at x.
    pub fn eval(&self, x: f64, extrap: Extrapolation) -> f64 {
        let n = self.xs.len();
        if x < self.xs[0] {
            return match extrap {
                Extrapolation::Clamp => self.a[0],
                Extrapolation::Linear => self.a[0] + self.b[0]*(x - self.xs[0]),
                Extrapolation::Nan => f64::NAN,
            };
        }
        if x > self.xs[n - 1] {
            let last = self.a.len() - 1;
            let dx = self.xs[n-1] - self.xs[n-2];
            let y_end = self.a[last]+self.b[last]*dx+self.c[last]*dx*dx+self.d[last]*dx*dx*dx;
            return match extrap {
                Extrapolation::Clamp => y_end,
                Extrapolation::Linear => {
                    let slope = self.b[last]+2.0*self.c[last]*dx+3.0*self.d[last]*dx*dx;
                    y_end + slope*(x - self.xs[n-1])
                }
                Extrapolation::Nan => f64::NAN,
            };
        }

        let i = find_interval(&self.xs, x);
        let dx = x - self.xs[i];
        self.a[i] + self.b[i]*dx + self.c[i]*dx*dx + self.d[i]*dx*dx*dx
    }

    /// Evaluate derivative at x.
    pub fn eval_deriv(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let x_clamped = x.clamp(self.xs[0], self.xs[n - 1]);
        let i = find_interval(&self.xs, x_clamped);
        let dx = x_clamped - self.xs[i];
        self.b[i] + 2.0*self.c[i]*dx + 3.0*self.d[i]*dx*dx
    }

    pub fn segments(&self) -> usize { self.a.len() }
}

// ── Lagrange polynomial ──────────────────────────────────────────

/// Lagrange polynomial interpolation at a single point x.
pub fn lagrange(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    assert_eq!(n, ys.len());
    assert!(n >= 1);
    let mut result = 0.0;
    for i in 0..n {
        let mut basis = 1.0;
        for j in 0..n {
            if i != j { basis *= (x - xs[j]) / (xs[i] - xs[j]); }
        }
        result += ys[i] * basis;
    }
    result
}

pub fn lagrange_multi(xs: &[f64], ys: &[f64], eval_pts: &[f64]) -> Vec<f64> {
    eval_pts.iter().map(|x| lagrange(xs, ys, *x)).collect()
}

// ── Newton divided differences ───────────────────────────────────

/// Newton's divided-difference interpolation.
#[derive(Debug, Clone)]
pub struct NewtonInterp {
    xs: Vec<f64>,
    coeffs: Vec<f64>,
}

impl NewtonInterp {
    /// Build Newton interpolant from data points.
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        assert_eq!(n, ys.len());
        assert!(n >= 1);

        // Build divided difference table
        let mut dd = ys.to_vec();
        let mut coeffs = vec![dd[0]];

        for j in 1..n {
            for i in (j..n).rev() {
                dd[i] = (dd[i] - dd[i - 1]) / (xs[i] - xs[i - j]);
            }
            coeffs.push(dd[j]);
        }

        Self { xs: xs.to_vec(), coeffs }
    }

    /// Evaluate the Newton polynomial at x using Horner's method.
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.coeffs.len();
        let mut result = self.coeffs[n - 1];
        for i in (0..n - 1).rev() {
            result = result * (x - self.xs[i]) + self.coeffs[i];
        }
        result
    }

    /// Degree of the interpolating polynomial.
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() { 0 } else { self.coeffs.len() - 1 }
    }
}

// ── Bilinear interpolation (2D grid) ─────────────────────────────

/// 2D grid for bilinear interpolation.
#[derive(Debug, Clone)]
pub struct BilinearGrid {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub values: Vec<f64>,
}

impl BilinearGrid {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), xs.len() * ys.len());
        Self { xs, ys, values }
    }

    pub fn eval(&self, x: f64, y: f64, extrap: Extrapolation) -> f64 {
        let nx = self.xs.len();
        let ny = self.ys.len();

        let x_clamped = match extrap {
            Extrapolation::Clamp => x.clamp(self.xs[0], self.xs[nx-1]),
            Extrapolation::Nan => {
                if x < self.xs[0] || x > self.xs[nx-1] { return f64::NAN; }
                x
            }
            Extrapolation::Linear => x,
        };
        let y_clamped = match extrap {
            Extrapolation::Clamp => y.clamp(self.ys[0], self.ys[ny-1]),
            Extrapolation::Nan => {
                if y < self.ys[0] || y > self.ys[ny-1] { return f64::NAN; }
                y
            }
            Extrapolation::Linear => y,
        };

        let xc = x_clamped.clamp(self.xs[0], self.xs[nx-1]);
        let yc = y_clamped.clamp(self.ys[0], self.ys[ny-1]);

        let ix = find_interval(&self.xs, xc);
        let iy = find_interval(&self.ys, yc);

        let tx = if (self.xs[ix+1]-self.xs[ix]).abs() < 1e-15 { 0.0 }
                 else { (xc-self.xs[ix])/(self.xs[ix+1]-self.xs[ix]) };
        let ty = if (self.ys[iy+1]-self.ys[iy]).abs() < 1e-15 { 0.0 }
                 else { (yc-self.ys[iy])/(self.ys[iy+1]-self.ys[iy]) };

        let c00 = self.values[iy*nx+ix];
        let c10 = self.values[iy*nx+ix+1];
        let c01 = self.values[(iy+1)*nx+ix];
        let c11 = self.values[(iy+1)*nx+ix+1];

        lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty)
    }
}

// ── Nearest neighbor ─────────────────────────────────────────────

pub fn nearest_neighbor(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    assert_eq!(xs.len(), ys.len());
    assert!(!xs.is_empty());
    let mut best_i = 0;
    let mut best_d = (x - xs[0]).abs();
    for i in 1..xs.len() {
        let d = (x - xs[i]).abs();
        if d < best_d { best_d = d; best_i = i; }
    }
    ys[best_i]
}

pub fn nearest_neighbor_multi(xs: &[f64], ys: &[f64], pts: &[f64]) -> Vec<f64> {
    pts.iter().map(|x| nearest_neighbor(xs, ys, *x)).collect()
}

// ── Hermite interpolation ────────────────────────────────────────

/// Cubic Hermite interpolation between two points with given tangents.
pub fn hermite(p0: f64, m0: f64, p1: f64, m1: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0*t3-3.0*t2+1.0)*p0 + (t3-2.0*t2+t)*m0 + (-2.0*t3+3.0*t2)*p1 + (t3-t2)*m1
}

/// Piecewise Hermite interpolation with specified tangents.
#[derive(Debug, Clone)]
pub struct HermiteSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    tangents: Vec<f64>,
}

impl HermiteSpline {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, tangents: Vec<f64>) -> Self {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(xs.len(), tangents.len());
        assert!(xs.len() >= 2);
        Self { xs, ys, tangents }
    }

    /// Create with Catmull-Rom tangent estimation.
    pub fn catmull_rom(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        let n = xs.len();
        assert!(n >= 2);
        let mut tangents = vec![0.0; n];
        if n == 2 {
            let s = (ys[1]-ys[0])/(xs[1]-xs[0]);
            tangents[0] = s; tangents[1] = s;
        } else {
            tangents[0] = (ys[1]-ys[0])/(xs[1]-xs[0]);
            tangents[n-1] = (ys[n-1]-ys[n-2])/(xs[n-1]-xs[n-2]);
            for i in 1..n-1 { tangents[i] = (ys[i+1]-ys[i-1])/(xs[i+1]-xs[i-1]); }
        }
        Self { xs, ys, tangents }
    }

    pub fn eval(&self, x: f64, extrap: Extrapolation) -> f64 {
        let n = self.xs.len();
        if x < self.xs[0] {
            return match extrap {
                Extrapolation::Clamp => self.ys[0],
                Extrapolation::Linear => self.ys[0]+self.tangents[0]*(x-self.xs[0]),
                Extrapolation::Nan => f64::NAN,
            };
        }
        if x > self.xs[n-1] {
            return match extrap {
                Extrapolation::Clamp => self.ys[n-1],
                Extrapolation::Linear => self.ys[n-1]+self.tangents[n-1]*(x-self.xs[n-1]),
                Extrapolation::Nan => f64::NAN,
            };
        }
        let i = find_interval(&self.xs, x);
        let h = self.xs[i+1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        hermite(self.ys[i], self.tangents[i]*h, self.ys[i+1], self.tangents[i+1]*h, t)
    }
}

// ── Interpolation error estimation ───────────────────────────────

/// Available interpolation methods for error estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    Linear,
    CubicSpline,
    Lagrange,
    NearestNeighbor,
}

impl fmt::Display for InterpolationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear => write!(f, "Linear"),
            Self::CubicSpline => write!(f, "CubicSpline"),
            Self::Lagrange => write!(f, "Lagrange"),
            Self::NearestNeighbor => write!(f, "NearestNeighbor"),
        }
    }
}

/// Leave-one-out cross-validation error estimate.
pub fn leave_one_out_error(xs: &[f64], ys: &[f64], method: InterpolationMethod) -> f64 {
    let n = xs.len();
    assert!(n >= 3);
    let mut total_err = 0.0;
    for skip in 0..n {
        let mut xs_sub = Vec::with_capacity(n-1);
        let mut ys_sub = Vec::with_capacity(n-1);
        for j in 0..n {
            if j != skip { xs_sub.push(xs[j]); ys_sub.push(ys[j]); }
        }
        let predicted = match method {
            InterpolationMethod::Linear => linear_interp(&xs_sub, &ys_sub, xs[skip], Extrapolation::Linear),
            InterpolationMethod::CubicSpline => CubicSpline::new(&xs_sub, &ys_sub).eval(xs[skip], Extrapolation::Linear),
            InterpolationMethod::Lagrange => lagrange(&xs_sub, &ys_sub, xs[skip]),
            InterpolationMethod::NearestNeighbor => nearest_neighbor(&xs_sub, &ys_sub, xs[skip]),
        };
        total_err += (predicted - ys[skip]).abs();
    }
    total_err / n as f64
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    #[test]
    fn test_lerp() {
        assert!(approx_eq(lerp(0.0, 10.0, 0.5), 5.0, 1e-12));
        assert!(approx_eq(lerp(0.0, 10.0, 0.0), 0.0, 1e-12));
        assert!(approx_eq(lerp(0.0, 10.0, 1.0), 10.0, 1e-12));
    }

    #[test]
    fn test_inverse_lerp() {
        assert!(approx_eq(inverse_lerp(0.0, 10.0, 5.0), 0.5, 1e-12));
    }

    #[test]
    fn test_linear_interp_basic() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 4.0, 9.0];
        assert!(approx_eq(linear_interp(&xs, &ys, 0.5, Extrapolation::Clamp), 0.5, 1e-12));
        assert!(approx_eq(linear_interp(&xs, &ys, 1.5, Extrapolation::Clamp), 2.5, 1e-12));
    }

    #[test]
    fn test_linear_interp_exact() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 2.0, 8.0];
        assert!(approx_eq(linear_interp(&xs, &ys, 1.0, Extrapolation::Clamp), 2.0, 1e-12));
    }

    #[test]
    fn test_linear_extrap_clamp() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        assert!(approx_eq(linear_interp(&xs, &ys, -1.0, Extrapolation::Clamp), 0.0, 1e-12));
        assert!(approx_eq(linear_interp(&xs, &ys, 5.0, Extrapolation::Clamp), 2.0, 1e-12));
    }

    #[test]
    fn test_linear_extrap_linear() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        assert!(approx_eq(linear_interp(&xs, &ys, -1.0, Extrapolation::Linear), -1.0, 1e-12));
        assert!(approx_eq(linear_interp(&xs, &ys, 3.0, Extrapolation::Linear), 3.0, 1e-12));
    }

    #[test]
    fn test_linear_extrap_nan() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        assert!(linear_interp(&xs, &ys, -0.1, Extrapolation::Nan).is_nan());
    }

    #[test]
    fn test_cubic_spline_exact_pts() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 0.0, 1.0];
        let spline = CubicSpline::new(&xs, &ys);
        for i in 0..xs.len()-1 {
            assert!(approx_eq(spline.eval(xs[i], Extrapolation::Clamp), ys[i], 1e-10));
        }
    }

    #[test]
    fn test_cubic_spline_smooth() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|x| x*x).collect();
        let spline = CubicSpline::new(&xs, &ys);
        assert!(approx_eq(spline.eval(0.5, Extrapolation::Clamp), 0.25, 0.15));
        assert!(approx_eq(spline.eval(2.5, Extrapolation::Clamp), 6.25, 0.1));
    }

    #[test]
    fn test_spline_derivative() {
        // y = x^2, derivative = 2x
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|x| x*x).collect();
        let spline = CubicSpline::new(&xs, &ys);
        assert!(approx_eq(spline.eval_deriv(2.0), 4.0, 0.3));
    }

    #[test]
    fn test_cubic_spline_two_pts() {
        let spline = CubicSpline::new(&[0.0, 1.0], &[0.0, 1.0]);
        assert!(approx_eq(spline.eval(0.5, Extrapolation::Clamp), 0.5, 1e-12));
    }

    #[test]
    fn test_lagrange_quadratic() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        assert!(approx_eq(lagrange(&xs, &ys, 0.5), 0.25, 1e-12));
        assert!(approx_eq(lagrange(&xs, &ys, 1.5), 2.25, 1e-12));
        assert!(approx_eq(lagrange(&xs, &ys, 3.0), 9.0, 1e-12));
    }

    #[test]
    fn test_lagrange_multi() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let r = lagrange_multi(&xs, &ys, &[0.25, 0.5, 0.75]);
        assert!(approx_eq(r[1], 0.5, 1e-12));
    }

    #[test]
    fn test_newton_interp_quadratic() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        let ni = NewtonInterp::new(&xs, &ys);
        assert_eq!(ni.degree(), 2);
        assert!(approx_eq(ni.eval(0.5), 0.25, 1e-12));
        assert!(approx_eq(ni.eval(1.5), 2.25, 1e-12));
        assert!(approx_eq(ni.eval(3.0), 9.0, 1e-12));
    }

    #[test]
    fn test_newton_linear() {
        let ni = NewtonInterp::new(&[0.0, 2.0], &[1.0, 5.0]);
        assert_eq!(ni.degree(), 1);
        assert!(approx_eq(ni.eval(1.0), 3.0, 1e-12));
    }

    #[test]
    fn test_newton_agrees_with_lagrange() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let ni = NewtonInterp::new(&xs, &ys);
        for &x in &[0.5, 1.5, 2.5, 3.5] {
            let newton_val = ni.eval(x);
            let lagrange_val = lagrange(&xs, &ys, x);
            assert!(approx_eq(newton_val, lagrange_val, 1e-10));
        }
    }

    #[test]
    fn test_bilinear_grid() {
        let grid = BilinearGrid::new(vec![0.0,1.0], vec![0.0,1.0], vec![0.0,1.0,1.0,2.0]);
        assert!(approx_eq(grid.eval(0.5, 0.5, Extrapolation::Clamp), 1.0, 1e-12));
        assert!(approx_eq(grid.eval(0.0, 0.0, Extrapolation::Clamp), 0.0, 1e-12));
        assert!(approx_eq(grid.eval(1.0, 1.0, Extrapolation::Clamp), 2.0, 1e-12));
    }

    #[test]
    fn test_bilinear_clamp() {
        let grid = BilinearGrid::new(vec![0.0,1.0], vec![0.0,1.0], vec![0.0,1.0,1.0,2.0]);
        let v = grid.eval(-1.0, 0.5, Extrapolation::Clamp);
        assert!(v.is_finite());
    }

    #[test]
    fn test_bilinear_nan() {
        let grid = BilinearGrid::new(vec![0.0,1.0], vec![0.0,1.0], vec![0.0,1.0,1.0,2.0]);
        assert!(grid.eval(-0.1, 0.5, Extrapolation::Nan).is_nan());
    }

    #[test]
    fn test_nearest_neighbor() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0];
        assert!(approx_eq(nearest_neighbor(&xs, &ys, 0.3), 10.0, 1e-12));
        assert!(approx_eq(nearest_neighbor(&xs, &ys, 0.6), 20.0, 1e-12));
    }

    #[test]
    fn test_hermite_basic() {
        assert!(approx_eq(hermite(0.0, 0.0, 1.0, 0.0, 0.5), 0.5, 1e-12));
    }

    #[test]
    fn test_hermite_spline() {
        let spline = HermiteSpline::new(
            vec![0.0,1.0,2.0,3.0], vec![0.0,1.0,0.0,1.0], vec![1.0,0.0,-1.0,0.0],
        );
        assert!(approx_eq(spline.eval(0.0, Extrapolation::Clamp), 0.0, 1e-10));
        assert!(approx_eq(spline.eval(1.0, Extrapolation::Clamp), 1.0, 1e-10));
    }

    #[test]
    fn test_catmull_rom() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| x*x).collect();
        let spline = HermiteSpline::catmull_rom(xs.clone(), ys.clone());
        for i in 0..xs.len() {
            assert!(approx_eq(spline.eval(xs[i], Extrapolation::Clamp), ys[i], 1e-10));
        }
    }

    #[test]
    fn test_leave_one_out() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.0*x+1.0).collect();
        let err = leave_one_out_error(&xs, &ys, InterpolationMethod::Linear);
        assert!(err < 1e-10);
    }

    #[test]
    fn test_method_display() {
        assert_eq!(format!("{}", InterpolationMethod::Lagrange), "Lagrange");
    }

    #[test]
    fn test_spline_segments() {
        let spline = CubicSpline::new(&[0.0,1.0,2.0,3.0], &[0.0,1.0,0.0,1.0]);
        assert_eq!(spline.segments(), 3);
    }
}
