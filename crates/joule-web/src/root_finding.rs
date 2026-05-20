//! Root finding algorithms — bisection, Newton-Raphson, secant, Brent's method,
//! bracketing, convergence criteria.
//!
//! Pure-Rust numerical root finders for f(x) = 0.

use std::fmt;

// ── Configuration ────────────────────────────────────────────────

/// Convergence criteria for root-finding.
#[derive(Debug, Clone)]
pub struct Convergence {
    pub max_iter: usize,
    pub f_tol: f64,
    pub x_tol: f64,
}

impl Default for Convergence {
    fn default() -> Self { Self { max_iter: 100, f_tol: 1e-12, x_tol: 1e-12 } }
}

/// Result of a root-finding computation.
#[derive(Debug, Clone)]
pub struct RootResult {
    pub root: f64,
    pub f_root: f64,
    pub iterations: usize,
    pub converged: bool,
    pub history: Vec<f64>,
}

impl fmt::Display for RootResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "root={:.10}, f(root)={:.2e}, iters={}, converged={}",
            self.root, self.f_root, self.iterations, self.converged)
    }
}

// ── Bisection method ─────────────────────────────────────────────

/// Bisection method. Requires f(a) and f(b) to have opposite signs.
pub fn bisection(
    f: fn(f64) -> f64, mut a: f64, mut b: f64, conv: &Convergence, record: bool,
) -> RootResult {
    let mut fa = f(a);
    let fb = f(b);
    assert!(fa * fb <= 0.0, "bisection requires opposite signs");

    let mut history = Vec::new();
    let mut mid = a;

    for iter in 0..conv.max_iter {
        mid = (a + b) / 2.0;
        let fm = f(mid);
        if record { history.push(mid); }

        if fm.abs() < conv.f_tol || (b - a) / 2.0 < conv.x_tol {
            return RootResult { root: mid, f_root: fm, iterations: iter+1, converged: true, history };
        }

        if fa * fm < 0.0 { b = mid; } else { a = mid; fa = fm; }
    }

    RootResult { root: mid, f_root: f(mid), iterations: conv.max_iter, converged: false, history }
}

// ── Newton-Raphson method ────────────────────────────────────────

/// Newton-Raphson method. Requires the derivative f'(x).
pub fn newton_raphson(
    f: fn(f64) -> f64, df: fn(f64) -> f64, x0: f64, conv: &Convergence, record: bool,
) -> RootResult {
    let mut x = x0;
    let mut history = Vec::new();
    if record { history.push(x); }

    for iter in 0..conv.max_iter {
        let fx = f(x);
        let dfx = df(x);
        if dfx.abs() < 1e-15 {
            return RootResult { root: x, f_root: fx, iterations: iter+1, converged: false, history };
        }
        let x_new = x - fx / dfx;
        if record { history.push(x_new); }
        if fx.abs() < conv.f_tol || (x_new - x).abs() < conv.x_tol {
            return RootResult { root: x_new, f_root: f(x_new), iterations: iter+1, converged: true, history };
        }
        x = x_new;
    }

    RootResult { root: x, f_root: f(x), iterations: conv.max_iter, converged: false, history }
}

// ── Secant method ────────────────────────────────────────────────

/// Secant method. Uses two initial guesses x0, x1.
pub fn secant(
    f: fn(f64) -> f64, mut x0: f64, mut x1: f64, conv: &Convergence, record: bool,
) -> RootResult {
    let mut history = Vec::new();
    let mut f0 = f(x0);
    let mut f1 = f(x1);
    if record { history.push(x0); history.push(x1); }

    for iter in 0..conv.max_iter {
        if (f1 - f0).abs() < 1e-15 {
            return RootResult { root: x1, f_root: f1, iterations: iter+1, converged: false, history };
        }
        let x_new = x1 - f1*(x1-x0)/(f1-f0);
        let f_new = f(x_new);
        if record { history.push(x_new); }
        if f_new.abs() < conv.f_tol || (x_new - x1).abs() < conv.x_tol {
            return RootResult { root: x_new, f_root: f_new, iterations: iter+1, converged: true, history };
        }
        x0 = x1; f0 = f1;
        x1 = x_new; f1 = f_new;
    }

    RootResult { root: x1, f_root: f1, iterations: conv.max_iter, converged: false, history }
}

// ── Brent's method ───────────────────────────────────────────────

/// Brent's method — combines bisection, secant, and inverse quadratic interpolation.
pub fn brent(
    f: fn(f64) -> f64, mut a: f64, mut b: f64, conv: &Convergence, record: bool,
) -> RootResult {
    let mut fa = f(a);
    let mut fb = f(b);
    assert!(fa * fb <= 0.0, "Brent requires opposite signs");

    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = 0.0;
    let mut history = Vec::new();
    if record { history.push(b); }

    for iter in 0..conv.max_iter {
        if fb.abs() < conv.f_tol || (b - a).abs() < conv.x_tol {
            return RootResult { root: b, f_root: fb, iterations: iter+1, converged: true, history };
        }

        let s = if (fa - fc).abs() > 1e-15 && (fb - fc).abs() > 1e-15 {
            // Inverse quadratic interpolation
            a*fb*fc/((fa-fb)*(fa-fc)) + b*fa*fc/((fb-fa)*(fb-fc)) + c*fa*fb/((fc-fa)*(fc-fb))
        } else {
            b - fb*(b-a)/(fb-fa)
        };

        let lo = ((3.0*a+b)/4.0).min(b);
        let hi = ((3.0*a+b)/4.0).max(b);
        let cond1 = s < lo || s > hi;
        let cond2 = mflag && (s-b).abs() >= (b-c).abs()/2.0;
        let cond3 = !mflag && (s-b).abs() >= (c-d).abs()/2.0;
        let cond4 = mflag && (b-c).abs() < conv.x_tol;
        let cond5 = !mflag && (c-d).abs() < conv.x_tol;

        let s = if cond1 || cond2 || cond3 || cond4 || cond5 {
            mflag = true;
            (a + b) / 2.0
        } else {
            mflag = false;
            s
        };
        let fs = f(s);

        d = c;
        c = b;
        fc = fb;

        // Maintain the bracket: a and b must have opposite signs of f.
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if record { history.push(b); }

        // Ensure |f(b)| <= |f(a)| so b remains the best approximation.
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    RootResult { root: b, f_root: fb, iterations: conv.max_iter, converged: false, history }
}

// ── Multiple roots / bracket detection ───────────────────────────

/// Detect sign-change brackets in [a, b] by subdividing into n intervals.
pub fn detect_brackets(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> Vec<(f64, f64)> {
    assert!(n >= 1);
    let dx = (b - a) / n as f64;
    let mut brackets = Vec::new();
    let mut x_prev = a;
    let mut f_prev = f(a);
    for i in 1..=n {
        let x = a + i as f64 * dx;
        let fx = f(x);
        if f_prev * fx <= 0.0 { brackets.push((x_prev, x)); }
        x_prev = x; f_prev = fx;
    }
    brackets
}

/// Find multiple roots using user-supplied bracket intervals.
pub fn find_multiple_roots(
    f: fn(f64) -> f64, initial_brackets: &[(f64, f64)], conv: &Convergence,
) -> Vec<RootResult> {
    let mut roots = Vec::new();
    for &(a, b) in initial_brackets {
        let fa = f(a);
        let fb = f(b);
        if fa * fb > 0.0 {
            let sub = detect_brackets(f, a, b, 20);
            for (sa, sb) in sub {
                let res = brent(f, sa, sb, conv, false);
                if res.converged && !roots.iter().any(|r: &RootResult| (r.root-res.root).abs() < conv.x_tol*10.0) {
                    roots.push(res);
                }
            }
        } else {
            let res = brent(f, a, b, conv, false);
            if res.converged && !roots.iter().any(|r: &RootResult| (r.root-res.root).abs() < conv.x_tol*10.0) {
                roots.push(res);
            }
        }
    }
    roots
}

/// Automatically find all roots in [a, b] by bracket detection then Brent.
pub fn find_all_roots(
    f: fn(f64) -> f64, a: f64, b: f64, n_sub: usize, conv: &Convergence,
) -> Vec<RootResult> {
    let brackets = detect_brackets(f, a, b, n_sub);
    let mut roots = Vec::new();
    for (ba, bb) in brackets {
        let res = brent(f, ba, bb, conv, false);
        if res.converged && !roots.iter().any(|r: &RootResult| (r.root-res.root).abs() < conv.x_tol*10.0) {
            roots.push(res);
        }
    }
    roots
}

// ── Complex roots ────────────────────────────────────────────────

/// A complex number represented as (real, imag).
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self { Self { re, im } }
    pub fn magnitude(&self) -> f64 { (self.re*self.re + self.im*self.im).sqrt() }
    pub fn conjugate(&self) -> Self { Self { re: self.re, im: -self.im } }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 { write!(f, "{:.6}+{:.6}i", self.re, self.im) }
        else { write!(f, "{:.6}{:.6}i", self.re, self.im) }
    }
}

/// Find complex roots of a quadratic ax^2 + bx + c = 0.
pub fn quadratic_roots(a: f64, b: f64, c: f64) -> Vec<Complex> {
    if a.abs() < 1e-15 {
        if b.abs() < 1e-15 { return vec![]; }
        return vec![Complex::new(-c/b, 0.0)];
    }
    let disc = b*b - 4.0*a*c;
    if disc >= 0.0 {
        let sq = disc.sqrt();
        vec![Complex::new((-b+sq)/(2.0*a), 0.0), Complex::new((-b-sq)/(2.0*a), 0.0)]
    } else {
        let sq = (-disc).sqrt();
        vec![Complex::new(-b/(2.0*a), sq/(2.0*a)), Complex::new(-b/(2.0*a), -sq/(2.0*a))]
    }
}

/// Depressed cubic x^3 + px + q = 0 via Cardano's formula.
pub fn cubic_roots_depressed(p: f64, q: f64) -> Vec<Complex> {
    let disc = -4.0*p*p*p - 27.0*q*q;
    if disc.abs() < 1e-12 {
        if p.abs() < 1e-12 && q.abs() < 1e-12 { return vec![Complex::new(0.0, 0.0)]; }
        if p.abs() < 1e-12 { return vec![Complex::new((-q).cbrt(), 0.0)]; }
        return vec![Complex::new(3.0*q/p, 0.0), Complex::new(-3.0*q/(2.0*p), 0.0)];
    }
    if disc > 0.0 {
        let m = 2.0*(-p/3.0).sqrt();
        let theta = (3.0*q/(p*m)).acos()/3.0;
        let pi23 = 2.0*std::f64::consts::PI/3.0;
        vec![
            Complex::new(m*theta.cos(), 0.0),
            Complex::new(m*(theta-pi23).cos(), 0.0),
            Complex::new(m*(theta+pi23).cos(), 0.0),
        ]
    } else {
        let sq = (q*q/4.0 + p*p*p/27.0).abs().sqrt();
        let u = { let a = -q/2.0+sq; if a >= 0.0 { a.cbrt() } else { -((-a).cbrt()) } };
        let v = { let a = -q/2.0-sq; if a >= 0.0 { a.cbrt() } else { -((-a).cbrt()) } };
        let real_root = u + v;
        let re = -real_root/2.0;
        let im = (3.0_f64).sqrt()*(u-v)/2.0;
        vec![Complex::new(real_root, 0.0), Complex::new(re, im), Complex::new(re, -im)]
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    fn f_sqrt2(x: f64) -> f64 { x*x - 2.0 }
    fn df_sqrt2(x: f64) -> f64 { 2.0*x }
    fn f_cubic(x: f64) -> f64 { x*x*x - x }
    fn f_sin(x: f64) -> f64 { x.sin() }
    fn f_exp_m2(x: f64) -> f64 { x.exp() - 2.0 }
    fn df_exp_m2(x: f64) -> f64 { x.exp() }

    #[test]
    fn test_bisection_sqrt2() {
        let res = bisection(f_sqrt2, 1.0, 2.0, &Convergence::default(), false);
        assert!(res.converged);
        assert!(approx_eq(res.root, std::f64::consts::SQRT_2, 1e-10));
    }

    #[test]
    fn test_bisection_history() {
        let conv = Convergence { max_iter: 10, ..Convergence::default() };
        let res = bisection(f_sqrt2, 1.0, 2.0, &conv, true);
        assert!(!res.history.is_empty());
    }

    #[test]
    fn test_newton_sqrt2() {
        let res = newton_raphson(f_sqrt2, df_sqrt2, 1.5, &Convergence::default(), false);
        assert!(res.converged);
        assert!(approx_eq(res.root, std::f64::consts::SQRT_2, 1e-10));
    }

    #[test]
    fn test_newton_exp() {
        let res = newton_raphson(f_exp_m2, df_exp_m2, 1.0, &Convergence::default(), false);
        assert!(res.converged);
        assert!(approx_eq(res.root, 2.0_f64.ln(), 1e-10));
    }

    #[test]
    fn test_secant_sqrt2() {
        let res = secant(f_sqrt2, 1.0, 2.0, &Convergence::default(), false);
        assert!(res.converged);
        assert!(approx_eq(res.root, std::f64::consts::SQRT_2, 1e-10));
    }

    #[test]
    fn test_secant_history() {
        let conv = Convergence { max_iter: 20, ..Convergence::default() };
        let res = secant(f_sqrt2, 1.0, 2.0, &conv, true);
        assert!(res.history.len() >= 2);
    }

    #[test]
    fn test_brent_sqrt2() {
        let res = brent(f_sqrt2, 1.0, 2.0, &Convergence::default(), false);
        assert!(res.converged);
        assert!(approx_eq(res.root, std::f64::consts::SQRT_2, 1e-10));
    }

    #[test]
    fn test_brent_sin() {
        let res = brent(f_sin, 3.0, 3.5, &Convergence::default(), false);
        assert!(res.converged);
        assert!(approx_eq(res.root, std::f64::consts::PI, 1e-10));
    }

    #[test]
    fn test_detect_brackets() {
        let brackets = detect_brackets(f_sin, 0.0, 10.0, 100);
        assert!(brackets.len() >= 3);
    }

    #[test]
    fn test_find_all_roots_sin() {
        let roots = find_all_roots(f_sin, 0.5, 10.0, 100, &Convergence::default());
        assert!(roots.len() >= 3);
        for r in &roots { assert!(r.converged); assert!(r.f_root.abs() < 1e-10); }
    }

    #[test]
    fn test_find_multiple_roots() {
        let roots = find_multiple_roots(
            f_cubic, &[(-1.5,-0.5),(-0.5,0.5),(0.5,1.5)], &Convergence::default(),
        );
        assert_eq!(roots.len(), 3);
    }

    #[test]
    fn test_quadratic_real() {
        let r = quadratic_roots(1.0, -3.0, 2.0);
        let mut v: Vec<f64> = r.iter().map(|c| c.re).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(v[0], 1.0, 1e-10));
        assert!(approx_eq(v[1], 2.0, 1e-10));
    }

    #[test]
    fn test_quadratic_complex() {
        let r = quadratic_roots(1.0, 0.0, 1.0);
        assert_eq!(r.len(), 2);
        assert!(approx_eq(r[0].re, 0.0, 1e-10));
        assert!(approx_eq(r[0].im.abs(), 1.0, 1e-10));
    }

    #[test]
    fn test_quadratic_linear() {
        let r = quadratic_roots(0.0, 2.0, -4.0);
        assert_eq!(r.len(), 1);
        assert!(approx_eq(r[0].re, 2.0, 1e-10));
    }

    #[test]
    fn test_cubic_depressed() {
        let r = cubic_roots_depressed(-1.0, 0.0);
        assert!(r.len() >= 2);
    }

    #[test]
    fn test_convergence_default() {
        let c = Convergence::default();
        assert_eq!(c.max_iter, 100);
    }

    #[test]
    fn test_result_display() {
        let res = bisection(f_sqrt2, 1.0, 2.0, &Convergence::default(), false);
        let s = format!("{}", res);
        assert!(s.contains("converged=true"));
    }

    #[test]
    fn test_complex_ops() {
        let c = Complex::new(3.0, 4.0);
        assert!(approx_eq(c.magnitude(), 5.0, 1e-12));
        let conj = c.conjugate();
        assert!(approx_eq(conj.im, -4.0, 1e-12));
    }

    #[test]
    fn test_complex_display() {
        let s = format!("{}", Complex::new(1.0, -2.0));
        assert!(s.contains("i"));
    }

    #[test]
    fn test_brent_vs_bisection_accuracy() {
        // Both should find the same root, but Brent in fewer iterations
        let conv = Convergence { max_iter: 50, ..Convergence::default() };
        let bi = bisection(f_sqrt2, 1.0, 2.0, &conv, true);
        let br = brent(f_sqrt2, 1.0, 2.0, &conv, true);
        assert!(bi.converged && br.converged);
        // Brent typically converges faster
        assert!(br.iterations <= bi.iterations);
    }
}
