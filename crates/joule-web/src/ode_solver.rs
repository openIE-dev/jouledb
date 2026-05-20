//! ODE solver — Euler, RK4, adaptive step size, systems of ODEs, event detection,
//! error estimation.
//!
//! Pure-Rust ordinary differential equation solvers with step recording,
//! convergence comparison, event detection, and error estimation.

use std::fmt;

// ── Types ────────────────────────────────────────────────────────

/// A single ODE dy/dt = f(t, y).
pub type OdeFunc = fn(t: f64, y: f64) -> f64;

/// A system of ODEs dy/dt = f(t, y) where y is a vector.
pub type OdeSystem = fn(t: f64, y: &[f64], dydt: &mut [f64]);

/// A recorded step during integration.
#[derive(Debug, Clone)]
pub struct Step {
    pub t: f64,
    pub y: Vec<f64>,
    pub dt: f64,
}

/// Result of an ODE integration.
#[derive(Debug, Clone)]
pub struct OdeResult {
    pub t_final: f64,
    pub y_final: Vec<f64>,
    pub steps: Vec<Step>,
    pub fn_evals: usize,
    pub step_count: usize,
    pub rejected_steps: usize,
}

impl fmt::Display for OdeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ODE: t={:.4}, steps={}, evals={}, rejected={}",
            self.t_final, self.step_count, self.fn_evals, self.rejected_steps)
    }
}

/// An event detected during integration.
#[derive(Debug, Clone)]
pub struct OdeEvent {
    pub t: f64,
    pub y: Vec<f64>,
    pub event_index: usize,
}

/// Result of error estimation between two solutions.
#[derive(Debug, Clone)]
pub struct ErrorEstimate {
    pub max_error: f64,
    pub rms_error: f64,
    pub sample_count: usize,
}

// ── Euler method (scalar) ────────────────────────────────────────

/// Euler method for a single ODE: dy/dt = f(t, y).
pub fn euler(f: OdeFunc, t0: f64, y0: f64, t_end: f64, dt: f64, record: bool) -> OdeResult {
    let mut t = t0;
    let mut y = y0;
    let mut steps = Vec::new();
    let mut evals = 0usize;
    let mut step_count = 0usize;

    if record { steps.push(Step { t, y: vec![y], dt }); }

    while t < t_end - dt * 1e-10 {
        let h = if t + dt > t_end { t_end - t } else { dt };
        let dy = f(t, y);
        evals += 1;
        y += h * dy;
        t += h;
        step_count += 1;
        if record { steps.push(Step { t, y: vec![y], dt: h }); }
    }

    OdeResult { t_final: t, y_final: vec![y], steps, fn_evals: evals, step_count, rejected_steps: 0 }
}

// ── Euler method (system) ────────────────────────────────────────

/// Euler method for a system of ODEs.
pub fn euler_system(
    f: OdeSystem, t0: f64, y0: &[f64], t_end: f64, dt: f64, record: bool,
) -> OdeResult {
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut dydt = vec![0.0; n];
    let mut steps = Vec::new();
    let mut evals = 0usize;
    let mut step_count = 0usize;

    if record { steps.push(Step { t, y: y.clone(), dt }); }

    while t < t_end - dt * 1e-10 {
        let h = if t + dt > t_end { t_end - t } else { dt };
        f(t, &y, &mut dydt);
        evals += 1;
        for i in 0..n { y[i] += h * dydt[i]; }
        t += h;
        step_count += 1;
        if record { steps.push(Step { t, y: y.clone(), dt: h }); }
    }

    OdeResult { t_final: t, y_final: y, steps, fn_evals: evals, step_count, rejected_steps: 0 }
}

// ── RK4 method (scalar) ─────────────────────────────────────────

/// Classical 4th-order Runge-Kutta for a single ODE.
pub fn rk4(f: OdeFunc, t0: f64, y0: f64, t_end: f64, dt: f64, record: bool) -> OdeResult {
    let mut t = t0;
    let mut y = y0;
    let mut steps = Vec::new();
    let mut evals = 0usize;
    let mut step_count = 0usize;

    if record { steps.push(Step { t, y: vec![y], dt }); }

    while t < t_end - dt * 1e-10 {
        let h = if t + dt > t_end { t_end - t } else { dt };
        let k1 = f(t, y);
        let k2 = f(t + h / 2.0, y + h / 2.0 * k1);
        let k3 = f(t + h / 2.0, y + h / 2.0 * k2);
        let k4 = f(t + h, y + h * k3);
        evals += 4;
        y += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += h;
        step_count += 1;
        if record { steps.push(Step { t, y: vec![y], dt: h }); }
    }

    OdeResult { t_final: t, y_final: vec![y], steps, fn_evals: evals, step_count, rejected_steps: 0 }
}

// ── RK4 method (system) ─────────────────────────────────────────

/// Classical 4th-order Runge-Kutta for a system of ODEs.
pub fn rk4_system(
    f: OdeSystem, t0: f64, y0: &[f64], t_end: f64, dt: f64, record: bool,
) -> OdeResult {
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut steps = Vec::new();
    let mut evals = 0usize;
    let mut step_count = 0usize;

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut tmp = vec![0.0; n];

    if record { steps.push(Step { t, y: y.clone(), dt }); }

    while t < t_end - dt * 1e-10 {
        let h = if t + dt > t_end { t_end - t } else { dt };

        f(t, &y, &mut k1);
        for i in 0..n { tmp[i] = y[i] + h / 2.0 * k1[i]; }
        f(t + h / 2.0, &tmp, &mut k2);
        for i in 0..n { tmp[i] = y[i] + h / 2.0 * k2[i]; }
        f(t + h / 2.0, &tmp, &mut k3);
        for i in 0..n { tmp[i] = y[i] + h * k3[i]; }
        f(t + h, &tmp, &mut k4);
        evals += 4;

        for i in 0..n {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        step_count += 1;
        if record { steps.push(Step { t, y: y.clone(), dt: h }); }
    }

    OdeResult { t_final: t, y_final: y, steps, fn_evals: evals, step_count, rejected_steps: 0 }
}

// ── RK45 Dormand-Prince (adaptive, system) ──────────────────────

/// Dormand-Prince RK45 adaptive step-size method for a system of ODEs.
pub fn rk45_system(
    f: OdeSystem, t0: f64, y0: &[f64], t_end: f64, dt_initial: f64,
    atol: f64, rtol: f64, record: bool,
) -> OdeResult {
    const A21: f64 = 1.0/5.0;
    const A31: f64 = 3.0/40.0; const A32: f64 = 9.0/40.0;
    const A41: f64 = 44.0/45.0; const A42: f64 = -56.0/15.0; const A43: f64 = 32.0/9.0;
    const A51: f64 = 19372.0/6561.0; const A52: f64 = -25360.0/2187.0;
    const A53: f64 = 64448.0/6561.0; const A54: f64 = -212.0/729.0;
    const A61: f64 = 9017.0/3168.0; const A62: f64 = -355.0/33.0;
    const A63: f64 = 46732.0/5247.0; const A64: f64 = 49.0/176.0; const A65: f64 = -5103.0/18656.0;
    const B1: f64 = 35.0/384.0; const B3: f64 = 500.0/1113.0;
    const B4: f64 = 125.0/192.0; const B5: f64 = -2187.0/6784.0; const B6: f64 = 11.0/84.0;
    const E1: f64 = 71.0/57600.0; const E3: f64 = -71.0/16695.0;
    const E4: f64 = 71.0/1920.0; const E5: f64 = -17253.0/339200.0;
    const E6: f64 = 22.0/525.0; const E7: f64 = -1.0/40.0;

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut dt = dt_initial;
    let mut steps = Vec::new();
    let mut evals = 0usize;
    let mut step_count = 0usize;
    let mut rejected = 0usize;

    let mut k1 = vec![0.0; n]; let mut k2 = vec![0.0; n]; let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n]; let mut k5 = vec![0.0; n]; let mut k6 = vec![0.0; n];
    let mut k7 = vec![0.0; n]; let mut tmp = vec![0.0; n]; let mut y_new = vec![0.0; n];

    if record { steps.push(Step { t, y: y.clone(), dt }); }

    let max_iter = 1_000_000;
    let mut iter = 0;
    while t < t_end - 1e-12 * dt_initial && iter < max_iter {
        iter += 1;
        let h = if t + dt > t_end { t_end - t } else { dt };

        f(t, &y, &mut k1);
        for i in 0..n { tmp[i] = y[i] + h * A21 * k1[i]; }
        f(t + h / 5.0, &tmp, &mut k2);
        for i in 0..n { tmp[i] = y[i] + h * (A31*k1[i] + A32*k2[i]); }
        f(t + 3.0*h/10.0, &tmp, &mut k3);
        for i in 0..n { tmp[i] = y[i] + h * (A41*k1[i] + A42*k2[i] + A43*k3[i]); }
        f(t + 4.0*h/5.0, &tmp, &mut k4);
        for i in 0..n { tmp[i] = y[i] + h*(A51*k1[i]+A52*k2[i]+A53*k3[i]+A54*k4[i]); }
        f(t + 8.0*h/9.0, &tmp, &mut k5);
        for i in 0..n { tmp[i] = y[i]+h*(A61*k1[i]+A62*k2[i]+A63*k3[i]+A64*k4[i]+A65*k5[i]); }
        f(t + h, &tmp, &mut k6);
        evals += 6;

        for i in 0..n {
            y_new[i] = y[i] + h*(B1*k1[i] + B3*k3[i] + B4*k4[i] + B5*k5[i] + B6*k6[i]);
        }

        f(t + h, &y_new, &mut k7);
        evals += 1;

        let mut err = 0.0;
        for i in 0..n {
            let sc = atol + rtol * y[i].abs().max(y_new[i].abs());
            let ei = h*(E1*k1[i]+E3*k3[i]+E4*k4[i]+E5*k5[i]+E6*k6[i]+E7*k7[i]);
            err += (ei / sc) * (ei / sc);
        }
        err = (err / n as f64).sqrt();

        if err <= 1.0 {
            t += h;
            for i in 0..n { y[i] = y_new[i]; }
            step_count += 1;
            if record { steps.push(Step { t, y: y.clone(), dt: h }); }
            let factor = if err < 1e-15 { 5.0 } else { (0.84 * (1.0/err).powf(0.2)).min(5.0) };
            dt = h * factor;
        } else {
            rejected += 1;
            dt = h * (0.84 * (1.0/err).powf(0.2)).max(0.1);
        }
    }

    OdeResult { t_final: t, y_final: y, steps, fn_evals: evals, step_count, rejected_steps: rejected }
}

// ── Event detection ──────────────────────────────────────────────

/// Detect events (zero-crossings of event functions) during RK4 integration of a system.
/// Each event function g(t, y) triggers when it crosses zero from positive to negative.
pub fn rk4_with_events(
    f: OdeSystem, t0: f64, y0: &[f64], t_end: f64, dt: f64,
    event_fns: &[fn(f64, &[f64]) -> f64],
) -> (OdeResult, Vec<OdeEvent>) {
    let n = y0.len();
    let ne = event_fns.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut steps = Vec::new();
    let mut evals = 0usize;
    let mut step_count = 0usize;
    let mut events = Vec::new();

    let mut k1 = vec![0.0; n]; let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n]; let mut k4 = vec![0.0; n];
    let mut tmp = vec![0.0; n];

    let mut prev_g: Vec<f64> = event_fns.iter().map(|g| g(t, &y)).collect();
    steps.push(Step { t, y: y.clone(), dt });

    while t < t_end - dt * 1e-10 {
        let h = if t + dt > t_end { t_end - t } else { dt };
        let y_old = y.clone();
        let t_old = t;

        f(t, &y, &mut k1);
        for i in 0..n { tmp[i] = y[i] + h/2.0*k1[i]; }
        f(t + h/2.0, &tmp, &mut k2);
        for i in 0..n { tmp[i] = y[i] + h/2.0*k2[i]; }
        f(t + h/2.0, &tmp, &mut k3);
        for i in 0..n { tmp[i] = y[i] + h*k3[i]; }
        f(t + h, &tmp, &mut k4);
        evals += 4;

        for i in 0..n {
            y[i] += h / 6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        }
        t += h;
        step_count += 1;

        // Check for zero-crossings
        for ei in 0..ne {
            let g_new = event_fns[ei](t, &y);
            if prev_g[ei] * g_new < 0.0 {
                // Bisect to find the event time
                let mut ta = t_old;
                let mut tb = t;
                let mut ya = y_old.clone();
                let mut yb = y.clone();
                for _ in 0..50 {
                    let tm = (ta + tb) / 2.0;
                    // Linear interpolation of state at tm
                    let frac = (tm - t_old) / h;
                    let ym: Vec<f64> = (0..n).map(|i| ya[i] + frac*(yb[i]-ya[i])).collect();
                    let gm = event_fns[ei](tm, &ym);
                    if gm.abs() < 1e-12 || (tb - ta) < 1e-14 {
                        events.push(OdeEvent { t: tm, y: ym, event_index: ei });
                        break;
                    }
                    let ga = event_fns[ei](ta, &ya);
                    if ga * gm < 0.0 {
                        tb = tm;
                        yb = ym;
                    } else {
                        ta = tm;
                        ya = ym;
                    }
                }
            }
            prev_g[ei] = g_new;
        }

        steps.push(Step { t, y: y.clone(), dt: h });
    }

    let result = OdeResult {
        t_final: t, y_final: y, steps, fn_evals: evals,
        step_count, rejected_steps: 0,
    };
    (result, events)
}

// ── Error estimation ─────────────────────────────────────────────

/// Estimate error by comparing solutions at two different step sizes.
/// Runs f with dt and dt/2, returns max and RMS error at the final time.
pub fn estimate_error(
    f: OdeFunc, t0: f64, y0: f64, t_end: f64, dt: f64,
) -> ErrorEstimate {
    let coarse = rk4(f, t0, y0, t_end, dt, true);
    let fine = rk4(f, t0, y0, t_end, dt / 2.0, true);

    let mut max_err = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    // Compare at coarse step times
    let mut fi = 0;
    for cs in &coarse.steps {
        // Find nearest fine step
        while fi + 1 < fine.steps.len() && (fine.steps[fi].t - cs.t).abs() > (fine.steps[fi + 1].t - cs.t).abs() {
            fi += 1;
        }
        if fi < fine.steps.len() {
            let err = (cs.y[0] - fine.steps[fi].y[0]).abs();
            if err > max_err { max_err = err; }
            sum_sq += err * err;
            count += 1;
        }
    }

    let rms = if count > 0 { (sum_sq / count as f64).sqrt() } else { 0.0 };
    ErrorEstimate { max_error: max_err, rms_error: rms, sample_count: count }
}

// ── Convergence comparison ───────────────────────────────────────

/// Compare convergence of Euler vs RK4 on a scalar ODE.
pub fn convergence_comparison(
    f: OdeFunc, t0: f64, y0: f64, t_end: f64,
    exact_solution: fn(f64) -> f64, step_sizes: &[f64],
) -> Vec<(f64, f64, f64)> {
    let exact = exact_solution(t_end);
    step_sizes.iter().map(|dt| {
        let euler_err = (euler(f, t0, y0, t_end, *dt, false).y_final[0] - exact).abs();
        let rk4_err = (rk4(f, t0, y0, t_end, *dt, false).y_final[0] - exact).abs();
        (*dt, euler_err, rk4_err)
    }).collect()
}

// ── Demo problems ────────────────────────────────────────────────

/// Lorenz system: standard parameters sigma=10, rho=28, beta=8/3.
pub fn lorenz_system(t: f64, y: &[f64], dydt: &mut [f64]) {
    let _ = t;
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    dydt[0] = sigma * (y[1] - y[0]);
    dydt[1] = y[0] * (rho - y[2]) - y[1];
    dydt[2] = y[0] * y[1] - beta * y[2];
}

pub fn solve_lorenz(t_end: f64, dt: f64) -> OdeResult {
    rk4_system(lorenz_system, 0.0, &[1.0, 1.0, 1.0], t_end, dt, true)
}

/// Harmonic oscillator: x'' + omega^2 * x = 0, unit omega.
pub fn harmonic_system(_t: f64, y: &[f64], dydt: &mut [f64]) {
    dydt[0] = y[1];
    dydt[1] = -y[0];
}

pub fn solve_harmonic(omega: f64, x0: f64, v0: f64, t_end: f64, dt: f64) -> OdeResult {
    if (omega - 1.0).abs() < 1e-15 {
        rk4_system(harmonic_system, 0.0, &[x0, v0], t_end, dt, true)
    } else {
        solve_harmonic_general(omega, x0, v0, t_end, dt)
    }
}

fn solve_harmonic_general(omega: f64, x0: f64, v0: f64, t_end: f64, dt: f64) -> OdeResult {
    let n = 2;
    let mut t = 0.0;
    let mut y = vec![x0, v0];
    let mut steps = Vec::new();
    let mut evals = 0usize;
    let mut step_count = 0usize;
    let omega2 = omega * omega;

    steps.push(Step { t, y: y.clone(), dt });
    let eval = |yv: &[f64]| -> Vec<f64> { vec![yv[1], -omega2 * yv[0]] };

    while t < t_end - dt * 1e-10 {
        let h = if t + dt > t_end { t_end - t } else { dt };
        let k1 = eval(&y);
        let tmp2: Vec<f64> = (0..n).map(|i| y[i] + h/2.0*k1[i]).collect();
        let k2 = eval(&tmp2);
        let tmp3: Vec<f64> = (0..n).map(|i| y[i] + h/2.0*k2[i]).collect();
        let k3 = eval(&tmp3);
        let tmp4: Vec<f64> = (0..n).map(|i| y[i] + h*k3[i]).collect();
        let k4 = eval(&tmp4);
        evals += 4;
        for i in 0..n { y[i] += h/6.0*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i]); }
        t += h;
        step_count += 1;
        steps.push(Step { t, y: y.clone(), dt: h });
    }

    OdeResult { t_final: t, y_final: y, steps, fn_evals: evals, step_count, rejected_steps: 0 }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    fn exp_ode(_t: f64, y: f64) -> f64 { y }
    fn exp_exact(t: f64) -> f64 { t.exp() }
    fn decay_ode(_t: f64, y: f64) -> f64 { -y }
    fn quad_ode(t: f64, _y: f64) -> f64 { 2.0 * t }

    #[test]
    fn test_euler_exponential() {
        let res = euler(exp_ode, 0.0, 1.0, 1.0, 0.001, false);
        assert!(approx_eq(res.y_final[0], exp_exact(1.0), 0.01));
    }

    #[test]
    fn test_euler_records_steps() {
        let res = euler(exp_ode, 0.0, 1.0, 0.1, 0.05, true);
        assert_eq!(res.steps.len(), 3);
    }

    #[test]
    fn test_rk4_exponential() {
        let res = rk4(exp_ode, 0.0, 1.0, 1.0, 0.1, false);
        assert!(approx_eq(res.y_final[0], exp_exact(1.0), 1e-4));
    }

    #[test]
    fn test_rk4_decay() {
        let res = rk4(decay_ode, 0.0, 1.0, 2.0, 0.01, false);
        assert!(approx_eq(res.y_final[0], (-2.0_f64).exp(), 1e-8));
    }

    #[test]
    fn test_rk4_quadratic() {
        let res = rk4(quad_ode, 0.0, 0.0, 1.0, 0.01, false);
        assert!(approx_eq(res.y_final[0], 1.0, 1e-8));
    }

    #[test]
    fn test_euler_system() {
        fn exp_sys(_t: f64, y: &[f64], dydt: &mut [f64]) { dydt[0] = y[0]; }
        let res = euler_system(exp_sys, 0.0, &[1.0], 1.0, 0.001, false);
        assert!(approx_eq(res.y_final[0], std::f64::consts::E, 0.01));
    }

    #[test]
    fn test_rk4_system_harmonic() {
        fn ho(_t: f64, y: &[f64], dydt: &mut [f64]) { dydt[0] = y[1]; dydt[1] = -y[0]; }
        let res = rk4_system(ho, 0.0, &[1.0, 0.0], std::f64::consts::PI, 0.01, false);
        assert!(approx_eq(res.y_final[0], -1.0, 1e-6));
        assert!(approx_eq(res.y_final[1], 0.0, 1e-4));
    }

    #[test]
    fn test_rk45_adaptive() {
        fn exp_sys(_t: f64, y: &[f64], dydt: &mut [f64]) { dydt[0] = y[0]; }
        let res = rk45_system(exp_sys, 0.0, &[1.0], 1.0, 0.1, 1e-8, 1e-8, false);
        assert!(approx_eq(res.y_final[0], std::f64::consts::E, 1e-6));
    }

    #[test]
    fn test_rk45_recording() {
        fn exp_sys(_t: f64, y: &[f64], dydt: &mut [f64]) { dydt[0] = y[0]; }
        let res = rk45_system(exp_sys, 0.0, &[1.0], 1.0, 0.1, 1e-6, 1e-6, true);
        assert!(res.steps.len() >= 2);
    }

    #[test]
    fn test_convergence_comparison() {
        let results = convergence_comparison(exp_ode, 0.0, 1.0, 1.0, exp_exact, &[0.1, 0.05, 0.01]);
        assert_eq!(results.len(), 3);
        assert!(results[2].1 < results[0].1); // Euler improves
        assert!(results[2].2 < results[0].2); // RK4 improves
        assert!(results[0].2 < results[0].1); // RK4 < Euler at same dt
    }

    #[test]
    fn test_event_detection() {
        // Bouncing ball: y''=-9.8, y(0)=10, v(0)=0
        // Event: y=0 (ground hit)
        fn ball(_t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = y[1];
            dydt[1] = -9.8;
        }
        fn ground_event(_t: f64, y: &[f64]) -> f64 { y[0] }

        let (result, events) = rk4_with_events(ball, 0.0, &[10.0, 0.0], 3.0, 0.001, &[ground_event]);
        assert!(result.y_final.len() == 2);
        assert!(!events.is_empty(), "should detect ground hit");
        // The ball should hit ground around t=sqrt(20/9.8) ~ 1.43
        assert!(approx_eq(events[0].t, (20.0_f64 / 9.8).sqrt(), 0.01));
    }

    #[test]
    fn test_error_estimation() {
        let est = estimate_error(exp_ode, 0.0, 1.0, 1.0, 0.1);
        assert!(est.max_error > 0.0);
        assert!(est.rms_error > 0.0);
        assert!(est.sample_count > 0);
        // Fine solution should be more accurate, so difference should be small
        assert!(est.max_error < 1e-4);
    }

    #[test]
    fn test_harmonic_oscillator() {
        let res = solve_harmonic(1.0, 1.0, 0.0, std::f64::consts::PI, 0.01);
        assert!(approx_eq(res.y_final[0], -1.0, 1e-4));
    }

    #[test]
    fn test_harmonic_general_omega() {
        let res = solve_harmonic(2.0, 1.0, 0.0, std::f64::consts::PI, 0.001);
        assert!(approx_eq(res.y_final[0], 1.0, 1e-3));
    }

    #[test]
    fn test_lorenz_runs() {
        let res = solve_lorenz(1.0, 0.01);
        assert!(res.steps.len() > 10);
        assert!(res.y_final.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lorenz_system_fn() {
        let mut dydt = [0.0; 3];
        lorenz_system(0.0, &[1.0, 1.0, 1.0], &mut dydt);
        assert!(approx_eq(dydt[0], 0.0, 1e-12));
        assert!(approx_eq(dydt[1], 26.0, 1e-12));
    }

    #[test]
    fn test_euler_step_count() {
        let res = euler(exp_ode, 0.0, 1.0, 1.0, 0.1, false);
        assert_eq!(res.step_count, 10);
        assert_eq!(res.fn_evals, 10);
    }

    #[test]
    fn test_rk4_evals() {
        let res = rk4(exp_ode, 0.0, 1.0, 1.0, 0.1, false);
        assert_eq!(res.fn_evals, 40);
    }

    #[test]
    fn test_ode_result_display() {
        let res = euler(exp_ode, 0.0, 1.0, 1.0, 0.1, false);
        let s = format!("{}", res);
        assert!(s.contains("ODE:"));
    }

    #[test]
    fn test_multiple_events() {
        // Oscillator: detect zero crossings of position
        fn osc(_t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = y[1]; dydt[1] = -y[0];
        }
        fn pos_cross(_t: f64, y: &[f64]) -> f64 { y[0] }

        let (_, events) = rk4_with_events(
            osc, 0.0, &[1.0, 0.0], 2.0 * std::f64::consts::PI, 0.01, &[pos_cross],
        );
        // cos(t)=0 at pi/2, 3pi/2 approximately — expect at least 2 crossings
        assert!(events.len() >= 2);
    }
}
