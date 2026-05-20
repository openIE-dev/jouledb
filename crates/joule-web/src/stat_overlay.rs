//! Statistical overlays: regression lines, confidence bands, trend lines, smoothing.
//!
//! Replaces ggplot2 geom_smooth, seaborn regplot, MATLAB polyfit+polyval+confint.
//! All output is SVG path/polygon strings that can be composed with any chart.

// ── Linear regression ──────────────────────────────────────────────

/// Ordinary least squares fit result.
#[derive(Debug, Clone)]
pub struct LinRegResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub std_err_slope: f64,
    pub std_err_intercept: f64,
    pub residual_std: f64,
}

/// Fit OLS linear regression: y = slope * x + intercept.
pub fn linear_regression(x: &[f64], y: &[f64]) -> LinRegResult {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 { return LinRegResult { slope: 0.0, intercept: 0.0, r_squared: 0.0, std_err_slope: 0.0, std_err_intercept: 0.0, residual_std: 0.0 }; }

    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for i in 0..n as usize {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }

    let slope = if sxx.abs() < 1e-15 { 0.0 } else { sxy / sxx };
    let intercept = my - slope * mx;

    let ss_res: f64 = (0..n as usize).map(|i| {
        let pred = slope * x[i] + intercept;
        (y[i] - pred).powi(2)
    }).sum();
    let ss_tot = syy;
    let r_squared = if ss_tot.abs() < 1e-15 { 1.0 } else { 1.0 - ss_res / ss_tot };

    let dof = (n - 2.0).max(1.0);
    let mse = ss_res / dof;
    let residual_std = mse.sqrt();
    let std_err_slope = if sxx.abs() < 1e-15 { 0.0 } else { (mse / sxx).sqrt() };
    let std_err_intercept = (mse * (1.0 / n + mx * mx / sxx)).sqrt();

    LinRegResult { slope, intercept, r_squared, std_err_slope, std_err_intercept, residual_std }
}

/// Polynomial regression: y = c0 + c1*x + c2*x^2 + ... + cn*x^n.
pub fn poly_fit(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    let m = degree + 1;
    if n < m { return vec![0.0; m]; }

    // Build normal equations: A'A * c = A'y
    let mut ata = vec![vec![0.0; m]; m];
    let mut aty = vec![0.0; m];

    for i in 0..n {
        let mut xi_pow = vec![1.0; m];
        for j in 1..m { xi_pow[j] = xi_pow[j - 1] * x[i]; }
        for j in 0..m {
            aty[j] += xi_pow[j] * y[i];
            for k in 0..m {
                ata[j][k] += xi_pow[j] * xi_pow[k];
            }
        }
    }

    // Solve via Gaussian elimination
    gauss_solve(&mut ata, &mut aty)
}

/// Evaluate polynomial at x.
pub fn poly_eval(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    let mut xp = 1.0;
    for &c in coeffs {
        result += c * xp;
        xp *= x;
    }
    result
}

fn gauss_solve(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Vec<f64> {
    let n = b.len();
    // Forward elimination
    for col in 0..n {
        let mut max_row = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[max_row][col].abs() { max_row = row; }
        }
        a.swap(col, max_row);
        b.swap(col, max_row);
        let pivot = a[col][col];
        if pivot.abs() < 1e-15 { continue; }
        for row in col + 1..n {
            let factor = a[row][col] / pivot;
            for k in col..n { a[row][k] -= factor * a[col][k]; }
            b[row] -= factor * b[col];
        }
    }
    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in i + 1..n { sum -= a[i][j] * x[j]; }
        x[i] = if a[i][i].abs() < 1e-15 { 0.0 } else { sum / a[i][i] };
    }
    x
}

// ── Confidence bands ───────────────────────────────────────────────

/// 95% confidence band for linear regression (approximate t ≈ 1.96 for large n).
pub fn confidence_band(x: &[f64], reg: &LinRegResult, n_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let sxx: f64 = x.iter().map(|&xi| (xi - mx).powi(2)).sum();
    let x_min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let t_crit = 1.96; // approx for large n

    let xs: Vec<f64> = (0..n_points).map(|i| x_min + (x_max - x_min) * i as f64 / (n_points - 1).max(1) as f64).collect();
    let mut upper = Vec::new();
    let mut lower = Vec::new();

    for &xi in &xs {
        let yhat = reg.slope * xi + reg.intercept;
        let se = reg.residual_std * (1.0 / n + (xi - mx).powi(2) / sxx).sqrt();
        upper.push(yhat + t_crit * se);
        lower.push(yhat - t_crit * se);
    }

    (xs, lower, upper)
}

// ── LOESS/LOWESS smoothing ─────────────────────────────────────────

/// Locally weighted scatterplot smoothing (LOWESS).
pub fn lowess(x: &[f64], y: &[f64], frac: f64, n_eval: usize) -> (Vec<f64>, Vec<f64>) {
    let n = x.len().min(y.len());
    if n < 3 { return (x.to_vec(), y.to_vec()); }

    let x_min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let eval_x: Vec<f64> = (0..n_eval).map(|i| x_min + (x_max - x_min) * i as f64 / (n_eval - 1).max(1) as f64).collect();
    let k = ((n as f64 * frac) as usize).max(3).min(n);

    let eval_y: Vec<f64> = eval_x.iter().map(|&xi| {
        // Find k nearest neighbors
        let mut dists: Vec<(f64, usize)> = (0..n).map(|j| ((x[j] - xi).abs(), j)).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let max_dist = dists[k - 1].0.max(1e-15);

        // Tricube weights
        let mut sw = 0.0;
        let mut swx = 0.0;
        let mut swy = 0.0;
        let mut swxx = 0.0;
        let mut swxy = 0.0;
        for &(d, j) in &dists[..k] {
            let u = d / max_dist;
            let w = if u < 1.0 { (1.0 - u * u * u).powi(3) } else { 0.0 };
            sw += w;
            swx += w * x[j];
            swy += w * y[j];
            swxx += w * x[j] * x[j];
            swxy += w * x[j] * y[j];
        }

        // Weighted linear regression at xi
        let det = sw * swxx - swx * swx;
        if det.abs() < 1e-15 { swy / sw.max(1e-15) }
        else { (swxx * swy - swx * swxy + (sw * swxy - swx * swy) * xi) / det }
    }).collect();

    (eval_x, eval_y)
}

// ── Moving average ─────────────────────────────────────────────────

/// Simple moving average.
pub fn moving_average(y: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || y.is_empty() { return y.to_vec(); }
    let w = window.min(y.len());
    let mut result = Vec::with_capacity(y.len());
    let mut sum: f64 = y[..w].iter().sum();
    // Pad first w-1 values with expanding window
    for i in 0..y.len() {
        if i >= w {
            sum += y[i] - y[i - w];
        } else if i > 0 {
            sum = y[..=i].iter().sum();
        }
        let count = (i + 1).min(w);
        result.push(sum / count as f64);
    }
    result
}

// ── SVG rendering helpers ──────────────────────────────────────────

/// Render a regression line as SVG.
pub fn regression_line_svg(
    reg: &LinRegResult, x_min: f64, x_max: f64,
    scale_x: impl Fn(f64) -> f64, scale_y: impl Fn(f64) -> f64,
    color: &str,
) -> String {
    let y0 = reg.slope * x_min + reg.intercept;
    let y1 = reg.slope * x_max + reg.intercept;
    format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{color}\" stroke-width=\"2\" stroke-dasharray=\"6,3\"/>",
        scale_x(x_min), scale_y(y0), scale_x(x_max), scale_y(y1)
    )
}

/// Render a confidence band as SVG polygon.
pub fn confidence_band_svg(
    xs: &[f64], lower: &[f64], upper: &[f64],
    scale_x: impl Fn(f64) -> f64, scale_y: impl Fn(f64) -> f64,
    color: &str, opacity: f64,
) -> String {
    let mut points = String::new();
    // Forward: upper bound
    for (i, &x) in xs.iter().enumerate() {
        let px = scale_x(x);
        let py = scale_y(upper[i]);
        if i == 0 { points.push_str(&format!("{px:.1},{py:.1}")); }
        else { points.push_str(&format!(" {px:.1},{py:.1}")); }
    }
    // Backward: lower bound
    for i in (0..xs.len()).rev() {
        let px = scale_x(xs[i]);
        let py = scale_y(lower[i]);
        points.push_str(&format!(" {px:.1},{py:.1}"));
    }
    format!("<polygon points=\"{points}\" fill=\"{color}\" fill-opacity=\"{opacity:.2}\" stroke=\"none\"/>")
}

/// Render a smoothed curve as SVG path.
pub fn smooth_curve_svg(
    xs: &[f64], ys: &[f64],
    scale_x: impl Fn(f64) -> f64, scale_y: impl Fn(f64) -> f64,
    color: &str, width: f64,
) -> String {
    let mut d = String::new();
    for (i, (&x, &y)) in xs.iter().zip(ys.iter()).enumerate() {
        let px = scale_x(x);
        let py = scale_y(y);
        if i == 0 { d.push_str(&format!("M{px:.1},{py:.1}")); }
        else { d.push_str(&format!(" L{px:.1},{py:.1}")); }
    }
    format!("<path d=\"{d}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"{width}\"/>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_regression_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = linear_regression(&x, &y);
        assert!((r.slope - 2.0).abs() < 0.01);
        assert!((r.intercept - 0.0).abs() < 0.01);
        assert!((r.r_squared - 1.0).abs() < 0.01);
    }

    #[test]
    fn linear_regression_noisy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
        let r = linear_regression(&x, &y);
        assert!((r.slope - 2.0).abs() < 0.2);
        assert!(r.r_squared > 0.99);
    }

    #[test]
    fn poly_fit_linear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0]; // y = 1 + 2x
        let c = poly_fit(&x, &y, 1);
        assert!((c[0] - 1.0).abs() < 0.01); // intercept
        assert!((c[1] - 2.0).abs() < 0.01); // slope
    }

    #[test]
    fn poly_fit_quadratic() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect(); // y = x^2
        let c = poly_fit(&x, &y, 2);
        assert!((c[0]).abs() < 0.1); // c0 ≈ 0
        assert!((c[1]).abs() < 0.1); // c1 ≈ 0
        assert!((c[2] - 1.0).abs() < 0.1); // c2 ≈ 1
    }

    #[test]
    fn poly_eval_correct() {
        let c = vec![1.0, 2.0, 3.0]; // 1 + 2x + 3x^2
        assert!((poly_eval(&c, 0.0) - 1.0).abs() < 0.01);
        assert!((poly_eval(&c, 1.0) - 6.0).abs() < 0.01);
        assert!((poly_eval(&c, 2.0) - 17.0).abs() < 0.01);
    }

    #[test]
    fn confidence_band_positive_width() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
        let reg = linear_regression(&x, &y);
        let (xs, lower, upper) = confidence_band(&x, &reg, 20);
        assert_eq!(xs.len(), 20);
        for i in 0..20 { assert!(upper[i] >= lower[i]); }
    }

    #[test]
    fn lowess_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let (xs, ys) = lowess(&x, &y, 0.5, 20);
        assert_eq!(xs.len(), 20);
        assert_eq!(ys.len(), 20);
    }

    #[test]
    fn moving_average_basic() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&y, 3);
        assert_eq!(ma.len(), 5);
        assert!((ma[2] - 2.0).abs() < 0.01); // (1+2+3)/3
        assert!((ma[4] - 4.0).abs() < 0.01); // (3+4+5)/3
    }
}
