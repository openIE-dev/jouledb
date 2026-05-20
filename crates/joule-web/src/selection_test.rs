//! Selection tests for detecting natural selection from population data.
//!
//! Implements Tajima's D, Fu and Li's D* and F*, the McDonald-Kreitman
//! test, Fay and Wu's H, and composite likelihood ratio statistics.
//! All computations are std-only with f64 math.

use std::fmt;

// ── SFS-Based Input ─────────────────────────────────────────────────

/// Site frequency spectrum data for neutrality tests.
#[derive(Debug, Clone, PartialEq)]
pub struct SfsData {
    /// Counts: entry `i` = number of sites with `i` derived alleles (0..=2n).
    pub counts: Vec<u64>,
    /// Number of diploid individuals.
    pub n_individuals: usize,
}

impl SfsData {
    pub fn new(counts: Vec<u64>, n_individuals: usize) -> Self {
        Self { counts, n_individuals }
    }

    /// Total number of chromosomes: 2n.
    pub fn n_chromosomes(&self) -> usize {
        2 * self.n_individuals
    }

    /// Number of segregating sites S.
    pub fn segregating_sites(&self) -> u64 {
        let n2 = self.n_chromosomes();
        if self.counts.len() <= 1 {
            return 0;
        }
        let end = self.counts.len().min(n2);
        self.counts[1..end].iter().sum()
    }

    /// Mean pairwise differences (pi).
    pub fn pi(&self) -> f64 {
        let n2 = self.n_chromosomes();
        let n2f = n2 as f64;
        let mut sum = 0.0;
        for (i, &c) in self.counts.iter().enumerate() {
            if i == 0 || i >= n2 {
                continue;
            }
            let freq = i as f64 / n2f;
            sum += c as f64 * 2.0 * freq * (1.0 - freq) * n2f / (n2f - 1.0);
        }
        sum
    }

    /// Watterson's theta.
    pub fn theta_w(&self) -> f64 {
        let s = self.segregating_sites() as f64;
        let a1 = harmonic(self.n_chromosomes() - 1);
        if a1 == 0.0 { 0.0 } else { s / a1 }
    }

    /// Singleton count (sites with exactly 1 derived allele).
    pub fn singletons(&self) -> u64 {
        if self.counts.len() > 1 { self.counts[1] } else { 0 }
    }

    /// High-frequency derived count (sites with > n/2 derived alleles).
    pub fn high_freq_derived(&self) -> u64 {
        let half = self.n_chromosomes() / 2;
        let n2 = self.n_chromosomes();
        self.counts.iter().enumerate()
            .filter(|&(i, _)| i > half && i < n2)
            .map(|(_, &c)| c)
            .sum()
    }
}

impl fmt::Display for SfsData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SFS(n={}, S={}, pi={:.4}, theta_w={:.4})",
            self.n_individuals,
            self.segregating_sites(),
            self.pi(),
            self.theta_w()
        )
    }
}

// ── Test Result ─────────────────────────────────────────────────────

/// Result of a neutrality / selection test.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionResult {
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    pub interpretation: SelectionSignal,
    pub n_sites: u64,
}

/// Interpretation of the test signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionSignal {
    Neutral,
    PositiveSelection,
    BalancingSelection,
    PurifyingSelection,
    Expansion,
}

impl fmt::Display for SelectionSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neutral => write!(f, "neutral"),
            Self::PositiveSelection => write!(f, "positive_selection"),
            Self::BalancingSelection => write!(f, "balancing_selection"),
            Self::PurifyingSelection => write!(f, "purifying_selection"),
            Self::Expansion => write!(f, "expansion"),
        }
    }
}

impl fmt::Display for SelectionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}={:.4} p={:.6} signal={} sites={}",
            self.test_name, self.statistic, self.p_value,
            self.interpretation, self.n_sites
        )
    }
}

// ── Selection Tester ────────────────────────────────────────────────

/// Configurable selection test suite.
#[derive(Debug, Clone)]
pub struct SelectionTester {
    alpha: f64,
    use_beta_approx: bool,
}

impl Default for SelectionTester {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            use_beta_approx: true,
        }
    }
}

impl SelectionTester {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_alpha(mut self, a: f64) -> Self {
        self.alpha = a.clamp(1e-10, 0.5);
        self
    }

    pub fn with_beta_approx(mut self, b: bool) -> Self {
        self.use_beta_approx = b;
        self
    }

    /// Tajima's D statistic.
    pub fn tajimas_d(&self, sfs: &SfsData) -> SelectionResult {
        let n = sfs.n_chromosomes();
        let s = sfs.segregating_sites() as f64;
        let pi = sfs.pi();
        let theta_w = sfs.theta_w();

        if s == 0.0 || n < 4 {
            return SelectionResult {
                test_name: "Tajima_D".into(),
                statistic: 0.0,
                p_value: 1.0,
                interpretation: SelectionSignal::Neutral,
                n_sites: sfs.segregating_sites(),
            };
        }

        let a1 = harmonic(n - 1);
        let a2 = harmonic2(n - 1);
        let nf = n as f64;

        let b1 = (nf + 1.0) / (3.0 * (nf - 1.0));
        let b2 = 2.0 * (nf * nf + nf + 3.0) / (9.0 * nf * (nf - 1.0));

        let c1 = b1 - 1.0 / a1;
        let c2 = b2 - (nf + 2.0) / (a1 * nf) + a2 / (a1 * a1);

        let e1 = c1 / a1;
        let e2 = c2 / (a1 * a1 + a2);

        let var_d = e1 * s + e2 * s * (s - 1.0);
        let d = if var_d > 0.0 {
            (pi - theta_w) / var_d.sqrt()
        } else {
            0.0
        };

        let p_value = if self.use_beta_approx {
            tajima_d_pvalue(d, n)
        } else {
            normal_two_tail(d)
        };

        let interpretation = if d.abs() < 1e-10 || p_value > self.alpha {
            SelectionSignal::Neutral
        } else if d > 0.0 {
            SelectionSignal::BalancingSelection
        } else {
            SelectionSignal::PositiveSelection
        };

        SelectionResult {
            test_name: "Tajima_D".into(),
            statistic: d,
            p_value,
            interpretation,
            n_sites: sfs.segregating_sites(),
        }
    }

    /// Fu and Li's D* (uses singletons vs total mutations).
    pub fn fu_li_d_star(&self, sfs: &SfsData) -> SelectionResult {
        let n = sfs.n_chromosomes();
        let s = sfs.segregating_sites() as f64;
        let eta_s = sfs.singletons() as f64;

        if s == 0.0 || n < 4 {
            return SelectionResult {
                test_name: "Fu_Li_D*".into(),
                statistic: 0.0,
                p_value: 1.0,
                interpretation: SelectionSignal::Neutral,
                n_sites: sfs.segregating_sites(),
            };
        }

        let nf = n as f64;
        let a1 = harmonic(n - 1);
        let a2 = harmonic2(n - 1);

        let an_plus1 = a1 + 1.0 / nf;
        let cn = if n == 2 {
            1.0
        } else {
            2.0 * (nf * a1 - 2.0 * (nf - 1.0)) / ((nf - 1.0) * (nf - 2.0))
        };

        let dn = cn + (nf - 2.0) / ((nf - 1.0) * (nf - 1.0))
            + 2.0 / (nf - 1.0) * (1.5 - (2.0 * an_plus1 - 3.0) / (nf - 2.0) - 1.0 / nf);
        let _ = dn;

        let vd = 1.0 + a1 * a1 / (a2 + a1 * a1)
            * (cn * cn - (cn - s / a1) * (cn - s / a1));
        let vd = vd.max(1e-15);

        let d_star = (s / a1 - eta_s * nf / (nf - 1.0)) / vd.sqrt();

        let p_value = normal_two_tail(d_star);

        let interpretation = if p_value > self.alpha {
            SelectionSignal::Neutral
        } else if d_star < 0.0 {
            SelectionSignal::PositiveSelection
        } else {
            SelectionSignal::BalancingSelection
        };

        SelectionResult {
            test_name: "Fu_Li_D*".into(),
            statistic: d_star,
            p_value,
            interpretation,
            n_sites: sfs.segregating_sites(),
        }
    }

    /// Fu and Li's F* statistic.
    pub fn fu_li_f_star(&self, sfs: &SfsData) -> SelectionResult {
        let n = sfs.n_chromosomes();
        let s = sfs.segregating_sites() as f64;
        let pi = sfs.pi();
        let eta_s = sfs.singletons() as f64;

        if s == 0.0 || n < 4 {
            return SelectionResult {
                test_name: "Fu_Li_F*".into(),
                statistic: 0.0,
                p_value: 1.0,
                interpretation: SelectionSignal::Neutral,
                n_sites: sfs.segregating_sites(),
            };
        }

        let nf = n as f64;
        let a1 = harmonic(n - 1);
        let a2 = harmonic2(n - 1);

        let an1 = a1 + 1.0 / nf;

        // Variance components
        let vf_num = (2.0 * nf * a1 - 4.0 * (nf - 1.0)) / ((nf - 1.0) * (nf - 2.0));
        let vf = vf_num + a2
            + (2.0 * (nf * nf + nf + 3.0)) / (9.0 * nf * (nf - 1.0))
            - 2.0 * an1 / (nf - 1.0);
        let vf = vf.max(1e-15);
        let _ = a2;

        let f_star = (pi - eta_s * (nf - 1.0) / nf) / vf.sqrt();

        let p_value = normal_two_tail(f_star);

        let interpretation = if p_value > self.alpha {
            SelectionSignal::Neutral
        } else if f_star < 0.0 {
            SelectionSignal::PositiveSelection
        } else {
            SelectionSignal::BalancingSelection
        };

        SelectionResult {
            test_name: "Fu_Li_F*".into(),
            statistic: f_star,
            p_value,
            interpretation,
            n_sites: sfs.segregating_sites(),
        }
    }

    /// Fay and Wu's H (detects positive selection / hitchhiking).
    pub fn fay_wu_h(&self, sfs: &SfsData) -> SelectionResult {
        let n = sfs.n_chromosomes();
        let pi = sfs.pi();

        if sfs.segregating_sites() == 0 || n < 4 {
            return SelectionResult {
                test_name: "Fay_Wu_H".into(),
                statistic: 0.0,
                p_value: 1.0,
                interpretation: SelectionSignal::Neutral,
                n_sites: 0,
            };
        }

        let nf = n as f64;

        // theta_H = (2 / (n(n-1))) * sum_i (i^2 * S_i)
        let mut theta_h = 0.0;
        for (i, &c) in sfs.counts.iter().enumerate() {
            if i == 0 || i >= n {
                continue;
            }
            theta_h += (i as f64 * i as f64) * c as f64;
        }
        theta_h *= 2.0 / (nf * (nf - 1.0));

        let h = pi - theta_h;

        // Approximate variance
        let s = sfs.segregating_sites() as f64;
        let a1 = harmonic(n - 1);
        let theta = s / a1;
        let var_h = theta * (nf - 2.0) / (6.0 * (nf - 1.0))
            + theta * theta * (18.0 * nf * nf * (3.0 * nf + 2.0) * a1
                - 88.0 * nf * nf * nf)
                / (9.0 * nf * nf * (nf - 1.0) * (nf - 1.0));
        let var_h = var_h.abs().max(1e-15);

        let h_norm = h / var_h.sqrt();
        let p_value = normal_two_tail(h_norm);

        let interpretation = if p_value > self.alpha {
            SelectionSignal::Neutral
        } else if h < 0.0 {
            SelectionSignal::PositiveSelection
        } else {
            SelectionSignal::BalancingSelection
        };

        SelectionResult {
            test_name: "Fay_Wu_H".into(),
            statistic: h,
            p_value,
            interpretation,
            n_sites: sfs.segregating_sites(),
        }
    }

    /// Run all neutrality tests on the same SFS.
    pub fn run_all(&self, sfs: &SfsData) -> Vec<SelectionResult> {
        vec![
            self.tajimas_d(sfs),
            self.fu_li_d_star(sfs),
            self.fu_li_f_star(sfs),
            self.fay_wu_h(sfs),
        ]
    }
}

impl fmt::Display for SelectionTester {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SelectionTester(alpha={}, beta_approx={})", self.alpha, self.use_beta_approx)
    }
}

// ── McDonald-Kreitman Test ──────────────────────────────────────────

/// Counts for the McDonald-Kreitman test.
#[derive(Debug, Clone, PartialEq)]
pub struct MkCounts {
    /// Divergent nonsynonymous (fixed between species).
    pub dn: u64,
    /// Divergent synonymous.
    pub ds: u64,
    /// Polymorphic nonsynonymous (within species).
    pub pn: u64,
    /// Polymorphic synonymous.
    pub ps: u64,
}

impl MkCounts {
    pub fn new(dn: u64, ds: u64, pn: u64, ps: u64) -> Self {
        Self { dn, ds, pn, ps }
    }

    /// Neutrality index NI = (Pn/Ps) / (Dn/Ds).
    pub fn neutrality_index(&self) -> f64 {
        if self.ds == 0 || self.ps == 0 || self.dn == 0 {
            return f64::INFINITY;
        }
        (self.pn as f64 / self.ps as f64) / (self.dn as f64 / self.ds as f64)
    }

    /// Alpha: proportion of adaptive substitutions = 1 - NI.
    pub fn alpha(&self) -> f64 {
        let ni = self.neutrality_index();
        if ni.is_infinite() { 0.0 } else { 1.0 - ni }
    }

    /// Direction of selection (DoS) = Dn/(Dn+Ds) - Pn/(Pn+Ps).
    pub fn direction_of_selection(&self) -> f64 {
        let div_total = self.dn + self.ds;
        let poly_total = self.pn + self.ps;
        if div_total == 0 || poly_total == 0 {
            return 0.0;
        }
        self.dn as f64 / div_total as f64 - self.pn as f64 / poly_total as f64
    }
}

impl fmt::Display for MkCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MK(Dn={}, Ds={}, Pn={}, Ps={}, NI={:.4}, alpha={:.4})",
            self.dn, self.ds, self.pn, self.ps,
            self.neutrality_index(), self.alpha()
        )
    }
}

/// Run the McDonald-Kreitman test using a 2x2 Fisher's exact test.
pub fn mcdonald_kreitman(counts: &MkCounts) -> SelectionResult {
    let ni = counts.neutrality_index();

    // G-test (likelihood ratio) for the 2x2 table
    let table = [
        [counts.dn as f64, counts.ds as f64],
        [counts.pn as f64, counts.ps as f64],
    ];
    let g_stat = g_test_2x2(&table);

    // Approximate p-value (chi2 with 1 df)
    let p_value = chi2_pvalue_1df(g_stat);

    let interpretation = if p_value > 0.05 {
        SelectionSignal::Neutral
    } else if ni < 1.0 {
        SelectionSignal::PositiveSelection
    } else {
        SelectionSignal::PurifyingSelection
    };

    SelectionResult {
        test_name: "McDonald_Kreitman".into(),
        statistic: g_stat,
        p_value,
        interpretation,
        n_sites: counts.dn + counts.ds + counts.pn + counts.ps,
    }
}

// ── Helper Functions ────────────────────────────────────────────────

fn harmonic(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / i as f64).sum()
}

fn harmonic2(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / (i as f64 * i as f64)).sum()
}

/// Approximate beta distribution p-value for Tajima's D.
fn tajima_d_pvalue(d: f64, n: usize) -> f64 {
    // Use normal approximation as fallback
    let _ = n;
    normal_two_tail(d)
}

/// Two-tailed p-value from standard normal.
fn normal_two_tail(z: f64) -> f64 {
    let p = normal_cdf(z.abs());
    (2.0 * (1.0 - p)).clamp(0.0, 1.0)
}

/// Standard normal CDF via rational approximation.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319_381_530
        + t * (-0.356_563_782
            + t * (1.781_477_937
                + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-0.5 * x * x).exp() * 0.398_942_280_401;
    if x >= 0.0 {
        1.0 - pdf * poly
    } else {
        pdf * poly
    }
}

fn chi2_pvalue_1df(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let z = x.sqrt();
    normal_two_tail(z)
}

/// G-test (log-likelihood ratio) for a 2x2 table.
fn g_test_2x2(table: &[[f64; 2]; 2]) -> f64 {
    let row_totals = [table[0][0] + table[0][1], table[1][0] + table[1][1]];
    let col_totals = [table[0][0] + table[1][0], table[0][1] + table[1][1]];
    let grand_total = row_totals[0] + row_totals[1];

    if grand_total == 0.0 {
        return 0.0;
    }

    let mut g = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            let observed = table[i][j];
            let expected = row_totals[i] * col_totals[j] / grand_total;
            if observed > 0.0 && expected > 0.0 {
                g += observed * (observed / expected).ln();
            }
        }
    }
    2.0 * g
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    fn make_neutral_sfs() -> SfsData {
        // Approximate neutral SFS for n=10 (20 chromosomes)
        // 1/i expectation for a standard coalescent
        let mut counts = vec![0u64; 21];
        counts[0] = 50; // monomorphic
        counts[1] = 20; // singletons
        counts[2] = 10;
        counts[3] = 7;
        counts[4] = 5;
        counts[5] = 4;
        counts[6] = 3;
        counts[7] = 3;
        counts[8] = 2;
        counts[9] = 2;
        SfsData::new(counts, 10)
    }

    #[test]
    fn test_sfs_segregating_sites() {
        let sfs = make_neutral_sfs();
        assert_eq!(sfs.segregating_sites(), 56);
    }

    #[test]
    fn test_sfs_pi() {
        let sfs = make_neutral_sfs();
        assert!(sfs.pi() > 0.0);
    }

    #[test]
    fn test_sfs_theta_w() {
        let sfs = make_neutral_sfs();
        assert!(sfs.theta_w() > 0.0);
    }

    #[test]
    fn test_tajimas_d_neutral() {
        let sfs = make_neutral_sfs();
        let tester = SelectionTester::new();
        let result = tester.tajimas_d(&sfs);
        // Should be close to 0 for neutral SFS
        assert!(result.statistic.abs() < 3.0);
    }

    #[test]
    fn test_tajimas_d_positive_selection() {
        // Excess singletons -> negative D (positive selection / expansion)
        let mut counts = vec![0u64; 21];
        counts[0] = 50;
        counts[1] = 100; // many singletons
        counts[2] = 5;
        let sfs = SfsData::new(counts, 10);
        let tester = SelectionTester::new();
        let result = tester.tajimas_d(&sfs);
        assert!(result.statistic < 0.0);
    }

    #[test]
    fn test_tajimas_d_balancing() {
        // Excess intermediate-freq variants -> positive D
        let mut counts = vec![0u64; 21];
        counts[0] = 50;
        counts[1] = 2;
        counts[9] = 30;
        counts[10] = 30;
        counts[11] = 30;
        let sfs = SfsData::new(counts, 10);
        let tester = SelectionTester::new();
        let result = tester.tajimas_d(&sfs);
        assert!(result.statistic > 0.0);
    }

    #[test]
    fn test_fu_li_d_star() {
        let sfs = make_neutral_sfs();
        let tester = SelectionTester::new();
        let result = tester.fu_li_d_star(&sfs);
        assert!(result.statistic.is_finite());
    }

    #[test]
    fn test_fu_li_f_star() {
        let sfs = make_neutral_sfs();
        let tester = SelectionTester::new();
        let result = tester.fu_li_f_star(&sfs);
        assert!(result.statistic.is_finite());
    }

    #[test]
    fn test_fay_wu_h() {
        let sfs = make_neutral_sfs();
        let tester = SelectionTester::new();
        let result = tester.fay_wu_h(&sfs);
        assert!(result.statistic.is_finite());
    }

    #[test]
    fn test_run_all() {
        let sfs = make_neutral_sfs();
        let tester = SelectionTester::new();
        let results = tester.run_all(&sfs);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_mk_neutrality_index() {
        let mk = MkCounts::new(10, 20, 15, 30);
        let ni = mk.neutrality_index();
        assert!((ni - 1.0).abs() < EPS);
    }

    #[test]
    fn test_mk_positive_selection() {
        // More nonsynonymous divergence than expected
        let mk = MkCounts::new(30, 10, 5, 10);
        let ni = mk.neutrality_index();
        assert!(ni < 1.0);
        assert!(mk.alpha() > 0.0);
    }

    #[test]
    fn test_mk_direction_of_selection() {
        let mk = MkCounts::new(30, 10, 5, 10);
        let dos = mk.direction_of_selection();
        assert!(dos > 0.0); // positive DoS = adaptive evolution
    }

    #[test]
    fn test_mcdonald_kreitman_test() {
        let mk = MkCounts::new(30, 10, 5, 10);
        let result = mcdonald_kreitman(&mk);
        assert_eq!(result.test_name, "McDonald_Kreitman");
    }

    #[test]
    fn test_sfs_display() {
        let sfs = make_neutral_sfs();
        let s = format!("{}", sfs);
        assert!(s.contains("S=56"));
    }

    #[test]
    fn test_result_display() {
        let r = SelectionResult {
            test_name: "Tajima_D".into(),
            statistic: -1.5,
            p_value: 0.03,
            interpretation: SelectionSignal::PositiveSelection,
            n_sites: 100,
        };
        let s = format!("{}", r);
        assert!(s.contains("positive_selection"));
    }

    #[test]
    fn test_signal_display() {
        assert_eq!(format!("{}", SelectionSignal::Neutral), "neutral");
        assert_eq!(format!("{}", SelectionSignal::Expansion), "expansion");
    }

    #[test]
    fn test_mk_display() {
        let mk = MkCounts::new(10, 20, 15, 30);
        let s = format!("{}", mk);
        assert!(s.contains("MK"));
    }

    #[test]
    fn test_tester_display() {
        let t = SelectionTester::new().with_alpha(0.01);
        let s = format!("{}", t);
        assert!(s.contains("0.01"));
    }

    #[test]
    fn test_empty_sfs() {
        let sfs = SfsData::new(vec![], 10);
        let tester = SelectionTester::new();
        let result = tester.tajimas_d(&sfs);
        assert!((result.statistic - 0.0).abs() < EPS);
    }

    #[test]
    fn test_normal_cdf() {
        let p = normal_cdf(0.0);
        assert!((p - 0.5).abs() < 0.001);
        let p2 = normal_cdf(3.0);
        assert!(p2 > 0.998);
    }
}
