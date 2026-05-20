//! Weight pruning: magnitude-based, structured, unstructured, lottery ticket
//! hypothesis, and gradual pruning schedules.
//!
//! Reduces model size by zeroing out low-importance weights. Supports
//! both unstructured (individual weight) and structured (entire filter/row)
//! pruning with configurable schedules for gradual sparsification.

use std::collections::HashMap;
use std::fmt;

// ── Sparsity Metric ────────────────────────────────────────────

/// Compute sparsity ratio (fraction of zeros) in a weight slice.
pub fn sparsity_ratio(weights: &[f64]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }
    let zeros = weights.iter().filter(|&&w| w == 0.0).count();
    zeros as f64 / weights.len() as f64
}

/// Count non-zero elements.
pub fn nnz(weights: &[f64]) -> usize {
    weights.iter().filter(|&&w| w != 0.0).count()
}

// ── Pruning Method ─────────────────────────────────────────────

/// Method for selecting which weights to prune.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningMethod {
    /// Remove weights with smallest absolute value.
    Magnitude,
    /// Remove entire rows/filters (structured).
    StructuredRow,
    /// Remove entire columns (structured).
    StructuredColumn,
    /// Random pruning (baseline comparison).
    Random { seed: u64 },
    /// Movement-based: prune weights that moved least during training.
    Movement,
}

impl fmt::Display for PruningMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PruningMethod::Magnitude => write!(f, "magnitude"),
            PruningMethod::StructuredRow => write!(f, "structured-row"),
            PruningMethod::StructuredColumn => write!(f, "structured-column"),
            PruningMethod::Random { seed } => write!(f, "random(seed={seed})"),
            PruningMethod::Movement => write!(f, "movement"),
        }
    }
}

// ── Pruning Schedule ───────────────────────────────────────────

/// Schedule for gradually increasing sparsity over training steps.
#[derive(Debug, Clone, PartialEq)]
pub enum PruningSchedule {
    /// Apply target sparsity in one shot.
    OneShot,
    /// Linearly ramp from initial to target sparsity.
    Linear {
        start_step: u64,
        end_step: u64,
        initial_sparsity: f64,
        target_sparsity: f64,
    },
    /// Polynomial (cubic) schedule from initial to target.
    Polynomial {
        start_step: u64,
        end_step: u64,
        initial_sparsity: f64,
        target_sparsity: f64,
        power: f64,
    },
    /// Step-wise: increase sparsity at fixed intervals.
    StepWise {
        steps: Vec<(u64, f64)>,
    },
}

impl PruningSchedule {
    /// Compute the target sparsity at a given training step.
    pub fn sparsity_at(&self, step: u64) -> f64 {
        match self {
            PruningSchedule::OneShot => 1.0, // caller provides the ratio
            PruningSchedule::Linear {
                start_step,
                end_step,
                initial_sparsity,
                target_sparsity,
            } => {
                if step <= *start_step {
                    return *initial_sparsity;
                }
                if step >= *end_step {
                    return *target_sparsity;
                }
                let t = (step - start_step) as f64 / (end_step - start_step) as f64;
                initial_sparsity + t * (target_sparsity - initial_sparsity)
            }
            PruningSchedule::Polynomial {
                start_step,
                end_step,
                initial_sparsity,
                target_sparsity,
                power,
            } => {
                if step <= *start_step {
                    return *initial_sparsity;
                }
                if step >= *end_step {
                    return *target_sparsity;
                }
                let t = (step - start_step) as f64 / (end_step - start_step) as f64;
                let factor = 1.0 - (1.0 - t).powf(*power);
                initial_sparsity + factor * (target_sparsity - initial_sparsity)
            }
            PruningSchedule::StepWise { steps } => {
                let mut current = 0.0;
                for &(s, ratio) in steps {
                    if step >= s {
                        current = ratio;
                    } else {
                        break;
                    }
                }
                current
            }
        }
    }
}

impl fmt::Display for PruningSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PruningSchedule::OneShot => write!(f, "one-shot"),
            PruningSchedule::Linear { target_sparsity, .. } => {
                write!(f, "linear(target={target_sparsity:.2})")
            }
            PruningSchedule::Polynomial { target_sparsity, power, .. } => {
                write!(f, "polynomial(target={target_sparsity:.2}, pow={power})")
            }
            PruningSchedule::StepWise { steps } => {
                write!(f, "step-wise({} steps)", steps.len())
            }
        }
    }
}

// ── Pruning Mask ───────────────────────────────────────────────

/// Binary mask: `true` means the weight is kept, `false` means pruned.
#[derive(Debug, Clone)]
pub struct PruningMask {
    pub shape: Vec<usize>,
    pub mask: Vec<bool>,
}

impl PruningMask {
    /// Create an all-ones (no pruning) mask.
    pub fn ones(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        Self { shape: shape.to_vec(), mask: vec![true; len] }
    }

    /// Create mask by magnitude pruning at given ratio.
    pub fn from_magnitude(weights: &[f64], shape: &[usize], ratio: f64) -> Self {
        let n = weights.len();
        let num_prune = (n as f64 * ratio).round() as usize;

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            weights[a]
                .abs()
                .partial_cmp(&weights[b].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut mask = vec![true; n];
        for &idx in indices.iter().take(num_prune) {
            mask[idx] = false;
        }

        Self { shape: shape.to_vec(), mask }
    }

    /// Create a structured row mask: prune rows with smallest L2 norm.
    pub fn from_structured_row(weights: &[f64], rows: usize, cols: usize, ratio: f64) -> Self {
        let num_prune_rows = (rows as f64 * ratio).round() as usize;

        let mut row_norms: Vec<(usize, f64)> = (0..rows)
            .map(|r| {
                let start = r * cols;
                let end = start + cols;
                let norm: f64 = weights[start..end].iter().map(|v| v * v).sum::<f64>().sqrt();
                (r, norm)
            })
            .collect();

        row_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut mask = vec![true; rows * cols];
        for &(row, _) in row_norms.iter().take(num_prune_rows) {
            for c in 0..cols {
                mask[row * cols + c] = false;
            }
        }

        Self { shape: vec![rows, cols], mask }
    }

    /// Create a structured column mask: prune columns with smallest L2 norm.
    pub fn from_structured_column(weights: &[f64], rows: usize, cols: usize, ratio: f64) -> Self {
        let num_prune_cols = (cols as f64 * ratio).round() as usize;

        let mut col_norms: Vec<(usize, f64)> = (0..cols)
            .map(|c| {
                let norm: f64 = (0..rows).map(|r| {
                    let v = weights[r * cols + c];
                    v * v
                }).sum::<f64>().sqrt();
                (c, norm)
            })
            .collect();

        col_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut mask = vec![true; rows * cols];
        for &(col, _) in col_norms.iter().take(num_prune_cols) {
            for r in 0..rows {
                mask[r * cols + col] = false;
            }
        }

        Self { shape: vec![rows, cols], mask }
    }

    /// Apply the mask to weights (zero out pruned positions).
    pub fn apply(&self, weights: &mut [f64]) {
        for (i, &keep) in self.mask.iter().enumerate() {
            if !keep {
                weights[i] = 0.0;
            }
        }
    }

    /// Achieved sparsity ratio.
    pub fn sparsity(&self) -> f64 {
        if self.mask.is_empty() {
            return 0.0;
        }
        let pruned = self.mask.iter().filter(|&&m| !m).count();
        pruned as f64 / self.mask.len() as f64
    }

    /// Number of remaining (non-pruned) weights.
    pub fn remaining(&self) -> usize {
        self.mask.iter().filter(|&&m| m).count()
    }
}

impl fmt::Display for PruningMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PruningMask(shape={:?}, sparsity={:.2}%, remaining={})",
            self.shape,
            self.sparsity() * 100.0,
            self.remaining()
        )
    }
}

// ── Lottery Ticket ─────────────────────────────────────────────

/// Lottery ticket state: stores initial weights for rewinding.
#[derive(Debug, Clone)]
pub struct LotteryTicket {
    /// Original (initialisation) weights.
    pub initial_weights: Vec<f64>,
    /// Current mask discovered through iterative pruning.
    pub mask: PruningMask,
    /// Current pruning round.
    pub round: u32,
    /// Target sparsity per round.
    pub per_round_ratio: f64,
}

impl LotteryTicket {
    pub fn new(initial_weights: Vec<f64>, shape: &[usize], per_round_ratio: f64) -> Self {
        Self {
            mask: PruningMask::ones(shape),
            initial_weights,
            round: 0,
            per_round_ratio,
        }
    }

    /// After training, prune and rewind to initial weights for next round.
    pub fn prune_and_rewind(&mut self, trained_weights: &[f64]) -> Vec<f64> {
        // Apply existing mask to trained weights, then compute new mask on unmasked subset
        let mut masked_abs: Vec<(usize, f64)> = trained_weights
            .iter()
            .enumerate()
            .filter(|(i, _)| self.mask.mask[*i])
            .map(|(i, &w)| (i, w.abs()))
            .collect();

        masked_abs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let num_to_prune = (masked_abs.len() as f64 * self.per_round_ratio).round() as usize;
        for &(idx, _) in masked_abs.iter().take(num_to_prune) {
            self.mask.mask[idx] = false;
        }

        self.round += 1;

        // Rewind: return initial weights with new mask applied
        let mut rewound = self.initial_weights.clone();
        self.mask.apply(&mut rewound);
        rewound
    }

    /// Current overall sparsity.
    pub fn sparsity(&self) -> f64 {
        self.mask.sparsity()
    }
}

impl fmt::Display for LotteryTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LotteryTicket(round={}, sparsity={:.2}%)",
            self.round,
            self.sparsity() * 100.0
        )
    }
}

// ── Pruning Config ─────────────────────────────────────────────

/// Configuration for the pruning pipeline.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    pub method: PruningMethod,
    pub schedule: PruningSchedule,
    pub target_sparsity: f64,
    pub layer_configs: HashMap<String, f64>,
}

impl PruningConfig {
    pub fn new(method: PruningMethod, target_sparsity: f64) -> Self {
        Self {
            method,
            schedule: PruningSchedule::OneShot,
            target_sparsity,
            layer_configs: HashMap::new(),
        }
    }

    pub fn with_schedule(mut self, schedule: PruningSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub fn with_layer_sparsity(mut self, layer: impl Into<String>, sparsity: f64) -> Self {
        self.layer_configs.insert(layer.into(), sparsity);
        self
    }

    /// Get the sparsity target for a specific layer (falls back to global).
    pub fn sparsity_for(&self, layer: &str) -> f64 {
        self.layer_configs.get(layer).copied().unwrap_or(self.target_sparsity)
    }
}

impl fmt::Display for PruningConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PruningConfig(method={}, target={:.2}%, schedule={})",
            self.method,
            self.target_sparsity * 100.0,
            self.schedule
        )
    }
}

// ── Pruning Report ─────────────────────────────────────────────

/// Summary report after pruning a model.
#[derive(Debug, Clone)]
pub struct PruningReport {
    pub layers: Vec<LayerPruneInfo>,
    pub total_params: usize,
    pub total_remaining: usize,
}

/// Per-layer pruning information.
#[derive(Debug, Clone)]
pub struct LayerPruneInfo {
    pub name: String,
    pub params: usize,
    pub remaining: usize,
    pub sparsity: f64,
}

impl PruningReport {
    pub fn new() -> Self {
        Self { layers: Vec::new(), total_params: 0, total_remaining: 0 }
    }

    pub fn add_layer(&mut self, name: impl Into<String>, params: usize, remaining: usize) {
        let sparsity = if params == 0 { 0.0 } else { 1.0 - remaining as f64 / params as f64 };
        self.layers.push(LayerPruneInfo {
            name: name.into(),
            params,
            remaining,
            sparsity,
        });
        self.total_params += params;
        self.total_remaining += remaining;
    }

    pub fn overall_sparsity(&self) -> f64 {
        if self.total_params == 0 {
            return 0.0;
        }
        1.0 - self.total_remaining as f64 / self.total_params as f64
    }
}

impl Default for PruningReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PruningReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PruningReport(layers={}, params={}, remaining={}, sparsity={:.2}%)",
            self.layers.len(),
            self.total_params,
            self.total_remaining,
            self.overall_sparsity() * 100.0
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparsity_ratio_all_zeros() {
        assert!((sparsity_ratio(&[0.0, 0.0, 0.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_ratio_no_zeros() {
        assert!((sparsity_ratio(&[1.0, -1.0, 0.5])).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_ratio_mixed() {
        assert!((sparsity_ratio(&[0.0, 1.0, 0.0, 2.0]) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_nnz() {
        assert_eq!(nnz(&[0.0, 1.0, 0.0, 2.0, 0.0]), 2);
    }

    #[test]
    fn test_magnitude_mask_50pct() {
        let weights = vec![0.1, -0.5, 0.3, -0.2, 0.8, -0.05];
        let mask = PruningMask::from_magnitude(&weights, &[6], 0.5);
        // 3 smallest: |0.05|, |0.1|, |0.2|
        assert!(!mask.mask[5]); // -0.05
        assert!(!mask.mask[0]); // 0.1
        assert!(!mask.mask[3]); // -0.2
        assert!(mask.mask[1]);  // -0.5 kept
        assert!(mask.mask[4]);  // 0.8 kept
    }

    #[test]
    fn test_mask_apply() {
        let mut weights = vec![1.0, 2.0, 3.0, 4.0];
        let mask = PruningMask::from_magnitude(&weights, &[4], 0.5);
        mask.apply(&mut weights);
        assert_eq!(nnz(&weights), 2);
    }

    #[test]
    fn test_mask_sparsity() {
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let mask = PruningMask::from_magnitude(&weights, &[4], 0.5);
        assert!((mask.sparsity() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_structured_row_pruning() {
        // 3 rows x 2 cols; row norms: sqrt(1+4)=2.24, sqrt(9+16)=5, sqrt(25+36)=~7.8
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = PruningMask::from_structured_row(&weights, 3, 2, 0.34);
        // Should prune 1 row (smallest norm = row 0)
        assert!(!mask.mask[0]);
        assert!(!mask.mask[1]);
        assert!(mask.mask[2]); // row 1 kept
        assert!(mask.mask[4]); // row 2 kept
    }

    #[test]
    fn test_structured_column_pruning() {
        // 2 rows x 3 cols
        let weights = vec![0.1, 5.0, 3.0, 0.2, 4.0, 2.0];
        let mask = PruningMask::from_structured_column(&weights, 2, 3, 0.34);
        // col norms: sqrt(0.01+0.04)=0.22, sqrt(25+16)=6.4, sqrt(9+4)=3.6
        // Prune 1 col: col 0
        assert!(!mask.mask[0]);
        assert!(!mask.mask[3]);
        assert!(mask.mask[1]);
    }

    #[test]
    fn test_linear_schedule() {
        let sched = PruningSchedule::Linear {
            start_step: 100,
            end_step: 200,
            initial_sparsity: 0.0,
            target_sparsity: 0.9,
        };
        assert!((sched.sparsity_at(100) - 0.0).abs() < 1e-10);
        assert!((sched.sparsity_at(150) - 0.45).abs() < 1e-10);
        assert!((sched.sparsity_at(200) - 0.9).abs() < 1e-10);
        assert!((sched.sparsity_at(50) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_schedule() {
        let sched = PruningSchedule::Polynomial {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            target_sparsity: 0.8,
            power: 3.0,
        };
        assert!(sched.sparsity_at(0) < 0.01);
        assert!(sched.sparsity_at(100) > 0.79);
        // Polynomial ramp with formula 1-(1-t)^3: convex curve, at midpoint exceeds linear midpoint
        assert!(sched.sparsity_at(50) > 0.5 * 0.8);
    }

    #[test]
    fn test_stepwise_schedule() {
        let sched = PruningSchedule::StepWise {
            steps: vec![(0, 0.2), (50, 0.5), (100, 0.8)],
        };
        assert!((sched.sparsity_at(0) - 0.2).abs() < 1e-10);
        assert!((sched.sparsity_at(49) - 0.2).abs() < 1e-10);
        assert!((sched.sparsity_at(50) - 0.5).abs() < 1e-10);
        assert!((sched.sparsity_at(100) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_lottery_ticket_rewind() {
        let init = vec![0.5, -0.3, 0.8, -0.1, 0.2];
        let mut lt = LotteryTicket::new(init.clone(), &[5], 0.4);
        assert_eq!(lt.round, 0);

        // Simulate training: weights changed
        let trained = vec![0.6, -0.01, 0.9, -0.02, 0.3];
        let rewound = lt.prune_and_rewind(&trained);
        assert_eq!(lt.round, 1);
        assert!(lt.sparsity() > 0.0);
        // Rewound values come from initial, not trained
        for (i, &r) in rewound.iter().enumerate() {
            if lt.mask.mask[i] {
                assert_eq!(r, init[i]);
            } else {
                assert_eq!(r, 0.0);
            }
        }
    }

    #[test]
    fn test_pruning_config_builder() {
        let cfg = PruningConfig::new(PruningMethod::Magnitude, 0.9)
            .with_schedule(PruningSchedule::OneShot)
            .with_layer_sparsity("fc1", 0.5);
        assert_eq!(cfg.sparsity_for("fc1"), 0.5);
        assert_eq!(cfg.sparsity_for("fc2"), 0.9);
    }

    #[test]
    fn test_pruning_report() {
        let mut report = PruningReport::new();
        report.add_layer("fc1", 1000, 500);
        report.add_layer("fc2", 500, 100);
        assert_eq!(report.total_params, 1500);
        assert_eq!(report.total_remaining, 600);
        assert!((report.overall_sparsity() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_pruning_report_default() {
        let report = PruningReport::default();
        assert_eq!(report.total_params, 0);
        assert_eq!(report.overall_sparsity(), 0.0);
    }

    #[test]
    fn test_display_impls() {
        assert!(format!("{}", PruningMethod::Magnitude).contains("magnitude"));
        assert!(format!("{}", PruningSchedule::OneShot).contains("one-shot"));

        let mask = PruningMask::ones(&[4]);
        assert!(format!("{mask}").contains("0.00%"));

        let lt = LotteryTicket::new(vec![1.0; 4], &[4], 0.5);
        assert!(format!("{lt}").contains("round=0"));

        let cfg = PruningConfig::new(PruningMethod::Magnitude, 0.5);
        assert!(format!("{cfg}").contains("50.00%"));

        let report = PruningReport::new();
        assert!(format!("{report}").contains("PruningReport"));
    }

    #[test]
    fn test_ones_mask_all_kept() {
        let mask = PruningMask::ones(&[3, 4]);
        assert_eq!(mask.remaining(), 12);
        assert_eq!(mask.sparsity(), 0.0);
    }
}
