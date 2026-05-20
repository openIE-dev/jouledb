//! Early stopping for neural network training.
//!
//! Monitors a validation metric during training and halts when the model
//! stops improving, preventing overfitting while preserving the best weights:
//!
//! - [`EarlyStopping`] — core monitor with patience and delta threshold
//! - [`MetricTracker`] — records metric history with smoothing
//! - [`BestModelCheckpoint`] — stores and restores best parameter snapshot
//! - [`TrainingStatus`] — current training phase and termination reason

use std::fmt;

// ── Metric Direction ───────────────────────────────────────────────

/// Whether a lower or higher metric value is better.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricDirection {
    /// Lower is better (e.g., loss, error rate).
    Minimize,
    /// Higher is better (e.g., accuracy, F1 score).
    Maximize,
}

impl MetricDirection {
    /// Returns `true` if `candidate` is an improvement over `current`.
    pub fn is_improvement(self, candidate: f64, current: f64, min_delta: f64) -> bool {
        match self {
            MetricDirection::Minimize => candidate < current - min_delta,
            MetricDirection::Maximize => candidate > current + min_delta,
        }
    }
}

impl fmt::Display for MetricDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricDirection::Minimize => write!(f, "minimize"),
            MetricDirection::Maximize => write!(f, "maximize"),
        }
    }
}

// ── Training Status ────────────────────────────────────────────────

/// Current state of the training loop.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingStatus {
    /// Training is ongoing and improving.
    Improving,
    /// No improvement seen for some epochs, but patience not exhausted.
    Stalled { epochs_without_improvement: u64 },
    /// Training should stop — patience exhausted.
    StopPatience,
    /// Training should stop — metric diverged (NaN or extreme).
    StopDiverged,
    /// Training should stop — reached maximum epochs.
    StopMaxEpochs,
}

impl TrainingStatus {
    pub fn should_stop(&self) -> bool {
        matches!(
            self,
            TrainingStatus::StopPatience
                | TrainingStatus::StopDiverged
                | TrainingStatus::StopMaxEpochs
        )
    }
}

impl fmt::Display for TrainingStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainingStatus::Improving => write!(f, "improving"),
            TrainingStatus::Stalled { epochs_without_improvement } => {
                write!(f, "stalled ({epochs_without_improvement} epochs)")
            }
            TrainingStatus::StopPatience => write!(f, "stopped (patience)"),
            TrainingStatus::StopDiverged => write!(f, "stopped (diverged)"),
            TrainingStatus::StopMaxEpochs => write!(f, "stopped (max epochs)"),
        }
    }
}

// ── Early Stopping ─────────────────────────────────────────────────

/// Monitors a validation metric and signals when training should stop.
///
/// The monitor tracks the best observed metric value and counts consecutive
/// epochs without improvement. When the counter exceeds `patience`, it
/// signals to stop training.
pub struct EarlyStopping {
    patience: u64,
    min_delta: f64,
    direction: MetricDirection,
    best_value: f64,
    best_epoch: u64,
    epochs_no_improve: u64,
    current_epoch: u64,
    max_epochs: Option<u64>,
    divergence_threshold: Option<f64>,
}

impl EarlyStopping {
    pub fn new(patience: u64, direction: MetricDirection) -> Self {
        let best_value = match direction {
            MetricDirection::Minimize => f64::INFINITY,
            MetricDirection::Maximize => f64::NEG_INFINITY,
        };
        Self {
            patience,
            min_delta: 0.0,
            direction,
            best_value,
            best_epoch: 0,
            epochs_no_improve: 0,
            current_epoch: 0,
            max_epochs: None,
            divergence_threshold: None,
        }
    }

    pub fn with_min_delta(mut self, delta: f64) -> Self {
        self.min_delta = delta.abs();
        self
    }

    pub fn with_max_epochs(mut self, max: u64) -> Self {
        self.max_epochs = Some(max);
        self
    }

    pub fn with_divergence_threshold(mut self, threshold: f64) -> Self {
        self.divergence_threshold = Some(threshold);
        self
    }

    pub fn with_patience(mut self, patience: u64) -> Self {
        self.patience = patience;
        self
    }

    /// Report a new metric value for the current epoch.
    /// Returns the resulting [`TrainingStatus`].
    pub fn step(&mut self, metric_value: f64) -> TrainingStatus {
        self.current_epoch += 1;

        // Check for divergence
        if metric_value.is_nan() || metric_value.is_infinite() {
            return TrainingStatus::StopDiverged;
        }
        if let Some(threshold) = self.divergence_threshold {
            if metric_value.abs() > threshold {
                return TrainingStatus::StopDiverged;
            }
        }

        // Check for improvement
        if self.direction.is_improvement(metric_value, self.best_value, self.min_delta) {
            self.best_value = metric_value;
            self.best_epoch = self.current_epoch;
            self.epochs_no_improve = 0;

            // Check max epochs even on improvement
            if let Some(max) = self.max_epochs {
                if self.current_epoch >= max {
                    return TrainingStatus::StopMaxEpochs;
                }
            }
            return TrainingStatus::Improving;
        }

        self.epochs_no_improve += 1;

        // Check max epochs
        if let Some(max) = self.max_epochs {
            if self.current_epoch >= max {
                return TrainingStatus::StopMaxEpochs;
            }
        }

        // Check patience
        if self.epochs_no_improve >= self.patience {
            TrainingStatus::StopPatience
        } else {
            TrainingStatus::Stalled {
                epochs_without_improvement: self.epochs_no_improve,
            }
        }
    }

    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    pub fn best_epoch(&self) -> u64 {
        self.best_epoch
    }

    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    pub fn epochs_without_improvement(&self) -> u64 {
        self.epochs_no_improve
    }

    pub fn patience(&self) -> u64 {
        self.patience
    }

    pub fn reset(&mut self) {
        self.best_value = match self.direction {
            MetricDirection::Minimize => f64::INFINITY,
            MetricDirection::Maximize => f64::NEG_INFINITY,
        };
        self.best_epoch = 0;
        self.epochs_no_improve = 0;
        self.current_epoch = 0;
    }
}

impl fmt::Display for EarlyStopping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EarlyStopping(patience={}, {}, best={:.6} @epoch {}, no_improve={})",
            self.patience,
            self.direction,
            self.best_value,
            self.best_epoch,
            self.epochs_no_improve
        )
    }
}

// ── Metric Tracker ─────────────────────────────────────────────────

/// Records metric history with optional exponential moving average smoothing.
pub struct MetricTracker {
    name: String,
    history: Vec<f64>,
    smoothed_history: Vec<f64>,
    smoothing_factor: f64,
    ema_value: f64,
    ema_initialized: bool,
}

impl MetricTracker {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            history: Vec::new(),
            smoothed_history: Vec::new(),
            smoothing_factor: 0.0,
            ema_value: 0.0,
            ema_initialized: false,
        }
    }

    pub fn with_smoothing(mut self, factor: f64) -> Self {
        self.smoothing_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Record a new metric value.
    pub fn record(&mut self, value: f64) {
        self.history.push(value);

        if self.smoothing_factor > 0.0 {
            if !self.ema_initialized {
                self.ema_value = value;
                self.ema_initialized = true;
            } else {
                self.ema_value = self.smoothing_factor * self.ema_value
                    + (1.0 - self.smoothing_factor) * value;
            }
            self.smoothed_history.push(self.ema_value);
        }
    }

    pub fn latest(&self) -> Option<f64> {
        self.history.last().copied()
    }

    pub fn smoothed_latest(&self) -> Option<f64> {
        self.smoothed_history.last().copied()
    }

    pub fn history(&self) -> &[f64] {
        &self.history
    }

    pub fn smoothed_history(&self) -> &[f64] {
        &self.smoothed_history
    }

    pub fn count(&self) -> usize {
        self.history.len()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Mean of all recorded values.
    pub fn mean(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().sum::<f64>() / self.history.len() as f64
    }

    /// Standard deviation of all recorded values.
    pub fn std_dev(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance =
            self.history.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / self.history.len() as f64;
        variance.sqrt()
    }

    /// Best (min or max depending on direction) value seen.
    pub fn best(&self, direction: MetricDirection) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        match direction {
            MetricDirection::Minimize => {
                self.history.iter().cloned().reduce(f64::min)
            }
            MetricDirection::Maximize => {
                self.history.iter().cloned().reduce(f64::max)
            }
        }
    }

    /// Detect if the metric is plateauing over the last `window` entries.
    pub fn is_plateau(&self, window: usize, threshold: f64) -> bool {
        if self.history.len() < window {
            return false;
        }
        let tail = &self.history[self.history.len() - window..];
        let min_val = tail.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = tail.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (max_val - min_val).abs() < threshold
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.smoothed_history.clear();
        self.ema_value = 0.0;
        self.ema_initialized = false;
    }
}

impl fmt::Display for MetricTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MetricTracker(\"{}\", n={}, latest={:.6})",
            self.name,
            self.history.len(),
            self.latest().unwrap_or(f64::NAN)
        )
    }
}

// ── Best Model Checkpoint ──────────────────────────────────────────

/// Stores a snapshot of model parameters at the best validation metric.
pub struct BestModelCheckpoint {
    direction: MetricDirection,
    best_params: Option<Vec<f64>>,
    best_metric: f64,
    best_epoch: u64,
    save_count: u64,
}

impl BestModelCheckpoint {
    pub fn new(direction: MetricDirection) -> Self {
        let best_metric = match direction {
            MetricDirection::Minimize => f64::INFINITY,
            MetricDirection::Maximize => f64::NEG_INFINITY,
        };
        Self {
            direction,
            best_params: None,
            best_metric,
            best_epoch: 0,
            save_count: 0,
        }
    }

    pub fn with_direction(mut self, direction: MetricDirection) -> Self {
        self.direction = direction;
        self.best_metric = match direction {
            MetricDirection::Minimize => f64::INFINITY,
            MetricDirection::Maximize => f64::NEG_INFINITY,
        };
        self
    }

    /// Check if the current metric is the best seen, and if so, save params.
    /// Returns `true` if a new checkpoint was saved.
    pub fn update(&mut self, metric: f64, params: &[f64], epoch: u64) -> bool {
        if self.direction.is_improvement(metric, self.best_metric, 0.0) {
            self.best_metric = metric;
            self.best_params = Some(params.to_vec());
            self.best_epoch = epoch;
            self.save_count += 1;
            true
        } else {
            false
        }
    }

    /// Restore the best parameters. Returns `None` if no checkpoint exists.
    pub fn restore(&self) -> Option<&[f64]> {
        self.best_params.as_deref()
    }

    /// Copy best parameters into the provided slice.
    pub fn restore_into(&self, params: &mut [f64]) -> bool {
        if let Some(best) = &self.best_params {
            if best.len() == params.len() {
                params.copy_from_slice(best);
                return true;
            }
        }
        false
    }

    pub fn best_metric(&self) -> f64 {
        self.best_metric
    }

    pub fn best_epoch(&self) -> u64 {
        self.best_epoch
    }

    pub fn has_checkpoint(&self) -> bool {
        self.best_params.is_some()
    }

    pub fn save_count(&self) -> u64 {
        self.save_count
    }
}

impl fmt::Display for BestModelCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BestModelCheckpoint({}, metric={:.6}, epoch={}, saves={})",
            self.direction, self.best_metric, self.best_epoch, self.save_count
        )
    }
}

// ── Training Progress ──────────────────────────────────────────────

/// Tracks overall training progress including loss and metric curves.
pub struct TrainingProgress {
    train_loss: MetricTracker,
    val_loss: MetricTracker,
    val_metric: MetricTracker,
    early_stop: EarlyStopping,
    checkpoint: BestModelCheckpoint,
}

impl TrainingProgress {
    pub fn new(patience: u64, direction: MetricDirection) -> Self {
        Self {
            train_loss: MetricTracker::new("train_loss"),
            val_loss: MetricTracker::new("val_loss"),
            val_metric: MetricTracker::new("val_metric"),
            early_stop: EarlyStopping::new(patience, direction),
            checkpoint: BestModelCheckpoint::new(direction),
        }
    }

    pub fn with_smoothing(mut self, factor: f64) -> Self {
        self.train_loss = self.train_loss.with_smoothing(factor);
        self.val_loss = self.val_loss.with_smoothing(factor);
        self.val_metric = self.val_metric.with_smoothing(factor);
        self
    }

    pub fn with_min_delta(mut self, delta: f64) -> Self {
        self.early_stop = self.early_stop.with_min_delta(delta);
        self
    }

    /// Record one epoch's results. Returns the training status.
    pub fn epoch(
        &mut self,
        train_loss: f64,
        val_loss: f64,
        val_metric: f64,
        params: &[f64],
    ) -> TrainingStatus {
        self.train_loss.record(train_loss);
        self.val_loss.record(val_loss);
        self.val_metric.record(val_metric);
        self.checkpoint
            .update(val_metric, params, self.early_stop.current_epoch + 1);
        self.early_stop.step(val_metric)
    }

    pub fn best_metric(&self) -> f64 {
        self.checkpoint.best_metric()
    }

    pub fn best_epoch(&self) -> u64 {
        self.checkpoint.best_epoch()
    }

    pub fn restore_best(&self, params: &mut [f64]) -> bool {
        self.checkpoint.restore_into(params)
    }

    pub fn train_loss_history(&self) -> &[f64] {
        self.train_loss.history()
    }

    pub fn val_loss_history(&self) -> &[f64] {
        self.val_loss.history()
    }

    pub fn val_metric_history(&self) -> &[f64] {
        self.val_metric.history()
    }

    pub fn total_epochs(&self) -> u64 {
        self.early_stop.current_epoch()
    }
}

impl fmt::Display for TrainingProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TrainingProgress(epochs={}, best={:.6} @epoch {})",
            self.early_stop.current_epoch(),
            self.checkpoint.best_metric(),
            self.checkpoint.best_epoch()
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn early_stopping_improves() {
        let mut es = EarlyStopping::new(5, MetricDirection::Minimize);
        let status = es.step(1.0);
        assert_eq!(status, TrainingStatus::Improving);
    }

    #[test]
    fn early_stopping_stalls() {
        let mut es = EarlyStopping::new(3, MetricDirection::Minimize);
        es.step(1.0); // best
        let status = es.step(1.5); // worse
        assert!(matches!(status, TrainingStatus::Stalled { .. }));
    }

    #[test]
    fn early_stopping_stops_patience() {
        let mut es = EarlyStopping::new(3, MetricDirection::Minimize);
        es.step(1.0);
        es.step(2.0);
        es.step(2.0);
        let status = es.step(2.0);
        assert_eq!(status, TrainingStatus::StopPatience);
        assert!(status.should_stop());
    }

    #[test]
    fn early_stopping_maximize() {
        let mut es = EarlyStopping::new(3, MetricDirection::Maximize);
        es.step(0.5);
        let status = es.step(0.8); // improvement
        assert_eq!(status, TrainingStatus::Improving);
        assert_eq!(es.best_epoch(), 2);
    }

    #[test]
    fn early_stopping_min_delta() {
        let mut es = EarlyStopping::new(3, MetricDirection::Minimize).with_min_delta(0.1);
        es.step(1.0);
        // 0.95 < 1.0 but not by min_delta=0.1
        let status = es.step(0.95);
        assert!(matches!(status, TrainingStatus::Stalled { .. }));
    }

    #[test]
    fn early_stopping_nan_diverges() {
        let mut es = EarlyStopping::new(5, MetricDirection::Minimize);
        let status = es.step(f64::NAN);
        assert_eq!(status, TrainingStatus::StopDiverged);
    }

    #[test]
    fn early_stopping_max_epochs() {
        let mut es = EarlyStopping::new(100, MetricDirection::Minimize).with_max_epochs(3);
        es.step(1.0);
        es.step(0.5);
        let status = es.step(0.2);
        assert_eq!(status, TrainingStatus::StopMaxEpochs);
    }

    #[test]
    fn early_stopping_divergence_threshold() {
        let mut es = EarlyStopping::new(5, MetricDirection::Minimize)
            .with_divergence_threshold(100.0);
        let status = es.step(500.0);
        assert_eq!(status, TrainingStatus::StopDiverged);
    }

    #[test]
    fn early_stopping_reset() {
        let mut es = EarlyStopping::new(3, MetricDirection::Minimize);
        es.step(1.0);
        es.step(2.0);
        es.reset();
        assert_eq!(es.current_epoch(), 0);
        assert_eq!(es.epochs_without_improvement(), 0);
    }

    #[test]
    fn early_stopping_display() {
        let es = EarlyStopping::new(5, MetricDirection::Minimize);
        assert!(format!("{es}").contains("EarlyStopping"));
    }

    #[test]
    fn metric_tracker_history() {
        let mut tracker = MetricTracker::new("loss");
        tracker.record(1.0);
        tracker.record(0.8);
        tracker.record(0.6);
        assert_eq!(tracker.count(), 3);
        assert!((tracker.latest().unwrap() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn metric_tracker_smoothing() {
        let mut tracker = MetricTracker::new("loss").with_smoothing(0.9);
        tracker.record(1.0);
        tracker.record(0.5);
        // EMA: 0.9 * 1.0 + 0.1 * 0.5 = 0.95
        assert!((tracker.smoothed_latest().unwrap() - 0.95).abs() < 1e-10);
    }

    #[test]
    fn metric_tracker_mean_std() {
        let mut tracker = MetricTracker::new("acc");
        for v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            tracker.record(*v);
        }
        assert!((tracker.mean() - 3.0).abs() < 1e-10);
        assert!(tracker.std_dev() > 1.0);
    }

    #[test]
    fn metric_tracker_plateau() {
        let mut tracker = MetricTracker::new("loss");
        for _ in 0..10 {
            tracker.record(0.5);
        }
        assert!(tracker.is_plateau(5, 0.01));
    }

    #[test]
    fn metric_tracker_no_plateau() {
        let mut tracker = MetricTracker::new("loss");
        for i in 0..10 {
            tracker.record(i as f64);
        }
        assert!(!tracker.is_plateau(5, 0.01));
    }

    #[test]
    fn checkpoint_saves_best() {
        let mut ckpt = BestModelCheckpoint::new(MetricDirection::Maximize);
        let params = vec![1.0, 2.0, 3.0];
        ckpt.update(0.5, &params, 1);
        ckpt.update(0.8, &[4.0, 5.0, 6.0], 2);
        ckpt.update(0.7, &[7.0, 8.0, 9.0], 3); // not best
        assert_eq!(ckpt.best_epoch(), 2);
        assert!((ckpt.best_metric() - 0.8).abs() < 1e-10);
        let restored = ckpt.restore().unwrap();
        assert_eq!(restored, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn checkpoint_restore_into() {
        let mut ckpt = BestModelCheckpoint::new(MetricDirection::Minimize);
        ckpt.update(1.0, &[10.0, 20.0], 1);
        let mut params = vec![0.0, 0.0];
        assert!(ckpt.restore_into(&mut params));
        assert_eq!(params, vec![10.0, 20.0]);
    }

    #[test]
    fn training_progress_integration() {
        let mut progress = TrainingProgress::new(3, MetricDirection::Minimize);
        let params = vec![1.0, 2.0];
        let s1 = progress.epoch(1.0, 0.9, 0.9, &params);
        assert_eq!(s1, TrainingStatus::Improving);
        let s2 = progress.epoch(0.8, 0.7, 0.7, &params);
        assert_eq!(s2, TrainingStatus::Improving);
        let _ = progress.epoch(0.7, 0.8, 0.8, &params); // worse
        let _ = progress.epoch(0.6, 0.9, 0.9, &params); // worse
        let s5 = progress.epoch(0.5, 1.0, 1.0, &params); // patience=3 exhausted
        assert!(s5.should_stop());
        assert_eq!(progress.best_epoch(), 2);
    }

    #[test]
    fn training_progress_display() {
        let progress = TrainingProgress::new(5, MetricDirection::Minimize);
        assert!(format!("{progress}").contains("TrainingProgress"));
    }

    #[test]
    fn training_status_display() {
        assert!(format!("{}", TrainingStatus::Improving).contains("improving"));
        assert!(format!("{}", TrainingStatus::StopPatience).contains("patience"));
    }
}
