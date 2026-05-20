//! Learning rate schedulers for training optimization.
//!
//! Provides strategies for dynamically adjusting the learning rate during
//! training to improve convergence and final model quality:
//!
//! - [`StepDecayScheduler`] — reduce LR by a factor every N steps
//! - [`CosineAnnealingScheduler`] — smooth cosine decay to a minimum LR
//! - [`WarmupScheduler`] — linear warmup from zero before handing off
//! - [`CyclicalLrScheduler`] — triangular/triangular2 cyclical policies
//! - [`OneCycleScheduler`] — super-convergence one-cycle policy
//! - [`ExponentialDecayScheduler`] — continuous exponential decay

use std::fmt;

// ── Step Decay ─────────────────────────────────────────────────────

/// Reduces the learning rate by a multiplicative factor every `step_size` steps.
///
/// ```text
/// lr_t = base_lr * gamma^(floor(step / step_size))
/// ```
pub struct StepDecayScheduler {
    base_lr: f64,
    gamma: f64,
    step_size: u64,
    current_step: u64,
}

impl StepDecayScheduler {
    pub fn new(base_lr: f64, step_size: u64, gamma: f64) -> Self {
        Self {
            base_lr,
            gamma,
            step_size: step_size.max(1),
            current_step: 0,
        }
    }

    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn with_step_size(mut self, step_size: u64) -> Self {
        self.step_size = step_size.max(1);
        self
    }

    pub fn get_lr(&self) -> f64 {
        let num_decays = self.current_step / self.step_size;
        self.base_lr * self.gamma.powi(num_decays as i32)
    }

    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.get_lr()
    }

    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

impl fmt::Display for StepDecayScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StepDecay(base={:.6}, γ={:.4}, every={}, lr={:.6})",
            self.base_lr, self.gamma, self.step_size, self.get_lr()
        )
    }
}

// ── Cosine Annealing ───────────────────────────────────────────────

/// Cosine annealing schedule that smoothly decreases LR from `base_lr`
/// to `min_lr` over `total_steps`, following a half-cosine curve.
///
/// ```text
/// lr_t = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * t / T))
/// ```
pub struct CosineAnnealingScheduler {
    base_lr: f64,
    min_lr: f64,
    total_steps: u64,
    current_step: u64,
}

impl CosineAnnealingScheduler {
    pub fn new(base_lr: f64, total_steps: u64) -> Self {
        Self {
            base_lr,
            min_lr: 0.0,
            total_steps: total_steps.max(1),
            current_step: 0,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn with_total_steps(mut self, steps: u64) -> Self {
        self.total_steps = steps.max(1);
        self
    }

    pub fn get_lr(&self) -> f64 {
        if self.current_step >= self.total_steps {
            return self.min_lr;
        }
        let progress = self.current_step as f64 / self.total_steps as f64;
        self.min_lr
            + 0.5
                * (self.base_lr - self.min_lr)
                * (1.0 + (std::f64::consts::PI * progress).cos())
    }

    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.get_lr()
    }

    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    pub fn progress(&self) -> f64 {
        self.current_step as f64 / self.total_steps as f64
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

impl fmt::Display for CosineAnnealingScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CosineAnnealing(base={:.6}, min={:.6}, {}/{}, lr={:.6})",
            self.base_lr,
            self.min_lr,
            self.current_step,
            self.total_steps,
            self.get_lr()
        )
    }
}

// ── Warmup Scheduler ───────────────────────────────────────────────

/// Linear warmup that ramps learning rate from `start_lr` to `target_lr`
/// over `warmup_steps`, then holds at `target_lr`.
///
/// Commonly used as a prefix to other schedulers to prevent early
/// instability from large updates on an untrained model.
pub struct WarmupScheduler {
    start_lr: f64,
    target_lr: f64,
    warmup_steps: u64,
    current_step: u64,
}

impl WarmupScheduler {
    pub fn new(target_lr: f64, warmup_steps: u64) -> Self {
        Self {
            start_lr: 0.0,
            target_lr,
            warmup_steps: warmup_steps.max(1),
            current_step: 0,
        }
    }

    pub fn with_start_lr(mut self, start_lr: f64) -> Self {
        self.start_lr = start_lr;
        self
    }

    pub fn with_warmup_steps(mut self, steps: u64) -> Self {
        self.warmup_steps = steps.max(1);
        self
    }

    pub fn get_lr(&self) -> f64 {
        if self.current_step >= self.warmup_steps {
            return self.target_lr;
        }
        let frac = self.current_step as f64 / self.warmup_steps as f64;
        self.start_lr + frac * (self.target_lr - self.start_lr)
    }

    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.get_lr()
    }

    pub fn is_warmup_complete(&self) -> bool {
        self.current_step >= self.warmup_steps
    }

    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

impl fmt::Display for WarmupScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.is_warmup_complete() {
            "complete"
        } else {
            "warming"
        };
        write!(
            f,
            "Warmup(target={:.6}, {}/{}, {}, lr={:.6})",
            self.target_lr,
            self.current_step,
            self.warmup_steps,
            status,
            self.get_lr()
        )
    }
}

// ── Cyclical Learning Rate ─────────────────────────────────────────

/// Policy variant for cyclical LR schedules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CyclicalPolicy {
    /// Linearly increase then decrease between bounds each cycle.
    Triangular,
    /// Same as triangular but halves the amplitude each cycle.
    Triangular2,
    /// Scales the amplitude by `gamma^cycle` each cycle.
    ExpRange(f64),
}

/// Cyclical learning rate schedule (Smith, 2017).
///
/// Oscillates the learning rate between `base_lr` and `max_lr` with a
/// period of `2 * step_size` steps.
pub struct CyclicalLrScheduler {
    base_lr: f64,
    max_lr: f64,
    step_size: u64,
    policy: CyclicalPolicy,
    current_step: u64,
}

impl CyclicalLrScheduler {
    pub fn new(base_lr: f64, max_lr: f64, step_size: u64) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size: step_size.max(1),
            policy: CyclicalPolicy::Triangular,
            current_step: 0,
        }
    }

    pub fn with_policy(mut self, policy: CyclicalPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn with_step_size(mut self, step_size: u64) -> Self {
        self.step_size = step_size.max(1);
        self
    }

    pub fn get_lr(&self) -> f64 {
        let cycle = (self.current_step as f64 / (2.0 * self.step_size as f64)).floor();
        let x = (self.current_step as f64 / self.step_size as f64 - 2.0 * cycle).abs();
        let triangle = (1.0 - (x - 1.0).abs()).max(0.0);

        let scale = match self.policy {
            CyclicalPolicy::Triangular => 1.0,
            CyclicalPolicy::Triangular2 => 1.0 / 2.0_f64.powf(cycle),
            CyclicalPolicy::ExpRange(gamma) => gamma.powf(self.current_step as f64),
        };

        self.base_lr + (self.max_lr - self.base_lr) * triangle * scale
    }

    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.get_lr()
    }

    pub fn current_cycle(&self) -> u64 {
        (self.current_step as f64 / (2.0 * self.step_size as f64)).floor() as u64
    }

    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

impl fmt::Display for CyclicalLrScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let policy_name = match self.policy {
            CyclicalPolicy::Triangular => "triangular",
            CyclicalPolicy::Triangular2 => "triangular2",
            CyclicalPolicy::ExpRange(_) => "exp_range",
        };
        write!(
            f,
            "CyclicalLR({}, base={:.6}, max={:.6}, lr={:.6})",
            policy_name,
            self.base_lr,
            self.max_lr,
            self.get_lr()
        )
    }
}

// ── One-Cycle Policy ───────────────────────────────────────────────

/// One-cycle learning rate policy (Smith & Topin, 2018).
///
/// Three phases:
/// 1. **Warmup**: linearly increase LR from `start_lr` to `max_lr` over
///    `pct_start * total_steps` steps.
/// 2. **Annealing**: cosine decay from `max_lr` to `final_lr` over the
///    remaining steps.
/// 3. **Momentum** (optional): inversely mirrors the LR cycle.
pub struct OneCycleScheduler {
    max_lr: f64,
    total_steps: u64,
    pct_start: f64,
    start_lr_div: f64,
    final_lr_div: f64,
    current_step: u64,
}

impl OneCycleScheduler {
    pub fn new(max_lr: f64, total_steps: u64) -> Self {
        Self {
            max_lr,
            total_steps: total_steps.max(1),
            pct_start: 0.3,
            start_lr_div: 25.0,
            final_lr_div: 10000.0,
            current_step: 0,
        }
    }

    pub fn with_pct_start(mut self, pct: f64) -> Self {
        self.pct_start = pct.clamp(0.01, 0.99);
        self
    }

    pub fn with_start_lr_div(mut self, div: f64) -> Self {
        self.start_lr_div = div.max(1.0);
        self
    }

    pub fn with_final_lr_div(mut self, div: f64) -> Self {
        self.final_lr_div = div.max(1.0);
        self
    }

    pub fn get_lr(&self) -> f64 {
        let warmup_steps = (self.pct_start * self.total_steps as f64) as u64;
        let start_lr = self.max_lr / self.start_lr_div;
        let final_lr = self.max_lr / self.final_lr_div;

        if self.current_step <= warmup_steps {
            // Phase 1: linear warmup
            let frac = self.current_step as f64 / warmup_steps.max(1) as f64;
            start_lr + frac * (self.max_lr - start_lr)
        } else {
            // Phase 2: cosine annealing
            let anneal_steps = self.total_steps - warmup_steps;
            let anneal_progress =
                (self.current_step - warmup_steps) as f64 / anneal_steps.max(1) as f64;
            let anneal_progress = anneal_progress.min(1.0);
            final_lr
                + 0.5
                    * (self.max_lr - final_lr)
                    * (1.0 + (std::f64::consts::PI * anneal_progress).cos())
        }
    }

    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.get_lr()
    }

    pub fn phase(&self) -> &str {
        let warmup_steps = (self.pct_start * self.total_steps as f64) as u64;
        if self.current_step <= warmup_steps {
            "warmup"
        } else if self.current_step < self.total_steps {
            "annealing"
        } else {
            "complete"
        }
    }

    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

impl fmt::Display for OneCycleScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OneCycle(max={:.6}, {}, {}/{}, lr={:.6})",
            self.max_lr,
            self.phase(),
            self.current_step,
            self.total_steps,
            self.get_lr()
        )
    }
}

// ── Exponential Decay ──────────────────────────────────────────────

/// Continuous exponential decay: `lr_t = base_lr * gamma^step`.
pub struct ExponentialDecayScheduler {
    base_lr: f64,
    gamma: f64,
    current_step: u64,
    min_lr: f64,
}

impl ExponentialDecayScheduler {
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        Self {
            base_lr,
            gamma,
            current_step: 0,
            min_lr: 0.0,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn get_lr(&self) -> f64 {
        let lr = self.base_lr * self.gamma.powf(self.current_step as f64);
        lr.max(self.min_lr)
    }

    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.get_lr()
    }

    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

impl fmt::Display for ExponentialDecayScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ExponentialDecay(base={:.6}, γ={:.6}, lr={:.6})",
            self.base_lr,
            self.gamma,
            self.get_lr()
        )
    }
}

// ── Composite Scheduler ────────────────────────────────────────────

/// Chains a warmup scheduler with a cosine annealing scheduler.
pub struct WarmupCosineScheduler {
    warmup: WarmupScheduler,
    cosine: CosineAnnealingScheduler,
}

impl WarmupCosineScheduler {
    pub fn new(max_lr: f64, warmup_steps: u64, total_steps: u64) -> Self {
        Self {
            warmup: WarmupScheduler::new(max_lr, warmup_steps),
            cosine: CosineAnnealingScheduler::new(max_lr, total_steps.saturating_sub(warmup_steps)),
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.cosine = self.cosine.with_min_lr(min_lr);
        self
    }

    pub fn get_lr(&self) -> f64 {
        if !self.warmup.is_warmup_complete() {
            self.warmup.get_lr()
        } else {
            self.cosine.get_lr()
        }
    }

    pub fn step(&mut self) -> f64 {
        if !self.warmup.is_warmup_complete() {
            self.warmup.step()
        } else {
            self.cosine.step()
        }
    }

    pub fn reset(&mut self) {
        self.warmup.reset();
        self.cosine.reset();
    }
}

impl fmt::Display for WarmupCosineScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.warmup.is_warmup_complete() {
            write!(f, "WarmupCosine(warmup, lr={:.6})", self.warmup.get_lr())
        } else {
            write!(f, "WarmupCosine(cosine, lr={:.6})", self.cosine.get_lr())
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_decay_basic() {
        let mut sched = StepDecayScheduler::new(0.1, 10, 0.5);
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
        for _ in 0..10 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn step_decay_multiple_drops() {
        let mut sched = StepDecayScheduler::new(1.0, 5, 0.1);
        for _ in 0..15 {
            sched.step();
        }
        // After 15 steps with step_size=5: 3 decays → 1.0 * 0.1^3 = 0.001
        assert!((sched.get_lr() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn step_decay_display() {
        let sched = StepDecayScheduler::new(0.01, 100, 0.9);
        assert!(format!("{sched}").contains("StepDecay"));
    }

    #[test]
    fn cosine_annealing_endpoints() {
        let mut sched = CosineAnnealingScheduler::new(0.1, 100);
        // Start: should be close to base_lr
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
        // End: should reach min_lr
        for _ in 0..100 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_annealing_midpoint() {
        let sched = CosineAnnealingScheduler {
            base_lr: 1.0,
            min_lr: 0.0,
            total_steps: 100,
            current_step: 50,
        };
        // At midpoint: cos(π/2) = 0, so lr = 0 + 0.5*(1-0)*(1+0) = 0.5
        assert!((sched.get_lr() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn cosine_annealing_with_min() {
        let mut sched = CosineAnnealingScheduler::new(0.1, 50).with_min_lr(0.001);
        for _ in 0..50 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn warmup_linear_ramp() {
        let mut sched = WarmupScheduler::new(0.1, 10);
        for _ in 0..5 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn warmup_holds_at_target() {
        let mut sched = WarmupScheduler::new(0.1, 5);
        for _ in 0..20 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
        assert!(sched.is_warmup_complete());
    }

    #[test]
    fn warmup_display() {
        let sched = WarmupScheduler::new(0.01, 100);
        assert!(format!("{sched}").contains("warming"));
    }

    #[test]
    fn cyclical_triangular_oscillates() {
        let mut sched = CyclicalLrScheduler::new(0.001, 0.01, 10);
        let mut lrs = Vec::new();
        for _ in 0..20 {
            lrs.push(sched.step());
        }
        // Should go up then down, completing one cycle in 20 steps
        assert!(lrs[4] > lrs[0]); // going up
        assert!(lrs[9] > lrs[14]); // going down
    }

    #[test]
    fn cyclical_triangular2_decreasing_amplitude() {
        let mut sched =
            CyclicalLrScheduler::new(0.001, 0.01, 10).with_policy(CyclicalPolicy::Triangular2);
        // Peak of first cycle
        for _ in 0..10 {
            sched.step();
        }
        let peak1 = sched.get_lr();
        // Complete first cycle, peak of second
        for _ in 0..20 {
            sched.step();
        }
        let peak2 = sched.get_lr();
        // Second cycle peak should be lower
        assert!(peak2 < peak1);
    }

    #[test]
    fn one_cycle_phases() {
        let mut sched = OneCycleScheduler::new(0.01, 100).with_pct_start(0.3);
        assert_eq!(sched.phase(), "warmup");
        for _ in 0..31 {
            sched.step();
        }
        assert_eq!(sched.phase(), "annealing");
        for _ in 0..70 {
            sched.step();
        }
        assert_eq!(sched.phase(), "complete");
    }

    #[test]
    fn one_cycle_reaches_max() {
        let mut sched = OneCycleScheduler::new(0.01, 100).with_pct_start(0.3);
        // After warmup (30 steps)
        for _ in 0..30 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn one_cycle_display() {
        let sched = OneCycleScheduler::new(0.01, 100);
        assert!(format!("{sched}").contains("OneCycle"));
    }

    #[test]
    fn exponential_decay_basic() {
        let mut sched = ExponentialDecayScheduler::new(1.0, 0.9);
        sched.step();
        assert!((sched.get_lr() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn exponential_decay_with_floor() {
        let mut sched = ExponentialDecayScheduler::new(1.0, 0.5).with_min_lr(0.1);
        for _ in 0..100 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn warmup_cosine_composite() {
        let mut sched = WarmupCosineScheduler::new(0.01, 10, 100);
        // During warmup
        for _ in 0..5 {
            sched.step();
        }
        assert!(sched.get_lr() < 0.01);
        // After warmup, during cosine
        for _ in 0..6 {
            sched.step();
        }
        let lr_after_warmup = sched.get_lr();
        for _ in 0..50 {
            sched.step();
        }
        assert!(sched.get_lr() < lr_after_warmup);
    }

    #[test]
    fn step_decay_reset() {
        let mut sched = StepDecayScheduler::new(0.1, 5, 0.5);
        for _ in 0..10 {
            sched.step();
        }
        sched.reset();
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
        assert_eq!(sched.current_step(), 0);
    }
}
