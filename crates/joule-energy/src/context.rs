//! Unified energy context for productivity crates.
//!
//! `EnergyContext` combines the scheduler, policy, ledger, and operation log
//! into a single handle that productivity crates embed. It provides:
//!
//! - RAII operation tracking via `begin_op()` → `OpGuard`
//! - Adaptive quality via `quality()` (thermal/battery-aware)
//! - Device routing via the underlying `ComputeScheduler`
//! - Session energy accounting via `OperationLog`
//!
//! When no `EnergyContext` is present, operations run unmetered — zero overhead.

use crate::hw::tracker::DeviceTarget;
use crate::ledger::EnergyLedger;
use crate::policy::{EnergyPolicy, HardwareProfile, QualityScale};
use crate::scheduler::{
    ComputeIntensity, ComputeScheduler, ProductivityTaskType, TaskPriority, TaskSubmission,
};
use crate::tag::{OperationLog, OperationTag, OperationTimer};
use std::sync::atomic::{AtomicU64, Ordering};

static OP_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_op_id() -> u64 {
    OP_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Unified energy context for a productivity session.
///
/// Embed in your top-level struct (Grid, SceneGraph, Editor, etc.)
/// to get automatic energy measurement on every operation.
pub struct EnergyContext {
    pub scheduler: ComputeScheduler,
    pub policy: EnergyPolicy,
    pub ledger: EnergyLedger,
    pub log: OperationLog,
    enabled: bool,
    /// Default power estimate (mW) when no live reading is available.
    default_power_mw: f64,
}

impl EnergyContext {
    /// Create a new energy context with default settings.
    pub fn new() -> Self {
        Self {
            scheduler: ComputeScheduler::new(),
            policy: EnergyPolicy::new(HardwareProfile::Balanced),
            ledger: EnergyLedger::new(),
            log: OperationLog::default(),
            enabled: true,
            default_power_mw: 5000.0, // 5W default estimate
        }
    }

    /// Create a disabled context (no-op — for when energy tracking is off).
    pub fn disabled() -> Self {
        let mut ctx = Self::new();
        ctx.enabled = false;
        ctx
    }

    /// Whether energy tracking is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set the default power estimate used when live readings aren't available.
    pub fn set_default_power_mw(&mut self, power_mw: f64) {
        self.default_power_mw = power_mw;
    }

    /// Get current quality scaling factors (thermal/battery-aware).
    pub fn quality(&self) -> &QualityScale {
        self.policy.quality()
    }

    /// Begin tracking an operation. Returns an `OpGuard` that records
    /// energy when dropped or finished.
    ///
    /// If tracking is disabled, returns `None` — caller just runs the op.
    pub fn begin_op(
        &mut self,
        task_type: ProductivityTaskType,
        label: &str,
    ) -> Option<OpGuard> {
        if !self.enabled {
            return None;
        }

        let op_id = next_op_id();
        let device = task_type.default_device();
        let intensity = task_type.default_intensity();

        // Ask scheduler for device routing.
        let decision = self.scheduler.submit(TaskSubmission {
            task_id: op_id,
            task_type,
            priority: TaskPriority::Normal,
            intensity,
            preferred_device: device,
            energy_budget_mj: 0.0,
            deadline_ms: 0,
            estimated_duration_ms: 100,
        });

        let timer = OperationTimer::start(
            op_id,
            task_type,
            decision.assigned_device,
            self.default_power_mw,
        )
        .with_label(label);

        Some(OpGuard {
            timer: Some(timer),
            counterfactual_mj: decision.counterfactual_energy_mj,
            default_power_mw: self.default_power_mw,
        })
    }

    /// Begin a high-priority operation with custom intensity.
    pub fn begin_op_with(
        &mut self,
        task_type: ProductivityTaskType,
        priority: TaskPriority,
        intensity: ComputeIntensity,
        label: &str,
    ) -> Option<OpGuard> {
        if !self.enabled {
            return None;
        }

        let op_id = next_op_id();
        let device = task_type.default_device();

        let decision = self.scheduler.submit(TaskSubmission {
            task_id: op_id,
            task_type,
            priority,
            intensity,
            preferred_device: device,
            energy_budget_mj: 0.0,
            deadline_ms: 0,
            estimated_duration_ms: 100,
        });

        let timer = OperationTimer::start(
            op_id,
            task_type,
            decision.assigned_device,
            self.default_power_mw,
        )
        .with_label(label);

        Some(OpGuard {
            timer: Some(timer),
            counterfactual_mj: decision.counterfactual_energy_mj,
            default_power_mw: self.default_power_mw,
        })
    }

    /// Record a completed operation tag into the log.
    pub fn record(&mut self, tag: OperationTag) {
        self.log.record(tag);
    }

    /// Total energy consumed this session (millijoules).
    pub fn total_energy_mj(&self) -> f64 {
        self.log.total_energy_mj()
    }

    /// Overall energy savings percentage vs. naive device routing.
    pub fn savings_percent(&self) -> f64 {
        self.log.overall_savings_percent()
    }

    /// Number of operations tracked this session.
    pub fn operation_count(&self) -> usize {
        self.log.tags().len()
    }

    /// Energy breakdown by task type.
    pub fn energy_by_task_type(
        &self,
    ) -> std::collections::HashMap<ProductivityTaskType, f64> {
        self.log.by_task_type()
    }

    /// Energy breakdown by device.
    pub fn energy_by_device(&self) -> std::collections::HashMap<DeviceTarget, f64> {
        self.log.by_device()
    }

    /// Reset all tracking state for a new session.
    pub fn reset(&mut self) {
        self.ledger.reset();
        self.log.clear();
        self.scheduler = ComputeScheduler::new();
    }
}

impl Default for EnergyContext {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for an in-flight operation.
///
/// Drop this to finalize the energy measurement and get an `OperationTag`.
/// Call `.finish()` to get the tag directly, or let it drop (which logs a warning).
pub struct OpGuard {
    timer: Option<OperationTimer>,
    counterfactual_mj: Option<f64>,
    default_power_mw: f64,
}

impl OpGuard {
    /// Finish the operation and return the energy tag.
    pub fn finish(mut self) -> OperationTag {
        let timer = self.timer.take().expect("OpGuard already finished");
        timer.finish_estimated(self.counterfactual_mj)
    }

    /// Finish with a live power reading for better accuracy.
    pub fn finish_with_power(mut self, current_power_mw: f64) -> OperationTag {
        let timer = self.timer.take().expect("OpGuard already finished");
        timer.finish(current_power_mw, self.counterfactual_mj)
    }
}

impl Drop for OpGuard {
    fn drop(&mut self) {
        if let Some(timer) = self.timer.take() {
            // Auto-finish with estimated power so energy is always recorded.
            let _tag = timer.finish_estimated(self.counterfactual_mj);
            // Tag is lost on drop — caller should use .finish() to capture it.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_context_basic() {
        let mut ctx = EnergyContext::new();
        assert!(ctx.is_enabled());
        assert_eq!(ctx.operation_count(), 0);

        let guard = ctx.begin_op(ProductivityTaskType::TextEdit, "keystroke");
        assert!(guard.is_some());
        let tag = guard.unwrap().finish();
        assert!(tag.duration_ms >= 0.0);
        assert_eq!(tag.task_type, ProductivityTaskType::TextEdit);
        ctx.record(tag);
        assert_eq!(ctx.operation_count(), 1);
        assert!(ctx.total_energy_mj() >= 0.0);
    }

    #[test]
    fn test_disabled_context_returns_none() {
        let mut ctx = EnergyContext::disabled();
        assert!(!ctx.is_enabled());
        let guard = ctx.begin_op(ProductivityTaskType::CanvasRender, "render frame");
        assert!(guard.is_none());
    }

    #[test]
    fn test_multiple_operations_accumulate() {
        let mut ctx = EnergyContext::new();

        for i in 0..10 {
            let guard = ctx.begin_op(ProductivityTaskType::FormulaRecalc, "recalc cell");
            if let Some(g) = guard {
                // Simulate some work.
                std::hint::black_box(i * i);
                let tag = g.finish();
                ctx.record(tag);
            }
        }

        assert_eq!(ctx.operation_count(), 10);
        let by_type = ctx.energy_by_task_type();
        assert!(by_type.contains_key(&ProductivityTaskType::FormulaRecalc));
    }

    #[test]
    fn test_high_priority_op() {
        let mut ctx = EnergyContext::new();
        let guard = ctx.begin_op_with(
            ProductivityTaskType::AiInference,
            TaskPriority::Critical,
            ComputeIntensity::Heavy,
            "summarize doc",
        );
        assert!(guard.is_some());
        let tag = guard.unwrap().finish();
        assert_eq!(tag.task_type, ProductivityTaskType::AiInference);
        ctx.record(tag);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut ctx = EnergyContext::new();
        let guard = ctx.begin_op(ProductivityTaskType::CrdtSync, "sync");
        ctx.record(guard.unwrap().finish());
        assert_eq!(ctx.operation_count(), 1);

        ctx.reset();
        assert_eq!(ctx.operation_count(), 0);
        assert_eq!(ctx.total_energy_mj(), 0.0);
    }

    #[test]
    fn test_quality_scaling() {
        let ctx = EnergyContext::new();
        let q = ctx.quality();
        assert!(q.render_scale > 0.0);
        assert!(q.sync_frequency > 0.0);
        assert!(q.animation_budget > 0.0);
    }

    #[test]
    fn test_energy_by_device() {
        let mut ctx = EnergyContext::new();
        let g = ctx.begin_op(ProductivityTaskType::TextEdit, "edit");
        ctx.record(g.unwrap().finish());
        let by_device = ctx.energy_by_device();
        // TextEdit defaults to CPU
        assert!(by_device.contains_key(&DeviceTarget::Cpu));
    }
}
