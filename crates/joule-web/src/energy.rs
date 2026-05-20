//! Energy tracking for web operations.
//!
//! Wraps `joule_energy::context::EnergyContext` to provide per-operation
//! energy measurement for web framework operations (render, route, fetch,
//! state updates, animations, form validation, i18n, and storage).

use joule_energy::context::{EnergyContext, OpGuard};
use joule_energy::scheduler::ProductivityTaskType;
use std::collections::HashMap;

// ── EnergyAwareWeb ─────────────────────────────────────────────

/// Energy tracking wrapper for web framework operations.
pub struct EnergyAwareWeb {
    ctx: EnergyContext,
}

impl EnergyAwareWeb {
    /// Create an energy-aware web context with tracking enabled.
    pub fn new() -> Self {
        Self {
            ctx: EnergyContext::new(),
        }
    }

    /// Create a disabled context (no-op, zero overhead).
    pub fn disabled() -> Self {
        Self {
            ctx: EnergyContext::disabled(),
        }
    }

    /// Access the underlying energy context.
    pub fn context(&self) -> &EnergyContext {
        &self.ctx
    }

    /// Access the underlying energy context mutably.
    pub fn context_mut(&mut self) -> &mut EnergyContext {
        &mut self.ctx
    }

    // ── Operation guards ───────────────────────────────────────

    /// Begin tracking a VDOM render/diff/patch operation.
    pub fn begin_render(&mut self, label: &str) -> Option<OpGuard> {
        self.ctx.begin_op(ProductivityTaskType::CanvasRender, label)
    }

    /// Begin tracking a route resolution (graph traversal).
    pub fn begin_route(&mut self, label: &str) -> Option<OpGuard> {
        self.ctx.begin_op(ProductivityTaskType::GraphQuery, label)
    }

    /// Begin tracking a network fetch (I/O).
    pub fn begin_fetch(&mut self, label: &str) -> Option<OpGuard> {
        self.ctx.begin_op(ProductivityTaskType::FileIo, label)
    }

    /// Begin tracking a state dispatch (CRDT-like sync).
    pub fn begin_state_update(&mut self, label: &str) -> Option<OpGuard> {
        self.ctx.begin_op(ProductivityTaskType::CrdtSync, label)
    }

    /// Begin tracking an animation frame.
    pub fn begin_animation(&mut self, label: &str) -> Option<OpGuard> {
        self.ctx.begin_op(ProductivityTaskType::CanvasRender, label)
    }

    /// Begin tracking form validation (computation).
    pub fn begin_form_validate(&mut self, label: &str) -> Option<OpGuard> {
        self.ctx
            .begin_op(ProductivityTaskType::FormulaRecalc, label)
    }

    /// Begin tracking i18n text formatting.
    pub fn begin_i18n_format(&mut self, label: &str) -> Option<OpGuard> {
        self.ctx.begin_op(ProductivityTaskType::TextEdit, label)
    }

    /// Begin tracking a storage operation (I/O).
    pub fn begin_storage_op(&mut self, label: &str) -> Option<OpGuard> {
        self.ctx.begin_op(ProductivityTaskType::FileIo, label)
    }

    // ── Summary methods ────────────────────────────────────────

    /// Total energy consumed in this session (millijoules).
    pub fn total_energy_mj(&self) -> f64 {
        self.ctx.total_energy_mj()
    }

    /// Overall energy savings percentage vs. naive device routing.
    pub fn savings_percent(&self) -> f64 {
        self.ctx.savings_percent()
    }

    /// Number of operations tracked this session.
    pub fn operation_count(&self) -> usize {
        self.ctx.operation_count()
    }

    /// Energy breakdown by productivity task type.
    pub fn energy_by_operation(&self) -> HashMap<ProductivityTaskType, f64> {
        self.ctx.energy_by_task_type()
    }

    /// Reset all tracking state for a new session.
    pub fn reset(&mut self) {
        self.ctx.reset();
    }

    /// Generate a structured report of web energy usage.
    pub fn report(&self) -> WebEnergyReport {
        let by_type = self.ctx.energy_by_task_type();
        let render_energy = by_type
            .get(&ProductivityTaskType::CanvasRender)
            .copied()
            .unwrap_or(0.0);
        let fetch_energy = by_type
            .get(&ProductivityTaskType::FileIo)
            .copied()
            .unwrap_or(0.0);
        let state_energy = by_type
            .get(&ProductivityTaskType::CrdtSync)
            .copied()
            .unwrap_or(0.0);
        let route_energy = by_type
            .get(&ProductivityTaskType::GraphQuery)
            .copied()
            .unwrap_or(0.0);

        WebEnergyReport {
            total_energy_mj: self.ctx.total_energy_mj(),
            total_operations: self.ctx.operation_count(),
            savings_percent: self.ctx.savings_percent(),
            render_energy_mj: render_energy,
            fetch_energy_mj: fetch_energy,
            state_energy_mj: state_energy,
            route_energy_mj: route_energy,
        }
    }
}

impl Default for EnergyAwareWeb {
    fn default() -> Self {
        Self::new()
    }
}

// ── WebEnergyReport ────────────────────────────────────────────

/// Structured energy report for web operations.
#[derive(Debug, Clone)]
pub struct WebEnergyReport {
    /// Total energy consumed (millijoules).
    pub total_energy_mj: f64,
    /// Total number of tracked operations.
    pub total_operations: usize,
    /// Overall energy savings percentage.
    pub savings_percent: f64,
    /// Energy consumed by render/animation operations (millijoules).
    pub render_energy_mj: f64,
    /// Energy consumed by fetch/storage operations (millijoules).
    pub fetch_energy_mj: f64,
    /// Energy consumed by state updates (millijoules).
    pub state_energy_mj: f64,
    /// Energy consumed by route resolutions (millijoules).
    pub route_energy_mj: f64,
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_tracking_records_energy() {
        let mut web = EnergyAwareWeb::new();
        let guard = web.begin_render("initial render");
        assert!(guard.is_some());
        let tag = guard.unwrap().finish();
        assert_eq!(tag.task_type, ProductivityTaskType::CanvasRender);
        web.context_mut().record(tag);
        assert_eq!(web.operation_count(), 1);
        assert!(web.total_energy_mj() >= 0.0);
    }

    #[test]
    fn disabled_returns_none() {
        let mut web = EnergyAwareWeb::disabled();
        assert!(web.begin_render("noop").is_none());
        assert!(web.begin_route("noop").is_none());
        assert!(web.begin_fetch("noop").is_none());
        assert!(web.begin_state_update("noop").is_none());
        assert_eq!(web.operation_count(), 0);
    }

    #[test]
    fn multiple_ops_accumulate() {
        let mut web = EnergyAwareWeb::new();

        // Render
        let tag = web.begin_render("render").unwrap().finish();
        web.context_mut().record(tag);

        // Route
        let tag = web.begin_route("route").unwrap().finish();
        web.context_mut().record(tag);

        // Fetch
        let tag = web.begin_fetch("fetch").unwrap().finish();
        web.context_mut().record(tag);

        // State update
        let tag = web.begin_state_update("dispatch").unwrap().finish();
        web.context_mut().record(tag);

        assert_eq!(web.operation_count(), 4);
        assert!(web.total_energy_mj() >= 0.0);
    }

    #[test]
    fn report_breaks_down_correctly() {
        let mut web = EnergyAwareWeb::new();

        // Two render ops
        let tag = web.begin_render("r1").unwrap().finish();
        web.context_mut().record(tag);
        let tag = web.begin_render("r2").unwrap().finish();
        web.context_mut().record(tag);

        // One fetch
        let tag = web.begin_fetch("f1").unwrap().finish();
        web.context_mut().record(tag);

        // One route
        let tag = web.begin_route("nav").unwrap().finish();
        web.context_mut().record(tag);

        // One state update
        let tag = web.begin_state_update("dispatch").unwrap().finish();
        web.context_mut().record(tag);

        let report = web.report();
        assert_eq!(report.total_operations, 5);
        assert!(report.render_energy_mj >= 0.0);
        assert!(report.fetch_energy_mj >= 0.0);
        assert!(report.route_energy_mj >= 0.0);
        assert!(report.state_energy_mj >= 0.0);
        assert!(report.total_energy_mj >= 0.0);
    }

    #[test]
    fn reset_clears() {
        let mut web = EnergyAwareWeb::new();
        let tag = web.begin_render("r").unwrap().finish();
        web.context_mut().record(tag);
        assert_eq!(web.operation_count(), 1);

        web.reset();
        assert_eq!(web.operation_count(), 0);
        assert_eq!(web.total_energy_mj(), 0.0);
    }

    #[test]
    fn savings_percent_computed() {
        let mut web = EnergyAwareWeb::new();
        let tag = web.begin_render("r").unwrap().finish();
        web.context_mut().record(tag);
        let s = web.savings_percent();
        assert!(s.is_finite());
    }

    #[test]
    fn all_operation_types_track() {
        let mut web = EnergyAwareWeb::new();

        let tag = web.begin_animation("anim").unwrap().finish();
        web.context_mut().record(tag);

        let tag = web.begin_form_validate("validate").unwrap().finish();
        web.context_mut().record(tag);

        let tag = web.begin_i18n_format("format").unwrap().finish();
        web.context_mut().record(tag);

        let tag = web.begin_storage_op("read").unwrap().finish();
        web.context_mut().record(tag);

        assert_eq!(web.operation_count(), 4);

        let by_op = web.energy_by_operation();
        assert!(by_op.contains_key(&ProductivityTaskType::CanvasRender));
        assert!(by_op.contains_key(&ProductivityTaskType::FormulaRecalc));
        assert!(by_op.contains_key(&ProductivityTaskType::TextEdit));
        assert!(by_op.contains_key(&ProductivityTaskType::FileIo));
    }
}
