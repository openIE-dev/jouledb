//! Per-operation energy tagging.
//!
//! RAII guard that tags a productivity operation with its energy cost,
//! the device it ran on, and the counterfactual (what it would have cost
//! on the default device). This is the bridge between the scheduler and
//! the user-facing energy display.

use crate::hw::tracker::DeviceTarget;
use crate::scheduler::ProductivityTaskType;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A completed energy-tagged operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationTag {
    /// Unique operation identifier.
    pub operation_id: u64,
    /// What kind of productivity operation this was.
    pub task_type: ProductivityTaskType,
    /// Which compute unit executed it.
    pub device: DeviceTarget,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: f64,
    /// Actual energy consumed in millijoules.
    pub energy_mj: f64,
    /// What the energy would have been on the naive/default device.
    pub counterfactual_mj: Option<f64>,
    /// Energy savings percentage (positive = saved, negative = cost more).
    pub savings_percent: Option<f64>,
    /// Timestamp when the operation started (Unix seconds).
    pub started_at: u64,
    /// Optional: entity/node ID this operation was performed on.
    pub entity_id: Option<u64>,
    /// Optional: human-readable label.
    pub label: Option<String>,
}

/// RAII guard for measuring an operation's energy.
///
/// Create with `OperationTimer::start()`, then drop or call `.finish()`
/// to record the measurement.
pub struct OperationTimer {
    operation_id: u64,
    task_type: ProductivityTaskType,
    device: DeviceTarget,
    start: Instant,
    started_at: u64,
    entity_id: Option<u64>,
    label: Option<String>,
    /// Power draw estimate in milliwatts at start (for integration).
    power_mw_at_start: f64,
    finished: bool,
}

impl OperationTimer {
    /// Start timing an operation.
    pub fn start(
        operation_id: u64,
        task_type: ProductivityTaskType,
        device: DeviceTarget,
        power_mw_estimate: f64,
    ) -> Self {
        Self {
            operation_id,
            task_type,
            device,
            start: Instant::now(),
            started_at: unix_timestamp(),
            entity_id: None,
            label: None,
            power_mw_at_start: power_mw_estimate,
            finished: false,
        }
    }

    /// Associate this operation with a graph entity / flow node.
    pub fn with_entity(mut self, entity_id: u64) -> Self {
        self.entity_id = Some(entity_id);
        self
    }

    /// Add a human-readable label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Finish the timer and produce an OperationTag.
    ///
    /// `power_mw_at_end`: current power draw for energy integration.
    /// `counterfactual_mj`: what it would have cost on a different device.
    pub fn finish(
        mut self,
        power_mw_at_end: f64,
        counterfactual_mj: Option<f64>,
    ) -> OperationTag {
        self.finished = true;
        let duration = self.start.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let dt_secs = duration.as_secs_f64();

        // Trapezoidal energy estimate.
        let avg_power_mw = (self.power_mw_at_start + power_mw_at_end) / 2.0;
        let energy_mj = avg_power_mw * dt_secs; // mW * s = mJ

        let savings_percent = counterfactual_mj.map(|cf| {
            if cf > 0.0 {
                ((cf - energy_mj) / cf) * 100.0
            } else {
                0.0
            }
        });

        OperationTag {
            operation_id: self.operation_id,
            task_type: self.task_type,
            device: self.device,
            duration_ms,
            energy_mj,
            counterfactual_mj,
            savings_percent,
            started_at: self.started_at,
            entity_id: self.entity_id,
            label: self.label.clone(),
        }
    }

    /// Finish with only duration-based estimation (no live power readings).
    pub fn finish_estimated(mut self, counterfactual_mj: Option<f64>) -> OperationTag {
        self.finished = true;
        let duration = self.start.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let dt_secs = duration.as_secs_f64();

        // Use start power as constant estimate.
        let energy_mj = self.power_mw_at_start * dt_secs;

        let savings_percent = counterfactual_mj.map(|cf| {
            if cf > 0.0 {
                ((cf - energy_mj) / cf) * 100.0
            } else {
                0.0
            }
        });

        OperationTag {
            operation_id: self.operation_id,
            task_type: self.task_type,
            device: self.device,
            duration_ms,
            energy_mj,
            counterfactual_mj,
            savings_percent,
            started_at: self.started_at,
            entity_id: self.entity_id,
            label: self.label.clone(),
        }
    }
}

impl Drop for OperationTimer {
    fn drop(&mut self) {
        if !self.finished {
            tracing::warn!(
                operation_id = self.operation_id,
                "OperationTimer dropped without finish() — energy not recorded"
            );
        }
    }
}

/// Accumulator for operation tags within a session.
pub struct OperationLog {
    tags: Vec<OperationTag>,
    max_entries: usize,
}

impl OperationLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            tags: Vec::with_capacity(max_entries.min(10_000)),
            max_entries,
        }
    }

    pub fn record(&mut self, tag: OperationTag) {
        if self.tags.len() >= self.max_entries {
            // Evict oldest 10%.
            let drain_count = self.max_entries / 10;
            self.tags.drain(..drain_count);
        }
        self.tags.push(tag);
    }

    pub fn tags(&self) -> &[OperationTag] {
        &self.tags
    }

    /// Total energy consumed across all logged operations.
    pub fn total_energy_mj(&self) -> f64 {
        self.tags.iter().map(|t| t.energy_mj).sum()
    }

    /// Total counterfactual energy (what it would have cost without optimization).
    pub fn total_counterfactual_mj(&self) -> f64 {
        self.tags
            .iter()
            .filter_map(|t| t.counterfactual_mj)
            .sum()
    }

    /// Overall savings percentage.
    pub fn overall_savings_percent(&self) -> f64 {
        let actual = self.total_energy_mj();
        let counterfactual = self.total_counterfactual_mj();
        if counterfactual > 0.0 {
            ((counterfactual - actual) / counterfactual) * 100.0
        } else {
            0.0
        }
    }

    /// Energy breakdown by task type.
    pub fn by_task_type(&self) -> std::collections::HashMap<ProductivityTaskType, f64> {
        let mut map = std::collections::HashMap::new();
        for tag in &self.tags {
            *map.entry(tag.task_type).or_insert(0.0) += tag.energy_mj;
        }
        map
    }

    /// Energy breakdown by device.
    pub fn by_device(&self) -> std::collections::HashMap<DeviceTarget, f64> {
        let mut map = std::collections::HashMap::new();
        for tag in &self.tags {
            *map.entry(tag.device).or_insert(0.0) += tag.energy_mj;
        }
        map
    }

    /// Energy breakdown by entity/node ID.
    pub fn by_entity(&self) -> std::collections::HashMap<u64, f64> {
        let mut map = std::collections::HashMap::new();
        for tag in &self.tags {
            if let Some(eid) = tag.entity_id {
                *map.entry(eid).or_insert(0.0) += tag.energy_mj;
            }
        }
        map
    }

    pub fn clear(&mut self) {
        self.tags.clear();
    }
}

impl Default for OperationLog {
    fn default() -> Self {
        Self::new(10_000)
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_timer() {
        let timer = OperationTimer::start(
            1,
            ProductivityTaskType::TextEdit,
            DeviceTarget::Cpu,
            5000.0,
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
        let tag = timer.finish(5000.0, Some(15000.0 * 0.005));
        assert!(tag.duration_ms > 0.0);
        assert!(tag.energy_mj > 0.0);
        assert!(tag.savings_percent.unwrap() > 0.0);
    }

    #[test]
    fn test_operation_log() {
        let mut log = OperationLog::new(100);
        log.record(OperationTag {
            operation_id: 1,
            task_type: ProductivityTaskType::TextEdit,
            device: DeviceTarget::Cpu,
            duration_ms: 5.0,
            energy_mj: 0.025,
            counterfactual_mj: None,
            savings_percent: None,
            started_at: 0,
            entity_id: Some(42),
            label: Some("keystroke".into()),
        });
        assert_eq!(log.tags().len(), 1);
        assert!(log.total_energy_mj() > 0.0);
    }

    #[test]
    fn test_by_entity() {
        let mut log = OperationLog::new(100);
        for i in 0..5 {
            log.record(OperationTag {
                operation_id: i,
                task_type: ProductivityTaskType::FlowNodeExec,
                device: DeviceTarget::Cpu,
                duration_ms: 10.0,
                energy_mj: 1.0,
                counterfactual_mj: None,
                savings_percent: None,
                started_at: 0,
                entity_id: Some(i % 2), // Two entities.
                label: None,
            });
        }
        let by_entity = log.by_entity();
        assert_eq!(by_entity.len(), 2);
    }
}
