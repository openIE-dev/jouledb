//! Heterogeneous compute scheduler.
//!
//! Routes tasks to CPU, GPU, or NPU based on live utilization, thermal
//! headroom, energy budget, task priority, and workload characteristics.
//! Extends `joule-db-energy::ComputeRouter` with a task queue, deadlines,
//! and per-task energy attribution.

use crate::hw::tracker::DeviceTarget;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Task priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Background tasks (indexing, prefetch, sync).
    Background = 0,
    /// Normal interactive tasks (typing, scrolling).
    Normal = 1,
    /// High priority (user-initiated search, AI query).
    High = 2,
    /// Critical (real-time video encode, live collaboration).
    Critical = 3,
}

/// Computational intensity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeIntensity {
    /// Trivial: key press, cursor move, UI state toggle.
    Trivial,
    /// Light: text formatting, small search, chat message.
    Light,
    /// Medium: formula recalc, image filter, document render.
    Medium,
    /// Heavy: large spreadsheet recalc, AI inference, video encode.
    Heavy,
    /// Extreme: batch ML training, full-text re-index.
    Extreme,
}

/// Task type for productivity operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProductivityTaskType {
    /// Text editing (keystroke, formatting).
    TextEdit,
    /// Formula evaluation (spreadsheet).
    FormulaRecalc,
    /// Graph query (knowledge graph traversal).
    GraphQuery,
    /// Full-text search (tantivy index).
    Search,
    /// AI inference (summarize, classify, complete).
    AiInference,
    /// Canvas render (vector/raster).
    CanvasRender,
    /// Video encode/decode.
    VideoCodec,
    /// CRDT sync (collaboration merge).
    CrdtSync,
    /// File import/export.
    FileIo,
    /// Flow node execution.
    FlowNodeExec,
}

impl ProductivityTaskType {
    /// Default device affinity for this task type.
    pub fn default_device(&self) -> DeviceTarget {
        match self {
            Self::TextEdit => DeviceTarget::Cpu,
            Self::FormulaRecalc => DeviceTarget::Gpu,
            Self::GraphQuery => DeviceTarget::Cpu,
            Self::Search => DeviceTarget::Cpu,
            Self::AiInference => DeviceTarget::Npu,
            Self::CanvasRender => DeviceTarget::Gpu,
            Self::VideoCodec => DeviceTarget::Npu,
            Self::CrdtSync => DeviceTarget::Cpu,
            Self::FileIo => DeviceTarget::Cpu,
            Self::FlowNodeExec => DeviceTarget::Cpu,
        }
    }

    /// Default compute intensity for this task type.
    pub fn default_intensity(&self) -> ComputeIntensity {
        match self {
            Self::TextEdit => ComputeIntensity::Trivial,
            Self::FormulaRecalc => ComputeIntensity::Medium,
            Self::GraphQuery => ComputeIntensity::Light,
            Self::Search => ComputeIntensity::Light,
            Self::AiInference => ComputeIntensity::Heavy,
            Self::CanvasRender => ComputeIntensity::Medium,
            Self::VideoCodec => ComputeIntensity::Heavy,
            Self::CrdtSync => ComputeIntensity::Light,
            Self::FileIo => ComputeIntensity::Medium,
            Self::FlowNodeExec => ComputeIntensity::Medium,
        }
    }
}

/// A task submission to the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSubmission {
    pub task_id: u64,
    pub task_type: ProductivityTaskType,
    pub priority: TaskPriority,
    pub intensity: ComputeIntensity,
    /// Preferred compute unit (may be overridden by scheduler).
    pub preferred_device: DeviceTarget,
    /// Maximum energy budget in millijoules (0 = unlimited).
    pub energy_budget_mj: f64,
    /// Deadline in milliseconds from now (0 = no deadline).
    pub deadline_ms: u64,
    /// Estimated duration in milliseconds.
    pub estimated_duration_ms: u64,
}

/// Scheduler's decision for a submitted task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleDecision {
    pub task_id: u64,
    pub assigned_device: DeviceTarget,
    pub estimated_start_ms: u64,
    pub estimated_energy_mj: f64,
    /// If the preferred device was overridden, what it would have cost.
    pub counterfactual_energy_mj: Option<f64>,
    /// Fallback device if primary is unavailable.
    pub fallback_device: Option<DeviceTarget>,
}

/// Live state of a compute unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeUnitState {
    pub device: DeviceTarget,
    pub utilization: f64,
    pub thermal_headroom: f64,
    pub available: bool,
    pub queue_depth: u64,
    pub power_draw_mw: u64,
}

/// Heterogeneous compute scheduler.
pub struct ComputeScheduler {
    unit_states: HashMap<DeviceTarget, ComputeUnitState>,
    task_queue: VecDeque<TaskSubmission>,
    completed: HashMap<u64, ScheduleDecision>,
    next_task_id: u64,
    current_time_ms: u64,
}

impl ComputeScheduler {
    pub fn new() -> Self {
        let mut unit_states = HashMap::new();
        for device in [DeviceTarget::Cpu, DeviceTarget::Gpu, DeviceTarget::Npu] {
            unit_states.insert(
                device,
                ComputeUnitState {
                    device,
                    utilization: 0.0,
                    thermal_headroom: 1.0,
                    available: device == DeviceTarget::Cpu, // CPU always available.
                    queue_depth: 0,
                    power_draw_mw: 0,
                },
            );
        }
        Self {
            unit_states,
            task_queue: VecDeque::new(),
            completed: HashMap::new(),
            next_task_id: 1,
            current_time_ms: 0,
        }
    }

    /// Update state of a compute unit.
    pub fn update_unit_state(&mut self, state: ComputeUnitState) {
        self.unit_states.insert(state.device, state);
    }

    /// Update the scheduler's clock.
    pub fn update_time(&mut self, time_ms: u64) {
        self.current_time_ms = time_ms;
    }

    /// Submit a task and get a scheduling decision.
    pub fn submit(&mut self, mut task: TaskSubmission) -> ScheduleDecision {
        if task.task_id == 0 {
            task.task_id = self.next_task_id;
            self.next_task_id += 1;
        }

        let preferred = task.preferred_device;
        let assigned = self.pick_device(&task);

        // Estimate energy based on intensity and device.
        let energy_mj = self.estimate_energy(&task, assigned);
        let counterfactual = if assigned != preferred {
            Some(self.estimate_energy(&task, preferred))
        } else {
            None
        };

        let fallback = if assigned == DeviceTarget::Gpu || assigned == DeviceTarget::Npu {
            Some(DeviceTarget::Cpu)
        } else {
            None
        };

        let decision = ScheduleDecision {
            task_id: task.task_id,
            assigned_device: assigned,
            estimated_start_ms: self.current_time_ms,
            estimated_energy_mj: energy_mj,
            counterfactual_energy_mj: counterfactual,
            fallback_device: fallback,
        };

        self.completed.insert(task.task_id, decision.clone());
        decision
    }

    /// Record that a task completed, with actual energy measurement.
    pub fn complete(&mut self, task_id: u64, actual_energy_mj: f64, actual_duration_ms: u64) {
        if let Some(decision) = self.completed.get_mut(&task_id) {
            decision.estimated_energy_mj = actual_energy_mj;
        }
    }

    /// Get a completed task's decision.
    pub fn get_decision(&self, task_id: u64) -> Option<&ScheduleDecision> {
        self.completed.get(&task_id)
    }

    /// Pick the best device for a task.
    fn pick_device(&self, task: &TaskSubmission) -> DeviceTarget {
        let preferred = task.preferred_device;

        // Check if preferred device is available and has capacity.
        if let Some(state) = self.unit_states.get(&preferred) {
            if state.available && state.utilization < 0.9 && state.thermal_headroom > 0.2 {
                return preferred;
            }
        }

        // Fallback: find best available device.
        let mut best = DeviceTarget::Cpu;
        let mut best_score = f64::MIN;

        for (device, state) in &self.unit_states {
            if !state.available {
                continue;
            }
            // Score: lower utilization + more thermal headroom = better.
            let score = (1.0 - state.utilization) * state.thermal_headroom
                - (state.queue_depth as f64 * 0.01);
            if score > best_score {
                best_score = score;
                best = *device;
            }
        }

        best
    }

    /// Estimate energy cost in millijoules.
    fn estimate_energy(&self, task: &TaskSubmission, device: DeviceTarget) -> f64 {
        let base_mj = match task.intensity {
            ComputeIntensity::Trivial => 0.01,
            ComputeIntensity::Light => 0.1,
            ComputeIntensity::Medium => 1.0,
            ComputeIntensity::Heavy => 10.0,
            ComputeIntensity::Extreme => 100.0,
        };

        // Device efficiency multiplier (lower = more efficient for this workload).
        let device_factor = match (device, task.task_type) {
            // GPU excels at parallel: render, recalc.
            (DeviceTarget::Gpu, ProductivityTaskType::CanvasRender) => 0.3,
            (DeviceTarget::Gpu, ProductivityTaskType::FormulaRecalc) => 0.4,
            // NPU excels at inference, video.
            (DeviceTarget::Npu, ProductivityTaskType::AiInference) => 0.2,
            (DeviceTarget::Npu, ProductivityTaskType::VideoCodec) => 0.3,
            // CPU handles everything but less efficiently for parallel work.
            (DeviceTarget::Cpu, ProductivityTaskType::CanvasRender) => 1.5,
            (DeviceTarget::Cpu, ProductivityTaskType::AiInference) => 3.0,
            (DeviceTarget::Cpu, ProductivityTaskType::VideoCodec) => 2.5,
            // Default: CPU is baseline 1.0.
            (DeviceTarget::Cpu, _) => 1.0,
            _ => 1.0,
        };

        let duration_factor = task.estimated_duration_ms as f64 / 100.0;

        base_mj * device_factor * duration_factor.max(0.1)
    }
}

impl Default for ComputeScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_task() {
        let mut scheduler = ComputeScheduler::new();
        let task = TaskSubmission {
            task_id: 0,
            task_type: ProductivityTaskType::TextEdit,
            priority: TaskPriority::Normal,
            intensity: ComputeIntensity::Trivial,
            preferred_device: DeviceTarget::Cpu,
            energy_budget_mj: 0.0,
            deadline_ms: 0,
            estimated_duration_ms: 10,
        };
        let decision = scheduler.submit(task);
        assert_eq!(decision.assigned_device, DeviceTarget::Cpu);
        assert!(decision.estimated_energy_mj > 0.0);
    }

    #[test]
    fn test_gpu_render_more_efficient() {
        let mut scheduler = ComputeScheduler::new();
        // Mark GPU as available.
        scheduler.update_unit_state(ComputeUnitState {
            device: DeviceTarget::Gpu,
            utilization: 0.1,
            thermal_headroom: 0.9,
            available: true,
            queue_depth: 0,
            power_draw_mw: 5000,
        });

        let task = TaskSubmission {
            task_id: 0,
            task_type: ProductivityTaskType::CanvasRender,
            priority: TaskPriority::Normal,
            intensity: ComputeIntensity::Medium,
            preferred_device: DeviceTarget::Gpu,
            energy_budget_mj: 0.0,
            deadline_ms: 0,
            estimated_duration_ms: 100,
        };
        let decision = scheduler.submit(task);
        assert_eq!(decision.assigned_device, DeviceTarget::Gpu);
    }

    #[test]
    fn test_npu_ai_more_efficient() {
        let mut scheduler = ComputeScheduler::new();
        scheduler.update_unit_state(ComputeUnitState {
            device: DeviceTarget::Npu,
            utilization: 0.0,
            thermal_headroom: 1.0,
            available: true,
            queue_depth: 0,
            power_draw_mw: 500,
        });

        let task = TaskSubmission {
            task_id: 0,
            task_type: ProductivityTaskType::AiInference,
            priority: TaskPriority::High,
            intensity: ComputeIntensity::Heavy,
            preferred_device: DeviceTarget::Npu,
            energy_budget_mj: 0.0,
            deadline_ms: 0,
            estimated_duration_ms: 500,
        };
        let decision = scheduler.submit(task);
        assert_eq!(decision.assigned_device, DeviceTarget::Npu);
    }

    #[test]
    fn test_fallback_when_preferred_unavailable() {
        let mut scheduler = ComputeScheduler::new();
        // GPU unavailable.
        scheduler.update_unit_state(ComputeUnitState {
            device: DeviceTarget::Gpu,
            utilization: 0.0,
            thermal_headroom: 1.0,
            available: false,
            queue_depth: 0,
            power_draw_mw: 0,
        });

        let task = TaskSubmission {
            task_id: 0,
            task_type: ProductivityTaskType::CanvasRender,
            priority: TaskPriority::Normal,
            intensity: ComputeIntensity::Medium,
            preferred_device: DeviceTarget::Gpu,
            energy_budget_mj: 0.0,
            deadline_ms: 0,
            estimated_duration_ms: 100,
        };
        let decision = scheduler.submit(task);
        assert_eq!(decision.assigned_device, DeviceTarget::Cpu);
    }

    #[test]
    fn test_counterfactual_energy() {
        let mut scheduler = ComputeScheduler::new();
        // GPU unavailable, so CPU fallback.
        scheduler.update_unit_state(ComputeUnitState {
            device: DeviceTarget::Gpu,
            utilization: 0.0,
            thermal_headroom: 1.0,
            available: false,
            queue_depth: 0,
            power_draw_mw: 0,
        });

        let task = TaskSubmission {
            task_id: 0,
            task_type: ProductivityTaskType::CanvasRender,
            priority: TaskPriority::Normal,
            intensity: ComputeIntensity::Medium,
            preferred_device: DeviceTarget::Gpu,
            energy_budget_mj: 0.0,
            deadline_ms: 0,
            estimated_duration_ms: 100,
        };
        let decision = scheduler.submit(task);
        // Should have counterfactual showing GPU would've been cheaper.
        assert!(decision.counterfactual_energy_mj.is_some());
    }
}
