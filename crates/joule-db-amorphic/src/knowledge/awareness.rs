//! Awareness: the system knows what it senses, what it's missing, and what it
//! doesn't know it's missing.
//!
//! Axiom 3: Known / Unknown / Unaware.
//!
//! Applied to the system's own sensor inventory:
//! - **Known**: I have this input stream and I'm reading from it.
//! - **Unknown**: I know this input stream should exist but I don't have it.
//!   The absence is information (negative.rs).
//! - **Unaware**: Senses I don't know exist. Detected only by unexplained
//!   contrast — the Promoter finds patterns that no known sensor explains.
//!
//! ## The Action Loop
//!
//! ```text
//! sense → Encode → Compare(expected, actual)
//!   → delta detected?
//!     → yes: Route(energy) → act → sense result → Reflect(did delta resolve?)
//!     → no:  check for absent inputs → absence delta?
//!       → yes: flag missing sensor, seek alternative
//!       → no:  idle (Resting state, near-zero energy)
//! ```
//!
//! The loop runs on deltas. No delta = no compute. Delta on presence = process.
//! Delta on ABSENCE = flag and seek. This is the spike gate applied to the
//! entire sensory system, not just individual records.

use crate::BinaryHV;
use std::collections::HashMap;
use std::time::Instant;

use super::grounded::Modality;
use super::negative::NegationOperator;

/// State of a sensor channel: axiom 3 applied to inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorState {
    /// Active: receiving input, producing contrast.
    Active,
    /// Silent: registered but not producing input. Could be normal (no change)
    /// or could indicate a problem.
    Silent,
    /// Absent: known to be missing. The system is aware of the gap.
    Absent,
    /// Unaware: the system doesn't know this channel could exist.
    /// (By definition, unaware channels aren't in the registry — they're
    /// discovered by the anomaly detector and promoted to Absent or Active.)
    Unaware,
}

/// A registered sensor channel.
#[derive(Clone, Debug)]
pub struct SensorChannel {
    /// Name of this channel.
    pub name: String,
    /// What modality it reads from.
    pub modality: Modality,
    /// Current state.
    pub state: SensorState,
    /// Last value received (as BinaryHV).
    pub last_value: Option<BinaryHV>,
    /// Last time this channel produced input.
    pub last_active_ms: u64,
    /// Expected update interval (ms). If exceeded, flag as Silent.
    pub expected_interval_ms: u64,
    /// Total readings received.
    pub readings: u64,
    /// Total deltas detected (readings that differed from previous).
    pub deltas: u64,
}

/// An action taken by the system in response to a delta.
#[derive(Clone, Debug)]
pub struct Action {
    /// What triggered this action.
    pub trigger: ActionTrigger,
    /// Description of the action.
    pub description: String,
    /// The delta that caused it.
    pub delta_magnitude: f64,
    /// Energy spent on this action.
    pub energy_joules: f64,
}

/// What triggered an action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionTrigger {
    /// A sensor produced a new reading that differed from the last.
    SensorDelta(String),
    /// A sensor went silent (was active, stopped producing).
    SensorSilent(String),
    /// A known-absent sensor was detected as now available.
    SensorRestored(String),
    /// Unexplained contrast detected — possible unaware channel.
    UnexplainedContrast,
    /// Periodic self-check (Test primitive).
    SelfTest,
}

/// The result of reflecting on an action: did it help?
#[derive(Clone, Debug)]
pub struct Reflection {
    /// The action that was taken.
    pub action: Action,
    /// Did the delta resolve after the action?
    pub resolved: bool,
    /// New delta magnitude after the action (lower = action helped).
    pub residual_delta: f64,
    /// Energy efficiency: delta_resolved / energy_spent.
    pub efficiency: f64,
}

/// The awareness system: sensor registry + action loop + reflection.
pub struct Awareness {
    /// Registered sensor channels.
    channels: HashMap<String, SensorChannel>,
    /// Negation operator for absence detection.
    negation: NegationOperator,
    /// Action history.
    pub action_log: Vec<Action>,
    /// Reflection history.
    pub reflection_log: Vec<Reflection>,
    /// Anomaly accumulator: unexplained contrast events that
    /// might indicate an unaware sensor channel.
    pub unexplained_contrasts: Vec<(u64, f64)>, // (timestamp_ms, magnitude)
    /// Threshold for promoting unexplained contrast to "possible missing sensor."
    pub anomaly_threshold: usize,
    /// Current tick (monotonic timestamp).
    tick_ms: u64,
    /// Dimension.
    dim: usize,
}

impl Awareness {
    pub fn new(dim: usize) -> Self {
        Self {
            channels: HashMap::new(),
            negation: NegationOperator::new(dim),
            action_log: Vec::new(),
            reflection_log: Vec::new(),
            unexplained_contrasts: Vec::new(),
            anomaly_threshold: 5,
            tick_ms: 0,
            dim,
        }
    }

    /// Register a sensor channel. It starts as Active with no readings.
    pub fn register_channel(
        &mut self,
        name: &str,
        modality: Modality,
        expected_interval_ms: u64,
    ) {
        self.channels.insert(
            name.to_lowercase(),
            SensorChannel {
                name: name.to_lowercase(),
                modality,
                state: SensorState::Active,
                last_value: None,
                last_active_ms: 0,
                expected_interval_ms,
                readings: 0,
                deltas: 0,
            },
        );
    }

    /// Register a known-absent channel. The system knows it SHOULD have this
    /// but doesn't. This is axiom 3: Unknown.
    pub fn register_absent(&mut self, name: &str, modality: Modality) {
        self.channels.insert(
            name.to_lowercase(),
            SensorChannel {
                name: name.to_lowercase(),
                modality,
                state: SensorState::Absent,
                last_value: None,
                last_active_ms: 0,
                expected_interval_ms: 0,
                readings: 0,
                deltas: 0,
            },
        );
    }

    /// Feed a sensor reading. Returns an action if delta detected.
    pub fn sense(&mut self, channel_name: &str, value: BinaryHV, timestamp_ms: u64) -> Option<Action> {
        self.tick_ms = timestamp_ms;

        let channel = self.channels.get_mut(&channel_name.to_lowercase())?;

        // If this channel was absent, it's now restored
        if channel.state == SensorState::Absent {
            channel.state = SensorState::Active;
            let action = Action {
                trigger: ActionTrigger::SensorRestored(channel.name.clone()),
                description: format!("sensor '{}' restored after absence", channel.name),
                delta_magnitude: 1.0,
                energy_joules: 0.000_001,
            };
            self.action_log.push(action.clone());
            channel.last_value = Some(value);
            channel.last_active_ms = timestamp_ms;
            channel.readings += 1;
            return Some(action);
        }

        // Compare with last value — detect delta
        let delta = match &channel.last_value {
            Some(prev) => 1.0 - prev.similarity(&value) as f64,
            None => 1.0, // First reading: max delta
        };

        channel.last_value = Some(value);
        channel.last_active_ms = timestamp_ms;
        channel.readings += 1;
        channel.state = SensorState::Active;

        // Delta detected?
        if delta > 0.05 {
            channel.deltas += 1;
            let action = Action {
                trigger: ActionTrigger::SensorDelta(channel.name.clone()),
                description: format!(
                    "delta {:.3} on sensor '{}'",
                    delta, channel.name
                ),
                delta_magnitude: delta,
                energy_joules: 0.000_01 * delta, // Energy proportional to delta
            };
            self.action_log.push(action.clone());
            Some(action)
        } else {
            None // No delta, no action, no energy
        }
    }

    /// Check all channels for silence. Returns actions for any that went silent.
    pub fn check_silence(&mut self, current_ms: u64) -> Vec<Action> {
        self.tick_ms = current_ms;
        let mut actions = Vec::new();

        let silent_channels: Vec<String> = self
            .channels
            .iter()
            .filter(|(_, ch)| {
                ch.state == SensorState::Active
                    && ch.expected_interval_ms > 0
                    && ch.readings > 0
                    && current_ms.saturating_sub(ch.last_active_ms) > ch.expected_interval_ms * 3
            })
            .map(|(name, _)| name.clone())
            .collect();

        for name in silent_channels {
            if let Some(ch) = self.channels.get_mut(&name) {
                ch.state = SensorState::Silent;
                let action = Action {
                    trigger: ActionTrigger::SensorSilent(ch.name.clone()),
                    description: format!(
                        "sensor '{}' went silent (expected every {}ms, last active {}ms ago)",
                        ch.name,
                        ch.expected_interval_ms,
                        current_ms.saturating_sub(ch.last_active_ms)
                    ),
                    delta_magnitude: 0.5, // Silence is a medium-priority delta
                    energy_joules: 0.000_001,
                };
                actions.push(action.clone());
                self.action_log.push(action);
            }
        }

        actions
    }

    /// Record unexplained contrast: something changed but no sensor explains it.
    /// After `anomaly_threshold` unexplained events, promote to "possible missing sensor."
    pub fn record_unexplained(&mut self, timestamp_ms: u64, magnitude: f64) -> Option<Action> {
        self.unexplained_contrasts.push((timestamp_ms, magnitude));

        if self.unexplained_contrasts.len() >= self.anomaly_threshold {
            let avg_magnitude = self
                .unexplained_contrasts
                .iter()
                .map(|(_, m)| m)
                .sum::<f64>()
                / self.unexplained_contrasts.len() as f64;

            let action = Action {
                trigger: ActionTrigger::UnexplainedContrast,
                description: format!(
                    "detected {} unexplained contrast events (avg magnitude {:.3}). Possible missing sensor channel.",
                    self.unexplained_contrasts.len(),
                    avg_magnitude
                ),
                delta_magnitude: avg_magnitude,
                energy_joules: 0.000_01,
            };

            self.unexplained_contrasts.clear();
            self.action_log.push(action.clone());
            Some(action)
        } else {
            None
        }
    }

    /// Reflect on an action: did it resolve the delta?
    pub fn reflect(&mut self, action: Action, resolved: bool, residual_delta: f64) -> Reflection {
        let efficiency = if action.energy_joules > 0.0 && resolved {
            action.delta_magnitude / action.energy_joules
        } else {
            0.0
        };

        let reflection = Reflection {
            action,
            resolved,
            residual_delta,
            efficiency,
        };

        self.reflection_log.push(reflection.clone());
        reflection
    }

    /// Self-test: check the health of all sensor channels.
    pub fn self_test(&mut self, current_ms: u64) -> Vec<Action> {
        let mut actions = self.check_silence(current_ms);

        // Count channels in each state
        let active = self.channels.values().filter(|c| c.state == SensorState::Active).count();
        let silent = self.channels.values().filter(|c| c.state == SensorState::Silent).count();
        let absent = self.channels.values().filter(|c| c.state == SensorState::Absent).count();

        if silent > 0 || absent > 0 {
            actions.push(Action {
                trigger: ActionTrigger::SelfTest,
                description: format!(
                    "self-test: {} active, {} silent, {} absent channels",
                    active, silent, absent
                ),
                delta_magnitude: (silent + absent) as f64 / self.channels.len().max(1) as f64,
                energy_joules: 0.000_001,
            });
        }

        actions
    }

    /// Get all channels and their states.
    pub fn channel_states(&self) -> Vec<(&str, SensorState, u64, u64)> {
        self.channels
            .values()
            .map(|ch| (ch.name.as_str(), ch.state, ch.readings, ch.deltas))
            .collect()
    }

    /// Number of registered channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Number of active channels.
    pub fn active_channels(&self) -> usize {
        self.channels
            .values()
            .filter(|ch| ch.state == SensorState::Active)
            .count()
    }

    /// Number of absent channels (known gaps).
    pub fn absent_channels(&self) -> usize {
        self.channels
            .values()
            .filter(|ch| ch.state == SensorState::Absent)
            .count()
    }

    /// Average delta rate across all active channels.
    pub fn delta_rate(&self) -> f64 {
        let active: Vec<&SensorChannel> = self
            .channels
            .values()
            .filter(|ch| ch.readings > 0)
            .collect();
        if active.is_empty() {
            return 0.0;
        }
        let total_deltas: u64 = active.iter().map(|ch| ch.deltas).sum();
        let total_readings: u64 = active.iter().map(|ch| ch.readings).sum();
        if total_readings == 0 {
            return 0.0;
        }
        total_deltas as f64 / total_readings as f64
    }

    /// Average reflection efficiency: how much delta resolved per joule spent.
    pub fn reflection_efficiency(&self) -> f64 {
        if self.reflection_log.is_empty() {
            return 0.0;
        }
        let total: f64 = self.reflection_log.iter().map(|r| r.efficiency).sum();
        total / self.reflection_log.len() as f64
    }
}

impl Default for Awareness {
    fn default() -> Self {
        Self::new(super::concept::KNOWLEDGE_DIM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::concept::KNOWLEDGE_DIM;

    #[test]
    fn test_register_and_sense() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        awareness.register_channel("temperature", Modality::Sensor, 1000);

        let reading = BinaryHV::random(KNOWLEDGE_DIM, 1);
        let action = awareness.sense("temperature", reading, 100);

        // First reading is always a delta (no previous value)
        assert!(action.is_some());
        assert_eq!(awareness.channels["temperature"].readings, 1);
    }

    #[test]
    fn test_no_delta_no_action() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        awareness.register_channel("temperature", Modality::Sensor, 1000);

        let reading = BinaryHV::random(KNOWLEDGE_DIM, 1);
        awareness.sense("temperature", reading.clone(), 100);

        // Same reading again: no delta, no action
        let action = awareness.sense("temperature", reading, 200);
        assert!(action.is_none());
    }

    #[test]
    fn test_delta_detected() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        awareness.register_channel("temperature", Modality::Sensor, 1000);

        awareness.sense("temperature", BinaryHV::random(KNOWLEDGE_DIM, 1), 100);
        let action = awareness.sense("temperature", BinaryHV::random(KNOWLEDGE_DIM, 2), 200);

        // Different reading: delta detected
        assert!(action.is_some());
        let a = action.unwrap();
        assert!(a.delta_magnitude > 0.05);
    }

    #[test]
    fn test_silence_detection() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        awareness.register_channel("heartbeat", Modality::Sensor, 100); // Expected every 100ms

        awareness.sense("heartbeat", BinaryHV::random(KNOWLEDGE_DIM, 1), 0);

        // Time passes without new readings
        let actions = awareness.check_silence(500); // 500ms > 100ms * 3

        assert!(!actions.is_empty());
        assert_eq!(
            awareness.channels["heartbeat"].state,
            SensorState::Silent
        );
    }

    #[test]
    fn test_absent_channel() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        awareness.register_absent("magnetoception", Modality::Sensor);

        assert_eq!(
            awareness.channels["magnetoception"].state,
            SensorState::Absent
        );
        assert_eq!(awareness.absent_channels(), 1);
    }

    #[test]
    fn test_absent_restored() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        awareness.register_absent("magnetoception", Modality::Sensor);

        // Suddenly we get a reading from an absent channel
        let action = awareness.sense("magnetoception", BinaryHV::random(KNOWLEDGE_DIM, 1), 100);
        assert!(action.is_some());
        assert!(matches!(
            action.unwrap().trigger,
            ActionTrigger::SensorRestored(_)
        ));
        assert_eq!(
            awareness.channels["magnetoception"].state,
            SensorState::Active
        );
    }

    #[test]
    fn test_unexplained_contrast_promotes() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        awareness.anomaly_threshold = 3;

        // Three unexplained events trigger promotion
        assert!(awareness.record_unexplained(100, 0.5).is_none());
        assert!(awareness.record_unexplained(200, 0.6).is_none());
        let action = awareness.record_unexplained(300, 0.7);
        assert!(action.is_some());
        assert!(matches!(
            action.unwrap().trigger,
            ActionTrigger::UnexplainedContrast
        ));
    }

    #[test]
    fn test_reflect_on_action() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        let action = Action {
            trigger: ActionTrigger::SensorDelta("temp".into()),
            description: "temp changed".into(),
            delta_magnitude: 0.5,
            energy_joules: 0.000_01,
        };

        let reflection = awareness.reflect(action, true, 0.05);
        assert!(reflection.resolved);
        assert!(reflection.efficiency > 0.0);
        assert_eq!(awareness.reflection_log.len(), 1);
    }

    #[test]
    fn test_self_test() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        awareness.register_channel("vision", Modality::Image, 33); // 30fps
        awareness.register_absent("echolocation", Modality::Audio);

        awareness.sense("vision", BinaryHV::random(KNOWLEDGE_DIM, 1), 0);

        let actions = awareness.self_test(1000); // Check at 1s
        // Should report: vision may be silent (33ms interval, 1000ms since last)
        // + absent echolocation
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_delta_rate() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);
        awareness.register_channel("sensor_a", Modality::Sensor, 100);

        // 5 readings, 3 deltas
        awareness.sense("sensor_a", BinaryHV::random(KNOWLEDGE_DIM, 1), 0);
        awareness.sense("sensor_a", BinaryHV::random(KNOWLEDGE_DIM, 2), 100);
        awareness.sense("sensor_a", BinaryHV::random(KNOWLEDGE_DIM, 3), 200);
        let same = awareness.channels["sensor_a"].last_value.clone().unwrap();
        awareness.sense("sensor_a", same.clone(), 300); // no delta
        awareness.sense("sensor_a", same, 400); // no delta

        let rate = awareness.delta_rate();
        assert!(rate > 0.0 && rate < 1.0);
    }

    #[test]
    fn test_full_awareness_loop() {
        let mut awareness = Awareness::new(KNOWLEDGE_DIM);

        // Register channels
        awareness.register_channel("vision", Modality::Image, 33);
        awareness.register_channel("hearing", Modality::Audio, 10);
        awareness.register_channel("touch", Modality::Sensor, 100);
        awareness.register_absent("thermoception", Modality::Sensor);

        // Sense → Act → Reflect loop
        let v1 = awareness.sense("vision", BinaryHV::random(KNOWLEDGE_DIM, 1), 0);
        if let Some(action) = v1 {
            // Took action on vision delta. Did it resolve?
            let _reflection = awareness.reflect(action, true, 0.01);
        }

        let h1 = awareness.sense("hearing", BinaryHV::random(KNOWLEDGE_DIM, 10), 10);
        if let Some(action) = h1 {
            let _reflection = awareness.reflect(action, true, 0.02);
        }

        // Check status
        assert_eq!(awareness.active_channels(), 3); // vision + hearing + touch (registered = active)
        assert_eq!(awareness.absent_channels(), 1); // thermoception
        assert_eq!(awareness.channel_count(), 4);
        assert!(awareness.reflection_efficiency() > 0.0);

        eprintln!("=== Awareness Loop ===");
        eprintln!("Channels: {} total, {} active, {} absent",
            awareness.channel_count(),
            awareness.active_channels(),
            awareness.absent_channels()
        );
        eprintln!("Delta rate: {:.1}%", awareness.delta_rate() * 100.0);
        eprintln!("Reflection efficiency: {:.1}", awareness.reflection_efficiency());
        eprintln!("Actions taken: {}", awareness.action_log.len());
    }
}
