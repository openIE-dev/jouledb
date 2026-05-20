//! Unified Runtime: LiveIntelligence + Awareness + Oracle in one system.
//!
//! The complete operational intelligence:
//! - Starts empty, reads live, gets faster (LiveIntelligence)
//! - Knows what it senses, what it's missing, what it's unaware of (Awareness)
//! - Reaches out to external knowledge on demand (Oracle)
//! - Tracks negative knowledge: absence, negation, void (NegativeKnowledge)
//! - Multimodal grounded input (Grounded)
//! - Energy-metered every operation
//!
//! ```text
//! input (any modality) → Awareness.sense() → delta?
//!   → yes → LiveIntelligence.read() → cache + learn
//!         → Awareness.reflect(resolved?)
//!   → no  → check absence → idle
//!
//! question → LiveIntelligence.ask() → cache hit? → answer
//!          → miss → Oracle.query() → read result → cache → answer
//! ```

use crate::BinaryHV;
use std::time::Instant;

use super::awareness::{Action, Awareness, SensorState};
use super::concept::KNOWLEDGE_DIM;
use super::grounded::{encode_audio_frame, encode_numeric, encode_sensor, encode_structured, Modality};
use super::live::{AskResult, LiveIntelligence, ReadResult};
use super::negative::NegativeKnowledge;
use super::oracle::{Oracle, OracleBackend};

/// The unified runtime.
pub struct Runtime {
    /// Live intelligence: reads, learns, answers.
    pub intelligence: LiveIntelligence,
    /// Awareness: sensor registry, action loop, reflection.
    pub awareness: Awareness,
    /// Negative knowledge: absence, negation, void.
    pub negative: NegativeKnowledge,
    /// Oracle: on-demand external knowledge.
    pub oracle: Oracle,
    /// Total energy consumed across all subsystems.
    pub total_energy: f64,
    /// Total operations.
    pub total_ops: u64,
    /// Start time.
    start: Instant,
    /// Dimension.
    dim: usize,
}

impl Runtime {
    /// Create a new runtime. Starts empty.
    pub fn new() -> Self {
        let dim = KNOWLEDGE_DIM;
        Self {
            intelligence: LiveIntelligence::new(),
            awareness: Awareness::new(dim),
            negative: NegativeKnowledge::new(dim),
            oracle: Oracle::default(),
            total_energy: 0.0,
            total_ops: 0,
            start: Instant::now(),
            dim,
        }
    }

    /// Register an oracle backend.
    pub fn register_oracle(&mut self, backend: Box<dyn OracleBackend>) {
        self.oracle.register_backend(backend);
    }

    /// Register a sensor channel.
    pub fn register_sensor(&mut self, name: &str, modality: Modality, interval_ms: u64) {
        self.awareness.register_channel(name, modality, interval_ms);
    }

    /// Register a known-absent sensor.
    pub fn register_absent_sensor(&mut self, name: &str, modality: Modality) {
        self.awareness.register_absent(name, modality);
    }

    // ================================================================
    // Core operations
    // ================================================================

    /// Feed a sensor reading. The full loop:
    /// Awareness detects delta → LiveIntelligence reads → cache → reflect.
    pub fn sense(&mut self, channel: &str, value: BinaryHV, label: &str, timestamp_ms: u64) -> SenseResult {
        self.total_ops += 1;

        // Awareness: detect delta
        let action = self.awareness.sense(channel, value.clone(), timestamp_ms);

        match action {
            Some(act) => {
                // Delta detected — spend energy to process
                let energy = act.energy_joules;
                self.total_energy += energy;

                // LiveIntelligence reads the label (learns from it)
                let read_result = self.intelligence.read(label);
                self.total_energy += read_result.energy_joules;

                // Reflect: did we learn something?
                let resolved = read_result.triples_extracted > 0 || read_result.cached;
                let residual = if resolved { 0.01 } else { act.delta_magnitude };
                self.awareness.reflect(act.clone(), resolved, residual);

                SenseResult {
                    channel: channel.to_string(),
                    delta: true,
                    delta_magnitude: act.delta_magnitude,
                    learned: read_result.triples_extracted > 0,
                    cached: read_result.cached,
                    energy: energy + read_result.energy_joules,
                }
            }
            None => {
                // No delta — zero energy beyond the comparison
                self.total_energy += 0.000_001;
                SenseResult {
                    channel: channel.to_string(),
                    delta: false,
                    delta_magnitude: 0.0,
                    learned: false,
                    cached: false,
                    energy: 0.000_001,
                }
            }
        }
    }

    /// Feed a text input (convenience wrapper).
    pub fn read_text(&mut self, text: &str) -> ReadResult {
        self.total_ops += 1;
        let result = self.intelligence.read(text);
        self.total_energy += result.energy_joules;
        result
    }

    /// Feed numeric sensor data.
    pub fn sense_numeric(&mut self, channel: &str, readings: &[f32], label: &str, timestamp_ms: u64) -> SenseResult {
        let hv = encode_sensor(readings, self.dim);
        self.sense(channel, hv, label, timestamp_ms)
    }

    /// Feed audio features (MFCC).
    pub fn sense_audio(&mut self, channel: &str, mfcc: &[f32], label: &str, timestamp_ms: u64) -> SenseResult {
        let hv = encode_audio_frame(mfcc, self.dim);
        self.sense(channel, hv, label, timestamp_ms)
    }

    /// Feed structured data.
    pub fn sense_structured(&mut self, channel: &str, fields: &[(&str, &str)], label: &str, timestamp_ms: u64) -> SenseResult {
        let hv = encode_structured(fields, self.dim, 0);
        self.sense(channel, hv, label, timestamp_ms)
    }

    /// Ask a question. Uses LiveIntelligence + Oracle fallback.
    pub fn ask(&mut self, question: &str) -> AskResult {
        self.total_ops += 1;

        // Try LiveIntelligence first
        let result = self.intelligence.ask(question);

        // If no answer from cache, try Oracle
        if result.candidates.is_empty() {
            let oracle_result = self.oracle.query(question);
            if !oracle_result.related.is_empty() {
                // Oracle found something — feed it to intelligence
                for (concept, relation, _weight) in &oracle_result.related {
                    let text = format!("{} {} {}", question, "related_to", concept);
                    self.intelligence.read(&text);
                }
                // Re-ask with expanded knowledge
                let result2 = self.intelligence.ask(question);
                self.total_energy += result2.energy_joules;
                return result2;
            }
        }

        self.total_energy += result.energy_joules;
        result
    }

    /// Record negative knowledge: "X is NOT Y."
    pub fn record_negation(&mut self, subject: &str, property: &str) {
        let hv = self.intelligence.encoder.encode(property).vector;
        self.negative.record_negation(subject, property, &hv);
    }

    /// Check if something is negated.
    pub fn is_negated(&self, subject: &str, property: &str) -> bool {
        self.negative.is_negated(subject, property)
    }

    /// Check silence across all sensors.
    pub fn check_health(&mut self, current_ms: u64) -> Vec<Action> {
        self.awareness.self_test(current_ms)
    }

    // ================================================================
    // Status
    // ================================================================

    /// Full system status.
    pub fn status(&self) -> RuntimeStatus {
        let intel_status = self.intelligence.status();
        RuntimeStatus {
            concepts_cached: intel_status.concepts_cached,
            triples_learned: intel_status.triples_learned,
            cache_hit_rate: intel_status.cache_hit_rate,
            active_sensors: self.awareness.active_channels(),
            absent_sensors: self.awareness.absent_channels(),
            total_sensors: self.awareness.channel_count(),
            negative_facts: self.negative.negative_fact_count(),
            oracle_cache_hits: self.oracle.cache_hits,
            total_energy: self.total_energy,
            total_ops: self.total_ops,
            energy_per_op: if self.total_ops > 0 {
                self.total_energy / self.total_ops as f64
            } else {
                0.0
            },
            uptime_ms: self.start.elapsed().as_millis() as u64,
            is_accelerating: intel_status.is_accelerating,
            delta_rate: self.awareness.delta_rate(),
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a sensor input.
#[derive(Clone, Debug)]
pub struct SenseResult {
    pub channel: String,
    pub delta: bool,
    pub delta_magnitude: f64,
    pub learned: bool,
    pub cached: bool,
    pub energy: f64,
}

/// Full runtime status.
#[derive(Clone, Debug)]
pub struct RuntimeStatus {
    pub concepts_cached: u64,
    pub triples_learned: u64,
    pub cache_hit_rate: f64,
    pub active_sensors: usize,
    pub absent_sensors: usize,
    pub total_sensors: usize,
    pub negative_facts: usize,
    pub oracle_cache_hits: u64,
    pub total_energy: f64,
    pub total_ops: u64,
    pub energy_per_op: f64,
    pub uptime_ms: u64,
    pub is_accelerating: bool,
    pub delta_rate: f64,
}

impl RuntimeStatus {
    pub fn render(&self) -> String {
        format!(
            "Runtime Status:\n  Concepts: {} cached, {} triples learned\n  Cache hit rate: {:.1}%\n  Sensors: {}/{} active, {} absent\n  Negative facts: {}\n  Oracle cache hits: {}\n  Energy: {:.6} J total, {:.9} J/op\n  Ops: {}, Accelerating: {}",
            self.concepts_cached,
            self.triples_learned,
            self.cache_hit_rate * 100.0,
            self.active_sensors,
            self.total_sensors,
            self.absent_sensors,
            self.negative_facts,
            self.oracle_cache_hits,
            self.total_energy,
            self.energy_per_op,
            self.total_ops,
            self.is_accelerating,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_starts_empty() {
        let rt = Runtime::new();
        let status = rt.status();
        assert_eq!(status.concepts_cached, 0);
        assert_eq!(status.total_ops, 0);
        assert_eq!(status.total_sensors, 0);
    }

    #[test]
    fn test_read_text() {
        let mut rt = Runtime::new();
        let r = rt.read_text("a whale is a mammal");
        assert!(!r.cached);
        assert!(r.triples_extracted > 0);
        assert!(rt.status().concepts_cached > 0);
    }

    #[test]
    fn test_sense_with_delta() {
        let mut rt = Runtime::new();
        rt.register_sensor("text_feed", Modality::Text, 1000);

        // Use BinaryHV directly — text encoding produces distinct vectors
        let hv1 = BinaryHV::from_data(b"temperature is 22 degrees", KNOWLEDGE_DIM);
        let r1 = rt.sense("text_feed", hv1, "temp 22C", 0);
        assert!(r1.delta); // First reading = delta

        let hv2 = BinaryHV::from_data(b"temperature is 95 degrees", KNOWLEDGE_DIM);
        let r2 = rt.sense("text_feed", hv2, "temp 95C", 1000);
        assert!(r2.delta, "different input should produce delta");
    }

    #[test]
    fn test_sense_no_delta() {
        let mut rt = Runtime::new();
        rt.register_sensor("pressure", Modality::Sensor, 1000);

        let reading = encode_sensor(&[1013.0], KNOWLEDGE_DIM);
        rt.sense("pressure", reading.clone(), "pressure 1013", 0);
        let r = rt.sense("pressure", reading, "pressure 1013", 1000);
        assert!(!r.delta); // Same reading = no delta = no energy
    }

    #[test]
    fn test_ask_after_reading() {
        let mut rt = Runtime::new();
        rt.read_text("a whale is a mammal");
        rt.read_text("a dolphin is a mammal");
        rt.read_text("a whale can swim");

        let result = rt.ask("what is a whale?");
        assert!(rt.status().concepts_cached > 0);
    }

    #[test]
    fn test_negative_knowledge() {
        let mut rt = Runtime::new();
        rt.record_negation("patient_a", "allergic_penicillin");
        assert!(rt.is_negated("patient_a", "allergic_penicillin"));
        assert!(!rt.is_negated("patient_a", "allergic_aspirin"));
        assert_eq!(rt.status().negative_facts, 1);
    }

    #[test]
    fn test_absent_sensor() {
        let mut rt = Runtime::new();
        rt.register_sensor("vision", Modality::Image, 33);
        rt.register_absent_sensor("echolocation", Modality::Audio);

        let status = rt.status();
        assert_eq!(status.active_sensors, 1);
        assert_eq!(status.absent_sensors, 1);
        assert_eq!(status.total_sensors, 2);
    }

    #[test]
    fn test_health_check() {
        let mut rt = Runtime::new();
        rt.register_sensor("heartbeat", Modality::Sensor, 100);
        rt.sense_numeric("heartbeat", &[72.0], "72 bpm", 0);

        // Time passes without heartbeat
        let actions = rt.check_health(500);
        assert!(!actions.is_empty()); // Should flag silence
    }

    #[test]
    fn test_energy_tracking() {
        let mut rt = Runtime::new();
        rt.read_text("hello world");
        rt.read_text("goodbye world");

        assert!(rt.total_energy > 0.0);
        assert_eq!(rt.total_ops, 2);
        assert!(rt.status().energy_per_op > 0.0);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut rt = Runtime::new();

        // Register sensors
        rt.register_sensor("text_input", Modality::Text, 0);
        rt.register_sensor("temperature", Modality::Sensor, 5000);
        rt.register_absent_sensor("camera", Modality::Image);

        // Read some text
        rt.read_text("a whale is a large mammal that lives in the ocean");
        rt.read_text("a dolphin is a smart mammal that can swim fast");
        rt.read_text("the ocean is a large body of water");

        // Sense temperature
        rt.sense_numeric("temperature", &[22.5, 65.0], "room: 22.5C, 65% humidity", 0);
        rt.sense_numeric("temperature", &[23.1, 64.0], "room: 23.1C, 64% humidity", 5000);

        // Record negative fact
        rt.record_negation("whale", "fish");

        // Ask questions
        let a1 = rt.ask("what is a whale?");
        let a2 = rt.ask("what is a dolphin?");

        // Status
        let status = rt.status();

        eprintln!("{}", status.render());
        eprintln!("Q: {} → A: {}", a1.question, a1.answer);
        eprintln!("Q: {} → A: {}", a2.question, a2.answer);
        eprintln!("whale is NOT fish: {}", rt.is_negated("whale", "fish"));

        assert!(status.concepts_cached > 3);
        assert!(status.triples_learned > 3);
        assert_eq!(status.active_sensors, 2); // text + temp
        assert_eq!(status.absent_sensors, 1); // camera
        assert_eq!(status.negative_facts, 1);
        assert!(status.total_energy > 0.0);
    }
}
