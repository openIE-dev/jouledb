//! Speech synthesis and recognition state management.
//!
//! Provides headless state machines for the Web Speech API — synthesis
//! (text-to-speech) and recognition (speech-to-text) — fully testable
//! without a browser runtime.

use std::collections::VecDeque;

// ── Speech Synthesis ────────────────────────────────────────────

/// A voice descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechVoice {
    pub name: String,
    pub lang: String,
    pub local: bool,
    pub default_: bool,
}

/// A speech utterance with synthesis parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechUtterance {
    pub text: String,
    pub voice: Option<String>,
    pub rate: f64,
    pub pitch: f64,
    pub volume: f64,
    pub lang: Option<String>,
}

impl SpeechUtterance {
    /// Create a new utterance with default parameters.
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            voice: None,
            rate: 1.0,
            pitch: 1.0,
            volume: 1.0,
            lang: None,
        }
    }

    /// Set the speaking rate (0.1 to 10.0).
    pub fn rate(mut self, r: f64) -> Self {
        self.rate = r.clamp(0.1, 10.0);
        self
    }

    /// Set the pitch (0.0 to 2.0).
    pub fn pitch(mut self, p: f64) -> Self {
        self.pitch = p.clamp(0.0, 2.0);
        self
    }

    /// Set the volume (0.0 to 1.0).
    pub fn volume(mut self, v: f64) -> Self {
        self.volume = v.clamp(0.0, 1.0);
        self
    }

    /// Set the language.
    pub fn lang(mut self, l: &str) -> Self {
        self.lang = Some(l.to_string());
        self
    }

    /// Set the voice name.
    pub fn voice(mut self, v: &str) -> Self {
        self.voice = Some(v.to_string());
        self
    }
}

/// Synthesis state machine.
#[derive(Debug, Clone, PartialEq)]
pub enum SynthState {
    Idle,
    Speaking,
    Paused,
}

/// Speech synthesizer managing a queue of utterances.
pub struct SpeechSynthesizer {
    state: SynthState,
    queue: VecDeque<SpeechUtterance>,
    voices: Vec<SpeechVoice>,
    current: Option<SpeechUtterance>,
}

impl SpeechSynthesizer {
    /// Create a new synthesizer in idle state.
    pub fn new() -> Self {
        Self {
            state: SynthState::Idle,
            queue: VecDeque::new(),
            voices: Vec::new(),
            current: None,
        }
    }

    /// Register an available voice.
    pub fn add_voice(&mut self, voice: SpeechVoice) {
        self.voices.push(voice);
    }

    /// Return available voices.
    pub fn voices(&self) -> &[SpeechVoice] {
        &self.voices
    }

    /// Queue an utterance for speaking. If idle, start speaking immediately.
    pub fn speak(&mut self, utterance: SpeechUtterance) {
        if self.state == SynthState::Idle {
            self.current = Some(utterance);
            self.state = SynthState::Speaking;
        } else {
            self.queue.push_back(utterance);
        }
    }

    /// Pause the current utterance.
    pub fn pause(&mut self) {
        if self.state == SynthState::Speaking {
            self.state = SynthState::Paused;
        }
    }

    /// Resume a paused utterance.
    pub fn resume(&mut self) {
        if self.state == SynthState::Paused {
            self.state = SynthState::Speaking;
        }
    }

    /// Cancel all speech and clear the queue.
    pub fn cancel(&mut self) {
        self.state = SynthState::Idle;
        self.current = None;
        self.queue.clear();
    }

    /// Simulate finishing the current utterance and advancing to next.
    pub fn finish_current(&mut self) {
        if self.state == SynthState::Speaking || self.state == SynthState::Paused {
            if let Some(next) = self.queue.pop_front() {
                self.current = Some(next);
                self.state = SynthState::Speaking;
            } else {
                self.current = None;
                self.state = SynthState::Idle;
            }
        }
    }

    /// Check if currently speaking.
    pub fn is_speaking(&self) -> bool {
        self.state == SynthState::Speaking
    }

    /// Number of queued utterances (not including current).
    pub fn queue_length(&self) -> usize {
        self.queue.len()
    }

    /// Current synthesizer state.
    pub fn state(&self) -> &SynthState {
        &self.state
    }

    /// The currently speaking utterance, if any.
    pub fn current(&self) -> Option<&SpeechUtterance> {
        self.current.as_ref()
    }
}

impl Default for SpeechSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Speech Recognition ──────────────────────────────────────────

/// A single recognition result.
#[derive(Debug, Clone, PartialEq)]
pub struct RecognitionResult {
    pub transcript: String,
    pub confidence: f64,
    pub is_final: bool,
    pub alternatives: Vec<(String, f64)>,
}

/// Configuration for speech recognition.
#[derive(Debug, Clone)]
pub struct RecognitionConfig {
    pub continuous: bool,
    pub interim_results: bool,
    pub lang: String,
    pub max_alternatives: usize,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            continuous: false,
            interim_results: false,
            lang: "en-US".to_string(),
            max_alternatives: 1,
        }
    }
}

/// Speech recognition state.
pub struct RecognitionState {
    config: RecognitionConfig,
    results: Vec<RecognitionResult>,
    listening: bool,
}

impl RecognitionState {
    /// Create a new recognition state with the given config.
    pub fn new(config: RecognitionConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            listening: false,
        }
    }

    /// Start listening.
    pub fn start(&mut self) {
        self.listening = true;
    }

    /// Stop listening.
    pub fn stop(&mut self) {
        self.listening = false;
    }

    /// Whether currently listening.
    pub fn is_listening(&self) -> bool {
        self.listening
    }

    /// Add a recognition result.
    pub fn add_result(&mut self, result: RecognitionResult) {
        self.results.push(result);
    }

    /// All accumulated results.
    pub fn results(&self) -> &[RecognitionResult] {
        &self.results
    }

    /// Concatenate all final transcripts.
    pub fn final_transcript(&self) -> String {
        self.results
            .iter()
            .filter(|r| r.is_final)
            .map(|r| r.transcript.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The recognition configuration.
    pub fn config(&self) -> &RecognitionConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speak_queues_and_starts() {
        let mut synth = SpeechSynthesizer::new();
        synth.speak(SpeechUtterance::new("Hello"));
        assert!(synth.is_speaking());
        assert_eq!(synth.queue_length(), 0);

        synth.speak(SpeechUtterance::new("World"));
        assert_eq!(synth.queue_length(), 1);
    }

    #[test]
    fn pause_and_resume() {
        let mut synth = SpeechSynthesizer::new();
        synth.speak(SpeechUtterance::new("Test"));
        synth.pause();
        assert_eq!(*synth.state(), SynthState::Paused);
        assert!(!synth.is_speaking());
        synth.resume();
        assert!(synth.is_speaking());
    }

    #[test]
    fn cancel_clears_everything() {
        let mut synth = SpeechSynthesizer::new();
        synth.speak(SpeechUtterance::new("A"));
        synth.speak(SpeechUtterance::new("B"));
        synth.speak(SpeechUtterance::new("C"));
        assert_eq!(synth.queue_length(), 2);

        synth.cancel();
        assert_eq!(*synth.state(), SynthState::Idle);
        assert_eq!(synth.queue_length(), 0);
        assert!(synth.current().is_none());
    }

    #[test]
    fn finish_advances_queue() {
        let mut synth = SpeechSynthesizer::new();
        synth.speak(SpeechUtterance::new("First"));
        synth.speak(SpeechUtterance::new("Second"));
        assert_eq!(synth.current().unwrap().text, "First");

        synth.finish_current();
        assert_eq!(synth.current().unwrap().text, "Second");
        assert!(synth.is_speaking());

        synth.finish_current();
        assert_eq!(*synth.state(), SynthState::Idle);
    }

    #[test]
    fn voice_selection() {
        let mut synth = SpeechSynthesizer::new();
        synth.add_voice(SpeechVoice {
            name: "Alice".into(),
            lang: "en-US".into(),
            local: true,
            default_: true,
        });
        synth.add_voice(SpeechVoice {
            name: "Bob".into(),
            lang: "en-GB".into(),
            local: false,
            default_: false,
        });
        assert_eq!(synth.voices().len(), 2);
        assert_eq!(synth.voices()[0].name, "Alice");
    }

    #[test]
    fn utterance_builder() {
        let u = SpeechUtterance::new("hello")
            .rate(1.5)
            .pitch(0.8)
            .volume(0.5)
            .lang("fr-FR")
            .voice("Marie");
        assert_eq!(u.text, "hello");
        assert_eq!(u.rate, 1.5);
        assert_eq!(u.pitch, 0.8);
        assert_eq!(u.volume, 0.5);
        assert_eq!(u.lang.as_deref(), Some("fr-FR"));
        assert_eq!(u.voice.as_deref(), Some("Marie"));
    }

    #[test]
    fn utterance_clamping() {
        let u = SpeechUtterance::new("test").rate(100.0).pitch(-5.0).volume(99.0);
        assert_eq!(u.rate, 10.0);
        assert_eq!(u.pitch, 0.0);
        assert_eq!(u.volume, 1.0);
    }

    #[test]
    fn recognition_results_accumulate() {
        let mut state = RecognitionState::new(RecognitionConfig::default());
        state.start();
        assert!(state.is_listening());

        state.add_result(RecognitionResult {
            transcript: "hello".into(),
            confidence: 0.9,
            is_final: true,
            alternatives: vec![("hi".into(), 0.5)],
        });
        state.add_result(RecognitionResult {
            transcript: "world".into(),
            confidence: 0.85,
            is_final: false,
            alternatives: vec![],
        });
        assert_eq!(state.results().len(), 2);
    }

    #[test]
    fn final_transcript_concatenates() {
        let mut state = RecognitionState::new(RecognitionConfig::default());
        state.add_result(RecognitionResult {
            transcript: "hello".into(),
            confidence: 0.9,
            is_final: true,
            alternatives: vec![],
        });
        state.add_result(RecognitionResult {
            transcript: "maybe".into(),
            confidence: 0.4,
            is_final: false,
            alternatives: vec![],
        });
        state.add_result(RecognitionResult {
            transcript: "world".into(),
            confidence: 0.88,
            is_final: true,
            alternatives: vec![],
        });
        assert_eq!(state.final_transcript(), "hello world");
    }

    #[test]
    fn recognition_stop() {
        let mut state = RecognitionState::new(RecognitionConfig::default());
        state.start();
        assert!(state.is_listening());
        state.stop();
        assert!(!state.is_listening());
    }
}
