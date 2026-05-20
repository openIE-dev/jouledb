//! Media recording state machine — chunk collection, timer-based data, blob assembly.

use std::collections::VecDeque;

// ── State ───────────────────────────────────────────────────────

/// Recording state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingState {
    Inactive,
    Recording,
    Paused,
}

// ── Events ──────────────────────────────────────────────────────

/// Events emitted by the recorder.
#[derive(Debug, Clone, PartialEq)]
pub enum RecorderEvent {
    Start,
    Stop,
    Pause,
    Resume,
    DataAvailable(Vec<u8>),
    Error(String),
}

// ── Config ──────────────────────────────────────────────────────

/// Recorder configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RecorderConfig {
    /// MIME type (e.g. "video/webm", "audio/ogg").
    pub mime_type: String,
    /// Target bitrate in bits/sec, if applicable.
    pub bitrate: Option<u64>,
    /// Interval in ms to request data chunks (0 = only on stop).
    pub timeslice_ms: u64,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            mime_type: "video/webm".into(),
            bitrate: None,
            timeslice_ms: 0,
        }
    }
}

// ── Recorder ────────────────────────────────────────────────────

/// Media recorder state machine.
#[derive(Debug, Clone)]
pub struct MediaRecorder {
    pub state: RecordingState,
    pub config: RecorderConfig,
    chunks: Vec<Vec<u8>>,
    current_chunk: Vec<u8>,
    events: VecDeque<RecorderEvent>,
    elapsed_ms: u64,
    last_flush_ms: u64,
}

impl MediaRecorder {
    /// Create a new recorder with the given config.
    pub fn new(config: RecorderConfig) -> Self {
        Self {
            state: RecordingState::Inactive,
            config,
            chunks: Vec::new(),
            current_chunk: Vec::new(),
            events: VecDeque::new(),
            elapsed_ms: 0,
            last_flush_ms: 0,
        }
    }

    /// Drain all pending events.
    pub fn drain_events(&mut self) -> Vec<RecorderEvent> {
        self.events.drain(..).collect()
    }

    /// Start recording.
    pub fn start(&mut self) {
        if self.state != RecordingState::Inactive {
            return;
        }
        self.state = RecordingState::Recording;
        self.chunks.clear();
        self.current_chunk.clear();
        self.elapsed_ms = 0;
        self.last_flush_ms = 0;
        self.events.push_back(RecorderEvent::Start);
    }

    /// Stop recording.
    pub fn stop(&mut self) {
        match self.state {
            RecordingState::Recording | RecordingState::Paused => {
                // Flush any remaining data
                if !self.current_chunk.is_empty() {
                    let chunk = std::mem::take(&mut self.current_chunk);
                    self.events.push_back(RecorderEvent::DataAvailable(chunk.clone()));
                    self.chunks.push(chunk);
                }
                self.state = RecordingState::Inactive;
                self.events.push_back(RecorderEvent::Stop);
            }
            RecordingState::Inactive => {}
        }
    }

    /// Pause recording.
    pub fn pause(&mut self) {
        if self.state == RecordingState::Recording {
            self.state = RecordingState::Paused;
            self.events.push_back(RecorderEvent::Pause);
        }
    }

    /// Resume recording.
    pub fn resume(&mut self) {
        if self.state == RecordingState::Paused {
            self.state = RecordingState::Recording;
            self.events.push_back(RecorderEvent::Resume);
        }
    }

    /// Feed raw data into the recorder (simulating media input).
    pub fn write_data(&mut self, data: &[u8]) {
        if self.state != RecordingState::Recording {
            return;
        }
        self.current_chunk.extend_from_slice(data);
    }

    /// Advance the simulated clock. Flushes chunks at timeslice intervals.
    pub fn tick(&mut self, delta_ms: u64) {
        if self.state != RecordingState::Recording {
            return;
        }
        self.elapsed_ms += delta_ms;

        if self.config.timeslice_ms > 0 {
            while self.elapsed_ms - self.last_flush_ms >= self.config.timeslice_ms {
                self.last_flush_ms += self.config.timeslice_ms;
                if !self.current_chunk.is_empty() {
                    let chunk = std::mem::take(&mut self.current_chunk);
                    self.events.push_back(RecorderEvent::DataAvailable(chunk.clone()));
                    self.chunks.push(chunk);
                }
            }
        }
    }

    /// Manually request the current data (like `requestData()` in the Web API).
    pub fn request_data(&mut self) {
        if self.state == RecordingState::Recording || self.state == RecordingState::Paused {
            if !self.current_chunk.is_empty() {
                let chunk = std::mem::take(&mut self.current_chunk);
                self.events.push_back(RecorderEvent::DataAvailable(chunk.clone()));
                self.chunks.push(chunk);
            }
        }
    }

    /// Assemble all collected chunks into a single blob.
    pub fn assemble_blob(&self) -> Vec<u8> {
        let total: usize = self.chunks.iter().map(|c| c.len()).sum();
        let mut blob = Vec::with_capacity(total);
        for chunk in &self.chunks {
            blob.extend_from_slice(chunk);
        }
        blob
    }

    /// Number of collected chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Total bytes recorded across all chunks.
    pub fn total_bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum::<usize>() + self.current_chunk.len()
    }

    /// Get the MIME type.
    pub fn mime_type(&self) -> &str {
        &self.config.mime_type
    }

    /// Report an error (transitions to inactive).
    pub fn report_error(&mut self, msg: impl Into<String>) {
        self.state = RecordingState::Inactive;
        self.events.push_back(RecorderEvent::Error(msg.into()));
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_recorder() -> MediaRecorder {
        MediaRecorder::new(RecorderConfig::default())
    }

    #[test]
    fn start_stop_lifecycle() {
        let mut r = make_recorder();
        assert_eq!(r.state, RecordingState::Inactive);
        r.start();
        assert_eq!(r.state, RecordingState::Recording);
        r.stop();
        assert_eq!(r.state, RecordingState::Inactive);
        let evts = r.drain_events();
        assert_eq!(evts[0], RecorderEvent::Start);
        assert_eq!(evts[1], RecorderEvent::Stop);
    }

    #[test]
    fn pause_resume() {
        let mut r = make_recorder();
        r.start();
        r.pause();
        assert_eq!(r.state, RecordingState::Paused);
        r.resume();
        assert_eq!(r.state, RecordingState::Recording);
    }

    #[test]
    fn write_and_stop_flushes() {
        let mut r = make_recorder();
        r.start();
        r.write_data(&[1, 2, 3]);
        r.write_data(&[4, 5]);
        r.stop();
        let blob = r.assemble_blob();
        assert_eq!(blob, vec![1, 2, 3, 4, 5]);
        assert_eq!(r.chunk_count(), 1);
    }

    #[test]
    fn timeslice_chunking() {
        let mut r = MediaRecorder::new(RecorderConfig {
            mime_type: "video/webm".into(),
            bitrate: None,
            timeslice_ms: 100,
        });
        r.start();
        r.write_data(&[1, 2, 3]);
        r.tick(50);
        assert_eq!(r.chunk_count(), 0); // not yet

        r.tick(60); // now at 110ms, >= 100ms
        assert_eq!(r.chunk_count(), 1);

        r.write_data(&[4, 5]);
        r.tick(100);
        assert_eq!(r.chunk_count(), 2);
    }

    #[test]
    fn request_data_manually() {
        let mut r = make_recorder();
        r.start();
        r.write_data(&[10, 20]);
        r.request_data();
        assert_eq!(r.chunk_count(), 1);
        r.write_data(&[30]);
        r.request_data();
        assert_eq!(r.chunk_count(), 2);
        let blob = r.assemble_blob();
        assert_eq!(blob, vec![10, 20, 30]);
    }

    #[test]
    fn no_write_when_paused() {
        let mut r = make_recorder();
        r.start();
        r.write_data(&[1]);
        r.pause();
        r.write_data(&[2]); // should be ignored
        r.resume();
        r.write_data(&[3]);
        r.stop();
        let blob = r.assemble_blob();
        assert_eq!(blob, vec![1, 3]);
    }

    #[test]
    fn no_write_when_inactive() {
        let mut r = make_recorder();
        r.write_data(&[1, 2, 3]);
        assert_eq!(r.total_bytes(), 0);
    }

    #[test]
    fn blob_assembly_multiple_chunks() {
        let mut r = make_recorder();
        r.start();
        r.write_data(&[0xAA; 100]);
        r.request_data();
        r.write_data(&[0xBB; 50]);
        r.request_data();
        let blob = r.assemble_blob();
        assert_eq!(blob.len(), 150);
        assert!(blob[..100].iter().all(|b| *b == 0xAA));
        assert!(blob[100..].iter().all(|b| *b == 0xBB));
    }

    #[test]
    fn error_transitions_to_inactive() {
        let mut r = make_recorder();
        r.start();
        r.report_error("stream ended unexpectedly");
        assert_eq!(r.state, RecordingState::Inactive);
        let evts = r.drain_events();
        assert!(evts.iter().any(|e| matches!(e, RecorderEvent::Error(m) if m.contains("stream ended"))));
    }

    #[test]
    fn total_bytes_includes_unflushed() {
        let mut r = make_recorder();
        r.start();
        r.write_data(&[1, 2, 3]);
        assert_eq!(r.total_bytes(), 3);
        r.request_data();
        assert_eq!(r.total_bytes(), 3);
        r.write_data(&[4, 5]);
        assert_eq!(r.total_bytes(), 5);
    }

    #[test]
    fn data_available_events() {
        let mut r = MediaRecorder::new(RecorderConfig {
            timeslice_ms: 50,
            ..RecorderConfig::default()
        });
        r.start();
        r.drain_events();
        r.write_data(&[1]);
        r.tick(60);
        let evts = r.drain_events();
        assert_eq!(evts.len(), 1);
        assert!(matches!(&evts[0], RecorderEvent::DataAvailable(d) if d == &[1]));
    }

    #[test]
    fn mime_type_config() {
        let r = MediaRecorder::new(RecorderConfig {
            mime_type: "audio/ogg".into(),
            ..RecorderConfig::default()
        });
        assert_eq!(r.mime_type(), "audio/ogg");
    }
}
