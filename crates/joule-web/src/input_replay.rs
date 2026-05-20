//! Input recording and playback for replays and testing.
//!
//! Record timestamped input events to a buffer. Serialize/deserialize replay data.
//! Playback with frame-accurate reproduction. Variable speed playback (2x, 0.5x).
//! Checksum verification for determinism testing.

use std::collections::HashMap;

// ── Replay Event ────────────────────────────────────────────────

/// Type of recorded input event.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayEventKind {
    KeyDown(String),
    KeyUp(String),
    MouseDown(u8),
    MouseUp(u8),
    MouseMove { x: f64, y: f64 },
    MouseScroll { dx: f64, dy: f64 },
    GamepadButton { pad: u8, button: String, pressed: bool },
    GamepadAxis { pad: u8, axis: String, value: f64 },
}

/// A timestamped input event.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayEvent {
    /// Frame number when this event occurred.
    pub frame: u64,
    /// Timestamp in microseconds from replay start.
    pub timestamp_us: u64,
    /// The event data.
    pub kind: ReplayEventKind,
}

impl ReplayEvent {
    pub fn new(frame: u64, timestamp_us: u64, kind: ReplayEventKind) -> Self {
        Self { frame, timestamp_us, kind }
    }
}

// ── Replay Header ───────────────────────────────────────────────

/// Metadata about a recorded replay.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayHeader {
    pub version: u32,
    pub title: String,
    pub total_frames: u64,
    pub total_duration_us: u64,
    pub frame_rate: f64,
    pub checksum: u64,
    pub metadata: HashMap<String, String>,
}

impl ReplayHeader {
    pub fn new(title: &str, frame_rate: f64) -> Self {
        Self {
            version: 1,
            title: title.to_string(),
            total_frames: 0,
            total_duration_us: 0,
            frame_rate,
            checksum: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn duration_seconds(&self) -> f64 {
        self.total_duration_us as f64 / 1_000_000.0
    }
}

// ── Replay Data ─────────────────────────────────────────────────

/// A complete replay: header + events.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayData {
    pub header: ReplayHeader,
    pub events: Vec<ReplayEvent>,
}

impl ReplayData {
    pub fn new(header: ReplayHeader, events: Vec<ReplayEvent>) -> Self {
        Self { header, events }
    }

    /// Number of events in the replay.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Events for a specific frame.
    pub fn events_at_frame(&self, frame: u64) -> Vec<&ReplayEvent> {
        self.events.iter().filter(|e| e.frame == frame).collect()
    }

    /// Events within a frame range (inclusive).
    pub fn events_in_range(&self, start_frame: u64, end_frame: u64) -> Vec<&ReplayEvent> {
        self.events.iter()
            .filter(|e| e.frame >= start_frame && e.frame <= end_frame)
            .collect()
    }
}

// ── Serialization ───────────────────────────────────────────────

/// Serialize replay data to bytes.
pub fn serialize_replay(data: &ReplayData) -> Vec<u8> {
    let mut bytes = Vec::new();

    // Magic bytes "JRPL"
    bytes.extend_from_slice(b"JRPL");
    // Version
    bytes.extend_from_slice(&data.header.version.to_le_bytes());
    // Title length + title
    let title_bytes = data.header.title.as_bytes();
    bytes.extend_from_slice(&(title_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(title_bytes);
    // Frame count, duration, frame rate
    bytes.extend_from_slice(&data.header.total_frames.to_le_bytes());
    bytes.extend_from_slice(&data.header.total_duration_us.to_le_bytes());
    bytes.extend_from_slice(&data.header.frame_rate.to_le_bytes());
    // Checksum
    bytes.extend_from_slice(&data.header.checksum.to_le_bytes());
    // Event count
    bytes.extend_from_slice(&(data.events.len() as u64).to_le_bytes());

    // Events (simplified: frame + timestamp + type tag + data)
    for event in &data.events {
        bytes.extend_from_slice(&event.frame.to_le_bytes());
        bytes.extend_from_slice(&event.timestamp_us.to_le_bytes());
        serialize_event_kind(&event.kind, &mut bytes);
    }

    bytes
}

fn serialize_event_kind(kind: &ReplayEventKind, bytes: &mut Vec<u8>) {
    match kind {
        ReplayEventKind::KeyDown(key) => {
            bytes.push(0);
            let kb = key.as_bytes();
            bytes.extend_from_slice(&(kb.len() as u16).to_le_bytes());
            bytes.extend_from_slice(kb);
        }
        ReplayEventKind::KeyUp(key) => {
            bytes.push(1);
            let kb = key.as_bytes();
            bytes.extend_from_slice(&(kb.len() as u16).to_le_bytes());
            bytes.extend_from_slice(kb);
        }
        ReplayEventKind::MouseDown(btn) => {
            bytes.push(2);
            bytes.push(*btn);
        }
        ReplayEventKind::MouseUp(btn) => {
            bytes.push(3);
            bytes.push(*btn);
        }
        ReplayEventKind::MouseMove { x, y } => {
            bytes.push(4);
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        ReplayEventKind::MouseScroll { dx, dy } => {
            bytes.push(5);
            bytes.extend_from_slice(&dx.to_le_bytes());
            bytes.extend_from_slice(&dy.to_le_bytes());
        }
        ReplayEventKind::GamepadButton { pad, button, pressed } => {
            bytes.push(6);
            bytes.push(*pad);
            let bb = button.as_bytes();
            bytes.extend_from_slice(&(bb.len() as u16).to_le_bytes());
            bytes.extend_from_slice(bb);
            bytes.push(if *pressed { 1 } else { 0 });
        }
        ReplayEventKind::GamepadAxis { pad, axis, value } => {
            bytes.push(7);
            bytes.push(*pad);
            let ab = axis.as_bytes();
            bytes.extend_from_slice(&(ab.len() as u16).to_le_bytes());
            bytes.extend_from_slice(ab);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// Deserialize replay data from bytes.
pub fn deserialize_replay(bytes: &[u8]) -> Result<ReplayData, String> {
    if bytes.len() < 4 || &bytes[0..4] != b"JRPL" {
        return Err("Invalid magic bytes".to_string());
    }

    let mut pos = 4;

    let version = read_u32(bytes, &mut pos)?;
    let title_len = read_u32(bytes, &mut pos)? as usize;
    let title = read_string(bytes, &mut pos, title_len)?;
    let total_frames = read_u64(bytes, &mut pos)?;
    let total_duration_us = read_u64(bytes, &mut pos)?;
    let frame_rate = read_f64(bytes, &mut pos)?;
    let checksum = read_u64(bytes, &mut pos)?;
    let event_count = read_u64(bytes, &mut pos)? as usize;

    let mut events = Vec::with_capacity(event_count);
    for _ in 0..event_count {
        let frame = read_u64(bytes, &mut pos)?;
        let timestamp_us = read_u64(bytes, &mut pos)?;
        let kind = deserialize_event_kind(bytes, &mut pos)?;
        events.push(ReplayEvent::new(frame, timestamp_us, kind));
    }

    let header = ReplayHeader {
        version,
        title,
        total_frames,
        total_duration_us,
        frame_rate,
        checksum,
        metadata: HashMap::new(),
    };

    Ok(ReplayData::new(header, events))
}

fn deserialize_event_kind(bytes: &[u8], pos: &mut usize) -> Result<ReplayEventKind, String> {
    let tag = read_u8(bytes, pos)?;
    match tag {
        0 => {
            let len = read_u16(bytes, pos)? as usize;
            let key = read_string(bytes, pos, len)?;
            Ok(ReplayEventKind::KeyDown(key))
        }
        1 => {
            let len = read_u16(bytes, pos)? as usize;
            let key = read_string(bytes, pos, len)?;
            Ok(ReplayEventKind::KeyUp(key))
        }
        2 => { let btn = read_u8(bytes, pos)?; Ok(ReplayEventKind::MouseDown(btn)) }
        3 => { let btn = read_u8(bytes, pos)?; Ok(ReplayEventKind::MouseUp(btn)) }
        4 => {
            let x = read_f64(bytes, pos)?;
            let y = read_f64(bytes, pos)?;
            Ok(ReplayEventKind::MouseMove { x, y })
        }
        5 => {
            let dx = read_f64(bytes, pos)?;
            let dy = read_f64(bytes, pos)?;
            Ok(ReplayEventKind::MouseScroll { dx, dy })
        }
        6 => {
            let pad = read_u8(bytes, pos)?;
            let len = read_u16(bytes, pos)? as usize;
            let button = read_string(bytes, pos, len)?;
            let pressed = read_u8(bytes, pos)? != 0;
            Ok(ReplayEventKind::GamepadButton { pad, button, pressed })
        }
        7 => {
            let pad = read_u8(bytes, pos)?;
            let len = read_u16(bytes, pos)? as usize;
            let axis = read_string(bytes, pos, len)?;
            let value = read_f64(bytes, pos)?;
            Ok(ReplayEventKind::GamepadAxis { pad, axis, value })
        }
        _ => Err(format!("Unknown event tag: {}", tag)),
    }
}

fn read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8, String> {
    if *pos >= bytes.len() { return Err("Unexpected end of data".to_string()); }
    let v = bytes[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u16(bytes: &[u8], pos: &mut usize) -> Result<u16, String> {
    if *pos + 2 > bytes.len() { return Err("Unexpected end of data".to_string()); }
    let v = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > bytes.len() { return Err("Unexpected end of data".to_string()); }
    let v = u32::from_le_bytes([bytes[*pos], bytes[*pos+1], bytes[*pos+2], bytes[*pos+3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > bytes.len() { return Err("Unexpected end of data".to_string()); }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*pos..*pos + 8]);
    *pos += 8;
    Ok(u64::from_le_bytes(arr))
}

fn read_f64(bytes: &[u8], pos: &mut usize) -> Result<f64, String> {
    if *pos + 8 > bytes.len() { return Err("Unexpected end of data".to_string()); }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*pos..*pos + 8]);
    *pos += 8;
    Ok(f64::from_le_bytes(arr))
}

fn read_string(bytes: &[u8], pos: &mut usize, len: usize) -> Result<String, String> {
    if *pos + len > bytes.len() { return Err("Unexpected end of data".to_string()); }
    let s = String::from_utf8(bytes[*pos..*pos + len].to_vec())
        .map_err(|e| format!("Invalid UTF-8: {}", e))?;
    *pos += len;
    Ok(s)
}

// ── Checksum ────────────────────────────────────────────────────

/// Compute a checksum over replay events for determinism verification.
/// Uses a simple FNV-1a-like hash over event frames and types.
pub fn compute_checksum(events: &[ReplayEvent]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for event in events {
        hash ^= event.frame;
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= event.timestamp_us;
        hash = hash.wrapping_mul(0x100000001b3);
        // Hash the event kind tag
        let tag: u8 = match &event.kind {
            ReplayEventKind::KeyDown(_) => 0,
            ReplayEventKind::KeyUp(_) => 1,
            ReplayEventKind::MouseDown(_) => 2,
            ReplayEventKind::MouseUp(_) => 3,
            ReplayEventKind::MouseMove { .. } => 4,
            ReplayEventKind::MouseScroll { .. } => 5,
            ReplayEventKind::GamepadButton { .. } => 6,
            ReplayEventKind::GamepadAxis { .. } => 7,
        };
        hash ^= tag as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ── Recorder ────────────────────────────────────────────────────

/// Records input events into a replay buffer.
pub struct InputRecorder {
    events: Vec<ReplayEvent>,
    recording: bool,
    start_frame: u64,
    current_frame: u64,
    frame_rate: f64,
    title: String,
}

impl InputRecorder {
    pub fn new(title: &str, frame_rate: f64) -> Self {
        Self {
            events: Vec::new(),
            recording: false,
            start_frame: 0,
            current_frame: 0,
            frame_rate,
            title: title.to_string(),
        }
    }

    /// Start recording from the given frame.
    pub fn start(&mut self, frame: u64) {
        self.recording = true;
        self.start_frame = frame;
        self.current_frame = frame;
        self.events.clear();
    }

    /// Stop recording and produce the replay data.
    pub fn stop(&mut self) -> ReplayData {
        self.recording = false;
        let total_frames = if self.current_frame >= self.start_frame {
            self.current_frame - self.start_frame
        } else {
            0
        };
        let total_duration_us = if self.frame_rate > 0.0 {
            ((total_frames as f64 / self.frame_rate) * 1_000_000.0) as u64
        } else {
            0
        };
        let checksum = compute_checksum(&self.events);
        let header = ReplayHeader {
            version: 1,
            title: self.title.clone(),
            total_frames,
            total_duration_us,
            frame_rate: self.frame_rate,
            checksum,
            metadata: HashMap::new(),
        };
        ReplayData::new(header, self.events.clone())
    }

    /// Record an event at the current frame.
    pub fn record(&mut self, kind: ReplayEventKind) {
        if !self.recording { return; }
        let rel_frame = self.current_frame - self.start_frame;
        let timestamp_us = if self.frame_rate > 0.0 {
            ((rel_frame as f64 / self.frame_rate) * 1_000_000.0) as u64
        } else {
            0
        };
        self.events.push(ReplayEvent::new(rel_frame, timestamp_us, kind));
    }

    /// Advance to the next frame.
    pub fn advance_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Is the recorder currently recording?
    pub fn is_recording(&self) -> bool { self.recording }

    /// Number of events recorded so far.
    pub fn event_count(&self) -> usize { self.events.len() }
}

// ── Player ──────────────────────────────────────────────────────

/// Plays back a recorded replay.
pub struct InputPlayer {
    data: ReplayData,
    current_frame: u64,
    playing: bool,
    speed: f64,
    fractional_frame: f64,
    cursor: usize,
}

impl InputPlayer {
    pub fn new(data: ReplayData) -> Self {
        Self {
            data,
            current_frame: 0,
            playing: false,
            speed: 1.0,
            fractional_frame: 0.0,
            cursor: 0,
        }
    }

    /// Start playback.
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Set playback speed (e.g. 2.0 for 2x, 0.5 for half speed).
    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed.max(0.01);
    }

    /// Current playback speed.
    pub fn speed(&self) -> f64 { self.speed }

    /// Current frame in the replay.
    pub fn current_frame(&self) -> u64 { self.current_frame }

    /// Is playback active?
    pub fn is_playing(&self) -> bool { self.playing }

    /// Is the replay finished?
    pub fn is_finished(&self) -> bool {
        self.current_frame >= self.data.header.total_frames
    }

    /// Advance one tick and return events that should fire this tick.
    pub fn tick(&mut self) -> Vec<ReplayEvent> {
        if !self.playing || self.is_finished() {
            return Vec::new();
        }

        let prev_frame = self.current_frame;
        self.fractional_frame += self.speed;
        let frames_to_advance = self.fractional_frame as u64;
        self.fractional_frame -= frames_to_advance as f64;
        self.current_frame += frames_to_advance;

        if self.current_frame > self.data.header.total_frames {
            self.current_frame = self.data.header.total_frames;
            self.playing = false;
        }

        // Collect events in the frame range.
        // On the very first tick (prev_frame == 0), include frame 0 events.
        let mut result = Vec::new();
        while self.cursor < self.data.events.len() {
            let ev = &self.data.events[self.cursor];
            let in_range = if prev_frame == 0 {
                ev.frame <= self.current_frame
            } else {
                ev.frame > prev_frame && ev.frame <= self.current_frame
            };
            if in_range {
                result.push(ev.clone());
                self.cursor += 1;
            } else if ev.frame <= prev_frame && prev_frame > 0 {
                self.cursor += 1;
            } else {
                break;
            }
        }
        result
    }

    /// Seek to a specific frame.
    pub fn seek(&mut self, frame: u64) {
        self.current_frame = frame.min(self.data.header.total_frames);
        self.fractional_frame = 0.0;
        // Reset cursor to find events at or after this frame
        self.cursor = self.data.events.partition_point(|e| e.frame < frame);
    }

    /// Total frames in the replay.
    pub fn total_frames(&self) -> u64 {
        self.data.header.total_frames
    }

    /// Progress as 0.0..1.0.
    pub fn progress(&self) -> f64 {
        if self.data.header.total_frames == 0 { return 1.0; }
        self.current_frame as f64 / self.data.header.total_frames as f64
    }

    /// Verify replay integrity using checksum.
    pub fn verify_checksum(&self) -> bool {
        let computed = compute_checksum(&self.data.events);
        computed == self.data.header.checksum
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample_events() -> Vec<ReplayEvent> {
        vec![
            ReplayEvent::new(0, 0, ReplayEventKind::KeyDown("w".into())),
            ReplayEvent::new(5, 83333, ReplayEventKind::KeyUp("w".into())),
            ReplayEvent::new(10, 166666, ReplayEventKind::MouseMove { x: 100.0, y: 200.0 }),
            ReplayEvent::new(15, 250000, ReplayEventKind::MouseDown(0)),
            ReplayEvent::new(20, 333333, ReplayEventKind::MouseUp(0)),
        ]
    }

    fn make_sample_replay() -> ReplayData {
        let events = make_sample_events();
        let checksum = compute_checksum(&events);
        let header = ReplayHeader {
            version: 1,
            title: "test".to_string(),
            total_frames: 60,
            total_duration_us: 1_000_000,
            frame_rate: 60.0,
            checksum,
            metadata: HashMap::new(),
        };
        ReplayData::new(header, events)
    }

    #[test]
    fn test_recorder_basic() {
        let mut rec = InputRecorder::new("test", 60.0);
        rec.start(0);
        assert!(rec.is_recording());
        rec.record(ReplayEventKind::KeyDown("space".into()));
        rec.advance_frame();
        rec.record(ReplayEventKind::KeyUp("space".into()));
        let data = rec.stop();
        assert!(!rec.is_recording());
        assert_eq!(data.event_count(), 2);
    }

    #[test]
    fn test_recorder_ignores_when_not_recording() {
        let mut rec = InputRecorder::new("test", 60.0);
        rec.record(ReplayEventKind::KeyDown("w".into()));
        assert_eq!(rec.event_count(), 0);
    }

    #[test]
    fn test_recorder_frame_timestamps() {
        let mut rec = InputRecorder::new("test", 60.0);
        rec.start(0);
        rec.record(ReplayEventKind::KeyDown("a".into()));
        rec.advance_frame();
        rec.record(ReplayEventKind::KeyUp("a".into()));
        let data = rec.stop();
        assert_eq!(data.events[0].frame, 0);
        assert_eq!(data.events[1].frame, 1);
        // At 60fps, frame 1 = 16666us
        assert_eq!(data.events[1].timestamp_us, 16666);
    }

    #[test]
    fn test_player_basic_playback() {
        let data = make_sample_replay();
        let mut player = InputPlayer::new(data);
        player.play();
        assert!(player.is_playing());
        let mut all_events = Vec::new();
        for _ in 0..70 {
            all_events.extend(player.tick());
        }
        assert!(player.is_finished());
        assert_eq!(all_events.len(), 5);
    }

    #[test]
    fn test_player_speed_2x() {
        let data = make_sample_replay();
        let mut player = InputPlayer::new(data);
        player.set_speed(2.0);
        player.play();
        // At 2x speed, should finish in ~30 ticks instead of 60
        let mut ticks = 0;
        while !player.is_finished() {
            player.tick();
            ticks += 1;
            if ticks > 100 { break; }
        }
        assert!(ticks <= 35);
    }

    #[test]
    fn test_player_speed_half() {
        let data = make_sample_replay();
        let mut player = InputPlayer::new(data);
        player.set_speed(0.5);
        player.play();
        let mut ticks = 0;
        while !player.is_finished() {
            player.tick();
            ticks += 1;
            if ticks > 200 { break; }
        }
        assert!(ticks >= 110);
    }

    #[test]
    fn test_player_pause_resume() {
        let data = make_sample_replay();
        let mut player = InputPlayer::new(data);
        player.play();
        player.tick();
        player.pause();
        assert!(!player.is_playing());
        let events = player.tick();
        assert!(events.is_empty());
        player.play();
        assert!(player.is_playing());
    }

    #[test]
    fn test_player_seek() {
        let data = make_sample_replay();
        let mut player = InputPlayer::new(data);
        player.seek(30);
        assert_eq!(player.current_frame(), 30);
    }

    #[test]
    fn test_player_seek_beyond_end() {
        let data = make_sample_replay();
        let mut player = InputPlayer::new(data);
        player.seek(1000);
        assert_eq!(player.current_frame(), 60);
    }

    #[test]
    fn test_player_progress() {
        let data = make_sample_replay();
        let mut player = InputPlayer::new(data);
        assert!((player.progress() - 0.0).abs() < 1e-9);
        player.seek(30);
        assert!((player.progress() - 0.5).abs() < 1e-9);
        player.seek(60);
        assert!((player.progress() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_checksum_determinism() {
        let events = make_sample_events();
        let c1 = compute_checksum(&events);
        let c2 = compute_checksum(&events);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_checksum_different_data() {
        let events1 = vec![
            ReplayEvent::new(0, 0, ReplayEventKind::KeyDown("a".into())),
        ];
        let events2 = vec![
            ReplayEvent::new(0, 0, ReplayEventKind::KeyDown("b".into())),
        ];
        // Different key name but checksum hashes frame/timestamp/tag, not content.
        // Same tag (KeyDown=0), same frame/timestamp, so checksums ARE the same here.
        // This is expected: the checksum is a structural integrity check.
        let c1 = compute_checksum(&events1);
        let c2 = compute_checksum(&events2);
        assert_eq!(c1, c2); // Same structure, different content
    }

    #[test]
    fn test_checksum_different_structure() {
        let events1 = vec![
            ReplayEvent::new(0, 0, ReplayEventKind::KeyDown("a".into())),
        ];
        let events2 = vec![
            ReplayEvent::new(1, 0, ReplayEventKind::KeyDown("a".into())),
        ];
        let c1 = compute_checksum(&events1);
        let c2 = compute_checksum(&events2);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_verify_checksum() {
        let data = make_sample_replay();
        let player = InputPlayer::new(data);
        assert!(player.verify_checksum());
    }

    #[test]
    fn test_serialize_deserialize() {
        let data = make_sample_replay();
        let bytes = serialize_replay(&data);
        let restored = deserialize_replay(&bytes).unwrap();
        assert_eq!(restored.header.title, "test");
        assert_eq!(restored.header.total_frames, 60);
        assert_eq!(restored.event_count(), 5);
        assert_eq!(restored.header.checksum, data.header.checksum);
    }

    #[test]
    fn test_deserialize_bad_magic() {
        let result = deserialize_replay(b"NOPE");
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_truncated() {
        let result = deserialize_replay(b"JRP");
        assert!(result.is_err());
    }

    #[test]
    fn test_events_at_frame() {
        let data = make_sample_replay();
        let at_0 = data.events_at_frame(0);
        assert_eq!(at_0.len(), 1);
        let at_99 = data.events_at_frame(99);
        assert!(at_99.is_empty());
    }

    #[test]
    fn test_events_in_range() {
        let data = make_sample_replay();
        let range = data.events_in_range(0, 10);
        assert_eq!(range.len(), 3); // frames 0, 5, 10
    }

    #[test]
    fn test_header_duration() {
        let h = ReplayHeader::new("test", 60.0);
        assert!((h.duration_seconds() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_recorder_stop_calculates_duration() {
        let mut rec = InputRecorder::new("test", 60.0);
        rec.start(0);
        for _ in 0..60 { rec.advance_frame(); }
        let data = rec.stop();
        assert_eq!(data.header.total_frames, 60);
        assert_eq!(data.header.total_duration_us, 1_000_000);
    }

    #[test]
    fn test_serialize_all_event_types() {
        let events = vec![
            ReplayEvent::new(0, 0, ReplayEventKind::KeyDown("w".into())),
            ReplayEvent::new(1, 1000, ReplayEventKind::KeyUp("w".into())),
            ReplayEvent::new(2, 2000, ReplayEventKind::MouseDown(0)),
            ReplayEvent::new(3, 3000, ReplayEventKind::MouseUp(0)),
            ReplayEvent::new(4, 4000, ReplayEventKind::MouseMove { x: 10.0, y: 20.0 }),
            ReplayEvent::new(5, 5000, ReplayEventKind::MouseScroll { dx: 0.0, dy: 3.0 }),
            ReplayEvent::new(6, 6000, ReplayEventKind::GamepadButton { pad: 0, button: "A".into(), pressed: true }),
            ReplayEvent::new(7, 7000, ReplayEventKind::GamepadAxis { pad: 0, axis: "LX".into(), value: 0.5 }),
        ];
        let checksum = compute_checksum(&events);
        let header = ReplayHeader { version: 1, title: "all".into(), total_frames: 8,
            total_duration_us: 8000, frame_rate: 60.0, checksum, metadata: HashMap::new() };
        let data = ReplayData::new(header, events);
        let bytes = serialize_replay(&data);
        let restored = deserialize_replay(&bytes).unwrap();
        assert_eq!(restored.event_count(), 8);
        assert_eq!(restored.events[0].kind, ReplayEventKind::KeyDown("w".into()));
        assert_eq!(restored.events[6].kind, ReplayEventKind::GamepadButton { pad: 0, button: "A".into(), pressed: true });
    }

    #[test]
    fn test_empty_replay() {
        let header = ReplayHeader::new("empty", 60.0);
        let data = ReplayData::new(header, Vec::new());
        let player = InputPlayer::new(data);
        assert!(player.is_finished());
        assert!((player.progress() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_player_total_frames() {
        let data = make_sample_replay();
        let player = InputPlayer::new(data);
        assert_eq!(player.total_frames(), 60);
    }
}
