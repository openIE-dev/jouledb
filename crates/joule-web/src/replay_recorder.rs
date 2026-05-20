//! Game replay recording and playback — timestamped inputs, keyframes, seek.
//!
//! Replaces replay.js / OpenReplay-game with pure Rust.
//! Record timestamped input events with initial state snapshots,
//! delta encoding, variable-length integers, keyframe snapshots,
//! playback speed control, seek, clip extraction, and file format.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    NoEventsRecorded,
    TimestampOutOfOrder { prev: u64, current: u64 },
    SeekOutOfRange { target_ms: u64, duration_ms: u64 },
    InvalidSpeed(String),
    CorruptedData(String),
    InvalidMagic,
    ChecksumMismatch { expected: u32, actual: u32 },
    ClipRangeInvalid { start_ms: u64, end_ms: u64 },
    NoKeyframeFound,
    EmptyReplay,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoEventsRecorded => write!(f, "no events recorded"),
            Self::TimestampOutOfOrder { prev, current } => {
                write!(f, "timestamp out of order: {current} < {prev}")
            }
            Self::SeekOutOfRange { target_ms, duration_ms } => {
                write!(f, "seek {target_ms}ms out of range (duration {duration_ms}ms)")
            }
            Self::InvalidSpeed(msg) => write!(f, "invalid speed: {msg}"),
            Self::CorruptedData(msg) => write!(f, "corrupted replay data: {msg}"),
            Self::InvalidMagic => write!(f, "invalid replay magic"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected:#010x}, got {actual:#010x}")
            }
            Self::ClipRangeInvalid { start_ms, end_ms } => {
                write!(f, "invalid clip range: {start_ms}..{end_ms}")
            }
            Self::NoKeyframeFound => write!(f, "no keyframe found for seek"),
            Self::EmptyReplay => write!(f, "replay is empty"),
        }
    }
}

impl std::error::Error for ReplayError {}

// ── Variable-Length Integer Encoding ────────────────────────────

fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

fn decode_varint(data: &[u8], offset: &mut usize) -> Result<u64, ReplayError> {
    let mut result = 0u64;
    let mut shift = 0;
    loop {
        if *offset >= data.len() {
            return Err(ReplayError::CorruptedData("truncated varint".into()));
        }
        let byte = data[*offset];
        *offset += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(ReplayError::CorruptedData("varint too long".into()));
        }
    }
    Ok(result)
}

// ── CRC32 ───────────────────────────────────────────────────────

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
}

// ── Input Event ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputKind {
    KeyDown(u16),
    KeyUp(u16),
    MouseMove { x: i32, y: i32 },
    MouseButton { button: u8, pressed: bool },
    GamepadAxis { axis: u8, value: i16 },
    GamepadButton { button: u8, pressed: bool },
    Custom(Vec<u8>),
}

impl InputKind {
    fn type_tag(&self) -> u8 {
        match self {
            Self::KeyDown(_) => 0,
            Self::KeyUp(_) => 1,
            Self::MouseMove { .. } => 2,
            Self::MouseButton { .. } => 3,
            Self::GamepadAxis { .. } => 4,
            Self::GamepadButton { .. } => 5,
            Self::Custom(_) => 6,
        }
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        buf.push(self.type_tag());
        match self {
            Self::KeyDown(code) => buf.extend_from_slice(&code.to_le_bytes()),
            Self::KeyUp(code) => buf.extend_from_slice(&code.to_le_bytes()),
            Self::MouseMove { x, y } => {
                buf.extend_from_slice(&x.to_le_bytes());
                buf.extend_from_slice(&y.to_le_bytes());
            }
            Self::MouseButton { button, pressed } => {
                buf.push(*button);
                buf.push(if *pressed { 1 } else { 0 });
            }
            Self::GamepadAxis { axis, value } => {
                buf.push(*axis);
                buf.extend_from_slice(&value.to_le_bytes());
            }
            Self::GamepadButton { button, pressed } => {
                buf.push(*button);
                buf.push(if *pressed { 1 } else { 0 });
            }
            Self::Custom(data) => {
                encode_varint(data.len() as u64, buf);
                buf.extend_from_slice(data);
            }
        }
    }

    fn decode(data: &[u8], offset: &mut usize) -> Result<Self, ReplayError> {
        if *offset >= data.len() {
            return Err(ReplayError::CorruptedData("truncated input kind".into()));
        }
        let tag = data[*offset];
        *offset += 1;
        match tag {
            0 => {
                if *offset + 2 > data.len() { return Err(ReplayError::CorruptedData("truncated key code".into())); }
                let code = u16::from_le_bytes([data[*offset], data[*offset + 1]]);
                *offset += 2;
                Ok(Self::KeyDown(code))
            }
            1 => {
                if *offset + 2 > data.len() { return Err(ReplayError::CorruptedData("truncated key code".into())); }
                let code = u16::from_le_bytes([data[*offset], data[*offset + 1]]);
                *offset += 2;
                Ok(Self::KeyUp(code))
            }
            2 => {
                if *offset + 8 > data.len() { return Err(ReplayError::CorruptedData("truncated mouse move".into())); }
                let x = i32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
                *offset += 4;
                let y = i32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
                *offset += 4;
                Ok(Self::MouseMove { x, y })
            }
            3 => {
                if *offset + 2 > data.len() { return Err(ReplayError::CorruptedData("truncated mouse button".into())); }
                let button = data[*offset];
                let pressed = data[*offset + 1] != 0;
                *offset += 2;
                Ok(Self::MouseButton { button, pressed })
            }
            4 => {
                if *offset + 3 > data.len() { return Err(ReplayError::CorruptedData("truncated gamepad axis".into())); }
                let axis = data[*offset];
                *offset += 1;
                let value = i16::from_le_bytes([data[*offset], data[*offset + 1]]);
                *offset += 2;
                Ok(Self::GamepadAxis { axis, value })
            }
            5 => {
                if *offset + 2 > data.len() { return Err(ReplayError::CorruptedData("truncated gamepad button".into())); }
                let button = data[*offset];
                let pressed = data[*offset + 1] != 0;
                *offset += 2;
                Ok(Self::GamepadButton { button, pressed })
            }
            6 => {
                let len = decode_varint(data, offset)? as usize;
                if *offset + len > data.len() { return Err(ReplayError::CorruptedData("truncated custom data".into())); }
                let custom_data = data[*offset..*offset + len].to_vec();
                *offset += len;
                Ok(Self::Custom(custom_data))
            }
            _ => Err(ReplayError::CorruptedData(format!("unknown input tag: {tag}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputEvent {
    pub timestamp_ms: u64,
    pub kind: InputKind,
}

// ── Keyframe ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Keyframe {
    pub timestamp_ms: u64,
    pub state_snapshot: Vec<u8>,
    pub event_index: usize,
}

// ── Replay Recorder ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ReplayRecorder {
    initial_state: Vec<u8>,
    events: Vec<InputEvent>,
    keyframes: Vec<Keyframe>,
    keyframe_interval_ms: u64,
    last_keyframe_ms: u64,
    version: u32,
}

const REPLAY_MAGIC: [u8; 4] = [0x52, 0x50, 0x4C, 0x59]; // "RPLY"
const REPLAY_VERSION: u32 = 1;

impl ReplayRecorder {
    pub fn new(initial_state: Vec<u8>, keyframe_interval_ms: u64) -> Self {
        Self {
            initial_state,
            events: Vec::new(),
            keyframes: Vec::new(),
            keyframe_interval_ms,
            last_keyframe_ms: 0,
            version: REPLAY_VERSION,
        }
    }

    pub fn record_event(&mut self, timestamp_ms: u64, kind: InputKind) -> Result<(), ReplayError> {
        if let Some(last) = self.events.last() {
            if timestamp_ms < last.timestamp_ms {
                return Err(ReplayError::TimestampOutOfOrder {
                    prev: last.timestamp_ms, current: timestamp_ms,
                });
            }
        }
        self.events.push(InputEvent { timestamp_ms, kind });
        Ok(())
    }

    pub fn add_keyframe(&mut self, timestamp_ms: u64, state_snapshot: Vec<u8>) {
        let event_index = self.events.len();
        self.keyframes.push(Keyframe { timestamp_ms, state_snapshot, event_index });
        self.last_keyframe_ms = timestamp_ms;
    }

    pub fn should_keyframe(&self, current_ms: u64) -> bool {
        current_ms >= self.last_keyframe_ms + self.keyframe_interval_ms
    }

    pub fn event_count(&self) -> usize { self.events.len() }
    pub fn keyframe_count(&self) -> usize { self.keyframes.len() }

    pub fn duration_ms(&self) -> u64 {
        self.events.last().map(|e| e.timestamp_ms).unwrap_or(0)
    }

    pub fn events(&self) -> &[InputEvent] { &self.events }
    pub fn initial_state(&self) -> &[u8] { &self.initial_state }

    /// Encode to binary replay file.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&REPLAY_MAGIC);
        buf.extend_from_slice(&self.version.to_le_bytes());
        // Placeholder for checksum
        let checksum_offset = buf.len();
        buf.extend_from_slice(&[0u8; 4]);
        // Duration
        buf.extend_from_slice(&self.duration_ms().to_le_bytes());
        // Initial state
        encode_varint(self.initial_state.len() as u64, &mut buf);
        buf.extend_from_slice(&self.initial_state);
        // Events (delta encoded)
        encode_varint(self.events.len() as u64, &mut buf);
        let mut prev_ts = 0u64;
        for event in &self.events {
            let delta = event.timestamp_ms - prev_ts;
            encode_varint(delta, &mut buf);
            prev_ts = event.timestamp_ms;
            event.kind.encode(&mut buf);
        }
        // Keyframes
        encode_varint(self.keyframes.len() as u64, &mut buf);
        for kf in &self.keyframes {
            buf.extend_from_slice(&kf.timestamp_ms.to_le_bytes());
            encode_varint(kf.event_index as u64, &mut buf);
            encode_varint(kf.state_snapshot.len() as u64, &mut buf);
            buf.extend_from_slice(&kf.state_snapshot);
        }
        // Checksum
        let checksum = crc32(&buf[checksum_offset + 4..]);
        buf[checksum_offset..checksum_offset + 4].copy_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Decode from binary replay file.
    pub fn decode(data: &[u8]) -> Result<Self, ReplayError> {
        if data.len() < 16 {
            return Err(ReplayError::CorruptedData("too small".into()));
        }
        if data[0..4] != REPLAY_MAGIC {
            return Err(ReplayError::InvalidMagic);
        }
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let stored_checksum = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let computed_checksum = crc32(&data[12..]);
        if stored_checksum != computed_checksum {
            return Err(ReplayError::ChecksumMismatch {
                expected: stored_checksum, actual: computed_checksum,
            });
        }
        let mut offset = 12;
        // Duration (skip, recomputed)
        if offset + 8 > data.len() { return Err(ReplayError::CorruptedData("truncated duration".into())); }
        let _duration = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;
        // Initial state
        let state_len = decode_varint(data, &mut offset)? as usize;
        if offset + state_len > data.len() { return Err(ReplayError::CorruptedData("truncated initial state".into())); }
        let initial_state = data[offset..offset + state_len].to_vec();
        offset += state_len;
        // Events
        let event_count = decode_varint(data, &mut offset)? as usize;
        let mut events = Vec::with_capacity(event_count);
        let mut prev_ts = 0u64;
        for _ in 0..event_count {
            let delta = decode_varint(data, &mut offset)?;
            let timestamp_ms = prev_ts + delta;
            prev_ts = timestamp_ms;
            let kind = InputKind::decode(data, &mut offset)?;
            events.push(InputEvent { timestamp_ms, kind });
        }
        // Keyframes
        let kf_count = decode_varint(data, &mut offset)? as usize;
        let mut keyframes = Vec::with_capacity(kf_count);
        for _ in 0..kf_count {
            if offset + 8 > data.len() { return Err(ReplayError::CorruptedData("truncated keyframe".into())); }
            let timestamp_ms = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let event_index = decode_varint(data, &mut offset)? as usize;
            let snap_len = decode_varint(data, &mut offset)? as usize;
            if offset + snap_len > data.len() { return Err(ReplayError::CorruptedData("truncated keyframe snapshot".into())); }
            let state_snapshot = data[offset..offset + snap_len].to_vec();
            offset += snap_len;
            keyframes.push(Keyframe { timestamp_ms, state_snapshot, event_index });
        }
        let last_kf_ms = keyframes.last().map(|kf| kf.timestamp_ms).unwrap_or(0);
        Ok(Self {
            initial_state,
            events,
            keyframes,
            keyframe_interval_ms: 5000, // default
            last_keyframe_ms: last_kf_ms,
            version,
        })
    }

    pub fn estimated_file_size(&self) -> usize {
        let header = 20;
        let state = self.initial_state.len() + 5;
        let events_est = self.events.len() * 8; // rough
        let keyframes_est: usize = self.keyframes.iter().map(|kf| 16 + kf.state_snapshot.len()).sum();
        header + state + events_est + keyframes_est
    }
}

// ── Playback ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ReplayPlayer {
    events: Vec<InputEvent>,
    keyframes: Vec<Keyframe>,
    initial_state: Vec<u8>,
    current_index: usize,
    current_time_ms: u64,
    duration_ms: u64,
    speed: f64,
    paused: bool,
}

impl ReplayPlayer {
    pub fn from_recorder(recorder: &ReplayRecorder) -> Result<Self, ReplayError> {
        if recorder.events.is_empty() {
            return Err(ReplayError::EmptyReplay);
        }
        Ok(Self {
            events: recorder.events.clone(),
            keyframes: recorder.keyframes.clone(),
            initial_state: recorder.initial_state.clone(),
            current_index: 0,
            current_time_ms: 0,
            duration_ms: recorder.duration_ms(),
            speed: 1.0,
            paused: false,
        })
    }

    pub fn set_speed(&mut self, speed: f64) -> Result<(), ReplayError> {
        if speed <= 0.0 || speed > 16.0 {
            return Err(ReplayError::InvalidSpeed(format!("{speed} not in (0, 16]")));
        }
        self.speed = speed;
        Ok(())
    }

    pub fn speed(&self) -> f64 { self.speed }
    pub fn is_paused(&self) -> bool { self.paused }
    pub fn pause(&mut self) { self.paused = true; }
    pub fn resume(&mut self) { self.paused = false; }
    pub fn current_time_ms(&self) -> u64 { self.current_time_ms }
    pub fn duration_ms(&self) -> u64 { self.duration_ms }
    pub fn is_finished(&self) -> bool { self.current_index >= self.events.len() }
    pub fn progress(&self) -> f64 {
        if self.duration_ms == 0 { return 1.0; }
        self.current_time_ms as f64 / self.duration_ms as f64
    }

    /// Advance by real-time delta, return events that fired.
    pub fn advance(&mut self, real_delta_ms: u64) -> Vec<&InputEvent> {
        if self.paused || self.is_finished() { return Vec::new(); }
        let game_delta = (real_delta_ms as f64 * self.speed) as u64;
        let target_time = self.current_time_ms + game_delta;
        let mut fired = Vec::new();
        while self.current_index < self.events.len() {
            if self.events[self.current_index].timestamp_ms <= target_time {
                fired.push(&self.events[self.current_index]);
                self.current_index += 1;
            } else {
                break;
            }
        }
        self.current_time_ms = target_time.min(self.duration_ms);
        fired
    }

    /// Seek to a timestamp, using nearest prior keyframe.
    pub fn seek(&mut self, target_ms: u64) -> Result<Option<&Keyframe>, ReplayError> {
        if target_ms > self.duration_ms {
            return Err(ReplayError::SeekOutOfRange {
                target_ms, duration_ms: self.duration_ms,
            });
        }
        // Find the nearest keyframe at or before target
        let kf = self.keyframes.iter().rev().find(|kf| kf.timestamp_ms <= target_ms);
        if let Some(kf) = kf {
            self.current_index = kf.event_index;
            self.current_time_ms = kf.timestamp_ms;
            // Advance to exact target
            while self.current_index < self.events.len()
                && self.events[self.current_index].timestamp_ms <= target_ms
            {
                self.current_index += 1;
            }
            self.current_time_ms = target_ms;
            // Return reference to the keyframe in self.keyframes
            let kf_idx = self.keyframes.iter().rposition(|k| k.timestamp_ms <= target_ms).unwrap();
            Ok(Some(&self.keyframes[kf_idx]))
        } else {
            // No keyframe — start from beginning
            self.current_index = 0;
            self.current_time_ms = 0;
            while self.current_index < self.events.len()
                && self.events[self.current_index].timestamp_ms <= target_ms
            {
                self.current_index += 1;
            }
            self.current_time_ms = target_ms;
            Ok(None)
        }
    }
}

// ── Clip Extraction ─────────────────────────────────────────────

pub fn extract_clip(
    recorder: &ReplayRecorder,
    start_ms: u64,
    end_ms: u64,
) -> Result<ReplayRecorder, ReplayError> {
    if start_ms >= end_ms {
        return Err(ReplayError::ClipRangeInvalid { start_ms, end_ms });
    }
    if end_ms > recorder.duration_ms() {
        return Err(ReplayError::ClipRangeInvalid { start_ms, end_ms });
    }
    // Find the closest keyframe at or before start
    let snap = recorder.keyframes.iter().rev()
        .find(|kf| kf.timestamp_ms <= start_ms)
        .map(|kf| kf.state_snapshot.clone())
        .unwrap_or_else(|| recorder.initial_state.clone());

    let events: Vec<InputEvent> = recorder.events.iter()
        .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
        .cloned()
        .map(|mut e| { e.timestamp_ms -= start_ms; e })
        .collect();

    let keyframes: Vec<Keyframe> = recorder.keyframes.iter()
        .filter(|kf| kf.timestamp_ms >= start_ms && kf.timestamp_ms <= end_ms)
        .cloned()
        .map(|mut kf| {
            kf.timestamp_ms -= start_ms;
            // Recompute event_index relative to clip
            kf.event_index = events.iter()
                .position(|e| e.timestamp_ms >= kf.timestamp_ms)
                .unwrap_or(events.len());
            kf
        })
        .collect();

    let mut clip = ReplayRecorder::new(snap, recorder.keyframe_interval_ms);
    clip.events = events;
    clip.keyframes = keyframes;
    Ok(clip)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_replay() -> ReplayRecorder {
        let mut rec = ReplayRecorder::new(vec![0, 1, 2, 3], 5000);
        rec.record_event(0, InputKind::KeyDown(65)).unwrap();
        rec.record_event(100, InputKind::MouseMove { x: 10, y: 20 }).unwrap();
        rec.record_event(200, InputKind::KeyUp(65)).unwrap();
        rec.record_event(500, InputKind::MouseButton { button: 0, pressed: true }).unwrap();
        rec.record_event(1000, InputKind::GamepadAxis { axis: 0, value: -100 }).unwrap();
        rec.add_keyframe(500, vec![10, 20, 30]);
        rec.record_event(2000, InputKind::KeyDown(66)).unwrap();
        rec.record_event(3000, InputKind::Custom(vec![42, 43])).unwrap();
        rec.add_keyframe(2000, vec![40, 50]);
        rec
    }

    #[test]
    fn record_events() {
        let rec = build_replay();
        assert_eq!(rec.event_count(), 7);
        assert_eq!(rec.duration_ms(), 3000);
    }

    #[test]
    fn timestamp_order() {
        let mut rec = ReplayRecorder::new(vec![], 5000);
        rec.record_event(100, InputKind::KeyDown(65)).unwrap();
        let err = rec.record_event(50, InputKind::KeyDown(66)).unwrap_err();
        assert!(matches!(err, ReplayError::TimestampOutOfOrder { .. }));
    }

    #[test]
    fn keyframe_tracking() {
        let rec = build_replay();
        assert_eq!(rec.keyframe_count(), 2);
    }

    #[test]
    fn should_keyframe() {
        let rec = ReplayRecorder::new(vec![], 5000);
        assert!(rec.should_keyframe(5000));
        assert!(!rec.should_keyframe(4999));
    }

    #[test]
    fn varint_roundtrip() {
        let values = [0, 1, 127, 128, 255, 300, 65535, 1_000_000, u64::MAX];
        for &val in &values {
            let mut buf = Vec::new();
            encode_varint(val, &mut buf);
            let mut offset = 0;
            let decoded = decode_varint(&buf, &mut offset).unwrap();
            assert_eq!(decoded, val, "varint roundtrip failed for {val}");
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let rec = build_replay();
        let encoded = rec.encode();
        let decoded = ReplayRecorder::decode(&encoded).unwrap();
        assert_eq!(decoded.event_count(), 7);
        assert_eq!(decoded.initial_state(), &[0, 1, 2, 3]);
        assert_eq!(decoded.duration_ms(), 3000);
        assert_eq!(decoded.keyframe_count(), 2);
        // Verify event content
        assert_eq!(decoded.events()[0].kind, InputKind::KeyDown(65));
        assert_eq!(decoded.events()[1].kind, InputKind::MouseMove { x: 10, y: 20 });
    }

    #[test]
    fn invalid_magic() {
        let mut data = build_replay().encode();
        data[0] = 0xFF;
        let err = ReplayRecorder::decode(&data).unwrap_err();
        assert!(matches!(err, ReplayError::InvalidMagic));
    }

    #[test]
    fn checksum_mismatch() {
        let mut data = build_replay().encode();
        let last = data.len() - 1;
        data[last] ^= 0xFF;
        let err = ReplayRecorder::decode(&data).unwrap_err();
        assert!(matches!(err, ReplayError::ChecksumMismatch { .. }));
    }

    #[test]
    fn playback_basic() {
        let rec = build_replay();
        let mut player = ReplayPlayer::from_recorder(&rec).unwrap();
        assert!(!player.is_finished());
        assert!((player.progress() - 0.0).abs() < 1e-6);
        // advance(150) at speed 1.0 means target = 150ms
        // Events at 0ms and 100ms fire (both <= 150). Event at 200ms does not.
        let events = player.advance(150);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn playback_advance_collects_events() {
        let rec = build_replay();
        let mut player = ReplayPlayer::from_recorder(&rec).unwrap();
        let events = player.advance(250);
        // Events at 0, 100, 200 all fire (all <= 250)
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn playback_speed() {
        let rec = build_replay();
        let mut player = ReplayPlayer::from_recorder(&rec).unwrap();
        player.set_speed(2.0).unwrap();
        // Advance 500 real ms = 1000 game ms at 2x
        let events = player.advance(500);
        // Events at 0, 100, 200, 500, 1000 all fire (<= 1000)
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn playback_pause() {
        let rec = build_replay();
        let mut player = ReplayPlayer::from_recorder(&rec).unwrap();
        player.pause();
        let events = player.advance(5000);
        assert!(events.is_empty());
        player.resume();
        let events = player.advance(150);
        assert_eq!(events.len(), 2); // 0, 100
    }

    #[test]
    fn playback_invalid_speed() {
        let rec = build_replay();
        let mut player = ReplayPlayer::from_recorder(&rec).unwrap();
        assert!(player.set_speed(0.0).is_err());
        assert!(player.set_speed(17.0).is_err());
        assert!(player.set_speed(0.25).is_ok());
    }

    #[test]
    fn seek_with_keyframe() {
        let rec = build_replay();
        let mut player = ReplayPlayer::from_recorder(&rec).unwrap();
        let kf = player.seek(1500).unwrap();
        assert!(kf.is_some());
        let kf = kf.unwrap();
        assert_eq!(kf.timestamp_ms, 500); // nearest keyframe before 1500
    }

    #[test]
    fn seek_before_first_keyframe() {
        let rec = build_replay();
        let mut player = ReplayPlayer::from_recorder(&rec).unwrap();
        let kf = player.seek(50).unwrap();
        assert!(kf.is_none()); // no keyframe before 50ms
    }

    #[test]
    fn seek_out_of_range() {
        let rec = build_replay();
        let mut player = ReplayPlayer::from_recorder(&rec).unwrap();
        let err = player.seek(99999).unwrap_err();
        assert!(matches!(err, ReplayError::SeekOutOfRange { .. }));
    }

    #[test]
    fn clip_extraction() {
        let rec = build_replay();
        let clip = extract_clip(&rec, 100, 2000).unwrap();
        // Events between 100 and 2000: at 100, 200, 500, 1000, 2000
        assert_eq!(clip.event_count(), 5);
        // Timestamps should be rebased: 0, 100, 400, 900, 1900
        assert_eq!(clip.events()[0].timestamp_ms, 0);
        assert_eq!(clip.events()[1].timestamp_ms, 100);
    }

    #[test]
    fn clip_invalid_range() {
        let rec = build_replay();
        let err = extract_clip(&rec, 2000, 1000).unwrap_err();
        assert!(matches!(err, ReplayError::ClipRangeInvalid { .. }));
    }

    #[test]
    fn clip_out_of_bounds() {
        let rec = build_replay();
        let err = extract_clip(&rec, 0, 99999).unwrap_err();
        assert!(matches!(err, ReplayError::ClipRangeInvalid { .. }));
    }

    #[test]
    fn estimated_file_size() {
        let rec = build_replay();
        let est = rec.estimated_file_size();
        let actual = rec.encode().len();
        // Estimate should be in the right ballpark (within 3x)
        assert!(est > 0);
        assert!((est as f64) < (actual as f64) * 3.0);
    }

    #[test]
    fn empty_replay_player() {
        let rec = ReplayRecorder::new(vec![], 5000);
        let err = ReplayPlayer::from_recorder(&rec).unwrap_err();
        assert!(matches!(err, ReplayError::EmptyReplay));
    }

    #[test]
    fn crc32_known_values() {
        assert_eq!(crc32(b""), 0x00000000);
        assert_eq!(crc32(b"hello"), 0x3610a686);
    }

    #[test]
    fn gamepad_button_roundtrip() {
        let mut rec = ReplayRecorder::new(vec![1], 5000);
        rec.record_event(0, InputKind::GamepadButton { button: 3, pressed: true }).unwrap();
        let data = rec.encode();
        let dec = ReplayRecorder::decode(&data).unwrap();
        assert_eq!(dec.events()[0].kind, InputKind::GamepadButton { button: 3, pressed: true });
    }

    #[test]
    fn custom_event_roundtrip() {
        let mut rec = ReplayRecorder::new(vec![1], 5000);
        let payload = vec![1, 2, 3, 4, 5, 6, 7, 8];
        rec.record_event(0, InputKind::Custom(payload.clone())).unwrap();
        let data = rec.encode();
        let dec = ReplayRecorder::decode(&data).unwrap();
        assert_eq!(dec.events()[0].kind, InputKind::Custom(payload));
    }

    #[test]
    fn playback_progress_and_finish() {
        let mut rec = ReplayRecorder::new(vec![1], 5000);
        rec.record_event(0, InputKind::KeyDown(65)).unwrap();
        rec.record_event(1000, InputKind::KeyUp(65)).unwrap();
        let mut player = ReplayPlayer::from_recorder(&rec).unwrap();
        player.advance(1500);
        assert!(player.is_finished());
        assert!((player.progress() - 1.0).abs() < 1e-6);
    }
}
