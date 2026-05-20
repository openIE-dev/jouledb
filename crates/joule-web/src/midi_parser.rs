//! MIDI file and message parser.
//!
//! Parses MIDI messages (Note On/Off, Control Change, Program Change,
//! Pitch Bend, System Exclusive), MIDI file formats 0 and 1, variable-length
//! quantity encoding, tempo maps, and note-name conversions. Pure Rust — no
//! external audio libraries.

use std::collections::HashMap;
use std::fmt;

// ── MIDI Note Names ──────────────────────────────────────────────

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Convert a note name like "C4" or "D#3" to a MIDI number (C4 = 60).
pub fn note_name_to_midi(name: &str) -> Option<u8> {
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    let (note_part, oct_part) = if name.len() >= 2 && name.as_bytes()[1] == b'#' {
        (&name[..2], &name[2..])
    } else if name.len() >= 2 && name.as_bytes()[1] == b'b' {
        // Flat: treat Db as C#, Eb as D#, etc.
        let base = match name.as_bytes()[0] {
            b'D' => "C#",
            b'E' => "D#",
            b'G' => "F#",
            b'A' => "G#",
            b'B' => "A#",
            b'C' => "B",
            b'F' => "E",
            _ => return None,
        };
        return note_name_to_midi(&format!("{}{}", base, &name[2..]));
    } else {
        (&name[..1], &name[1..])
    };

    let semitone = NOTE_NAMES.iter().position(|n| *n == note_part)? as i16;
    let octave: i16 = oct_part.parse().ok()?;
    let midi = (octave + 1) * 12 + semitone;
    if midi < 0 || midi > 127 {
        None
    } else {
        Some(midi as u8)
    }
}

/// Convert a MIDI number to a note name (e.g. 60 -> "C4").
pub fn midi_to_note_name(midi: u8) -> String {
    let note = NOTE_NAMES[(midi % 12) as usize];
    let octave = (midi as i16 / 12) - 1;
    format!("{}{}", note, octave)
}

// ── Variable-Length Quantity ──────────────────────────────────────

/// Encode a u32 value as MIDI variable-length quantity bytes.
pub fn encode_vlq(mut value: u32) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }
    let mut bytes = Vec::new();
    bytes.push((value & 0x7F) as u8);
    value >>= 7;
    while value > 0 {
        bytes.push((value & 0x7F) as u8 | 0x80);
        value >>= 7;
    }
    bytes.reverse();
    bytes
}

/// Decode a variable-length quantity from a byte slice. Returns (value, bytes_consumed).
pub fn decode_vlq(data: &[u8]) -> Option<(u32, usize)> {
    let mut value: u32 = 0;
    for (i, &byte) in data.iter().enumerate() {
        if i >= 4 {
            return None; // VLQ should not exceed 4 bytes
        }
        value = (value << 7) | (byte & 0x7F) as u32;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

// ── MIDI Messages ────────────────────────────────────────────────

/// A parsed MIDI channel message.
#[derive(Debug, Clone, PartialEq)]
pub enum MidiMessage {
    NoteOff { channel: u8, note: u8, velocity: u8 },
    NoteOn { channel: u8, note: u8, velocity: u8 },
    ControlChange { channel: u8, controller: u8, value: u8 },
    ProgramChange { channel: u8, program: u8 },
    ChannelPressure { channel: u8, pressure: u8 },
    PitchBend { channel: u8, value: u16 },
    SystemExclusive { data: Vec<u8> },
    MetaTempo { microseconds_per_beat: u32 },
    MetaTimeSignature { numerator: u8, denominator_power: u8 },
    MetaEndOfTrack,
    MetaTrackName { name: String },
    Unknown { status: u8, data: Vec<u8> },
}

impl fmt::Display for MidiMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MidiMessage::NoteOn { channel, note, velocity } => {
                write!(f, "NoteOn ch={} note={} vel={}", channel, midi_to_note_name(*note), velocity)
            }
            MidiMessage::NoteOff { channel, note, velocity } => {
                write!(f, "NoteOff ch={} note={} vel={}", channel, midi_to_note_name(*note), velocity)
            }
            MidiMessage::ControlChange { channel, controller, value } => {
                write!(f, "CC ch={} ctrl={} val={}", channel, controller, value)
            }
            MidiMessage::ProgramChange { channel, program } => {
                write!(f, "PC ch={} prog={}", channel, program)
            }
            MidiMessage::PitchBend { channel, value } => {
                write!(f, "PitchBend ch={} val={}", channel, value)
            }
            MidiMessage::SystemExclusive { data } => {
                write!(f, "SysEx [{} bytes]", data.len())
            }
            MidiMessage::MetaTempo { microseconds_per_beat } => {
                let bpm = 60_000_000.0 / *microseconds_per_beat as f64;
                write!(f, "Tempo {:.1} BPM", bpm)
            }
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Serialize a MIDI message back to bytes (channel messages only).
pub fn serialize_message(msg: &MidiMessage) -> Vec<u8> {
    match msg {
        MidiMessage::NoteOff { channel, note, velocity } => {
            vec![0x80 | (channel & 0x0F), *note & 0x7F, *velocity & 0x7F]
        }
        MidiMessage::NoteOn { channel, note, velocity } => {
            vec![0x90 | (channel & 0x0F), *note & 0x7F, *velocity & 0x7F]
        }
        MidiMessage::ControlChange { channel, controller, value } => {
            vec![0xB0 | (channel & 0x0F), *controller & 0x7F, *value & 0x7F]
        }
        MidiMessage::ProgramChange { channel, program } => {
            vec![0xC0 | (channel & 0x0F), *program & 0x7F]
        }
        MidiMessage::ChannelPressure { channel, pressure } => {
            vec![0xD0 | (channel & 0x0F), *pressure & 0x7F]
        }
        MidiMessage::PitchBend { channel, value } => {
            let lsb = (*value & 0x7F) as u8;
            let msb = ((*value >> 7) & 0x7F) as u8;
            vec![0xE0 | (channel & 0x0F), lsb, msb]
        }
        MidiMessage::SystemExclusive { data } => {
            let mut out = vec![0xF0];
            out.extend_from_slice(data);
            out.push(0xF7);
            out
        }
        MidiMessage::MetaTempo { microseconds_per_beat } => {
            let v = *microseconds_per_beat;
            vec![0xFF, 0x51, 0x03, (v >> 16) as u8, (v >> 8) as u8, v as u8]
        }
        MidiMessage::MetaEndOfTrack => vec![0xFF, 0x2F, 0x00],
        MidiMessage::MetaTimeSignature { numerator, denominator_power } => {
            vec![0xFF, 0x58, 0x04, *numerator, *denominator_power, 24, 8]
        }
        MidiMessage::MetaTrackName { name } => {
            let mut out = vec![0xFF, 0x03];
            out.extend_from_slice(&encode_vlq(name.len() as u32));
            out.extend_from_slice(name.as_bytes());
            out
        }
        MidiMessage::Unknown { status, data } => {
            let mut out = vec![*status];
            out.extend_from_slice(data);
            out
        }
    }
}

/// A timed MIDI event (delta time + message).
#[derive(Debug, Clone, PartialEq)]
pub struct MidiEvent {
    pub delta_ticks: u32,
    pub message: MidiMessage,
}

// ── MIDI File Parsing ────────────────────────────────────────────

/// MIDI file format type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MidiFormat {
    SingleTrack,   // Format 0
    MultiTrack,    // Format 1
}

/// A parsed MIDI file.
#[derive(Debug, Clone, PartialEq)]
pub struct MidiFile {
    pub format: MidiFormat,
    pub ticks_per_quarter: u16,
    pub tracks: Vec<Vec<MidiEvent>>,
}

/// Tempo map entry: at a given tick, the tempo changes.
#[derive(Debug, Clone)]
pub struct TempoEntry {
    pub tick: u64,
    pub microseconds_per_beat: u32,
}

/// Build a tempo map from a track's events.
pub fn build_tempo_map(events: &[MidiEvent]) -> Vec<TempoEntry> {
    let mut map = Vec::new();
    let mut tick: u64 = 0;
    for ev in events {
        tick += ev.delta_ticks as u64;
        if let MidiMessage::MetaTempo { microseconds_per_beat } = &ev.message {
            map.push(TempoEntry { tick, microseconds_per_beat: *microseconds_per_beat });
        }
    }
    if map.is_empty() || map[0].tick != 0 {
        map.insert(0, TempoEntry { tick: 0, microseconds_per_beat: 500_000 }); // 120 BPM default
    }
    map
}

/// Convert a tick position to microseconds using a tempo map.
pub fn tick_to_microseconds(tick: u64, ticks_per_quarter: u16, tempo_map: &[TempoEntry]) -> f64 {
    let mut us = 0.0;
    let mut prev_tick: u64 = 0;
    let mut prev_tempo: u32 = 500_000;
    let tpq = ticks_per_quarter as f64;

    for entry in tempo_map {
        if entry.tick >= tick {
            break;
        }
        let dt = entry.tick - prev_tick;
        us += dt as f64 * prev_tempo as f64 / tpq;
        prev_tick = entry.tick;
        prev_tempo = entry.microseconds_per_beat;
    }

    let remaining = tick - prev_tick;
    us += remaining as f64 * prev_tempo as f64 / tpq;
    us
}

/// Parse a MIDI message from raw bytes with running status support.
/// Returns (message, bytes_consumed).
pub fn parse_message(data: &[u8], running_status: u8) -> Option<(MidiMessage, usize)> {
    if data.is_empty() {
        return None;
    }

    let (status, offset) = if data[0] & 0x80 != 0 {
        (data[0], 1)
    } else {
        (running_status, 0)
    };

    let remaining = &data[offset..];

    match status & 0xF0 {
        0x80 => {
            if remaining.len() < 2 { return None; }
            let ch = status & 0x0F;
            Some((MidiMessage::NoteOff { channel: ch, note: remaining[0], velocity: remaining[1] }, offset + 2))
        }
        0x90 => {
            if remaining.len() < 2 { return None; }
            let ch = status & 0x0F;
            let vel = remaining[1];
            if vel == 0 {
                Some((MidiMessage::NoteOff { channel: ch, note: remaining[0], velocity: 0 }, offset + 2))
            } else {
                Some((MidiMessage::NoteOn { channel: ch, note: remaining[0], velocity: vel }, offset + 2))
            }
        }
        0xA0 => {
            if remaining.len() < 2 { return None; }
            Some((MidiMessage::Unknown { status, data: remaining[..2].to_vec() }, offset + 2))
        }
        0xB0 => {
            if remaining.len() < 2 { return None; }
            let ch = status & 0x0F;
            Some((MidiMessage::ControlChange { channel: ch, controller: remaining[0], value: remaining[1] }, offset + 2))
        }
        0xC0 => {
            if remaining.is_empty() { return None; }
            let ch = status & 0x0F;
            Some((MidiMessage::ProgramChange { channel: ch, program: remaining[0] }, offset + 1))
        }
        0xD0 => {
            if remaining.is_empty() { return None; }
            let ch = status & 0x0F;
            Some((MidiMessage::ChannelPressure { channel: ch, pressure: remaining[0] }, offset + 1))
        }
        0xE0 => {
            if remaining.len() < 2 { return None; }
            let ch = status & 0x0F;
            let value = (remaining[0] as u16) | ((remaining[1] as u16) << 7);
            Some((MidiMessage::PitchBend { channel: ch, value }, offset + 2))
        }
        0xF0 => {
            match status {
                0xF0 => {
                    // SysEx: read until 0xF7
                    if let Some(end) = remaining.iter().position(|b| *b == 0xF7) {
                        Some((MidiMessage::SystemExclusive { data: remaining[..end].to_vec() }, offset + end + 1))
                    } else {
                        Some((MidiMessage::SystemExclusive { data: remaining.to_vec() }, offset + remaining.len()))
                    }
                }
                0xFF => {
                    // Meta event
                    if remaining.len() < 2 { return None; }
                    let meta_type = remaining[0];
                    let (length, vlq_len) = decode_vlq(&remaining[1..])?;
                    let data_start = 1 + vlq_len;
                    let data_end = data_start + length as usize;
                    if remaining.len() < data_end { return None; }
                    let meta_data = &remaining[data_start..data_end];

                    let msg = match meta_type {
                        0x03 => MidiMessage::MetaTrackName {
                            name: String::from_utf8_lossy(meta_data).into_owned(),
                        },
                        0x2F => MidiMessage::MetaEndOfTrack,
                        0x51 if meta_data.len() >= 3 => {
                            let uspb = ((meta_data[0] as u32) << 16)
                                | ((meta_data[1] as u32) << 8)
                                | meta_data[2] as u32;
                            MidiMessage::MetaTempo { microseconds_per_beat: uspb }
                        }
                        0x58 if meta_data.len() >= 2 => {
                            MidiMessage::MetaTimeSignature {
                                numerator: meta_data[0],
                                denominator_power: meta_data[1],
                            }
                        }
                        _ => MidiMessage::Unknown { status: 0xFF, data: meta_data.to_vec() },
                    };
                    Some((msg, offset + data_end))
                }
                _ => Some((MidiMessage::Unknown { status, data: vec![] }, offset)),
            }
        }
        _ => None,
    }
}

/// Parse raw MIDI file bytes into a MidiFile structure.
pub fn parse_midi_file(data: &[u8]) -> Option<MidiFile> {
    if data.len() < 14 { return None; }
    // Header chunk: MThd
    if &data[0..4] != b"MThd" { return None; }
    let header_len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if header_len < 6 { return None; }
    let format_raw = u16::from_be_bytes([data[8], data[9]]);
    let num_tracks = u16::from_be_bytes([data[10], data[11]]);
    let tpq = u16::from_be_bytes([data[12], data[13]]);

    let format = match format_raw {
        0 => MidiFormat::SingleTrack,
        1 => MidiFormat::MultiTrack,
        _ => return None,
    };

    let mut pos = 8 + header_len;
    let mut tracks = Vec::new();

    for _ in 0..num_tracks {
        if pos + 8 > data.len() { break; }
        if &data[pos..pos + 4] != b"MTrk" { break; }
        let track_len = u32::from_be_bytes([
            data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7],
        ]) as usize;
        pos += 8;
        let track_end = pos + track_len;
        if track_end > data.len() { break; }

        let mut events = Vec::new();
        let mut tpos = pos;
        let mut running = 0u8;

        while tpos < track_end {
            let (delta, vlq_len) = match decode_vlq(&data[tpos..track_end]) {
                Some(v) => v,
                None => break,
            };
            tpos += vlq_len;
            if tpos >= track_end { break; }

            match parse_message(&data[tpos..track_end], running) {
                Some((msg, consumed)) => {
                    if data[tpos] & 0x80 != 0 && data[tpos] < 0xF0 {
                        running = data[tpos];
                    }
                    events.push(MidiEvent { delta_ticks: delta, message: msg });
                    tpos += consumed;
                }
                None => break,
            }
        }

        tracks.push(events);
        pos = track_end;
    }

    Some(MidiFile { format, ticks_per_quarter: tpq, tracks })
}

/// Build a minimal MIDI file (format 0) from events.
pub fn build_midi_file(events: &[MidiEvent], tpq: u16) -> Vec<u8> {
    let mut track_data = Vec::new();
    for ev in events {
        track_data.extend_from_slice(&encode_vlq(ev.delta_ticks));
        track_data.extend_from_slice(&serialize_message(&ev.message));
    }

    let mut out = Vec::new();
    // MThd header
    out.extend_from_slice(b"MThd");
    out.extend_from_slice(&6u32.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // format 0
    out.extend_from_slice(&1u16.to_be_bytes()); // 1 track
    out.extend_from_slice(&tpq.to_be_bytes());

    // MTrk
    out.extend_from_slice(b"MTrk");
    out.extend_from_slice(&(track_data.len() as u32).to_be_bytes());
    out.extend_from_slice(&track_data);

    out
}

/// Extract all note events from a track, returning (tick, channel, note, velocity, duration_ticks).
pub fn extract_notes(events: &[MidiEvent]) -> Vec<(u64, u8, u8, u8, u64)> {
    let mut notes = Vec::new();
    let mut pending: HashMap<(u8, u8), (u64, u8)> = HashMap::new();
    let mut tick: u64 = 0;

    for ev in events {
        tick += ev.delta_ticks as u64;
        match &ev.message {
            MidiMessage::NoteOn { channel, note, velocity } => {
                pending.insert((*channel, *note), (tick, *velocity));
            }
            MidiMessage::NoteOff { channel, note, .. } => {
                if let Some((start_tick, vel)) = pending.remove(&(*channel, *note)) {
                    let dur = tick.saturating_sub(start_tick);
                    notes.push((start_tick, *channel, *note, vel, dur));
                }
            }
            _ => {}
        }
    }
    notes.sort_by_key(|n| n.0);
    notes
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_name_to_midi_c4() {
        assert_eq!(note_name_to_midi("C4"), Some(60));
    }

    #[test]
    fn test_note_name_to_midi_a4() {
        assert_eq!(note_name_to_midi("A4"), Some(69));
    }

    #[test]
    fn test_note_name_sharp() {
        assert_eq!(note_name_to_midi("C#4"), Some(61));
        assert_eq!(note_name_to_midi("F#3"), Some(54));
    }

    #[test]
    fn test_note_name_flat() {
        assert_eq!(note_name_to_midi("Db4"), note_name_to_midi("C#4"));
        assert_eq!(note_name_to_midi("Eb4"), note_name_to_midi("D#4"));
    }

    #[test]
    fn test_midi_to_note_name_roundtrip() {
        for midi in 0..=127u8 {
            let name = midi_to_note_name(midi);
            let back = note_name_to_midi(&name).unwrap();
            assert_eq!(back, midi);
        }
    }

    #[test]
    fn test_vlq_encode_zero() {
        assert_eq!(encode_vlq(0), vec![0]);
    }

    #[test]
    fn test_vlq_encode_small() {
        assert_eq!(encode_vlq(0x7F), vec![0x7F]);
    }

    #[test]
    fn test_vlq_encode_two_bytes() {
        assert_eq!(encode_vlq(0x80), vec![0x81, 0x00]);
    }

    #[test]
    fn test_vlq_encode_large() {
        assert_eq!(encode_vlq(0x0FFF_FFFF), vec![0xFF, 0xFF, 0xFF, 0x7F]);
    }

    #[test]
    fn test_vlq_roundtrip() {
        for val in [0, 1, 127, 128, 255, 1000, 16383, 2097151, 0x0FFF_FFFF] {
            let encoded = encode_vlq(val);
            let (decoded, len) = decode_vlq(&encoded).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(len, encoded.len());
        }
    }

    #[test]
    fn test_parse_note_on() {
        let data = [0x90, 60, 100];
        let (msg, consumed) = parse_message(&data, 0).unwrap();
        assert_eq!(consumed, 3);
        assert_eq!(msg, MidiMessage::NoteOn { channel: 0, note: 60, velocity: 100 });
    }

    #[test]
    fn test_parse_note_on_vel_zero_is_note_off() {
        let data = [0x91, 64, 0];
        let (msg, _) = parse_message(&data, 0).unwrap();
        assert_eq!(msg, MidiMessage::NoteOff { channel: 1, note: 64, velocity: 0 });
    }

    #[test]
    fn test_parse_control_change() {
        let data = [0xB2, 7, 100]; // CC#7 (volume) on ch2
        let (msg, _) = parse_message(&data, 0).unwrap();
        assert_eq!(msg, MidiMessage::ControlChange { channel: 2, controller: 7, value: 100 });
    }

    #[test]
    fn test_parse_program_change() {
        let data = [0xC5, 42];
        let (msg, consumed) = parse_message(&data, 0).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(msg, MidiMessage::ProgramChange { channel: 5, program: 42 });
    }

    #[test]
    fn test_parse_pitch_bend() {
        let data = [0xE0, 0x00, 0x40]; // center
        let (msg, _) = parse_message(&data, 0).unwrap();
        assert_eq!(msg, MidiMessage::PitchBend { channel: 0, value: 0x2000 });
    }

    #[test]
    fn test_running_status() {
        // Running status: second message uses running status from first
        let data = [60, 80]; // note=60, vel=80, running_status=0x90
        let (msg, consumed) = parse_message(&data, 0x90).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(msg, MidiMessage::NoteOn { channel: 0, note: 60, velocity: 80 });
    }

    #[test]
    fn test_serialize_roundtrip_note_on() {
        let msg = MidiMessage::NoteOn { channel: 3, note: 72, velocity: 110 };
        let bytes = serialize_message(&msg);
        let (parsed, _) = parse_message(&bytes, 0).unwrap();
        assert_eq!(parsed, msg);
    }

    #[test]
    fn test_serialize_roundtrip_cc() {
        let msg = MidiMessage::ControlChange { channel: 0, controller: 64, value: 127 };
        let bytes = serialize_message(&msg);
        let (parsed, _) = parse_message(&bytes, 0).unwrap();
        assert_eq!(parsed, msg);
    }

    #[test]
    fn test_serialize_sysex() {
        let msg = MidiMessage::SystemExclusive { data: vec![0x7E, 0x7F, 0x09, 0x01] };
        let bytes = serialize_message(&msg);
        assert_eq!(bytes[0], 0xF0);
        assert_eq!(*bytes.last().unwrap(), 0xF7);
    }

    #[test]
    fn test_build_and_parse_midi_file() {
        let events = vec![
            MidiEvent { delta_ticks: 0, message: MidiMessage::NoteOn { channel: 0, note: 60, velocity: 100 } },
            MidiEvent { delta_ticks: 480, message: MidiMessage::NoteOff { channel: 0, note: 60, velocity: 0 } },
            MidiEvent { delta_ticks: 0, message: MidiMessage::MetaEndOfTrack },
        ];
        let data = build_midi_file(&events, 480);
        let parsed = parse_midi_file(&data).unwrap();
        assert_eq!(parsed.format, MidiFormat::SingleTrack);
        assert_eq!(parsed.ticks_per_quarter, 480);
        assert_eq!(parsed.tracks.len(), 1);
        assert_eq!(parsed.tracks[0].len(), 3);
    }

    #[test]
    fn test_tempo_map_default() {
        let events = vec![
            MidiEvent { delta_ticks: 0, message: MidiMessage::NoteOn { channel: 0, note: 60, velocity: 100 } },
        ];
        let map = build_tempo_map(&events);
        assert_eq!(map.len(), 1);
        assert_eq!(map[0].microseconds_per_beat, 500_000); // 120 BPM
    }

    #[test]
    fn test_tempo_map_with_change() {
        let events = vec![
            MidiEvent { delta_ticks: 0, message: MidiMessage::MetaTempo { microseconds_per_beat: 600_000 } },
            MidiEvent { delta_ticks: 960, message: MidiMessage::MetaTempo { microseconds_per_beat: 400_000 } },
        ];
        let map = build_tempo_map(&events);
        assert_eq!(map.len(), 2);
        assert_eq!(map[0].microseconds_per_beat, 600_000);
        assert_eq!(map[1].microseconds_per_beat, 400_000);
    }

    #[test]
    fn test_tick_to_microseconds() {
        let map = vec![TempoEntry { tick: 0, microseconds_per_beat: 500_000 }];
        // At 120 BPM, 480 ticks = 1 beat = 500,000 us
        let us = tick_to_microseconds(480, 480, &map);
        assert!((us - 500_000.0).abs() < 1.0);
    }

    #[test]
    fn test_extract_notes() {
        let events = vec![
            MidiEvent { delta_ticks: 0, message: MidiMessage::NoteOn { channel: 0, note: 60, velocity: 100 } },
            MidiEvent { delta_ticks: 480, message: MidiMessage::NoteOff { channel: 0, note: 60, velocity: 0 } },
            MidiEvent { delta_ticks: 0, message: MidiMessage::NoteOn { channel: 0, note: 64, velocity: 90 } },
            MidiEvent { delta_ticks: 240, message: MidiMessage::NoteOff { channel: 0, note: 64, velocity: 0 } },
        ];
        let notes = extract_notes(&events);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0], (0, 0, 60, 100, 480));
        assert_eq!(notes[1], (480, 0, 64, 90, 240));
    }

    #[test]
    fn test_midi_display() {
        let msg = MidiMessage::NoteOn { channel: 0, note: 60, velocity: 100 };
        let s = format!("{}", msg);
        assert!(s.contains("NoteOn"));
        assert!(s.contains("C4"));
    }

    #[test]
    fn test_invalid_note_name() {
        assert_eq!(note_name_to_midi(""), None);
        assert_eq!(note_name_to_midi("X4"), None);
    }

    #[test]
    fn test_note_out_of_range() {
        assert_eq!(note_name_to_midi("C-2"), None); // would be negative
    }
}
