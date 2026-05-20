//! MIDI message parsing and file format concepts.
//!
//! Parses MIDI channel messages (NoteOn/Off, ControlChange, ProgramChange,
//! PitchBend, ChannelPressure, PolyPressure), system messages (SysEx, clock,
//! start/stop), and meta events (tempo, time signature, key signature).
//! Provides MIDI file header/track parsing, tempo mapping, note name conversion,
//! and note-to-frequency conversion.

use std::collections::BTreeMap;

// ── MIDI Messages ───────────────────────────────────────────────

/// MIDI message types — channel, system, and meta messages.
#[derive(Debug, Clone, PartialEq)]
pub enum MidiMessage {
    NoteOn {
        channel: u8,
        note: u8,
        velocity: u8,
    },
    NoteOff {
        channel: u8,
        note: u8,
        velocity: u8,
    },
    ControlChange {
        channel: u8,
        controller: u8,
        value: u8,
    },
    ProgramChange {
        channel: u8,
        program: u8,
    },
    PitchBend {
        channel: u8,
        /// Signed value: -8192..8191.
        value: i16,
    },
    ChannelPressure {
        channel: u8,
        pressure: u8,
    },
    PolyPressure {
        channel: u8,
        note: u8,
        pressure: u8,
    },
    SysEx {
        data: Vec<u8>,
    },
    /// System real-time: timing clock (0xF8).
    TimingClock,
    /// System real-time: start (0xFA).
    Start,
    /// System real-time: continue (0xFB).
    Continue,
    /// System real-time: stop (0xFC).
    Stop,
    /// System real-time: active sensing (0xFE).
    ActiveSensing,
    /// System real-time: system reset (0xFF in real-time context).
    SystemReset,
    MetaTempo {
        microseconds_per_beat: u32,
    },
    MetaTimeSignature {
        numerator: u8,
        denominator_power: u8,
    },
    MetaKeySignature {
        /// Sharps (positive) or flats (negative).
        key: i8,
        /// 0 = major, 1 = minor.
        scale: u8,
    },
    MetaTrackName {
        name: String,
    },
    MetaEndOfTrack,
    Unknown {
        status: u8,
        data: Vec<u8>,
    },
}

impl MidiMessage {
    /// Returns true if this is a channel voice message.
    pub fn is_channel_voice(&self) -> bool {
        matches!(
            self,
            Self::NoteOn { .. }
                | Self::NoteOff { .. }
                | Self::ControlChange { .. }
                | Self::ProgramChange { .. }
                | Self::PitchBend { .. }
                | Self::ChannelPressure { .. }
                | Self::PolyPressure { .. }
        )
    }

    /// Returns the channel number (0..15) if this is a channel message.
    pub fn channel(&self) -> Option<u8> {
        match self {
            Self::NoteOn { channel, .. }
            | Self::NoteOff { channel, .. }
            | Self::ControlChange { channel, .. }
            | Self::ProgramChange { channel, .. }
            | Self::PitchBend { channel, .. }
            | Self::ChannelPressure { channel, .. }
            | Self::PolyPressure { channel, .. } => Some(*channel),
            _ => None,
        }
    }

    /// Encode the message back to raw MIDI bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::NoteOn {
                channel,
                note,
                velocity,
            } => vec![0x90 | (channel & 0x0F), *note, *velocity],
            Self::NoteOff {
                channel,
                note,
                velocity,
            } => vec![0x80 | (channel & 0x0F), *note, *velocity],
            Self::ControlChange {
                channel,
                controller,
                value,
            } => vec![0xB0 | (channel & 0x0F), *controller, *value],
            Self::ProgramChange { channel, program } => {
                vec![0xC0 | (channel & 0x0F), *program]
            }
            Self::PitchBend { channel, value } => {
                let unsigned = (*value + 8192) as u16;
                let lsb = (unsigned & 0x7F) as u8;
                let msb = ((unsigned >> 7) & 0x7F) as u8;
                vec![0xE0 | (channel & 0x0F), lsb, msb]
            }
            Self::ChannelPressure { channel, pressure } => {
                vec![0xD0 | (channel & 0x0F), *pressure]
            }
            Self::PolyPressure {
                channel,
                note,
                pressure,
            } => vec![0xA0 | (channel & 0x0F), *note, *pressure],
            _ => Vec::new(),
        }
    }
}

/// Parse a MIDI message from raw bytes. Returns the message and bytes consumed.
pub fn parse_midi_message(data: &[u8], running_status: u8) -> Option<(MidiMessage, usize)> {
    if data.is_empty() {
        return None;
    }

    let (status, offset) = if data[0] & 0x80 != 0 {
        (data[0], 1)
    } else {
        (running_status, 0)
    };

    // System real-time (single byte, can interleave)
    match status {
        0xF8 => return Some((MidiMessage::TimingClock, offset)),
        0xFA => return Some((MidiMessage::Start, offset)),
        0xFB => return Some((MidiMessage::Continue, offset)),
        0xFC => return Some((MidiMessage::Stop, offset)),
        0xFE => return Some((MidiMessage::ActiveSensing, offset)),
        _ => {}
    }

    let remaining = &data[offset..];
    let msg_type = status & 0xF0;
    let channel = status & 0x0F;

    match msg_type {
        0x90 => {
            if remaining.len() < 2 {
                return None;
            }
            let note = remaining[0];
            let velocity = remaining[1];
            let msg = if velocity == 0 {
                MidiMessage::NoteOff {
                    channel,
                    note,
                    velocity: 0,
                }
            } else {
                MidiMessage::NoteOn {
                    channel,
                    note,
                    velocity,
                }
            };
            Some((msg, offset + 2))
        }
        0x80 => {
            if remaining.len() < 2 {
                return None;
            }
            Some((
                MidiMessage::NoteOff {
                    channel,
                    note: remaining[0],
                    velocity: remaining[1],
                },
                offset + 2,
            ))
        }
        0xA0 => {
            if remaining.len() < 2 {
                return None;
            }
            Some((
                MidiMessage::PolyPressure {
                    channel,
                    note: remaining[0],
                    pressure: remaining[1],
                },
                offset + 2,
            ))
        }
        0xB0 => {
            if remaining.len() < 2 {
                return None;
            }
            Some((
                MidiMessage::ControlChange {
                    channel,
                    controller: remaining[0],
                    value: remaining[1],
                },
                offset + 2,
            ))
        }
        0xC0 => {
            if remaining.is_empty() {
                return None;
            }
            Some((
                MidiMessage::ProgramChange {
                    channel,
                    program: remaining[0],
                },
                offset + 1,
            ))
        }
        0xD0 => {
            if remaining.is_empty() {
                return None;
            }
            Some((
                MidiMessage::ChannelPressure {
                    channel,
                    pressure: remaining[0],
                },
                offset + 1,
            ))
        }
        0xE0 => {
            if remaining.len() < 2 {
                return None;
            }
            let lsb = remaining[0] as i16;
            let msb = remaining[1] as i16;
            let value = ((msb << 7) | lsb) - 8192;
            Some((MidiMessage::PitchBend { channel, value }, offset + 2))
        }
        _ if status == 0xF0 => {
            // SysEx: read until 0xF7
            let end = remaining.iter().position(|b| *b == 0xF7);
            let end_pos = end.map_or(remaining.len(), |p| p + 1);
            Some((
                MidiMessage::SysEx {
                    data: remaining[..end_pos].to_vec(),
                },
                offset + end_pos,
            ))
        }
        _ if status == 0xFF => {
            // Meta event
            if remaining.len() < 2 {
                return None;
            }
            let meta_type = remaining[0];
            let (length, vlq_len) = read_variable_length(&remaining[1..]);
            let data_start = 1 + vlq_len;
            let data_end = data_start + length as usize;

            match meta_type {
                0x03 => {
                    // Track Name
                    if remaining.len() >= data_end {
                        let name_bytes = &remaining[data_start..data_end];
                        let name = String::from_utf8_lossy(name_bytes).to_string();
                        Some((MidiMessage::MetaTrackName { name }, offset + data_end))
                    } else {
                        None
                    }
                }
                0x51 => {
                    // Tempo
                    if remaining.len() >= data_end && length >= 3 {
                        let d = &remaining[data_start..data_end];
                        let usec =
                            ((d[0] as u32) << 16) | ((d[1] as u32) << 8) | (d[2] as u32);
                        Some((
                            MidiMessage::MetaTempo {
                                microseconds_per_beat: usec,
                            },
                            offset + data_end,
                        ))
                    } else {
                        None
                    }
                }
                0x58 => {
                    // Time Signature
                    if remaining.len() >= data_end && length >= 2 {
                        let d = &remaining[data_start..data_end];
                        Some((
                            MidiMessage::MetaTimeSignature {
                                numerator: d[0],
                                denominator_power: d[1],
                            },
                            offset + data_end,
                        ))
                    } else {
                        None
                    }
                }
                0x59 => {
                    // Key Signature
                    if remaining.len() >= data_end && length >= 2 {
                        let d = &remaining[data_start..data_end];
                        Some((
                            MidiMessage::MetaKeySignature {
                                key: d[0] as i8,
                                scale: d[1],
                            },
                            offset + data_end,
                        ))
                    } else {
                        None
                    }
                }
                0x2F => Some((MidiMessage::MetaEndOfTrack, offset + data_end)),
                _ => Some((
                    MidiMessage::Unknown {
                        status: 0xFF,
                        data: if remaining.len() >= data_end {
                            remaining[data_start..data_end].to_vec()
                        } else {
                            Vec::new()
                        },
                    },
                    offset + data_end.min(remaining.len()),
                )),
            }
        }
        _ => Some((
            MidiMessage::Unknown {
                status,
                data: Vec::new(),
            },
            offset,
        )),
    }
}

// ── Variable-Length Quantity ─────────────────────────────────────

/// Read a MIDI variable-length quantity. Returns (value, bytes_consumed).
pub fn read_variable_length(data: &[u8]) -> (u32, usize) {
    let mut value: u32 = 0;
    let mut bytes = 0;

    for byte in data {
        value = (value << 7) | (byte & 0x7F) as u32;
        bytes += 1;
        if byte & 0x80 == 0 {
            break;
        }
        if bytes >= 4 {
            break;
        }
    }

    (value, bytes)
}

/// Encode a value as a MIDI variable-length quantity.
pub fn write_variable_length(value: u32) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }

    let mut bytes = Vec::new();
    let mut v = value;

    while v > 0 {
        bytes.push((v & 0x7F) as u8);
        v >>= 7;
    }

    bytes.reverse();
    let len = bytes.len();
    for b in &mut bytes[..len - 1] {
        *b |= 0x80;
    }

    bytes
}

// ── MIDI File Reader ────────────────────────────────────────────

/// MIDI file format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidiFormat {
    SingleTrack,
    MultiTrack,
    MultiSong,
}

/// MIDI file header.
#[derive(Debug, Clone)]
pub struct MidiHeader {
    pub format: MidiFormat,
    pub num_tracks: u16,
    pub ticks_per_beat: u16,
}

/// A timed MIDI event in a track.
#[derive(Debug, Clone)]
pub struct MidiEvent {
    pub delta_ticks: u32,
    pub absolute_ticks: u32,
    pub message: MidiMessage,
}

/// A MIDI track.
#[derive(Debug, Clone)]
pub struct MidiTrack {
    pub events: Vec<MidiEvent>,
    pub name: Option<String>,
}

/// Parsed MIDI file.
#[derive(Debug, Clone)]
pub struct MidiFile {
    pub header: MidiHeader,
    pub tracks: Vec<MidiTrack>,
}

impl MidiFile {
    /// Get the total duration in ticks.
    pub fn total_ticks(&self) -> u32 {
        self.tracks
            .iter()
            .flat_map(|t| t.events.last())
            .map(|e| e.absolute_ticks)
            .max()
            .unwrap_or(0)
    }

    /// Get all note events across all tracks.
    pub fn note_events(&self) -> Vec<&MidiEvent> {
        self.tracks
            .iter()
            .flat_map(|t| t.events.iter())
            .filter(|e| {
                matches!(
                    e.message,
                    MidiMessage::NoteOn { .. } | MidiMessage::NoteOff { .. }
                )
            })
            .collect()
    }
}

/// Parse a MIDI file from raw bytes.
pub fn parse_midi_file(data: &[u8]) -> Option<MidiFile> {
    if data.len() < 14 {
        return None;
    }

    if &data[0..4] != b"MThd" {
        return None;
    }

    let header_len = read_u32_be(&data[4..8]) as usize;
    if data.len() < 8 + header_len {
        return None;
    }

    let format = match read_u16_be(&data[8..10]) {
        0 => MidiFormat::SingleTrack,
        1 => MidiFormat::MultiTrack,
        2 => MidiFormat::MultiSong,
        _ => return None,
    };

    let num_tracks = read_u16_be(&data[10..12]);
    let ticks_per_beat = read_u16_be(&data[12..14]);

    let header = MidiHeader {
        format,
        num_tracks,
        ticks_per_beat,
    };

    let mut tracks = Vec::new();
    let mut pos = 8 + header_len;

    for _ in 0..num_tracks {
        if pos + 8 > data.len() {
            break;
        }
        if &data[pos..pos + 4] != b"MTrk" {
            break;
        }
        let track_len = read_u32_be(&data[pos + 4..pos + 8]) as usize;
        pos += 8;

        let track_end = (pos + track_len).min(data.len());
        let track_data = &data[pos..track_end];

        let track = parse_track(track_data);
        tracks.push(track);

        pos = track_end;
    }

    Some(MidiFile { header, tracks })
}

fn parse_track(data: &[u8]) -> MidiTrack {
    let mut events = Vec::new();
    let mut pos = 0;
    let mut absolute_ticks: u32 = 0;
    let mut running_status: u8 = 0;
    let mut track_name: Option<String> = None;

    while pos < data.len() {
        let (delta, vlq_len) = read_variable_length(&data[pos..]);
        pos += vlq_len;
        absolute_ticks = absolute_ticks.wrapping_add(delta);

        if pos >= data.len() {
            break;
        }

        if let Some((msg, consumed)) = parse_midi_message(&data[pos..], running_status) {
            if data[pos] & 0x80 != 0 && data[pos] < 0xF0 {
                running_status = data[pos];
            }

            if let MidiMessage::MetaTrackName { name: n } = &msg {
                track_name = Some(n.clone());
            }

            events.push(MidiEvent {
                delta_ticks: delta,
                absolute_ticks,
                message: msg,
            });
            pos += consumed;
        } else {
            break;
        }
    }

    MidiTrack {
        events,
        name: track_name,
    }
}

fn read_u16_be(data: &[u8]) -> u16 {
    ((data[0] as u16) << 8) | (data[1] as u16)
}

fn read_u32_be(data: &[u8]) -> u32 {
    ((data[0] as u32) << 24)
        | ((data[1] as u32) << 16)
        | ((data[2] as u32) << 8)
        | (data[3] as u32)
}

// ── Tempo Map ───────────────────────────────────────────────────

/// Maps MIDI ticks to real time using tempo changes.
#[derive(Debug, Clone)]
pub struct TempoMap {
    ticks_per_beat: u16,
    tempo_changes: BTreeMap<u32, u32>,
}

impl TempoMap {
    pub fn new(ticks_per_beat: u16) -> Self {
        let mut tempo_changes = BTreeMap::new();
        tempo_changes.insert(0, 500_000); // Default 120 BPM
        Self {
            ticks_per_beat,
            tempo_changes,
        }
    }

    /// Build a tempo map from a MIDI file.
    pub fn from_midi_file(file: &MidiFile) -> Self {
        let mut map = Self::new(file.header.ticks_per_beat);
        for track in &file.tracks {
            for event in &track.events {
                if let MidiMessage::MetaTempo {
                    microseconds_per_beat,
                } = &event.message
                {
                    map.add_tempo_change(event.absolute_ticks, *microseconds_per_beat);
                }
            }
        }
        map
    }

    pub fn add_tempo_change(&mut self, tick: u32, microseconds_per_beat: u32) {
        self.tempo_changes.insert(tick, microseconds_per_beat);
    }

    /// Convert a tick position to time in seconds.
    pub fn tick_to_seconds(&self, tick: u32) -> f64 {
        let mut seconds = 0.0;
        let mut prev_tick: u32 = 0;
        let mut current_tempo: u32 = 500_000;

        for (&change_tick, &tempo) in &self.tempo_changes {
            if change_tick >= tick {
                break;
            }
            let delta_ticks = change_tick - prev_tick;
            seconds += self.ticks_to_seconds_at_tempo(delta_ticks, current_tempo);
            prev_tick = change_tick;
            current_tempo = tempo;
        }

        let remaining = tick - prev_tick;
        seconds += self.ticks_to_seconds_at_tempo(remaining, current_tempo);
        seconds
    }

    /// Convert seconds to the nearest tick position.
    pub fn seconds_to_tick(&self, target_seconds: f64) -> u32 {
        let mut seconds = 0.0;
        let mut prev_tick: u32 = 0;
        let mut current_tempo: u32 = 500_000;

        for (&change_tick, &tempo) in &self.tempo_changes {
            let delta = change_tick - prev_tick;
            let seg_dur = self.ticks_to_seconds_at_tempo(delta, current_tempo);
            if seconds + seg_dur >= target_seconds {
                let remaining_time = target_seconds - seconds;
                let remaining_ticks = remaining_time * 1_000_000.0 * self.ticks_per_beat as f64
                    / current_tempo as f64;
                return prev_tick + remaining_ticks as u32;
            }
            seconds += seg_dur;
            prev_tick = change_tick;
            current_tempo = tempo;
        }

        let remaining_time = target_seconds - seconds;
        let remaining_ticks =
            remaining_time * 1_000_000.0 * self.ticks_per_beat as f64 / current_tempo as f64;
        prev_tick + remaining_ticks.max(0.0) as u32
    }

    fn ticks_to_seconds_at_tempo(&self, ticks: u32, usec_per_beat: u32) -> f64 {
        let beats = ticks as f64 / self.ticks_per_beat as f64;
        beats * usec_per_beat as f64 / 1_000_000.0
    }

    /// Get BPM at a given tick.
    pub fn bpm_at_tick(&self, tick: u32) -> f64 {
        let mut tempo = 500_000u32;
        for (&t, &usec) in &self.tempo_changes {
            if t > tick {
                break;
            }
            tempo = usec;
        }
        60_000_000.0 / tempo as f64
    }

    /// Get total duration in seconds for a given total tick count.
    pub fn duration_seconds(&self, total_ticks: u32) -> f64 {
        self.tick_to_seconds(total_ticks)
    }
}

// ── Time Signature ──────────────────────────────────────────────

/// Represents a time signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSignature {
    pub numerator: u8,
    pub denominator: u8,
}

impl TimeSignature {
    pub fn new(numerator: u8, denominator: u8) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Create from MIDI meta event values.
    pub fn from_midi(numerator: u8, denominator_power: u8) -> Self {
        Self {
            numerator,
            denominator: 1u8.wrapping_shl(denominator_power as u32),
        }
    }

    /// Beats per measure.
    pub fn beats_per_measure(&self) -> u8 {
        self.numerator
    }
}

// ── Note Names ──────────────────────────────────────────────────

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Convert MIDI note number to note name (e.g., 60 -> "C4").
pub fn note_name(note: u8) -> String {
    let octave = (note as i16 / 12) - 1;
    let name = NOTE_NAMES[(note % 12) as usize];
    format!("{name}{octave}")
}

/// Convert note name to MIDI note number (e.g., "C4" -> 60).
pub fn note_number(name: &str) -> Option<u8> {
    if name.len() < 2 {
        return None;
    }

    let (note_part, octave_str) = if name.len() >= 3 && &name[1..2] == "#" {
        (&name[..2], &name[2..])
    } else if name.len() >= 3 && &name[1..2] == "b" {
        // Support flat notation
        let sharp_equivalent = match &name[..2] {
            "Db" => "C#",
            "Eb" => "D#",
            "Gb" => "F#",
            "Ab" => "G#",
            "Bb" => "A#",
            _ => return None,
        };
        let idx = NOTE_NAMES.iter().position(|n| *n == sharp_equivalent)?;
        let octave: i16 = name[2..].parse().ok()?;
        let midi = (octave + 1) * 12 + idx as i16;
        if midi >= 0 && midi <= 127 {
            return Some(midi as u8);
        } else {
            return None;
        }
    } else {
        (&name[..1], &name[1..])
    };

    let note_idx = NOTE_NAMES.iter().position(|n| *n == note_part)?;
    let octave: i16 = octave_str.parse().ok()?;
    let midi = (octave + 1) * 12 + note_idx as i16;

    if midi >= 0 && midi <= 127 {
        Some(midi as u8)
    } else {
        None
    }
}

/// Convert MIDI note to frequency in Hz (A4 = 440 Hz).
pub fn note_to_frequency(note: u8) -> f64 {
    440.0 * 2.0f64.powf((note as f64 - 69.0) / 12.0)
}

/// Convert frequency to nearest MIDI note.
pub fn frequency_to_note(freq: f64) -> u8 {
    let note = 69.0 + 12.0 * (freq / 440.0).log2();
    note.round().clamp(0.0, 127.0) as u8
}

/// Convert MIDI note to frequency with cent offset.
pub fn note_to_frequency_with_cents(note: u8, cents: f64) -> f64 {
    440.0 * 2.0f64.powf((note as f64 - 69.0 + cents / 100.0) / 12.0)
}

/// Calculate the interval in semitones between two frequencies.
pub fn frequency_interval(freq1: f64, freq2: f64) -> f64 {
    12.0 * (freq2 / freq1).log2()
}

// ── Sequence Playback ───────────────────────────────────────────

/// A scheduled event for playback.
#[derive(Debug, Clone)]
pub struct ScheduledEvent {
    pub time_seconds: f64,
    pub message: MidiMessage,
}

/// Build a playback timeline from a MIDI file.
pub fn build_timeline(file: &MidiFile) -> Vec<ScheduledEvent> {
    let tempo_map = TempoMap::from_midi_file(file);
    let mut timeline = Vec::new();

    for track in &file.tracks {
        for event in &track.events {
            let time = tempo_map.tick_to_seconds(event.absolute_ticks);
            timeline.push(ScheduledEvent {
                time_seconds: time,
                message: event.message.clone(),
            });
        }
    }

    timeline.sort_by(|a, b| {
        a.time_seconds
            .partial_cmp(&b.time_seconds)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    timeline
}

// ── Well-Known Controller Numbers ───────────────────────────────

/// Standard MIDI controller numbers.
pub mod controllers {
    pub const BANK_SELECT_MSB: u8 = 0;
    pub const MOD_WHEEL: u8 = 1;
    pub const BREATH: u8 = 2;
    pub const FOOT: u8 = 4;
    pub const PORTAMENTO_TIME: u8 = 5;
    pub const DATA_ENTRY_MSB: u8 = 6;
    pub const VOLUME: u8 = 7;
    pub const BALANCE: u8 = 8;
    pub const PAN: u8 = 10;
    pub const EXPRESSION: u8 = 11;
    pub const SUSTAIN_PEDAL: u8 = 64;
    pub const PORTAMENTO: u8 = 65;
    pub const SOSTENUTO: u8 = 66;
    pub const SOFT_PEDAL: u8 = 67;
    pub const ALL_SOUND_OFF: u8 = 120;
    pub const RESET_ALL: u8 = 121;
    pub const ALL_NOTES_OFF: u8 = 123;
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_name_c4() {
        assert_eq!(note_name(60), "C4");
    }

    #[test]
    fn note_name_a4() {
        assert_eq!(note_name(69), "A4");
    }

    #[test]
    fn note_number_roundtrip() {
        for n in 0..128u8 {
            let name = note_name(n);
            let back = note_number(&name).unwrap();
            assert_eq!(back, n, "Roundtrip failed for note {n} -> {name}");
        }
    }

    #[test]
    fn note_number_flat_notation() {
        assert_eq!(note_number("Bb4"), Some(70));
        assert_eq!(note_number("Eb3"), Some(51));
    }

    #[test]
    fn note_to_freq_a4() {
        assert!((note_to_frequency(69) - 440.0).abs() < 0.01);
    }

    #[test]
    fn note_to_freq_c4() {
        assert!((note_to_frequency(60) - 261.63).abs() < 0.1);
    }

    #[test]
    fn freq_to_note_440() {
        assert_eq!(frequency_to_note(440.0), 69);
    }

    #[test]
    fn note_to_freq_with_cents() {
        let freq = note_to_frequency_with_cents(69, 0.0);
        assert!((freq - 440.0).abs() < 0.01);
        let freq_up = note_to_frequency_with_cents(69, 100.0);
        // +100 cents = +1 semitone = A#4
        assert!((freq_up - note_to_frequency(70)).abs() < 0.01);
    }

    #[test]
    fn frequency_interval_octave() {
        let interval = frequency_interval(220.0, 440.0);
        assert!((interval - 12.0).abs() < 0.01);
    }

    #[test]
    fn variable_length_read() {
        assert_eq!(read_variable_length(&[0x00]), (0, 1));
        assert_eq!(read_variable_length(&[0x7F]), (127, 1));
        assert_eq!(read_variable_length(&[0x81, 0x00]), (128, 2));
        assert_eq!(read_variable_length(&[0xC0, 0x00]), (0x2000, 2));
        assert_eq!(read_variable_length(&[0xFF, 0x7F]), (0x3FFF, 2));
    }

    #[test]
    fn variable_length_write() {
        assert_eq!(write_variable_length(0), vec![0x00]);
        assert_eq!(write_variable_length(127), vec![0x7F]);
        assert_eq!(write_variable_length(128), vec![0x81, 0x00]);
    }

    #[test]
    fn variable_length_roundtrip() {
        for val in [0, 1, 127, 128, 255, 256, 16383, 16384, 0x1FFFFF] {
            let encoded = write_variable_length(val);
            let (decoded, _) = read_variable_length(&encoded);
            assert_eq!(decoded, val, "VLQ roundtrip failed for {val}");
        }
    }

    #[test]
    fn parse_note_on() {
        let data = [0x90, 60, 100];
        let (msg, consumed) = parse_midi_message(&data, 0).unwrap();
        assert_eq!(consumed, 3);
        assert_eq!(
            msg,
            MidiMessage::NoteOn {
                channel: 0,
                note: 60,
                velocity: 100
            }
        );
    }

    #[test]
    fn parse_note_on_velocity_zero_is_note_off() {
        let data = [0x90, 60, 0];
        let (msg, _) = parse_midi_message(&data, 0).unwrap();
        assert_eq!(
            msg,
            MidiMessage::NoteOff {
                channel: 0,
                note: 60,
                velocity: 0
            }
        );
    }

    #[test]
    fn parse_control_change() {
        let data = [0xB0, 7, 100];
        let (msg, _) = parse_midi_message(&data, 0).unwrap();
        assert_eq!(
            msg,
            MidiMessage::ControlChange {
                channel: 0,
                controller: 7,
                value: 100
            }
        );
    }

    #[test]
    fn parse_channel_pressure() {
        let data = [0xD0, 80];
        let (msg, consumed) = parse_midi_message(&data, 0).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(
            msg,
            MidiMessage::ChannelPressure {
                channel: 0,
                pressure: 80
            }
        );
    }

    #[test]
    fn parse_poly_pressure() {
        let data = [0xA0, 60, 90];
        let (msg, consumed) = parse_midi_message(&data, 0).unwrap();
        assert_eq!(consumed, 3);
        assert_eq!(
            msg,
            MidiMessage::PolyPressure {
                channel: 0,
                note: 60,
                pressure: 90
            }
        );
    }

    #[test]
    fn parse_pitch_bend_center() {
        let data = [0xE0, 0x00, 0x40];
        let (msg, _) = parse_midi_message(&data, 0).unwrap();
        assert_eq!(msg, MidiMessage::PitchBend { channel: 0, value: 0 });
    }

    #[test]
    fn parse_system_realtime() {
        let (msg, consumed) = parse_midi_message(&[0xF8], 0).unwrap();
        assert_eq!(msg, MidiMessage::TimingClock);
        assert_eq!(consumed, 1);

        let (msg, _) = parse_midi_message(&[0xFA], 0).unwrap();
        assert_eq!(msg, MidiMessage::Start);

        let (msg, _) = parse_midi_message(&[0xFC], 0).unwrap();
        assert_eq!(msg, MidiMessage::Stop);
    }

    #[test]
    fn message_to_bytes_roundtrip() {
        let msg = MidiMessage::NoteOn {
            channel: 3,
            note: 60,
            velocity: 100,
        };
        let bytes = msg.to_bytes();
        assert_eq!(bytes, vec![0x93, 60, 100]);
        let (parsed, _) = parse_midi_message(&bytes, 0).unwrap();
        assert_eq!(parsed, msg);
    }

    #[test]
    fn pitch_bend_to_bytes_roundtrip() {
        for value in [-8192i16, -1000, 0, 1000, 8191] {
            let msg = MidiMessage::PitchBend { channel: 0, value };
            let bytes = msg.to_bytes();
            let (parsed, _) = parse_midi_message(&bytes, 0).unwrap();
            assert_eq!(parsed, msg, "PitchBend roundtrip failed for value={value}");
        }
    }

    #[test]
    fn message_is_channel_voice() {
        let note = MidiMessage::NoteOn {
            channel: 0,
            note: 60,
            velocity: 100,
        };
        assert!(note.is_channel_voice());
        assert_eq!(note.channel(), Some(0));

        let clock = MidiMessage::TimingClock;
        assert!(!clock.is_channel_voice());
        assert_eq!(clock.channel(), None);
    }

    #[test]
    fn parse_midi_file_basic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"MThd");
        data.extend_from_slice(&6u32.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());
        data.extend_from_slice(&1u16.to_be_bytes());
        data.extend_from_slice(&480u16.to_be_bytes());

        let mut track_data = Vec::new();
        track_data.push(0x00);
        track_data.extend_from_slice(&[0x90, 60, 100]);
        track_data.push(0x83);
        track_data.push(0x60);
        track_data.extend_from_slice(&[0x80, 60, 0]);
        track_data.push(0x00);
        track_data.extend_from_slice(&[0xFF, 0x2F, 0x00]);

        data.extend_from_slice(b"MTrk");
        data.extend_from_slice(&(track_data.len() as u32).to_be_bytes());
        data.extend_from_slice(&track_data);

        let file = parse_midi_file(&data).unwrap();
        assert_eq!(file.header.format, MidiFormat::SingleTrack);
        assert_eq!(file.header.ticks_per_beat, 480);
        assert_eq!(file.tracks.len(), 1);
        assert!(file.tracks[0].events.len() >= 2);
        assert!(file.total_ticks() > 0);
        assert!(!file.note_events().is_empty());
    }

    #[test]
    fn tempo_map_default() {
        let map = TempoMap::new(480);
        assert!((map.bpm_at_tick(0) - 120.0).abs() < 0.01);
    }

    #[test]
    fn tempo_map_tick_to_seconds() {
        let map = TempoMap::new(480);
        let secs = map.tick_to_seconds(480);
        assert!((secs - 0.5).abs() < 0.001);
    }

    #[test]
    fn tempo_map_seconds_to_tick() {
        let map = TempoMap::new(480);
        let tick = map.seconds_to_tick(0.5);
        assert!((tick as f64 - 480.0).abs() < 2.0);
    }

    #[test]
    fn time_signature_from_midi() {
        let ts = TimeSignature::from_midi(6, 3); // 6/8
        assert_eq!(ts.numerator, 6);
        assert_eq!(ts.denominator, 8);
        assert_eq!(ts.beats_per_measure(), 6);
    }

    #[test]
    fn controllers_values() {
        assert_eq!(controllers::VOLUME, 7);
        assert_eq!(controllers::SUSTAIN_PEDAL, 64);
        assert_eq!(controllers::ALL_NOTES_OFF, 123);
    }
}
