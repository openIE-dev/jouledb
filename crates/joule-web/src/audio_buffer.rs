//! Audio buffer management — interleaved/planar formats, sample conversion, mixing.
//!
//! Provides `AudioBuf` for multi-channel audio with support for interleaved and
//! planar storage layouts. Handles i16/f32 conversion, gain application,
//! fade in/out, silence detection, and buffer splitting/joining.

use std::fmt;

// ── Sample Format ───────────────────────────────────────────────

/// Supported sample formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 16-bit signed integer.
    I16,
    /// 32-bit float (normalized to [-1.0, 1.0]).
    F32,
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I16 => write!(f, "i16"),
            Self::F32 => write!(f, "f32"),
        }
    }
}

// ── Buffer Layout ───────────────────────────────────────────────

/// Memory layout of audio samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLayout {
    /// Samples interleaved: L0 R0 L1 R1 L2 R2 ...
    Interleaved,
    /// Samples planar: L0 L1 L2 ... R0 R1 R2 ...
    Planar,
}

// ── AudioBuf ────────────────────────────────────────────────────

/// Multi-channel audio buffer with configurable layout.
#[derive(Debug, Clone)]
pub struct AudioBuf {
    /// Stored as f32 internally regardless of original format.
    data: Vec<f32>,
    channels: usize,
    sample_rate: u32,
    frames: usize,
    layout: BufferLayout,
}

impl AudioBuf {
    /// Create a silent audio buffer.
    pub fn new(channels: usize, frames: usize, sample_rate: u32, layout: BufferLayout) -> Self {
        let ch = channels.max(1);
        Self {
            data: vec![0.0; ch * frames],
            channels: ch,
            sample_rate,
            frames,
            layout,
        }
    }

    /// Create a buffer from interleaved f32 data.
    pub fn from_interleaved_f32(data: &[f32], channels: usize, sample_rate: u32) -> Self {
        let ch = channels.max(1);
        let frames = if ch > 0 { data.len() / ch } else { 0 };
        Self {
            data: data[..frames * ch].to_vec(),
            channels: ch,
            sample_rate,
            frames,
            layout: BufferLayout::Interleaved,
        }
    }

    /// Create a buffer from interleaved i16 data, converting to f32.
    pub fn from_interleaved_i16(data: &[i16], channels: usize, sample_rate: u32) -> Self {
        let ch = channels.max(1);
        let frames = if ch > 0 { data.len() / ch } else { 0 };
        let f32_data: Vec<f32> = data[..frames * ch]
            .iter()
            .map(|s| i16_to_f32(*s))
            .collect();
        Self {
            data: f32_data,
            channels: ch,
            sample_rate,
            frames,
            layout: BufferLayout::Interleaved,
        }
    }

    /// Create a buffer from planar channel data (one Vec per channel).
    pub fn from_planar(channel_data: &[Vec<f32>], sample_rate: u32) -> Self {
        if channel_data.is_empty() {
            return Self::new(1, 0, sample_rate, BufferLayout::Planar);
        }
        let channels = channel_data.len();
        let frames = channel_data.iter().map(|c| c.len()).min().unwrap_or(0);
        let mut data = Vec::with_capacity(channels * frames);
        for ch_buf in channel_data {
            data.extend_from_slice(&ch_buf[..frames]);
        }
        Self {
            data,
            channels,
            sample_rate,
            frames,
            layout: BufferLayout::Planar,
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn layout(&self) -> BufferLayout {
        self.layout
    }

    pub fn total_samples(&self) -> usize {
        self.channels * self.frames
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frames as f64 / self.sample_rate as f64
    }

    /// Get a sample by channel and frame index.
    pub fn get_sample(&self, channel: usize, frame: usize) -> Option<f32> {
        if channel >= self.channels || frame >= self.frames {
            return None;
        }
        let idx = match self.layout {
            BufferLayout::Interleaved => frame * self.channels + channel,
            BufferLayout::Planar => channel * self.frames + frame,
        };
        self.data.get(idx).copied()
    }

    /// Set a sample by channel and frame index.
    pub fn set_sample(&mut self, channel: usize, frame: usize, value: f32) {
        if channel >= self.channels || frame >= self.frames {
            return;
        }
        let idx = match self.layout {
            BufferLayout::Interleaved => frame * self.channels + channel,
            BufferLayout::Planar => channel * self.frames + frame,
        };
        if let Some(s) = self.data.get_mut(idx) {
            *s = value;
        }
    }

    /// Get immutable access to channel data (works efficiently for planar layout).
    /// Returns a Vec for interleaved since data must be gathered.
    pub fn channel_data(&self, channel: usize) -> Vec<f32> {
        if channel >= self.channels {
            return Vec::new();
        }
        match self.layout {
            BufferLayout::Planar => {
                let start = channel * self.frames;
                let end = start + self.frames;
                self.data[start..end].to_vec()
            }
            BufferLayout::Interleaved => {
                (0..self.frames)
                    .map(|f| self.data[f * self.channels + channel])
                    .collect()
            }
        }
    }

    /// Get raw data slice.
    pub fn raw_data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable raw data slice.
    pub fn raw_data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Convert to interleaved layout (in place).
    pub fn to_interleaved(&mut self) {
        if self.layout == BufferLayout::Interleaved {
            return;
        }
        let mut interleaved = vec![0.0f32; self.data.len()];
        for ch in 0..self.channels {
            for f in 0..self.frames {
                let src = ch * self.frames + f;
                let dst = f * self.channels + ch;
                interleaved[dst] = self.data[src];
            }
        }
        self.data = interleaved;
        self.layout = BufferLayout::Interleaved;
    }

    /// Convert to planar layout (in place).
    pub fn to_planar(&mut self) {
        if self.layout == BufferLayout::Planar {
            return;
        }
        let mut planar = vec![0.0f32; self.data.len()];
        for ch in 0..self.channels {
            for f in 0..self.frames {
                let src = f * self.channels + ch;
                let dst = ch * self.frames + f;
                planar[dst] = self.data[src];
            }
        }
        self.data = planar;
        self.layout = BufferLayout::Planar;
    }

    /// Export as interleaved i16 samples.
    pub fn to_interleaved_i16(&self) -> Vec<i16> {
        match self.layout {
            BufferLayout::Interleaved => self.data.iter().map(|s| f32_to_i16(*s)).collect(),
            BufferLayout::Planar => {
                let mut out = Vec::with_capacity(self.total_samples());
                for f in 0..self.frames {
                    for ch in 0..self.channels {
                        let idx = ch * self.frames + f;
                        out.push(f32_to_i16(self.data[idx]));
                    }
                }
                out
            }
        }
    }

    /// Apply linear gain to all samples.
    pub fn apply_gain(&mut self, gain: f32) {
        for s in &mut self.data {
            *s *= gain;
        }
    }

    /// Apply gain to a specific channel.
    pub fn apply_channel_gain(&mut self, channel: usize, gain: f32) {
        if channel >= self.channels {
            return;
        }
        for f in 0..self.frames {
            let idx = match self.layout {
                BufferLayout::Interleaved => f * self.channels + channel,
                BufferLayout::Planar => channel * self.frames + f,
            };
            self.data[idx] *= gain;
        }
    }

    /// Apply linear fade in over `fade_frames` at the start.
    pub fn fade_in(&mut self, fade_frames: usize) {
        let fade_len = fade_frames.min(self.frames);
        if fade_len == 0 {
            return;
        }
        for f in 0..fade_len {
            let gain = f as f32 / fade_len as f32;
            for ch in 0..self.channels {
                let idx = match self.layout {
                    BufferLayout::Interleaved => f * self.channels + ch,
                    BufferLayout::Planar => ch * self.frames + f,
                };
                self.data[idx] *= gain;
            }
        }
    }

    /// Apply linear fade out over `fade_frames` at the end.
    pub fn fade_out(&mut self, fade_frames: usize) {
        let fade_len = fade_frames.min(self.frames);
        if fade_len == 0 {
            return;
        }
        let start = self.frames - fade_len;
        for f in start..self.frames {
            let progress = (f - start) as f32 / fade_len as f32;
            let gain = 1.0 - progress;
            for ch in 0..self.channels {
                let idx = match self.layout {
                    BufferLayout::Interleaved => f * self.channels + ch,
                    BufferLayout::Planar => ch * self.frames + f,
                };
                self.data[idx] *= gain;
            }
        }
    }

    /// Detect if the buffer is silent (all samples below threshold).
    pub fn is_silent(&self, threshold: f32) -> bool {
        self.data.iter().all(|s| s.abs() < threshold)
    }

    /// Calculate RMS level across all channels.
    pub fn rms(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.data.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        (sum / self.data.len() as f64).sqrt() as f32
    }

    /// Calculate peak level across all channels.
    pub fn peak(&self) -> f32 {
        self.data
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max)
    }

    /// Mix another buffer into this one (additive).
    pub fn mix(&mut self, other: &AudioBuf) {
        let frames = self.frames.min(other.frames);
        let channels = self.channels.min(other.channels);
        for ch in 0..channels {
            for f in 0..frames {
                let self_idx = match self.layout {
                    BufferLayout::Interleaved => f * self.channels + ch,
                    BufferLayout::Planar => ch * self.frames + f,
                };
                let other_val = other.get_sample(ch, f).unwrap_or(0.0);
                self.data[self_idx] += other_val;
            }
        }
    }

    /// Mix with a gain factor applied to the other buffer.
    pub fn mix_with_gain(&mut self, other: &AudioBuf, gain: f32) {
        let frames = self.frames.min(other.frames);
        let channels = self.channels.min(other.channels);
        for ch in 0..channels {
            for f in 0..frames {
                let self_idx = match self.layout {
                    BufferLayout::Interleaved => f * self.channels + ch,
                    BufferLayout::Planar => ch * self.frames + f,
                };
                let other_val = other.get_sample(ch, f).unwrap_or(0.0);
                self.data[self_idx] += other_val * gain;
            }
        }
    }

    /// Split a buffer at a frame boundary, returning (left, right).
    pub fn split_at_frame(&self, frame: usize) -> (AudioBuf, AudioBuf) {
        let split = frame.min(self.frames);
        let left_frames = split;
        let right_frames = self.frames - split;

        let mut left = AudioBuf::new(self.channels, left_frames, self.sample_rate, self.layout);
        let mut right = AudioBuf::new(self.channels, right_frames, self.sample_rate, self.layout);

        for ch in 0..self.channels {
            for f in 0..left_frames {
                let val = self.get_sample(ch, f).unwrap_or(0.0);
                left.set_sample(ch, f, val);
            }
            for f in 0..right_frames {
                let val = self.get_sample(ch, split + f).unwrap_or(0.0);
                right.set_sample(ch, f, val);
            }
        }

        (left, right)
    }

    /// Join two buffers end-to-end. Both must have same channel count and sample rate.
    pub fn join(a: &AudioBuf, b: &AudioBuf) -> Option<AudioBuf> {
        if a.channels != b.channels || a.sample_rate != b.sample_rate {
            return None;
        }
        let layout = a.layout;
        let total_frames = a.frames + b.frames;
        let mut result = AudioBuf::new(a.channels, total_frames, a.sample_rate, layout);

        for ch in 0..a.channels {
            for f in 0..a.frames {
                let val = a.get_sample(ch, f).unwrap_or(0.0);
                result.set_sample(ch, f, val);
            }
            for f in 0..b.frames {
                let val = b.get_sample(ch, f).unwrap_or(0.0);
                result.set_sample(ch, a.frames + f, val);
            }
        }

        Some(result)
    }

    /// Clamp all samples to [-1.0, 1.0].
    pub fn clamp(&mut self) {
        for s in &mut self.data {
            *s = s.clamp(-1.0, 1.0);
        }
    }

    /// Normalize the buffer so the peak is at `target_peak`.
    pub fn normalize(&mut self, target_peak: f32) {
        let current_peak = self.peak();
        if current_peak < 1e-10 {
            return;
        }
        let gain = target_peak / current_peak;
        self.apply_gain(gain);
    }

    /// Reverse the buffer in place.
    pub fn reverse(&mut self) {
        for ch in 0..self.channels {
            let half = self.frames / 2;
            for f in 0..half {
                let rev = self.frames - 1 - f;
                let a = self.get_sample(ch, f).unwrap_or(0.0);
                let b = self.get_sample(ch, rev).unwrap_or(0.0);
                self.set_sample(ch, f, b);
                self.set_sample(ch, rev, a);
            }
        }
    }

    /// Downmix to mono by averaging all channels.
    pub fn to_mono(&self) -> AudioBuf {
        let mut mono = AudioBuf::new(1, self.frames, self.sample_rate, BufferLayout::Planar);
        let ch_f = self.channels as f32;
        for f in 0..self.frames {
            let mut sum = 0.0f32;
            for ch in 0..self.channels {
                sum += self.get_sample(ch, f).unwrap_or(0.0);
            }
            mono.set_sample(0, f, sum / ch_f);
        }
        mono
    }
}

// ── Conversion Helpers ──────────────────────────────────────────

/// Convert i16 sample to f32 [-1.0, 1.0].
pub fn i16_to_f32(sample: i16) -> f32 {
    if sample >= 0 {
        sample as f32 / 32767.0
    } else {
        sample as f32 / 32768.0
    }
}

/// Convert f32 sample [-1.0, 1.0] to i16.
pub fn f32_to_i16(sample: f32) -> i16 {
    let clamped = sample.clamp(-1.0, 1.0);
    if clamped >= 0.0 {
        (clamped * 32767.0) as i16
    } else {
        (clamped * 32768.0) as i16
    }
}

/// Convert dB to linear gain.
pub fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Convert linear gain to dB.
pub fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        return f32::NEG_INFINITY;
    }
    20.0 * linear.log10()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_silent_buffer() {
        let buf = AudioBuf::new(2, 1024, 44100, BufferLayout::Interleaved);
        assert_eq!(buf.channels(), 2);
        assert_eq!(buf.frames(), 1024);
        assert_eq!(buf.sample_rate(), 44100);
        assert!(buf.is_silent(0.001));
    }

    #[test]
    fn interleaved_get_set() {
        let mut buf = AudioBuf::new(2, 4, 44100, BufferLayout::Interleaved);
        buf.set_sample(0, 0, 0.5);
        buf.set_sample(1, 0, -0.3);
        buf.set_sample(0, 3, 0.8);
        assert!((buf.get_sample(0, 0).unwrap() - 0.5).abs() < 1e-6);
        assert!((buf.get_sample(1, 0).unwrap() - (-0.3)).abs() < 1e-6);
        assert!((buf.get_sample(0, 3).unwrap() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn planar_get_set() {
        let mut buf = AudioBuf::new(2, 4, 44100, BufferLayout::Planar);
        buf.set_sample(0, 2, 0.7);
        buf.set_sample(1, 1, -0.4);
        assert!((buf.get_sample(0, 2).unwrap() - 0.7).abs() < 1e-6);
        assert!((buf.get_sample(1, 1).unwrap() - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn from_interleaved_f32() {
        let data = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3];
        let buf = AudioBuf::from_interleaved_f32(&data, 2, 48000);
        assert_eq!(buf.frames(), 3);
        assert!((buf.get_sample(0, 0).unwrap() - 0.1).abs() < 1e-6);
        assert!((buf.get_sample(1, 0).unwrap() - (-0.1)).abs() < 1e-6);
        assert!((buf.get_sample(0, 2).unwrap() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn from_interleaved_i16() {
        let data = [16383i16, -16384, 0, 0];
        let buf = AudioBuf::from_interleaved_i16(&data, 2, 44100);
        assert_eq!(buf.frames(), 2);
        let s0 = buf.get_sample(0, 0).unwrap();
        assert!(s0 > 0.49 && s0 < 0.51);
        let s1 = buf.get_sample(1, 0).unwrap();
        assert!(s1 > -0.51 && s1 < -0.49);
    }

    #[test]
    fn from_planar() {
        let ch0 = vec![0.1, 0.2, 0.3];
        let ch1 = vec![-0.1, -0.2, -0.3];
        let buf = AudioBuf::from_planar(&[ch0, ch1], 44100);
        assert_eq!(buf.channels(), 2);
        assert_eq!(buf.frames(), 3);
        assert!((buf.get_sample(0, 1).unwrap() - 0.2).abs() < 1e-6);
        assert!((buf.get_sample(1, 2).unwrap() - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn interleaved_to_planar_roundtrip() {
        let data = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3];
        let mut buf = AudioBuf::from_interleaved_f32(&data, 2, 44100);
        buf.to_planar();
        assert_eq!(buf.layout(), BufferLayout::Planar);
        assert!((buf.get_sample(0, 0).unwrap() - 0.1).abs() < 1e-6);
        assert!((buf.get_sample(1, 2).unwrap() - (-0.3)).abs() < 1e-6);
        buf.to_interleaved();
        assert_eq!(buf.layout(), BufferLayout::Interleaved);
        assert!((buf.get_sample(0, 0).unwrap() - 0.1).abs() < 1e-6);
        assert!((buf.get_sample(1, 2).unwrap() - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn i16_f32_roundtrip() {
        for val in [-32768i16, -16384, -1, 0, 1, 16383, 32767] {
            let f = i16_to_f32(val);
            assert!(f >= -1.0 && f <= 1.0);
            let back = f32_to_i16(f);
            assert!((val - back).unsigned_abs() <= 1, "val={val} f={f} back={back}");
        }
    }

    #[test]
    fn to_interleaved_i16() {
        let ch0 = vec![0.5, -0.5];
        let ch1 = vec![-0.25, 0.25];
        let buf = AudioBuf::from_planar(&[ch0, ch1], 44100);
        let i16_data = buf.to_interleaved_i16();
        assert_eq!(i16_data.len(), 4);
        assert!((i16_data[0] - 16383).abs() <= 1);
        assert!((i16_data[1] - (-8192)).abs() <= 1);
    }

    #[test]
    fn apply_gain() {
        let mut buf = AudioBuf::from_interleaved_f32(&[0.5, -0.5, 0.25, -0.25], 2, 44100);
        buf.apply_gain(0.5);
        assert!((buf.get_sample(0, 0).unwrap() - 0.25).abs() < 1e-6);
        assert!((buf.get_sample(1, 0).unwrap() - (-0.25)).abs() < 1e-6);
    }

    #[test]
    fn apply_channel_gain() {
        let mut buf = AudioBuf::from_interleaved_f32(&[1.0, 1.0, 1.0, 1.0], 2, 44100);
        buf.apply_channel_gain(0, 0.5);
        assert!((buf.get_sample(0, 0).unwrap() - 0.5).abs() < 1e-6);
        assert!((buf.get_sample(1, 0).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fade_in() {
        let data = vec![1.0; 8];
        let mut buf = AudioBuf::from_interleaved_f32(&data, 1, 44100);
        buf.fade_in(4);
        assert!(buf.get_sample(0, 0).unwrap().abs() < 1e-6);
        assert!((buf.get_sample(0, 2).unwrap() - 0.5).abs() < 1e-6);
        assert!((buf.get_sample(0, 4).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fade_out() {
        let data = vec![1.0; 8];
        let mut buf = AudioBuf::from_interleaved_f32(&data, 1, 44100);
        buf.fade_out(4);
        assert!((buf.get_sample(0, 3).unwrap() - 1.0).abs() < 1e-6);
        assert!((buf.get_sample(0, 6).unwrap() - 0.5).abs() < 0.26);
        assert!(buf.get_sample(0, 7).unwrap().abs() < 0.26);
    }

    #[test]
    fn silence_detection() {
        let mut buf = AudioBuf::new(2, 100, 44100, BufferLayout::Interleaved);
        assert!(buf.is_silent(0.001));
        buf.set_sample(0, 50, 0.5);
        assert!(!buf.is_silent(0.001));
    }

    #[test]
    fn rms_and_peak() {
        let mut buf = AudioBuf::new(1, 4, 44100, BufferLayout::Planar);
        buf.set_sample(0, 0, 0.5);
        buf.set_sample(0, 1, -0.5);
        buf.set_sample(0, 2, 0.5);
        buf.set_sample(0, 3, -0.5);
        assert!((buf.rms() - 0.5).abs() < 1e-6);
        assert!((buf.peak() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mix_buffers() {
        let mut a = AudioBuf::from_interleaved_f32(&[0.3, 0.3], 1, 44100);
        let b = AudioBuf::from_interleaved_f32(&[0.2, 0.2], 1, 44100);
        a.mix(&b);
        assert!((a.get_sample(0, 0).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mix_with_gain() {
        let mut a = AudioBuf::from_interleaved_f32(&[0.3], 1, 44100);
        let b = AudioBuf::from_interleaved_f32(&[1.0], 1, 44100);
        a.mix_with_gain(&b, 0.2);
        assert!((a.get_sample(0, 0).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn split_and_join() {
        let data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let buf = AudioBuf::from_interleaved_f32(&data, 1, 44100);
        let (left, right) = buf.split_at_frame(3);
        assert_eq!(left.frames(), 3);
        assert_eq!(right.frames(), 5);
        let joined = AudioBuf::join(&left, &right).unwrap();
        assert_eq!(joined.frames(), 8);
        for i in 0..8 {
            let expected = i as f32 * 0.1;
            assert!((joined.get_sample(0, i).unwrap() - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn normalize() {
        let mut buf = AudioBuf::from_interleaved_f32(&[0.25, -0.25, 0.5, -0.5], 1, 44100);
        buf.normalize(1.0);
        assert!((buf.peak() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reverse() {
        let mut buf = AudioBuf::from_interleaved_f32(&[0.1, 0.2, 0.3, 0.4], 1, 44100);
        buf.reverse();
        assert!((buf.get_sample(0, 0).unwrap() - 0.4).abs() < 1e-6);
        assert!((buf.get_sample(0, 3).unwrap() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn to_mono() {
        let ch0 = vec![0.6, 0.4];
        let ch1 = vec![0.4, 0.6];
        let buf = AudioBuf::from_planar(&[ch0, ch1], 44100);
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert!((mono.get_sample(0, 0).unwrap() - 0.5).abs() < 1e-6);
        assert!((mono.get_sample(0, 1).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn db_conversion() {
        let linear = db_to_linear(0.0);
        assert!((linear - 1.0).abs() < 1e-6);
        let linear_6 = db_to_linear(-6.0);
        assert!((linear_6 - 0.5012).abs() < 0.01);
        let db = linear_to_db(1.0);
        assert!(db.abs() < 1e-6);
    }

    #[test]
    fn duration_secs() {
        let buf = AudioBuf::new(2, 44100, 44100, BufferLayout::Interleaved);
        assert!((buf.duration_secs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn channel_data_extraction() {
        let data = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3];
        let buf = AudioBuf::from_interleaved_f32(&data, 2, 44100);
        let ch0 = buf.channel_data(0);
        assert_eq!(ch0.len(), 3);
        assert!((ch0[0] - 0.1).abs() < 1e-6);
        assert!((ch0[2] - 0.3).abs() < 1e-6);
        let ch1 = buf.channel_data(1);
        assert!((ch1[0] - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn out_of_bounds_returns_none() {
        let buf = AudioBuf::new(2, 4, 44100, BufferLayout::Interleaved);
        assert!(buf.get_sample(3, 0).is_none());
        assert!(buf.get_sample(0, 5).is_none());
    }

    #[test]
    fn clamp_samples() {
        let mut buf = AudioBuf::from_interleaved_f32(&[1.5, -1.5, 0.5, -0.5], 1, 44100);
        buf.clamp();
        assert!((buf.get_sample(0, 0).unwrap() - 1.0).abs() < 1e-6);
        assert!((buf.get_sample(0, 1).unwrap() - (-1.0)).abs() < 1e-6);
        assert!((buf.get_sample(0, 2).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn join_mismatched_channels_fails() {
        let a = AudioBuf::new(1, 4, 44100, BufferLayout::Planar);
        let b = AudioBuf::new(2, 4, 44100, BufferLayout::Planar);
        assert!(AudioBuf::join(&a, &b).is_none());
    }

    #[test]
    fn sample_format_display() {
        assert_eq!(format!("{}", SampleFormat::I16), "i16");
        assert_eq!(format!("{}", SampleFormat::F32), "f32");
    }
}
