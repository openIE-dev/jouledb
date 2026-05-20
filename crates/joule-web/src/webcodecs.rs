//! WebCodecs — video frames, audio data, encoded chunks, encoder/decoder state machines.

use std::collections::VecDeque;

// ── Pixel Format ────────────────────────────────────────────────

/// Pixel format for video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    I420,
    NV12,
    RGBA,
    BGRA,
}

impl PixelFormat {
    /// Bytes needed for a frame of this format at the given dimensions.
    pub fn frame_size(&self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;
        match self {
            Self::I420 => w * h * 3 / 2,
            Self::NV12 => w * h * 3 / 2,
            Self::RGBA | Self::BGRA => w * h * 4,
        }
    }
}

// ── Audio Format ────────────────────────────────────────────────

/// Sample format for audio data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSampleFormat {
    F32,
    S16,
    U8,
}

impl AudioSampleFormat {
    /// Bytes per sample for this format.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::S16 => 2,
            Self::U8 => 1,
        }
    }
}

// ── Video Frame ─────────────────────────────────────────────────

/// A raw video frame.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub timestamp_us: i64,
    pub duration_us: Option<i64>,
    pub format: PixelFormat,
    pub data: Vec<u8>,
}

impl VideoFrame {
    /// Create a new video frame. Returns `None` if data length doesn't match format.
    pub fn new(
        width: u32,
        height: u32,
        timestamp_us: i64,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> Option<Self> {
        let expected = format.frame_size(width, height);
        if data.len() != expected {
            return None;
        }
        Some(Self {
            width,
            height,
            timestamp_us,
            duration_us: None,
            format,
            data,
        })
    }

    /// Total number of pixels.
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

// ── Audio Data ──────────────────────────────────────────────────

/// A chunk of raw audio data.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioData {
    pub sample_rate: u32,
    pub channels: u32,
    pub format: AudioSampleFormat,
    pub timestamp_us: i64,
    pub data: Vec<u8>,
}

impl AudioData {
    /// Number of audio frames (samples per channel).
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        let bytes_per_frame = self.format.bytes_per_sample() * self.channels as usize;
        if bytes_per_frame == 0 {
            return 0;
        }
        self.data.len() / bytes_per_frame
    }

    /// Duration in microseconds.
    pub fn duration_us(&self) -> i64 {
        if self.sample_rate == 0 {
            return 0;
        }
        (self.frame_count() as i64 * 1_000_000) / self.sample_rate as i64
    }
}

// ── Encoded Chunk ───────────────────────────────────────────────

/// An encoded video or audio chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedChunk {
    pub data: Vec<u8>,
    pub timestamp_us: i64,
    pub duration_us: Option<i64>,
    pub is_keyframe: bool,
}

// ── Codec Config ────────────────────────────────────────────────

/// Video encoder/decoder configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoCodecConfig {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub bitrate: Option<u64>,
    pub framerate: Option<f64>,
    pub hardware_acceleration: HardwareAcceleration,
}

/// Audio encoder/decoder configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioCodecConfig {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub bitrate: Option<u64>,
}

/// Hardware acceleration preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareAcceleration {
    NoPreference,
    PreferHardware,
    PreferSoftware,
}

// ── Codec State ─────────────────────────────────────────────────

/// State of an encoder or decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecState {
    Unconfigured,
    Configured,
    Closed,
}

// ── Video Encoder ───────────────────────────────────────────────

/// Simulated video encoder.
#[derive(Debug, Clone)]
pub struct VideoEncoder {
    pub state: CodecState,
    pub config: Option<VideoCodecConfig>,
    pub encode_count: u64,
    output: VecDeque<EncodedChunk>,
    keyframe_interval: u64,
}

impl VideoEncoder {
    pub fn new() -> Self {
        Self {
            state: CodecState::Unconfigured,
            config: None,
            encode_count: 0,
            output: VecDeque::new(),
            keyframe_interval: 30,
        }
    }

    /// Configure the encoder.
    pub fn configure(&mut self, config: VideoCodecConfig) {
        self.config = Some(config);
        self.state = CodecState::Configured;
        self.encode_count = 0;
    }

    /// Set keyframe interval (every N frames).
    pub fn set_keyframe_interval(&mut self, interval: u64) {
        self.keyframe_interval = interval;
    }

    /// Encode a video frame. Returns the number of output chunks produced.
    pub fn encode(&mut self, frame: &VideoFrame, force_keyframe: bool) -> usize {
        if self.state != CodecState::Configured {
            return 0;
        }
        let is_keyframe = force_keyframe
            || self.encode_count == 0
            || (self.keyframe_interval > 0 && self.encode_count % self.keyframe_interval == 0);

        // Simulate encoding: produce a smaller chunk
        let encoded_size = frame.data.len() / 4 + 1;
        let chunk = EncodedChunk {
            data: vec![0u8; encoded_size],
            timestamp_us: frame.timestamp_us,
            duration_us: frame.duration_us,
            is_keyframe,
        };
        self.output.push_back(chunk);
        self.encode_count += 1;
        1
    }

    /// Drain output chunks.
    pub fn drain_output(&mut self) -> Vec<EncodedChunk> {
        self.output.drain(..).collect()
    }

    /// Close the encoder.
    pub fn close(&mut self) {
        self.state = CodecState::Closed;
    }

    /// Flush any pending frames.
    pub fn flush(&mut self) -> Vec<EncodedChunk> {
        self.drain_output()
    }
}

impl Default for VideoEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Video Decoder ───────────────────────────────────────────────

/// Simulated video decoder.
#[derive(Debug, Clone)]
pub struct VideoDecoder {
    pub state: CodecState,
    pub config: Option<VideoCodecConfig>,
    pub decode_count: u64,
    output: VecDeque<VideoFrame>,
}

impl VideoDecoder {
    pub fn new() -> Self {
        Self {
            state: CodecState::Unconfigured,
            config: None,
            decode_count: 0,
            output: VecDeque::new(),
        }
    }

    /// Configure the decoder.
    pub fn configure(&mut self, config: VideoCodecConfig) {
        self.config = Some(config);
        self.state = CodecState::Configured;
        self.decode_count = 0;
    }

    /// Decode an encoded chunk. Returns number of frames produced.
    pub fn decode(&mut self, chunk: &EncodedChunk) -> usize {
        if self.state != CodecState::Configured {
            return 0;
        }
        let cfg = self.config.as_ref().unwrap();
        let format = PixelFormat::I420;
        let frame_size = format.frame_size(cfg.width, cfg.height);
        let frame = VideoFrame {
            width: cfg.width,
            height: cfg.height,
            timestamp_us: chunk.timestamp_us,
            duration_us: chunk.duration_us,
            format,
            data: vec![0u8; frame_size],
        };
        self.output.push_back(frame);
        self.decode_count += 1;
        1
    }

    /// Drain output frames.
    pub fn drain_output(&mut self) -> Vec<VideoFrame> {
        self.output.drain(..).collect()
    }

    /// Close the decoder.
    pub fn close(&mut self) {
        self.state = CodecState::Closed;
    }
}

impl Default for VideoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Audio Encoder ───────────────────────────────────────────────

/// Simulated audio encoder.
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    pub state: CodecState,
    pub config: Option<AudioCodecConfig>,
    pub encode_count: u64,
    output: VecDeque<EncodedChunk>,
}

impl AudioEncoder {
    pub fn new() -> Self {
        Self {
            state: CodecState::Unconfigured,
            config: None,
            encode_count: 0,
            output: VecDeque::new(),
        }
    }

    /// Configure the encoder.
    pub fn configure(&mut self, config: AudioCodecConfig) {
        self.config = Some(config);
        self.state = CodecState::Configured;
        self.encode_count = 0;
    }

    /// Encode audio data.
    pub fn encode(&mut self, audio: &AudioData) -> usize {
        if self.state != CodecState::Configured {
            return 0;
        }
        let encoded_size = audio.data.len() / 2 + 1;
        let chunk = EncodedChunk {
            data: vec![0u8; encoded_size],
            timestamp_us: audio.timestamp_us,
            duration_us: Some(audio.duration_us()),
            is_keyframe: true, // audio chunks are always keyframes
        };
        self.output.push_back(chunk);
        self.encode_count += 1;
        1
    }

    /// Drain output.
    pub fn drain_output(&mut self) -> Vec<EncodedChunk> {
        self.output.drain(..).collect()
    }

    /// Close the encoder.
    pub fn close(&mut self) {
        self.state = CodecState::Closed;
    }
}

impl Default for AudioEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame(ts: i64) -> VideoFrame {
        let format = PixelFormat::RGBA;
        let w = 4;
        let h = 4;
        let data = vec![0u8; format.frame_size(w, h)];
        VideoFrame::new(w, h, ts, format, data).unwrap()
    }

    #[test]
    fn pixel_format_sizes() {
        assert_eq!(PixelFormat::I420.frame_size(1920, 1080), 1920 * 1080 * 3 / 2);
        assert_eq!(PixelFormat::RGBA.frame_size(100, 100), 40_000);
        assert_eq!(PixelFormat::NV12.frame_size(640, 480), 640 * 480 * 3 / 2);
    }

    #[test]
    fn video_frame_validation() {
        let good = VideoFrame::new(2, 2, 0, PixelFormat::RGBA, vec![0u8; 16]);
        assert!(good.is_some());
        let bad = VideoFrame::new(2, 2, 0, PixelFormat::RGBA, vec![0u8; 10]);
        assert!(bad.is_none());
    }

    #[test]
    fn video_frame_pixel_count() {
        let f = test_frame(0);
        assert_eq!(f.pixel_count(), 16);
    }

    #[test]
    fn audio_data_frame_count() {
        let audio = AudioData {
            sample_rate: 48000,
            channels: 2,
            format: AudioSampleFormat::F32,
            timestamp_us: 0,
            data: vec![0u8; 48000 * 2 * 4], // 1 second of stereo f32
        };
        assert_eq!(audio.frame_count(), 48000);
    }

    #[test]
    fn audio_duration() {
        let audio = AudioData {
            sample_rate: 48000,
            channels: 1,
            format: AudioSampleFormat::S16,
            timestamp_us: 0,
            data: vec![0u8; 48000 * 2], // 1 second mono s16
        };
        assert_eq!(audio.duration_us(), 1_000_000);
    }

    #[test]
    fn encoder_lifecycle() {
        let mut enc = VideoEncoder::new();
        assert_eq!(enc.state, CodecState::Unconfigured);

        enc.configure(VideoCodecConfig {
            codec: "avc1.42001E".into(),
            width: 4,
            height: 4,
            bitrate: Some(1_000_000),
            framerate: Some(30.0),
            hardware_acceleration: HardwareAcceleration::NoPreference,
        });
        assert_eq!(enc.state, CodecState::Configured);

        let frame = test_frame(0);
        let n = enc.encode(&frame, false);
        assert_eq!(n, 1);
        assert_eq!(enc.encode_count, 1);

        let output = enc.drain_output();
        assert_eq!(output.len(), 1);
        assert!(output[0].is_keyframe); // first frame is always keyframe

        enc.close();
        assert_eq!(enc.state, CodecState::Closed);
    }

    #[test]
    fn encoder_keyframe_interval() {
        let mut enc = VideoEncoder::new();
        enc.configure(VideoCodecConfig {
            codec: "avc1".into(),
            width: 4, height: 4,
            bitrate: None, framerate: None,
            hardware_acceleration: HardwareAcceleration::NoPreference,
        });
        enc.set_keyframe_interval(3);

        for i in 0..6 {
            enc.encode(&test_frame(i * 33333), false);
        }
        let chunks = enc.drain_output();
        assert!(chunks[0].is_keyframe);  // frame 0
        assert!(!chunks[1].is_keyframe); // frame 1
        assert!(!chunks[2].is_keyframe); // frame 2
        assert!(chunks[3].is_keyframe);  // frame 3
        assert!(!chunks[4].is_keyframe); // frame 4
        assert!(!chunks[5].is_keyframe); // frame 5
    }

    #[test]
    fn force_keyframe() {
        let mut enc = VideoEncoder::new();
        enc.configure(VideoCodecConfig {
            codec: "avc1".into(),
            width: 4, height: 4,
            bitrate: None, framerate: None,
            hardware_acceleration: HardwareAcceleration::NoPreference,
        });
        enc.encode(&test_frame(0), false); // keyframe (first)
        enc.encode(&test_frame(1), true);  // forced keyframe
        let chunks = enc.drain_output();
        assert!(chunks[1].is_keyframe);
    }

    #[test]
    fn decoder_produces_frames() {
        let mut dec = VideoDecoder::new();
        dec.configure(VideoCodecConfig {
            codec: "avc1".into(),
            width: 4, height: 4,
            bitrate: None, framerate: None,
            hardware_acceleration: HardwareAcceleration::NoPreference,
        });

        let chunk = EncodedChunk {
            data: vec![0u8; 10],
            timestamp_us: 1000,
            duration_us: Some(33333),
            is_keyframe: true,
        };
        let n = dec.decode(&chunk);
        assert_eq!(n, 1);

        let frames = dec.drain_output();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].width, 4);
        assert_eq!(frames[0].timestamp_us, 1000);
    }

    #[test]
    fn unconfigured_encoder_produces_nothing() {
        let mut enc = VideoEncoder::new();
        let n = enc.encode(&test_frame(0), false);
        assert_eq!(n, 0);
    }

    #[test]
    fn audio_encoder_lifecycle() {
        let mut enc = AudioEncoder::new();
        enc.configure(AudioCodecConfig {
            codec: "opus".into(),
            sample_rate: 48000,
            channels: 2,
            bitrate: Some(128_000),
        });

        let audio = AudioData {
            sample_rate: 48000,
            channels: 2,
            format: AudioSampleFormat::F32,
            timestamp_us: 0,
            data: vec![0u8; 4800 * 2 * 4], // 100ms
        };
        let n = enc.encode(&audio);
        assert_eq!(n, 1);
        let output = enc.drain_output();
        assert!(output[0].is_keyframe);
        assert!(output[0].data.len() < audio.data.len());
    }

    #[test]
    fn audio_sample_format_sizes() {
        assert_eq!(AudioSampleFormat::F32.bytes_per_sample(), 4);
        assert_eq!(AudioSampleFormat::S16.bytes_per_sample(), 2);
        assert_eq!(AudioSampleFormat::U8.bytes_per_sample(), 1);
    }
}
