//! Signal Types - First-class signal representations
//!
//! Signals carry their sample rate, dimensionality, and provenance.

use smol_str::SmolStr;

use super::units::SampleRate;

/// A time-domain signal with known sample rate and value type.
///
/// The type system enforces that operations between signals
/// with different sample rates require explicit resampling.
#[derive(Debug, Clone)]
pub struct Signal<T, const FS: u32> {
    /// Signal samples
    pub samples: Vec<T>,
    /// Channel identifier (e.g., "left_hand.accel.x")
    pub channel: SmolStr,
    /// Start timestamp (nanoseconds since epoch)
    pub start_ns: i64,
    /// Metadata about signal provenance
    pub metadata: SignalMetadata,
}

impl<T, const FS: u32> Signal<T, FS> {
    /// Create a new signal with given samples
    pub fn new(channel: impl Into<SmolStr>, samples: Vec<T>, start_ns: i64) -> Self {
        Self {
            samples,
            channel: channel.into(),
            start_ns,
            metadata: SignalMetadata::default(),
        }
    }

    /// Get the sample rate as a value
    #[inline]
    pub const fn sample_rate(&self) -> SampleRate {
        SampleRate::new(FS)
    }

    /// Get duration in nanoseconds
    #[inline]
    pub fn duration_ns(&self) -> i64 {
        let samples = self.samples.len() as i64;
        (samples * 1_000_000_000) / (FS as i64)
    }

    /// Get the number of samples
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if signal is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Metadata about signal provenance and quality
#[derive(Debug, Clone, Default)]
pub struct SignalMetadata {
    /// Source device/sensor identifier
    pub source: Option<SmolStr>,
    /// Calibration status
    pub calibrated: bool,
    /// Known noise floor (in signal units)
    pub noise_floor: Option<f64>,
    /// Artifact flags for each sample (bit flags)
    pub artifact_mask: Option<Vec<u8>>,
    /// Units (e.g., "m/s²", "µV")
    pub units: Option<SmolStr>,
}

/// A signal bundle - multiple synchronized signals
#[derive(Debug, Clone)]
pub struct SignalBundle {
    /// Named signals in the bundle
    pub signals: Vec<(SmolStr, DynamicSignal)>,
    /// Common start timestamp
    pub start_ns: i64,
    /// Bundle identifier
    pub bundle_id: SmolStr,
}

/// Type-erased signal for runtime flexibility
#[derive(Debug, Clone)]
pub enum DynamicSignal {
    F32_50(Signal<f32, 50>),
    F32_100(Signal<f32, 100>),
    F32_200(Signal<f32, 200>),
    F32_500(Signal<f32, 500>),
    F32_1000(Signal<f32, 1000>),
    F64_50(Signal<f64, 50>),
    F64_100(Signal<f64, 100>),
    F64_200(Signal<f64, 200>),
    F64_500(Signal<f64, 500>),
    F64_1000(Signal<f64, 1000>),
    /// For arbitrary sample rates
    F32Dyn(DynSignal<f32>),
    F64Dyn(DynSignal<f64>),
}

/// Dynamically-typed signal (runtime sample rate)
#[derive(Debug, Clone)]
pub struct DynSignal<T> {
    pub samples: Vec<T>,
    pub sample_rate: u32,
    pub channel: SmolStr,
    pub start_ns: i64,
    pub metadata: SignalMetadata,
}

impl<T> DynSignal<T> {
    pub fn new(
        channel: impl Into<SmolStr>,
        samples: Vec<T>,
        sample_rate: u32,
        start_ns: i64,
    ) -> Self {
        Self {
            samples,
            sample_rate,
            channel: channel.into(),
            start_ns,
            metadata: SignalMetadata::default(),
        }
    }
}

/// Discrete event with timestamp
#[derive(Debug, Clone)]
pub struct Event<T> {
    /// Event payload
    pub value: T,
    /// Timestamp in nanoseconds
    pub timestamp_ns: i64,
    /// Event label/type
    pub label: SmolStr,
}

/// A stream of events
#[derive(Debug, Clone)]
pub struct EventStream<T> {
    pub events: Vec<Event<T>>,
    pub channel: SmolStr,
}

/// Time interval with start and end
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Start timestamp (nanoseconds)
    pub start_ns: i64,
    /// End timestamp (nanoseconds)  
    pub end_ns: i64,
}

impl Interval {
    pub fn new(start_ns: i64, end_ns: i64) -> Self {
        Self { start_ns, end_ns }
    }

    pub fn duration_ns(&self) -> i64 {
        self.end_ns - self.start_ns
    }

    pub fn contains(&self, timestamp_ns: i64) -> bool {
        timestamp_ns >= self.start_ns && timestamp_ns <= self.end_ns
    }

    pub fn overlaps(&self, other: &Interval) -> bool {
        self.start_ns <= other.end_ns && other.start_ns <= self.end_ns
    }
}

/// Channel specification for query parsing
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelSpec {
    /// Device/source path (e.g., "controller.right_hand")
    pub path: SmolStr,
    /// Specific channel (e.g., "accel.x")
    pub channel: SmolStr,
    /// Expected sample rate (optional, for validation)
    pub expected_rate: Option<u32>,
    /// Expected units (optional, for validation)
    pub expected_units: Option<SmolStr>,
}

impl ChannelSpec {
    pub fn new(path: impl Into<SmolStr>, channel: impl Into<SmolStr>) -> Self {
        Self {
            path: path.into(),
            channel: channel.into(),
            expected_rate: None,
            expected_units: None,
        }
    }

    /// Full channel identifier
    pub fn full_path(&self) -> SmolStr {
        SmolStr::new(format!("{}.{}", self.path, self.channel))
    }
}

// ====== MediaQL: Multi-Dimensional Signal Types ======

/// A 2D signal (image or single video frame).
/// Stored as a flat Vec in row-major order with `width * height * channels` elements.
#[derive(Debug, Clone)]
pub struct Signal2D<T> {
    pub data: Vec<T>,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub color_space: MediaColorSpace,
    pub metadata: SignalMetadata,
}

/// A video signal: a temporal sequence of 2D frames.
#[derive(Debug, Clone)]
pub struct VideoSignal<T> {
    pub frames: Vec<Signal2D<T>>,
    pub fps: f32,
    pub start_ns: i64,
    pub metadata: SignalMetadata,
}

/// Color space for media signals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaColorSpace {
    Rgb,
    Srgb,
    YCbCr,
    Hsv,
    Lab,
    Grayscale,
}

impl Default for MediaColorSpace {
    fn default() -> Self {
        MediaColorSpace::Srgb
    }
}

/// Frequency-domain representation of media (the "middle" of middle-out).
/// Stores coefficients that can reconstruct the original AND be queried directly.
#[derive(Debug, Clone)]
pub struct FrequencyCoefficients {
    /// Sparse DCT/FFT coefficient entries
    pub entries: Vec<CoefficientEntry>,
    /// Original dimensions (width, height) or (time_bins, freq_bins)
    pub shape: (u32, u32),
    /// Transform type used
    pub transform: FreqTransform,
    /// Quality factor (for lossy quantization)
    pub quality: u8,
}

/// A single frequency coefficient
#[derive(Debug, Clone, Copy)]
pub struct CoefficientEntry {
    /// Position in frequency space (x, y) or (time, freq)
    pub position: (u32, u32),
    /// Coefficient magnitude
    pub magnitude: f32,
    /// Coefficient phase (radians)
    pub phase: f32,
}

/// Transform type used for frequency encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreqTransform {
    /// 2D Discrete Cosine Transform (image)
    Dct2d,
    /// Short-Time Fourier Transform (audio)
    Stft,
    /// 2D FFT (image)
    Fft2d,
    /// Discrete Wavelet Transform (multi-scale)
    Dwt2d,
}

/// Ingested media: frequency coefficients + HDC hologram + perceptual hash.
/// This is the "middle-out" triple — one storage artifact, three capabilities.
#[derive(Debug, Clone)]
pub struct IngestedMedia {
    /// Frequency-domain coefficients (for lossless/lossy reconstruction)
    pub coefficients: FrequencyCoefficients,
    /// Perceptual hash (for fast dedup)
    pub phash: u64,
    /// Original media dimensions
    pub width: u32,
    pub height: u32,
    /// Media type tag
    pub media_type: MediaTypeTag,
}

/// Media type discriminator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaTypeTag {
    Image,
    Audio,
    Video,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_duration() {
        let sig: Signal<f32, 100> = Signal::new("test", vec![0.0; 100], 0);
        assert_eq!(sig.duration_ns(), 1_000_000_000); // 1 second
    }

    #[test]
    fn test_interval_overlap() {
        let a = Interval::new(0, 100);
        let b = Interval::new(50, 150);
        let c = Interval::new(200, 300);

        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }
}
