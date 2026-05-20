//! AST Expression Types
//!
//! The recursive expression structure that represents signal operations.

use smol_str::SmolStr;

use crate::types::{FrequencyBand, Hertz, SampleRate, Seconds};

/// Root expression type for signal operations
#[derive(Debug, Clone, PartialEq)]
pub enum SignalExpr {
    /// Reference to a signal source
    Source(SourceRef),

    /// Signal literal (inline data)
    Literal(SignalLiteral),

    /// Transform operation on a signal
    Transform {
        input: Box<SignalExpr>,
        op: TransformOp,
    },

    /// Window operation
    Window {
        input: Box<SignalExpr>,
        spec: WindowSpec,
    },

    /// Aggregation operation
    Aggregate {
        input: Box<SignalExpr>,
        op: AggregateOp,
    },

    /// Cross-signal correlation/operation
    Correlate {
        inputs: Vec<SignalExpr>,
        op: CorrelateOp,
    },

    /// Epoch alignment (event-locked analysis)
    Align {
        signal: Box<SignalExpr>,
        events: Box<SignalExpr>,
        spec: AlignSpec,
    },

    /// Signal fusion (multi-signal combination)
    Fuse {
        inputs: Vec<SignalExpr>,
        method: FuseMethod,
    },

    /// Let binding (named intermediate result)
    Let {
        name: SmolStr,
        value: Box<SignalExpr>,
        body: Box<SignalExpr>,
    },

    /// Variable reference
    Var(SmolStr),

    /// Conditional expression
    If {
        condition: Box<ScalarExpr>,
        then_expr: Box<SignalExpr>,
        else_expr: Box<SignalExpr>,
    },

    /// Pipeline operator (syntactic sugar for chained transforms)
    Pipeline(Vec<SignalExpr>),
}

/// Reference to a signal source
#[derive(Debug, Clone, PartialEq)]
pub struct SourceRef {
    /// Path to signal (e.g., "controller.right_hand.accel")
    pub path: SmolStr,
    /// Optional alias
    pub alias: Option<SmolStr>,
    /// Optional type hint
    pub type_hint: Option<SignalTypeHint>,
}

impl SourceRef {
    pub fn new(path: impl Into<SmolStr>) -> Self {
        Self {
            path: path.into(),
            alias: None,
            type_hint: None,
        }
    }

    pub fn with_alias(mut self, alias: impl Into<SmolStr>) -> Self {
        self.alias = Some(alias.into());
        self
    }
}

/// Signal type hint for validation
#[derive(Debug, Clone, PartialEq)]
pub struct SignalTypeHint {
    pub sample_rate: Option<SampleRate>,
    pub value_type: Option<ValueType>,
    pub units: Option<SmolStr>,
}

/// Value type for signals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    F32,
    F64,
    I16,
    I32,
    Bool,
    Complex64,
    Complex128,
}

/// Inline signal literal
#[derive(Debug, Clone, PartialEq)]
pub struct SignalLiteral {
    pub values: Vec<f64>,
    pub sample_rate: u32,
    pub channel: SmolStr,
}

/// DSP Transform operations
#[derive(Debug, Clone, PartialEq)]
pub enum TransformOp {
    // Frequency transforms
    Fft(FftParams),
    Ifft,
    Stft(StftParams),
    Wavelet(WaveletParams),
    Hilbert,

    // Filtering
    Bandpass(FilterParams),
    Lowpass(FilterParams),
    Highpass(FilterParams),
    Notch(NotchParams),
    Median(MedianParams),

    // Resampling
    Resample(ResampleParams),
    Decimate(DecimateParams),
    Interpolate(InterpolateParams),

    // Normalization
    ZScore(ZScoreParams),
    Detrend(DetrendParams),
    BaselineCorrect(BaselineParams),

    // Envelope
    Envelope,
    InstantaneousPhase,
    InstantaneousFrequency,

    // Artifact handling
    Reject(RejectParams),
    InterpolateArtifacts(ArtifactInterpolateParams),

    // Math
    Abs,
    Square,
    Sqrt,
    Log,
    Log10,
    Exp,
    Diff,
    Cumsum,
    Scale(f64),
    Offset(f64),

    // ====== MediaQL: 2D Frequency Transforms ======

    /// 2D FFT for image data
    Fft2d(Fft2dParams),
    /// Inverse 2D FFT
    Ifft2d,
    /// 2D Discrete Cosine Transform (JPEG-style block encoding)
    Dct2d(Dct2dParams),
    /// Inverse 2D DCT (reconstruction from frequency coefficients)
    Idct2d,
    /// 2D Discrete Wavelet Transform (multi-scale analysis)
    Dwt2d(Dwt2dParams),

    // ====== MediaQL: Audio-Specific ======

    /// Mel-frequency cepstral coefficients
    Mfcc(MfccParams),
    /// Chroma features (pitch class profile)
    ChromaFeatures(ChromaParams),
    /// Audio fingerprinting (spectrogram peak constellation)
    AudioFingerprint(FingerprintParams),

    // ====== MediaQL: Image-Specific ======

    /// Perceptual hash (DCT-based content fingerprint)
    PerceptualHash(PHashParams),
    /// Edge detection (Sobel, Canny)
    EdgeDetect(EdgeParams),
    /// Histogram equalization
    HistogramEqualize,
    /// Color space conversion
    ColorConvert(ColorSpace),

    // ====== MediaQL: Video-Specific ======

    /// Optical flow between frames
    OpticalFlow(OpticalFlowParams),
    /// Shot/scene boundary detection
    ShotDetect(ShotDetectParams),
    /// Extract frames at specified rate
    FrameExtract(FrameExtractParams),

    // Custom (user-defined)
    Custom {
        name: SmolStr,
        params: Vec<(SmolStr, Literal)>,
    },
}

// ====== MediaQL Parameter Structs ======

/// 2D FFT parameters
#[derive(Debug, Clone, PartialEq)]
pub struct Fft2dParams {
    pub window: WindowFunction,
    pub zero_pad: bool,
}

impl Default for Fft2dParams {
    fn default() -> Self {
        Self {
            window: WindowFunction::Hann,
            zero_pad: true,
        }
    }
}

/// 2D DCT parameters (block-based, JPEG-style)
#[derive(Debug, Clone, PartialEq)]
pub struct Dct2dParams {
    /// Block size for block-DCT (default 8x8)
    pub block_size: usize,
    /// Quality factor (1-100, controls coefficient quantization)
    pub quality: u8,
    /// DCT type (II is standard JPEG)
    pub dct_type: DctType,
}

impl Default for Dct2dParams {
    fn default() -> Self {
        Self {
            block_size: 8,
            quality: 85,
            dct_type: DctType::II,
        }
    }
}

/// DCT type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctType {
    I,
    II,
    III,
    IV,
}

/// 2D Discrete Wavelet Transform parameters
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt2dParams {
    pub wavelet: WaveletType,
    pub levels: usize,
}

impl Default for Dwt2dParams {
    fn default() -> Self {
        Self {
            wavelet: WaveletType::Morlet,
            levels: 4,
        }
    }
}

/// MFCC extraction parameters
#[derive(Debug, Clone, PartialEq)]
pub struct MfccParams {
    pub n_coefficients: usize,
    pub n_mels: usize,
    pub fft_size: usize,
    pub hop_size: usize,
    pub include_deltas: bool,
}

impl Default for MfccParams {
    fn default() -> Self {
        Self {
            n_coefficients: 13,
            n_mels: 40,
            fft_size: 2048,
            hop_size: 512,
            include_deltas: true,
        }
    }
}

/// Chroma feature parameters
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaParams {
    pub n_chroma: usize,
    pub hop_size: usize,
}

impl Default for ChromaParams {
    fn default() -> Self {
        Self {
            n_chroma: 12,
            hop_size: 512,
        }
    }
}

/// Audio fingerprint parameters
#[derive(Debug, Clone, PartialEq)]
pub struct FingerprintParams {
    /// Peak constellation density
    pub fan_value: usize,
    /// Time window for peak pairs
    pub max_time_delta: usize,
}

impl Default for FingerprintParams {
    fn default() -> Self {
        Self {
            fan_value: 15,
            max_time_delta: 200,
        }
    }
}

/// Perceptual hash parameters
#[derive(Debug, Clone, PartialEq)]
pub struct PHashParams {
    /// Hash size in bits (default 64)
    pub hash_bits: usize,
    /// Downscale size before DCT
    pub dct_size: usize,
}

impl Default for PHashParams {
    fn default() -> Self {
        Self {
            hash_bits: 64,
            dct_size: 32,
        }
    }
}

/// Edge detection parameters
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeParams {
    pub method: EdgeMethod,
    pub threshold: f64,
}

/// Edge detection method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMethod {
    Sobel,
    Canny,
    Laplacian,
}

/// Color space for conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Rgb,
    Srgb,
    YCbCr,
    Hsv,
    Lab,
    Grayscale,
}

/// Optical flow parameters
#[derive(Debug, Clone, PartialEq)]
pub struct OpticalFlowParams {
    pub method: FlowMethod,
}

/// Optical flow method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowMethod {
    LucasKanade,
    HornSchunck,
    Farneback,
}

/// Shot detection parameters
#[derive(Debug, Clone, PartialEq)]
pub struct ShotDetectParams {
    /// Threshold for scene change (0.0 - 1.0)
    pub threshold: f64,
    /// Minimum shot length in frames
    pub min_shot_frames: usize,
}

impl Default for ShotDetectParams {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            min_shot_frames: 15,
        }
    }
}

/// Frame extraction parameters
#[derive(Debug, Clone, PartialEq)]
pub struct FrameExtractParams {
    /// Target frames per second (None = all frames)
    pub fps: Option<f32>,
    /// Extract only keyframes
    pub keyframes_only: bool,
}

/// FFT parameters
#[derive(Debug, Clone, PartialEq)]
pub struct FftParams {
    pub size: Option<usize>,
    pub window: WindowFunction,
    pub zero_pad: bool,
}

impl Default for FftParams {
    fn default() -> Self {
        Self {
            size: None,
            window: WindowFunction::Hann,
            zero_pad: true,
        }
    }
}

/// STFT parameters
#[derive(Debug, Clone, PartialEq)]
pub struct StftParams {
    pub window_size: usize,
    pub hop_size: usize,
    pub window: WindowFunction,
    pub fft_size: Option<usize>,
}

impl Default for StftParams {
    fn default() -> Self {
        Self {
            window_size: 256,
            hop_size: 64,
            window: WindowFunction::Hann,
            fft_size: None,
        }
    }
}

/// Wavelet transform parameters
#[derive(Debug, Clone, PartialEq)]
pub struct WaveletParams {
    pub mother: WaveletType,
    pub scales: ScaleSpec,
}

/// Wavelet types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletType {
    Morlet,
    Mexican,
    Paul,
    Dog,
}

/// Scale specification for wavelet transforms
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleSpec {
    Linear {
        start: f64,
        end: f64,
        count: usize,
    },
    Log {
        start: f64,
        end: f64,
        count: usize,
    },
    Frequencies(Vec<Hertz>),
    Octaves {
        start: f64,
        voices_per_octave: usize,
        num_octaves: usize,
    },
}

/// Filter parameters
#[derive(Debug, Clone, PartialEq)]
pub struct FilterParams {
    pub cutoff_low: Option<Hertz>,
    pub cutoff_high: Option<Hertz>,
    pub order: u8,
    pub filter_type: FilterType,
}

/// Filter design type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterType {
    #[default]
    Butterworth,
    Chebyshev {
        ripple_db: u8,
    },
    Bessel,
    Elliptic {
        ripple_db: u8,
        stopband_db: u8,
    },
}

/// Notch filter parameters
#[derive(Debug, Clone, PartialEq)]
pub struct NotchParams {
    pub frequency: Hertz,
    pub q_factor: f64,
}

/// Median filter parameters
#[derive(Debug, Clone, PartialEq)]
pub struct MedianParams {
    pub kernel_size: usize,
}

/// Resample parameters
#[derive(Debug, Clone, PartialEq)]
pub struct ResampleParams {
    pub target_rate: SampleRate,
    pub method: ResampleMethod,
}

/// Resampling methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResampleMethod {
    #[default]
    Sinc,
    Linear,
    Cubic,
    Polyphase,
}

/// Decimation parameters
#[derive(Debug, Clone, PartialEq)]
pub struct DecimateParams {
    pub factor: usize,
    pub antialias: bool,
}

/// Interpolation parameters
#[derive(Debug, Clone, PartialEq)]
pub struct InterpolateParams {
    pub factor: usize,
    pub method: ResampleMethod,
}

/// Z-score normalization parameters
#[derive(Debug, Clone, PartialEq)]
pub struct ZScoreParams {
    pub baseline: BaselineReference,
}

/// Baseline reference specification
#[derive(Debug, Clone, PartialEq)]
pub enum BaselineReference {
    Full,
    First(Seconds),
    Last(Seconds),
    Range { start: Seconds, end: Seconds },
    External(SmolStr),
}

/// Detrend parameters
#[derive(Debug, Clone, PartialEq)]
pub struct DetrendParams {
    pub order: u8,
}

/// Baseline correction parameters
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineParams {
    pub reference: BaselineReference,
    pub method: BaselineMethod,
}

/// Baseline correction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BaselineMethod {
    #[default]
    Subtract,
    Divide,
    Percent,
}

/// Artifact rejection parameters
#[derive(Debug, Clone, PartialEq)]
pub struct RejectParams {
    pub conditions: Vec<RejectCondition>,
}

/// Conditions for artifact rejection
#[derive(Debug, Clone, PartialEq)]
pub enum RejectCondition {
    AmplitudeThreshold { factor: f64 },
    Flatline { duration: Seconds },
    Saturation { threshold: f64 },
    Gradient { max_rate: f64 },
    Custom(Box<ScalarExpr>),
}

/// Artifact interpolation parameters
#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactInterpolateParams {
    pub method: InterpolateMethod,
}

/// Interpolation methods for artifacts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolateMethod {
    #[default]
    Linear,
    Cubic,
    Spline,
    NearestNeighbor,
}

/// Window function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowFunction {
    Rectangular,
    #[default]
    Hann,
    Hamming,
    Blackman,
    Kaiser {
        beta: u8,
    },
    FlatTop,
    Gaussian {
        sigma: u8,
    },
}

/// Window specification for temporal windowing
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSpec {
    pub kind: WindowKind,
    pub causality: Causality,
}

/// Types of temporal windows
#[derive(Debug, Clone, PartialEq)]
pub enum WindowKind {
    /// Non-overlapping windows
    Tumbling { duration: Seconds },
    /// Overlapping windows
    Sliding { duration: Seconds, step: Seconds },
    /// Gap-based sessions
    Session {
        gap: Seconds,
        max_duration: Option<Seconds>,
    },
    /// Event-triggered windows
    Landmark { start: EventRef, end: EventRef },
    /// Frequency band window (for spectral queries)
    FrequencyBand(FrequencyBand),
    /// Frequency bins
    FrequencyBins { count: usize, scale: FrequencyScale },
}

/// Causality constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Causality {
    /// Only past data (real-time safe)
    #[default]
    Causal,
    /// Can use future data (offline only)
    Acausal,
}

/// Frequency scale for binning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrequencyScale {
    #[default]
    Linear,
    Log,
    Mel,
    Bark,
    Erb,
}

/// Event reference for landmark windows
#[derive(Debug, Clone, PartialEq)]
pub struct EventRef {
    pub source: SmolStr,
    pub offset: Option<Seconds>,
}

/// Correlation operations between signals
#[derive(Debug, Clone, PartialEq)]
pub enum CorrelateOp {
    /// Cross-correlation with optional max lag
    CrossCorrelation { max_lag: Option<Seconds> },
    /// Magnitude-squared coherence
    Coherence { band: Option<FrequencyBand> },
    /// Granger causality test
    GrangerCausality {
        order: u8,
        direction: CausalDirection,
    },
    /// Phase locking value
    PhaseLockingValue { band: FrequencyBand },
    /// Transfer entropy
    TransferEntropy { history: usize },
    /// Mutual information
    MutualInformation { bins: usize },
    /// Pearson correlation
    Pearson,
    /// Spearman correlation
    Spearman,
    /// Custom correlation function
    Custom {
        name: SmolStr,
        params: Vec<(SmolStr, Literal)>,
    },
}

/// Direction for causal analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalDirection {
    AtoB,
    BtoA,
    Bidirectional,
}

/// Epoch alignment specification
#[derive(Debug, Clone, PartialEq)]
pub struct AlignSpec {
    /// Time before event
    pub pre: Seconds,
    /// Time after event
    pub post: Seconds,
    /// Baseline period for normalization
    pub baseline: Option<BaselineReference>,
    /// Aggregation across epochs
    pub aggregate: Option<EpochAggregate>,
}

/// Aggregation across epochs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochAggregate {
    Mean,
    Median,
    GrandMean,
    Concatenate,
}

/// Signal fusion methods
#[derive(Debug, Clone, PartialEq)]
pub enum FuseMethod {
    /// Kalman filter fusion
    Kalman { state_model: StateModel },
    /// Complementary filter
    Complementary { alpha: f64 },
    /// Simple concatenation
    Concatenate,
    /// Weighted average
    WeightedAverage { weights: Vec<f64> },
    /// Principal component
    Pca { components: usize },
}

/// State model for Kalman filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateModel {
    Position,
    Velocity,
    Acceleration,
    RigidBody6Dof,
    Custom,
}

/// Aggregate operations
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateOp {
    // Time domain
    Mean,
    Std,
    Var,
    Rms,
    Peak,
    Trough,
    PeakToPeak,
    ZeroCrossings,
    Slope,
    Percentile(f64),

    // Frequency domain
    DominantFrequency,
    SpectralCentroid,
    SpectralEntropy,
    SpectralFlatness,
    BandPower(FrequencyBand),
    FrequencyRatio {
        low: FrequencyBand,
        high: FrequencyBand,
    },

    // Statistical
    Kurtosis,
    Skewness,
    HurstExponent,
    SampleEntropy {
        m: usize,
        r: f64,
    },
    LyapunovExponent,

    // Clinical (domain-specific)
    TremorSeverity {
        scale: ClinicalScale,
    },
    ReactionTime {
        stimulus: EventRef,
        response: EventRef,
    },
    MovementSmoothness {
        method: SmoothnessMethod,
    },

    // ====== MediaQL: Image Aggregations ======

    /// Spatial frequency content of an image region
    SpatialFrequencyContent,
    /// Texture entropy (randomness measure)
    TextureEntropy,
    /// Color histogram distribution
    ColorHistogram { bins: usize },
    /// Edge density (fraction of edge pixels)
    EdgeDensity,

    // ====== MediaQL: Audio Aggregations ======

    /// Pitch contour over time
    PitchContour,
    /// Onset detection strength
    OnsetStrength,
    /// Beat spectrum (rhythmic pattern)
    BeatSpectrum,
    /// Loudness in LUFS
    Loudness,

    // ====== MediaQL: Video Aggregations ======

    /// Scene change count in a video segment
    SceneChangeCount,
    /// Average motion magnitude across frames
    MotionMagnitude,
    /// Temporal flicker metric
    FlickerMetric,

    // ====== MediaQL: Cross-Modal ======

    /// Content-based similarity score
    MediaSimilarity { metric: SimilarityMetric },

    // Custom
    Custom {
        name: SmolStr,
        params: Vec<(SmolStr, Literal)>,
    },
}

/// Similarity metric for media comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Hamming distance on perceptual hashes
    Perceptual,
    /// Cosine similarity on feature vectors
    Cosine,
    /// HDC holographic similarity
    Holographic,
}

/// Clinical scales for tremor severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClinicalScale {
    Updrs,
    Fahn,
    Bain,
    Custom,
}

/// Movement smoothness metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothnessMethod {
    Sparc,
    Ldlj,
    Jerk,
}

/// Scalar expressions (for conditions, thresholds, etc.)
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarExpr {
    Literal(Literal),
    Var(SmolStr),
    Field {
        base: Box<ScalarExpr>,
        field: SmolStr,
    },
    Binary {
        op: BinaryOp,
        left: Box<ScalarExpr>,
        right: Box<ScalarExpr>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<ScalarExpr>,
    },
    Call {
        name: SmolStr,
        args: Vec<ScalarExpr>,
    },
    Aggregate {
        input: Box<SignalExpr>,
        op: AggregateOp,
    },
}

/// Literal values
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(SmolStr),
    Duration(Seconds),
    Frequency(Hertz),
    FrequencyBand(FrequencyBand),
    Array(Vec<Literal>),
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    Abs,
}
