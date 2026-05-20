//! Error types for JouleDB Novel

use thiserror::Error;

/// Main error type for novel database operations
#[derive(Error, Debug)]
pub enum NovelError {
    /// SDM-related error
    #[error("SDM error: {0}")]
    #[cfg(feature = "sdm")]
    Sdm(#[from] crate::sdm::SDMError),

    /// Holographic memory error
    #[error("Holographic error: {0}")]
    #[cfg(feature = "holographic")]
    Holographic(String),

    /// Hyperdimensional computing error
    #[error("Hyperdimensional error: {0}")]
    #[cfg(feature = "hyperdimensional")]
    Hyperdimensional(String),

    /// Predictor error
    #[error("Predictor error: {0}")]
    #[cfg(feature = "predictive")]
    Predictor(String),

    /// Thermodynamic optimizer error
    #[error("Thermodynamic error: {0}")]
    #[cfg(feature = "thermodynamic")]
    Thermodynamic(String),

    /// Manifold error
    #[error("Manifold error: {0}")]
    #[cfg(feature = "manifold")]
    Manifold(String),

    /// SNN error
    #[error("SNN error: {0}")]
    #[cfg(feature = "spiking")]
    Spiking(String),

    /// Learned index error
    #[error("Learned index error: {0}")]
    #[cfg(feature = "learned")]
    Learned(String),

    /// AmorphicEngine unified error
    #[error("AmorphicEngine error: {0}")]
    #[cfg(feature = "hdc-research")]
    AmorphicEngine(String),

    /// Neurosymbolic error
    #[error("Neurosymbolic error: {0}")]
    #[cfg(feature = "neurosymbolic")]
    Neurosymbolic(String),

    /// Invertible visualization error
    #[error("Invertible error: {0}")]
    #[cfg(feature = "invertible")]
    Invertible(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// Result type for novel database operations
pub type NovelResult<T> = Result<T, NovelError>;
