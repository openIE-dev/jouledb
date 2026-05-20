//! # JouleDB HDC
//!
//! Hyperdimensional Computing (HDC) library for high-performance similarity search.
//!
//! ## Hyperdimensional Computing / Vector Symbolic Architecture
//!
//! A unified architecture combining:
//!
//! - **Binary Hypervectors** - 10,000-dimensional binary vectors with XOR binding
//! - **Holographic Memory** - Distributed pattern storage with superposition
//! - **Sparse Distributed Memory (SDM)** - Content-addressable Kanerva memory
//! - **Predictive Prefetching** - Markov/N-gram query prediction
//!
//! ## Core Concepts
//!
//! HDC represents data as high-dimensional binary vectors where:
//! - **Bind (XOR)** - Creates associations between concepts
//! - **Bundle (majority vote)** - Combines multiple vectors into one
//! - **Permute (rotate)** - Encodes sequence/position information
//! - **Similarity (Hamming)** - Measures how related two vectors are
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use joule_db_hdc::{BinaryHV, BundleAccumulator};
//!
//! // Create random hypervectors for concepts
//! let apple = BinaryHV::random(10000, 42);
//! let red = BinaryHV::random(10000, 43);
//!
//! // Bind creates "red apple" association
//! let red_apple = apple.bind(&red);
//!
//! // Bundle multiple items with majority vote
//! let mut acc = BundleAccumulator::new(10000);
//! acc.add(&apple);
//! acc.add(&red_apple);
//! let combined = acc.threshold();
//!
//! // High similarity = related concepts
//! println!("Similarity: {}", combined.similarity(&apple));
//! ```
//!
//! ## Design Principles
//!
//! 1. **Pure Rust** - No WASM dependencies in core
//! 2. **Feature-gated** - Enable only what you need
//! 3. **Platform-agnostic** - Works on native and WASM targets
//!
//! ## Extended HDC Components
//!
//! Feature-gated modules for advanced database paradigms:
//!
//! - **Neurosymbolic** (`neurosymbolic`) - Neural + symbolic reasoning hybrid
//! - **Spiking Neural Networks** (`spiking`) - Temporal data processing
//! - **Invertible Encodings** (`invertible`) - Data embedded in visualizations
//! - **Post-Quantum Cryptography** (`pqc`) - NIST FIPS 203/204/205, HQC KEM
//! - **Thermodynamic Optimizer** (`thermodynamic`) - Simulated annealing query optimization
//! - **Information Manifold** (`manifold`) - Geodesic-based similarity search
//! - **Learned Indexes** (`learned`) - ML-optimized data access, Neural LSH
//! - **Quantum-Inspired** (`quantum-inspired`) - Simulated quantum annealing
//! - **Tensor Networks** (`tensor-network`) - MPS compression
//! - **Variational Encoding** (`variational`) - Gradient-learned binary encoding
//! - **Radix Spline** (`experimental-radix-spline`) - Learned index interpolation
//! - **Amorphic Engine** (`hdc-research`) - Unified HDC database engine

#![allow(dead_code, unused_imports, unused_assignments)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod bridge;
pub mod error;

/// The 15 Primitives of Hyperdimensional Computing.
/// First-class APIs for the irreducible operations of a contrast recognition runtime.
pub mod primitives;

/// Common trait for domain-specific HDC encoding modules
pub mod domain;

/// Macro for generating domain-specific HDC encoder structs
pub mod domain_macro;

/// SIMD-accelerated operations (always available, auto-detects CPU features)
pub mod simd;

/// GPU-accelerated HDC operations via ComputeBackend dispatch
pub mod gpu_dispatch;

/// WASM SIMD-accelerated operations (only on wasm32 targets)
#[cfg(feature = "hyperdimensional")]
pub mod simd_wasm;

/// Binary Resonator Network — iterative factorization for BinaryHV.
/// Recovers bound pairs from bundled composites, raising effective capacity.
#[cfg(feature = "holographic")]
pub mod resonator;

// Core HDC Components
#[cfg(feature = "sdm")]
pub mod sdm;

#[cfg(feature = "holographic")]
pub mod holographic;

#[cfg(feature = "holographic")]
pub mod holographic_kv;

#[cfg(feature = "holographic")]
pub mod turbo_holographic;

#[cfg(feature = "hyperdimensional")]
pub mod hyperdimensional;

#[cfg(feature = "hyperdimensional")]
pub mod binary_hd;

/// Ternary HDC: {-1, 0, +1} hypervectors with packed trit encoding
#[cfg(feature = "hyperdimensional")]
pub mod ternary_hv;

#[cfg(feature = "hyperdimensional")]
pub mod fourier_bind;

#[cfg(feature = "hyperdimensional")]
pub mod map_bind;

#[cfg(feature = "predictive")]
pub mod predictive;

// Extended HDC components

/// Neurosymbolic reasoning: neural + symbolic hybrid
#[cfg(feature = "neurosymbolic")]
pub mod neurosymbolic;

/// Spiking Neural Networks: temporal data processing
#[cfg(feature = "spiking")]
pub mod spiking;

/// Invertible encodings: data embedded in visualizations
#[cfg(feature = "invertible")]
pub mod invertible;

/// Post-Quantum Cryptography: lattice KEM, hash signatures, HQC
#[cfg(feature = "pqc")]
pub mod pqc;

/// Thermodynamic optimizer: simulated annealing for query optimization
#[cfg(feature = "thermodynamic")]
pub mod thermodynamic;

/// Information manifold: geodesic-based similarity search
#[cfg(feature = "manifold")]
pub mod manifold;

/// Learned indexes: ML-optimized data access patterns
#[cfg(feature = "learned")]
pub mod learned;

/// Neural locality-sensitive hashing
#[cfg(feature = "learned")]
pub mod neural_lsh;

/// Quantum-inspired optimization: simulated quantum annealing
#[cfg(feature = "quantum-inspired")]
pub mod quantum_inspired;

/// Tensor network compression: MPS for high-dimensional data
#[cfg(feature = "tensor-network")]
pub mod tensor_network;

/// Variational encoding: gradient-learned binary encoding
#[cfg(feature = "variational")]
pub mod variational;

/// Radix spline: learned index with radix spline interpolation
#[cfg(feature = "experimental-radix-spline")]
pub mod radix_spline;

/// Amorphic Engine: unified HDC API
#[cfg(feature = "hdc-research")]
pub mod amorphic_engine;

/// Amorphic Engine storage: persistent engine with amorphic records
#[cfg(feature = "hdc-research")]
pub mod amorphic_engine_storage;

// Re-exports
pub use domain::DomainEncoder;
pub use error::{NovelError, NovelResult};

#[cfg(feature = "sdm")]
pub use sdm::{SDMAddress, SDMError, SDMStats, SparseDistributedMemory};

pub use holographic_kv::{
    HolographicKV, HolographicKVConfig, HolographicKVDynamic, HolographicKVError,
    HolographicKVMetrics,
};
pub use turbo_holographic::{
    BinaryHV, BinaryHolographicDirect, BundleAccumulator, ByteCodebook, HolographicStore,
    HybridHolographic, LazyHolographicWriter, LazyWriteConfig, LazyWriteStats, TurboConfig,
    TurboHolographic, UltraHolographic, UltraHolographicFast, UltraHolographicV2,
};

#[cfg(feature = "holographic")]
pub use holographic::{Complex, HolographicError, HolographicStorage};

#[cfg(feature = "hyperdimensional")]
pub use hyperdimensional::{HDError, HyperVector, HyperdimensionalStorage};

#[cfg(feature = "hyperdimensional")]
pub use binary_hd::{
    AdaptiveDimensionConfig, AssociativeMemory, BatchOps, BinaryHyperVector, Codebook,
    DEFAULT_DIMENSIONS, LEGACY_DIMENSIONS, SparseBinaryHyperVector, estimate_capacity,
    expected_random_similarity, generate_seed, optimal_dimensions,
};

#[cfg(feature = "hyperdimensional")]
pub use ternary_hv::{BundleAccumulatorTernary, TernaryHV, TernaryHolographicKV};

#[cfg(feature = "hyperdimensional")]
pub use fourier_bind::{
    FourierBenchmark, FourierBindable, FourierBinder, FourierBinderStats, FourierConfig,
};

#[cfg(feature = "hyperdimensional")]
pub use map_bind::{BindingStrategy, HLBBinder, MAPBinder};

#[cfg(feature = "predictive")]
pub use predictive::{NGramPredictor, Prediction, PredictorError, PredictorStats, QueryPredictor};

// Extended HDC re-exports

#[cfg(feature = "neurosymbolic")]
pub use neurosymbolic::{
    NeuralLayer, NeurosymbolicDB, NeurosymbolicError, QueryType, SymbolicReasoner,
};

#[cfg(feature = "invertible")]
pub use invertible::{EncodingMode, InvVis, InvertibleError, VisCode};

#[cfg(feature = "thermodynamic")]
pub use thermodynamic::{OptimizerStats, QueryPlan, ThermoError, ThermodynamicOptimizer};

#[cfg(feature = "manifold")]
pub use manifold::{
    DistanceMetric, HNSWIndex, HNSWResult, InformationManifold, ManifoldError, ManifoldPoint,
};

#[cfg(feature = "holographic")]
pub use resonator::{BinaryCodebook, BinaryResonator, FactorResult, ResonatorResult};

#[cfg(feature = "spiking")]
pub use spiking::{Neuron, SNNError, SpikeEvent, SpikeNeuralNetwork};

#[cfg(feature = "learned")]
pub use learned::{LearnedError, LearnedIndex, LearnedIndexModel};

#[cfg(feature = "learned")]
pub use neural_lsh::{NeuralHashFunction, NeuralLSHConfig, NeuralLSHIndex};

#[cfg(feature = "experimental-radix-spline")]
pub use radix_spline::{LearnedBTreeHybrid, RadixSpline, RadixSplineConfig, SplinePoint};

#[cfg(feature = "quantum-inspired")]
pub use quantum_inspired::{AnnealingConfig, QuantumAnnealer, QuantumTunneler, QueryPlanOptimizer};

#[cfg(feature = "tensor-network")]
pub use tensor_network::{MPS, MPSConfig, MPSTensor, TensorNetworkIndex, TreeTensorNetwork};

#[cfg(feature = "variational")]
pub use variational::{
    OnlineVariationalLearner, TrainingConfig, VariationalBinaryEncoder, VariationalEncoder,
    VariationalIndex,
};

#[cfg(feature = "pqc")]
pub use pqc::{
    ConstantTime, HqcCiphertext, HqcKeyPair, HqcParams, HybridCiphertext, HybridEncryption,
    HybridHqcCiphertext, HybridHqcKem, HybridHqcKeyPair, HybridHqcPublicKey, HybridHqcSecretKey,
    HybridKem, KeyMetadata, ML_DSA_44_PARAMS, ML_DSA_65_PARAMS, ML_DSA_87_PARAMS,
    ML_KEM_512_PARAMS, ML_KEM_768_PARAMS, ML_KEM_1024_PARAMS, MlDsa44, MlDsa65, MlDsa87,
    MlDsaParams, MlDsaSignature, MlDsaSigningKey, MlDsaVerificationKey, MlKem512, MlKem768,
    MlKem1024, MlKemCiphertext, MlKemDecapsulationKey, MlKemEncapsulationKey, MlKemParams,
    MlKemSharedSecret, PqcError, PqcKeyStore, PqcResult, SecureZeroingVec, Sha3_256, Sha3_512,
    Shake128, Shake256, SlhDsa128f, SlhDsa128s, SlhDsa192f, SlhDsa192s, SlhDsa256f, SlhDsa256s,
    SlhDsaParams, SlhDsaPublicKey, SlhDsaSecretKey, SlhSignature, StoredKey,
};

#[cfg(feature = "hdc-research")]
pub use amorphic_engine::{
    AmorphicEngine, AmorphicEngineConfig, AmorphicEngineError, AmorphicEngineResult,
    AmorphicEngineStats,
};

#[cfg(feature = "hdc-research")]
pub use amorphic_engine_storage::{
    AmorphicStorageConfig, AmorphicStorageEngine, AmorphicStorageStats, HybridQueryResult,
    QuerySource,
};
