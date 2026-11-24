//! # Layer 1: The Knowledge Core
//!
//! A small, compressed database representing the structural skeleton of human
//! knowledge. Not the facts — the *shape* of how facts relate to each other.
//!
//! Architecture:
//! ```text
//! ConceptNet/Wikidata → relationship triples → holographic encoding
//!   → SDM storage → traversal function → contrast baseline
//! ```
//!
//! The core answers: "given concept X, what is structurally nearby?"
//! Not by lookup — by navigation through a manifold of compressed relationships.
//!
//! ## Encoding
//! - Each **concept** is a BinaryHV (character n-gram binding)
//! - Each **relation type** is a fixed BinaryHV (codebook of ~40 types)
//! - Each **triple** (subject, relation, object) is holographically bound:
//!   `triple_hv = subject ⊗ relation ⊗ Permute(object)`
//! - All triples are **bundled** into the core via majority-vote accumulation
//!
//! ## Retrieval
//! - Query: unbind the query concept from the core → what's bound to it?
//! - Traverse: follow relation chains through repeated unbind + compare
//! - Contrast: measure novelty against the core centroid

pub mod ask;
pub mod attractor_tier;
pub mod awareness;
pub mod benchmark;
pub mod bpe;
pub mod cleanup;
pub mod concept;
pub mod context;
pub mod core;
pub mod cortex_gen;
pub mod decoder;
pub mod eigenbasis;
pub mod encoder_tuning;
pub mod energy_receipt;
pub mod expand;
pub mod flowr_executor;
pub mod generate;
pub mod ghrr;
pub mod grounded;
pub mod ingest;
pub mod live;
pub mod mamba_decoder;
pub mod materializer;
pub mod negative;
pub mod oracle;
pub mod pathstore;
pub mod pattern_resolver;
pub mod relation;
pub mod runtime;
pub mod spatial_tier;
pub mod structural_encoder;
pub mod tier0;
pub mod traverse;
pub mod triple;
pub mod ucg_backend;
pub mod ucg_live;

pub use ask::{Answer, Ask};
pub use awareness::{
    Action, ActionTrigger, Awareness, Reflection as AwarenessReflection, SensorChannel, SensorState,
};
pub use bpe::BpeTokenizer;
pub use cleanup::{CleanupMemory, CleanupResult};
pub use concept::{ConceptEncoder, EncodedConcept};
pub use context::ContextWindow;
pub use core::KnowledgeCore;
pub use decoder::{
    DecoderContext, DecoderResult, DecoderStyle, TemplateDecoder, TextDecoder, build_prompt,
};
pub use eigenbasis::{Eigenbasis, NUM_EIGENDIMS, NUM_PATTERNS, PATTERN_NAMES, PatternScores};
pub use encoder_tuning::{
    AdaptiveEncoder, EncoderMetrics, measure_encoding, recommended_dimension,
};
pub use energy_receipt::{
    EnergyReceipt, GPU_BASELINE_PJ, LLM_BASELINE_PJ, format_picojoules, format_ratio,
    tier_native_floor_pj,
};
pub use expand::{Expander, ExpansionResult, KnowledgeSource, TextSource, WebSearchSource};
pub use generate::{GenerationResult, Generator, SequenceMemory};
pub use ghrr::GhrrVector;
pub use grounded::{
    Groundable, GroundedInput, Modality, encode_audio_frame, encode_image_patch, encode_numeric,
    encode_sensor, encode_structured, fuse_modalities,
};
pub use ingest::ConceptNetParser;
pub use live::LiveIntelligence;
pub use mamba_decoder::MambaDecoder;
pub use materializer::{EntropyLevel, MaterializeResult, Materializer, Skill, Source};
pub use negative::{Absence, NegationOperator, NegativeKnowledge};
pub use oracle::{InMemoryBackend, Oracle, OracleBackend, OracleResult, OracleSource};
pub use pathstore::PathStore;
pub use pattern_resolver::PatternLangResolver;
pub use relation::{RelationCodebook, RelationType};
pub use runtime::{Runtime, RuntimeStatus};
pub use structural_encoder::StructuralEncoder;
pub use tier0::{QueryType, Tier0, Tier0Result, Tier0Source};
pub use traverse::{TraversalResult, TraversalStep, Traverser};
pub use triple::{EncodedTriple, Triple};
pub use ucg_backend::{UcgConfig, UcgFileBackend};
pub use ucg_live::UcgLive;
