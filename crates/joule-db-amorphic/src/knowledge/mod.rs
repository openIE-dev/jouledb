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

pub mod concept;
pub mod relation;
pub mod triple;
pub mod core;
pub mod traverse;
pub mod ingest;
pub mod generate;
pub mod expand;
pub mod oracle;
pub mod ghrr;
pub mod cleanup;
pub mod context;
pub mod encoder_tuning;
pub mod eigenbasis;
pub mod tier0;
pub mod ucg_live;
pub mod decoder;
pub mod mamba_decoder;
pub mod pattern_resolver;
pub mod materializer;
pub mod energy_receipt;
pub mod spatial_tier;
pub mod attractor_tier;
pub mod cortex_gen;
pub mod flowr_executor;
pub mod structural_encoder;
pub mod pathstore;
pub mod ucg_backend;
pub mod bpe;
pub mod negative;
pub mod grounded;
pub mod awareness;
pub mod benchmark;
pub mod live;
pub mod runtime;
pub mod ask;

pub use concept::{ConceptEncoder, EncodedConcept};
pub use relation::{RelationType, RelationCodebook};
pub use triple::{Triple, EncodedTriple};
pub use core::KnowledgeCore;
pub use traverse::{Traverser, TraversalResult, TraversalStep};
pub use ingest::ConceptNetParser;
pub use generate::{Generator, GenerationResult, SequenceMemory};
pub use expand::{Expander, ExpansionResult, KnowledgeSource, TextSource, WebSearchSource};
pub use oracle::{Oracle, OracleBackend, OracleResult, OracleSource, InMemoryBackend};
pub use ghrr::GhrrVector;
pub use cleanup::{CleanupMemory, CleanupResult};
pub use context::ContextWindow;
pub use eigenbasis::{Eigenbasis, PatternScores, PATTERN_NAMES, NUM_PATTERNS, NUM_EIGENDIMS};
pub use tier0::{Tier0, Tier0Result, Tier0Source, QueryType};
pub use ucg_live::UcgLive;
pub use decoder::{TextDecoder, DecoderContext, DecoderResult, DecoderStyle, TemplateDecoder, build_prompt};
pub use mamba_decoder::MambaDecoder;
pub use pattern_resolver::PatternLangResolver;
pub use materializer::{Materializer, MaterializeResult, EntropyLevel, Source, Skill};
pub use energy_receipt::{EnergyReceipt, tier_native_floor_pj, format_picojoules, format_ratio, GPU_BASELINE_PJ, LLM_BASELINE_PJ};
pub use encoder_tuning::{AdaptiveEncoder, EncoderMetrics, measure_encoding, recommended_dimension};
pub use structural_encoder::StructuralEncoder;
pub use pathstore::PathStore;
pub use ucg_backend::{UcgConfig, UcgFileBackend};
pub use bpe::BpeTokenizer;
pub use negative::{NegationOperator, NegativeKnowledge, Absence};
pub use grounded::{GroundedInput, Modality, Groundable, encode_numeric, encode_structured, encode_audio_frame, encode_image_patch, encode_sensor, fuse_modalities};
pub use awareness::{Awareness, SensorState, SensorChannel, Action, ActionTrigger, Reflection as AwarenessReflection};
pub use live::LiveIntelligence;
pub use runtime::{Runtime, RuntimeStatus};
pub use ask::{Ask, Answer};
