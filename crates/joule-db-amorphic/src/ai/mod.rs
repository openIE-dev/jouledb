//! JouleDB AI — tiered inference runtime.
//!
//! 4 tiers from sub-microsecond to frontier:
//! - **Tier 1 (Holographic)**: Pure HDC ops. Always available. ~0.2µJ. WASM/IoT/browser.
//! - **Tier 2 (Embedded)**: On-device ONNX. Feature-gated. ~2mJ. Edge/NPU.
//! - **Tier 3 (Local)**: Local LLM. Feature-gated. ~1J. Server/GPU.
//! - **Tier 4 (Frontier)**: Cloud API. Feature-gated. ~0.5J. Anywhere.
//!
//! Same interface, same energy receipts, auto-selects tier per query.
//!
//! ```rust,ignore
//! let mut ai = JouleDbAi::new();
//! let result = ai.infer("find movies like Inception", &store, Default::default())?;
//! println!("Tier: {:?}, Energy: {} J", result.receipt.tier, result.receipt.energy_joules);
//! ```

pub mod backends;
pub mod contrast;
pub mod facade;
pub mod flow_bridge;
pub mod flowqit;
pub mod holo;
pub mod metabolic;
pub mod pattern_bridge;
pub mod promoter;
pub mod question;
pub mod receipt;
pub mod selector;
pub mod tier;
pub mod traits;
pub mod ucg;
pub mod vector_bridge;

pub use backends::{CascadeFrontier, CascadeLocal, HdcEmbedded};
pub use contrast::{Contrast, ContrastEngine, ContrastSource, FieldContrast, SpikeCluster};
pub use facade::JouleDbAi;
pub use flow_bridge::{
    ComputeRegime, DomainLut, FlowGraph, FlowGraphBuilder, FlowNode, FlowNodeKind, FlowReasoning,
    FlowResult, FlowWire, LocalFlowReasoner, TraceStep, WireKind,
};
pub use flowqit::{
    ContrastDynamics, DensityState, FlowQitEngine, LANDAUER_FLOOR, MeasurementResult, QitPriors,
    QitRegime, QitSummary,
};
pub use metabolic::{ComputeBudget, MetabolicController, MetabolicState};
pub use pattern_bridge::{
    MatchKind, OodaPosition, PatternBridge, PatternMatch, PatternResolution, PatternResolver,
};
pub use promoter::{AwarenessPromoter, DiscoveredDimension, DiscoverySource};
pub use question::{Frontier, FrontierSource, Question, QuestionEngine, QuestionOutcome};
pub use receipt::{AiReceipt, EnergyProvenance, TokenCount};
pub use selector::{ComplexityScore, HardwareProfile, TierAvailability, classify_complexity};
pub use tier::{InferenceTier, TierConstraints};
pub use traits::{
    AiError, AiMessage, AiOutput, AiResult, AiTool, EmbeddedInference, EnrichedFields, Entity,
    FrontierInference, HolographicInference, LocalInference, ReasoningResult, Sentiment,
    SentimentLabel,
};
pub use ucg::{ContrastMap, ContrastState, NUM_ORBITS, OrbitScores, OrbitWeights, UcgEngine};
pub use vector_bridge::{PhasorVector, binaryhv_to_phasor, cross_similarity, phasor_to_binaryhv};
