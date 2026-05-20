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

pub mod receipt;
pub mod tier;
pub mod traits;
pub mod holo;
pub mod selector;
pub mod facade;
pub mod question;
pub mod promoter;
pub mod contrast;
pub mod metabolic;
pub mod vector_bridge;
pub mod ucg;
pub mod flow_bridge;
pub mod pattern_bridge;
pub mod flowqit;

pub use facade::JouleDbAi;
pub use receipt::{AiReceipt, EnergyProvenance, TokenCount};
pub use tier::{InferenceTier, TierConstraints};
pub use traits::{
    AiError, AiMessage, AiOutput, AiResult, AiTool, EmbeddedInference, EnrichedFields, Entity,
    FrontierInference, HolographicInference, LocalInference, ReasoningResult, Sentiment,
    SentimentLabel,
};
pub use selector::{ComplexityScore, HardwareProfile, TierAvailability, classify_complexity};
pub use question::{Frontier, FrontierSource, Question, QuestionEngine, QuestionOutcome};
pub use promoter::{AwarenessPromoter, DiscoveredDimension, DiscoverySource};
pub use contrast::{Contrast, ContrastEngine, ContrastSource, FieldContrast, SpikeCluster};
pub use metabolic::{ComputeBudget, MetabolicController, MetabolicState};
pub use vector_bridge::{PhasorVector, binaryhv_to_phasor, phasor_to_binaryhv, cross_similarity};
pub use ucg::{ContrastMap, ContrastState, OrbitScores, OrbitWeights, UcgEngine, NUM_ORBITS};
pub use flow_bridge::{
    ComputeRegime, DomainLut, FlowGraph, FlowGraphBuilder, FlowNode, FlowNodeKind, FlowReasoning,
    FlowResult, FlowWire, LocalFlowReasoner, TraceStep, WireKind,
};
pub use pattern_bridge::{
    MatchKind, OodaPosition, PatternBridge, PatternMatch, PatternResolution, PatternResolver,
};
pub use flowqit::{
    ContrastDynamics, DensityState, FlowQitEngine, MeasurementResult, QitPriors, QitRegime,
    QitSummary, LANDAUER_FLOOR,
};
