//! Pluggable tier traits — implementations live outside this crate.
//!
//! Only Tier 1 (HolographicInference) is implemented here.
//! Tiers 2-4 are injected as trait objects by the consumer.

use joule_db_hdc::BinaryHV;
use serde::{Deserialize, Serialize};

use super::receipt::AiReceipt;
use crate::{AmorphicError, RecordId};

/// Error type for AI operations.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    #[error("AI operation failed: {0}")]
    OperationFailed(String),
    #[error("Tier not available: {0}")]
    TierNotAvailable(String),
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("Energy budget exceeded")]
    EnergyBudgetExceeded,
    #[error("Amorphic error: {0}")]
    Amorphic(#[from] AmorphicError),
}

/// Result of an AI inference operation.
#[derive(Debug, Clone)]
pub struct AiResult {
    /// Primary output (text, classification label, etc.)
    pub output: AiOutput,
    /// Energy receipt
    pub receipt: AiReceipt,
}

/// AI operation output variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AiOutput {
    /// Text response (from generation, summarization, etc.)
    Text(String),
    /// Classification result
    Classification { label: String, confidence: f32 },
    /// Similarity results (record_id, score)
    Similarity(Vec<(RecordId, f32)>),
    /// Embedding (as BinaryHV words for storage)
    Embedding(Vec<u64>),
    /// Structured data (JSON)
    Structured(serde_json::Value),
    /// Multiple tags
    Tags(Vec<String>),
}

/// Named entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

/// Sentiment analysis result.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Sentiment {
    /// -1.0 (very negative) to 1.0 (very positive)
    pub score: f32,
    pub label: SentimentLabel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SentimentLabel {
    Positive,
    Neutral,
    Negative,
}

/// Reasoning result with chain-of-thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    pub answer: String,
    pub reasoning_steps: Vec<String>,
    pub confidence: f32,
    pub sources_used: Vec<String>,
}

/// AI message for multi-turn conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiMessage {
    pub role: String, // "user", "assistant", "system"
    pub content: String,
}

/// AI tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiTool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Enriched fields added by AI to a record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedFields {
    pub sentiment: Option<Sentiment>,
    pub language: Option<String>,
    pub entities: Vec<Entity>,
    pub tags: Vec<String>,
    pub summary: Option<String>,
}

// ============================================================================
// Tier Traits — pluggable backends
// ============================================================================

/// Tier 1: Holographic inference — pure HDC operations.
/// Always available. Uses only joule-db-hdc.
pub trait HolographicInference: Send + Sync {
    /// Find nearest holograms to query.
    fn similarity(&self, query: &BinaryHV, k: usize) -> Vec<(RecordId, f32)>;

    /// Attention-SDM read: content-addressable memory lookup.
    fn attention_read(&self, query: &BinaryHV) -> Vec<f64>;

    /// Classify by holographic similarity to category prototypes.
    fn classify_holo(&self, input: &BinaryHV, categories: &[(String, BinaryHV)]) -> (String, f32);

    /// Encode text into BinaryHV using character n-gram binding.
    fn encode_text(&self, text: &str) -> BinaryHV;
}

/// Tier 2: Embedded inference — small on-device models.
/// Behind `feature = "ai-embedded"`.
pub trait EmbeddedInference: Send + Sync {
    /// Generate embedding, returned as BinaryHV for uniform storage.
    fn embed(&self, text: &str) -> Result<(BinaryHV, AiReceipt), AiError>;

    /// Classify text into categories.
    fn classify(&self, text: &str, categories: &[String]) -> Result<(String, f32, AiReceipt), AiError>;

    /// Named entity recognition.
    fn extract_entities(&self, text: &str) -> Result<(Vec<Entity>, AiReceipt), AiError>;

    /// Sentiment analysis.
    fn sentiment(&self, text: &str) -> Result<(Sentiment, AiReceipt), AiError>;

    /// Language detection.
    fn detect_language(&self, text: &str) -> Result<(String, AiReceipt), AiError>;
}

/// Tier 3: Local LLM inference.
/// Behind `feature = "ai-local"`.
pub trait LocalInference: Send + Sync {
    /// Text generation with context.
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<(String, AiReceipt), AiError>;

    /// Natural language to query (MediaQL/SQL).
    fn nl_to_query(&self, nl: &str, schema_hint: &str) -> Result<(String, AiReceipt), AiError>;

    /// Summarize text.
    fn summarize(&self, text: &str, max_words: u32) -> Result<(String, AiReceipt), AiError>;
}

/// Tier 4: Frontier API inference.
/// Behind `feature = "ai-frontier"`.
pub trait FrontierInference: Send + Sync {
    /// Full completion with tool support.
    fn complete(
        &self,
        messages: Vec<AiMessage>,
        tools: &[AiTool],
    ) -> Result<(String, AiReceipt), AiError>;

    /// Reasoning with chain-of-thought.
    fn reason(
        &self,
        question: &str,
        context: &str,
    ) -> Result<(ReasoningResult, AiReceipt), AiError>;

    /// Dense embedding via API.
    fn embed_api(&self, text: &str) -> Result<(Vec<f32>, AiReceipt), AiError>;
}
