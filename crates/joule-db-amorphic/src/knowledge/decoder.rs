//! SSM Decoder: lightweight language model for fluent text generation.
//!
//! JouleDB AI does the reasoning (34 patterns, 15 primitives, eigenbasis).
//! The decoder does the talking (converts structured knowledge into fluent text).
//!
//! Architecture:
//! ```text
//! JouleDB AI: "The answer is: cancer and war share structural patterns
//!              (replication, feedback, emergence) at 93% similarity"
//!
//!      ↓ (structured context)
//!
//! SSM Decoder: "Cancer and war are remarkably similar in their fundamental
//!              structure. Both rely on replication to spread, feedback loops
//!              to accelerate, and emergence to produce complex outcomes
//!              from simple rules."
//! ```
//!
//! The decoder is a TRANSLATOR, not a reasoner. It adds fluency, not knowledge.
//! Think of it as a text-to-speech system for structured thoughts.
//!
//! Target backends:
//! - mamba.rs (130M params, pure Rust, Apple Silicon) — preferred
//! - Any GGUF model via llama.cpp bindings
//! - API fallback (Tier 4) for highest quality

/// Trait for any text decoder (SSM, LLM, template, etc.)
pub trait TextDecoder: Send + Sync {
    /// Generate fluent text from structured context.
    /// `context` is the structured knowledge (entities, relations, patterns).
    /// `max_tokens` limits output length.
    /// Returns generated text + energy estimate.
    fn decode(&self, context: &DecoderContext, max_tokens: usize) -> DecoderResult;

    /// Model name (for receipts).
    fn model_name(&self) -> &str;

    /// Estimated energy per token (joules).
    fn energy_per_token(&self) -> f64;
}

/// Structured context passed to the decoder.
#[derive(Clone, Debug)]
pub struct DecoderContext {
    /// The query/question that triggered this response.
    pub query: String,
    /// Key entities mentioned.
    pub entities: Vec<String>,
    /// Key relationships discovered.
    pub relationships: Vec<(String, String, String)>, // (subject, relation, object)
    /// Structural patterns involved.
    pub patterns: Vec<(String, f64)>, // (pattern_name, score)
    /// Structural similarity score (if comparing two things).
    pub similarity: Option<f64>,
    /// Raw Tier 0 answer (if available — decoder should improve on this).
    pub raw_answer: Option<String>,
    /// Desired tone/style.
    pub style: DecoderStyle,
}

/// Desired output style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderStyle {
    /// Brief, factual answer.
    Concise,
    /// Explanatory paragraph.
    Explanatory,
    /// Technical/scientific.
    Technical,
    /// Conversational.
    Conversational,
}

impl Default for DecoderStyle {
    fn default() -> Self {
        Self::Concise
    }
}

/// Result from the decoder.
#[derive(Clone, Debug)]
pub struct DecoderResult {
    /// The fluent text output.
    pub text: String,
    /// Number of tokens generated.
    pub tokens: usize,
    /// Energy consumed (joules).
    pub energy: f64,
    /// Latency (microseconds).
    pub latency_us: u64,
}

/// Template-based decoder: zero-cost, no model needed.
/// Uses the raw Tier 0 answer with light formatting.
/// This is the fallback when no SSM/LLM is available.
pub struct TemplateDecoder;

impl TextDecoder for TemplateDecoder {
    fn decode(&self, context: &DecoderContext, _max_tokens: usize) -> DecoderResult {
        let text = if let Some(ref raw) = context.raw_answer {
            // Light formatting of the raw answer
            match context.style {
                DecoderStyle::Concise => raw.clone(),
                DecoderStyle::Explanatory => {
                    let mut out = String::new();
                    if !context.entities.is_empty() {
                        out.push_str(&context.entities[0]);
                    }
                    if let Some(sim) = context.similarity {
                        if context.entities.len() >= 2 {
                            out.push_str(&format!(
                                " and {} share {:.0}% structural similarity",
                                context.entities[1],
                                sim * 100.0
                            ));
                        }
                    }
                    if !context.patterns.is_empty() {
                        let pattern_str: Vec<String> = context
                            .patterns
                            .iter()
                            .take(5)
                            .map(|(p, s)| format!("{} ({:.0}%)", p, s * 100.0))
                            .collect();
                        out.push_str(&format!(
                            ". Key patterns: {}",
                            pattern_str.join(", ")
                        ));
                    }
                    if out.is_empty() {
                        raw.clone()
                    } else {
                        out
                    }
                }
                _ => raw.clone(),
            }
        } else {
            // Generate from entities and patterns
            let mut parts = Vec::new();
            for entity in &context.entities {
                parts.push(entity.clone());
            }
            if !context.patterns.is_empty() {
                let patterns: Vec<&str> = context.patterns.iter().take(3).map(|(p, _)| p.as_str()).collect();
                parts.push(format!("exhibits {}", patterns.join(", ")));
            }
            parts.join(". ")
        };

        let tokens = text.split_whitespace().count();
        DecoderResult {
            text,
            tokens,
            energy: tokens as f64 * 0.000_000_01, // 10 nJ per token (template is free)
            latency_us: 1, // <1µs
        }
    }

    fn model_name(&self) -> &str {
        "template"
    }

    fn energy_per_token(&self) -> f64 {
        0.000_000_01 // 10 nJ
    }
}

/// Prompt builder: constructs a prompt for an SSM/LLM decoder
/// from structured context. This is the bridge between JouleDB AI's
/// structured knowledge and the decoder's text input.
pub fn build_prompt(context: &DecoderContext) -> String {
    let mut prompt = String::new();

    prompt.push_str("Given the following knowledge, write a ");
    prompt.push_str(match context.style {
        DecoderStyle::Concise => "brief, factual",
        DecoderStyle::Explanatory => "clear, explanatory",
        DecoderStyle::Technical => "precise, technical",
        DecoderStyle::Conversational => "friendly, conversational",
    });
    prompt.push_str(" response.\n\n");

    prompt.push_str(&format!("Question: {}\n\n", context.query));

    if !context.entities.is_empty() {
        prompt.push_str(&format!("Entities: {}\n", context.entities.join(", ")));
    }

    if !context.relationships.is_empty() {
        prompt.push_str("Facts:\n");
        for (s, r, o) in &context.relationships {
            prompt.push_str(&format!("- {} {} {}\n", s, r, o));
        }
    }

    if !context.patterns.is_empty() {
        prompt.push_str("Structural patterns:\n");
        for (p, s) in &context.patterns {
            prompt.push_str(&format!("- {} ({:.0}%)\n", p, s * 100.0));
        }
    }

    if let Some(sim) = context.similarity {
        prompt.push_str(&format!("\nStructural similarity: {:.0}%\n", sim * 100.0));
    }

    prompt.push_str("\nResponse:");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_decoder_concise() {
        let decoder = TemplateDecoder;
        let ctx = DecoderContext {
            query: "What is cancer?".into(),
            entities: vec!["cancer".into()],
            relationships: vec![],
            patterns: vec![
                ("replication".into(), 0.9),
                ("feedback".into(), 0.8),
                ("emergence".into(), 0.7),
            ],
            similarity: None,
            raw_answer: Some("cancer exhibits replication, feedback, emergence".into()),
            style: DecoderStyle::Concise,
        };

        let result = decoder.decode(&ctx, 100);
        assert!(!result.text.is_empty());
        assert!(result.energy < 0.000_001); // Sub-µJ
    }

    #[test]
    fn test_template_decoder_explanatory() {
        let decoder = TemplateDecoder;
        let ctx = DecoderContext {
            query: "How are cancer and war related?".into(),
            entities: vec!["cancer".into(), "war".into()],
            relationships: vec![],
            patterns: vec![
                ("replication".into(), 0.85),
                ("feedback".into(), 0.75),
                ("emergence".into(), 0.8),
            ],
            similarity: Some(0.93),
            raw_answer: Some("93% structurally similar".into()),
            style: DecoderStyle::Explanatory,
        };

        let result = decoder.decode(&ctx, 100);
        assert!(result.text.contains("93%"));
        assert!(result.text.contains("cancer"));
    }

    #[test]
    fn test_build_prompt() {
        let ctx = DecoderContext {
            query: "How are cancer and war related?".into(),
            entities: vec!["cancer".into(), "war".into()],
            relationships: vec![
                ("cancer".into(), "exhibits".into(), "replication".into()),
                ("war".into(), "exhibits".into(), "replication".into()),
            ],
            patterns: vec![("replication".into(), 0.9)],
            similarity: Some(0.93),
            raw_answer: None,
            style: DecoderStyle::Explanatory,
        };

        let prompt = build_prompt(&ctx);
        assert!(prompt.contains("Question:"));
        assert!(prompt.contains("cancer"));
        assert!(prompt.contains("war"));
        assert!(prompt.contains("replication"));
        assert!(prompt.contains("93%"));
        assert!(prompt.contains("Response:"));
    }

    #[test]
    fn test_decoder_energy() {
        let decoder = TemplateDecoder;
        assert!(decoder.energy_per_token() < 0.000_001);
        assert_eq!(decoder.model_name(), "template");
    }
}
