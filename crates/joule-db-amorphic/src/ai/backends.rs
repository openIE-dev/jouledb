//! Deterministic tier backends.
//!
//! Tiers 2–4 are implemented on the holographic encoder, the materializer
//! cascade, and SigQL statistics. Paths that need model weights return
//! [`AiError::ModelNotLoaded`] instead of a fabricated completion.

use std::sync::Mutex;
use std::time::Instant;

use super::holo::HoloEngine;
use super::receipt::{AiReceipt, EnergyProvenance};
use super::tier::InferenceTier;
use super::traits::{
    AiError, AiMessage, AiTool, EmbeddedInference, Entity, FrontierInference,
    HolographicInference, LocalInference, ReasoningResult, Sentiment, SentimentLabel,
};
use crate::knowledge::{Materializer, Source};
use joule_db_hdc::BinaryHV;

const POSITIVE: &[&str] = &[
    "good", "great", "excellent", "happy", "love", "wonderful", "best", "amazing", "pleasant",
    "joy",
];
const NEGATIVE: &[&str] = &[
    "bad", "terrible", "awful", "hate", "worst", "horrible", "poor", "sad", "angry",
    "disappointing",
];

fn missing_weights(tier: &str, prompt: &str, probe: &BinaryHV, source: &str) -> AiError {
    let mix = probe.as_words().iter().fold(0u64, |acc, w| acc ^ *w);
    AiError::ModelNotLoaded(format!(
        "{tier} weights are not linked (no local/frontier model in this build); \
         holographic probe dim={} mix={mix:#x}; cascade={source}; prompt_len={}",
        probe.dimension(),
        prompt.len()
    ))
}

fn receipt(tier: InferenceTier, model: &str, joules: f64, started: Instant) -> AiReceipt {
    AiReceipt {
        tier,
        model: model.to_string(),
        energy_joules: joules,
        provenance: EnergyProvenance::Calculated,
        latency_us: started.elapsed().as_micros() as u64,
        tokens: None,
    }
}

fn truncate_tokens(text: &str, max_tokens: u32) -> String {
    if max_tokens == 0 {
        return String::new();
    }
    text.split_whitespace()
        .take(max_tokens as usize)
        .collect::<Vec<_>>()
        .join(" ")
}

fn cascade_hit(materializer: &mut Materializer, prompt: &str) -> Option<(String, f64, Source)> {
    let resolved = materializer.materialize(prompt);
    if resolved.source == Source::Neural || !resolved.verified || resolved.output.is_empty() {
        return None;
    }
    Some((resolved.output, resolved.energy_joules, resolved.source))
}

// ============================================================================
// Tier 2 — embedded HDC (no ONNX weights in tree)
// ============================================================================

/// On-device tier backed by holographic encoding, not an ONNX runtime.
pub struct HdcEmbedded {
    holo: HoloEngine,
}

impl HdcEmbedded {
    pub fn new() -> Self {
        Self {
            holo: HoloEngine::new(),
        }
    }

    fn polarity_signal(&self, text: &str) -> (f32, f64) {
        let mut series = Vec::new();
        for token in text.split(|c: char| !c.is_alphanumeric()) {
            let word = token.to_lowercase();
            if word.is_empty() {
                continue;
            }
            let sample = if POSITIVE.contains(&word.as_str()) {
                1.0
            } else if NEGATIVE.contains(&word.as_str()) {
                -1.0
            } else {
                0.0
            };
            series.push(sample);
        }
        let lexical = if series.is_empty() {
            0.0
        } else {
            sigql::dsp::statistics::compute_mean(&series)
                .map(|v| v.value)
                .unwrap_or(0.0)
        };
        let text_hv = self.holo.encode_text(text);
        let pos = self.holo.encode_text(&POSITIVE.join(" "));
        let neg = self.holo.encode_text(&NEGATIVE.join(" "));
        let holo = text_hv.similarity(&pos) - text_hv.similarity(&neg);
        let score = (0.5 * lexical as f32 + 0.5 * holo).clamp(-1.0, 1.0);
        (score, lexical)
    }
}

impl Default for HdcEmbedded {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddedInference for HdcEmbedded {
    fn embed(&self, text: &str) -> Result<(BinaryHV, AiReceipt), AiError> {
        let started = Instant::now();
        let hv = self.holo.encode_text(text);
        Ok((
            hv,
            receipt(
                InferenceTier::Embedded,
                "hdc-trigram-embed",
                0.000_002,
                started,
            ),
        ))
    }

    fn classify(
        &self,
        text: &str,
        categories: &[String],
    ) -> Result<(String, f32, AiReceipt), AiError> {
        if categories.is_empty() {
            return Err(AiError::OperationFailed(
                "classify requires at least one category".into(),
            ));
        }
        let started = Instant::now();
        let input = self.holo.encode_text(text);
        let protos: Vec<(String, BinaryHV)> = categories
            .iter()
            .map(|c| (c.clone(), self.holo.encode_text(c)))
            .collect();
        let (label, confidence) = self.holo.classify_holo(&input, &protos);
        Ok((
            label,
            confidence,
            receipt(
                InferenceTier::Embedded,
                "hdc-prototype-classify",
                0.000_000_2 * categories.len() as f64,
                started,
            ),
        ))
    }

    fn extract_entities(&self, text: &str) -> Result<(Vec<Entity>, AiReceipt), AiError> {
        let started = Instant::now();
        let mut entities = Vec::new();
        let raw_tokens = tokenize_spans(text);
        let person = self.holo.encode_text("person human individual named");
        let org = self.holo.encode_text("organization company corporation works at");
        let place = self.holo.encode_text("place city country location in from");

        for (idx, (token, start, end)) in raw_tokens.iter().enumerate() {
            if let Ok(number) = token.parse::<f64>() {
                if token.chars().any(|c| c.is_ascii_digit()) {
                    let _ = number;
                    entities.push(Entity {
                        text: token.clone(),
                        entity_type: "number".into(),
                        start: *start,
                        end: *end,
                        confidence: 1.0,
                    });
                    continue;
                }
            }
            let chars: Vec<char> = token.chars().collect();
            let capitalized = chars.first().is_some_and(|c| c.is_uppercase())
                && chars.len() > 1
                && chars.iter().any(|c| c.is_alphabetic());
            let acronym = token.len() >= 2
                && token.len() <= 6
                && token.chars().all(|c| c.is_ascii_uppercase());
            if !capitalized && !acronym {
                continue;
            }
            let prev = idx
                .checked_sub(1)
                .and_then(|i| raw_tokens.get(i))
                .map(|(t, _, _)| t.to_lowercase());
            let (entity_type, proto) = match prev.as_deref() {
                Some("at") | Some("for") => ("organization", &org),
                Some("in") | Some("from") | Some("to") => ("place", &place),
                _ if acronym => ("organization", &org),
                _ => ("person", &person),
            };
            let ctx = format!("{} {}", prev.unwrap_or_default(), token);
            let confidence = self.holo.encode_text(&ctx).similarity(proto).clamp(0.0, 1.0);
            entities.push(Entity {
                text: token.clone(),
                entity_type: entity_type.to_string(),
                start: *start,
                end: *end,
                confidence,
            });
        }
        Ok((
            entities,
            receipt(
                InferenceTier::Embedded,
                "hdc-span-ner",
                0.000_001,
                started,
            ),
        ))
    }

    fn sentiment(&self, text: &str) -> Result<(Sentiment, AiReceipt), AiError> {
        if text.trim().is_empty() {
            return Err(AiError::OperationFailed(
                "sentiment requires non-empty text".into(),
            ));
        }
        let started = Instant::now();
        let (score, _) = self.polarity_signal(text);
        let label = if score > 0.02 {
            SentimentLabel::Positive
        } else if score < -0.02 {
            SentimentLabel::Negative
        } else {
            SentimentLabel::Neutral
        };
        Ok((
            Sentiment { score, label },
            receipt(
                InferenceTier::Embedded,
                "hdc-polarity+sigql-mean",
                0.000_002,
                started,
            ),
        ))
    }

    fn detect_language(&self, text: &str) -> Result<(String, AiReceipt), AiError> {
        if text.trim().is_empty() {
            return Err(AiError::OperationFailed(
                "detect_language requires non-empty text".into(),
            ));
        }
        let started = Instant::now();
        let input = self.holo.encode_text(text);
        let langs = [
            ("en", "the and of to a in is that for"),
            ("es", "el la de que y en los las por"),
            ("fr", "le la les de et un des que dans"),
            ("de", "der die das und ein nicht mit von"),
        ];
        let mut best = ("und", 0.0f32);
        for (code, proto) in langs {
            let score = input.similarity(&self.holo.encode_text(proto));
            if score > best.1 {
                best = (code, score);
            }
        }
        Ok((
            best.0.to_string(),
            receipt(
                InferenceTier::Embedded,
                "hdc-function-word-lang",
                0.000_001,
                started,
            ),
        ))
    }
}

fn tokenize_spans(text: &str) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() || is_sep(bytes[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() && !is_sep(bytes[i]) {
            i += 1;
        }
        let token = text[start..i].trim_matches(|c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\''));
        if !token.is_empty() {
            let rel = text[start..i].find(token).unwrap_or(0);
            let s = start + rel;
            out.push((token.to_string(), s, s + token.len()));
        }
    }
    out
}

fn is_sep(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'[' | b']' | b'{' | b'}')
}

// ============================================================================
// Tier 3 — local cascade
// ============================================================================

/// Local tier: materializer cascade for closed forms, HDC extractive summary,
/// and schema-grounded NL→SQL. Open generation returns [`AiError::ModelNotLoaded`].
pub struct CascadeLocal {
    materializer: Mutex<Materializer>,
    holo: HoloEngine,
}

impl CascadeLocal {
    pub fn new() -> Self {
        Self {
            materializer: Mutex::new(Materializer::new()),
            holo: HoloEngine::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Materializer> {
        self.materializer
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl Default for CascadeLocal {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalInference for CascadeLocal {
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<(String, AiReceipt), AiError> {
        let started = Instant::now();
        let mut mat = self.lock();
        if let Some((text, joules, source)) = cascade_hit(&mut mat, prompt) {
            return Ok((
                truncate_tokens(&text, max_tokens),
                receipt(
                    InferenceTier::Local,
                    &format!("cascade:{source:?}"),
                    joules,
                    started,
                ),
            ));
        }
        let probe = self.holo.encode_text(prompt);
        Err(missing_weights("local-generate", prompt, &probe, "neural"))
    }

    fn nl_to_query(&self, nl: &str, schema_hint: &str) -> Result<(String, AiReceipt), AiError> {
        let started = Instant::now();
        let tables = parse_schema_hint(schema_hint);
        if tables.is_empty() {
            return Err(AiError::OperationFailed(
                "schema_hint did not name any tables".into(),
            ));
        }
        let nl_l = nl.to_lowercase();
        let chosen = tables
            .iter()
            .find(|t| nl_l.split(|c: char| !c.is_alphanumeric()).any(|w| w == t.name))
            .cloned()
            .unwrap_or_else(|| {
                let q = self.holo.encode_text(nl);
                tables
                    .iter()
                    .max_by(|a, b| {
                        let sa = q.similarity(&self.holo.encode_text(&a.signature()));
                        let sb = q.similarity(&self.holo.encode_text(&b.signature()));
                        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned()
                    .expect("tables non-empty")
            });
        let mentioned: Vec<&str> = chosen
            .columns
            .iter()
            .map(|c| c.as_str())
            .filter(|c| nl_l.split(|ch: char| !ch.is_alphanumeric()).any(|w| w == *c))
            .collect();
        let projection = if mentioned.is_empty() {
            "*".to_string()
        } else {
            mentioned.join(", ")
        };
        let mut sql = format!("SELECT {projection} FROM {}", chosen.name);
        if let Some((col, value)) = predicate_from_nl(nl, &chosen.columns) {
            let escaped = value.replace('\'', "''");
            sql.push_str(&format!(" WHERE {col} = '{escaped}'"));
        }
        Ok((
            sql,
            receipt(
                InferenceTier::Local,
                "hdc-schema-nl2sql",
                0.000_01,
                started,
            ),
        ))
    }

    fn summarize(&self, text: &str, max_words: u32) -> Result<(String, AiReceipt), AiError> {
        if text.trim().is_empty() {
            return Err(AiError::OperationFailed(
                "summarize requires non-empty text".into(),
            ));
        }
        let started = Instant::now();
        let doc = self.holo.encode_text(text);
        let sentences = split_sentences(text);
        let mut ranked: Vec<(usize, f32)> = sentences
            .iter()
            .enumerate()
            .map(|(i, s)| (i, doc.similarity(&self.holo.encode_text(s))))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut words_left = max_words as usize;
        let mut picked: Vec<usize> = Vec::new();
        if max_words > 0 {
            for (idx, _) in ranked {
                if words_left == 0 {
                    break;
                }
                let wc = sentences[idx].split_whitespace().count();
                if wc == 0 {
                    continue;
                }
                picked.push(idx);
                words_left = words_left.saturating_sub(wc);
            }
        }
        picked.sort_unstable();
        let summary = truncate_tokens(
            &picked
                .iter()
                .map(|i| sentences[*i].as_str())
                .collect::<Vec<_>>()
                .join(" "),
            max_words,
        );
        Ok((
            summary,
            receipt(
                InferenceTier::Local,
                "hdc-extractive-summary",
                0.000_002 * sentences.len() as f64,
                started,
            ),
        ))
    }
}

#[derive(Clone)]
struct TableHint {
    name: String,
    columns: Vec<String>,
}

impl TableHint {
    fn signature(&self) -> String {
        format!("{} {}", self.name, self.columns.join(" "))
    }
}

fn parse_schema_hint(hint: &str) -> Vec<TableHint> {
    let mut tables = Vec::new();
    let lower_src = hint;
    let bytes = lower_src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() || matches!(bytes[i], b',' | b';') {
            i += 1;
            continue;
        }
        // Skip CREATE / TABLE keywords.
        if hint[i..].len() >= 6 && hint[i..i + 6].eq_ignore_ascii_case("create") {
            i += 6;
            continue;
        }
        if hint[i..].len() >= 5 && hint[i..i + 5].eq_ignore_ascii_case("table") {
            i += 5;
            continue;
        }
        if !bytes[i].is_ascii_alphabetic() && bytes[i] != b'_' {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        let name = hint[start..i].to_string();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let mut columns = Vec::new();
        if i < bytes.len() && bytes[i] == b'(' {
            i += 1;
            let col_start = i;
            while i < bytes.len() && bytes[i] != b')' {
                i += 1;
            }
            let inner = &hint[col_start..i];
            for part in inner.split(',') {
                if let Some(col) = part.split_whitespace().next() {
                    let col = col.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
                    if !col.is_empty() {
                        columns.push(col.to_string());
                    }
                }
            }
            if i < bytes.len() && bytes[i] == b')' {
                i += 1;
            }
        }
        if !name.is_empty() {
            tables.push(TableHint { name, columns });
        }
    }
    tables
}

fn predicate_from_nl(nl: &str, columns: &[String]) -> Option<(String, String)> {
    let tokens: Vec<&str> = nl.split_whitespace().collect();
    for i in 0..tokens.len() {
        let bare = tokens[i].trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
        let Some(col) = columns.iter().find(|c| c.eq_ignore_ascii_case(bare)) else {
            continue;
        };
        let op = tokens.get(i + 1).copied().unwrap_or("");
        if op.eq_ignore_ascii_case("is")
            || op == "="
            || op.eq_ignore_ascii_case("equals")
            || op == "=="
        {
            let value = tokens.get(i + 2)?.trim_matches(|c: char| {
                matches!(c, ',' | '.' | ';' | '?' | '!' | '"' | '\'')
            });
            if !value.is_empty() {
                return Some((col.clone(), value.to_string()));
            }
        }
    }
    None
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = cur.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            cur.clear();
        }
    }
    let tail = cur.trim();
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }
    if sentences.is_empty() {
        sentences.push(text.trim().to_string());
    }
    sentences
}

// ============================================================================
// Tier 4 — frontier without provider weights
// ============================================================================

/// Frontier-shaped API. Deterministic cascade and tool-name binding run
/// locally; anything that needs a provider model returns [`AiError::ModelNotLoaded`].
pub struct CascadeFrontier {
    materializer: Mutex<Materializer>,
    holo: HoloEngine,
}

impl CascadeFrontier {
    pub fn new() -> Self {
        Self {
            materializer: Mutex::new(Materializer::new()),
            holo: HoloEngine::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Materializer> {
        self.materializer
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn last_user<'a>(&self, messages: &'a [AiMessage]) -> &'a str {
        messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("")
    }
}

impl Default for CascadeFrontier {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontierInference for CascadeFrontier {
    fn complete(
        &self,
        messages: Vec<AiMessage>,
        tools: &[AiTool],
    ) -> Result<(String, AiReceipt), AiError> {
        let started = Instant::now();
        let user = self.last_user(&messages).to_string();
        if user.is_empty() && messages.is_empty() {
            return Err(AiError::OperationFailed(
                "complete requires at least one message".into(),
            ));
        }
        let prompt = if user.is_empty() {
            messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default()
        } else {
            user
        };
        {
            let mut mat = self.lock();
            if let Some((text, joules, source)) = cascade_hit(&mut mat, &prompt) {
                return Ok((
                    text,
                    receipt(
                        InferenceTier::Frontier,
                        &format!("cascade:{source:?}"),
                        joules,
                        started,
                    ),
                ));
            }
        }
        if let Some(bound) = bind_named_tool(&prompt, tools) {
            return Ok((
                bound,
                receipt(
                    InferenceTier::Frontier,
                    "hdc-tool-name-bind",
                    0.000_01,
                    started,
                ),
            ));
        }
        let probe = self.holo.encode_text(&prompt);
        Err(missing_weights(
            "frontier-complete",
            &prompt,
            &probe,
            "neural",
        ))
    }

    fn reason(
        &self,
        question: &str,
        context: &str,
    ) -> Result<(ReasoningResult, AiReceipt), AiError> {
        let started = Instant::now();
        let combined = if context.is_empty() {
            question.to_string()
        } else {
            format!("{question}\n{context}")
        };
        let mut mat = self.lock();
        if let Some((answer, joules, source)) = cascade_hit(&mut mat, &combined) {
            let q_hv = self.holo.encode_text(question);
            let c_hv = self.holo.encode_text(if context.is_empty() { question } else { context });
            let agreement = q_hv.similarity(&c_hv);
            return Ok((
                ReasoningResult {
                    answer,
                    reasoning_steps: vec![
                        format!("cascade source {source:?}"),
                        format!("holographic question/context agreement {agreement:.4}"),
                    ],
                    confidence: agreement.clamp(0.0, 1.0),
                    sources_used: vec![format!("materializer:{source:?}")],
                },
                receipt(
                    InferenceTier::Frontier,
                    &format!("cascade-reason:{source:?}"),
                    joules,
                    started,
                ),
            ));
        }
        let probe = self.holo.encode_text(&combined);
        Err(missing_weights(
            "frontier-reason",
            question,
            &probe,
            "neural",
        ))
    }

    fn embed_api(&self, text: &str) -> Result<(Vec<f32>, AiReceipt), AiError> {
        // Dense provider embeddings are not linked. Project the holographic
        // code to a bipolar f32 vector (bit 1 → +1, bit 0 → −1). This is the
        // HDC path, not a cloud embedding.
        let started = Instant::now();
        let hv = self.holo.encode_text(text);
        let mut out = Vec::with_capacity(hv.dimension());
        for bit in 0..hv.dimension() {
            let word = bit / 64;
            let offset = bit % 64;
            let on = hv
                .as_words()
                .get(word)
                .map(|w| (w >> offset) & 1 == 1)
                .unwrap_or(false);
            out.push(if on { 1.0 } else { -1.0 });
        }
        Ok((
            out,
            receipt(
                InferenceTier::Frontier,
                "hdc-bit-projection (frontier API weights not linked)",
                0.000_002,
                started,
            ),
        ))
    }
}

/// Bind a tool only when its name appears in the prompt. Required string
/// parameters are filled from the text after the name; missing required
/// params refuse the bind (no invented arguments).
fn bind_named_tool(prompt: &str, tools: &[AiTool]) -> Option<String> {
    let lower = prompt.to_lowercase();
    for tool in tools {
        let name_l = tool.name.to_lowercase();
        let Some(pos) = lower.find(&name_l) else {
            continue;
        };
        let after = prompt[pos + tool.name.len()..].trim();
        let required = required_params(&tool.parameters);
        let mut args = serde_json::Map::new();
        if required.is_empty() {
            // no required fields
        } else if required.len() == 1 {
            if after.is_empty() {
                return None;
            }
            args.insert(required[0].clone(), serde_json::Value::String(after.to_string()));
        } else {
            // Multiple required params cannot be recovered without a schema
            // parser that knows types and a real model. Refuse.
            return None;
        }
        let payload = serde_json::json!({
            "tool": tool.name,
            "arguments": args,
        });
        return Some(payload.to_string());
    }
    None
}

fn required_params(schema: &serde_json::Value) -> Vec<String> {
    schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::traits::{AiOutput, EmbeddedInference, FrontierInference, LocalInference};

    #[test]
    fn embedded_classify_and_sentiment() {
        let emb = HdcEmbedded::new();
        let (label, conf, _) = emb
            .classify(
                "basketball game was exciting",
                &["sports football soccer basketball".into(), "cooking recipe kitchen".into()],
            )
            .unwrap();
        assert_eq!(label, "sports football soccer basketball");
        assert!(conf > 0.5, "conf {conf}");

        let (pos, _) = emb.sentiment("I love this excellent wonderful day").unwrap();
        let (neg, _) = emb.sentiment("this is terrible awful horrible").unwrap();
        assert!(
            pos.score > neg.score,
            "pos {} vs neg {}",
            pos.score,
            neg.score
        );
        assert_eq!(pos.label, SentimentLabel::Positive);
        assert_eq!(neg.label, SentimentLabel::Negative);
    }

    #[test]
    fn embedded_entities_are_input_spans() {
        let emb = HdcEmbedded::new();
        let text = "Ada Lovelace works at NASA in Paris";
        let (ents, _) = emb.extract_entities(text).unwrap();
        assert!(ents.iter().any(|e| e.text == "NASA" && e.entity_type == "organization"));
        assert!(ents.iter().any(|e| e.text == "Paris" && e.entity_type == "place"));
        assert!(ents.iter().all(|e| text[e.start..e.end] == e.text));
    }

    #[test]
    fn local_arithmetic_and_missing_weights() {
        let local = CascadeLocal::new();
        let (text, _) = local.generate("2 + 3", 16).unwrap();
        assert_eq!(text, "5");
        let err = local.generate("Write me a sonnet about entropy", 32).unwrap_err();
        match err {
            AiError::ModelNotLoaded(msg) => {
                assert!(msg.contains("local-generate"), "{msg}");
                assert!(msg.contains("dim="), "{msg}");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn local_nl_to_query_uses_schema() {
        let local = CascadeLocal::new();
        let (sql, _) = local
            .nl_to_query(
                "films where genre is scifi",
                "films(id int, title text, genre text)",
            )
            .unwrap();
        assert!(sql.starts_with("SELECT"), "{sql}");
        assert!(sql.contains("FROM films"), "{sql}");
        assert!(sql.contains("genre = 'scifi'"), "{sql}");
    }

    #[test]
    fn local_summary_is_extractive() {
        let local = CascadeLocal::new();
        let text = "Alpha particles were observed. Baking bread takes time. Alpha particles scatter off gold foil.";
        let (summary, _) = local.summarize(text, 12).unwrap();
        assert!(!summary.is_empty());
        for word in summary.split_whitespace() {
            let bare = word.trim_matches(|c: char| !c.is_alphanumeric());
            assert!(
                text.to_lowercase().contains(&bare.to_lowercase()),
                "invented word {bare} in {summary}"
            );
        }
    }

    #[test]
    fn frontier_projects_bits_and_refuses_open_generation() {
        let frontier = CascadeFrontier::new();
        let (a, _) = frontier.embed_api("action movie").unwrap();
        let (b, _) = frontier.embed_api("action film").unwrap();
        let (c, _) = frontier.embed_api("chocolate cake recipe").unwrap();
        assert_eq!(a.len(), b.len());
        assert!(a.len() > 64);
        let ab = cosine(&a, &b);
        let ac = cosine(&a, &c);
        assert!(ab > ac, "related {ab} should beat unrelated {ac}");

        let err = frontier
            .complete(
                vec![AiMessage {
                    role: "user".into(),
                    content: "Write me a sonnet about entropy".into(),
                }],
                &[],
            )
            .unwrap_err();
        assert!(matches!(err, AiError::ModelNotLoaded(_)));

        let (done, _) = frontier
            .complete(
                vec![AiMessage {
                    role: "user".into(),
                    content: "2 + 3".into(),
                }],
                &[],
            )
            .unwrap();
        assert_eq!(done, "5");
    }

    #[test]
    fn frontier_binds_named_tool_only() {
        let frontier = CascadeFrontier::new();
        let tools = vec![AiTool {
            name: "db.query".into(),
            description: "run sql".into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["sql"]
            }),
        }];
        let (bound, _) = frontier
            .complete(
                vec![AiMessage {
                    role: "user".into(),
                    content: "db.query SELECT 1".into(),
                }],
                &tools,
            )
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&bound).unwrap();
        assert_eq!(v["tool"], "db.query");
        assert_eq!(v["arguments"]["sql"], "SELECT 1");
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[allow(dead_code)]
    fn _ai_output_mention() -> AiOutput {
        AiOutput::Text(String::new())
    }
}
