//! NLP tokenization: word, sentence, and subword tokenizers with span tracking.
//!
//! Provides a configurable tokenization pipeline for natural language processing:
//! word tokenizer (whitespace + punctuation split), sentence splitter, subword
//! tokenizer (BPE-like merge rules), token span tracking, special token handling
//! ([CLS]/[SEP]/[PAD]), and vocabulary management.

use std::collections::HashMap;

// ── Token types ──────────────────────────────────────────────────

/// A single token with its source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NlpToken {
    pub text: String,
    /// Byte offset of start in the original text.
    pub start: usize,
    /// Byte offset of end (exclusive) in the original text.
    pub end: usize,
    pub kind: NlpTokenKind,
}

/// Classification of an NLP token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NlpTokenKind {
    Word,
    Punctuation,
    Number,
    Whitespace,
    Special,
    Subword,
    Unknown,
}

/// Special tokens used by transformer-style models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialToken {
    Cls,
    Sep,
    Pad,
    Unk,
    Mask,
    Bos,
    Eos,
}

impl SpecialToken {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpecialToken::Cls => "[CLS]",
            SpecialToken::Sep => "[SEP]",
            SpecialToken::Pad => "[PAD]",
            SpecialToken::Unk => "[UNK]",
            SpecialToken::Mask => "[MASK]",
            SpecialToken::Bos => "[BOS]",
            SpecialToken::Eos => "[EOS]",
        }
    }

    pub fn from_str(s: &str) -> Option<SpecialToken> {
        match s {
            "[CLS]" => Some(SpecialToken::Cls),
            "[SEP]" => Some(SpecialToken::Sep),
            "[PAD]" => Some(SpecialToken::Pad),
            "[UNK]" => Some(SpecialToken::Unk),
            "[MASK]" => Some(SpecialToken::Mask),
            "[BOS]" => Some(SpecialToken::Bos),
            "[EOS]" => Some(SpecialToken::Eos),
            _ => None,
        }
    }
}

// ── Vocabulary ───────────────────────────────────────────────────

/// Manages token-to-id mapping.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        }
    }

    /// Build a vocabulary from special tokens first, then word tokens.
    pub fn with_special_tokens(specials: &[SpecialToken]) -> Self {
        let mut vocab = Self::new();
        for s in specials {
            vocab.add(s.as_str().to_string());
        }
        vocab
    }

    /// Add a token, returning its id. No-op if already present.
    pub fn add(&mut self, token: String) -> u32 {
        if let Some(id) = self.token_to_id.get(&token) {
            return *id;
        }
        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.push(token);
        id
    }

    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Encode a token string into an id, using UNK id if not found.
    pub fn encode(&self, token: &str, unk_id: u32) -> u32 {
        self.token_to_id.get(token).copied().unwrap_or(unk_id)
    }

    /// Build vocabulary from a corpus of already-tokenized text.
    pub fn build_from_tokens(tokens: &[&str], specials: &[SpecialToken]) -> Self {
        let mut vocab = Self::with_special_tokens(specials);
        for token in tokens {
            vocab.add(token.to_string());
        }
        vocab
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

// ── Word tokenizer ───────────────────────────────────────────────

fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || matches!(c,
        '\u{2010}'..='\u{2027}' | '\u{2030}'..='\u{205E}' | '\u{2190}'..='\u{2BFF}'
    )
}

/// Tokenize text into words by splitting on whitespace and separating punctuation.
pub fn word_tokenize(text: &str) -> Vec<NlpToken> {
    let mut tokens = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < text.len() {
        // Skip and emit whitespace runs.
        if bytes[i].is_ascii_whitespace() || text[i..].starts_with(|c: char| c.is_whitespace()) {
            let start = i;
            while i < text.len() {
                let ch = text[i..].chars().next().unwrap();
                if !ch.is_whitespace() {
                    break;
                }
                i += ch.len_utf8();
            }
            tokens.push(NlpToken {
                text: text[start..i].to_string(),
                start,
                end: i,
                kind: NlpTokenKind::Whitespace,
            });
            continue;
        }

        let ch = text[i..].chars().next().unwrap();

        // Punctuation as individual tokens.
        if is_punctuation(ch) {
            tokens.push(NlpToken {
                text: ch.to_string(),
                start: i,
                end: i + ch.len_utf8(),
                kind: NlpTokenKind::Punctuation,
            });
            i += ch.len_utf8();
            continue;
        }

        // Number runs.
        if ch.is_ascii_digit() {
            let start = i;
            while i < text.len() {
                let c = text[i..].chars().next().unwrap();
                if !c.is_ascii_digit() && c != '.' {
                    break;
                }
                i += c.len_utf8();
            }
            tokens.push(NlpToken {
                text: text[start..i].to_string(),
                start,
                end: i,
                kind: NlpTokenKind::Number,
            });
            continue;
        }

        // Word: alphanumeric + apostrophes for contractions.
        let start = i;
        while i < text.len() {
            let c = text[i..].chars().next().unwrap();
            if c.is_whitespace() || (is_punctuation(c) && c != '\'') {
                break;
            }
            // Break on apostrophe only if NOT followed by an alpha (end of word).
            if c == '\'' {
                let next_pos = i + 1;
                if next_pos < text.len() {
                    let nc = text[next_pos..].chars().next().unwrap();
                    if !nc.is_alphabetic() {
                        break;
                    }
                } else {
                    break;
                }
            }
            i += c.len_utf8();
        }
        if i > start {
            tokens.push(NlpToken {
                text: text[start..i].to_string(),
                start,
                end: i,
                kind: NlpTokenKind::Word,
            });
        }
    }

    tokens
}

/// Tokenize and return only word-type tokens (no whitespace/punctuation).
pub fn word_tokenize_clean(text: &str) -> Vec<NlpToken> {
    word_tokenize(text)
        .into_iter()
        .filter(|t| matches!(t.kind, NlpTokenKind::Word | NlpTokenKind::Number))
        .collect()
}

// ── Sentence splitter ────────────────────────────────────────────

/// Abbreviations that should not end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "dept", "est", "vol", "vs", "etc", "inc", "ltd", "co", "corp",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
    "nov", "dec", "fig", "eq", "approx", "govt", "assn",
];

/// Split text into sentences.
pub fn sentence_split(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let mut current_start = 0;
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        if ch == '.' || ch == '!' || ch == '?' {
            // Collect consecutive sentence-ending punctuation (e.g. "..." or "?!")
            let punct_start = i;
            while i < len && (chars[i] == '.' || chars[i] == '!' || chars[i] == '?') {
                i += 1;
            }

            // Check if this is an abbreviation (word before period).
            if ch == '.' && (i - punct_start) == 1 {
                // Find the word before the period.
                let byte_pos = chars[..punct_start].iter().map(|c| c.len_utf8()).sum::<usize>();
                let text_before = &text[current_start..byte_pos];
                let last_word = text_before.split_whitespace().next_back().unwrap_or("");
                let clean_word = last_word.trim_start_matches(|c: char| !c.is_alphabetic());
                if ABBREVIATIONS.contains(&clean_word.to_lowercase().as_str()) {
                    continue;
                }
            }

            // Skip trailing whitespace.
            while i < len && chars[i].is_whitespace() {
                i += 1;
            }

            // Check if next char is uppercase or end of text → sentence boundary.
            let is_boundary = i >= len || chars[i].is_uppercase() || chars[i] == '"' || chars[i] == '\'';

            if is_boundary {
                let byte_end: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
                let sentence = text[current_start..byte_end].trim().to_string();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                current_start = byte_end;
                // Skip any leading whitespace of next sentence.
                while current_start < text.len() && text.as_bytes().get(current_start).map_or(false, |b| b.is_ascii_whitespace()) {
                    current_start += 1;
                }
            }
        } else {
            i += 1;
        }
    }

    // Remainder.
    if current_start < text.len() {
        let remainder = text[current_start..].trim().to_string();
        if !remainder.is_empty() {
            sentences.push(remainder);
        }
    }

    sentences
}

// ── BPE-like subword tokenizer ───────────────────────────────────

/// A merge rule for BPE: merge `left` + `right` → `merged`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub merged: String,
    pub priority: u32,
}

/// BPE-like subword tokenizer.
#[derive(Debug, Clone)]
pub struct SubwordTokenizer {
    merge_rules: Vec<MergeRule>,
    vocab: Vocabulary,
    continuing_prefix: String,
}

impl SubwordTokenizer {
    pub fn new(continuing_prefix: &str) -> Self {
        Self {
            merge_rules: Vec::new(),
            vocab: Vocabulary::with_special_tokens(&[
                SpecialToken::Unk,
                SpecialToken::Cls,
                SpecialToken::Sep,
                SpecialToken::Pad,
            ]),
            continuing_prefix: continuing_prefix.to_string(),
        }
    }

    /// Add a merge rule. Rules are applied in priority order (lower = first).
    pub fn add_merge(&mut self, left: &str, right: &str, priority: u32) {
        let merged = format!("{left}{right}");
        self.merge_rules.push(MergeRule {
            left: left.to_string(),
            right: right.to_string(),
            merged: merged.clone(),
            priority,
        });
        self.merge_rules.sort_by_key(|r| r.priority);
        self.vocab.add(merged);
    }

    /// Add a token to the vocabulary.
    pub fn add_vocab(&mut self, token: &str) -> u32 {
        self.vocab.add(token.to_string())
    }

    /// Tokenize a single word into subword pieces.
    pub fn tokenize_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return Vec::new();
        }

        // Start with individual characters.
        let mut pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // Apply merge rules iteratively.
        let mut changed = true;
        while changed {
            changed = false;
            for rule in &self.merge_rules {
                let mut i = 0;
                while i + 1 < pieces.len() {
                    if pieces[i] == rule.left && pieces[i + 1] == rule.right {
                        pieces[i] = rule.merged.clone();
                        pieces.remove(i + 1);
                        changed = true;
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Add continuing prefix to all pieces after the first.
        let prefix = &self.continuing_prefix;
        for piece in pieces.iter_mut().skip(1) {
            *piece = format!("{prefix}{piece}");
        }

        pieces
    }

    /// Tokenize a full text into subword tokens.
    pub fn tokenize(&self, text: &str) -> Vec<NlpToken> {
        let words = word_tokenize_clean(text);
        let mut result = Vec::new();

        for word_tok in &words {
            let subwords = self.tokenize_word(&word_tok.text);
            let mut offset = word_tok.start;
            for (i, sw) in subwords.iter().enumerate() {
                let raw_len = if i == 0 {
                    sw.len()
                } else {
                    sw.len().saturating_sub(self.continuing_prefix.len())
                };
                let end = (offset + raw_len).min(word_tok.end);
                result.push(NlpToken {
                    text: sw.clone(),
                    start: offset,
                    end,
                    kind: NlpTokenKind::Subword,
                });
                offset = end;
            }
        }

        result
    }

    /// Encode text into vocabulary ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let unk_id = self.vocab.get_id("[UNK]").unwrap_or(0);
        self.tokenize(text)
            .iter()
            .map(|t| self.vocab.encode(&t.text, unk_id))
            .collect()
    }

    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }
}

// ── Special token wrapping ───────────────────────────────────────

/// Wrap a token sequence with [CLS] at start and [SEP] at end.
pub fn add_cls_sep(tokens: Vec<NlpToken>) -> Vec<NlpToken> {
    let mut result = Vec::with_capacity(tokens.len() + 2);
    result.push(NlpToken {
        text: "[CLS]".to_string(),
        start: 0,
        end: 0,
        kind: NlpTokenKind::Special,
    });
    result.extend(tokens);
    let last_end = result.last().map(|t| t.end).unwrap_or(0);
    result.push(NlpToken {
        text: "[SEP]".to_string(),
        start: last_end,
        end: last_end,
        kind: NlpTokenKind::Special,
    });
    result
}

/// Pad a token sequence to the specified length with [PAD] tokens.
pub fn pad_to_length(mut tokens: Vec<NlpToken>, target_len: usize) -> Vec<NlpToken> {
    let last_end = tokens.last().map(|t| t.end).unwrap_or(0);
    while tokens.len() < target_len {
        tokens.push(NlpToken {
            text: "[PAD]".to_string(),
            start: last_end,
            end: last_end,
            kind: NlpTokenKind::Special,
        });
    }
    tokens
}

/// Truncate a token sequence to the specified maximum length.
pub fn truncate_tokens(mut tokens: Vec<NlpToken>, max_len: usize) -> Vec<NlpToken> {
    tokens.truncate(max_len);
    tokens
}

/// Prepare tokens for a model: add [CLS]/[SEP], truncate, pad.
pub fn prepare_for_model(tokens: Vec<NlpToken>, max_len: usize) -> Vec<NlpToken> {
    let wrapped = add_cls_sep(tokens);
    let truncated = truncate_tokens(wrapped, max_len);
    pad_to_length(truncated, max_len)
}

// ── Pair encoding ────────────────────────────────────────────────

/// Encode a pair of token sequences (e.g., for sentence-pair tasks).
/// Returns: [CLS] tokens_a [SEP] tokens_b [SEP]
pub fn encode_pair(tokens_a: Vec<NlpToken>, tokens_b: Vec<NlpToken>) -> Vec<NlpToken> {
    let mut result = Vec::with_capacity(tokens_a.len() + tokens_b.len() + 3);

    result.push(NlpToken {
        text: "[CLS]".to_string(),
        start: 0,
        end: 0,
        kind: NlpTokenKind::Special,
    });

    let last_a = tokens_a.last().map(|t| t.end).unwrap_or(0);
    result.extend(tokens_a);

    result.push(NlpToken {
        text: "[SEP]".to_string(),
        start: last_a,
        end: last_a,
        kind: NlpTokenKind::Special,
    });

    let last_b = tokens_b.last().map(|t| t.end).unwrap_or(0);
    result.extend(tokens_b);

    result.push(NlpToken {
        text: "[SEP]".to_string(),
        start: last_b,
        end: last_b,
        kind: NlpTokenKind::Special,
    });

    result
}

/// Generate segment ids for a pair-encoded sequence (0 for A, 1 for B).
pub fn segment_ids(tokens: &[NlpToken]) -> Vec<u8> {
    let mut ids = Vec::with_capacity(tokens.len());
    let mut segment = 0u8;
    let mut sep_count = 0;

    for tok in tokens {
        ids.push(segment);
        if tok.text == "[SEP]" {
            sep_count += 1;
            if sep_count == 1 {
                segment = 1;
            }
        }
    }

    ids
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_tokenize_basic() {
        let tokens = word_tokenize("Hello world");
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(words, ["Hello", " ", "world"]);
    }

    #[test]
    fn test_word_tokenize_punctuation() {
        let tokens = word_tokenize("Hello, world!");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, ["Hello", ",", " ", "world", "!"]);
    }

    #[test]
    fn test_word_tokenize_spans() {
        let tokens = word_tokenize("Hi there");
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 2);
        assert_eq!(tokens[0].text, "Hi");
        // whitespace
        assert_eq!(tokens[1].start, 2);
        assert_eq!(tokens[1].end, 3);
        // "there"
        assert_eq!(tokens[2].start, 3);
        assert_eq!(tokens[2].end, 8);
    }

    #[test]
    fn test_word_tokenize_numbers() {
        let tokens = word_tokenize("Test 42 end");
        let nums: Vec<&NlpToken> = tokens.iter().filter(|t| t.kind == NlpTokenKind::Number).collect();
        assert_eq!(nums.len(), 1);
        assert_eq!(nums[0].text, "42");
    }

    #[test]
    fn test_word_tokenize_contractions() {
        let tokens = word_tokenize("don't can't");
        let words: Vec<&str> = tokens.iter()
            .filter(|t| t.kind == NlpTokenKind::Word)
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(words, ["don't", "can't"]);
    }

    #[test]
    fn test_word_tokenize_clean() {
        let tokens = word_tokenize_clean("Hello, world! 42");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, ["Hello", "world", "42"]);
    }

    #[test]
    fn test_sentence_split_basic() {
        let sents = sentence_split("Hello world. How are you? I am fine!");
        assert_eq!(sents.len(), 3);
        assert_eq!(sents[0], "Hello world.");
        assert_eq!(sents[1], "How are you?");
        assert_eq!(sents[2], "I am fine!");
    }

    #[test]
    fn test_sentence_split_abbreviation() {
        let sents = sentence_split("Dr. Smith went home. He was tired.");
        assert_eq!(sents.len(), 2);
        assert!(sents[0].contains("Dr."));
    }

    #[test]
    fn test_sentence_split_empty() {
        let sents = sentence_split("");
        assert!(sents.is_empty());
    }

    #[test]
    fn test_sentence_split_single() {
        let sents = sentence_split("No period here");
        assert_eq!(sents.len(), 1);
        assert_eq!(sents[0], "No period here");
    }

    #[test]
    fn test_vocabulary_basic() {
        let mut vocab = Vocabulary::new();
        let id0 = vocab.add("hello".to_string());
        let id1 = vocab.add("world".to_string());
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(vocab.get_id("hello"), Some(0));
        assert_eq!(vocab.get_token(1), Some("world"));
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_vocabulary_dedup() {
        let mut vocab = Vocabulary::new();
        let id0 = vocab.add("hello".to_string());
        let id1 = vocab.add("hello".to_string());
        assert_eq!(id0, id1);
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_vocabulary_with_specials() {
        let vocab = Vocabulary::with_special_tokens(&[SpecialToken::Cls, SpecialToken::Sep]);
        assert_eq!(vocab.get_id("[CLS]"), Some(0));
        assert_eq!(vocab.get_id("[SEP]"), Some(1));
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_vocabulary_encode_unk() {
        let vocab = Vocabulary::with_special_tokens(&[SpecialToken::Unk]);
        let unk_id = vocab.get_id("[UNK]").unwrap();
        assert_eq!(vocab.encode("missing", unk_id), unk_id);
    }

    #[test]
    fn test_subword_tokenizer() {
        let mut sw = SubwordTokenizer::new("##");
        sw.add_vocab("h");
        sw.add_vocab("e");
        sw.add_vocab("l");
        sw.add_vocab("o");
        sw.add_merge("h", "e", 1);
        sw.add_merge("l", "o", 2);

        let pieces = sw.tokenize_word("hello");
        // h+e→he, then l+l can't merge, l+o→lo
        // chars: h,e,l,l,o → he,l,l,o → he,l,lo
        assert_eq!(pieces, ["he", "##l", "##lo"]);
    }

    #[test]
    fn test_subword_tokenizer_full_text() {
        let mut sw = SubwordTokenizer::new("##");
        sw.add_vocab("h");
        sw.add_vocab("i");
        sw.add_merge("h", "i", 1);
        let tokens = sw.tokenize("hi hi");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hi");
        assert_eq!(tokens[1].text, "hi");
    }

    #[test]
    fn test_add_cls_sep() {
        let tokens = word_tokenize_clean("test");
        let wrapped = add_cls_sep(tokens);
        assert_eq!(wrapped[0].text, "[CLS]");
        assert_eq!(wrapped[1].text, "test");
        assert_eq!(wrapped[2].text, "[SEP]");
    }

    #[test]
    fn test_pad_to_length() {
        let tokens = word_tokenize_clean("a b");
        let padded = pad_to_length(tokens, 5);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[2].text, "[PAD]");
        assert_eq!(padded[2].kind, NlpTokenKind::Special);
    }

    #[test]
    fn test_prepare_for_model() {
        let tokens = word_tokenize_clean("hello world");
        // CLS hello world SEP → len 4, pad to 6
        let prepared = prepare_for_model(tokens, 6);
        assert_eq!(prepared.len(), 6);
        assert_eq!(prepared[0].text, "[CLS]");
        assert_eq!(prepared[3].text, "[SEP]");
        assert_eq!(prepared[4].text, "[PAD]");
    }

    #[test]
    fn test_encode_pair() {
        let a = word_tokenize_clean("hello");
        let b = word_tokenize_clean("world");
        let encoded = encode_pair(a, b);
        let texts: Vec<&str> = encoded.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, ["[CLS]", "hello", "[SEP]", "world", "[SEP]"]);
    }

    #[test]
    fn test_segment_ids() {
        let a = word_tokenize_clean("hello");
        let b = word_tokenize_clean("world");
        let encoded = encode_pair(a, b);
        let seg = segment_ids(&encoded);
        // [CLS]=0, hello=0, [SEP]=0, world=1, [SEP]=1
        assert_eq!(seg, [0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_special_token_roundtrip() {
        for st in &[SpecialToken::Cls, SpecialToken::Sep, SpecialToken::Pad, SpecialToken::Unk, SpecialToken::Mask] {
            let s = st.as_str();
            let parsed = SpecialToken::from_str(s).unwrap();
            assert_eq!(*st, parsed);
        }
    }

    #[test]
    fn test_truncate_tokens() {
        let tokens = word_tokenize_clean("one two three four five");
        let truncated = truncate_tokens(tokens, 3);
        assert_eq!(truncated.len(), 3);
    }

    #[test]
    fn test_build_vocab_from_tokens() {
        let vocab = Vocabulary::build_from_tokens(
            &["hello", "world", "hello"],
            &[SpecialToken::Unk],
        );
        assert_eq!(vocab.len(), 3); // [UNK], hello, world
        assert_eq!(vocab.get_id("hello"), Some(1));
    }
}
