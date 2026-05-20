//! Text tokenization pipeline.
//!
//! Whitespace / unicode tokenizer, token filters (lowercase, stemming, stop
//! words, synonyms, n-gram, edge n-gram), analyzer chains, and character
//! filters.

use std::collections::{HashMap, HashSet};

// ── Token ───────────────────────────────────────────────────────

/// A single token produced by a tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The token text.
    pub text: String,
    /// Start byte offset in the original text.
    pub start: usize,
    /// End byte offset in the original text.
    pub end: usize,
    /// Position index (0-based term position).
    pub position: usize,
}

// ── Character Filters ───────────────────────────────────────────

/// Character-level filter applied before tokenization.
#[derive(Debug, Clone)]
pub enum CharFilter {
    /// Map individual characters (e.g., ligature expansion).
    Mapping(HashMap<char, String>),
    /// Strip characters matching a predicate description.
    /// The bool selects: true = keep alphanumeric + whitespace, false = keep all.
    StripNonAlphanumeric,
    /// Normalize unicode (NFC-like: decompose then compose basic accents).
    AsciiNormalize,
    /// HTML tag removal (simple angle-bracket stripping).
    HtmlStrip,
}

/// Apply a character filter to input text.
pub fn apply_char_filter(text: &str, filter: &CharFilter) -> String {
    match filter {
        CharFilter::Mapping(map) => {
            let mut result = String::with_capacity(text.len());
            for ch in text.chars() {
                if let Some(replacement) = map.get(&ch) {
                    result.push_str(replacement);
                } else {
                    result.push(ch);
                }
            }
            result
        }
        CharFilter::StripNonAlphanumeric => text
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect(),
        CharFilter::AsciiNormalize => {
            text.chars()
                .map(|c| match c {
                    '\u{00C0}'..='\u{00C5}' => 'A',
                    '\u{00E0}'..='\u{00E5}' => 'a',
                    '\u{00C8}'..='\u{00CB}' => 'E',
                    '\u{00E8}'..='\u{00EB}' => 'e',
                    '\u{00CC}'..='\u{00CF}' => 'I',
                    '\u{00EC}'..='\u{00EF}' => 'i',
                    '\u{00D2}'..='\u{00D6}' => 'O',
                    '\u{00F2}'..='\u{00F6}' => 'o',
                    '\u{00D9}'..='\u{00DC}' => 'U',
                    '\u{00F9}'..='\u{00FC}' => 'u',
                    '\u{00D1}' => 'N',
                    '\u{00F1}' => 'n',
                    '\u{00C7}' => 'C',
                    '\u{00E7}' => 'c',
                    other => other,
                })
                .collect()
        }
        CharFilter::HtmlStrip => {
            let mut result = String::with_capacity(text.len());
            let mut in_tag = false;
            for ch in text.chars() {
                if ch == '<' {
                    in_tag = true;
                } else if ch == '>' {
                    in_tag = false;
                } else if !in_tag {
                    result.push(ch);
                }
            }
            result
        }
    }
}

// ── Tokenizers ──────────────────────────────────────────────────

/// Tokenize on whitespace boundaries.
pub fn whitespace_tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut position = 0;

    let bytes = text.as_bytes();
    let mut i = 0;
    let len = bytes.len();

    while i < len {
        // Skip whitespace.
        while i < len && text[i..].starts_with(|c: char| c.is_whitespace()) {
            i += text[i..].chars().next().map_or(1, |c| c.len_utf8());
        }
        if i >= len {
            break;
        }
        let start = i;
        // Consume non-whitespace.
        while i < len && !text[i..].starts_with(|c: char| c.is_whitespace()) {
            i += text[i..].chars().next().map_or(1, |c| c.len_utf8());
        }
        if i > start {
            tokens.push(Token {
                text: text[start..i].to_string(),
                start,
                end: i,
                position,
            });
            position += 1;
        }
    }
    tokens
}

/// Tokenize on unicode word boundaries: split on non-alphanumeric characters.
pub fn unicode_tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut position = 0;
    let mut start_byte = 0;
    let mut current = String::new();
    let mut byte_offset = 0;

    for ch in text.chars() {
        let ch_len = ch.len_utf8();
        if ch.is_alphanumeric() || ch == '_' {
            if current.is_empty() {
                start_byte = byte_offset;
            }
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(Token {
                text: std::mem::take(&mut current),
                start: start_byte,
                end: byte_offset,
                position,
            });
            position += 1;
        }
        byte_offset += ch_len;
    }

    if !current.is_empty() {
        tokens.push(Token {
            text: current,
            start: start_byte,
            end: byte_offset,
            position,
        });
    }

    tokens
}

// ── Token Filters ───────────────────────────────────────────────

/// Lowercase all tokens.
pub fn lowercase_filter(tokens: Vec<Token>) -> Vec<Token> {
    tokens
        .into_iter()
        .map(|mut t| {
            t.text = t.text.to_lowercase();
            t
        })
        .collect()
}

/// Remove stop words.
pub fn stop_word_filter(tokens: Vec<Token>, stop_words: &HashSet<String>) -> Vec<Token> {
    tokens
        .into_iter()
        .filter(|t| !stop_words.contains(&t.text.to_lowercase()))
        .collect()
}

/// Build a default English stop-word set.
pub fn english_stop_words() -> HashSet<String> {
    [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "as", "be", "was", "were",
        "been", "are", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "this",
        "that", "these", "those", "not", "no", "nor", "so", "if", "then",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Simple Porter-like stemmer (suffix stripping for common English suffixes).
pub fn stem(word: &str) -> String {
    let w = word.to_lowercase();

    // Order matters: try longer suffixes first.
    let suffixes = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("izing", "ize"),
        ("ating", "ate"),
        ("ities", "ity"),
        ("ingly", ""),
        ("edly", ""),
        ("ness", ""),
        ("ment", ""),
        ("able", ""),
        ("ible", ""),
        ("tion", "t"),
        ("sion", "s"),
        ("ally", "al"),
        ("ful", ""),
        ("ing", ""),
        ("ous", ""),
        ("ive", ""),
        ("ant", ""),
        ("ent", ""),
        ("ism", ""),
        ("ist", ""),
        ("ity", ""),
        ("er", ""),
        ("ed", ""),
        ("ly", ""),
        ("es", ""),
        ("s", ""),
    ];

    for (suffix, replacement) in &suffixes {
        if w.len() > suffix.len() + 2 && w.ends_with(suffix) {
            let stem_part = &w[..w.len() - suffix.len()];
            return format!("{}{}", stem_part, replacement);
        }
    }
    w
}

/// Apply stemming filter.
pub fn stemming_filter(tokens: Vec<Token>) -> Vec<Token> {
    tokens
        .into_iter()
        .map(|mut t| {
            t.text = stem(&t.text);
            t
        })
        .collect()
}

/// Synonym expansion: for each token, if it has synonyms, add them.
pub fn synonym_filter(tokens: Vec<Token>, synonyms: &HashMap<String, Vec<String>>) -> Vec<Token> {
    let mut result = Vec::new();
    for t in tokens {
        let lower = t.text.to_lowercase();
        result.push(t.clone());
        if let Some(syns) = synonyms.get(&lower) {
            for syn in syns {
                result.push(Token {
                    text: syn.clone(),
                    start: t.start,
                    end: t.end,
                    position: t.position,
                });
            }
        }
    }
    result
}

/// Generate n-grams from tokens.
pub fn ngram_filter(tokens: Vec<Token>, min_n: usize, max_n: usize) -> Vec<Token> {
    let mut result = Vec::new();
    for t in &tokens {
        let chars: Vec<char> = t.text.chars().collect();
        for n in min_n..=max_n {
            if n > chars.len() {
                continue;
            }
            for i in 0..=chars.len() - n {
                let ngram: String = chars[i..i + n].iter().collect();
                result.push(Token {
                    text: ngram,
                    start: t.start,
                    end: t.end,
                    position: t.position,
                });
            }
        }
    }
    result
}

/// Generate edge n-grams (prefixes) from tokens.
pub fn edge_ngram_filter(tokens: Vec<Token>, min_n: usize, max_n: usize) -> Vec<Token> {
    let mut result = Vec::new();
    for t in &tokens {
        let chars: Vec<char> = t.text.chars().collect();
        for n in min_n..=max_n.min(chars.len()) {
            let prefix: String = chars[..n].iter().collect();
            result.push(Token {
                text: prefix,
                start: t.start,
                end: t.end,
                position: t.position,
            });
        }
    }
    result
}

/// Remove tokens shorter than a minimum length.
pub fn length_filter(tokens: Vec<Token>, min_len: usize) -> Vec<Token> {
    tokens.into_iter().filter(|t| t.text.len() >= min_len).collect()
}

/// Trim whitespace from token text.
pub fn trim_filter(tokens: Vec<Token>) -> Vec<Token> {
    tokens
        .into_iter()
        .map(|mut t| {
            t.text = t.text.trim().to_string();
            t
        })
        .filter(|t| !t.text.is_empty())
        .collect()
}

// ── Analyzer ────────────────────────────────────────────────────

/// A filter step in the analysis pipeline.
#[derive(Debug, Clone)]
pub enum FilterStep {
    Lowercase,
    StopWords(HashSet<String>),
    Stemming,
    Synonyms(HashMap<String, Vec<String>>),
    Ngram { min: usize, max: usize },
    EdgeNgram { min: usize, max: usize },
    MinLength(usize),
    Trim,
}

/// Which tokenizer an analyzer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// Split on whitespace.
    Whitespace,
    /// Split on non-alphanumeric unicode boundaries.
    Unicode,
    /// Entire input is one token (for keyword / exact-match fields).
    Keyword,
}

/// A complete text analysis pipeline: char filters + tokenizer + token filters.
#[derive(Debug, Clone)]
pub struct Analyzer {
    pub name: String,
    pub char_filters: Vec<CharFilter>,
    pub tokenizer: TokenizerKind,
    pub token_filters: Vec<FilterStep>,
}

impl Analyzer {
    /// Create a standard analyzer (lowercase + stop words + stemming).
    pub fn standard() -> Self {
        Self {
            name: "standard".to_string(),
            char_filters: vec![],
            tokenizer: TokenizerKind::Unicode,
            token_filters: vec![
                FilterStep::Lowercase,
                FilterStep::StopWords(english_stop_words()),
                FilterStep::Stemming,
            ],
        }
    }

    /// Create a simple analyzer (whitespace + lowercase).
    pub fn simple() -> Self {
        Self {
            name: "simple".to_string(),
            char_filters: vec![],
            tokenizer: TokenizerKind::Whitespace,
            token_filters: vec![FilterStep::Lowercase],
        }
    }

    /// Create a keyword analyzer (no tokenization — treats entire input as one token).
    pub fn keyword() -> Self {
        Self {
            name: "keyword".to_string(),
            char_filters: vec![],
            tokenizer: TokenizerKind::Keyword,
            token_filters: vec![FilterStep::Trim],
        }
    }

    /// Create a custom analyzer.
    pub fn custom(name: &str) -> Self {
        Self {
            name: name.to_string(),
            char_filters: vec![],
            tokenizer: TokenizerKind::Unicode,
            token_filters: vec![],
        }
    }

    /// Add a character filter.
    pub fn with_char_filter(mut self, filter: CharFilter) -> Self {
        self.char_filters.push(filter);
        self
    }

    /// Set unicode tokenizer.
    pub fn with_unicode_tokenizer(mut self) -> Self {
        self.tokenizer = TokenizerKind::Unicode;
        self
    }

    /// Set whitespace tokenizer.
    pub fn with_whitespace_tokenizer(mut self) -> Self {
        self.tokenizer = TokenizerKind::Whitespace;
        self
    }

    /// Set keyword tokenizer (entire input is one token).
    pub fn with_keyword_tokenizer(mut self) -> Self {
        self.tokenizer = TokenizerKind::Keyword;
        self
    }

    /// Add a token filter step.
    pub fn with_filter(mut self, step: FilterStep) -> Self {
        self.token_filters.push(step);
        self
    }

    /// Analyze text through the full pipeline.
    pub fn analyze(&self, text: &str) -> Vec<Token> {
        // Apply character filters.
        let mut processed = text.to_string();
        for cf in &self.char_filters {
            processed = apply_char_filter(&processed, cf);
        }

        // Tokenize.
        let mut tokens = match self.tokenizer {
            TokenizerKind::Unicode => unicode_tokenize(&processed),
            TokenizerKind::Whitespace => whitespace_tokenize(&processed),
            TokenizerKind::Keyword => {
                // Entire input is a single token.
                vec![Token {
                    text: processed.clone(),
                    start: 0,
                    end: processed.len(),
                    position: 0,
                }]
            }
        };

        // Apply token filters.
        for step in &self.token_filters {
            tokens = match step {
                FilterStep::Lowercase => lowercase_filter(tokens),
                FilterStep::StopWords(sw) => stop_word_filter(tokens, sw),
                FilterStep::Stemming => stemming_filter(tokens),
                FilterStep::Synonyms(syn) => synonym_filter(tokens, syn),
                FilterStep::Ngram { min, max } => ngram_filter(tokens, *min, *max),
                FilterStep::EdgeNgram { min, max } => edge_ngram_filter(tokens, *min, *max),
                FilterStep::MinLength(n) => length_filter(tokens, *n),
                FilterStep::Trim => trim_filter(tokens),
            };
        }

        tokens
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenize() {
        let tokens = whitespace_tokenize("hello world  foo");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "foo");
    }

    #[test]
    fn test_whitespace_positions() {
        let tokens = whitespace_tokenize("hello world");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[1].position, 1);
    }

    #[test]
    fn test_whitespace_offsets() {
        let tokens = whitespace_tokenize("hello world");
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 5);
        assert_eq!(tokens[1].start, 6);
        assert_eq!(tokens[1].end, 11);
    }

    #[test]
    fn test_unicode_tokenize() {
        let tokens = unicode_tokenize("hello, world! foo-bar");
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "foo");
        assert_eq!(tokens[3].text, "bar");
    }

    #[test]
    fn test_unicode_tokenize_empty() {
        let tokens = unicode_tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_lowercase_filter() {
        let tokens = whitespace_tokenize("Hello WORLD");
        let filtered = lowercase_filter(tokens);
        assert_eq!(filtered[0].text, "hello");
        assert_eq!(filtered[1].text, "world");
    }

    #[test]
    fn test_stop_word_filter() {
        let tokens = whitespace_tokenize("the quick brown fox");
        let sw = english_stop_words();
        let filtered = stop_word_filter(tokens, &sw);
        // "the" should be removed
        assert!(filtered.iter().all(|t| t.text != "the"));
        assert!(filtered.iter().any(|t| t.text == "quick"));
    }

    #[test]
    fn test_stemming() {
        assert_eq!(stem("running"), "runn");
        assert_eq!(stem("cats"), "cat");
        assert_eq!(stem("happiness"), "happi");
    }

    #[test]
    fn test_stemming_filter() {
        let tokens = whitespace_tokenize("running cats");
        let stemmed = stemming_filter(tokens);
        assert_eq!(stemmed[0].text, "runn");
        assert_eq!(stemmed[1].text, "cat");
    }

    #[test]
    fn test_synonym_filter() {
        let tokens = whitespace_tokenize("fast car");
        let mut syns = HashMap::new();
        syns.insert("fast".to_string(), vec!["quick".to_string(), "rapid".to_string()]);
        let expanded = synonym_filter(tokens, &syns);
        // Should have original + 2 synonyms + "car"
        assert_eq!(expanded.len(), 4);
        assert!(expanded.iter().any(|t| t.text == "fast"));
        assert!(expanded.iter().any(|t| t.text == "quick"));
        assert!(expanded.iter().any(|t| t.text == "rapid"));
    }

    #[test]
    fn test_ngram_filter() {
        let tokens = vec![Token {
            text: "hello".to_string(),
            start: 0,
            end: 5,
            position: 0,
        }];
        let ngrams = ngram_filter(tokens, 2, 3);
        // 2-grams: he, el, ll, lo = 4; 3-grams: hel, ell, llo = 3; total = 7
        assert_eq!(ngrams.len(), 7);
        assert!(ngrams.iter().any(|t| t.text == "he"));
        assert!(ngrams.iter().any(|t| t.text == "hel"));
    }

    #[test]
    fn test_edge_ngram_filter() {
        let tokens = vec![Token {
            text: "hello".to_string(),
            start: 0,
            end: 5,
            position: 0,
        }];
        let edge = edge_ngram_filter(tokens, 1, 4);
        assert_eq!(edge.len(), 4);
        assert_eq!(edge[0].text, "h");
        assert_eq!(edge[1].text, "he");
        assert_eq!(edge[2].text, "hel");
        assert_eq!(edge[3].text, "hell");
    }

    #[test]
    fn test_length_filter() {
        let tokens = whitespace_tokenize("a bb ccc dddd");
        let filtered = length_filter(tokens, 3);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].text, "ccc");
        assert_eq!(filtered[1].text, "dddd");
    }

    #[test]
    fn test_html_strip_char_filter() {
        let result = apply_char_filter("<p>hello <b>world</b></p>", &CharFilter::HtmlStrip);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_ascii_normalize_char_filter() {
        let result = apply_char_filter("caf\u{00E9} na\u{00EF}ve", &CharFilter::AsciiNormalize);
        assert_eq!(result, "cafe naive");
    }

    #[test]
    fn test_strip_non_alphanumeric() {
        let result = apply_char_filter("hello, world! 123", &CharFilter::StripNonAlphanumeric);
        assert_eq!(result, "hello world 123");
    }

    #[test]
    fn test_mapping_char_filter() {
        let mut map = HashMap::new();
        map.insert('\u{00E6}', "ae".to_string()); // ae ligature
        let result = apply_char_filter("\u{00E6}sop", &CharFilter::Mapping(map));
        assert_eq!(result, "aesop");
    }

    #[test]
    fn test_standard_analyzer() {
        let analyzer = Analyzer::standard();
        let tokens = analyzer.analyze("The Quick Brown Foxes are running");
        // "the" and "are" removed (stop words), remaining lowercase + stemmed
        assert!(tokens.iter().all(|t| t.text != "the"));
        assert!(tokens.iter().all(|t| t.text != "are"));
        // "quick" should be there (lowercased)
        assert!(tokens.iter().any(|t| t.text == "quick"));
    }

    #[test]
    fn test_simple_analyzer() {
        let analyzer = Analyzer::simple();
        let tokens = analyzer.analyze("Hello World");
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_custom_analyzer() {
        let analyzer = Analyzer::custom("my_analyzer")
            .with_char_filter(CharFilter::HtmlStrip)
            .with_unicode_tokenizer()
            .with_filter(FilterStep::Lowercase)
            .with_filter(FilterStep::MinLength(3));
        let tokens = analyzer.analyze("<p>A Big Cat</p>");
        // "a" removed by min length
        assert!(tokens.iter().all(|t| t.text != "a"));
        assert!(tokens.iter().any(|t| t.text == "big"));
        assert!(tokens.iter().any(|t| t.text == "cat"));
    }

    #[test]
    fn test_keyword_analyzer() {
        let analyzer = Analyzer::keyword();
        let tokens = analyzer.analyze("  hello world  ");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "hello world");
    }

    #[test]
    fn test_english_stop_words() {
        let sw = english_stop_words();
        assert!(sw.contains("the"));
        assert!(sw.contains("is"));
        assert!(!sw.contains("hello"));
    }
}
