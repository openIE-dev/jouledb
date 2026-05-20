//! Full-Text Search analyzers for JouleDB.
//!
//! Provides pluggable text analysis with tokenization, stopword removal,
//! and stemming for FTS index construction and query processing.

use std::collections::HashSet;

/// A positioned token from text analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The normalized token text.
    pub text: String,
    /// Token position in the original text (0-indexed).
    pub position: usize,
}

/// Trait for text analyzers used in full-text search.
pub trait Analyzer: Send + Sync {
    /// Tokenize input text into positioned tokens.
    fn tokenize(&self, text: &str) -> Vec<Token>;
    /// Analyzer name for serialization.
    fn name(&self) -> &str;
}

/// Standard analyzer: Unicode word boundary tokenization, lowercase,
/// stopword removal, and Porter stemming.
pub struct StandardAnalyzer {
    stopwords: HashSet<&'static str>,
}

impl Default for StandardAnalyzer {
    fn default() -> Self {
        Self {
            stopwords: ENGLISH_STOPWORDS.iter().copied().collect(),
        }
    }
}

impl Analyzer for StandardAnalyzer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::with_capacity(text.len() / 6 + 1);
        let mut pos = 0;
        for word in unicode_word_split(text) {
            let lower = word.to_lowercase();
            if lower.len() < 2 || self.stopwords.contains(lower.as_str()) {
                continue;
            }
            let stemmed = porter_stem(&lower);
            tokens.push(Token {
                text: stemmed,
                position: pos,
            });
            pos += 1;
        }
        tokens
    }

    fn name(&self) -> &str {
        "standard"
    }
}

/// Simple analyzer: whitespace split + lowercase. No stemming, no stopwords.
pub struct SimpleAnalyzer;

impl Analyzer for SimpleAnalyzer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::with_capacity(text.len() / 6 + 1);
        let mut pos = 0;
        for word in text.split_whitespace() {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();
            if cleaned.len() >= 2 {
                tokens.push(Token {
                    text: cleaned,
                    position: pos,
                });
                pos += 1;
            }
        }
        tokens
    }

    fn name(&self) -> &str {
        "simple"
    }
}

/// Whitespace analyzer: whitespace split only. No case folding, no stemming.
pub struct WhitespaceAnalyzer;

impl Analyzer for WhitespaceAnalyzer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        text.split_whitespace()
            .enumerate()
            .filter(|(_, w)| !w.is_empty())
            .map(|(i, w)| Token {
                text: w.to_string(),
                position: i,
            })
            .collect()
    }

    fn name(&self) -> &str {
        "whitespace"
    }
}

/// Keyword analyzer: treats entire input as a single token.
pub struct KeywordAnalyzer;

impl Analyzer for KeywordAnalyzer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        if text.is_empty() {
            Vec::new()
        } else {
            vec![Token {
                text: text.to_string(),
                position: 0,
            }]
        }
    }

    fn name(&self) -> &str {
        "keyword"
    }
}

/// N-gram analyzer: generates character n-grams for substring matching.
/// Useful for CJK text and partial-word search.
pub struct NgramAnalyzer {
    min_gram: usize,
    max_gram: usize,
}

impl NgramAnalyzer {
    pub fn new(min_gram: usize, max_gram: usize) -> Self {
        Self {
            min_gram: min_gram.max(1),
            max_gram: max_gram.max(min_gram.max(1)),
        }
    }
}

impl Default for NgramAnalyzer {
    fn default() -> Self {
        Self {
            min_gram: 2,
            max_gram: 3,
        }
    }
}

impl Analyzer for NgramAnalyzer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::with_capacity(text.len());
        let mut pos = 0;
        // Split on whitespace, then generate n-grams from each word
        for word in unicode_word_split(text) {
            let lower = word.to_lowercase();
            let chars: Vec<char> = lower.chars().collect();
            if chars.is_empty() {
                continue;
            }
            for n in self.min_gram..=self.max_gram {
                if n > chars.len() {
                    continue;
                }
                for i in 0..=(chars.len() - n) {
                    let gram: String = chars[i..i + n].iter().collect();
                    tokens.push(Token {
                        text: gram,
                        position: pos,
                    });
                    pos += 1;
                }
            }
        }
        // Also generate n-grams from CJK runs (contiguous CJK chars across word boundaries)
        let chars: Vec<char> = text.chars().collect();
        let mut cjk_start = None;
        for (i, &ch) in chars.iter().enumerate() {
            if is_cjk(ch) {
                if cjk_start.is_none() {
                    cjk_start = Some(i);
                }
            } else if let Some(start) = cjk_start {
                let run: Vec<char> = chars[start..i].to_vec();
                self.emit_cjk_ngrams(&run, &mut tokens, &mut pos);
                cjk_start = None;
            }
        }
        if let Some(start) = cjk_start {
            let run: Vec<char> = chars[start..].to_vec();
            self.emit_cjk_ngrams(&run, &mut tokens, &mut pos);
        }
        tokens
    }

    fn name(&self) -> &str {
        "ngram"
    }
}

impl NgramAnalyzer {
    fn emit_cjk_ngrams(&self, chars: &[char], tokens: &mut Vec<Token>, pos: &mut usize) {
        for n in self.min_gram..=self.max_gram {
            if n > chars.len() {
                continue;
            }
            for i in 0..=(chars.len() - n) {
                let gram: String = chars[i..i + n].iter().collect();
                tokens.push(Token {
                    text: gram,
                    position: *pos,
                });
                *pos += 1;
            }
        }
    }
}

/// Check if a character is in a CJK Unicode block.
fn is_cjk(ch: char) -> bool {
    let cp = ch as u32;
    // CJK Unified Ideographs
    (0x4E00..=0x9FFF).contains(&cp)
    // CJK Extension A
    || (0x3400..=0x4DBF).contains(&cp)
    // CJK Extension B
    || (0x20000..=0x2A6DF).contains(&cp)
    // CJK Compatibility Ideographs
    || (0xF900..=0xFAFF).contains(&cp)
    // Hiragana
    || (0x3040..=0x309F).contains(&cp)
    // Katakana
    || (0x30A0..=0x30FF).contains(&cp)
    // Hangul Syllables
    || (0xAC00..=0xD7AF).contains(&cp)
}

/// Create an analyzer by name. Defaults to "standard".
/// For ngram, use "ngram" or "ngram:min:max" (e.g., "ngram:2:4").
pub fn create_analyzer(name: &str) -> Box<dyn Analyzer> {
    let lower = name.to_lowercase();
    if lower.starts_with("ngram") {
        // Parse optional min:max from "ngram:2:4"
        let parts: Vec<&str> = lower.split(':').collect();
        let min = parts
            .get(1)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);
        let max = parts
            .get(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3);
        return Box::new(NgramAnalyzer::new(min, max));
    }
    match lower.as_str() {
        "simple" => Box::new(SimpleAnalyzer),
        "whitespace" => Box::new(WhitespaceAnalyzer),
        "keyword" => Box::new(KeywordAnalyzer),
        _ => Box::new(StandardAnalyzer::default()),
    }
}

// ==================== Unicode Word Splitting ====================

/// Split text on non-alphanumeric boundaries, yielding word-like segments.
fn unicode_word_split(text: &str) -> Vec<&str> {
    // Estimate ~1 word per 5 chars on average
    let mut words = Vec::with_capacity(text.len() / 5 + 1);
    let bytes = text.as_bytes();
    let mut start = None;

    for (i, ch) in text.char_indices() {
        if ch.is_alphanumeric() || ch == '_' {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            words.push(&text[s..i]);
            start = None;
        }
    }
    if let Some(s) = start {
        words.push(&text[s..]);
    }
    let _ = bytes; // suppress unused warning
    words
}

// ==================== Porter Stemmer ====================
// Implements Porter's algorithm (1980) Steps 1a, 1b, 1c, 2, 3, 4, 5a, 5b.

/// Apply Porter stemming to a lowercase word.
pub fn porter_stem(word: &str) -> String {
    if word.len() <= 2 {
        return word.to_string();
    }

    let mut s = word.to_string();
    s = step1a(s);
    s = step1b(s);
    s = step1c(s);
    s = step2(s);
    s = step3(s);
    s = step4(s);
    s = step5a(s);
    s = step5b(s);
    s
}

fn measure(stem: &str) -> usize {
    // Count VC sequences (consonant-vowel transitions)
    let mut m = 0;
    let mut prev_vowel = false;
    for ch in stem.chars() {
        let is_v = is_vowel_char(ch, prev_vowel);
        if !is_v && prev_vowel {
            m += 1;
        }
        prev_vowel = is_v;
    }
    m
}

fn is_vowel_char(ch: char, _prev: bool) -> bool {
    matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u')
}

fn has_vowel(stem: &str) -> bool {
    stem.chars().any(|ch| is_vowel_char(ch, false))
}

fn ends_double_consonant(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() < 2 {
        return false;
    }
    let last = bytes[bytes.len() - 1];
    let prev = bytes[bytes.len() - 2];
    last == prev && !is_vowel_char(last as char, false)
}

fn ends_cvc(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() < 3 {
        return false;
    }
    let c1 = bytes[bytes.len() - 3] as char;
    let v = bytes[bytes.len() - 2] as char;
    let c2 = bytes[bytes.len() - 1] as char;
    !is_vowel_char(c2, false)
        && is_vowel_char(v, false)
        && !is_vowel_char(c1, false)
        && !matches!(c2, 'w' | 'x' | 'y')
}

// Step 1a: plurals
fn step1a(s: String) -> String {
    if s.ends_with("sses") {
        return s[..s.len() - 2].to_string();
    }
    if s.ends_with("ies") {
        return s[..s.len() - 2].to_string();
    }
    if s.ends_with("ss") {
        return s;
    }
    if s.ends_with('s') && s.len() > 3 {
        return s[..s.len() - 1].to_string();
    }
    s
}

// Step 1b: -ed, -ing
fn step1b(s: String) -> String {
    if s.ends_with("eed") {
        let stem = &s[..s.len() - 3];
        if measure(stem) > 0 {
            return s[..s.len() - 1].to_string(); // -eed -> -ee
        }
        return s;
    }

    let (trimmed, matched) = if s.ends_with("ed") && s.len() > 4 {
        (s[..s.len() - 2].to_string(), true)
    } else if s.ends_with("ing") && s.len() > 5 {
        (s[..s.len() - 3].to_string(), true)
    } else {
        (s, false)
    };

    if matched && has_vowel(&trimmed) {
        return step1b_fixup(trimmed);
    }
    trimmed
}

fn step1b_fixup(s: String) -> String {
    if s.ends_with("at") || s.ends_with("bl") || s.ends_with("iz") {
        return format!("{}e", s);
    }
    if ends_double_consonant(&s) {
        let last = s.as_bytes()[s.len() - 1] as char;
        if !matches!(last, 'l' | 's' | 'z') {
            return s[..s.len() - 1].to_string();
        }
    }
    if measure(&s) == 1 && ends_cvc(&s) {
        return format!("{}e", s);
    }
    s
}

// Step 1c: y -> i
fn step1c(s: String) -> String {
    if s.ends_with('y') && s.len() > 2 {
        let stem = &s[..s.len() - 1];
        if has_vowel(stem) {
            return format!("{}i", stem);
        }
    }
    s
}

// Step 2: suffix normalization (m > 0)
fn step2(s: String) -> String {
    let replacements: &[(&str, &str)] = &[
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ];
    for &(suffix, replacement) in replacements {
        if s.ends_with(suffix) {
            let stem = &s[..s.len() - suffix.len()];
            if measure(stem) > 0 {
                return format!("{}{}", stem, replacement);
            }
            return s;
        }
    }
    s
}

// Step 3: suffix normalization (m > 0)
fn step3(s: String) -> String {
    let replacements: &[(&str, &str)] = &[
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];
    for &(suffix, replacement) in replacements {
        if s.ends_with(suffix) {
            let stem = &s[..s.len() - suffix.len()];
            if measure(stem) > 0 {
                return format!("{}{}", stem, replacement);
            }
            return s;
        }
    }
    s
}

// Step 4: suffix removal (m > 1)
fn step4(s: String) -> String {
    let suffixes: &[&str] = &[
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
        "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ];
    for &suffix in suffixes {
        if s.ends_with(suffix) {
            let stem = &s[..s.len() - suffix.len()];
            if suffix == "ion" {
                // Special: stem must end in s or t
                if measure(stem) > 1 && (stem.ends_with('s') || stem.ends_with('t')) {
                    return stem.to_string();
                }
            } else if measure(stem) > 1 {
                return stem.to_string();
            }
            return s;
        }
    }
    s
}

// Step 5a: remove trailing e
fn step5a(s: String) -> String {
    if s.ends_with('e') {
        let stem = &s[..s.len() - 1];
        if measure(stem) > 1 {
            return stem.to_string();
        }
        if measure(stem) == 1 && !ends_cvc(stem) {
            return stem.to_string();
        }
    }
    s
}

// Step 5b: -ll -> -l if m > 1
fn step5b(s: String) -> String {
    if s.ends_with("ll") && measure(&s[..s.len() - 1]) > 1 {
        return s[..s.len() - 1].to_string();
    }
    s
}

// ==================== Stopwords ====================

/// 175 common English stopwords.
pub static ENGLISH_STOPWORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "get",
    "got",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "its",
    "itself",
    "just",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "us",
    "very",
    "was",
    "wasn't",
    "we",
    "were",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

/// German stopwords (common).
pub static GERMAN_STOPWORDS: &[&str] = &[
    "aber",
    "alle",
    "allem",
    "allen",
    "aller",
    "allerdings",
    "alles",
    "also",
    "am",
    "an",
    "ander",
    "andere",
    "anderem",
    "anderen",
    "anderer",
    "anderes",
    "als",
    "auf",
    "aus",
    "bei",
    "beim",
    "bereits",
    "bin",
    "bis",
    "bist",
    "da",
    "damit",
    "dann",
    "das",
    "dass",
    "dein",
    "deine",
    "deinem",
    "deinen",
    "deiner",
    "dem",
    "den",
    "denn",
    "der",
    "des",
    "die",
    "dies",
    "diese",
    "diesem",
    "diesen",
    "dieser",
    "doch",
    "du",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "er",
    "es",
    "etwas",
    "euch",
    "euer",
    "eure",
    "eurem",
    "euren",
    "eurer",
    "fur",
    "gegen",
    "hat",
    "hatte",
    "ich",
    "ihm",
    "ihn",
    "ihnen",
    "ihr",
    "ihre",
    "ihrem",
    "ihren",
    "ihrer",
    "im",
    "in",
    "ist",
    "ja",
    "jede",
    "jedem",
    "jeden",
    "jeder",
    "jedes",
    "jene",
    "jenem",
    "jenen",
    "jener",
    "jenes",
    "kann",
    "kein",
    "keine",
    "keinem",
    "keinen",
    "keiner",
    "man",
    "mein",
    "meine",
    "meinem",
    "meinen",
    "meiner",
    "mir",
    "mit",
    "muss",
    "nach",
    "nicht",
    "nichts",
    "noch",
    "nun",
    "nur",
    "ob",
    "oder",
    "ohne",
    "sehr",
    "sein",
    "seine",
    "seinem",
    "seinen",
    "seiner",
    "sich",
    "sie",
    "sind",
    "so",
    "soll",
    "sollte",
    "uber",
    "um",
    "und",
    "uns",
    "unser",
    "unsere",
    "unserem",
    "unseren",
    "unserer",
    "von",
    "vor",
    "war",
    "warum",
    "was",
    "weil",
    "welch",
    "welche",
    "welchem",
    "welchen",
    "welcher",
    "wenn",
    "wer",
    "werde",
    "wie",
    "wieder",
    "will",
    "wir",
    "wird",
    "wo",
    "worden",
    "wurde",
    "zu",
    "zum",
    "zur",
];

/// French stopwords (common).
pub static FRENCH_STOPWORDS: &[&str] = &[
    "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux", "il",
    "je", "la", "le", "leur", "lui", "ma", "mais", "me", "mes", "mon", "ne", "nos", "notre",
    "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur",
    "ta", "te", "tes", "ton", "tu", "un", "une", "vos", "votre", "vous", "est", "sont", "ont",
    "fait", "comme", "tout", "tous",
];

/// Spanish stopwords (common).
pub static SPANISH_STOPWORDS: &[&str] = &[
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con",
    "no", "una", "su", "al", "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "fue", "este",
    "ha", "si", "porque", "esta", "son", "entre", "cuando", "muy", "sin", "sobre", "ser",
    "tambien", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "estas", "algunos",
];

/// Get stopwords for a language code.
pub fn stopwords_for_language(lang: &str) -> &'static [&'static str] {
    match lang.to_lowercase().as_str() {
        "en" | "english" => ENGLISH_STOPWORDS,
        "de" | "german" => GERMAN_STOPWORDS,
        "fr" | "french" => FRENCH_STOPWORDS,
        "es" | "spanish" => SPANISH_STOPWORDS,
        _ => ENGLISH_STOPWORDS,
    }
}

// ==================== Boolean Query Parsing ====================

/// A parsed boolean query clause.
#[derive(Debug, Clone, PartialEq)]
pub enum BooleanClause {
    /// `+term` — required
    Required(String),
    /// `-term` — excluded
    Excluded(String),
    /// `"exact phrase"` — phrase match
    Phrase(String),
    /// bare term — optional (boosts score if present)
    Optional(String),
}

/// Parse a boolean query string into clauses.
///
/// Syntax:
/// - `+term` — term is required (doc MUST contain it)
/// - `-term` — term is excluded (doc MUST NOT contain it)
/// - `"exact phrase"` — phrase must appear consecutively
/// - `term` — optional (increases score but not required)
///
/// Example: `+database -mysql "full text search" indexing`
pub fn parse_boolean_query(query: &str) -> Vec<BooleanClause> {
    let mut clauses = Vec::new();
    let chars: Vec<char> = query.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Skip whitespace
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        // Quoted phrase
        if chars[i] == '"' {
            i += 1; // skip opening quote
            let start = i;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            let phrase: String = chars[start..i].iter().collect();
            if !phrase.trim().is_empty() {
                clauses.push(BooleanClause::Phrase(phrase.trim().to_string()));
            }
            if i < chars.len() {
                i += 1;
            } // skip closing quote
            continue;
        }

        // + or - prefix
        let prefix = if chars[i] == '+' {
            i += 1;
            Some('+')
        } else if chars[i] == '-' {
            i += 1;
            Some('-')
        } else {
            None
        };

        // Read the term
        let start = i;
        while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '"' {
            i += 1;
        }
        let term: String = chars[start..i].iter().collect();
        if term.is_empty() {
            continue;
        }

        let term_lower = term.to_lowercase();
        match prefix {
            Some('+') => clauses.push(BooleanClause::Required(term_lower)),
            Some('-') => clauses.push(BooleanClause::Excluded(term_lower)),
            _ => clauses.push(BooleanClause::Optional(term_lower)),
        }
    }

    clauses
}

/// Check if a document matches all boolean clauses.
/// Returns true if the doc passes all required/excluded/phrase constraints.
/// `doc_terms` is the set of stemmed terms in the document.
/// `doc_text` is the raw text (for phrase matching).
pub fn matches_boolean_query(
    clauses: &[BooleanClause],
    doc_terms: &HashSet<&str>,
    doc_text: &str,
    analyzer: &dyn Analyzer,
) -> bool {
    for clause in clauses {
        match clause {
            BooleanClause::Required(term) => {
                // Stem the required term to match indexed form
                let stemmed = porter_stem(term);
                if !doc_terms.contains(stemmed.as_str()) {
                    return false;
                }
            }
            BooleanClause::Excluded(term) => {
                let stemmed = porter_stem(term);
                if doc_terms.contains(stemmed.as_str()) {
                    return false;
                }
            }
            BooleanClause::Phrase(phrase) => {
                if !phrase_match(phrase, doc_text) {
                    return false;
                }
            }
            BooleanClause::Optional(_) => {
                // Optional terms don't filter, just boost score
            }
        }
    }
    let _ = analyzer; // available for future use
    true
}

/// Extract all scoring terms from boolean clauses (required + optional).
pub fn boolean_query_scoring_terms(clauses: &[BooleanClause]) -> Vec<String> {
    let mut terms = Vec::new();
    for clause in clauses {
        match clause {
            BooleanClause::Required(t) | BooleanClause::Optional(t) => {
                terms.push(porter_stem(t));
            }
            BooleanClause::Phrase(phrase) => {
                // Score individual words from the phrase
                for word in phrase.split_whitespace() {
                    let lower = word.to_lowercase();
                    if lower.len() >= 2 {
                        terms.push(porter_stem(&lower));
                    }
                }
            }
            BooleanClause::Excluded(_) => {} // excluded terms don't contribute to score
        }
    }
    terms
}

// ==================== Field Boost Helpers ====================

/// Parse a field boost specification like "title^2.0,body^1.0,tags^0.5".
/// Returns a map of column name → boost weight.
pub fn parse_field_boosts(spec: &str) -> std::collections::HashMap<String, f64> {
    let mut boosts = std::collections::HashMap::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some(caret_pos) = part.find('^') {
            let col = part[..caret_pos].trim().to_string();
            let weight: f64 = part[caret_pos + 1..].trim().parse().unwrap_or(1.0);
            if !col.is_empty() {
                boosts.insert(col, weight);
            }
        } else {
            boosts.insert(part.to_string(), 1.0);
        }
    }
    boosts
}

// ==================== BM25 Helpers ====================

/// Compute BM25 IDF component: ln((N - df + 0.5) / (df + 0.5) + 1)
/// where N = total documents, df = documents containing term.
pub fn bm25_idf(total_docs: f64, doc_freq: f64) -> f64 {
    ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln()
}

/// Compute BM25 term score given tf, doc length, avg doc length.
/// k1=1.2, b=0.75 (standard defaults).
pub fn bm25_term_score(tf: f64, doc_len: f64, avg_dl: f64, idf: f64) -> f64 {
    let k1 = 1.2_f64;
    let b = 0.75_f64;
    let numerator = tf * (k1 + 1.0);
    let denominator = tf + k1 * (1.0 - b + b * doc_len / avg_dl.max(1.0));
    idf * numerator / denominator
}

// ==================== Fuzzy Matching ====================

/// Levenshtein edit distance between two strings.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Check if any token in text fuzzy-matches any query token within max_distance.
pub fn fuzzy_match(query: &str, text: &str, max_distance: usize) -> bool {
    let query_tokens: Vec<String> = query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    let text_tokens: Vec<String> = text
        .split_whitespace()
        .map(|t| {
            t.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect();

    for qt in &query_tokens {
        let mut found = false;
        for tt in &text_tokens {
            if levenshtein_distance(qt, tt) <= max_distance {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    !query_tokens.is_empty()
}

// ==================== Phrase Matching ====================

/// Check if a phrase (sequence of words) appears consecutively in text.
pub fn phrase_match(phrase: &str, text: &str) -> bool {
    let phrase_tokens: Vec<String> = phrase
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    if phrase_tokens.is_empty() {
        return false;
    }

    let text_tokens: Vec<String> = unicode_word_split(text)
        .iter()
        .map(|w| w.to_lowercase())
        .collect();

    if text_tokens.len() < phrase_tokens.len() {
        return false;
    }

    for i in 0..=(text_tokens.len() - phrase_tokens.len()) {
        if text_tokens[i..i + phrase_tokens.len()] == phrase_tokens[..] {
            return true;
        }
    }
    false
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_analyzer_basic() {
        let analyzer = StandardAnalyzer::default();
        let tokens = analyzer.tokenize("The quick brown fox jumps over the lazy dog");
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        // "the" is a stopword; remaining get stemmed
        assert!(!words.contains(&"the"));
        assert!(words.contains(&"quick"));
        assert!(words.contains(&"brown"));
        assert!(words.contains(&"fox"));
        assert!(words.contains(&"lazi")); // "lazy" -> "lazi" via step1c y->i
    }

    #[test]
    fn test_standard_analyzer_stopword_removal() {
        let analyzer = StandardAnalyzer::default();
        let tokens = analyzer.tokenize("this is a test of the analyzer");
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(!words.contains(&"this"));
        assert!(!words.contains(&"is"));
        assert!(!words.contains(&"a"));
        assert!(!words.contains(&"of"));
        assert!(!words.contains(&"the"));
        assert!(words.contains(&"test"));
        assert!(words.contains(&"analyz")); // stemmed
    }

    #[test]
    fn test_standard_analyzer_positions() {
        let analyzer = StandardAnalyzer::default();
        let tokens = analyzer.tokenize("hello world test");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[1].position, 1);
        assert_eq!(tokens[2].position, 2);
    }

    #[test]
    fn test_simple_analyzer() {
        let analyzer = SimpleAnalyzer;
        let tokens = analyzer.tokenize("Hello World! This is GREAT.");
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(words, vec!["hello", "world", "this", "is", "great"]);
    }

    #[test]
    fn test_whitespace_analyzer() {
        let analyzer = WhitespaceAnalyzer;
        let tokens = analyzer.tokenize("Hello World! MiXeD");
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(words, vec!["Hello", "World!", "MiXeD"]);
    }

    #[test]
    fn test_keyword_analyzer() {
        let analyzer = KeywordAnalyzer;
        let tokens = analyzer.tokenize("hello world test");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "hello world test");
    }

    #[test]
    fn test_keyword_analyzer_empty() {
        let analyzer = KeywordAnalyzer;
        assert!(analyzer.tokenize("").is_empty());
    }

    #[test]
    fn test_create_analyzer_factory() {
        assert_eq!(create_analyzer("standard").name(), "standard");
        assert_eq!(create_analyzer("simple").name(), "simple");
        assert_eq!(create_analyzer("whitespace").name(), "whitespace");
        assert_eq!(create_analyzer("keyword").name(), "keyword");
        assert_eq!(create_analyzer("unknown").name(), "standard"); // default
    }

    // ---- Porter Stemmer Tests ----

    #[test]
    fn test_porter_stem_basics() {
        assert_eq!(porter_stem("caresses"), "caress");
        assert_eq!(porter_stem("ponies"), "poni");
        assert_eq!(porter_stem("cats"), "cat");
    }

    #[test]
    fn test_porter_stem_ing_ed() {
        assert_eq!(porter_stem("agreed"), "agre");
        assert_eq!(porter_stem("plastered"), "plaster");
        assert_eq!(porter_stem("motoring"), "motor");
        assert_eq!(porter_stem("singing"), "sing");
    }

    #[test]
    fn test_porter_stem_ational() {
        assert_eq!(porter_stem("relational"), "relat");
        assert_eq!(porter_stem("conditional"), "condit");
    }

    #[test]
    fn test_porter_stem_short_words() {
        assert_eq!(porter_stem("a"), "a");
        assert_eq!(porter_stem("is"), "is");
        assert_eq!(porter_stem("go"), "go");
    }

    // ---- BM25 Tests ----

    #[test]
    fn test_bm25_idf() {
        // With 100 docs and df=10: ln((100-10+0.5)/(10+0.5)+1) = ln(9.619...) ≈ 2.26
        let idf = bm25_idf(100.0, 10.0);
        assert!(idf > 2.0 && idf < 3.0);

        // Rare term: df=1 in 1000 docs -> high IDF
        let idf_rare = bm25_idf(1000.0, 1.0);
        assert!(idf_rare > 6.0);
    }

    #[test]
    fn test_bm25_term_score() {
        let idf = bm25_idf(100.0, 10.0);
        let score1 = bm25_term_score(1.0, 100.0, 100.0, idf);
        let score3 = bm25_term_score(3.0, 100.0, 100.0, idf);
        // Higher tf -> higher score (but sublinear)
        assert!(score3 > score1);
        assert!(score3 < score1 * 3.0); // sublinear saturation
    }

    // ---- Levenshtein Tests ----

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", ""), 3);
    }

    #[test]
    fn test_fuzzy_match() {
        assert!(fuzzy_match("hello", "hello world", 0));
        assert!(fuzzy_match("helo", "hello world", 1));
        assert!(!fuzzy_match("helo", "hello world", 0));
        assert!(fuzzy_match("wrld", "hello world", 1));
    }

    // ---- Phrase Match Tests ----

    #[test]
    fn test_phrase_match() {
        assert!(phrase_match("quick brown", "The quick brown fox jumps"));
        assert!(!phrase_match("quick fox", "The quick brown fox jumps"));
        assert!(phrase_match("hello", "hello world"));
        assert!(!phrase_match("missing", "hello world"));
    }

    #[test]
    fn test_phrase_match_case_insensitive() {
        assert!(phrase_match("QUICK BROWN", "The quick brown fox"));
    }

    // ---- N-gram Analyzer Tests ----

    #[test]
    fn test_ngram_analyzer_basic() {
        let analyzer = NgramAnalyzer::new(2, 3);
        let tokens = analyzer.tokenize("hello");
        let grams: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        // 2-grams: he, el, ll, lo = 4
        // 3-grams: hel, ell, llo = 3
        assert!(grams.contains(&"he"));
        assert!(grams.contains(&"el"));
        assert!(grams.contains(&"hel"));
        assert!(grams.contains(&"llo"));
        assert_eq!(grams.len(), 7);
    }

    #[test]
    fn test_ngram_analyzer_factory() {
        let analyzer = create_analyzer("ngram:2:4");
        assert_eq!(analyzer.name(), "ngram");
        let tokens = analyzer.tokenize("test");
        // 2-grams: te, es, st = 3
        // 3-grams: tes, est = 2
        // 4-grams: test = 1
        assert_eq!(tokens.len(), 6);
    }

    #[test]
    fn test_ngram_cjk() {
        let analyzer = NgramAnalyzer::new(1, 2);
        let tokens = analyzer.tokenize("数据库");
        let grams: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        // CJK n-grams: 1-grams: 数, 据, 库 = 3; 2-grams: 数据, 据库 = 2
        assert!(grams.contains(&"数"));
        assert!(grams.contains(&"据"));
        assert!(grams.contains(&"数据"));
        assert!(grams.contains(&"据库"));
    }

    // ---- Boolean Query Tests ----

    #[test]
    fn test_parse_boolean_query_basic() {
        let clauses = parse_boolean_query("+database -mysql indexing");
        assert_eq!(clauses.len(), 3);
        assert_eq!(clauses[0], BooleanClause::Required("database".to_string()));
        assert_eq!(clauses[1], BooleanClause::Excluded("mysql".to_string()));
        assert_eq!(clauses[2], BooleanClause::Optional("indexing".to_string()));
    }

    #[test]
    fn test_parse_boolean_query_phrase() {
        let clauses = parse_boolean_query(r#"+required "exact phrase" optional"#);
        assert_eq!(clauses.len(), 3);
        assert_eq!(clauses[0], BooleanClause::Required("required".to_string()));
        assert_eq!(
            clauses[1],
            BooleanClause::Phrase("exact phrase".to_string())
        );
        assert_eq!(clauses[2], BooleanClause::Optional("optional".to_string()));
    }

    #[test]
    fn test_matches_boolean_required() {
        let clauses = parse_boolean_query("+database");
        let analyzer = StandardAnalyzer::default();
        let terms: HashSet<&str> = ["databas"].into_iter().collect(); // stemmed "database"
        assert!(matches_boolean_query(
            &clauses,
            &terms,
            "database engine",
            &analyzer
        ));

        let terms2: HashSet<&str> = ["engin"].into_iter().collect();
        assert!(!matches_boolean_query(
            &clauses,
            &terms2,
            "engine only",
            &analyzer
        ));
    }

    #[test]
    fn test_matches_boolean_excluded() {
        let clauses = parse_boolean_query("-mysql +database");
        let analyzer = StandardAnalyzer::default();
        // Has database but also mysql → excluded
        let terms: HashSet<&str> = ["databas", "mysql"].into_iter().collect();
        assert!(!matches_boolean_query(
            &clauses,
            &terms,
            "database mysql",
            &analyzer
        ));

        // Has database, no mysql → passes
        let terms2: HashSet<&str> = ["databas"].into_iter().collect();
        assert!(matches_boolean_query(
            &clauses,
            &terms2,
            "database only",
            &analyzer
        ));
    }

    #[test]
    fn test_matches_boolean_phrase() {
        let clauses = parse_boolean_query(r#""full text search""#);
        let analyzer = StandardAnalyzer::default();
        let terms: HashSet<&str> = ["full", "text", "search"].into_iter().collect();
        assert!(matches_boolean_query(
            &clauses,
            &terms,
            "full text search engine",
            &analyzer
        ));
        assert!(!matches_boolean_query(
            &clauses,
            &terms,
            "full search text engine",
            &analyzer
        ));
    }

    #[test]
    fn test_boolean_scoring_terms() {
        let clauses = parse_boolean_query("+database -mysql indexing \"full text\"");
        let terms = boolean_query_scoring_terms(&clauses);
        // Required: database → stemmed, Optional: indexing → stemmed, Phrase: full + text → stemmed
        assert!(terms.len() >= 3);
        // Excluded: mysql should NOT appear
        assert!(!terms.contains(&"mysql".to_string()));
    }

    // ---- Field Boost Tests ----

    #[test]
    fn test_parse_field_boosts() {
        let boosts = parse_field_boosts("title^2.0,body^1.0,tags^0.5");
        assert_eq!(boosts.get("title"), Some(&2.0));
        assert_eq!(boosts.get("body"), Some(&1.0));
        assert_eq!(boosts.get("tags"), Some(&0.5));
    }

    #[test]
    fn test_parse_field_boosts_no_weight() {
        let boosts = parse_field_boosts("title,body");
        assert_eq!(boosts.get("title"), Some(&1.0));
        assert_eq!(boosts.get("body"), Some(&1.0));
    }

    // ---- Language Stopwords Tests ----

    #[test]
    fn test_stopwords_for_language() {
        let en = stopwords_for_language("en");
        assert!(en.contains(&"the"));

        let de = stopwords_for_language("de");
        assert!(de.contains(&"und"));

        let fr = stopwords_for_language("fr");
        assert!(fr.contains(&"avec"));

        let es = stopwords_for_language("es");
        assert!(es.contains(&"para"));
    }
}
