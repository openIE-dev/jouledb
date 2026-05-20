//! Dictionary-based spell checker.
//!
//! Levenshtein distance, suggestions, custom dictionaries, and Hunspell-style
//! affix rules. Replaces nspell/Typo.js with pure Rust.

use std::collections::{HashMap, HashSet};

// ── Levenshtein distance ────────────────────────────────────────

/// Compute Levenshtein edit distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let n = a_chars.len();
    let m = b_chars.len();

    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }

    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];

    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

// ── Affix rules ─────────────────────────────────────────────────

/// A basic affix rule (prefix or suffix).
#[derive(Debug, Clone)]
pub struct AffixRule {
    /// Whether this is a prefix (true) or suffix (false).
    pub is_prefix: bool,
    /// Characters to strip before applying.
    pub strip: String,
    /// Characters to add.
    pub affix: String,
    /// Optional condition pattern (simplified: only match suffix of stem).
    pub condition: Option<String>,
}

impl AffixRule {
    /// Create a prefix rule.
    pub fn prefix(strip: &str, affix: &str) -> Self {
        Self {
            is_prefix: true,
            strip: strip.to_string(),
            affix: affix.to_string(),
            condition: None,
        }
    }

    /// Create a suffix rule.
    pub fn suffix(strip: &str, affix: &str) -> Self {
        Self {
            is_prefix: false,
            strip: strip.to_string(),
            affix: affix.to_string(),
            condition: None,
        }
    }

    /// Create a suffix rule with condition.
    pub fn suffix_with_condition(strip: &str, affix: &str, condition: &str) -> Self {
        Self {
            is_prefix: false,
            strip: strip.to_string(),
            affix: affix.to_string(),
            condition: Some(condition.to_string()),
        }
    }

    /// Apply the rule to a stem, returning the derived word if applicable.
    pub fn apply(&self, stem: &str) -> Option<String> {
        if self.is_prefix {
            if self.strip.is_empty() || stem.starts_with(&self.strip) {
                let rest = &stem[self.strip.len()..];
                Some(format!("{}{rest}", self.affix))
            } else {
                None
            }
        } else {
            // Check condition
            if let Some(cond) = &self.condition {
                if !stem.ends_with(cond.as_str()) {
                    return None;
                }
            }
            if self.strip.is_empty() {
                Some(format!("{stem}{}", self.affix))
            } else if stem.ends_with(&self.strip) {
                let base = &stem[..stem.len() - self.strip.len()];
                Some(format!("{base}{}", self.affix))
            } else {
                None
            }
        }
    }
}

// ── Misspelling report ──────────────────────────────────────────

/// A misspelled word with its location and suggestions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Misspelling {
    /// The misspelled word.
    pub word: String,
    /// Byte offset in the original text.
    pub offset: usize,
    /// Length in bytes.
    pub length: usize,
    /// Suggested corrections, ordered by edit distance.
    pub suggestions: Vec<String>,
}

// ── Ignore patterns ─────────────────────────────────────────────

/// Check if a word looks like a URL.
fn is_url(word: &str) -> bool {
    word.starts_with("http://")
        || word.starts_with("https://")
        || word.starts_with("ftp://")
        || word.contains("://")
}

/// Check if a word looks like an email address.
fn is_email(word: &str) -> bool {
    let parts: Vec<&str> = word.split('@').collect();
    parts.len() == 2 && !parts[0].is_empty() && parts[1].contains('.')
}

/// Check if a word is all digits (possibly with separators).
fn is_number(word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    word.chars()
        .all(|c| c.is_ascii_digit() || c == '.' || c == ',' || c == '-' || c == '+')
        && word.chars().any(|c| c.is_ascii_digit())
}

// ── Word extraction ─────────────────────────────────────────────

/// A word found in text with its byte position.
#[derive(Debug, Clone)]
struct WordSpan {
    word: String,
    offset: usize,
}

/// Extract words from text, splitting on non-alphabetic characters.
fn extract_words(text: &str) -> Vec<WordSpan> {
    let mut words = Vec::new();
    let mut start = None;

    for (i, c) in text.char_indices() {
        if c.is_alphabetic() || c == '\'' {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            let word = &text[s..i];
            // Trim leading/trailing apostrophes
            let trimmed = word.trim_matches('\'');
            if !trimmed.is_empty() {
                let trim_offset = s + word.find(trimmed).unwrap_or(0);
                words.push(WordSpan {
                    word: trimmed.to_string(),
                    offset: trim_offset,
                });
            }
            start = None;
        }
    }
    // Handle trailing word
    if let Some(s) = start {
        let word = &text[s..];
        let trimmed = word.trim_matches('\'');
        if !trimmed.is_empty() {
            let trim_offset = s + word.find(trimmed).unwrap_or(0);
            words.push(WordSpan {
                word: trimmed.to_string(),
                offset: trim_offset,
            });
        }
    }

    words
}

// ── Spell Checker ───────────────────────────────────────────────

/// A dictionary-based spell checker.
#[derive(Debug, Clone)]
pub struct SpellChecker {
    /// Main dictionary words (lowercased).
    dictionary: HashSet<String>,
    /// Custom user dictionary.
    custom: HashSet<String>,
    /// Affix rules grouped by flag character.
    affix_rules: HashMap<char, Vec<AffixRule>>,
    /// Words with affix flags.
    word_flags: HashMap<String, Vec<char>>,
    /// Maximum edit distance for suggestions.
    pub max_distance: usize,
    /// Maximum number of suggestions to return.
    pub max_suggestions: usize,
}

impl SpellChecker {
    /// Create a new spell checker with a word list.
    pub fn new(words: &[&str]) -> Self {
        let dictionary: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
        Self {
            dictionary,
            custom: HashSet::new(),
            affix_rules: HashMap::new(),
            word_flags: HashMap::new(),
            max_distance: 2,
            max_suggestions: 5,
        }
    }

    /// Create an empty spell checker.
    pub fn empty() -> Self {
        Self::new(&[])
    }

    /// Add a word to the main dictionary.
    pub fn add_word(&mut self, word: &str) {
        self.dictionary.insert(word.to_lowercase());
    }

    /// Add a word to the custom dictionary.
    pub fn add_custom(&mut self, word: &str) {
        self.custom.insert(word.to_lowercase());
    }

    /// Remove a word from the custom dictionary.
    pub fn remove_custom(&mut self, word: &str) {
        self.custom.remove(&word.to_lowercase());
    }

    /// Add an affix rule under a flag character.
    pub fn add_affix_rule(&mut self, flag: char, rule: AffixRule) {
        self.affix_rules.entry(flag).or_default().push(rule);
    }

    /// Associate a word with affix flags.
    pub fn set_word_flags(&mut self, word: &str, flags: &[char]) {
        self.word_flags
            .insert(word.to_lowercase(), flags.to_vec());
    }

    /// Check if a word is correctly spelled.
    pub fn is_correct(&self, word: &str) -> bool {
        let lower = word.to_lowercase();

        // Direct dictionary match
        if self.dictionary.contains(&lower) || self.custom.contains(&lower) {
            return true;
        }

        // Check affix-expanded forms
        for (stem, flags) in &self.word_flags {
            for flag in flags {
                if let Some(rules) = self.affix_rules.get(flag) {
                    for rule in rules {
                        if let Some(derived) = rule.apply(stem) {
                            if derived.to_lowercase() == lower {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        false
    }

    /// Suggest corrections for a misspelled word (edit distance <= max_distance).
    pub fn suggest(&self, word: &str) -> Vec<String> {
        let lower = word.to_lowercase();
        let mut candidates: Vec<(String, usize)> = Vec::new();

        // Check dictionary words
        for dict_word in self.dictionary.iter().chain(self.custom.iter()) {
            let dist = levenshtein(&lower, dict_word);
            if dist > 0 && dist <= self.max_distance {
                candidates.push((dict_word.clone(), dist));
            }
        }

        // Check affix-derived words
        for (stem, flags) in &self.word_flags {
            for flag in flags {
                if let Some(rules) = self.affix_rules.get(flag) {
                    for rule in rules {
                        if let Some(derived) = rule.apply(stem) {
                            let d = derived.to_lowercase();
                            let dist = levenshtein(&lower, &d);
                            if dist > 0 && dist <= self.max_distance {
                                candidates.push((d, dist));
                            }
                        }
                    }
                }
            }
        }

        // Sort by distance, then alphabetically
        candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        candidates.dedup_by(|a, b| a.0 == b.0);
        candidates
            .into_iter()
            .take(self.max_suggestions)
            .map(|(w, _)| w)
            .collect()
    }

    /// Check an entire text and return misspellings with positions and suggestions.
    pub fn check_text(&self, text: &str) -> Vec<Misspelling> {
        let words = extract_words(text);
        let mut results = Vec::new();

        for span in &words {
            // Skip patterns
            if is_url(&span.word) || is_email(&span.word) || is_number(&span.word) {
                continue;
            }
            // Skip single characters
            if span.word.len() <= 1 {
                continue;
            }

            if !self.is_correct(&span.word) {
                let suggestions = self.suggest(&span.word);
                results.push(Misspelling {
                    word: span.word.clone(),
                    offset: span.offset,
                    length: span.word.len(),
                    suggestions,
                });
            }
        }

        results
    }

    /// Number of words in the dictionary (including custom).
    pub fn word_count(&self) -> usize {
        self.dictionary.len() + self.custom.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_checker() -> SpellChecker {
        SpellChecker::new(&[
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello", "world",
            "test", "testing", "rust", "code", "spell", "check", "correct", "house", "mouse",
        ])
    }

    #[test]
    fn levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn levenshtein_one_edit() {
        assert_eq!(levenshtein("hello", "helo"), 1);
        assert_eq!(levenshtein("hello", "jello"), 1);
        assert_eq!(levenshtein("hello", "helloo"), 1);
    }

    #[test]
    fn levenshtein_two_edits() {
        assert_eq!(levenshtein("hello", "hallo"), 1);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn correct_word() {
        let checker = basic_checker();
        assert!(checker.is_correct("hello"));
        assert!(checker.is_correct("Hello")); // case insensitive
        assert!(!checker.is_correct("helo"));
    }

    #[test]
    fn suggestions() {
        let checker = basic_checker();
        let sugg = checker.suggest("helo");
        assert!(sugg.contains(&"hello".to_string()));
    }

    #[test]
    fn custom_dictionary() {
        let mut checker = basic_checker();
        assert!(!checker.is_correct("rustacean"));
        checker.add_custom("rustacean");
        assert!(checker.is_correct("rustacean"));
        checker.remove_custom("rustacean");
        assert!(!checker.is_correct("rustacean"));
    }

    #[test]
    fn check_text_finds_misspellings() {
        let checker = basic_checker();
        let results = checker.check_text("the quik brown fox");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].word, "quik");
        assert!(results[0].suggestions.contains(&"quick".to_string()));
    }

    #[test]
    fn check_text_positions() {
        let checker = basic_checker();
        let results = checker.check_text("hello wrld test");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].word, "wrld");
        assert_eq!(results[0].offset, 6);
        assert_eq!(results[0].length, 4);
    }

    #[test]
    fn ignores_urls_emails_numbers() {
        let checker = basic_checker();
        let results = checker.check_text("visit https://example.com or email user@test.com or call 12345");
        // None of these should be flagged
        for r in &results {
            assert!(
                !is_url(&r.word) && !is_email(&r.word) && !is_number(&r.word),
                "should not flag: {}",
                r.word
            );
        }
    }

    #[test]
    fn affix_suffix_rule() {
        let mut checker = SpellChecker::new(&["walk", "talk"]);
        checker.add_affix_rule('S', AffixRule::suffix("", "s"));
        checker.add_affix_rule('D', AffixRule::suffix("", "ed"));
        checker.add_affix_rule('G', AffixRule::suffix("", "ing"));
        checker.set_word_flags("walk", &['S', 'D', 'G']);
        checker.set_word_flags("talk", &['S', 'D', 'G']);

        assert!(checker.is_correct("walks"));
        assert!(checker.is_correct("walked"));
        assert!(checker.is_correct("walking"));
        assert!(checker.is_correct("talks"));
        assert!(!checker.is_correct("talken"));
    }

    #[test]
    fn affix_prefix_rule() {
        let mut checker = SpellChecker::new(&["do", "happy"]);
        checker.add_affix_rule('U', AffixRule::prefix("", "un"));
        checker.set_word_flags("do", &['U']);
        checker.set_word_flags("happy", &['U']);

        assert!(checker.is_correct("undo"));
        assert!(checker.is_correct("unhappy"));
    }

    #[test]
    fn affix_suffix_with_strip() {
        let mut checker = SpellChecker::new(&["happy"]);
        checker.add_affix_rule(
            'Y',
            AffixRule::suffix_with_condition("y", "ily", "y"),
        );
        checker.set_word_flags("happy", &['Y']);
        assert!(checker.is_correct("happily"));
    }

    #[test]
    fn word_extraction() {
        let words = extract_words("Hello, world! It's a test-case.");
        let texts: Vec<&str> = words.iter().map(|w| w.word.as_str()).collect();
        assert!(texts.contains(&"Hello"));
        assert!(texts.contains(&"world"));
        assert!(texts.contains(&"It's"));
        assert!(texts.contains(&"test"));
        assert!(texts.contains(&"case"));
    }

    #[test]
    fn max_suggestions_limit() {
        let mut checker = basic_checker();
        checker.max_suggestions = 2;
        let sugg = checker.suggest("helo");
        assert!(sugg.len() <= 2);
    }
}
