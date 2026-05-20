//! Spell checker with edit distance, keyboard-distance weighting, phonetic
//! matching (Soundex), custom dictionaries, suggestion ranking, context-aware
//! corrections, and word frequency tracking.
//!
//! Differs from the `spell_check` module (Rich Text & Editors) in that this
//! module targets search-engine query correction rather than editor integration.

use std::collections::{HashMap, HashSet};

// ── Errors ──────────────────────────────────────────────────────

/// Spell-checker errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SpellError {
    #[error("dictionary is empty")]
    EmptyDictionary,
    #[error("word is empty")]
    EmptyWord,
}

// ── Levenshtein ─────────────────────────────────────────────────

/// Standard Levenshtein edit distance.
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
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

// ── Keyboard distance ───────────────────────────────────────────

/// QWERTY keyboard layout for proximity weighting.
fn key_position(c: char) -> Option<(f64, f64)> {
    let row0 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'];
    let row1 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'];
    let row2 = ['z', 'x', 'c', 'v', 'b', 'n', 'm'];

    let lower = c.to_ascii_lowercase();

    for (i, &k) in row0.iter().enumerate() {
        if k == lower {
            return Some((i as f64, 0.0));
        }
    }
    for (i, &k) in row1.iter().enumerate() {
        if k == lower {
            return Some((i as f64 + 0.25, 1.0));
        }
    }
    for (i, &k) in row2.iter().enumerate() {
        if k == lower {
            return Some((i as f64 + 0.75, 2.0));
        }
    }
    None
}

/// Keyboard distance between two characters (Euclidean on QWERTY layout).
pub fn keyboard_distance(a: char, b: char) -> f64 {
    match (key_position(a), key_position(b)) {
        (Some((ax, ay)), Some((bx, by))) => {
            let dx = ax - bx;
            let dy = ay - by;
            (dx * dx + dy * dy).sqrt()
        }
        _ => 3.0, // max penalty for unknown
    }
}

/// Weighted edit distance incorporating keyboard proximity.
pub fn weighted_edit_distance(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let n = a_chars.len();
    let m = b_chars.len();

    if n == 0 {
        return m as f64;
    }
    if m == 0 {
        return n as f64;
    }

    let mut prev: Vec<f64> = (0..=m).map(|i| i as f64).collect();
    let mut curr = vec![0.0f64; m + 1];

    for i in 1..=n {
        curr[0] = i as f64;
        for j in 1..=m {
            let sub_cost = if a_chars[i - 1] == b_chars[j - 1] {
                0.0
            } else {
                // Nearby keys cost less.
                let kd = keyboard_distance(a_chars[i - 1], b_chars[j - 1]);
                (kd / 3.0).min(1.0) // normalize to [0, 1]
            };
            curr[j] = (prev[j] + 1.0)
                .min(curr[j - 1] + 1.0)
                .min(prev[j - 1] + sub_cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

// ── Soundex ─────────────────────────────────────────────────────

/// Compute the Soundex phonetic code for a word.
pub fn soundex(word: &str) -> String {
    if word.is_empty() {
        return String::new();
    }

    let chars: Vec<char> = word.chars().collect();
    let first = chars[0].to_ascii_uppercase();

    let code_for = |c: char| -> Option<char> {
        match c.to_ascii_lowercase() {
            'b' | 'f' | 'p' | 'v' => Some('1'),
            'c' | 'g' | 'j' | 'k' | 'q' | 's' | 'x' | 'z' => Some('2'),
            'd' | 't' => Some('3'),
            'l' => Some('4'),
            'm' | 'n' => Some('5'),
            'r' => Some('6'),
            _ => None,
        }
    };

    let mut result = String::with_capacity(4);
    result.push(first);

    let mut last_code = code_for(first);

    for &ch in &chars[1..] {
        let code = code_for(ch);
        if let Some(c) = code {
            if code != last_code {
                result.push(c);
                if result.len() == 4 {
                    break;
                }
            }
        }
        last_code = code;
    }

    // Pad with zeros.
    while result.len() < 4 {
        result.push('0');
    }

    result
}

// ── Suggestion ──────────────────────────────────────────────────

/// A spelling suggestion with scoring metadata.
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub word: String,
    pub edit_distance: usize,
    pub keyboard_score: f64,
    pub frequency: u64,
    pub phonetic_match: bool,
    /// Combined rank score (lower is better).
    pub rank_score: f64,
}

// ── SpellChecker ────────────────────────────────────────────────

/// Dictionary-based spell checker.
#[derive(Debug, Clone)]
pub struct SpellChecker {
    /// word -> frequency
    dictionary: HashMap<String, u64>,
    /// soundex code -> list of words
    phonetic_index: HashMap<String, Vec<String>>,
    /// Maximum edit distance for suggestions.
    max_edit_distance: usize,
    /// Maximum number of suggestions to return.
    max_suggestions: usize,
}

impl SpellChecker {
    /// Create a new spell checker.
    pub fn new() -> Self {
        Self {
            dictionary: HashMap::new(),
            phonetic_index: HashMap::new(),
            max_edit_distance: 2,
            max_suggestions: 5,
        }
    }

    /// Set maximum edit distance.
    pub fn with_max_distance(mut self, d: usize) -> Self {
        self.max_edit_distance = d;
        self
    }

    /// Set maximum suggestions.
    pub fn with_max_suggestions(mut self, n: usize) -> Self {
        self.max_suggestions = n;
        self
    }

    /// Number of words in the dictionary.
    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }

    /// Add a word to the dictionary with a frequency count.
    pub fn add_word(&mut self, word: &str, frequency: u64) {
        let lower = word.to_lowercase();
        *self.dictionary.entry(lower.clone()).or_insert(0) += frequency;

        let code = soundex(&lower);
        if !code.is_empty() {
            let words = self.phonetic_index.entry(code).or_default();
            if !words.contains(&lower) {
                words.push(lower);
            }
        }
    }

    /// Add many words with frequency 1.
    pub fn add_words(&mut self, words: &[&str]) {
        for w in words {
            self.add_word(w, 1);
        }
    }

    /// Add words from text (split on whitespace).
    pub fn add_text(&mut self, text: &str) {
        for word in text.split_whitespace() {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect();
            if !cleaned.is_empty() {
                self.add_word(&cleaned, 1);
            }
        }
    }

    /// Check if a word is in the dictionary.
    pub fn is_correct(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.dictionary.contains_key(&lower)
    }

    /// Get the frequency of a word.
    pub fn frequency(&self, word: &str) -> u64 {
        let lower = word.to_lowercase();
        self.dictionary.get(&lower).copied().unwrap_or(0)
    }

    /// Get spelling suggestions for a misspelled word.
    pub fn suggest(&self, word: &str) -> Vec<Suggestion> {
        let lower = word.to_lowercase();
        if lower.is_empty() {
            return Vec::new();
        }

        let word_soundex = soundex(&lower);

        let mut suggestions: Vec<Suggestion> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        // Exclude the word itself from suggestions.
        seen.insert(lower.clone());

        // Phase 1: edit-distance candidates.
        for (dict_word, &freq) in &self.dictionary {
            let ed = levenshtein(&lower, dict_word);
            if ed <= self.max_edit_distance && ed > 0 {
                if seen.insert(dict_word.clone()) {
                    let kbd = weighted_edit_distance(&lower, dict_word);
                    let phon = soundex(dict_word) == word_soundex;
                    suggestions.push(Suggestion {
                        word: dict_word.clone(),
                        edit_distance: ed,
                        keyboard_score: kbd,
                        frequency: freq,
                        phonetic_match: phon,
                        rank_score: 0.0, // computed below
                    });
                }
            }
        }

        // Phase 2: phonetic candidates (may add more).
        if let Some(phon_words) = self.phonetic_index.get(&word_soundex) {
            for pw in phon_words {
                if seen.insert(pw.clone()) {
                    let ed = levenshtein(&lower, pw);
                    let freq = self.dictionary.get(pw).copied().unwrap_or(0);
                    let kbd = weighted_edit_distance(&lower, pw);
                    suggestions.push(Suggestion {
                        word: pw.clone(),
                        edit_distance: ed,
                        keyboard_score: kbd,
                        frequency: freq,
                        phonetic_match: true,
                        rank_score: 0.0,
                    });
                }
            }
        }

        // Rank: lower is better.
        let max_freq = suggestions.iter().map(|s| s.frequency).max().unwrap_or(1).max(1);
        for s in &mut suggestions {
            let freq_factor = 1.0 - (s.frequency as f64 / max_freq as f64);
            let phon_bonus = if s.phonetic_match { -0.5 } else { 0.0 };
            s.rank_score = (s.edit_distance as f64) * 2.0
                + s.keyboard_score
                + freq_factor
                + phon_bonus;
        }

        suggestions.sort_by(|a, b| {
            a.rank_score
                .partial_cmp(&b.rank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.truncate(self.max_suggestions);
        suggestions
    }

    /// Context-aware suggestion: given a list of context words, boost suggestions
    /// that commonly co-occur (simple frequency-based heuristic).
    pub fn suggest_in_context(&self, word: &str, context_words: &[&str]) -> Vec<Suggestion> {
        let mut suggestions = self.suggest(word);

        // Boost suggestions that share a phonetic code with any context word.
        let context_codes: HashSet<String> = context_words
            .iter()
            .map(|w| soundex(w))
            .collect();

        for s in &mut suggestions {
            let code = soundex(&s.word);
            if context_codes.contains(&code) {
                s.rank_score -= 1.0; // boost
            }
        }

        suggestions.sort_by(|a, b| {
            a.rank_score
                .partial_cmp(&b.rank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions
    }

    /// Check a sentence and return corrections for misspelled words.
    pub fn check_text(&self, text: &str) -> Vec<(String, Vec<Suggestion>)> {
        let mut results = Vec::new();
        for word in text.split_whitespace() {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect();
            if cleaned.is_empty() {
                continue;
            }
            if !self.is_correct(&cleaned) {
                let suggestions = self.suggest(&cleaned);
                results.push((cleaned, suggestions));
            }
        }
        results
    }

    /// Get all words in the dictionary.
    pub fn words(&self) -> Vec<String> {
        let mut words: Vec<String> = self.dictionary.keys().cloned().collect();
        words.sort();
        words
    }
}

impl Default for SpellChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_checker() -> SpellChecker {
        let mut sc = SpellChecker::new().with_max_distance(2).with_max_suggestions(5);
        sc.add_words(&[
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "cat", "hello", "world", "rust", "programming", "language",
            "search", "engine", "spell", "check", "correct", "dictionary",
        ]);
        // Boost some frequencies.
        sc.add_word("the", 100);
        sc.add_word("hello", 50);
        sc
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_one_edit() {
        assert_eq!(levenshtein("hello", "hallo"), 1);
        assert_eq!(levenshtein("hello", "hell"), 1);
        assert_eq!(levenshtein("hello", "helloo"), 1);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn test_keyboard_distance_same() {
        assert_eq!(keyboard_distance('a', 'a'), 0.0);
    }

    #[test]
    fn test_keyboard_distance_adjacent() {
        let d = keyboard_distance('a', 's');
        assert!(d > 0.0 && d < 2.0);
    }

    #[test]
    fn test_keyboard_distance_far() {
        let d_close = keyboard_distance('a', 's');
        let d_far = keyboard_distance('a', 'p');
        assert!(d_far > d_close);
    }

    #[test]
    fn test_weighted_edit_distance() {
        // Adjacent key typo should cost less than distant key typo
        let d_adj = weighted_edit_distance("hello", "jello"); // h->j are close
        let d_far = weighted_edit_distance("hello", "zello"); // h->z are far
        assert!(d_adj <= d_far);
    }

    #[test]
    fn test_soundex_basic() {
        assert_eq!(soundex("Robert"), "R163");
        assert_eq!(soundex("Rupert"), "R163");
    }

    #[test]
    fn test_soundex_empty() {
        assert_eq!(soundex(""), "");
    }

    #[test]
    fn test_soundex_short() {
        let code = soundex("A");
        assert_eq!(code.len(), 4);
        assert_eq!(code, "A000");
    }

    #[test]
    fn test_is_correct() {
        let sc = build_checker();
        assert!(sc.is_correct("hello"));
        assert!(sc.is_correct("Hello")); // case insensitive
        assert!(!sc.is_correct("helllo"));
    }

    #[test]
    fn test_frequency() {
        let sc = build_checker();
        assert!(sc.frequency("the") > 1);
        assert_eq!(sc.frequency("nonexistent"), 0);
    }

    #[test]
    fn test_suggest_returns_results() {
        let sc = build_checker();
        let sugs = sc.suggest("helo");
        assert!(!sugs.is_empty());
        // "hello" should be among suggestions
        assert!(sugs.iter().any(|s| s.word == "hello"));
    }

    #[test]
    fn test_suggest_correct_word_no_self() {
        let sc = build_checker();
        let sugs = sc.suggest("hello");
        // Should not suggest the same word as a correction
        assert!(sugs.iter().all(|s| s.word != "hello" || s.edit_distance > 0));
    }

    #[test]
    fn test_suggest_max_limit() {
        let sc = build_checker();
        let sugs = sc.suggest("helo");
        assert!(sugs.len() <= 5);
    }

    #[test]
    fn test_suggest_ranking() {
        let sc = build_checker();
        let sugs = sc.suggest("helo");
        // Verify sorted by rank_score ascending
        for window in sugs.windows(2) {
            assert!(window[0].rank_score <= window[1].rank_score);
        }
    }

    #[test]
    fn test_check_text() {
        let sc = build_checker();
        let errors = sc.check_text("the quik brown fx");
        // "quik" and "fx" should be flagged
        assert!(errors.iter().any(|(w, _)| w == "quik"));
        assert!(errors.iter().any(|(w, _)| w == "fx"));
        // "the" and "brown" should NOT be flagged
        assert!(errors.iter().all(|(w, _)| w != "the" && w != "brown"));
    }

    #[test]
    fn test_context_suggestion() {
        let sc = build_checker();
        let sugs = sc.suggest_in_context("helo", &["world"]);
        assert!(!sugs.is_empty());
    }

    #[test]
    fn test_add_text() {
        let mut sc = SpellChecker::new();
        sc.add_text("hello world rust programming");
        assert!(sc.is_correct("hello"));
        assert!(sc.is_correct("rust"));
        assert_eq!(sc.dictionary_size(), 4);
    }

    #[test]
    fn test_phonetic_matching() {
        let mut sc = SpellChecker::new().with_max_distance(3);
        sc.add_word("smith", 10);
        sc.add_word("smyth", 5);
        // Both have same soundex: S530
        assert_eq!(soundex("smith"), soundex("smyth"));
        let sugs = sc.suggest("smth");
        // At least one should be a phonetic match
        assert!(sugs.iter().any(|s| s.phonetic_match));
    }

    #[test]
    fn test_words_sorted() {
        let sc = build_checker();
        let words = sc.words();
        for window in words.windows(2) {
            assert!(window[0] <= window[1]);
        }
    }

    #[test]
    fn test_default_trait() {
        let sc = SpellChecker::default();
        assert_eq!(sc.dictionary_size(), 0);
    }
}
