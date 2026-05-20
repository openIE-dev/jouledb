//! Stop word management: built-in English list, custom lists, case-insensitive
//! matching, and stop word removal from token streams.

use std::collections::HashSet;

// ── Built-in English stop words (125+) ───────────────────────────

const ENGLISH_STOP_WORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "just", "let's", "like", "me",
    "might", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shall",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "us", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "will", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
];

/// Hint for which language's stop words to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    English,
    Spanish,
    French,
    German,
    Italian,
    Portuguese,
    Custom,
}

// ── Stop word set ────────────────────────────────────────────────

/// A case-insensitive set of stop words.
#[derive(Debug, Clone)]
pub struct StopWordSet {
    words: HashSet<String>,
    language: Language,
}

impl StopWordSet {
    /// Create an empty set.
    pub fn new(language: Language) -> Self {
        Self {
            words: HashSet::new(),
            language,
        }
    }

    /// Create a set with the built-in English stop words.
    pub fn english() -> Self {
        let words = ENGLISH_STOP_WORDS
            .iter()
            .map(|w| w.to_lowercase())
            .collect();
        Self {
            words,
            language: Language::English,
        }
    }

    /// Create a set from a custom word list.
    pub fn from_words(words: &[&str], language: Language) -> Self {
        let set = words.iter().map(|w| w.to_lowercase()).collect();
        Self {
            words: set,
            language,
        }
    }

    /// Create a Spanish stop word set (common subset).
    pub fn spanish() -> Self {
        let words: &[&str] = &[
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
            "las", "por", "un", "para", "con", "no", "una", "su", "al",
            "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "este",
            "si", "porque", "esta", "entre", "cuando", "muy", "sin",
            "sobre", "tambien", "me", "hasta", "hay", "donde", "quien",
            "desde", "todo", "nos", "durante", "todos", "uno", "les",
            "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
            "esto", "mi", "antes", "algunos", "que", "unos", "yo",
            "otro", "otras", "otra", "el", "tan", "poco", "ella",
        ];
        Self::from_words(words, Language::Spanish)
    }

    /// Create a French stop word set (common subset).
    pub fn french() -> Self {
        let words: &[&str] = &[
            "le", "la", "les", "de", "des", "du", "un", "une", "et",
            "en", "que", "qui", "dans", "ce", "il", "je", "ne", "se",
            "pas", "plus", "son", "par", "au", "sur", "avec", "tout",
            "mais", "ou", "comme", "on", "pour", "nous", "vous", "elle",
            "est", "sont", "leur", "cette", "ces", "ont", "sa", "ses",
            "mon", "mes", "aux", "te", "lui", "me", "si", "tu",
        ];
        Self::from_words(words, Language::French)
    }

    /// Create a German stop word set (common subset).
    pub fn german() -> Self {
        let words: &[&str] = &[
            "der", "die", "das", "und", "in", "den", "von", "zu", "mit",
            "ist", "auf", "ein", "eine", "dem", "nicht", "sich", "des",
            "es", "als", "an", "auch", "so", "dass", "kann", "aber",
            "um", "am", "aus", "wenn", "wie", "man", "nach", "noch",
            "nur", "da", "hat", "bei", "im", "ich", "er", "sie", "wir",
        ];
        Self::from_words(words, Language::German)
    }

    /// Add a word to the set.
    pub fn add(&mut self, word: &str) {
        self.words.insert(word.to_lowercase());
    }

    /// Remove a word from the set.
    pub fn remove(&mut self, word: &str) -> bool {
        self.words.remove(&word.to_lowercase())
    }

    /// Check if a word is a stop word (case-insensitive).
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.words.contains(&word.to_lowercase())
    }

    /// Number of stop words in the set.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub fn language(&self) -> Language {
        self.language
    }

    /// Merge another set into this one.
    pub fn merge(&mut self, other: &StopWordSet) {
        for word in &other.words {
            self.words.insert(word.clone());
        }
    }

    /// Return an iterator over all stop words (in no particular order).
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.words.iter().map(|s| s.as_str())
    }
}

impl Default for StopWordSet {
    fn default() -> Self {
        Self::english()
    }
}

// ── Token stream filtering ───────────────────────────────────────

/// A simple token for stop word filtering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextToken {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Remove stop words from a token stream.
pub fn remove_stop_words(tokens: &[TextToken], stop_words: &StopWordSet) -> Vec<TextToken> {
    tokens
        .iter()
        .filter(|t| !stop_words.is_stop_word(&t.text))
        .cloned()
        .collect()
}

/// Remove stop words from a slice of strings.
pub fn filter_strings(words: &[&str], stop_words: &StopWordSet) -> Vec<String> {
    words
        .iter()
        .filter(|w| !stop_words.is_stop_word(w))
        .map(|w| w.to_string())
        .collect()
}

/// Remove stop words from a text string, preserving spacing.
pub fn remove_stop_words_text(text: &str, stop_words: &StopWordSet) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let filtered: Vec<&str> = words
        .into_iter()
        .filter(|w| {
            // Strip punctuation for matching but keep the original form.
            let clean: String = w.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect();
            !stop_words.is_stop_word(&clean)
        })
        .collect();
    filtered.join(" ")
}

// ── Language detection heuristic ─────────────────────────────────

/// A simple heuristic to guess the language of a text based on stop word frequency.
pub fn detect_language_hint(text: &str) -> Language {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    if words.is_empty() {
        return Language::English;
    }

    let sets = [
        (Language::English, StopWordSet::english()),
        (Language::Spanish, StopWordSet::spanish()),
        (Language::French, StopWordSet::french()),
        (Language::German, StopWordSet::german()),
    ];

    let mut best_lang = Language::English;
    let mut best_count = 0usize;

    for (lang, set) in &sets {
        let count = words.iter().filter(|w| set.is_stop_word(w)).count();
        if count > best_count {
            best_count = count;
            best_lang = *lang;
        }
    }

    best_lang
}

/// Get the default stop word set for a detected or specified language.
pub fn get_stop_words_for_language(lang: Language) -> StopWordSet {
    match lang {
        Language::English => StopWordSet::english(),
        Language::Spanish => StopWordSet::spanish(),
        Language::French => StopWordSet::french(),
        Language::German => StopWordSet::german(),
        _ => StopWordSet::english(),
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_stop_words_count() {
        let set = StopWordSet::english();
        assert!(set.len() >= 100, "Expected 100+ stop words, got {}", set.len());
    }

    #[test]
    fn test_is_stop_word_case_insensitive() {
        let set = StopWordSet::english();
        assert!(set.is_stop_word("the"));
        assert!(set.is_stop_word("The"));
        assert!(set.is_stop_word("THE"));
    }

    #[test]
    fn test_non_stop_word() {
        let set = StopWordSet::english();
        assert!(!set.is_stop_word("computer"));
        assert!(!set.is_stop_word("algorithm"));
    }

    #[test]
    fn test_custom_stop_word_set() {
        let set = StopWordSet::from_words(&["foo", "bar", "baz"], Language::Custom);
        assert!(set.is_stop_word("foo"));
        assert!(set.is_stop_word("FOO"));
        assert!(!set.is_stop_word("qux"));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_add_remove() {
        let mut set = StopWordSet::new(Language::Custom);
        set.add("hello");
        assert!(set.is_stop_word("hello"));
        assert!(set.remove("hello"));
        assert!(!set.is_stop_word("hello"));
    }

    #[test]
    fn test_remove_stop_words_tokens() {
        let set = StopWordSet::english();
        let tokens = vec![
            TextToken { text: "The".to_string(), start: 0, end: 3 },
            TextToken { text: "quick".to_string(), start: 4, end: 9 },
            TextToken { text: "fox".to_string(), start: 10, end: 13 },
            TextToken { text: "is".to_string(), start: 14, end: 16 },
            TextToken { text: "here".to_string(), start: 17, end: 21 },
        ];
        let filtered = remove_stop_words(&tokens, &set);
        let texts: Vec<&str> = filtered.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, ["quick", "fox"]);
    }

    #[test]
    fn test_filter_strings() {
        let set = StopWordSet::english();
        let words = &["the", "quick", "brown", "fox", "is", "a"];
        let filtered = filter_strings(words, &set);
        assert_eq!(filtered, ["quick", "brown", "fox"]);
    }

    #[test]
    fn test_remove_stop_words_text() {
        let set = StopWordSet::english();
        let result = remove_stop_words_text("The quick brown fox is a test", &set);
        assert_eq!(result, "quick brown fox test");
    }

    #[test]
    fn test_spanish_stop_words() {
        let set = StopWordSet::spanish();
        assert!(set.is_stop_word("de"));
        assert!(set.is_stop_word("la"));
        assert!(!set.is_stop_word("computadora"));
    }

    #[test]
    fn test_french_stop_words() {
        let set = StopWordSet::french();
        assert!(set.is_stop_word("le"));
        assert!(set.is_stop_word("la"));
        assert!(!set.is_stop_word("ordinateur"));
    }

    #[test]
    fn test_german_stop_words() {
        let set = StopWordSet::german();
        assert!(set.is_stop_word("der"));
        assert!(set.is_stop_word("die"));
        assert!(!set.is_stop_word("computer"));
    }

    #[test]
    fn test_detect_english() {
        let lang = detect_language_hint("The quick brown fox is not a very good animal");
        assert_eq!(lang, Language::English);
    }

    #[test]
    fn test_detect_spanish() {
        let lang = detect_language_hint("de la que el en y a los del se las por un para con");
        assert_eq!(lang, Language::Spanish);
    }

    #[test]
    fn test_merge_sets() {
        let mut set = StopWordSet::new(Language::Custom);
        set.add("foo");
        let other = StopWordSet::from_words(&["bar", "baz"], Language::Custom);
        set.merge(&other);
        assert!(set.is_stop_word("foo"));
        assert!(set.is_stop_word("bar"));
        assert!(set.is_stop_word("baz"));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_empty_set() {
        let set = StopWordSet::new(Language::Custom);
        assert!(set.is_empty());
        assert!(!set.is_stop_word("anything"));
    }

    #[test]
    fn test_get_stop_words_for_language() {
        let set = get_stop_words_for_language(Language::English);
        assert!(set.is_stop_word("the"));
        let set = get_stop_words_for_language(Language::French);
        assert!(set.is_stop_word("le"));
    }

    #[test]
    fn test_detect_language_empty() {
        let lang = detect_language_hint("");
        assert_eq!(lang, Language::English);
    }

    #[test]
    fn test_contractions_in_stop_words() {
        let set = StopWordSet::english();
        assert!(set.is_stop_word("don't"));
        assert!(set.is_stop_word("can't"));
        assert!(set.is_stop_word("won't"));
    }
}
