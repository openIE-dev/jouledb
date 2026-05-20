//! Word stemming: Porter stemmer for English with custom rule support.
//!
//! Implements the Porter stemming algorithm with step-by-step suffix stripping,
//! a rule engine for custom stemming rules, batch stemming, and a stem cache
//! for repeated lookups.

use std::collections::HashMap;

// ── Porter stemmer ───────────────────────────────────────────────

/// Measure: the number of VC (vowel-consonant) sequences in a stem.
fn measure(word: &str) -> usize {
    let mut m = 0;
    let mut in_vowel = false;
    for c in word.chars() {
        if is_vowel_in_context(c, in_vowel) {
            in_vowel = true;
        } else {
            if in_vowel {
                m += 1;
            }
            in_vowel = false;
        }
    }
    m
}

fn is_vowel_char(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

fn is_vowel_in_context(c: char, _prev_was_vowel: bool) -> bool {
    is_vowel_char(c)
}

/// Check if word contains a vowel.
fn has_vowel(word: &str) -> bool {
    word.chars().any(is_vowel_char)
}

/// Check if word ends with a double consonant.
fn ends_double_consonant(word: &str) -> bool {
    let bytes = word.as_bytes();
    if bytes.len() < 2 {
        return false;
    }
    let last = bytes[bytes.len() - 1];
    let prev = bytes[bytes.len() - 2];
    last == prev && !is_vowel_char(last as char)
}

/// Check if word ends with consonant-vowel-consonant where the last consonant
/// is not w, x, or y.
fn ends_cvc(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 3 {
        return false;
    }
    let c3 = chars[chars.len() - 1];
    let c2 = chars[chars.len() - 2];
    let c1 = chars[chars.len() - 3];
    !is_vowel_char(c3) && is_vowel_char(c2) && !is_vowel_char(c1)
        && c3 != 'w' && c3 != 'x' && c3 != 'y'
}

fn replace_suffix(word: &str, suffix: &str, replacement: &str) -> String {
    if let Some(stem) = word.strip_suffix(suffix) {
        format!("{stem}{replacement}")
    } else {
        word.to_string()
    }
}

fn strip_suffix_if<F>(word: &str, suffix: &str, replacement: &str, condition: F) -> String
where
    F: Fn(&str) -> bool,
{
    if let Some(stem) = word.strip_suffix(suffix) {
        if condition(stem) {
            return format!("{stem}{replacement}");
        }
    }
    word.to_string()
}

/// Porter stemmer step 1a: plurals.
fn step1a(word: &str) -> String {
    if let Some(stem) = word.strip_suffix("sses") {
        return format!("{stem}ss");
    }
    if let Some(stem) = word.strip_suffix("ies") {
        return format!("{stem}i");
    }
    if word.ends_with("ss") {
        return word.to_string();
    }
    if let Some(stem) = word.strip_suffix('s') {
        if !stem.is_empty() {
            return stem.to_string();
        }
    }
    word.to_string()
}

/// Porter stemmer step 1b: -eed, -ed, -ing.
fn step1b(word: &str) -> String {
    if let Some(stem) = word.strip_suffix("eed") {
        if measure(stem) > 0 {
            return format!("{stem}ee");
        }
        return word.to_string();
    }

    let mut result = word.to_string();
    let mut did_strip = false;

    if let Some(stem) = word.strip_suffix("ed") {
        if has_vowel(stem) {
            result = stem.to_string();
            did_strip = true;
        }
    }

    if !did_strip {
        if let Some(stem) = word.strip_suffix("ing") {
            if has_vowel(stem) {
                result = stem.to_string();
                did_strip = true;
            }
        }
    }

    if did_strip {
        if result.ends_with("at") || result.ends_with("bl") || result.ends_with("iz") {
            result.push('e');
        } else if ends_double_consonant(&result) {
            let last = result.chars().last().unwrap();
            if last != 'l' && last != 's' && last != 'z' {
                result.pop();
            }
        } else if measure(&result) == 1 && ends_cvc(&result) {
            result.push('e');
        }
    }

    result
}

/// Porter stemmer step 1c: y → i.
fn step1c(word: &str) -> String {
    if let Some(stem) = word.strip_suffix('y') {
        if has_vowel(stem) {
            return format!("{stem}i");
        }
    }
    word.to_string()
}

/// Porter stemmer step 2: map double suffixes to single.
fn step2(word: &str) -> String {
    let mappings: &[(&str, &str)] = &[
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

    for (suffix, replacement) in mappings {
        if word.ends_with(suffix) {
            let result = strip_suffix_if(word, suffix, replacement, |stem| measure(stem) > 0);
            if result != word {
                return result;
            }
            return word.to_string();
        }
    }
    word.to_string()
}

/// Porter stemmer step 3: -icate, -ative, etc.
fn step3(word: &str) -> String {
    let mappings: &[(&str, &str)] = &[
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];

    for (suffix, replacement) in mappings {
        if word.ends_with(suffix) {
            let result = strip_suffix_if(word, suffix, replacement, |stem| measure(stem) > 0);
            if result != word {
                return result;
            }
            return word.to_string();
        }
    }
    word.to_string()
}

/// Porter stemmer step 4: remove -ance, -ence, etc.
fn step4(word: &str) -> String {
    let suffixes: &[&str] = &[
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
        "ous", "ive", "ize",
    ];

    for suffix in suffixes {
        if let Some(stem) = word.strip_suffix(suffix) {
            if measure(stem) > 1 {
                // Special case for -ion: stem must end in s or t.
                if *suffix == "ion" {
                    if stem.ends_with('s') || stem.ends_with('t') {
                        return stem.to_string();
                    }
                    continue;
                }
                return stem.to_string();
            }
        }
    }
    word.to_string()
}

/// Porter stemmer step 5a: remove trailing -e.
fn step5a(word: &str) -> String {
    if let Some(stem) = word.strip_suffix('e') {
        let m = measure(stem);
        if m > 1 {
            return stem.to_string();
        }
        if m == 1 && !ends_cvc(stem) {
            return stem.to_string();
        }
    }
    word.to_string()
}

/// Porter stemmer step 5b: remove double l if m > 1.
fn step5b(word: &str) -> String {
    if word.ends_with("ll") && measure(word) > 1 {
        return word[..word.len() - 1].to_string();
    }
    word.to_string()
}

/// Apply the full Porter stemming algorithm to a single word.
pub fn porter_stem(word: &str) -> String {
    if word.len() <= 2 {
        return word.to_lowercase();
    }

    let w = word.to_lowercase();
    let w = step1a(&w);
    let w = step1b(&w);
    let w = step1c(&w);
    let w = step2(&w);
    let w = step3(&w);
    let w = step4(&w);
    let w = step5a(&w);
    step5b(&w)
}

// ── Custom stemmer rules ─────────────────────────────────────────

/// A custom stemming rule: if word ends with `suffix`, replace with `replacement`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StemRule {
    pub suffix: String,
    pub replacement: String,
    /// Minimum stem length after removing suffix (0 = no minimum).
    pub min_stem_len: usize,
}

/// Rule engine for custom stemming.
#[derive(Debug, Clone)]
pub struct StemmerRulesEngine {
    rules: Vec<StemRule>,
}

impl StemmerRulesEngine {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, suffix: &str, replacement: &str, min_stem_len: usize) {
        self.rules.push(StemRule {
            suffix: suffix.to_string(),
            replacement: replacement.to_string(),
            min_stem_len,
        });
    }

    /// Apply rules in order; first matching rule wins.
    pub fn stem(&self, word: &str) -> String {
        let lower = word.to_lowercase();
        for rule in &self.rules {
            if let Some(stem) = lower.strip_suffix(rule.suffix.as_str()) {
                if stem.len() >= rule.min_stem_len {
                    return format!("{stem}{}", rule.replacement);
                }
            }
        }
        lower
    }

    pub fn rules(&self) -> &[StemRule] {
        &self.rules
    }
}

impl Default for StemmerRulesEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Cached stemmer ───────────────────────────────────────────────

/// Stemmer with caching for repeated lookups.
#[derive(Debug, Clone)]
pub struct CachedStemmer {
    cache: HashMap<String, String>,
    custom_rules: Option<StemmerRulesEngine>,
}

impl CachedStemmer {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            custom_rules: None,
        }
    }

    pub fn with_rules(rules: StemmerRulesEngine) -> Self {
        Self {
            cache: HashMap::new(),
            custom_rules: Some(rules),
        }
    }

    pub fn stem(&mut self, word: &str) -> String {
        let lower = word.to_lowercase();
        if let Some(cached) = self.cache.get(&lower) {
            return cached.clone();
        }

        let result = if let Some(rules) = &self.custom_rules {
            rules.stem(&lower)
        } else {
            porter_stem(&lower)
        };

        self.cache.insert(lower, result.clone());
        result
    }

    /// Batch stem a list of words.
    pub fn stem_batch(&mut self, words: &[&str]) -> Vec<String> {
        words.iter().map(|w| self.stem(w)).collect()
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for CachedStemmer {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch stem without caching.
pub fn stem_batch(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| porter_stem(w)).collect()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_porter_caresses() {
        assert_eq!(porter_stem("caresses"), "caress");
    }

    #[test]
    fn test_porter_ponies() {
        assert_eq!(porter_stem("ponies"), "poni");
    }

    #[test]
    fn test_porter_cats() {
        assert_eq!(porter_stem("cats"), "cat");
    }

    #[test]
    fn test_porter_feed() {
        // feed → step1a removes trailing s? No. feed → feed through step1a.
        // step1b: -eed with m(f)=0 → no change. "feed" stays.
        assert_eq!(porter_stem("feed"), "feed");
    }

    #[test]
    fn test_porter_agreed() {
        // agreed → step1a no change, step1b: -eed, stem "agr", m=1>0 → agree
        assert_eq!(porter_stem("agreed"), "agre");
    }

    #[test]
    fn test_porter_plastered() {
        // plastered → step1b: -ed, stem=plaster, has_vowel → plaster
        assert_eq!(porter_stem("plastered"), "plaster");
    }

    #[test]
    fn test_porter_motoring() {
        // motoring → step1b: -ing, stem=motor, has_vowel → motor
        assert_eq!(porter_stem("motoring"), "motor");
    }

    #[test]
    fn test_porter_sing() {
        // sing → step1b: -ing, stem=s, has_vowel(s)=false → no change
        assert_eq!(porter_stem("sing"), "sing");
    }

    #[test]
    fn test_porter_happy() {
        // happy → step1c: -y with vowel → happi
        assert_eq!(porter_stem("happy"), "happi");
    }

    #[test]
    fn test_porter_short_words() {
        assert_eq!(porter_stem("a"), "a");
        assert_eq!(porter_stem("be"), "be");
    }

    #[test]
    fn test_measure() {
        assert_eq!(measure("tr"), 0);
        // trouble: [C]tr (V)ou (C)bl [V]e = m=1
        assert_eq!(measure("trouble"), 1);
        assert_eq!(measure("troubles"), 2);
        // oaten: [V]oa (C)t (V)e (C)n = m=2
        assert_eq!(measure("oaten"), 2);
    }

    #[test]
    fn test_has_vowel() {
        assert!(has_vowel("hello"));
        assert!(!has_vowel("rhythms".strip_suffix("ythms").unwrap())); // "rh" has no vowel? "rh" → no
        assert!(!has_vowel("bcd"));
    }

    #[test]
    fn test_custom_rules_engine() {
        let mut engine = StemmerRulesEngine::new();
        engine.add_rule("ing", "", 3);
        engine.add_rule("ed", "", 3);

        assert_eq!(engine.stem("running"), "runn");
        assert_eq!(engine.stem("played"), "play");
        assert_eq!(engine.stem("bed"), "bed"); // stem "b" < min_stem_len 3
    }

    #[test]
    fn test_cached_stemmer() {
        let mut stemmer = CachedStemmer::new();
        let s1 = stemmer.stem("running");
        let s2 = stemmer.stem("running");
        assert_eq!(s1, s2);
        assert_eq!(stemmer.cache_size(), 1);
    }

    #[test]
    fn test_cached_stemmer_batch() {
        let mut stemmer = CachedStemmer::new();
        let results = stemmer.stem_batch(&["cats", "running", "cats"]);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], results[2]); // same word same stem
        assert_eq!(stemmer.cache_size(), 2);
    }

    #[test]
    fn test_cached_stemmer_with_rules() {
        let mut engine = StemmerRulesEngine::new();
        engine.add_rule("ly", "", 2);
        let mut stemmer = CachedStemmer::with_rules(engine);
        assert_eq!(stemmer.stem("quickly"), "quick");
    }

    #[test]
    fn test_stem_batch_fn() {
        let results = stem_batch(&["cats", "dogs"]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "cat");
        assert_eq!(results[1], "dog");
    }

    #[test]
    fn test_clear_cache() {
        let mut stemmer = CachedStemmer::new();
        stemmer.stem("hello");
        assert_eq!(stemmer.cache_size(), 1);
        stemmer.clear_cache();
        assert_eq!(stemmer.cache_size(), 0);
    }

    #[test]
    fn test_ends_double_consonant() {
        assert!(ends_double_consonant("hopp"));
        assert!(!ends_double_consonant("hop"));
        assert!(!ends_double_consonant("bee")); // ee are vowels
    }

    #[test]
    fn test_replace_suffix() {
        assert_eq!(replace_suffix("running", "ing", ""), "runn");
        assert_eq!(replace_suffix("cats", "xyz", ""), "cats");
    }
}
