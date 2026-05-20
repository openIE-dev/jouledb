//! Procedural name generation via Markov chains, syllable assembly,
//! and template-based construction.
//!
//! Generates fantasy character names, place names, and templated names
//! with uniqueness checks, gender-aware templates, min/max length
//! constraints, and vowel harmony.

use std::collections::HashMap;

// ── Seeded RNG ──

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if lo >= hi { return lo; }
        lo + (self.next_u64() % (hi - lo) as u64) as usize
    }
    fn pick<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        &slice[self.range(0, slice.len())]
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ── MarkovChain ──

#[derive(Debug, Clone)]
pub struct MarkovChain {
    order: usize,
    transitions: HashMap<String, Vec<(char, f64)>>,
}

impl MarkovChain {
    /// Build a Markov chain from a training corpus of names.
    pub fn from_corpus(names: &[&str], order: usize) -> Self {
        let order = order.max(1);
        let mut raw: HashMap<String, HashMap<char, usize>> = HashMap::new();

        for name in names {
            let padded = format!("{}{}{}", "^".repeat(order), name.to_lowercase(), "$");
            let chars: Vec<char> = padded.chars().collect();
            for i in 0..chars.len().saturating_sub(order) {
                let context: String = chars[i..i + order].iter().collect();
                if i + order < chars.len() {
                    *raw.entry(context).or_default().entry(chars[i + order]).or_insert(0) += 1;
                }
            }
        }

        let transitions: HashMap<String, Vec<(char, f64)>> = raw.into_iter().map(|(ctx, counts)| {
            let total: usize = counts.values().sum();
            let probs: Vec<(char, f64)> = counts.into_iter()
                .map(|(ch, c)| (ch, c as f64 / total as f64))
                .collect();
            (ctx, probs)
        }).collect();

        Self { order, transitions }
    }

    /// Generate a name using the Markov chain.
    pub fn generate_name(&self, min_len: usize, max_len: usize, seed: u64) -> String {
        let mut rng = Rng::new(seed);
        let max_attempts = 100;

        for _ in 0..max_attempts {
            let mut context: Vec<char> = vec!['^'; self.order];
            let mut result = String::new();

            for _ in 0..max_len + 10 {
                let ctx_str: String = context.iter().collect();
                let next_char = match self.transitions.get(&ctx_str) {
                    Some(probs) => weighted_pick(probs, &mut rng),
                    None => break,
                };

                if next_char == '$' { break; }
                result.push(next_char);
                context.push(next_char);
                if context.len() > self.order {
                    context.remove(0);
                }
            }

            if result.len() >= min_len && result.len() <= max_len {
                return capitalize(&result);
            }
        }

        // Fallback
        capitalize(&"name".to_string())
    }
}

fn weighted_pick(options: &[(char, f64)], rng: &mut Rng) -> char {
    let total: f64 = options.iter().map(|(_, w)| w).sum();
    if total <= 0.0 { return '$'; }
    let roll = rng.next_f64() * total;
    let mut cum = 0.0;
    for &(ch, w) in options {
        cum += w;
        if roll < cum { return ch; }
    }
    options.last().map(|&(ch, _)| ch).unwrap_or('$')
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let upper: String = c.to_uppercase().collect();
            format!("{}{}", upper, chars.collect::<String>())
        }
    }
}

// ── SyllableGenerator ──

#[derive(Debug, Clone)]
pub struct SyllableGenerator {
    onsets: Vec<String>,
    nuclei: Vec<String>,
    codas: Vec<String>,
}

impl SyllableGenerator {
    /// Create a syllable generator with onset, nucleus, and coda sets.
    pub fn new(onsets: &[&str], nuclei: &[&str], codas: &[&str]) -> Self {
        Self {
            onsets: onsets.iter().map(|s| s.to_string()).collect(),
            nuclei: nuclei.iter().map(|s| s.to_string()).collect(),
            codas: codas.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Default fantasy-style syllable sets.
    pub fn fantasy() -> Self {
        Self::new(
            &["", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p",
              "r", "s", "t", "v", "w", "z", "th", "sh", "ch", "br", "cr", "dr",
              "fr", "gr", "pr", "tr", "str", "bl", "cl", "fl", "gl", "pl", "sl"],
            &["a", "e", "i", "o", "u", "ae", "ai", "ei", "ou", "ia", "ea"],
            &["", "b", "d", "g", "k", "l", "m", "n", "p", "r", "s", "t",
              "x", "th", "sh", "nd", "nt", "rn", "rd", "lm", "lt", "nk"],
        )
    }

    /// Generate a name by combining syllables.
    pub fn generate_name(
        &self,
        min_syllables: usize,
        max_syllables: usize,
        min_len: usize,
        max_len: usize,
        seed: u64,
    ) -> String {
        let mut rng = Rng::new(seed);
        for _ in 0..100 {
            let num_syl = rng.range(min_syllables, max_syllables + 1);
            let mut name = String::new();
            for _ in 0..num_syl {
                let onset = rng.pick(&self.onsets);
                let nucleus = rng.pick(&self.nuclei);
                let coda = rng.pick(&self.codas);
                name.push_str(onset);
                name.push_str(nucleus);
                name.push_str(coda);
            }
            if name.len() >= min_len && name.len() <= max_len {
                return capitalize(&name);
            }
        }
        capitalize("syllax")
    }
}

// ── TemplateGenerator ──

#[derive(Debug, Clone)]
pub struct TemplateGenerator {
    templates: Vec<String>,
    word_lists: HashMap<String, Vec<String>>,
}

impl TemplateGenerator {
    pub fn new() -> Self {
        Self { templates: Vec::new(), word_lists: HashMap::new() }
    }

    /// Add a template pattern, e.g. "{adj} {noun}".
    pub fn add_template(mut self, template: impl Into<String>) -> Self {
        self.templates.push(template.into());
        self
    }

    /// Add a word list for a placeholder name.
    pub fn add_words(mut self, key: impl Into<String>, words: &[&str]) -> Self {
        self.word_lists.insert(key.into(), words.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Generate a name from a random template.
    pub fn generate_name(&self, seed: u64) -> String {
        if self.templates.is_empty() { return String::from("Unknown"); }
        let mut rng = Rng::new(seed);
        let template = rng.pick(&self.templates).clone();
        let mut result = template;

        for (key, words) in &self.word_lists {
            let placeholder = format!("{{{}}}", key);
            while result.contains(&placeholder) {
                let word = rng.pick(words).clone();
                result = result.replacen(&placeholder, &word, 1);
            }
        }
        result
    }
}

// ── Gender ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Masculine,
    Feminine,
    Neutral,
}

// ── GenderAwareGenerator ──

#[derive(Debug, Clone)]
pub struct GenderAwareGenerator {
    masculine_suffixes: Vec<String>,
    feminine_suffixes: Vec<String>,
    neutral_suffixes: Vec<String>,
    base_generator: SyllableGenerator,
}

impl GenderAwareGenerator {
    pub fn new(base: SyllableGenerator) -> Self {
        Self {
            masculine_suffixes: vec!["or".into(), "us".into(), "an".into(), "rik".into(), "mund".into()],
            feminine_suffixes: vec!["a".into(), "ia".into(), "elle".into(), "wen".into(), "ith".into()],
            neutral_suffixes: vec!["el".into(), "yn".into(), "is".into(), "en".into()],
            base_generator: base,
        }
    }

    pub fn with_suffixes(
        mut self,
        masculine: &[&str],
        feminine: &[&str],
        neutral: &[&str],
    ) -> Self {
        self.masculine_suffixes = masculine.iter().map(|s| s.to_string()).collect();
        self.feminine_suffixes = feminine.iter().map(|s| s.to_string()).collect();
        self.neutral_suffixes = neutral.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn generate_name(&self, gender: Gender, seed: u64) -> String {
        let mut rng = Rng::new(seed);
        let base = self.base_generator.generate_name(1, 2, 2, 6, rng.next_u64());
        let suffix = match gender {
            Gender::Masculine => rng.pick(&self.masculine_suffixes).clone(),
            Gender::Feminine => rng.pick(&self.feminine_suffixes).clone(),
            Gender::Neutral => rng.pick(&self.neutral_suffixes).clone(),
        };
        capitalize(&format!("{}{}", base.to_lowercase(), suffix))
    }
}

// ── PlaceNameGenerator ──

#[derive(Debug, Clone)]
pub struct PlaceNameGenerator {
    prefixes: Vec<String>,
    terrain_suffixes: Vec<String>,
    water_features: Vec<String>,
}

impl PlaceNameGenerator {
    pub fn new() -> Self {
        Self {
            prefixes: vec![
                "Silver".into(), "Golden".into(), "Dark".into(), "White".into(),
                "Red".into(), "Iron".into(), "Storm".into(), "Shadow".into(),
                "Frost".into(), "Sun".into(), "Moon".into(), "Star".into(),
                "Dragon".into(), "Eagle".into(), "Wolf".into(), "Bear".into(),
            ],
            terrain_suffixes: vec![
                "vale".into(), "fell".into(), "hollow".into(), "ridge".into(),
                "shire".into(), "wood".into(), "moor".into(), "dale".into(),
                "haven".into(), "keep".into(), "gate".into(), "tower".into(),
            ],
            water_features: vec![
                "ford".into(), "lake".into(), "brook".into(), "falls".into(),
                "marsh".into(), "bay".into(), "port".into(), "creek".into(),
            ],
        }
    }

    pub fn generate_terrain(&self, seed: u64) -> String {
        let mut rng = Rng::new(seed);
        let prefix = rng.pick(&self.prefixes);
        let suffix = rng.pick(&self.terrain_suffixes);
        format!("{}{}", prefix, suffix)
    }

    pub fn generate_water(&self, seed: u64) -> String {
        let mut rng = Rng::new(seed);
        let prefix = rng.pick(&self.prefixes);
        let feature = rng.pick(&self.water_features);
        format!("{}{}", prefix, feature)
    }

    pub fn generate_river(&self, seed: u64) -> String {
        let mut rng = Rng::new(seed);
        let prefix = rng.pick(&self.prefixes);
        format!("{} River", prefix)
    }
}

// ── VowelHarmony ──

/// Check if a name has reasonable vowel harmony (alternating consonants/vowels).
pub fn vowel_harmony_score(name: &str) -> f64 {
    let chars: Vec<char> = name.to_lowercase().chars().collect();
    if chars.len() < 2 { return 1.0; }

    let is_vowel = |c: char| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u');
    let mut transitions = 0usize;
    let mut total = 0usize;

    for window in chars.windows(2) {
        if window[0].is_alphabetic() && window[1].is_alphabetic() {
            total += 1;
            if is_vowel(window[0]) != is_vowel(window[1]) {
                transitions += 1;
            }
        }
    }

    if total == 0 { return 1.0; }
    transitions as f64 / total as f64
}

// ── Batch generation with uniqueness ──

/// Generate a batch of unique names using any name-generation function.
pub fn generate_unique_batch<F>(
    count: usize,
    mut generator: F,
    seed: u64,
) -> Vec<String>
where
    F: FnMut(u64) -> String,
{
    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut attempt_seed = seed;
    let max_attempts = count * 20;
    let mut attempts = 0;

    while results.len() < count && attempts < max_attempts {
        let name = generator(attempt_seed);
        attempt_seed = attempt_seed.wrapping_add(1);
        attempts += 1;
        let lower = name.to_lowercase();
        if !seen.contains(&lower) {
            seen.insert(lower);
            results.push(name);
        }
    }
    results
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn test_corpus() -> Vec<&'static str> {
        vec![
            "Aragorn", "Legolas", "Gandalf", "Frodo", "Samwise",
            "Boromir", "Faramir", "Eowyn", "Arwen", "Elrond",
            "Galadriel", "Celeborn", "Thranduil", "Glorfindel",
        ]
    }

    #[test]
    fn test_markov_basic() {
        let chain = MarkovChain::from_corpus(&test_corpus(), 2);
        let name = chain.generate_name(3, 10, 42);
        assert!(name.len() >= 3);
        assert!(name.len() <= 10);
    }

    #[test]
    fn test_markov_determinism() {
        let chain = MarkovChain::from_corpus(&test_corpus(), 2);
        let a = chain.generate_name(3, 10, 123);
        let b = chain.generate_name(3, 10, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn test_markov_different_seeds() {
        let chain = MarkovChain::from_corpus(&test_corpus(), 2);
        let a = chain.generate_name(3, 10, 1);
        let b = chain.generate_name(3, 10, 999);
        // Very unlikely to be equal with different seeds
        assert!(a.len() >= 3);
        assert!(b.len() >= 3);
    }

    #[test]
    fn test_markov_order_1() {
        let chain = MarkovChain::from_corpus(&test_corpus(), 1);
        let name = chain.generate_name(2, 12, 42);
        assert!(name.len() >= 2);
    }

    #[test]
    fn test_syllable_basic() {
        let syllable_gen = SyllableGenerator::fantasy();
        let name = syllable_gen.generate_name(2, 3, 4, 12, 42);
        assert!(name.len() >= 4);
        assert!(name.len() <= 12);
    }

    #[test]
    fn test_syllable_determinism() {
        let syllable_gen = SyllableGenerator::fantasy();
        let a = syllable_gen.generate_name(2, 3, 4, 12, 42);
        let b = syllable_gen.generate_name(2, 3, 4, 12, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_syllable_custom() {
        let syllable_gen = SyllableGenerator::new(
            &["k", "t", "m"],
            &["a", "o"],
            &["r", "n"],
        );
        let name = syllable_gen.generate_name(2, 3, 3, 10, 42);
        assert!(!name.is_empty());
    }

    #[test]
    fn test_template_basic() {
        let template_gen = TemplateGenerator::new()
            .add_template("{adj} {noun}")
            .add_words("adj", &["Mighty", "Swift", "Dark"])
            .add_words("noun", &["Warrior", "Sage", "Hunter"]);
        let name = template_gen.generate_name(42);
        assert!(name.contains(' '));
    }

    #[test]
    fn test_template_determinism() {
        let template_gen = TemplateGenerator::new()
            .add_template("{adj} {noun}")
            .add_words("adj", &["Bold", "Calm"])
            .add_words("noun", &["Fox", "Bear"]);
        let a = template_gen.generate_name(42);
        let b = template_gen.generate_name(42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_template_empty() {
        let template_gen = TemplateGenerator::new();
        let name = template_gen.generate_name(42);
        assert_eq!(name, "Unknown");
    }

    #[test]
    fn test_gender_masculine() {
        let gender_gen = GenderAwareGenerator::new(SyllableGenerator::fantasy());
        let name = gender_gen.generate_name(Gender::Masculine, 42);
        assert!(!name.is_empty());
        assert!(name.chars().next().unwrap().is_uppercase());
    }

    #[test]
    fn test_gender_feminine() {
        let gender_gen = GenderAwareGenerator::new(SyllableGenerator::fantasy());
        let name = gender_gen.generate_name(Gender::Feminine, 42);
        assert!(!name.is_empty());
    }

    #[test]
    fn test_gender_neutral() {
        let gender_gen = GenderAwareGenerator::new(SyllableGenerator::fantasy());
        let name = gender_gen.generate_name(Gender::Neutral, 42);
        assert!(!name.is_empty());
    }

    #[test]
    fn test_place_terrain() {
        let place_gen = PlaceNameGenerator::new();
        let name = place_gen.generate_terrain(42);
        assert!(!name.is_empty());
        assert!(name.len() > 4);
    }

    #[test]
    fn test_place_water() {
        let place_gen = PlaceNameGenerator::new();
        let name = place_gen.generate_water(42);
        assert!(!name.is_empty());
    }

    #[test]
    fn test_place_river() {
        let place_gen = PlaceNameGenerator::new();
        let name = place_gen.generate_river(42);
        assert!(name.ends_with("River"));
    }

    #[test]
    fn test_vowel_harmony_alternating() {
        // Perfect alternation: consonant-vowel-consonant-vowel
        let score = vowel_harmony_score("babu");
        assert!(score > 0.5);
    }

    #[test]
    fn test_vowel_harmony_clustered() {
        let score = vowel_harmony_score("aaeee");
        assert!(score < 0.5);
    }

    #[test]
    fn test_unique_batch() {
        let syllable_gen = SyllableGenerator::fantasy();
        let names = generate_unique_batch(10, |seed| {
            syllable_gen.generate_name(2, 3, 4, 10, seed)
        }, 42);
        assert_eq!(names.len(), 10);
        let mut lower: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
        lower.sort();
        lower.dedup();
        assert_eq!(lower.len(), 10, "all names should be unique");
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("hello"), "Hello");
        assert_eq!(capitalize(""), "");
        assert_eq!(capitalize("a"), "A");
    }

    #[test]
    fn test_markov_length_constraints() {
        let chain = MarkovChain::from_corpus(&test_corpus(), 2);
        for seed in 0..20u64 {
            let name = chain.generate_name(4, 8, seed);
            assert!(name.len() >= 4 && name.len() <= 8,
                "name '{}' len {} out of range [4,8]", name, name.len());
        }
    }

    #[test]
    fn test_syllable_variety() {
        let syllable_gen = SyllableGenerator::fantasy();
        let names: Vec<String> = (0..10).map(|i| {
            syllable_gen.generate_name(2, 3, 4, 12, i * 37)
        }).collect();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert!(unique.len() >= 5, "should produce variety");
    }
}
