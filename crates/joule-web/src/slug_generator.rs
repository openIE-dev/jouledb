//! URL slug generation — Unicode to ASCII transliteration, special character
//! removal, space-to-hyphen, consecutive hyphen collapse, max length
//! truncation, uniqueness suffix, custom separator, and stop word removal.
//!
//! Pure-Rust replacement for slugify, speakingurl, url-slug, and similar
//! Node.js slug generation libraries.

use std::collections::HashSet;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from slug generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlugError {
    EmptyInput,
    InvalidConfig(String),
}

impl fmt::Display for SlugError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "cannot generate slug from empty input"),
            Self::InvalidConfig(msg) => write!(f, "invalid slug config: {msg}"),
        }
    }
}

impl std::error::Error for SlugError {}

// ── Configuration ───────────────────────────────────────────────

/// Configuration for slug generation.
#[derive(Debug, Clone)]
pub struct SlugConfig {
    /// Separator character (default: '-').
    pub separator: char,
    /// Maximum slug length (0 = no limit).
    pub max_length: usize,
    /// Whether to remove stop words.
    pub remove_stop_words: bool,
    /// Custom stop words to remove (in addition to defaults).
    pub custom_stop_words: Vec<String>,
    /// Whether to transliterate Unicode characters.
    pub transliterate: bool,
    /// Whether to lowercase the slug.
    pub lowercase: bool,
}

impl Default for SlugConfig {
    fn default() -> Self {
        Self {
            separator: '-',
            max_length: 0,
            remove_stop_words: false,
            custom_stop_words: Vec::new(),
            transliterate: true,
            lowercase: true,
        }
    }
}

impl SlugConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_separator(mut self, sep: char) -> Self {
        self.separator = sep;
        self
    }

    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = max;
        self
    }

    pub fn with_stop_word_removal(mut self, enable: bool) -> Self {
        self.remove_stop_words = enable;
        self
    }

    pub fn with_custom_stop_words(mut self, words: Vec<String>) -> Self {
        self.custom_stop_words = words;
        self
    }

    pub fn with_transliterate(mut self, enable: bool) -> Self {
        self.transliterate = enable;
        self
    }

    pub fn with_lowercase(mut self, enable: bool) -> Self {
        self.lowercase = enable;
        self
    }
}

// ── Default Stop Words ──────────────────────────────────────────

/// English stop words commonly removed from slugs.
const DEFAULT_STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "is", "it", "as", "be", "was", "are", "were", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "this", "that", "these", "those", "not", "no", "nor", "so", "if", "then",
];

// ── Unicode Transliteration ─────────────────────────────────────

/// Transliterate common Unicode characters to ASCII equivalents.
fn transliterate_char(ch: char) -> Option<&'static str> {
    Some(match ch {
        // Latin letters with diacritics
        '\u{00C0}'..='\u{00C5}' => "A",   // À Á Â Ã Ä Å
        '\u{00C6}' => "AE",               // Æ
        '\u{00C7}' => "C",                // Ç
        '\u{00C8}'..='\u{00CB}' => "E",   // È É Ê Ë
        '\u{00CC}'..='\u{00CF}' => "I",   // Ì Í Î Ï
        '\u{00D0}' => "D",                // Ð
        '\u{00D1}' => "N",                // Ñ
        '\u{00D2}'..='\u{00D6}' => "O",   // Ò Ó Ô Õ Ö
        '\u{00D8}' => "O",                // Ø
        '\u{00D9}'..='\u{00DC}' => "U",   // Ù Ú Û Ü
        '\u{00DD}' => "Y",                // Ý
        '\u{00DE}' => "Th",               // Þ
        '\u{00DF}' => "ss",               // ß
        '\u{00E0}'..='\u{00E5}' => "a",   // à á â ã ä å
        '\u{00E6}' => "ae",               // æ
        '\u{00E7}' => "c",                // ç
        '\u{00E8}'..='\u{00EB}' => "e",   // è é ê ë
        '\u{00EC}'..='\u{00EF}' => "i",   // ì í î ï
        '\u{00F0}' => "d",                // ð
        '\u{00F1}' => "n",                // ñ
        '\u{00F2}'..='\u{00F6}' => "o",   // ò ó ô õ ö
        '\u{00F8}' => "o",                // ø
        '\u{00F9}'..='\u{00FC}' => "u",   // ù ú û ü
        '\u{00FD}' | '\u{00FF}' => "y",   // ý ÿ
        '\u{00FE}' => "th",               // þ

        // Additional common accented characters
        '\u{0100}' | '\u{0102}' => "A",   // Ā Ă
        '\u{0101}' | '\u{0103}' => "a",   // ā ă
        '\u{0106}' | '\u{0108}' | '\u{010C}' => "C", // Ć Ĉ Č
        '\u{0107}' | '\u{0109}' | '\u{010D}' => "c", // ć ĉ č
        '\u{010E}' | '\u{0110}' => "D",   // Ď Đ
        '\u{010F}' | '\u{0111}' => "d",   // ď đ
        '\u{0112}' | '\u{0114}' | '\u{0116}' | '\u{0118}' | '\u{011A}' => "E",
        '\u{0113}' | '\u{0115}' | '\u{0117}' | '\u{0119}' | '\u{011B}' => "e",
        '\u{011E}' | '\u{0120}' => "G",   // Ğ Ġ
        '\u{011F}' | '\u{0121}' => "g",   // ğ ġ
        '\u{0130}' => "I",                // İ
        '\u{0131}' => "i",                // ı
        '\u{0141}' => "L",                // Ł
        '\u{0142}' => "l",                // ł
        '\u{0143}' | '\u{0147}' => "N",   // Ń Ň
        '\u{0144}' | '\u{0148}' => "n",   // ń ň
        '\u{0150}' | '\u{0152}' => "O",   // Ő Œ (mapped to O for slug purposes)
        '\u{0151}' | '\u{0153}' => "o",   // ő œ
        '\u{0158}' | '\u{0154}' => "R",   // Ř Ŕ
        '\u{0159}' | '\u{0155}' => "r",   // ř ŕ
        '\u{015A}' | '\u{015E}' | '\u{0160}' => "S", // Ś Ş Š
        '\u{015B}' | '\u{015F}' | '\u{0161}' => "s", // ś ş š
        '\u{0164}' | '\u{0162}' => "T",   // Ť Ţ
        '\u{0165}' | '\u{0163}' => "t",   // ť ţ
        '\u{016E}' | '\u{0170}' => "U",   // Ů Ű
        '\u{016F}' | '\u{0171}' => "u",   // ů ű
        '\u{0178}' => "Y",                // Ÿ
        '\u{0179}' | '\u{017B}' | '\u{017D}' => "Z", // Ź Ż Ž
        '\u{017A}' | '\u{017C}' | '\u{017E}' => "z", // ź ż ž

        // Common symbols to text
        '\u{0026}' => "and",              // &
        '\u{0040}' => "at",               // @

        _ => return None,
    })
}

// ── Slug Generation ─────────────────────────────────────────────

/// Generate a slug from the given text with the given config.
pub fn generate_slug(input: &str, config: &SlugConfig) -> Result<String, SlugError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(SlugError::EmptyInput);
    }

    let sep = config.separator;

    // Build stop-word set
    let stop_words: HashSet<&str> = if config.remove_stop_words {
        let mut set: HashSet<&str> = DEFAULT_STOP_WORDS.iter().copied().collect();
        for w in &config.custom_stop_words {
            set.insert(w.as_str());
        }
        set
    } else {
        HashSet::new()
    };

    // Process character by character
    let mut slug = String::new();
    let mut last_was_sep = true; // treat start as separator to avoid leading sep

    for ch in trimmed.chars() {
        if ch.is_ascii_alphanumeric() {
            if config.lowercase {
                for lc in ch.to_lowercase() {
                    slug.push(lc);
                }
            } else {
                slug.push(ch);
            }
            last_was_sep = false;
        } else if ch == ' ' || ch == '-' || ch == '_' || ch == '/' || ch == '\\' || ch == '.' {
            if !last_was_sep {
                slug.push(sep);
                last_was_sep = true;
            }
        } else if config.transliterate {
            if let Some(replacement) = transliterate_char(ch) {
                if config.lowercase {
                    slug.push_str(&replacement.to_lowercase());
                } else {
                    slug.push_str(replacement);
                }
                last_was_sep = false;
            }
            // else: skip non-ASCII, non-transliteratable chars
        }
    }

    // Remove trailing separator
    while slug.ends_with(sep) {
        slug.pop();
    }

    // Remove stop words if enabled
    if config.remove_stop_words && !stop_words.is_empty() {
        let sep_str = sep.to_string();
        let parts: Vec<&str> = slug.split(sep).collect();
        let filtered: Vec<&str> = parts
            .into_iter()
            .filter(|part| !stop_words.contains(part))
            .collect();
        slug = filtered.join(&sep_str);
    }

    // Truncate to max length
    if config.max_length > 0 && slug.len() > config.max_length {
        // Try to truncate on a separator boundary
        let truncated = &slug[..config.max_length];
        if let Some(last_sep) = truncated.rfind(sep) {
            slug = truncated[..last_sep].to_string();
        } else {
            slug = truncated.to_string();
        }
    }

    // Remove trailing separator again after truncation
    while slug.ends_with(sep) {
        slug.pop();
    }

    if slug.is_empty() {
        return Err(SlugError::EmptyInput);
    }

    Ok(slug)
}

/// Generate a slug with default configuration.
pub fn slugify(input: &str) -> Result<String, SlugError> {
    generate_slug(input, &SlugConfig::default())
}

// ── Uniqueness ──────────────────────────────────────────────────

/// Ensure a slug is unique by appending a numeric suffix if needed.
///
/// Given a base slug and a set of existing slugs, returns either the
/// base slug (if unique) or the slug with `-1`, `-2`, etc. appended.
pub fn make_unique(base_slug: &str, existing: &HashSet<String>, sep: char) -> String {
    if !existing.contains(base_slug) {
        return base_slug.to_string();
    }

    let mut counter = 1u64;
    loop {
        let candidate = format!("{}{}{}", base_slug, sep, counter);
        if !existing.contains(&candidate) {
            return candidate;
        }
        counter += 1;
    }
}

/// Generate a unique slug, automatically suffixing if needed.
pub fn slugify_unique(
    input: &str,
    existing: &HashSet<String>,
    config: &SlugConfig,
) -> Result<String, SlugError> {
    let base = generate_slug(input, config)?;
    Ok(make_unique(&base, existing, config.separator))
}

// ── Batch Slug Generation ───────────────────────────────────────

/// Generate unique slugs for a batch of inputs.
pub fn slugify_batch(inputs: &[&str], config: &SlugConfig) -> Result<Vec<String>, SlugError> {
    let mut used = HashSet::new();
    let mut result = Vec::with_capacity(inputs.len());

    for input in inputs {
        let slug = slugify_unique(input, &used, config)?;
        used.insert(slug.clone());
        result.push(slug);
    }

    Ok(result)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_slug() {
        assert_eq!(slugify("Hello World").unwrap(), "hello-world");
    }

    #[test]
    fn test_special_characters_removed() {
        assert_eq!(
            slugify("Hello! World# $2026").unwrap(),
            "hello-world-2026"
        );
    }

    #[test]
    fn test_consecutive_spaces() {
        assert_eq!(slugify("Hello   World").unwrap(), "hello-world");
    }

    #[test]
    fn test_unicode_transliteration() {
        assert_eq!(slugify("Crème Brûlée").unwrap(), "creme-brulee");
    }

    #[test]
    fn test_german_characters() {
        assert_eq!(slugify("Straße").unwrap(), "strasse");
    }

    #[test]
    fn test_scandinavian_characters() {
        assert_eq!(slugify("Ångström").unwrap(), "angstrom");
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(slugify(""), Err(SlugError::EmptyInput));
    }

    #[test]
    fn test_only_special_chars() {
        assert_eq!(slugify("!!!???"), Err(SlugError::EmptyInput));
    }

    #[test]
    fn test_max_length() {
        let config = SlugConfig::new().with_max_length(10);
        let slug = generate_slug("Hello World from Rust Programming", &config).unwrap();
        assert!(slug.len() <= 10);
    }

    #[test]
    fn test_max_length_word_boundary() {
        let config = SlugConfig::new().with_max_length(12);
        let slug = generate_slug("Hello Beautiful World", &config).unwrap();
        // Should truncate on separator boundary
        assert!(slug.len() <= 12);
        assert!(!slug.ends_with('-'));
    }

    #[test]
    fn test_custom_separator() {
        let config = SlugConfig::new().with_separator('_');
        let slug = generate_slug("Hello World", &config).unwrap();
        assert_eq!(slug, "hello_world");
    }

    #[test]
    fn test_stop_word_removal() {
        let config = SlugConfig::new().with_stop_word_removal(true);
        let slug = generate_slug("The Quick Brown Fox and the Lazy Dog", &config).unwrap();
        assert!(!slug.contains("-the-"));
        assert!(!slug.contains("-and-"));
        assert!(slug.contains("quick"));
        assert!(slug.contains("fox"));
    }

    #[test]
    fn test_custom_stop_words() {
        let config = SlugConfig::new()
            .with_stop_word_removal(true)
            .with_custom_stop_words(vec!["hello".into()]);
        let slug = generate_slug("Hello World", &config).unwrap();
        assert!(!slug.contains("hello"));
    }

    #[test]
    fn test_no_transliteration() {
        let config = SlugConfig::new().with_transliterate(false);
        let slug = generate_slug("Café Résumé", &config).unwrap();
        // Without transliteration, accented chars are dropped
        assert_eq!(slug, "caf-rsum");
    }

    #[test]
    fn test_preserve_case() {
        let config = SlugConfig::new().with_lowercase(false);
        let slug = generate_slug("Hello World", &config).unwrap();
        assert_eq!(slug, "Hello-World");
    }

    #[test]
    fn test_make_unique_no_collision() {
        let existing = HashSet::new();
        assert_eq!(make_unique("hello", &existing, '-'), "hello");
    }

    #[test]
    fn test_make_unique_with_collision() {
        let mut existing = HashSet::new();
        existing.insert("hello".into());
        assert_eq!(make_unique("hello", &existing, '-'), "hello-1");
    }

    #[test]
    fn test_make_unique_multiple_collisions() {
        let mut existing = HashSet::new();
        existing.insert("hello".into());
        existing.insert("hello-1".into());
        existing.insert("hello-2".into());
        assert_eq!(make_unique("hello", &existing, '-'), "hello-3");
    }

    #[test]
    fn test_slugify_unique() {
        let mut existing = HashSet::new();
        existing.insert("hello-world".into());
        let config = SlugConfig::default();
        let slug = slugify_unique("Hello World", &existing, &config).unwrap();
        assert_eq!(slug, "hello-world-1");
    }

    #[test]
    fn test_slugify_batch() {
        let inputs = vec!["Hello World", "Hello World", "Hello World"];
        let config = SlugConfig::default();
        let slugs = slugify_batch(&inputs, &config).unwrap();
        assert_eq!(slugs[0], "hello-world");
        assert_eq!(slugs[1], "hello-world-1");
        assert_eq!(slugs[2], "hello-world-2");
    }

    #[test]
    fn test_ampersand_transliteration() {
        assert_eq!(slugify("Rock & Roll").unwrap(), "rock-and-roll");
    }

    #[test]
    fn test_at_sign_transliteration() {
        assert_eq!(slugify("user @ domain").unwrap(), "user-at-domain");
    }

    #[test]
    fn test_leading_trailing_spaces() {
        assert_eq!(slugify("  Hello World  ").unwrap(), "hello-world");
    }

    #[test]
    fn test_numbers_preserved() {
        assert_eq!(slugify("Article 42 is great").unwrap(), "article-42-is-great");
    }

    #[test]
    fn test_mixed_separators() {
        assert_eq!(slugify("Hello/World-Test_Page").unwrap(), "hello-world-test-page");
    }

    #[test]
    fn test_ae_ligature() {
        assert_eq!(slugify("Ærodynamic").unwrap(), "aerodynamic");
    }
}
