//! Named entity recognition (rule-based): pattern and dictionary NER,
//! entity types (PERSON/ORG/LOCATION/DATE/MONEY/EMAIL/URL), span extraction,
//! entity linking, overlap resolution, and custom entity types.

use std::collections::{HashMap, HashSet};

// ── Entity types ─────────────────────────────────────────────────

/// Standard named entity types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Time,
    Money,
    Percent,
    Email,
    Url,
    Phone,
    Number,
    Custom(u32),
}

impl EntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::Person => "PERSON",
            EntityType::Organization => "ORG",
            EntityType::Location => "LOCATION",
            EntityType::Date => "DATE",
            EntityType::Time => "TIME",
            EntityType::Money => "MONEY",
            EntityType::Percent => "PERCENT",
            EntityType::Email => "EMAIL",
            EntityType::Url => "URL",
            EntityType::Phone => "PHONE",
            EntityType::Number => "NUMBER",
            EntityType::Custom(_) => "CUSTOM",
        }
    }
}

/// A recognized entity span.
#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    /// Byte offset start in original text.
    pub start: usize,
    /// Byte offset end (exclusive) in original text.
    pub end: usize,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Optional linked entity id/label.
    pub linked_id: Option<String>,
}

// ── Pattern rules ────────────────────────────────────────────────

/// A pattern-matching rule for NER.
#[derive(Debug, Clone)]
pub struct NerRule {
    pub entity_type: EntityType,
    pub kind: RuleKind,
    pub confidence: f64,
}

/// Kind of matching rule.
#[derive(Debug, Clone)]
pub enum RuleKind {
    /// Match against a set of known strings (dictionary lookup).
    Dictionary(HashSet<String>),
    /// Match words starting with uppercase (title case heuristic).
    TitleCase,
    /// Custom pattern function name (matched by built-in patterns).
    BuiltinPattern(BuiltinPattern),
}

/// Built-in regex-like patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPattern {
    Email,
    Url,
    Money,
    Date,
    Time,
    Phone,
    Percent,
    Number,
}

// ── Pattern matchers ─────────────────────────────────────────────

fn match_email(text: &str, start: usize) -> Option<(usize, usize)> {
    // Simple email pattern: chars@chars.chars
    let bytes = text.as_bytes();
    let mut i = start;

    // Local part: alphanumeric, dots, underscores, hyphens.
    let local_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'.' || bytes[i] == b'_' || bytes[i] == b'-' || bytes[i] == b'+') {
        i += 1;
    }
    if i == local_start || i >= bytes.len() || bytes[i] != b'@' {
        return None;
    }
    i += 1; // skip @

    // Domain part.
    let domain_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'.' || bytes[i] == b'-') {
        i += 1;
    }
    if i == domain_start {
        return None;
    }

    // Must have at least one dot in domain.
    let domain = &text[domain_start..i];
    if !domain.contains('.') {
        return None;
    }

    Some((start, i))
}

fn match_url(text: &str, start: usize) -> Option<(usize, usize)> {
    let slice = &text[start..];
    if !slice.starts_with("http://") && !slice.starts_with("https://") && !slice.starts_with("www.") {
        return None;
    }

    let mut i = start;
    let bytes = text.as_bytes();
    while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' && bytes[i] != b')' && bytes[i] != b']' {
        i += 1;
    }

    // Strip trailing punctuation.
    while i > start && matches!(bytes[i - 1], b'.' | b',' | b';' | b':' | b'!' | b'?') {
        i -= 1;
    }

    if i > start + 4 {
        Some((start, i))
    } else {
        None
    }
}

fn match_money(text: &str, start: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    if start >= bytes.len() {
        return None;
    }

    // Currency symbol: $, EUR, GBP, etc.
    let mut i = start;
    let has_symbol = bytes[i] == b'$';
    if has_symbol {
        i += 1;
    } else {
        return None;
    }

    // Number with optional commas and decimal.
    let num_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b',' || bytes[i] == b'.') {
        i += 1;
    }
    if i == num_start {
        return None;
    }

    Some((start, i))
}

fn match_date(text: &str, start: usize) -> Option<(usize, usize)> {
    let slice = &text[start..];

    // Patterns: MM/DD/YYYY, YYYY-MM-DD, DD-MM-YYYY
    let bytes = slice.as_bytes();
    if bytes.len() < 8 {
        return None;
    }

    // YYYY-MM-DD or YYYY/MM/DD
    if bytes.len() >= 10
        && bytes[0].is_ascii_digit() && bytes[1].is_ascii_digit()
        && bytes[2].is_ascii_digit() && bytes[3].is_ascii_digit()
        && (bytes[4] == b'-' || bytes[4] == b'/')
        && bytes[5].is_ascii_digit() && bytes[6].is_ascii_digit()
        && (bytes[7] == b'-' || bytes[7] == b'/')
        && bytes[8].is_ascii_digit() && bytes[9].is_ascii_digit()
    {
        return Some((start, start + 10));
    }

    // MM/DD/YYYY or DD/MM/YYYY
    if bytes.len() >= 10
        && bytes[0].is_ascii_digit() && bytes[1].is_ascii_digit()
        && (bytes[2] == b'/' || bytes[2] == b'-')
        && bytes[3].is_ascii_digit() && bytes[4].is_ascii_digit()
        && (bytes[5] == b'/' || bytes[5] == b'-')
        && bytes[6].is_ascii_digit() && bytes[7].is_ascii_digit()
        && bytes[8].is_ascii_digit() && bytes[9].is_ascii_digit()
    {
        return Some((start, start + 10));
    }

    // Month names: January 1, 2024 etc.
    let months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    let lower = slice.to_lowercase();
    for month in &months {
        if lower.starts_with(month) {
            // Find end of date expression (month + day + optional year).
            let mut end = month.len();
            // Skip whitespace and digits.
            while end < slice.len() && (slice.as_bytes()[end].is_ascii_whitespace()
                || slice.as_bytes()[end].is_ascii_digit()
                || slice.as_bytes()[end] == b',')
            {
                end += 1;
            }
            if end > month.len() + 1 {
                return Some((start, start + end));
            }
        }
    }

    None
}

fn match_time(text: &str, start: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    if start + 4 > bytes.len() {
        return None;
    }

    // HH:MM or H:MM
    let mut i = start;
    if !bytes[i].is_ascii_digit() {
        return None;
    }
    i += 1;
    if i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return None;
    }
    i += 1;
    if i + 1 >= bytes.len() || !bytes[i].is_ascii_digit() || !bytes[i + 1].is_ascii_digit() {
        return None;
    }
    i += 2;

    // Optional :SS
    if i + 2 < bytes.len() && bytes[i] == b':' && bytes[i + 1].is_ascii_digit() && bytes[i + 2].is_ascii_digit() {
        i += 3;
    }

    // Optional AM/PM
    if i + 1 < bytes.len() {
        let suffix = &text[i..].to_lowercase();
        if suffix.starts_with("am") || suffix.starts_with("pm") {
            i += 2;
        } else if i < bytes.len() && bytes[i] == b' ' && i + 2 < bytes.len() {
            let suffix = &text[i + 1..].to_lowercase();
            if suffix.starts_with("am") || suffix.starts_with("pm") {
                i += 3;
            }
        }
    }

    Some((start, i))
}

fn match_phone(text: &str, start: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut i = start;
    let mut digit_count = 0;

    // Optional + prefix.
    if i < bytes.len() && bytes[i] == b'+' {
        i += 1;
    }

    // Optional opening paren.
    if i < bytes.len() && bytes[i] == b'(' {
        i += 1;
    }

    // First character after prefix must be a digit.
    if i >= bytes.len() || !bytes[i].is_ascii_digit() {
        return None;
    }

    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            digit_count += 1;
            i += 1;
        } else if bytes[i] == b'-' || bytes[i] == b' ' || bytes[i] == b'.' || bytes[i] == b')' || bytes[i] == b'(' {
            i += 1;
        } else {
            break;
        }
    }

    // Strip trailing non-digit separators (space, hyphen, dot, paren).
    while i > start && !bytes[i - 1].is_ascii_digit() {
        i -= 1;
    }

    if digit_count >= 7 && digit_count <= 15 {
        Some((start, i))
    } else {
        None
    }
}

fn match_percent(text: &str, start: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut i = start;

    // Number.
    let num_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
    }
    if i == num_start {
        return None;
    }

    // % sign.
    if i < bytes.len() && bytes[i] == b'%' {
        i += 1;
        return Some((start, i));
    }

    // "percent" or "pct"
    if i < bytes.len() && bytes[i] == b' ' {
        let rest = &text[i + 1..].to_lowercase();
        if rest.starts_with("percent") {
            return Some((start, i + 1 + 7));
        }
        if rest.starts_with("pct") {
            return Some((start, i + 1 + 3));
        }
    }

    None
}

fn match_number(text: &str, start: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut i = start;
    let num_start = i;

    // Optional sign.
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }

    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b',' || bytes[i] == b'.') {
        i += 1;
    }

    if i > num_start && text[num_start..i].chars().any(|c| c.is_ascii_digit()) {
        Some((start, i))
    } else {
        None
    }
}

// ── NER Tagger ───────────────────────────────────────────────────

/// Rule-based named entity tagger.
#[derive(Debug, Clone)]
pub struct NerTagger {
    rules: Vec<NerRule>,
    custom_type_names: HashMap<u32, String>,
    entity_links: HashMap<String, String>,
}

impl NerTagger {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            custom_type_names: HashMap::new(),
            entity_links: HashMap::new(),
        }
    }

    /// Create a tagger with default built-in pattern rules.
    pub fn with_defaults() -> Self {
        let mut tagger = Self::new();
        tagger.add_pattern_rule(EntityType::Email, BuiltinPattern::Email, 0.95);
        tagger.add_pattern_rule(EntityType::Url, BuiltinPattern::Url, 0.95);
        tagger.add_pattern_rule(EntityType::Money, BuiltinPattern::Money, 0.9);
        tagger.add_pattern_rule(EntityType::Date, BuiltinPattern::Date, 0.85);
        tagger.add_pattern_rule(EntityType::Time, BuiltinPattern::Time, 0.85);
        tagger.add_pattern_rule(EntityType::Phone, BuiltinPattern::Phone, 0.8);
        tagger.add_pattern_rule(EntityType::Percent, BuiltinPattern::Percent, 0.9);
        tagger
    }

    /// Add a built-in pattern rule.
    pub fn add_pattern_rule(&mut self, entity_type: EntityType, pattern: BuiltinPattern, confidence: f64) {
        self.rules.push(NerRule {
            entity_type,
            kind: RuleKind::BuiltinPattern(pattern),
            confidence,
        });
    }

    /// Add a dictionary-based rule.
    pub fn add_dictionary(&mut self, entity_type: EntityType, words: &[&str], confidence: f64) {
        let set: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
        self.rules.push(NerRule {
            entity_type,
            kind: RuleKind::Dictionary(set),
            confidence,
        });
    }

    /// Add a title-case heuristic rule (for person/org names).
    pub fn add_title_case_rule(&mut self, entity_type: EntityType, confidence: f64) {
        self.rules.push(NerRule {
            entity_type,
            kind: RuleKind::TitleCase,
            confidence,
        });
    }

    /// Register a custom entity type name.
    pub fn register_custom_type(&mut self, id: u32, name: &str) {
        self.custom_type_names.insert(id, name.to_string());
    }

    /// Add an entity link (surface form → canonical id).
    pub fn add_entity_link(&mut self, surface: &str, canonical_id: &str) {
        self.entity_links.insert(surface.to_lowercase(), canonical_id.to_string());
    }

    /// Tag entities in text.
    pub fn tag(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        for rule in &self.rules {
            match &rule.kind {
                RuleKind::BuiltinPattern(pattern) => {
                    self.apply_pattern(text, *pattern, rule.entity_type, rule.confidence, &mut entities);
                }
                RuleKind::Dictionary(dict) => {
                    self.apply_dictionary(text, dict, rule.entity_type, rule.confidence, &mut entities);
                }
                RuleKind::TitleCase => {
                    self.apply_title_case(text, rule.entity_type, rule.confidence, &mut entities);
                }
            }
        }

        // Resolve overlaps and link.
        let mut resolved = resolve_overlaps(entities);
        self.link_entities(&mut resolved);
        resolved
    }

    fn apply_pattern(
        &self,
        text: &str,
        pattern: BuiltinPattern,
        entity_type: EntityType,
        confidence: f64,
        entities: &mut Vec<Entity>,
    ) {
        let matcher: fn(&str, usize) -> Option<(usize, usize)> = match pattern {
            BuiltinPattern::Email => match_email,
            BuiltinPattern::Url => match_url,
            BuiltinPattern::Money => match_money,
            BuiltinPattern::Date => match_date,
            BuiltinPattern::Time => match_time,
            BuiltinPattern::Phone => match_phone,
            BuiltinPattern::Percent => match_percent,
            BuiltinPattern::Number => match_number,
        };

        let mut pos = 0;
        while pos < text.len() {
            if let Some((start, end)) = matcher(text, pos) {
                entities.push(Entity {
                    text: text[start..end].to_string(),
                    entity_type,
                    start,
                    end,
                    confidence,
                    linked_id: None,
                });
                pos = end;
            } else {
                pos += 1;
            }
        }
    }

    fn apply_dictionary(
        &self,
        text: &str,
        dict: &HashSet<String>,
        entity_type: EntityType,
        confidence: f64,
        entities: &mut Vec<Entity>,
    ) {
        let lower = text.to_lowercase();
        for entry in dict {
            let mut search_start = 0;
            while let Some(pos) = lower[search_start..].find(entry.as_str()) {
                let abs_start = search_start + pos;
                let abs_end = abs_start + entry.len();

                // Ensure word boundary.
                let at_word_start = abs_start == 0
                    || !text.as_bytes()[abs_start - 1].is_ascii_alphanumeric();
                let at_word_end = abs_end >= text.len()
                    || !text.as_bytes()[abs_end].is_ascii_alphanumeric();

                if at_word_start && at_word_end {
                    entities.push(Entity {
                        text: text[abs_start..abs_end].to_string(),
                        entity_type,
                        start: abs_start,
                        end: abs_end,
                        confidence,
                        linked_id: None,
                    });
                }

                search_start = abs_end;
            }
        }
    }

    fn apply_title_case(
        &self,
        text: &str,
        entity_type: EntityType,
        confidence: f64,
        entities: &mut Vec<Entity>,
    ) {
        // Find runs of title-cased words (starting with uppercase).
        let words: Vec<(usize, &str)> = text_word_spans(text);

        let mut i = 0;
        while i < words.len() {
            let (start, word) = words[i];
            if starts_uppercase(word) && word.len() > 1 {
                // Extend to consecutive title-case words.
                let entity_start = start;
                let mut entity_end = start + word.len();
                let mut j = i + 1;
                while j < words.len() {
                    let (next_start, next_word) = words[j];
                    if starts_uppercase(next_word) && next_word.len() > 1 {
                        entity_end = next_start + next_word.len();
                        j += 1;
                    } else {
                        break;
                    }
                }

                // Skip if at the very beginning of a sentence (common false positive).
                let is_sentence_start = entity_start == 0
                    || (entity_start >= 2 && text.as_bytes()[entity_start - 2] == b'.');

                if !is_sentence_start {
                    entities.push(Entity {
                        text: text[entity_start..entity_end].to_string(),
                        entity_type,
                        start: entity_start,
                        end: entity_end,
                        confidence,
                        linked_id: None,
                    });
                }

                i = j;
            } else {
                i += 1;
            }
        }
    }

    fn link_entities(&self, entities: &mut [Entity]) {
        for entity in entities.iter_mut() {
            if let Some(linked) = self.entity_links.get(&entity.text.to_lowercase()) {
                entity.linked_id = Some(linked.clone());
            }
        }
    }
}

impl Default for NerTagger {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn starts_uppercase(word: &str) -> bool {
    word.chars().next().map_or(false, |c| c.is_uppercase())
}

fn text_word_spans(text: &str) -> Vec<(usize, &str)> {
    let mut spans = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        spans.push((start, &text[start..i]));
    }

    spans
}

// ── Overlap resolution ───────────────────────────────────────────

/// Resolve overlapping entities: prefer longer spans, then higher confidence.
pub fn resolve_overlaps(mut entities: Vec<Entity>) -> Vec<Entity> {
    if entities.is_empty() {
        return entities;
    }

    // Sort by start, then by length descending, then by confidence descending.
    entities.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then((b.end - b.start).cmp(&(a.end - a.start)))
            .then(b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut result = Vec::new();
    let mut last_end = 0;

    for entity in entities {
        if entity.start >= last_end {
            last_end = entity.end;
            result.push(entity);
        }
    }

    result
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_detection() {
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag("Contact us at info@example.com for help");
        let emails: Vec<&Entity> = entities.iter().filter(|e| e.entity_type == EntityType::Email).collect();
        assert_eq!(emails.len(), 1);
        assert_eq!(emails[0].text, "info@example.com");
    }

    #[test]
    fn test_url_detection() {
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag("Visit https://example.com/page for details");
        let urls: Vec<&Entity> = entities.iter().filter(|e| e.entity_type == EntityType::Url).collect();
        assert_eq!(urls.len(), 1);
        assert!(urls[0].text.starts_with("https://"));
    }

    #[test]
    fn test_money_detection() {
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag("The price is $49.99 today");
        let money: Vec<&Entity> = entities.iter().filter(|e| e.entity_type == EntityType::Money).collect();
        assert_eq!(money.len(), 1);
        assert_eq!(money[0].text, "$49.99");
    }

    #[test]
    fn test_date_detection_iso() {
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag("The event is on 2024-03-15 sharp");
        let dates: Vec<&Entity> = entities.iter().filter(|e| e.entity_type == EntityType::Date).collect();
        assert_eq!(dates.len(), 1);
        assert_eq!(dates[0].text, "2024-03-15");
    }

    #[test]
    fn test_percent_detection() {
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag("Growth was 42.5% this quarter");
        let pcts: Vec<&Entity> = entities.iter().filter(|e| e.entity_type == EntityType::Percent).collect();
        assert_eq!(pcts.len(), 1);
        assert_eq!(pcts[0].text, "42.5%");
    }

    #[test]
    fn test_dictionary_ner() {
        let mut tagger = NerTagger::new();
        tagger.add_dictionary(
            EntityType::Location,
            &["new york", "london", "paris"],
            0.9,
        );
        let entities = tagger.tag("She traveled to London and then Paris");
        assert_eq!(entities.len(), 2);
        let locs: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(locs.contains(&"London"));
        assert!(locs.contains(&"Paris"));
    }

    #[test]
    fn test_dictionary_word_boundary() {
        let mut tagger = NerTagger::new();
        tagger.add_dictionary(EntityType::Person, &["al"], 0.9);
        let entities = tagger.tag("The algorithm is good");
        // "al" should NOT match inside "algorithm"
        assert!(entities.is_empty());
    }

    #[test]
    fn test_title_case_detection() {
        let mut tagger = NerTagger::new();
        tagger.add_title_case_rule(EntityType::Person, 0.6);
        let entities = tagger.tag("I spoke with John Smith yesterday");
        assert!(!entities.is_empty());
        let person_texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(person_texts.iter().any(|t| t.contains("John")));
    }

    #[test]
    fn test_overlap_resolution() {
        let entities = vec![
            Entity { text: "New".to_string(), entity_type: EntityType::Location, start: 0, end: 3, confidence: 0.5, linked_id: None },
            Entity { text: "New York".to_string(), entity_type: EntityType::Location, start: 0, end: 8, confidence: 0.9, linked_id: None },
        ];
        let resolved = resolve_overlaps(entities);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].text, "New York");
    }

    #[test]
    fn test_entity_linking() {
        let mut tagger = NerTagger::new();
        tagger.add_dictionary(EntityType::Organization, &["apple"], 0.8);
        tagger.add_entity_link("apple", "AAPL");
        let entities = tagger.tag("Apple announced new products");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].linked_id.as_deref(), Some("AAPL"));
    }

    #[test]
    fn test_custom_entity_type() {
        let mut tagger = NerTagger::new();
        tagger.register_custom_type(1, "PRODUCT");
        tagger.add_dictionary(EntityType::Custom(1), &["iphone", "macbook"], 0.85);
        let entities = tagger.tag("The iPhone is amazing");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, EntityType::Custom(1));
    }

    #[test]
    fn test_multiple_emails() {
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag("Send to alice@test.com or bob@test.com");
        let emails: Vec<&Entity> = entities.iter().filter(|e| e.entity_type == EntityType::Email).collect();
        assert_eq!(emails.len(), 2);
    }

    #[test]
    fn test_empty_text() {
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag("");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_entity_spans_correct() {
        let text = "Price: $100.00 end";
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag(text);
        let money: Vec<&Entity> = entities.iter().filter(|e| e.entity_type == EntityType::Money).collect();
        assert!(!money.is_empty());
        let e = money[0];
        assert_eq!(&text[e.start..e.end], e.text.as_str());
    }

    #[test]
    fn test_confidence_scores() {
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag("user@domain.com costs $50");
        for e in &entities {
            assert!(e.confidence > 0.0 && e.confidence <= 1.0);
        }
    }

    #[test]
    fn test_time_detection() {
        let tagger = NerTagger::with_defaults();
        let entities = tagger.tag("The meeting is at 14:30 today");
        let times: Vec<&Entity> = entities.iter().filter(|e| e.entity_type == EntityType::Time).collect();
        assert!(!times.is_empty());
        assert!(times[0].text.contains("14:30"));
    }
}
