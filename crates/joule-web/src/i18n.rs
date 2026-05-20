//! Internationalization — message catalogs, locale negotiation, plural rules,
//! ICU-like message format, interpolation, fallback chains, BCP 47 language tags.
//!
//! Pure-Rust replacement for i18next / react-intl / FormatJS.

use std::collections::HashMap;

// ── BCP 47 Language Tag ─────────────────────────────────────────

/// A parsed BCP 47 language tag (language-script-region-variant).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LanguageTag {
    pub language: String,
    pub script: Option<String>,
    pub region: Option<String>,
    pub variant: Option<String>,
}

impl LanguageTag {
    /// Parse a BCP 47 tag like `en`, `en-US`, `zh-Hant-TW`, `de-DE-1996`.
    pub fn parse(tag: &str) -> Option<Self> {
        let tag = tag.trim();
        if tag.is_empty() {
            return None;
        }
        let parts: Vec<&str> = tag.split(['-', '_']).collect();
        if parts.is_empty() || parts[0].len() < 2 || parts[0].len() > 3 {
            return None;
        }
        if !parts[0].chars().all(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        let language = parts[0].to_lowercase();
        let mut script = None;
        let mut region = None;
        let mut variant = None;
        let mut idx = 1;
        // Script: 4-letter title-case subtag
        if idx < parts.len() && parts[idx].len() == 4 && parts[idx].chars().all(|c| c.is_ascii_alphabetic()) {
            let s = parts[idx];
            let mut sc = String::with_capacity(4);
            for (i, ch) in s.chars().enumerate() {
                if i == 0 { sc.push(ch.to_ascii_uppercase()); } else { sc.push(ch.to_ascii_lowercase()); }
            }
            script = Some(sc);
            idx += 1;
        }
        // Region: 2-letter alpha or 3-digit numeric
        if idx < parts.len() {
            let p = parts[idx];
            if p.len() == 2 && p.chars().all(|c| c.is_ascii_alphabetic()) {
                region = Some(p.to_uppercase());
                idx += 1;
            } else if p.len() == 3 && p.chars().all(|c| c.is_ascii_digit()) {
                region = Some(p.to_string());
                idx += 1;
            }
        }
        // Variant: anything remaining (5-8 chars or digit + 3 chars)
        if idx < parts.len() {
            let p = parts[idx];
            if p.len() >= 4 {
                variant = Some(p.to_lowercase());
            }
        }
        Some(LanguageTag { language, script, region, variant })
    }

    /// Produce the canonical tag string.
    pub fn to_string(&self) -> String {
        let mut s = self.language.clone();
        if let Some(ref sc) = self.script {
            s.push('-');
            s.push_str(sc);
        }
        if let Some(ref r) = self.region {
            s.push('-');
            s.push_str(r);
        }
        if let Some(ref v) = self.variant {
            s.push('-');
            s.push_str(v);
        }
        s
    }

    /// Simple language-region tag for backward compat.
    pub fn language_region_tag(&self) -> String {
        match &self.region {
            Some(r) => format!("{}-{r}", self.language),
            None => self.language.clone(),
        }
    }
}

// ── Locale (simplified wrapper) ─────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Locale {
    pub language: String,
    pub region: Option<String>,
}

impl Locale {
    pub fn new(language: &str) -> Self {
        Self { language: language.to_lowercase(), region: None }
    }

    pub fn new_with_region(language: &str, region: &str) -> Self {
        Self { language: language.to_lowercase(), region: Some(region.to_uppercase()) }
    }

    pub fn language_tag(&self) -> String {
        match &self.region {
            Some(r) => format!("{}-{r}", self.language),
            None => self.language.clone(),
        }
    }

    pub fn parse(tag: &str) -> Option<Locale> {
        let bt = LanguageTag::parse(tag)?;
        Some(Locale { language: bt.language, region: bt.region })
    }

    pub fn from_language_tag(bt: &LanguageTag) -> Self {
        Locale { language: bt.language.clone(), region: bt.region.clone() }
    }
}

// ── Locale Negotiation ──────────────────────────────────────────

/// Negotiate best locale from requested list against available locales.
/// Returns the first match using fallback: exact -> language+region -> language.
pub fn negotiate_locale<'a>(
    requested: &[&str],
    available: &[&'a str],
    default: &'a str,
) -> &'a str {
    for req in requested {
        let req_tag = match LanguageTag::parse(req) {
            Some(t) => t,
            None => continue,
        };
        // Exact match
        for &avail in available {
            if let Some(avail_tag) = LanguageTag::parse(avail) {
                if avail_tag.language == req_tag.language
                    && avail_tag.region == req_tag.region
                    && avail_tag.script == req_tag.script
                {
                    return avail;
                }
            }
        }
        // Language + region match (ignoring script)
        if req_tag.region.is_some() {
            for &avail in available {
                if let Some(avail_tag) = LanguageTag::parse(avail) {
                    if avail_tag.language == req_tag.language && avail_tag.region == req_tag.region {
                        return avail;
                    }
                }
            }
        }
        // Language-only match
        for &avail in available {
            if let Some(avail_tag) = LanguageTag::parse(avail) {
                if avail_tag.language == req_tag.language {
                    return avail;
                }
            }
        }
    }
    default
}

// ── Plural Categories ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PluralCategory { Zero, One, Two, Few, Many, Other }

pub trait PluralRule: Send + Sync {
    fn select(&self, n: f64) -> PluralCategory;
}

#[derive(Debug, Clone)]
pub struct EnglishPlural;
impl PluralRule for EnglishPlural {
    fn select(&self, n: f64) -> PluralCategory {
        if (n - 1.0).abs() < f64::EPSILON { PluralCategory::One } else { PluralCategory::Other }
    }
}

#[derive(Debug, Clone)]
pub struct FrenchPlural;
impl PluralRule for FrenchPlural {
    fn select(&self, n: f64) -> PluralCategory {
        if n.abs() < 1.5 { PluralCategory::One } else { PluralCategory::Other }
    }
}

#[derive(Debug, Clone)]
pub struct ArabicPlural;
impl PluralRule for ArabicPlural {
    fn select(&self, n: f64) -> PluralCategory {
        let ni = n.abs() as u64;
        match ni {
            0 => PluralCategory::Zero,
            1 => PluralCategory::One,
            2 => PluralCategory::Two,
            3..=10 => PluralCategory::Few,
            11..=99 => PluralCategory::Many,
            _ => PluralCategory::Other,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JapanesePlural;
impl PluralRule for JapanesePlural {
    fn select(&self, _n: f64) -> PluralCategory { PluralCategory::Other }
}

#[derive(Debug, Clone)]
pub struct PolishPlural;
impl PluralRule for PolishPlural {
    fn select(&self, n: f64) -> PluralCategory {
        let ni = n.abs() as u64;
        if ni == 1 { return PluralCategory::One; }
        let mod10 = ni % 10;
        let mod100 = ni % 100;
        if (2..=4).contains(&mod10) && !(12..=14).contains(&mod100) {
            PluralCategory::Few
        } else {
            PluralCategory::Other
        }
    }
}

#[derive(Debug, Clone)]
pub struct RussianPlural;
impl PluralRule for RussianPlural {
    fn select(&self, n: f64) -> PluralCategory {
        let ni = n.abs() as u64;
        let mod10 = ni % 10;
        let mod100 = ni % 100;
        if mod10 == 1 && mod100 != 11 {
            PluralCategory::One
        } else if (2..=4).contains(&mod10) && !(12..=14).contains(&mod100) {
            PluralCategory::Few
        } else {
            PluralCategory::Many
        }
    }
}

fn default_plural_rule(lang: &str) -> Box<dyn PluralRule> {
    match lang {
        "en" | "de" | "es" | "it" | "pt" | "nl" | "sv" | "da" | "no" => Box::new(EnglishPlural),
        "fr" => Box::new(FrenchPlural),
        "ar" => Box::new(ArabicPlural),
        "ja" | "zh" | "ko" | "vi" | "th" => Box::new(JapanesePlural),
        "pl" | "cs" | "sk" | "hr" => Box::new(PolishPlural),
        "ru" | "uk" | "be" => Box::new(RussianPlural),
        _ => Box::new(EnglishPlural),
    }
}

// ── ICU-like Message Format ─────────────────────────────────────

/// Parsed ICU message token.
#[derive(Debug, Clone, PartialEq)]
enum IcuToken {
    Literal(String),
    Argument(String),
    Plural { arg: String, forms: Vec<(String, String)> },
    Select { arg: String, forms: Vec<(String, String)> },
}

/// Parse an ICU-like message format string.
/// Supports: `{name}`, `{count, plural, one{...} other{...}}`, `{gender, select, male{...} female{...} other{...}}`.
fn parse_icu_message(input: &str) -> Vec<IcuToken> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        if chars[i] == '{' {
            i += 1;
            let start = i;
            let mut depth = 1;
            let mut content = String::new();
            while i < len && depth > 0 {
                if chars[i] == '{' { depth += 1; }
                if chars[i] == '}' { depth -= 1; }
                if depth > 0 { content.push(chars[i]); }
                i += 1;
            }
            let _ = start;
            let parts: Vec<&str> = content.splitn(3, ',').collect();
            if parts.len() == 1 {
                tokens.push(IcuToken::Argument(parts[0].trim().to_string()));
            } else if parts.len() >= 3 {
                let arg = parts[0].trim().to_string();
                let kind = parts[1].trim();
                let forms_str = parts[2..].join(",");
                let forms = parse_icu_forms(forms_str.trim());
                match kind {
                    "plural" => tokens.push(IcuToken::Plural { arg, forms }),
                    "select" => tokens.push(IcuToken::Select { arg, forms }),
                    _ => tokens.push(IcuToken::Argument(arg)),
                }
            } else {
                tokens.push(IcuToken::Argument(parts[0].trim().to_string()));
            }
        } else {
            let mut lit = String::new();
            while i < len && chars[i] != '{' {
                lit.push(chars[i]);
                i += 1;
            }
            tokens.push(IcuToken::Literal(lit));
        }
    }
    tokens
}

fn parse_icu_forms(input: &str) -> Vec<(String, String)> {
    let mut forms = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        // skip whitespace
        while i < len && chars[i].is_whitespace() { i += 1; }
        if i >= len { break; }
        // read keyword (e.g., "one", "other", "=0", "male")
        let mut keyword = String::new();
        while i < len && chars[i] != '{' && !chars[i].is_whitespace() {
            keyword.push(chars[i]);
            i += 1;
        }
        while i < len && chars[i].is_whitespace() { i += 1; }
        if i >= len || chars[i] != '{' { break; }
        i += 1; // skip opening brace
        let mut depth = 1;
        let mut body = String::new();
        while i < len && depth > 0 {
            if chars[i] == '{' { depth += 1; }
            if chars[i] == '}' { depth -= 1; }
            if depth > 0 { body.push(chars[i]); }
            i += 1;
        }
        if !keyword.is_empty() {
            forms.push((keyword, body));
        }
    }
    forms
}

/// Format an ICU message with arguments.
pub fn format_icu(template: &str, args: &HashMap<String, String>, plural_rule: &dyn PluralRule) -> String {
    let tokens = parse_icu_message(template);
    let mut result = String::new();
    for tok in &tokens {
        match tok {
            IcuToken::Literal(s) => result.push_str(s),
            IcuToken::Argument(name) => {
                if let Some(val) = args.get(name) {
                    result.push_str(val);
                }
            }
            IcuToken::Plural { arg, forms } => {
                if let Some(val_str) = args.get(arg) {
                    if let Ok(n) = val_str.parse::<f64>() {
                        // Check exact match first (=0, =1, etc.)
                        let exact_key = format!("={}", n as i64);
                        let mut body = None;
                        for (k, v) in forms {
                            if *k == exact_key { body = Some(v.as_str()); break; }
                        }
                        if body.is_none() {
                            let cat = plural_rule.select(n);
                            let cat_str = match cat {
                                PluralCategory::Zero => "zero",
                                PluralCategory::One => "one",
                                PluralCategory::Two => "two",
                                PluralCategory::Few => "few",
                                PluralCategory::Many => "many",
                                PluralCategory::Other => "other",
                            };
                            for (k, v) in forms {
                                if k == cat_str { body = Some(v.as_str()); break; }
                            }
                        }
                        if body.is_none() {
                            for (k, v) in forms {
                                if k == "other" { body = Some(v.as_str()); break; }
                            }
                        }
                        if let Some(b) = body {
                            let replaced = b.replace('#', val_str);
                            result.push_str(&replaced);
                        }
                    }
                }
            }
            IcuToken::Select { arg, forms } => {
                if let Some(val) = args.get(arg) {
                    let mut body = None;
                    for (k, v) in forms {
                        if k == val { body = Some(v.as_str()); break; }
                    }
                    if body.is_none() {
                        for (k, v) in forms {
                            if k == "other" { body = Some(v.as_str()); break; }
                        }
                    }
                    if let Some(b) = body {
                        result.push_str(b);
                    }
                }
            }
        }
    }
    result
}

// ── Message Entries ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MessageEntry {
    Simple(String),
    WithPlaceholders(String),
    Plural { forms: HashMap<PluralCategory, String> },
    IcuFormat(String),
}

// ── Message Catalog ─────────────────────────────────────────────

pub struct MessageCatalog {
    pub locale: Locale,
    pub messages: HashMap<String, MessageEntry>,
    plural_rule: Box<dyn PluralRule>,
}

impl MessageCatalog {
    pub fn new(locale: Locale) -> Self {
        let rule = default_plural_rule(&locale.language);
        Self { locale, messages: HashMap::new(), plural_rule: rule }
    }

    pub fn add(&mut self, key: &str, entry: MessageEntry) -> &mut Self {
        self.messages.insert(key.into(), entry);
        self
    }

    pub fn get(&self, key: &str) -> Option<&MessageEntry> {
        self.messages.get(key)
    }

    pub fn format(&self, key: &str, args: &HashMap<String, String>) -> Option<String> {
        let entry = self.messages.get(key)?;
        match entry {
            MessageEntry::Simple(s) => Some(s.clone()),
            MessageEntry::WithPlaceholders(template) => Some(substitute_placeholders(template, args)),
            MessageEntry::Plural { forms } => {
                let count_str = args.get("count")?;
                let count: f64 = count_str.parse().ok()?;
                let category = self.plural_rule.select(count);
                let template = forms.get(&category).or_else(|| forms.get(&PluralCategory::Other))?;
                Some(substitute_placeholders(template, args))
            }
            MessageEntry::IcuFormat(template) => {
                Some(format_icu(template, args, self.plural_rule.as_ref()))
            }
        }
    }
}

fn substitute_placeholders(template: &str, args: &HashMap<String, String>) -> String {
    let mut result = template.to_string();
    for (key, value) in args {
        result = result.replace(&format!("{{{key}}}"), value);
    }
    result
}

// ── Fallback Chain ──────────────────────────────────────────────

/// Compute a fallback chain for a locale, e.g. `zh-Hant-TW` -> `[zh-Hant-TW, zh-TW, zh]`.
pub fn fallback_chain(tag: &str) -> Vec<String> {
    let mut chain = Vec::new();
    let parts: Vec<&str> = tag.split(['-', '_']).collect();
    // Full tag
    chain.push(parts.join("-"));
    // Drop subtags from the right
    for i in (1..parts.len()).rev() {
        chain.push(parts[..i].join("-"));
    }
    chain
}

// ── I18n Manager ────────────────────────────────────────────────

pub struct I18n {
    catalogs: HashMap<String, MessageCatalog>,
    current_locale: String,
    fallback_locale: String,
}

impl I18n {
    pub fn new(default_locale: &str) -> Self {
        Self { catalogs: HashMap::new(), current_locale: default_locale.into(), fallback_locale: default_locale.into() }
    }

    pub fn add_catalog(&mut self, catalog: MessageCatalog) {
        let tag = catalog.locale.language_tag();
        self.catalogs.insert(tag, catalog);
    }

    pub fn set_locale(&mut self, locale: &str) -> bool {
        if self.catalogs.contains_key(locale) {
            self.current_locale = locale.into();
            true
        } else {
            false
        }
    }

    pub fn set_fallback(&mut self, locale: &str) {
        self.fallback_locale = locale.into();
    }

    pub fn t(&self, key: &str) -> String {
        self.t_args(key, &HashMap::new())
    }

    pub fn t_args(&self, key: &str, args: &HashMap<String, String>) -> String {
        // Try current locale's fallback chain first
        let chain = fallback_chain(&self.current_locale);
        for tag in &chain {
            if let Some(catalog) = self.catalogs.get(tag) {
                if let Some(result) = catalog.format(key, args) {
                    return result;
                }
            }
        }
        // Then try the designated fallback locale chain
        if self.current_locale != self.fallback_locale {
            let fb_chain = fallback_chain(&self.fallback_locale);
            for tag in &fb_chain {
                if let Some(catalog) = self.catalogs.get(tag) {
                    if let Some(result) = catalog.format(key, args) {
                        return result;
                    }
                }
            }
        }
        key.to_string()
    }

    pub fn t_plural(&self, key: &str, count: f64, args: &HashMap<String, String>) -> String {
        let mut full_args = args.clone();
        full_args.insert("count".into(), format_number_plain(count));
        self.t_args(key, &full_args)
    }

    pub fn available_locales(&self) -> Vec<&str> {
        self.catalogs.keys().map(|s| s.as_str()).collect()
    }
}

fn format_number_plain(n: f64) -> String {
    if n == n.trunc() { format!("{}", n as i64) } else { format!("{n}") }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_args(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    // -- BCP 47 parsing --

    #[test]
    fn bcp47_simple_language() {
        let tag = LanguageTag::parse("en").unwrap();
        assert_eq!(tag.language, "en");
        assert!(tag.script.is_none());
        assert!(tag.region.is_none());
        assert!(tag.variant.is_none());
    }

    #[test]
    fn bcp47_language_region() {
        let tag = LanguageTag::parse("en-US").unwrap();
        assert_eq!(tag.language, "en");
        assert_eq!(tag.region.as_deref(), Some("US"));
    }

    #[test]
    fn bcp47_with_script() {
        let tag = LanguageTag::parse("zh-Hant-TW").unwrap();
        assert_eq!(tag.language, "zh");
        assert_eq!(tag.script.as_deref(), Some("Hant"));
        assert_eq!(tag.region.as_deref(), Some("TW"));
    }

    #[test]
    fn bcp47_with_variant() {
        let tag = LanguageTag::parse("de-DE-1996").unwrap();
        assert_eq!(tag.language, "de");
        assert_eq!(tag.region.as_deref(), Some("DE"));
        assert_eq!(tag.variant.as_deref(), Some("1996"));
    }

    #[test]
    fn bcp47_underscore_separator() {
        let tag = LanguageTag::parse("en_GB").unwrap();
        assert_eq!(tag.language, "en");
        assert_eq!(tag.region.as_deref(), Some("GB"));
    }

    #[test]
    fn bcp47_case_normalization() {
        let tag = LanguageTag::parse("ZH-hANT-tw").unwrap();
        assert_eq!(tag.language, "zh");
        assert_eq!(tag.script.as_deref(), Some("Hant"));
        assert_eq!(tag.region.as_deref(), Some("TW"));
    }

    #[test]
    fn bcp47_to_string() {
        let tag = LanguageTag::parse("zh-Hant-TW").unwrap();
        assert_eq!(tag.to_string(), "zh-Hant-TW");
    }

    #[test]
    fn bcp47_empty_returns_none() {
        assert!(LanguageTag::parse("").is_none());
    }

    #[test]
    fn bcp47_numeric_region() {
        let tag = LanguageTag::parse("es-419").unwrap();
        assert_eq!(tag.language, "es");
        assert_eq!(tag.region.as_deref(), Some("419"));
    }

    // -- Locale negotiation --

    #[test]
    fn negotiate_exact_match() {
        let avail = &["en", "fr", "de"];
        assert_eq!(negotiate_locale(&["fr"], avail, "en"), "fr");
    }

    #[test]
    fn negotiate_language_fallback() {
        let avail = &["en", "fr", "de"];
        assert_eq!(negotiate_locale(&["fr-CA"], avail, "en"), "fr");
    }

    #[test]
    fn negotiate_returns_default() {
        let avail = &["en", "fr"];
        assert_eq!(negotiate_locale(&["ja"], avail, "en"), "en");
    }

    #[test]
    fn negotiate_priority_order() {
        let avail = &["en", "fr", "de"];
        assert_eq!(negotiate_locale(&["de", "fr"], avail, "en"), "de");
    }

    #[test]
    fn negotiate_region_match() {
        let avail = &["en-US", "en-GB", "fr"];
        assert_eq!(negotiate_locale(&["en-GB"], avail, "en-US"), "en-GB");
    }

    // -- Fallback chain --

    #[test]
    fn fallback_chain_simple() {
        assert_eq!(fallback_chain("en"), vec!["en"]);
    }

    #[test]
    fn fallback_chain_region() {
        assert_eq!(fallback_chain("en-US"), vec!["en-US", "en"]);
    }

    #[test]
    fn fallback_chain_script_region() {
        assert_eq!(fallback_chain("zh-Hant-TW"), vec!["zh-Hant-TW", "zh-Hant", "zh"]);
    }

    // -- Plural rules --

    #[test]
    fn english_plural() {
        let r = EnglishPlural;
        assert_eq!(r.select(0.0), PluralCategory::Other);
        assert_eq!(r.select(1.0), PluralCategory::One);
        assert_eq!(r.select(2.0), PluralCategory::Other);
    }

    #[test]
    fn french_plural() {
        let r = FrenchPlural;
        assert_eq!(r.select(0.0), PluralCategory::One);
        assert_eq!(r.select(1.0), PluralCategory::One);
        assert_eq!(r.select(2.0), PluralCategory::Other);
    }

    #[test]
    fn arabic_plural() {
        let r = ArabicPlural;
        assert_eq!(r.select(0.0), PluralCategory::Zero);
        assert_eq!(r.select(1.0), PluralCategory::One);
        assert_eq!(r.select(2.0), PluralCategory::Two);
        assert_eq!(r.select(5.0), PluralCategory::Few);
        assert_eq!(r.select(50.0), PluralCategory::Many);
        assert_eq!(r.select(100.0), PluralCategory::Other);
    }

    #[test]
    fn polish_plural() {
        let r = PolishPlural;
        assert_eq!(r.select(1.0), PluralCategory::One);
        assert_eq!(r.select(2.0), PluralCategory::Few);
        assert_eq!(r.select(5.0), PluralCategory::Other);
        assert_eq!(r.select(12.0), PluralCategory::Other);
        assert_eq!(r.select(22.0), PluralCategory::Few);
    }

    #[test]
    fn russian_plural() {
        let r = RussianPlural;
        assert_eq!(r.select(1.0), PluralCategory::One);
        assert_eq!(r.select(2.0), PluralCategory::Few);
        assert_eq!(r.select(5.0), PluralCategory::Many);
        assert_eq!(r.select(11.0), PluralCategory::Many);
        assert_eq!(r.select(21.0), PluralCategory::One);
    }

    // -- ICU message format --

    #[test]
    fn icu_simple_arg() {
        let args = make_args(&[("name", "World")]);
        let result = format_icu("Hello, {name}!", &args, &EnglishPlural);
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn icu_plural() {
        let template = "{count, plural, one{# item} other{# items}}";
        let args = make_args(&[("count", "1")]);
        assert_eq!(format_icu(template, &args, &EnglishPlural), "1 item");
        let args = make_args(&[("count", "5")]);
        assert_eq!(format_icu(template, &args, &EnglishPlural), "5 items");
    }

    #[test]
    fn icu_plural_exact_match() {
        let template = "{count, plural, =0{no items} one{# item} other{# items}}";
        let args = make_args(&[("count", "0")]);
        assert_eq!(format_icu(template, &args, &EnglishPlural), "no items");
    }

    #[test]
    fn icu_select() {
        let template = "{gender, select, male{He} female{She} other{They}} went home.";
        let args = make_args(&[("gender", "female")]);
        assert_eq!(format_icu(template, &args, &EnglishPlural), "She went home.");
        let args = make_args(&[("gender", "nonbinary")]);
        assert_eq!(format_icu(template, &args, &EnglishPlural), "They went home.");
    }

    // -- Message catalog --

    #[test]
    fn simple_message_lookup() {
        let mut cat = MessageCatalog::new(Locale::new("en"));
        cat.add("greeting", MessageEntry::Simple("Hello".into()));
        assert_eq!(cat.format("greeting", &HashMap::new()), Some("Hello".into()));
    }

    #[test]
    fn placeholder_substitution() {
        let mut cat = MessageCatalog::new(Locale::new("en"));
        cat.add("hello", MessageEntry::WithPlaceholders("Hello, {name}!".into()));
        let args = make_args(&[("name", "World")]);
        assert_eq!(cat.format("hello", &args), Some("Hello, World!".into()));
    }

    #[test]
    fn plural_english_one_and_other() {
        let mut cat = MessageCatalog::new(Locale::new("en"));
        let mut forms = HashMap::new();
        forms.insert(PluralCategory::One, "{count} item".into());
        forms.insert(PluralCategory::Other, "{count} items".into());
        cat.add("items", MessageEntry::Plural { forms });
        assert_eq!(cat.format("items", &make_args(&[("count", "1")])), Some("1 item".into()));
        assert_eq!(cat.format("items", &make_args(&[("count", "5")])), Some("5 items".into()));
    }

    #[test]
    fn icu_format_entry() {
        let mut cat = MessageCatalog::new(Locale::new("en"));
        cat.add("items", MessageEntry::IcuFormat("{count, plural, one{# item} other{# items}}".into()));
        assert_eq!(cat.format("items", &make_args(&[("count", "1")])), Some("1 item".into()));
        assert_eq!(cat.format("items", &make_args(&[("count", "3")])), Some("3 items".into()));
    }

    // -- I18n with fallback --

    #[test]
    fn fallback_locale() {
        let mut i18n = I18n::new("en");
        let mut en = MessageCatalog::new(Locale::new("en"));
        en.add("hello", MessageEntry::Simple("Hello".into()));
        en.add("bye", MessageEntry::Simple("Goodbye".into()));
        i18n.add_catalog(en);
        let mut fr = MessageCatalog::new(Locale::new("fr"));
        fr.add("hello", MessageEntry::Simple("Bonjour".into()));
        i18n.add_catalog(fr);
        i18n.set_locale("fr");
        assert_eq!(i18n.t("hello"), "Bonjour");
        assert_eq!(i18n.t("bye"), "Goodbye");
    }

    #[test]
    fn missing_key_returns_key() {
        let i18n = I18n::new("en");
        assert_eq!(i18n.t("nonexistent.key"), "nonexistent.key");
    }

    #[test]
    fn set_locale_switches() {
        let mut i18n = I18n::new("en");
        let mut en = MessageCatalog::new(Locale::new("en"));
        en.add("greeting", MessageEntry::Simple("Hello".into()));
        i18n.add_catalog(en);
        let mut de = MessageCatalog::new(Locale::new("de"));
        de.add("greeting", MessageEntry::Simple("Hallo".into()));
        i18n.add_catalog(de);
        assert_eq!(i18n.t("greeting"), "Hello");
        assert!(i18n.set_locale("de"));
        assert_eq!(i18n.t("greeting"), "Hallo");
        assert!(!i18n.set_locale("xx"));
    }

    #[test]
    fn locale_parsing() {
        let en = Locale::parse("en").expect("en");
        assert_eq!(en.language, "en");
        assert!(en.region.is_none());
        let en_us = Locale::parse("en-US").expect("en-US");
        assert_eq!(en_us.language, "en");
        assert_eq!(en_us.region.as_deref(), Some("US"));
    }

    #[test]
    fn t_plural_integration() {
        let mut i18n = I18n::new("en");
        let mut en = MessageCatalog::new(Locale::new("en"));
        let mut forms = HashMap::new();
        forms.insert(PluralCategory::One, "{count} file".into());
        forms.insert(PluralCategory::Other, "{count} files".into());
        en.add("files", MessageEntry::Plural { forms });
        i18n.add_catalog(en);
        assert_eq!(i18n.t_plural("files", 1.0, &HashMap::new()), "1 file");
        assert_eq!(i18n.t_plural("files", 42.0, &HashMap::new()), "42 files");
    }

    #[test]
    fn available_locales_lists_all() {
        let mut i18n = I18n::new("en");
        i18n.add_catalog(MessageCatalog::new(Locale::new("en")));
        i18n.add_catalog(MessageCatalog::new(Locale::new("fr")));
        let mut locales = i18n.available_locales();
        locales.sort();
        assert_eq!(locales, vec!["en", "fr"]);
    }

    #[test]
    fn set_fallback_locale() {
        let mut i18n = I18n::new("en");
        let mut en = MessageCatalog::new(Locale::new("en"));
        en.add("msg", MessageEntry::Simple("English".into()));
        i18n.add_catalog(en);
        let mut de = MessageCatalog::new(Locale::new("de"));
        de.add("other", MessageEntry::Simple("German".into()));
        i18n.add_catalog(de);
        i18n.set_locale("de");
        i18n.set_fallback("en");
        assert_eq!(i18n.t("msg"), "English");
    }

    #[test]
    fn icu_multiple_args() {
        let template = "{name} has {count, plural, one{# new message} other{# new messages}}.";
        let args = make_args(&[("name", "Alice"), ("count", "3")]);
        assert_eq!(format_icu(template, &args, &EnglishPlural), "Alice has 3 new messages.");
    }
}
