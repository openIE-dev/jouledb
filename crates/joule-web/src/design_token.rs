//! Design tokens: typed token definitions, groups, theme-aware resolution,
//! CSS custom property generation, JSON export/import, and token aliasing.
//!
//! Tokens are the atomic values of a design system — colors, spacing, typography,
//! shadows, borders, radii, and z-indices. This module provides a structured way
//! to define, group, alias, and export them.

use std::collections::HashMap;
use std::fmt;

// ── Token Value Types ───────────────────────────────────────────

/// The category of a design token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenCategory {
    Color,
    Spacing,
    Typography,
    Shadow,
    Border,
    Radius,
    ZIndex,
}

impl fmt::Display for TokenCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenCategory::Color => write!(f, "color"),
            TokenCategory::Spacing => write!(f, "spacing"),
            TokenCategory::Typography => write!(f, "typography"),
            TokenCategory::Shadow => write!(f, "shadow"),
            TokenCategory::Border => write!(f, "border"),
            TokenCategory::Radius => write!(f, "radius"),
            TokenCategory::ZIndex => write!(f, "z-index"),
        }
    }
}

/// A typed token value.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenValue {
    /// A CSS color string (hex, rgb, hsl, etc.).
    Color(String),
    /// A dimension with unit (e.g., `16px`, `1rem`).
    Dimension(f64, String),
    /// A numeric value (e.g., z-index, font-weight).
    Number(f64),
    /// A raw string (e.g., font-family, shadow shorthand).
    Str(String),
    /// An alias to another token by name.
    Alias(String),
}

impl TokenValue {
    pub fn css_value(&self) -> String {
        match self {
            TokenValue::Color(c) => c.clone(),
            TokenValue::Dimension(v, unit) => format!("{v}{unit}"),
            TokenValue::Number(n) => format!("{n}"),
            TokenValue::Str(s) => s.clone(),
            TokenValue::Alias(name) => format!("var(--{name})"),
        }
    }
}

impl fmt::Display for TokenValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.css_value())
    }
}

// ── Token ───────────────────────────────────────────────────────

/// A single design token.
#[derive(Debug, Clone)]
pub struct Token {
    /// Dot-separated name, e.g. `color.primary.500`.
    pub name: String,
    pub value: TokenValue,
    pub category: TokenCategory,
    pub description: Option<String>,
}

impl Token {
    pub fn new(
        name: impl Into<String>,
        value: TokenValue,
        category: TokenCategory,
    ) -> Self {
        Self {
            name: name.into(),
            value,
            category,
            description: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// CSS custom property name: `color.primary.500` → `--color-primary-500`.
    pub fn css_property_name(&self) -> String {
        format!("--{}", self.name.replace('.', "-"))
    }

    /// Full CSS custom property declaration.
    pub fn to_css_declaration(&self) -> String {
        format!("{}: {};", self.css_property_name(), self.value.css_value())
    }
}

// ── Token Group ─────────────────────────────────────────────────

/// A named group of tokens (e.g., "colors", "spacing").
#[derive(Debug, Clone)]
pub struct TokenGroup {
    pub name: String,
    pub tokens: Vec<Token>,
}

impl TokenGroup {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tokens: Vec::new(),
        }
    }

    pub fn add(mut self, token: Token) -> Self {
        self.tokens.push(token);
        self
    }

    pub fn get(&self, name: &str) -> Option<&Token> {
        self.tokens.iter().find(|t| t.name == name)
    }

    pub fn to_css(&self) -> String {
        let mut css = format!("/* {} */\n", self.name);
        for token in &self.tokens {
            css.push_str(&format!("  {}\n", token.to_css_declaration()));
        }
        css
    }
}

// ── Theme ───────────────────────────────────────────────────────

/// A theme is a named collection of token overrides.
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    overrides: HashMap<String, TokenValue>,
}

impl Theme {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            overrides: HashMap::new(),
        }
    }

    pub fn set(mut self, token_name: impl Into<String>, value: TokenValue) -> Self {
        self.overrides.insert(token_name.into(), value);
        self
    }

    pub fn get(&self, token_name: &str) -> Option<&TokenValue> {
        self.overrides.get(token_name)
    }
}

// ── Token Store ─────────────────────────────────────────────────

/// Central store of all tokens with theme-aware resolution.
#[derive(Debug, Clone)]
pub struct TokenStore {
    tokens: HashMap<String, Token>,
    themes: HashMap<String, Theme>,
    active_theme: Option<String>,
}

impl TokenStore {
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            themes: HashMap::new(),
            active_theme: None,
        }
    }

    pub fn register(&mut self, token: Token) {
        self.tokens.insert(token.name.clone(), token);
    }

    pub fn add_theme(&mut self, theme: Theme) {
        self.themes.insert(theme.name.clone(), theme);
    }

    pub fn set_active_theme(&mut self, name: impl Into<String>) {
        self.active_theme = Some(name.into());
    }

    /// Resolve a token value, applying theme overrides and following aliases.
    pub fn resolve(&self, name: &str) -> Option<TokenValue> {
        // Check theme override first.
        if let Some(theme_name) = &self.active_theme {
            if let Some(theme) = self.themes.get(theme_name) {
                if let Some(val) = theme.get(name) {
                    return Some(self.resolve_aliases(val, 10));
                }
            }
        }

        // Fall back to base token.
        let token = self.tokens.get(name)?;
        Some(self.resolve_aliases(&token.value, 10))
    }

    fn resolve_aliases(&self, value: &TokenValue, depth: usize) -> TokenValue {
        if depth == 0 {
            return value.clone();
        }
        if let TokenValue::Alias(target) = value {
            if let Some(token) = self.tokens.get(target) {
                return self.resolve_aliases(&token.value, depth - 1);
            }
        }
        value.clone()
    }

    /// Generate CSS custom properties for all tokens.
    pub fn to_css(&self) -> String {
        let mut css = String::from(":root {\n");
        let mut names: Vec<&String> = self.tokens.keys().collect();
        names.sort();
        for name in names {
            let token = &self.tokens[name];
            css.push_str(&format!("  {}\n", token.to_css_declaration()));
        }
        css.push_str("}\n");
        css
    }

    /// Generate CSS for a specific theme (as a data-attribute selector).
    pub fn theme_css(&self, theme_name: &str) -> Option<String> {
        let theme = self.themes.get(theme_name)?;
        let mut css = format!("[data-theme=\"{theme_name}\"] {{\n");
        let mut names: Vec<&String> = theme.overrides.keys().collect();
        names.sort();
        for name in names {
            let value = &theme.overrides[name];
            let prop = format!("--{}", name.replace('.', "-"));
            css.push_str(&format!("  {prop}: {};\n", value.css_value()));
        }
        css.push_str("}\n");
        Some(css)
    }

    /// Export all tokens as JSON.
    pub fn to_json(&self) -> String {
        let mut entries = Vec::new();
        let mut names: Vec<&String> = self.tokens.keys().collect();
        names.sort();
        for name in names {
            let token = &self.tokens[name];
            let val_json = match &token.value {
                TokenValue::Color(c) => format!("\"{}\"", c),
                TokenValue::Dimension(v, u) => format!("\"{}{}\"", v, u),
                TokenValue::Number(n) => format!("{}", n),
                TokenValue::Str(s) => format!("\"{}\"", s),
                TokenValue::Alias(a) => format!("\"${}\"", a),
            };
            let desc = token
                .description
                .as_deref()
                .map(|d| format!(", \"description\": \"{}\"", d))
                .unwrap_or_default();
            entries.push(format!(
                "  \"{}\": {{ \"value\": {}, \"category\": \"{}\"{}}}",
                name, val_json, token.category, desc
            ));
        }
        format!("{{\n{}\n}}", entries.join(",\n"))
    }

    /// Import tokens from a simple JSON object.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let parsed: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {e}"))?;

        let obj = parsed.as_object().ok_or("Expected JSON object")?;
        let mut store = TokenStore::new();

        for (name, val) in obj {
            let token_obj = val.as_object().ok_or(format!("Expected object for {name}"))?;
            let category_str = token_obj
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("color");
            let category = match category_str {
                "color" => TokenCategory::Color,
                "spacing" => TokenCategory::Spacing,
                "typography" => TokenCategory::Typography,
                "shadow" => TokenCategory::Shadow,
                "border" => TokenCategory::Border,
                "radius" => TokenCategory::Radius,
                "z-index" => TokenCategory::ZIndex,
                _ => TokenCategory::Color,
            };

            let value = if let Some(v) = token_obj.get("value") {
                if let Some(n) = v.as_f64() {
                    TokenValue::Number(n)
                } else if let Some(s) = v.as_str() {
                    if let Some(alias) = s.strip_prefix('$') {
                        TokenValue::Alias(alias.to_owned())
                    } else {
                        TokenValue::Str(s.to_owned())
                    }
                } else {
                    TokenValue::Str(v.to_string())
                }
            } else {
                continue;
            };

            let mut token = Token::new(name.clone(), value, category);
            if let Some(desc) = token_obj.get("description").and_then(|v| v.as_str()) {
                token = token.with_description(desc);
            }
            store.register(token);
        }

        Ok(store)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

impl Default for TokenStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_store() -> TokenStore {
        let mut store = TokenStore::new();
        store.register(Token::new(
            "color.primary",
            TokenValue::Color("#3b82f6".into()),
            TokenCategory::Color,
        ));
        store.register(Token::new(
            "color.bg",
            TokenValue::Color("#ffffff".into()),
            TokenCategory::Color,
        ));
        store.register(Token::new(
            "spacing.md",
            TokenValue::Dimension(16.0, "px".into()),
            TokenCategory::Spacing,
        ));
        store.register(Token::new(
            "z.modal",
            TokenValue::Number(1000.0),
            TokenCategory::ZIndex,
        ));
        store
    }

    #[test]
    fn test_token_css_property_name() {
        let t = Token::new("color.primary.500", TokenValue::Color("#f00".into()), TokenCategory::Color);
        assert_eq!(t.css_property_name(), "--color-primary-500");
    }

    #[test]
    fn test_token_css_declaration() {
        let t = Token::new("spacing.md", TokenValue::Dimension(16.0, "px".into()), TokenCategory::Spacing);
        assert_eq!(t.to_css_declaration(), "--spacing-md: 16px;");
    }

    #[test]
    fn test_store_resolve_base() {
        let store = sample_store();
        let val = store.resolve("color.primary").unwrap();
        assert_eq!(val.css_value(), "#3b82f6");
    }

    #[test]
    fn test_theme_override() {
        let mut store = sample_store();
        let dark = Theme::new("dark")
            .set("color.bg", TokenValue::Color("#1a1a2e".into()));
        store.add_theme(dark);
        store.set_active_theme("dark");
        let val = store.resolve("color.bg").unwrap();
        assert_eq!(val.css_value(), "#1a1a2e");
    }

    #[test]
    fn test_alias_resolution() {
        let mut store = TokenStore::new();
        store.register(Token::new("color.blue.500", TokenValue::Color("#3b82f6".into()), TokenCategory::Color));
        store.register(Token::new("color.primary", TokenValue::Alias("color.blue.500".into()), TokenCategory::Color));
        let val = store.resolve("color.primary").unwrap();
        assert_eq!(val.css_value(), "#3b82f6");
    }

    #[test]
    fn test_to_css() {
        let store = sample_store();
        let css = store.to_css();
        assert!(css.contains(":root {"));
        assert!(css.contains("--color-primary: #3b82f6;"));
        assert!(css.contains("--spacing-md: 16px;"));
    }

    #[test]
    fn test_theme_css() {
        let mut store = sample_store();
        let dark = Theme::new("dark")
            .set("color.bg", TokenValue::Color("#111".into()));
        store.add_theme(dark);
        let css = store.theme_css("dark").unwrap();
        assert!(css.contains("[data-theme=\"dark\"]"));
        assert!(css.contains("--color-bg: #111;"));
    }

    #[test]
    fn test_to_json() {
        let store = sample_store();
        let json = store.to_json();
        assert!(json.contains("\"color.primary\""));
        assert!(json.contains("#3b82f6"));
    }

    #[test]
    fn test_from_json_roundtrip() {
        let store = sample_store();
        let json = store.to_json();
        let store2 = TokenStore::from_json(&json).unwrap();
        assert_eq!(store2.len(), store.len());
        let val = store2.resolve("spacing.md").unwrap();
        assert_eq!(val.css_value(), "16px");
    }

    #[test]
    fn test_token_group() {
        let group = TokenGroup::new("colors")
            .add(Token::new("color.red", TokenValue::Color("#f00".into()), TokenCategory::Color))
            .add(Token::new("color.blue", TokenValue::Color("#00f".into()), TokenCategory::Color));
        assert_eq!(group.tokens.len(), 2);
        assert!(group.get("color.red").is_some());
        let css = group.to_css();
        assert!(css.contains("/* colors */"));
    }

    #[test]
    fn test_category_display() {
        assert_eq!(TokenCategory::ZIndex.to_string(), "z-index");
        assert_eq!(TokenCategory::Color.to_string(), "color");
    }

    #[test]
    fn test_token_value_display() {
        assert_eq!(TokenValue::Number(42.0).to_string(), "42");
        assert_eq!(TokenValue::Alias("foo".into()).to_string(), "var(--foo)");
    }

    #[test]
    fn test_from_json_invalid() {
        assert!(TokenStore::from_json("not json").is_err());
    }

    #[test]
    fn test_store_default() {
        let store = TokenStore::default();
        assert!(store.is_empty());
    }
}
