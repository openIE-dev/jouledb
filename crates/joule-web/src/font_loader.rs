//! Font metadata and loading — registry, fallback chains, CSS generation.
//!
//! Replaces FontFaceObserver and google-fonts-helper with a pure Rust
//! font metadata system. Manages font families, weights, styles, loading
//! states, and generates `@font-face` CSS declarations.

use std::collections::HashMap;
use std::fmt;

// ── Font Weight ────────────────────────────────────────────────

/// Font weight from 100 (Thin) to 900 (Black).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FontWeight(pub u16);

impl FontWeight {
    pub const THIN: Self = Self(100);
    pub const EXTRA_LIGHT: Self = Self(200);
    pub const LIGHT: Self = Self(300);
    pub const NORMAL: Self = Self(400);
    pub const MEDIUM: Self = Self(500);
    pub const SEMI_BOLD: Self = Self(600);
    pub const BOLD: Self = Self(700);
    pub const EXTRA_BOLD: Self = Self(800);
    pub const BLACK: Self = Self(900);

    pub fn new(weight: u16) -> Option<Self> {
        if (100..=900).contains(&weight) && weight % 100 == 0 {
            Some(Self(weight))
        } else {
            None
        }
    }
}

impl fmt::Display for FontWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── Font Style ─────────────────────────────────────────────────

/// Font style variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

impl fmt::Display for FontStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Italic => write!(f, "italic"),
            Self::Oblique => write!(f, "oblique"),
        }
    }
}

// ── Font Family ────────────────────────────────────────────────

/// A named font family.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FontFamily(pub String);

impl FontFamily {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Quote family name for CSS if it contains spaces.
    pub fn css_value(&self) -> String {
        if self.0.contains(' ') {
            format!("\"{}\"", self.0)
        } else {
            self.0.clone()
        }
    }
}

impl fmt::Display for FontFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── Loading State ──────────────────────────────────────────────

/// The loading state of a font face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontLoadState {
    Unloaded,
    Loading,
    Loaded,
    Error,
}

// ── Font Face ──────────────────────────────────────────────────

/// A single font face descriptor (one weight/style combo).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FontFace {
    pub family: FontFamily,
    pub weight: FontWeight,
    pub style: FontStyle,
    pub source_url: String,
    pub unicode_range: Option<String>,
    pub load_state: FontLoadState,
    pub format: Option<String>,
}

impl FontFace {
    pub fn new(
        family: impl Into<String>,
        weight: FontWeight,
        style: FontStyle,
        url: impl Into<String>,
    ) -> Self {
        Self {
            family: FontFamily::new(family),
            weight,
            style,
            source_url: url.into(),
            unicode_range: None,
            load_state: FontLoadState::Unloaded,
            format: None,
        }
    }

    pub fn unicode_range(mut self, range: impl Into<String>) -> Self {
        self.unicode_range = Some(range.into());
        self
    }

    pub fn format(mut self, fmt: impl Into<String>) -> Self {
        self.format = Some(fmt.into());
        self
    }

    /// Generate a `@font-face` CSS rule for this face.
    pub fn to_css(&self) -> String {
        let mut css = String::from("@font-face {\n");
        css.push_str(&format!(
            "  font-family: {};\n",
            self.family.css_value()
        ));
        css.push_str(&format!("  font-weight: {};\n", self.weight));
        css.push_str(&format!("  font-style: {};\n", self.style));

        if let Some(fmt) = &self.format {
            css.push_str(&format!(
                "  src: url(\"{}\") format(\"{fmt}\");\n",
                self.source_url
            ));
        } else {
            css.push_str(&format!("  src: url(\"{}\");\n", self.source_url));
        }

        if let Some(range) = &self.unicode_range {
            css.push_str(&format!("  unicode-range: {range};\n"));
        }

        css.push_str("}\n");
        css
    }

    fn lookup_key(&self) -> (String, FontWeight, FontStyle) {
        (self.family.0.to_lowercase(), self.weight, self.style)
    }
}

// ── Font Registry ──────────────────────────────────────────────

/// Registry for managing font faces with fallback chains.
#[derive(Debug, Clone)]
pub struct FontRegistry {
    faces: Vec<FontFace>,
    fallback_chains: HashMap<String, Vec<FontFamily>>,
}

impl FontRegistry {
    pub fn new() -> Self {
        Self {
            faces: Vec::new(),
            fallback_chains: HashMap::new(),
        }
    }

    /// Register a font face.
    pub fn register(&mut self, face: FontFace) {
        self.faces.push(face);
    }

    /// Look up a font face by family, weight, and style.
    pub fn lookup(
        &self,
        family: &str,
        weight: FontWeight,
        style: FontStyle,
    ) -> Option<&FontFace> {
        let key = (family.to_lowercase(), weight, style);
        self.faces.iter().find(|f| f.lookup_key() == key)
    }

    /// Look up with fallback: try exact match, then fallback chain.
    pub fn lookup_with_fallback(
        &self,
        family: &str,
        weight: FontWeight,
        style: FontStyle,
    ) -> Option<&FontFace> {
        // Try exact match first.
        if let Some(face) = self.lookup(family, weight, style) {
            return Some(face);
        }
        // Try fallback chain.
        let lower = family.to_lowercase();
        if let Some(chain) = self.fallback_chains.get(&lower) {
            for fallback in chain {
                if let Some(face) = self.lookup(&fallback.0, weight, style) {
                    return Some(face);
                }
            }
        }
        None
    }

    /// Define a fallback chain for a family.
    pub fn set_fallback_chain(&mut self, family: &str, chain: Vec<FontFamily>) {
        self.fallback_chains
            .insert(family.to_lowercase(), chain);
    }

    /// Get all registered faces for a family.
    pub fn faces_for_family(&self, family: &str) -> Vec<&FontFace> {
        let lower = family.to_lowercase();
        self.faces
            .iter()
            .filter(|f| f.family.0.to_lowercase() == lower)
            .collect()
    }

    /// Generate all `@font-face` CSS rules.
    pub fn to_css(&self) -> String {
        self.faces.iter().map(|f| f.to_css()).collect::<Vec<_>>().join("\n")
    }

    /// Total number of registered faces.
    pub fn len(&self) -> usize {
        self.faces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// Set loading state for a specific face.
    pub fn set_load_state(
        &mut self,
        family: &str,
        weight: FontWeight,
        style: FontStyle,
        state: FontLoadState,
    ) -> bool {
        let key = (family.to_lowercase(), weight, style);
        for face in &mut self.faces {
            if face.lookup_key() == key {
                face.load_state = state;
                return true;
            }
        }
        false
    }
}

impl Default for FontRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── System Font Stacks ─────────────────────────────────────────

/// Pre-defined system font stacks.
pub struct SystemFontStack;

impl SystemFontStack {
    /// Sans-serif system font stack.
    pub fn sans() -> Vec<FontFamily> {
        vec![
            FontFamily::new("system-ui"),
            FontFamily::new("-apple-system"),
            FontFamily::new("BlinkMacSystemFont"),
            FontFamily::new("Segoe UI"),
            FontFamily::new("Roboto"),
            FontFamily::new("Helvetica Neue"),
            FontFamily::new("Arial"),
            FontFamily::new("sans-serif"),
        ]
    }

    /// Serif system font stack.
    pub fn serif() -> Vec<FontFamily> {
        vec![
            FontFamily::new("Iowan Old Style"),
            FontFamily::new("Apple Garamond"),
            FontFamily::new("Baskerville"),
            FontFamily::new("Times New Roman"),
            FontFamily::new("Droid Serif"),
            FontFamily::new("Times"),
            FontFamily::new("Source Serif Pro"),
            FontFamily::new("serif"),
        ]
    }

    /// Monospace system font stack.
    pub fn mono() -> Vec<FontFamily> {
        vec![
            FontFamily::new("ui-monospace"),
            FontFamily::new("SFMono-Regular"),
            FontFamily::new("SF Mono"),
            FontFamily::new("Menlo"),
            FontFamily::new("Consolas"),
            FontFamily::new("Liberation Mono"),
            FontFamily::new("monospace"),
        ]
    }

    /// Generate CSS `font-family` value from a stack.
    pub fn to_css_value(stack: &[FontFamily]) -> String {
        stack.iter().map(|f| f.css_value()).collect::<Vec<_>>().join(", ")
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_weight_valid() {
        assert!(FontWeight::new(400).is_some());
        assert!(FontWeight::new(700).is_some());
        assert!(FontWeight::new(150).is_none());
        assert!(FontWeight::new(0).is_none());
        assert!(FontWeight::new(1000).is_none());
    }

    #[test]
    fn font_face_css_generation() {
        let face = FontFace::new(
            "Inter",
            FontWeight::NORMAL,
            FontStyle::Normal,
            "/fonts/inter-400.woff2",
        )
        .format("woff2");

        let css = face.to_css();
        assert!(css.contains("font-family: Inter;"));
        assert!(css.contains("font-weight: 400;"));
        assert!(css.contains("font-style: normal;"));
        assert!(css.contains("format(\"woff2\")"));
    }

    #[test]
    fn font_face_unicode_range() {
        let face = FontFace::new(
            "Noto Sans",
            FontWeight::NORMAL,
            FontStyle::Normal,
            "/fonts/noto.woff2",
        )
        .unicode_range("U+0000-00FF");

        let css = face.to_css();
        assert!(css.contains("unicode-range: U+0000-00FF;"));
        assert!(css.contains("font-family: \"Noto Sans\";"));
    }

    #[test]
    fn registry_register_lookup() {
        let mut reg = FontRegistry::new();
        reg.register(FontFace::new(
            "Inter",
            FontWeight::NORMAL,
            FontStyle::Normal,
            "/fonts/inter-400.woff2",
        ));
        reg.register(FontFace::new(
            "Inter",
            FontWeight::BOLD,
            FontStyle::Normal,
            "/fonts/inter-700.woff2",
        ));

        assert_eq!(reg.len(), 2);
        let face = reg.lookup("Inter", FontWeight::NORMAL, FontStyle::Normal);
        assert!(face.is_some());
        assert_eq!(face.unwrap().source_url, "/fonts/inter-400.woff2");

        assert!(reg
            .lookup("Inter", FontWeight::NORMAL, FontStyle::Italic)
            .is_none());
    }

    #[test]
    fn registry_case_insensitive() {
        let mut reg = FontRegistry::new();
        reg.register(FontFace::new(
            "Inter",
            FontWeight::NORMAL,
            FontStyle::Normal,
            "/fonts/inter.woff2",
        ));
        assert!(reg.lookup("inter", FontWeight::NORMAL, FontStyle::Normal).is_some());
        assert!(reg.lookup("INTER", FontWeight::NORMAL, FontStyle::Normal).is_some());
    }

    #[test]
    fn fallback_chain() {
        let mut reg = FontRegistry::new();
        reg.register(FontFace::new(
            "Fallback Sans",
            FontWeight::NORMAL,
            FontStyle::Normal,
            "/fonts/fallback.woff2",
        ));
        reg.set_fallback_chain(
            "Primary Sans",
            vec![FontFamily::new("Fallback Sans")],
        );

        // Direct lookup fails.
        assert!(reg
            .lookup("Primary Sans", FontWeight::NORMAL, FontStyle::Normal)
            .is_none());
        // Fallback succeeds.
        let face = reg.lookup_with_fallback(
            "Primary Sans",
            FontWeight::NORMAL,
            FontStyle::Normal,
        );
        assert!(face.is_some());
        assert_eq!(face.unwrap().family.0, "Fallback Sans");
    }

    #[test]
    fn load_state_transition() {
        let mut reg = FontRegistry::new();
        reg.register(FontFace::new(
            "Test",
            FontWeight::NORMAL,
            FontStyle::Normal,
            "/test.woff2",
        ));
        let face = reg.lookup("Test", FontWeight::NORMAL, FontStyle::Normal).unwrap();
        assert_eq!(face.load_state, FontLoadState::Unloaded);

        reg.set_load_state("Test", FontWeight::NORMAL, FontStyle::Normal, FontLoadState::Loading);
        let face = reg.lookup("Test", FontWeight::NORMAL, FontStyle::Normal).unwrap();
        assert_eq!(face.load_state, FontLoadState::Loading);

        reg.set_load_state("Test", FontWeight::NORMAL, FontStyle::Normal, FontLoadState::Loaded);
        let face = reg.lookup("Test", FontWeight::NORMAL, FontStyle::Normal).unwrap();
        assert_eq!(face.load_state, FontLoadState::Loaded);
    }

    #[test]
    fn faces_for_family() {
        let mut reg = FontRegistry::new();
        reg.register(FontFace::new("Roboto", FontWeight::LIGHT, FontStyle::Normal, "/r300.woff2"));
        reg.register(FontFace::new("Roboto", FontWeight::NORMAL, FontStyle::Normal, "/r400.woff2"));
        reg.register(FontFace::new("Roboto", FontWeight::BOLD, FontStyle::Normal, "/r700.woff2"));
        reg.register(FontFace::new("Open Sans", FontWeight::NORMAL, FontStyle::Normal, "/os.woff2"));

        assert_eq!(reg.faces_for_family("Roboto").len(), 3);
        assert_eq!(reg.faces_for_family("Open Sans").len(), 1);
    }

    #[test]
    fn system_stacks() {
        let sans = SystemFontStack::sans();
        assert!(!sans.is_empty());
        assert_eq!(sans[0].0, "system-ui");

        let mono = SystemFontStack::mono();
        assert!(mono.iter().any(|f| f.0 == "monospace"));

        let css = SystemFontStack::to_css_value(&sans);
        assert!(css.contains("system-ui"));
        assert!(css.contains("\"Segoe UI\""));
    }

    #[test]
    fn generate_all_css() {
        let mut reg = FontRegistry::new();
        reg.register(FontFace::new("A", FontWeight::NORMAL, FontStyle::Normal, "/a.woff2"));
        reg.register(FontFace::new("B", FontWeight::BOLD, FontStyle::Italic, "/b.woff2"));
        let css = reg.to_css();
        assert!(css.contains("font-family: A;"));
        assert!(css.contains("font-family: B;"));
        assert!(css.contains("font-style: italic;"));
    }

    #[test]
    fn font_family_quoting() {
        let plain = FontFamily::new("Arial");
        assert_eq!(plain.css_value(), "Arial");
        let spaced = FontFamily::new("Open Sans");
        assert_eq!(spaced.css_value(), "\"Open Sans\"");
    }
}
