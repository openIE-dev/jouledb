//! CSS-in-Rust — style object builder, property types, pseudo-classes/elements,
//! media queries, keyframe animations, className generation, and style sheet output.
//!
//! Build type-safe CSS in Rust without string templates. Pure Rust, no browser APIs.

use std::collections::BTreeMap;
use std::fmt;

// ── Style Property ───────────────────────────────────────────────

/// A CSS property-value pair.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Property {
    pub name: String,
    pub value: String,
}

impl Property {
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self { name: name.into(), value: value.into() }
    }
}

impl fmt::Display for Property {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {};", self.name, self.value)
    }
}

// ── Pseudo Selector ──────────────────────────────────────────────

/// CSS pseudo-class or pseudo-element.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pseudo {
    // Pseudo-classes
    Hover,
    Focus,
    FocusVisible,
    FocusWithin,
    Active,
    Visited,
    FirstChild,
    LastChild,
    NthChild(String),
    Disabled,
    Checked,
    Placeholder,
    // Pseudo-elements
    Before,
    After,
    FirstLine,
    FirstLetter,
    Selection,
    Marker,
    // Custom
    Custom(String),
}

impl fmt::Display for Pseudo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hover => write!(f, ":hover"),
            Self::Focus => write!(f, ":focus"),
            Self::FocusVisible => write!(f, ":focus-visible"),
            Self::FocusWithin => write!(f, ":focus-within"),
            Self::Active => write!(f, ":active"),
            Self::Visited => write!(f, ":visited"),
            Self::FirstChild => write!(f, ":first-child"),
            Self::LastChild => write!(f, ":last-child"),
            Self::NthChild(s) => write!(f, ":nth-child({s})"),
            Self::Disabled => write!(f, ":disabled"),
            Self::Checked => write!(f, ":checked"),
            Self::Placeholder => write!(f, "::placeholder"),
            Self::Before => write!(f, "::before"),
            Self::After => write!(f, "::after"),
            Self::FirstLine => write!(f, "::first-line"),
            Self::FirstLetter => write!(f, "::first-letter"),
            Self::Selection => write!(f, "::selection"),
            Self::Marker => write!(f, "::marker"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

// ── Keyframe ─────────────────────────────────────────────────────

/// A single keyframe step.
#[derive(Debug, Clone)]
pub struct KeyframeStep {
    pub position: String,  // "0%", "50%", "from", "to"
    pub properties: Vec<Property>,
}

impl KeyframeStep {
    pub fn new(position: impl Into<String>) -> Self {
        Self { position: position.into(), properties: Vec::new() }
    }

    pub fn at(pct: u8) -> Self { Self::new(format!("{pct}%")) }
    pub fn from() -> Self { Self::new("from") }
    pub fn to() -> Self { Self::new("to") }

    pub fn prop(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.push(Property::new(name, value));
        self
    }

    /// Render this step as CSS text.
    pub fn to_css(&self, indent: &str) -> String {
        let mut out = format!("{indent}{} {{\n", self.position);
        for p in &self.properties {
            out.push_str(&format!("{indent}  {}\n", p));
        }
        out.push_str(&format!("{indent}}}\n"));
        out
    }
}

/// Named keyframe animation.
#[derive(Debug, Clone)]
pub struct Keyframes {
    pub name: String,
    pub steps: Vec<KeyframeStep>,
}

impl Keyframes {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), steps: Vec::new() }
    }

    pub fn step(mut self, step: KeyframeStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Render as CSS @keyframes block.
    pub fn to_css(&self) -> String {
        let mut out = format!("@keyframes {} {{\n", self.name);
        for s in &self.steps {
            out.push_str(&s.to_css("  "));
        }
        out.push_str("}\n");
        out
    }
}

// ── Style Rule ───────────────────────────────────────────────────

/// A single CSS rule (selector + properties).
#[derive(Debug, Clone)]
pub struct StyleRule {
    pub selector: String,
    pub properties: Vec<Property>,
    pub pseudo_rules: BTreeMap<Pseudo, Vec<Property>>,
    pub media_rules: Vec<(String, Vec<Property>)>,
}

impl StyleRule {
    pub fn new(selector: impl Into<String>) -> Self {
        Self {
            selector: selector.into(),
            properties: Vec::new(),
            pseudo_rules: BTreeMap::new(),
            media_rules: Vec::new(),
        }
    }

    /// Add a CSS property.
    pub fn prop(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.push(Property::new(name, value));
        self
    }

    /// Add property under a pseudo-class/element.
    pub fn pseudo(mut self, pseudo: Pseudo, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.pseudo_rules
            .entry(pseudo)
            .or_default()
            .push(Property::new(name, value));
        self
    }

    /// Add property under a media query.
    pub fn media(mut self, query: impl Into<String>, name: impl Into<String>, value: impl Into<String>) -> Self {
        let q = query.into();
        if let Some(entry) = self.media_rules.iter_mut().find(|(mq, _)| mq == &q) {
            entry.1.push(Property::new(name, value));
        } else {
            self.media_rules.push((q, vec![Property::new(name, value)]));
        }
        self
    }

    /// Render as CSS text.
    pub fn to_css(&self) -> String {
        let mut out = String::new();

        // Main rule
        if !self.properties.is_empty() {
            out.push_str(&format!("{} {{\n", self.selector));
            for p in &self.properties {
                out.push_str(&format!("  {}\n", p));
            }
            out.push_str("}\n");
        }

        // Pseudo rules
        for (pseudo, props) in &self.pseudo_rules {
            out.push_str(&format!("{}{} {{\n", self.selector, pseudo));
            for p in props {
                out.push_str(&format!("  {}\n", p));
            }
            out.push_str("}\n");
        }

        // Media rules
        for (query, props) in &self.media_rules {
            out.push_str(&format!("@media {} {{\n", query));
            out.push_str(&format!("  {} {{\n", self.selector));
            for p in props {
                out.push_str(&format!("    {}\n", p));
            }
            out.push_str("  }\n");
            out.push_str("}\n");
        }

        out
    }

    /// Count of base properties.
    pub fn property_count(&self) -> usize { self.properties.len() }
}

// ── Class Name Generator ─────────────────────────────────────────

/// Simple deterministic hash-based class name generator.
pub struct ClassNameGenerator {
    prefix: String,
    counter: u64,
}

impl ClassNameGenerator {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self { prefix: prefix.into(), counter: 0 }
    }

    /// Generate a unique class name.
    pub fn next(&mut self) -> String {
        let id = self.counter;
        self.counter += 1;
        format!("{}{}", self.prefix, to_base36(id))
    }

    /// Generate a class name from a hash of the content.
    pub fn hashed(prefix: &str, content: &str) -> String {
        let hash = simple_hash(content);
        format!("{}{}", prefix, to_base36(hash))
    }
}

fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

fn to_base36(mut n: u64) -> String {
    if n == 0 { return "0".to_string(); }
    let chars: Vec<char> = "0123456789abcdefghijklmnopqrstuvwxyz".chars().collect();
    let mut result = Vec::new();
    while n > 0 {
        result.push(chars[(n % 36) as usize]);
        n /= 36;
    }
    result.reverse();
    result.into_iter().collect()
}

// ── Style Sheet ──────────────────────────────────────────────────

/// Collection of style rules forming a stylesheet.
#[derive(Debug, Clone)]
pub struct StyleSheet {
    rules: Vec<StyleRule>,
    keyframes: Vec<Keyframes>,
    global_vars: BTreeMap<String, String>,
}

impl StyleSheet {
    pub fn new() -> Self {
        Self { rules: Vec::new(), keyframes: Vec::new(), global_vars: BTreeMap::new() }
    }

    /// Add a style rule.
    pub fn rule(mut self, rule: StyleRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add keyframe animation.
    pub fn keyframes(mut self, kf: Keyframes) -> Self {
        self.keyframes.push(kf);
        self
    }

    /// Set a CSS custom property on :root.
    pub fn var(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.global_vars.insert(name.into(), value.into());
        self
    }

    /// Render the entire stylesheet as CSS text.
    pub fn to_css(&self) -> String {
        let mut out = String::new();

        // CSS custom properties
        if !self.global_vars.is_empty() {
            out.push_str(":root {\n");
            for (name, value) in &self.global_vars {
                out.push_str(&format!("  --{}: {};\n", name, value));
            }
            out.push_str("}\n");
        }

        // Keyframes
        for kf in &self.keyframes {
            out.push_str(&kf.to_css());
        }

        // Rules
        for rule in &self.rules {
            out.push_str(&rule.to_css());
        }

        out
    }

    /// Number of rules.
    pub fn rule_count(&self) -> usize { self.rules.len() }

    /// Number of keyframe animations.
    pub fn keyframe_count(&self) -> usize { self.keyframes.len() }
}

impl Default for StyleSheet {
    fn default() -> Self { Self::new() }
}

// ── Convenience Builders ─────────────────────────────────────────

/// Quick display property.
pub fn display(value: &str) -> Property { Property::new("display", value) }
pub fn flex() -> Property { Property::new("display", "flex") }
pub fn grid() -> Property { Property::new("display", "grid") }
pub fn margin(value: &str) -> Property { Property::new("margin", value) }
pub fn padding(value: &str) -> Property { Property::new("padding", value) }
pub fn color(value: &str) -> Property { Property::new("color", value) }
pub fn background(value: &str) -> Property { Property::new("background", value) }
pub fn font_size(value: &str) -> Property { Property::new("font-size", value) }
pub fn width(value: &str) -> Property { Property::new("width", value) }
pub fn height(value: &str) -> Property { Property::new("height", value) }

/// Create a fade-in animation.
pub fn fade_in(name: &str, duration_ms: u32) -> (Keyframes, Property) {
    let kf = Keyframes::new(name)
        .step(KeyframeStep::from().prop("opacity", "0"))
        .step(KeyframeStep::to().prop("opacity", "1"));
    let anim = Property::new("animation", format!("{name} {duration_ms}ms ease-in"));
    (kf, anim)
}

/// Create a slide-in-from-bottom animation.
pub fn slide_in_up(name: &str, distance_px: u32, duration_ms: u32) -> (Keyframes, Property) {
    let kf = Keyframes::new(name)
        .step(KeyframeStep::from()
            .prop("transform", format!("translateY({distance_px}px)"))
            .prop("opacity", "0"))
        .step(KeyframeStep::to()
            .prop("transform", "translateY(0)")
            .prop("opacity", "1"));
    let anim = Property::new("animation", format!("{name} {duration_ms}ms ease-out"));
    (kf, anim)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_display() {
        let p = Property::new("color", "red");
        assert_eq!(p.to_string(), "color: red;");
    }

    #[test]
    fn test_style_rule_basic() {
        let rule = StyleRule::new(".btn")
            .prop("display", "inline-flex")
            .prop("padding", "8px 16px")
            .prop("border-radius", "4px");

        let css = rule.to_css();
        assert!(css.contains(".btn {"));
        assert!(css.contains("display: inline-flex;"));
        assert!(css.contains("padding: 8px 16px;"));
    }

    #[test]
    fn test_style_rule_pseudo() {
        let rule = StyleRule::new(".link")
            .prop("color", "blue")
            .pseudo(Pseudo::Hover, "color", "darkblue")
            .pseudo(Pseudo::Visited, "color", "purple");

        let css = rule.to_css();
        assert!(css.contains(".link:hover {"));
        assert!(css.contains("color: darkblue;"));
        assert!(css.contains(".link:visited {"));
    }

    #[test]
    fn test_style_rule_media() {
        let rule = StyleRule::new(".container")
            .prop("width", "100%")
            .media("(min-width: 768px)", "width", "750px")
            .media("(min-width: 1200px)", "width", "1170px");

        let css = rule.to_css();
        assert!(css.contains("@media (min-width: 768px)"));
        assert!(css.contains("width: 750px;"));
        assert!(css.contains("@media (min-width: 1200px)"));
    }

    #[test]
    fn test_keyframes() {
        let kf = Keyframes::new("fadeIn")
            .step(KeyframeStep::from().prop("opacity", "0"))
            .step(KeyframeStep::to().prop("opacity", "1"));

        let css = kf.to_css();
        assert!(css.contains("@keyframes fadeIn {"));
        assert!(css.contains("from {"));
        assert!(css.contains("to {"));
        assert!(css.contains("opacity: 0;"));
        assert!(css.contains("opacity: 1;"));
    }

    #[test]
    fn test_keyframes_percentage() {
        let kf = Keyframes::new("bounce")
            .step(KeyframeStep::at(0).prop("transform", "translateY(0)"))
            .step(KeyframeStep::at(50).prop("transform", "translateY(-20px)"))
            .step(KeyframeStep::at(100).prop("transform", "translateY(0)"));

        let css = kf.to_css();
        assert!(css.contains("0% {"));
        assert!(css.contains("50% {"));
        assert!(css.contains("100% {"));
    }

    #[test]
    fn test_class_name_generator_sequential() {
        let mut cng = ClassNameGenerator::new("css-");
        assert_eq!(cng.next(), "css-0");
        assert_eq!(cng.next(), "css-1");
        assert_eq!(cng.next(), "css-2");
    }

    #[test]
    fn test_class_name_generator_hashed() {
        let a = ClassNameGenerator::hashed("s-", "color: red;");
        let b = ClassNameGenerator::hashed("s-", "color: blue;");
        assert_ne!(a, b);

        // Same input = same hash
        let c = ClassNameGenerator::hashed("s-", "color: red;");
        assert_eq!(a, c);
    }

    #[test]
    fn test_stylesheet_full() {
        let sheet = StyleSheet::new()
            .var("primary", "#007bff")
            .var("spacing", "8px")
            .keyframes(Keyframes::new("fadeIn")
                .step(KeyframeStep::from().prop("opacity", "0"))
                .step(KeyframeStep::to().prop("opacity", "1")))
            .rule(StyleRule::new(".card")
                .prop("padding", "var(--spacing)")
                .prop("background", "white")
                .pseudo(Pseudo::Hover, "box-shadow", "0 2px 8px rgba(0,0,0,0.1)"));

        let css = sheet.to_css();
        assert!(css.contains(":root {"));
        assert!(css.contains("--primary: #007bff;"));
        assert!(css.contains("@keyframes fadeIn"));
        assert!(css.contains(".card {"));
        assert!(css.contains(".card:hover {"));
    }

    #[test]
    fn test_stylesheet_counts() {
        let sheet = StyleSheet::new()
            .rule(StyleRule::new(".a").prop("color", "red"))
            .rule(StyleRule::new(".b").prop("color", "blue"))
            .keyframes(Keyframes::new("spin"));

        assert_eq!(sheet.rule_count(), 2);
        assert_eq!(sheet.keyframe_count(), 1);
    }

    #[test]
    fn test_pseudo_display() {
        assert_eq!(Pseudo::Hover.to_string(), ":hover");
        assert_eq!(Pseudo::Before.to_string(), "::before");
        assert_eq!(Pseudo::After.to_string(), "::after");
        assert_eq!(Pseudo::FocusVisible.to_string(), ":focus-visible");
        assert_eq!(Pseudo::NthChild("2n+1".into()).to_string(), ":nth-child(2n+1)");
        assert_eq!(Pseudo::Selection.to_string(), "::selection");
        assert_eq!(Pseudo::Marker.to_string(), "::marker");
    }

    #[test]
    fn test_convenience_builders() {
        assert_eq!(flex().to_string(), "display: flex;");
        assert_eq!(grid().to_string(), "display: grid;");
        assert_eq!(margin("0 auto").to_string(), "margin: 0 auto;");
        assert_eq!(padding("16px").to_string(), "padding: 16px;");
        assert_eq!(color("red").to_string(), "color: red;");
        assert_eq!(font_size("14px").to_string(), "font-size: 14px;");
    }

    #[test]
    fn test_fade_in_helper() {
        let (kf, anim) = fade_in("fadeIn", 300);
        assert_eq!(kf.name, "fadeIn");
        assert_eq!(kf.steps.len(), 2);
        assert!(anim.value.contains("300ms"));
    }

    #[test]
    fn test_slide_in_up_helper() {
        let (kf, anim) = slide_in_up("slideUp", 20, 500);
        assert_eq!(kf.name, "slideUp");
        assert_eq!(kf.steps.len(), 2);
        assert!(anim.value.contains("500ms"));
    }

    #[test]
    fn test_style_rule_property_count() {
        let rule = StyleRule::new(".x")
            .prop("a", "1")
            .prop("b", "2")
            .prop("c", "3");
        assert_eq!(rule.property_count(), 3);
    }

    #[test]
    fn test_media_rule_grouping() {
        let rule = StyleRule::new(".box")
            .media("(min-width: 768px)", "width", "50%")
            .media("(min-width: 768px)", "padding", "16px");

        // Both props should be in one @media block
        let css = rule.to_css();
        let media_count = css.matches("@media (min-width: 768px)").count();
        assert_eq!(media_count, 1);
    }

    #[test]
    fn test_empty_rule_no_output() {
        let rule = StyleRule::new(".empty");
        let css = rule.to_css();
        assert!(!css.contains(".empty {"));
    }

    #[test]
    fn test_to_base36() {
        assert_eq!(to_base36(0), "0");
        assert_eq!(to_base36(35), "z");
        assert_eq!(to_base36(36), "10");
    }

    #[test]
    fn test_custom_pseudo() {
        let rule = StyleRule::new(".nav")
            .pseudo(Pseudo::Custom(":is(.active, .current)".into()), "font-weight", "bold");

        let css = rule.to_css();
        assert!(css.contains(".nav:is(.active, .current) {"));
    }

    #[test]
    fn test_pseudo_focus_within() {
        assert_eq!(Pseudo::FocusWithin.to_string(), ":focus-within");
    }

    #[test]
    fn test_pseudo_first_last_child() {
        assert_eq!(Pseudo::FirstChild.to_string(), ":first-child");
        assert_eq!(Pseudo::LastChild.to_string(), ":last-child");
    }

    #[test]
    fn test_pseudo_disabled_checked() {
        assert_eq!(Pseudo::Disabled.to_string(), ":disabled");
        assert_eq!(Pseudo::Checked.to_string(), ":checked");
    }

    #[test]
    fn test_pseudo_placeholder() {
        assert_eq!(Pseudo::Placeholder.to_string(), "::placeholder");
    }

    #[test]
    fn test_pseudo_first_line_letter() {
        assert_eq!(Pseudo::FirstLine.to_string(), "::first-line");
        assert_eq!(Pseudo::FirstLetter.to_string(), "::first-letter");
    }

    #[test]
    fn test_stylesheet_vars_only() {
        let sheet = StyleSheet::new()
            .var("bg", "white")
            .var("fg", "black");

        let css = sheet.to_css();
        assert!(css.contains(":root {"));
        assert!(css.contains("--bg: white;"));
        assert!(css.contains("--fg: black;"));
    }

    #[test]
    fn test_keyframe_step_to_css() {
        let step = KeyframeStep::at(50)
            .prop("transform", "scale(1.2)")
            .prop("opacity", "0.5");

        let css = step.to_css("  ");
        assert!(css.contains("50% {"));
        assert!(css.contains("transform: scale(1.2);"));
    }
}
