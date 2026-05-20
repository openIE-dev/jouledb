//! CSS style resolution — specificity calculation, cascade ordering, inheritance,
//! computed values, style rule matching, and selector matching.
//!
//! Replaces browser CSS resolution logic. Computes specificity per selector,
//! resolves the cascade (origin, specificity, source order), applies inheritance,
//! resolves computed values, and matches selectors against elements.

use std::collections::HashMap;

// ── Specificity ─────────────────────────────────────────────────────────

/// CSS specificity as (inline, ids, classes, elements).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Specificity {
    pub inline: u16,
    pub ids: u16,
    pub classes: u16,
    pub elements: u16,
}

impl Specificity {
    pub fn new(inline: u16, ids: u16, classes: u16, elements: u16) -> Self {
        Self { inline, ids, classes, elements }
    }

    pub fn zero() -> Self {
        Self { inline: 0, ids: 0, classes: 0, elements: 0 }
    }

    /// Numeric value for comparison (inline*1000000 + ids*10000 + classes*100 + elements).
    pub fn numeric_value(&self) -> u64 {
        (self.inline as u64) * 1_000_000
            + (self.ids as u64) * 10_000
            + (self.classes as u64) * 100
            + self.elements as u64
    }

    /// Calculate specificity from a simple selector string.
    pub fn from_selector(selector: &str) -> Self {
        let mut ids = 0u16;
        let mut classes = 0u16;
        let mut elements = 0u16;

        // Split on whitespace for descendant combinators, then parse each part
        for part in selector.split_whitespace() {
            // Split on combinators > + ~
            for segment in split_combinators(part) {
                let seg = segment.trim();
                if seg.is_empty() {
                    continue;
                }

                let mut chars = seg.chars().peekable();
                while let Some(&ch) = chars.peek() {
                    match ch {
                        '#' => {
                            ids += 1;
                            chars.next();
                            // consume the identifier
                            while let Some(&c) = chars.peek() {
                                if c.is_alphanumeric() || c == '-' || c == '_' {
                                    chars.next();
                                } else {
                                    break;
                                }
                            }
                        }
                        '.' => {
                            classes += 1;
                            chars.next();
                            while let Some(&c) = chars.peek() {
                                if c.is_alphanumeric() || c == '-' || c == '_' {
                                    chars.next();
                                } else {
                                    break;
                                }
                            }
                        }
                        '[' => {
                            classes += 1;
                            chars.next();
                            // consume until ]
                            while let Some(&c) = chars.peek() {
                                chars.next();
                                if c == ']' {
                                    break;
                                }
                            }
                        }
                        ':' => {
                            chars.next();
                            if let Some(&next_ch) = chars.peek() {
                                if next_ch == ':' {
                                    // pseudo-element
                                    elements += 1;
                                    chars.next();
                                } else {
                                    // pseudo-class
                                    classes += 1;
                                }
                            }
                            while let Some(&c) = chars.peek() {
                                if c.is_alphanumeric() || c == '-' || c == '_' {
                                    chars.next();
                                } else {
                                    break;
                                }
                            }
                        }
                        '*' => {
                            // universal selector — no specificity
                            chars.next();
                        }
                        _ if ch.is_alphabetic() => {
                            elements += 1;
                            while let Some(&c) = chars.peek() {
                                if c.is_alphanumeric() || c == '-' || c == '_' {
                                    chars.next();
                                } else {
                                    break;
                                }
                            }
                        }
                        _ => {
                            chars.next();
                        }
                    }
                }
            }
        }

        Self { inline: 0, ids, classes, elements }
    }
}

fn split_combinators(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for ch in s.chars() {
        if ch == '>' || ch == '+' || ch == '~' {
            if !current.is_empty() {
                parts.push(current.clone());
                current.clear();
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

// ── Cascade origin ──────────────────────────────────────────────────────

/// Origin of a style rule in the cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CascadeOrigin {
    UserAgent = 0,
    User = 1,
    Author = 2,
    AuthorImportant = 3,
    UserImportant = 4,
    UserAgentImportant = 5,
}

// ── Style rules ─────────────────────────────────────────────────────────

/// A CSS property value.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyValue {
    pub value: String,
    pub important: bool,
}

impl PropertyValue {
    pub fn new(value: &str) -> Self {
        Self { value: value.to_string(), important: false }
    }

    pub fn important(value: &str) -> Self {
        Self { value: value.to_string(), important: true }
    }
}

/// A single style rule with selector, declarations, origin, and source order.
#[derive(Debug, Clone)]
pub struct StyleRule {
    pub selector: String,
    pub declarations: HashMap<String, PropertyValue>,
    pub origin: CascadeOrigin,
    pub source_order: u32,
}

impl StyleRule {
    pub fn new(selector: &str, origin: CascadeOrigin, source_order: u32) -> Self {
        Self {
            selector: selector.to_string(),
            declarations: HashMap::new(),
            origin,
            source_order,
        }
    }

    pub fn with_property(mut self, property: &str, value: &str) -> Self {
        self.declarations.insert(property.to_string(), PropertyValue::new(value));
        self
    }

    pub fn with_important(mut self, property: &str, value: &str) -> Self {
        self.declarations.insert(property.to_string(), PropertyValue::important(value));
        self
    }

    pub fn specificity(&self) -> Specificity {
        Specificity::from_selector(&self.selector)
    }
}

// ── Element matching ────────────────────────────────────────────────────

/// A simplified element for selector matching.
#[derive(Debug, Clone)]
pub struct Element {
    pub tag: String,
    pub id: Option<String>,
    pub classes: Vec<String>,
    pub attributes: HashMap<String, String>,
    pub parent: Option<Box<Element>>,
}

impl Element {
    pub fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            id: None,
            classes: Vec::new(),
            attributes: HashMap::new(),
            parent: None,
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    pub fn with_class(mut self, class: &str) -> Self {
        self.classes.push(class.to_string());
        self
    }

    pub fn with_attr(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_parent(mut self, parent: Element) -> Self {
        self.parent = Some(Box::new(parent));
        self
    }
}

/// Test if a simple selector matches an element.
/// Supports: tag, #id, .class, [attr], * (universal), and combinations like div.class#id.
pub fn selector_matches(selector: &str, element: &Element) -> bool {
    // Split compound selector into parts
    let parts = parse_simple_parts(selector.trim());
    for part in &parts {
        if !simple_part_matches(part, element) {
            return false;
        }
    }
    true
}

/// A parsed simple selector part.
#[derive(Debug)]
enum SelectorPart {
    Tag(String),
    Id(String),
    Class(String),
    Attr(String),
    Universal,
}

fn parse_simple_parts(selector: &str) -> Vec<SelectorPart> {
    let mut parts = Vec::new();
    let mut chars = selector.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            '#' => {
                chars.next();
                let mut name = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '-' || c == '_' {
                        name.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if !name.is_empty() {
                    parts.push(SelectorPart::Id(name));
                }
            }
            '.' => {
                chars.next();
                let mut name = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '-' || c == '_' {
                        name.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if !name.is_empty() {
                    parts.push(SelectorPart::Class(name));
                }
            }
            '[' => {
                chars.next();
                let mut attr = String::new();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == ']' {
                        break;
                    }
                    attr.push(c);
                }
                if !attr.is_empty() {
                    parts.push(SelectorPart::Attr(attr));
                }
            }
            '*' => {
                chars.next();
                parts.push(SelectorPart::Universal);
            }
            _ if ch.is_alphabetic() => {
                let mut name = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '-' || c == '_' {
                        name.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if !name.is_empty() {
                    parts.push(SelectorPart::Tag(name));
                }
            }
            _ => {
                chars.next();
            }
        }
    }

    parts
}

fn simple_part_matches(part: &SelectorPart, element: &Element) -> bool {
    match part {
        SelectorPart::Tag(tag) => element.tag == *tag,
        SelectorPart::Id(id) => element.id.as_deref() == Some(id.as_str()),
        SelectorPart::Class(cls) => element.classes.contains(cls),
        SelectorPart::Attr(attr) => {
            // Support [attr] and [attr=value]
            if let Some(eq_idx) = attr.find('=') {
                let key = &attr[..eq_idx];
                let val = attr[eq_idx + 1..].trim_matches('"').trim_matches('\'');
                element.attributes.get(key).map_or(false, |v| v == val)
            } else {
                element.attributes.contains_key(attr.as_str())
            }
        }
        SelectorPart::Universal => true,
    }
}

// ── Style resolver ──────────────────────────────────────────────────────

/// Properties known to be inherited by default in CSS.
const INHERITED_PROPERTIES: &[&str] = &[
    "color", "font-family", "font-size", "font-style", "font-weight",
    "line-height", "text-align", "visibility", "cursor", "letter-spacing",
    "word-spacing", "white-space", "direction", "list-style",
];

/// Default property values.
fn default_value(property: &str) -> &str {
    match property {
        "display" => "block",
        "position" => "static",
        "color" => "black",
        "background-color" => "transparent",
        "font-size" => "16px",
        "font-weight" => "normal",
        "font-family" => "serif",
        "font-style" => "normal",
        "text-align" => "start",
        "line-height" => "normal",
        "margin" => "0",
        "padding" => "0",
        "border" => "none",
        "width" => "auto",
        "height" => "auto",
        "opacity" => "1",
        "visibility" => "visible",
        "cursor" => "auto",
        "overflow" => "visible",
        _ => "initial",
    }
}

/// Check if a property is inherited.
pub fn is_inherited(property: &str) -> bool {
    INHERITED_PROPERTIES.contains(&property)
}

/// Resolved style for an element — the final computed properties.
#[derive(Debug, Clone)]
pub struct ComputedStyle {
    pub properties: HashMap<String, String>,
}

impl ComputedStyle {
    pub fn new() -> Self {
        Self { properties: HashMap::new() }
    }

    pub fn get(&self, property: &str) -> Option<&str> {
        self.properties.get(property).map(|s| s.as_str())
    }

    pub fn set(&mut self, property: &str, value: &str) {
        self.properties.insert(property.to_string(), value.to_string());
    }
}

impl Default for ComputedStyle {
    fn default() -> Self {
        Self::new()
    }
}

/// Cascade entry for sorting.
#[derive(Debug)]
struct CascadeEntry {
    property: String,
    value: String,
    important: bool,
    origin: CascadeOrigin,
    specificity: Specificity,
    source_order: u32,
}

/// Resolve styles for an element given a set of rules and optional parent style.
pub fn resolve_style(
    element: &Element,
    rules: &[StyleRule],
    parent_style: Option<&ComputedStyle>,
) -> ComputedStyle {
    let mut entries: Vec<CascadeEntry> = Vec::new();

    // Collect all matching rules
    for rule in rules {
        if selector_matches(&rule.selector, element) {
            let spec = rule.specificity();
            for (prop, pv) in &rule.declarations {
                let origin = if pv.important {
                    match rule.origin {
                        CascadeOrigin::Author => CascadeOrigin::AuthorImportant,
                        CascadeOrigin::User => CascadeOrigin::UserImportant,
                        CascadeOrigin::UserAgent => CascadeOrigin::UserAgentImportant,
                        other => other,
                    }
                } else {
                    rule.origin
                };

                entries.push(CascadeEntry {
                    property: prop.clone(),
                    value: pv.value.clone(),
                    important: pv.important,
                    origin,
                    specificity: spec,
                    source_order: rule.source_order,
                });
            }
        }
    }

    // Sort by cascade: origin, specificity, source_order
    entries.sort_by(|a, b| {
        a.origin.cmp(&b.origin)
            .then(a.specificity.cmp(&b.specificity))
            .then(a.source_order.cmp(&b.source_order))
    });

    // Apply last-wins per property (highest cascade wins)
    let mut property_map: HashMap<String, String> = HashMap::new();
    for entry in &entries {
        property_map.insert(entry.property.clone(), entry.value.clone());
    }

    // Apply inheritance
    if let Some(parent) = parent_style {
        for &prop in INHERITED_PROPERTIES {
            if !property_map.contains_key(prop) {
                if let Some(parent_val) = parent.get(prop) {
                    property_map.insert(prop.to_string(), parent_val.to_string());
                }
            }
        }
    }

    // Apply defaults for missing properties
    let needed_defaults = ["display", "position", "color", "font-size", "visibility"];
    for prop in needed_defaults {
        if !property_map.contains_key(prop) {
            property_map.insert(prop.to_string(), default_value(prop).to_string());
        }
    }

    ComputedStyle { properties: property_map }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specificity_zero() {
        let s = Specificity::zero();
        assert_eq!(s.numeric_value(), 0);
    }

    #[test]
    fn specificity_inline() {
        let s = Specificity::new(1, 0, 0, 0);
        assert!(s > Specificity::new(0, 10, 10, 10));
    }

    #[test]
    fn specificity_from_tag() {
        let s = Specificity::from_selector("div");
        assert_eq!(s, Specificity::new(0, 0, 0, 1));
    }

    #[test]
    fn specificity_from_class() {
        let s = Specificity::from_selector(".btn");
        assert_eq!(s, Specificity::new(0, 0, 1, 0));
    }

    #[test]
    fn specificity_from_id() {
        let s = Specificity::from_selector("#main");
        assert_eq!(s, Specificity::new(0, 1, 0, 0));
    }

    #[test]
    fn specificity_compound_selector() {
        let s = Specificity::from_selector("div.btn#submit");
        assert_eq!(s, Specificity::new(0, 1, 1, 1));
    }

    #[test]
    fn specificity_multiple_classes() {
        let s = Specificity::from_selector(".a.b.c");
        assert_eq!(s, Specificity::new(0, 0, 3, 0));
    }

    #[test]
    fn specificity_attribute_selector() {
        let s = Specificity::from_selector("[type=text]");
        assert_eq!(s, Specificity::new(0, 0, 1, 0));
    }

    #[test]
    fn specificity_universal_zero() {
        let s = Specificity::from_selector("*");
        assert_eq!(s.numeric_value(), 0);
    }

    #[test]
    fn specificity_descendant() {
        let s = Specificity::from_selector("div p span");
        assert_eq!(s, Specificity::new(0, 0, 0, 3));
    }

    #[test]
    fn selector_matches_tag() {
        let el = Element::new("div");
        assert!(selector_matches("div", &el));
        assert!(!selector_matches("span", &el));
    }

    #[test]
    fn selector_matches_id() {
        let el = Element::new("div").with_id("main");
        assert!(selector_matches("#main", &el));
        assert!(!selector_matches("#other", &el));
    }

    #[test]
    fn selector_matches_class() {
        let el = Element::new("div").with_class("btn").with_class("primary");
        assert!(selector_matches(".btn", &el));
        assert!(selector_matches(".primary", &el));
        assert!(!selector_matches(".secondary", &el));
    }

    #[test]
    fn selector_matches_compound() {
        let el = Element::new("div").with_id("x").with_class("y");
        assert!(selector_matches("div#x.y", &el));
        assert!(!selector_matches("span#x.y", &el));
    }

    #[test]
    fn selector_matches_universal() {
        let el = Element::new("anything");
        assert!(selector_matches("*", &el));
    }

    #[test]
    fn selector_matches_attribute() {
        let el = Element::new("input").with_attr("type", "text");
        assert!(selector_matches("[type=text]", &el));
        assert!(!selector_matches("[type=password]", &el));
    }

    #[test]
    fn cascade_last_wins() {
        let el = Element::new("div");
        let rules = vec![
            StyleRule::new("div", CascadeOrigin::Author, 0)
                .with_property("color", "red"),
            StyleRule::new("div", CascadeOrigin::Author, 1)
                .with_property("color", "blue"),
        ];
        let style = resolve_style(&el, &rules, None);
        assert_eq!(style.get("color"), Some("blue"));
    }

    #[test]
    fn cascade_specificity_wins() {
        let el = Element::new("div").with_class("special");
        let rules = vec![
            StyleRule::new("div", CascadeOrigin::Author, 0)
                .with_property("color", "red"),
            StyleRule::new(".special", CascadeOrigin::Author, 1)
                .with_property("color", "blue"),
        ];
        let style = resolve_style(&el, &rules, None);
        assert_eq!(style.get("color"), Some("blue"));
    }

    #[test]
    fn cascade_important_wins() {
        let el = Element::new("div");
        let rules = vec![
            StyleRule::new("div", CascadeOrigin::Author, 0)
                .with_important("color", "red"),
            StyleRule::new("div", CascadeOrigin::Author, 1)
                .with_property("color", "blue"),
        ];
        let style = resolve_style(&el, &rules, None);
        assert_eq!(style.get("color"), Some("red"));
    }

    #[test]
    fn inheritance_color() {
        let mut parent_style = ComputedStyle::new();
        parent_style.set("color", "green");
        parent_style.set("font-size", "20px");

        let el = Element::new("span");
        let style = resolve_style(&el, &[], Some(&parent_style));
        assert_eq!(style.get("color"), Some("green"));
        assert_eq!(style.get("font-size"), Some("20px"));
    }

    #[test]
    fn non_inherited_property_gets_default() {
        let mut parent_style = ComputedStyle::new();
        parent_style.set("display", "flex");

        let el = Element::new("div");
        let style = resolve_style(&el, &[], Some(&parent_style));
        // display is not inherited — gets default "block"
        assert_eq!(style.get("display"), Some("block"));
    }

    #[test]
    fn is_inherited_check() {
        assert!(is_inherited("color"));
        assert!(is_inherited("font-size"));
        assert!(!is_inherited("display"));
        assert!(!is_inherited("margin"));
    }

    #[test]
    fn default_values() {
        assert_eq!(default_value("display"), "block");
        assert_eq!(default_value("color"), "black");
        assert_eq!(default_value("unknown-prop"), "initial");
    }
}
