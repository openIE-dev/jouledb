//! CSS cascade and inheritance resolver: match style rules to elements,
//! sort by specificity + source order, and compute final styles with
//! property inheritance and shorthand expansion.

use std::collections::HashMap;

// ── Specificity (self-contained) ─────────────────────────────────

/// CSS specificity triple (id, class, type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CascadeSpecificity {
    pub id: u32,
    pub class: u32,
    pub tag: u32,
}

impl CascadeSpecificity {
    pub fn new(id: u32, class: u32, tag: u32) -> Self {
        Self { id, class, tag }
    }

    pub fn zero() -> Self {
        Self { id: 0, class: 0, tag: 0 }
    }
}

impl Ord for CascadeSpecificity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id
            .cmp(&other.id)
            .then(self.class.cmp(&other.class))
            .then(self.tag.cmp(&other.tag))
    }
}

impl PartialOrd for CascadeSpecificity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ── Style Rule ───────────────────────────────────────────────────

/// A CSS style rule: a selector string and a list of (property, value) declarations.
#[derive(Debug, Clone)]
pub struct StyleRule {
    pub selector_text: String,
    pub declarations: Vec<(String, String)>,
}

// ── Element representation ───────────────────────────────────────

/// A minimal element description for cascade matching.
#[derive(Debug, Clone)]
pub struct ElementInfo {
    pub tag: String,
    pub id: Option<String>,
    pub classes: Vec<String>,
    pub parent_style: Option<HashMap<String, String>>,
}

impl ElementInfo {
    pub fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            id: None,
            classes: Vec::new(),
            parent_style: None,
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

    pub fn with_parent_style(mut self, style: HashMap<String, String>) -> Self {
        self.parent_style = Some(style);
        self
    }
}

// ── Matched Rule ─────────────────────────────────────────────────

/// A rule that matched an element, annotated with specificity and source order.
#[derive(Debug, Clone)]
pub struct MatchedRule {
    pub rule: StyleRule,
    pub specificity: CascadeSpecificity,
    pub source_order: usize,
}

// ── Selector Parsing & Matching ──────────────────────────────────

/// A single simple selector component.
#[derive(Debug, Clone, PartialEq)]
enum SimpleSelector {
    Tag(String),
    Class(String),
    Id(String),
}

fn parse_selector(text: &str) -> Vec<SimpleSelector> {
    let mut parts = Vec::new();
    let trimmed = text.trim();
    let mut chars = trimmed.chars().peekable();

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
                    parts.push(SimpleSelector::Id(name));
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
                    parts.push(SimpleSelector::Class(name));
                }
            }
            '*' => {
                chars.next();
            }
            c if c.is_alphanumeric() || c == '-' || c == '_' => {
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
                    parts.push(SimpleSelector::Tag(name));
                }
            }
            _ => {
                chars.next();
            }
        }
    }

    parts
}

fn compute_specificity(parts: &[SimpleSelector]) -> CascadeSpecificity {
    let mut spec = CascadeSpecificity::zero();
    for part in parts {
        match part {
            SimpleSelector::Id(_) => spec.id += 1,
            SimpleSelector::Class(_) => spec.class += 1,
            SimpleSelector::Tag(_) => spec.tag += 1,
        }
    }
    spec
}

fn matches_element(parts: &[SimpleSelector], element: &ElementInfo) -> bool {
    for part in parts {
        match part {
            SimpleSelector::Tag(t) => {
                if *t != element.tag {
                    return false;
                }
            }
            SimpleSelector::Class(c) => {
                if !element.classes.contains(c) {
                    return false;
                }
            }
            SimpleSelector::Id(id) => {
                if element.id.as_deref() != Some(id.as_str()) {
                    return false;
                }
            }
        }
    }
    true
}

// ── Inheritable Properties ───────────────────────────────────────

const INHERITABLE: &[&str] = &[
    "color",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
    "font-variant",
    "letter-spacing",
    "line-height",
    "text-align",
    "text-indent",
    "text-transform",
    "visibility",
    "white-space",
    "word-spacing",
    "cursor",
    "direction",
    "list-style",
    "list-style-type",
    "list-style-position",
    "quotes",
];

fn is_inheritable(property: &str) -> bool {
    INHERITABLE.contains(&property)
}

// ── Initial Values ───────────────────────────────────────────────

fn initial_value(property: &str) -> &str {
    match property {
        "color" => "black",
        "font-family" => "serif",
        "font-size" => "16px",
        "font-style" => "normal",
        "font-weight" => "normal",
        "line-height" => "normal",
        "text-align" => "start",
        "visibility" => "visible",
        "display" => "inline",
        "position" => "static",
        "margin" | "margin-top" | "margin-right" | "margin-bottom" | "margin-left" => "0",
        "padding" | "padding-top" | "padding-right" | "padding-bottom" | "padding-left" => "0",
        "border-width" | "border-top-width" | "border-right-width"
        | "border-bottom-width" | "border-left-width" => "medium",
        "background-color" => "transparent",
        "opacity" => "1",
        "width" | "height" => "auto",
        _ => "",
    }
}

// ── Shorthand Expansion ──────────────────────────────────────────

fn expand_shorthand(property: &str, value: &str) -> Vec<(String, String)> {
    match property {
        "margin" => expand_four_sides(value, "margin"),
        "padding" => expand_four_sides(value, "padding"),
        "border-width" => expand_four_sides(value, "border"),
        "border-color" => expand_four_sides_suffix(value, "border", "color"),
        "border-style" => expand_four_sides_suffix(value, "border", "style"),
        _ => vec![(property.to_string(), value.to_string())],
    }
}

fn expand_four_sides(value: &str, prefix: &str) -> Vec<(String, String)> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let (top, right, bottom, left) = match parts.len() {
        1 => (parts[0], parts[0], parts[0], parts[0]),
        2 => (parts[0], parts[1], parts[0], parts[1]),
        3 => (parts[0], parts[1], parts[2], parts[1]),
        4 => (parts[0], parts[1], parts[2], parts[3]),
        _ => (value, value, value, value),
    };
    vec![
        (format!("{prefix}-top"), top.to_string()),
        (format!("{prefix}-right"), right.to_string()),
        (format!("{prefix}-bottom"), bottom.to_string()),
        (format!("{prefix}-left"), left.to_string()),
    ]
}

fn expand_four_sides_suffix(value: &str, prefix: &str, suffix: &str) -> Vec<(String, String)> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let (top, right, bottom, left) = match parts.len() {
        1 => (parts[0], parts[0], parts[0], parts[0]),
        2 => (parts[0], parts[1], parts[0], parts[1]),
        3 => (parts[0], parts[1], parts[2], parts[1]),
        4 => (parts[0], parts[1], parts[2], parts[3]),
        _ => (value, value, value, value),
    };
    vec![
        (format!("{prefix}-top-{suffix}"), top.to_string()),
        (format!("{prefix}-right-{suffix}"), right.to_string()),
        (format!("{prefix}-bottom-{suffix}"), bottom.to_string()),
        (format!("{prefix}-left-{suffix}"), left.to_string()),
    ]
}

// ── Cascade Resolver ─────────────────────────────────────────────

/// The main cascade resolver. Holds a stylesheet and resolves styles for elements.
#[derive(Debug, Clone)]
pub struct CascadeResolver {
    rules: Vec<StyleRule>,
}

impl CascadeResolver {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: StyleRule) {
        self.rules.push(rule);
    }

    /// Match all rules against the given element.
    pub fn match_rules(&self, element: &ElementInfo) -> Vec<MatchedRule> {
        let mut matched = Vec::new();
        for (i, rule) in self.rules.iter().enumerate() {
            let parts = parse_selector(&rule.selector_text);
            if parts.is_empty() || matches_element(&parts, element) {
                let spec = if parts.is_empty() {
                    CascadeSpecificity::zero()
                } else {
                    compute_specificity(&parts)
                };
                matched.push(MatchedRule {
                    rule: rule.clone(),
                    specificity: spec,
                    source_order: i,
                });
            }
        }
        // Sort by specificity then source order (stable)
        matched.sort_by(|a, b| a.specificity.cmp(&b.specificity).then(a.source_order.cmp(&b.source_order)));
        matched
    }

    /// Compute the final cascaded style for an element.
    pub fn compute_style(&self, element: &ElementInfo) -> HashMap<String, String> {
        let matched = self.match_rules(element);
        let mut style: HashMap<String, String> = HashMap::new();

        // Apply declarations in cascade order (lower specificity first, later wins)
        for m in &matched {
            for (prop, val) in &m.rule.declarations {
                let expanded = expand_shorthand(prop, val);
                for (ep, ev) in expanded {
                    style.insert(ep, ev);
                }
            }
        }

        // Inheritance: for inheritable properties not set, take from parent
        if let Some(parent) = &element.parent_style {
            for prop in INHERITABLE {
                let key = prop.to_string();
                if !style.contains_key(&key) {
                    if let Some(pv) = parent.get(&key) {
                        style.insert(key, pv.clone());
                    }
                }
            }
        }

        style
    }

    /// Resolve a specific property, returning the cascaded value, inherited value,
    /// or initial value.
    pub fn resolve_property(&self, element: &ElementInfo, property: &str) -> String {
        let style = self.compute_style(element);
        if let Some(v) = style.get(property) {
            return v.clone();
        }
        // Inherit from parent if inheritable
        if is_inheritable(property) {
            if let Some(parent) = &element.parent_style {
                if let Some(v) = parent.get(property) {
                    return v.clone();
                }
            }
        }
        initial_value(property).to_string()
    }
}

impl Default for CascadeResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(selector: &str, decls: &[(&str, &str)]) -> StyleRule {
        StyleRule {
            selector_text: selector.to_string(),
            declarations: decls.iter().map(|(p, v)| (p.to_string(), v.to_string())).collect(),
        }
    }

    #[test]
    fn test_tag_matching() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule("div", &[("color", "red")]));
        resolver.add_rule(rule("span", &[("color", "blue")]));

        let el = ElementInfo::new("div");
        let matched = resolver.match_rules(&el);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].rule.selector_text, "div");
    }

    #[test]
    fn test_class_matching() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule(".active", &[("color", "green")]));

        let el = ElementInfo::new("div").with_class("active");
        let matched = resolver.match_rules(&el);
        assert_eq!(matched.len(), 1);
    }

    #[test]
    fn test_id_matching() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule("#main", &[("width", "100px")]));

        let el = ElementInfo::new("div").with_id("main");
        let matched = resolver.match_rules(&el);
        assert_eq!(matched.len(), 1);
    }

    #[test]
    fn test_compound_selector() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule("div.active", &[("color", "red")]));

        let el1 = ElementInfo::new("div").with_class("active");
        assert_eq!(resolver.match_rules(&el1).len(), 1);

        let el2 = ElementInfo::new("span").with_class("active");
        assert_eq!(resolver.match_rules(&el2).len(), 0);
    }

    #[test]
    fn test_specificity_ordering() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule("div", &[("color", "red")]));         // (0,0,1)
        resolver.add_rule(rule(".cls", &[("color", "green")]));      // (0,1,0)
        resolver.add_rule(rule("#id", &[("color", "blue")]));        // (1,0,0)

        let el = ElementInfo::new("div").with_class("cls").with_id("id");
        let matched = resolver.match_rules(&el);
        assert_eq!(matched.len(), 3);
        // Sorted ascending: tag < class < id
        assert_eq!(matched[0].specificity, CascadeSpecificity::new(0, 0, 1));
        assert_eq!(matched[1].specificity, CascadeSpecificity::new(0, 1, 0));
        assert_eq!(matched[2].specificity, CascadeSpecificity::new(1, 0, 0));
    }

    #[test]
    fn test_cascade_last_wins() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule("div", &[("color", "red")]));
        resolver.add_rule(rule("div", &[("color", "blue")]));

        let el = ElementInfo::new("div");
        let style = resolver.compute_style(&el);
        // Both have same specificity; source order later wins
        assert_eq!(style.get("color").unwrap(), "blue");
    }

    #[test]
    fn test_higher_specificity_wins() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule(".cls", &[("color", "green")]));
        resolver.add_rule(rule("div", &[("color", "red")]));

        let el = ElementInfo::new("div").with_class("cls");
        let style = resolver.compute_style(&el);
        assert_eq!(style.get("color").unwrap(), "green");
    }

    #[test]
    fn test_shorthand_expansion_margin() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule("div", &[("margin", "10px 20px")]));

        let el = ElementInfo::new("div");
        let style = resolver.compute_style(&el);
        assert_eq!(style.get("margin-top").unwrap(), "10px");
        assert_eq!(style.get("margin-right").unwrap(), "20px");
        assert_eq!(style.get("margin-bottom").unwrap(), "10px");
        assert_eq!(style.get("margin-left").unwrap(), "20px");
    }

    #[test]
    fn test_shorthand_expansion_four_values() {
        let expanded = expand_shorthand("padding", "1px 2px 3px 4px");
        assert_eq!(expanded.len(), 4);
        assert_eq!(expanded[0], ("padding-top".to_string(), "1px".to_string()));
        assert_eq!(expanded[1], ("padding-right".to_string(), "2px".to_string()));
        assert_eq!(expanded[2], ("padding-bottom".to_string(), "3px".to_string()));
        assert_eq!(expanded[3], ("padding-left".to_string(), "4px".to_string()));
    }

    #[test]
    fn test_inheritance_from_parent() {
        let mut parent_style = HashMap::new();
        parent_style.insert("color".to_string(), "blue".to_string());
        parent_style.insert("display".to_string(), "flex".to_string());

        let resolver = CascadeResolver::new();
        let el = ElementInfo::new("span").with_parent_style(parent_style);
        let style = resolver.compute_style(&el);
        // color is inheritable
        assert_eq!(style.get("color").unwrap(), "blue");
        // display is NOT inheritable
        assert!(!style.contains_key("display"));
    }

    #[test]
    fn test_resolve_property_initial() {
        let resolver = CascadeResolver::new();
        let el = ElementInfo::new("div");
        assert_eq!(resolver.resolve_property(&el, "color"), "black");
        assert_eq!(resolver.resolve_property(&el, "display"), "inline");
    }

    #[test]
    fn test_resolve_property_inherited() {
        let mut parent_style = HashMap::new();
        parent_style.insert("font-size".to_string(), "20px".to_string());

        let resolver = CascadeResolver::new();
        let el = ElementInfo::new("span").with_parent_style(parent_style);
        assert_eq!(resolver.resolve_property(&el, "font-size"), "20px");
    }

    #[test]
    fn test_no_match_wrong_tag() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule("p", &[("color", "red")]));
        let el = ElementInfo::new("div");
        assert!(resolver.match_rules(&el).is_empty());
    }

    #[test]
    fn test_multiple_classes() {
        let mut resolver = CascadeResolver::new();
        resolver.add_rule(rule(".a.b", &[("color", "red")]));

        let el_both = ElementInfo::new("div").with_class("a").with_class("b");
        assert_eq!(resolver.match_rules(&el_both).len(), 1);

        let el_one = ElementInfo::new("div").with_class("a");
        assert_eq!(resolver.match_rules(&el_one).len(), 0);
    }

    #[test]
    fn test_shorthand_single_value() {
        let expanded = expand_shorthand("margin", "5px");
        assert_eq!(expanded.len(), 4);
        for (_, v) in &expanded {
            assert_eq!(v, "5px");
        }
    }
}
