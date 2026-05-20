//! Accessibility — ARIA role validation, required properties, focus management,
//! WCAG color contrast checking (AA/AAA), landmark regions, skip-link generation.
//!
//! Pure Rust a11y toolkit. No browser APIs or DOM dependencies.

use std::collections::HashMap;
use std::fmt;

// ── ARIA Roles ───────────────────────────────────────────────────

/// WAI-ARIA role categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleCategory {
    Landmark,
    Widget,
    Document,
    LiveRegion,
    Window,
    Abstract,
}

/// An ARIA role with its category and required properties.
#[derive(Debug, Clone)]
pub struct AriaRole {
    pub name: &'static str,
    pub category: RoleCategory,
    pub required_props: &'static [&'static str],
    pub supported_props: &'static [&'static str],
}

/// Registry of known ARIA roles.
pub fn known_roles() -> Vec<AriaRole> {
    vec![
        AriaRole { name: "alert", category: RoleCategory::LiveRegion, required_props: &[], supported_props: &["aria-atomic", "aria-live"] },
        AriaRole { name: "alertdialog", category: RoleCategory::Window, required_props: &["aria-label"], supported_props: &["aria-describedby"] },
        AriaRole { name: "banner", category: RoleCategory::Landmark, required_props: &[], supported_props: &["aria-label"] },
        AriaRole { name: "button", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-pressed", "aria-expanded", "aria-disabled"] },
        AriaRole { name: "checkbox", category: RoleCategory::Widget, required_props: &["aria-checked"], supported_props: &["aria-disabled", "aria-readonly"] },
        AriaRole { name: "complementary", category: RoleCategory::Landmark, required_props: &[], supported_props: &["aria-label"] },
        AriaRole { name: "contentinfo", category: RoleCategory::Landmark, required_props: &[], supported_props: &["aria-label"] },
        AriaRole { name: "dialog", category: RoleCategory::Window, required_props: &["aria-label"], supported_props: &["aria-describedby", "aria-modal"] },
        AriaRole { name: "form", category: RoleCategory::Landmark, required_props: &[], supported_props: &["aria-label"] },
        AriaRole { name: "heading", category: RoleCategory::Document, required_props: &["aria-level"], supported_props: &[] },
        AriaRole { name: "img", category: RoleCategory::Document, required_props: &["aria-label"], supported_props: &[] },
        AriaRole { name: "link", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-disabled", "aria-expanded"] },
        AriaRole { name: "list", category: RoleCategory::Document, required_props: &[], supported_props: &["aria-label"] },
        AriaRole { name: "listbox", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-multiselectable", "aria-required", "aria-activedescendant"] },
        AriaRole { name: "listitem", category: RoleCategory::Document, required_props: &[], supported_props: &["aria-level"] },
        AriaRole { name: "log", category: RoleCategory::LiveRegion, required_props: &[], supported_props: &["aria-live", "aria-atomic"] },
        AriaRole { name: "main", category: RoleCategory::Landmark, required_props: &[], supported_props: &["aria-label"] },
        AriaRole { name: "menu", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-activedescendant", "aria-orientation"] },
        AriaRole { name: "menuitem", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-disabled"] },
        AriaRole { name: "navigation", category: RoleCategory::Landmark, required_props: &[], supported_props: &["aria-label"] },
        AriaRole { name: "option", category: RoleCategory::Widget, required_props: &["aria-selected"], supported_props: &["aria-disabled"] },
        AriaRole { name: "progressbar", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-valuenow", "aria-valuemin", "aria-valuemax"] },
        AriaRole { name: "radio", category: RoleCategory::Widget, required_props: &["aria-checked"], supported_props: &["aria-disabled"] },
        AriaRole { name: "radiogroup", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-required", "aria-orientation"] },
        AriaRole { name: "region", category: RoleCategory::Landmark, required_props: &["aria-label"], supported_props: &[] },
        AriaRole { name: "search", category: RoleCategory::Landmark, required_props: &[], supported_props: &["aria-label"] },
        AriaRole { name: "slider", category: RoleCategory::Widget, required_props: &["aria-valuenow", "aria-valuemin", "aria-valuemax"], supported_props: &["aria-orientation", "aria-disabled"] },
        AriaRole { name: "spinbutton", category: RoleCategory::Widget, required_props: &["aria-valuenow", "aria-valuemin", "aria-valuemax"], supported_props: &["aria-readonly", "aria-required"] },
        AriaRole { name: "status", category: RoleCategory::LiveRegion, required_props: &[], supported_props: &["aria-live", "aria-atomic"] },
        AriaRole { name: "switch", category: RoleCategory::Widget, required_props: &["aria-checked"], supported_props: &["aria-disabled", "aria-readonly"] },
        AriaRole { name: "tab", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-selected", "aria-disabled"] },
        AriaRole { name: "tablist", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-orientation", "aria-multiselectable"] },
        AriaRole { name: "tabpanel", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-labelledby"] },
        AriaRole { name: "textbox", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-placeholder", "aria-readonly", "aria-required", "aria-multiline"] },
        AriaRole { name: "timer", category: RoleCategory::LiveRegion, required_props: &[], supported_props: &["aria-live"] },
        AriaRole { name: "tooltip", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-describedby"] },
        AriaRole { name: "tree", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-multiselectable", "aria-activedescendant"] },
        AriaRole { name: "treeitem", category: RoleCategory::Widget, required_props: &[], supported_props: &["aria-expanded", "aria-selected", "aria-level"] },
    ]
}

/// Look up an ARIA role by name.
pub fn find_role(name: &str) -> Option<AriaRole> {
    known_roles().into_iter().find(|r| r.name == name)
}

/// Validate that required ARIA properties are present.
pub fn validate_aria(role_name: &str, provided_props: &[&str]) -> Vec<String> {
    let mut errors = Vec::new();
    match find_role(role_name) {
        Some(role) => {
            for req in role.required_props {
                if !provided_props.contains(req) {
                    errors.push(format!("role '{}' requires property '{}'", role_name, req));
                }
            }
        }
        None => {
            errors.push(format!("unknown ARIA role: '{}'", role_name));
        }
    }
    errors
}

// ── Focus Management ─────────────────────────────────────────────

/// Focus trap: ordered list of focusable element IDs.
#[derive(Debug, Clone)]
pub struct FocusTrap {
    elements: Vec<String>,
    current: usize,
    wrap: bool,
}

impl FocusTrap {
    pub fn new(elements: Vec<String>) -> Self {
        Self { elements, current: 0, wrap: true }
    }

    pub fn no_wrap(mut self) -> Self { self.wrap = false; self }

    /// Move focus to next element. Returns the element ID.
    pub fn next(&mut self) -> Option<&str> {
        if self.elements.is_empty() { return None; }
        if self.current + 1 < self.elements.len() {
            self.current += 1;
        } else if self.wrap {
            self.current = 0;
        }
        Some(&self.elements[self.current])
    }

    /// Move focus to previous element.
    pub fn prev(&mut self) -> Option<&str> {
        if self.elements.is_empty() { return None; }
        if self.current > 0 {
            self.current -= 1;
        } else if self.wrap {
            self.current = self.elements.len() - 1;
        }
        Some(&self.elements[self.current])
    }

    /// Get currently focused element.
    pub fn current(&self) -> Option<&str> {
        self.elements.get(self.current).map(|s| s.as_str())
    }

    /// Reset to first element.
    pub fn reset(&mut self) {
        self.current = 0;
    }

    /// Focus a specific element by ID. Returns true if found.
    pub fn focus(&mut self, id: &str) -> bool {
        if let Some(idx) = self.elements.iter().position(|e| e == id) {
            self.current = idx;
            true
        } else {
            false
        }
    }

    /// Number of focusable elements.
    pub fn len(&self) -> usize { self.elements.len() }

    /// Whether the trap is empty.
    pub fn is_empty(&self) -> bool { self.elements.is_empty() }
}

// ── Color Contrast (WCAG 2.1) ───────────────────────────────────

/// sRGB color for contrast calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SrgbColor {
    pub r: f64,  // 0..1
    pub g: f64,
    pub b: f64,
}

impl SrgbColor {
    /// Create from 0-1 range floats.
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Create from 0-255 byte values.
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self::new(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0)
    }

    /// Parse hex color string (#RGB, #RRGGBB).
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?;
                let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?;
                let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?;
                Some(Self::from_u8(r, g, b))
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(Self::from_u8(r, g, b))
            }
            _ => None,
        }
    }

    /// Relative luminance per WCAG 2.1.
    pub fn relative_luminance(&self) -> f64 {
        let linearize = |c: f64| -> f64 {
            if c <= 0.03928 { c / 12.92 }
            else { ((c + 0.055) / 1.055).powf(2.4) }
        };
        0.2126 * linearize(self.r) + 0.7152 * linearize(self.g) + 0.0722 * linearize(self.b)
    }
}

impl fmt::Display for SrgbColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "#{:02x}{:02x}{:02x}",
            (self.r * 255.0).round() as u8,
            (self.g * 255.0).round() as u8,
            (self.b * 255.0).round() as u8,
        )
    }
}

/// WCAG conformance levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WcagLevel {
    Fail,
    AA,
    AAA,
}

impl fmt::Display for WcagLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fail => write!(f, "Fail"),
            Self::AA => write!(f, "AA"),
            Self::AAA => write!(f, "AAA"),
        }
    }
}

/// Compute contrast ratio between two colors (WCAG 2.1 algorithm).
/// Returns a value between 1:1 and 21:1.
pub fn contrast_ratio(c1: &SrgbColor, c2: &SrgbColor) -> f64 {
    let l1 = c1.relative_luminance();
    let l2 = c2.relative_luminance();
    let lighter = l1.max(l2);
    let darker = l1.min(l2);
    (lighter + 0.05) / (darker + 0.05)
}

/// Check WCAG conformance for normal text (>= 4.5:1 for AA, >= 7:1 for AAA).
pub fn check_contrast_normal(fg: &SrgbColor, bg: &SrgbColor) -> WcagLevel {
    let ratio = contrast_ratio(fg, bg);
    if ratio >= 7.0 { WcagLevel::AAA }
    else if ratio >= 4.5 { WcagLevel::AA }
    else { WcagLevel::Fail }
}

/// Check WCAG conformance for large text (>= 3:1 for AA, >= 4.5:1 for AAA).
pub fn check_contrast_large(fg: &SrgbColor, bg: &SrgbColor) -> WcagLevel {
    let ratio = contrast_ratio(fg, bg);
    if ratio >= 4.5 { WcagLevel::AAA }
    else if ratio >= 3.0 { WcagLevel::AA }
    else { WcagLevel::Fail }
}

/// Suggest a foreground color (black or white) for best contrast against a background.
pub fn suggest_foreground(bg: &SrgbColor) -> SrgbColor {
    let white = SrgbColor::new(1.0, 1.0, 1.0);
    let black = SrgbColor::new(0.0, 0.0, 0.0);
    if contrast_ratio(bg, &white) > contrast_ratio(bg, &black) {
        white
    } else {
        black
    }
}

// ── Landmark Regions ─────────────────────────────────────────────

/// Landmark region configuration.
#[derive(Debug, Clone)]
pub struct LandmarkRegion {
    pub role: String,
    pub label: Option<String>,
    pub element: String,
}

impl LandmarkRegion {
    pub fn new(role: impl Into<String>, element: impl Into<String>) -> Self {
        Self { role: role.into(), label: None, element: element.into() }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Generate HTML opening tag with ARIA attributes.
    pub fn open_tag(&self) -> String {
        let mut attrs = format!(r#"role="{}""#, self.role);
        if let Some(ref label) = self.label {
            attrs.push_str(&format!(r#" aria-label="{}""#, label));
        }
        format!("<{} {}>", self.element, attrs)
    }

    /// Generate HTML closing tag.
    pub fn close_tag(&self) -> String {
        format!("</{}>", self.element)
    }
}

/// Standard page landmark structure.
pub fn standard_landmarks() -> Vec<LandmarkRegion> {
    vec![
        LandmarkRegion::new("banner", "header"),
        LandmarkRegion::new("navigation", "nav").with_label("Main navigation"),
        LandmarkRegion::new("main", "main"),
        LandmarkRegion::new("complementary", "aside"),
        LandmarkRegion::new("contentinfo", "footer"),
    ]
}

// ── Skip Link ────────────────────────────────────────────────────

/// Generate a skip-to-content link.
pub fn skip_link(target_id: &str, text: &str) -> String {
    format!(
        r##"<a href="#{target_id}" class="skip-link">{text}</a>"##
    )
}

/// Generate CSS for visually-hidden skip link that appears on focus.
pub fn skip_link_css(class: &str) -> String {
    format!(
        r#".{class} {{
  position: absolute;
  top: -40px;
  left: 0;
  background: #000;
  color: #fff;
  padding: 8px 16px;
  z-index: 100;
  transition: top 0.2s;
}}
.{class}:focus {{
  top: 0;
}}"#
    )
}

// ── A11y Audit ───────────────────────────────────────────────────

/// Issue severity for accessibility audits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// A single accessibility audit finding.
#[derive(Debug, Clone)]
pub struct AuditIssue {
    pub severity: Severity,
    pub rule: String,
    pub message: String,
    pub element: Option<String>,
}

impl fmt::Display for AuditIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.rule, self.message)?;
        if let Some(ref el) = self.element {
            write!(f, " ({})", el)?;
        }
        Ok(())
    }
}

/// Audit a set of element descriptions for common a11y issues.
pub fn audit_elements(elements: &[ElementInfo]) -> Vec<AuditIssue> {
    let mut issues = Vec::new();

    for el in elements {
        // Images without alt text
        if el.tag == "img" && el.attrs.get("alt").is_none() {
            issues.push(AuditIssue {
                severity: Severity::Error,
                rule: "img-alt".into(),
                message: "Image missing alt attribute".into(),
                element: el.id.clone(),
            });
        }

        // Interactive elements without accessible name
        let interactive = ["button", "a", "input", "select", "textarea"];
        if interactive.contains(&el.tag.as_str()) {
            let has_name = el.text_content.is_some()
                || el.attrs.contains_key("aria-label")
                || el.attrs.contains_key("aria-labelledby")
                || el.attrs.contains_key("title");

            if !has_name {
                issues.push(AuditIssue {
                    severity: Severity::Error,
                    rule: "accessible-name".into(),
                    message: format!("<{}> missing accessible name", el.tag),
                    element: el.id.clone(),
                });
            }
        }

        // Form inputs without labels
        if (el.tag == "input" || el.tag == "select" || el.tag == "textarea")
            && el.attrs.get("type").map(|t| t.as_str()) != Some("hidden")
        {
            let has_label = el.attrs.contains_key("aria-label")
                || el.attrs.contains_key("aria-labelledby")
                || el.has_associated_label;

            if !has_label {
                issues.push(AuditIssue {
                    severity: Severity::Warning,
                    rule: "label".into(),
                    message: format!("<{}> should have an associated label", el.tag),
                    element: el.id.clone(),
                });
            }
        }

        // ARIA role validation
        if let Some(role) = el.attrs.get("role") {
            let provided: Vec<&str> = el.attrs.keys().map(|k| k.as_str()).collect();
            let errs = validate_aria(role, &provided);
            for err in errs {
                issues.push(AuditIssue {
                    severity: Severity::Error,
                    rule: "aria-required-props".into(),
                    message: err,
                    element: el.id.clone(),
                });
            }
        }

        // Tabindex > 0
        if let Some(ti) = el.attrs.get("tabindex") {
            if let Ok(n) = ti.parse::<i32>() {
                if n > 0 {
                    issues.push(AuditIssue {
                        severity: Severity::Warning,
                        rule: "tabindex".into(),
                        message: format!("tabindex={} — positive tabindex disrupts natural tab order", n),
                        element: el.id.clone(),
                    });
                }
            }
        }
    }

    issues
}

/// Simplified element info for auditing.
#[derive(Debug, Clone)]
pub struct ElementInfo {
    pub tag: String,
    pub id: Option<String>,
    pub attrs: HashMap<String, String>,
    pub text_content: Option<String>,
    pub has_associated_label: bool,
}

impl ElementInfo {
    pub fn new(tag: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            id: None,
            attrs: HashMap::new(),
            text_content: None,
            has_associated_label: false,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self { self.id = Some(id.into()); self }
    pub fn with_attr(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.attrs.insert(key.into(), val.into());
        self
    }
    pub fn with_text(mut self, text: impl Into<String>) -> Self { self.text_content = Some(text.into()); self }
    pub fn with_label(mut self) -> Self { self.has_associated_label = true; self }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_role() {
        let btn = find_role("button").unwrap();
        assert_eq!(btn.category, RoleCategory::Widget);
        assert!(btn.required_props.is_empty());

        let cb = find_role("checkbox").unwrap();
        assert!(cb.required_props.contains(&"aria-checked"));
    }

    #[test]
    fn test_find_role_unknown() {
        assert!(find_role("nonexistent").is_none());
    }

    #[test]
    fn test_validate_aria_ok() {
        let errors = validate_aria("button", &["aria-pressed"]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_aria_missing_required() {
        let errors = validate_aria("checkbox", &[]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("aria-checked"));
    }

    #[test]
    fn test_validate_aria_unknown_role() {
        let errors = validate_aria("foobar", &[]);
        assert!(errors[0].contains("unknown"));
    }

    #[test]
    fn test_validate_slider() {
        let errors = validate_aria("slider", &["aria-valuenow"]);
        assert_eq!(errors.len(), 2); // missing valuemin + valuemax
    }

    #[test]
    fn test_focus_trap_navigation() {
        let mut trap = FocusTrap::new(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(trap.current(), Some("a"));

        assert_eq!(trap.next(), Some("b"));
        assert_eq!(trap.next(), Some("c"));
        assert_eq!(trap.next(), Some("a")); // wraps

        assert_eq!(trap.prev(), Some("c")); // wraps back
    }

    #[test]
    fn test_focus_trap_no_wrap() {
        let mut trap = FocusTrap::new(vec!["x".into(), "y".into()]).no_wrap();
        assert_eq!(trap.next(), Some("y"));
        assert_eq!(trap.next(), Some("y")); // stays at end
    }

    #[test]
    fn test_focus_trap_prev_wrap() {
        let mut trap = FocusTrap::new(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(trap.prev(), Some("c")); // wraps from 0 to last
    }

    #[test]
    fn test_focus_trap_focus_by_id() {
        let mut trap = FocusTrap::new(vec!["a".into(), "b".into(), "c".into()]);
        assert!(trap.focus("b"));
        assert_eq!(trap.current(), Some("b"));
        assert!(!trap.focus("z"));
    }

    #[test]
    fn test_focus_trap_empty() {
        let mut trap = FocusTrap::new(vec![]);
        assert!(trap.is_empty());
        assert!(trap.next().is_none());
        assert!(trap.prev().is_none());
    }

    #[test]
    fn test_focus_trap_reset() {
        let mut trap = FocusTrap::new(vec!["a".into(), "b".into()]);
        trap.next();
        trap.reset();
        assert_eq!(trap.current(), Some("a"));
    }

    #[test]
    fn test_color_from_hex_6() {
        let c = SrgbColor::from_hex("#ff0000").unwrap();
        assert!((c.r - 1.0).abs() < 0.01);
        assert!(c.g.abs() < 0.01);
        assert!(c.b.abs() < 0.01);
    }

    #[test]
    fn test_color_from_hex_3() {
        let c = SrgbColor::from_hex("#fff").unwrap();
        assert!((c.r - 1.0).abs() < 0.01);
        assert!((c.g - 1.0).abs() < 0.01);
        assert!((c.b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_color_from_hex_invalid() {
        assert!(SrgbColor::from_hex("#gg").is_none());
        assert!(SrgbColor::from_hex("#12345").is_none());
    }

    #[test]
    fn test_color_display() {
        let c = SrgbColor::from_u8(255, 128, 0);
        assert_eq!(c.to_string(), "#ff8000");
    }

    #[test]
    fn test_luminance_black() {
        let black = SrgbColor::new(0.0, 0.0, 0.0);
        assert!(black.relative_luminance().abs() < 1e-10);
    }

    #[test]
    fn test_luminance_white() {
        let white = SrgbColor::new(1.0, 1.0, 1.0);
        assert!((white.relative_luminance() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_contrast_ratio_bw() {
        let black = SrgbColor::new(0.0, 0.0, 0.0);
        let white = SrgbColor::new(1.0, 1.0, 1.0);
        let ratio = contrast_ratio(&black, &white);
        assert!((ratio - 21.0).abs() < 0.1);
    }

    #[test]
    fn test_contrast_ratio_same() {
        let c = SrgbColor::from_hex("#808080").unwrap();
        let ratio = contrast_ratio(&c, &c);
        assert!((ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_check_contrast_normal_pass() {
        let black = SrgbColor::new(0.0, 0.0, 0.0);
        let white = SrgbColor::new(1.0, 1.0, 1.0);
        assert_eq!(check_contrast_normal(&black, &white), WcagLevel::AAA);
    }

    #[test]
    fn test_check_contrast_normal_fail() {
        let light1 = SrgbColor::from_hex("#cccccc").unwrap();
        let light2 = SrgbColor::from_hex("#ffffff").unwrap();
        assert_eq!(check_contrast_normal(&light1, &light2), WcagLevel::Fail);
    }

    #[test]
    fn test_check_contrast_large() {
        let fg = SrgbColor::from_hex("#767676").unwrap();
        let bg = SrgbColor::new(1.0, 1.0, 1.0);
        // #767676 on white is approximately 4.54:1
        assert!(check_contrast_large(&fg, &bg) >= WcagLevel::AA);
    }

    #[test]
    fn test_suggest_foreground() {
        let dark_bg = SrgbColor::from_hex("#1a1a2e").unwrap();
        let light_bg = SrgbColor::from_hex("#f0f0f0").unwrap();

        let fg_dark = suggest_foreground(&dark_bg);
        assert!((fg_dark.r - 1.0).abs() < 0.01); // white

        let fg_light = suggest_foreground(&light_bg);
        assert!(fg_light.r.abs() < 0.01); // black
    }

    #[test]
    fn test_landmark_region() {
        let nav = LandmarkRegion::new("navigation", "nav")
            .with_label("Main");
        assert_eq!(nav.open_tag(), r#"<nav role="navigation" aria-label="Main">"#);
        assert_eq!(nav.close_tag(), "</nav>");
    }

    #[test]
    fn test_standard_landmarks() {
        let landmarks = standard_landmarks();
        assert_eq!(landmarks.len(), 5);
        assert!(landmarks.iter().any(|l| l.role == "main"));
        assert!(landmarks.iter().any(|l| l.role == "banner"));
    }

    #[test]
    fn test_skip_link() {
        let link = skip_link("main-content", "Skip to main content");
        assert!(link.contains(r##"href="#main-content""##));
        assert!(link.contains("Skip to main content"));
    }

    #[test]
    fn test_skip_link_css() {
        let css = skip_link_css("skip-link");
        assert!(css.contains(".skip-link"));
        assert!(css.contains("position: absolute"));
        assert!(css.contains(":focus"));
    }

    #[test]
    fn test_audit_img_no_alt() {
        let elements = vec![ElementInfo::new("img")];
        let issues = audit_elements(&elements);
        assert!(issues.iter().any(|i| i.rule == "img-alt"));
    }

    #[test]
    fn test_audit_img_with_alt() {
        let elements = vec![
            ElementInfo::new("img").with_attr("alt", "A photo"),
        ];
        let issues = audit_elements(&elements);
        assert!(!issues.iter().any(|i| i.rule == "img-alt"));
    }

    #[test]
    fn test_audit_button_no_name() {
        let elements = vec![ElementInfo::new("button")];
        let issues = audit_elements(&elements);
        assert!(issues.iter().any(|i| i.rule == "accessible-name"));
    }

    #[test]
    fn test_audit_button_with_text() {
        let elements = vec![
            ElementInfo::new("button").with_text("Submit"),
        ];
        let issues = audit_elements(&elements);
        assert!(!issues.iter().any(|i| i.rule == "accessible-name"));
    }

    #[test]
    fn test_audit_input_no_label() {
        let elements = vec![
            ElementInfo::new("input").with_attr("type", "text"),
        ];
        let issues = audit_elements(&elements);
        assert!(issues.iter().any(|i| i.rule == "label"));
    }

    #[test]
    fn test_audit_input_hidden_ok() {
        let elements = vec![
            ElementInfo::new("input").with_attr("type", "hidden"),
        ];
        let issues = audit_elements(&elements);
        assert!(!issues.iter().any(|i| i.rule == "label"));
    }

    #[test]
    fn test_audit_positive_tabindex() {
        let elements = vec![
            ElementInfo::new("div").with_attr("tabindex", "5"),
        ];
        let issues = audit_elements(&elements);
        assert!(issues.iter().any(|i| i.rule == "tabindex"));
    }

    #[test]
    fn test_audit_aria_role_validation() {
        let elements = vec![
            ElementInfo::new("div").with_attr("role", "checkbox"),
        ];
        let issues = audit_elements(&elements);
        assert!(issues.iter().any(|i| i.rule == "aria-required-props"));
    }

    #[test]
    fn test_wcag_level_ordering() {
        assert!(WcagLevel::AAA > WcagLevel::AA);
        assert!(WcagLevel::AA > WcagLevel::Fail);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::Error.to_string(), "ERROR");
        assert_eq!(Severity::Warning.to_string(), "WARNING");
        assert_eq!(Severity::Info.to_string(), "INFO");
    }

    #[test]
    fn test_audit_issue_display() {
        let issue = AuditIssue {
            severity: Severity::Error,
            rule: "img-alt".into(),
            message: "Missing alt".into(),
            element: Some("hero-img".into()),
        };
        let s = issue.to_string();
        assert!(s.contains("[ERROR]"));
        assert!(s.contains("img-alt"));
        assert!(s.contains("hero-img"));
    }

    #[test]
    fn test_focus_trap_len() {
        let trap = FocusTrap::new(vec!["a".into(), "b".into()]);
        assert_eq!(trap.len(), 2);
    }

    #[test]
    fn test_color_clamp() {
        let c = SrgbColor::new(1.5, -0.2, 0.5);
        assert!((c.r - 1.0).abs() < 0.001);
        assert!(c.g.abs() < 0.001);
        assert!((c.b - 0.5).abs() < 0.001);
    }
}
