//! CSS-in-Rust: type-safe CSS properties, stylesheets, and media queries.
//!
//! Replaces Tailwind / styled-components / CSS-in-JS with compile-time
//! checked CSS generation. Zero runtime overhead — all output is plain
//! CSS strings.

use std::fmt;

// ── CSS Values ──────────────────────────────────────────────────

/// A typed CSS value.
#[derive(Debug, Clone, PartialEq)]
pub enum CssValue {
    Px(f64),
    Em(f64),
    Rem(f64),
    Percent(f64),
    Vh(f64),
    Vw(f64),
    Auto,
    Zero,
    Inherit,
    Color(String),
    Str(String),
    Number(f64),
    Calc(String),
}

impl fmt::Display for CssValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CssValue::Px(v) => write!(f, "{v}px"),
            CssValue::Em(v) => write!(f, "{v}em"),
            CssValue::Rem(v) => write!(f, "{v}rem"),
            CssValue::Percent(v) => write!(f, "{v}%"),
            CssValue::Vh(v) => write!(f, "{v}vh"),
            CssValue::Vw(v) => write!(f, "{v}vw"),
            CssValue::Auto => write!(f, "auto"),
            CssValue::Zero => write!(f, "0"),
            CssValue::Inherit => write!(f, "inherit"),
            CssValue::Color(c) => write!(f, "{c}"),
            CssValue::Str(s) => write!(f, "{s}"),
            CssValue::Number(n) => write!(f, "{n}"),
            CssValue::Calc(expr) => write!(f, "calc({expr})"),
        }
    }
}

// ── CSS Properties ──────────────────────────────────────────────

/// Common CSS properties (type-safe enum instead of raw strings).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CssProperty {
    Display,
    Position,
    Width,
    Height,
    MinWidth,
    MaxWidth,
    MinHeight,
    MaxHeight,
    Margin,
    MarginTop,
    MarginRight,
    MarginBottom,
    MarginLeft,
    Padding,
    PaddingTop,
    PaddingRight,
    PaddingBottom,
    PaddingLeft,
    Color,
    BackgroundColor,
    Background,
    FontSize,
    FontWeight,
    FontFamily,
    TextAlign,
    LineHeight,
    LetterSpacing,
    TextDecoration,
    TextTransform,
    Border,
    BorderRadius,
    BorderColor,
    BorderWidth,
    BorderStyle,
    Opacity,
    Overflow,
    ZIndex,
    Cursor,
    BoxShadow,
    Transition,
    Transform,
    FlexDirection,
    JustifyContent,
    AlignItems,
    FlexWrap,
    Gap,
    FlexGrow,
    FlexShrink,
    GridTemplateColumns,
    GridTemplateRows,
    GridGap,
    GridColumn,
    GridRow,
    Top,
    Right,
    Bottom,
    Left,
}

impl fmt::Display for CssProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CssProperty::Display => "display",
            CssProperty::Position => "position",
            CssProperty::Width => "width",
            CssProperty::Height => "height",
            CssProperty::MinWidth => "min-width",
            CssProperty::MaxWidth => "max-width",
            CssProperty::MinHeight => "min-height",
            CssProperty::MaxHeight => "max-height",
            CssProperty::Margin => "margin",
            CssProperty::MarginTop => "margin-top",
            CssProperty::MarginRight => "margin-right",
            CssProperty::MarginBottom => "margin-bottom",
            CssProperty::MarginLeft => "margin-left",
            CssProperty::Padding => "padding",
            CssProperty::PaddingTop => "padding-top",
            CssProperty::PaddingRight => "padding-right",
            CssProperty::PaddingBottom => "padding-bottom",
            CssProperty::PaddingLeft => "padding-left",
            CssProperty::Color => "color",
            CssProperty::BackgroundColor => "background-color",
            CssProperty::Background => "background",
            CssProperty::FontSize => "font-size",
            CssProperty::FontWeight => "font-weight",
            CssProperty::FontFamily => "font-family",
            CssProperty::TextAlign => "text-align",
            CssProperty::LineHeight => "line-height",
            CssProperty::LetterSpacing => "letter-spacing",
            CssProperty::TextDecoration => "text-decoration",
            CssProperty::TextTransform => "text-transform",
            CssProperty::Border => "border",
            CssProperty::BorderRadius => "border-radius",
            CssProperty::BorderColor => "border-color",
            CssProperty::BorderWidth => "border-width",
            CssProperty::BorderStyle => "border-style",
            CssProperty::Opacity => "opacity",
            CssProperty::Overflow => "overflow",
            CssProperty::ZIndex => "z-index",
            CssProperty::Cursor => "cursor",
            CssProperty::BoxShadow => "box-shadow",
            CssProperty::Transition => "transition",
            CssProperty::Transform => "transform",
            CssProperty::FlexDirection => "flex-direction",
            CssProperty::JustifyContent => "justify-content",
            CssProperty::AlignItems => "align-items",
            CssProperty::FlexWrap => "flex-wrap",
            CssProperty::Gap => "gap",
            CssProperty::FlexGrow => "flex-grow",
            CssProperty::FlexShrink => "flex-shrink",
            CssProperty::GridTemplateColumns => "grid-template-columns",
            CssProperty::GridTemplateRows => "grid-template-rows",
            CssProperty::GridGap => "grid-gap",
            CssProperty::GridColumn => "grid-column",
            CssProperty::GridRow => "grid-row",
            CssProperty::Top => "top",
            CssProperty::Right => "right",
            CssProperty::Bottom => "bottom",
            CssProperty::Left => "left",
        };
        write!(f, "{s}")
    }
}

// ── Style Rule ──────────────────────────────────────────────────

/// A single CSS property-value pair.
#[derive(Debug, Clone, PartialEq)]
pub struct StyleRule {
    pub property: CssProperty,
    pub value: CssValue,
}

// ── Style (fluent builder) ──────────────────────────────────────

/// A collection of CSS rules with a fluent builder API.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Style {
    pub rules: Vec<StyleRule>,
}

impl Style {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add an arbitrary property-value rule.
    pub fn rule(mut self, property: CssProperty, value: CssValue) -> Self {
        self.rules.push(StyleRule { property, value });
        self
    }

    // ── Fluent shortcuts ────────────────────────────────────────

    pub fn display(self, v: CssValue) -> Self { self.rule(CssProperty::Display, v) }
    pub fn position(self, v: CssValue) -> Self { self.rule(CssProperty::Position, v) }
    pub fn width(self, v: CssValue) -> Self { self.rule(CssProperty::Width, v) }
    pub fn height(self, v: CssValue) -> Self { self.rule(CssProperty::Height, v) }
    pub fn min_width(self, v: CssValue) -> Self { self.rule(CssProperty::MinWidth, v) }
    pub fn max_width(self, v: CssValue) -> Self { self.rule(CssProperty::MaxWidth, v) }
    pub fn min_height(self, v: CssValue) -> Self { self.rule(CssProperty::MinHeight, v) }
    pub fn max_height(self, v: CssValue) -> Self { self.rule(CssProperty::MaxHeight, v) }
    pub fn margin(self, v: CssValue) -> Self { self.rule(CssProperty::Margin, v) }
    pub fn padding(self, v: CssValue) -> Self { self.rule(CssProperty::Padding, v) }
    pub fn color(self, v: CssValue) -> Self { self.rule(CssProperty::Color, v) }
    pub fn bg(self, v: CssValue) -> Self { self.rule(CssProperty::BackgroundColor, v) }
    pub fn font_size(self, v: CssValue) -> Self { self.rule(CssProperty::FontSize, v) }
    pub fn font_weight(self, v: CssValue) -> Self { self.rule(CssProperty::FontWeight, v) }
    pub fn border_radius(self, v: CssValue) -> Self { self.rule(CssProperty::BorderRadius, v) }
    pub fn border(self, v: CssValue) -> Self { self.rule(CssProperty::Border, v) }
    pub fn opacity(self, v: CssValue) -> Self { self.rule(CssProperty::Opacity, v) }
    pub fn z_index(self, v: CssValue) -> Self { self.rule(CssProperty::ZIndex, v) }
    pub fn cursor(self, v: CssValue) -> Self { self.rule(CssProperty::Cursor, v) }
    pub fn box_shadow(self, v: CssValue) -> Self { self.rule(CssProperty::BoxShadow, v) }
    pub fn transition(self, v: CssValue) -> Self { self.rule(CssProperty::Transition, v) }
    pub fn transform(self, v: CssValue) -> Self { self.rule(CssProperty::Transform, v) }
    pub fn gap(self, v: CssValue) -> Self { self.rule(CssProperty::Gap, v) }
    pub fn flex_grow(self, v: CssValue) -> Self { self.rule(CssProperty::FlexGrow, v) }
    pub fn flex_shrink(self, v: CssValue) -> Self { self.rule(CssProperty::FlexShrink, v) }
    pub fn top(self, v: CssValue) -> Self { self.rule(CssProperty::Top, v) }
    pub fn right(self, v: CssValue) -> Self { self.rule(CssProperty::Right, v) }
    pub fn bottom(self, v: CssValue) -> Self { self.rule(CssProperty::Bottom, v) }
    pub fn left(self, v: CssValue) -> Self { self.rule(CssProperty::Left, v) }
    pub fn flex_direction(self, v: CssValue) -> Self { self.rule(CssProperty::FlexDirection, v) }
    pub fn justify_content(self, v: CssValue) -> Self { self.rule(CssProperty::JustifyContent, v) }
    pub fn align_items(self, v: CssValue) -> Self { self.rule(CssProperty::AlignItems, v) }
    pub fn grid_template_columns(self, v: CssValue) -> Self { self.rule(CssProperty::GridTemplateColumns, v) }
    pub fn grid_template_rows(self, v: CssValue) -> Self { self.rule(CssProperty::GridTemplateRows, v) }
    pub fn grid_gap(self, v: CssValue) -> Self { self.rule(CssProperty::GridGap, v) }
    pub fn grid_column(self, v: CssValue) -> Self { self.rule(CssProperty::GridColumn, v) }
    pub fn grid_row(self, v: CssValue) -> Self { self.rule(CssProperty::GridRow, v) }
    pub fn text_align(self, v: CssValue) -> Self { self.rule(CssProperty::TextAlign, v) }
    pub fn line_height(self, v: CssValue) -> Self { self.rule(CssProperty::LineHeight, v) }
    pub fn overflow(self, v: CssValue) -> Self { self.rule(CssProperty::Overflow, v) }

    /// Shorthand: sets `display: flex`.
    pub fn flex(self) -> Self { self.display(CssValue::Str("flex".into())) }

    /// Shorthand: sets `display: grid`.
    pub fn grid(self) -> Self { self.display(CssValue::Str("grid".into())) }

    // ── Rendering ───────────────────────────────────────────────

    /// Render as CSS declaration block: `display: flex; width: 100px;`
    pub fn to_css_string(&self) -> String {
        self.rules
            .iter()
            .map(|r| format!("{}: {}", r.property, r.value))
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Alias for inline style attribute usage.
    pub fn to_inline_style(&self) -> String {
        self.to_css_string()
    }
}

// ── Media Query ─────────────────────────────────────────────────

/// A CSS `@media` query containing scoped style rules.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaQuery {
    pub condition: String,
    pub styles: Vec<(String, Style)>,
}

impl MediaQuery {
    pub fn min_width(px: f64) -> Self {
        Self { condition: format!("(min-width: {px}px)"), styles: Vec::new() }
    }

    pub fn max_width(px: f64) -> Self {
        Self { condition: format!("(max-width: {px}px)"), styles: Vec::new() }
    }

    pub fn prefers_dark() -> Self {
        Self { condition: "(prefers-color-scheme: dark)".into(), styles: Vec::new() }
    }

    pub fn prefers_reduced_motion() -> Self {
        Self { condition: "(prefers-reduced-motion: reduce)".into(), styles: Vec::new() }
    }

    /// Add a selector + style pair to this media query.
    pub fn rule(mut self, selector: &str, style: Style) -> Self {
        self.styles.push((selector.to_string(), style));
        self
    }

    /// Render the full `@media` block.
    pub fn to_css(&self) -> String {
        let mut out = format!("@media {} {{\n", self.condition);
        for (sel, style) in &self.styles {
            out.push_str(&format!("  {} {{ {} }}\n", sel, style.to_css_string()));
        }
        out.push('}');
        out
    }
}

// ── Stylesheet ──────────────────────────────────────────────────

/// A single entry in a stylesheet.
#[derive(Debug, Clone, PartialEq)]
pub enum CssRule {
    Style { selector: String, style: Style },
    Media(MediaQuery),
    Keyframes { name: String, steps: Vec<(f64, Style)> },
    Raw(String),
}

/// A full CSS stylesheet built from typed rules.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Stylesheet {
    pub rules: Vec<CssRule>,
}

impl Stylesheet {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn rule(mut self, selector: &str, style: Style) -> Self {
        self.rules.push(CssRule::Style { selector: selector.to_string(), style });
        self
    }

    pub fn media(mut self, query: MediaQuery) -> Self {
        self.rules.push(CssRule::Media(query));
        self
    }

    pub fn keyframes(mut self, name: &str, steps: Vec<(f64, Style)>) -> Self {
        self.rules.push(CssRule::Keyframes { name: name.to_string(), steps });
        self
    }

    pub fn raw(mut self, css: &str) -> Self {
        self.rules.push(CssRule::Raw(css.to_string()));
        self
    }

    /// Create a scoped version — all selectors get a prefix prepended.
    pub fn scoped(mut self, prefix: &str) -> Stylesheet {
        for rule in &mut self.rules {
            match rule {
                CssRule::Style { selector, .. } => {
                    *selector = format!("{prefix} {selector}");
                }
                CssRule::Media(mq) => {
                    for (sel, _) in &mut mq.styles {
                        *sel = format!("{prefix} {sel}");
                    }
                }
                CssRule::Keyframes { .. } | CssRule::Raw(_) => {}
            }
        }
        self
    }

    /// Render the entire stylesheet as a CSS string.
    pub fn to_css(&self) -> String {
        let mut out = String::new();
        for rule in &self.rules {
            match rule {
                CssRule::Style { selector, style } => {
                    out.push_str(&format!("{selector} {{ {} }}\n", style.to_css_string()));
                }
                CssRule::Media(mq) => {
                    out.push_str(&mq.to_css());
                    out.push('\n');
                }
                CssRule::Keyframes { name, steps } => {
                    out.push_str(&format!("@keyframes {name} {{\n"));
                    for (pct, style) in steps {
                        let pct_str = if (*pct - 0.0).abs() < f64::EPSILON {
                            "from".to_string()
                        } else if (*pct - 100.0).abs() < f64::EPSILON {
                            "to".to_string()
                        } else {
                            format!("{pct}%")
                        };
                        out.push_str(&format!("  {pct_str} {{ {} }}\n", style.to_css_string()));
                    }
                    out.push_str("}\n");
                }
                CssRule::Raw(raw) => {
                    out.push_str(raw);
                    out.push('\n');
                }
            }
        }
        out
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn css_value_px_format() {
        assert_eq!(CssValue::Px(12.0).to_string(), "12px");
    }

    #[test]
    fn css_value_em_format() {
        assert_eq!(CssValue::Em(1.5).to_string(), "1.5em");
    }

    #[test]
    fn css_value_rem_format() {
        assert_eq!(CssValue::Rem(2.0).to_string(), "2rem");
    }

    #[test]
    fn css_value_percent_format() {
        assert_eq!(CssValue::Percent(50.0).to_string(), "50%");
    }

    #[test]
    fn css_value_auto_inherit_zero() {
        assert_eq!(CssValue::Auto.to_string(), "auto");
        assert_eq!(CssValue::Inherit.to_string(), "inherit");
        assert_eq!(CssValue::Zero.to_string(), "0");
    }

    #[test]
    fn css_value_calc() {
        assert_eq!(CssValue::Calc("100% - 20px".into()).to_string(), "calc(100% - 20px)");
    }

    #[test]
    fn css_value_viewport_units() {
        assert_eq!(CssValue::Vh(100.0).to_string(), "100vh");
        assert_eq!(CssValue::Vw(50.0).to_string(), "50vw");
    }

    #[test]
    fn style_fluent_api_and_css_string() {
        let style = Style::new()
            .display(CssValue::Str("flex".into()))
            .width(CssValue::Px(100.0))
            .height(CssValue::Percent(50.0));
        let css = style.to_css_string();
        assert_eq!(css, "display: flex; width: 100px; height: 50%");
    }

    #[test]
    fn style_inline_same_as_css_string() {
        let style = Style::new().color(CssValue::Color("#fff".into()));
        assert_eq!(style.to_css_string(), style.to_inline_style());
    }

    #[test]
    fn flex_shorthand() {
        let style = Style::new()
            .flex()
            .justify_content(CssValue::Str("center".into()))
            .align_items(CssValue::Str("center".into()));
        let css = style.to_css_string();
        assert!(css.contains("display: flex"));
        assert!(css.contains("justify-content: center"));
        assert!(css.contains("align-items: center"));
    }

    #[test]
    fn grid_properties() {
        let style = Style::new()
            .grid()
            .grid_template_columns(CssValue::Str("1fr 1fr 1fr".into()))
            .grid_gap(CssValue::Px(16.0));
        let css = style.to_css_string();
        assert!(css.contains("display: grid"));
        assert!(css.contains("grid-template-columns: 1fr 1fr 1fr"));
        assert!(css.contains("grid-gap: 16px"));
    }

    #[test]
    fn media_query_min_width() {
        let mq = MediaQuery::min_width(768.0)
            .rule(".container", Style::new().width(CssValue::Percent(100.0)));
        let css = mq.to_css();
        assert!(css.contains("@media (min-width: 768px)"));
        assert!(css.contains(".container"));
        assert!(css.contains("width: 100%"));
    }

    #[test]
    fn media_query_prefers_dark() {
        let mq = MediaQuery::prefers_dark()
            .rule("body", Style::new().bg(CssValue::Color("#000".into())));
        let css = mq.to_css();
        assert!(css.contains("prefers-color-scheme: dark"));
    }

    #[test]
    fn stylesheet_multiple_rules() {
        let sheet = Stylesheet::new()
            .rule("body", Style::new().margin(CssValue::Zero))
            .rule(".header", Style::new().height(CssValue::Px(60.0)));
        let css = sheet.to_css();
        assert!(css.contains("body { margin: 0 }"));
        assert!(css.contains(".header { height: 60px }"));
    }

    #[test]
    fn stylesheet_keyframes() {
        let sheet = Stylesheet::new().keyframes(
            "fadeIn",
            vec![
                (0.0, Style::new().opacity(CssValue::Number(0.0))),
                (100.0, Style::new().opacity(CssValue::Number(1.0))),
            ],
        );
        let css = sheet.to_css();
        assert!(css.contains("@keyframes fadeIn"));
        assert!(css.contains("from { opacity: 0 }"));
        assert!(css.contains("to { opacity: 1 }"));
    }

    #[test]
    fn stylesheet_scoped_prefix() {
        let sheet = Stylesheet::new()
            .rule(".btn", Style::new().color(CssValue::Color("red".into())))
            .scoped("[data-c123]");
        let css = sheet.to_css();
        assert!(css.contains("[data-c123] .btn"));
    }

    #[test]
    fn stylesheet_raw_rule() {
        let sheet = Stylesheet::new().raw("/* custom */\n* { box-sizing: border-box; }");
        let css = sheet.to_css();
        assert!(css.contains("box-sizing: border-box"));
    }

    #[test]
    fn complex_nested_selectors() {
        let sheet = Stylesheet::new()
            .rule("nav > ul > li:first-child a:hover", Style::new().color(CssValue::Color("blue".into())));
        let css = sheet.to_css();
        assert!(css.contains("nav > ul > li:first-child a:hover { color: blue }"));
    }

    #[test]
    fn css_property_display_names() {
        assert_eq!(CssProperty::BackgroundColor.to_string(), "background-color");
        assert_eq!(CssProperty::FlexDirection.to_string(), "flex-direction");
        assert_eq!(CssProperty::GridTemplateColumns.to_string(), "grid-template-columns");
        assert_eq!(CssProperty::ZIndex.to_string(), "z-index");
        assert_eq!(CssProperty::MarginTop.to_string(), "margin-top");
    }
}
