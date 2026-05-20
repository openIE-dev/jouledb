//! Media query builder: feature queries (width/height/orientation/resolution/
//! color-scheme/prefers-reduced-motion/prefers-contrast), combinators
//! (and/or/not), media type (screen/print/all), and evaluation against
//! viewport state.
//!
//! Pure Rust — emits CSS media query strings and evaluates queries
//! against a `ViewportState` struct for server-side or test usage.

use std::fmt;

// ── Media Type ──────────────────────────────────────────────────

/// CSS media type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    All,
    Screen,
    Print,
}

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediaType::All => write!(f, "all"),
            MediaType::Screen => write!(f, "screen"),
            MediaType::Print => write!(f, "print"),
        }
    }
}

// ── Color Scheme ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    Light,
    Dark,
}

impl fmt::Display for ColorScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColorScheme::Light => write!(f, "light"),
            ColorScheme::Dark => write!(f, "dark"),
        }
    }
}

// ── Orientation ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Portrait,
    Landscape,
}

impl fmt::Display for Orientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Orientation::Portrait => write!(f, "portrait"),
            Orientation::Landscape => write!(f, "landscape"),
        }
    }
}

// ── Contrast Preference ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContrastPreference {
    NoPreference,
    More,
    Less,
    Custom,
}

impl fmt::Display for ContrastPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContrastPreference::NoPreference => write!(f, "no-preference"),
            ContrastPreference::More => write!(f, "more"),
            ContrastPreference::Less => write!(f, "less"),
            ContrastPreference::Custom => write!(f, "custom"),
        }
    }
}

// ── Media Feature ───────────────────────────────────────────────

/// A single media feature condition.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaFeature {
    MinWidth(f64),
    MaxWidth(f64),
    Width(f64),
    MinHeight(f64),
    MaxHeight(f64),
    Height(f64),
    Orientation(Orientation),
    /// Resolution in dpi.
    MinResolution(f64),
    MaxResolution(f64),
    ColorScheme(ColorScheme),
    PrefersReducedMotion(bool),
    PrefersContrast(ContrastPreference),
    /// Hover capability.
    Hover(bool),
    /// Pointer precision: `fine` (true) or `coarse` (false).
    FinePointer(bool),
    /// Display mode (standalone, fullscreen, etc.).
    DisplayMode(String),
}

impl MediaFeature {
    /// CSS fragment for this feature.
    pub fn to_css(&self) -> String {
        match self {
            MediaFeature::MinWidth(v) => format!("(min-width: {v}px)"),
            MediaFeature::MaxWidth(v) => format!("(max-width: {v}px)"),
            MediaFeature::Width(v) => format!("(width: {v}px)"),
            MediaFeature::MinHeight(v) => format!("(min-height: {v}px)"),
            MediaFeature::MaxHeight(v) => format!("(max-height: {v}px)"),
            MediaFeature::Height(v) => format!("(height: {v}px)"),
            MediaFeature::Orientation(o) => format!("(orientation: {o})"),
            MediaFeature::MinResolution(dpi) => format!("(min-resolution: {dpi}dpi)"),
            MediaFeature::MaxResolution(dpi) => format!("(max-resolution: {dpi}dpi)"),
            MediaFeature::ColorScheme(cs) => format!("(prefers-color-scheme: {cs})"),
            MediaFeature::PrefersReducedMotion(reduce) => {
                if *reduce {
                    "(prefers-reduced-motion: reduce)".to_owned()
                } else {
                    "(prefers-reduced-motion: no-preference)".to_owned()
                }
            }
            MediaFeature::PrefersContrast(c) => format!("(prefers-contrast: {c})"),
            MediaFeature::Hover(can_hover) => {
                if *can_hover {
                    "(hover: hover)".to_owned()
                } else {
                    "(hover: none)".to_owned()
                }
            }
            MediaFeature::FinePointer(fine) => {
                if *fine {
                    "(pointer: fine)".to_owned()
                } else {
                    "(pointer: coarse)".to_owned()
                }
            }
            MediaFeature::DisplayMode(mode) => format!("(display-mode: {mode})"),
        }
    }
}

// ── Query Combinator ────────────────────────────────────────────

/// How features are combined within a media query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaCombinator {
    And,
    Or,
    Not,
}

// ── Media Query ─────────────────────────────────────────────────

/// A complete media query expression.
#[derive(Debug, Clone)]
pub struct MediaQuery {
    pub media_type: Option<MediaType>,
    pub features: Vec<MediaFeature>,
    pub combinator: MediaCombinator,
}

impl MediaQuery {
    pub fn new() -> Self {
        Self {
            media_type: None,
            features: Vec::new(),
            combinator: MediaCombinator::And,
        }
    }

    pub fn screen() -> Self {
        Self {
            media_type: Some(MediaType::Screen),
            features: Vec::new(),
            combinator: MediaCombinator::And,
        }
    }

    pub fn print() -> Self {
        Self {
            media_type: Some(MediaType::Print),
            features: Vec::new(),
            combinator: MediaCombinator::And,
        }
    }

    pub fn with_type(mut self, media_type: MediaType) -> Self {
        self.media_type = Some(media_type);
        self
    }

    pub fn with_combinator(mut self, combinator: MediaCombinator) -> Self {
        self.combinator = combinator;
        self
    }

    pub fn and(mut self, feature: MediaFeature) -> Self {
        self.features.push(feature);
        self
    }

    /// Generate the CSS `@media` query string (without braces).
    pub fn to_css(&self) -> String {
        let type_str = self.media_type.map(|t| t.to_string());

        if self.features.is_empty() {
            let prefix = if matches!(self.combinator, MediaCombinator::Not) { "not " } else { "" };
            return match type_str {
                Some(t) => format!("@media {prefix}{t}"),
                None => format!("@media {prefix}all"),
            };
        }

        let features_css: Vec<String> = self.features.iter().map(|f| f.to_css()).collect();

        match self.combinator {
            MediaCombinator::And => {
                let joined = features_css.join(" and ");
                match type_str {
                    Some(t) => format!("@media {t} and {joined}"),
                    None => format!("@media {joined}"),
                }
            }
            MediaCombinator::Or => {
                // CSS `or` in media queries uses comma separation.
                let parts: Vec<String> = features_css
                    .iter()
                    .map(|f| {
                        match &type_str {
                            Some(t) => format!("{t} and {f}"),
                            None => f.clone(),
                        }
                    })
                    .collect();
                format!("@media {}", parts.join(", "))
            }
            MediaCombinator::Not => {
                let joined = features_css.join(" and ");
                match type_str {
                    Some(t) => format!("@media not {t} and {joined}"),
                    None => format!("@media not all and {joined}"),
                }
            }
        }
    }

    /// Wrap CSS rules in this media query.
    pub fn wrap(&self, css_rules: &str) -> String {
        format!("{} {{\n{css_rules}}}\n", self.to_css())
    }
}

impl Default for MediaQuery {
    fn default() -> Self {
        Self::new()
    }
}

// ── Viewport State (for evaluation) ─────────────────────────────

/// The current viewport/environment state for evaluating media queries.
#[derive(Debug, Clone)]
pub struct ViewportState {
    pub width: f64,
    pub height: f64,
    pub media_type: MediaType,
    pub orientation: Orientation,
    pub resolution_dpi: f64,
    pub color_scheme: ColorScheme,
    pub prefers_reduced_motion: bool,
    pub prefers_contrast: ContrastPreference,
    pub can_hover: bool,
    pub fine_pointer: bool,
    pub display_mode: String,
}

impl ViewportState {
    /// Desktop defaults.
    pub fn desktop(width: f64, height: f64) -> Self {
        Self {
            width,
            height,
            media_type: MediaType::Screen,
            orientation: if width > height {
                Orientation::Landscape
            } else {
                Orientation::Portrait
            },
            resolution_dpi: 96.0,
            color_scheme: ColorScheme::Light,
            prefers_reduced_motion: false,
            prefers_contrast: ContrastPreference::NoPreference,
            can_hover: true,
            fine_pointer: true,
            display_mode: "browser".to_owned(),
        }
    }

    /// Mobile defaults.
    pub fn mobile(width: f64, height: f64) -> Self {
        Self {
            width,
            height,
            media_type: MediaType::Screen,
            orientation: if width > height {
                Orientation::Landscape
            } else {
                Orientation::Portrait
            },
            resolution_dpi: 326.0,
            color_scheme: ColorScheme::Light,
            prefers_reduced_motion: false,
            prefers_contrast: ContrastPreference::NoPreference,
            can_hover: false,
            fine_pointer: false,
            display_mode: "browser".to_owned(),
        }
    }

    pub fn with_color_scheme(mut self, scheme: ColorScheme) -> Self {
        self.color_scheme = scheme;
        self
    }

    pub fn with_reduced_motion(mut self, reduced: bool) -> Self {
        self.prefers_reduced_motion = reduced;
        self
    }

    /// Evaluate a single media feature against this state.
    pub fn matches_feature(&self, feature: &MediaFeature) -> bool {
        match feature {
            MediaFeature::MinWidth(v) => self.width >= *v,
            MediaFeature::MaxWidth(v) => self.width <= *v,
            MediaFeature::Width(v) => (self.width - *v).abs() < 0.5,
            MediaFeature::MinHeight(v) => self.height >= *v,
            MediaFeature::MaxHeight(v) => self.height <= *v,
            MediaFeature::Height(v) => (self.height - *v).abs() < 0.5,
            MediaFeature::Orientation(o) => self.orientation == *o,
            MediaFeature::MinResolution(dpi) => self.resolution_dpi >= *dpi,
            MediaFeature::MaxResolution(dpi) => self.resolution_dpi <= *dpi,
            MediaFeature::ColorScheme(cs) => self.color_scheme == *cs,
            MediaFeature::PrefersReducedMotion(r) => self.prefers_reduced_motion == *r,
            MediaFeature::PrefersContrast(c) => self.prefers_contrast == *c,
            MediaFeature::Hover(h) => self.can_hover == *h,
            MediaFeature::FinePointer(f) => self.fine_pointer == *f,
            MediaFeature::DisplayMode(m) => self.display_mode == *m,
        }
    }

    /// Evaluate a media type match.
    fn matches_type(&self, query_type: Option<MediaType>) -> bool {
        match query_type {
            None | Some(MediaType::All) => true,
            Some(t) => self.media_type == t,
        }
    }

    /// Evaluate a complete media query against this state.
    pub fn matches(&self, query: &MediaQuery) -> bool {
        if !self.matches_type(query.media_type) {
            return match query.combinator {
                MediaCombinator::Not => true,
                _ => false,
            };
        }

        if query.features.is_empty() {
            return match query.combinator {
                MediaCombinator::Not => false,
                _ => true,
            };
        }

        let result = match query.combinator {
            MediaCombinator::And => query.features.iter().all(|f| self.matches_feature(f)),
            MediaCombinator::Or => query.features.iter().any(|f| self.matches_feature(f)),
            MediaCombinator::Not => !query.features.iter().all(|f| self.matches_feature(f)),
        };

        result
    }
}

// ── Convenience Builders ────────────────────────────────────────

/// Shorthand: `@media (min-width: Npx)`.
pub fn min_width(px: f64) -> MediaQuery {
    MediaQuery::new().and(MediaFeature::MinWidth(px))
}

/// Shorthand: `@media (max-width: Npx)`.
pub fn max_width(px: f64) -> MediaQuery {
    MediaQuery::new().and(MediaFeature::MaxWidth(px))
}

/// Shorthand: dark mode query.
pub fn dark_mode() -> MediaQuery {
    MediaQuery::new().and(MediaFeature::ColorScheme(ColorScheme::Dark))
}

/// Shorthand: reduced motion query.
pub fn reduced_motion() -> MediaQuery {
    MediaQuery::new().and(MediaFeature::PrefersReducedMotion(true))
}

/// Shorthand: print media.
pub fn print_only() -> MediaQuery {
    MediaQuery::print()
}

/// Shorthand: between two widths.
pub fn width_between(min: f64, max: f64) -> MediaQuery {
    MediaQuery::new()
        .and(MediaFeature::MinWidth(min))
        .and(MediaFeature::MaxWidth(max))
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_width_query() {
        let q = min_width(768.0);
        assert_eq!(q.to_css(), "@media (min-width: 768px)");
    }

    #[test]
    fn test_max_width_query() {
        let q = max_width(1024.0);
        assert_eq!(q.to_css(), "@media (max-width: 1024px)");
    }

    #[test]
    fn test_dark_mode_query() {
        let q = dark_mode();
        assert_eq!(q.to_css(), "@media (prefers-color-scheme: dark)");
    }

    #[test]
    fn test_reduced_motion_query() {
        let q = reduced_motion();
        assert_eq!(q.to_css(), "@media (prefers-reduced-motion: reduce)");
    }

    #[test]
    fn test_combined_and_query() {
        let q = MediaQuery::screen()
            .and(MediaFeature::MinWidth(768.0))
            .and(MediaFeature::Orientation(Orientation::Landscape));
        let css = q.to_css();
        assert!(css.contains("screen"));
        assert!(css.contains("min-width: 768px"));
        assert!(css.contains("orientation: landscape"));
        assert!(css.contains(" and "));
    }

    #[test]
    fn test_or_query() {
        let q = MediaQuery::new()
            .with_combinator(MediaCombinator::Or)
            .and(MediaFeature::MaxWidth(480.0))
            .and(MediaFeature::Orientation(Orientation::Portrait));
        let css = q.to_css();
        assert!(css.contains(", "));
    }

    #[test]
    fn test_not_query() {
        let q = MediaQuery::new()
            .with_type(MediaType::Print)
            .with_combinator(MediaCombinator::Not);
        let css = q.to_css();
        assert!(css.contains("not print"));
    }

    #[test]
    fn test_print_query() {
        let q = print_only();
        assert_eq!(q.to_css(), "@media print");
    }

    #[test]
    fn test_width_between() {
        let q = width_between(640.0, 1024.0);
        let css = q.to_css();
        assert!(css.contains("min-width: 640px"));
        assert!(css.contains("max-width: 1024px"));
    }

    #[test]
    fn test_wrap_css() {
        let q = min_width(768.0);
        let wrapped = q.wrap("  .container { max-width: 720px; }\n");
        assert!(wrapped.starts_with("@media (min-width: 768px) {"));
        assert!(wrapped.contains(".container"));
    }

    #[test]
    fn test_viewport_state_desktop() {
        let state = ViewportState::desktop(1920.0, 1080.0);
        assert_eq!(state.orientation, Orientation::Landscape);
        assert!(state.can_hover);
        assert!(state.fine_pointer);
    }

    #[test]
    fn test_viewport_state_mobile() {
        let state = ViewportState::mobile(375.0, 812.0);
        assert_eq!(state.orientation, Orientation::Portrait);
        assert!(!state.can_hover);
        assert!(!state.fine_pointer);
    }

    #[test]
    fn test_evaluate_min_width() {
        let state = ViewportState::desktop(1200.0, 800.0);
        let q = min_width(768.0);
        assert!(state.matches(&q));
        let q2 = min_width(1400.0);
        assert!(!state.matches(&q2));
    }

    #[test]
    fn test_evaluate_dark_mode() {
        let state = ViewportState::desktop(1920.0, 1080.0)
            .with_color_scheme(ColorScheme::Dark);
        assert!(state.matches(&dark_mode()));

        let light = ViewportState::desktop(1920.0, 1080.0);
        assert!(!light.matches(&dark_mode()));
    }

    #[test]
    fn test_evaluate_reduced_motion() {
        let state = ViewportState::desktop(1920.0, 1080.0)
            .with_reduced_motion(true);
        assert!(state.matches(&reduced_motion()));
    }

    #[test]
    fn test_evaluate_not() {
        let state = ViewportState::desktop(1920.0, 1080.0);
        let q = MediaQuery::new()
            .with_combinator(MediaCombinator::Not)
            .and(MediaFeature::MaxWidth(480.0));
        // Width is 1920, so MaxWidth(480) is false. NOT false = true.
        assert!(state.matches(&q));
    }

    #[test]
    fn test_evaluate_print_mismatch() {
        let state = ViewportState::desktop(1920.0, 1080.0);
        let q = print_only();
        assert!(!state.matches(&q));
    }

    #[test]
    fn test_hover_feature() {
        let q = MediaQuery::new().and(MediaFeature::Hover(true));
        let desktop = ViewportState::desktop(1920.0, 1080.0);
        assert!(desktop.matches(&q));
        let mobile = ViewportState::mobile(375.0, 812.0);
        assert!(!mobile.matches(&q));
    }

    #[test]
    fn test_resolution_feature() {
        let q = MediaQuery::new().and(MediaFeature::MinResolution(200.0));
        let retina = ViewportState::mobile(375.0, 812.0); // 326 dpi
        assert!(retina.matches(&q));
        let standard = ViewportState::desktop(1920.0, 1080.0); // 96 dpi
        assert!(!standard.matches(&q));
    }
}
