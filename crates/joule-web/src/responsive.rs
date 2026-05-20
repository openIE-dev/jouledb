//! Responsive design utilities — breakpoints, media queries, container queries,
//! fluid typography, aspect ratios, and viewport units.
//!
//! Pure Rust builder for CSS responsive design patterns. No browser APIs.

use std::fmt;

// ── Breakpoint ───────────────────────────────────────────────────

/// Named breakpoint with a minimum width in pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Breakpoint {
    pub name: String,
    pub min_width_px: u32,
}

impl Breakpoint {
    pub fn new(name: impl Into<String>, min_width_px: u32) -> Self {
        Self { name: name.into(), min_width_px }
    }
}

/// Standard responsive breakpoint system.
#[derive(Debug, Clone)]
pub struct BreakpointSystem {
    breakpoints: Vec<Breakpoint>,
}

impl BreakpointSystem {
    /// Create an empty breakpoint system.
    pub fn new() -> Self {
        Self { breakpoints: Vec::new() }
    }

    /// Bootstrap 5 breakpoints.
    pub fn bootstrap() -> Self {
        Self {
            breakpoints: vec![
                Breakpoint::new("xs", 0),
                Breakpoint::new("sm", 576),
                Breakpoint::new("md", 768),
                Breakpoint::new("lg", 992),
                Breakpoint::new("xl", 1200),
                Breakpoint::new("xxl", 1400),
            ],
        }
    }

    /// Tailwind CSS breakpoints.
    pub fn tailwind() -> Self {
        Self {
            breakpoints: vec![
                Breakpoint::new("sm", 640),
                Breakpoint::new("md", 768),
                Breakpoint::new("lg", 1024),
                Breakpoint::new("xl", 1280),
                Breakpoint::new("2xl", 1536),
            ],
        }
    }

    /// Add a custom breakpoint. Keeps sorted by min_width.
    pub fn add(mut self, bp: Breakpoint) -> Self {
        self.breakpoints.push(bp);
        self.breakpoints.sort_by_key(|b| b.min_width_px);
        self
    }

    /// Get all breakpoints, sorted ascending.
    pub fn all(&self) -> &[Breakpoint] {
        &self.breakpoints
    }

    /// Find breakpoint by name.
    pub fn get(&self, name: &str) -> Option<&Breakpoint> {
        self.breakpoints.iter().find(|b| b.name == name)
    }

    /// Generate min-width media query for a breakpoint name.
    pub fn up(&self, name: &str) -> Option<String> {
        self.get(name).map(|bp| {
            if bp.min_width_px == 0 {
                String::new() // No media query needed for 0px
            } else {
                format!("@media (min-width: {}px)", bp.min_width_px)
            }
        })
    }

    /// Generate max-width media query (next breakpoint - 0.02px).
    pub fn down(&self, name: &str) -> Option<String> {
        let idx = self.breakpoints.iter().position(|b| b.name == name)?;
        if idx + 1 < self.breakpoints.len() {
            let next = &self.breakpoints[idx + 1];
            let max = next.min_width_px as f64 - 0.02;
            Some(format!("@media (max-width: {:.2}px)", max))
        } else {
            // Last breakpoint has no upper bound
            None
        }
    }

    /// Generate range media query between two breakpoints.
    pub fn between(&self, lower: &str, upper: &str) -> Option<String> {
        let low = self.get(lower)?;
        let up_idx = self.breakpoints.iter().position(|b| b.name == upper)?;
        if up_idx + 1 < self.breakpoints.len() {
            let next = &self.breakpoints[up_idx + 1];
            let max = next.min_width_px as f64 - 0.02;
            Some(format!(
                "@media (min-width: {}px) and (max-width: {:.2}px)",
                low.min_width_px, max
            ))
        } else {
            Some(format!("@media (min-width: {}px)", low.min_width_px))
        }
    }

    /// Determine which breakpoint a given width falls into.
    pub fn current_breakpoint(&self, width_px: u32) -> Option<&Breakpoint> {
        self.breakpoints.iter().rev().find(|bp| width_px >= bp.min_width_px)
    }

    /// Number of defined breakpoints.
    pub fn count(&self) -> usize { self.breakpoints.len() }
}

impl Default for BreakpointSystem {
    fn default() -> Self { Self::new() }
}

// ── Media Query Builder ──────────────────────────────────────────

/// Media feature for building queries.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaFeature {
    MinWidth(f64),
    MaxWidth(f64),
    MinHeight(f64),
    MaxHeight(f64),
    Orientation(MediaOrientation),
    PrefersColorScheme(ColorScheme),
    PrefersReducedMotion,
    PrefersContrast(ContrastPref),
    Hover(bool),
    Pointer(PointerType),
    DisplayMode(String),
    AspectRatio(u32, u32),
    Resolution(f64),
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaOrientation { Portrait, Landscape }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme { Light, Dark }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContrastPref { More, Less, NoPreference }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerType { Fine, Coarse, None }

impl fmt::Display for MediaFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinWidth(v) => write!(f, "(min-width: {v}px)"),
            Self::MaxWidth(v) => write!(f, "(max-width: {v}px)"),
            Self::MinHeight(v) => write!(f, "(min-height: {v}px)"),
            Self::MaxHeight(v) => write!(f, "(max-height: {v}px)"),
            Self::Orientation(o) => {
                let s = match o {
                    MediaOrientation::Portrait => "portrait",
                    MediaOrientation::Landscape => "landscape",
                };
                write!(f, "(orientation: {s})")
            }
            Self::PrefersColorScheme(cs) => {
                let s = match cs { ColorScheme::Light => "light", ColorScheme::Dark => "dark" };
                write!(f, "(prefers-color-scheme: {s})")
            }
            Self::PrefersReducedMotion => write!(f, "(prefers-reduced-motion: reduce)"),
            Self::PrefersContrast(c) => {
                let s = match c {
                    ContrastPref::More => "more",
                    ContrastPref::Less => "less",
                    ContrastPref::NoPreference => "no-preference",
                };
                write!(f, "(prefers-contrast: {s})")
            }
            Self::Hover(h) => write!(f, "(hover: {})", if *h { "hover" } else { "none" }),
            Self::Pointer(p) => {
                let s = match p { PointerType::Fine => "fine", PointerType::Coarse => "coarse", PointerType::None => "none" };
                write!(f, "(pointer: {s})")
            }
            Self::DisplayMode(m) => write!(f, "(display-mode: {m})"),
            Self::AspectRatio(w, h) => write!(f, "(aspect-ratio: {w}/{h})"),
            Self::Resolution(dpi) => write!(f, "(min-resolution: {dpi}dpi)"),
            Self::Custom(c) => write!(f, "{c}"),
        }
    }
}

/// Builder for CSS media queries.
#[derive(Debug, Clone)]
pub struct MediaQuery {
    media_type: Option<String>,
    features: Vec<MediaFeature>,
    negate: bool,
}

impl MediaQuery {
    pub fn new() -> Self {
        Self { media_type: None, features: Vec::new(), negate: false }
    }

    pub fn screen() -> Self { Self::new().media_type("screen") }
    pub fn print() -> Self { Self::new().media_type("print") }

    pub fn media_type(mut self, t: impl Into<String>) -> Self { self.media_type = Some(t.into()); self }
    pub fn not(mut self) -> Self { self.negate = true; self }

    pub fn feature(mut self, f: MediaFeature) -> Self { self.features.push(f); self }
    pub fn min_width(self, px: f64) -> Self { self.feature(MediaFeature::MinWidth(px)) }
    pub fn max_width(self, px: f64) -> Self { self.feature(MediaFeature::MaxWidth(px)) }
    pub fn min_height(self, px: f64) -> Self { self.feature(MediaFeature::MinHeight(px)) }
    pub fn max_height(self, px: f64) -> Self { self.feature(MediaFeature::MaxHeight(px)) }
    pub fn dark_mode(self) -> Self { self.feature(MediaFeature::PrefersColorScheme(ColorScheme::Dark)) }
    pub fn light_mode(self) -> Self { self.feature(MediaFeature::PrefersColorScheme(ColorScheme::Light)) }
    pub fn reduced_motion(self) -> Self { self.feature(MediaFeature::PrefersReducedMotion) }
    pub fn landscape(self) -> Self { self.feature(MediaFeature::Orientation(MediaOrientation::Landscape)) }
    pub fn portrait(self) -> Self { self.feature(MediaFeature::Orientation(MediaOrientation::Portrait)) }

    /// Render the full @media query string.
    pub fn build(&self) -> String {
        let mut parts = Vec::new();
        if self.negate { parts.push("not".to_string()); }
        if let Some(ref mt) = self.media_type { parts.push(mt.clone()); }
        for feat in &self.features {
            parts.push(feat.to_string());
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!("@media {}", parts.join(" and "))
        }
    }
}

impl Default for MediaQuery {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MediaQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.build())
    }
}

// ── Container Query ──────────────────────────────────────────────

/// CSS container query builder.
#[derive(Debug, Clone)]
pub struct ContainerQuery {
    pub container_name: Option<String>,
    pub min_width: Option<f64>,
    pub max_width: Option<f64>,
    pub min_height: Option<f64>,
    pub max_height: Option<f64>,
}

impl ContainerQuery {
    pub fn new() -> Self {
        Self { container_name: None, min_width: None, max_width: None, min_height: None, max_height: None }
    }

    pub fn named(name: impl Into<String>) -> Self {
        Self { container_name: Some(name.into()), ..Self::new() }
    }

    pub fn min_width(mut self, px: f64) -> Self { self.min_width = Some(px); self }
    pub fn max_width(mut self, px: f64) -> Self { self.max_width = Some(px); self }
    pub fn min_height(mut self, px: f64) -> Self { self.min_height = Some(px); self }
    pub fn max_height(mut self, px: f64) -> Self { self.max_height = Some(px); self }

    /// Generate the @container rule string.
    pub fn build(&self) -> String {
        let mut conditions = Vec::new();
        if let Some(v) = self.min_width { conditions.push(format!("(min-width: {v}px)")); }
        if let Some(v) = self.max_width { conditions.push(format!("(max-width: {v}px)")); }
        if let Some(v) = self.min_height { conditions.push(format!("(min-height: {v}px)")); }
        if let Some(v) = self.max_height { conditions.push(format!("(max-height: {v}px)")); }

        let cond = conditions.join(" and ");
        match &self.container_name {
            Some(name) => format!("@container {name} {cond}"),
            None => format!("@container {cond}"),
        }
    }

    /// Generate container-type CSS property for the container element.
    pub fn container_type_css(container_type: ContainerType) -> String {
        match container_type {
            ContainerType::InlineSize => "container-type: inline-size;".to_string(),
            ContainerType::Size => "container-type: size;".to_string(),
            ContainerType::Normal => "container-type: normal;".to_string(),
        }
    }
}

impl Default for ContainerQuery {
    fn default() -> Self { Self::new() }
}

/// Container sizing types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerType { InlineSize, Size, Normal }

// ── Fluid Typography ─────────────────────────────────────────────

/// Calculate CSS clamp() value for fluid typography.
///
/// Generates `clamp(min_size, preferred, max_size)` where preferred is
/// a viewport-relative calculation between min_viewport and max_viewport.
pub fn fluid_type(
    min_size_px: f64,
    max_size_px: f64,
    min_viewport_px: f64,
    max_viewport_px: f64,
) -> String {
    let slope = (max_size_px - min_size_px) / (max_viewport_px - min_viewport_px);
    let intercept = min_size_px - slope * min_viewport_px;
    let vw = slope * 100.0;
    let rem_intercept = intercept / 16.0;

    format!(
        "clamp({:.4}rem, {:.4}rem + {:.4}vw, {:.4}rem)",
        min_size_px / 16.0,
        rem_intercept,
        vw,
        max_size_px / 16.0,
    )
}

/// Generate a fluid spacing scale.
pub fn fluid_space(
    min_px: f64,
    max_px: f64,
    min_vp: f64,
    max_vp: f64,
) -> String {
    fluid_type(min_px, max_px, min_vp, max_vp)
}

// ── Aspect Ratio ─────────────────────────────────────────────────

/// Aspect ratio helper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AspectRatio {
    pub width: f64,
    pub height: f64,
}

impl AspectRatio {
    pub fn new(width: f64, height: f64) -> Self { Self { width, height } }

    /// Common 16:9 ratio.
    pub fn widescreen() -> Self { Self::new(16.0, 9.0) }
    /// Common 4:3 ratio.
    pub fn standard() -> Self { Self::new(4.0, 3.0) }
    /// Square 1:1.
    pub fn square() -> Self { Self::new(1.0, 1.0) }
    /// Ultrawide 21:9.
    pub fn ultrawide() -> Self { Self::new(21.0, 9.0) }

    /// Decimal ratio value.
    pub fn ratio(&self) -> f64 { self.width / self.height }

    /// CSS aspect-ratio property value.
    pub fn css_value(&self) -> String {
        if self.width == self.width.floor() && self.height == self.height.floor() {
            format!("{} / {}", self.width as u32, self.height as u32)
        } else {
            format!("{:.2} / {:.2}", self.width, self.height)
        }
    }

    /// Padding-bottom trick for older browsers.
    pub fn padding_bottom_percent(&self) -> f64 {
        (self.height / self.width) * 100.0
    }

    /// Calculate height for a given width.
    pub fn height_for_width(&self, w: f64) -> f64 {
        w / self.ratio()
    }

    /// Calculate width for a given height.
    pub fn width_for_height(&self, h: f64) -> f64 {
        h * self.ratio()
    }
}

impl fmt::Display for AspectRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.css_value())
    }
}

// ── Viewport Units ───────────────────────────────────────────────

/// Viewport unit types (including new logical units).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportUnit {
    Vw, Vh, Vmin, Vmax,
    Svw, Svh,  // Small viewport
    Lvw, Lvh,  // Large viewport
    Dvw, Dvh,  // Dynamic viewport
}

impl fmt::Display for ViewportUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vw => write!(f, "vw"),
            Self::Vh => write!(f, "vh"),
            Self::Vmin => write!(f, "vmin"),
            Self::Vmax => write!(f, "vmax"),
            Self::Svw => write!(f, "svw"),
            Self::Svh => write!(f, "svh"),
            Self::Lvw => write!(f, "lvw"),
            Self::Lvh => write!(f, "lvh"),
            Self::Dvw => write!(f, "dvw"),
            Self::Dvh => write!(f, "dvh"),
        }
    }
}

/// Create a viewport-unit CSS value.
pub fn vp(value: f64, unit: ViewportUnit) -> String {
    format!("{value}{unit}")
}

/// Full-viewport-height that accounts for mobile browser chrome.
/// Uses `min(100vh, 100dvh)` when available, else `100vh`.
pub fn full_height_safe() -> String {
    "min(100vh, 100dvh)".to_string()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_breakpoints() {
        let bs = BreakpointSystem::bootstrap();
        assert_eq!(bs.count(), 6);
        assert_eq!(bs.get("sm").unwrap().min_width_px, 576);
        assert_eq!(bs.get("xl").unwrap().min_width_px, 1200);
    }

    #[test]
    fn test_tailwind_breakpoints() {
        let tw = BreakpointSystem::tailwind();
        assert_eq!(tw.count(), 5);
        assert_eq!(tw.get("lg").unwrap().min_width_px, 1024);
    }

    #[test]
    fn test_breakpoint_up() {
        let bs = BreakpointSystem::bootstrap();
        assert_eq!(bs.up("xs").unwrap(), ""); // 0px = no query
        assert_eq!(bs.up("md").unwrap(), "@media (min-width: 768px)");
    }

    #[test]
    fn test_breakpoint_down() {
        let bs = BreakpointSystem::bootstrap();
        let down_sm = bs.down("sm").unwrap();
        assert!(down_sm.contains("max-width"));
        assert!(down_sm.contains("767.98"));
        assert!(bs.down("xxl").is_none()); // last = no upper bound
    }

    #[test]
    fn test_breakpoint_between() {
        let bs = BreakpointSystem::bootstrap();
        let btwn = bs.between("sm", "lg").unwrap();
        assert!(btwn.contains("min-width: 576px"));
        assert!(btwn.contains("max-width"));
    }

    #[test]
    fn test_current_breakpoint() {
        let bs = BreakpointSystem::bootstrap();
        assert_eq!(bs.current_breakpoint(400).unwrap().name, "xs");
        assert_eq!(bs.current_breakpoint(800).unwrap().name, "md");
        assert_eq!(bs.current_breakpoint(1500).unwrap().name, "xxl");
    }

    #[test]
    fn test_custom_breakpoints() {
        let bs = BreakpointSystem::new()
            .add(Breakpoint::new("phone", 0))
            .add(Breakpoint::new("tablet", 600))
            .add(Breakpoint::new("desktop", 1024));

        assert_eq!(bs.count(), 3);
        assert_eq!(bs.all()[0].name, "phone");
        assert_eq!(bs.all()[2].name, "desktop");
    }

    #[test]
    fn test_media_query_basic() {
        let mq = MediaQuery::screen().min_width(768.0);
        let result = mq.build();
        assert!(result.contains("@media"));
        assert!(result.contains("screen"));
        assert!(result.contains("min-width: 768px"));
    }

    #[test]
    fn test_media_query_dark_mode() {
        let mq = MediaQuery::new().dark_mode();
        assert!(mq.build().contains("prefers-color-scheme: dark"));
    }

    #[test]
    fn test_media_query_reduced_motion() {
        let mq = MediaQuery::new().reduced_motion();
        assert!(mq.build().contains("prefers-reduced-motion: reduce"));
    }

    #[test]
    fn test_media_query_combined() {
        let mq = MediaQuery::screen()
            .min_width(768.0)
            .max_width(1024.0)
            .landscape();

        let result = mq.build();
        assert!(result.contains("screen"));
        assert!(result.contains("min-width: 768px"));
        assert!(result.contains("max-width: 1024px"));
        assert!(result.contains("orientation: landscape"));
    }

    #[test]
    fn test_media_query_print() {
        let mq = MediaQuery::print();
        assert!(mq.build().contains("print"));
    }

    #[test]
    fn test_media_query_not() {
        let mq = MediaQuery::new().not().media_type("print");
        assert!(mq.build().contains("not"));
    }

    #[test]
    fn test_media_query_pointer() {
        let mq = MediaQuery::new().feature(MediaFeature::Pointer(PointerType::Coarse));
        assert!(mq.build().contains("pointer: coarse"));
    }

    #[test]
    fn test_media_query_hover() {
        let mq = MediaQuery::new().feature(MediaFeature::Hover(true));
        assert!(mq.build().contains("hover: hover"));
    }

    #[test]
    fn test_media_query_display_trait() {
        let mq = MediaQuery::screen().min_width(640.0);
        let s = format!("{mq}");
        assert!(s.starts_with("@media"));
    }

    #[test]
    fn test_container_query_basic() {
        let cq = ContainerQuery::new().min_width(400.0);
        let result = cq.build();
        assert!(result.contains("@container"));
        assert!(result.contains("min-width: 400px"));
    }

    #[test]
    fn test_container_query_named() {
        let cq = ContainerQuery::named("sidebar").min_width(300.0).max_width(600.0);
        let result = cq.build();
        assert!(result.contains("@container sidebar"));
        assert!(result.contains("min-width: 300px"));
        assert!(result.contains("max-width: 600px"));
    }

    #[test]
    fn test_container_type_css() {
        assert_eq!(
            ContainerQuery::container_type_css(ContainerType::InlineSize),
            "container-type: inline-size;"
        );
    }

    #[test]
    fn test_fluid_type() {
        let clamp = fluid_type(16.0, 24.0, 320.0, 1200.0);
        assert!(clamp.starts_with("clamp("));
        assert!(clamp.contains("rem"));
        assert!(clamp.contains("vw"));
    }

    #[test]
    fn test_fluid_type_values() {
        let clamp = fluid_type(16.0, 32.0, 400.0, 1200.0);
        // min = 16/16 = 1rem, max = 32/16 = 2rem
        assert!(clamp.contains("1.0000rem"));
        assert!(clamp.contains("2.0000rem"));
    }

    #[test]
    fn test_aspect_ratio_widescreen() {
        let ar = AspectRatio::widescreen();
        assert!((ar.ratio() - 16.0 / 9.0).abs() < 1e-10);
        assert_eq!(ar.css_value(), "16 / 9");
    }

    #[test]
    fn test_aspect_ratio_square() {
        let ar = AspectRatio::square();
        assert!((ar.ratio() - 1.0).abs() < 1e-10);
        assert_eq!(ar.padding_bottom_percent(), 100.0);
    }

    #[test]
    fn test_aspect_ratio_dimensions() {
        let ar = AspectRatio::widescreen();
        let h = ar.height_for_width(1920.0);
        assert!((h - 1080.0).abs() < 0.01);

        let w = ar.width_for_height(1080.0);
        assert!((w - 1920.0).abs() < 0.01);
    }

    #[test]
    fn test_aspect_ratio_display() {
        let ar = AspectRatio::standard();
        assert_eq!(format!("{ar}"), "4 / 3");
    }

    #[test]
    fn test_viewport_units() {
        assert_eq!(vp(100.0, ViewportUnit::Vh), "100vh");
        assert_eq!(vp(50.0, ViewportUnit::Vw), "50vw");
        assert_eq!(vp(100.0, ViewportUnit::Dvh), "100dvh");
        assert_eq!(vp(100.0, ViewportUnit::Svh), "100svh");
    }

    #[test]
    fn test_viewport_unit_display() {
        assert_eq!(ViewportUnit::Vmin.to_string(), "vmin");
        assert_eq!(ViewportUnit::Vmax.to_string(), "vmax");
        assert_eq!(ViewportUnit::Lvw.to_string(), "lvw");
        assert_eq!(ViewportUnit::Lvh.to_string(), "lvh");
    }

    #[test]
    fn test_full_height_safe() {
        let h = full_height_safe();
        assert!(h.contains("100vh"));
        assert!(h.contains("100dvh"));
    }

    #[test]
    fn test_media_feature_resolution() {
        let f = MediaFeature::Resolution(192.0);
        assert_eq!(f.to_string(), "(min-resolution: 192dpi)");
    }

    #[test]
    fn test_media_feature_aspect_ratio() {
        let f = MediaFeature::AspectRatio(16, 9);
        assert_eq!(f.to_string(), "(aspect-ratio: 16/9)");
    }

    #[test]
    fn test_media_feature_custom() {
        let f = MediaFeature::Custom("(scripting: enabled)".into());
        assert_eq!(f.to_string(), "(scripting: enabled)");
    }

    #[test]
    fn test_prefers_contrast() {
        let mq = MediaQuery::new().feature(MediaFeature::PrefersContrast(ContrastPref::More));
        assert!(mq.build().contains("prefers-contrast: more"));
    }

    #[test]
    fn test_breakpoint_get_missing() {
        let bs = BreakpointSystem::bootstrap();
        assert!(bs.get("nonexistent").is_none());
        assert!(bs.up("nonexistent").is_none());
    }

    #[test]
    fn test_aspect_ratio_ultrawide() {
        let ar = AspectRatio::ultrawide();
        assert!((ar.ratio() - 21.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_fluid_space_alias() {
        let a = fluid_type(8.0, 16.0, 320.0, 1200.0);
        let b = fluid_space(8.0, 16.0, 320.0, 1200.0);
        assert_eq!(a, b);
    }
}
