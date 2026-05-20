//! PWA Web App Manifest — JSON manifest builder for Progressive Web Apps.
//!
//! Generates W3C-compliant `manifest.json` with icons, display modes,
//! theme/background colors, shortcuts, share targets, and related apps.
//! Pure Rust — no web-sys, serializes to JSON via serde.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Display Mode ─────────────────────────────────────────────────

/// Display mode for the PWA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DisplayMode {
    Fullscreen,
    Standalone,
    MinimalUi,
    Browser,
    WindowControlsOverlay,
}

impl fmt::Display for DisplayMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fullscreen => write!(f, "fullscreen"),
            Self::Standalone => write!(f, "standalone"),
            Self::MinimalUi => write!(f, "minimal-ui"),
            Self::Browser => write!(f, "browser"),
            Self::WindowControlsOverlay => write!(f, "window-controls-overlay"),
        }
    }
}

// ── Orientation ──────────────────────────────────────────────────

/// Screen orientation lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Orientation {
    Any,
    Natural,
    Landscape,
    LandscapePrimary,
    LandscapeSecondary,
    Portrait,
    PortraitPrimary,
    PortraitSecondary,
}

// ── Icon ─────────────────────────────────────────────────────────

/// A single icon entry in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Icon {
    pub src: String,
    pub sizes: String,
    #[serde(rename = "type")]
    pub mime_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
}

impl Icon {
    /// Create icon with src, size string (e.g. "192x192"), and MIME type.
    pub fn new(src: impl Into<String>, sizes: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self { src: src.into(), sizes: sizes.into(), mime_type: mime_type.into(), purpose: None }
    }

    /// Set icon purpose (e.g. "any", "maskable", "monochrome").
    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = Some(purpose.into());
        self
    }

    /// Common PNG icon at a given square size.
    pub fn png(src: impl Into<String>, size: u32) -> Self {
        Self::new(src, format!("{size}x{size}"), "image/png")
    }

    /// SVG icon (any size).
    pub fn svg(src: impl Into<String>) -> Self {
        Self::new(src, "any", "image/svg+xml")
    }
}

// ── Shortcut ─────────────────────────────────────────────────────

/// App shortcut (context menu on long-press / right-click).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shortcut {
    pub name: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub icons: Vec<Icon>,
}

impl Shortcut {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            short_name: None,
            description: None,
            icons: Vec::new(),
        }
    }

    pub fn with_short_name(mut self, sn: impl Into<String>) -> Self {
        self.short_name = Some(sn.into());
        self
    }

    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }

    pub fn with_icon(mut self, icon: Icon) -> Self {
        self.icons.push(icon);
        self
    }
}

// ── Share Target ─────────────────────────────────────────────────

/// Web Share Target configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareTarget {
    pub action: String,
    pub method: ShareMethod,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enctype: Option<String>,
    pub params: ShareParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ShareMethod {
    Get,
    Post,
}

/// Parameter mapping for share target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub files: Vec<ShareFile>,
}

impl ShareParams {
    pub fn new() -> Self {
        Self { title: None, text: None, url: None, files: Vec::new() }
    }
}

impl Default for ShareParams {
    fn default() -> Self { Self::new() }
}

/// File type accepted by share target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareFile {
    pub name: String,
    pub accept: Vec<String>,
}

// ── Related Application ──────────────────────────────────────────

/// Reference to a related native application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelatedApp {
    pub platform: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

// ── Protocol Handler ─────────────────────────────────────────────

/// Custom protocol handler registration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolHandler {
    pub protocol: String,
    pub url: String,
}

// ── Manifest ─────────────────────────────────────────────────────

/// Complete W3C Web App Manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<DisplayMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation: Option<Orientation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_color: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub icons: Vec<Icon>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub shortcuts: Vec<Shortcut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share_target: Option<ShareTarget>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub related_applications: Vec<RelatedApp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefer_related_applications: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub protocol_handlers: Vec<ProtocolHandler>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub categories: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dir: Option<String>,
}

impl Manifest {
    /// Create a new manifest with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            short_name: None,
            description: None,
            start_url: None,
            scope: None,
            display: None,
            orientation: None,
            theme_color: None,
            background_color: None,
            icons: Vec::new(),
            shortcuts: Vec::new(),
            share_target: None,
            related_applications: Vec::new(),
            prefer_related_applications: None,
            protocol_handlers: Vec::new(),
            id: None,
            categories: Vec::new(),
            lang: None,
            dir: None,
        }
    }

    pub fn short_name(mut self, n: impl Into<String>) -> Self { self.short_name = Some(n.into()); self }
    pub fn description(mut self, d: impl Into<String>) -> Self { self.description = Some(d.into()); self }
    pub fn start_url(mut self, u: impl Into<String>) -> Self { self.start_url = Some(u.into()); self }
    pub fn scope(mut self, s: impl Into<String>) -> Self { self.scope = Some(s.into()); self }
    pub fn display(mut self, d: DisplayMode) -> Self { self.display = Some(d); self }
    pub fn orientation(mut self, o: Orientation) -> Self { self.orientation = Some(o); self }
    pub fn theme_color(mut self, c: impl Into<String>) -> Self { self.theme_color = Some(c.into()); self }
    pub fn background_color(mut self, c: impl Into<String>) -> Self { self.background_color = Some(c.into()); self }
    pub fn id(mut self, i: impl Into<String>) -> Self { self.id = Some(i.into()); self }
    pub fn lang(mut self, l: impl Into<String>) -> Self { self.lang = Some(l.into()); self }
    pub fn dir(mut self, d: impl Into<String>) -> Self { self.dir = Some(d.into()); self }

    pub fn add_icon(mut self, icon: Icon) -> Self { self.icons.push(icon); self }
    pub fn add_shortcut(mut self, s: Shortcut) -> Self { self.shortcuts.push(s); self }
    pub fn add_related_app(mut self, app: RelatedApp) -> Self { self.related_applications.push(app); self }
    pub fn add_protocol_handler(mut self, p: ProtocolHandler) -> Self { self.protocol_handlers.push(p); self }
    pub fn add_category(mut self, c: impl Into<String>) -> Self { self.categories.push(c.into()); self }

    pub fn share_target(mut self, st: ShareTarget) -> Self { self.share_target = Some(st); self }
    pub fn prefer_related(mut self, p: bool) -> Self { self.prefer_related_applications = Some(p); self }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("manifest serializes to JSON")
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Generate HTML `<link>` tag for this manifest.
    pub fn link_tag(&self, href: &str) -> String {
        format!(r#"<link rel="manifest" href="{href}">"#)
    }

    /// Generate HTML meta tags for theme-color and apple-mobile-web-app.
    pub fn meta_tags(&self) -> Vec<String> {
        let mut tags = Vec::new();
        if let Some(ref tc) = self.theme_color {
            tags.push(format!(r#"<meta name="theme-color" content="{tc}">"#));
        }
        if self.display == Some(DisplayMode::Standalone) || self.display == Some(DisplayMode::Fullscreen) {
            tags.push(r#"<meta name="apple-mobile-web-app-capable" content="yes">"#.to_string());
        }
        if let Some(ref sn) = self.short_name {
            tags.push(format!(r#"<meta name="apple-mobile-web-app-title" content="{sn}">"#));
        } else {
            tags.push(format!(r#"<meta name="apple-mobile-web-app-title" content="{}">"#, self.name));
        }
        tags
    }

    /// Validate the manifest for common issues.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.name.is_empty() {
            warnings.push("name is empty".into());
        }
        if self.name.len() > 45 {
            warnings.push(format!("name is {} chars (recommended max 45)", self.name.len()));
        }
        if let Some(ref sn) = self.short_name {
            if sn.len() > 12 {
                warnings.push(format!("short_name is {} chars (recommended max 12)", sn.len()));
            }
        }
        if self.icons.is_empty() {
            warnings.push("no icons defined".into());
        } else {
            let has_192 = self.icons.iter().any(|i| i.sizes.contains("192"));
            let has_512 = self.icons.iter().any(|i| i.sizes.contains("512"));
            if !has_192 { warnings.push("missing 192x192 icon".into()); }
            if !has_512 { warnings.push("missing 512x512 icon".into()); }
            let has_maskable = self.icons.iter().any(|i| {
                i.purpose.as_deref().map_or(false, |p| p.contains("maskable"))
            });
            if !has_maskable { warnings.push("no maskable icon for adaptive icon support".into()); }
        }
        if self.start_url.is_none() {
            warnings.push("start_url not set".into());
        }
        if self.display.is_none() {
            warnings.push("display mode not set (defaults to browser)".into());
        }
        if self.theme_color.is_none() {
            warnings.push("theme_color not set".into());
        }
        warnings
    }

    /// Count of defined icons.
    pub fn icon_count(&self) -> usize { self.icons.len() }

    /// Count of defined shortcuts.
    pub fn shortcut_count(&self) -> usize { self.shortcuts.len() }
}

// ── Standard icon sets ───────────────────────────────────────────

/// Generate a standard set of PNG icons at common sizes.
pub fn standard_icon_set(base_path: &str) -> Vec<Icon> {
    let sizes: &[u32] = &[48, 72, 96, 128, 144, 152, 192, 384, 512];
    sizes.iter().map(|s| {
        Icon::png(format!("{base_path}/icon-{s}x{s}.png"), *s)
    }).collect()
}

/// Generate maskable + any icon set.
pub fn dual_purpose_icons(base_path: &str) -> Vec<Icon> {
    vec![
        Icon::png(format!("{base_path}/icon-192x192.png"), 192).with_purpose("any"),
        Icon::png(format!("{base_path}/icon-512x512.png"), 512).with_purpose("any"),
        Icon::png(format!("{base_path}/maskable-192x192.png"), 192).with_purpose("maskable"),
        Icon::png(format!("{base_path}/maskable-512x512.png"), 512).with_purpose("maskable"),
    ]
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_manifest() {
        let m = Manifest::new("My App")
            .short_name("App")
            .start_url("/")
            .display(DisplayMode::Standalone)
            .theme_color("#1a1a2e")
            .background_color("#ffffff");

        assert_eq!(m.name, "My App");
        assert_eq!(m.short_name.as_deref(), Some("App"));
        assert_eq!(m.display, Some(DisplayMode::Standalone));
    }

    #[test]
    fn test_json_roundtrip() {
        let m = Manifest::new("Test PWA")
            .short_name("Test")
            .start_url("/app")
            .scope("/")
            .display(DisplayMode::Standalone)
            .theme_color("#000000")
            .background_color("#ffffff")
            .add_icon(Icon::png("/icon-192.png", 192))
            .add_icon(Icon::png("/icon-512.png", 512));

        let json = m.to_json();
        let parsed = Manifest::from_json(&json).unwrap();
        assert_eq!(parsed.name, "Test PWA");
        assert_eq!(parsed.icons.len(), 2);
        assert_eq!(parsed.display, Some(DisplayMode::Standalone));
    }

    #[test]
    fn test_display_mode_serialize() {
        let m = Manifest::new("X").display(DisplayMode::MinimalUi);
        let json = m.to_json();
        assert!(json.contains("minimal-ui"));
    }

    #[test]
    fn test_icon_constructors() {
        let png = Icon::png("/icon.png", 192);
        assert_eq!(png.sizes, "192x192");
        assert_eq!(png.mime_type, "image/png");

        let svg = Icon::svg("/icon.svg");
        assert_eq!(svg.sizes, "any");
        assert_eq!(svg.mime_type, "image/svg+xml");

        let maskable = Icon::png("/m.png", 512).with_purpose("maskable");
        assert_eq!(maskable.purpose.as_deref(), Some("maskable"));
    }

    #[test]
    fn test_shortcuts() {
        let s = Shortcut::new("New Doc", "/new")
            .with_short_name("New")
            .with_description("Create a new document")
            .with_icon(Icon::png("/new-icon.png", 96));

        assert_eq!(s.name, "New Doc");
        assert_eq!(s.url, "/new");
        assert_eq!(s.icons.len(), 1);

        let m = Manifest::new("App").add_shortcut(s);
        let json = m.to_json();
        assert!(json.contains("New Doc"));
        assert!(json.contains("/new"));
    }

    #[test]
    fn test_share_target() {
        let st = ShareTarget {
            action: "/share".into(),
            method: ShareMethod::Post,
            enctype: Some("multipart/form-data".into()),
            params: ShareParams {
                title: Some("title".into()),
                text: Some("text".into()),
                url: Some("url".into()),
                files: vec![ShareFile {
                    name: "media".into(),
                    accept: vec!["image/*".into(), "video/*".into()],
                }],
            },
        };

        let m = Manifest::new("Share App").share_target(st);
        let json = m.to_json();
        assert!(json.contains("/share"));
        assert!(json.contains("POST"));
        assert!(json.contains("multipart/form-data"));
    }

    #[test]
    fn test_validation_warnings() {
        let m = Manifest::new("");
        let w = m.validate();
        assert!(w.iter().any(|s| s.contains("name is empty")));
        assert!(w.iter().any(|s| s.contains("no icons")));
        assert!(w.iter().any(|s| s.contains("start_url")));
    }

    #[test]
    fn test_validation_good_manifest() {
        let m = Manifest::new("Good App")
            .start_url("/")
            .display(DisplayMode::Standalone)
            .theme_color("#fff")
            .add_icon(Icon::png("/i-192.png", 192))
            .add_icon(Icon::png("/i-512.png", 512).with_purpose("maskable"));

        let w = m.validate();
        assert!(!w.iter().any(|s| s.contains("name is empty")));
        assert!(!w.iter().any(|s| s.contains("start_url")));
    }

    #[test]
    fn test_validation_long_short_name() {
        let m = Manifest::new("A").short_name("VeryLongShortName");
        let w = m.validate();
        assert!(w.iter().any(|s| s.contains("short_name")));
    }

    #[test]
    fn test_meta_tags_standalone() {
        let m = Manifest::new("App")
            .display(DisplayMode::Standalone)
            .theme_color("#123456");

        let tags = m.meta_tags();
        assert!(tags.iter().any(|t| t.contains("theme-color")));
        assert!(tags.iter().any(|t| t.contains("apple-mobile-web-app-capable")));
    }

    #[test]
    fn test_meta_tags_browser_mode() {
        let m = Manifest::new("App").display(DisplayMode::Browser);
        let tags = m.meta_tags();
        assert!(!tags.iter().any(|t| t.contains("apple-mobile-web-app-capable")));
    }

    #[test]
    fn test_link_tag() {
        let m = Manifest::new("App");
        let link = m.link_tag("/manifest.json");
        assert_eq!(link, r#"<link rel="manifest" href="/manifest.json">"#);
    }

    #[test]
    fn test_standard_icon_set() {
        let icons = standard_icon_set("/icons");
        assert_eq!(icons.len(), 9);
        assert!(icons.iter().any(|i| i.sizes == "192x192"));
        assert!(icons.iter().any(|i| i.sizes == "512x512"));
        assert!(icons[0].src.starts_with("/icons/"));
    }

    #[test]
    fn test_dual_purpose_icons() {
        let icons = dual_purpose_icons("/img");
        assert_eq!(icons.len(), 4);
        let maskable_count = icons.iter().filter(|i| {
            i.purpose.as_deref() == Some("maskable")
        }).count();
        assert_eq!(maskable_count, 2);
    }

    #[test]
    fn test_related_apps() {
        let m = Manifest::new("App")
            .add_related_app(RelatedApp {
                platform: "play".into(),
                url: "https://play.google.com/store/apps/details?id=com.example".into(),
                id: Some("com.example".into()),
            })
            .prefer_related(true);

        let json = m.to_json();
        assert!(json.contains("play"));
        assert!(json.contains("prefer_related_applications"));
    }

    #[test]
    fn test_protocol_handlers() {
        let m = Manifest::new("App")
            .add_protocol_handler(ProtocolHandler {
                protocol: "web+myapp".into(),
                url: "/handle?url=%s".into(),
            });

        assert_eq!(m.protocol_handlers.len(), 1);
        assert_eq!(m.protocol_handlers[0].protocol, "web+myapp");
    }

    #[test]
    fn test_categories_and_lang() {
        let m = Manifest::new("App")
            .add_category("productivity")
            .add_category("utilities")
            .lang("en-US")
            .dir("ltr");

        assert_eq!(m.categories.len(), 2);
        assert_eq!(m.lang.as_deref(), Some("en-US"));
        assert_eq!(m.dir.as_deref(), Some("ltr"));
    }

    #[test]
    fn test_orientation_serialize() {
        let m = Manifest::new("Game").orientation(Orientation::LandscapePrimary);
        let json = m.to_json();
        assert!(json.contains("landscape-primary"));
    }

    #[test]
    fn test_icon_count_and_shortcut_count() {
        let m = Manifest::new("App")
            .add_icon(Icon::png("/a.png", 192))
            .add_icon(Icon::png("/b.png", 512))
            .add_shortcut(Shortcut::new("A", "/a"));

        assert_eq!(m.icon_count(), 2);
        assert_eq!(m.shortcut_count(), 1);
    }

    #[test]
    fn test_manifest_id() {
        let m = Manifest::new("App").id("com.example.app");
        let json = m.to_json();
        assert!(json.contains("com.example.app"));
    }

    #[test]
    fn test_empty_optional_fields_omitted() {
        let m = Manifest::new("Minimal");
        let json = m.to_json();
        assert!(!json.contains("short_name"));
        assert!(!json.contains("icons"));
        assert!(!json.contains("shortcuts"));
    }

    #[test]
    fn test_share_params_default() {
        let sp = ShareParams::default();
        assert!(sp.title.is_none());
        assert!(sp.text.is_none());
        assert!(sp.url.is_none());
        assert!(sp.files.is_empty());
    }

    #[test]
    fn test_display_mode_display_trait() {
        assert_eq!(DisplayMode::Fullscreen.to_string(), "fullscreen");
        assert_eq!(DisplayMode::Standalone.to_string(), "standalone");
        assert_eq!(DisplayMode::MinimalUi.to_string(), "minimal-ui");
        assert_eq!(DisplayMode::Browser.to_string(), "browser");
        assert_eq!(DisplayMode::WindowControlsOverlay.to_string(), "window-controls-overlay");
    }

    #[test]
    fn test_full_manifest_json_fields() {
        let m = Manifest::new("Full App")
            .short_name("Full")
            .description("A complete test app")
            .start_url("/start")
            .scope("/")
            .display(DisplayMode::Standalone)
            .orientation(Orientation::Portrait)
            .theme_color("#ff0000")
            .background_color("#00ff00")
            .id("full-app-id")
            .lang("en")
            .dir("ltr")
            .add_icon(Icon::png("/icon.png", 192))
            .add_category("test");

        let json = m.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["name"], "Full App");
        assert_eq!(parsed["short_name"], "Full");
        assert_eq!(parsed["start_url"], "/start");
        assert_eq!(parsed["scope"], "/");
        assert_eq!(parsed["display"], "standalone");
        assert_eq!(parsed["orientation"], "portrait");
    }
}
