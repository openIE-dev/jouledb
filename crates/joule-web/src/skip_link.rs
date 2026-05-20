//! Skip navigation links: landmark registration, skip link generation,
//! landmark ordering, dynamic updates, section IDs, and keyboard shortcuts.
//!
//! Pure data — no browser dependency. Maintains a registry of page landmarks
//! and generates skip navigation HTML for accessibility.

use std::collections::HashMap;

// ── Landmark Role ─────────────────────────────────────────────

/// Standard ARIA landmark roles for skip links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LandmarkRole {
    Main,
    Navigation,
    Complementary,
    ContentInfo,
    Banner,
    Search,
    Form,
    Region,
}

impl LandmarkRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            LandmarkRole::Main => "main",
            LandmarkRole::Navigation => "navigation",
            LandmarkRole::Complementary => "complementary",
            LandmarkRole::ContentInfo => "contentinfo",
            LandmarkRole::Banner => "banner",
            LandmarkRole::Search => "search",
            LandmarkRole::Form => "form",
            LandmarkRole::Region => "region",
        }
    }

    /// Default display label for skip links.
    pub fn default_label(&self) -> &'static str {
        match self {
            LandmarkRole::Main => "Skip to main content",
            LandmarkRole::Navigation => "Skip to navigation",
            LandmarkRole::Complementary => "Skip to sidebar",
            LandmarkRole::ContentInfo => "Skip to footer",
            LandmarkRole::Banner => "Skip to header",
            LandmarkRole::Search => "Skip to search",
            LandmarkRole::Form => "Skip to form",
            LandmarkRole::Region => "Skip to region",
        }
    }

    /// Default sort order (lower = higher priority in skip link list).
    pub fn default_order(&self) -> u32 {
        match self {
            LandmarkRole::Main => 0,
            LandmarkRole::Navigation => 1,
            LandmarkRole::Search => 2,
            LandmarkRole::Form => 3,
            LandmarkRole::Complementary => 4,
            LandmarkRole::ContentInfo => 5,
            LandmarkRole::Banner => 6,
            LandmarkRole::Region => 7,
        }
    }
}

// ── Landmark ──────────────────────────────────────────────────

/// A registered landmark on the page.
#[derive(Debug, Clone)]
pub struct Landmark {
    /// Unique element ID (used as href target).
    pub id: String,
    pub role: LandmarkRole,
    /// Human-readable label for the skip link.
    pub label: String,
    /// Sort order override. `None` uses `role.default_order()`.
    pub order: Option<u32>,
    /// Optional keyboard shortcut (e.g. "Alt+1").
    pub shortcut: Option<String>,
    /// Whether this landmark is currently visible/active.
    pub visible: bool,
}

impl Landmark {
    pub fn new(id: &str, role: LandmarkRole) -> Self {
        Self {
            id: id.into(),
            role,
            label: role.default_label().into(),
            order: None,
            shortcut: None,
            visible: true,
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.into();
        self
    }

    pub fn with_order(mut self, order: u32) -> Self {
        self.order = Some(order);
        self
    }

    pub fn with_shortcut(mut self, shortcut: &str) -> Self {
        self.shortcut = Some(shortcut.into());
        self
    }

    fn effective_order(&self) -> u32 {
        self.order.unwrap_or_else(|| self.role.default_order())
    }
}

// ── Skip Link ─────────────────────────────────────────────────

/// A rendered skip link.
#[derive(Debug, Clone)]
pub struct SkipLink {
    pub href: String,
    pub label: String,
    pub shortcut: Option<String>,
}

impl SkipLink {
    /// Render as an HTML anchor.
    pub fn to_html(&self) -> String {
        let mut attrs = vec![
            format!("href=\"#{}\"", self.href),
            "class=\"skip-link\"".into(),
        ];
        if let Some(sc) = &self.shortcut {
            attrs.push(format!("accesskey=\"{}\"", sc));
        }
        format!("<a {}>{}</a>", attrs.join(" "), self.label)
    }
}

// ── Keyboard Shortcut ─────────────────────────────────────────

/// A keyboard shortcut mapping to a landmark.
#[derive(Debug, Clone)]
pub struct LandmarkShortcut {
    pub key: String,
    pub modifiers: Vec<String>,
    pub landmark_id: String,
    pub description: String,
}

impl LandmarkShortcut {
    pub fn new(key: &str, modifiers: &[&str], landmark_id: &str, description: &str) -> Self {
        Self {
            key: key.into(),
            modifiers: modifiers.iter().map(|s| s.to_string()).collect(),
            landmark_id: landmark_id.into(),
            description: description.into(),
        }
    }

    /// Human-readable shortcut string.
    pub fn display(&self) -> String {
        let mut parts: Vec<&str> = self.modifiers.iter().map(|s| s.as_str()).collect();
        parts.push(&self.key);
        parts.join("+")
    }
}

// ── Landmark Registry ─────────────────────────────────────────

/// Registry of page landmarks for skip navigation.
#[derive(Debug, Default)]
pub struct LandmarkRegistry {
    landmarks: Vec<Landmark>,
    shortcuts: HashMap<String, LandmarkShortcut>,
}

impl LandmarkRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a landmark.
    pub fn register(&mut self, landmark: Landmark) {
        // Replace if same ID exists.
        self.landmarks.retain(|l| l.id != landmark.id);
        self.landmarks.push(landmark);
    }

    /// Unregister a landmark by ID.
    pub fn unregister(&mut self, id: &str) {
        self.landmarks.retain(|l| l.id != id);
        self.shortcuts.retain(|_, sc| sc.landmark_id != id);
    }

    /// Update visibility of a landmark.
    pub fn set_visible(&mut self, id: &str, visible: bool) {
        if let Some(lm) = self.landmarks.iter_mut().find(|l| l.id == id) {
            lm.visible = visible;
        }
    }

    /// Update the label of a landmark.
    pub fn set_label(&mut self, id: &str, label: &str) {
        if let Some(lm) = self.landmarks.iter_mut().find(|l| l.id == id) {
            lm.label = label.into();
        }
    }

    /// Register a keyboard shortcut for a landmark.
    pub fn add_shortcut(&mut self, shortcut: LandmarkShortcut) {
        let key = shortcut.display();
        self.shortcuts.insert(key, shortcut);
    }

    /// Get a landmark by ID.
    pub fn get(&self, id: &str) -> Option<&Landmark> {
        self.landmarks.iter().find(|l| l.id == id)
    }

    /// Get landmarks sorted by order, visible only.
    pub fn sorted_landmarks(&self) -> Vec<&Landmark> {
        let mut visible: Vec<_> = self.landmarks.iter().filter(|l| l.visible).collect();
        visible.sort_by_key(|l| l.effective_order());
        visible
    }

    /// Generate skip links for all visible landmarks in order.
    pub fn skip_links(&self) -> Vec<SkipLink> {
        self.sorted_landmarks()
            .iter()
            .map(|lm| SkipLink {
                href: lm.id.clone(),
                label: lm.label.clone(),
                shortcut: lm.shortcut.clone(),
            })
            .collect()
    }

    /// Render the full skip navigation bar as HTML.
    pub fn render_skip_nav(&self) -> String {
        let links = self.skip_links();
        if links.is_empty() {
            return String::new();
        }
        let inner: Vec<_> = links.iter().map(|l| l.to_html()).collect();
        format!("<nav aria-label=\"Skip links\" class=\"skip-nav\">{}</nav>", inner.join(""))
    }

    /// Look up which landmark a keyboard shortcut maps to.
    pub fn resolve_shortcut(&self, key_combo: &str) -> Option<&str> {
        self.shortcuts.get(key_combo).map(|sc| sc.landmark_id.as_str())
    }

    /// Total registered landmarks.
    pub fn count(&self) -> usize {
        self.landmarks.len()
    }

    /// All registered shortcuts.
    pub fn shortcuts(&self) -> Vec<&LandmarkShortcut> {
        self.shortcuts.values().collect()
    }

    /// Generate a section ID from a label (slugify).
    pub fn generate_section_id(label: &str) -> String {
        label
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_registry() -> LandmarkRegistry {
        let mut reg = LandmarkRegistry::new();
        reg.register(Landmark::new("main-content", LandmarkRole::Main));
        reg.register(Landmark::new("nav-primary", LandmarkRole::Navigation));
        reg.register(Landmark::new("search-box", LandmarkRole::Search));
        reg.register(Landmark::new("footer", LandmarkRole::ContentInfo));
        reg
    }

    #[test]
    fn register_and_count() {
        let reg = sample_registry();
        assert_eq!(reg.count(), 4);
    }

    #[test]
    fn sorted_by_default_order() {
        let reg = sample_registry();
        let sorted = reg.sorted_landmarks();
        assert_eq!(sorted[0].id, "main-content");
        assert_eq!(sorted[1].id, "nav-primary");
        assert_eq!(sorted[2].id, "search-box");
        assert_eq!(sorted[3].id, "footer");
    }

    #[test]
    fn custom_order_override() {
        let mut reg = LandmarkRegistry::new();
        reg.register(Landmark::new("footer", LandmarkRole::ContentInfo).with_order(0));
        reg.register(Landmark::new("main", LandmarkRole::Main).with_order(1));
        let sorted = reg.sorted_landmarks();
        assert_eq!(sorted[0].id, "footer");
    }

    #[test]
    fn skip_links_generation() {
        let reg = sample_registry();
        let links = reg.skip_links();
        assert_eq!(links.len(), 4);
        assert_eq!(links[0].label, "Skip to main content");
    }

    #[test]
    fn skip_link_html() {
        let link = SkipLink {
            href: "main".into(),
            label: "Skip to main content".into(),
            shortcut: None,
        };
        let html = link.to_html();
        assert!(html.contains("href=\"#main\""));
        assert!(html.contains("Skip to main content"));
    }

    #[test]
    fn skip_link_with_accesskey() {
        let link = SkipLink {
            href: "nav".into(),
            label: "Navigation".into(),
            shortcut: Some("n".into()),
        };
        let html = link.to_html();
        assert!(html.contains("accesskey=\"n\""));
    }

    #[test]
    fn visibility_filtering() {
        let mut reg = sample_registry();
        reg.set_visible("search-box", false);
        let sorted = reg.sorted_landmarks();
        assert_eq!(sorted.len(), 3);
        assert!(sorted.iter().all(|l| l.id != "search-box"));
    }

    #[test]
    fn dynamic_label_update() {
        let mut reg = sample_registry();
        reg.set_label("nav-primary", "Skip to site navigation");
        let lm = reg.get("nav-primary").unwrap();
        assert_eq!(lm.label, "Skip to site navigation");
    }

    #[test]
    fn unregister() {
        let mut reg = sample_registry();
        reg.unregister("footer");
        assert_eq!(reg.count(), 3);
        assert!(reg.get("footer").is_none());
    }

    #[test]
    fn render_skip_nav() {
        let mut reg = LandmarkRegistry::new();
        reg.register(Landmark::new("main", LandmarkRole::Main));
        let html = reg.render_skip_nav();
        assert!(html.contains("<nav"));
        assert!(html.contains("aria-label=\"Skip links\""));
        assert!(html.contains("href=\"#main\""));
    }

    #[test]
    fn keyboard_shortcut() {
        let mut reg = sample_registry();
        reg.add_shortcut(LandmarkShortcut::new("1", &["Alt"], "main-content", "Go to main"));
        let resolved = reg.resolve_shortcut("Alt+1");
        assert_eq!(resolved, Some("main-content"));
    }

    #[test]
    fn generate_section_id() {
        assert_eq!(LandmarkRegistry::generate_section_id("My Cool Section!"), "my-cool-section");
        assert_eq!(LandmarkRegistry::generate_section_id("hello world"), "hello-world");
    }

    #[test]
    fn replace_duplicate_id() {
        let mut reg = LandmarkRegistry::new();
        reg.register(Landmark::new("main", LandmarkRole::Main).with_label("Old"));
        reg.register(Landmark::new("main", LandmarkRole::Main).with_label("New"));
        assert_eq!(reg.count(), 1);
        assert_eq!(reg.get("main").unwrap().label, "New");
    }

    #[test]
    fn landmark_role_as_str() {
        assert_eq!(LandmarkRole::Main.as_str(), "main");
        assert_eq!(LandmarkRole::Navigation.as_str(), "navigation");
        assert_eq!(LandmarkRole::Complementary.as_str(), "complementary");
    }
}
