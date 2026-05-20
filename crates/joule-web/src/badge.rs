//! Badge component: content (text/number), variants (primary/secondary/
//! success/warning/danger), dot mode (no content), max count with "99+"
//! overflow, positioning (top-right, etc.), pulse animation flag, visibility toggle.

// ── Variant ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadgeVariant {
    Primary,
    Secondary,
    Success,
    Warning,
    Danger,
}

impl BadgeVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Secondary => "secondary",
            Self::Success => "success",
            Self::Warning => "warning",
            Self::Danger => "danger",
        }
    }

    pub fn color(self) -> &'static str {
        match self {
            Self::Primary => "#3498db",
            Self::Secondary => "#95a5a6",
            Self::Success => "#2ecc71",
            Self::Warning => "#f39c12",
            Self::Danger => "#e74c3c",
        }
    }
}

// ── Position ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadgePosition {
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
    Inline,
}

impl BadgePosition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TopRight => "top-right",
            Self::TopLeft => "top-left",
            Self::BottomRight => "bottom-right",
            Self::BottomLeft => "bottom-left",
            Self::Inline => "inline",
        }
    }

    pub fn css_properties(self) -> (&'static str, &'static str) {
        match self {
            Self::TopRight => ("top:0;right:0", "translate(50%,-50%)"),
            Self::TopLeft => ("top:0;left:0", "translate(-50%,-50%)"),
            Self::BottomRight => ("bottom:0;right:0", "translate(50%,50%)"),
            Self::BottomLeft => ("bottom:0;left:0", "translate(-50%,50%)"),
            Self::Inline => ("position:relative", "none"),
        }
    }
}

// ── Badge content ──────────────────────────────────────────────────

/// What's displayed inside the badge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BadgeContent {
    /// A numeric count.
    Count(u32),
    /// Arbitrary text.
    Text(String),
    /// Dot mode — no visible content, just the dot.
    Dot,
}

// ── Badge ──────────────────────────────────────────────────────────

/// A badge/notification indicator component.
#[derive(Debug, Clone)]
pub struct Badge {
    pub content: BadgeContent,
    pub variant: BadgeVariant,
    pub position: BadgePosition,
    /// Maximum count before showing "N+".
    pub max_count: Option<u32>,
    /// Show a pulse animation.
    pub pulse: bool,
    /// Whether the badge is visible.
    pub visible: bool,
}

impl Badge {
    pub fn count(n: u32) -> Self {
        Self {
            content: BadgeContent::Count(n),
            variant: BadgeVariant::Danger,
            position: BadgePosition::TopRight,
            max_count: Some(99),
            pulse: false,
            visible: true,
        }
    }

    pub fn text(t: impl Into<String>) -> Self {
        Self {
            content: BadgeContent::Text(t.into()),
            variant: BadgeVariant::Primary,
            position: BadgePosition::TopRight,
            max_count: None,
            pulse: false,
            visible: true,
        }
    }

    pub fn dot() -> Self {
        Self {
            content: BadgeContent::Dot,
            variant: BadgeVariant::Danger,
            position: BadgePosition::TopRight,
            max_count: None,
            pulse: false,
            visible: true,
        }
    }

    pub fn variant(mut self, v: BadgeVariant) -> Self {
        self.variant = v;
        self
    }

    pub fn position(mut self, p: BadgePosition) -> Self {
        self.position = p;
        self
    }

    pub fn max_count(mut self, m: u32) -> Self {
        self.max_count = Some(m);
        self
    }

    pub fn pulse(mut self, p: bool) -> Self {
        self.pulse = p;
        self
    }

    pub fn visible(mut self, v: bool) -> Self {
        self.visible = v;
        self
    }

    /// The display text, applying max_count overflow formatting.
    pub fn display_text(&self) -> String {
        match &self.content {
            BadgeContent::Count(n) => {
                if let Some(max) = self.max_count {
                    if *n > max {
                        return format!("{}+", max);
                    }
                }
                n.to_string()
            }
            BadgeContent::Text(t) => t.clone(),
            BadgeContent::Dot => String::new(),
        }
    }

    /// Whether the badge should be hidden (zero count hides by default).
    pub fn should_show(&self) -> bool {
        if !self.visible {
            return false;
        }
        match &self.content {
            BadgeContent::Count(0) => false,
            _ => true,
        }
    }

    /// Render the badge to HTML.
    pub fn render(&self) -> String {
        if !self.should_show() {
            return String::new();
        }

        let variant_class = self.variant.as_str();
        let position_class = self.position.as_str();
        let is_dot = matches!(self.content, BadgeContent::Dot);
        let dot_class = if is_dot { " badge--dot" } else { "" };
        let pulse_class = if self.pulse { " badge--pulse" } else { "" };
        let text = self.display_text();
        let (pos_css, transform) = self.position.css_properties();

        format!(
            "<span class=\"badge badge--{} badge--{}{}{}\" \
             style=\"{};transform:{};background:{}\">{}</span>",
            variant_class,
            position_class,
            dot_class,
            pulse_class,
            pos_css,
            transform,
            self.variant.color(),
            text,
        )
    }

    /// Render a badge wrapping a child element.
    pub fn render_with_child(&self, child_html: &str) -> String {
        let badge_html = self.render();
        format!(
            "<div class=\"badge-container\" style=\"position:relative;display:inline-block\">\
             {}{}</div>",
            child_html, badge_html,
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_display() {
        let b = Badge::count(42);
        assert_eq!(b.display_text(), "42");
    }

    #[test]
    fn test_count_overflow() {
        let b = Badge::count(150).max_count(99);
        assert_eq!(b.display_text(), "99+");
    }

    #[test]
    fn test_count_at_max() {
        let b = Badge::count(99).max_count(99);
        assert_eq!(b.display_text(), "99");
    }

    #[test]
    fn test_text_display() {
        let b = Badge::text("New");
        assert_eq!(b.display_text(), "New");
    }

    #[test]
    fn test_dot_display() {
        let b = Badge::dot();
        assert_eq!(b.display_text(), "");
    }

    #[test]
    fn test_zero_count_hidden() {
        let b = Badge::count(0);
        assert!(!b.should_show());
        assert!(b.render().is_empty());
    }

    #[test]
    fn test_visibility_toggle() {
        let b = Badge::count(5).visible(false);
        assert!(!b.should_show());
        assert!(b.render().is_empty());
    }

    #[test]
    fn test_dot_is_shown() {
        let b = Badge::dot();
        assert!(b.should_show());
        let html = b.render();
        assert!(html.contains("badge--dot"));
    }

    #[test]
    fn test_pulse_class() {
        let b = Badge::count(3).pulse(true);
        let html = b.render();
        assert!(html.contains("badge--pulse"));
    }

    #[test]
    fn test_variant_colors() {
        assert_eq!(BadgeVariant::Primary.color(), "#3498db");
        assert_eq!(BadgeVariant::Danger.color(), "#e74c3c");
        assert_eq!(BadgeVariant::Success.color(), "#2ecc71");
    }

    #[test]
    fn test_position_class() {
        let b = Badge::count(1).position(BadgePosition::BottomLeft);
        let html = b.render();
        assert!(html.contains("badge--bottom-left"));
    }

    #[test]
    fn test_render_contains_background() {
        let b = Badge::count(5).variant(BadgeVariant::Success);
        let html = b.render();
        assert!(html.contains("background:#2ecc71"));
    }

    #[test]
    fn test_render_with_child() {
        let b = Badge::count(3);
        let html = b.render_with_child("<button>Inbox</button>");
        assert!(html.contains("<button>Inbox</button>"));
        assert!(html.contains("badge-container"));
        assert!(html.contains("3"));
    }

    #[test]
    fn test_custom_max_count() {
        let b = Badge::count(500).max_count(9);
        assert_eq!(b.display_text(), "9+");
    }
}
