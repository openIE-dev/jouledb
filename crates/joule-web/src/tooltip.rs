//! Tooltip system: show/hide with delay, placement (top/right/bottom/left
//! with auto-flip), arrow positioning, hover/focus/click triggers, dismissal
//! on scroll, tooltip content (text/rich), max width.

// ── Placement ──────────────────────────────────────────────────────

/// Preferred placement for a tooltip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TooltipPlacement {
    Top,
    Right,
    Bottom,
    Left,
}

impl TooltipPlacement {
    /// The opposite placement for auto-flip.
    pub fn opposite(self) -> Self {
        match self {
            Self::Top => Self::Bottom,
            Self::Bottom => Self::Top,
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Top => "top",
            Self::Right => "right",
            Self::Bottom => "bottom",
            Self::Left => "left",
        }
    }
}

// ── Trigger mode ───────────────────────────────────────────────────

/// What triggers tooltip visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerMode {
    Hover,
    Focus,
    Click,
}

// ── Tooltip content ────────────────────────────────────────────────

/// Content rendered inside the tooltip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TooltipContent {
    Text(String),
    Rich(String), // HTML string
}

impl TooltipContent {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text(s) | Self::Rich(s) => s,
        }
    }
}

// ── Rect helper ────────────────────────────────────────────────────

/// Simple bounding rectangle.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self {
            x,
            y,
            width: w,
            height: h,
        }
    }

    pub fn right(&self) -> f64 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f64 {
        self.y + self.height
    }

    pub fn center_x(&self) -> f64 {
        self.x + self.width / 2.0
    }

    pub fn center_y(&self) -> f64 {
        self.y + self.height / 2.0
    }
}

// ── Tooltip config ─────────────────────────────────────────────────

/// Configuration for a tooltip instance.
#[derive(Debug, Clone)]
pub struct TooltipConfig {
    pub content: TooltipContent,
    pub placement: TooltipPlacement,
    pub auto_flip: bool,
    pub trigger: TriggerMode,
    /// Delay before showing (ms).
    pub show_delay_ms: u64,
    /// Delay before hiding (ms).
    pub hide_delay_ms: u64,
    /// Dismiss when the page scrolls.
    pub dismiss_on_scroll: bool,
    /// Max width in pixels.
    pub max_width: f64,
    /// Offset from anchor edge in pixels.
    pub offset: f64,
    /// Show an arrow pointing to the anchor.
    pub show_arrow: bool,
    /// Arrow size in pixels.
    pub arrow_size: f64,
}

impl Default for TooltipConfig {
    fn default() -> Self {
        Self {
            content: TooltipContent::Text(String::new()),
            placement: TooltipPlacement::Top,
            auto_flip: true,
            trigger: TriggerMode::Hover,
            show_delay_ms: 200,
            hide_delay_ms: 0,
            dismiss_on_scroll: true,
            max_width: 300.0,
            offset: 8.0,
            show_arrow: true,
            arrow_size: 6.0,
        }
    }
}

impl TooltipConfig {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content: TooltipContent::Text(text.into()),
            ..Default::default()
        }
    }

    pub fn rich(html: impl Into<String>) -> Self {
        Self {
            content: TooltipContent::Rich(html.into()),
            ..Default::default()
        }
    }

    pub fn placement(mut self, p: TooltipPlacement) -> Self {
        self.placement = p;
        self
    }

    pub fn trigger(mut self, t: TriggerMode) -> Self {
        self.trigger = t;
        self
    }

    pub fn show_delay(mut self, ms: u64) -> Self {
        self.show_delay_ms = ms;
        self
    }

    pub fn hide_delay(mut self, ms: u64) -> Self {
        self.hide_delay_ms = ms;
        self
    }

    pub fn max_width(mut self, w: f64) -> Self {
        self.max_width = w;
        self
    }

    pub fn offset(mut self, o: f64) -> Self {
        self.offset = o;
        self
    }

    pub fn arrow(mut self, show: bool) -> Self {
        self.show_arrow = show;
        self
    }
}

// ── Tooltip state ──────────────────────────────────────────────────

/// Runtime state for a tooltip.
#[derive(Debug, Clone)]
pub struct Tooltip {
    pub config: TooltipConfig,
    pub visible: bool,
    /// Resolved placement after auto-flip.
    pub resolved_placement: TooltipPlacement,
    /// Computed position (x, y) of the tooltip.
    pub position: (f64, f64),
    /// Arrow offset from tooltip edge (px).
    pub arrow_offset: f64,
    /// Timestamp (ms) when show was requested.
    pub show_requested_at: Option<u64>,
    /// Timestamp (ms) when hide was requested.
    pub hide_requested_at: Option<u64>,
}

impl Tooltip {
    pub fn new(config: TooltipConfig) -> Self {
        Self {
            resolved_placement: config.placement,
            config,
            visible: false,
            position: (0.0, 0.0),
            arrow_offset: 0.0,
            show_requested_at: None,
            hide_requested_at: None,
        }
    }

    /// Request to show the tooltip at a given time.
    pub fn request_show(&mut self, now_ms: u64) {
        self.hide_requested_at = None;
        if self.config.show_delay_ms == 0 {
            self.visible = true;
        } else {
            self.show_requested_at = Some(now_ms);
        }
    }

    /// Request to hide the tooltip at a given time.
    pub fn request_hide(&mut self, now_ms: u64) {
        self.show_requested_at = None;
        if self.config.hide_delay_ms == 0 {
            self.visible = false;
        } else {
            self.hide_requested_at = Some(now_ms);
        }
    }

    /// Tick: check if pending show/hide should resolve.
    pub fn tick(&mut self, now_ms: u64) {
        if let Some(t) = self.show_requested_at {
            if now_ms >= t + self.config.show_delay_ms {
                self.visible = true;
                self.show_requested_at = None;
            }
        }
        if let Some(t) = self.hide_requested_at {
            if now_ms >= t + self.config.hide_delay_ms {
                self.visible = false;
                self.hide_requested_at = None;
            }
        }
    }

    /// Handle scroll event.
    pub fn handle_scroll(&mut self) {
        if self.config.dismiss_on_scroll && self.visible {
            self.visible = false;
            self.show_requested_at = None;
        }
    }

    /// Compute position relative to an anchor rect, within a viewport.
    pub fn compute_position(
        &mut self,
        anchor: Rect,
        tooltip_size: (f64, f64),
        viewport: Rect,
    ) {
        let (tw, th) = tooltip_size;
        let offset = self.config.offset;

        let mut placement = self.config.placement;

        // Auto-flip if needed
        if self.config.auto_flip {
            let fits = |p: TooltipPlacement| -> bool {
                match p {
                    TooltipPlacement::Top => anchor.y - th - offset >= viewport.y,
                    TooltipPlacement::Bottom => {
                        anchor.bottom() + th + offset <= viewport.bottom()
                    }
                    TooltipPlacement::Left => anchor.x - tw - offset >= viewport.x,
                    TooltipPlacement::Right => {
                        anchor.right() + tw + offset <= viewport.right()
                    }
                }
            };

            if !fits(placement) && fits(placement.opposite()) {
                placement = placement.opposite();
            }
        }

        let (x, y) = match placement {
            TooltipPlacement::Top => (anchor.center_x() - tw / 2.0, anchor.y - th - offset),
            TooltipPlacement::Bottom => {
                (anchor.center_x() - tw / 2.0, anchor.bottom() + offset)
            }
            TooltipPlacement::Left => (anchor.x - tw - offset, anchor.center_y() - th / 2.0),
            TooltipPlacement::Right => {
                (anchor.right() + offset, anchor.center_y() - th / 2.0)
            }
        };

        self.resolved_placement = placement;
        self.position = (x, y);

        // Arrow offset: centered on the anchor
        self.arrow_offset = match placement {
            TooltipPlacement::Top | TooltipPlacement::Bottom => {
                (anchor.center_x() - x).max(self.config.arrow_size)
            }
            TooltipPlacement::Left | TooltipPlacement::Right => {
                (anchor.center_y() - y).max(self.config.arrow_size)
            }
        };
    }

    /// Render tooltip to HTML.
    pub fn render(&self) -> String {
        if !self.visible {
            return String::new();
        }

        let placement_str = self.resolved_placement.as_str();
        let (x, y) = self.position;
        let max_w = self.config.max_width;

        let content_html = match &self.config.content {
            TooltipContent::Text(t) => format!("<span>{}</span>", t),
            TooltipContent::Rich(h) => h.clone(),
        };

        let arrow_html = if self.config.show_arrow {
            format!(
                "<div class=\"tooltip-arrow tooltip-arrow--{}\" style=\"--arrow-offset:{}px\"></div>",
                placement_str, self.arrow_offset
            )
        } else {
            String::new()
        };

        format!(
            "<div class=\"tooltip tooltip--{}\" role=\"tooltip\" \
             style=\"left:{}px;top:{}px;max-width:{}px\">\
             {}{}</div>",
            placement_str, x, y, max_w, content_html, arrow_html
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_show_immediately_when_no_delay() {
        let mut tt = Tooltip::new(TooltipConfig::new("Hello").show_delay(0));
        tt.request_show(100);
        assert!(tt.visible);
    }

    #[test]
    fn test_show_with_delay() {
        let mut tt = Tooltip::new(TooltipConfig::new("Hello").show_delay(200));
        tt.request_show(100);
        assert!(!tt.visible);
        tt.tick(200);
        assert!(!tt.visible);
        tt.tick(300);
        assert!(tt.visible);
    }

    #[test]
    fn test_hide_immediately() {
        let mut tt = Tooltip::new(TooltipConfig::new("Hello").show_delay(0).hide_delay(0));
        tt.request_show(0);
        assert!(tt.visible);
        tt.request_hide(100);
        assert!(!tt.visible);
    }

    #[test]
    fn test_hide_with_delay() {
        let mut tt = Tooltip::new(TooltipConfig::new("Hello").show_delay(0).hide_delay(100));
        tt.request_show(0);
        tt.request_hide(50);
        assert!(tt.visible);
        tt.tick(149);
        assert!(tt.visible);
        tt.tick(150);
        assert!(!tt.visible);
    }

    #[test]
    fn test_dismiss_on_scroll() {
        let mut tt = Tooltip::new(TooltipConfig::new("Hello").show_delay(0));
        tt.request_show(0);
        assert!(tt.visible);
        tt.handle_scroll();
        assert!(!tt.visible);
    }

    #[test]
    fn test_no_dismiss_on_scroll_when_disabled() {
        let mut tt = Tooltip::new(TooltipConfig::new("Hello").show_delay(0));
        tt.config.dismiss_on_scroll = false;
        tt.request_show(0);
        tt.handle_scroll();
        assert!(tt.visible);
    }

    #[test]
    fn test_placement_opposite() {
        assert_eq!(TooltipPlacement::Top.opposite(), TooltipPlacement::Bottom);
        assert_eq!(TooltipPlacement::Left.opposite(), TooltipPlacement::Right);
    }

    #[test]
    fn test_auto_flip() {
        let mut tt = Tooltip::new(
            TooltipConfig::new("Tip").placement(TooltipPlacement::Top),
        );
        let anchor = Rect::new(100.0, 10.0, 50.0, 20.0);
        let viewport = Rect::new(0.0, 0.0, 500.0, 500.0);
        // Not enough room above (anchor.y=10, tooltip height=30+8=38 > 10)
        tt.compute_position(anchor, (80.0, 30.0), viewport);
        assert_eq!(tt.resolved_placement, TooltipPlacement::Bottom);
    }

    #[test]
    fn test_position_bottom() {
        let mut tt = Tooltip::new(
            TooltipConfig::new("Tip").placement(TooltipPlacement::Bottom).offset(10.0),
        );
        let anchor = Rect::new(100.0, 50.0, 40.0, 20.0);
        let viewport = Rect::new(0.0, 0.0, 800.0, 600.0);
        tt.compute_position(anchor, (60.0, 24.0), viewport);
        assert_eq!(tt.resolved_placement, TooltipPlacement::Bottom);
        // x = anchor.center_x - tw/2 = 120 - 30 = 90
        assert!((tt.position.0 - 90.0).abs() < 0.1);
        // y = anchor.bottom + offset = 70 + 10 = 80
        assert!((tt.position.1 - 80.0).abs() < 0.1);
    }

    #[test]
    fn test_render_hidden_empty() {
        let tt = Tooltip::new(TooltipConfig::new("Hello"));
        assert!(tt.render().is_empty());
    }

    #[test]
    fn test_render_visible_html() {
        let mut tt = Tooltip::new(TooltipConfig::new("Hello").show_delay(0));
        tt.request_show(0);
        tt.position = (100.0, 50.0);
        let html = tt.render();
        assert!(html.contains("role=\"tooltip\""));
        assert!(html.contains("Hello"));
        assert!(html.contains("tooltip-arrow"));
    }

    #[test]
    fn test_render_no_arrow() {
        let mut tt = Tooltip::new(TooltipConfig::new("Hello").show_delay(0).arrow(false));
        tt.request_show(0);
        let html = tt.render();
        assert!(!html.contains("tooltip-arrow"));
    }

    #[test]
    fn test_rich_content() {
        let cfg = TooltipConfig::rich("<b>Bold</b>");
        match &cfg.content {
            TooltipContent::Rich(h) => assert_eq!(h, "<b>Bold</b>"),
            _ => panic!("Expected rich content"),
        }
    }
}
