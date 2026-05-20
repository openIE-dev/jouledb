//! Popover manager: anchor positioning, placement with collision detection,
//! offset from anchor, arrow element, click-outside dismissal, portal rendering
//! flag, interactive content, trigger modes (hover/click/focus).

// ── Placement ──────────────────────────────────────────────────────

/// Placement side for the popover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopoverPlacement {
    Top,
    TopStart,
    TopEnd,
    Bottom,
    BottomStart,
    BottomEnd,
    Left,
    LeftStart,
    LeftEnd,
    Right,
    RightStart,
    RightEnd,
}

impl PopoverPlacement {
    /// Primary side of this placement.
    pub fn side(self) -> Side {
        match self {
            Self::Top | Self::TopStart | Self::TopEnd => Side::Top,
            Self::Bottom | Self::BottomStart | Self::BottomEnd => Side::Bottom,
            Self::Left | Self::LeftStart | Self::LeftEnd => Side::Left,
            Self::Right | Self::RightStart | Self::RightEnd => Side::Right,
        }
    }

    /// Opposite placement (same alignment, flipped side).
    pub fn opposite(self) -> Self {
        match self {
            Self::Top => Self::Bottom,
            Self::TopStart => Self::BottomStart,
            Self::TopEnd => Self::BottomEnd,
            Self::Bottom => Self::Top,
            Self::BottomStart => Self::TopStart,
            Self::BottomEnd => Self::TopEnd,
            Self::Left => Self::Right,
            Self::LeftStart => Self::RightStart,
            Self::LeftEnd => Self::RightEnd,
            Self::Right => Self::Left,
            Self::RightStart => Self::LeftStart,
            Self::RightEnd => Self::LeftEnd,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Top => "top",
            Self::TopStart => "top-start",
            Self::TopEnd => "top-end",
            Self::Bottom => "bottom",
            Self::BottomStart => "bottom-start",
            Self::BottomEnd => "bottom-end",
            Self::Left => "left",
            Self::LeftStart => "left-start",
            Self::LeftEnd => "left-end",
            Self::Right => "right",
            Self::RightStart => "right-start",
            Self::RightEnd => "right-end",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Top,
    Bottom,
    Left,
    Right,
}

// ── Trigger mode ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopoverTrigger {
    Click,
    Hover,
    Focus,
}

// ── Rect ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, width: w, height: h }
    }
    pub fn right(&self) -> f64 { self.x + self.width }
    pub fn bottom(&self) -> f64 { self.y + self.height }
    pub fn center_x(&self) -> f64 { self.x + self.width / 2.0 }
    pub fn center_y(&self) -> f64 { self.y + self.height / 2.0 }
}

// ── Popover config ─────────────────────────────────────────────────

/// Configuration for a popover.
#[derive(Debug, Clone)]
pub struct PopoverConfig {
    pub id: String,
    pub content: String,
    pub placement: PopoverPlacement,
    pub trigger: PopoverTrigger,
    pub offset: f64,
    pub show_arrow: bool,
    pub arrow_size: f64,
    pub portal: bool,
    pub interactive: bool,
    pub dismiss_on_click_outside: bool,
    pub class: Option<String>,
}

impl PopoverConfig {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            placement: PopoverPlacement::Bottom,
            trigger: PopoverTrigger::Click,
            offset: 8.0,
            show_arrow: true,
            arrow_size: 8.0,
            portal: false,
            interactive: true,
            dismiss_on_click_outside: true,
            class: None,
        }
    }

    pub fn placement(mut self, p: PopoverPlacement) -> Self {
        self.placement = p;
        self
    }

    pub fn trigger(mut self, t: PopoverTrigger) -> Self {
        self.trigger = t;
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

    pub fn portal(mut self, enabled: bool) -> Self {
        self.portal = enabled;
        self
    }

    pub fn interactive(mut self, enabled: bool) -> Self {
        self.interactive = enabled;
        self
    }

    pub fn dismiss_on_click_outside(mut self, enabled: bool) -> Self {
        self.dismiss_on_click_outside = enabled;
        self
    }
}

// ── Popover ────────────────────────────────────────────────────────

/// A positioned popover with collision detection.
#[derive(Debug)]
pub struct Popover {
    pub config: PopoverConfig,
    pub open: bool,
    pub resolved_placement: PopoverPlacement,
    pub position: (f64, f64),
    pub arrow_offset: f64,
}

impl Popover {
    pub fn new(config: PopoverConfig) -> Self {
        let placement = config.placement;
        Self {
            config,
            open: false,
            resolved_placement: placement,
            position: (0.0, 0.0),
            arrow_offset: 0.0,
        }
    }

    /// Toggle open/close.
    pub fn toggle(&mut self) {
        self.open = !self.open;
    }

    /// Show the popover.
    pub fn show(&mut self) {
        self.open = true;
    }

    /// Hide the popover.
    pub fn hide(&mut self) {
        self.open = false;
    }

    /// Handle click outside: dismiss if configured.
    pub fn handle_click_outside(&mut self) -> bool {
        if self.open && self.config.dismiss_on_click_outside {
            self.open = false;
            true
        } else {
            false
        }
    }

    /// Compute position with collision detection against viewport.
    pub fn compute_position(
        &mut self,
        anchor: Rect,
        popover_size: (f64, f64),
        viewport: Rect,
    ) {
        let (pw, ph) = popover_size;
        let offset = self.config.offset;

        let mut placement = self.config.placement;

        // Check if preferred placement fits
        if !Self::fits(placement.side(), anchor, pw, ph, offset, viewport) {
            let opp = placement.opposite();
            if Self::fits(opp.side(), anchor, pw, ph, offset, viewport) {
                placement = opp;
            }
            // else keep preferred (best effort)
        }

        let (x, y) = Self::compute_xy(placement, anchor, pw, ph, offset);

        self.resolved_placement = placement;
        self.position = (x, y);

        // Arrow offset (centered on anchor edge)
        self.arrow_offset = match placement.side() {
            Side::Top | Side::Bottom => (anchor.center_x() - x).max(self.config.arrow_size),
            Side::Left | Side::Right => (anchor.center_y() - y).max(self.config.arrow_size),
        };
    }

    fn fits(
        side: Side,
        anchor: Rect,
        pw: f64,
        ph: f64,
        offset: f64,
        viewport: Rect,
    ) -> bool {
        match side {
            Side::Top => anchor.y - ph - offset >= viewport.y,
            Side::Bottom => anchor.bottom() + ph + offset <= viewport.bottom(),
            Side::Left => anchor.x - pw - offset >= viewport.x,
            Side::Right => anchor.right() + pw + offset <= viewport.right(),
        }
    }

    fn compute_xy(
        placement: PopoverPlacement,
        anchor: Rect,
        pw: f64,
        ph: f64,
        offset: f64,
    ) -> (f64, f64) {
        match placement {
            PopoverPlacement::Top => (anchor.center_x() - pw / 2.0, anchor.y - ph - offset),
            PopoverPlacement::TopStart => (anchor.x, anchor.y - ph - offset),
            PopoverPlacement::TopEnd => (anchor.right() - pw, anchor.y - ph - offset),
            PopoverPlacement::Bottom => (anchor.center_x() - pw / 2.0, anchor.bottom() + offset),
            PopoverPlacement::BottomStart => (anchor.x, anchor.bottom() + offset),
            PopoverPlacement::BottomEnd => (anchor.right() - pw, anchor.bottom() + offset),
            PopoverPlacement::Left => (anchor.x - pw - offset, anchor.center_y() - ph / 2.0),
            PopoverPlacement::LeftStart => (anchor.x - pw - offset, anchor.y),
            PopoverPlacement::LeftEnd => (anchor.x - pw - offset, anchor.bottom() - ph),
            PopoverPlacement::Right => (anchor.right() + offset, anchor.center_y() - ph / 2.0),
            PopoverPlacement::RightStart => (anchor.right() + offset, anchor.y),
            PopoverPlacement::RightEnd => (anchor.right() + offset, anchor.bottom() - ph),
        }
    }

    /// Render to HTML.
    pub fn render(&self) -> String {
        if !self.open {
            return String::new();
        }

        let (x, y) = self.position;
        let placement_str = self.resolved_placement.as_str();
        let class = self.config.class.as_deref().unwrap_or("popover");
        let portal_attr = if self.config.portal { " data-portal=\"true\"" } else { "" };

        let arrow_html = if self.config.show_arrow {
            format!(
                "<div class=\"popover-arrow popover-arrow--{}\" style=\"--arrow-offset:{}px\"></div>",
                placement_str, self.arrow_offset
            )
        } else {
            String::new()
        };

        format!(
            "<div class=\"{} popover--{}\" role=\"dialog\" \
             style=\"left:{}px;top:{}px\"{}>\
             <div class=\"popover-content\">{}</div>{}</div>",
            class, placement_str, x, y, portal_attr,
            self.config.content, arrow_html,
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn viewport() -> Rect {
        Rect::new(0.0, 0.0, 800.0, 600.0)
    }

    #[test]
    fn test_toggle() {
        let mut pop = Popover::new(PopoverConfig::new("p1", "Content"));
        assert!(!pop.open);
        pop.toggle();
        assert!(pop.open);
        pop.toggle();
        assert!(!pop.open);
    }

    #[test]
    fn test_show_hide() {
        let mut pop = Popover::new(PopoverConfig::new("p1", "Content"));
        pop.show();
        assert!(pop.open);
        pop.hide();
        assert!(!pop.open);
    }

    #[test]
    fn test_click_outside_dismissal() {
        let mut pop = Popover::new(PopoverConfig::new("p1", "Content"));
        pop.show();
        assert!(pop.handle_click_outside());
        assert!(!pop.open);
    }

    #[test]
    fn test_click_outside_disabled() {
        let mut pop = Popover::new(
            PopoverConfig::new("p1", "Content").dismiss_on_click_outside(false),
        );
        pop.show();
        assert!(!pop.handle_click_outside());
        assert!(pop.open);
    }

    #[test]
    fn test_position_bottom() {
        let mut pop = Popover::new(
            PopoverConfig::new("p1", "Content")
                .placement(PopoverPlacement::Bottom)
                .offset(10.0),
        );
        let anchor = Rect::new(100.0, 50.0, 40.0, 20.0);
        pop.compute_position(anchor, (80.0, 30.0), viewport());
        assert_eq!(pop.resolved_placement, PopoverPlacement::Bottom);
        // y = anchor.bottom + offset = 70 + 10 = 80
        assert!((pop.position.1 - 80.0).abs() < 0.1);
    }

    #[test]
    fn test_collision_flip() {
        let mut pop = Popover::new(
            PopoverConfig::new("p1", "Content")
                .placement(PopoverPlacement::Top)
                .offset(8.0),
        );
        // Anchor near the top of viewport — no room above
        let anchor = Rect::new(100.0, 5.0, 40.0, 20.0);
        pop.compute_position(anchor, (80.0, 30.0), viewport());
        assert_eq!(pop.resolved_placement.side(), Side::Bottom);
    }

    #[test]
    fn test_placement_opposite() {
        assert_eq!(PopoverPlacement::Top.opposite(), PopoverPlacement::Bottom);
        assert_eq!(PopoverPlacement::LeftStart.opposite(), PopoverPlacement::RightStart);
    }

    #[test]
    fn test_bottom_start_alignment() {
        let mut pop = Popover::new(
            PopoverConfig::new("p1", "Content")
                .placement(PopoverPlacement::BottomStart)
                .offset(0.0),
        );
        let anchor = Rect::new(100.0, 50.0, 40.0, 20.0);
        pop.compute_position(anchor, (80.0, 30.0), viewport());
        // x should align with anchor left
        assert!((pop.position.0 - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_render_hidden_empty() {
        let pop = Popover::new(PopoverConfig::new("p1", "Content"));
        assert!(pop.render().is_empty());
    }

    #[test]
    fn test_render_visible() {
        let mut pop = Popover::new(PopoverConfig::new("p1", "Hello"));
        pop.show();
        pop.position = (50.0, 100.0);
        let html = pop.render();
        assert!(html.contains("role=\"dialog\""));
        assert!(html.contains("Hello"));
        assert!(html.contains("popover-arrow"));
    }

    #[test]
    fn test_portal_attribute() {
        let mut pop = Popover::new(
            PopoverConfig::new("p1", "Content").portal(true),
        );
        pop.show();
        let html = pop.render();
        assert!(html.contains("data-portal=\"true\""));
    }

    #[test]
    fn test_no_arrow() {
        let mut pop = Popover::new(
            PopoverConfig::new("p1", "Content").arrow(false),
        );
        pop.show();
        let html = pop.render();
        assert!(!html.contains("popover-arrow"));
    }
}
