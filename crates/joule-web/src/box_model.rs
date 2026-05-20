//! CSS box model computation: edges, dimensions, rects, margin collapsing,
//! auto-margin centering, box-sizing adjustment, and min/max clamping.

// ── Edge ─────────────────────────────────────────────────────────

/// The four-side edge values (top, right, bottom, left) in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Edge {
    pub fn new(top: f64, right: f64, bottom: f64, left: f64) -> Self {
        Self { top, right, bottom, left }
    }

    pub fn uniform(value: f64) -> Self {
        Self { top: value, right: value, bottom: value, left: value }
    }

    pub fn zero() -> Self {
        Self::uniform(0.0)
    }

    /// Horizontal sum (left + right).
    pub fn horizontal(&self) -> f64 {
        self.left + self.right
    }

    /// Vertical sum (top + bottom).
    pub fn vertical(&self) -> f64 {
        self.top + self.bottom
    }

    /// Parse a CSS shorthand value like "10px", "10px 20px", "10px 20px 30px",
    /// or "10px 20px 30px 40px" into an Edge.
    pub fn parse_shorthand(value: &str) -> Option<Self> {
        let parts: Vec<f64> = value
            .split_whitespace()
            .filter_map(|s| parse_px(s))
            .collect();

        match parts.len() {
            1 => Some(Self::uniform(parts[0])),
            2 => Some(Self::new(parts[0], parts[1], parts[0], parts[1])),
            3 => Some(Self::new(parts[0], parts[1], parts[2], parts[1])),
            4 => Some(Self::new(parts[0], parts[1], parts[2], parts[3])),
            _ => None,
        }
    }
}

fn parse_px(s: &str) -> Option<f64> {
    let s = s.trim();
    if s == "0" {
        return Some(0.0);
    }
    let num_str = s.strip_suffix("px").unwrap_or(s);
    num_str.parse::<f64>().ok()
}

// ── Rect ─────────────────────────────────────────────────────────

/// A rectangle defined by position and size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width, height }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, width: 0.0, height: 0.0 }
    }

    pub fn right(&self) -> f64 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f64 {
        self.y + self.height
    }
}

// ── BoxSizing ────────────────────────────────────────────────────

/// CSS box-sizing property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxSizing {
    ContentBox,
    BorderBox,
}

// ── BoxDimensions ────────────────────────────────────────────────

/// Complete box model dimensions for a CSS element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxDimensions {
    pub content_width: f64,
    pub content_height: f64,
    pub padding: Edge,
    pub border: Edge,
    pub margin: Edge,
}

impl BoxDimensions {
    pub fn new(width: f64, height: f64) -> Self {
        Self {
            content_width: width,
            content_height: height,
            padding: Edge::zero(),
            border: Edge::zero(),
            margin: Edge::zero(),
        }
    }

    pub fn with_padding(mut self, padding: Edge) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_border(mut self, border: Edge) -> Self {
        self.border = border;
        self
    }

    pub fn with_margin(mut self, margin: Edge) -> Self {
        self.margin = margin;
        self
    }

    /// The content box rect (at origin 0,0 offset by margin+border+padding).
    pub fn content_box(&self) -> Rect {
        let x = self.margin.left + self.border.left + self.padding.left;
        let y = self.margin.top + self.border.top + self.padding.top;
        Rect::new(x, y, self.content_width, self.content_height)
    }

    /// The padding box (content + padding).
    pub fn padding_box(&self) -> Rect {
        let x = self.margin.left + self.border.left;
        let y = self.margin.top + self.border.top;
        Rect::new(
            x,
            y,
            self.content_width + self.padding.horizontal(),
            self.content_height + self.padding.vertical(),
        )
    }

    /// The border box (content + padding + border).
    pub fn border_box(&self) -> Rect {
        let x = self.margin.left;
        let y = self.margin.top;
        Rect::new(
            x,
            y,
            self.content_width + self.padding.horizontal() + self.border.horizontal(),
            self.content_height + self.padding.vertical() + self.border.vertical(),
        )
    }

    /// The margin box (content + padding + border + margin).
    pub fn margin_box(&self) -> Rect {
        Rect::new(
            0.0,
            0.0,
            self.content_width
                + self.padding.horizontal()
                + self.border.horizontal()
                + self.margin.horizontal(),
            self.content_height
                + self.padding.vertical()
                + self.border.vertical()
                + self.margin.vertical(),
        )
    }

    /// Adjust dimensions for box-sizing. When `border-box`, the given width/height
    /// includes padding and border, so content dimensions are smaller.
    pub fn apply_box_sizing(mut self, sizing: BoxSizing) -> Self {
        if sizing == BoxSizing::BorderBox {
            self.content_width = (self.content_width
                - self.padding.horizontal()
                - self.border.horizontal())
            .max(0.0);
            self.content_height = (self.content_height
                - self.padding.vertical()
                - self.border.vertical())
            .max(0.0);
        }
        self
    }

    /// Clamp content dimensions to min/max constraints.
    pub fn clamp(mut self, min_w: Option<f64>, max_w: Option<f64>, min_h: Option<f64>, max_h: Option<f64>) -> Self {
        if let Some(min) = min_w {
            self.content_width = self.content_width.max(min);
        }
        if let Some(max) = max_w {
            self.content_width = self.content_width.min(max);
        }
        if let Some(min) = min_h {
            self.content_height = self.content_height.max(min);
        }
        if let Some(max) = max_h {
            self.content_height = self.content_height.min(max);
        }
        self
    }

    /// Center horizontally via auto margins within a containing width.
    pub fn center_horizontal(mut self, container_width: f64) -> Self {
        let border_box_w =
            self.content_width + self.padding.horizontal() + self.border.horizontal();
        let available = (container_width - border_box_w).max(0.0);
        let half = available / 2.0;
        self.margin.left = half;
        self.margin.right = half;
        self
    }
}

// ── Margin Collapsing ────────────────────────────────────────────

/// Collapse adjacent vertical margins: the larger margin wins.
pub fn collapse_margins(margin_a_bottom: f64, margin_b_top: f64) -> f64 {
    if margin_a_bottom >= 0.0 && margin_b_top >= 0.0 {
        margin_a_bottom.max(margin_b_top)
    } else if margin_a_bottom < 0.0 && margin_b_top < 0.0 {
        margin_a_bottom.min(margin_b_top)
    } else {
        margin_a_bottom + margin_b_top
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_parse_single() {
        let e = Edge::parse_shorthand("10px").unwrap();
        assert_eq!(e, Edge::uniform(10.0));
    }

    #[test]
    fn test_edge_parse_two() {
        let e = Edge::parse_shorthand("10px 20px").unwrap();
        assert_eq!(e, Edge::new(10.0, 20.0, 10.0, 20.0));
    }

    #[test]
    fn test_edge_parse_three() {
        let e = Edge::parse_shorthand("10px 20px 30px").unwrap();
        assert_eq!(e, Edge::new(10.0, 20.0, 30.0, 20.0));
    }

    #[test]
    fn test_edge_parse_four() {
        let e = Edge::parse_shorthand("1px 2px 3px 4px").unwrap();
        assert_eq!(e, Edge::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_content_box() {
        let b = BoxDimensions::new(100.0, 50.0)
            .with_padding(Edge::uniform(10.0))
            .with_border(Edge::uniform(2.0))
            .with_margin(Edge::uniform(5.0));

        let cb = b.content_box();
        assert_eq!(cb.x, 5.0 + 2.0 + 10.0);
        assert_eq!(cb.y, 5.0 + 2.0 + 10.0);
        assert_eq!(cb.width, 100.0);
        assert_eq!(cb.height, 50.0);
    }

    #[test]
    fn test_padding_box() {
        let b = BoxDimensions::new(100.0, 50.0)
            .with_padding(Edge::uniform(10.0))
            .with_border(Edge::uniform(2.0))
            .with_margin(Edge::uniform(5.0));

        let pb = b.padding_box();
        assert_eq!(pb.width, 120.0);
        assert_eq!(pb.height, 70.0);
    }

    #[test]
    fn test_border_box() {
        let b = BoxDimensions::new(100.0, 50.0)
            .with_padding(Edge::uniform(10.0))
            .with_border(Edge::uniform(2.0))
            .with_margin(Edge::uniform(5.0));

        let bb = b.border_box();
        assert_eq!(bb.width, 124.0);
        assert_eq!(bb.height, 74.0);
    }

    #[test]
    fn test_margin_box() {
        let b = BoxDimensions::new(100.0, 50.0)
            .with_padding(Edge::uniform(10.0))
            .with_border(Edge::uniform(2.0))
            .with_margin(Edge::uniform(5.0));

        let mb = b.margin_box();
        assert_eq!(mb.width, 134.0);
        assert_eq!(mb.height, 84.0);
    }

    #[test]
    fn test_box_sizing_border_box() {
        let b = BoxDimensions::new(200.0, 100.0)
            .with_padding(Edge::uniform(20.0))
            .with_border(Edge::uniform(5.0))
            .apply_box_sizing(BoxSizing::BorderBox);

        assert_eq!(b.content_width, 200.0 - 40.0 - 10.0); // 150
        assert_eq!(b.content_height, 100.0 - 40.0 - 10.0); // 50
    }

    #[test]
    fn test_box_sizing_content_box_noop() {
        let b = BoxDimensions::new(200.0, 100.0)
            .with_padding(Edge::uniform(20.0))
            .apply_box_sizing(BoxSizing::ContentBox);

        assert_eq!(b.content_width, 200.0);
    }

    #[test]
    fn test_clamp() {
        let b = BoxDimensions::new(50.0, 300.0)
            .clamp(Some(100.0), Some(400.0), Some(0.0), Some(200.0));

        assert_eq!(b.content_width, 100.0);
        assert_eq!(b.content_height, 200.0);
    }

    #[test]
    fn test_center_horizontal() {
        let b = BoxDimensions::new(200.0, 100.0)
            .with_padding(Edge::zero())
            .with_border(Edge::zero())
            .center_horizontal(800.0);

        assert_eq!(b.margin.left, 300.0);
        assert_eq!(b.margin.right, 300.0);
    }

    #[test]
    fn test_margin_collapse_positive() {
        assert_eq!(collapse_margins(20.0, 30.0), 30.0);
    }

    #[test]
    fn test_margin_collapse_negative() {
        assert_eq!(collapse_margins(-10.0, -20.0), -20.0);
    }

    #[test]
    fn test_margin_collapse_mixed() {
        assert_eq!(collapse_margins(20.0, -5.0), 15.0);
    }

    #[test]
    fn test_edge_horizontal_vertical() {
        let e = Edge::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(e.horizontal(), 60.0);
        assert_eq!(e.vertical(), 40.0);
    }

    #[test]
    fn test_border_box_no_negative() {
        let b = BoxDimensions::new(10.0, 10.0)
            .with_padding(Edge::uniform(20.0))
            .with_border(Edge::uniform(5.0))
            .apply_box_sizing(BoxSizing::BorderBox);

        assert_eq!(b.content_width, 0.0);
        assert_eq!(b.content_height, 0.0);
    }
}
