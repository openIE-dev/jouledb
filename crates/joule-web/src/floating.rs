//! Tooltip / popover positioning engine.
//!
//! Replaces Floating UI / Popper.js. Calculates optimal placement for
//! a floating element relative to a reference element, with flip, shift,
//! and arrow positioning. Pure math — no browser dependency.

// ── Placement ──────────────────────────────────────────────────

/// Desired placement of the floating element relative to the reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Top, TopStart, TopEnd,
    Bottom, BottomStart, BottomEnd,
    Left, LeftStart, LeftEnd,
    Right, RightStart, RightEnd,
}

impl Placement {
    /// Return the opposite placement (Top <-> Bottom, Left <-> Right, etc.).
    pub fn opposite(&self) -> Self {
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

    pub fn is_horizontal(&self) -> bool {
        matches!(self, Self::Left | Self::LeftStart | Self::LeftEnd
            | Self::Right | Self::RightStart | Self::RightEnd)
    }

    pub fn is_vertical(&self) -> bool {
        !self.is_horizontal()
    }
}

// ── Geometry ───────────────────────────────────────────────────

/// Axis-aligned bounding rectangle.
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

    fn right(&self) -> f64 { self.x + self.width }
    fn bottom(&self) -> f64 { self.y + self.height }
    fn center_x(&self) -> f64 { self.x + self.width / 2.0 }
    fn center_y(&self) -> f64 { self.y + self.height / 2.0 }
}

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// ── Config / Result ────────────────────────────────────────────

/// Configuration for `compute_position`.
#[derive(Debug, Clone)]
pub struct FloatingConfig {
    pub placement: Placement,
    pub offset: f64,
    pub flip: bool,
    pub shift: bool,
    pub arrow: bool,
    pub boundary: Option<Rect>,
}

impl Default for FloatingConfig {
    fn default() -> Self {
        Self {
            placement: Placement::Bottom,
            offset: 8.0,
            flip: true,
            shift: true,
            arrow: false,
            boundary: None,
        }
    }
}

/// Result of `compute_position`.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatingResult {
    pub x: f64,
    pub y: f64,
    pub placement: Placement,
    pub arrow_x: Option<f64>,
    pub arrow_y: Option<f64>,
}

// ── Core algorithm ─────────────────────────────────────────────

/// Compute the position of a floating element relative to a reference element.
pub fn compute_position(reference: &Rect, floating: &Rect, config: &FloatingConfig) -> FloatingResult {
    let mut placement = config.placement;

    // 1. Initial position.
    let (mut x, mut y) = initial_coords(reference, floating, placement, config.offset);

    // 2. Flip if the floating element overflows the boundary.
    if config.flip {
        if let Some(ref boundary) = config.boundary {
            if overflows(x, y, floating, boundary) {
                let opp = placement.opposite();
                let (ox, oy) = initial_coords(reference, floating, opp, config.offset);
                if !overflows(ox, oy, floating, boundary) {
                    placement = opp;
                    x = ox;
                    y = oy;
                }
            }
        }
    }

    // 3. Shift to stay within boundary.
    if config.shift {
        if let Some(ref boundary) = config.boundary {
            if placement.is_vertical() {
                if x < boundary.x { x = boundary.x; }
                if x + floating.width > boundary.right() {
                    x = boundary.right() - floating.width;
                }
            } else {
                if y < boundary.y { y = boundary.y; }
                if y + floating.height > boundary.bottom() {
                    y = boundary.bottom() - floating.height;
                }
            }
        }
    }

    // 4. Arrow position.
    let (arrow_x, arrow_y) = if config.arrow {
        compute_arrow(reference, floating, x, y, placement)
    } else {
        (None, None)
    };

    FloatingResult { x, y, placement, arrow_x, arrow_y }
}

fn initial_coords(reference: &Rect, floating: &Rect, placement: Placement, offset: f64) -> (f64, f64) {
    match placement {
        Placement::Top => (
            reference.center_x() - floating.width / 2.0,
            reference.y - floating.height - offset,
        ),
        Placement::TopStart => (
            reference.x,
            reference.y - floating.height - offset,
        ),
        Placement::TopEnd => (
            reference.right() - floating.width,
            reference.y - floating.height - offset,
        ),
        Placement::Bottom => (
            reference.center_x() - floating.width / 2.0,
            reference.bottom() + offset,
        ),
        Placement::BottomStart => (
            reference.x,
            reference.bottom() + offset,
        ),
        Placement::BottomEnd => (
            reference.right() - floating.width,
            reference.bottom() + offset,
        ),
        Placement::Left => (
            reference.x - floating.width - offset,
            reference.center_y() - floating.height / 2.0,
        ),
        Placement::LeftStart => (
            reference.x - floating.width - offset,
            reference.y,
        ),
        Placement::LeftEnd => (
            reference.x - floating.width - offset,
            reference.bottom() - floating.height,
        ),
        Placement::Right => (
            reference.right() + offset,
            reference.center_y() - floating.height / 2.0,
        ),
        Placement::RightStart => (
            reference.right() + offset,
            reference.y,
        ),
        Placement::RightEnd => (
            reference.right() + offset,
            reference.bottom() - floating.height,
        ),
    }
}

fn overflows(x: f64, y: f64, floating: &Rect, boundary: &Rect) -> bool {
    x < boundary.x
        || y < boundary.y
        || x + floating.width > boundary.right()
        || y + floating.height > boundary.bottom()
}

fn compute_arrow(
    reference: &Rect,
    floating: &Rect,
    fx: f64,
    fy: f64,
    placement: Placement,
) -> (Option<f64>, Option<f64>) {
    if placement.is_vertical() {
        let arrow_x = (reference.center_x() - fx).clamp(8.0, floating.width - 8.0);
        (Some(arrow_x), None)
    } else {
        let arrow_y = (reference.center_y() - fy).clamp(8.0, floating.height - 8.0);
        (None, Some(arrow_y))
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> Rect { Rect::new(100.0, 100.0, 80.0, 30.0) }
    fn tooltip() -> Rect { Rect::new(0.0, 0.0, 120.0, 40.0) }

    #[test]
    fn top_placement_positions_above() {
        let cfg = FloatingConfig { placement: Placement::Top, offset: 8.0, flip: false, shift: false, arrow: false, boundary: None };
        let r = compute_position(&reference(), &tooltip(), &cfg);
        assert!(r.y < reference().y, "floating y={} should be above ref y={}", r.y, reference().y);
        assert_eq!(r.placement, Placement::Top);
    }

    #[test]
    fn bottom_placement_positions_below() {
        let cfg = FloatingConfig { placement: Placement::Bottom, offset: 8.0, flip: false, shift: false, arrow: false, boundary: None };
        let r = compute_position(&reference(), &tooltip(), &cfg);
        assert!(r.y > reference().bottom(), "floating y={} should be below ref bottom={}", r.y, reference().bottom());
    }

    #[test]
    fn left_placement() {
        let cfg = FloatingConfig { placement: Placement::Left, offset: 8.0, flip: false, shift: false, arrow: false, boundary: None };
        let r = compute_position(&reference(), &tooltip(), &cfg);
        assert!(r.x + tooltip().width <= reference().x);
    }

    #[test]
    fn right_placement() {
        let cfg = FloatingConfig { placement: Placement::Right, offset: 8.0, flip: false, shift: false, arrow: false, boundary: None };
        let r = compute_position(&reference(), &tooltip(), &cfg);
        assert!(r.x >= reference().right());
    }

    #[test]
    fn offset_pushes_away() {
        let cfg0 = FloatingConfig { placement: Placement::Bottom, offset: 0.0, flip: false, shift: false, arrow: false, boundary: None };
        let cfg20 = FloatingConfig { placement: Placement::Bottom, offset: 20.0, flip: false, shift: false, arrow: false, boundary: None };
        let r0 = compute_position(&reference(), &tooltip(), &cfg0);
        let r20 = compute_position(&reference(), &tooltip(), &cfg20);
        assert!((r20.y - r0.y - 20.0).abs() < 1e-6);
    }

    #[test]
    fn flip_when_overflow() {
        let boundary = Rect::new(0.0, 90.0, 400.0, 300.0);
        let cfg = FloatingConfig {
            placement: Placement::Top, offset: 8.0, flip: true, shift: false, arrow: false,
            boundary: Some(boundary),
        };
        let r = compute_position(&reference(), &tooltip(), &cfg);
        assert_eq!(r.placement, Placement::Bottom);
        assert!(r.y > reference().bottom());
    }

    #[test]
    fn shift_keeps_in_bounds() {
        let boundary = Rect::new(0.0, 0.0, 180.0, 400.0);
        let cfg = FloatingConfig {
            placement: Placement::Bottom, offset: 8.0, flip: false, shift: true, arrow: false,
            boundary: Some(boundary),
        };
        let r = compute_position(&reference(), &tooltip(), &cfg);
        assert!(r.x >= boundary.x);
        assert!(r.x + tooltip().width <= boundary.right());
    }

    #[test]
    fn arrow_centered_on_reference() {
        let cfg = FloatingConfig {
            placement: Placement::Bottom, offset: 8.0, flip: false, shift: false, arrow: true, boundary: None,
        };
        let r = compute_position(&reference(), &tooltip(), &cfg);
        assert!(r.arrow_x.is_some());
        let ax = r.arrow_x.unwrap();
        let expected = reference().center_x() - r.x;
        assert!((ax - expected).abs() < 1.0, "arrow_x={ax}, expected~{expected}");
    }

    #[test]
    fn start_end_alignment() {
        let cfg_start = FloatingConfig { placement: Placement::BottomStart, offset: 0.0, flip: false, shift: false, arrow: false, boundary: None };
        let cfg_end = FloatingConfig { placement: Placement::BottomEnd, offset: 0.0, flip: false, shift: false, arrow: false, boundary: None };
        let rs = compute_position(&reference(), &tooltip(), &cfg_start);
        let re = compute_position(&reference(), &tooltip(), &cfg_end);
        assert!((rs.x - reference().x).abs() < 1e-6, "start should align left");
        assert!((re.x + tooltip().width - reference().right()).abs() < 1e-6, "end should align right");
    }

    #[test]
    fn no_boundary_no_flip() {
        let cfg = FloatingConfig {
            placement: Placement::Top, offset: 8.0, flip: true, shift: true, arrow: false, boundary: None,
        };
        let r = compute_position(&reference(), &tooltip(), &cfg);
        assert_eq!(r.placement, Placement::Top, "no boundary means no flip");
    }

    #[test]
    fn placement_opposite_round_trip() {
        for p in [Placement::Top, Placement::Bottom, Placement::Left, Placement::Right,
                   Placement::TopStart, Placement::BottomEnd, Placement::LeftEnd, Placement::RightStart] {
            assert_eq!(p.opposite().opposite(), p);
        }
    }

    #[test]
    fn horizontal_vertical() {
        assert!(Placement::Top.is_vertical());
        assert!(Placement::Bottom.is_vertical());
        assert!(Placement::Left.is_horizontal());
        assert!(Placement::Right.is_horizontal());
        assert!(!Placement::Top.is_horizontal());
        assert!(!Placement::Left.is_vertical());
    }
}
