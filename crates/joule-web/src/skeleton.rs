//! Skeleton / loading-state primitives for perceived-performance UX.
//!
//! Replaces react-loading-skeleton / react-content-loader. Renders
//! placeholder shapes as SVG with optional pulse/wave animation.
//! Pure Rust — no browser dependency.

// ── Shape ──────────────────────────────────────────────────────

/// Geometric primitive used to build a skeleton placeholder.
#[derive(Debug, Clone, PartialEq)]
pub enum SkeletonShape {
    Rectangle { width: f64, height: f64 },
    Circle { radius: f64 },
    Text { lines: usize, line_height: f64 },
    Avatar { size: f64 },
    Button { width: f64, height: f64 },
    Custom { width: f64, height: f64, border_radius: f64 },
}

// ── Animation ──────────────────────────────────────────────────

/// Animation style for skeleton placeholders.
#[derive(Debug, Clone, PartialEq)]
pub enum SkeletonAnimation {
    Pulse,
    Wave,
    None_,
}

// ── Config ─────────────────────────────────────────────────────

/// Visual configuration shared by all items in a skeleton layout.
#[derive(Debug, Clone)]
pub struct SkeletonConfig {
    pub animation: SkeletonAnimation,
    pub base_color: String,
    pub highlight_color: String,
    pub duration_ms: u64,
}

impl Default for SkeletonConfig {
    fn default() -> Self {
        Self {
            animation: SkeletonAnimation::Pulse,
            base_color: "#e0e0e0".into(),
            highlight_color: "#f0f0f0".into(),
            duration_ms: 1500,
        }
    }
}

// ── Item / Layout ──────────────────────────────────────────────

/// A single positioned skeleton shape.
#[derive(Debug, Clone)]
pub struct SkeletonItem {
    pub shape: SkeletonShape,
    pub x: f64,
    pub y: f64,
    pub delay_ms: u64,
}

/// A collection of skeleton items that renders as a single SVG.
#[derive(Debug, Clone)]
pub struct SkeletonLayout {
    pub items: Vec<SkeletonItem>,
}

impl Default for SkeletonLayout {
    fn default() -> Self { Self::new() }
}

impl SkeletonLayout {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn add(mut self, shape: SkeletonShape, x: f64, y: f64) -> Self {
        self.items.push(SkeletonItem { shape, x, y, delay_ms: 0 });
        self
    }

    pub fn add_with_delay(mut self, shape: SkeletonShape, x: f64, y: f64, delay: u64) -> Self {
        self.items.push(SkeletonItem { shape, x, y, delay_ms: delay });
        self
    }

    /// Render the skeleton layout as an SVG string.
    pub fn to_svg(&self, config: &SkeletonConfig) -> String {
        // Determine viewport size.
        let (mut max_x, mut max_y): (f64, f64) = (0.0, 0.0);
        for item in &self.items {
            let (w, h) = shape_bounds(&item.shape);
            let ex = item.x + w;
            let ey = item.y + h;
            if ex > max_x { max_x = ex; }
            if ey > max_y { max_y = ey; }
        }
        let vw = max_x.ceil().max(1.0);
        let vh = max_y.ceil().max(1.0);

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{vw}\" height=\"{vh}\" viewBox=\"0 0 {vw} {vh}\">\n"
        );

        // Defs — animation
        svg.push_str("<defs>\n");
        match config.animation {
            SkeletonAnimation::Pulse => {
                svg.push_str(&format!(
                    "<style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}.sk{{fill:{};animation:pulse {}ms ease-in-out infinite}}</style>\n",
                    config.base_color, config.duration_ms
                ));
            }
            SkeletonAnimation::Wave => {
                svg.push_str(&format!(
                    "<linearGradient id=\"wave\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\">\
                     <stop offset=\"0%\" stop-color=\"{}\"/>\
                     <stop offset=\"50%\" stop-color=\"{}\"/>\
                     <stop offset=\"100%\" stop-color=\"{}\"/>\
                     <animateTransform attributeName=\"gradientTransform\" type=\"translate\" from=\"-1 0\" to=\"1 0\" dur=\"{}ms\" repeatCount=\"indefinite\"/>\
                     </linearGradient>\n",
                    config.base_color, config.highlight_color, config.base_color, config.duration_ms
                ));
                svg.push_str("<style>.sk{fill:url(#wave)}</style>\n");
            }
            SkeletonAnimation::None_ => {
                svg.push_str(&format!(
                    "<style>.sk{{fill:{}}}</style>\n",
                    config.base_color
                ));
            }
        }
        svg.push_str("</defs>\n");

        for item in &self.items {
            let delay_attr = if item.delay_ms > 0 {
                format!(" style=\"animation-delay:{}ms\"", item.delay_ms)
            } else {
                String::new()
            };
            svg.push_str(&shape_to_svg_element(&item.shape, item.x, item.y, &delay_attr));
            svg.push('\n');
        }

        svg.push_str("</svg>");
        svg
    }

    // ── Presets ────────────────────────────────────────────────

    /// Card skeleton: avatar + title + two text lines + button.
    pub fn card_layout() -> Self {
        Self::new()
            .add(SkeletonShape::Avatar { size: 48.0 }, 16.0, 16.0)
            .add(SkeletonShape::Rectangle { width: 200.0, height: 20.0 }, 80.0, 20.0)
            .add(SkeletonShape::Text { lines: 2, line_height: 16.0 }, 16.0, 80.0)
            .add(SkeletonShape::Button { width: 100.0, height: 36.0 }, 16.0, 130.0)
    }

    /// List skeleton with `rows` identical rows.
    pub fn list_layout(rows: usize) -> Self {
        let mut layout = Self::new();
        for i in 0..rows {
            let y = 16.0 + (i as f64) * 56.0;
            layout = layout
                .add(SkeletonShape::Circle { radius: 20.0 }, 16.0, y)
                .add(SkeletonShape::Rectangle { width: 260.0, height: 16.0 }, 56.0, y + 4.0)
                .add(SkeletonShape::Rectangle { width: 180.0, height: 12.0 }, 56.0, y + 28.0);
        }
        layout
    }

    /// Article skeleton: title + hero image + text block.
    pub fn article_layout() -> Self {
        Self::new()
            .add(SkeletonShape::Rectangle { width: 320.0, height: 28.0 }, 16.0, 16.0)
            .add(SkeletonShape::Rectangle { width: 360.0, height: 200.0 }, 16.0, 60.0)
            .add(SkeletonShape::Text { lines: 4, line_height: 18.0 }, 16.0, 280.0)
    }
}

fn shape_bounds(shape: &SkeletonShape) -> (f64, f64) {
    match shape {
        SkeletonShape::Rectangle { width, height } => (*width, *height),
        SkeletonShape::Circle { radius } => (radius * 2.0, radius * 2.0),
        SkeletonShape::Text { lines, line_height } => (280.0, *lines as f64 * line_height),
        SkeletonShape::Avatar { size } => (*size, *size),
        SkeletonShape::Button { width, height } => (*width, *height),
        SkeletonShape::Custom { width, height, .. } => (*width, *height),
    }
}

fn shape_to_svg_element(shape: &SkeletonShape, x: f64, y: f64, extra: &str) -> String {
    match shape {
        SkeletonShape::Rectangle { width, height } => {
            format!("<rect class=\"sk\" x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"4\"{extra}/>")
        }
        SkeletonShape::Circle { radius } => {
            let cx = x + radius;
            let cy = y + radius;
            format!("<circle class=\"sk\" cx=\"{cx}\" cy=\"{cy}\" r=\"{radius}\"{extra}/>")
        }
        SkeletonShape::Text { lines, line_height } => {
            let mut out = String::new();
            for i in 0..*lines {
                let ly = y + (i as f64) * line_height;
                let w = if i == lines - 1 { 180.0 } else { 280.0 };
                out.push_str(&format!(
                    "<rect class=\"sk\" x=\"{x}\" y=\"{ly}\" width=\"{w}\" height=\"{h}\" rx=\"3\"{extra}/>",
                    h = line_height * 0.7
                ));
            }
            out
        }
        SkeletonShape::Avatar { size } => {
            let cx = x + size / 2.0;
            let cy = y + size / 2.0;
            let r = size / 2.0;
            format!("<circle class=\"sk\" cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\"{extra}/>")
        }
        SkeletonShape::Button { width, height } => {
            format!("<rect class=\"sk\" x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"8\"{extra}/>")
        }
        SkeletonShape::Custom { width, height, border_radius } => {
            format!("<rect class=\"sk\" x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"{border_radius}\"{extra}/>")
        }
    }
}

// ── LoadingState ───────────────────────────────────────────────

/// Tri-state wrapper for data that may still be loading.
#[derive(Debug, Clone)]
pub enum LoadingState<T> {
    Loading(SkeletonLayout),
    Loaded(T),
    Error(String),
}

impl<T> LoadingState<T> {
    pub fn is_loading(&self) -> bool { matches!(self, Self::Loading(_)) }
    pub fn is_loaded(&self) -> bool { matches!(self, Self::Loaded(_)) }
    pub fn data(&self) -> Option<&T> {
        match self { Self::Loaded(t) => Some(t), _ => None }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_layout_has_items() {
        let layout = SkeletonLayout::card_layout();
        assert!(layout.items.len() >= 3, "card should have multiple items");
    }

    #[test]
    fn list_layout_correct_rows() {
        let layout = SkeletonLayout::list_layout(5);
        // Each row contributes 3 items (circle + 2 rects).
        assert_eq!(layout.items.len(), 15);
    }

    #[test]
    fn article_layout_has_items() {
        let layout = SkeletonLayout::article_layout();
        assert!(layout.items.len() >= 3);
    }

    #[test]
    fn svg_output_contains_svg_tag() {
        let layout = SkeletonLayout::card_layout();
        let config = SkeletonConfig::default();
        let svg = layout.to_svg(&config);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("class=\"sk\""));
    }

    #[test]
    fn svg_pulse_animation() {
        let config = SkeletonConfig { animation: SkeletonAnimation::Pulse, ..Default::default() };
        let svg = SkeletonLayout::new()
            .add(SkeletonShape::Rectangle { width: 100.0, height: 20.0 }, 0.0, 0.0)
            .to_svg(&config);
        assert!(svg.contains("@keyframes pulse"));
    }

    #[test]
    fn svg_wave_animation() {
        let config = SkeletonConfig { animation: SkeletonAnimation::Wave, ..Default::default() };
        let svg = SkeletonLayout::new()
            .add(SkeletonShape::Rectangle { width: 100.0, height: 20.0 }, 0.0, 0.0)
            .to_svg(&config);
        assert!(svg.contains("linearGradient"));
    }

    #[test]
    fn svg_none_animation() {
        let config = SkeletonConfig { animation: SkeletonAnimation::None_, ..Default::default() };
        let svg = SkeletonLayout::new()
            .add(SkeletonShape::Circle { radius: 10.0 }, 5.0, 5.0)
            .to_svg(&config);
        assert!(svg.contains("circle"));
        assert!(!svg.contains("@keyframes"));
    }

    #[test]
    fn loading_state_transitions() {
        let state: LoadingState<String> = LoadingState::Loading(SkeletonLayout::card_layout());
        assert!(state.is_loading());
        assert!(!state.is_loaded());
        assert!(state.data().is_none());

        let loaded: LoadingState<String> = LoadingState::Loaded("hello".into());
        assert!(!loaded.is_loading());
        assert!(loaded.is_loaded());
        assert_eq!(loaded.data(), Some(&"hello".to_string()));

        let err: LoadingState<String> = LoadingState::Error("oops".into());
        assert!(!err.is_loading());
        assert!(!err.is_loaded());
    }

    #[test]
    fn custom_shape_renders() {
        let layout = SkeletonLayout::new()
            .add(SkeletonShape::Custom { width: 80.0, height: 40.0, border_radius: 12.0 }, 10.0, 10.0);
        let svg = layout.to_svg(&SkeletonConfig::default());
        assert!(svg.contains("rx=\"12\""));
    }

    #[test]
    fn add_with_delay_sets_delay() {
        let layout = SkeletonLayout::new()
            .add_with_delay(SkeletonShape::Rectangle { width: 50.0, height: 10.0 }, 0.0, 0.0, 300);
        assert_eq!(layout.items[0].delay_ms, 300);
        let svg = layout.to_svg(&SkeletonConfig::default());
        assert!(svg.contains("animation-delay:300ms"));
    }

    #[test]
    fn text_shape_renders_lines() {
        let layout = SkeletonLayout::new()
            .add(SkeletonShape::Text { lines: 3, line_height: 16.0 }, 0.0, 0.0);
        let svg = layout.to_svg(&SkeletonConfig::default());
        let count = svg.matches("<rect").count();
        assert_eq!(count, 3);
    }
}
