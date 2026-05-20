//! Avatar component: image URL, fallback to initials, size variants
//! (xs/sm/md/lg/xl), shape (circle/rounded-square), status indicator
//! (online/offline/busy/away), group/stack with overlap, color from name hash.

// ── Size ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvatarSize {
    Xs,
    Sm,
    Md,
    Lg,
    Xl,
}

impl AvatarSize {
    pub fn pixels(self) -> u32 {
        match self {
            Self::Xs => 24,
            Self::Sm => 32,
            Self::Md => 40,
            Self::Lg => 56,
            Self::Xl => 80,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Xs => "xs",
            Self::Sm => "sm",
            Self::Md => "md",
            Self::Lg => "lg",
            Self::Xl => "xl",
        }
    }
}

// ── Shape ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvatarShape {
    Circle,
    RoundedSquare,
}

impl AvatarShape {
    pub fn border_radius(self, size_px: u32) -> String {
        match self {
            Self::Circle => "50%".into(),
            Self::RoundedSquare => format!("{}px", size_px / 5),
        }
    }
}

// ── Status ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusIndicator {
    Online,
    Offline,
    Busy,
    Away,
}

impl StatusIndicator {
    pub fn color(self) -> &'static str {
        match self {
            Self::Online => "#2ecc71",
            Self::Offline => "#95a5a6",
            Self::Busy => "#e74c3c",
            Self::Away => "#f39c12",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Offline => "offline",
            Self::Busy => "busy",
            Self::Away => "away",
        }
    }
}

// ── Avatar ─────────────────────────────────────────────────────────

/// A single avatar component.
#[derive(Debug, Clone)]
pub struct Avatar {
    pub name: String,
    pub image_url: Option<String>,
    pub size: AvatarSize,
    pub shape: AvatarShape,
    pub status: Option<StatusIndicator>,
    pub alt: Option<String>,
}

impl Avatar {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            image_url: None,
            size: AvatarSize::Md,
            shape: AvatarShape::Circle,
            status: None,
            alt: None,
        }
    }

    pub fn image(mut self, url: impl Into<String>) -> Self {
        self.image_url = Some(url.into());
        self
    }

    pub fn size(mut self, s: AvatarSize) -> Self {
        self.size = s;
        self
    }

    pub fn shape(mut self, s: AvatarShape) -> Self {
        self.shape = s;
        self
    }

    pub fn status(mut self, s: StatusIndicator) -> Self {
        self.status = Some(s);
        self
    }

    pub fn alt(mut self, a: impl Into<String>) -> Self {
        self.alt = Some(a.into());
        self
    }

    /// Extract initials from the name (up to 2 characters).
    pub fn initials(&self) -> String {
        let parts: Vec<&str> = self.name.split_whitespace().collect();
        match parts.len() {
            0 => "?".into(),
            1 => {
                let mut chars = parts[0].chars();
                match chars.next() {
                    Some(c) => c.to_uppercase().to_string(),
                    None => "?".into(),
                }
            }
            _ => {
                let first = parts[0].chars().next().unwrap_or('?');
                let last = parts[parts.len() - 1].chars().next().unwrap_or('?');
                format!("{}{}", first.to_uppercase(), last.to_uppercase())
            }
        }
    }

    /// Deterministic background color derived from the name.
    pub fn color_from_name(&self) -> String {
        let hash = self.name_hash();
        let hue = hash % 360;
        format!("hsl({}, 65%, 55%)", hue)
    }

    fn name_hash(&self) -> u32 {
        let mut hash: u32 = 5381;
        for byte in self.name.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u32);
        }
        hash
    }

    /// Render to HTML.
    pub fn render(&self) -> String {
        let px = self.size.pixels();
        let radius = self.shape.border_radius(px);
        let size_class = self.size.as_str();

        let inner = if let Some(url) = &self.image_url {
            let alt_text = self.alt.as_deref().unwrap_or(&self.name);
            format!(
                "<img src=\"{}\" alt=\"{}\" width=\"{}\" height=\"{}\" \
                 style=\"border-radius:{}\" />",
                url, alt_text, px, px, radius
            )
        } else {
            let bg = self.color_from_name();
            let initials = self.initials();
            let font_size = px * 2 / 5;
            format!(
                "<span class=\"avatar-initials\" \
                 style=\"width:{}px;height:{}px;border-radius:{};background:{};font-size:{}px;line-height:{}px\">\
                 {}</span>",
                px, px, radius, bg, font_size, px, initials
            )
        };

        let status_html = if let Some(st) = &self.status {
            let dot_size = px / 4;
            format!(
                "<span class=\"avatar-status avatar-status--{}\" \
                 style=\"width:{}px;height:{}px;background:{}\"></span>",
                st.as_str(),
                dot_size,
                dot_size,
                st.color()
            )
        } else {
            String::new()
        };

        format!(
            "<div class=\"avatar avatar--{} avatar--{}\" role=\"img\" aria-label=\"{}\">\
             {}{}</div>",
            size_class,
            if self.shape == AvatarShape::Circle { "circle" } else { "rounded" },
            self.alt.as_deref().unwrap_or(&self.name),
            inner,
            status_html,
        )
    }
}

// ── Avatar group ───────────────────────────────────────────────────

/// A group of avatars displayed with overlap (like a stack).
#[derive(Debug)]
pub struct AvatarGroup {
    pub avatars: Vec<Avatar>,
    /// Maximum number of avatars to display before showing "+N".
    pub max_display: Option<usize>,
    /// Overlap in pixels between adjacent avatars.
    pub overlap_px: u32,
}

impl AvatarGroup {
    pub fn new(avatars: Vec<Avatar>) -> Self {
        Self {
            avatars,
            max_display: None,
            overlap_px: 8,
        }
    }

    pub fn max_display(mut self, n: usize) -> Self {
        self.max_display = Some(n);
        self
    }

    pub fn overlap(mut self, px: u32) -> Self {
        self.overlap_px = px;
        self
    }

    /// Number of avatars hidden behind the overflow indicator.
    pub fn overflow_count(&self) -> usize {
        match self.max_display {
            Some(max) if self.avatars.len() > max => self.avatars.len() - max,
            _ => 0,
        }
    }

    /// Render the group to HTML.
    pub fn render(&self) -> String {
        let display_count = self
            .max_display
            .unwrap_or(self.avatars.len())
            .min(self.avatars.len());

        let mut html = String::from("<div class=\"avatar-group\">");

        for (i, avatar) in self.avatars.iter().take(display_count).enumerate() {
            let offset = if i > 0 {
                format!(" style=\"margin-left:-{}px\"", self.overlap_px)
            } else {
                String::new()
            };
            html.push_str(&format!("<div class=\"avatar-group-item\"{}>{}</div>", offset, avatar.render()));
        }

        let overflow = self.overflow_count();
        if overflow > 0 {
            let size = self
                .avatars
                .first()
                .map(|a| a.size)
                .unwrap_or(AvatarSize::Md);
            let px = size.pixels();
            html.push_str(&format!(
                "<div class=\"avatar-overflow\" \
                 style=\"width:{}px;height:{}px;margin-left:-{}px\">\
                 +{}</div>",
                px, px, self.overlap_px, overflow
            ));
        }

        html.push_str("</div>");
        html
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initials_two_names() {
        let a = Avatar::new("John Doe");
        assert_eq!(a.initials(), "JD");
    }

    #[test]
    fn test_initials_single_name() {
        let a = Avatar::new("Alice");
        assert_eq!(a.initials(), "A");
    }

    #[test]
    fn test_initials_three_names() {
        let a = Avatar::new("Mary Jane Watson");
        assert_eq!(a.initials(), "MW");
    }

    #[test]
    fn test_initials_empty() {
        let a = Avatar::new("");
        assert_eq!(a.initials(), "?");
    }

    #[test]
    fn test_color_deterministic() {
        let a = Avatar::new("Alice");
        let c1 = a.color_from_name();
        let c2 = a.color_from_name();
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_different_names_different_colors() {
        let a = Avatar::new("Alice");
        let b = Avatar::new("Bob");
        // Very likely different (not guaranteed, but good enough for a hash)
        assert_ne!(a.color_from_name(), b.color_from_name());
    }

    #[test]
    fn test_size_pixels() {
        assert_eq!(AvatarSize::Xs.pixels(), 24);
        assert_eq!(AvatarSize::Xl.pixels(), 80);
    }

    #[test]
    fn test_shape_border_radius() {
        assert_eq!(AvatarShape::Circle.border_radius(40), "50%");
        assert_eq!(AvatarShape::RoundedSquare.border_radius(40), "8px");
    }

    #[test]
    fn test_render_with_image() {
        let a = Avatar::new("Alice").image("https://example.com/alice.png");
        let html = a.render();
        assert!(html.contains("img src=\"https://example.com/alice.png\""));
        assert!(html.contains("role=\"img\""));
    }

    #[test]
    fn test_render_with_initials() {
        let a = Avatar::new("Bob Smith");
        let html = a.render();
        assert!(html.contains("BS"));
        assert!(html.contains("avatar-initials"));
    }

    #[test]
    fn test_render_with_status() {
        let a = Avatar::new("Alice").status(StatusIndicator::Online);
        let html = a.render();
        assert!(html.contains("avatar-status--online"));
        assert!(html.contains("#2ecc71"));
    }

    #[test]
    fn test_avatar_group_overflow() {
        let avatars = vec![
            Avatar::new("A"),
            Avatar::new("B"),
            Avatar::new("C"),
            Avatar::new("D"),
        ];
        let group = AvatarGroup::new(avatars).max_display(2);
        assert_eq!(group.overflow_count(), 2);
        let html = group.render();
        assert!(html.contains("+2"));
    }

    #[test]
    fn test_avatar_group_no_overflow() {
        let avatars = vec![Avatar::new("A"), Avatar::new("B")];
        let group = AvatarGroup::new(avatars).max_display(5);
        assert_eq!(group.overflow_count(), 0);
    }

    #[test]
    fn test_status_colors() {
        assert_eq!(StatusIndicator::Online.color(), "#2ecc71");
        assert_eq!(StatusIndicator::Busy.color(), "#e74c3c");
        assert_eq!(StatusIndicator::Away.color(), "#f39c12");
        assert_eq!(StatusIndicator::Offline.color(), "#95a5a6");
    }
}
