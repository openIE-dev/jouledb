//! Responsive breakpoints: named breakpoints, media query generation,
//! container query support, breakpoint matching, between-ranges, and
//! orientation queries.
//!
//! Pure data — no browser dependency. Emits CSS media/container query strings.

use std::fmt;

// ── Named Breakpoints ───────────────────────────────────────────

/// Standard responsive breakpoint names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BreakpointName {
    Xs,
    Sm,
    Md,
    Lg,
    Xl,
    Xxl,
}

impl BreakpointName {
    /// Default min-width in pixels for each breakpoint.
    pub fn default_min_width(self) -> f64 {
        match self {
            BreakpointName::Xs => 0.0,
            BreakpointName::Sm => 640.0,
            BreakpointName::Md => 768.0,
            BreakpointName::Lg => 1024.0,
            BreakpointName::Xl => 1280.0,
            BreakpointName::Xxl => 1536.0,
        }
    }

    /// All names in ascending order.
    pub fn all() -> &'static [BreakpointName] {
        &[
            BreakpointName::Xs,
            BreakpointName::Sm,
            BreakpointName::Md,
            BreakpointName::Lg,
            BreakpointName::Xl,
            BreakpointName::Xxl,
        ]
    }
}

impl fmt::Display for BreakpointName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BreakpointName::Xs => write!(f, "xs"),
            BreakpointName::Sm => write!(f, "sm"),
            BreakpointName::Md => write!(f, "md"),
            BreakpointName::Lg => write!(f, "lg"),
            BreakpointName::Xl => write!(f, "xl"),
            BreakpointName::Xxl => write!(f, "2xl"),
        }
    }
}

// ── Breakpoint Definition ───────────────────────────────────────

/// A single breakpoint with a name and min-width.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub name: String,
    pub min_width: f64,
}

impl Breakpoint {
    pub fn new(name: impl Into<String>, min_width: f64) -> Self {
        Self {
            name: name.into(),
            min_width,
        }
    }

    pub fn from_named(bp: BreakpointName) -> Self {
        Self::new(bp.to_string(), bp.default_min_width())
    }
}

// ── Breakpoint Set ──────────────────────────────────────────────

/// A collection of breakpoints, sorted by min_width ascending.
#[derive(Debug, Clone)]
pub struct BreakpointSet {
    breakpoints: Vec<Breakpoint>,
}

impl BreakpointSet {
    /// Create from a list of breakpoints (will be sorted).
    pub fn new(mut breakpoints: Vec<Breakpoint>) -> Self {
        breakpoints.sort_by(|a, b| {
            a.min_width
                .partial_cmp(&b.min_width)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { breakpoints }
    }

    /// Default Tailwind-style breakpoints.
    pub fn tailwind() -> Self {
        let bps: Vec<Breakpoint> = BreakpointName::all()
            .iter()
            .map(|n| Breakpoint::from_named(*n))
            .collect();
        Self::new(bps)
    }

    /// Bootstrap-style breakpoints.
    pub fn bootstrap() -> Self {
        Self::new(vec![
            Breakpoint::new("xs", 0.0),
            Breakpoint::new("sm", 576.0),
            Breakpoint::new("md", 768.0),
            Breakpoint::new("lg", 992.0),
            Breakpoint::new("xl", 1200.0),
            Breakpoint::new("xxl", 1400.0),
        ])
    }

    /// Find which breakpoint matches a viewport width (returns the largest that fits).
    pub fn match_width(&self, width: f64) -> Option<&Breakpoint> {
        self.breakpoints
            .iter()
            .rev()
            .find(|bp| width >= bp.min_width)
    }

    /// Get a breakpoint by name.
    pub fn get(&self, name: &str) -> Option<&Breakpoint> {
        self.breakpoints.iter().find(|bp| bp.name == name)
    }

    /// Generate a `min-width` media query for a breakpoint.
    pub fn up(&self, name: &str) -> Option<String> {
        self.get(name).map(|bp| {
            if bp.min_width == 0.0 {
                String::new() // xs needs no query
            } else {
                format!("@media (min-width: {}px)", bp.min_width)
            }
        })
    }

    /// Generate a `max-width` media query (exclusive of the *next* breakpoint).
    pub fn down(&self, name: &str) -> Option<String> {
        let idx = self.breakpoints.iter().position(|bp| bp.name == name)?;
        if idx + 1 < self.breakpoints.len() {
            let max = self.breakpoints[idx + 1].min_width - 0.02;
            Some(format!("@media (max-width: {max}px)"))
        } else {
            // Largest breakpoint — no upper bound.
            None
        }
    }

    /// Generate a between-range media query.
    pub fn between(&self, lower: &str, upper: &str) -> Option<String> {
        let lo = self.get(lower)?;
        let hi = self.get(upper)?;
        if lo.min_width >= hi.min_width {
            return None;
        }
        let max = hi.min_width - 0.02;
        if lo.min_width == 0.0 {
            Some(format!("@media (max-width: {max}px)"))
        } else {
            Some(format!(
                "@media (min-width: {}px) and (max-width: {max}px)",
                lo.min_width
            ))
        }
    }

    /// Generate a container query instead of a media query.
    pub fn container_up(&self, name: &str, container_name: Option<&str>) -> Option<String> {
        let bp = self.get(name)?;
        let container = container_name.unwrap_or_default();
        if bp.min_width == 0.0 {
            Some(String::new())
        } else if container.is_empty() {
            Some(format!("@container (min-width: {}px)", bp.min_width))
        } else {
            Some(format!(
                "@container {container} (min-width: {}px)",
                bp.min_width
            ))
        }
    }

    /// All breakpoints.
    pub fn iter(&self) -> impl Iterator<Item = &Breakpoint> {
        self.breakpoints.iter()
    }

    pub fn len(&self) -> usize {
        self.breakpoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.breakpoints.is_empty()
    }
}

// ── Orientation ─────────────────────────────────────────────────

/// Device orientation for media queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Portrait,
    Landscape,
}

impl Orientation {
    /// Determine orientation from dimensions.
    pub fn from_dimensions(width: f64, height: f64) -> Self {
        if height >= width {
            Orientation::Portrait
        } else {
            Orientation::Landscape
        }
    }

    /// CSS media query fragment.
    pub fn media_query(self) -> String {
        match self {
            Orientation::Portrait => "@media (orientation: portrait)".to_owned(),
            Orientation::Landscape => "@media (orientation: landscape)".to_owned(),
        }
    }
}

impl fmt::Display for Orientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Orientation::Portrait => write!(f, "portrait"),
            Orientation::Landscape => write!(f, "landscape"),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_breakpoints() {
        assert_eq!(BreakpointName::Sm.default_min_width(), 640.0);
        assert_eq!(BreakpointName::Lg.default_min_width(), 1024.0);
    }

    #[test]
    fn test_tailwind_set() {
        let set = BreakpointSet::tailwind();
        assert_eq!(set.len(), 6);
    }

    #[test]
    fn test_bootstrap_set() {
        let set = BreakpointSet::bootstrap();
        assert_eq!(set.len(), 6);
        assert!(set.get("sm").unwrap().min_width == 576.0);
    }

    #[test]
    fn test_match_width() {
        let set = BreakpointSet::tailwind();
        let bp = set.match_width(800.0).unwrap();
        assert_eq!(bp.name, "md"); // 768 <= 800 < 1024
    }

    #[test]
    fn test_up_query() {
        let set = BreakpointSet::tailwind();
        assert_eq!(set.up("xs").unwrap(), "");
        assert_eq!(set.up("md").unwrap(), "@media (min-width: 768px)");
    }

    #[test]
    fn test_down_query() {
        let set = BreakpointSet::tailwind();
        let down = set.down("sm").unwrap();
        assert!(down.contains("max-width"));
        assert!(down.contains("767.98"));
    }

    #[test]
    fn test_down_largest() {
        let set = BreakpointSet::tailwind();
        assert!(set.down("2xl").is_none());
    }

    #[test]
    fn test_between() {
        let set = BreakpointSet::tailwind();
        let q = set.between("sm", "lg").unwrap();
        assert!(q.contains("min-width: 640px"));
        assert!(q.contains("max-width: 1023.98px"));
    }

    #[test]
    fn test_between_invalid_order() {
        let set = BreakpointSet::tailwind();
        assert!(set.between("lg", "sm").is_none());
    }

    #[test]
    fn test_container_up() {
        let set = BreakpointSet::tailwind();
        let q = set.container_up("md", Some("sidebar")).unwrap();
        assert_eq!(q, "@container sidebar (min-width: 768px)");
    }

    #[test]
    fn test_container_up_unnamed() {
        let set = BreakpointSet::tailwind();
        let q = set.container_up("lg", None).unwrap();
        assert_eq!(q, "@container (min-width: 1024px)");
    }

    #[test]
    fn test_orientation_from_dimensions() {
        assert_eq!(
            Orientation::from_dimensions(375.0, 812.0),
            Orientation::Portrait
        );
        assert_eq!(
            Orientation::from_dimensions(1920.0, 1080.0),
            Orientation::Landscape
        );
    }

    #[test]
    fn test_orientation_media_query() {
        assert_eq!(
            Orientation::Portrait.media_query(),
            "@media (orientation: portrait)"
        );
    }

    #[test]
    fn test_breakpoint_name_display() {
        assert_eq!(BreakpointName::Xxl.to_string(), "2xl");
        assert_eq!(BreakpointName::Xs.to_string(), "xs");
    }

    #[test]
    fn test_get_nonexistent() {
        let set = BreakpointSet::tailwind();
        assert!(set.get("nonexistent").is_none());
    }
}
