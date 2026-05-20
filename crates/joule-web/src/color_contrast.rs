//! WCAG color contrast checker: contrast ratio calculation, AA/AAA compliance,
//! relative luminance, suggested adjustments, and bulk page audit.
//!
//! Pure math — no browser dependency. Implements WCAG 2.1 contrast algorithms.

// ── Color ─────────────────────────────────────────────────────

/// An sRGB color (0–255 per channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub const BLACK: Color = Color::new(0, 0, 0);
    pub const WHITE: Color = Color::new(255, 255, 255);

    /// Parse a hex color string ("#RGB", "#RRGGBB").
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix('#').unwrap_or(hex);
        match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                Some(Color::new(r, g, b))
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(Color::new(r, g, b))
            }
            _ => None,
        }
    }

    /// Render as "#RRGGBB".
    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    /// Relative luminance per WCAG 2.1 (0.0 = darkest, 1.0 = lightest).
    pub fn relative_luminance(&self) -> f64 {
        fn linearize(c: u8) -> f64 {
            let s = c as f64 / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        }
        let r = linearize(self.r);
        let g = linearize(self.g);
        let b = linearize(self.b);
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }
}

// ── Contrast Ratio ────────────────────────────────────────────

/// Compute the WCAG contrast ratio between two colors (1.0 to 21.0).
pub fn contrast_ratio(c1: Color, c2: Color) -> f64 {
    let l1 = c1.relative_luminance();
    let l2 = c2.relative_luminance();
    let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
    (lighter + 0.05) / (darker + 0.05)
}

// ── WCAG Compliance ───────────────────────────────────────────

/// WCAG conformance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WcagLevel {
    /// Does not meet AA.
    Fail,
    /// Meets AA for the given text size.
    AA,
    /// Meets AAA for the given text size.
    AAA,
}

/// Text size classification for WCAG thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextSize {
    /// Normal text (< 18pt, or < 14pt bold).
    Normal,
    /// Large text (>= 18pt, or >= 14pt bold).
    Large,
}

/// Check WCAG compliance for a color pair at a given text size.
pub fn check_compliance(fg: Color, bg: Color, text_size: TextSize) -> WcagLevel {
    let ratio = contrast_ratio(fg, bg);
    let (aa_threshold, aaa_threshold) = match text_size {
        TextSize::Normal => (4.5, 7.0),
        TextSize::Large => (3.0, 4.5),
    };
    if ratio >= aaa_threshold {
        WcagLevel::AAA
    } else if ratio >= aa_threshold {
        WcagLevel::AA
    } else {
        WcagLevel::Fail
    }
}

// ── Color Adjustment ──────────────────────────────────────────

/// Suggest a darker or lighter foreground color to meet a target contrast ratio.
/// Returns `None` if impossible (e.g., both black and white fail, which can't happen).
pub fn suggest_foreground(bg: Color, target_ratio: f64) -> Option<Color> {
    let bg_lum = bg.relative_luminance();

    // Try darkening: find a dark color with enough contrast.
    // Target luminance for dark fg: (bg_lum + 0.05) / target_ratio - 0.05
    let dark_lum = (bg_lum + 0.05) / target_ratio - 0.05;
    if dark_lum >= 0.0 {
        if let Some(c) = find_gray_for_luminance(dark_lum) {
            if contrast_ratio(c, bg) >= target_ratio - 0.01 {
                return Some(c);
            }
        }
    }

    // Try lightening: target_ratio * (bg_lum + 0.05) - 0.05 = light_lum  (when fg is lighter)
    // But we want fg lighter than bg only if darkening failed.
    let light_lum = target_ratio * (bg_lum + 0.05) - 0.05;
    if light_lum <= 1.0 {
        if let Some(c) = find_gray_for_luminance(light_lum) {
            if contrast_ratio(c, bg) >= target_ratio - 0.01 {
                return Some(c);
            }
        }
    }

    // Fallback: black or white, whichever has higher contrast.
    if contrast_ratio(Color::BLACK, bg) >= contrast_ratio(Color::WHITE, bg) {
        Some(Color::BLACK)
    } else {
        Some(Color::WHITE)
    }
}

/// Find a gray (r=g=b) color closest to a target relative luminance.
fn find_gray_for_luminance(target: f64) -> Option<Color> {
    let target = target.clamp(0.0, 1.0);
    let mut best_v = 0u8;
    let mut best_diff = f64::MAX;
    // Binary-ish search over 0–255.
    for v in 0..=255u8 {
        let c = Color::new(v, v, v);
        let diff = (c.relative_luminance() - target).abs();
        if diff < best_diff {
            best_diff = diff;
            best_v = v;
        }
    }
    Some(Color::new(best_v, best_v, best_v))
}

// ── Color Pair Evaluation ─────────────────────────────────────

/// Result of evaluating a color pair.
#[derive(Debug, Clone)]
pub struct ColorPairResult {
    pub foreground: Color,
    pub background: Color,
    pub contrast_ratio: f64,
    pub normal_text: WcagLevel,
    pub large_text: WcagLevel,
}

/// Evaluate a foreground/background color pair.
pub fn evaluate_pair(fg: Color, bg: Color) -> ColorPairResult {
    let ratio = contrast_ratio(fg, bg);
    ColorPairResult {
        foreground: fg,
        background: bg,
        contrast_ratio: ratio,
        normal_text: check_compliance(fg, bg, TextSize::Normal),
        large_text: check_compliance(fg, bg, TextSize::Large),
    }
}

// ── Bulk Audit ────────────────────────────────────────────────

/// An element in a page audit.
#[derive(Debug, Clone)]
pub struct AuditElement {
    pub selector: String,
    pub foreground: Color,
    pub background: Color,
    pub text_size: TextSize,
}

/// Result of auditing one element.
#[derive(Debug, Clone)]
pub struct AuditResult {
    pub selector: String,
    pub contrast_ratio: f64,
    pub level: WcagLevel,
    pub suggestion: Option<Color>,
}

/// Audit a list of elements for WCAG AA compliance.
pub fn audit_page(elements: &[AuditElement]) -> Vec<AuditResult> {
    elements
        .iter()
        .map(|el| {
            let ratio = contrast_ratio(el.foreground, el.background);
            let level = check_compliance(el.foreground, el.background, el.text_size);
            let suggestion = if level == WcagLevel::Fail {
                let target = match el.text_size {
                    TextSize::Normal => 4.5,
                    TextSize::Large => 3.0,
                };
                suggest_foreground(el.background, target)
            } else {
                None
            };
            AuditResult {
                selector: el.selector.clone(),
                contrast_ratio: ratio,
                level,
                suggestion,
            }
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_on_white_max_contrast() {
        let ratio = contrast_ratio(Color::BLACK, Color::WHITE);
        assert!((ratio - 21.0).abs() < 0.1);
    }

    #[test]
    fn same_color_min_contrast() {
        let ratio = contrast_ratio(Color::new(128, 128, 128), Color::new(128, 128, 128));
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn symmetric() {
        let c1 = Color::new(255, 0, 0);
        let c2 = Color::new(0, 0, 255);
        assert!((contrast_ratio(c1, c2) - contrast_ratio(c2, c1)).abs() < 0.001);
    }

    #[test]
    fn hex_parsing() {
        assert_eq!(Color::from_hex("#ff0000"), Some(Color::new(255, 0, 0)));
        assert_eq!(Color::from_hex("#fff"), Some(Color::new(255, 255, 255)));
        assert_eq!(Color::from_hex("000000"), Some(Color::BLACK));
        assert_eq!(Color::from_hex("#zz"), None);
    }

    #[test]
    fn hex_rendering() {
        assert_eq!(Color::new(255, 128, 0).to_hex(), "#ff8000");
    }

    #[test]
    fn luminance_black_zero() {
        assert!(Color::BLACK.relative_luminance().abs() < 0.001);
    }

    #[test]
    fn luminance_white_one() {
        assert!((Color::WHITE.relative_luminance() - 1.0).abs() < 0.001);
    }

    #[test]
    fn aaa_normal_text() {
        // Black on white must pass AAA for normal text (ratio ~21).
        assert_eq!(
            check_compliance(Color::BLACK, Color::WHITE, TextSize::Normal),
            WcagLevel::AAA
        );
    }

    #[test]
    fn aa_large_text_lower_threshold() {
        // Gray on white — passes AA large (ratio >= 3.0) but not AA normal (needs 4.5).
        // #767676 (118,118,118) has ratio ~4.54, too close. Use lighter gray.
        let gray = Color::new(145, 145, 145);
        let level_large = check_compliance(gray, Color::WHITE, TextSize::Large);
        let level_normal = check_compliance(gray, Color::WHITE, TextSize::Normal);
        assert!(level_large >= WcagLevel::AA);
        // gray #969696 on white has ratio ~2.8, fails both for normal.
        assert_eq!(level_normal, WcagLevel::Fail);
    }

    #[test]
    fn evaluate_pair_result() {
        let result = evaluate_pair(Color::BLACK, Color::WHITE);
        assert!((result.contrast_ratio - 21.0).abs() < 0.1);
        assert_eq!(result.normal_text, WcagLevel::AAA);
        assert_eq!(result.large_text, WcagLevel::AAA);
    }

    #[test]
    fn suggest_foreground_for_white_bg() {
        let suggestion = suggest_foreground(Color::WHITE, 4.5).unwrap();
        let ratio = contrast_ratio(suggestion, Color::WHITE);
        assert!(ratio >= 4.4, "ratio was {}", ratio);
    }

    #[test]
    fn suggest_foreground_for_dark_bg() {
        let dark = Color::new(30, 30, 30);
        let suggestion = suggest_foreground(dark, 4.5).unwrap();
        let ratio = contrast_ratio(suggestion, dark);
        assert!(ratio >= 4.4, "ratio was {}", ratio);
    }

    #[test]
    fn bulk_audit() {
        let elements = vec![
            AuditElement {
                selector: "body".into(),
                foreground: Color::BLACK,
                background: Color::WHITE,
                text_size: TextSize::Normal,
            },
            AuditElement {
                selector: ".muted".into(),
                foreground: Color::new(200, 200, 200),
                background: Color::WHITE,
                text_size: TextSize::Normal,
            },
        ];
        let results = audit_page(&elements);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].level, WcagLevel::AAA);
        assert_eq!(results[1].level, WcagLevel::Fail);
        assert!(results[1].suggestion.is_some());
    }

    #[test]
    fn wcag_level_ordering() {
        assert!(WcagLevel::Fail < WcagLevel::AA);
        assert!(WcagLevel::AA < WcagLevel::AAA);
    }
}
