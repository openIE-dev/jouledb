//! Calendar heatmap (GitHub contribution-style).  Day cells arranged in weekly
//! columns, month labels, intensity-based colouring, value tooltip data, year
//! view, configurable colour scale.  Pure Rust SVG output.

use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// A single day's data point.
#[derive(Debug, Clone)]
pub struct DayValue {
    /// Year (e.g. 2026).
    pub year: i32,
    /// Month (1-12).
    pub month: u32,
    /// Day of month (1-31).
    pub day: u32,
    /// The numeric value for this day.
    pub value: f64,
    /// Optional tooltip text.
    pub tooltip: Option<String>,
}

impl DayValue {
    pub fn new(year: i32, month: u32, day: u32, value: f64) -> Self {
        Self {
            year,
            month,
            day,
            value,
            tooltip: None,
        }
    }

    pub fn with_tooltip(mut self, tip: impl Into<String>) -> Self {
        self.tooltip = Some(tip.into());
        self
    }

    /// Day-of-year (1-based, approximate — sufficient for layout).
    pub fn day_of_year(&self) -> u32 {
        let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let leap = is_leap_year(self.year);
        let mut doy = 0u32;
        for m in 1..self.month {
            let mut d = days_in_month[m as usize];
            if m == 2 && leap {
                d += 1;
            }
            doy += d;
        }
        doy + self.day
    }
}

/// Whether a year is a leap year.
fn is_leap_year(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Days in a given year.
fn days_in_year(y: i32) -> u32 {
    if is_leap_year(y) { 366 } else { 365 }
}

// ── Colour scale ────────────────────────────────────────────────

/// A colour stop in the intensity scale.
#[derive(Debug, Clone)]
pub struct ColorStop {
    /// Normalized position (0.0 = min, 1.0 = max).
    pub position: f64,
    /// Named SVG colour.
    pub color: String,
}

impl ColorStop {
    pub fn new(position: f64, color: impl Into<String>) -> Self {
        Self {
            position: position.clamp(0.0, 1.0),
            color: color.into(),
        }
    }
}

/// A multi-stop colour scale for mapping values to colours.
#[derive(Debug, Clone)]
pub struct ColorScale {
    pub stops: Vec<ColorStop>,
    /// Colour for days with no data.
    pub empty_color: String,
}

impl Default for ColorScale {
    fn default() -> Self {
        Self {
            stops: vec![
                ColorStop::new(0.0, "honeydew"),
                ColorStop::new(0.25, "palegreen"),
                ColorStop::new(0.5, "mediumseagreen"),
                ColorStop::new(0.75, "seagreen"),
                ColorStop::new(1.0, "darkgreen"),
            ],
            empty_color: "gainsboro".into(),
        }
    }
}

impl ColorScale {
    /// Map a normalized value (0..1) to the nearest colour stop.
    pub fn color_for(&self, normalized: f64) -> &str {
        if self.stops.is_empty() {
            return &self.empty_color;
        }
        let val = normalized.clamp(0.0, 1.0);
        let mut best = &self.stops[0];
        let mut best_dist = (val - best.position).abs();
        for stop in &self.stops[1..] {
            let d = (val - stop.position).abs();
            if d < best_dist {
                best = stop;
                best_dist = d;
            }
        }
        &best.color
    }
}

// ── Day of week ─────────────────────────────────────────────────

/// Day of the week (0 = Sunday, 6 = Saturday) for a given date.
/// Uses Zeller-like formula (Tomohiko Sakamoto's algorithm).
fn day_of_week(mut year: i32, month: u32, day: u32) -> u32 {
    let t = [0i32, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    if month < 3 {
        year -= 1;
    }
    let m = month as i32;
    let d = day as i32;
    let dow = (year + year / 4 - year / 100 + year / 400 + t[(m - 1) as usize] + d) % 7;
    dow.unsigned_abs()
}

// ── Config ──────────────────────────────────────────────────────

/// Day label style for the left margin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DayLabelStyle {
    /// No day labels.
    None,
    /// Mon, Wed, Fri only.
    Abbreviated,
    /// All seven days.
    Full,
}

/// Configuration for the calendar heatmap.
#[derive(Debug, Clone)]
pub struct CalendarHeatmapConfig {
    pub year: i32,
    pub cell_size: f64,
    pub cell_gap: f64,
    pub padding_left: f64,
    pub padding_top: f64,
    pub font_size: f64,
    pub color_scale: ColorScale,
    pub day_labels: DayLabelStyle,
    /// Corner radius for day cells.
    pub cell_radius: f64,
}

impl Default for CalendarHeatmapConfig {
    fn default() -> Self {
        Self {
            year: 2026,
            cell_size: 12.0,
            cell_gap: 2.0,
            padding_left: 40.0,
            padding_top: 24.0,
            font_size: 10.0,
            color_scale: ColorScale::default(),
            day_labels: DayLabelStyle::Abbreviated,
            cell_radius: 2.0,
        }
    }
}

impl CalendarHeatmapConfig {
    /// Total width of the SVG.
    pub fn svg_width(&self) -> f64 {
        // 53 weeks max
        self.padding_left + 53.0 * (self.cell_size + self.cell_gap) + 20.0
    }

    /// Total height of the SVG.
    pub fn svg_height(&self) -> f64 {
        self.padding_top + 7.0 * (self.cell_size + self.cell_gap) + 20.0
    }
}

// ── Layout ──────────────────────────────────────────────────────

/// A laid-out day cell.
#[derive(Debug, Clone)]
pub struct DayCell {
    pub x: f64,
    pub y: f64,
    pub color: String,
    pub value: f64,
    pub tooltip: Option<String>,
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

/// Compute the grid of day cells for a year.
pub fn compute_cells(
    data: &[DayValue],
    cfg: &CalendarHeatmapConfig,
) -> Vec<DayCell> {
    // Build lookup: day_of_year -> &DayValue
    let lookup: std::collections::HashMap<u32, &DayValue> = data
        .iter()
        .filter(|d| d.year == cfg.year)
        .map(|d| (d.day_of_year(), d))
        .collect();

    let min_val = data.iter().map(|d| d.value).fold(f64::INFINITY, f64::min);
    let max_val = data
        .iter()
        .map(|d| d.value)
        .fold(f64::NEG_INFINITY, f64::max);
    let val_range = max_val - min_val;

    let jan1_dow = day_of_week(cfg.year, 1, 1);
    let total_days = days_in_year(cfg.year);
    let step = cfg.cell_size + cfg.cell_gap;

    let days_per_month = [0u32, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let leap = is_leap_year(cfg.year);

    let mut cells = Vec::with_capacity(total_days as usize);
    let mut current_month = 1u32;
    let mut current_day = 1u32;

    for doy in 1..=total_days {
        let dow = (jan1_dow + doy - 1) % 7;
        let week = (jan1_dow + doy - 1) / 7;

        let x = cfg.padding_left + week as f64 * step;
        let y = cfg.padding_top + dow as f64 * step;

        let (color, value, tooltip) = if let Some(dv) = lookup.get(&doy) {
            let norm = if val_range.abs() < f64::EPSILON { 1.0 } else { (dv.value - min_val) / val_range };
            let c = cfg.color_scale.color_for(norm);
            (c.to_string(), dv.value, dv.tooltip.clone())
        } else {
            (cfg.color_scale.empty_color.clone(), 0.0, None)
        };

        cells.push(DayCell {
            x,
            y,
            color,
            value,
            tooltip,
            year: cfg.year,
            month: current_month,
            day: current_day,
        });

        // Advance date
        let month_days = if current_month == 2 && leap {
            29
        } else {
            days_per_month[current_month as usize]
        };
        current_day += 1;
        if current_day > month_days {
            current_day = 1;
            current_month += 1;
        }
    }

    cells
}

// ── Rendering ───────────────────────────────────────────────────

/// Month labels along the top.
fn render_month_labels(cfg: &CalendarHeatmapConfig) -> String {
    let month_names = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
        "Nov", "Dec",
    ];
    let jan1_dow = day_of_week(cfg.year, 1, 1);
    let step = cfg.cell_size + cfg.cell_gap;
    let mut svg = String::new();

    let days_per_month = [0u32, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let leap = is_leap_year(cfg.year);

    let mut doy = 1u32;
    for m in 1..=12u32 {
        let week = (jan1_dow + doy - 1) / 7;
        let x = cfg.padding_left + week as f64 * step;
        let y = cfg.padding_top - 6.0;
        let fs = cfg.font_size;
        let _ = write!(
            svg,
            "<text x=\"{x}\" y=\"{y}\" font-size=\"{fs}\" fill=\"gray\">{}</text>",
            month_names[m as usize]
        );
        let md = if m == 2 && leap {
            29
        } else {
            days_per_month[m as usize]
        };
        doy += md;
    }
    svg
}

/// Day-of-week labels on the left.
fn render_day_labels(cfg: &CalendarHeatmapConfig) -> String {
    let step = cfg.cell_size + cfg.cell_gap;
    let mut svg = String::new();

    let labels: Vec<(u32, &str)> = match cfg.day_labels {
        DayLabelStyle::None => return svg,
        DayLabelStyle::Abbreviated => vec![(1, "Mon"), (3, "Wed"), (5, "Fri")],
        DayLabelStyle::Full => vec![
            (0, "Sun"),
            (1, "Mon"),
            (2, "Tue"),
            (3, "Wed"),
            (4, "Thu"),
            (5, "Fri"),
            (6, "Sat"),
        ],
    };

    let fs = cfg.font_size;
    for (dow, label) in labels {
        let y = cfg.padding_top + dow as f64 * step + cfg.cell_size * 0.75;
        let x = cfg.padding_left - 6.0;
        let _ = write!(
            svg,
            "<text x=\"{x}\" y=\"{y}\" font-size=\"{fs}\" \
             text-anchor=\"end\" fill=\"gray\">{label}</text>"
        );
    }
    svg
}

/// Render the complete calendar heatmap as SVG.
pub fn render_calendar_heatmap(
    data: &[DayValue],
    cfg: &CalendarHeatmapConfig,
) -> String {
    let cells = compute_cells(data, cfg);
    let w = cfg.svg_width();
    let h = cfg.svg_height();

    let mut svg = String::with_capacity(8192);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
         viewBox=\"0 0 {w} {h}\">"
    );

    // Month labels
    svg.push_str(&render_month_labels(cfg));

    // Day labels
    svg.push_str(&render_day_labels(cfg));

    // Day cells
    svg.push_str("<g class=\"cells\">");
    for cell in &cells {
        let cs = cfg.cell_size;
        let cr = cfg.cell_radius;
        let _ = write!(
            svg,
            "<rect x=\"{}\" y=\"{}\" width=\"{cs}\" height=\"{cs}\" \
             rx=\"{cr}\" ry=\"{cr}\" fill=\"{}\" stroke=\"white\" stroke-width=\"0.5\">",
            cell.x, cell.y, cell.color
        );
        if let Some(tip) = &cell.tooltip {
            let _ = write!(svg, "<title>{tip}</title>");
        }
        svg.push_str("</rect>");
    }
    svg.push_str("</g>");

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_value_new() {
        let d = DayValue::new(2026, 3, 9, 42.0);
        assert_eq!(d.year, 2026);
        assert_eq!(d.month, 3);
        assert_eq!(d.day, 9);
        assert!((d.value - 42.0).abs() < 1e-9);
    }

    #[test]
    fn day_value_tooltip() {
        let d = DayValue::new(2026, 1, 1, 1.0).with_tooltip("New Year");
        assert_eq!(d.tooltip.as_deref(), Some("New Year"));
    }

    #[test]
    fn day_of_year_jan1() {
        let d = DayValue::new(2026, 1, 1, 0.0);
        assert_eq!(d.day_of_year(), 1);
    }

    #[test]
    fn day_of_year_dec31_non_leap() {
        let d = DayValue::new(2026, 12, 31, 0.0);
        assert_eq!(d.day_of_year(), 365);
    }

    #[test]
    fn day_of_year_dec31_leap() {
        let d = DayValue::new(2024, 12, 31, 0.0);
        assert_eq!(d.day_of_year(), 366);
    }

    #[test]
    fn is_leap() {
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(2026));
        assert!(is_leap_year(2000));
        assert!(!is_leap_year(1900));
    }

    #[test]
    fn color_scale_default() {
        let cs = ColorScale::default();
        assert_eq!(cs.stops.len(), 5);
    }

    #[test]
    fn color_scale_min() {
        let cs = ColorScale::default();
        let c = cs.color_for(0.0);
        assert_eq!(c, "honeydew");
    }

    #[test]
    fn color_scale_max() {
        let cs = ColorScale::default();
        let c = cs.color_for(1.0);
        assert_eq!(c, "darkgreen");
    }

    #[test]
    fn color_scale_mid() {
        let cs = ColorScale::default();
        let c = cs.color_for(0.5);
        assert_eq!(c, "mediumseagreen");
    }

    #[test]
    fn color_scale_empty() {
        let cs = ColorScale {
            stops: vec![],
            empty_color: "white".into(),
        };
        assert_eq!(cs.color_for(0.5), "white");
    }

    #[test]
    fn day_of_week_known() {
        // 2026-01-01 is Thursday (4)
        assert_eq!(day_of_week(2026, 1, 1), 4);
        // 2024-01-01 is Monday (1)
        assert_eq!(day_of_week(2024, 1, 1), 1);
    }

    #[test]
    fn config_default_sane() {
        let cfg = CalendarHeatmapConfig::default();
        assert!(cfg.cell_size > 0.0);
        assert!(cfg.svg_width() > 0.0);
        assert!(cfg.svg_height() > 0.0);
    }

    #[test]
    fn compute_cells_full_year() {
        let cfg = CalendarHeatmapConfig {
            year: 2026,
            ..Default::default()
        };
        let cells = compute_cells(&[], &cfg);
        assert_eq!(cells.len(), 365);
    }

    #[test]
    fn compute_cells_leap_year() {
        let cfg = CalendarHeatmapConfig {
            year: 2024,
            ..Default::default()
        };
        let cells = compute_cells(&[], &cfg);
        assert_eq!(cells.len(), 366);
    }

    #[test]
    fn cells_use_data_color() {
        let data = vec![DayValue::new(2026, 6, 15, 100.0)];
        let cfg = CalendarHeatmapConfig::default();
        let cells = compute_cells(&data, &cfg);
        let june15 = cells
            .iter()
            .find(|c| c.month == 6 && c.day == 15)
            .unwrap();
        assert_eq!(june15.color, "darkgreen"); // max value
    }

    #[test]
    fn render_produces_svg() {
        let cfg = CalendarHeatmapConfig::default();
        let svg = render_calendar_heatmap(&[], &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn render_has_rects() {
        let cfg = CalendarHeatmapConfig::default();
        let svg = render_calendar_heatmap(&[], &cfg);
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn render_has_month_labels() {
        let cfg = CalendarHeatmapConfig::default();
        let svg = render_calendar_heatmap(&[], &cfg);
        assert!(svg.contains("Jan"));
        assert!(svg.contains("Dec"));
    }

    #[test]
    fn render_has_day_labels_abbreviated() {
        let cfg = CalendarHeatmapConfig::default();
        let svg = render_calendar_heatmap(&[], &cfg);
        assert!(svg.contains("Mon"));
        assert!(svg.contains("Wed"));
        assert!(svg.contains("Fri"));
    }

    #[test]
    fn tooltip_appears_in_svg() {
        let data = vec![DayValue::new(2026, 1, 1, 5.0).with_tooltip("Holiday")];
        let cfg = CalendarHeatmapConfig::default();
        let svg = render_calendar_heatmap(&data, &cfg);
        assert!(svg.contains("<title>Holiday</title>"));
    }
}
