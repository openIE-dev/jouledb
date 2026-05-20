//! Timeline visualization: event placement on time axis, duration bars,
//! milestone markers, overlapping event handling (swim lanes), zoom levels
//! (day/week/month/year), grouping by category, today marker.  Pure Rust SVG.

use std::fmt::Write as FmtWrite;

// ── Time representation ──────────────────────────────────────────

/// A simple timestamp as days since epoch (Jan 1, 2000).
/// Avoids external deps while giving meaningful time arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimeStamp(pub i64);

impl TimeStamp {
    /// Create from year/month/day.
    pub fn ymd(year: i32, month: u32, day: u32) -> Self {
        // Simple days-since-2000-01-01 approximation.
        let y = year as i64 - 2000;
        let m = month as i64;
        let d = day as i64;
        Self(y * 365 + y / 4 + (m - 1) * 30 + d)
    }

    /// Difference in days.
    pub fn days_between(self, other: TimeStamp) -> i64 {
        (other.0 - self.0).abs()
    }

    pub fn add_days(self, days: i64) -> Self {
        Self(self.0 + days)
    }

    /// Approximate label.
    pub fn label(&self) -> String {
        // Reverse the approximation for display.
        let total = self.0;
        let year = 2000 + total / 365;
        let rem = total % 365;
        let month = (rem / 30).max(0) + 1;
        let day = (rem % 30).max(0) + 1;
        format!("{year}-{month:02}-{day:02}")
    }
}

// ── Event types ──────────────────────────────────────────────────

/// A timeline event — either an instant (milestone) or a duration.
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    pub id: String,
    pub label: String,
    pub start: TimeStamp,
    /// If `None`, this is a milestone (point event).
    pub end: Option<TimeStamp>,
    pub category: Option<String>,
    pub color: String,
}

impl TimelineEvent {
    /// Create a duration event.
    pub fn duration(
        id: impl Into<String>,
        label: impl Into<String>,
        start: TimeStamp,
        end: TimeStamp,
        color: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            start,
            end: Some(end),
            category: None,
            color: color.into(),
        }
    }

    /// Create a milestone event.
    pub fn milestone(
        id: impl Into<String>,
        label: impl Into<String>,
        at: TimeStamp,
        color: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            start: at,
            end: None,
            category: None,
            color: color.into(),
        }
    }

    pub fn with_category(mut self, cat: impl Into<String>) -> Self {
        self.category = Some(cat.into());
        self
    }

    pub fn is_milestone(&self) -> bool {
        self.end.is_none()
    }

    pub fn effective_end(&self) -> TimeStamp {
        self.end.unwrap_or(self.start)
    }

    /// Duration in days (0 for milestones).
    pub fn duration_days(&self) -> i64 {
        self.start.days_between(self.effective_end())
    }
}

// ── Zoom levels ──────────────────────────────────────────────────

/// Zoom level for the time axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoomLevel {
    Day,
    Week,
    Month,
    Year,
}

impl ZoomLevel {
    /// Approximate days per tick at this zoom level.
    pub fn days_per_tick(&self) -> i64 {
        match self {
            ZoomLevel::Day => 1,
            ZoomLevel::Week => 7,
            ZoomLevel::Month => 30,
            ZoomLevel::Year => 365,
        }
    }
}

// ── Swim lane assignment ─────────────────────────────────────────

/// Assign non-overlapping swim lanes to events.
pub fn assign_lanes(events: &[TimelineEvent]) -> Vec<usize> {
    let mut indexed: Vec<(usize, &TimelineEvent)> = events.iter().enumerate().collect();
    indexed.sort_by_key(|(_, e)| e.start);

    let mut lane_ends: Vec<TimeStamp> = Vec::new();
    let mut assignments = vec![0_usize; events.len()];

    for (orig_idx, event) in &indexed {
        let ev_end = event.effective_end();
        // Find first lane where the event fits.
        let lane = lane_ends
            .iter()
            .position(|end| event.start >= *end);
        match lane {
            Some(l) => {
                lane_ends[l] = ev_end.add_days(1);
                assignments[*orig_idx] = l;
            }
            None => {
                assignments[*orig_idx] = lane_ends.len();
                lane_ends.push(ev_end.add_days(1));
            }
        }
    }

    assignments
}

/// Number of swim lanes needed.
pub fn lane_count(lanes: &[usize]) -> usize {
    lanes.iter().copied().max().map_or(0, |m| m + 1)
}

// ── Grouping ─────────────────────────────────────────────────────

/// Group events by category.
pub fn group_by_category(events: &[TimelineEvent]) -> Vec<(String, Vec<usize>)> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, ev) in events.iter().enumerate() {
        let cat = ev.category.clone().unwrap_or_else(|| "Uncategorized".into());
        if let Some(grp) = groups.iter_mut().find(|(c, _)| *c == cat) {
            grp.1.push(i);
        } else {
            groups.push((cat, vec![i]));
        }
    }
    groups
}

// ── Config & rendering ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TimelineConfig {
    pub width: f64,
    pub lane_height: f64,
    pub margin: f64,
    pub header_height: f64,
    pub zoom: ZoomLevel,
    pub today: Option<TimeStamp>,
    pub today_color: String,
    pub bar_height: f64,
    pub milestone_radius: f64,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            width: 1000.0,
            lane_height: 36.0,
            margin: 40.0,
            header_height: 30.0,
            zoom: ZoomLevel::Month,
            today: None,
            today_color: "#e74c3c".into(),
            bar_height: 20.0,
            milestone_radius: 6.0,
        }
    }
}

pub fn render_svg(events: &[TimelineEvent], config: &TimelineConfig) -> String {
    if events.is_empty() {
        return format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}"></svg>"#,
            config.width, config.margin * 2.0 + config.header_height,
        );
    }

    let lanes = assign_lanes(events);
    let n_lanes = lane_count(&lanes);

    let total_h = config.margin + config.header_height + n_lanes as f64 * config.lane_height + config.margin;

    // Time range.
    let t_min = events.iter().map(|e| e.start).min().unwrap();
    let t_max = events.iter().map(|e| e.effective_end()).max().unwrap();
    let t_range = (t_max.0 - t_min.0).max(1) as f64;
    let plot_w = config.width - 2.0 * config.margin;

    let to_x = |t: TimeStamp| -> f64 {
        config.margin + (t.0 - t_min.0) as f64 / t_range * plot_w
    };

    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{total_h}">"#,
        config.width,
    );

    // Time axis ticks.
    let tick_step = config.zoom.days_per_tick();
    let mut tick = t_min;
    while tick <= t_max {
        let x = to_x(tick);
        let _ = write!(
            svg,
            "<line x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{total_h:.1}\" stroke=\"lightgray\" stroke-width=\"1\"/>",
            config.margin,
        );
        let _ = write!(
            svg,
            "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"9\" fill=\"gray\">{}</text>",
            config.margin + config.header_height - 5.0,
            tick.label(),
        );
        tick = tick.add_days(tick_step);
    }

    // Today marker.
    if let Some(today) = config.today {
        if today >= t_min && today <= t_max {
            let x = to_x(today);
            let _ = write!(
                svg,
                r#"<line x1="{x:.1}" y1="{:.1}" x2="{x:.1}" y2="{total_h:.1}" stroke="{}" stroke-width="2" stroke-dasharray="4,3"/>"#,
                config.margin, config.today_color,
            );
        }
    }

    // Events.
    let y_base = config.margin + config.header_height;
    for (i, ev) in events.iter().enumerate() {
        let lane = lanes[i];
        let lane_y = y_base + lane as f64 * config.lane_height;
        let bar_y = lane_y + (config.lane_height - config.bar_height) / 2.0;

        if ev.is_milestone() {
            let cx = to_x(ev.start);
            let cy = lane_y + config.lane_height / 2.0;
            let r = config.milestone_radius;
            // Diamond shape.
            let _ = write!(
                svg,
                r#"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="{}"/>"#,
                cx, cy - r, cx + r, cy, cx, cy + r, cx - r, cy, ev.color,
            );
            let _ = write!(
                svg,
                r#"<text x="{:.1}" y="{:.1}" text-anchor="start" font-size="10">{}</text>"#,
                cx + r + 3.0, cy + 4.0, ev.label,
            );
        } else {
            let x1 = to_x(ev.start);
            let x2 = to_x(ev.effective_end());
            let w = (x2 - x1).max(2.0);
            let _ = write!(
                svg,
                r#"<rect x="{x1:.1}" y="{bar_y:.1}" width="{w:.1}" height="{:.1}" rx="3" fill="{}" opacity="0.85"/>"#,
                config.bar_height, ev.color,
            );
            // Label inside bar if it fits.
            let label_x = x1 + 4.0;
            let label_y = bar_y + config.bar_height / 2.0 + 4.0;
            let _ = write!(
                svg,
                "<text x=\"{label_x:.1}\" y=\"{label_y:.1}\" font-size=\"10\" fill=\"white\" clip-path=\"url(#clip-{i})\">{}</text>",
                ev.label,
            );
        }
    }

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_events() -> Vec<TimelineEvent> {
        vec![
            TimelineEvent::duration("a", "Project A", TimeStamp::ymd(2026, 1, 1), TimeStamp::ymd(2026, 3, 15), "#3498db"),
            TimelineEvent::duration("b", "Project B", TimeStamp::ymd(2026, 2, 1), TimeStamp::ymd(2026, 5, 1), "#2ecc71"),
            TimelineEvent::milestone("m1", "Launch", TimeStamp::ymd(2026, 4, 1), "#e74c3c"),
            TimelineEvent::duration("c", "Phase 2", TimeStamp::ymd(2026, 4, 15), TimeStamp::ymd(2026, 6, 30), "#9b59b6"),
        ]
    }

    #[test]
    fn timestamp_ymd() {
        let t = TimeStamp::ymd(2026, 3, 15);
        assert!(t.0 > 0);
    }

    #[test]
    fn timestamp_days_between() {
        let a = TimeStamp::ymd(2026, 1, 1);
        let b = TimeStamp::ymd(2026, 2, 1);
        assert!(a.days_between(b) > 0);
    }

    #[test]
    fn timestamp_label() {
        let t = TimeStamp::ymd(2026, 3, 15);
        let lbl = t.label();
        assert!(lbl.contains("2026"));
    }

    #[test]
    fn event_duration_days() {
        let ev = TimelineEvent::duration("x", "X", TimeStamp::ymd(2026, 1, 1), TimeStamp::ymd(2026, 2, 1), "#000");
        assert!(ev.duration_days() > 0);
    }

    #[test]
    fn milestone_zero_duration() {
        let ev = TimelineEvent::milestone("m", "M", TimeStamp::ymd(2026, 5, 1), "#000");
        assert_eq!(ev.duration_days(), 0);
        assert!(ev.is_milestone());
    }

    #[test]
    fn lane_assignment_no_overlap() {
        let events = sample_events();
        let lanes = assign_lanes(&events);
        assert_eq!(lanes.len(), events.len());
        // Overlapping events should be in different lanes.
        // A and B overlap, so they should differ.
        assert_ne!(lanes[0], lanes[1]);
    }

    #[test]
    fn lane_count_correct() {
        let events = sample_events();
        let lanes = assign_lanes(&events);
        let n = lane_count(&lanes);
        assert!(n >= 2);
    }

    #[test]
    fn group_by_category_works() {
        let events = vec![
            TimelineEvent::duration("a", "A", TimeStamp::ymd(2026, 1, 1), TimeStamp::ymd(2026, 2, 1), "#000")
                .with_category("Dev"),
            TimelineEvent::duration("b", "B", TimeStamp::ymd(2026, 2, 1), TimeStamp::ymd(2026, 3, 1), "#000")
                .with_category("Dev"),
            TimelineEvent::milestone("c", "C", TimeStamp::ymd(2026, 3, 1), "#000")
                .with_category("Ops"),
        ];
        let groups = group_by_category(&events);
        assert_eq!(groups.len(), 2);
        let dev = groups.iter().find(|(c, _)| c == "Dev").unwrap();
        assert_eq!(dev.1.len(), 2);
    }

    #[test]
    fn render_svg_basic() {
        let events = sample_events();
        let svg = render_svg(&events, &TimelineConfig::default());
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));      // duration bars
        assert!(svg.contains("<polygon"));   // milestone diamond
    }

    #[test]
    fn render_with_today() {
        let events = sample_events();
        let cfg = TimelineConfig {
            today: Some(TimeStamp::ymd(2026, 3, 8)),
            ..Default::default()
        };
        let svg = render_svg(&events, &cfg);
        assert!(svg.contains("stroke-dasharray")); // today marker
    }

    #[test]
    fn zoom_day_ticks() {
        assert_eq!(ZoomLevel::Day.days_per_tick(), 1);
        assert_eq!(ZoomLevel::Week.days_per_tick(), 7);
        assert_eq!(ZoomLevel::Month.days_per_tick(), 30);
        assert_eq!(ZoomLevel::Year.days_per_tick(), 365);
    }

    #[test]
    fn empty_events() {
        let svg = render_svg(&[], &TimelineConfig::default());
        assert!(svg.contains("svg"));
    }

    #[test]
    fn non_overlapping_share_lane() {
        let events = vec![
            TimelineEvent::duration("a", "A", TimeStamp::ymd(2026, 1, 1), TimeStamp::ymd(2026, 2, 1), "#000"),
            TimelineEvent::duration("b", "B", TimeStamp::ymd(2026, 3, 1), TimeStamp::ymd(2026, 4, 1), "#000"),
        ];
        let lanes = assign_lanes(&events);
        assert_eq!(lanes[0], lanes[1]); // no overlap → same lane
    }

    #[test]
    fn uncategorized_default() {
        let events = vec![
            TimelineEvent::milestone("x", "X", TimeStamp::ymd(2026, 6, 1), "#000"),
        ];
        let groups = group_by_category(&events);
        assert_eq!(groups[0].0, "Uncategorized");
    }
}
