//! Date/time utilities for web applications.
//!
//! Replaces date-fns, dayjs, and Moment.js with a friendlier Rust API
//! built on `chrono`.

use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Timelike, Utc, Weekday};

// ── Relative Time ───────────────────────────────────────────────

/// Human-readable "time ago" string relative to now.
pub fn time_ago(dt: &DateTime<Utc>) -> String {
    let now = Utc::now();
    format_relative(dt, &now)
}

/// Human-readable "time until" string relative to now.
pub fn time_until(dt: &DateTime<Utc>) -> String {
    let now = Utc::now();
    let diff = *dt - now;
    let secs = diff.num_seconds();

    if secs < 0 {
        return time_ago(dt);
    }
    if secs < 10 {
        return "just now".to_string();
    }
    if secs < 60 {
        return format!("in {} seconds", secs);
    }
    let mins = diff.num_minutes();
    if mins < 60 {
        return format!("in {} minute{}", mins, plural(mins));
    }
    let hours = diff.num_hours();
    if hours < 24 {
        return format!("in {} hour{}", hours, plural(hours));
    }
    let days = diff.num_days();
    if days == 1 {
        return "tomorrow".to_string();
    }
    if days < 14 {
        return format!("in {} days", days);
    }
    let weeks = days / 7;
    if weeks < 5 {
        return format!("in {} week{}", weeks, plural(weeks));
    }
    let months = days / 30;
    if months < 12 {
        return format!("in {} month{}", months, plural(months));
    }
    let years = days / 365;
    format!("in {} year{}", years, plural(years))
}

/// Relative time description from `dt` relative to `base`.
pub fn format_relative(dt: &DateTime<Utc>, base: &DateTime<Utc>) -> String {
    let diff = *base - *dt;
    let secs = diff.num_seconds();

    if secs < 0 {
        // Future — delegate
        let future_diff = *dt - *base;
        let fs = future_diff.num_seconds();
        if fs < 10 {
            return "just now".to_string();
        }
        return format!("in {} seconds", fs);
    }

    if secs < 10 {
        return "just now".to_string();
    }
    if secs < 60 {
        return format!("{} seconds ago", secs);
    }
    let mins = diff.num_minutes();
    if mins < 60 {
        return format!("{} minute{} ago", mins, plural(mins));
    }
    let hours = diff.num_hours();
    if hours < 24 {
        return format!("{} hour{} ago", hours, plural(hours));
    }
    let days = diff.num_days();
    if days == 1 {
        return "yesterday".to_string();
    }
    if days < 14 {
        return format!("{} days ago", days);
    }
    let weeks = days / 7;
    if weeks < 5 {
        return format!("{} week{} ago", weeks, plural(weeks));
    }
    let months = days / 30;
    if months < 12 {
        return format!("{} month{} ago", months, plural(months));
    }
    let years = days / 365;
    format!("{} year{} ago", years, plural(years))
}

fn plural(n: i64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

// ── Formatting ──────────────────────────────────────────────────

/// Format a `DateTime<Utc>` using a pattern string.
///
/// Supported tokens: `YYYY`, `YY`, `MM`, `DD`, `HH`, `mm`, `ss`,
/// `ddd` (Mon), `dddd` (Monday), `MMM` (Jan), `MMMM` (January),
/// `A` (AM/PM), `hh` (12-hour).
pub fn format_date(dt: &DateTime<Utc>, pattern: &str) -> String {
    let mut result = pattern.to_string();

    // Order matters — replace longer tokens first.
    result = result.replace("YYYY", &format!("{:04}", dt.year()));
    result = result.replace("YY", &format!("{:02}", dt.year() % 100));

    // Day names before DD
    result = result.replace("dddd", weekday_full(dt.weekday()));
    result = result.replace("ddd", weekday_short(dt.weekday()));

    // Month names before MM
    result = result.replace("MMMM", month_full(dt.month()));
    result = result.replace("MMM", month_short(dt.month()));

    result = result.replace("MM", &format!("{:02}", dt.month()));
    result = result.replace("DD", &format!("{:02}", dt.day()));

    // AM/PM
    let ampm = if dt.hour() < 12 { "AM" } else { "PM" };
    result = result.replace('A', ampm);

    // 12-hour (hh) before 24-hour (HH) since hh doesn't overlap with HH
    let h12 = {
        let h = dt.hour() % 12;
        if h == 0 { 12 } else { h }
    };
    result = result.replace("hh", &format!("{:02}", h12));
    result = result.replace("HH", &format!("{:02}", dt.hour()));
    result = result.replace("mm", &format!("{:02}", dt.minute()));
    result = result.replace("ss", &format!("{:02}", dt.second()));

    result
}

fn weekday_short(w: Weekday) -> &'static str {
    match w {
        Weekday::Mon => "Mon",
        Weekday::Tue => "Tue",
        Weekday::Wed => "Wed",
        Weekday::Thu => "Thu",
        Weekday::Fri => "Fri",
        Weekday::Sat => "Sat",
        Weekday::Sun => "Sun",
    }
}

fn weekday_full(w: Weekday) -> &'static str {
    match w {
        Weekday::Mon => "Monday",
        Weekday::Tue => "Tuesday",
        Weekday::Wed => "Wednesday",
        Weekday::Thu => "Thursday",
        Weekday::Fri => "Friday",
        Weekday::Sat => "Saturday",
        Weekday::Sun => "Sunday",
    }
}

fn month_short(m: u32) -> &'static str {
    match m {
        1 => "Jan", 2 => "Feb", 3 => "Mar", 4 => "Apr",
        5 => "May", 6 => "Jun", 7 => "Jul", 8 => "Aug",
        9 => "Sep", 10 => "Oct", 11 => "Nov", 12 => "Dec",
        _ => "???",
    }
}

fn month_full(m: u32) -> &'static str {
    match m {
        1 => "January", 2 => "February", 3 => "March", 4 => "April",
        5 => "May", 6 => "June", 7 => "July", 8 => "August",
        9 => "September", 10 => "October", 11 => "November", 12 => "December",
        _ => "Unknown",
    }
}

// ── Duration ────────────────────────────────────────────────────

/// Broken-down duration between two points in time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurationParts {
    pub years: i64,
    pub months: i64,
    pub days: i64,
    pub hours: i64,
    pub minutes: i64,
    pub seconds: i64,
}

/// Compute the broken-down duration between two `DateTime`s.
pub fn duration_between(a: &DateTime<Utc>, b: &DateTime<Utc>) -> DurationParts {
    let (earlier, later) = if a <= b { (a, b) } else { (b, a) };

    let mut years = later.year() as i64 - earlier.year() as i64;
    let mut months = later.month() as i64 - earlier.month() as i64;
    let mut days = later.day() as i64 - earlier.day() as i64;
    let mut hours = later.hour() as i64 - earlier.hour() as i64;
    let mut minutes = later.minute() as i64 - earlier.minute() as i64;
    let mut seconds = later.second() as i64 - earlier.second() as i64;

    if seconds < 0 {
        seconds += 60;
        minutes -= 1;
    }
    if minutes < 0 {
        minutes += 60;
        hours -= 1;
    }
    if hours < 0 {
        hours += 24;
        days -= 1;
    }
    if days < 0 {
        months -= 1;
        let prev_month = if later.month() == 1 { 12 } else { later.month() - 1 };
        let prev_year = if later.month() == 1 {
            later.year() - 1
        } else {
            later.year()
        };
        days += days_in_month(prev_year, prev_month) as i64;
    }
    if months < 0 {
        months += 12;
        years -= 1;
    }

    DurationParts {
        years,
        months,
        days,
        hours,
        minutes,
        seconds,
    }
}

/// Format a `DurationParts` as a human-readable string.
pub fn format_duration(parts: &DurationParts) -> String {
    let mut segments = Vec::new();
    if parts.years > 0 {
        segments.push(format!("{} year{}", parts.years, plural(parts.years)));
    }
    if parts.months > 0 {
        segments.push(format!("{} month{}", parts.months, plural(parts.months)));
    }
    if parts.days > 0 {
        segments.push(format!("{} day{}", parts.days, plural(parts.days)));
    }
    if parts.hours > 0 {
        segments.push(format!("{} hour{}", parts.hours, plural(parts.hours)));
    }
    if parts.minutes > 0 {
        segments.push(format!("{} minute{}", parts.minutes, plural(parts.minutes)));
    }
    if parts.seconds > 0 || segments.is_empty() {
        segments.push(format!("{} second{}", parts.seconds, plural(parts.seconds)));
    }
    segments.join(", ")
}

/// Short format: "2y 3mo 1d".
pub fn format_duration_short(parts: &DurationParts) -> String {
    let mut segments = Vec::new();
    if parts.years > 0 {
        segments.push(format!("{}y", parts.years));
    }
    if parts.months > 0 {
        segments.push(format!("{}mo", parts.months));
    }
    if parts.days > 0 {
        segments.push(format!("{}d", parts.days));
    }
    if parts.hours > 0 {
        segments.push(format!("{}h", parts.hours));
    }
    if parts.minutes > 0 {
        segments.push(format!("{}m", parts.minutes));
    }
    if parts.seconds > 0 || segments.is_empty() {
        segments.push(format!("{}s", parts.seconds));
    }
    segments.join(" ")
}

// ── Calendar Helpers ────────────────────────────────────────────

/// Start of day (midnight UTC).
pub fn start_of_day(dt: &DateTime<Utc>) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(dt.year(), dt.month(), dt.day(), 0, 0, 0)
        .single()
        .unwrap_or(*dt)
}

/// End of day (23:59:59 UTC).
pub fn end_of_day(dt: &DateTime<Utc>) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(dt.year(), dt.month(), dt.day(), 23, 59, 59)
        .single()
        .unwrap_or(*dt)
}

/// Start of the ISO week (Monday).
pub fn start_of_week(dt: &DateTime<Utc>) -> DateTime<Utc> {
    let days_since_monday = dt.weekday().num_days_from_monday() as i64;
    let monday = *dt - Duration::days(days_since_monday);
    start_of_day(&monday)
}

/// First day of the month.
pub fn start_of_month(dt: &DateTime<Utc>) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(dt.year(), dt.month(), 1, 0, 0, 0)
        .single()
        .unwrap_or(*dt)
}

/// Last moment of the last day of the month.
pub fn end_of_month(dt: &DateTime<Utc>) -> DateTime<Utc> {
    let dim = days_in_month(dt.year(), dt.month());
    Utc.with_ymd_and_hms(dt.year(), dt.month(), dim, 23, 59, 59)
        .single()
        .unwrap_or(*dt)
}

/// Number of days in a given month.
pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Whether a year is a leap year.
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Whether two `DateTime`s fall on the same calendar day (UTC).
pub fn is_same_day(a: &DateTime<Utc>, b: &DateTime<Utc>) -> bool {
    a.year() == b.year() && a.month() == b.month() && a.day() == b.day()
}

/// Whether the given `DateTime` is today (UTC).
pub fn is_today(dt: &DateTime<Utc>) -> bool {
    is_same_day(dt, &Utc::now())
}

/// Whether the given `DateTime` falls on a Saturday or Sunday.
pub fn is_weekend(dt: &DateTime<Utc>) -> bool {
    matches!(dt.weekday(), Weekday::Sat | Weekday::Sun)
}

// ── Calendar Grid ───────────────────────────────────────────────

/// A calendar month grid.
#[derive(Debug, Clone)]
pub struct CalendarMonth {
    pub year: i32,
    pub month: u32,
    /// 6 weeks x 7 days. `None` for days outside the month. Week starts Monday.
    pub weeks: Vec<[Option<u32>; 7]>,
}

/// Build a 6-week calendar grid for a given month.
pub fn calendar_month(year: i32, month: u32) -> CalendarMonth {
    let dim = days_in_month(year, month);
    let first = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
    let start_weekday = first.weekday().num_days_from_monday() as usize;

    let mut weeks = Vec::new();
    let mut day = 1u32;

    for week_idx in 0..6 {
        let mut row = [None; 7];
        for col in 0..7 {
            let cell_offset = week_idx * 7 + col;
            if cell_offset >= start_weekday && day <= dim {
                row[col] = Some(day);
                day += 1;
            }
        }
        weeks.push(row);
    }

    CalendarMonth {
        year,
        month,
        weeks,
    }
}

// ── Date Range ──────────────────────────────────────────────────

/// A range of dates (inclusive of start, exclusive of end for iteration).
#[derive(Debug, Clone)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl DateRange {
    /// Whether the range contains the given datetime.
    pub fn contains(&self, dt: &DateTime<Utc>) -> bool {
        *dt >= self.start && *dt <= self.end
    }

    /// Whether this range overlaps with another.
    pub fn overlaps(&self, other: &DateRange) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Number of full days in the range.
    pub fn days(&self) -> i64 {
        (self.end - self.start).num_days()
    }

    /// Iterate over each day (at midnight UTC) from start to end (exclusive).
    pub fn iter_days(&self) -> impl Iterator<Item = DateTime<Utc>> + '_ {
        let total = self.days().max(0) as usize;
        (0..total).map(move |i| self.start + Duration::days(i as i64))
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dt(y: i32, m: u32, d: u32, h: u32, mi: u32, s: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, h, mi, s).unwrap()
    }

    #[test]
    fn time_ago_just_now() {
        let now = Utc::now();
        let result = format_relative(&now, &now);
        assert_eq!(result, "just now");
    }

    #[test]
    fn time_ago_five_minutes() {
        let base = dt(2026, 3, 8, 12, 0, 0);
        let five_min_ago = base - Duration::minutes(5);
        let result = format_relative(&five_min_ago, &base);
        assert_eq!(result, "5 minutes ago");
    }

    #[test]
    fn format_date_yyyy_mm_dd() {
        let d = dt(2026, 3, 8, 15, 30, 45);
        assert_eq!(format_date(&d, "YYYY-MM-DD"), "2026-03-08");
    }

    #[test]
    fn format_date_with_day_names() {
        let d = dt(2026, 3, 8, 0, 0, 0); // Sunday
        let result = format_date(&d, "dddd, MMMM DD, YYYY");
        assert_eq!(result, "Sunday, March 08, 2026");
    }

    #[test]
    fn duration_between_dates() {
        let a = dt(2024, 1, 1, 0, 0, 0);
        let b = dt(2026, 4, 5, 3, 30, 15);
        let parts = duration_between(&a, &b);
        assert_eq!(parts.years, 2);
        assert_eq!(parts.months, 3);
        assert_eq!(parts.days, 4);
        assert_eq!(parts.hours, 3);
        assert_eq!(parts.minutes, 30);
        assert_eq!(parts.seconds, 15);
    }

    #[test]
    fn format_duration_test() {
        let parts = DurationParts {
            years: 2,
            months: 3,
            days: 1,
            hours: 0,
            minutes: 0,
            seconds: 0,
        };
        assert_eq!(format_duration(&parts), "2 years, 3 months, 1 day");
        assert_eq!(format_duration_short(&parts), "2y 3mo 1d");
    }

    #[test]
    fn start_of_day_zeros_time() {
        let d = dt(2026, 3, 8, 15, 30, 45);
        let s = start_of_day(&d);
        assert_eq!(s.hour(), 0);
        assert_eq!(s.minute(), 0);
        assert_eq!(s.second(), 0);
        assert_eq!(s.day(), 8);
    }

    #[test]
    fn start_of_month_test() {
        let d = dt(2026, 3, 15, 10, 0, 0);
        let s = start_of_month(&d);
        assert_eq!(s.day(), 1);
        assert_eq!(s.hour(), 0);
    }

    #[test]
    fn days_in_month_feb_leap() {
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2025, 2), 28);
        assert_eq!(days_in_month(2000, 2), 29);
        assert_eq!(days_in_month(1900, 2), 28);
    }

    #[test]
    fn is_weekend_sat_sun() {
        let sat = dt(2026, 3, 7, 0, 0, 0); // Saturday
        let sun = dt(2026, 3, 8, 0, 0, 0); // Sunday
        let mon = dt(2026, 3, 9, 0, 0, 0); // Monday
        assert!(is_weekend(&sat));
        assert!(is_weekend(&sun));
        assert!(!is_weekend(&mon));
    }

    #[test]
    fn calendar_month_grid_shape() {
        let cal = calendar_month(2026, 3);
        assert_eq!(cal.weeks.len(), 6);
        assert_eq!(cal.year, 2026);
        assert_eq!(cal.month, 3);
        let has_day_1 = cal.weeks[0].iter().any(|d| *d == Some(1));
        assert!(has_day_1);
        let has_day_31 = cal.weeks.iter().any(|w| w.iter().any(|d| *d == Some(31)));
        assert!(has_day_31);
    }

    #[test]
    fn date_range_contains_overlaps() {
        let range1 = DateRange {
            start: dt(2026, 1, 1, 0, 0, 0),
            end: dt(2026, 1, 31, 23, 59, 59),
        };
        let range2 = DateRange {
            start: dt(2026, 1, 15, 0, 0, 0),
            end: dt(2026, 2, 15, 0, 0, 0),
        };

        assert!(range1.contains(&dt(2026, 1, 15, 12, 0, 0)));
        assert!(!range1.contains(&dt(2026, 2, 1, 0, 0, 0)));
        assert!(range1.overlaps(&range2));
    }

    #[test]
    fn is_same_day_test() {
        let a = dt(2026, 3, 8, 10, 0, 0);
        let b = dt(2026, 3, 8, 22, 30, 0);
        let c = dt(2026, 3, 9, 0, 0, 0);
        assert!(is_same_day(&a, &b));
        assert!(!is_same_day(&a, &c));
    }

    #[test]
    fn end_of_month_test() {
        let d = dt(2026, 2, 15, 0, 0, 0);
        let eom = end_of_month(&d);
        assert_eq!(eom.day(), 28);
        assert_eq!(eom.hour(), 23);
        assert_eq!(eom.minute(), 59);
    }
}
