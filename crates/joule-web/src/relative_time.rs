//! Relative time formatting — "5 minutes ago", "in 2 hours", etc.
//!
//! Replaces timeago.js / date-fns `formatDistanceToNow` with a configurable,
//! locale-aware pure-Rust formatter. Supports past, future, and auto-update
//! interval suggestions.

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};

// ── Thresholds ─────────────────────────────────────────────────

/// Configurable thresholds for switching between units.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Thresholds {
    /// Seconds under which we say "just now".
    pub just_now_seconds: i64,
    /// Seconds threshold for switching from seconds to minutes.
    pub seconds_limit: i64,
    /// Minutes threshold for switching from minutes to hours.
    pub minutes_limit: i64,
    /// Hours threshold for switching from hours to days.
    pub hours_limit: i64,
    /// Days threshold for switching from days to weeks.
    pub days_limit: i64,
    /// Days threshold for switching from weeks to months.
    pub weeks_limit_days: i64,
    /// Days threshold for switching from months to years.
    pub months_limit_days: i64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            just_now_seconds: 10,
            seconds_limit: 60,
            minutes_limit: 60,
            hours_limit: 24,
            days_limit: 7,
            weeks_limit_days: 30,
            months_limit_days: 365,
        }
    }
}

// ── Locale ─────────────────────────────────────────────────────

/// Locale for relative time strings.
pub trait RelativeTimeLocale {
    fn just_now(&self) -> &str;
    fn seconds_ago(&self, n: i64) -> String;
    fn minutes_ago(&self, n: i64) -> String;
    fn hours_ago(&self, n: i64) -> String;
    fn yesterday(&self) -> String;
    fn days_ago(&self, n: i64) -> String;
    fn weeks_ago(&self, n: i64) -> String;
    fn last_month(&self) -> String;
    fn months_ago(&self, n: i64) -> String;
    fn last_year(&self) -> String;
    fn years_ago(&self, n: i64) -> String;

    fn in_seconds(&self, n: i64) -> String;
    fn in_minutes(&self, n: i64) -> String;
    fn in_hours(&self, n: i64) -> String;
    fn tomorrow(&self) -> String;
    fn in_days(&self, n: i64) -> String;
    fn in_weeks(&self, n: i64) -> String;
    fn in_months(&self, n: i64) -> String;
    fn in_years(&self, n: i64) -> String;
}

/// English locale (default).
#[derive(Debug, Clone, Copy)]
pub struct EnglishLocale;

fn plural(n: i64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

impl RelativeTimeLocale for EnglishLocale {
    fn just_now(&self) -> &str { "just now" }
    fn seconds_ago(&self, n: i64) -> String { format!("{n} second{} ago", plural(n)) }
    fn minutes_ago(&self, n: i64) -> String { format!("{n} minute{} ago", plural(n)) }
    fn hours_ago(&self, n: i64) -> String { format!("{n} hour{} ago", plural(n)) }
    fn yesterday(&self) -> String { "yesterday".to_string() }
    fn days_ago(&self, n: i64) -> String { format!("{n} day{} ago", plural(n)) }
    fn weeks_ago(&self, n: i64) -> String { format!("{n} week{} ago", plural(n)) }
    fn last_month(&self) -> String { "last month".to_string() }
    fn months_ago(&self, n: i64) -> String { format!("{n} month{} ago", plural(n)) }
    fn last_year(&self) -> String { "last year".to_string() }
    fn years_ago(&self, n: i64) -> String { format!("{n} year{} ago", plural(n)) }

    fn in_seconds(&self, n: i64) -> String { format!("in {n} second{}", plural(n)) }
    fn in_minutes(&self, n: i64) -> String { format!("in {n} minute{}", plural(n)) }
    fn in_hours(&self, n: i64) -> String { format!("in {n} hour{}", plural(n)) }
    fn tomorrow(&self) -> String { "tomorrow".to_string() }
    fn in_days(&self, n: i64) -> String { format!("in {n} day{}", plural(n)) }
    fn in_weeks(&self, n: i64) -> String { format!("in {n} week{}", plural(n)) }
    fn in_months(&self, n: i64) -> String { format!("in {n} month{}", plural(n)) }
    fn in_years(&self, n: i64) -> String { format!("in {n} year{}", plural(n)) }
}

// ── Formatter ──────────────────────────────────────────────────

/// Relative time formatter with configurable thresholds and locale.
pub struct RelativeTimeFormatter<L: RelativeTimeLocale = EnglishLocale> {
    pub thresholds: Thresholds,
    pub locale: L,
}

impl RelativeTimeFormatter<EnglishLocale> {
    pub fn new() -> Self {
        Self {
            thresholds: Thresholds::default(),
            locale: EnglishLocale,
        }
    }
}

impl Default for RelativeTimeFormatter<EnglishLocale> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: RelativeTimeLocale> RelativeTimeFormatter<L> {
    pub fn with_locale<L2: RelativeTimeLocale>(self, locale: L2) -> RelativeTimeFormatter<L2> {
        RelativeTimeFormatter {
            thresholds: self.thresholds,
            locale,
        }
    }

    pub fn with_thresholds(mut self, thresholds: Thresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Format the relative time between `dt` and `now`.
    pub fn format(&self, dt: NaiveDateTime, now: NaiveDateTime) -> String {
        let diff_secs = (now - dt).num_seconds();
        if diff_secs >= 0 {
            self.format_past(diff_secs)
        } else {
            self.format_future(-diff_secs)
        }
    }

    /// Format a `DateTime<Utc>` relative to another `DateTime<Utc>`.
    pub fn format_utc(&self, dt: DateTime<Utc>, now: DateTime<Utc>) -> String {
        self.format(dt.naive_utc(), now.naive_utc())
    }

    fn format_past(&self, secs: i64) -> String {
        let t = &self.thresholds;
        let l = &self.locale;

        if secs < t.just_now_seconds {
            return l.just_now().to_string();
        }
        if secs < t.seconds_limit {
            return l.seconds_ago(secs);
        }
        let mins = secs / 60;
        if mins < t.minutes_limit {
            return l.minutes_ago(mins);
        }
        let hours = secs / 3600;
        if hours < t.hours_limit {
            return l.hours_ago(hours);
        }
        let days = secs / 86400;
        if days == 1 {
            return l.yesterday();
        }
        if days < t.days_limit {
            return l.days_ago(days);
        }
        if days < t.weeks_limit_days {
            let weeks = days / 7;
            return l.weeks_ago(weeks);
        }
        if days < t.months_limit_days {
            let months = days / 30;
            if months <= 1 {
                return l.last_month();
            }
            return l.months_ago(months);
        }
        let years = days / 365;
        if years <= 1 {
            return l.last_year();
        }
        l.years_ago(years)
    }

    fn format_future(&self, secs: i64) -> String {
        let t = &self.thresholds;
        let l = &self.locale;

        if secs < t.just_now_seconds {
            return l.just_now().to_string();
        }
        if secs < t.seconds_limit {
            return l.in_seconds(secs);
        }
        let mins = secs / 60;
        if mins < t.minutes_limit {
            return l.in_minutes(mins);
        }
        let hours = secs / 3600;
        if hours < t.hours_limit {
            return l.in_hours(hours);
        }
        let days = secs / 86400;
        if days == 1 {
            return l.tomorrow();
        }
        if days < t.days_limit {
            return l.in_days(days);
        }
        if days < t.weeks_limit_days {
            return l.in_weeks(days / 7);
        }
        if days < t.months_limit_days {
            return l.in_months(days / 30);
        }
        l.in_years(days / 365)
    }

    /// Suggest an auto-update interval (in seconds) based on how far away
    /// the datetime is. Closer times need more frequent updates.
    pub fn suggested_update_interval(&self, dt: NaiveDateTime, now: NaiveDateTime) -> u64 {
        let secs = (now - dt).num_seconds().unsigned_abs();
        if secs < 60 {
            1 // update every second
        } else if secs < 3600 {
            30 // every 30 seconds
        } else if secs < 86400 {
            300 // every 5 minutes
        } else if secs < 86400 * 7 {
            3600 // every hour
        } else {
            86400 // daily
        }
    }
}

// ── Convenience Functions ──────────────────────────────────────

/// Format relative time with default English locale and thresholds.
pub fn format_relative(dt: NaiveDateTime, now: NaiveDateTime) -> String {
    RelativeTimeFormatter::new().format(dt, now)
}

/// Format relative time from `DateTime<Utc>`.
pub fn format_relative_utc(dt: DateTime<Utc>, now: DateTime<Utc>) -> String {
    RelativeTimeFormatter::new().format_utc(dt, now)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn dt(y: i32, m: u32, d: u32, h: u32, mi: u32, s: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(h, mi, s).unwrap()
    }

    fn now() -> NaiveDateTime {
        dt(2026, 3, 8, 12, 0, 0)
    }

    #[test]
    fn just_now() {
        let result = format_relative(dt(2026, 3, 8, 11, 59, 55), now());
        assert_eq!(result, "just now");
    }

    #[test]
    fn seconds_ago() {
        let result = format_relative(dt(2026, 3, 8, 11, 59, 30), now());
        assert_eq!(result, "30 seconds ago");
    }

    #[test]
    fn minutes_ago() {
        let result = format_relative(dt(2026, 3, 8, 11, 55, 0), now());
        assert_eq!(result, "5 minutes ago");
    }

    #[test]
    fn hours_ago() {
        let result = format_relative(dt(2026, 3, 8, 10, 0, 0), now());
        assert_eq!(result, "2 hours ago");
    }

    #[test]
    fn yesterday() {
        let result = format_relative(dt(2026, 3, 7, 12, 0, 0), now());
        assert_eq!(result, "yesterday");
    }

    #[test]
    fn days_ago() {
        let result = format_relative(dt(2026, 3, 5, 12, 0, 0), now());
        assert_eq!(result, "3 days ago");
    }

    #[test]
    fn weeks_ago() {
        let result = format_relative(dt(2026, 2, 22, 12, 0, 0), now());
        assert_eq!(result, "2 weeks ago");
    }

    #[test]
    fn last_month() {
        // 35 days ago crosses the weeks_limit_days (30) threshold
        let result = format_relative(dt(2026, 2, 1, 12, 0, 0), now());
        assert_eq!(result, "last month");
    }

    #[test]
    fn months_ago() {
        let result = format_relative(dt(2025, 9, 8, 12, 0, 0), now());
        assert_eq!(result, "6 months ago");
    }

    #[test]
    fn last_year() {
        let result = format_relative(dt(2025, 3, 8, 12, 0, 0), now());
        assert_eq!(result, "last year");
    }

    #[test]
    fn years_ago() {
        let result = format_relative(dt(2023, 3, 8, 12, 0, 0), now());
        assert_eq!(result, "3 years ago");
    }

    #[test]
    fn future_minutes() {
        let result = format_relative(dt(2026, 3, 8, 12, 5, 0), now());
        assert_eq!(result, "in 5 minutes");
    }

    #[test]
    fn future_tomorrow() {
        let result = format_relative(dt(2026, 3, 9, 12, 0, 0), now());
        assert_eq!(result, "tomorrow");
    }

    #[test]
    fn future_weeks() {
        let result = format_relative(dt(2026, 3, 22, 12, 0, 0), now());
        assert_eq!(result, "in 2 weeks");
    }

    #[test]
    fn singular_forms() {
        let result = format_relative(dt(2026, 3, 8, 11, 59, 49), now());
        assert_eq!(result, "11 seconds ago");
        let result = format_relative(dt(2026, 3, 8, 11, 59, 0), now());
        assert_eq!(result, "1 minute ago");
        let result = format_relative(dt(2026, 3, 8, 11, 0, 0), now());
        assert_eq!(result, "1 hour ago");
    }

    #[test]
    fn custom_thresholds() {
        let fmt = RelativeTimeFormatter::new().with_thresholds(Thresholds {
            just_now_seconds: 30,
            ..Thresholds::default()
        });
        // 20 seconds ago should now be "just now" with 30s threshold.
        let result = fmt.format(dt(2026, 3, 8, 11, 59, 40), now());
        assert_eq!(result, "just now");
    }

    #[test]
    fn update_interval_suggestion() {
        let fmt = RelativeTimeFormatter::new();
        // Recent: update every second.
        assert_eq!(fmt.suggested_update_interval(dt(2026, 3, 8, 11, 59, 50), now()), 1);
        // Minutes ago: every 30s.
        assert_eq!(fmt.suggested_update_interval(dt(2026, 3, 8, 11, 50, 0), now()), 30);
        // Hours ago: every 5 min.
        assert_eq!(fmt.suggested_update_interval(dt(2026, 3, 8, 8, 0, 0), now()), 300);
    }
}
