//! Date/time formatter — pattern-based, named formats, relative dates.
//!
//! Supports yyyy-MM-dd patterns, short/medium/long/full named formats,
//! relative date output ("yesterday", "in 3 days"), locale-aware month/day
//! names, era formatting, and week-of-year — pure Rust, no ICU dependency.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Date formatting errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DateFormatError {
    /// Unknown pattern token.
    UnknownToken(String),
    /// Invalid date components.
    InvalidDate(String),
}

impl fmt::Display for DateFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownToken(t) => write!(f, "unknown pattern token: {t}"),
            Self::InvalidDate(msg) => write!(f, "invalid date: {msg}"),
        }
    }
}

impl std::error::Error for DateFormatError {}

// ── DateTime ────────────────────────────────────────────────────

/// A simple date-time struct (no time zone).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateTime {
    pub year: i32,
    pub month: u32,  // 1..12
    pub day: u32,    // 1..31
    pub hour: u32,   // 0..23
    pub minute: u32, // 0..59
    pub second: u32, // 0..59
}

impl DateTime {
    /// Create a new date-time.
    pub fn new(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: u32) -> Self {
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    /// Create a date (midnight).
    pub fn date(year: i32, month: u32, day: u32) -> Self {
        Self::new(year, month, day, 0, 0, 0)
    }

    /// Day of week (0 = Sunday, 6 = Saturday) using Tomohiko Sakamoto's algorithm.
    pub fn day_of_week(&self) -> u32 {
        let t = [0i32, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
        let mut y = self.year;
        if self.month < 3 {
            y -= 1;
        }
        ((y + y / 4 - y / 100 + y / 400 + t[(self.month - 1) as usize] + self.day as i32)
            .rem_euclid(7)) as u32
    }

    /// Day of year (1-based).
    pub fn day_of_year(&self) -> u32 {
        let leap = is_leap_year(self.year);
        let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut doy: u32 = 0;
        for i in 0..(self.month - 1) as usize {
            doy += month_days[i];
        }
        doy + self.day
    }

    /// ISO week number (1-based).
    pub fn week_of_year(&self) -> u32 {
        let doy = self.day_of_year();
        let dow = self.day_of_week(); // 0=Sun
        // ISO: Monday=1, Sunday=7
        let iso_dow = if dow == 0 { 7 } else { dow };
        // ISO week calculation (simplified)
        let w = (doy + 7 - iso_dow) / 7;
        if w == 0 { 1 } else { w }
    }

    /// Era: "BC" or "AD".
    pub fn era(&self) -> &'static str {
        if self.year <= 0 { "BC" } else { "AD" }
    }

    /// 12-hour clock hour.
    pub fn hour12(&self) -> u32 {
        match self.hour % 12 {
            0 => 12,
            h => h,
        }
    }

    /// AM/PM designator.
    pub fn am_pm(&self) -> &'static str {
        if self.hour < 12 { "AM" } else { "PM" }
    }

    /// Difference in days from `other` (self - other).
    pub fn days_from(&self, other: &DateTime) -> i64 {
        self.to_day_number() - other.to_day_number()
    }

    /// Convert to a day number (for relative calculations).
    fn to_day_number(&self) -> i64 {
        // Simplified Julian Day Number
        let a = (14 - self.month as i64) / 12;
        let y = self.year as i64 + 4800 - a;
        let m = self.month as i64 + 12 * a - 3;
        self.day as i64 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

// ── Locale data ─────────────────────────────────────────────────

/// Locale-specific month and day names.
#[derive(Debug, Clone)]
pub struct DateLocale {
    pub months_long: [&'static str; 12],
    pub months_short: [&'static str; 12],
    pub days_long: [&'static str; 7],   // Sunday first
    pub days_short: [&'static str; 7],
}

impl DateLocale {
    /// English locale.
    pub fn english() -> Self {
        Self {
            months_long: [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            ],
            months_short: [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ],
            days_long: [
                "Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday",
            ],
            days_short: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        }
    }

    /// French locale.
    pub fn french() -> Self {
        Self {
            months_long: [
                "janvier", "février", "mars", "avril", "mai", "juin",
                "juillet", "août", "septembre", "octobre", "novembre", "décembre",
            ],
            months_short: [
                "janv.", "févr.", "mars", "avr.", "mai", "juin",
                "juil.", "août", "sept.", "oct.", "nov.", "déc.",
            ],
            days_long: [
                "dimanche", "lundi", "mardi", "mercredi",
                "jeudi", "vendredi", "samedi",
            ],
            days_short: ["dim.", "lun.", "mar.", "mer.", "jeu.", "ven.", "sam."],
        }
    }
}

impl Default for DateLocale {
    fn default() -> Self {
        Self::english()
    }
}

// ── Named formats ───────────────────────────────────────────────

/// Predefined format styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DateStyle {
    /// "3/14/2026"
    Short,
    /// "Mar 14, 2026"
    Medium,
    /// "March 14, 2026"
    Long,
    /// "Saturday, March 14, 2026"
    Full,
}

impl DateStyle {
    /// Return the pattern string for this style.
    pub fn pattern(&self) -> &'static str {
        match self {
            Self::Short => "M/d/yyyy",
            Self::Medium => "MMM d, yyyy",
            Self::Long => "MMMM d, yyyy",
            Self::Full => "EEEE, MMMM d, yyyy",
        }
    }
}

/// Predefined time format styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeStyle {
    /// "2:30 PM"
    Short,
    /// "2:30:00 PM"
    Medium,
}

impl TimeStyle {
    /// Return the pattern string for this style.
    pub fn pattern(&self) -> &'static str {
        match self {
            Self::Short => "h:mm a",
            Self::Medium => "h:mm:ss a",
        }
    }
}

// ── Formatter ───────────────────────────────────────────────────

/// Date/time formatter.
#[derive(Debug, Clone)]
pub struct DateFormatter {
    pub locale: DateLocale,
}

impl Default for DateFormatter {
    fn default() -> Self {
        Self {
            locale: DateLocale::default(),
        }
    }
}

impl DateFormatter {
    /// Create a new formatter with English locale.
    pub fn new() -> Self {
        Self::default()
    }

    /// Format a date-time using a pattern string.
    ///
    /// Supported tokens: yyyy, yy, MMMM, MMM, MM, M, dd, d,
    /// EEEE, EEE, HH, H, hh, h, mm, m, ss, s, a, G, w.
    pub fn format_pattern(&self, dt: &DateTime, pattern: &str) -> String {
        let mut result = String::new();
        let chars: Vec<char> = pattern.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Try to match longest token first
            if let Some(token_len) = self.try_token(&chars, i, dt, &mut result) {
                i += token_len;
            } else {
                // Literal character
                result.push(chars[i]);
                i += 1;
            }
        }
        result
    }

    fn try_token(
        &self,
        chars: &[char],
        start: usize,
        dt: &DateTime,
        out: &mut String,
    ) -> Option<usize> {
        let remaining = &chars[start..];
        let s: String = remaining.iter().collect();

        // 4-char tokens
        if s.starts_with("yyyy") {
            out.push_str(&format!("{:04}", dt.year.unsigned_abs()));
            return Some(4);
        }
        if s.starts_with("MMMM") {
            out.push_str(self.locale.months_long[(dt.month - 1) as usize]);
            return Some(4);
        }
        if s.starts_with("EEEE") {
            out.push_str(self.locale.days_long[dt.day_of_week() as usize]);
            return Some(4);
        }

        // 3-char tokens
        if s.starts_with("MMM") {
            out.push_str(self.locale.months_short[(dt.month - 1) as usize]);
            return Some(3);
        }
        if s.starts_with("EEE") {
            out.push_str(self.locale.days_short[dt.day_of_week() as usize]);
            return Some(3);
        }

        // 2-char tokens
        if s.starts_with("yy") {
            out.push_str(&format!("{:02}", dt.year % 100));
            return Some(2);
        }
        if s.starts_with("MM") {
            out.push_str(&format!("{:02}", dt.month));
            return Some(2);
        }
        if s.starts_with("dd") {
            out.push_str(&format!("{:02}", dt.day));
            return Some(2);
        }
        if s.starts_with("HH") {
            out.push_str(&format!("{:02}", dt.hour));
            return Some(2);
        }
        if s.starts_with("hh") {
            out.push_str(&format!("{:02}", dt.hour12()));
            return Some(2);
        }
        if s.starts_with("mm") {
            out.push_str(&format!("{:02}", dt.minute));
            return Some(2);
        }
        if s.starts_with("ss") {
            out.push_str(&format!("{:02}", dt.second));
            return Some(2);
        }

        // 1-char tokens (only if not followed by same char → already handled above)
        if remaining[0] == 'M' && (remaining.len() < 2 || remaining[1] != 'M') {
            out.push_str(&dt.month.to_string());
            return Some(1);
        }
        if remaining[0] == 'd' && (remaining.len() < 2 || remaining[1] != 'd') {
            out.push_str(&dt.day.to_string());
            return Some(1);
        }
        if remaining[0] == 'H' && (remaining.len() < 2 || remaining[1] != 'H') {
            out.push_str(&dt.hour.to_string());
            return Some(1);
        }
        if remaining[0] == 'h' && (remaining.len() < 2 || remaining[1] != 'h') {
            out.push_str(&dt.hour12().to_string());
            return Some(1);
        }
        if remaining[0] == 'm' && (remaining.len() < 2 || remaining[1] != 'm') {
            out.push_str(&dt.minute.to_string());
            return Some(1);
        }
        if remaining[0] == 's' && (remaining.len() < 2 || remaining[1] != 's') {
            out.push_str(&dt.second.to_string());
            return Some(1);
        }
        if remaining[0] == 'a' {
            out.push_str(dt.am_pm());
            return Some(1);
        }
        if remaining[0] == 'G' {
            out.push_str(dt.era());
            return Some(1);
        }
        if remaining[0] == 'w' {
            out.push_str(&dt.week_of_year().to_string());
            return Some(1);
        }

        None
    }

    /// Format with a named date style.
    pub fn format_date_style(&self, dt: &DateTime, style: DateStyle) -> String {
        self.format_pattern(dt, style.pattern())
    }

    /// Format with a named time style.
    pub fn format_time_style(&self, dt: &DateTime, style: TimeStyle) -> String {
        self.format_pattern(dt, style.pattern())
    }

    /// Format as a relative date ("yesterday", "in 3 days", "2 days ago").
    pub fn format_relative(&self, dt: &DateTime, now: &DateTime) -> String {
        let diff = dt.days_from(now);
        match diff {
            0 => "today".to_string(),
            1 => "tomorrow".to_string(),
            -1 => "yesterday".to_string(),
            d if d > 1 => format!("in {d} days"),
            d => format!("{} days ago", -d),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dt() -> DateTime {
        DateTime::new(2026, 3, 14, 14, 30, 45)
    }

    #[test]
    fn iso_pattern() {
        let f = DateFormatter::new();
        assert_eq!(f.format_pattern(&sample_dt(), "yyyy-MM-dd"), "2026-03-14");
    }

    #[test]
    fn time_pattern() {
        let f = DateFormatter::new();
        assert_eq!(
            f.format_pattern(&sample_dt(), "HH:mm:ss"),
            "14:30:45"
        );
    }

    #[test]
    fn twelve_hour_clock() {
        let f = DateFormatter::new();
        assert_eq!(f.format_pattern(&sample_dt(), "h:mm a"), "2:30 PM");
    }

    #[test]
    fn short_style() {
        let f = DateFormatter::new();
        assert_eq!(f.format_date_style(&sample_dt(), DateStyle::Short), "3/14/2026");
    }

    #[test]
    fn medium_style() {
        let f = DateFormatter::new();
        assert_eq!(
            f.format_date_style(&sample_dt(), DateStyle::Medium),
            "Mar 14, 2026"
        );
    }

    #[test]
    fn long_style() {
        let f = DateFormatter::new();
        assert_eq!(
            f.format_date_style(&sample_dt(), DateStyle::Long),
            "March 14, 2026"
        );
    }

    #[test]
    fn full_style() {
        let f = DateFormatter::new();
        let result = f.format_date_style(&sample_dt(), DateStyle::Full);
        assert_eq!(result, "Saturday, March 14, 2026");
    }

    #[test]
    fn relative_today() {
        let f = DateFormatter::new();
        let now = DateTime::date(2026, 3, 14);
        assert_eq!(f.format_relative(&now, &now), "today");
    }

    #[test]
    fn relative_yesterday_tomorrow() {
        let f = DateFormatter::new();
        let now = DateTime::date(2026, 3, 14);
        let yesterday = DateTime::date(2026, 3, 13);
        let tomorrow = DateTime::date(2026, 3, 15);
        assert_eq!(f.format_relative(&yesterday, &now), "yesterday");
        assert_eq!(f.format_relative(&tomorrow, &now), "tomorrow");
    }

    #[test]
    fn relative_days_ago() {
        let f = DateFormatter::new();
        let now = DateTime::date(2026, 3, 14);
        let past = DateTime::date(2026, 3, 10);
        assert_eq!(f.format_relative(&past, &now), "4 days ago");
    }

    #[test]
    fn relative_in_days() {
        let f = DateFormatter::new();
        let now = DateTime::date(2026, 3, 14);
        let future = DateTime::date(2026, 3, 17);
        assert_eq!(f.format_relative(&future, &now), "in 3 days");
    }

    #[test]
    fn era_ad() {
        let f = DateFormatter::new();
        let dt = DateTime::date(2026, 1, 1);
        assert_eq!(f.format_pattern(&dt, "G"), "AD");
    }

    #[test]
    fn week_of_year() {
        let dt = DateTime::date(2026, 1, 15);
        assert!(dt.week_of_year() >= 2);
    }

    #[test]
    fn french_locale() {
        let f = DateFormatter {
            locale: DateLocale::french(),
        };
        let dt = DateTime::date(2026, 3, 14);
        assert_eq!(f.format_pattern(&dt, "d MMMM yyyy"), "14 mars 2026");
    }

    #[test]
    fn midnight_12hr() {
        let dt = DateTime::new(2026, 1, 1, 0, 0, 0);
        assert_eq!(dt.hour12(), 12);
        assert_eq!(dt.am_pm(), "AM");
    }

    #[test]
    fn day_of_week_known() {
        // 2026-03-14 is a Saturday
        let dt = DateTime::date(2026, 3, 14);
        assert_eq!(dt.day_of_week(), 6); // Saturday
    }
}
