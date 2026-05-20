//! Duration parsing and formatting — ISO 8601 and human-readable.
//!
//! Replaces date-fns `intervalToDuration` / ISO 8601 duration parsers with
//! a pure-Rust implementation. Supports ISO 8601 duration strings (P1Y2M3DT4H5M6S),
//! human-readable parsing ("2h 30m"), formatting, arithmetic, and comparison.

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

// ── Duration ───────────────────────────────────────────────────

/// A duration with calendar (year/month) and clock (day/hour/min/sec) components.
/// Unlike `chrono::Duration`, this preserves the original components —
/// "1 month" stays "1 month" rather than being converted to 30 days.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Duration {
    pub years: u32,
    pub months: u32,
    pub days: u32,
    pub hours: u32,
    pub minutes: u32,
    pub seconds: u32,
}

impl Duration {
    pub const ZERO: Duration = Duration { years: 0, months: 0, days: 0, hours: 0, minutes: 0, seconds: 0 };

    pub fn new(years: u32, months: u32, days: u32, hours: u32, minutes: u32, seconds: u32) -> Self {
        Self { years, months, days, hours, minutes, seconds }
    }

    /// Create from hours and minutes only.
    pub fn hm(hours: u32, minutes: u32) -> Self {
        Self { hours, minutes, ..Self::ZERO }
    }

    /// Create from days only.
    pub fn from_days(days: u32) -> Self {
        Self { days, ..Self::ZERO }
    }

    /// Create from seconds only.
    pub fn from_seconds(seconds: u32) -> Self {
        Self { seconds, ..Self::ZERO }
    }

    /// Is this a zero duration?
    pub fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    /// Total approximate seconds (using 365d/year, 30d/month).
    pub fn total_seconds(&self) -> u64 {
        let y = self.years as u64 * 365 * 86400;
        let m = self.months as u64 * 30 * 86400;
        let d = self.days as u64 * 86400;
        let h = self.hours as u64 * 3600;
        let mi = self.minutes as u64 * 60;
        y + m + d + h + mi + self.seconds as u64
    }

    /// Total approximate minutes.
    pub fn total_minutes(&self) -> u64 {
        self.total_seconds() / 60
    }

    /// Total approximate hours.
    pub fn total_hours(&self) -> u64 {
        self.total_seconds() / 3600
    }

    /// Total approximate days.
    pub fn total_days(&self) -> u64 {
        self.total_seconds() / 86400
    }

    /// Normalize: carry overflow (e.g. 90 seconds → 1 minute 30 seconds).
    /// Note: months/years are not normalized against each other by default
    /// since month lengths vary.
    pub fn normalize(&self) -> Self {
        let mut s = self.seconds;
        let mut mi = self.minutes;
        let mut h = self.hours;
        let mut d = self.days;
        let mut mo = self.months;
        let y = self.years;

        mi += s / 60;
        s %= 60;
        h += mi / 60;
        mi %= 60;
        d += h / 24;
        h %= 24;
        mo += d / 30; // approximate
        d %= 30;

        Self { years: y + mo / 12, months: mo % 12, days: d, hours: h, minutes: mi, seconds: s }
    }

    /// Add two durations component-wise.
    pub fn add(&self, other: &Duration) -> Self {
        Self {
            years: self.years + other.years,
            months: self.months + other.months,
            days: self.days + other.days,
            hours: self.hours + other.hours,
            minutes: self.minutes + other.minutes,
            seconds: self.seconds + other.seconds,
        }
    }

    /// Subtract component-wise (saturating at zero).
    pub fn subtract(&self, other: &Duration) -> Self {
        Self {
            years: self.years.saturating_sub(other.years),
            months: self.months.saturating_sub(other.months),
            days: self.days.saturating_sub(other.days),
            hours: self.hours.saturating_sub(other.hours),
            minutes: self.minutes.saturating_sub(other.minutes),
            seconds: self.seconds.saturating_sub(other.seconds),
        }
    }

    /// Duration between two datetimes (absolute value, clock components only).
    pub fn between(a: NaiveDateTime, b: NaiveDateTime) -> Self {
        let diff = if b > a { b - a } else { a - b };
        let total_secs = diff.num_seconds() as u64;
        let days = (total_secs / 86400) as u32;
        let hours = ((total_secs % 86400) / 3600) as u32;
        let minutes = ((total_secs % 3600) / 60) as u32;
        let seconds = (total_secs % 60) as u32;
        Self { days, hours, minutes, seconds, ..Self::ZERO }
    }
}

impl PartialOrd for Duration {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Duration {
    fn cmp(&self, other: &Self) -> Ordering {
        self.total_seconds().cmp(&other.total_seconds())
    }
}

impl std::fmt::Display for Duration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format_human(self))
    }
}

// ── ISO 8601 Parsing ───────────────────────────────────────────

/// Parse an ISO 8601 duration string (e.g. "P1Y2M3DT4H5M6S").
pub fn parse_iso(input: &str) -> Result<Duration, ParseError> {
    let s = input.trim();
    if !s.starts_with('P') {
        return Err(ParseError::InvalidFormat("must start with 'P'".into()));
    }

    let s = &s[1..];
    let (date_part, time_part) = if let Some(t_pos) = s.find('T') {
        (&s[..t_pos], Some(&s[t_pos + 1..]))
    } else {
        (s, None)
    };

    let mut dur = Duration::ZERO;

    // Parse date part: nY nM nD
    let mut num_buf = String::new();
    for ch in date_part.chars() {
        if ch.is_ascii_digit() {
            num_buf.push(ch);
        } else {
            let n: u32 = num_buf.parse().map_err(|_| ParseError::InvalidNumber(num_buf.clone()))?;
            num_buf.clear();
            match ch {
                'Y' => dur.years = n,
                'M' => dur.months = n,
                'D' => dur.days = n,
                'W' => dur.days = n * 7,
                _ => return Err(ParseError::InvalidFormat(format!("unexpected '{ch}' in date part"))),
            }
        }
    }

    // Parse time part: nH nM nS
    if let Some(tp) = time_part {
        num_buf.clear();
        for ch in tp.chars() {
            if ch.is_ascii_digit() {
                num_buf.push(ch);
            } else {
                let n: u32 = num_buf.parse().map_err(|_| ParseError::InvalidNumber(num_buf.clone()))?;
                num_buf.clear();
                match ch {
                    'H' => dur.hours = n,
                    'M' => dur.minutes = n,
                    'S' => dur.seconds = n,
                    _ => return Err(ParseError::InvalidFormat(format!("unexpected '{ch}' in time part"))),
                }
            }
        }
    }

    Ok(dur)
}

/// Parse a human-readable duration string (e.g. "2h 30m", "1 day 6 hours").
pub fn parse_human(input: &str) -> Result<Duration, ParseError> {
    let s = input.trim().to_lowercase();
    if s.is_empty() {
        return Err(ParseError::InvalidFormat("empty string".into()));
    }

    // Insert spaces between letter→digit transitions to handle "3d5h" → "3d 5h"
    let mut expanded = String::with_capacity(s.len() + 8);
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if i > 0 && c.is_ascii_digit() && chars[i - 1].is_ascii_alphabetic() {
            expanded.push(' ');
        }
        expanded.push(c);
    }

    let mut dur = Duration::ZERO;
    let mut num_buf = String::new();
    let mut unit_buf = String::new();
    let mut found_any = false;

    let tokens: Vec<&str> = expanded.split_whitespace().collect();
    let mut i = 0;

    while i < tokens.len() {
        let token = tokens[i];

        // Try to split "2h", "30m", "1d" etc.
        let (num_part, unit_part) = split_number_unit(token);

        if let Some(num_str) = num_part {
            let n: u32 = num_str.parse().map_err(|_| ParseError::InvalidNumber(num_str.to_string()))?;
            let unit = if let Some(u) = unit_part {
                u.to_string()
            } else if i + 1 < tokens.len() {
                // Unit is the next token.
                i += 1;
                tokens[i].to_string()
            } else {
                return Err(ParseError::InvalidFormat("number without unit".into()));
            };

            match unit.as_str() {
                "y" | "yr" | "yrs" | "year" | "years" => dur.years += n,
                "mo" | "mon" | "mos" | "month" | "months" => dur.months += n,
                "w" | "wk" | "wks" | "week" | "weeks" => dur.days += n * 7,
                "d" | "day" | "days" => dur.days += n,
                "h" | "hr" | "hrs" | "hour" | "hours" => dur.hours += n,
                "m" | "min" | "mins" | "minute" | "minutes" => dur.minutes += n,
                "s" | "sec" | "secs" | "second" | "seconds" => dur.seconds += n,
                other => return Err(ParseError::InvalidFormat(format!("unknown unit '{other}'"))),
            }
            found_any = true;
        }
        i += 1;
    }

    if !found_any {
        return Err(ParseError::InvalidFormat("no duration components found".into()));
    }

    Ok(dur)
}

/// Split "2h" into (Some("2"), Some("h")).
fn split_number_unit(s: &str) -> (Option<&str>, Option<&str>) {
    let boundary = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if boundary == 0 {
        return (None, None);
    }
    let num = &s[..boundary];
    let unit = if boundary < s.len() { Some(&s[boundary..]) } else { None };
    (Some(num), unit)
}

// ── Formatting ─────────────────────────────────────────────────

/// Format as ISO 8601 duration (e.g. "P1Y2M3DT4H5M6S").
pub fn format_iso(d: &Duration) -> String {
    let mut out = String::from("P");
    if d.years > 0 { out.push_str(&format!("{}Y", d.years)); }
    if d.months > 0 { out.push_str(&format!("{}M", d.months)); }
    if d.days > 0 { out.push_str(&format!("{}D", d.days)); }
    if d.hours > 0 || d.minutes > 0 || d.seconds > 0 {
        out.push('T');
        if d.hours > 0 { out.push_str(&format!("{}H", d.hours)); }
        if d.minutes > 0 { out.push_str(&format!("{}M", d.minutes)); }
        if d.seconds > 0 { out.push_str(&format!("{}S", d.seconds)); }
    }
    if out == "P" {
        out.push_str("T0S");
    }
    out
}

fn plural(n: u32) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// Format as human-readable (e.g. "1 year 2 months 3 days 4 hours 5 minutes 6 seconds").
pub fn format_human(d: &Duration) -> String {
    let mut parts = Vec::new();
    if d.years > 0 { parts.push(format!("{} year{}", d.years, plural(d.years))); }
    if d.months > 0 { parts.push(format!("{} month{}", d.months, plural(d.months))); }
    if d.days > 0 { parts.push(format!("{} day{}", d.days, plural(d.days))); }
    if d.hours > 0 { parts.push(format!("{} hour{}", d.hours, plural(d.hours))); }
    if d.minutes > 0 { parts.push(format!("{} minute{}", d.minutes, plural(d.minutes))); }
    if d.seconds > 0 { parts.push(format!("{} second{}", d.seconds, plural(d.seconds))); }
    if parts.is_empty() {
        "0 seconds".to_string()
    } else {
        parts.join(" ")
    }
}

/// Format as short human-readable (e.g. "1y 2mo 3d 4h 5m 6s").
pub fn format_short(d: &Duration) -> String {
    let mut parts = Vec::new();
    if d.years > 0 { parts.push(format!("{}y", d.years)); }
    if d.months > 0 { parts.push(format!("{}mo", d.months)); }
    if d.days > 0 { parts.push(format!("{}d", d.days)); }
    if d.hours > 0 { parts.push(format!("{}h", d.hours)); }
    if d.minutes > 0 { parts.push(format!("{}m", d.minutes)); }
    if d.seconds > 0 { parts.push(format!("{}s", d.seconds)); }
    if parts.is_empty() {
        "0s".to_string()
    } else {
        parts.join(" ")
    }
}

// ── Errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    InvalidFormat(String),
    InvalidNumber(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "invalid duration format: {msg}"),
            Self::InvalidNumber(n) => write!(f, "invalid number: {n}"),
        }
    }
}

impl std::error::Error for ParseError {}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn dt(y: i32, m: u32, d: u32, h: u32, mi: u32, s: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(h, mi, s).unwrap()
    }

    #[test]
    fn parse_iso_full() {
        let d = parse_iso("P1Y2M3DT4H5M6S").unwrap();
        assert_eq!(d, Duration::new(1, 2, 3, 4, 5, 6));
    }

    #[test]
    fn parse_iso_date_only() {
        let d = parse_iso("P30D").unwrap();
        assert_eq!(d, Duration::from_days(30));
    }

    #[test]
    fn parse_iso_time_only() {
        let d = parse_iso("PT2H30M").unwrap();
        assert_eq!(d, Duration::hm(2, 30));
    }

    #[test]
    fn parse_iso_weeks() {
        let d = parse_iso("P2W").unwrap();
        assert_eq!(d.days, 14);
    }

    #[test]
    fn parse_iso_invalid() {
        assert!(parse_iso("not a duration").is_err());
        assert!(parse_iso("").is_err());
    }

    #[test]
    fn parse_human_combined() {
        let d = parse_human("2h 30m").unwrap();
        assert_eq!(d.hours, 2);
        assert_eq!(d.minutes, 30);
    }

    #[test]
    fn parse_human_long_form() {
        let d = parse_human("1 day 6 hours").unwrap();
        assert_eq!(d.days, 1);
        assert_eq!(d.hours, 6);
    }

    #[test]
    fn parse_human_compact() {
        let d = parse_human("3d5h").unwrap();
        assert_eq!(d.days, 3);
        assert_eq!(d.hours, 5);
    }

    #[test]
    fn format_iso_roundtrip() {
        let d = Duration::new(1, 2, 3, 4, 5, 6);
        let s = format_iso(&d);
        assert_eq!(s, "P1Y2M3DT4H5M6S");
        let parsed = parse_iso(&s).unwrap();
        assert_eq!(parsed, d);
    }

    #[test]
    fn format_human_output() {
        let d = Duration::new(0, 0, 2, 3, 0, 0);
        assert_eq!(format_human(&d), "2 days 3 hours");
    }

    #[test]
    fn format_short_output() {
        let d = Duration::new(1, 0, 0, 2, 30, 0);
        assert_eq!(format_short(&d), "1y 2h 30m");
    }

    #[test]
    fn format_zero() {
        assert_eq!(format_human(&Duration::ZERO), "0 seconds");
        assert_eq!(format_iso(&Duration::ZERO), "PT0S");
    }

    #[test]
    fn add_durations() {
        let a = Duration::hm(2, 30);
        let b = Duration::hm(1, 45);
        let sum = a.add(&b);
        assert_eq!(sum.hours, 3);
        assert_eq!(sum.minutes, 75);
        let normalized = sum.normalize();
        assert_eq!(normalized.hours, 4);
        assert_eq!(normalized.minutes, 15);
    }

    #[test]
    fn subtract_durations() {
        let a = Duration::hm(5, 30);
        let b = Duration::hm(2, 15);
        let diff = a.subtract(&b);
        assert_eq!(diff.hours, 3);
        assert_eq!(diff.minutes, 15);
    }

    #[test]
    fn subtract_saturates() {
        let a = Duration::hm(1, 0);
        let b = Duration::hm(2, 0);
        let diff = a.subtract(&b);
        assert_eq!(diff.hours, 0);
    }

    #[test]
    fn between_datetimes() {
        let a = dt(2026, 3, 8, 10, 0, 0);
        let b = dt(2026, 3, 8, 12, 30, 45);
        let d = Duration::between(a, b);
        assert_eq!(d.hours, 2);
        assert_eq!(d.minutes, 30);
        assert_eq!(d.seconds, 45);
    }

    #[test]
    fn between_multiday() {
        let a = dt(2026, 3, 1, 0, 0, 0);
        let b = dt(2026, 3, 4, 6, 0, 0);
        let d = Duration::between(a, b);
        assert_eq!(d.days, 3);
        assert_eq!(d.hours, 6);
    }

    #[test]
    fn total_conversions() {
        let d = Duration::new(0, 0, 1, 2, 30, 0);
        assert_eq!(d.total_seconds(), 86400 + 7200 + 1800);
        assert_eq!(d.total_minutes(), (86400 + 7200 + 1800) / 60);
        assert_eq!(d.total_hours(), (86400 + 7200 + 1800) / 3600);
    }

    #[test]
    fn comparison() {
        let a = Duration::hm(2, 30);
        let b = Duration::hm(3, 0);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a.cmp(&a), Ordering::Equal);
    }

    #[test]
    fn normalize_overflow() {
        let d = Duration::new(0, 0, 0, 0, 0, 3661); // 3661 seconds
        let n = d.normalize();
        assert_eq!(n.hours, 1);
        assert_eq!(n.minutes, 1);
        assert_eq!(n.seconds, 1);
    }
}
