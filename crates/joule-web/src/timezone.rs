//! Timezone handling — offsets, DST transitions, and conversions.
//!
//! Replaces Luxon / date-fns-tz with a pure-Rust timezone model.
//! No IANA database dependency — ships a curated list of common timezones
//! with simplified DST rules sufficient for display and conversion.

use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike, Weekday};
use serde::{Deserialize, Serialize};

// ── Offset ─────────────────────────────────────────────────────

/// A UTC offset in hours and minutes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TzOffset {
    /// Hours from UTC (-12 to +14).
    pub hours: i8,
    /// Additional minutes (0 or 30 or 45).
    pub minutes: i8,
}

impl TzOffset {
    pub const fn new(hours: i8, minutes: i8) -> Self {
        Self { hours, minutes }
    }

    /// Total offset in minutes.
    pub fn total_minutes(&self) -> i32 {
        self.hours as i32 * 60 + self.minutes as i32
    }

    /// Total offset in seconds.
    pub fn total_seconds(&self) -> i64 {
        self.total_minutes() as i64 * 60
    }

    /// Format as "+HH:MM" or "-HH:MM".
    pub fn format(&self) -> String {
        let sign = if self.total_minutes() >= 0 { '+' } else { '-' };
        let abs_h = self.hours.unsigned_abs();
        let abs_m = self.minutes.unsigned_abs();
        format!("{sign}{abs_h:02}:{abs_m:02}")
    }
}

impl std::fmt::Display for TzOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

// ── DST Rule ───────────────────────────────────────────────────

/// Simplified DST transition rule: "second Sunday of March at 2:00 AM".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DstTransition {
    pub month: u32,
    /// Nth occurrence of weekday (1-based). Negative means from end.
    pub nth_weekday: i8,
    pub weekday: Weekday,
    pub hour: u32,
}

impl DstTransition {
    /// Find the actual date of this transition in the given year.
    pub fn date_in_year(&self, year: i32) -> Option<NaiveDate> {
        let dim = days_in_month(year, self.month);
        let mut matches = Vec::new();
        for day in 1..=dim {
            if let Some(d) = NaiveDate::from_ymd_opt(year, self.month, day) {
                if d.weekday() == self.weekday {
                    matches.push(d);
                }
            }
        }
        if self.nth_weekday > 0 {
            matches.get((self.nth_weekday - 1) as usize).copied()
        } else {
            let idx = matches.len() as i8 + self.nth_weekday;
            if idx >= 0 { matches.get(idx as usize).copied() } else { None }
        }
    }

    /// The datetime of this transition in the given year (local time).
    pub fn datetime_in_year(&self, year: i32) -> Option<NaiveDateTime> {
        self.date_in_year(year).and_then(|d| d.and_hms_opt(self.hour, 0, 0))
    }
}

/// DST rules: when to spring forward and fall back, and by how much.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DstRules {
    pub start: DstTransition,
    pub end: DstTransition,
    /// Additional minutes added during DST (usually 60).
    pub offset_minutes: i32,
}

// ── Named Timezone ─────────────────────────────────────────────

/// A named timezone with standard offset and optional DST rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedTimezone {
    pub name: &'static str,
    pub abbreviation: &'static str,
    pub dst_abbreviation: Option<&'static str>,
    pub standard_offset: TzOffset,
    pub dst_rules: Option<DstRules>,
}

impl NamedTimezone {
    /// Is DST active at the given UTC datetime?
    pub fn is_dst(&self, utc_dt: NaiveDateTime) -> bool {
        let Some(rules) = &self.dst_rules else { return false };
        let local_approx = utc_dt + chrono::Duration::seconds(self.standard_offset.total_seconds());
        let year = local_approx.year();
        let Some(start) = rules.start.datetime_in_year(year) else { return false };
        let Some(end) = rules.end.datetime_in_year(year) else { return false };

        if start < end {
            // Northern hemisphere: DST between start and end.
            local_approx >= start && local_approx < end
        } else {
            // Southern hemisphere: DST from start through year-end and year-start through end.
            local_approx >= start || local_approx < end
        }
    }

    /// Get the current offset (accounting for DST) at a given UTC datetime.
    pub fn offset_at(&self, utc_dt: NaiveDateTime) -> TzOffset {
        if self.is_dst(utc_dt) {
            let extra = self.dst_rules.as_ref().map_or(0, |r| r.offset_minutes);
            let total = self.standard_offset.total_minutes() + extra;
            TzOffset::new((total / 60) as i8, (total % 60) as i8)
        } else {
            self.standard_offset
        }
    }

    /// The abbreviation to display at a given UTC datetime.
    pub fn abbreviation_at(&self, utc_dt: NaiveDateTime) -> &str {
        if self.is_dst(utc_dt) {
            self.dst_abbreviation.unwrap_or(self.abbreviation)
        } else {
            self.abbreviation
        }
    }

    /// Convert a UTC datetime to this timezone's local time.
    pub fn to_local(&self, utc_dt: NaiveDateTime) -> NaiveDateTime {
        let offset = self.offset_at(utc_dt);
        utc_dt + chrono::Duration::seconds(offset.total_seconds())
    }

    /// Convert a local datetime in this timezone to UTC.
    /// Note: ambiguous times near DST transitions are resolved using standard offset.
    pub fn to_utc(&self, local_dt: NaiveDateTime) -> NaiveDateTime {
        // First approximation using standard offset.
        let approx_utc = local_dt - chrono::Duration::seconds(self.standard_offset.total_seconds());
        let actual_offset = self.offset_at(approx_utc);
        local_dt - chrono::Duration::seconds(actual_offset.total_seconds())
    }

    /// Format a UTC datetime with timezone abbreviation.
    pub fn format_datetime(&self, utc_dt: NaiveDateTime, fmt: &str) -> String {
        let local = self.to_local(utc_dt);
        let abbr = self.abbreviation_at(utc_dt);
        // Simple template: replace %Z with abbreviation, %Y %m %d %H %M %S with values.
        fmt.replace("%Y", &format!("{:04}", local.year()))
            .replace("%m", &format!("{:02}", local.month()))
            .replace("%d", &format!("{:02}", local.day()))
            .replace("%H", &format!("{:02}", local.hour()))
            .replace("%M", &format!("{:02}", local.minute()))
            .replace("%S", &format!("{:02}", local.second()))
            .replace("%Z", abbr)
    }
}

// ── Convert Between Timezones ──────────────────────────────────

/// Convert a datetime from one timezone to another.
pub fn convert(dt_local: NaiveDateTime, from: &NamedTimezone, to: &NamedTimezone) -> NaiveDateTime {
    let utc = from.to_utc(dt_local);
    to.to_local(utc)
}

// ── Common Timezones ───────────────────────────────────────────

// US DST: second Sunday of March at 2am → first Sunday of November at 2am.
const US_DST_START: DstTransition = DstTransition { month: 3, nth_weekday: 2, weekday: Weekday::Sun, hour: 2 };
const US_DST_END: DstTransition = DstTransition { month: 11, nth_weekday: 1, weekday: Weekday::Sun, hour: 2 };
const US_DST: DstRules = DstRules { start: US_DST_START, end: US_DST_END, offset_minutes: 60 };

// EU DST: last Sunday of March at 1am UTC → last Sunday of October at 1am UTC.
const EU_DST_START: DstTransition = DstTransition { month: 3, nth_weekday: -1, weekday: Weekday::Sun, hour: 1 };
const EU_DST_END: DstTransition = DstTransition { month: 10, nth_weekday: -1, weekday: Weekday::Sun, hour: 1 };
const EU_DST: DstRules = DstRules { start: EU_DST_START, end: EU_DST_END, offset_minutes: 60 };

pub const UTC: NamedTimezone = NamedTimezone {
    name: "UTC", abbreviation: "UTC", dst_abbreviation: None,
    standard_offset: TzOffset::new(0, 0), dst_rules: None,
};

pub const US_EASTERN: NamedTimezone = NamedTimezone {
    name: "America/New_York", abbreviation: "EST", dst_abbreviation: Some("EDT"),
    standard_offset: TzOffset::new(-5, 0), dst_rules: Some(US_DST),
};

pub const US_CENTRAL: NamedTimezone = NamedTimezone {
    name: "America/Chicago", abbreviation: "CST", dst_abbreviation: Some("CDT"),
    standard_offset: TzOffset::new(-6, 0), dst_rules: Some(US_DST),
};

pub const US_MOUNTAIN: NamedTimezone = NamedTimezone {
    name: "America/Denver", abbreviation: "MST", dst_abbreviation: Some("MDT"),
    standard_offset: TzOffset::new(-7, 0), dst_rules: Some(US_DST),
};

pub const US_PACIFIC: NamedTimezone = NamedTimezone {
    name: "America/Los_Angeles", abbreviation: "PST", dst_abbreviation: Some("PDT"),
    standard_offset: TzOffset::new(-8, 0), dst_rules: Some(US_DST),
};

pub const EU_CET: NamedTimezone = NamedTimezone {
    name: "Europe/Berlin", abbreviation: "CET", dst_abbreviation: Some("CEST"),
    standard_offset: TzOffset::new(1, 0), dst_rules: Some(EU_DST),
};

pub const EU_EET: NamedTimezone = NamedTimezone {
    name: "Europe/Athens", abbreviation: "EET", dst_abbreviation: Some("EEST"),
    standard_offset: TzOffset::new(2, 0), dst_rules: Some(EU_DST),
};

pub const EU_GMT: NamedTimezone = NamedTimezone {
    name: "Europe/London", abbreviation: "GMT", dst_abbreviation: Some("BST"),
    standard_offset: TzOffset::new(0, 0), dst_rules: Some(EU_DST),
};

pub const ASIA_TOKYO: NamedTimezone = NamedTimezone {
    name: "Asia/Tokyo", abbreviation: "JST", dst_abbreviation: None,
    standard_offset: TzOffset::new(9, 0), dst_rules: None,
};

pub const ASIA_KOLKATA: NamedTimezone = NamedTimezone {
    name: "Asia/Kolkata", abbreviation: "IST", dst_abbreviation: None,
    standard_offset: TzOffset::new(5, 30), dst_rules: None,
};

pub const AUSTRALIA_SYDNEY: NamedTimezone = NamedTimezone {
    name: "Australia/Sydney", abbreviation: "AEST", dst_abbreviation: Some("AEDT"),
    standard_offset: TzOffset::new(10, 0),
    // Australia DST: first Sunday of October → first Sunday of April.
    dst_rules: Some(DstRules {
        start: DstTransition { month: 10, nth_weekday: 1, weekday: Weekday::Sun, hour: 2 },
        end: DstTransition { month: 4, nth_weekday: 1, weekday: Weekday::Sun, hour: 3 },
        offset_minutes: 60,
    }),
};

/// All built-in timezones.
pub fn all_timezones() -> Vec<&'static NamedTimezone> {
    vec![
        &UTC, &US_EASTERN, &US_CENTRAL, &US_MOUNTAIN, &US_PACIFIC,
        &EU_GMT, &EU_CET, &EU_EET,
        &ASIA_TOKYO, &ASIA_KOLKATA, &AUSTRALIA_SYDNEY,
    ]
}

/// Find a timezone by name.
pub fn find_timezone(name: &str) -> Option<&'static NamedTimezone> {
    all_timezones().into_iter().find(|tz| tz.name == name || tz.abbreviation == name)
}

// ── Helpers ────────────────────────────────────────────────────

fn days_in_month(year: i32, month: u32) -> u32 {
    if month == 12 {
        31
    } else {
        let next = NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap();
        (next - chrono::Duration::days(1)).day()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dt(y: i32, m: u32, d: u32, h: u32, mi: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(h, mi, 0).unwrap()
    }

    #[test]
    fn offset_format() {
        assert_eq!(TzOffset::new(-5, 0).format(), "-05:00");
        assert_eq!(TzOffset::new(5, 30).format(), "+05:30");
        assert_eq!(TzOffset::new(0, 0).format(), "+00:00");
    }

    #[test]
    fn utc_no_dst() {
        let winter = dt(2026, 1, 15, 12, 0);
        let summer = dt(2026, 7, 15, 12, 0);
        assert!(!UTC.is_dst(winter));
        assert!(!UTC.is_dst(summer));
        assert_eq!(UTC.to_local(winter), winter);
    }

    #[test]
    fn us_eastern_winter() {
        // January: EST (-5)
        let utc = dt(2026, 1, 15, 17, 0);
        let local = US_EASTERN.to_local(utc);
        assert_eq!(local, dt(2026, 1, 15, 12, 0));
        assert_eq!(US_EASTERN.abbreviation_at(utc), "EST");
    }

    #[test]
    fn us_eastern_summer() {
        // July: EDT (-4)
        let utc = dt(2026, 7, 15, 16, 0);
        let local = US_EASTERN.to_local(utc);
        assert_eq!(local, dt(2026, 7, 15, 12, 0));
        assert_eq!(US_EASTERN.abbreviation_at(utc), "EDT");
    }

    #[test]
    fn to_utc_roundtrip() {
        let local = dt(2026, 7, 15, 12, 0);
        let utc = US_EASTERN.to_utc(local);
        let back = US_EASTERN.to_local(utc);
        assert_eq!(back, local);
    }

    #[test]
    fn tokyo_no_dst() {
        let utc = dt(2026, 7, 15, 3, 0);
        let local = ASIA_TOKYO.to_local(utc);
        assert_eq!(local, dt(2026, 7, 15, 12, 0));
        assert!(!ASIA_TOKYO.is_dst(utc));
    }

    #[test]
    fn kolkata_half_hour_offset() {
        let utc = dt(2026, 1, 15, 6, 30);
        let local = ASIA_KOLKATA.to_local(utc);
        assert_eq!(local, dt(2026, 1, 15, 12, 0));
    }

    #[test]
    fn convert_between_timezones() {
        // 12pm Eastern in summer → what time in Tokyo?
        let eastern_local = dt(2026, 7, 15, 12, 0);
        let tokyo_local = convert(eastern_local, &US_EASTERN, &ASIA_TOKYO);
        // EDT is -4, JST is +9, difference is 13 hours.
        assert_eq!(tokyo_local, dt(2026, 7, 16, 1, 0));
    }

    #[test]
    fn format_with_abbreviation() {
        let utc = dt(2026, 1, 15, 17, 0);
        let formatted = US_EASTERN.format_datetime(utc, "%Y-%m-%d %H:%M %Z");
        assert_eq!(formatted, "2026-01-15 12:00 EST");
    }

    #[test]
    fn dst_transition_date() {
        // Second Sunday of March 2026 is March 8.
        let date = US_DST_START.date_in_year(2026);
        assert_eq!(date, Some(NaiveDate::from_ymd_opt(2026, 3, 8).unwrap()));
    }

    #[test]
    fn find_timezone_by_name() {
        assert_eq!(find_timezone("Asia/Tokyo").unwrap().abbreviation, "JST");
        assert_eq!(find_timezone("EST").unwrap().name, "America/New_York");
        assert!(find_timezone("Mars/Olympus_Mons").is_none());
    }

    #[test]
    fn all_timezones_list() {
        let all = all_timezones();
        assert!(all.len() >= 10);
        assert!(all.iter().any(|tz| tz.name == "UTC"));
    }

    #[test]
    fn eu_cet_summer() {
        // July: CEST (+2)
        let utc = dt(2026, 7, 15, 10, 0);
        assert!(EU_CET.is_dst(utc));
        let local = EU_CET.to_local(utc);
        assert_eq!(local, dt(2026, 7, 15, 12, 0));
        assert_eq!(EU_CET.abbreviation_at(utc), "CEST");
    }
}
