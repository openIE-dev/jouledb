//! Cron expression parser — parse, validate, compute next occurrences, and describe.
//!
//! Replaces node-cron / cron-parser with a pure-Rust cron expression engine.
//! Supports standard 5-field cron syntax: minute hour day-of-month month day-of-week.

use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ── Field ──────────────────────────────────────────────────────

/// A single cron field value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CronField {
    /// Match any value (*).
    Any,
    /// A specific value (e.g. 5).
    Value(u32),
    /// A range (e.g. 1-5).
    Range(u32, u32),
    /// A step (e.g. */15 or 1-30/5).
    Step { base: Box<CronField>, step: u32 },
    /// A list of values/ranges (e.g. 1,3,5 or 1-5,10,15-20).
    List(Vec<CronField>),
}

impl CronField {
    /// Does this field match the given value?
    pub fn matches(&self, value: u32) -> bool {
        match self {
            Self::Any => true,
            Self::Value(v) => *v == value,
            Self::Range(lo, hi) => value >= *lo && value <= *hi,
            Self::Step { base, step } => {
                if !base.matches(value) {
                    return false;
                }
                let start = match base.as_ref() {
                    Self::Any => 0,
                    Self::Range(lo, _) => *lo,
                    Self::Value(v) => *v,
                    _ => 0,
                };
                if value < start {
                    return false;
                }
                (value - start) % step == 0
            }
            Self::List(items) => items.iter().any(|f| f.matches(value)),
        }
    }

    /// All matching values within a given range [min, max].
    pub fn expand(&self, min: u32, max: u32) -> BTreeSet<u32> {
        let mut set = BTreeSet::new();
        for v in min..=max {
            if self.matches(v) {
                set.insert(v);
            }
        }
        set
    }
}

// ── Cron Expression ────────────────────────────────────────────

/// A parsed cron expression (5 fields).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronExpr {
    pub minute: CronField,
    pub hour: CronField,
    pub day_of_month: CronField,
    pub month: CronField,
    pub day_of_week: CronField,
}

impl CronExpr {
    /// Parse a standard cron expression ("minute hour day month weekday").
    pub fn parse(expr: &str) -> Result<Self, CronError> {
        let parts: Vec<&str> = expr.trim().split_whitespace().collect();
        if parts.len() != 5 {
            return Err(CronError::InvalidFieldCount(parts.len()));
        }

        Ok(Self {
            minute: parse_field(parts[0], 0, 59)?,
            hour: parse_field(parts[1], 0, 23)?,
            day_of_month: parse_field(parts[2], 1, 31)?,
            month: parse_field(parts[3], 1, 12)?,
            day_of_week: parse_field(parts[4], 0, 6)?,
        })
    }

    /// Does this expression match the given datetime?
    pub fn matches(&self, dt: NaiveDateTime) -> bool {
        let wd = dt.weekday().num_days_from_sunday();
        self.minute.matches(dt.minute())
            && self.hour.matches(dt.hour())
            && self.day_of_month.matches(dt.day())
            && self.month.matches(dt.month())
            && self.day_of_week.matches(wd)
    }

    /// Find the next occurrence after `after` (exclusive).
    /// Searches up to ~4 years ahead to avoid infinite loops.
    pub fn next_after(&self, after: NaiveDateTime) -> Option<NaiveDateTime> {
        // Start from the next minute.
        let mut dt = after + chrono::Duration::minutes(1);
        // Zero out seconds.
        dt = dt.date().and_hms_opt(dt.hour(), dt.minute(), 0)?;

        let limit = after + chrono::Duration::days(366 * 4);

        while dt < limit {
            if self.matches(dt) {
                return Some(dt);
            }

            // Advance intelligently.
            if !self.month.matches(dt.month()) {
                // Skip to first day of next month.
                dt = advance_to_next_month(dt)?;
                continue;
            }
            if !self.day_of_month.matches(dt.day()) || !self.day_of_week.matches(dt.weekday().num_days_from_sunday()) {
                // Skip to next day.
                dt = (dt.date() + chrono::Duration::days(1)).and_hms_opt(0, 0, 0)?;
                continue;
            }
            if !self.hour.matches(dt.hour()) {
                // Skip to next hour.
                dt = dt.date().and_hms_opt(dt.hour() + 1, 0, 0).unwrap_or_else(|| {
                    (dt.date() + chrono::Duration::days(1)).and_hms_opt(0, 0, 0).unwrap()
                });
                continue;
            }
            // Skip to next minute.
            dt += chrono::Duration::minutes(1);
        }

        None
    }

    /// Find the next N occurrences after `after`.
    pub fn next_n(&self, after: NaiveDateTime, n: usize) -> Vec<NaiveDateTime> {
        let mut results = Vec::with_capacity(n);
        let mut cursor = after;
        for _ in 0..n {
            if let Some(next) = self.next_after(cursor) {
                results.push(next);
                cursor = next;
            } else {
                break;
            }
        }
        results
    }

    /// Validate that the expression is well-formed and can produce occurrences.
    pub fn validate(&self) -> Result<(), CronError> {
        if self.minute.expand(0, 59).is_empty() {
            return Err(CronError::InvalidField("minute".into()));
        }
        if self.hour.expand(0, 23).is_empty() {
            return Err(CronError::InvalidField("hour".into()));
        }
        if self.day_of_month.expand(1, 31).is_empty() {
            return Err(CronError::InvalidField("day_of_month".into()));
        }
        if self.month.expand(1, 12).is_empty() {
            return Err(CronError::InvalidField("month".into()));
        }
        if self.day_of_week.expand(0, 6).is_empty() {
            return Err(CronError::InvalidField("day_of_week".into()));
        }
        Ok(())
    }

    /// Human-readable description of the schedule.
    pub fn describe(&self) -> String {
        let time_desc = describe_time(&self.minute, &self.hour);
        let day_desc = describe_day(&self.day_of_month, &self.month, &self.day_of_week);

        if day_desc.is_empty() {
            time_desc
        } else {
            format!("{time_desc} {day_desc}")
        }
    }
}

impl std::fmt::Display for CronExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}

// ── Parsing Helpers ────────────────────────────────────────────

fn parse_field(s: &str, min: u32, max: u32) -> Result<CronField, CronError> {
    if s == "*" {
        return Ok(CronField::Any);
    }

    // List: "1,3,5"
    if s.contains(',') {
        let items: Result<Vec<CronField>, _> = s.split(',').map(|part| parse_field(part, min, max)).collect();
        return Ok(CronField::List(items?));
    }

    // Step: "*/15" or "1-30/5"
    if s.contains('/') {
        let parts: Vec<&str> = s.splitn(2, '/').collect();
        let base = parse_field(parts[0], min, max)?;
        let step: u32 = parts[1].parse().map_err(|_| CronError::InvalidValue(parts[1].to_string()))?;
        if step == 0 {
            return Err(CronError::InvalidValue("step cannot be 0".into()));
        }
        return Ok(CronField::Step { base: Box::new(base), step });
    }

    // Range: "1-5"
    if s.contains('-') {
        let parts: Vec<&str> = s.splitn(2, '-').collect();
        let lo: u32 = parts[0].parse().map_err(|_| CronError::InvalidValue(parts[0].to_string()))?;
        let hi: u32 = parts[1].parse().map_err(|_| CronError::InvalidValue(parts[1].to_string()))?;
        if lo > max || hi > max {
            return Err(CronError::OutOfRange { value: hi.max(lo), min, max });
        }
        return Ok(CronField::Range(lo, hi));
    }

    // Single value.
    let v: u32 = s.parse().map_err(|_| CronError::InvalidValue(s.to_string()))?;
    if v < min || v > max {
        return Err(CronError::OutOfRange { value: v, min, max });
    }
    Ok(CronField::Value(v))
}

fn advance_to_next_month(dt: NaiveDateTime) -> Option<NaiveDateTime> {
    let (y, m) = if dt.month() == 12 { (dt.year() + 1, 1) } else { (dt.year(), dt.month() + 1) };
    NaiveDate::from_ymd_opt(y, m, 1)?.and_hms_opt(0, 0, 0)
}

// ── Description Helpers ────────────────────────────────────────

fn describe_time(minute: &CronField, hour: &CronField) -> String {
    match (minute, hour) {
        (CronField::Value(m), CronField::Value(h)) => {
            let (display_h, ampm) = if *h == 0 {
                (12, "AM")
            } else if *h < 12 {
                (*h, "AM")
            } else if *h == 12 {
                (12, "PM")
            } else {
                (h - 12, "PM")
            };
            format!("At {display_h}:{m:02} {ampm}")
        }
        (CronField::Value(m), CronField::Any) => format!("At minute {m} of every hour"),
        (CronField::Step { step, .. }, CronField::Any) => format!("Every {step} minutes"),
        (CronField::Any, CronField::Any) => "Every minute".to_string(),
        _ => {
            let min_str = field_to_string(minute);
            let hr_str = field_to_string(hour);
            format!("At minute {min_str} past hour {hr_str}")
        }
    }
}

fn describe_day(dom: &CronField, month: &CronField, dow: &CronField) -> String {
    let mut parts = Vec::new();

    match dow {
        CronField::Any => {}
        CronField::Range(1, 5) => parts.push("on weekdays".to_string()),
        CronField::Range(lo, hi) => {
            parts.push(format!("on {} through {}", weekday_name(*lo), weekday_name(*hi)));
        }
        CronField::Value(d) => parts.push(format!("on {}", weekday_name(*d))),
        _ => parts.push(format!("on day-of-week {}", field_to_string(dow))),
    }

    match dom {
        CronField::Any => {}
        CronField::Value(d) => parts.push(format!("on day {d} of the month")),
        _ => parts.push(format!("on day {}", field_to_string(dom))),
    }

    match month {
        CronField::Any => {}
        CronField::Value(m) => parts.push(format!("in {}", month_name(*m))),
        _ => parts.push(format!("in month {}", field_to_string(month))),
    }

    parts.join(" ")
}

fn field_to_string(f: &CronField) -> String {
    match f {
        CronField::Any => "*".to_string(),
        CronField::Value(v) => v.to_string(),
        CronField::Range(lo, hi) => format!("{lo}-{hi}"),
        CronField::Step { base, step } => format!("{}/{step}", field_to_string(base)),
        CronField::List(items) => items.iter().map(field_to_string).collect::<Vec<_>>().join(","),
    }
}

fn weekday_name(n: u32) -> &'static str {
    match n {
        0 => "Sunday",
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6 => "Saturday",
        _ => "Unknown",
    }
}

fn month_name(n: u32) -> &'static str {
    match n {
        1 => "January", 2 => "February", 3 => "March", 4 => "April",
        5 => "May", 6 => "June", 7 => "July", 8 => "August",
        9 => "September", 10 => "October", 11 => "November", 12 => "December",
        _ => "Unknown",
    }
}

// ── Errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CronError {
    InvalidFieldCount(usize),
    InvalidValue(String),
    InvalidField(String),
    OutOfRange { value: u32, min: u32, max: u32 },
}

impl std::fmt::Display for CronError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFieldCount(n) => write!(f, "expected 5 fields, got {n}"),
            Self::InvalidValue(v) => write!(f, "invalid value: {v}"),
            Self::InvalidField(name) => write!(f, "invalid field: {name}"),
            Self::OutOfRange { value, min, max } => {
                write!(f, "value {value} out of range [{min}, {max}]")
            }
        }
    }
}

impl std::error::Error for CronError {}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dt(y: i32, m: u32, d: u32, h: u32, mi: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(h, mi, 0).unwrap()
    }

    #[test]
    fn parse_every_minute() {
        let cron = CronExpr::parse("* * * * *").unwrap();
        assert_eq!(cron.minute, CronField::Any);
        assert!(cron.matches(dt(2026, 3, 8, 10, 30)));
    }

    #[test]
    fn parse_specific_time() {
        let cron = CronExpr::parse("30 9 * * *").unwrap();
        assert!(cron.matches(dt(2026, 3, 8, 9, 30)));
        assert!(!cron.matches(dt(2026, 3, 8, 9, 31)));
    }

    #[test]
    fn parse_weekdays_at_9am() {
        let cron = CronExpr::parse("0 9 * * 1-5").unwrap();
        // March 9, 2026 is Monday.
        assert!(cron.matches(dt(2026, 3, 9, 9, 0)));
        // March 8, 2026 is Sunday.
        assert!(!cron.matches(dt(2026, 3, 8, 9, 0)));
    }

    #[test]
    fn parse_every_15_minutes() {
        let cron = CronExpr::parse("*/15 * * * *").unwrap();
        assert!(cron.matches(dt(2026, 3, 8, 10, 0)));
        assert!(cron.matches(dt(2026, 3, 8, 10, 15)));
        assert!(cron.matches(dt(2026, 3, 8, 10, 30)));
        assert!(cron.matches(dt(2026, 3, 8, 10, 45)));
        assert!(!cron.matches(dt(2026, 3, 8, 10, 10)));
    }

    #[test]
    fn parse_list() {
        let cron = CronExpr::parse("0 9,12,17 * * *").unwrap();
        assert!(cron.matches(dt(2026, 3, 8, 9, 0)));
        assert!(cron.matches(dt(2026, 3, 8, 12, 0)));
        assert!(cron.matches(dt(2026, 3, 8, 17, 0)));
        assert!(!cron.matches(dt(2026, 3, 8, 10, 0)));
    }

    #[test]
    fn parse_invalid() {
        assert!(CronExpr::parse("* * *").is_err()); // too few fields
        assert!(CronExpr::parse("60 * * * *").is_err()); // minute out of range
    }

    #[test]
    fn next_occurrence() {
        let cron = CronExpr::parse("0 9 * * *").unwrap();
        let after = dt(2026, 3, 8, 10, 0);
        let next = cron.next_after(after).unwrap();
        assert_eq!(next, dt(2026, 3, 9, 9, 0));
    }

    #[test]
    fn next_occurrence_same_day() {
        let cron = CronExpr::parse("30 14 * * *").unwrap();
        let after = dt(2026, 3, 8, 10, 0);
        let next = cron.next_after(after).unwrap();
        assert_eq!(next, dt(2026, 3, 8, 14, 30));
    }

    #[test]
    fn next_n_occurrences() {
        let cron = CronExpr::parse("0 9 * * 1-5").unwrap();
        let after = dt(2026, 3, 6, 10, 0); // Friday
        let next_3 = cron.next_n(after, 3);
        assert_eq!(next_3.len(), 3);
        // Next weekday after Friday 10am is Monday.
        assert_eq!(next_3[0], dt(2026, 3, 9, 9, 0)); // Monday
        assert_eq!(next_3[1], dt(2026, 3, 10, 9, 0)); // Tuesday
        assert_eq!(next_3[2], dt(2026, 3, 11, 9, 0)); // Wednesday
    }

    #[test]
    fn describe_every_weekday_9am() {
        let cron = CronExpr::parse("0 9 * * 1-5").unwrap();
        let desc = cron.describe();
        assert!(desc.contains("9:00 AM"), "got: {desc}");
        assert!(desc.contains("weekdays"), "got: {desc}");
    }

    #[test]
    fn describe_every_15_min() {
        let cron = CronExpr::parse("*/15 * * * *").unwrap();
        let desc = cron.describe();
        assert!(desc.contains("15 minutes"), "got: {desc}");
    }

    #[test]
    fn validate_valid() {
        let cron = CronExpr::parse("0 9 * * 1-5").unwrap();
        assert!(cron.validate().is_ok());
    }

    #[test]
    fn field_expand() {
        let field = CronField::Step { base: Box::new(CronField::Any), step: 15 };
        let values = field.expand(0, 59);
        assert_eq!(values, [0, 15, 30, 45].into_iter().collect());
    }

    #[test]
    fn specific_month_and_day() {
        let cron = CronExpr::parse("0 0 25 12 *").unwrap();
        assert!(cron.matches(dt(2026, 12, 25, 0, 0)));
        assert!(!cron.matches(dt(2026, 12, 26, 0, 0)));
        let desc = cron.describe();
        assert!(desc.contains("December"), "got: {desc}");
    }
}
