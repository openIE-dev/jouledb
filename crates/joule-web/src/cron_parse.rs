//! Cron expression parser and scheduler.
//!
//! Supports standard 5-field cron, optional seconds (6-field) and year (7-field).
//! Wildcards, ranges, steps, lists, last-day-of-month (L), nth-weekday (3#2).
//! Computes the next N occurrences from a given starting time.

use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike};
use std::collections::BTreeSet;
use std::fmt;
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum CronError {
    #[error("invalid cron expression: {0}")]
    Invalid(String),
    #[error("field '{field}' value {value} out of range [{min}..{max}]")]
    OutOfRange { field: String, value: u32, min: u32, max: u32 },
    #[error("wrong number of fields: expected 5-7, got {0}")]
    WrongFieldCount(usize),
}

// ── Field Value ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum FieldValue {
    /// Match any value.
    Any,
    /// A single value.
    Single(u32),
    /// A range inclusive.
    Range(u32, u32),
    /// A step: base/step.
    Step { base: Box<FieldValue>, step: u32 },
    /// A list of values.
    List(Vec<FieldValue>),
    /// Last day of month.
    Last,
    /// Nth weekday: weekday#nth (e.g. 5#3 = third Friday).
    NthWeekday { weekday: u32, nth: u32 },
    /// Last weekday: weekdayL (e.g. 5L = last Friday).
    LastWeekday(u32),
}

impl FieldValue {
    /// Expand this value into a set of matching values within the given range.
    fn expand(&self, min: u32, max: u32) -> BTreeSet<u32> {
        let mut set = BTreeSet::new();
        match self {
            Self::Any => {
                for v in min..=max { set.insert(v); }
            }
            Self::Single(v) => { set.insert(*v); }
            Self::Range(lo, hi) => {
                for v in *lo..=*hi { set.insert(v); }
            }
            Self::Step { base, step } => {
                let base_vals = base.expand(min, max);
                if base_vals.is_empty() { return set; }
                let start = *base_vals.iter().next().unwrap();
                let end = *base_vals.iter().next_back().unwrap();
                let mut v = start;
                while v <= end {
                    set.insert(v);
                    v += step;
                }
            }
            Self::List(items) => {
                for item in items {
                    set.extend(item.expand(min, max));
                }
            }
            Self::Last | Self::NthWeekday { .. } | Self::LastWeekday(_) => {
                // These are context-dependent; handled separately
            }
        }
        set
    }
}

// ── Cron Expression ─────────────────────────────────────────────

/// A parsed cron expression.
#[derive(Debug, Clone)]
pub struct CronExpr {
    seconds: FieldValue,
    minutes: FieldValue,
    hours: FieldValue,
    day_of_month: FieldValue,
    month: FieldValue,
    day_of_week: FieldValue,
    year: Option<FieldValue>,
    raw: String,
}

impl CronExpr {
    /// Parse a cron expression string.
    ///
    /// Supports:
    /// - 5 fields: minute hour day-of-month month day-of-week
    /// - 6 fields: second minute hour day-of-month month day-of-week
    /// - 7 fields: second minute hour day-of-month month day-of-week year
    pub fn parse(expr: &str) -> Result<Self, CronError> {
        let parts: Vec<&str> = expr.split_whitespace().collect();
        let (sec, min, hour, dom, mon, dow, year) = match parts.len() {
            5 => {
                (FieldValue::Single(0),
                 parse_field(parts[0], "minute", 0, 59)?,
                 parse_field(parts[1], "hour", 0, 23)?,
                 parse_dom_field(parts[2])?,
                 parse_month_field(parts[3])?,
                 parse_dow_field(parts[4])?,
                 None)
            }
            6 => {
                (parse_field(parts[0], "second", 0, 59)?,
                 parse_field(parts[1], "minute", 0, 59)?,
                 parse_field(parts[2], "hour", 0, 23)?,
                 parse_dom_field(parts[3])?,
                 parse_month_field(parts[4])?,
                 parse_dow_field(parts[5])?,
                 None)
            }
            7 => {
                (parse_field(parts[0], "second", 0, 59)?,
                 parse_field(parts[1], "minute", 0, 59)?,
                 parse_field(parts[2], "hour", 0, 23)?,
                 parse_dom_field(parts[3])?,
                 parse_month_field(parts[4])?,
                 parse_dow_field(parts[5])?,
                 Some(parse_field(parts[6], "year", 1970, 2099)?))
            }
            n => return Err(CronError::WrongFieldCount(n)),
        };

        Ok(Self {
            seconds: sec,
            minutes: min,
            hours: hour,
            day_of_month: dom,
            month: mon,
            day_of_week: dow,
            year,
            raw: expr.to_string(),
        })
    }

    /// Check if the given datetime matches this cron expression.
    pub fn matches(&self, dt: &NaiveDateTime) -> bool {
        let sec = dt.second();
        let min = dt.minute();
        let hour = dt.hour();
        let dom = dt.day();
        let mon = dt.month();
        let dow = dt.weekday().num_days_from_sunday();

        if !self.seconds.expand(0, 59).contains(&sec) { return false; }
        if !self.minutes.expand(0, 59).contains(&min) { return false; }
        if !self.hours.expand(0, 23).contains(&hour) { return false; }
        if !self.month.expand(1, 12).contains(&mon) { return false; }

        // Day of month
        let dom_matches = match &self.day_of_month {
            FieldValue::Last => dom == last_day_of_month(dt.year(), mon),
            FieldValue::Any => true,
            other => other.expand(1, 31).contains(&dom),
        };

        // Day of week
        let dow_matches = match &self.day_of_week {
            FieldValue::NthWeekday { weekday, nth } => {
                dow == *weekday && is_nth_weekday(dt.date(), *weekday, *nth)
            }
            FieldValue::LastWeekday(wd) => {
                dow == *wd && is_last_weekday_of_month(dt.date(), *wd)
            }
            FieldValue::Any => true,
            other => other.expand(0, 6).contains(&dow),
        };

        if !dom_matches || !dow_matches { return false; }

        if let Some(yr) = &self.year {
            let y = dt.year() as u32;
            if !yr.expand(1970, 2099).contains(&y) { return false; }
        }

        true
    }

    /// Compute the next occurrence after `from`.
    pub fn next_after(&self, from: &NaiveDateTime) -> Option<NaiveDateTime> {
        self.next_n_after(from, 1).into_iter().next()
    }

    /// Compute the next N occurrences after `from`.
    pub fn next_n_after(&self, from: &NaiveDateTime, n: usize) -> Vec<NaiveDateTime> {
        let mut results = Vec::with_capacity(n);
        let mut current = *from + chrono::Duration::seconds(1);
        let max_iterations = 366 * 24 * 60 * 60; // 1 year of seconds
        let mut iterations = 0;

        while results.len() < n && iterations < max_iterations {
            if self.matches(&current) {
                results.push(current);
                current = current + chrono::Duration::seconds(1);
            } else {
                // Skip ahead more aggressively
                current = advance_to_next_candidate(&current, self);
            }
            iterations += 1;
        }
        results
    }

    /// Describe this cron expression in human-readable form.
    pub fn describe(&self) -> String {
        let sec_desc = describe_field(&self.seconds, "second");
        let min_desc = describe_field(&self.minutes, "minute");
        let hour_desc = describe_field(&self.hours, "hour");

        if sec_desc == "0" {
            format!("At {} {}", min_desc, hour_desc)
        } else {
            format!("At {} {} {}", sec_desc, min_desc, hour_desc)
        }
    }

    /// Get the raw expression string.
    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

fn advance_to_next_candidate(dt: &NaiveDateTime, _expr: &CronExpr) -> NaiveDateTime {
    // Simple strategy: advance by 1 second.
    // A production scheduler would skip non-matching months/days.
    *dt + chrono::Duration::seconds(1)
}

fn describe_field(field: &FieldValue, name: &str) -> String {
    match field {
        FieldValue::Any => format!("every {}", name),
        FieldValue::Single(v) => format!("{}", v),
        FieldValue::Range(lo, hi) => format!("{}-{}", lo, hi),
        FieldValue::Step { step, .. } => format!("every {} {}s", step, name),
        FieldValue::List(items) => {
            let parts: Vec<String> = items.iter().map(|i| describe_field(i, name)).collect();
            parts.join(",")
        }
        FieldValue::Last => "last day".to_string(),
        FieldValue::NthWeekday { weekday, nth } => format!("{}#{}", weekday, nth),
        FieldValue::LastWeekday(wd) => format!("{}L", wd),
    }
}

fn last_day_of_month(year: i32, month: u32) -> u32 {
    if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)
    }
    .map(|d| d.pred_opt().map(|p| p.day()).unwrap_or(28))
    .unwrap_or(28)
}

fn is_nth_weekday(date: NaiveDate, weekday: u32, nth: u32) -> bool {
    if date.weekday().num_days_from_sunday() != weekday { return false; }
    let day = date.day();
    let occurrence = (day - 1) / 7 + 1;
    occurrence == nth
}

fn is_last_weekday_of_month(date: NaiveDate, weekday: u32) -> bool {
    if date.weekday().num_days_from_sunday() != weekday { return false; }
    let last = last_day_of_month(date.year(), date.month());
    date.day() + 7 > last
}

// ── Field Parsing ───────────────────────────────────────────────

fn parse_field(s: &str, name: &str, min: u32, max: u32) -> Result<FieldValue, CronError> {
    if s == "*" { return Ok(FieldValue::Any); }

    // List (comma separated)
    if s.contains(',') {
        let items: Result<Vec<FieldValue>, CronError> = s.split(',')
            .map(|p| parse_field(p.trim(), name, min, max))
            .collect();
        return Ok(FieldValue::List(items?));
    }

    // Step
    if s.contains('/') {
        let parts: Vec<&str> = s.splitn(2, '/').collect();
        let base = parse_field(parts[0], name, min, max)?;
        let step: u32 = parts[1].parse().map_err(|_| CronError::Invalid(format!("bad step in {}", s)))?;
        return Ok(FieldValue::Step { base: Box::new(base), step });
    }

    // Range
    if s.contains('-') {
        let parts: Vec<&str> = s.splitn(2, '-').collect();
        let lo: u32 = parts[0].parse().map_err(|_| CronError::Invalid(format!("bad range start: {}", parts[0])))?;
        let hi: u32 = parts[1].parse().map_err(|_| CronError::Invalid(format!("bad range end: {}", parts[1])))?;
        if lo > max || hi > max {
            return Err(CronError::OutOfRange { field: name.to_string(), value: hi.max(lo), min, max });
        }
        return Ok(FieldValue::Range(lo, hi));
    }

    // Single value
    let v: u32 = s.parse().map_err(|_| CronError::Invalid(format!("bad value: {}", s)))?;
    if v < min || v > max {
        return Err(CronError::OutOfRange { field: name.to_string(), value: v, min, max });
    }
    Ok(FieldValue::Single(v))
}

fn parse_dom_field(s: &str) -> Result<FieldValue, CronError> {
    if s == "L" { return Ok(FieldValue::Last); }
    parse_field(s, "day_of_month", 1, 31)
}

fn parse_month_field(s: &str) -> Result<FieldValue, CronError> {
    let s = replace_month_names(s);
    parse_field(&s, "month", 1, 12)
}

fn parse_dow_field(s: &str) -> Result<FieldValue, CronError> {
    let s_lower = s.to_uppercase();
    // Nth weekday: e.g. 5#3
    if s_lower.contains('#') {
        let parts: Vec<&str> = s_lower.splitn(2, '#').collect();
        let wd: u32 = parse_dow_value(parts[0])?;
        let nth: u32 = parts[1].parse().map_err(|_| CronError::Invalid(format!("bad nth: {}", parts[1])))?;
        return Ok(FieldValue::NthWeekday { weekday: wd, nth });
    }
    // Last weekday: e.g. 5L
    if s_lower.ends_with('L') {
        let wd_str = &s_lower[..s_lower.len() - 1];
        let wd = parse_dow_value(wd_str)?;
        return Ok(FieldValue::LastWeekday(wd));
    }
    let replaced = replace_dow_names(&s_lower);
    parse_field(&replaced, "day_of_week", 0, 6)
}

fn parse_dow_value(s: &str) -> Result<u32, CronError> {
    let replaced = replace_dow_names(s);
    replaced.parse().map_err(|_| CronError::Invalid(format!("bad day-of-week: {}", s)))
}

fn replace_month_names(s: &str) -> String {
    let mut result = s.to_uppercase();
    let months = [
        ("JAN", "1"), ("FEB", "2"), ("MAR", "3"), ("APR", "4"),
        ("MAY", "5"), ("JUN", "6"), ("JUL", "7"), ("AUG", "8"),
        ("SEP", "9"), ("OCT", "10"), ("NOV", "11"), ("DEC", "12"),
    ];
    for (name, num) in &months {
        result = result.replace(name, num);
    }
    result
}

fn replace_dow_names(s: &str) -> String {
    let mut result = s.to_uppercase();
    let days = [
        ("SUN", "0"), ("MON", "1"), ("TUE", "2"), ("WED", "3"),
        ("THU", "4"), ("FRI", "5"), ("SAT", "6"),
    ];
    for (name, num) in &days {
        result = result.replace(name, num);
    }
    result
}

impl fmt::Display for CronExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dt(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, mo, d).unwrap()
            .and_hms_opt(h, mi, s).unwrap()
    }

    #[test]
    fn test_every_minute() {
        let cron = CronExpr::parse("* * * * *").unwrap();
        assert!(cron.matches(&dt(2026, 3, 8, 10, 30, 0)));
    }

    #[test]
    fn test_specific_time() {
        let cron = CronExpr::parse("30 10 * * *").unwrap();
        assert!(cron.matches(&dt(2026, 3, 8, 10, 30, 0)));
        assert!(!cron.matches(&dt(2026, 3, 8, 10, 31, 0)));
    }

    #[test]
    fn test_range() {
        let cron = CronExpr::parse("0-5 * * * *").unwrap();
        assert!(cron.matches(&dt(2026, 3, 8, 10, 3, 0)));
        assert!(!cron.matches(&dt(2026, 3, 8, 10, 10, 0)));
    }

    #[test]
    fn test_step() {
        let cron = CronExpr::parse("*/15 * * * *").unwrap();
        assert!(cron.matches(&dt(2026, 3, 8, 10, 0, 0)));
        assert!(cron.matches(&dt(2026, 3, 8, 10, 15, 0)));
        assert!(cron.matches(&dt(2026, 3, 8, 10, 30, 0)));
        assert!(!cron.matches(&dt(2026, 3, 8, 10, 7, 0)));
    }

    #[test]
    fn test_list() {
        let cron = CronExpr::parse("0,15,30,45 * * * *").unwrap();
        assert!(cron.matches(&dt(2026, 3, 8, 10, 15, 0)));
        assert!(!cron.matches(&dt(2026, 3, 8, 10, 20, 0)));
    }

    #[test]
    fn test_day_of_week() {
        // Sunday = 0
        let cron = CronExpr::parse("0 0 * * 0").unwrap();
        // 2026-03-08 is a Sunday
        assert!(cron.matches(&dt(2026, 3, 8, 0, 0, 0)));
        assert!(!cron.matches(&dt(2026, 3, 9, 0, 0, 0))); // Monday
    }

    #[test]
    fn test_month_names() {
        let cron = CronExpr::parse("0 0 1 JAN *").unwrap();
        assert!(cron.matches(&dt(2026, 1, 1, 0, 0, 0)));
        assert!(!cron.matches(&dt(2026, 2, 1, 0, 0, 0)));
    }

    #[test]
    fn test_dow_names() {
        let cron = CronExpr::parse("0 0 * * MON").unwrap();
        assert!(cron.matches(&dt(2026, 3, 9, 0, 0, 0))); // Monday
        assert!(!cron.matches(&dt(2026, 3, 8, 0, 0, 0))); // Sunday
    }

    #[test]
    fn test_six_field_with_seconds() {
        let cron = CronExpr::parse("30 0 12 * * *").unwrap();
        assert!(cron.matches(&dt(2026, 3, 8, 12, 0, 30)));
        assert!(!cron.matches(&dt(2026, 3, 8, 12, 0, 0)));
    }

    #[test]
    fn test_seven_field_with_year() {
        let cron = CronExpr::parse("0 0 12 1 1 * 2026").unwrap();
        assert!(cron.matches(&dt(2026, 1, 1, 12, 0, 0)));
        assert!(!cron.matches(&dt(2027, 1, 1, 12, 0, 0)));
    }

    #[test]
    fn test_last_day_of_month() {
        let cron = CronExpr::parse("0 0 L * *").unwrap();
        // Jan 31
        assert!(cron.matches(&dt(2026, 1, 31, 0, 0, 0)));
        // Feb 28 (2026 is not leap)
        assert!(cron.matches(&dt(2026, 2, 28, 0, 0, 0)));
        assert!(!cron.matches(&dt(2026, 2, 27, 0, 0, 0)));
    }

    #[test]
    fn test_nth_weekday() {
        // 5#3 = third Friday
        let cron = CronExpr::parse("0 0 * * 5#3").unwrap();
        // 2026-03-20 is the third Friday of March 2026
        assert!(cron.matches(&dt(2026, 3, 20, 0, 0, 0)));
        assert!(!cron.matches(&dt(2026, 3, 13, 0, 0, 0))); // second Friday
    }

    #[test]
    fn test_next_occurrence() {
        let cron = CronExpr::parse("0 12 * * *").unwrap();
        let from = dt(2026, 3, 8, 10, 0, 0);
        let next = cron.next_after(&from).unwrap();
        assert_eq!(next, dt(2026, 3, 8, 12, 0, 0));
    }

    #[test]
    fn test_next_n() {
        let cron = CronExpr::parse("0 0 * * *").unwrap();
        let from = dt(2026, 3, 8, 0, 0, 0);
        let results = cron.next_n_after(&from, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], dt(2026, 3, 9, 0, 0, 0));
        assert_eq!(results[1], dt(2026, 3, 10, 0, 0, 0));
        assert_eq!(results[2], dt(2026, 3, 11, 0, 0, 0));
    }

    #[test]
    fn test_invalid_field_count() {
        assert!(CronExpr::parse("* * *").is_err());
        assert!(CronExpr::parse("* * * * * * * *").is_err());
    }

    #[test]
    fn test_out_of_range() {
        assert!(CronExpr::parse("60 * * * *").is_err()); // minute > 59
    }

    #[test]
    fn test_describe() {
        let cron = CronExpr::parse("30 10 * * *").unwrap();
        let desc = cron.describe();
        assert!(desc.contains("30"));
        assert!(desc.contains("10"));
    }

    #[test]
    fn test_display() {
        let expr = "*/5 * * * *";
        let cron = CronExpr::parse(expr).unwrap();
        assert_eq!(cron.to_string(), expr);
    }

    #[test]
    fn test_as_str() {
        let expr = "0 0 1 1 *";
        let cron = CronExpr::parse(expr).unwrap();
        assert_eq!(cron.as_str(), expr);
    }
}
