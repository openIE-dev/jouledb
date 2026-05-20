//! RFC 5545 recurrence rules (subset) — generate recurring event dates.
//!
//! Replaces rrule.js / date-fns-tz recurrence logic with a pure-Rust
//! implementation. Supports daily, weekly, monthly, and yearly frequencies
//! with BYDAY, BYMONTHDAY, BYMONTH, COUNT, UNTIL, EXDATE, and RDATE.

use chrono::{Datelike, NaiveDate, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ── Frequency ──────────────────────────────────────────────────

/// Recurrence frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Frequency {
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

// ── Weekday wrapper ────────────────────────────────────────────

/// A day-of-week with optional ordinal (e.g. 2nd Tuesday = (Some(2), Tue)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeekdaySpec {
    /// Optional ordinal: 1=first, 2=second, -1=last, etc. None means "every".
    pub ordinal: Option<i8>,
    pub weekday: Weekday,
}

impl WeekdaySpec {
    pub fn every(wd: Weekday) -> Self {
        Self { ordinal: None, weekday: wd }
    }

    pub fn nth(n: i8, wd: Weekday) -> Self {
        Self { ordinal: Some(n), weekday: wd }
    }
}

// ── Recurrence Rule ────────────────────────────────────────────

/// Termination condition for the recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bound {
    Count(u32),
    Until(NaiveDate),
}

/// A recurrence rule (subset of RFC 5545 RRULE).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RRule {
    pub freq: Frequency,
    /// Interval between occurrences (default 1).
    pub interval: u32,
    /// Termination: count or until date.
    pub bound: Option<Bound>,
    /// BYDAY — filter/expand by weekday.
    pub by_day: Vec<WeekdaySpec>,
    /// BYMONTHDAY — filter/expand by day-of-month (1–31).
    pub by_month_day: Vec<u32>,
    /// BYMONTH — filter/expand by month (1–12).
    pub by_month: Vec<u32>,
}

impl RRule {
    /// Create a simple daily recurrence.
    pub fn daily() -> Self {
        Self {
            freq: Frequency::Daily,
            interval: 1,
            bound: None,
            by_day: Vec::new(),
            by_month_day: Vec::new(),
            by_month: Vec::new(),
        }
    }

    /// Create a simple weekly recurrence.
    pub fn weekly() -> Self {
        Self { freq: Frequency::Weekly, ..Self::daily() }
    }

    /// Create a simple monthly recurrence.
    pub fn monthly() -> Self {
        Self { freq: Frequency::Monthly, ..Self::daily() }
    }

    /// Create a simple yearly recurrence.
    pub fn yearly() -> Self {
        Self { freq: Frequency::Yearly, ..Self::daily() }
    }

    pub fn with_interval(mut self, n: u32) -> Self {
        self.interval = n;
        self
    }

    pub fn with_count(mut self, n: u32) -> Self {
        self.bound = Some(Bound::Count(n));
        self
    }

    pub fn with_until(mut self, date: NaiveDate) -> Self {
        self.bound = Some(Bound::Until(date));
        self
    }

    pub fn with_by_day(mut self, days: Vec<WeekdaySpec>) -> Self {
        self.by_day = days;
        self
    }

    pub fn with_by_month_day(mut self, days: Vec<u32>) -> Self {
        self.by_month_day = days;
        self
    }

    pub fn with_by_month(mut self, months: Vec<u32>) -> Self {
        self.by_month = months;
        self
    }
}

// ── Common Presets ─────────────────────────────────────────────

/// Every weekday (Mon–Fri).
pub fn every_weekday() -> RRule {
    RRule::daily().with_by_day(vec![
        WeekdaySpec::every(Weekday::Mon),
        WeekdaySpec::every(Weekday::Tue),
        WeekdaySpec::every(Weekday::Wed),
        WeekdaySpec::every(Weekday::Thu),
        WeekdaySpec::every(Weekday::Fri),
    ])
}

/// First Monday of every month.
pub fn first_monday_of_month() -> RRule {
    RRule::monthly().with_by_day(vec![WeekdaySpec::nth(1, Weekday::Mon)])
}

/// Last Friday of every month.
pub fn last_friday_of_month() -> RRule {
    RRule::monthly().with_by_day(vec![WeekdaySpec::nth(-1, Weekday::Fri)])
}

/// Every other week.
pub fn biweekly() -> RRule {
    RRule::weekly().with_interval(2)
}

// ── Recurrence Set ─────────────────────────────────────────────

/// A full recurrence set: RRULE + exception dates + additional dates.
#[derive(Debug, Clone, Default)]
pub struct RecurrenceSet {
    pub dtstart: Option<NaiveDate>,
    pub rrule: Option<RRule>,
    /// EXDATE — dates to exclude from the recurrence.
    pub exdates: BTreeSet<NaiveDate>,
    /// RDATE — additional one-off dates.
    pub rdates: BTreeSet<NaiveDate>,
}

impl RecurrenceSet {
    pub fn new(start: NaiveDate, rule: RRule) -> Self {
        Self {
            dtstart: Some(start),
            rrule: Some(rule),
            exdates: BTreeSet::new(),
            rdates: BTreeSet::new(),
        }
    }

    pub fn add_exdate(&mut self, date: NaiveDate) {
        self.exdates.insert(date);
    }

    pub fn add_rdate(&mut self, date: NaiveDate) {
        self.rdates.insert(date);
    }

    /// Generate all occurrences. Capped at `max` to prevent infinite loops.
    pub fn occurrences(&self, max: usize) -> Vec<NaiveDate> {
        let mut dates = BTreeSet::new();

        if let (Some(start), Some(rule)) = (self.dtstart, &self.rrule) {
            for d in expand(start, rule, max) {
                if !self.exdates.contains(&d) {
                    dates.insert(d);
                }
            }
        }

        for rd in &self.rdates {
            if !self.exdates.contains(rd) {
                dates.insert(*rd);
            }
        }

        dates.into_iter().take(max).collect()
    }
}

// ── Expansion Engine ───────────────────────────────────────────

/// Generate occurrences from a start date and recurrence rule.
/// The caller should cap at a reasonable number.
pub fn expand(start: NaiveDate, rule: &RRule, max: usize) -> Vec<NaiveDate> {
    let mut results = Vec::new();
    let mut cursor = start;
    let mut count = 0u32;
    // Safety valve: never generate more than 10_000 candidates.
    let hard_limit = max.max(10_000);
    let mut iterations = 0usize;

    loop {
        if results.len() >= max {
            break;
        }
        iterations += 1;
        if iterations > hard_limit * 10 {
            break; // safety
        }

        let candidates = expand_candidates(cursor, rule);

        for candidate in candidates {
            if candidate < start {
                continue;
            }
            if let Some(Bound::Until(until)) = rule.bound {
                if candidate > until {
                    return results;
                }
            }
            if matches_filters(candidate, rule) {
                results.push(candidate);
                count += 1;
                if let Some(Bound::Count(c)) = rule.bound {
                    if count >= c {
                        return results;
                    }
                }
                if results.len() >= max {
                    return results;
                }
            }
        }

        cursor = advance_cursor(cursor, rule);
    }

    results
}

/// For a given cursor position, produce the candidate dates for this period.
fn expand_candidates(cursor: NaiveDate, rule: &RRule) -> Vec<NaiveDate> {
    match rule.freq {
        Frequency::Daily => vec![cursor],
        Frequency::Weekly => {
            if rule.by_day.is_empty() {
                vec![cursor]
            } else {
                // The week containing `cursor` — find matching days.
                let week_start = cursor - chrono::Duration::days(cursor.weekday().num_days_from_monday() as i64);
                rule.by_day
                    .iter()
                    .filter_map(|spec| {
                        let offset = spec.weekday.num_days_from_monday() as i64;
                        let d = week_start + chrono::Duration::days(offset);
                        Some(d)
                    })
                    .collect()
            }
        }
        Frequency::Monthly => {
            if !rule.by_month_day.is_empty() {
                rule.by_month_day
                    .iter()
                    .filter_map(|day| NaiveDate::from_ymd_opt(cursor.year(), cursor.month(), *day))
                    .collect()
            } else if !rule.by_day.is_empty() {
                expand_by_day_in_month(cursor.year(), cursor.month(), &rule.by_day)
            } else {
                vec![NaiveDate::from_ymd_opt(cursor.year(), cursor.month(), cursor.day().min(days_in_month(cursor.year(), cursor.month())))
                    .unwrap_or(cursor)]
            }
        }
        Frequency::Yearly => {
            if !rule.by_month.is_empty() {
                rule.by_month
                    .iter()
                    .filter_map(|m| {
                        let day = cursor.day().min(days_in_month(cursor.year(), *m));
                        NaiveDate::from_ymd_opt(cursor.year(), *m, day)
                    })
                    .collect()
            } else {
                let day = cursor.day().min(days_in_month(cursor.year(), cursor.month()));
                vec![NaiveDate::from_ymd_opt(cursor.year(), cursor.month(), day).unwrap_or(cursor)]
            }
        }
    }
}

/// Expand BYDAY specs within a specific month (for monthly frequency).
fn expand_by_day_in_month(year: i32, month: u32, specs: &[WeekdaySpec]) -> Vec<NaiveDate> {
    let mut results = Vec::new();
    let dim = days_in_month(year, month);

    for spec in specs {
        match spec.ordinal {
            None => {
                // Every occurrence of this weekday in the month.
                for day in 1..=dim {
                    if let Some(d) = NaiveDate::from_ymd_opt(year, month, day) {
                        if d.weekday() == spec.weekday {
                            results.push(d);
                        }
                    }
                }
            }
            Some(n) if n > 0 => {
                // Nth occurrence (1-indexed).
                let mut count = 0i8;
                for day in 1..=dim {
                    if let Some(d) = NaiveDate::from_ymd_opt(year, month, day) {
                        if d.weekday() == spec.weekday {
                            count += 1;
                            if count == n {
                                results.push(d);
                                break;
                            }
                        }
                    }
                }
            }
            Some(n) if n < 0 => {
                // Nth-from-last occurrence.
                let mut matching: Vec<NaiveDate> = Vec::new();
                for day in 1..=dim {
                    if let Some(d) = NaiveDate::from_ymd_opt(year, month, day) {
                        if d.weekday() == spec.weekday {
                            matching.push(d);
                        }
                    }
                }
                let idx = matching.len() as i8 + n; // n is negative
                if idx >= 0 && (idx as usize) < matching.len() {
                    results.push(matching[idx as usize]);
                }
            }
            _ => {}
        }
    }

    results.sort();
    results
}

/// Does a candidate date pass the by_day/by_month filters?
fn matches_filters(date: NaiveDate, rule: &RRule) -> bool {
    // For daily frequency with by_day, filter by weekday.
    if rule.freq == Frequency::Daily && !rule.by_day.is_empty() {
        if !rule.by_day.iter().any(|spec| spec.weekday == date.weekday()) {
            return false;
        }
    }
    // For yearly with by_month, already handled in candidates.
    true
}

/// Advance the cursor by one period.
fn advance_cursor(cursor: NaiveDate, rule: &RRule) -> NaiveDate {
    let interval = rule.interval.max(1) as i64;
    match rule.freq {
        Frequency::Daily => cursor + chrono::Duration::days(interval),
        Frequency::Weekly => cursor + chrono::Duration::weeks(interval),
        Frequency::Monthly => add_months(cursor, interval as i32),
        Frequency::Yearly => add_months(cursor, (interval * 12) as i32),
    }
}

fn add_months(date: NaiveDate, months: i32) -> NaiveDate {
    let total_months = date.year() * 12 + date.month() as i32 - 1 + months;
    let new_year = total_months / 12;
    let new_month = (total_months % 12 + 1) as u32;
    let new_day = date.day().min(days_in_month(new_year, new_month));
    NaiveDate::from_ymd_opt(new_year, new_month, new_day).unwrap_or(date)
}

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

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn daily_with_count() {
        let rule = RRule::daily().with_count(5);
        let dates = expand(d(2026, 3, 1), &rule, 100);
        assert_eq!(dates.len(), 5);
        assert_eq!(dates[0], d(2026, 3, 1));
        assert_eq!(dates[4], d(2026, 3, 5));
    }

    #[test]
    fn daily_with_until() {
        let rule = RRule::daily().with_until(d(2026, 3, 5));
        let dates = expand(d(2026, 3, 1), &rule, 100);
        assert_eq!(dates.len(), 5);
        assert_eq!(dates.last(), Some(&d(2026, 3, 5)));
    }

    #[test]
    fn weekly_interval_2() {
        let rule = RRule::weekly().with_interval(2).with_count(4);
        let dates = expand(d(2026, 3, 1), &rule, 100);
        assert_eq!(dates.len(), 4);
        assert_eq!(dates[1], d(2026, 3, 15));
        assert_eq!(dates[2], d(2026, 3, 29));
    }

    #[test]
    fn monthly_same_day() {
        let rule = RRule::monthly().with_count(3);
        let dates = expand(d(2026, 1, 15), &rule, 100);
        assert_eq!(dates, vec![d(2026, 1, 15), d(2026, 2, 15), d(2026, 3, 15)]);
    }

    #[test]
    fn monthly_31st_clamped() {
        // Months with fewer than 31 days should clamp.
        let rule = RRule::monthly().with_count(3);
        let dates = expand(d(2026, 1, 31), &rule, 100);
        assert_eq!(dates[0], d(2026, 1, 31));
        assert_eq!(dates[1], d(2026, 2, 28)); // Feb 2026 is not a leap year
        // Cursor advances from Feb 28 → Mar 28 (day clamped to cursor day)
        assert_eq!(dates[2], d(2026, 3, 28));
    }

    #[test]
    fn yearly() {
        let rule = RRule::yearly().with_count(3);
        let dates = expand(d(2026, 3, 8), &rule, 100);
        assert_eq!(dates, vec![d(2026, 3, 8), d(2027, 3, 8), d(2028, 3, 8)]);
    }

    #[test]
    fn every_weekday_preset() {
        let rule = every_weekday().with_count(5);
        let dates = expand(d(2026, 3, 9), &rule, 100); // Monday
        assert_eq!(dates.len(), 5);
        for d in &dates {
            let wd = d.weekday();
            assert!(wd != Weekday::Sat && wd != Weekday::Sun);
        }
    }

    #[test]
    fn first_monday_of_month_preset() {
        let rule = first_monday_of_month().with_count(3);
        let dates = expand(d(2026, 1, 1), &rule, 100);
        assert_eq!(dates.len(), 3);
        // First Monday of Jan 2026 is the 5th.
        assert_eq!(dates[0], d(2026, 1, 5));
        // First Monday of Feb 2026 is the 2nd.
        assert_eq!(dates[1], d(2026, 2, 2));
        // First Monday of Mar 2026 is the 2nd.
        assert_eq!(dates[2], d(2026, 3, 2));
    }

    #[test]
    fn last_friday_of_month_preset() {
        let rule = last_friday_of_month().with_count(2);
        let dates = expand(d(2026, 1, 1), &rule, 100);
        assert_eq!(dates.len(), 2);
        // Last Friday of Jan 2026 is the 30th.
        assert_eq!(dates[0], d(2026, 1, 30));
        // Last Friday of Feb 2026 is the 27th.
        assert_eq!(dates[1], d(2026, 2, 27));
    }

    #[test]
    fn exdate_exclusion() {
        let mut rset = RecurrenceSet::new(d(2026, 3, 1), RRule::daily().with_count(5));
        rset.add_exdate(d(2026, 3, 3));
        let dates = rset.occurrences(100);
        assert_eq!(dates.len(), 4);
        assert!(!dates.contains(&d(2026, 3, 3)));
    }

    #[test]
    fn rdate_addition() {
        let mut rset = RecurrenceSet::new(d(2026, 3, 1), RRule::weekly().with_count(2));
        rset.add_rdate(d(2026, 3, 4)); // extra date
        let dates = rset.occurrences(100);
        assert!(dates.contains(&d(2026, 3, 1)));
        assert!(dates.contains(&d(2026, 3, 4)));
        assert!(dates.contains(&d(2026, 3, 8)));
    }

    #[test]
    fn exdate_overrides_rdate() {
        let mut rset = RecurrenceSet::new(d(2026, 3, 1), RRule::daily().with_count(3));
        rset.add_rdate(d(2026, 3, 10));
        rset.add_exdate(d(2026, 3, 10)); // exclude the RDATE
        let dates = rset.occurrences(100);
        assert!(!dates.contains(&d(2026, 3, 10)));
    }

    #[test]
    fn by_month_day_monthly() {
        let rule = RRule::monthly().with_by_month_day(vec![1, 15]).with_count(4);
        let dates = expand(d(2026, 1, 1), &rule, 100);
        assert_eq!(dates.len(), 4);
        assert_eq!(dates[0], d(2026, 1, 1));
        assert_eq!(dates[1], d(2026, 1, 15));
        assert_eq!(dates[2], d(2026, 2, 1));
        assert_eq!(dates[3], d(2026, 2, 15));
    }
}
