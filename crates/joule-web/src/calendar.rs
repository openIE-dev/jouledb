//! Calendar model — month grids, week views, and event queries.
//!
//! Replaces FullCalendar / react-big-calendar with a pure-Rust calendar domain
//! model. No DOM — only data structures for month grids, day metadata, and
//! calendar events.

use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Weekday};
use serde::{Deserialize, Serialize};

// ── Day Metadata ───────────────────────────────────────────────

/// Metadata about a single day in a calendar grid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DayCell {
    pub date: NaiveDate,
    pub is_today: bool,
    pub is_weekend: bool,
    pub is_other_month: bool,
}

impl DayCell {
    /// Create a day cell, marking today/weekend/other-month flags.
    pub fn new(date: NaiveDate, display_month: u32, today: NaiveDate) -> Self {
        let wd = date.weekday();
        Self {
            date,
            is_today: date == today,
            is_weekend: wd == Weekday::Sat || wd == Weekday::Sun,
            is_other_month: date.month() != display_month,
        }
    }
}

// ── Calendar Month ─────────────────────────────────────────────

/// A calendar month grid: rows of 7-day weeks that cover the month, padded
/// with days from the previous and next months.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalendarMonth {
    pub year: i32,
    pub month: u32,
    /// Each inner Vec has exactly 7 elements (Sun–Sat).
    pub weeks: Vec<Vec<DayCell>>,
}

impl CalendarMonth {
    /// Generate a month grid. `today` is used to mark `is_today`.
    /// Weeks start on Sunday.
    pub fn generate(year: i32, month: u32, today: NaiveDate) -> Self {
        let first = NaiveDate::from_ymd_opt(year, month, 1).expect("valid month");
        // Number of days from Sunday to the first of the month.
        let start_offset = first.weekday().num_days_from_sunday() as i64;
        let grid_start = first - chrono::Duration::days(start_offset);

        let last_day = last_day_of_month(year, month);
        // We need enough weeks to cover the month.
        let end_of_last_week = {
            let remaining = (6 - last_day.weekday().num_days_from_sunday()) as i64;
            last_day + chrono::Duration::days(remaining)
        };

        let total_days = (end_of_last_week - grid_start).num_days() + 1;
        let num_weeks = (total_days / 7) as usize;

        let mut weeks = Vec::with_capacity(num_weeks);
        let mut cursor = grid_start;
        for _ in 0..num_weeks {
            let mut week = Vec::with_capacity(7);
            for _ in 0..7 {
                week.push(DayCell::new(cursor, month, today));
                cursor += chrono::Duration::days(1);
            }
            weeks.push(week);
        }

        Self { year, month, weeks }
    }

    /// Flat iterator over every day cell in the grid.
    pub fn days(&self) -> impl Iterator<Item = &DayCell> {
        self.weeks.iter().flat_map(|w| w.iter())
    }

    /// Days that belong to the displayed month (not other-month).
    pub fn month_days(&self) -> Vec<&DayCell> {
        self.days().filter(|d| !d.is_other_month).collect()
    }
}

// ── Calendar Event ─────────────────────────────────────────────

/// A calendar event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalendarEvent {
    pub id: String,
    pub title: String,
    pub start: NaiveDateTime,
    pub end: NaiveDateTime,
    pub all_day: bool,
    pub color: Option<String>,
}

impl CalendarEvent {
    /// Create a new timed event.
    pub fn new(id: impl Into<String>, title: impl Into<String>, start: NaiveDateTime, end: NaiveDateTime) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            start,
            end,
            all_day: false,
            color: None,
        }
    }

    /// Create an all-day event spanning the given date.
    pub fn all_day(id: impl Into<String>, title: impl Into<String>, date: NaiveDate) -> Self {
        let start = date.and_hms_opt(0, 0, 0).unwrap();
        let end = date.and_hms_opt(23, 59, 59).unwrap();
        Self {
            id: id.into(),
            title: title.into(),
            start,
            end,
            all_day: true,
            color: None,
        }
    }

    /// Set the color.
    pub fn with_color(mut self, color: impl Into<String>) -> Self {
        self.color = Some(color.into());
        self
    }

    /// Does this event overlap with the given date range (inclusive)?
    pub fn overlaps_range(&self, range_start: NaiveDate, range_end: NaiveDate) -> bool {
        let rs = range_start.and_hms_opt(0, 0, 0).unwrap();
        let re = range_end.and_hms_opt(23, 59, 59).unwrap();
        self.start <= re && self.end >= rs
    }

    /// Does this event fall on the given date?
    pub fn is_on_date(&self, date: NaiveDate) -> bool {
        self.overlaps_range(date, date)
    }
}

// ── Event Store ────────────────────────────────────────────────

/// A simple in-memory event collection with range queries.
#[derive(Debug, Clone, Default)]
pub struct EventStore {
    events: Vec<CalendarEvent>,
}

impl EventStore {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn add(&mut self, event: CalendarEvent) {
        self.events.push(event);
    }

    pub fn remove(&mut self, id: &str) -> Option<CalendarEvent> {
        if let Some(pos) = self.events.iter().position(|e| e.id == id) {
            Some(self.events.remove(pos))
        } else {
            None
        }
    }

    /// Events overlapping the given date range.
    pub fn events_in_range(&self, start: NaiveDate, end: NaiveDate) -> Vec<&CalendarEvent> {
        self.events.iter().filter(|e| e.overlaps_range(start, end)).collect()
    }

    /// Events on a specific date.
    pub fn events_on_date(&self, date: NaiveDate) -> Vec<&CalendarEvent> {
        self.events.iter().filter(|e| e.is_on_date(date)).collect()
    }

    pub fn all(&self) -> &[CalendarEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ── Week View ──────────────────────────────────────────────────

/// A time slot in a week/day view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSlot {
    pub time: NaiveTime,
    pub date: NaiveDate,
}

/// A week view: 7 columns of time slots.
#[derive(Debug, Clone)]
pub struct WeekView {
    pub start_date: NaiveDate,
    pub columns: Vec<WeekColumn>,
}

/// A single day column in a week view.
#[derive(Debug, Clone)]
pub struct WeekColumn {
    pub date: NaiveDate,
    pub slots: Vec<TimeSlot>,
}

impl WeekView {
    /// Generate a week view starting on `start_date` with time slots at the
    /// given `slot_minutes` interval (e.g. 30 for half-hour slots).
    /// `start_hour` and `end_hour` bound the visible time range.
    pub fn generate(
        start_date: NaiveDate,
        slot_minutes: u32,
        start_hour: u32,
        end_hour: u32,
    ) -> Self {
        let mut columns = Vec::with_capacity(7);
        for day_offset in 0..7 {
            let date = start_date + chrono::Duration::days(day_offset);
            let mut slots = Vec::new();
            let mut hour = start_hour;
            let mut minute = 0u32;
            loop {
                if hour >= end_hour {
                    break;
                }
                if let Some(time) = NaiveTime::from_hms_opt(hour, minute, 0) {
                    slots.push(TimeSlot { time, date });
                }
                minute += slot_minutes;
                while minute >= 60 {
                    minute -= 60;
                    hour += 1;
                }
            }
            columns.push(WeekColumn { date, slots });
        }
        Self { start_date, columns }
    }

    /// Find the column for a given date.
    pub fn column_for_date(&self, date: NaiveDate) -> Option<&WeekColumn> {
        self.columns.iter().find(|c| c.date == date)
    }
}

// ── Helpers ────────────────────────────────────────────────────

fn last_day_of_month(year: i32, month: u32) -> NaiveDate {
    if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap() - chrono::Duration::days(1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap() - chrono::Duration::days(1)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn dt(y: i32, m: u32, day: u32, h: u32, mi: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
            .and_hms_opt(h, mi, 0).unwrap()
    }

    #[test]
    fn month_grid_has_7_day_weeks() {
        let cal = CalendarMonth::generate(2026, 3, d(2026, 3, 8));
        for week in &cal.weeks {
            assert_eq!(week.len(), 7);
        }
    }

    #[test]
    fn march_2026_starts_on_sunday() {
        // March 1, 2026 is a Sunday.
        let cal = CalendarMonth::generate(2026, 3, d(2026, 3, 8));
        assert_eq!(cal.weeks[0][0].date, d(2026, 3, 1));
        assert!(!cal.weeks[0][0].is_other_month);
    }

    #[test]
    fn february_2024_leap_year() {
        let cal = CalendarMonth::generate(2024, 2, d(2024, 2, 15));
        let month_days = cal.month_days();
        assert_eq!(month_days.len(), 29); // leap year
    }

    #[test]
    fn today_flag_set_correctly() {
        let today = d(2026, 3, 8);
        let cal = CalendarMonth::generate(2026, 3, today);
        let cell = cal.days().find(|c| c.date == today).unwrap();
        assert!(cell.is_today);
        // Other days are not today.
        let other = cal.days().find(|c| c.date == d(2026, 3, 1)).unwrap();
        assert!(!other.is_today);
    }

    #[test]
    fn weekend_flags() {
        let cal = CalendarMonth::generate(2026, 3, d(2026, 3, 8));
        // March 7, 2026 is a Saturday.
        let sat = cal.days().find(|c| c.date == d(2026, 3, 7)).unwrap();
        assert!(sat.is_weekend);
        // March 8, 2026 is a Sunday.
        let sun = cal.days().find(|c| c.date == d(2026, 3, 8)).unwrap();
        assert!(sun.is_weekend);
        // March 9, 2026 is a Monday.
        let mon = cal.days().find(|c| c.date == d(2026, 3, 9)).unwrap();
        assert!(!mon.is_weekend);
    }

    #[test]
    fn other_month_padding() {
        // April 2026 starts on a Wednesday.
        let cal = CalendarMonth::generate(2026, 4, d(2026, 4, 1));
        // First row should have some March days.
        let first_day = &cal.weeks[0][0];
        assert!(first_day.is_other_month);
        assert_eq!(first_day.date.month(), 3);
    }

    #[test]
    fn event_on_date() {
        let evt = CalendarEvent::new("1", "Meeting", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0));
        assert!(evt.is_on_date(d(2026, 3, 8)));
        assert!(!evt.is_on_date(d(2026, 3, 9)));
    }

    #[test]
    fn multi_day_event_range() {
        let evt = CalendarEvent::new("1", "Conference", dt(2026, 3, 5, 9, 0), dt(2026, 3, 7, 17, 0));
        assert!(evt.overlaps_range(d(2026, 3, 6), d(2026, 3, 6)));
        assert!(evt.overlaps_range(d(2026, 3, 1), d(2026, 3, 5)));
        assert!(!evt.overlaps_range(d(2026, 3, 8), d(2026, 3, 10)));
    }

    #[test]
    fn event_store_range_query() {
        let mut store = EventStore::new();
        store.add(CalendarEvent::new("1", "A", dt(2026, 3, 5, 9, 0), dt(2026, 3, 5, 10, 0)));
        store.add(CalendarEvent::new("2", "B", dt(2026, 3, 10, 14, 0), dt(2026, 3, 10, 15, 0)));
        store.add(CalendarEvent::new("3", "C", dt(2026, 3, 20, 8, 0), dt(2026, 3, 20, 9, 0)));

        let found = store.events_in_range(d(2026, 3, 1), d(2026, 3, 7));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "1");

        let found_all = store.events_in_range(d(2026, 3, 1), d(2026, 3, 31));
        assert_eq!(found_all.len(), 3);
    }

    #[test]
    fn event_store_remove() {
        let mut store = EventStore::new();
        store.add(CalendarEvent::new("1", "A", dt(2026, 3, 5, 9, 0), dt(2026, 3, 5, 10, 0)));
        assert_eq!(store.len(), 1);
        let removed = store.remove("1");
        assert!(removed.is_some());
        assert_eq!(store.len(), 0);
        assert!(store.remove("nonexistent").is_none());
    }

    #[test]
    fn all_day_event() {
        let evt = CalendarEvent::all_day("1", "Holiday", d(2026, 3, 8));
        assert!(evt.all_day);
        assert!(evt.is_on_date(d(2026, 3, 8)));
        assert!(!evt.is_on_date(d(2026, 3, 9)));
    }

    #[test]
    fn week_view_slots() {
        let wv = WeekView::generate(d(2026, 3, 8), 30, 8, 18);
        assert_eq!(wv.columns.len(), 7);
        // 8:00 to 17:30 in 30-min slots = 20 slots per day.
        assert_eq!(wv.columns[0].slots.len(), 20);
        assert_eq!(wv.columns[0].slots[0].time, NaiveTime::from_hms_opt(8, 0, 0).unwrap());
    }

    #[test]
    fn week_view_column_lookup() {
        let wv = WeekView::generate(d(2026, 3, 8), 60, 9, 17);
        assert!(wv.column_for_date(d(2026, 3, 10)).is_some());
        assert!(wv.column_for_date(d(2026, 3, 20)).is_none());
    }

    #[test]
    fn event_with_color() {
        let evt = CalendarEvent::new("1", "Meeting", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0))
            .with_color("#ff0000");
        assert_eq!(evt.color.as_deref(), Some("#ff0000"));
    }
}
