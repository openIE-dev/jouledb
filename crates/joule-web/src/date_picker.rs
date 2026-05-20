//! Date picker state machine — single-date and range selection with constraints.
//!
//! Replaces react-datepicker / flatpickr with a headless, pure-Rust state machine.
//! No DOM — only state transitions for month/year/decade navigation, selection,
//! and constraint enforcement.

use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};

// ── View Mode ──────────────────────────────────────────────────

/// Which level of the picker is currently displayed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PickerView {
    Month,
    Year,
    Decade,
}

// ── Date Picker State ──────────────────────────────────────────

/// Headless state for a single-date picker.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(unpredictable_function_pointer_comparisons)]
pub struct DatePickerState {
    pub view: PickerView,
    pub selected_date: Option<NaiveDate>,
    pub viewing_year: i32,
    pub viewing_month: u32,
    pub min_date: Option<NaiveDate>,
    pub max_date: Option<NaiveDate>,
    /// External predicate: returns true if the date should be disabled.
    disabled_fn: Option<fn(NaiveDate) -> bool>,
    pub disable_weekends: bool,
}

impl DatePickerState {
    /// Create a new date picker focused on the given year/month.
    pub fn new(year: i32, month: u32) -> Self {
        Self {
            view: PickerView::Month,
            selected_date: None,
            viewing_year: year,
            viewing_month: month,
            min_date: None,
            max_date: None,
            disabled_fn: None,
            disable_weekends: false,
        }
    }

    /// Set the minimum selectable date.
    pub fn with_min(mut self, date: NaiveDate) -> Self {
        self.min_date = Some(date);
        self
    }

    /// Set the maximum selectable date.
    pub fn with_max(mut self, date: NaiveDate) -> Self {
        self.max_date = Some(date);
        self
    }

    /// Disable weekends.
    pub fn with_weekends_disabled(mut self) -> Self {
        self.disable_weekends = true;
        self
    }

    /// Set a custom disabled-date predicate.
    pub fn with_disabled_fn(mut self, f: fn(NaiveDate) -> bool) -> Self {
        self.disabled_fn = Some(f);
        self
    }

    /// Is this date disabled?
    pub fn is_disabled(&self, date: NaiveDate) -> bool {
        if let Some(min) = self.min_date {
            if date < min {
                return true;
            }
        }
        if let Some(max) = self.max_date {
            if date > max {
                return true;
            }
        }
        if self.disable_weekends {
            let wd = date.weekday();
            if wd == chrono::Weekday::Sat || wd == chrono::Weekday::Sun {
                return true;
            }
        }
        if let Some(f) = self.disabled_fn {
            if f(date) {
                return true;
            }
        }
        false
    }

    /// Navigate to the previous month.
    pub fn prev_month(&mut self) {
        if self.viewing_month == 1 {
            self.viewing_month = 12;
            self.viewing_year -= 1;
        } else {
            self.viewing_month -= 1;
        }
    }

    /// Navigate to the next month.
    pub fn next_month(&mut self) {
        if self.viewing_month == 12 {
            self.viewing_month = 1;
            self.viewing_year += 1;
        } else {
            self.viewing_month += 1;
        }
    }

    /// Navigate to the previous year.
    pub fn prev_year(&mut self) {
        self.viewing_year -= 1;
    }

    /// Navigate to the next year.
    pub fn next_year(&mut self) {
        self.viewing_year += 1;
    }

    /// Switch to month view.
    pub fn show_months(&mut self) {
        self.view = PickerView::Month;
    }

    /// Switch to year view (12 months).
    pub fn show_years(&mut self) {
        self.view = PickerView::Year;
    }

    /// Switch to decade view.
    pub fn show_decade(&mut self) {
        self.view = PickerView::Decade;
    }

    /// Try to select a date. Returns false if the date is disabled.
    pub fn select(&mut self, date: NaiveDate) -> bool {
        if self.is_disabled(date) {
            return false;
        }
        self.selected_date = Some(date);
        self.viewing_year = date.year();
        self.viewing_month = date.month();
        true
    }

    /// Select a month from the year view.
    pub fn select_month(&mut self, month: u32) {
        self.viewing_month = month;
        self.view = PickerView::Month;
    }

    /// Select a year from the decade view.
    pub fn select_year(&mut self, year: i32) {
        self.viewing_year = year;
        self.view = PickerView::Year;
    }

    /// Jump to today's date (navigates the view, does not select).
    pub fn go_to_today(&mut self, today: NaiveDate) {
        self.viewing_year = today.year();
        self.viewing_month = today.month();
        self.view = PickerView::Month;
    }

    /// Select today's date if it's not disabled.
    pub fn select_today(&mut self, today: NaiveDate) -> bool {
        self.go_to_today(today);
        self.select(today)
    }

    /// The decade range for decade view (e.g. 2020..=2029).
    pub fn decade_range(&self) -> (i32, i32) {
        let start = (self.viewing_year / 10) * 10;
        (start, start + 9)
    }
}

// ── Range Picker ───────────────────────────────────────────────

/// Which end of the range is being selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RangeEnd {
    Start,
    End,
}

/// State for a date range picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DateRangePickerState {
    pub inner: DatePickerState,
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    pub selecting: RangeEnd,
}

impl DateRangePickerState {
    pub fn new(year: i32, month: u32) -> Self {
        Self {
            inner: DatePickerState::new(year, month),
            start_date: None,
            end_date: None,
            selecting: RangeEnd::Start,
        }
    }

    /// Select a date in the range. Automatically advances from start to end.
    pub fn select(&mut self, date: NaiveDate) -> bool {
        if self.inner.is_disabled(date) {
            return false;
        }
        match self.selecting {
            RangeEnd::Start => {
                self.start_date = Some(date);
                self.end_date = None;
                self.selecting = RangeEnd::End;
            }
            RangeEnd::End => {
                if let Some(start) = self.start_date {
                    if date < start {
                        // Swap: the clicked date becomes the new start.
                        self.start_date = Some(date);
                        self.end_date = Some(start);
                    } else {
                        self.end_date = Some(date);
                    }
                } else {
                    self.start_date = Some(date);
                }
                self.selecting = RangeEnd::Start;
            }
        }
        self.inner.viewing_year = date.year();
        self.inner.viewing_month = date.month();
        true
    }

    /// Is the given date within the selected range (inclusive)?
    pub fn is_in_range(&self, date: NaiveDate) -> bool {
        match (self.start_date, self.end_date) {
            (Some(s), Some(e)) => date >= s && date <= e,
            _ => false,
        }
    }

    /// Number of days in the selected range (inclusive), or None.
    pub fn range_days(&self) -> Option<i64> {
        match (self.start_date, self.end_date) {
            (Some(s), Some(e)) => Some((e - s).num_days() + 1),
            _ => None,
        }
    }

    /// Reset the range selection.
    pub fn clear(&mut self) {
        self.start_date = None;
        self.end_date = None;
        self.selecting = RangeEnd::Start;
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
    fn initial_state() {
        let picker = DatePickerState::new(2026, 3);
        assert_eq!(picker.view, PickerView::Month);
        assert_eq!(picker.viewing_year, 2026);
        assert_eq!(picker.viewing_month, 3);
        assert!(picker.selected_date.is_none());
    }

    #[test]
    fn navigate_months() {
        let mut picker = DatePickerState::new(2026, 1);
        picker.prev_month();
        assert_eq!(picker.viewing_month, 12);
        assert_eq!(picker.viewing_year, 2025);
        picker.next_month();
        assert_eq!(picker.viewing_month, 1);
        assert_eq!(picker.viewing_year, 2026);
        picker.next_month();
        assert_eq!(picker.viewing_month, 2);
    }

    #[test]
    fn navigate_years() {
        let mut picker = DatePickerState::new(2026, 6);
        picker.prev_year();
        assert_eq!(picker.viewing_year, 2025);
        picker.next_year();
        picker.next_year();
        assert_eq!(picker.viewing_year, 2027);
    }

    #[test]
    fn select_date() {
        let mut picker = DatePickerState::new(2026, 3);
        assert!(picker.select(d(2026, 3, 15)));
        assert_eq!(picker.selected_date, Some(d(2026, 3, 15)));
    }

    #[test]
    fn min_max_constraints() {
        let picker = DatePickerState::new(2026, 3)
            .with_min(d(2026, 3, 5))
            .with_max(d(2026, 3, 25));

        assert!(picker.is_disabled(d(2026, 3, 1)));
        assert!(!picker.is_disabled(d(2026, 3, 10)));
        assert!(picker.is_disabled(d(2026, 3, 30)));
    }

    #[test]
    fn select_disabled_date_fails() {
        let mut picker = DatePickerState::new(2026, 3)
            .with_min(d(2026, 3, 5));
        assert!(!picker.select(d(2026, 3, 1)));
        assert!(picker.selected_date.is_none());
    }

    #[test]
    fn disable_weekends() {
        let picker = DatePickerState::new(2026, 3).with_weekends_disabled();
        // March 7, 2026 = Saturday
        assert!(picker.is_disabled(d(2026, 3, 7)));
        // March 8, 2026 = Sunday
        assert!(picker.is_disabled(d(2026, 3, 8)));
        // March 9, 2026 = Monday
        assert!(!picker.is_disabled(d(2026, 3, 9)));
    }

    #[test]
    fn custom_disabled_fn() {
        // Disable the 13th of any month.
        let picker = DatePickerState::new(2026, 3)
            .with_disabled_fn(|d| d.day() == 13);
        assert!(picker.is_disabled(d(2026, 3, 13)));
        assert!(!picker.is_disabled(d(2026, 3, 14)));
    }

    #[test]
    fn today_shortcut() {
        let mut picker = DatePickerState::new(2026, 1);
        let today = d(2026, 3, 8);
        assert!(picker.select_today(today));
        assert_eq!(picker.selected_date, Some(today));
        assert_eq!(picker.viewing_month, 3);
    }

    #[test]
    fn decade_range() {
        let picker = DatePickerState::new(2026, 3);
        assert_eq!(picker.decade_range(), (2020, 2029));
    }

    #[test]
    fn view_switching() {
        let mut picker = DatePickerState::new(2026, 3);
        picker.show_years();
        assert_eq!(picker.view, PickerView::Year);
        picker.select_month(6);
        assert_eq!(picker.view, PickerView::Month);
        assert_eq!(picker.viewing_month, 6);

        picker.show_decade();
        assert_eq!(picker.view, PickerView::Decade);
        picker.select_year(2028);
        assert_eq!(picker.view, PickerView::Year);
        assert_eq!(picker.viewing_year, 2028);
    }

    #[test]
    fn range_picker_basic() {
        let mut rp = DateRangePickerState::new(2026, 3);
        assert!(rp.select(d(2026, 3, 5)));
        assert_eq!(rp.start_date, Some(d(2026, 3, 5)));
        assert!(rp.end_date.is_none());
        assert_eq!(rp.selecting, RangeEnd::End);

        assert!(rp.select(d(2026, 3, 10)));
        assert_eq!(rp.start_date, Some(d(2026, 3, 5)));
        assert_eq!(rp.end_date, Some(d(2026, 3, 10)));
        assert_eq!(rp.range_days(), Some(6));
    }

    #[test]
    fn range_picker_swap_on_earlier_end() {
        let mut rp = DateRangePickerState::new(2026, 3);
        rp.select(d(2026, 3, 10));
        rp.select(d(2026, 3, 5)); // earlier than start
        assert_eq!(rp.start_date, Some(d(2026, 3, 5)));
        assert_eq!(rp.end_date, Some(d(2026, 3, 10)));
    }

    #[test]
    fn range_picker_is_in_range() {
        let mut rp = DateRangePickerState::new(2026, 3);
        rp.select(d(2026, 3, 5));
        rp.select(d(2026, 3, 10));
        assert!(rp.is_in_range(d(2026, 3, 7)));
        assert!(!rp.is_in_range(d(2026, 3, 11)));
    }

    #[test]
    fn range_picker_clear() {
        let mut rp = DateRangePickerState::new(2026, 3);
        rp.select(d(2026, 3, 5));
        rp.select(d(2026, 3, 10));
        rp.clear();
        assert!(rp.start_date.is_none());
        assert!(rp.end_date.is_none());
        assert_eq!(rp.selecting, RangeEnd::Start);
    }
}
