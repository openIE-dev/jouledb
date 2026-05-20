//! Event scheduling — resources, conflict detection, and time slot management.
//!
//! Replaces DayPilot / DHTMLX Scheduler with a pure-Rust scheduling engine.
//! No DOM — only the data model for events, resources, availability, and
//! drag-to-move/resize operations.

use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike, Weekday};
use serde::{Deserialize, Serialize};

// ── Time Slot Snapping ─────────────────────────────────────────

/// Granularity for time slot snapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapInterval {
    FifteenMin,
    ThirtyMin,
    OneHour,
}

impl SnapInterval {
    /// Minutes per snap interval.
    pub fn minutes(&self) -> u32 {
        match self {
            Self::FifteenMin => 15,
            Self::ThirtyMin => 30,
            Self::OneHour => 60,
        }
    }

    /// Snap a time to the nearest interval boundary (rounding down).
    pub fn snap(&self, time: NaiveTime) -> NaiveTime {
        let total_min = Timelike::hour(&time) * 60 + Timelike::minute(&time);
        let snapped = (total_min / self.minutes()) * self.minutes();
        let h = snapped / 60;
        let m = snapped % 60;
        NaiveTime::from_hms_opt(h.min(23), m, 0).unwrap()
    }

    /// Snap a datetime to the nearest interval boundary (rounding down).
    pub fn snap_datetime(&self, dt: NaiveDateTime) -> NaiveDateTime {
        let snapped_time = self.snap(dt.time());
        dt.date().and_time(snapped_time)
    }
}

// ── Resource ───────────────────────────────────────────────────

/// A schedulable resource (room, person, equipment, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resource {
    pub id: String,
    pub name: String,
    /// Working hours for this resource (None = 24h availability).
    pub working_hours: Option<WorkingHours>,
}

/// Working hours definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkingHours {
    pub start: NaiveTime,
    pub end: NaiveTime,
    /// Which days are working days (true = working).
    pub weekdays: [bool; 7], // Sun=0 .. Sat=6
}

impl WorkingHours {
    /// Standard Mon–Fri 9am–5pm.
    pub fn business_hours() -> Self {
        Self {
            start: NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            end: NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            weekdays: [false, true, true, true, true, true, false],
        }
    }

    /// Is this datetime within working hours?
    pub fn is_working(&self, dt: NaiveDateTime) -> bool {
        let day_idx = dt.weekday().num_days_from_sunday() as usize;
        if !self.weekdays[day_idx] {
            return false;
        }
        let t = dt.time();
        t >= self.start && t < self.end
    }
}

impl Resource {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            working_hours: None,
        }
    }

    pub fn with_working_hours(mut self, wh: WorkingHours) -> Self {
        self.working_hours = Some(wh);
        self
    }
}

// ── Scheduler Event ────────────────────────────────────────────

/// An event scheduled on a resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerEvent {
    pub id: String,
    pub title: String,
    pub start: NaiveDateTime,
    pub end: NaiveDateTime,
    pub resource_id: String,
}

impl SchedulerEvent {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        start: NaiveDateTime,
        end: NaiveDateTime,
        resource_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            start,
            end,
            resource_id: resource_id.into(),
        }
    }

    /// Duration in minutes.
    pub fn duration_minutes(&self) -> i64 {
        (self.end - self.start).num_minutes()
    }

    /// Do two events overlap in time?
    pub fn overlaps(&self, other: &SchedulerEvent) -> bool {
        self.start < other.end && self.end > other.start
    }
}

// ── Scheduler ──────────────────────────────────────────────────

/// The core scheduler: holds resources and events, provides conflict
/// detection and availability queries.
#[derive(Debug, Clone, Default)]
pub struct Scheduler {
    pub resources: Vec<Resource>,
    pub events: Vec<SchedulerEvent>,
    pub snap: Option<SnapInterval>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_snap(mut self, snap: SnapInterval) -> Self {
        self.snap = Some(snap);
        self
    }

    pub fn add_resource(&mut self, resource: Resource) {
        self.resources.push(resource);
    }

    /// Add an event, returning any conflicting event IDs.
    pub fn add_event(&mut self, event: SchedulerEvent) -> Vec<String> {
        let conflicts = self.find_conflicts(&event);
        self.events.push(event);
        conflicts
    }

    /// Find events that conflict with the given event (same resource, overlapping time).
    pub fn find_conflicts(&self, event: &SchedulerEvent) -> Vec<String> {
        self.events
            .iter()
            .filter(|e| e.resource_id == event.resource_id && e.id != event.id && e.overlaps(event))
            .map(|e| e.id.clone())
            .collect()
    }

    /// Is the resource free during the given time range?
    pub fn is_available(&self, resource_id: &str, start: NaiveDateTime, end: NaiveDateTime) -> bool {
        let probe = SchedulerEvent::new("__probe__", "", start, end, resource_id);
        self.find_conflicts(&probe).is_empty()
    }

    /// Events on a specific resource.
    pub fn events_for_resource(&self, resource_id: &str) -> Vec<&SchedulerEvent> {
        self.events.iter().filter(|e| e.resource_id == resource_id).collect()
    }

    /// Events on a specific date across all resources.
    pub fn events_on_date(&self, date: NaiveDate) -> Vec<&SchedulerEvent> {
        let day_start = date.and_hms_opt(0, 0, 0).unwrap();
        let day_end = date.and_hms_opt(23, 59, 59).unwrap();
        self.events
            .iter()
            .filter(|e| e.start <= day_end && e.end >= day_start)
            .collect()
    }

    /// Events in a date range.
    pub fn events_in_range(&self, start: NaiveDate, end: NaiveDate) -> Vec<&SchedulerEvent> {
        let range_start = start.and_hms_opt(0, 0, 0).unwrap();
        let range_end = end.and_hms_opt(23, 59, 59).unwrap();
        self.events
            .iter()
            .filter(|e| e.start <= range_end && e.end >= range_start)
            .collect()
    }

    /// Remove an event by ID.
    pub fn remove_event(&mut self, id: &str) -> Option<SchedulerEvent> {
        if let Some(pos) = self.events.iter().position(|e| e.id == id) {
            Some(self.events.remove(pos))
        } else {
            None
        }
    }

    /// Move an event to a new time (drag model). Snaps if configured.
    /// Returns conflicting event IDs at the new position.
    pub fn move_event(&mut self, id: &str, new_start: NaiveDateTime) -> Option<Vec<String>> {
        let idx = self.events.iter().position(|e| e.id == id)?;
        let duration = self.events[idx].end - self.events[idx].start;
        let snapped_start = if let Some(snap) = self.snap {
            snap.snap_datetime(new_start)
        } else {
            new_start
        };
        self.events[idx].start = snapped_start;
        self.events[idx].end = snapped_start + duration;

        let event_clone = self.events[idx].clone();
        Some(self.find_conflicts(&event_clone))
    }

    /// Resize an event's end time (drag-to-resize model). Snaps if configured.
    pub fn resize_event(&mut self, id: &str, new_end: NaiveDateTime) -> Option<Vec<String>> {
        let idx = self.events.iter().position(|e| e.id == id)?;
        let snapped_end = if let Some(snap) = self.snap {
            snap.snap_datetime(new_end)
        } else {
            new_end
        };
        if snapped_end <= self.events[idx].start {
            return Some(vec![]); // can't resize to zero or negative
        }
        self.events[idx].end = snapped_end;
        let event_clone = self.events[idx].clone();
        Some(self.find_conflicts(&event_clone))
    }
}

// ── View Helpers ───────────────────────────────────────────────

/// Day view: events for a single day grouped by resource.
pub fn day_view(scheduler: &Scheduler, date: NaiveDate) -> Vec<(&Resource, Vec<&SchedulerEvent>)> {
    scheduler
        .resources
        .iter()
        .map(|r| {
            let events: Vec<_> = scheduler
                .events_on_date(date)
                .into_iter()
                .filter(|e| e.resource_id == r.id)
                .collect();
            (r, events)
        })
        .collect()
}

/// Week view: events for 7 days starting from `start_date`.
pub fn week_view(scheduler: &Scheduler, start_date: NaiveDate) -> Vec<(NaiveDate, Vec<&SchedulerEvent>)> {
    (0..7)
        .map(|offset| {
            let date = start_date + chrono::Duration::days(offset);
            let events = scheduler.events_on_date(date);
            (date, events)
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dt(y: i32, m: u32, day: u32, h: u32, mi: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
            .and_hms_opt(h, mi, 0).unwrap()
    }

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn no_conflict_different_resources() {
        let mut sched = Scheduler::new();
        sched.add_resource(Resource::new("r1", "Room A"));
        sched.add_resource(Resource::new("r2", "Room B"));

        let c1 = sched.add_event(SchedulerEvent::new("1", "Meeting", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0), "r1"));
        assert!(c1.is_empty());

        let c2 = sched.add_event(SchedulerEvent::new("2", "Workshop", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0), "r2"));
        assert!(c2.is_empty());
    }

    #[test]
    fn conflict_same_resource_overlapping() {
        let mut sched = Scheduler::new();
        sched.add_resource(Resource::new("r1", "Room A"));
        sched.add_event(SchedulerEvent::new("1", "A", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 12, 0), "r1"));

        let conflicts = sched.add_event(SchedulerEvent::new("2", "B", dt(2026, 3, 8, 11, 0), dt(2026, 3, 8, 13, 0), "r1"));
        assert_eq!(conflicts, vec!["1"]);
    }

    #[test]
    fn no_conflict_adjacent() {
        let mut sched = Scheduler::new();
        sched.add_resource(Resource::new("r1", "Room A"));
        sched.add_event(SchedulerEvent::new("1", "A", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0), "r1"));

        // Starts exactly when the other ends — no overlap.
        let c = sched.add_event(SchedulerEvent::new("2", "B", dt(2026, 3, 8, 11, 0), dt(2026, 3, 8, 12, 0), "r1"));
        assert!(c.is_empty());
    }

    #[test]
    fn availability_check() {
        let mut sched = Scheduler::new();
        sched.add_resource(Resource::new("r1", "Room A"));
        sched.add_event(SchedulerEvent::new("1", "A", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 12, 0), "r1"));

        assert!(!sched.is_available("r1", dt(2026, 3, 8, 11, 0), dt(2026, 3, 8, 13, 0)));
        assert!(sched.is_available("r1", dt(2026, 3, 8, 12, 0), dt(2026, 3, 8, 14, 0)));
        assert!(sched.is_available("r1", dt(2026, 3, 8, 8, 0), dt(2026, 3, 8, 10, 0)));
    }

    #[test]
    fn events_on_date() {
        let mut sched = Scheduler::new();
        sched.add_resource(Resource::new("r1", "Room A"));
        sched.add_event(SchedulerEvent::new("1", "A", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0), "r1"));
        sched.add_event(SchedulerEvent::new("2", "B", dt(2026, 3, 9, 10, 0), dt(2026, 3, 9, 11, 0), "r1"));

        let on_8 = sched.events_on_date(d(2026, 3, 8));
        assert_eq!(on_8.len(), 1);
        assert_eq!(on_8[0].id, "1");
    }

    #[test]
    fn snap_fifteen_min() {
        let snap = SnapInterval::FifteenMin;
        let t = NaiveTime::from_hms_opt(10, 22, 0).unwrap();
        assert_eq!(snap.snap(t), NaiveTime::from_hms_opt(10, 15, 0).unwrap());
    }

    #[test]
    fn snap_one_hour() {
        let snap = SnapInterval::OneHour;
        let t = NaiveTime::from_hms_opt(10, 45, 0).unwrap();
        assert_eq!(snap.snap(t), NaiveTime::from_hms_opt(10, 0, 0).unwrap());
    }

    #[test]
    fn move_event() {
        let mut sched = Scheduler::new().with_snap(SnapInterval::ThirtyMin);
        sched.add_resource(Resource::new("r1", "Room A"));
        sched.add_event(SchedulerEvent::new("1", "A", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0), "r1"));

        let conflicts = sched.move_event("1", dt(2026, 3, 8, 14, 17)).unwrap();
        assert!(conflicts.is_empty());
        // Should snap to 14:00.
        assert_eq!(sched.events[0].start, dt(2026, 3, 8, 14, 0));
        assert_eq!(sched.events[0].end, dt(2026, 3, 8, 15, 0));
    }

    #[test]
    fn resize_event() {
        let mut sched = Scheduler::new();
        sched.add_resource(Resource::new("r1", "Room A"));
        sched.add_event(SchedulerEvent::new("1", "A", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0), "r1"));

        let conflicts = sched.resize_event("1", dt(2026, 3, 8, 12, 30)).unwrap();
        assert!(conflicts.is_empty());
        assert_eq!(sched.events[0].end, dt(2026, 3, 8, 12, 30));
    }

    #[test]
    fn working_hours() {
        let wh = WorkingHours::business_hours();
        // Monday 10am — working
        assert!(wh.is_working(dt(2026, 3, 9, 10, 0)));
        // Sunday 10am — not working
        assert!(!wh.is_working(dt(2026, 3, 8, 10, 0)));
        // Monday 6pm — after hours
        assert!(!wh.is_working(dt(2026, 3, 9, 18, 0)));
    }

    #[test]
    fn day_view_groups_by_resource() {
        let mut sched = Scheduler::new();
        sched.add_resource(Resource::new("r1", "Room A"));
        sched.add_resource(Resource::new("r2", "Room B"));
        sched.add_event(SchedulerEvent::new("1", "A", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0), "r1"));
        sched.add_event(SchedulerEvent::new("2", "B", dt(2026, 3, 8, 14, 0), dt(2026, 3, 8, 15, 0), "r2"));

        let view = day_view(&sched, d(2026, 3, 8));
        assert_eq!(view.len(), 2);
        assert_eq!(view[0].1.len(), 1);
        assert_eq!(view[0].1[0].id, "1");
        assert_eq!(view[1].1.len(), 1);
        assert_eq!(view[1].1[0].id, "2");
    }

    #[test]
    fn duration_minutes() {
        let evt = SchedulerEvent::new("1", "A", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 30), "r1");
        assert_eq!(evt.duration_minutes(), 90);
    }

    #[test]
    fn remove_event() {
        let mut sched = Scheduler::new();
        sched.add_resource(Resource::new("r1", "Room A"));
        sched.add_event(SchedulerEvent::new("1", "A", dt(2026, 3, 8, 10, 0), dt(2026, 3, 8, 11, 0), "r1"));
        assert_eq!(sched.events.len(), 1);
        sched.remove_event("1");
        assert_eq!(sched.events.len(), 0);
    }
}
