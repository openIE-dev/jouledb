//! Stream windowing — tumbling, sliding, and session windows with watermarks,
//! late arrival handling, window triggers, and window aggregation.
//!
//! Replaces JS stream-processing window libraries (Flink-style, RxJS windowTime)
//! with a pure-Rust windowing engine that tracks energy per window operation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Windowing errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowError {
    /// Window not found.
    WindowNotFound(String),
    /// Invalid window configuration.
    InvalidConfig(String),
    /// Event arrived too late (past watermark + allowed lateness).
    TooLate { event_time: i64, watermark: i64 },
    /// Window already closed.
    WindowClosed(String),
    /// Empty window — no events to aggregate.
    EmptyWindow(String),
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WindowNotFound(id) => write!(f, "window not found: {id}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::TooLate { event_time, watermark } => {
                write!(f, "event time {event_time} is past watermark {watermark}")
            }
            Self::WindowClosed(id) => write!(f, "window closed: {id}"),
            Self::EmptyWindow(id) => write!(f, "empty window: {id}"),
        }
    }
}

impl std::error::Error for WindowError {}

// ── Window Types ────────────────────────────────────────────────

/// The kind of window.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WindowKind {
    /// Fixed-size, non-overlapping windows.
    Tumbling,
    /// Fixed-size, overlapping windows with a slide interval.
    Sliding,
    /// Activity-based windows that close after a gap of inactivity.
    Session,
}

/// Trigger condition for firing a window.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TriggerKind {
    /// Fire when watermark passes window end.
    OnWatermark,
    /// Fire after N events arrive.
    OnCount(u64),
    /// Fire at a specific processing time interval (ms).
    OnProcessingTime(u64),
    /// Fire on every event (continuous).
    OnEveryEvent,
}

/// Window state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WindowState {
    Open,
    Closed,
    Fired,
}

// ── Structs ─────────────────────────────────────────────────────

/// Configuration for a windowing strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    pub kind: WindowKind,
    /// Window size in milliseconds.
    pub size_ms: u64,
    /// Slide interval (only for Sliding windows) in ms.
    pub slide_ms: Option<u64>,
    /// Session gap timeout in milliseconds (only for Session windows).
    pub session_gap_ms: Option<u64>,
    /// Maximum allowed lateness in milliseconds.
    pub allowed_lateness_ms: u64,
    /// Trigger kind.
    pub trigger: TriggerKind,
}

/// A single event entering the windowing system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    pub key: String,
    pub event_time_ms: i64,
    pub processing_time_ms: i64,
    pub value: f64,
    pub metadata: HashMap<String, String>,
}

/// A materialized window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Window {
    pub id: String,
    pub key: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub state: WindowState,
    pub events: Vec<StreamEvent>,
    pub created_at: DateTime<Utc>,
}

impl Window {
    fn contains(&self, event_time_ms: i64) -> bool {
        event_time_ms >= self.start_ms && event_time_ms < self.end_ms
    }
}

/// Aggregation result for a window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowAggregate {
    pub window_id: String,
    pub key: String,
    pub count: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub avg: f64,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Late event that was accepted within allowed lateness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateEvent {
    pub event: StreamEvent,
    pub assigned_window_id: String,
    pub lateness_ms: i64,
}

// ── Watermark ───────────────────────────────────────────────────

/// Watermark tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Watermark {
    /// Current watermark position in event time ms.
    pub position_ms: i64,
    /// How far behind max event time the watermark trails.
    pub lag_ms: u64,
    /// Maximum observed event time.
    max_event_time_ms: i64,
}

impl Watermark {
    pub fn new(lag_ms: u64) -> Self {
        Self {
            position_ms: 0,
            lag_ms,
            max_event_time_ms: 0,
        }
    }

    /// Update watermark with a new event time.
    pub fn advance(&mut self, event_time_ms: i64) {
        if event_time_ms > self.max_event_time_ms {
            self.max_event_time_ms = event_time_ms;
            self.position_ms = self.max_event_time_ms - self.lag_ms as i64;
        }
    }
}

// ── WindowManager ───────────────────────────────────────────────

/// Manages windows for streaming events.
#[derive(Debug, Clone)]
pub struct WindowManager {
    config: WindowConfig,
    windows: Vec<Window>,
    watermark: Watermark,
    late_events: Vec<LateEvent>,
    next_window_seq: u64,
    total_energy_uj: u64,
}

impl WindowManager {
    pub fn new(config: WindowConfig, watermark_lag_ms: u64) -> Result<Self, WindowError> {
        if config.size_ms == 0 {
            return Err(WindowError::InvalidConfig("size_ms must be > 0".into()));
        }
        if config.kind == WindowKind::Sliding {
            if let Some(slide) = config.slide_ms {
                if slide == 0 || slide > config.size_ms {
                    return Err(WindowError::InvalidConfig(
                        "slide_ms must be > 0 and <= size_ms".into(),
                    ));
                }
            } else {
                return Err(WindowError::InvalidConfig(
                    "sliding window requires slide_ms".into(),
                ));
            }
        }
        if config.kind == WindowKind::Session {
            if config.session_gap_ms.is_none() || config.session_gap_ms == Some(0) {
                return Err(WindowError::InvalidConfig(
                    "session window requires session_gap_ms > 0".into(),
                ));
            }
        }
        Ok(Self {
            config,
            windows: Vec::new(),
            watermark: Watermark::new(watermark_lag_ms),
            late_events: Vec::new(),
            next_window_seq: 0,
            total_energy_uj: 0,
        })
    }

    /// Ingest a single event into the appropriate window(s).
    pub fn ingest(&mut self, event: StreamEvent) -> Result<Vec<String>, WindowError> {
        self.total_energy_uj += 5;
        self.watermark.advance(event.event_time_ms);

        match self.config.kind {
            WindowKind::Tumbling => self.ingest_tumbling(event),
            WindowKind::Sliding => self.ingest_sliding(event),
            WindowKind::Session => self.ingest_session(event),
        }
    }

    /// Fire (close + aggregate) all windows whose end time is at or before the watermark.
    pub fn fire_eligible(&mut self) -> Vec<WindowAggregate> {
        let wm = self.watermark.position_ms;
        let mut aggregates = Vec::new();

        for window in &mut self.windows {
            if window.state == WindowState::Open && window.end_ms <= wm {
                window.state = WindowState::Fired;
                if !window.events.is_empty() {
                    aggregates.push(Self::compute_aggregate(window));
                }
            }
        }
        self.total_energy_uj += aggregates.len() as u64 * 10;
        aggregates
    }

    /// Force-fire a specific window by id.
    pub fn force_fire(&mut self, window_id: &str) -> Result<WindowAggregate, WindowError> {
        let window = self
            .windows
            .iter_mut()
            .find(|w| w.id == window_id)
            .ok_or_else(|| WindowError::WindowNotFound(window_id.to_string()))?;

        if window.state == WindowState::Fired || window.state == WindowState::Closed {
            return Err(WindowError::WindowClosed(window_id.to_string()));
        }
        if window.events.is_empty() {
            return Err(WindowError::EmptyWindow(window_id.to_string()));
        }
        window.state = WindowState::Fired;
        self.total_energy_uj += 10;
        Ok(Self::compute_aggregate(window))
    }

    /// Close all fired windows (mark as Closed, preventing further additions).
    pub fn close_fired(&mut self) -> usize {
        let mut count = 0;
        for w in &mut self.windows {
            if w.state == WindowState::Fired {
                w.state = WindowState::Closed;
                count += 1;
            }
        }
        count
    }

    /// Get a window by id.
    pub fn get_window(&self, window_id: &str) -> Option<&Window> {
        self.windows.iter().find(|w| w.id == window_id)
    }

    /// All current windows.
    pub fn windows(&self) -> &[Window] {
        &self.windows
    }

    /// All late events collected so far.
    pub fn late_events(&self) -> &[LateEvent] {
        &self.late_events
    }

    /// Current watermark position.
    pub fn watermark(&self) -> &Watermark {
        &self.watermark
    }

    /// Total energy consumed.
    pub fn total_energy_uj(&self) -> u64 {
        self.total_energy_uj
    }

    /// Count open windows.
    pub fn open_window_count(&self) -> usize {
        self.windows.iter().filter(|w| w.state == WindowState::Open).count()
    }

    // ── Internal ────────────────────────────────────────────────

    fn ingest_tumbling(&mut self, event: StreamEvent) -> Result<Vec<String>, WindowError> {
        let size = self.config.size_ms as i64;
        let window_start = (event.event_time_ms / size) * size;
        let window_end = window_start + size;

        // Check if we have this window for this key.
        let existing = self.windows.iter_mut().find(|w| {
            w.key == event.key && w.start_ms == window_start && w.end_ms == window_end
        });

        match existing {
            Some(w) => {
                if w.state == WindowState::Closed {
                    return Err(WindowError::WindowClosed(w.id.clone()));
                }
                let is_late = event.event_time_ms < self.watermark.position_ms;
                if is_late {
                    let lateness = self.watermark.position_ms - event.event_time_ms;
                    if lateness > self.config.allowed_lateness_ms as i64 {
                        return Err(WindowError::TooLate {
                            event_time: event.event_time_ms,
                            watermark: self.watermark.position_ms,
                        });
                    }
                    self.late_events.push(LateEvent {
                        assigned_window_id: w.id.clone(),
                        lateness_ms: lateness,
                        event,
                    });
                    let wid = self.late_events.last().unwrap().assigned_window_id.clone();
                    return Ok(vec![wid]);
                }
                let wid = w.id.clone();
                w.events.push(event);
                Ok(vec![wid])
            }
            None => {
                let wid = self.create_window(&event.key, window_start, window_end);
                let w = self.windows.last_mut().unwrap();
                w.events.push(event);
                Ok(vec![wid])
            }
        }
    }

    fn ingest_sliding(&mut self, event: StreamEvent) -> Result<Vec<String>, WindowError> {
        let size = self.config.size_ms as i64;
        let slide = self.config.slide_ms.unwrap() as i64;
        let mut assigned = Vec::new();

        // Find all windows this event belongs to.
        let earliest_start = ((event.event_time_ms - size).div_euclid(slide) + 1) * slide;
        let mut start = earliest_start.max(0);
        while start <= event.event_time_ms {
            let end = start + size;
            if event.event_time_ms >= start && event.event_time_ms < end {
                let existing = self.windows.iter_mut().find(|w| {
                    w.key == event.key && w.start_ms == start && w.end_ms == end
                });
                match existing {
                    Some(w) if w.state == WindowState::Open => {
                        assigned.push(w.id.clone());
                        w.events.push(event.clone());
                    }
                    Some(w) if w.state == WindowState::Fired => {
                        // Late event into a fired window.
                        let lateness = self.watermark.position_ms - event.event_time_ms;
                        if lateness <= self.config.allowed_lateness_ms as i64 {
                            self.late_events.push(LateEvent {
                                assigned_window_id: w.id.clone(),
                                lateness_ms: lateness.max(0),
                                event: event.clone(),
                            });
                            assigned.push(w.id.clone());
                        }
                    }
                    Some(_) => {} // Closed, skip.
                    None => {
                        let wid = self.create_window(&event.key, start, end);
                        let w = self.windows.last_mut().unwrap();
                        w.events.push(event.clone());
                        assigned.push(wid);
                    }
                }
            }
            start += slide;
        }
        Ok(assigned)
    }

    fn ingest_session(&mut self, event: StreamEvent) -> Result<Vec<String>, WindowError> {
        let gap = self.config.session_gap_ms.unwrap() as i64;

        // Find an existing open session window for this key where the event fits.
        let existing = self.windows.iter_mut().find(|w| {
            w.key == event.key
                && w.state == WindowState::Open
                && event.event_time_ms >= w.start_ms
                && event.event_time_ms < w.end_ms + gap
        });

        match existing {
            Some(w) => {
                let wid = w.id.clone();
                // Extend session window end if needed.
                let new_end = event.event_time_ms + gap;
                if new_end > w.end_ms {
                    w.end_ms = new_end;
                }
                w.events.push(event);
                Ok(vec![wid])
            }
            None => {
                let start = event.event_time_ms;
                let end = start + gap;
                let wid = self.create_window(&event.key, start, end);
                let w = self.windows.last_mut().unwrap();
                w.events.push(event);
                Ok(vec![wid])
            }
        }
    }

    fn create_window(&mut self, key: &str, start: i64, end: i64) -> String {
        let id = format!("w-{}", self.next_window_seq);
        self.next_window_seq += 1;
        self.windows.push(Window {
            id: id.clone(),
            key: key.to_string(),
            start_ms: start,
            end_ms: end,
            state: WindowState::Open,
            events: Vec::new(),
            created_at: Utc::now(),
        });
        id
    }

    fn compute_aggregate(window: &Window) -> WindowAggregate {
        let count = window.events.len() as u64;
        let sum: f64 = window.events.iter().map(|e| e.value).sum();
        let min = window
            .events
            .iter()
            .map(|e| e.value)
            .fold(f64::INFINITY, f64::min);
        let max = window
            .events
            .iter()
            .map(|e| e.value)
            .fold(f64::NEG_INFINITY, f64::max);
        let avg = if count > 0 { sum / count as f64 } else { 0.0 };

        WindowAggregate {
            window_id: window.id.clone(),
            key: window.key.clone(),
            count,
            sum,
            min,
            max,
            avg,
            start_ms: window.start_ms,
            end_ms: window.end_ms,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(key: &str, time_ms: i64, value: f64) -> StreamEvent {
        StreamEvent {
            key: key.to_string(),
            event_time_ms: time_ms,
            processing_time_ms: time_ms + 10,
            value,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_invalid_config_zero_size() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 0,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        assert!(WindowManager::new(cfg, 0).is_err());
    }

    #[test]
    fn test_invalid_sliding_no_slide() {
        let cfg = WindowConfig {
            kind: WindowKind::Sliding,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        assert!(WindowManager::new(cfg, 0).is_err());
    }

    #[test]
    fn test_invalid_sliding_slide_too_large() {
        let cfg = WindowConfig {
            kind: WindowKind::Sliding,
            size_ms: 1000,
            slide_ms: Some(2000),
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        assert!(WindowManager::new(cfg, 0).is_err());
    }

    #[test]
    fn test_invalid_session_no_gap() {
        let cfg = WindowConfig {
            kind: WindowKind::Session,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        assert!(WindowManager::new(cfg, 0).is_err());
    }

    #[test]
    fn test_tumbling_basic() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 100).unwrap();

        let ids1 = mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        assert_eq!(ids1.len(), 1);

        let ids2 = mgr.ingest(make_event("k1", 500, 2.0)).unwrap();
        assert_eq!(ids2[0], ids1[0]); // Same window [0, 1000).

        let ids3 = mgr.ingest(make_event("k1", 1100, 3.0)).unwrap();
        assert_ne!(ids3[0], ids1[0]); // Different window [1000, 2000).

        assert_eq!(mgr.windows().len(), 2);
    }

    #[test]
    fn test_tumbling_different_keys() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 100).unwrap();

        mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        mgr.ingest(make_event("k2", 200, 2.0)).unwrap();

        assert_eq!(mgr.windows().len(), 2);
        assert_eq!(mgr.windows()[0].key, "k1");
        assert_eq!(mgr.windows()[1].key, "k2");
    }

    #[test]
    fn test_fire_eligible() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 0).unwrap();

        mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        mgr.ingest(make_event("k1", 500, 2.0)).unwrap();
        // Advance watermark past window end (1000).
        mgr.ingest(make_event("k1", 1500, 3.0)).unwrap();

        let aggs = mgr.fire_eligible();
        assert_eq!(aggs.len(), 1);
        assert_eq!(aggs[0].count, 2);
        assert!((aggs[0].sum - 3.0).abs() < f64::EPSILON);
        assert!((aggs[0].avg - 1.5).abs() < f64::EPSILON);
        assert!((aggs[0].min - 1.0).abs() < f64::EPSILON);
        assert!((aggs[0].max - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_force_fire() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 1000).unwrap();

        let ids = mgr.ingest(make_event("k1", 100, 5.0)).unwrap();
        let agg = mgr.force_fire(&ids[0]).unwrap();
        assert_eq!(agg.count, 1);
        assert!((agg.sum - 5.0).abs() < f64::EPSILON);

        // Can't fire again.
        assert!(mgr.force_fire(&ids[0]).is_err());
    }

    #[test]
    fn test_force_fire_empty() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 1000).unwrap();

        assert!(mgr.force_fire("nonexistent").is_err());
    }

    #[test]
    fn test_close_fired() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 0).unwrap();

        mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        mgr.ingest(make_event("k1", 1500, 2.0)).unwrap();
        mgr.fire_eligible();

        let closed = mgr.close_fired();
        assert_eq!(closed, 1);
        assert_eq!(mgr.windows()[0].state, WindowState::Closed);
    }

    #[test]
    fn test_sliding_window() {
        let cfg = WindowConfig {
            kind: WindowKind::Sliding,
            size_ms: 1000,
            slide_ms: Some(500),
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 100).unwrap();

        // Event at 600 should go into window [0, 1000) and [500, 1500).
        let ids = mgr.ingest(make_event("k1", 600, 1.0)).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(mgr.windows().len(), 2);
    }

    #[test]
    fn test_session_window_basic() {
        let cfg = WindowConfig {
            kind: WindowKind::Session,
            size_ms: 1000, // Not used for sessions, but must be > 0.
            slide_ms: None,
            session_gap_ms: Some(500),
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 100).unwrap();

        // Events within 500ms gap.
        mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        mgr.ingest(make_event("k1", 300, 2.0)).unwrap();
        assert_eq!(mgr.windows().len(), 1);

        // Event far apart — new session.
        mgr.ingest(make_event("k1", 1500, 3.0)).unwrap();
        assert_eq!(mgr.windows().len(), 2);
    }

    #[test]
    fn test_session_window_extends() {
        let cfg = WindowConfig {
            kind: WindowKind::Session,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: Some(500),
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 100).unwrap();

        mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        // Window end is 100 + 500 = 600.
        let w = &mgr.windows()[0];
        assert_eq!(w.end_ms, 600);

        mgr.ingest(make_event("k1", 400, 2.0)).unwrap();
        // Window end extended to 400 + 500 = 900.
        let w = &mgr.windows()[0];
        assert_eq!(w.end_ms, 900);
    }

    #[test]
    fn test_watermark_advance() {
        let mut wm = Watermark::new(100);
        assert_eq!(wm.position_ms, 0);

        wm.advance(500);
        assert_eq!(wm.position_ms, 400);

        wm.advance(300); // Not advancing — old event.
        assert_eq!(wm.position_ms, 400);

        wm.advance(1000);
        assert_eq!(wm.position_ms, 900);
    }

    #[test]
    fn test_late_event_within_lateness() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 500,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 0).unwrap();

        // Create window [0, 1000).
        mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        // Advance watermark to 1100.
        mgr.ingest(make_event("k1", 1100, 2.0)).unwrap();

        // Late event at 800 — lateness = 1100 - 800 = 300 < 500.
        let ids = mgr.ingest(make_event("k1", 800, 3.0)).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(mgr.late_events().len(), 1);
        assert_eq!(mgr.late_events()[0].lateness_ms, 300);
    }

    #[test]
    fn test_too_late_event() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 100,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 0).unwrap();

        mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        mgr.ingest(make_event("k1", 2000, 2.0)).unwrap();

        // Late event at 200 — lateness = 2000 - 200 = 1800 >> 100.
        let result = mgr.ingest(make_event("k1", 200, 3.0));
        assert!(matches!(result, Err(WindowError::TooLate { .. })));
    }

    #[test]
    fn test_window_kind_serde() {
        let json = serde_json::to_string(&WindowKind::Session).unwrap();
        let parsed: WindowKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, WindowKind::Session);
    }

    #[test]
    fn test_trigger_kind_serde() {
        let t = TriggerKind::OnCount(42);
        let json = serde_json::to_string(&t).unwrap();
        let parsed: TriggerKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, TriggerKind::OnCount(42));
    }

    #[test]
    fn test_energy_tracking() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 100).unwrap();
        assert_eq!(mgr.total_energy_uj(), 0);

        mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        assert!(mgr.total_energy_uj() > 0);
    }

    #[test]
    fn test_open_window_count() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 0).unwrap();

        mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        mgr.ingest(make_event("k1", 1500, 2.0)).unwrap();
        assert_eq!(mgr.open_window_count(), 2);

        mgr.fire_eligible();
        assert_eq!(mgr.open_window_count(), 1);
    }

    #[test]
    fn test_get_window() {
        let cfg = WindowConfig {
            kind: WindowKind::Tumbling,
            size_ms: 1000,
            slide_ms: None,
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        let mut mgr = WindowManager::new(cfg, 100).unwrap();

        let ids = mgr.ingest(make_event("k1", 100, 1.0)).unwrap();
        let w = mgr.get_window(&ids[0]).unwrap();
        assert_eq!(w.key, "k1");
        assert!(mgr.get_window("bad").is_none());
    }

    #[test]
    fn test_window_contains() {
        let w = Window {
            id: "w-0".into(),
            key: "k".into(),
            start_ms: 100,
            end_ms: 200,
            state: WindowState::Open,
            events: Vec::new(),
            created_at: Utc::now(),
        };
        assert!(w.contains(100));
        assert!(w.contains(150));
        assert!(!w.contains(200)); // Exclusive end.
        assert!(!w.contains(99));
    }

    #[test]
    fn test_error_display() {
        let e = WindowError::TooLate {
            event_time: 100,
            watermark: 500,
        };
        assert!(e.to_string().contains("100"));
        assert!(e.to_string().contains("500"));
    }

    #[test]
    fn test_sliding_zero_slide() {
        let cfg = WindowConfig {
            kind: WindowKind::Sliding,
            size_ms: 1000,
            slide_ms: Some(0),
            session_gap_ms: None,
            allowed_lateness_ms: 0,
            trigger: TriggerKind::OnWatermark,
        };
        assert!(WindowManager::new(cfg, 0).is_err());
    }
}
