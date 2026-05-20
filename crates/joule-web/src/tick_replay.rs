//! Tick replay — event-driven replay engine, time-scaling (speed up/slow
//! down), bookmark/resume, filtered replay (by symbol/type), replay
//! statistics, ReplayConfig builder.
//!
//! Pure-Rust tick replay infrastructure for backtesting and analysis:
//!
//! - [`TickEvent`] — generic timestamped event (quote or trade)
//! - [`ReplayEngine`] — event-driven replay with time scaling
//! - [`Bookmark`] — checkpoint for pause/resume
//! - [`ReplayFilter`] — filter events by symbol, type, or time range
//! - [`ReplayStats`] — throughput and latency statistics
//! - [`ReplayConfig`] — builder for replay parameters

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ReplayError {
    EmptyFeed(String),
    InvalidSpeed(String),
    InvalidBookmark(String),
    AlreadyRunning,
    NotStarted,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFeed(s) => write!(f, "empty feed: {s}"),
            Self::InvalidSpeed(s) => write!(f, "invalid speed: {s}"),
            Self::InvalidBookmark(s) => write!(f, "invalid bookmark: {s}"),
            Self::AlreadyRunning => write!(f, "replay already running"),
            Self::NotStarted => write!(f, "replay not started"),
        }
    }
}

impl std::error::Error for ReplayError {}

// ── Tick Event Types ────────────────────────────────────────────

/// Classification of tick events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    Quote,
    Trade,
    Level2Update,
    Auction,
    Halt,
    Resume,
}

impl fmt::Display for EventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quote => write!(f, "Quote"),
            Self::Trade => write!(f, "Trade"),
            Self::Level2Update => write!(f, "L2Update"),
            Self::Auction => write!(f, "Auction"),
            Self::Halt => write!(f, "Halt"),
            Self::Resume => write!(f, "Resume"),
        }
    }
}

// ── Tick Event ──────────────────────────────────────────────────

/// A single timestamped market event for replay.
#[derive(Debug, Clone, PartialEq)]
pub struct TickEvent {
    pub timestamp_us: u64,
    pub symbol: String,
    pub event_type: EventType,
    pub price: f64,
    pub size: f64,
    pub sequence: u64,
    pub venue: String,
    /// Extra key-value data for extensibility.
    pub metadata: HashMap<String, String>,
}

impl TickEvent {
    pub fn new(timestamp_us: u64, symbol: &str, event_type: EventType,
               price: f64, size: f64, sequence: u64, venue: &str) -> Self {
        Self {
            timestamp_us,
            symbol: symbol.to_string(),
            event_type,
            price, size, sequence,
            venue: venue.to_string(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn notional(&self) -> f64 { self.price * self.size }
}

impl fmt::Display for TickEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{} {} {:.4}x{:.0} seq={} t={}",
               self.symbol, self.venue, self.event_type,
               self.price, self.size, self.sequence, self.timestamp_us)
    }
}

// ── Replay Config ───────────────────────────────────────────────

/// Builder for replay configuration.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    pub speed_factor: f64,
    pub start_us: Option<u64>,
    pub end_us: Option<u64>,
    pub symbol_filter: Vec<String>,
    pub event_type_filter: Vec<EventType>,
    pub loop_replay: bool,
    pub skip_gaps_above_us: Option<u64>,
}

impl ReplayConfig {
    pub fn new() -> Self {
        Self {
            speed_factor: 1.0,
            start_us: None,
            end_us: None,
            symbol_filter: Vec::new(),
            event_type_filter: Vec::new(),
            loop_replay: false,
            skip_gaps_above_us: None,
        }
    }

    pub fn with_speed(mut self, factor: f64) -> Self {
        self.speed_factor = factor;
        self
    }

    pub fn with_start(mut self, start_us: u64) -> Self {
        self.start_us = Some(start_us);
        self
    }

    pub fn with_end(mut self, end_us: u64) -> Self {
        self.end_us = Some(end_us);
        self
    }

    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol_filter.push(symbol.to_string());
        self
    }

    pub fn with_event_type(mut self, et: EventType) -> Self {
        self.event_type_filter.push(et);
        self
    }

    pub fn with_loop(mut self, do_loop: bool) -> Self {
        self.loop_replay = do_loop;
        self
    }

    pub fn with_skip_gaps(mut self, threshold_us: u64) -> Self {
        self.skip_gaps_above_us = Some(threshold_us);
        self
    }

    pub fn validate(&self) -> Result<(), ReplayError> {
        if self.speed_factor <= 0.0 {
            return Err(ReplayError::InvalidSpeed(
                format!("speed must be positive, got {}", self.speed_factor)));
        }
        Ok(())
    }
}

impl fmt::Display for ReplayConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReplayConfig[speed={:.1}x loop={} syms={} types={}]",
               self.speed_factor, self.loop_replay,
               self.symbol_filter.len(), self.event_type_filter.len())
    }
}

// ── Replay Filter ───────────────────────────────────────────────

/// Filters events during replay.
pub struct ReplayFilter {
    symbols: Vec<String>,
    event_types: Vec<EventType>,
    min_price: Option<f64>,
    max_price: Option<f64>,
    min_size: Option<f64>,
}

impl ReplayFilter {
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            event_types: Vec::new(),
            min_price: None,
            max_price: None,
            min_size: None,
        }
    }

    pub fn with_symbol(mut self, sym: &str) -> Self {
        self.symbols.push(sym.to_string());
        self
    }

    pub fn with_event_type(mut self, et: EventType) -> Self {
        self.event_types.push(et);
        self
    }

    pub fn with_min_price(mut self, p: f64) -> Self { self.min_price = Some(p); self }

    pub fn with_max_price(mut self, p: f64) -> Self { self.max_price = Some(p); self }

    pub fn with_min_size(mut self, s: f64) -> Self { self.min_size = Some(s); self }

    /// Returns true if the event passes all filters.
    pub fn matches(&self, event: &TickEvent) -> bool {
        if !self.symbols.is_empty() && !self.symbols.contains(&event.symbol) {
            return false;
        }
        if !self.event_types.is_empty() && !self.event_types.contains(&event.event_type) {
            return false;
        }
        if let Some(min) = self.min_price {
            if event.price < min { return false; }
        }
        if let Some(max) = self.max_price {
            if event.price > max { return false; }
        }
        if let Some(min_s) = self.min_size {
            if event.size < min_s { return false; }
        }
        true
    }

    pub fn filter_events<'a>(&self, events: &'a [TickEvent]) -> Vec<&'a TickEvent> {
        events.iter().filter(|e| self.matches(e)).collect()
    }
}

impl fmt::Display for ReplayFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReplayFilter[syms={} types={}]",
               self.symbols.len(), self.event_types.len())
    }
}

// ── Bookmark ────────────────────────────────────────────────────

/// Checkpoint for pause/resume of replay.
#[derive(Debug, Clone, PartialEq)]
pub struct Bookmark {
    pub event_index: usize,
    pub timestamp_us: u64,
    pub events_replayed: u64,
    pub label: String,
}

impl Bookmark {
    pub fn new(event_index: usize, timestamp_us: u64, events_replayed: u64, label: &str) -> Self {
        Self { event_index, timestamp_us, events_replayed, label: label.to_string() }
    }
}

impl fmt::Display for Bookmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bookmark[\"{}\" idx={} ts={} replayed={}]",
               self.label, self.event_index, self.timestamp_us, self.events_replayed)
    }
}

// ── Replay Statistics ───────────────────────────────────────────

/// Aggregate statistics for a replay session.
#[derive(Debug, Clone)]
pub struct ReplayStats {
    pub events_total: u64,
    pub events_replayed: u64,
    pub events_filtered: u64,
    pub symbols_seen: usize,
    pub first_timestamp_us: u64,
    pub last_timestamp_us: u64,
    pub wall_clock_us: u64,
    per_symbol: HashMap<String, u64>,
    per_type: HashMap<EventType, u64>,
}

impl ReplayStats {
    pub fn new() -> Self {
        Self {
            events_total: 0,
            events_replayed: 0,
            events_filtered: 0,
            symbols_seen: 0,
            first_timestamp_us: u64::MAX,
            last_timestamp_us: 0,
            wall_clock_us: 0,
            per_symbol: HashMap::new(),
            per_type: HashMap::new(),
        }
    }

    pub fn record_event(&mut self, event: &TickEvent) {
        self.events_replayed += 1;
        if event.timestamp_us < self.first_timestamp_us {
            self.first_timestamp_us = event.timestamp_us;
        }
        if event.timestamp_us > self.last_timestamp_us {
            self.last_timestamp_us = event.timestamp_us;
        }
        *self.per_symbol.entry(event.symbol.clone()).or_insert(0) += 1;
        *self.per_type.entry(event.event_type).or_insert(0) += 1;
        self.symbols_seen = self.per_symbol.len();
    }

    pub fn record_filtered(&mut self) {
        self.events_filtered += 1;
    }

    pub fn data_span_us(&self) -> u64 {
        if self.first_timestamp_us == u64::MAX { return 0; }
        self.last_timestamp_us.saturating_sub(self.first_timestamp_us)
    }

    pub fn events_per_second(&self) -> f64 {
        if self.wall_clock_us == 0 { return 0.0; }
        (self.events_replayed as f64) / (self.wall_clock_us as f64 / 1_000_000.0)
    }

    pub fn symbol_count(&self, symbol: &str) -> u64 {
        self.per_symbol.get(symbol).copied().unwrap_or(0)
    }

    pub fn type_count(&self, et: EventType) -> u64 {
        self.per_type.get(&et).copied().unwrap_or(0)
    }
}

impl fmt::Display for ReplayStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReplayStats[replayed={} filtered={} syms={} span={}us]",
               self.events_replayed, self.events_filtered,
               self.symbols_seen, self.data_span_us())
    }
}

// ── Replay State ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayState {
    Idle,
    Running,
    Paused,
    Completed,
}

impl fmt::Display for ReplayState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Running => write!(f, "Running"),
            Self::Paused => write!(f, "Paused"),
            Self::Completed => write!(f, "Completed"),
        }
    }
}

// ── Replay Engine ───────────────────────────────────────────────

/// Event-driven tick replay engine.
pub struct ReplayEngine {
    events: Vec<TickEvent>,
    config: ReplayConfig,
    filter: Option<ReplayFilter>,
    state: ReplayState,
    cursor: usize,
    stats: ReplayStats,
    bookmarks: Vec<Bookmark>,
    loop_count: u64,
}

impl ReplayEngine {
    pub fn new(events: Vec<TickEvent>, config: ReplayConfig) -> Result<Self, ReplayError> {
        config.validate()?;
        let total = events.len() as u64;
        Ok(Self {
            events,
            config,
            filter: None,
            state: ReplayState::Idle,
            cursor: 0,
            stats: ReplayStats { events_total: total, ..ReplayStats::new() },
            bookmarks: Vec::new(),
            loop_count: 0,
        })
    }

    pub fn with_filter(mut self, filter: ReplayFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn state(&self) -> ReplayState { self.state }
    pub fn stats(&self) -> &ReplayStats { &self.stats }
    pub fn cursor(&self) -> usize { self.cursor }
    pub fn loop_count(&self) -> u64 { self.loop_count }

    pub fn start(&mut self) -> Result<(), ReplayError> {
        if self.events.is_empty() {
            return Err(ReplayError::EmptyFeed("no events loaded".into()));
        }
        if self.state == ReplayState::Running {
            return Err(ReplayError::AlreadyRunning);
        }
        self.state = ReplayState::Running;
        // Apply start time filter
        if let Some(start) = self.config.start_us {
            while self.cursor < self.events.len()
                && self.events[self.cursor].timestamp_us < start
            {
                self.cursor += 1;
            }
        }
        Ok(())
    }

    /// Advance replay by one event. Returns the event or None if done.
    pub fn next_event(&mut self) -> Option<&TickEvent> {
        if self.state != ReplayState::Running {
            return None;
        }

        loop {
            if self.cursor >= self.events.len() {
                if self.config.loop_replay {
                    self.cursor = 0;
                    self.loop_count += 1;
                } else {
                    self.state = ReplayState::Completed;
                    return None;
                }
            }

            // Check end time
            if let Some(end) = self.config.end_us {
                if self.events[self.cursor].timestamp_us > end {
                    self.state = ReplayState::Completed;
                    return None;
                }
            }

            let event = &self.events[self.cursor];
            self.cursor += 1;

            // Apply filter
            if let Some(ref filt) = self.filter {
                if !filt.matches(event) {
                    self.stats.record_filtered();
                    continue;
                }
            }

            self.stats.record_event(event);
            return Some(event);
        }
    }

    /// Replay all remaining events, returning count processed.
    pub fn replay_all(&mut self) -> Result<u64, ReplayError> {
        if self.state == ReplayState::Idle {
            self.start()?;
        }
        let mut count = 0u64;
        while self.next_event().is_some() {
            count += 1;
        }
        Ok(count)
    }

    pub fn pause(&mut self) -> Result<(), ReplayError> {
        if self.state != ReplayState::Running {
            return Err(ReplayError::NotStarted);
        }
        self.state = ReplayState::Paused;
        Ok(())
    }

    pub fn resume(&mut self) -> Result<(), ReplayError> {
        if self.state != ReplayState::Paused {
            return Err(ReplayError::NotStarted);
        }
        self.state = ReplayState::Running;
        Ok(())
    }

    pub fn bookmark(&mut self, label: &str) -> Bookmark {
        let ts = if self.cursor > 0 && self.cursor <= self.events.len() {
            self.events[self.cursor - 1].timestamp_us
        } else {
            0
        };
        let bm = Bookmark::new(self.cursor, ts, self.stats.events_replayed, label);
        self.bookmarks.push(bm.clone());
        bm
    }

    pub fn restore_bookmark(&mut self, bookmark: &Bookmark) -> Result<(), ReplayError> {
        if bookmark.event_index > self.events.len() {
            return Err(ReplayError::InvalidBookmark(
                format!("index {} beyond event count {}", bookmark.event_index, self.events.len())));
        }
        self.cursor = bookmark.event_index;
        self.state = ReplayState::Paused;
        Ok(())
    }

    pub fn bookmarks(&self) -> &[Bookmark] { &self.bookmarks }

    /// Compute scaled inter-event delay in microseconds.
    pub fn scaled_delay_us(&self, event_a: &TickEvent, event_b: &TickEvent) -> u64 {
        let raw = event_b.timestamp_us.saturating_sub(event_a.timestamp_us);
        let scaled = raw as f64 / self.config.speed_factor;

        if let Some(max_gap) = self.config.skip_gaps_above_us {
            if raw > max_gap {
                return (max_gap as f64 / self.config.speed_factor) as u64;
            }
        }
        scaled as u64
    }
}

impl fmt::Display for ReplayEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReplayEngine[state={} cursor={}/{} loops={}]",
               self.state, self.cursor, self.events.len(), self.loop_count)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_events() -> Vec<TickEvent> {
        vec![
            TickEvent::new(1000, "AAPL", EventType::Quote, 150.0, 100.0, 1, "NYSE"),
            TickEvent::new(2000, "AAPL", EventType::Trade, 150.05, 50.0, 2, "NYSE"),
            TickEvent::new(3000, "GOOG", EventType::Quote, 2800.0, 10.0, 3, "ARCA"),
            TickEvent::new(4000, "AAPL", EventType::Trade, 150.10, 75.0, 4, "BATS"),
            TickEvent::new(5000, "GOOG", EventType::Trade, 2801.0, 5.0, 5, "ARCA"),
        ]
    }

    #[test]
    fn tick_event_display() {
        let e = TickEvent::new(1000, "AAPL", EventType::Trade, 150.0, 100.0, 1, "NYSE");
        let s = format!("{e}");
        assert!(s.contains("AAPL"));
        assert!(s.contains("Trade"));
    }

    #[test]
    fn tick_event_metadata() {
        let e = TickEvent::new(1000, "AAPL", EventType::Trade, 150.0, 100.0, 1, "NYSE")
            .with_metadata("condition", "F");
        assert_eq!(e.metadata.get("condition").unwrap(), "F");
    }

    #[test]
    fn config_builder() {
        let cfg = ReplayConfig::new()
            .with_speed(2.0)
            .with_start(1000)
            .with_end(5000)
            .with_symbol("AAPL")
            .with_loop(true);
        assert!((cfg.speed_factor - 2.0).abs() < 1e-9);
        assert_eq!(cfg.start_us, Some(1000));
        assert!(cfg.loop_replay);
    }

    #[test]
    fn config_invalid_speed() {
        let cfg = ReplayConfig::new().with_speed(-1.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn filter_by_symbol() {
        let filt = ReplayFilter::new().with_symbol("AAPL");
        let events = sample_events();
        let matched = filt.filter_events(&events);
        assert_eq!(matched.len(), 3);
    }

    #[test]
    fn filter_by_event_type() {
        let filt = ReplayFilter::new().with_event_type(EventType::Trade);
        let events = sample_events();
        let matched = filt.filter_events(&events);
        assert_eq!(matched.len(), 3);
    }

    #[test]
    fn filter_by_price_range() {
        let filt = ReplayFilter::new().with_min_price(200.0);
        let events = sample_events();
        let matched = filt.filter_events(&events);
        assert_eq!(matched.len(), 2); // GOOG only
    }

    #[test]
    fn filter_combined() {
        let filt = ReplayFilter::new()
            .with_symbol("AAPL")
            .with_event_type(EventType::Trade);
        let events = sample_events();
        let matched = filt.filter_events(&events);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn replay_all_events() {
        let events = sample_events();
        let cfg = ReplayConfig::new();
        let mut engine = ReplayEngine::new(events, cfg).unwrap();
        let count = engine.replay_all().unwrap();
        assert_eq!(count, 5);
        assert_eq!(engine.state(), ReplayState::Completed);
    }

    #[test]
    fn replay_with_filter() {
        let events = sample_events();
        let cfg = ReplayConfig::new();
        let filt = ReplayFilter::new().with_symbol("GOOG");
        let mut engine = ReplayEngine::new(events, cfg).unwrap().with_filter(filt);
        let count = engine.replay_all().unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn replay_with_time_range() {
        let events = sample_events();
        let cfg = ReplayConfig::new().with_start(2000).with_end(4000);
        let mut engine = ReplayEngine::new(events, cfg).unwrap();
        let count = engine.replay_all().unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn replay_pause_resume() {
        let events = sample_events();
        let cfg = ReplayConfig::new();
        let mut engine = ReplayEngine::new(events, cfg).unwrap();
        engine.start().unwrap();
        engine.next_event();
        engine.pause().unwrap();
        assert_eq!(engine.state(), ReplayState::Paused);
        engine.resume().unwrap();
        assert_eq!(engine.state(), ReplayState::Running);
    }

    #[test]
    fn replay_bookmark() {
        let events = sample_events();
        let cfg = ReplayConfig::new();
        let mut engine = ReplayEngine::new(events, cfg).unwrap();
        engine.start().unwrap();
        engine.next_event();
        engine.next_event();
        let bm = engine.bookmark("mid");
        assert_eq!(bm.event_index, 2);
        engine.replay_all().unwrap();
        engine.restore_bookmark(&bm).unwrap();
        assert_eq!(engine.cursor(), 2);
    }

    #[test]
    fn replay_loop() {
        let events = sample_events();
        let cfg = ReplayConfig::new().with_loop(true).with_end(5000);
        let mut engine = ReplayEngine::new(events, cfg).unwrap();
        engine.start().unwrap();
        // Consume through first loop
        for _ in 0..5 { engine.next_event(); }
        assert_eq!(engine.loop_count(), 0);
    }

    #[test]
    fn replay_scaled_delay() {
        let events = sample_events();
        let cfg = ReplayConfig::new().with_speed(2.0);
        let engine = ReplayEngine::new(events.clone(), cfg).unwrap();
        let delay = engine.scaled_delay_us(&events[0], &events[1]);
        assert_eq!(delay, 500);
    }

    #[test]
    fn replay_skip_gaps() {
        let mut events = sample_events();
        events[2].timestamp_us = 10_000_000; // huge gap
        let cfg = ReplayConfig::new().with_skip_gaps(1_000_000);
        let engine = ReplayEngine::new(events.clone(), cfg).unwrap();
        let delay = engine.scaled_delay_us(&events[1], &events[2]);
        assert_eq!(delay, 1_000_000);
    }

    #[test]
    fn replay_stats_tracking() {
        let events = sample_events();
        let cfg = ReplayConfig::new();
        let mut engine = ReplayEngine::new(events, cfg).unwrap();
        engine.replay_all().unwrap();
        let stats = engine.stats();
        assert_eq!(stats.events_replayed, 5);
        assert_eq!(stats.symbols_seen, 2);
        assert_eq!(stats.symbol_count("AAPL"), 3);
        assert_eq!(stats.type_count(EventType::Trade), 3);
    }

    #[test]
    fn replay_empty_feed() {
        let cfg = ReplayConfig::new();
        let mut engine = ReplayEngine::new(Vec::new(), cfg).unwrap();
        assert!(engine.start().is_err());
    }

    #[test]
    fn replay_display() {
        let events = sample_events();
        let cfg = ReplayConfig::new();
        let engine = ReplayEngine::new(events, cfg).unwrap();
        assert!(format!("{engine}").contains("Idle"));
    }
}
