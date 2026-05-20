//! Trending/Hot-Content Index — time-decayed engagement scoring.
//!
//! Content platforms need "what's trending right now" with sub-second freshness.
//! This module provides:
//! - Sliding window counters (1min, 5min, 1hr, 24hr)
//! - Exponential time decay (older events worth less)
//! - Momentum scoring (rate of engagement change)
//! - Lock-free atomic counters for concurrent updates

use crate::RecordId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Time windows for trending calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrendWindow {
    OneMinute,
    FiveMinutes,
    OneHour,
    TwentyFourHours,
}

impl TrendWindow {
    pub fn duration(&self) -> Duration {
        match self {
            TrendWindow::OneMinute => Duration::from_secs(60),
            TrendWindow::FiveMinutes => Duration::from_secs(300),
            TrendWindow::OneHour => Duration::from_secs(3600),
            TrendWindow::TwentyFourHours => Duration::from_secs(86400),
        }
    }
}

/// Sliding window counter for a single content item.
pub struct ContentCounter {
    /// Event counts per window
    counts: [AtomicU64; 4], // 1min, 5min, 1hr, 24hr
    /// Last event timestamp (epoch ms)
    last_event_ms: AtomicU64,
    /// Associated record ID
    pub record_id: RecordId,
    /// Optional category for filtered queries
    pub category: Option<String>,
}

impl ContentCounter {
    pub fn new(record_id: RecordId, category: Option<String>) -> Self {
        Self {
            counts: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            last_event_ms: AtomicU64::new(0),
            record_id,
            category,
        }
    }

    /// Record an engagement event.
    pub fn record_event(&self, now_ms: u64) {
        for count in &self.counts {
            count.fetch_add(1, Ordering::Relaxed);
        }
        self.last_event_ms.store(now_ms, Ordering::Relaxed);
    }

    /// Get count for a specific window.
    pub fn count(&self, window: TrendWindow) -> u64 {
        self.counts[window as usize].load(Ordering::Relaxed)
    }

    /// Time-decayed score: count × exp(-λ × age_seconds).
    pub fn decayed_score(&self, window: TrendWindow, now_ms: u64, half_life_secs: f64) -> f64 {
        let count = self.count(window) as f64;
        if count == 0.0 {
            return 0.0;
        }
        let last = self.last_event_ms.load(Ordering::Relaxed);
        let age_secs = (now_ms.saturating_sub(last)) as f64 / 1000.0;
        let lambda = 0.693 / half_life_secs; // ln(2) / half_life
        count * (-lambda * age_secs).exp()
    }

    /// Momentum: ratio of short-term to long-term engagement rate.
    /// Values > 1.0 = accelerating (trending up), < 1.0 = decelerating.
    pub fn momentum(&self) -> f64 {
        let short = self.counts[0].load(Ordering::Relaxed) as f64; // 1min
        let long = self.counts[2].load(Ordering::Relaxed) as f64;  // 1hr

        if long == 0.0 {
            return if short > 0.0 { f64::INFINITY } else { 0.0 };
        }

        // Normalize by window duration: (count_1min / 1) / (count_1hr / 60)
        (short / 1.0) / (long / 60.0)
    }

    /// Reset window counters (called periodically by the decay sweep).
    pub fn decay_windows(&self, now_ms: u64) {
        let last = self.last_event_ms.load(Ordering::Relaxed);
        if last == 0 {
            return;
        }
        let age_ms = now_ms.saturating_sub(last);

        // Reset 1-minute counter if >1 minute old
        if age_ms > 60_000 {
            self.counts[0].store(0, Ordering::Relaxed);
        }
        if age_ms > 300_000 {
            self.counts[1].store(0, Ordering::Relaxed);
        }
        if age_ms > 3_600_000 {
            self.counts[2].store(0, Ordering::Relaxed);
        }
        if age_ms > 86_400_000 {
            self.counts[3].store(0, Ordering::Relaxed);
        }
    }
}

/// The trending index: tracks engagement velocity for all content.
pub struct TrendingIndex {
    /// Content counters indexed by content ID
    counters: HashMap<String, ContentCounter>,
    /// Decay half-life in seconds (default: 300 = 5 minutes)
    pub half_life_secs: f64,
    /// Creation time for relative timestamps
    created: Instant,
}

impl TrendingIndex {
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            half_life_secs: 300.0,
            created: Instant::now(),
        }
    }

    pub fn with_half_life(half_life_secs: f64) -> Self {
        Self {
            half_life_secs,
            ..Self::new()
        }
    }

    /// Register a content item for trending tracking.
    pub fn register(&mut self, content_id: &str, record_id: RecordId, category: Option<String>) {
        self.counters.insert(
            content_id.to_string(),
            ContentCounter::new(record_id, category),
        );
    }

    /// Record an engagement event for a content item.
    /// If the content is not yet registered, it will be auto-registered without a category.
    pub fn record_event(&mut self, content_id: &str, record_id: RecordId) {
        let now_ms = self.now_ms();
        if !self.counters.contains_key(content_id) {
            self.counters.insert(
                content_id.to_string(),
                ContentCounter::new(record_id, None),
            );
        }
        self.counters.get(content_id).unwrap().record_event(now_ms);
    }

    /// Query the top-k trending content items.
    pub fn query_trending(
        &self,
        k: usize,
        window: TrendWindow,
        category: Option<&str>,
    ) -> Vec<TrendingItem> {
        let now_ms = self.now_ms();

        let mut scored: Vec<TrendingItem> = self
            .counters
            .iter()
            .filter(|(_, counter)| {
                category
                    .map(|cat| {
                        counter
                            .category
                            .as_ref()
                            .map(|c| c == cat)
                            .unwrap_or(false)
                    })
                    .unwrap_or(true)
            })
            .map(|(content_id, counter)| TrendingItem {
                content_id: content_id.clone(),
                record_id: counter.record_id,
                score: counter.decayed_score(window, now_ms, self.half_life_secs),
                momentum: counter.momentum(),
                count: counter.count(window),
            })
            .filter(|item| item.score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Run decay sweep: reset expired window counters.
    pub fn sweep(&self) {
        let now_ms = self.now_ms();
        for counter in self.counters.values() {
            counter.decay_windows(now_ms);
        }
    }

    /// Number of tracked content items.
    pub fn tracked_count(&self) -> usize {
        self.counters.len()
    }

    fn now_ms(&self) -> u64 {
        self.created.elapsed().as_millis() as u64
    }
}

impl Default for TrendingIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// A trending content item with score and momentum.
#[derive(Debug, Clone)]
pub struct TrendingItem {
    pub content_id: String,
    pub record_id: RecordId,
    /// Time-decayed engagement score
    pub score: f64,
    /// Momentum: >1 = accelerating, <1 = decelerating
    pub momentum: f64,
    /// Raw event count in the queried window
    pub count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_query() {
        let mut index = TrendingIndex::new();

        // Record events for different content
        for _ in 0..10 {
            index.record_event("video_a", 1);
        }
        for _ in 0..5 {
            index.record_event("video_b", 2);
        }
        for _ in 0..1 {
            index.record_event("video_c", 3);
        }

        let trending = index.query_trending(10, TrendWindow::OneMinute, None);

        // video_a should be first (most events)
        assert!(!trending.is_empty());
        assert_eq!(trending[0].content_id, "video_a");
        assert_eq!(trending[0].count, 10);
        assert_eq!(trending[1].content_id, "video_b");
    }

    #[test]
    fn test_category_filter() {
        let mut index = TrendingIndex::new();

        index.register("music_1", 1, Some("music".into()));
        index.register("sports_1", 2, Some("sports".into()));
        index.register("music_2", 3, Some("music".into()));

        for _ in 0..5 {
            index.record_event("music_1", 1);
            index.record_event("sports_1", 2);
            index.record_event("music_2", 3);
        }

        let music = index.query_trending(10, TrendWindow::OneMinute, Some("music"));
        assert_eq!(music.len(), 2);

        let sports = index.query_trending(10, TrendWindow::OneMinute, Some("sports"));
        assert_eq!(sports.len(), 1);
    }

    #[test]
    fn test_momentum() {
        let counter = ContentCounter::new(1, None);

        // No events = no momentum
        assert_eq!(counter.momentum(), 0.0);

        // Add events to 1min window only
        counter.counts[0].store(10, Ordering::Relaxed);
        counter.counts[2].store(10, Ordering::Relaxed);

        // Even distribution: (10/1) / (10/60) = 60
        let m = counter.momentum();
        assert!((m - 60.0).abs() < 0.01);
    }
}
