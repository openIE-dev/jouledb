//! FAST Channel Scheduler — Free Ad-Supported Streaming TV.
//!
//! $9B market by 2026. Uses TrendingIndex for content selection and
//! AdTargetingEngine for ad break optimization.
//! Auto-schedules 24/7 linear channels from a content library.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::trending::{TrendWindow, TrendingIndex};
use crate::ad_targeting::AdTargetingEngine;
use crate::RecordId;

/// Type of content in a schedule slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotType {
    /// Regular content
    Content,
    /// Ad break
    AdBreak,
    /// Bumper/interstitial
    Bumper,
    /// Live event
    Live,
}

/// A single slot in a FAST channel schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleSlot {
    /// Content record ID (or ad creative ID)
    pub content_id: RecordId,
    /// Content title
    pub title: String,
    /// Slot type
    pub slot_type: SlotType,
    /// Start time (unix ms)
    pub start_ms: u64,
    /// Duration (ms)
    pub duration_ms: u64,
    /// Category/genre
    pub category: Option<String>,
}

/// Strategy for filling schedule slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScheduleStrategy {
    /// Fill with trending content (maximize engagement)
    Trending,
    /// Round-robin through categories
    CategoryRotation,
    /// Maximize ad revenue (schedule content that drives highest CPM)
    RevenueOptimized,
    /// Custom manual schedule
    Manual,
}

/// FAST channel scheduler.
pub struct FastScheduler {
    /// Channel identifier
    pub channel_id: String,
    /// Channel name
    pub channel_name: String,
    /// Scheduling strategy
    pub strategy: ScheduleStrategy,
    /// Target ad break interval (ms)
    pub ad_interval_ms: u64,
    /// Target ad break duration (ms)
    pub ad_duration_ms: u64,
    /// Generated schedule
    schedule: Vec<ScheduleSlot>,
}

impl FastScheduler {
    pub fn new(channel_id: &str, name: &str, strategy: ScheduleStrategy) -> Self {
        Self {
            channel_id: channel_id.to_string(),
            channel_name: name.to_string(),
            strategy,
            ad_interval_ms: 15 * 60 * 1000,  // 15 min between ad breaks
            ad_duration_ms: 2 * 60 * 1000,    // 2 min ad breaks
            schedule: Vec::new(),
        }
    }

    /// Generate a schedule for the given time window.
    ///
    /// Uses the trending index to select content and inserts ad breaks
    /// at regular intervals.
    pub fn generate_schedule(
        &mut self,
        start_ms: u64,
        duration_ms: u64,
        content_catalog: &[(RecordId, String, u64, Option<String>)], // (id, title, duration_ms, category)
        trending: Option<&TrendingIndex>,
    ) -> &[ScheduleSlot] {
        self.schedule.clear();

        let mut current_ms = start_ms;
        let end_ms = start_ms + duration_ms;
        let mut last_ad_ms = start_ms;
        let mut content_idx = 0;

        // Sort content by trending score if available
        let mut catalog: Vec<(RecordId, String, u64, Option<String>, f64)> = content_catalog
            .iter()
            .map(|(id, title, dur, cat)| {
                let trend_score = trending
                    .map(|t| {
                        t.query_trending(1000, TrendWindow::OneHour, cat.as_deref())
                            .iter()
                            .find(|item| item.record_id == *id)
                            .map(|item| item.score)
                            .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0);
                (*id, title.clone(), *dur, cat.clone(), trend_score)
            })
            .collect();

        // Sort by trending score descending
        catalog.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

        while current_ms < end_ms {
            // Check if it's time for an ad break
            if current_ms - last_ad_ms >= self.ad_interval_ms {
                let ad_dur = self.ad_duration_ms.min(end_ms - current_ms);
                self.schedule.push(ScheduleSlot {
                    content_id: 0, // Ad creative selected at playout time
                    title: "Ad Break".into(),
                    slot_type: SlotType::AdBreak,
                    start_ms: current_ms,
                    duration_ms: ad_dur,
                    category: None,
                });
                current_ms += ad_dur;
                last_ad_ms = current_ms;
                continue;
            }

            // Select next content item
            if catalog.is_empty() {
                break;
            }

            let (id, title, dur, cat, _score) = &catalog[content_idx % catalog.len()];
            let slot_dur = (*dur).min(end_ms - current_ms);

            self.schedule.push(ScheduleSlot {
                content_id: *id,
                title: title.clone(),
                slot_type: SlotType::Content,
                start_ms: current_ms,
                duration_ms: slot_dur,
                category: cat.clone(),
            });

            current_ms += slot_dur;
            content_idx += 1;
        }

        &self.schedule
    }

    /// Get the current schedule.
    pub fn schedule(&self) -> &[ScheduleSlot] {
        &self.schedule
    }

    /// What's playing at a specific timestamp?
    pub fn at_time(&self, timestamp_ms: u64) -> Option<&ScheduleSlot> {
        self.schedule.iter().find(|slot| {
            timestamp_ms >= slot.start_ms && timestamp_ms < slot.start_ms + slot.duration_ms
        })
    }

    /// Total ad break duration in the schedule (ms).
    pub fn total_ad_time_ms(&self) -> u64 {
        self.schedule
            .iter()
            .filter(|s| s.slot_type == SlotType::AdBreak)
            .map(|s| s.duration_ms)
            .sum()
    }

    /// Total content duration (ms).
    pub fn total_content_time_ms(&self) -> u64 {
        self.schedule
            .iter()
            .filter(|s| s.slot_type == SlotType::Content)
            .map(|s| s.duration_ms)
            .sum()
    }

    /// Ad load percentage.
    pub fn ad_load_percent(&self) -> f64 {
        let total = self.total_ad_time_ms() + self.total_content_time_ms();
        if total == 0 {
            return 0.0;
        }
        self.total_ad_time_ms() as f64 / total as f64 * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_schedule() {
        let mut scheduler = FastScheduler::new("ch1", "Action Channel", ScheduleStrategy::Trending);
        scheduler.ad_interval_ms = 30_000; // 30s for testing
        scheduler.ad_duration_ms = 10_000; // 10s breaks

        let catalog = vec![
            (1, "Movie A".into(), 20_000u64, Some("action".to_string())),
            (2, "Movie B".into(), 20_000u64, Some("action".to_string())),
            (3, "Movie C".into(), 20_000u64, Some("action".to_string())),
        ];

        let schedule = scheduler.generate_schedule(0, 120_000, &catalog, None); // 2 minutes

        assert!(!schedule.is_empty());

        // Should have both content and ad breaks
        let has_content = schedule.iter().any(|s| s.slot_type == SlotType::Content);
        let has_ads = schedule.iter().any(|s| s.slot_type == SlotType::AdBreak);
        assert!(has_content);
        assert!(has_ads);
    }

    #[test]
    fn test_at_time() {
        let mut scheduler = FastScheduler::new("ch1", "Test", ScheduleStrategy::Manual);
        scheduler.ad_interval_ms = 1_000_000; // No ads for this test

        let catalog = vec![
            (1, "First".into(), 30_000u64, None),
            (2, "Second".into(), 30_000u64, None),
        ];

        scheduler.generate_schedule(0, 60_000, &catalog, None);

        let first = scheduler.at_time(15_000).unwrap();
        assert_eq!(first.title, "First");

        let second = scheduler.at_time(45_000).unwrap();
        assert_eq!(second.title, "Second");
    }

    #[test]
    fn test_ad_load() {
        let mut scheduler = FastScheduler::new("ch1", "Test", ScheduleStrategy::Trending);
        scheduler.ad_interval_ms = 50_000;
        scheduler.ad_duration_ms = 10_000;

        let catalog = vec![
            (1, "Content".into(), 40_000u64, None),
        ];

        scheduler.generate_schedule(0, 120_000, &catalog, None);

        let ad_load = scheduler.ad_load_percent();
        assert!(ad_load > 0.0 && ad_load < 50.0); // Reasonable ad load
    }
}
