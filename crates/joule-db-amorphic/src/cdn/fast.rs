//! FAST Channel Scheduler — Scheduled linear channels.
//!
//! Uses TrendingIndex for content selection.
//! Auto-schedules 24/7 linear channels from a content library.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::trending::{TrendWindow, TrendingIndex};
use crate::RecordId;

/// Type of content in a schedule slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotType {
    /// Regular content
    Content,
    /// Bumper/interstitial
    Bumper,
    /// Live event
    Live,
}

/// A single slot in a channel schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleSlot {
    /// Content record ID
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
    /// Generated schedule
    schedule: Vec<ScheduleSlot>,
}

impl FastScheduler {
    pub fn new(channel_id: &str, name: &str, strategy: ScheduleStrategy) -> Self {
        Self {
            channel_id: channel_id.to_string(),
            channel_name: name.to_string(),
            strategy,
            schedule: Vec::new(),
        }
    }

    /// Generate a schedule for the given time window.
    ///
    /// Uses the trending index to select content.
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

    /// Total content duration (ms).
    pub fn total_content_time_ms(&self) -> u64 {
        self.schedule
            .iter()
            .filter(|s| s.slot_type == SlotType::Content)
            .map(|s| s.duration_ms)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_schedule() {
        let mut scheduler = FastScheduler::new("ch1", "Action Channel", ScheduleStrategy::Trending);

        let catalog = vec![
            (1, "Movie A".into(), 20_000u64, Some("action".to_string())),
            (2, "Movie B".into(), 20_000u64, Some("action".to_string())),
            (3, "Movie C".into(), 20_000u64, Some("action".to_string())),
        ];

        let schedule = scheduler.generate_schedule(0, 120_000, &catalog, None); // 2 minutes

        assert!(!schedule.is_empty());

        let has_content = schedule.iter().any(|s| s.slot_type == SlotType::Content);
        assert!(has_content);
    }

    #[test]
    fn test_at_time() {
        let mut scheduler = FastScheduler::new("ch1", "Test", ScheduleStrategy::Manual);

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
}
