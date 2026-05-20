//! Content Event Schema — typed engagement events for real-time personalization.
//!
//! Recommendation engines (TikTok, Netflix, YouTube) need typed user interactions
//! flowing into the feature store in real-time. Each event updates the user's
//! profile hologram via incremental HDC bundling.
//!
//! # Event Flow
//! ```text
//! User action → ContentEvent → EventIngester → BundleAccumulator → Updated user BinaryHV
//! ```

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{AmorphicError, AmorphicResult, DIMENSION};

/// Typed content engagement events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentEvent {
    /// User watched content for a duration
    Watch {
        user_id: String,
        content_id: String,
        duration_ms: u64,
        completion_pct: f32,
    },
    /// User liked/hearted content
    Like {
        user_id: String,
        content_id: String,
    },
    /// User shared content
    Share {
        user_id: String,
        content_id: String,
        platform: String,
    },
    /// User skipped/scrolled past content
    Skip {
        user_id: String,
        content_id: String,
        after_ms: u64,
    },
    /// User searched for content
    Search {
        user_id: String,
        query: String,
    },
    /// Content was shown to user (for CTR calculation)
    Impression {
        user_id: String,
        content_id: String,
        position: u32,
    },
    /// User scrolled in feed
    Scroll {
        user_id: String,
        velocity: f32,
    },
}

impl ContentEvent {
    /// Extract the user ID from any event type.
    pub fn user_id(&self) -> &str {
        match self {
            ContentEvent::Watch { user_id, .. }
            | ContentEvent::Like { user_id, .. }
            | ContentEvent::Share { user_id, .. }
            | ContentEvent::Skip { user_id, .. }
            | ContentEvent::Search { user_id, .. }
            | ContentEvent::Impression { user_id, .. }
            | ContentEvent::Scroll { user_id, .. } => user_id,
        }
    }

    /// Extract the content ID if this event references specific content.
    pub fn content_id(&self) -> Option<&str> {
        match self {
            ContentEvent::Watch { content_id, .. }
            | ContentEvent::Like { content_id, .. }
            | ContentEvent::Share { content_id, .. }
            | ContentEvent::Skip { content_id, .. }
            | ContentEvent::Impression { content_id, .. } => Some(content_id),
            ContentEvent::Search { .. } | ContentEvent::Scroll { .. } => None,
        }
    }

    /// Engagement weight: how strong a signal is this event?
    /// Higher = stronger positive signal for recommendation.
    pub fn engagement_weight(&self) -> f32 {
        match self {
            ContentEvent::Watch { completion_pct, .. } => {
                // Completion percentage is the strongest signal
                *completion_pct * 3.0
            }
            ContentEvent::Like { .. } => 2.0,
            ContentEvent::Share { .. } => 2.5,      // Sharing is high intent
            ContentEvent::Skip { after_ms, .. } => {
                // Quick skip = negative signal
                if *after_ms < 2000 { -1.0 } else { -0.5 }
            }
            ContentEvent::Search { .. } => 1.0,
            ContentEvent::Impression { .. } => 0.1,  // Weak positive (saw it)
            ContentEvent::Scroll { velocity, .. } => {
                // Fast scroll = disengaged
                if *velocity > 5.0 { -0.3 } else { 0.0 }
            }
        }
    }
}

/// User profile built from engagement events via incremental HDC bundling.
pub struct UserProfile {
    /// The user's holographic profile (bundled content interactions)
    pub hologram: BinaryHV,
    /// Accumulator for incremental updates
    accumulator: BundleAccumulator,
    /// Number of events processed
    pub event_count: u64,
    /// Engagement counters by content ID
    pub content_engagement: HashMap<String, f32>,
}

impl UserProfile {
    pub fn new() -> Self {
        Self {
            hologram: BinaryHV::zeros(DIMENSION),
            accumulator: BundleAccumulator::new(DIMENSION),
            event_count: 0,
            content_engagement: HashMap::new(),
        }
    }

    /// Update the profile with a new engagement event.
    /// The content's hologram is weighted by engagement strength and bundled in.
    pub fn update(&mut self, content_hologram: &BinaryHV, weight: f32) {
        if weight > 0.0 {
            // Positive engagement: bundle the content hologram into the profile
            let repeats = (weight.abs() as usize).max(1).min(5);
            for _ in 0..repeats {
                self.accumulator.add(content_hologram);
            }
        }
        // Negative engagement: we don't unbundle (it would add noise)
        // Instead, the absence of positive signal naturally reduces similarity
        self.event_count += 1;
        self.hologram = self.accumulator.threshold();
    }
}

impl Default for UserProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages user profiles and processes engagement events.
pub struct EventProcessor {
    /// User profiles indexed by user_id
    profiles: HashMap<String, UserProfile>,
    /// Total events processed
    total_events: AtomicU64,
}

impl EventProcessor {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            total_events: AtomicU64::new(0),
        }
    }

    /// Process an engagement event.
    /// Returns the updated user profile hologram for immediate use in recommendations.
    pub fn process_event(
        &mut self,
        event: &ContentEvent,
        content_hologram: Option<&BinaryHV>,
    ) -> AmorphicResult<&BinaryHV> {
        let user_id = event.user_id().to_string();
        let weight = event.engagement_weight();

        let profile = self
            .profiles
            .entry(user_id.clone())
            .or_insert_with(UserProfile::new);

        // Update content engagement score
        if let Some(cid) = event.content_id() {
            *profile.content_engagement.entry(cid.to_string()).or_default() += weight;
        }

        // Update holographic profile if we have the content's hologram
        if let Some(hologram) = content_hologram {
            profile.update(hologram, weight);
        }

        self.total_events.fetch_add(1, Ordering::Relaxed);

        Ok(&self.profiles[&user_id].hologram)
    }

    /// Get a user's profile hologram (for recommendation queries).
    pub fn user_hologram(&self, user_id: &str) -> Option<&BinaryHV> {
        self.profiles.get(user_id).map(|p| &p.hologram)
    }

    /// Get engagement stats for a user.
    pub fn user_stats(&self, user_id: &str) -> Option<UserStats> {
        self.profiles.get(user_id).map(|p| UserStats {
            event_count: p.event_count,
            content_interactions: p.content_engagement.len(),
            top_content: p
                .content_engagement
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, v)| (k.clone(), *v)),
        })
    }

    /// Total events processed across all users.
    pub fn total_events(&self) -> u64 {
        self.total_events.load(Ordering::Relaxed)
    }
}

impl Default for EventProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary stats for a user.
#[derive(Debug, Clone)]
pub struct UserStats {
    pub event_count: u64,
    pub content_interactions: usize,
    pub top_content: Option<(String, f32)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_processing() {
        let mut processor = EventProcessor::new();

        // Create a content hologram
        let content_hv = BinaryHV::from_hash(b"movie:inception", DIMENSION);

        // User watches 80% of the movie
        let event = ContentEvent::Watch {
            user_id: "user:1".into(),
            content_id: "movie:inception".into(),
            duration_ms: 7200000,
            completion_pct: 0.8,
        };

        let profile = processor.process_event(&event, Some(&content_hv)).unwrap();
        assert!(profile.similarity(&content_hv) > 0.5);
    }

    #[test]
    fn test_engagement_weights() {
        let watch_full = ContentEvent::Watch {
            user_id: "u".into(),
            content_id: "c".into(),
            duration_ms: 1000,
            completion_pct: 1.0,
        };
        assert!(watch_full.engagement_weight() > 2.0);

        let skip = ContentEvent::Skip {
            user_id: "u".into(),
            content_id: "c".into(),
            after_ms: 500,
        };
        assert!(skip.engagement_weight() < 0.0);
    }

    #[test]
    fn test_user_profile_similarity() {
        let mut processor = EventProcessor::new();

        let action_hv = BinaryHV::from_hash(b"genre:action", DIMENSION);
        let comedy_hv = BinaryHV::from_hash(b"genre:comedy", DIMENSION);

        // User watches 3 action movies
        for i in 0..3 {
            processor
                .process_event(
                    &ContentEvent::Watch {
                        user_id: "user:1".into(),
                        content_id: format!("action_{}", i),
                        duration_ms: 7200000,
                        completion_pct: 0.9,
                    },
                    Some(&action_hv),
                )
                .unwrap();
        }

        let profile = processor.user_hologram("user:1").unwrap();

        // Profile should be more similar to action than comedy
        let action_sim = profile.similarity(&action_hv);
        let comedy_sim = profile.similarity(&comedy_hv);
        assert!(
            action_sim > comedy_sim,
            "Action sim ({}) should be > comedy sim ({})",
            action_sim,
            comedy_sim
        );
    }
}
