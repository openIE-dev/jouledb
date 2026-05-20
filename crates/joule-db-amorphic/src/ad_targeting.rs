//! Ad Tech Targeting Engine — programmatic ad decisioning via HDC.
//!
//! $200B+ digital ad market. Every streaming platform with an ad tier needs:
//! - Real-time bid decisioning (sub-10ms)
//! - Brand safety matching (ad ↔ content)
//! - Frequency capping (time-windowed per-user counters)
//! - Audience segment matching (user profile → targeting criteria)
//!
//! Novel approach: encode targeting criteria as BinaryHV holograms,
//! match against user holograms via holographic similarity.

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{RecordId, DIMENSION};

/// An ad campaign with targeting criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdCampaign {
    pub campaign_id: String,
    pub advertiser: String,
    pub creative_id: String,
    /// Budget remaining (microcents)
    pub budget_remaining: u64,
    /// Maximum bid (microcents per impression)
    pub max_bid: u64,
    /// Targeting criteria
    pub targeting: TargetingCriteria,
    /// Frequency cap
    pub frequency_cap: FrequencyCap,
}

/// Targeting criteria for ad placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetingCriteria {
    /// Audience segments (e.g., "sports_fans", "tech_enthusiasts")
    pub segments: Vec<String>,
    /// Content categories where this ad is allowed
    pub allowed_categories: Vec<String>,
    /// Content categories where this ad is NOT allowed (brand safety)
    pub blocked_categories: Vec<String>,
    /// Territory restrictions
    pub territories: Vec<String>,
    /// Time-of-day targeting (hours 0-23, empty = all hours)
    pub hours: Vec<u8>,
}

/// Frequency cap: limit ad exposure per user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyCap {
    /// Maximum impressions per user per window
    pub max_impressions: u32,
    /// Window duration in seconds
    pub window_secs: u64,
}

/// Result of an ad decision: which ad to show.
#[derive(Debug, Clone)]
pub struct AdDecision {
    pub campaign_id: String,
    pub creative_id: String,
    pub bid: u64,
    pub relevance_score: f32,
    pub brand_safety_score: f32,
}

/// Per-user frequency tracking.
struct UserFrequency {
    /// campaign_id → (impression_count, window_start_ms)
    impressions: HashMap<String, (u32, u64)>,
}

impl UserFrequency {
    fn new() -> Self {
        Self {
            impressions: HashMap::new(),
        }
    }

    fn record_impression(&mut self, campaign_id: &str, now_ms: u64) {
        let entry = self
            .impressions
            .entry(campaign_id.to_string())
            .or_insert((0, now_ms));
        entry.0 += 1;
    }

    fn can_show(&self, campaign_id: &str, cap: &FrequencyCap, now_ms: u64) -> bool {
        match self.impressions.get(campaign_id) {
            None => true,
            Some((count, window_start)) => {
                let window_ms = cap.window_secs * 1000;
                if now_ms - window_start > window_ms {
                    true // Window expired, reset
                } else {
                    *count < cap.max_impressions
                }
            }
        }
    }
}

/// The ad targeting engine.
pub struct AdTargetingEngine {
    /// Registered campaigns
    campaigns: Vec<AdCampaign>,
    /// Campaign targeting encoded as holograms (for similarity matching)
    campaign_holograms: Vec<(String, BinaryHV)>,
    /// Per-user frequency tracking
    user_frequencies: HashMap<String, UserFrequency>,
    /// Total impressions served
    pub total_impressions: AtomicU64,
}

impl AdTargetingEngine {
    pub fn new() -> Self {
        Self {
            campaigns: Vec::new(),
            campaign_holograms: Vec::new(),
            user_frequencies: HashMap::new(),
            total_impressions: AtomicU64::new(0),
        }
    }

    /// Register an ad campaign.
    pub fn register_campaign(&mut self, campaign: AdCampaign) {
        // Encode targeting criteria as a hologram
        let hologram = encode_targeting(&campaign.targeting);
        self.campaign_holograms
            .push((campaign.campaign_id.clone(), hologram));
        self.campaigns.push(campaign);
    }

    /// Decide which ad to show for a given user in a given content context.
    ///
    /// This is the core real-time decisioning function. Must be sub-10ms.
    pub fn decide(
        &mut self,
        user_hologram: &BinaryHV,
        content_category: &str,
        territory: &str,
        hour: u8,
        user_id: &str,
        now_ms: u64,
    ) -> Option<AdDecision> {
        let content_hv = BinaryHV::from_hash(content_category.as_bytes(), DIMENSION);

        let mut candidates: Vec<AdDecision> = self
            .campaigns
            .iter()
            .zip(self.campaign_holograms.iter())
            .filter_map(|(campaign, (_, hologram))| {
                // Budget check
                if campaign.budget_remaining == 0 {
                    return None;
                }

                // Territory check
                if !campaign.targeting.territories.is_empty()
                    && !campaign.targeting.territories.iter().any(|t| t == territory)
                {
                    return None;
                }

                // Time-of-day check
                if !campaign.targeting.hours.is_empty()
                    && !campaign.targeting.hours.contains(&hour)
                {
                    return None;
                }

                // Brand safety: content must not be in blocked categories
                if campaign
                    .targeting
                    .blocked_categories
                    .iter()
                    .any(|c| c == content_category)
                {
                    return None;
                }

                // Content category check
                if !campaign.targeting.allowed_categories.is_empty()
                    && !campaign
                        .targeting
                        .allowed_categories
                        .iter()
                        .any(|c| c == content_category)
                {
                    return None;
                }

                // Frequency cap check
                let user_freq = self.user_frequencies.get(user_id);
                if let Some(freq) = user_freq {
                    if !freq.can_show(
                        &campaign.campaign_id,
                        &campaign.frequency_cap,
                        now_ms,
                    ) {
                        return None;
                    }
                }

                // Relevance: holographic similarity between user profile and targeting
                let relevance = user_hologram.similarity(hologram);

                // Brand safety score: similarity between ad targeting and content
                let safety = hologram.similarity(&content_hv);

                Some(AdDecision {
                    campaign_id: campaign.campaign_id.clone(),
                    creative_id: campaign.creative_id.clone(),
                    bid: campaign.max_bid,
                    relevance_score: relevance,
                    brand_safety_score: safety,
                })
            })
            .collect();

        // Sort by bid × relevance (second-price auction with relevance weighting)
        candidates.sort_by(|a, b| {
            let score_a = a.bid as f64 * a.relevance_score as f64;
            let score_b = b.bid as f64 * b.relevance_score as f64;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(winner) = candidates.first() {
            // Record impression
            self.user_frequencies
                .entry(user_id.to_string())
                .or_insert_with(UserFrequency::new)
                .record_impression(&winner.campaign_id, now_ms);
            self.total_impressions.fetch_add(1, Ordering::Relaxed);
            Some(winner.clone())
        } else {
            None
        }
    }

    /// Number of registered campaigns.
    pub fn campaign_count(&self) -> usize {
        self.campaigns.len()
    }
}

impl Default for AdTargetingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode targeting criteria as a BinaryHV for similarity matching.
fn encode_targeting(criteria: &TargetingCriteria) -> BinaryHV {
    let mut acc = BundleAccumulator::new(DIMENSION);

    for segment in &criteria.segments {
        let hv = BinaryHV::from_hash(format!("seg:{}", segment).as_bytes(), DIMENSION);
        acc.add(&hv);
    }
    for cat in &criteria.allowed_categories {
        let hv = BinaryHV::from_hash(format!("cat:{}", cat).as_bytes(), DIMENSION);
        acc.add(&hv);
    }
    for territory in &criteria.territories {
        let hv = BinaryHV::from_hash(format!("geo:{}", territory).as_bytes(), DIMENSION);
        acc.add(&hv);
    }

    acc.threshold()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_campaign(id: &str, segments: &[&str], categories: &[&str]) -> AdCampaign {
        AdCampaign {
            campaign_id: id.to_string(),
            advertiser: "Test Advertiser".to_string(),
            creative_id: format!("creative_{}", id),
            budget_remaining: 1_000_000,
            max_bid: 500,
            targeting: TargetingCriteria {
                segments: segments.iter().map(|s| s.to_string()).collect(),
                allowed_categories: categories.iter().map(|s| s.to_string()).collect(),
                blocked_categories: vec![],
                territories: vec!["US".to_string()],
                hours: vec![],
            },
            frequency_cap: FrequencyCap {
                max_impressions: 3,
                window_secs: 3600,
            },
        }
    }

    #[test]
    fn test_ad_decision() {
        let mut engine = AdTargetingEngine::new();

        engine.register_campaign(make_campaign("sports_ad", &["sports_fans"], &["sports"]));
        engine.register_campaign(make_campaign("tech_ad", &["tech_enthusiasts"], &["technology"]));

        // User who likes sports
        let user = BinaryHV::from_hash(b"seg:sports_fans", DIMENSION);

        let decision = engine.decide(&user, "sports", "US", 12, "user_1", 1000);
        assert!(decision.is_some());
        assert_eq!(decision.unwrap().campaign_id, "sports_ad");
    }

    #[test]
    fn test_brand_safety_blocking() {
        let mut engine = AdTargetingEngine::new();

        let mut campaign = make_campaign("family_ad", &["families"], &["family"]);
        campaign.targeting.blocked_categories = vec!["violence".to_string()];
        engine.register_campaign(campaign);

        let user = BinaryHV::from_hash(b"seg:families", DIMENSION);

        // Should not show on violent content
        let decision = engine.decide(&user, "violence", "US", 12, "user_1", 1000);
        assert!(decision.is_none());

        // Should show on family content
        let decision = engine.decide(&user, "family", "US", 12, "user_1", 1000);
        assert!(decision.is_some());
    }

    #[test]
    fn test_frequency_capping() {
        let mut engine = AdTargetingEngine::new();

        let mut campaign = make_campaign("capped_ad", &["all"], &["entertainment"]);
        campaign.frequency_cap = FrequencyCap {
            max_impressions: 2,
            window_secs: 3600,
        };
        engine.register_campaign(campaign);

        let user = BinaryHV::from_hash(b"seg:all", DIMENSION);

        // First two impressions: OK
        assert!(engine.decide(&user, "entertainment", "US", 12, "user_1", 1000).is_some());
        assert!(engine.decide(&user, "entertainment", "US", 12, "user_1", 2000).is_some());
        // Third: blocked by frequency cap
        assert!(engine.decide(&user, "entertainment", "US", 12, "user_1", 3000).is_none());
    }

    #[test]
    fn test_territory_filtering() {
        let mut engine = AdTargetingEngine::new();
        engine.register_campaign(make_campaign("us_only", &["all"], &["entertainment"]));

        let user = BinaryHV::from_hash(b"seg:all", DIMENSION);

        // US: OK
        assert!(engine.decide(&user, "entertainment", "US", 12, "user_1", 1000).is_some());
        // FR: blocked (campaign targets US only)
        assert!(engine.decide(&user, "entertainment", "FR", 12, "user_2", 1000).is_none());
    }
}
