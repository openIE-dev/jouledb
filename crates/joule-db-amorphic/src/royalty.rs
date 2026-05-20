//! Royalty Calculation Engine — usage-based revenue distribution.
//!
//! YouTube pays $15B+/year to creators. Spotify's royalty system is constantly
//! criticized. This module orchestrates the existing JouleDB primitives
//! (temporal rights, events, graph relationships) into actual payment calculations.
//!
//! Pipeline: consumption events → usage aggregation → rights graph → revenue splits → payments

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{RecordId, Value};

/// A rights holder in the revenue graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightsHolder {
    pub holder_id: String,
    pub name: String,
    pub role: RightsRole,
    /// Revenue share as a fraction (0.0 - 1.0)
    pub share: f64,
}

/// Role of a rights holder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RightsRole {
    Creator,
    Writer,
    Performer,
    Producer,
    Publisher,
    Label,
    Distributor,
    Platform,
}

// ============================================================================
// E9: Production Royalty Depth
// ============================================================================

/// Type of right being licensed.
/// Different right types have different statutory rates and collection paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RightType {
    /// Mechanical: reproduction/distribution of a composition.
    /// US statutory rate: $0.12/unit or $0.0231/minute (2024-2027).
    Mechanical,
    /// Performance: public performance/broadcast/streaming.
    /// Collected by PROs (ASCAP, BMI, SESAC in US).
    Performance,
    /// Sync: use of music in visual media (film, TV, ads).
    /// Negotiated per-use, no statutory rate.
    Sync,
    /// Master: use of a specific recording (vs the composition).
    /// Negotiated with the label/owner of the master recording.
    Master,
    /// Neighboring: performer/producer rights in the recording.
    /// Collected by SoundExchange in US, PPL in UK.
    Neighboring,
}

/// Collection society that administers rights in a territory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSociety {
    pub code: String,
    pub name: String,
    pub territory: String,
    pub right_types: Vec<RightType>,
    /// Commission rate (fraction taken by the society)
    pub commission_rate: f64,
}

/// Statutory rate for a right type in a territory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatutoryRate {
    pub right_type: RightType,
    pub territory: String,
    /// Rate per stream/play (microcents)
    pub per_play_microcents: u64,
    /// Rate per minute of content (microcents), if duration-based
    pub per_minute_microcents: Option<u64>,
    /// Effective period
    pub valid_from_year: u16,
    pub valid_to_year: u16,
}

/// An advance payment against future royalties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Advance {
    pub advance_id: String,
    pub holder_id: String,
    /// Total advance amount (microcents)
    pub amount: u64,
    /// Amount already recouped (microcents)
    pub recouped: u64,
    /// Recoupment rate (fraction of royalties applied to recoup, typically 1.0)
    pub recoup_rate: f64,
}

impl Advance {
    /// Remaining unrecouped balance.
    pub fn balance(&self) -> u64 {
        self.amount.saturating_sub(self.recouped)
    }

    /// Is the advance fully recouped?
    pub fn is_recouped(&self) -> bool {
        self.recouped >= self.amount
    }

    /// Apply a royalty payment toward recoupment.
    /// Returns (amount_to_holder, amount_recouped).
    pub fn apply_recoupment(&mut self, royalty_amount: u64) -> (u64, u64) {
        if self.is_recouped() {
            return (royalty_amount, 0);
        }
        let recoup_amount = ((royalty_amount as f64 * self.recoup_rate) as u64)
            .min(self.balance());
        self.recouped += recoup_amount;
        let to_holder = royalty_amount - recoup_amount;
        (to_holder, recoup_amount)
    }
}

/// Registry of collection societies and statutory rates.
pub struct RightsRegistry {
    /// Collection societies by territory+right_type
    pub societies: Vec<CollectionSociety>,
    /// Statutory rates by territory+right_type
    pub rates: Vec<StatutoryRate>,
    /// Outstanding advances
    pub advances: HashMap<String, Advance>,
}

impl RightsRegistry {
    pub fn new() -> Self {
        Self {
            societies: Vec::new(),
            rates: Vec::new(),
            advances: HashMap::new(),
        }
    }

    /// Add default US collection societies and rates.
    pub fn with_us_defaults(mut self) -> Self {
        self.societies.extend(vec![
            CollectionSociety {
                code: "ASCAP".into(),
                name: "American Society of Composers, Authors and Publishers".into(),
                territory: "US".into(),
                right_types: vec![RightType::Performance],
                commission_rate: 0.115, // ~11.5%
            },
            CollectionSociety {
                code: "BMI".into(),
                name: "Broadcast Music, Inc.".into(),
                territory: "US".into(),
                right_types: vec![RightType::Performance],
                commission_rate: 0.135, // ~13.5%
            },
            CollectionSociety {
                code: "HFA".into(),
                name: "Harry Fox Agency".into(),
                territory: "US".into(),
                right_types: vec![RightType::Mechanical],
                commission_rate: 0.065, // ~6.5%
            },
            CollectionSociety {
                code: "SX".into(),
                name: "SoundExchange".into(),
                territory: "US".into(),
                right_types: vec![RightType::Neighboring],
                commission_rate: 0.05, // 5%
            },
        ]);

        // US statutory mechanical rate (2024-2027)
        self.rates.push(StatutoryRate {
            right_type: RightType::Mechanical,
            territory: "US".into(),
            per_play_microcents: 12000, // $0.12 = 12000 microcents
            per_minute_microcents: Some(2310), // $0.0231/min
            valid_from_year: 2024,
            valid_to_year: 2027,
        });

        self
    }

    /// Register an advance against a rights holder.
    pub fn register_advance(&mut self, advance: Advance) {
        self.advances.insert(advance.advance_id.clone(), advance);
    }

    /// Get applicable statutory rate for a right type in a territory.
    pub fn get_rate(&self, right_type: RightType, territory: &str, year: u16) -> Option<&StatutoryRate> {
        self.rates.iter().find(|r| {
            r.right_type == right_type
                && r.territory == territory
                && year >= r.valid_from_year
                && year <= r.valid_to_year
        })
    }

    /// Get collection societies for a territory and right type.
    pub fn get_societies(&self, territory: &str, right_type: RightType) -> Vec<&CollectionSociety> {
        self.societies
            .iter()
            .filter(|s| s.territory == territory && s.right_types.contains(&right_type))
            .collect()
    }

    /// Apply recoupment to a royalty payment.
    pub fn apply_recoupment(&mut self, holder_id: &str, amount: u64) -> (u64, u64) {
        let mut total_recouped = 0u64;
        let mut remaining = amount;

        for advance in self.advances.values_mut() {
            if advance.holder_id == holder_id && !advance.is_recouped() {
                let (to_holder, recouped) = advance.apply_recoupment(remaining);
                total_recouped += recouped;
                remaining = to_holder;
                if remaining == 0 {
                    break;
                }
            }
        }

        (remaining, total_recouped)
    }
}

impl Default for RightsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Usage event for royalty calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    pub content_id: String,
    pub user_id: String,
    pub territory: String,
    /// Duration consumed in milliseconds
    pub duration_ms: u64,
    /// Total content duration in milliseconds (for completion calculation)
    pub total_duration_ms: u64,
    /// Timestamp (unix ms)
    pub timestamp_ms: u64,
    /// Platform where consumption occurred
    pub platform: String,
}

impl UsageEvent {
    /// Completion ratio (0.0 - 1.0)
    pub fn completion(&self) -> f64 {
        if self.total_duration_ms == 0 {
            return 0.0;
        }
        (self.duration_ms as f64 / self.total_duration_ms as f64).min(1.0)
    }
}

/// Revenue pool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevenuePool {
    /// Total revenue to distribute (microcents)
    pub total_revenue: u64,
    /// Period start (unix ms)
    pub period_start_ms: u64,
    /// Period end (unix ms)
    pub period_end_ms: u64,
    /// Minimum completion ratio to count as a "play" (e.g., 0.3 = 30%)
    pub min_completion: f64,
    /// Distribution model
    pub model: DistributionModel,
}

/// How revenue is distributed across content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionModel {
    /// Pro-rata: each stream gets equal share of total pool.
    /// (Spotify model — criticized because heavy listeners subsidize casual ones)
    ProRata,
    /// User-centric: each user's subscription is split only among
    /// content THEY consumed. (Fairer to niche creators)
    UserCentric,
}

/// A calculated royalty payment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoyaltyPayment {
    pub holder_id: String,
    pub holder_name: String,
    pub role: RightsRole,
    pub content_id: String,
    /// Amount in microcents
    pub amount: u64,
    /// Number of qualifying plays
    pub play_count: u64,
    /// Territory breakdown
    pub territory_breakdown: HashMap<String, u64>,
}

/// The royalty calculator.
pub struct RoyaltyCalculator {
    /// Content → rights holders mapping
    rights_graph: HashMap<String, Vec<RightsHolder>>,
}

impl RoyaltyCalculator {
    pub fn new() -> Self {
        Self {
            rights_graph: HashMap::new(),
        }
    }

    /// Register rights holders for a content item.
    /// Shares should sum to 1.0 (or less if platform takes remainder).
    pub fn register_rights(
        &mut self,
        content_id: &str,
        holders: Vec<RightsHolder>,
    ) {
        self.rights_graph
            .insert(content_id.to_string(), holders);
    }

    /// Calculate royalty payments for a revenue pool given usage events.
    pub fn calculate(
        &self,
        pool: &RevenuePool,
        events: &[UsageEvent],
    ) -> Vec<RoyaltyPayment> {
        // Filter events to the pool's period and minimum completion
        let qualifying: Vec<&UsageEvent> = events
            .iter()
            .filter(|e| {
                e.timestamp_ms >= pool.period_start_ms
                    && e.timestamp_ms < pool.period_end_ms
                    && e.completion() >= pool.min_completion
            })
            .collect();

        match pool.model {
            DistributionModel::ProRata => self.calculate_pro_rata(pool, &qualifying),
            DistributionModel::UserCentric => self.calculate_user_centric(pool, &qualifying),
        }
    }

    /// Pro-rata: total pool / total plays × plays per content × rights shares.
    fn calculate_pro_rata(
        &self,
        pool: &RevenuePool,
        events: &[&UsageEvent],
    ) -> Vec<RoyaltyPayment> {
        let total_plays = events.len() as u64;
        if total_plays == 0 {
            return vec![];
        }

        let per_play = pool.total_revenue / total_plays;

        // Aggregate plays per content
        let mut content_plays: HashMap<String, ContentPlayStats> = HashMap::new();
        for event in events {
            let stats = content_plays
                .entry(event.content_id.clone())
                .or_insert_with(|| ContentPlayStats::new());
            stats.play_count += 1;
            *stats.territory_plays.entry(event.territory.clone()).or_default() += 1;
        }

        // Distribute to rights holders
        self.distribute_to_holders(&content_plays, per_play)
    }

    /// User-centric: each user's share split only among their consumed content.
    fn calculate_user_centric(
        &self,
        pool: &RevenuePool,
        events: &[&UsageEvent],
    ) -> Vec<RoyaltyPayment> {
        // Group events by user
        let mut user_events: HashMap<String, Vec<&UsageEvent>> = HashMap::new();
        for event in events {
            user_events
                .entry(event.user_id.clone())
                .or_default()
                .push(event);
        }

        let unique_users = user_events.len() as u64;
        if unique_users == 0 {
            return vec![];
        }

        let per_user = pool.total_revenue / unique_users;

        // For each user, split their share among their consumed content
        let mut content_plays: HashMap<String, ContentPlayStats> = HashMap::new();

        for (_, user_evts) in &user_events {
            let user_content_count = user_evts.len() as u64;
            if user_content_count == 0 {
                continue;
            }
            let per_content_per_user = per_user / user_content_count;

            for event in user_evts {
                let stats = content_plays
                    .entry(event.content_id.clone())
                    .or_insert_with(ContentPlayStats::new);
                stats.play_count += 1;
                stats.revenue += per_content_per_user;
                *stats.territory_plays.entry(event.territory.clone()).or_default() += 1;
            }
        }

        // For user-centric, per_play is already embedded in content_plays.revenue
        self.distribute_to_holders_absolute(&content_plays)
    }

    /// Distribute per-play revenue to rights holders.
    fn distribute_to_holders(
        &self,
        content_plays: &HashMap<String, ContentPlayStats>,
        per_play: u64,
    ) -> Vec<RoyaltyPayment> {
        let mut payments = Vec::new();

        for (content_id, stats) in content_plays {
            let content_revenue = per_play * stats.play_count;

            if let Some(holders) = self.rights_graph.get(content_id) {
                for holder in holders {
                    let amount = (content_revenue as f64 * holder.share) as u64;
                    if amount > 0 {
                        let territory_breakdown: HashMap<String, u64> = stats
                            .territory_plays
                            .iter()
                            .map(|(t, &count)| {
                                (t.clone(), (per_play as f64 * count as f64 * holder.share) as u64)
                            })
                            .collect();

                        payments.push(RoyaltyPayment {
                            holder_id: holder.holder_id.clone(),
                            holder_name: holder.name.clone(),
                            role: holder.role,
                            content_id: content_id.clone(),
                            amount,
                            play_count: stats.play_count,
                            territory_breakdown,
                        });
                    }
                }
            }
        }

        payments
    }

    /// Distribute absolute revenue amounts (for user-centric model).
    fn distribute_to_holders_absolute(
        &self,
        content_plays: &HashMap<String, ContentPlayStats>,
    ) -> Vec<RoyaltyPayment> {
        let mut payments = Vec::new();

        for (content_id, stats) in content_plays {
            if let Some(holders) = self.rights_graph.get(content_id) {
                for holder in holders {
                    let amount = (stats.revenue as f64 * holder.share) as u64;
                    if amount > 0 {
                        payments.push(RoyaltyPayment {
                            holder_id: holder.holder_id.clone(),
                            holder_name: holder.name.clone(),
                            role: holder.role,
                            content_id: content_id.clone(),
                            amount,
                            play_count: stats.play_count,
                            territory_breakdown: stats.territory_plays.iter()
                                .map(|(t, &c)| (t.clone(), (stats.revenue / stats.play_count.max(1) * c as u64) as u64))
                                .collect(),
                        });
                    }
                }
            }
        }

        payments
    }

    /// Number of content items with registered rights.
    pub fn content_count(&self) -> usize {
        self.rights_graph.len()
    }
}

impl Default for RoyaltyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

struct ContentPlayStats {
    play_count: u64,
    revenue: u64,
    territory_plays: HashMap<String, u64>,
}

impl ContentPlayStats {
    fn new() -> Self {
        Self {
            play_count: 0,
            revenue: 0,
            territory_plays: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_events() -> Vec<UsageEvent> {
        vec![
            UsageEvent {
                content_id: "song_a".into(),
                user_id: "user_1".into(),
                territory: "US".into(),
                duration_ms: 180_000,
                total_duration_ms: 200_000,
                timestamp_ms: 5000,
                platform: "streaming".into(),
            },
            UsageEvent {
                content_id: "song_a".into(),
                user_id: "user_2".into(),
                territory: "UK".into(),
                duration_ms: 190_000,
                total_duration_ms: 200_000,
                timestamp_ms: 6000,
                platform: "streaming".into(),
            },
            UsageEvent {
                content_id: "song_b".into(),
                user_id: "user_1".into(),
                territory: "US".into(),
                duration_ms: 150_000,
                total_duration_ms: 200_000,
                timestamp_ms: 7000,
                platform: "streaming".into(),
            },
        ]
    }

    #[test]
    fn test_pro_rata_distribution() {
        let mut calc = RoyaltyCalculator::new();

        calc.register_rights("song_a", vec![
            RightsHolder {
                holder_id: "artist_1".into(),
                name: "Artist One".into(),
                role: RightsRole::Performer,
                share: 0.5,
            },
            RightsHolder {
                holder_id: "label_1".into(),
                name: "Label One".into(),
                role: RightsRole::Label,
                share: 0.3,
            },
        ]);
        calc.register_rights("song_b", vec![
            RightsHolder {
                holder_id: "artist_2".into(),
                name: "Artist Two".into(),
                role: RightsRole::Performer,
                share: 0.6,
            },
        ]);

        let pool = RevenuePool {
            total_revenue: 3_000_000, // $30 in microcents
            period_start_ms: 0,
            period_end_ms: 100_000,
            min_completion: 0.5,
            model: DistributionModel::ProRata,
        };

        let payments = calc.calculate(&pool, &make_events());
        assert!(!payments.is_empty());

        // Total distributed should be <= pool
        let total: u64 = payments.iter().map(|p| p.amount).sum();
        assert!(total <= pool.total_revenue);

        // song_a has 2 plays, song_b has 1 play. Per play = 1M.
        // artist_1 gets 2 * 1M * 0.5 = 1M
        let artist_1: u64 = payments.iter()
            .filter(|p| p.holder_id == "artist_1")
            .map(|p| p.amount)
            .sum();
        assert_eq!(artist_1, 1_000_000);
    }

    #[test]
    fn test_user_centric_distribution() {
        let mut calc = RoyaltyCalculator::new();

        calc.register_rights("song_a", vec![
            RightsHolder {
                holder_id: "artist_1".into(),
                name: "Artist One".into(),
                role: RightsRole::Performer,
                share: 1.0,
            },
        ]);
        calc.register_rights("song_b", vec![
            RightsHolder {
                holder_id: "artist_2".into(),
                name: "Artist Two".into(),
                role: RightsRole::Performer,
                share: 1.0,
            },
        ]);

        let pool = RevenuePool {
            total_revenue: 2_000_000, // $20
            period_start_ms: 0,
            period_end_ms: 100_000,
            min_completion: 0.5,
            model: DistributionModel::UserCentric,
        };

        let payments = calc.calculate(&pool, &make_events());
        assert!(!payments.is_empty());

        // User-centric: user_1 pays for song_a + song_b equally
        // user_2 pays only for song_a
        // So song_a gets more total than pro-rata would give song_b
    }

    #[test]
    fn test_minimum_completion_filter() {
        let mut calc = RoyaltyCalculator::new();
        calc.register_rights("song_a", vec![
            RightsHolder {
                holder_id: "a".into(),
                name: "A".into(),
                role: RightsRole::Creator,
                share: 1.0,
            },
        ]);

        let pool = RevenuePool {
            total_revenue: 1_000_000,
            period_start_ms: 0,
            period_end_ms: 100_000,
            min_completion: 0.95, // Very strict
            model: DistributionModel::ProRata,
        };

        let events = vec![UsageEvent {
            content_id: "song_a".into(),
            user_id: "u".into(),
            territory: "US".into(),
            duration_ms: 100_000,  // 50% completion
            total_duration_ms: 200_000,
            timestamp_ms: 5000,
            platform: "streaming".into(),
        }];

        let payments = calc.calculate(&pool, &events);
        // Should be empty — completion < 95%
        assert!(payments.is_empty());
    }

    #[test]
    fn test_territory_breakdown() {
        let mut calc = RoyaltyCalculator::new();
        calc.register_rights("song_a", vec![
            RightsHolder {
                holder_id: "a".into(),
                name: "A".into(),
                role: RightsRole::Creator,
                share: 1.0,
            },
        ]);

        let pool = RevenuePool {
            total_revenue: 2_000_000,
            period_start_ms: 0,
            period_end_ms: 100_000,
            min_completion: 0.5,
            model: DistributionModel::ProRata,
        };

        let payments = calc.calculate(&pool, &make_events());
        let song_a_payment = payments.iter().find(|p| p.content_id == "song_a").unwrap();

        // song_a: 1 US play + 1 UK play
        assert!(song_a_payment.territory_breakdown.contains_key("US"));
        assert!(song_a_payment.territory_breakdown.contains_key("UK"));
    }
}
