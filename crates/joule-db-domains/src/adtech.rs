//! JouleDB AdTech Link
//!
//! HDC-powered Advertising Technology and Real-Time Bidding module.
//! Provides sub-millisecond audience matching, bid optimization, and fraud detection.

pub use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Types
// ============================================================================

/// A user profile for targeting
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub segments: Vec<String>,
    pub interests: Vec<String>,
    pub demographics: Demographics,
    pub device_types: Vec<DeviceType>,
    pub geo: Option<GeoInfo>,
    pub recency_score: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Demographics {
    pub age_range: Option<(u8, u8)>,
    pub gender: Option<Gender>,
    pub income_bracket: Option<IncomeBracket>,
    pub education: Option<Education>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Gender {
    Male,
    Female,
    Other,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum IncomeBracket {
    Low,
    LowerMiddle,
    Middle,
    UpperMiddle,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Education {
    HighSchool,
    SomeCollege,
    Bachelors,
    Graduate,
    Professional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeviceType {
    Desktop,
    Mobile,
    Tablet,
    Ctv,
    Other,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeoInfo {
    pub country: String,
    pub region: Option<String>,
    pub city: Option<String>,
    pub zip: Option<String>,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
}

/// A bid request from an exchange
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BidRequest {
    pub request_id: String,
    pub user_id: Option<String>,
    pub device_type: DeviceType,
    pub site_domain: String,
    pub page_url: Option<String>,
    pub ad_slot: AdSlot,
    pub floor_price: f64,
    pub geo: Option<GeoInfo>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdSlot {
    pub slot_id: String,
    pub width: u32,
    pub height: u32,
    pub position: AdPosition,
    pub ad_format: AdFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AdPosition {
    AboveFold,
    BelowFold,
    Sidebar,
    InContent,
    Footer,
    Interstitial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AdFormat {
    Banner,
    Video,
    Native,
    Audio,
    Rich,
}

/// A campaign targeting configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Campaign {
    pub campaign_id: String,
    pub advertiser_id: String,
    pub target_segments: Vec<String>,
    pub target_interests: Vec<String>,
    pub target_demographics: Option<Demographics>,
    pub target_devices: Vec<DeviceType>,
    pub target_geos: Vec<String>,
    pub blacklist_domains: Vec<String>,
    pub max_bid: f64,
    pub daily_budget: f64,
    pub total_budget: f64,
}

/// Creative (ad unit)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Creative {
    pub creative_id: String,
    pub campaign_id: String,
    pub format: AdFormat,
    pub width: u32,
    pub height: u32,
    pub content_tags: Vec<String>,
    pub landing_url: String,
}

/// An impression event
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Impression {
    pub impression_id: String,
    pub request_id: String,
    pub campaign_id: String,
    pub creative_id: String,
    pub user_id: Option<String>,
    pub winning_bid: f64,
    pub viewable: bool,
    pub timestamp: u64,
}

// ============================================================================
// AdTech Link Encoder
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// VSA Encoder for advertising data
    pub struct AdTechLink {
        seed: 0xAD7E_C4B1,
        dimension: 10000,
        fields: ["segments", "interests", "device", "gender", "income", "education",
                 "geo", "domain", "position", "format", "size", "price"],
        scalars: ["age", "price", "size", "recency"],
        enums: {
            device_vectors: DeviceType => [DeviceType::Desktop, DeviceType::Mobile, DeviceType::Tablet, DeviceType::Ctv, DeviceType::Other],
            gender_vectors: Gender => [Gender::Male, Gender::Female, Gender::Other, Gender::Unknown],
            income_vectors: IncomeBracket => [IncomeBracket::Low, IncomeBracket::LowerMiddle, IncomeBracket::Middle, IncomeBracket::UpperMiddle, IncomeBracket::High],
            education_vectors: Education => [Education::HighSchool, Education::SomeCollege, Education::Bachelors, Education::Graduate, Education::Professional],
            position_vectors: AdPosition => [AdPosition::AboveFold, AdPosition::BelowFold, AdPosition::Sidebar, AdPosition::InContent, AdPosition::Footer, AdPosition::Interstitial],
            format_vectors: AdFormat => [AdFormat::Banner, AdFormat::Video, AdFormat::Native, AdFormat::Audio, AdFormat::Rich]
        },
    }
}

impl AdTechLink {
    /// Encode a user profile
    pub fn encode_user(&self, user: &UserProfile) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Segments (superposition)
        if !user.segments.is_empty() {
            let mut seg_acc = BundleAccumulator::new(DIMENSION);
            for seg in &user.segments {
                seg_acc.add(&BinaryHV::from_hash(seg.as_bytes(), DIMENSION));
            }
            acc.add(&self.field_vectors["segments"].bind(&seg_acc.threshold()));
        }

        // Interests (superposition)
        if !user.interests.is_empty() {
            let mut int_acc = BundleAccumulator::new(DIMENSION);
            for interest in &user.interests {
                int_acc.add(&BinaryHV::from_hash(interest.as_bytes(), DIMENSION));
            }
            acc.add(&self.field_vectors["interests"].bind(&int_acc.threshold()));
        }

        // Devices (superposition)
        if !user.device_types.is_empty() {
            let mut dev_acc = BundleAccumulator::new(DIMENSION);
            for device in &user.device_types {
                dev_acc.add(&self.device_vectors[device]);
            }
            acc.add(&self.field_vectors["device"].bind(&dev_acc.threshold()));
        }

        // Demographics
        if let Some(gender) = &user.demographics.gender {
            acc.add(&self.field_vectors["gender"].bind(&self.gender_vectors[gender]));
        }
        if let Some(income) = &user.demographics.income_bracket {
            acc.add(&self.field_vectors["income"].bind(&self.income_vectors[income]));
        }
        if let Some(education) = &user.demographics.education {
            acc.add(&self.field_vectors["education"].bind(&self.education_vectors[education]));
        }

        // Geo
        if let Some(geo) = &user.geo {
            let geo_hv = BinaryHV::from_hash(geo.country.as_bytes(), DIMENSION);
            acc.add(&self.field_vectors["geo"].bind(&geo_hv));
        }

        // Recency
        let rec_shift = (user.recency_score * 100.0) as usize % 157;
        let rec_vec = self.scalar_bases["recency"].permute_words(rec_shift);
        acc.add(&rec_vec);

        acc.threshold()
    }

    /// Encode a campaign targeting
    pub fn encode_campaign(&self, campaign: &Campaign) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Target segments
        if !campaign.target_segments.is_empty() {
            let mut seg_acc = BundleAccumulator::new(DIMENSION);
            for seg in &campaign.target_segments {
                seg_acc.add(&BinaryHV::from_hash(seg.as_bytes(), DIMENSION));
            }
            acc.add(&self.field_vectors["segments"].bind(&seg_acc.threshold()));
        }

        // Target interests
        if !campaign.target_interests.is_empty() {
            let mut int_acc = BundleAccumulator::new(DIMENSION);
            for interest in &campaign.target_interests {
                int_acc.add(&BinaryHV::from_hash(interest.as_bytes(), DIMENSION));
            }
            acc.add(&self.field_vectors["interests"].bind(&int_acc.threshold()));
        }

        // Target devices
        if !campaign.target_devices.is_empty() {
            let mut dev_acc = BundleAccumulator::new(DIMENSION);
            for device in &campaign.target_devices {
                dev_acc.add(&self.device_vectors[device]);
            }
            acc.add(&self.field_vectors["device"].bind(&dev_acc.threshold()));
        }

        // Target geos
        if !campaign.target_geos.is_empty() {
            let mut geo_acc = BundleAccumulator::new(DIMENSION);
            for geo in &campaign.target_geos {
                geo_acc.add(&BinaryHV::from_hash(geo.as_bytes(), DIMENSION));
            }
            acc.add(&self.field_vectors["geo"].bind(&geo_acc.threshold()));
        }

        acc.threshold()
    }

    /// Encode a bid request
    pub fn encode_bid_request(&self, request: &BidRequest) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Device
        acc.add(&self.field_vectors["device"].bind(&self.device_vectors[&request.device_type]));

        // Domain
        let domain_hv = BinaryHV::from_hash(request.site_domain.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["domain"].bind(&domain_hv));

        // Ad slot
        acc.add(
            &self.field_vectors["position"].bind(&self.position_vectors[&request.ad_slot.position]),
        );
        acc.add(
            &self.field_vectors["format"].bind(&self.format_vectors[&request.ad_slot.ad_format]),
        );

        // Size (width * height bucket)
        let size = request.ad_slot.width * request.ad_slot.height;
        let size_shift = ((size as f64).log10() * 30.0) as usize % 157;
        let size_vec = self.scalar_bases["size"].permute_words(size_shift);
        acc.add(&self.field_vectors["size"].bind(&size_vec));

        // Geo
        if let Some(geo) = &request.geo {
            let geo_hv = BinaryHV::from_hash(geo.country.as_bytes(), DIMENSION);
            acc.add(&self.field_vectors["geo"].bind(&geo_hv));
        }

        acc.threshold()
    }
}

// ============================================================================
// Audience Matcher
// ============================================================================

/// Real-time audience matcher using holographic similarity
pub struct AudienceMatcher {
    /// Campaign vectors
    campaigns: HashMap<String, BinaryHV>,
    /// Campaign metadata
    campaign_data: HashMap<String, Campaign>,
    /// Encoder
    encoder: AdTechLink,
}

impl AudienceMatcher {
    pub fn new() -> Self {
        Self {
            campaigns: HashMap::new(),
            campaign_data: HashMap::new(),
            encoder: AdTechLink::new(),
        }
    }

    /// Add a campaign
    pub fn add_campaign(&mut self, campaign: Campaign) {
        let hv = self.encoder.encode_campaign(&campaign);
        self.campaigns.insert(campaign.campaign_id.clone(), hv);
        self.campaign_data
            .insert(campaign.campaign_id.clone(), campaign);
    }

    /// Find matching campaigns for a user (O(N) but each check is O(1))
    pub fn find_matches(&self, user: &UserProfile, min_score: f32) -> Vec<CampaignMatch> {
        let user_hv = self.encoder.encode_user(user);

        let mut matches: Vec<CampaignMatch> = self
            .campaigns
            .iter()
            .filter_map(|(id, hv)| {
                let similarity = user_hv.similarity(hv);
                if similarity >= min_score {
                    let campaign = self.campaign_data.get(id)?;
                    Some(CampaignMatch {
                        campaign_id: id.clone(),
                        match_score: similarity,
                        max_bid: campaign.max_bid,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by match score descending
        matches.sort_by(|a, b| {
            b.match_score
                .partial_cmp(&a.match_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }

    /// Quick bid decision for a request
    pub fn decide_bid(
        &self,
        request: &BidRequest,
        user: Option<&UserProfile>,
    ) -> Option<BidDecision> {
        let request_hv = self.encoder.encode_bid_request(request);

        let user_hv = user.map(|u| self.encoder.encode_user(u));

        let mut best_match: Option<(String, f32, f64)> = None;

        for (id, campaign_hv) in &self.campaigns {
            // Match request to campaign
            let request_score = request_hv.similarity(campaign_hv);

            // If we have user data, also check user match
            let user_score = user_hv
                .as_ref()
                .map(|u| u.similarity(campaign_hv))
                .unwrap_or(0.5);

            let combined_score = request_score * 0.4 + user_score * 0.6;

            if combined_score > 0.5 {
                if let Some(campaign) = self.campaign_data.get(id) {
                    if best_match.is_none() || combined_score > best_match.as_ref().unwrap().1 {
                        best_match = Some((id.clone(), combined_score, campaign.max_bid));
                    }
                }
            }
        }

        best_match.map(|(campaign_id, score, max_bid)| {
            // Calculate bid price based on score and floor
            let bid_price =
                (request.floor_price + (max_bid - request.floor_price) * score as f64).min(max_bid);

            BidDecision {
                campaign_id,
                bid_price,
                match_score: score,
            }
        })
    }

    /// Campaign count
    pub fn campaign_count(&self) -> usize {
        self.campaigns.len()
    }
}

impl Default for AudienceMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CampaignMatch {
    pub campaign_id: String,
    pub match_score: f32,
    pub max_bid: f64,
}

#[derive(Debug, Clone)]
pub struct BidDecision {
    pub campaign_id: String,
    pub bid_price: f64,
    pub match_score: f32,
}

// ============================================================================
// Fraud Detection
// ============================================================================

/// Holographic fraud detector
pub struct FraudDetector {
    /// Known fraudulent patterns
    fraud_bundle: BundleAccumulator,
    /// Normal traffic baseline
    normal_bundle: BundleAccumulator,
    /// Encoder
    encoder: AdTechLink,
    /// Observation counts
    fraud_count: usize,
    normal_count: usize,
}

impl FraudDetector {
    pub fn new() -> Self {
        Self {
            fraud_bundle: BundleAccumulator::new(DIMENSION),
            normal_bundle: BundleAccumulator::new(DIMENSION),
            encoder: AdTechLink::new(),
            fraud_count: 0,
            normal_count: 0,
        }
    }

    /// Learn from known fraudulent request
    pub fn learn_fraud(&mut self, request: &BidRequest) {
        let hv = self.encoder.encode_bid_request(request);
        self.fraud_bundle.add(&hv);
        self.fraud_count += 1;
    }

    /// Learn from known legitimate request
    pub fn learn_normal(&mut self, request: &BidRequest) {
        let hv = self.encoder.encode_bid_request(request);
        self.normal_bundle.add(&hv);
        self.normal_count += 1;
    }

    /// Check if a request looks fraudulent
    pub fn check_request(&self, request: &BidRequest) -> FraudScore {
        let hv = self.encoder.encode_bid_request(request);

        let fraud_similarity = if self.fraud_count > 0 {
            hv.similarity(&self.fraud_bundle.threshold())
        } else {
            0.5
        };

        let normal_similarity = if self.normal_count > 0 {
            hv.similarity(&self.normal_bundle.threshold())
        } else {
            0.5
        };

        // Higher fraud_similarity relative to normal = more suspicious
        let risk_score = fraud_similarity / (fraud_similarity + normal_similarity);

        FraudScore {
            risk_score,
            fraud_similarity,
            normal_similarity,
            is_suspicious: risk_score > 0.6,
        }
    }
}

impl Default for FraudDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct FraudScore {
    pub risk_score: f32,
    pub fraud_similarity: f32,
    pub normal_similarity: f32,
    pub is_suspicious: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_user(segments: Vec<&str>, device: DeviceType) -> UserProfile {
        UserProfile {
            user_id: "test_user".to_string(),
            segments: segments.into_iter().map(|s| s.to_string()).collect(),
            interests: vec!["tech".to_string()],
            demographics: Demographics {
                age_range: Some((25, 34)),
                gender: Some(Gender::Male),
                income_bracket: Some(IncomeBracket::Middle),
                education: Some(Education::Bachelors),
            },
            device_types: vec![device],
            geo: Some(GeoInfo {
                country: "US".to_string(),
                region: Some("CA".to_string()),
                city: Some("San Francisco".to_string()),
                zip: None,
                lat: None,
                lon: None,
            }),
            recency_score: 0.8,
        }
    }

    #[test]
    fn test_user_encoding() {
        let link = AdTechLink::new();

        let user = make_user(
            vec!["auto_intenders", "tech_enthusiasts"],
            DeviceType::Mobile,
        );
        let hv = link.encode_user(&user);

        // Same user should encode consistently
        let hv2 = link.encode_user(&user);
        assert!((hv.similarity(&hv2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_audience_matching() {
        let mut matcher = AudienceMatcher::new();

        // Add a campaign targeting tech enthusiasts on mobile
        matcher.add_campaign(Campaign {
            campaign_id: "camp_001".to_string(),
            advertiser_id: "adv_001".to_string(),
            target_segments: vec!["tech_enthusiasts".to_string()],
            target_interests: vec!["tech".to_string()],
            target_demographics: None,
            target_devices: vec![DeviceType::Mobile],
            target_geos: vec!["US".to_string()],
            blacklist_domains: vec![],
            max_bid: 5.0,
            daily_budget: 1000.0,
            total_budget: 10000.0,
        });

        // User that matches
        let user = make_user(vec!["tech_enthusiasts"], DeviceType::Mobile);
        let matches = matcher.find_matches(&user, 0.5);

        assert!(!matches.is_empty());
        println!("Match score: {}", matches[0].match_score);
    }

    #[test]
    fn test_bid_decision() {
        let mut matcher = AudienceMatcher::new();

        matcher.add_campaign(Campaign {
            campaign_id: "camp_001".to_string(),
            advertiser_id: "adv_001".to_string(),
            target_segments: vec!["auto_intenders".to_string()],
            target_interests: vec![],
            target_demographics: None,
            target_devices: vec![DeviceType::Desktop],
            target_geos: vec!["US".to_string()],
            blacklist_domains: vec![],
            max_bid: 10.0,
            daily_budget: 5000.0,
            total_budget: 50000.0,
        });

        let request = BidRequest {
            request_id: "req_001".to_string(),
            user_id: Some("user_001".to_string()),
            device_type: DeviceType::Desktop,
            site_domain: "cars.com".to_string(),
            page_url: None,
            ad_slot: AdSlot {
                slot_id: "slot_001".to_string(),
                width: 300,
                height: 250,
                position: AdPosition::AboveFold,
                ad_format: AdFormat::Banner,
            },
            floor_price: 1.0,
            geo: Some(GeoInfo {
                country: "US".to_string(),
                region: None,
                city: None,
                zip: None,
                lat: None,
                lon: None,
            }),
            timestamp: 1000,
        };

        let user = make_user(vec!["auto_intenders"], DeviceType::Desktop);
        let decision = matcher.decide_bid(&request, Some(&user));

        if let Some(d) = decision {
            println!("Bid: ${:.2}, Score: {:.2}", d.bid_price, d.match_score);
            assert!(d.bid_price >= request.floor_price);
            assert!(d.bid_price <= 10.0);
        }
    }

    #[test]
    fn test_fraud_detection() {
        let mut detector = FraudDetector::new();

        // Learn normal traffic patterns
        for _ in 0..100 {
            detector.learn_normal(&BidRequest {
                request_id: "normal".to_string(),
                user_id: Some("user".to_string()),
                device_type: DeviceType::Mobile,
                site_domain: "news.com".to_string(),
                page_url: None,
                ad_slot: AdSlot {
                    slot_id: "slot".to_string(),
                    width: 300,
                    height: 250,
                    position: AdPosition::AboveFold,
                    ad_format: AdFormat::Banner,
                },
                floor_price: 1.0,
                geo: Some(GeoInfo {
                    country: "US".to_string(),
                    region: None,
                    city: None,
                    zip: None,
                    lat: None,
                    lon: None,
                }),
                timestamp: 1000,
            });
        }

        // Learn fraud patterns
        for _ in 0..50 {
            detector.learn_fraud(&BidRequest {
                request_id: "fraud".to_string(),
                user_id: None, // No user ID often indicates bot
                device_type: DeviceType::Other,
                site_domain: "sketchy-site.xyz".to_string(),
                page_url: None,
                ad_slot: AdSlot {
                    slot_id: "slot".to_string(),
                    width: 1,
                    height: 1, // Tiny hidden ad
                    position: AdPosition::Footer,
                    ad_format: AdFormat::Banner,
                },
                floor_price: 0.01,
                geo: None,
                timestamp: 1000,
            });
        }

        // Check suspicious request
        let score = detector.check_request(&BidRequest {
            request_id: "test".to_string(),
            user_id: None,
            device_type: DeviceType::Other,
            site_domain: "another-sketchy.xyz".to_string(),
            page_url: None,
            ad_slot: AdSlot {
                slot_id: "slot".to_string(),
                width: 1,
                height: 1,
                position: AdPosition::Footer,
                ad_format: AdFormat::Banner,
            },
            floor_price: 0.01,
            geo: None,
            timestamp: 2000,
        });

        println!(
            "Fraud score: {:.2}, Suspicious: {}",
            score.risk_score, score.is_suspicious
        );
    }
}
