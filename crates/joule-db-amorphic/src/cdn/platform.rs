//! Platform API Connectors — sync distribution status with external platforms.
//!
//! Tracks distribution state across YouTube, TikTok, Meta, etc.
//! Maps JouleDB distribution manifests to platform-specific APIs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::distribution::{DistributionStatus, PlatformDistribution};

/// Supported external platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlatformType {
    YouTube,
    TikTok,
    Instagram,
    Facebook,
    Twitter,
    Twitch,
    Roku,
    PlutoTV,
    Tubi,
    Samsung,
    LG,
    Vizio,
    Custom,
}

impl PlatformType {
    pub fn name(&self) -> &str {
        match self {
            PlatformType::YouTube => "YouTube",
            PlatformType::TikTok => "TikTok",
            PlatformType::Instagram => "Instagram",
            PlatformType::Facebook => "Facebook",
            PlatformType::Twitter => "Twitter/X",
            PlatformType::Twitch => "Twitch",
            PlatformType::Roku => "Roku Channel",
            PlatformType::PlutoTV => "Pluto TV",
            PlatformType::Tubi => "Tubi",
            PlatformType::Samsung => "Samsung TV Plus",
            PlatformType::LG => "LG Channels",
            PlatformType::Vizio => "Vizio WatchFree+",
            PlatformType::Custom => "Custom",
        }
    }

    /// API base URL for status checks.
    pub fn api_base(&self) -> &str {
        match self {
            PlatformType::YouTube => "https://www.googleapis.com/youtube/v3",
            PlatformType::TikTok => "https://open.tiktokapis.com/v2",
            PlatformType::Instagram => "https://graph.instagram.com/v18.0",
            PlatformType::Facebook => "https://graph.facebook.com/v18.0",
            _ => "",
        }
    }
}

/// Sync status for a platform connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatus {
    pub platform: PlatformType,
    pub external_id: Option<String>,
    pub status: DistributionStatus,
    pub last_synced_ms: u64,
    pub error: Option<String>,
    pub metrics: Option<PlatformMetrics>,
}

/// Platform-reported metrics for a distributed content item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformMetrics {
    pub views: u64,
    pub likes: u64,
    pub shares: u64,
    pub comments: u64,
    pub watch_time_hours: f64,
    pub revenue_microcents: Option<u64>,
}

/// A connector to an external platform's API.
///
/// This is the trait that platform-specific adapters implement.
/// The connector handles authentication, status sync, and metrics retrieval.
pub trait PlatformConnector: Send + Sync {
    /// Platform type.
    fn platform_type(&self) -> PlatformType;

    /// Check if content is live on the platform.
    fn check_status(&self, external_id: &str) -> Result<SyncStatus, String>;

    /// Upload/publish content to the platform.
    fn publish(&self, content_id: &str, metadata: &HashMap<String, String>) -> Result<String, String>;

    /// Retrieve metrics for published content.
    fn get_metrics(&self, external_id: &str) -> Result<PlatformMetrics, String>;

    /// Remove content from the platform.
    fn unpublish(&self, external_id: &str) -> Result<(), String>;
}

/// Manages connections to multiple platforms and tracks sync state.
pub struct PlatformManager {
    /// Platform type → last known sync status per content
    statuses: HashMap<(String, PlatformType), SyncStatus>,
}

impl PlatformManager {
    pub fn new() -> Self {
        Self {
            statuses: HashMap::new(),
        }
    }

    /// Record a sync status update.
    pub fn update_status(&mut self, content_id: &str, status: SyncStatus) {
        self.statuses
            .insert((content_id.to_string(), status.platform), status);
    }

    /// Get the current sync status for a content item on a platform.
    pub fn get_status(&self, content_id: &str, platform: PlatformType) -> Option<&SyncStatus> {
        self.statuses.get(&(content_id.to_string(), platform))
    }

    /// Get all platforms where content is distributed.
    pub fn content_platforms(&self, content_id: &str) -> Vec<&SyncStatus> {
        self.statuses
            .iter()
            .filter(|((cid, _), _)| cid == content_id)
            .map(|(_, status)| status)
            .collect()
    }

    /// Get all content distributed to a specific platform.
    pub fn platform_content(&self, platform: PlatformType) -> Vec<(&str, &SyncStatus)> {
        self.statuses
            .iter()
            .filter(|((_, p), _)| *p == platform)
            .map(|((cid, _), status)| (cid.as_str(), status))
            .collect()
    }

    /// Aggregate metrics across all platforms for a content item.
    pub fn aggregate_metrics(&self, content_id: &str) -> PlatformMetrics {
        let mut total = PlatformMetrics {
            views: 0,
            likes: 0,
            shares: 0,
            comments: 0,
            watch_time_hours: 0.0,
            revenue_microcents: Some(0),
        };

        for status in self.content_platforms(content_id) {
            if let Some(ref m) = status.metrics {
                total.views += m.views;
                total.likes += m.likes;
                total.shares += m.shares;
                total.comments += m.comments;
                total.watch_time_hours += m.watch_time_hours;
                if let (Some(total_rev), Some(rev)) = (&mut total.revenue_microcents, m.revenue_microcents) {
                    *total_rev += rev;
                }
            }
        }

        total
    }

    pub fn tracked_count(&self) -> usize {
        self.statuses.len()
    }
}

impl Default for PlatformManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_manager() {
        let mut mgr = PlatformManager::new();

        mgr.update_status("video_1", SyncStatus {
            platform: PlatformType::YouTube,
            external_id: Some("yt_abc123".into()),
            status: DistributionStatus::Active,
            last_synced_ms: 1000,
            error: None,
            metrics: Some(PlatformMetrics {
                views: 10000,
                likes: 500,
                shares: 50,
                comments: 100,
                watch_time_hours: 500.0,
                revenue_microcents: Some(50_000_000),
            }),
        });

        mgr.update_status("video_1", SyncStatus {
            platform: PlatformType::TikTok,
            external_id: Some("tt_xyz789".into()),
            status: DistributionStatus::Active,
            last_synced_ms: 1000,
            error: None,
            metrics: Some(PlatformMetrics {
                views: 50000,
                likes: 5000,
                shares: 2000,
                comments: 300,
                watch_time_hours: 200.0,
                revenue_microcents: None,
            }),
        });

        let platforms = mgr.content_platforms("video_1");
        assert_eq!(platforms.len(), 2);

        let agg = mgr.aggregate_metrics("video_1");
        assert_eq!(agg.views, 60000);
        assert_eq!(agg.likes, 5500);
    }

    #[test]
    fn test_platform_api_base() {
        assert!(PlatformType::YouTube.api_base().contains("googleapis"));
        assert!(PlatformType::TikTok.api_base().contains("tiktok"));
    }
}
