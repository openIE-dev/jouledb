//! Distribution Manifest System — multi-platform content syndication.
//!
//! Content goes to Netflix AND YouTube AND TikTok AND Instagram — each
//! with different format requirements, metadata schemas, and rights windows.
//!
//! This module maps one content item to N platform-specific representations,
//! leveraging the amorphic engine's ability to materialize different views
//! from the same underlying record.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::RecordId;

/// A distribution platform with its requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Platform {
    pub platform_id: String,
    pub name: String,
    /// Required metadata fields for this platform
    pub required_fields: Vec<String>,
    /// Supported video formats
    pub video_formats: Vec<VideoFormat>,
    /// Maximum file size in bytes (0 = unlimited)
    pub max_file_size: u64,
    /// Platform-specific metadata mapping (our field → their field)
    pub field_mapping: HashMap<String, String>,
}

/// Video encoding format specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFormat {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub bitrate_kbps: u32,
    pub frame_rate: f32,
    pub hdr: bool,
}

/// A distribution manifest: maps one content item to its platform variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionManifest {
    pub content_id: String,
    pub title: String,
    /// Per-platform distribution entries
    pub distributions: Vec<PlatformDistribution>,
}

/// Distribution details for one platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformDistribution {
    pub platform_id: String,
    /// Platform-specific content ID (e.g., YouTube video ID)
    pub external_id: Option<String>,
    /// Distribution status
    pub status: DistributionStatus,
    /// Syndication window (unix ms)
    pub available_from: u64,
    pub available_until: u64,
    /// Selected encoding profile
    pub format: Option<VideoFormat>,
    /// Platform-mapped metadata
    pub metadata: HashMap<String, String>,
}

/// Distribution status for a platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionStatus {
    /// Not yet distributed
    Pending,
    /// Currently processing/encoding
    Processing,
    /// Live and available
    Active,
    /// Temporarily unavailable (windowing)
    Windowed,
    /// Permanently removed
    Removed,
    /// Distribution failed
    Failed,
}

/// Manages distribution manifests across platforms.
pub struct DistributionManager {
    /// Registered platforms
    platforms: HashMap<String, Platform>,
    /// Content → manifest mapping
    manifests: HashMap<String, DistributionManifest>,
}

impl DistributionManager {
    pub fn new() -> Self {
        Self {
            platforms: HashMap::new(),
            manifests: HashMap::new(),
        }
    }

    /// Register a distribution platform.
    pub fn register_platform(&mut self, platform: Platform) {
        self.platforms
            .insert(platform.platform_id.clone(), platform);
    }

    /// Create a distribution manifest for content across specified platforms.
    pub fn create_manifest(
        &mut self,
        content_id: &str,
        title: &str,
        platform_ids: &[&str],
        metadata: &HashMap<String, String>,
        available_from: u64,
        available_until: u64,
    ) -> DistributionManifest {
        let distributions: Vec<PlatformDistribution> = platform_ids
            .iter()
            .filter_map(|&pid| {
                self.platforms.get(pid).map(|platform| {
                    // Map metadata fields to platform-specific names
                    let mapped: HashMap<String, String> = platform
                        .field_mapping
                        .iter()
                        .filter_map(|(our_field, their_field)| {
                            metadata
                                .get(our_field)
                                .map(|v| (their_field.clone(), v.clone()))
                        })
                        .collect();

                    // Select best encoding format for platform
                    let format = platform.video_formats.first().cloned();

                    PlatformDistribution {
                        platform_id: pid.to_string(),
                        external_id: None,
                        status: DistributionStatus::Pending,
                        available_from,
                        available_until,
                        format,
                        metadata: mapped,
                    }
                })
            })
            .collect();

        let manifest = DistributionManifest {
            content_id: content_id.to_string(),
            title: title.to_string(),
            distributions,
        };

        self.manifests
            .insert(content_id.to_string(), manifest.clone());
        manifest
    }

    /// Update distribution status for a platform.
    pub fn update_status(
        &mut self,
        content_id: &str,
        platform_id: &str,
        status: DistributionStatus,
        external_id: Option<String>,
    ) -> bool {
        if let Some(manifest) = self.manifests.get_mut(content_id) {
            if let Some(dist) = manifest
                .distributions
                .iter_mut()
                .find(|d| d.platform_id == platform_id)
            {
                dist.status = status;
                if let Some(eid) = external_id {
                    dist.external_id = Some(eid);
                }
                return true;
            }
        }
        false
    }

    /// Get all active distributions for a content item.
    pub fn active_distributions(&self, content_id: &str, now_ms: u64) -> Vec<&PlatformDistribution> {
        self.manifests
            .get(content_id)
            .map(|m| {
                m.distributions
                    .iter()
                    .filter(|d| {
                        d.status == DistributionStatus::Active
                            && now_ms >= d.available_from
                            && now_ms < d.available_until
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all content distributed to a specific platform.
    pub fn platform_catalog(&self, platform_id: &str) -> Vec<(&str, &PlatformDistribution)> {
        self.manifests
            .iter()
            .flat_map(|(content_id, manifest)| {
                manifest
                    .distributions
                    .iter()
                    .filter(|d| d.platform_id == platform_id)
                    .map(move |d| (content_id.as_str(), d))
            })
            .collect()
    }

    /// Validate that all required metadata fields are present for a platform.
    pub fn validate_metadata(
        &self,
        platform_id: &str,
        metadata: &HashMap<String, String>,
    ) -> Vec<String> {
        self.platforms
            .get(platform_id)
            .map(|platform| {
                platform
                    .required_fields
                    .iter()
                    .filter(|f| !metadata.contains_key(f.as_str()))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn platform_count(&self) -> usize {
        self.platforms.len()
    }

    pub fn manifest_count(&self) -> usize {
        self.manifests.len()
    }
}

impl Default for DistributionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn youtube_platform() -> Platform {
        Platform {
            platform_id: "youtube".into(),
            name: "YouTube".into(),
            required_fields: vec!["title".into(), "description".into()],
            video_formats: vec![VideoFormat {
                codec: "h264".into(),
                width: 1920,
                height: 1080,
                bitrate_kbps: 8000,
                frame_rate: 30.0,
                hdr: false,
            }],
            max_file_size: 128_000_000_000, // 128GB
            field_mapping: [("title", "snippet.title"), ("description", "snippet.description")]
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn tiktok_platform() -> Platform {
        Platform {
            platform_id: "tiktok".into(),
            name: "TikTok".into(),
            required_fields: vec!["title".into()],
            video_formats: vec![VideoFormat {
                codec: "h264".into(),
                width: 1080,
                height: 1920,
                bitrate_kbps: 4000,
                frame_rate: 30.0,
                hdr: false,
            }],
            max_file_size: 10_000_000_000, // 10GB
            field_mapping: [("title", "caption")]
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    #[test]
    fn test_create_manifest() {
        let mut mgr = DistributionManager::new();
        mgr.register_platform(youtube_platform());
        mgr.register_platform(tiktok_platform());

        let metadata: HashMap<String, String> = [
            ("title", "My Video"),
            ("description", "A great video"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

        let manifest = mgr.create_manifest(
            "video_1",
            "My Video",
            &["youtube", "tiktok"],
            &metadata,
            0,
            u64::MAX,
        );

        assert_eq!(manifest.distributions.len(), 2);
        assert_eq!(manifest.distributions[0].status, DistributionStatus::Pending);
    }

    #[test]
    fn test_update_status() {
        let mut mgr = DistributionManager::new();
        mgr.register_platform(youtube_platform());

        let metadata = HashMap::new();
        mgr.create_manifest("v1", "V1", &["youtube"], &metadata, 0, u64::MAX);

        assert!(mgr.update_status(
            "v1",
            "youtube",
            DistributionStatus::Active,
            Some("yt_abc123".into()),
        ));

        let active = mgr.active_distributions("v1", 1000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].external_id.as_deref(), Some("yt_abc123"));
    }

    #[test]
    fn test_validate_metadata() {
        let mut mgr = DistributionManager::new();
        mgr.register_platform(youtube_platform());

        // Missing "description"
        let metadata: HashMap<String, String> =
            [("title", "test")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();

        let missing = mgr.validate_metadata("youtube", &metadata);
        assert_eq!(missing, vec!["description"]);
    }

    #[test]
    fn test_platform_catalog() {
        let mut mgr = DistributionManager::new();
        mgr.register_platform(youtube_platform());

        let m = HashMap::new();
        mgr.create_manifest("v1", "V1", &["youtube"], &m, 0, u64::MAX);
        mgr.create_manifest("v2", "V2", &["youtube"], &m, 0, u64::MAX);

        let catalog = mgr.platform_catalog("youtube");
        assert_eq!(catalog.len(), 2);
    }
}
