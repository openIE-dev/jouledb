//! Social sharing — share URLs, OG metadata, short URLs, share sheet model.
//!
//! Replaces react-share / AddThis / ShareThis with a pure-Rust sharing model.
//! Generates platform-specific share URLs with proper encoding.

use std::collections::HashMap;
use std::fmt::Write;

// ── ShareTarget ─────────────────────────────────────────────────

/// Social sharing target platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShareTarget {
    Twitter,
    Facebook,
    LinkedIn,
    Reddit,
    Email,
    WhatsApp,
    Telegram,
    Copy,
}

impl ShareTarget {
    /// All available targets (excluding Copy, which is local-only).
    pub const ALL: &'static [ShareTarget] = &[
        ShareTarget::Twitter,
        ShareTarget::Facebook,
        ShareTarget::LinkedIn,
        ShareTarget::Reddit,
        ShareTarget::Email,
        ShareTarget::WhatsApp,
        ShareTarget::Telegram,
        ShareTarget::Copy,
    ];

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Twitter => "Twitter",
            Self::Facebook => "Facebook",
            Self::LinkedIn => "LinkedIn",
            Self::Reddit => "Reddit",
            Self::Email => "Email",
            Self::WhatsApp => "WhatsApp",
            Self::Telegram => "Telegram",
            Self::Copy => "Copy Link",
        }
    }
}

// ── URL Encoding ────────────────────────────────────────────────

/// Percent-encode a string for use in URLs.
fn url_encode(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len() * 3);
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                let _ = write!(encoded, "%{byte:02X}");
            }
        }
    }
    encoded
}

// ── ShareData ───────────────────────────────────────────────────

/// Data to share.
#[derive(Debug, Clone)]
pub struct ShareData {
    pub title: String,
    pub text: String,
    pub url: String,
    pub image_url: Option<String>,
}

impl ShareData {
    pub fn new(title: &str, text: &str, url: &str) -> Self {
        Self {
            title: title.to_string(),
            text: text.to_string(),
            url: url.to_string(),
            image_url: None,
        }
    }

    pub fn with_image(mut self, image_url: &str) -> Self {
        self.image_url = Some(image_url.to_string());
        self
    }

    /// Generate the share URL for a given target.
    pub fn share_url(&self, target: ShareTarget) -> Option<String> {
        let encoded_url = url_encode(&self.url);
        let encoded_title = url_encode(&self.title);
        let encoded_text = url_encode(&self.text);

        match target {
            ShareTarget::Twitter => {
                Some(format!(
                    "https://twitter.com/intent/tweet?text={encoded_text}&url={encoded_url}"
                ))
            }
            ShareTarget::Facebook => {
                Some(format!(
                    "https://www.facebook.com/sharer/sharer.php?u={encoded_url}"
                ))
            }
            ShareTarget::LinkedIn => {
                Some(format!(
                    "https://www.linkedin.com/sharing/share-offsite/?url={encoded_url}"
                ))
            }
            ShareTarget::Reddit => {
                Some(format!(
                    "https://www.reddit.com/submit?url={encoded_url}&title={encoded_title}"
                ))
            }
            ShareTarget::Email => {
                Some(format!(
                    "mailto:?subject={encoded_title}&body={encoded_text}%0A{encoded_url}"
                ))
            }
            ShareTarget::WhatsApp => {
                Some(format!(
                    "https://api.whatsapp.com/send?text={encoded_text}%20{encoded_url}"
                ))
            }
            ShareTarget::Telegram => {
                Some(format!(
                    "https://t.me/share/url?url={encoded_url}&text={encoded_text}"
                ))
            }
            ShareTarget::Copy => None, // Copy is handled client-side
        }
    }
}

// ── Short URL ───────────────────────────────────────────────────

/// Generate a short URL identifier from a URL using a simple hash.
pub fn short_url_hash(url: &str) -> String {
    // FNV-1a 64-bit hash, then base36 encode.
    let hash = fnv1a_64(url.as_bytes());
    base36_encode(hash)
}

/// FNV-1a 64-bit hash.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x00000100000001B3);
    }
    hash
}

/// Encode a u64 as base-36 (0-9a-z), truncated to 8 chars.
fn base36_encode(mut value: u64) -> String {
    const CHARS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut result = Vec::with_capacity(13);
    if value == 0 {
        return "0".to_string();
    }
    while value > 0 {
        let rem = (value % 36) as usize;
        result.push(CHARS[rem]);
        value /= 36;
    }
    result.reverse();
    let s: String = result.into_iter().map(|b| b as char).collect();
    // Truncate to 8 chars for short URL.
    if s.len() > 8 { s[..8].to_string() } else { s }
}

// ── OG Metadata ─────────────────────────────────────────────────

/// Open Graph metadata for social previews.
#[derive(Debug, Clone)]
pub struct OgMetadata {
    pub title: String,
    pub description: String,
    pub image: Option<String>,
    pub url: Option<String>,
    pub site_name: Option<String>,
    pub og_type: Option<String>,
}

impl OgMetadata {
    pub fn new(title: &str, description: &str) -> Self {
        Self {
            title: title.to_string(),
            description: description.to_string(),
            image: None,
            url: None,
            site_name: None,
            og_type: None,
        }
    }

    pub fn with_image(mut self, image: &str) -> Self {
        self.image = Some(image.to_string());
        self
    }

    pub fn with_url(mut self, url: &str) -> Self {
        self.url = Some(url.to_string());
        self
    }

    pub fn with_site_name(mut self, name: &str) -> Self {
        self.site_name = Some(name.to_string());
        self
    }

    pub fn with_type(mut self, og_type: &str) -> Self {
        self.og_type = Some(og_type.to_string());
        self
    }

    /// Render OG meta tags as HTML.
    pub fn to_html(&self) -> String {
        let mut html = String::new();
        let _ = write!(
            html,
            "<meta property=\"og:title\" content=\"{}\">",
            html_escape(&self.title)
        );
        let _ = write!(
            html,
            "<meta property=\"og:description\" content=\"{}\">",
            html_escape(&self.description)
        );
        if let Some(image) = &self.image {
            let _ = write!(
                html,
                "<meta property=\"og:image\" content=\"{}\">",
                html_escape(image)
            );
        }
        if let Some(url) = &self.url {
            let _ = write!(
                html,
                "<meta property=\"og:url\" content=\"{}\">",
                html_escape(url)
            );
        }
        if let Some(site_name) = &self.site_name {
            let _ = write!(
                html,
                "<meta property=\"og:site_name\" content=\"{}\">",
                html_escape(site_name)
            );
        }
        if let Some(og_type) = &self.og_type {
            let _ = write!(
                html,
                "<meta property=\"og:type\" content=\"{}\">",
                html_escape(og_type)
            );
        }
        html
    }

    /// Build OG metadata from ShareData.
    pub fn from_share_data(data: &ShareData) -> Self {
        let mut og = Self::new(&data.title, &data.text).with_url(&data.url);
        if let Some(img) = &data.image_url {
            og = og.with_image(img);
        }
        og
    }
}

/// Minimal HTML escaping for attribute values.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

// ── ShareCountTracker ───────────────────────────────────────────

/// Tracks share counts per URL and target.
#[derive(Debug, Clone, Default)]
pub struct ShareCountTracker {
    counts: HashMap<String, HashMap<ShareTarget, u64>>,
}

impl ShareCountTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a share event.
    pub fn record(&mut self, url: &str, target: ShareTarget) {
        *self
            .counts
            .entry(url.to_string())
            .or_default()
            .entry(target)
            .or_insert(0) += 1;
    }

    /// Get total shares for a URL.
    pub fn total_for_url(&self, url: &str) -> u64 {
        self.counts
            .get(url)
            .map(|m| m.values().sum())
            .unwrap_or(0)
    }

    /// Get shares for a URL on a specific target.
    pub fn count_for(&self, url: &str, target: ShareTarget) -> u64 {
        self.counts
            .get(url)
            .and_then(|m| m.get(&target))
            .copied()
            .unwrap_or(0)
    }

    /// Get breakdown per target for a URL.
    pub fn breakdown(&self, url: &str) -> HashMap<ShareTarget, u64> {
        self.counts.get(url).cloned().unwrap_or_default()
    }
}

// ── ShareSheet ──────────────────────────────────────────────────

/// Model for a share sheet UI: lists available targets.
#[derive(Debug, Clone)]
pub struct ShareSheet {
    pub available_targets: Vec<ShareTarget>,
}

impl ShareSheet {
    /// Default share sheet with all targets.
    pub fn new() -> Self {
        Self {
            available_targets: ShareTarget::ALL.to_vec(),
        }
    }

    /// Create with specific targets.
    pub fn with_targets(targets: &[ShareTarget]) -> Self {
        Self {
            available_targets: targets.to_vec(),
        }
    }

    /// Generate share URLs for all available targets.
    pub fn urls_for(&self, data: &ShareData) -> Vec<(ShareTarget, Option<String>)> {
        self.available_targets
            .iter()
            .map(|t| (*t, data.share_url(*t)))
            .collect()
    }
}

impl Default for ShareSheet {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twitter_share_url() {
        let data = ShareData::new("Title", "Check this out", "https://example.com");
        let url = data.share_url(ShareTarget::Twitter).unwrap();
        assert!(url.starts_with("https://twitter.com/intent/tweet"));
        assert!(url.contains("Check%20this%20out"));
        assert!(url.contains("https%3A%2F%2Fexample.com"));
    }

    #[test]
    fn test_facebook_share_url() {
        let data = ShareData::new("Title", "Text", "https://example.com");
        let url = data.share_url(ShareTarget::Facebook).unwrap();
        assert!(url.starts_with("https://www.facebook.com/sharer"));
    }

    #[test]
    fn test_linkedin_share_url() {
        let data = ShareData::new("Title", "Text", "https://example.com");
        let url = data.share_url(ShareTarget::LinkedIn).unwrap();
        assert!(url.contains("linkedin.com"));
    }

    #[test]
    fn test_reddit_share_url() {
        let data = ShareData::new("My Post", "Text", "https://example.com");
        let url = data.share_url(ShareTarget::Reddit).unwrap();
        assert!(url.contains("reddit.com/submit"));
        assert!(url.contains("My%20Post"));
    }

    #[test]
    fn test_email_share_url() {
        let data = ShareData::new("Subject", "Body text", "https://example.com");
        let url = data.share_url(ShareTarget::Email).unwrap();
        assert!(url.starts_with("mailto:"));
        assert!(url.contains("subject=Subject"));
    }

    #[test]
    fn test_whatsapp_share_url() {
        let data = ShareData::new("Title", "Check this", "https://example.com");
        let url = data.share_url(ShareTarget::WhatsApp).unwrap();
        assert!(url.contains("whatsapp.com"));
    }

    #[test]
    fn test_telegram_share_url() {
        let data = ShareData::new("Title", "Text", "https://example.com");
        let url = data.share_url(ShareTarget::Telegram).unwrap();
        assert!(url.contains("t.me/share"));
    }

    #[test]
    fn test_copy_returns_none() {
        let data = ShareData::new("Title", "Text", "https://example.com");
        assert!(data.share_url(ShareTarget::Copy).is_none());
    }

    #[test]
    fn test_short_url_hash_deterministic() {
        let h1 = short_url_hash("https://example.com/page");
        let h2 = short_url_hash("https://example.com/page");
        assert_eq!(h1, h2);
        assert!(!h1.is_empty());
        assert!(h1.len() <= 8);
    }

    #[test]
    fn test_short_url_hash_different() {
        let h1 = short_url_hash("https://example.com/a");
        let h2 = short_url_hash("https://example.com/b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_og_metadata_html() {
        let og = OgMetadata::new("My Page", "A great page")
            .with_image("https://example.com/img.png")
            .with_url("https://example.com");
        let html = og.to_html();
        assert!(html.contains("og:title"));
        assert!(html.contains("My Page"));
        assert!(html.contains("og:description"));
        assert!(html.contains("og:image"));
        assert!(html.contains("og:url"));
    }

    #[test]
    fn test_og_from_share_data() {
        let data = ShareData::new("Title", "Description", "https://example.com")
            .with_image("https://example.com/img.jpg");
        let og = OgMetadata::from_share_data(&data);
        assert_eq!(og.title, "Title");
        assert_eq!(og.description, "Description");
        assert_eq!(og.image.as_deref(), Some("https://example.com/img.jpg"));
    }

    #[test]
    fn test_share_count_tracker() {
        let mut tracker = ShareCountTracker::new();
        tracker.record("https://example.com", ShareTarget::Twitter);
        tracker.record("https://example.com", ShareTarget::Twitter);
        tracker.record("https://example.com", ShareTarget::Facebook);

        assert_eq!(tracker.total_for_url("https://example.com"), 3);
        assert_eq!(
            tracker.count_for("https://example.com", ShareTarget::Twitter),
            2
        );
        assert_eq!(
            tracker.count_for("https://example.com", ShareTarget::Facebook),
            1
        );
        assert_eq!(tracker.total_for_url("https://other.com"), 0);
    }

    #[test]
    fn test_share_sheet() {
        let sheet = ShareSheet::new();
        assert_eq!(sheet.available_targets.len(), ShareTarget::ALL.len());

        let custom = ShareSheet::with_targets(&[ShareTarget::Twitter, ShareTarget::Email]);
        assert_eq!(custom.available_targets.len(), 2);
    }

    #[test]
    fn test_share_sheet_urls() {
        let sheet = ShareSheet::with_targets(&[ShareTarget::Twitter, ShareTarget::Copy]);
        let data = ShareData::new("T", "X", "https://example.com");
        let urls = sheet.urls_for(&data);
        assert_eq!(urls.len(), 2);
        assert!(urls[0].1.is_some()); // Twitter has URL
        assert!(urls[1].1.is_none()); // Copy has None
    }

    #[test]
    fn test_url_encode_special_chars() {
        let encoded = url_encode("hello world & more");
        assert_eq!(encoded, "hello%20world%20%26%20more");
    }

    #[test]
    fn test_html_escape() {
        let escaped = html_escape("A & B <C> \"D\"");
        assert_eq!(escaped, "A &amp; B &lt;C&gt; &quot;D&quot;");
    }
}
