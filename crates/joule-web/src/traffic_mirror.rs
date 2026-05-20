// traffic_mirror.rs — Traffic mirroring for dark launches, A/B splits,
// and percentage-based routing via deterministic hashing.

use std::collections::HashMap;

/// FNV-1a 64-bit hash for deterministic, order-independent selection.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x00000100000001B3);
    }
    h
}

/// Mirror configuration: where mirrored traffic goes.
#[derive(Debug, Clone)]
pub struct MirrorTarget {
    pub name: String,
    pub endpoint: String,
    /// If true, the caller should await the mirror response; if false, fire-and-forget.
    pub synchronous: bool,
}

/// Percentage-based traffic selection.
/// Uses deterministic FNV-1a hash of the key so the same key always
/// maps to the same bucket (consistent across restarts).
#[derive(Debug, Clone)]
pub struct PercentageSelector {
    /// 0..=100
    percent: u8,
}

impl PercentageSelector {
    pub fn new(percent: u8) -> Self {
        Self {
            percent: percent.min(100),
        }
    }

    pub fn percent(&self) -> u8 {
        self.percent
    }

    /// Returns `true` if the given key falls within the selected percentage.
    pub fn should_mirror(&self, key: &str) -> bool {
        if self.percent == 0 {
            return false;
        }
        if self.percent >= 100 {
            return true;
        }
        let h = fnv1a_64(key.as_bytes());
        (h % 100) < self.percent as u64
    }
}

/// Header-based routing rule: match a header name/value pair.
#[derive(Debug, Clone)]
pub struct HeaderRule {
    pub header_name: String,
    /// If `None`, the rule matches when the header is present (any value).
    pub expected_value: Option<String>,
}

impl HeaderRule {
    pub fn present(name: &str) -> Self {
        Self {
            header_name: name.to_lowercase(),
            expected_value: None,
        }
    }

    pub fn equals(name: &str, value: &str) -> Self {
        Self {
            header_name: name.to_lowercase(),
            expected_value: Some(value.to_string()),
        }
    }

    pub fn matches(&self, headers: &HashMap<String, String>) -> bool {
        match headers.get(&self.header_name) {
            None => false,
            Some(v) => match &self.expected_value {
                None => true,
                Some(expected) => v == expected,
            },
        }
    }
}

/// A/B traffic splitting with named variants.
#[derive(Debug, Clone)]
pub struct AbSplit {
    /// Variant name -> weight (relative, not percentage).
    variants: Vec<(String, u32)>,
    total_weight: u32,
}

impl AbSplit {
    pub fn new() -> Self {
        Self {
            variants: Vec::new(),
            total_weight: 0,
        }
    }

    pub fn add_variant(&mut self, name: &str, weight: u32) {
        self.variants.push((name.to_string(), weight));
        self.total_weight += weight;
    }

    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }

    /// Deterministically assign a key to a variant.
    /// Returns `None` if no variants are defined.
    pub fn assign(&self, key: &str) -> Option<&str> {
        if self.total_weight == 0 || self.variants.is_empty() {
            return None;
        }
        let h = fnv1a_64(key.as_bytes());
        let bucket = (h % self.total_weight as u64) as u32;
        let mut cumulative = 0u32;
        for (name, weight) in &self.variants {
            cumulative += weight;
            if bucket < cumulative {
                return Some(name.as_str());
            }
        }
        // Fallback (shouldn't happen with correct math).
        Some(self.variants.last().unwrap().0.as_str())
    }
}

impl Default for AbSplit {
    fn default() -> Self {
        Self::new()
    }
}

/// Dark launch flags — feature gates for mirrored traffic.
#[derive(Debug, Clone, Default)]
pub struct DarkLaunchFlags {
    flags: HashMap<String, bool>,
}

impl DarkLaunchFlags {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, flag: &str, enabled: bool) {
        self.flags.insert(flag.to_string(), enabled);
    }

    pub fn is_enabled(&self, flag: &str) -> bool {
        self.flags.get(flag).copied().unwrap_or(false)
    }

    pub fn enabled_flags(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self
            .flags
            .iter()
            .filter(|(_, v)| **v)
            .map(|(k, _)| k.as_str())
            .collect();
        out.sort();
        out
    }

    pub fn flag_count(&self) -> usize {
        self.flags.len()
    }
}

/// Result of mirroring a single request.
#[derive(Debug, Clone)]
pub struct MirrorResult {
    pub target_name: String,
    pub original_status: u16,
    pub mirror_status: Option<u16>,
    pub latency_us: u64,
    pub matched: bool,
}

/// Tracks mirror results for reporting.
#[derive(Debug, Clone, Default)]
pub struct MirrorTracker {
    results: Vec<MirrorResult>,
}

impl MirrorTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, result: MirrorResult) {
        self.results.push(result);
    }

    pub fn total(&self) -> usize {
        self.results.len()
    }

    pub fn matched_count(&self) -> usize {
        self.results.iter().filter(|r| r.matched).count()
    }

    pub fn mismatched_count(&self) -> usize {
        self.results.iter().filter(|r| !r.matched).count()
    }

    pub fn average_latency_us(&self) -> u64 {
        if self.results.is_empty() {
            return 0;
        }
        let sum: u64 = self.results.iter().map(|r| r.latency_us).sum();
        sum / self.results.len() as u64
    }

    pub fn results_for_target(&self, name: &str) -> Vec<&MirrorResult> {
        self.results.iter().filter(|r| r.target_name == name).collect()
    }

    pub fn clear(&mut self) {
        self.results.clear();
    }

    pub fn mismatch_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        self.mismatched_count() as f64 / self.results.len() as f64
    }
}

/// Full mirror configuration combining target, selector, header rules,
/// A/B split, and dark launch flags.
#[derive(Debug, Clone)]
pub struct MirrorConfig {
    pub target: MirrorTarget,
    pub selector: PercentageSelector,
    pub header_rules: Vec<HeaderRule>,
    pub dark_flags: DarkLaunchFlags,
}

impl MirrorConfig {
    pub fn new(target: MirrorTarget, percent: u8) -> Self {
        Self {
            target,
            selector: PercentageSelector::new(percent),
            header_rules: Vec::new(),
            dark_flags: DarkLaunchFlags::new(),
        }
    }

    pub fn add_header_rule(&mut self, rule: HeaderRule) {
        self.header_rules.push(rule);
    }

    /// Check whether a request (identified by key + headers) should be mirrored.
    pub fn should_mirror(&self, key: &str, headers: &HashMap<String, String>) -> bool {
        // Percentage gate first.
        if !self.selector.should_mirror(key) {
            return false;
        }
        // All header rules must match (AND logic).
        for rule in &self.header_rules {
            if !rule.matches(headers) {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Deterministic hash helpers (exposed for testing)
// ---------------------------------------------------------------------------

pub fn hash_key(key: &str) -> u64 {
    fnv1a_64(key.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_deterministic() {
        let a = fnv1a_64(b"hello");
        let b = fnv1a_64(b"hello");
        assert_eq!(a, b);
        let c = fnv1a_64(b"world");
        assert_ne!(a, c);
    }

    #[test]
    fn test_percentage_selector_zero() {
        let sel = PercentageSelector::new(0);
        assert!(!sel.should_mirror("any-key"));
        assert_eq!(sel.percent(), 0);
    }

    #[test]
    fn test_percentage_selector_hundred() {
        let sel = PercentageSelector::new(100);
        assert!(sel.should_mirror("any-key"));
        assert!(sel.should_mirror("another-key"));
    }

    #[test]
    fn test_percentage_selector_clamp() {
        let sel = PercentageSelector::new(200);
        assert_eq!(sel.percent(), 100);
        assert!(sel.should_mirror("key"));
    }

    #[test]
    fn test_percentage_selector_deterministic() {
        let sel = PercentageSelector::new(50);
        let r1 = sel.should_mirror("user-42");
        let r2 = sel.should_mirror("user-42");
        assert_eq!(r1, r2, "same key must produce same result");
    }

    #[test]
    fn test_percentage_selector_distribution() {
        // With 50%, roughly half of 1000 keys should be selected.
        let sel = PercentageSelector::new(50);
        let count = (0..1000)
            .filter(|i| sel.should_mirror(&format!("key-{i}")))
            .count();
        // Allow generous range for hash distribution.
        assert!(count > 350 && count < 650, "got {count} out of 1000");
    }

    #[test]
    fn test_header_rule_present() {
        let rule = HeaderRule::present("X-Debug");
        let mut h = HashMap::new();
        h.insert("x-debug".to_string(), "anything".to_string());
        assert!(rule.matches(&h));

        let empty: HashMap<String, String> = HashMap::new();
        assert!(!rule.matches(&empty));
    }

    #[test]
    fn test_header_rule_equals() {
        let rule = HeaderRule::equals("X-Env", "staging");
        let mut h = HashMap::new();
        h.insert("x-env".to_string(), "staging".to_string());
        assert!(rule.matches(&h));

        h.insert("x-env".to_string(), "prod".to_string());
        assert!(!rule.matches(&h));
    }

    #[test]
    fn test_ab_split_empty() {
        let split = AbSplit::new();
        assert!(split.assign("key").is_none());
        assert_eq!(split.variant_count(), 0);
    }

    #[test]
    fn test_ab_split_single_variant() {
        let mut split = AbSplit::new();
        split.add_variant("control", 1);
        assert_eq!(split.assign("anything").unwrap(), "control");
    }

    #[test]
    fn test_ab_split_deterministic() {
        let mut split = AbSplit::new();
        split.add_variant("A", 50);
        split.add_variant("B", 50);
        let v1 = split.assign("user-123").unwrap().to_string();
        let v2 = split.assign("user-123").unwrap().to_string();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_ab_split_distribution() {
        let mut split = AbSplit::new();
        split.add_variant("A", 50);
        split.add_variant("B", 50);
        let mut a_count = 0usize;
        for i in 0..1000 {
            if split.assign(&format!("user-{i}")).unwrap() == "A" {
                a_count += 1;
            }
        }
        assert!(a_count > 350 && a_count < 650, "A got {a_count}/1000");
    }

    #[test]
    fn test_dark_launch_flags() {
        let mut flags = DarkLaunchFlags::new();
        assert!(!flags.is_enabled("new-search"));
        flags.set("new-search", true);
        flags.set("old-feature", false);
        assert!(flags.is_enabled("new-search"));
        assert!(!flags.is_enabled("old-feature"));
        assert_eq!(flags.flag_count(), 2);
        assert_eq!(flags.enabled_flags(), vec!["new-search"]);
    }

    #[test]
    fn test_mirror_tracker() {
        let mut tracker = MirrorTracker::new();
        tracker.record(MirrorResult {
            target_name: "shadow".into(),
            original_status: 200,
            mirror_status: Some(200),
            latency_us: 100,
            matched: true,
        });
        tracker.record(MirrorResult {
            target_name: "shadow".into(),
            original_status: 200,
            mirror_status: Some(500),
            latency_us: 200,
            matched: false,
        });
        assert_eq!(tracker.total(), 2);
        assert_eq!(tracker.matched_count(), 1);
        assert_eq!(tracker.mismatched_count(), 1);
        assert_eq!(tracker.average_latency_us(), 150);
        assert!((tracker.mismatch_rate() - 0.5).abs() < f64::EPSILON);
        assert_eq!(tracker.results_for_target("shadow").len(), 2);
        assert_eq!(tracker.results_for_target("other").len(), 0);
    }

    #[test]
    fn test_mirror_tracker_empty() {
        let tracker = MirrorTracker::new();
        assert_eq!(tracker.average_latency_us(), 0);
        assert!((tracker.mismatch_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mirror_tracker_clear() {
        let mut tracker = MirrorTracker::new();
        tracker.record(MirrorResult {
            target_name: "t".into(),
            original_status: 200,
            mirror_status: None,
            latency_us: 50,
            matched: true,
        });
        assert_eq!(tracker.total(), 1);
        tracker.clear();
        assert_eq!(tracker.total(), 0);
    }

    #[test]
    fn test_mirror_config_should_mirror() {
        let target = MirrorTarget {
            name: "shadow-v2".into(),
            endpoint: "http://shadow:8080".into(),
            synchronous: false,
        };
        let mut config = MirrorConfig::new(target, 100);
        config.add_header_rule(HeaderRule::equals("x-env", "staging"));

        let mut headers = HashMap::new();
        headers.insert("x-env".to_string(), "staging".to_string());
        assert!(config.should_mirror("req-1", &headers));

        headers.insert("x-env".to_string(), "prod".to_string());
        assert!(!config.should_mirror("req-1", &headers));
    }

    #[test]
    fn test_mirror_config_percent_gate() {
        let target = MirrorTarget {
            name: "t".into(),
            endpoint: "http://t".into(),
            synchronous: false,
        };
        let config = MirrorConfig::new(target, 0);
        let headers = HashMap::new();
        assert!(!config.should_mirror("any", &headers));
    }

    #[test]
    fn test_hash_key_exposed() {
        let h1 = hash_key("test");
        let h2 = hash_key("test");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_mirror_target_fields() {
        let t = MirrorTarget {
            name: "beta".into(),
            endpoint: "http://beta:9090".into(),
            synchronous: true,
        };
        assert_eq!(t.name, "beta");
        assert!(t.synchronous);
    }

    #[test]
    fn test_ab_split_three_variants() {
        let mut split = AbSplit::new();
        split.add_variant("A", 1);
        split.add_variant("B", 1);
        split.add_variant("C", 1);
        assert_eq!(split.variant_count(), 3);
        // Every key must land somewhere.
        for i in 0..100 {
            let v = split.assign(&format!("k{i}"));
            assert!(v.is_some());
            let name = v.unwrap();
            assert!(name == "A" || name == "B" || name == "C");
        }
    }
}
