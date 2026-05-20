//! Progressive Web App manifest builder and service worker configuration.
//!
//! Replaces Workbox and next-pwa with pure-Rust config generation.
//! No browser APIs — just JSON manifests and JavaScript source generation.

use serde::{Deserialize, Serialize};

// ── Display & Orientation ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DisplayMode {
    Fullscreen,
    Standalone,
    MinimalUi,
    Browser,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Orientation {
    Any,
    Natural,
    Landscape,
    Portrait,
}

// ── Manifest Sub-types ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestIcon {
    pub src: String,
    pub sizes: String,
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shortcut {
    pub name: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub icons: Vec<ManifestIcon>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Screenshot {
    pub src: String,
    pub sizes: String,
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

// ── Web App Manifest ──

/// W3C Web App Manifest builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebAppManifest {
    pub name: String,
    pub short_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub start_url: String,
    pub display: DisplayMode,
    pub orientation: Orientation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_color: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub icons: Vec<ManifestIcon>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub categories: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub shortcuts: Vec<Shortcut>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub screenshots: Vec<Screenshot>,
}

impl WebAppManifest {
    pub fn new(name: &str, short_name: &str) -> Self {
        Self {
            name: name.to_string(),
            short_name: short_name.to_string(),
            description: None,
            start_url: "/".to_string(),
            display: DisplayMode::Standalone,
            orientation: Orientation::Any,
            theme_color: None,
            background_color: None,
            icons: Vec::new(),
            categories: Vec::new(),
            lang: None,
            dir: None,
            scope: None,
            id: None,
            shortcuts: Vec::new(),
            screenshots: Vec::new(),
        }
    }

    pub fn start_url(mut self, url: &str) -> Self {
        self.start_url = url.to_string();
        self
    }

    pub fn display(mut self, mode: DisplayMode) -> Self {
        self.display = mode;
        self
    }

    pub fn orientation(mut self, o: Orientation) -> Self {
        self.orientation = o;
        self
    }

    pub fn theme_color(mut self, c: &str) -> Self {
        self.theme_color = Some(c.to_string());
        self
    }

    pub fn background_color(mut self, c: &str) -> Self {
        self.background_color = Some(c.to_string());
        self
    }

    pub fn icon(mut self, src: &str, sizes: &str, type_: &str) -> Self {
        self.icons.push(ManifestIcon {
            src: src.to_string(),
            sizes: sizes.to_string(),
            type_: type_.to_string(),
            purpose: None,
        });
        self
    }

    pub fn icon_with_purpose(
        mut self,
        src: &str,
        sizes: &str,
        type_: &str,
        purpose: &str,
    ) -> Self {
        self.icons.push(ManifestIcon {
            src: src.to_string(),
            sizes: sizes.to_string(),
            type_: type_.to_string(),
            purpose: Some(purpose.to_string()),
        });
        self
    }

    pub fn shortcut(mut self, name: &str, url: &str) -> Self {
        self.shortcuts.push(Shortcut {
            name: name.to_string(),
            url: url.to_string(),
            description: None,
            icons: Vec::new(),
        });
        self
    }

    pub fn screenshot(mut self, src: &str, sizes: &str, type_: &str) -> Self {
        self.screenshots.push(Screenshot {
            src: src.to_string(),
            sizes: sizes.to_string(),
            type_: type_.to_string(),
            label: None,
        });
        self
    }

    pub fn category(mut self, cat: &str) -> Self {
        self.categories.push(cat.to_string());
        self
    }

    pub fn description(mut self, d: &str) -> Self {
        self.description = Some(d.to_string());
        self
    }

    pub fn lang(mut self, l: &str) -> Self {
        self.lang = Some(l.to_string());
        self
    }

    pub fn scope(mut self, s: &str) -> Self {
        self.scope = Some(s.to_string());
        self
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("manifest serialization should not fail")
    }

    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(self).expect("manifest serialization should not fail")
    }
}

// ── Caching Strategy ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CachingStrategy {
    CacheFirst,
    NetworkFirst,
    StaleWhileRevalidate,
    NetworkOnly,
    CacheOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStrategy {
    pub pattern: String,
    pub strategy: CachingStrategy,
    pub max_entries: Option<usize>,
    pub max_age_seconds: Option<u64>,
}

// ── Service Worker Config ──

/// Configuration for generating a service worker script.
#[derive(Debug, Clone)]
pub struct ServiceWorkerConfig {
    pub cache_name: String,
    pub version: String,
    pub precache: Vec<String>,
    pub runtime_cache: Vec<CacheStrategy>,
    pub offline_fallback: Option<String>,
    pub skip_waiting: bool,
    pub navigation_preload: bool,
}

impl ServiceWorkerConfig {
    pub fn new(cache_name: &str) -> Self {
        Self {
            cache_name: cache_name.to_string(),
            version: "1".to_string(),
            precache: Vec::new(),
            runtime_cache: Vec::new(),
            offline_fallback: None,
            skip_waiting: false,
            navigation_preload: false,
        }
    }

    pub fn precache_url(&mut self, url: &str) -> &mut Self {
        self.precache.push(url.to_string());
        self
    }

    pub fn runtime_cache(
        &mut self,
        pattern: &str,
        strategy: CachingStrategy,
    ) -> &mut Self {
        self.runtime_cache.push(CacheStrategy {
            pattern: pattern.to_string(),
            strategy,
            max_entries: None,
            max_age_seconds: None,
        });
        self
    }

    pub fn offline_fallback(&mut self, url: &str) -> &mut Self {
        self.offline_fallback = Some(url.to_string());
        self
    }

    pub fn precache_urls(&self) -> &[String] {
        &self.precache
    }

    /// Match a URL against runtime cache patterns, returning the strategy.
    /// Falls back to `NetworkFirst` if no pattern matches.
    pub fn cache_strategy_for(&self, url: &str) -> CachingStrategy {
        for rule in &self.runtime_cache {
            if url_matches_pattern(url, &rule.pattern) {
                return rule.strategy.clone();
            }
        }
        CachingStrategy::NetworkFirst
    }

    /// Generate a complete service worker JavaScript source.
    pub fn generate_sw_js(&self) -> String {
        let versioned_cache = format!("{}-v{}", self.cache_name, self.version);
        let mut js = String::new();

        // Cache name constant
        js.push_str(&format!(
            "const CACHE_NAME = '{versioned_cache}';\n"
        ));

        // Precache list
        js.push_str("const PRECACHE_URLS = [\n");
        for url in &self.precache {
            js.push_str(&format!("  '{url}',\n"));
        }
        js.push_str("];\n\n");

        // Install event — precache
        js.push_str("self.addEventListener('install', (event) => {\n");
        if self.skip_waiting {
            js.push_str("  self.skipWaiting();\n");
        }
        js.push_str("  event.waitUntil(\n");
        js.push_str("    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))\n");
        js.push_str("  );\n");
        js.push_str("});\n\n");

        // Activate event — cleanup old caches
        js.push_str("self.addEventListener('activate', (event) => {\n");
        js.push_str("  event.waitUntil(\n");
        js.push_str("    caches.keys().then((keys) => Promise.all(\n");
        js.push_str(
            "      keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))\n",
        );
        js.push_str("    ))\n");
        js.push_str("  );\n");
        js.push_str("});\n\n");

        // Fetch event
        js.push_str("self.addEventListener('fetch', (event) => {\n");
        js.push_str("  const url = event.request.url;\n");

        for rule in &self.runtime_cache {
            let condition = if rule.pattern.contains('*') {
                let prefix = rule.pattern.replace('*', "");
                format!("url.includes('{prefix}')")
            } else {
                format!("url.includes('{}')", rule.pattern)
            };
            let handler = match rule.strategy {
                CachingStrategy::CacheFirst => {
                    "    event.respondWith(\n      caches.match(event.request).then((r) => r || fetch(event.request))\n    );"
                }
                CachingStrategy::NetworkFirst => {
                    "    event.respondWith(\n      fetch(event.request).catch(() => caches.match(event.request))\n    );"
                }
                CachingStrategy::StaleWhileRevalidate => {
                    "    event.respondWith(\n      caches.match(event.request).then((r) => {\n        const f = fetch(event.request).then((resp) => {\n          caches.open(CACHE_NAME).then((c) => c.put(event.request, resp.clone()));\n          return resp;\n        });\n        return r || f;\n      })\n    );"
                }
                CachingStrategy::NetworkOnly => {
                    "    event.respondWith(fetch(event.request));"
                }
                CachingStrategy::CacheOnly => {
                    "    event.respondWith(caches.match(event.request));"
                }
            };
            js.push_str(&format!("  if ({condition}) {{\n{handler}\n    return;\n  }}\n"));
        }

        // Default: network-first with optional offline fallback
        if let Some(ref fallback) = self.offline_fallback {
            js.push_str(&format!(
                "  event.respondWith(\n    fetch(event.request).catch(() => caches.match('{fallback}'))\n  );\n"
            ));
        } else {
            js.push_str(
                "  event.respondWith(\n    fetch(event.request).catch(() => caches.match(event.request))\n  );\n",
            );
        }

        js.push_str("});\n");
        js
    }
}

/// Simple URL pattern matching: exact substring or prefix wildcard.
fn url_matches_pattern(url: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        url.contains(prefix)
    } else {
        url.contains(pattern)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_to_json_has_required_fields() {
        let m = WebAppManifest::new("My App", "App");
        let json = m.to_json();
        assert_eq!(json["name"], "My App");
        assert_eq!(json["short_name"], "App");
        assert_eq!(json["start_url"], "/");
        assert_eq!(json["display"], "standalone");
    }

    #[test]
    fn icon_added() {
        let m = WebAppManifest::new("App", "A")
            .icon("/icon.png", "192x192", "image/png");
        let json = m.to_json();
        assert_eq!(json["icons"][0]["src"], "/icon.png");
        assert_eq!(json["icons"][0]["sizes"], "192x192");
    }

    #[test]
    fn shortcut_added() {
        let m = WebAppManifest::new("App", "A").shortcut("Home", "/");
        let json = m.to_json();
        assert_eq!(json["shortcuts"][0]["name"], "Home");
        assert_eq!(json["shortcuts"][0]["url"], "/");
    }

    #[test]
    fn display_mode_serialized() {
        let m = WebAppManifest::new("App", "A").display(DisplayMode::Fullscreen);
        let json = m.to_json();
        assert_eq!(json["display"], "fullscreen");
    }

    #[test]
    fn sw_js_includes_install_event() {
        let mut sw = ServiceWorkerConfig::new("my-cache");
        sw.precache_url("/index.html");
        let js = sw.generate_sw_js();
        assert!(js.contains("addEventListener('install'"));
        assert!(js.contains("/index.html"));
    }

    #[test]
    fn precache_urls_listed() {
        let mut sw = ServiceWorkerConfig::new("c");
        sw.precache_url("/a.js");
        sw.precache_url("/b.css");
        assert_eq!(sw.precache_urls(), &["/a.js", "/b.css"]);
    }

    #[test]
    fn runtime_cache_pattern_matching() {
        let mut sw = ServiceWorkerConfig::new("c");
        sw.runtime_cache("/api/*", CachingStrategy::NetworkFirst);
        assert_eq!(
            sw.cache_strategy_for("/api/data"),
            CachingStrategy::NetworkFirst
        );
    }

    #[test]
    fn cache_first_strategy() {
        let mut sw = ServiceWorkerConfig::new("c");
        sw.runtime_cache("/static/", CachingStrategy::CacheFirst);
        assert_eq!(
            sw.cache_strategy_for("/static/app.js"),
            CachingStrategy::CacheFirst
        );
    }

    #[test]
    fn network_first_fallback() {
        let sw = ServiceWorkerConfig::new("c");
        assert_eq!(
            sw.cache_strategy_for("/unknown"),
            CachingStrategy::NetworkFirst
        );
    }

    #[test]
    fn offline_fallback_in_sw_js() {
        let mut sw = ServiceWorkerConfig::new("c");
        sw.offline_fallback("/offline.html");
        let js = sw.generate_sw_js();
        assert!(js.contains("/offline.html"));
    }

    #[test]
    fn version_in_cache_name() {
        let mut sw = ServiceWorkerConfig::new("app");
        sw.version = "42".to_string();
        let js = sw.generate_sw_js();
        assert!(js.contains("app-v42"));
    }

    #[test]
    fn manifest_pretty_printed_valid_json() {
        let m = WebAppManifest::new("Test", "T")
            .theme_color("#000")
            .icon("/i.png", "48x48", "image/png");
        let json_str = m.to_json_string();
        assert!(json_str.contains('\n'));
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["name"], "Test");
    }
}
