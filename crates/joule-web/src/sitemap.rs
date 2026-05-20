//! Sitemap/robots.txt — sitemap XML generation, sitemap index,
//! lastmod/changefreq/priority, robots.txt parser/builder, crawl-delay.
//!
//! Pure-Rust replacement for sitemap-generator, robots-txt-parser, etc.

use std::fmt;

// ── Change Frequency ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeFreq { Always, Hourly, Daily, Weekly, Monthly, Yearly, Never }

impl ChangeFreq {
    pub fn as_str(&self) -> &str {
        match self {
            ChangeFreq::Always => "always", ChangeFreq::Hourly => "hourly",
            ChangeFreq::Daily => "daily", ChangeFreq::Weekly => "weekly",
            ChangeFreq::Monthly => "monthly", ChangeFreq::Yearly => "yearly",
            ChangeFreq::Never => "never",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "always" => Some(ChangeFreq::Always),
            "hourly" => Some(ChangeFreq::Hourly),
            "daily" => Some(ChangeFreq::Daily),
            "weekly" => Some(ChangeFreq::Weekly),
            "monthly" => Some(ChangeFreq::Monthly),
            "yearly" => Some(ChangeFreq::Yearly),
            "never" => Some(ChangeFreq::Never),
            _ => None,
        }
    }
}

impl fmt::Display for ChangeFreq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Sitemap URL Entry ───────────────────────────────────────────

/// A single URL entry in a sitemap.
#[derive(Debug, Clone)]
pub struct SitemapUrl {
    pub loc: String,
    pub lastmod: Option<String>,
    pub changefreq: Option<ChangeFreq>,
    pub priority: Option<f64>,
}

impl SitemapUrl {
    pub fn new(loc: &str) -> Self {
        Self { loc: loc.to_string(), lastmod: None, changefreq: None, priority: None }
    }

    pub fn lastmod(mut self, date: &str) -> Self { self.lastmod = Some(date.to_string()); self }
    pub fn changefreq(mut self, freq: ChangeFreq) -> Self { self.changefreq = Some(freq); self }
    pub fn priority(mut self, p: f64) -> Self { self.priority = Some(p.max(0.0).min(1.0)); self }
}

// ── Sitemap Builder ─────────────────────────────────────────────

/// Builder for XML sitemaps (sitemaps.org protocol).
#[derive(Debug, Clone)]
pub struct SitemapBuilder {
    urls: Vec<SitemapUrl>,
}

impl SitemapBuilder {
    pub fn new() -> Self { Self { urls: Vec::new() } }

    pub fn add(&mut self, url: SitemapUrl) -> &mut Self {
        self.urls.push(url);
        self
    }

    pub fn url_count(&self) -> usize { self.urls.len() }

    /// Build the sitemap XML string.
    pub fn build(&self) -> String {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n");
        for url in &self.urls {
            xml.push_str("  <url>\n");
            xml.push_str(&format!("    <loc>{}</loc>\n", xml_escape(&url.loc)));
            if let Some(ref lm) = url.lastmod {
                xml.push_str(&format!("    <lastmod>{lm}</lastmod>\n"));
            }
            if let Some(cf) = url.changefreq {
                xml.push_str(&format!("    <changefreq>{}</changefreq>\n", cf.as_str()));
            }
            if let Some(p) = url.priority {
                xml.push_str(&format!("    <priority>{:.1}</priority>\n", p));
            }
            xml.push_str("  </url>\n");
        }
        xml.push_str("</urlset>\n");
        xml
    }

    /// Split into multiple sitemaps of at most `max_urls` each.
    pub fn split(&self, max_urls: usize) -> Vec<SitemapBuilder> {
        let mut parts = Vec::new();
        for chunk in self.urls.chunks(max_urls) {
            let mut builder = SitemapBuilder::new();
            for url in chunk {
                builder.add(url.clone());
            }
            parts.push(builder);
        }
        if parts.is_empty() {
            parts.push(SitemapBuilder::new());
        }
        parts
    }
}

impl Default for SitemapBuilder {
    fn default() -> Self { Self::new() }
}

// ── Sitemap Index ───────────────────────────────────────────────

/// An entry in a sitemap index.
#[derive(Debug, Clone)]
pub struct SitemapIndexEntry {
    pub loc: String,
    pub lastmod: Option<String>,
}

impl SitemapIndexEntry {
    pub fn new(loc: &str) -> Self {
        Self { loc: loc.to_string(), lastmod: None }
    }

    pub fn lastmod(mut self, date: &str) -> Self {
        self.lastmod = Some(date.to_string());
        self
    }
}

/// Builder for sitemap index files.
#[derive(Debug, Clone)]
pub struct SitemapIndexBuilder {
    entries: Vec<SitemapIndexEntry>,
}

impl SitemapIndexBuilder {
    pub fn new() -> Self { Self { entries: Vec::new() } }

    pub fn add(&mut self, entry: SitemapIndexEntry) -> &mut Self {
        self.entries.push(entry);
        self
    }

    pub fn build(&self) -> String {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<sitemapindex xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n");
        for entry in &self.entries {
            xml.push_str("  <sitemap>\n");
            xml.push_str(&format!("    <loc>{}</loc>\n", xml_escape(&entry.loc)));
            if let Some(ref lm) = entry.lastmod {
                xml.push_str(&format!("    <lastmod>{lm}</lastmod>\n"));
            }
            xml.push_str("  </sitemap>\n");
        }
        xml.push_str("</sitemapindex>\n");
        xml
    }
}

impl Default for SitemapIndexBuilder {
    fn default() -> Self { Self::new() }
}

// ── Robots.txt ──────────────────────────────────────────────────

/// A rule in robots.txt.
#[derive(Debug, Clone, PartialEq)]
pub enum RobotsRule {
    Allow(String),
    Disallow(String),
}

/// A parsed user-agent group in robots.txt.
#[derive(Debug, Clone)]
pub struct RobotsGroup {
    pub user_agents: Vec<String>,
    pub rules: Vec<RobotsRule>,
    pub crawl_delay: Option<f64>,
}

impl RobotsGroup {
    pub fn new(user_agent: &str) -> Self {
        Self {
            user_agents: vec![user_agent.to_string()],
            rules: Vec::new(),
            crawl_delay: None,
        }
    }

    pub fn add_user_agent(&mut self, ua: &str) -> &mut Self {
        self.user_agents.push(ua.to_string());
        self
    }

    pub fn allow(&mut self, path: &str) -> &mut Self {
        self.rules.push(RobotsRule::Allow(path.to_string()));
        self
    }

    pub fn disallow(&mut self, path: &str) -> &mut Self {
        self.rules.push(RobotsRule::Disallow(path.to_string()));
        self
    }

    pub fn crawl_delay(&mut self, seconds: f64) -> &mut Self {
        self.crawl_delay = Some(seconds);
        self
    }

    /// Check if a path is allowed for this group.
    pub fn is_allowed(&self, path: &str) -> bool {
        let mut best_match_len = 0;
        let mut allowed = true;
        for rule in &self.rules {
            let (pattern, is_allow) = match rule {
                RobotsRule::Allow(p) => (p.as_str(), true),
                RobotsRule::Disallow(p) => (p.as_str(), false),
            };
            if pattern.is_empty() { continue; }
            if path_matches(path, pattern) && pattern.len() >= best_match_len {
                best_match_len = pattern.len();
                allowed = is_allow;
            }
        }
        allowed
    }
}

fn path_matches(path: &str, pattern: &str) -> bool {
    if pattern.ends_with('*') {
        path.starts_with(&pattern[..pattern.len() - 1])
    } else if pattern.ends_with('$') {
        path == &pattern[..pattern.len() - 1]
    } else {
        path.starts_with(pattern)
    }
}

/// A parsed robots.txt file.
#[derive(Debug, Clone)]
pub struct RobotsTxt {
    pub groups: Vec<RobotsGroup>,
    pub sitemaps: Vec<String>,
}

impl RobotsTxt {
    /// Parse a robots.txt string.
    pub fn parse(input: &str) -> Self {
        let mut groups: Vec<RobotsGroup> = Vec::new();
        let mut sitemaps = Vec::new();
        let mut current_group: Option<RobotsGroup> = None;

        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }

            let (directive, value) = match line.split_once(':') {
                Some((d, v)) => (d.trim().to_lowercase(), v.trim().to_string()),
                None => continue,
            };

            match directive.as_str() {
                "user-agent" => {
                    if let Some(group) = current_group.take() {
                        if !group.rules.is_empty() || group.crawl_delay.is_some() {
                            groups.push(group);
                        }
                    }
                    current_group = Some(RobotsGroup::new(&value));
                }
                "allow" => {
                    if let Some(ref mut group) = current_group {
                        group.allow(&value);
                    }
                }
                "disallow" => {
                    if let Some(ref mut group) = current_group {
                        group.disallow(&value);
                    }
                }
                "crawl-delay" => {
                    if let Some(ref mut group) = current_group {
                        if let Ok(delay) = value.parse::<f64>() {
                            group.crawl_delay(delay);
                        }
                    }
                }
                "sitemap" => {
                    sitemaps.push(value);
                }
                _ => {}
            }
        }

        if let Some(group) = current_group {
            groups.push(group);
        }

        RobotsTxt { groups, sitemaps }
    }

    /// Check if a path is allowed for a user-agent.
    pub fn is_allowed(&self, user_agent: &str, path: &str) -> bool {
        let ua_lower = user_agent.to_lowercase();
        // Find most specific matching group
        let mut best_group: Option<&RobotsGroup> = None;
        let mut best_specificity = 0;

        for group in &self.groups {
            for ga in &group.user_agents {
                let ga_lower = ga.to_lowercase();
                if ga_lower == "*" && best_specificity == 0 {
                    best_group = Some(group);
                    best_specificity = 1;
                } else if ua_lower.contains(&ga_lower) && ga_lower.len() >= best_specificity {
                    best_group = Some(group);
                    best_specificity = ga_lower.len();
                }
            }
        }

        match best_group {
            Some(group) => group.is_allowed(path),
            None => true, // No matching group means allowed
        }
    }

    /// Get crawl delay for a user-agent.
    pub fn crawl_delay(&self, user_agent: &str) -> Option<f64> {
        let ua_lower = user_agent.to_lowercase();
        for group in &self.groups {
            for ga in &group.user_agents {
                if ga.to_lowercase() == ua_lower || ga == "*" {
                    return group.crawl_delay;
                }
            }
        }
        None
    }
}

/// Build a robots.txt string from groups and sitemaps.
pub fn build_robots_txt(groups: &[RobotsGroup], sitemaps: &[&str]) -> String {
    let mut out = String::new();
    for (i, group) in groups.iter().enumerate() {
        if i > 0 { out.push('\n'); }
        for ua in &group.user_agents {
            out.push_str(&format!("User-agent: {ua}\n"));
        }
        for rule in &group.rules {
            match rule {
                RobotsRule::Allow(p) => out.push_str(&format!("Allow: {p}\n")),
                RobotsRule::Disallow(p) => out.push_str(&format!("Disallow: {p}\n")),
            }
        }
        if let Some(delay) = group.crawl_delay {
            if delay == delay.trunc() {
                out.push_str(&format!("Crawl-delay: {}\n", delay as i64));
            } else {
                out.push_str(&format!("Crawl-delay: {delay}\n"));
            }
        }
    }
    if !sitemaps.is_empty() {
        out.push('\n');
        for sm in sitemaps {
            out.push_str(&format!("Sitemap: {sm}\n"));
        }
    }
    out
}

// ── Helpers ─────────────────────────────────────────────────────

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sitemap_basic() {
        let mut builder = SitemapBuilder::new();
        builder.add(SitemapUrl::new("https://example.com/"));
        let xml = builder.build();
        assert!(xml.contains("<?xml version=\"1.0\""));
        assert!(xml.contains("<urlset"));
        assert!(xml.contains("https://example.com/"));
        assert!(xml.contains("</urlset>"));
    }

    #[test]
    fn sitemap_with_metadata() {
        let mut builder = SitemapBuilder::new();
        builder.add(
            SitemapUrl::new("https://example.com/page")
                .lastmod("2026-03-09")
                .changefreq(ChangeFreq::Weekly)
                .priority(0.8)
        );
        let xml = builder.build();
        assert!(xml.contains("<lastmod>2026-03-09</lastmod>"));
        assert!(xml.contains("<changefreq>weekly</changefreq>"));
        assert!(xml.contains("<priority>0.8</priority>"));
    }

    #[test]
    fn sitemap_priority_clamping() {
        let url = SitemapUrl::new("https://example.com/").priority(1.5);
        assert!((url.priority.unwrap() - 1.0).abs() < f64::EPSILON);
        let url2 = SitemapUrl::new("https://example.com/").priority(-0.5);
        assert!((url2.priority.unwrap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sitemap_xml_escaping() {
        let mut builder = SitemapBuilder::new();
        builder.add(SitemapUrl::new("https://example.com/?a=1&b=2"));
        let xml = builder.build();
        assert!(xml.contains("&amp;"));
    }

    #[test]
    fn sitemap_url_count() {
        let mut builder = SitemapBuilder::new();
        builder.add(SitemapUrl::new("https://a.com"));
        builder.add(SitemapUrl::new("https://b.com"));
        assert_eq!(builder.url_count(), 2);
    }

    #[test]
    fn sitemap_split() {
        let mut builder = SitemapBuilder::new();
        for i in 0..5 {
            builder.add(SitemapUrl::new(&format!("https://example.com/{i}")));
        }
        let parts = builder.split(2);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].url_count(), 2);
        assert_eq!(parts[2].url_count(), 1);
    }

    #[test]
    fn sitemap_index_basic() {
        let mut index = SitemapIndexBuilder::new();
        index.add(SitemapIndexEntry::new("https://example.com/sitemap1.xml").lastmod("2026-03-09"));
        index.add(SitemapIndexEntry::new("https://example.com/sitemap2.xml"));
        let xml = index.build();
        assert!(xml.contains("<sitemapindex"));
        assert!(xml.contains("sitemap1.xml"));
        assert!(xml.contains("sitemap2.xml"));
        assert!(xml.contains("<lastmod>2026-03-09</lastmod>"));
    }

    #[test]
    fn robots_txt_parse_basic() {
        let input = "User-agent: *\nDisallow: /admin\nAllow: /admin/login\n";
        let robots = RobotsTxt::parse(input);
        assert_eq!(robots.groups.len(), 1);
        assert_eq!(robots.groups[0].user_agents[0], "*");
        assert_eq!(robots.groups[0].rules.len(), 2);
    }

    #[test]
    fn robots_txt_parse_sitemaps() {
        let input = "User-agent: *\nDisallow:\n\nSitemap: https://example.com/sitemap.xml\n";
        let robots = RobotsTxt::parse(input);
        assert_eq!(robots.sitemaps.len(), 1);
        assert_eq!(robots.sitemaps[0], "https://example.com/sitemap.xml");
    }

    #[test]
    fn robots_txt_parse_crawl_delay() {
        let input = "User-agent: Googlebot\nCrawl-delay: 10\nDisallow: /private\n";
        let robots = RobotsTxt::parse(input);
        assert_eq!(robots.groups[0].crawl_delay, Some(10.0));
    }

    #[test]
    fn robots_is_allowed() {
        let input = "User-agent: *\nDisallow: /admin\nAllow: /admin/public\n";
        let robots = RobotsTxt::parse(input);
        assert!(robots.is_allowed("Googlebot", "/"));
        assert!(!robots.is_allowed("Googlebot", "/admin"));
        assert!(robots.is_allowed("Googlebot", "/admin/public"));
    }

    #[test]
    fn robots_no_matching_group() {
        let robots = RobotsTxt::parse("");
        assert!(robots.is_allowed("AnyBot", "/anything"));
    }

    #[test]
    fn robots_wildcard_pattern() {
        let mut group = RobotsGroup::new("*");
        group.disallow("/private*");
        assert!(!group.is_allowed("/private/page"));
        assert!(group.is_allowed("/public/page"));
    }

    #[test]
    fn robots_exact_match_pattern() {
        let mut group = RobotsGroup::new("*");
        group.disallow("/exact$");
        assert!(!group.is_allowed("/exact"));
        assert!(group.is_allowed("/exact/more"));
    }

    #[test]
    fn robots_crawl_delay_lookup() {
        let input = "User-agent: Googlebot\nCrawl-delay: 5\nDisallow:\n";
        let robots = RobotsTxt::parse(input);
        assert_eq!(robots.crawl_delay("Googlebot"), Some(5.0));
        assert!(robots.crawl_delay("OtherBot").is_none());
    }

    #[test]
    fn build_robots_txt_basic() {
        let mut group = RobotsGroup::new("*");
        group.disallow("/private");
        group.allow("/private/public");
        group.crawl_delay(10.0);
        let txt = build_robots_txt(&[group], &["https://example.com/sitemap.xml"]);
        assert!(txt.contains("User-agent: *"));
        assert!(txt.contains("Disallow: /private"));
        assert!(txt.contains("Allow: /private/public"));
        assert!(txt.contains("Crawl-delay: 10"));
        assert!(txt.contains("Sitemap: https://example.com/sitemap.xml"));
    }

    #[test]
    fn build_robots_txt_multiple_groups() {
        let mut g1 = RobotsGroup::new("Googlebot");
        g1.allow("/");
        let mut g2 = RobotsGroup::new("BadBot");
        g2.disallow("/");
        let txt = build_robots_txt(&[g1, g2], &[]);
        assert!(txt.contains("User-agent: Googlebot"));
        assert!(txt.contains("User-agent: BadBot"));
    }

    #[test]
    fn changefreq_parse() {
        assert_eq!(ChangeFreq::parse("daily"), Some(ChangeFreq::Daily));
        assert_eq!(ChangeFreq::parse("WEEKLY"), Some(ChangeFreq::Weekly));
        assert_eq!(ChangeFreq::parse("invalid"), None);
    }

    #[test]
    fn changefreq_display() {
        assert_eq!(format!("{}", ChangeFreq::Monthly), "monthly");
    }

    #[test]
    fn empty_sitemap() {
        let builder = SitemapBuilder::new();
        let xml = builder.build();
        assert!(xml.contains("<urlset"));
        assert!(xml.contains("</urlset>"));
    }

    #[test]
    fn empty_sitemap_split() {
        let builder = SitemapBuilder::new();
        let parts = builder.split(50000);
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn robots_parse_comments() {
        let input = "# This is a comment\nUser-agent: *\n# Another comment\nDisallow: /secret\n";
        let robots = RobotsTxt::parse(input);
        assert_eq!(robots.groups.len(), 1);
        assert_eq!(robots.groups[0].rules.len(), 1);
    }

    #[test]
    fn robots_multiple_user_agents() {
        let mut group = RobotsGroup::new("Bot1");
        group.add_user_agent("Bot2");
        assert_eq!(group.user_agents.len(), 2);
    }

    #[test]
    fn sitemap_index_entry_builder() {
        let entry = SitemapIndexEntry::new("https://example.com/sm.xml")
            .lastmod("2026-01-01");
        assert_eq!(entry.loc, "https://example.com/sm.xml");
        assert_eq!(entry.lastmod.as_deref(), Some("2026-01-01"));
    }

    #[test]
    fn changefreq_all_variants() {
        let variants = [
            (ChangeFreq::Always, "always"),
            (ChangeFreq::Hourly, "hourly"),
            (ChangeFreq::Daily, "daily"),
            (ChangeFreq::Weekly, "weekly"),
            (ChangeFreq::Monthly, "monthly"),
            (ChangeFreq::Yearly, "yearly"),
            (ChangeFreq::Never, "never"),
        ];
        for (cf, expected) in &variants {
            assert_eq!(cf.as_str(), *expected);
        }
    }
}
