//! DNS resolution abstraction — record types, TTL caching, SRV selection.
//!
//! Replaces `dns`, `c-ares`, and `trust-dns` resolver logic with pure Rust.
//! Record types (A, AAAA, CNAME, MX, TXT, SRV), TTL-based caching,
//! round-robin selection, SRV priority/weight selection, hosts file parsing.

use std::collections::HashMap;
use std::fmt;

// ── Record types ───────────────────────────────────────────────

/// DNS record type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecordType {
    A,
    Aaaa,
    Cname,
    Mx,
    Txt,
    Srv,
}

impl RecordType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::A => "A",
            Self::Aaaa => "AAAA",
            Self::Cname => "CNAME",
            Self::Mx => "MX",
            Self::Txt => "TXT",
            Self::Srv => "SRV",
        }
    }
}

impl fmt::Display for RecordType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A DNS record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DnsRecord {
    A { address: [u8; 4] },
    Aaaa { address: [u16; 8] },
    Cname { target: String },
    Mx { priority: u16, exchange: String },
    Txt { text: String },
    Srv { priority: u16, weight: u16, port: u16, target: String },
}

impl DnsRecord {
    pub fn record_type(&self) -> RecordType {
        match self {
            Self::A { .. } => RecordType::A,
            Self::Aaaa { .. } => RecordType::Aaaa,
            Self::Cname { .. } => RecordType::Cname,
            Self::Mx { .. } => RecordType::Mx,
            Self::Txt { .. } => RecordType::Txt,
            Self::Srv { .. } => RecordType::Srv,
        }
    }

    pub fn a(a: u8, b: u8, c: u8, d: u8) -> Self {
        Self::A { address: [a, b, c, d] }
    }

    pub fn aaaa(segments: [u16; 8]) -> Self {
        Self::Aaaa { address: segments }
    }

    pub fn cname(target: &str) -> Self {
        Self::Cname { target: target.to_string() }
    }

    pub fn mx(priority: u16, exchange: &str) -> Self {
        Self::Mx { priority, exchange: exchange.to_string() }
    }

    pub fn txt(text: &str) -> Self {
        Self::Txt { text: text.to_string() }
    }

    pub fn srv(priority: u16, weight: u16, port: u16, target: &str) -> Self {
        Self::Srv { priority, weight, port, target: target.to_string() }
    }

    /// Format an A record as dotted-quad.
    pub fn a_str(&self) -> Option<String> {
        if let Self::A { address } = self {
            Some(format!("{}.{}.{}.{}", address[0], address[1], address[2], address[3]))
        } else {
            None
        }
    }
}

// ── DNS cache ──────────────────────────────────────────────────

/// A cached DNS entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub records: Vec<DnsRecord>,
    pub ttl_seconds: u32,
    pub inserted_epoch_s: u64,
    /// Round-robin counter for selection.
    rr_idx: usize,
}

impl CacheEntry {
    pub fn new(records: Vec<DnsRecord>, ttl_seconds: u32, now_epoch_s: u64) -> Self {
        Self {
            records,
            ttl_seconds,
            inserted_epoch_s: now_epoch_s,
            rr_idx: 0,
        }
    }

    pub fn is_expired(&self, now_epoch_s: u64) -> bool {
        now_epoch_s > self.inserted_epoch_s + self.ttl_seconds as u64
    }

    /// Round-robin select the next record.
    pub fn next_record(&mut self) -> Option<&DnsRecord> {
        if self.records.is_empty() {
            return None;
        }
        let idx = self.rr_idx % self.records.len();
        self.rr_idx = self.rr_idx.wrapping_add(1);
        Some(&self.records[idx])
    }
}

/// DNS cache keyed by (name, record type).
#[derive(Debug, Default)]
pub struct DnsCache {
    entries: HashMap<(String, String), CacheEntry>,
}

impl DnsCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        name: &str,
        rtype: &RecordType,
        records: Vec<DnsRecord>,
        ttl: u32,
        now_epoch_s: u64,
    ) {
        let key = (name.to_ascii_lowercase(), rtype.as_str().to_string());
        self.entries.insert(key, CacheEntry::new(records, ttl, now_epoch_s));
    }

    pub fn lookup(
        &mut self,
        name: &str,
        rtype: &RecordType,
        now_epoch_s: u64,
    ) -> Option<&mut CacheEntry> {
        let key = (name.to_ascii_lowercase(), rtype.as_str().to_string());
        self.entries
            .get_mut(&key)
            .filter(|e| !e.is_expired(now_epoch_s))
    }

    pub fn evict_expired(&mut self, now_epoch_s: u64) {
        self.entries.retain(|_, e| !e.is_expired(now_epoch_s));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ── SRV selection ──────────────────────────────────────────────

/// Select a target from SRV records using RFC 2782 priority/weight algorithm.
///
/// 1. Group by priority (lowest first).
/// 2. Within a priority group, select weighted random using a deterministic seed.
pub fn select_srv(records: &[DnsRecord], seed: u64) -> Option<(String, u16)> {
    let mut srvs: Vec<(u16, u16, u16, &str)> = records
        .iter()
        .filter_map(|r| match r {
            DnsRecord::Srv { priority, weight, port, target } => {
                Some((*priority, *weight, *port, target.as_str()))
            }
            _ => None,
        })
        .collect();

    if srvs.is_empty() {
        return None;
    }

    // Sort by priority.
    srvs.sort_by_key(|(p, _, _, _)| *p);
    let min_priority = srvs[0].0;

    // Take the lowest priority group.
    let group: Vec<_> = srvs
        .iter()
        .filter(|(p, _, _, _)| *p == min_priority)
        .collect();

    if group.len() == 1 {
        return Some((group[0].3.to_string(), group[0].2));
    }

    // Weighted selection.
    let total_weight: u64 = group.iter().map(|(_, w, _, _)| *w as u64).sum();
    if total_weight == 0 {
        // All zero weight: pick based on seed.
        let idx = (seed as usize) % group.len();
        return Some((group[idx].3.to_string(), group[idx].2));
    }

    let target_weight = seed % total_weight;
    let mut acc = 0u64;
    for entry in &group {
        acc += entry.1 as u64;
        if target_weight < acc {
            return Some((entry.3.to_string(), entry.2));
        }
    }

    let last = group.last().unwrap();
    Some((last.3.to_string(), last.2))
}

/// Select from MX records (lowest priority preferred).
pub fn select_mx(records: &[DnsRecord]) -> Option<String> {
    records
        .iter()
        .filter_map(|r| match r {
            DnsRecord::Mx { priority, exchange } => Some((*priority, exchange.as_str())),
            _ => None,
        })
        .min_by_key(|(p, _)| *p)
        .map(|(_, e)| e.to_string())
}

// ── Hosts file parser ──────────────────────────────────────────

/// Parsed hosts file entries.
#[derive(Debug, Default)]
pub struct HostsFile {
    entries: HashMap<String, Vec<String>>,
}

impl HostsFile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse hosts file content (e.g. `/etc/hosts`).
    pub fn parse(content: &str) -> Self {
        let mut entries: HashMap<String, Vec<String>> = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            // Strip comments.
            let line = if let Some(idx) = line.find('#') {
                &line[..idx]
            } else {
                line
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let ip = parts[0];
            for &hostname in &parts[1..] {
                entries
                    .entry(hostname.to_ascii_lowercase())
                    .or_default()
                    .push(ip.to_string());
            }
        }

        Self { entries }
    }

    /// Look up a hostname in the hosts file.
    pub fn lookup(&self, hostname: &str) -> Option<&[String]> {
        self.entries
            .get(&hostname.to_ascii_lowercase())
            .map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Resolver ───────────────────────────────────────────────────

/// DNS resolver with cache and hosts file.
#[derive(Debug)]
pub struct Resolver {
    pub cache: DnsCache,
    pub hosts: HostsFile,
}

impl Resolver {
    pub fn new() -> Self {
        Self {
            cache: DnsCache::new(),
            hosts: HostsFile::new(),
        }
    }

    pub fn with_hosts(mut self, hosts: HostsFile) -> Self {
        self.hosts = hosts;
        self
    }

    /// Resolve A records: check hosts file first, then cache.
    pub fn resolve_a(&mut self, name: &str, now_epoch_s: u64) -> Vec<DnsRecord> {
        // Hosts file first.
        if let Some(ips) = self.hosts.lookup(name) {
            return ips
                .iter()
                .filter_map(|ip| parse_ipv4(ip))
                .map(|addr| DnsRecord::A { address: addr })
                .collect();
        }
        // Cache.
        if let Some(entry) = self.cache.lookup(name, &RecordType::A, now_epoch_s) {
            return entry.records.clone();
        }
        Vec::new()
    }

    /// Add records to cache.
    pub fn cache_records(
        &mut self,
        name: &str,
        rtype: RecordType,
        records: Vec<DnsRecord>,
        ttl: u32,
        now_epoch_s: u64,
    ) {
        self.cache.insert(name, &rtype, records, ttl, now_epoch_s);
    }
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_ipv4(s: &str) -> Option<[u8; 4]> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let a = parts[0].parse::<u8>().ok()?;
    let b = parts[1].parse::<u8>().ok()?;
    let c = parts[2].parse::<u8>().ok()?;
    let d = parts[3].parse::<u8>().ok()?;
    Some([a, b, c, d])
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_types() {
        let a = DnsRecord::a(192, 168, 1, 1);
        assert_eq!(a.record_type(), RecordType::A);
        assert_eq!(a.a_str(), Some("192.168.1.1".to_string()));

        let mx = DnsRecord::mx(10, "mail.example.com");
        assert_eq!(mx.record_type(), RecordType::Mx);
    }

    #[test]
    fn cache_insert_and_lookup() {
        let mut cache = DnsCache::new();
        cache.insert(
            "example.com",
            &RecordType::A,
            vec![DnsRecord::a(1, 2, 3, 4)],
            300,
            1000,
        );
        assert!(cache.lookup("example.com", &RecordType::A, 1100).is_some());
        assert!(cache.lookup("example.com", &RecordType::A, 2000).is_none());
    }

    #[test]
    fn cache_case_insensitive() {
        let mut cache = DnsCache::new();
        cache.insert(
            "Example.COM",
            &RecordType::A,
            vec![DnsRecord::a(1, 1, 1, 1)],
            60,
            0,
        );
        assert!(cache.lookup("example.com", &RecordType::A, 0).is_some());
    }

    #[test]
    fn cache_round_robin() {
        let mut cache = DnsCache::new();
        cache.insert(
            "x.com",
            &RecordType::A,
            vec![DnsRecord::a(1, 0, 0, 1), DnsRecord::a(1, 0, 0, 2)],
            600,
            0,
        );
        let entry = cache.lookup("x.com", &RecordType::A, 0).unwrap();
        let r1 = entry.next_record().unwrap().clone();
        let r2 = entry.next_record().unwrap().clone();
        assert_ne!(r1, r2);
    }

    #[test]
    fn cache_evict_expired() {
        let mut cache = DnsCache::new();
        cache.insert("a.com", &RecordType::A, vec![], 10, 100);
        cache.insert("b.com", &RecordType::A, vec![], 1000, 100);
        cache.evict_expired(200);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn srv_selection_by_priority() {
        let records = vec![
            DnsRecord::srv(20, 10, 8080, "backup.example.com"),
            DnsRecord::srv(10, 10, 80, "primary.example.com"),
        ];
        let (target, port) = select_srv(&records, 0).unwrap();
        assert_eq!(target, "primary.example.com");
        assert_eq!(port, 80);
    }

    #[test]
    fn srv_selection_weighted() {
        let records = vec![
            DnsRecord::srv(10, 70, 80, "heavy.example.com"),
            DnsRecord::srv(10, 30, 80, "light.example.com"),
        ];
        // seed=0 -> target_weight=0, should pick first (weight 70).
        let (target, _) = select_srv(&records, 0).unwrap();
        assert_eq!(target, "heavy.example.com");
        // seed=69 -> still first (0..70).
        let (target, _) = select_srv(&records, 69).unwrap();
        assert_eq!(target, "heavy.example.com");
        // seed=70 -> second.
        let (target, _) = select_srv(&records, 70).unwrap();
        assert_eq!(target, "light.example.com");
    }

    #[test]
    fn mx_selection() {
        let records = vec![
            DnsRecord::mx(20, "backup-mx.example.com"),
            DnsRecord::mx(10, "primary-mx.example.com"),
        ];
        assert_eq!(
            select_mx(&records),
            Some("primary-mx.example.com".to_string())
        );
    }

    #[test]
    fn hosts_file_parse() {
        let content = r#"
127.0.0.1   localhost
::1         localhost
192.168.1.100 myhost.local myhost # comment
# full comment line
"#;
        let hosts = HostsFile::parse(content);
        let addrs = hosts.lookup("localhost").unwrap();
        assert_eq!(addrs.len(), 2);
        assert!(addrs.contains(&"127.0.0.1".to_string()));
        let my = hosts.lookup("myhost.local").unwrap();
        assert_eq!(my[0], "192.168.1.100");
        let my2 = hosts.lookup("myhost").unwrap();
        assert_eq!(my2[0], "192.168.1.100");
    }

    #[test]
    fn resolver_hosts_override() {
        let hosts = HostsFile::parse("10.0.0.1 custom.local\n");
        let mut resolver = Resolver::new().with_hosts(hosts);
        let records = resolver.resolve_a("custom.local", 0);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].a_str(), Some("10.0.0.1".to_string()));
    }

    #[test]
    fn resolver_cache_fallback() {
        let mut resolver = Resolver::new();
        resolver.cache_records(
            "api.example.com",
            RecordType::A,
            vec![DnsRecord::a(203, 0, 113, 1)],
            300,
            0,
        );
        let records = resolver.resolve_a("api.example.com", 100);
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn parse_ipv4_valid() {
        assert_eq!(parse_ipv4("192.168.1.1"), Some([192, 168, 1, 1]));
        assert_eq!(parse_ipv4("0.0.0.0"), Some([0, 0, 0, 0]));
    }

    #[test]
    fn parse_ipv4_invalid() {
        assert_eq!(parse_ipv4("not-an-ip"), None);
        assert_eq!(parse_ipv4("256.0.0.1"), None);
        assert_eq!(parse_ipv4("1.2.3"), None);
    }
}
