//! IP address filtering — allow/deny lists, CIDR range matching, geo-classification
//! (private/public/reserved), rate limiting per IP, IP reputation scoring, filter
//! rule ordering, and IPv4/IPv6 support.
//!
//! Replaces `express-ipfilter`, `node-ipgeo`, and similar JS middleware with a
//! pure-Rust IP filtering engine supporting CIDR matching and reputation tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// IP filter errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpFilterError {
    /// Invalid IP address format.
    InvalidAddress(String),
    /// Invalid CIDR notation.
    InvalidCidr(String),
    /// IP address blocked.
    Blocked(String),
    /// Rate limit exceeded for IP.
    RateLimited { ip: String, limit: u64, window_ms: u64 },
    /// Duplicate rule ID.
    DuplicateRule(String),
}

impl fmt::Display for IpFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAddress(addr) => write!(f, "invalid IP address: {addr}"),
            Self::InvalidCidr(cidr) => write!(f, "invalid CIDR: {cidr}"),
            Self::Blocked(ip) => write!(f, "IP blocked: {ip}"),
            Self::RateLimited { ip, limit, window_ms } => {
                write!(f, "rate limited: {ip} ({limit} req/{window_ms}ms)")
            }
            Self::DuplicateRule(id) => write!(f, "duplicate rule: {id}"),
        }
    }
}

impl std::error::Error for IpFilterError {}

// ── IP Address Types ───────────────────────────────────────────

/// Parsed IPv4 address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ipv4Addr {
    pub octets: [u8; 4],
}

impl Ipv4Addr {
    pub fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        Self { octets: [a, b, c, d] }
    }

    pub fn to_u32(self) -> u32 {
        ((self.octets[0] as u32) << 24)
            | ((self.octets[1] as u32) << 16)
            | ((self.octets[2] as u32) << 8)
            | (self.octets[3] as u32)
    }

    pub fn from_u32(val: u32) -> Self {
        Self {
            octets: [
                ((val >> 24) & 0xFF) as u8,
                ((val >> 16) & 0xFF) as u8,
                ((val >> 8) & 0xFF) as u8,
                (val & 0xFF) as u8,
            ],
        }
    }

    pub fn parse(s: &str) -> Result<Self, IpFilterError> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 {
            return Err(IpFilterError::InvalidAddress(s.to_string()));
        }
        let mut octets = [0u8; 4];
        for (i, part) in parts.iter().enumerate() {
            octets[i] = part
                .parse::<u8>()
                .map_err(|_| IpFilterError::InvalidAddress(s.to_string()))?;
        }
        Ok(Self { octets })
    }
}

impl fmt::Display for Ipv4Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}.{}", self.octets[0], self.octets[1], self.octets[2], self.octets[3])
    }
}

/// Parsed IPv6 address (stored as 128-bit value via two u64s).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ipv6Addr {
    pub high: u64,
    pub low: u64,
}

impl Ipv6Addr {
    pub fn new(high: u64, low: u64) -> Self {
        Self { high, low }
    }

    pub fn from_segments(segments: [u16; 8]) -> Self {
        let high = ((segments[0] as u64) << 48)
            | ((segments[1] as u64) << 32)
            | ((segments[2] as u64) << 16)
            | (segments[3] as u64);
        let low = ((segments[4] as u64) << 48)
            | ((segments[5] as u64) << 32)
            | ((segments[6] as u64) << 16)
            | (segments[7] as u64);
        Self { high, low }
    }

    pub fn to_segments(self) -> [u16; 8] {
        [
            ((self.high >> 48) & 0xFFFF) as u16,
            ((self.high >> 32) & 0xFFFF) as u16,
            ((self.high >> 16) & 0xFFFF) as u16,
            (self.high & 0xFFFF) as u16,
            ((self.low >> 48) & 0xFFFF) as u16,
            ((self.low >> 32) & 0xFFFF) as u16,
            ((self.low >> 16) & 0xFFFF) as u16,
            (self.low & 0xFFFF) as u16,
        ]
    }

    pub fn parse(s: &str) -> Result<Self, IpFilterError> {
        // Handle :: expansion
        let err = || IpFilterError::InvalidAddress(s.to_string());

        let expanded = expand_ipv6(s).map_err(|_| err())?;
        let parts: Vec<&str> = expanded.split(':').collect();
        if parts.len() != 8 {
            return Err(err());
        }
        let mut segments = [0u16; 8];
        for (i, part) in parts.iter().enumerate() {
            segments[i] = u16::from_str_radix(part, 16).map_err(|_| err())?;
        }
        Ok(Self::from_segments(segments))
    }

    pub fn is_loopback(&self) -> bool {
        self.high == 0 && self.low == 1
    }
}

impl fmt::Display for Ipv6Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let seg = self.to_segments();
        write!(
            f,
            "{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}",
            seg[0], seg[1], seg[2], seg[3], seg[4], seg[5], seg[6], seg[7]
        )
    }
}

fn expand_ipv6(s: &str) -> Result<String, ()> {
    if !s.contains("::") {
        return Ok(s.to_string());
    }
    let parts: Vec<&str> = s.split("::").collect();
    if parts.len() > 2 {
        return Err(());
    }
    let left: Vec<&str> = if parts[0].is_empty() {
        Vec::new()
    } else {
        parts[0].split(':').collect()
    };
    let right: Vec<&str> = if parts.len() > 1 && !parts[1].is_empty() {
        parts[1].split(':').collect()
    } else {
        Vec::new()
    };
    let missing = 8 - left.len() - right.len();
    let zeros: Vec<&str> = vec!["0"; missing];
    let all: Vec<&str> = left.iter().chain(zeros.iter()).chain(right.iter()).copied().collect();
    Ok(all.join(":"))
}

/// A unified IP address (v4 or v6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IpAddr {
    V4(Ipv4Addr),
    V6(Ipv6Addr),
}

impl IpAddr {
    pub fn parse(s: &str) -> Result<Self, IpFilterError> {
        if s.contains(':') {
            Ok(Self::V6(Ipv6Addr::parse(s)?))
        } else {
            Ok(Self::V4(Ipv4Addr::parse(s)?))
        }
    }

    pub fn is_v4(&self) -> bool {
        matches!(self, Self::V4(_))
    }

    pub fn is_v6(&self) -> bool {
        matches!(self, Self::V6(_))
    }
}

impl fmt::Display for IpAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V4(v4) => write!(f, "{v4}"),
            Self::V6(v6) => write!(f, "{v6}"),
        }
    }
}

/// Geo-classification of an IP address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IpClassification {
    /// Private/RFC1918 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16).
    Private,
    /// Loopback (127.0.0.0/8, ::1).
    Loopback,
    /// Link-local (169.254.0.0/16, fe80::/10).
    LinkLocal,
    /// Multicast (224.0.0.0/4, ff00::/8).
    Multicast,
    /// Reserved (0.0.0.0/8, 240.0.0.0/4, etc.).
    Reserved,
    /// Public/routable.
    Public,
}

impl IpClassification {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Private => "private",
            Self::Loopback => "loopback",
            Self::LinkLocal => "link_local",
            Self::Multicast => "multicast",
            Self::Reserved => "reserved",
            Self::Public => "public",
        }
    }
}

/// Classify an IP address.
pub fn classify_ip(addr: &IpAddr) -> IpClassification {
    match addr {
        IpAddr::V4(v4) => classify_ipv4(v4),
        IpAddr::V6(v6) => classify_ipv6(v6),
    }
}

fn classify_ipv4(addr: &Ipv4Addr) -> IpClassification {
    let o = addr.octets;
    // Loopback
    if o[0] == 127 {
        return IpClassification::Loopback;
    }
    // Private
    if o[0] == 10 {
        return IpClassification::Private;
    }
    if o[0] == 172 && (16..=31).contains(&o[1]) {
        return IpClassification::Private;
    }
    if o[0] == 192 && o[1] == 168 {
        return IpClassification::Private;
    }
    // Link-local
    if o[0] == 169 && o[1] == 254 {
        return IpClassification::LinkLocal;
    }
    // Multicast
    if (224..=239).contains(&o[0]) {
        return IpClassification::Multicast;
    }
    // Reserved
    if o[0] == 0 || o[0] >= 240 {
        return IpClassification::Reserved;
    }
    IpClassification::Public
}

fn classify_ipv6(addr: &Ipv6Addr) -> IpClassification {
    if addr.is_loopback() {
        return IpClassification::Loopback;
    }
    let first_seg = ((addr.high >> 48) & 0xFFFF) as u16;
    // Link-local: fe80::/10
    if first_seg & 0xFFC0 == 0xFE80 {
        return IpClassification::LinkLocal;
    }
    // Multicast: ff00::/8
    if first_seg & 0xFF00 == 0xFF00 {
        return IpClassification::Multicast;
    }
    // Unique local: fc00::/7
    if first_seg & 0xFE00 == 0xFC00 {
        return IpClassification::Private;
    }
    // Unspecified
    if addr.high == 0 && addr.low == 0 {
        return IpClassification::Reserved;
    }
    IpClassification::Public
}

/// CIDR range (supports v4 and v6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CidrRange {
    pub addr: IpAddr,
    pub prefix_len: u8,
}

impl CidrRange {
    pub fn parse(s: &str) -> Result<Self, IpFilterError> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() != 2 {
            return Err(IpFilterError::InvalidCidr(s.to_string()));
        }
        let addr = IpAddr::parse(parts[0])?;
        let prefix_len: u8 = parts[1]
            .parse()
            .map_err(|_| IpFilterError::InvalidCidr(s.to_string()))?;

        let max_prefix = match &addr {
            IpAddr::V4(_) => 32,
            IpAddr::V6(_) => 128,
        };
        if prefix_len > max_prefix {
            return Err(IpFilterError::InvalidCidr(s.to_string()));
        }

        Ok(Self { addr, prefix_len })
    }

    /// Check if an IP is within this CIDR range.
    pub fn contains(&self, ip: &IpAddr) -> bool {
        match (&self.addr, ip) {
            (IpAddr::V4(net), IpAddr::V4(target)) => {
                if self.prefix_len == 0 {
                    return true;
                }
                let mask = if self.prefix_len >= 32 {
                    u32::MAX
                } else {
                    u32::MAX << (32 - self.prefix_len)
                };
                (net.to_u32() & mask) == (target.to_u32() & mask)
            }
            (IpAddr::V6(net), IpAddr::V6(target)) => {
                if self.prefix_len == 0 {
                    return true;
                }
                if self.prefix_len <= 64 {
                    let shift = 64 - self.prefix_len as u32;
                    let mask = if shift >= 64 { 0 } else { u64::MAX << shift };
                    (net.high & mask) == (target.high & mask)
                } else {
                    if net.high != target.high {
                        return false;
                    }
                    let low_bits = self.prefix_len - 64;
                    let shift = 64 - low_bits as u32;
                    let mask = if shift >= 64 { 0 } else { u64::MAX << shift };
                    (net.low & mask) == (target.low & mask)
                }
            }
            _ => false, // v4/v6 mismatch
        }
    }
}

impl fmt::Display for CidrRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.addr, self.prefix_len)
    }
}

/// Action for a filter rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterAction {
    Allow,
    Deny,
}

/// A single IP filter rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRule {
    /// Unique rule ID.
    pub id: String,
    /// CIDR range (or single IP as /32 or /128).
    pub cidr: CidrRange,
    /// Allow or deny.
    pub action: FilterAction,
    /// Priority (lower = evaluated first).
    pub priority: u32,
    /// Description.
    pub description: String,
}

impl FilterRule {
    pub fn new(id: &str, cidr: CidrRange, action: FilterAction) -> Self {
        Self {
            id: id.to_string(),
            cidr,
            action,
            priority: 100,
            description: String::new(),
        }
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }
}

/// Per-IP rate limit state.
#[derive(Debug, Clone)]
struct IpRateState {
    /// Request timestamps within the current window (epoch ms).
    requests: Vec<u64>,
}

/// IP reputation entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpReputation {
    /// Current score (0-100, higher = more trusted).
    pub score: u32,
    /// Number of violations.
    pub violations: u32,
    /// Number of successful accesses.
    pub successes: u32,
    /// Last seen timestamp (epoch ms).
    pub last_seen_ms: u64,
}

impl IpReputation {
    fn new() -> Self {
        Self {
            score: 50, // neutral starting score
            violations: 0,
            successes: 0,
            last_seen_ms: 0,
        }
    }
}

/// The IP filter engine.
pub struct IpFilter {
    rules: Vec<FilterRule>,
    /// Default action if no rule matches.
    pub default_action: FilterAction,
    /// Rate limit: max requests per window.
    pub rate_limit: Option<u64>,
    /// Rate limit window in milliseconds.
    pub rate_window_ms: u64,
    /// Per-IP rate state.
    rate_states: HashMap<String, IpRateState>,
    /// IP reputation scores.
    reputations: HashMap<String, IpReputation>,
    /// Minimum reputation score to allow access (0 = disabled).
    pub min_reputation_score: u32,
}

impl IpFilter {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            default_action: FilterAction::Allow,
            rate_limit: None,
            rate_window_ms: 60_000,
            rate_states: HashMap::new(),
            reputations: HashMap::new(),
            min_reputation_score: 0,
        }
    }

    /// Add a filter rule.
    pub fn add_rule(&mut self, rule: FilterRule) -> Result<(), IpFilterError> {
        if self.rules.iter().any(|r| r.id == rule.id) {
            return Err(IpFilterError::DuplicateRule(rule.id));
        }
        self.rules.push(rule);
        self.rules.sort_by_key(|r| r.priority);
        Ok(())
    }

    /// Remove a rule by ID.
    pub fn remove_rule(&mut self, id: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.id != id);
        self.rules.len() < before
    }

    /// Number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Check whether an IP is allowed (combines rules, rate limits, reputation).
    pub fn check(&mut self, ip_str: &str, now_ms: u64) -> Result<FilterAction, IpFilterError> {
        let ip = IpAddr::parse(ip_str)?;

        // 1. Check filter rules (ordered by priority).
        let mut action = self.default_action;
        for rule in &self.rules {
            if rule.cidr.contains(&ip) {
                action = rule.action;
                break; // first match wins (rules are sorted by priority)
            }
        }

        if action == FilterAction::Deny {
            self.record_violation(ip_str, now_ms);
            return Err(IpFilterError::Blocked(ip_str.to_string()));
        }

        // 2. Check rate limit.
        if let Some(limit) = self.rate_limit {
            let state = self
                .rate_states
                .entry(ip_str.to_string())
                .or_insert_with(|| IpRateState { requests: Vec::new() });

            // Remove expired entries.
            let cutoff = now_ms.saturating_sub(self.rate_window_ms);
            state.requests.retain(|ts| *ts > cutoff);

            if state.requests.len() as u64 >= limit {
                self.record_violation(ip_str, now_ms);
                return Err(IpFilterError::RateLimited {
                    ip: ip_str.to_string(),
                    limit,
                    window_ms: self.rate_window_ms,
                });
            }
            state.requests.push(now_ms);
        }

        // 3. Check reputation.
        if self.min_reputation_score > 0 {
            let rep = self.reputations.get(ip_str);
            if let Some(rep) = rep {
                if rep.score < self.min_reputation_score {
                    return Err(IpFilterError::Blocked(ip_str.to_string()));
                }
            }
        }

        // Record success
        self.record_success(ip_str, now_ms);

        Ok(FilterAction::Allow)
    }

    fn record_violation(&mut self, ip: &str, now_ms: u64) {
        let rep = self.reputations.entry(ip.to_string()).or_insert_with(IpReputation::new);
        rep.violations += 1;
        rep.score = rep.score.saturating_sub(5);
        rep.last_seen_ms = now_ms;
    }

    fn record_success(&mut self, ip: &str, now_ms: u64) {
        let rep = self.reputations.entry(ip.to_string()).or_insert_with(IpReputation::new);
        rep.successes += 1;
        rep.score = std::cmp::min(100, rep.score + 1);
        rep.last_seen_ms = now_ms;
    }

    /// Get reputation for an IP.
    pub fn reputation(&self, ip: &str) -> Option<&IpReputation> {
        self.reputations.get(ip)
    }

    /// Manually set reputation score.
    pub fn set_reputation_score(&mut self, ip: &str, score: u32) {
        let rep = self.reputations.entry(ip.to_string()).or_insert_with(IpReputation::new);
        rep.score = std::cmp::min(100, score);
    }

    /// Classify an IP address.
    pub fn classify(&self, ip_str: &str) -> Result<IpClassification, IpFilterError> {
        let ip = IpAddr::parse(ip_str)?;
        Ok(classify_ip(&ip))
    }

    /// Check if an IP matches a CIDR range.
    pub fn matches_cidr(&self, ip_str: &str, cidr_str: &str) -> Result<bool, IpFilterError> {
        let ip = IpAddr::parse(ip_str)?;
        let cidr = CidrRange::parse(cidr_str)?;
        Ok(cidr.contains(&ip))
    }
}

impl Default for IpFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipv4_parse() {
        let ip = Ipv4Addr::parse("192.168.1.1").unwrap();
        assert_eq!(ip.octets, [192, 168, 1, 1]);
    }

    #[test]
    fn test_ipv4_invalid() {
        assert!(Ipv4Addr::parse("999.0.0.1").is_err());
        assert!(Ipv4Addr::parse("1.2.3").is_err());
    }

    #[test]
    fn test_ipv4_to_from_u32() {
        let ip = Ipv4Addr::new(10, 0, 0, 1);
        let val = ip.to_u32();
        let back = Ipv4Addr::from_u32(val);
        assert_eq!(ip, back);
    }

    #[test]
    fn test_ipv6_parse_full() {
        let ip = Ipv6Addr::parse("2001:0db8:0000:0000:0000:0000:0000:0001").unwrap();
        let seg = ip.to_segments();
        assert_eq!(seg[0], 0x2001);
        assert_eq!(seg[7], 1);
    }

    #[test]
    fn test_ipv6_parse_compressed() {
        let ip = Ipv6Addr::parse("::1").unwrap();
        assert!(ip.is_loopback());
    }

    #[test]
    fn test_ipv6_display() {
        let ip = Ipv6Addr::parse("::1").unwrap();
        assert_eq!(ip.to_string(), "0:0:0:0:0:0:0:1");
    }

    #[test]
    fn test_cidr_v4_contains() {
        let cidr = CidrRange::parse("192.168.1.0/24").unwrap();
        let ip_in = IpAddr::parse("192.168.1.100").unwrap();
        let ip_out = IpAddr::parse("192.168.2.1").unwrap();
        assert!(cidr.contains(&ip_in));
        assert!(!cidr.contains(&ip_out));
    }

    #[test]
    fn test_cidr_v4_host() {
        let cidr = CidrRange::parse("10.0.0.1/32").unwrap();
        assert!(cidr.contains(&IpAddr::parse("10.0.0.1").unwrap()));
        assert!(!cidr.contains(&IpAddr::parse("10.0.0.2").unwrap()));
    }

    #[test]
    fn test_cidr_v6_contains() {
        let cidr = CidrRange::parse("2001:db8::/32").unwrap();
        let ip_in = IpAddr::parse("2001:0db8:0000:0000:0000:0000:0000:0001").unwrap();
        let ip_out = IpAddr::parse("2001:0db9:0000:0000:0000:0000:0000:0001").unwrap();
        assert!(cidr.contains(&ip_in));
        assert!(!cidr.contains(&ip_out));
    }

    #[test]
    fn test_classify_private() {
        let filter = IpFilter::new();
        assert_eq!(filter.classify("10.0.0.1").unwrap(), IpClassification::Private);
        assert_eq!(filter.classify("172.16.0.1").unwrap(), IpClassification::Private);
        assert_eq!(filter.classify("192.168.1.1").unwrap(), IpClassification::Private);
    }

    #[test]
    fn test_classify_loopback() {
        let filter = IpFilter::new();
        assert_eq!(filter.classify("127.0.0.1").unwrap(), IpClassification::Loopback);
    }

    #[test]
    fn test_classify_public() {
        let filter = IpFilter::new();
        assert_eq!(filter.classify("8.8.8.8").unwrap(), IpClassification::Public);
    }

    #[test]
    fn test_classify_multicast() {
        let filter = IpFilter::new();
        assert_eq!(filter.classify("224.0.0.1").unwrap(), IpClassification::Multicast);
    }

    #[test]
    fn test_classify_link_local() {
        let filter = IpFilter::new();
        assert_eq!(filter.classify("169.254.1.1").unwrap(), IpClassification::LinkLocal);
    }

    #[test]
    fn test_filter_allow_rule() {
        let mut filter = IpFilter::new();
        filter.default_action = FilterAction::Deny;
        filter
            .add_rule(FilterRule::new(
                "allow-private",
                CidrRange::parse("192.168.0.0/16").unwrap(),
                FilterAction::Allow,
            ).with_priority(1))
            .unwrap();
        assert_eq!(filter.check("192.168.1.100", 1000).unwrap(), FilterAction::Allow);
        assert!(filter.check("8.8.8.8", 1000).is_err()); // default deny
    }

    #[test]
    fn test_filter_deny_rule() {
        let mut filter = IpFilter::new();
        filter
            .add_rule(FilterRule::new(
                "block-bad",
                CidrRange::parse("10.0.0.0/8").unwrap(),
                FilterAction::Deny,
            ))
            .unwrap();
        assert!(filter.check("10.0.0.5", 1000).is_err());
        assert_eq!(filter.check("8.8.8.8", 1000).unwrap(), FilterAction::Allow);
    }

    #[test]
    fn test_rate_limiting() {
        let mut filter = IpFilter::new();
        filter.rate_limit = Some(3);
        filter.rate_window_ms = 10_000;

        assert!(filter.check("1.2.3.4", 1000).is_ok());
        assert!(filter.check("1.2.3.4", 2000).is_ok());
        assert!(filter.check("1.2.3.4", 3000).is_ok());
        let err = filter.check("1.2.3.4", 4000).unwrap_err();
        match err {
            IpFilterError::RateLimited { .. } => {}
            other => panic!("expected RateLimited, got: {other}"),
        }
    }

    #[test]
    fn test_rate_limit_window_expiry() {
        let mut filter = IpFilter::new();
        filter.rate_limit = Some(2);
        filter.rate_window_ms = 5000;

        assert!(filter.check("1.2.3.4", 1000).is_ok());
        assert!(filter.check("1.2.3.4", 2000).is_ok());
        assert!(filter.check("1.2.3.4", 3000).is_err()); // limit hit
        // After window expires
        assert!(filter.check("1.2.3.4", 10_000).is_ok());
    }

    #[test]
    fn test_reputation_tracking() {
        let mut filter = IpFilter::new();
        filter.check("1.2.3.4", 1000).unwrap();
        let rep = filter.reputation("1.2.3.4").unwrap();
        assert_eq!(rep.successes, 1);
        assert!(rep.score > 50); // increased from default
    }

    #[test]
    fn test_reputation_violation() {
        let mut filter = IpFilter::new();
        filter
            .add_rule(FilterRule::new(
                "block",
                CidrRange::parse("10.0.0.0/8").unwrap(),
                FilterAction::Deny,
            ))
            .unwrap();
        let _ = filter.check("10.0.0.1", 1000);
        let rep = filter.reputation("10.0.0.1").unwrap();
        assert_eq!(rep.violations, 1);
        assert!(rep.score < 50); // decreased
    }

    #[test]
    fn test_min_reputation_score() {
        let mut filter = IpFilter::new();
        filter.min_reputation_score = 30;
        filter.set_reputation_score("1.2.3.4", 10);
        let err = filter.check("1.2.3.4", 1000).unwrap_err();
        match err {
            IpFilterError::Blocked(_) => {}
            other => panic!("expected Blocked, got: {other}"),
        }
    }

    #[test]
    fn test_duplicate_rule() {
        let mut filter = IpFilter::new();
        let cidr = CidrRange::parse("10.0.0.0/8").unwrap();
        filter.add_rule(FilterRule::new("r1", cidr.clone(), FilterAction::Allow)).unwrap();
        let err = filter.add_rule(FilterRule::new("r1", cidr, FilterAction::Deny)).unwrap_err();
        assert_eq!(err, IpFilterError::DuplicateRule("r1".into()));
    }

    #[test]
    fn test_remove_rule() {
        let mut filter = IpFilter::new();
        filter
            .add_rule(FilterRule::new(
                "r1",
                CidrRange::parse("10.0.0.0/8").unwrap(),
                FilterAction::Deny,
            ))
            .unwrap();
        assert!(filter.remove_rule("r1"));
        assert_eq!(filter.rule_count(), 0);
    }

    #[test]
    fn test_matches_cidr() {
        let filter = IpFilter::new();
        assert!(filter.matches_cidr("192.168.1.1", "192.168.0.0/16").unwrap());
        assert!(!filter.matches_cidr("10.0.0.1", "192.168.0.0/16").unwrap());
    }

    #[test]
    fn test_rule_priority_ordering() {
        let mut filter = IpFilter::new();
        filter.default_action = FilterAction::Deny;
        // Higher priority deny (priority 1) should override lower priority allow (priority 10)
        filter
            .add_rule(
                FilterRule::new("allow-all", CidrRange::parse("0.0.0.0/0").unwrap(), FilterAction::Allow)
                    .with_priority(10),
            )
            .unwrap();
        filter
            .add_rule(
                FilterRule::new(
                    "deny-specific",
                    CidrRange::parse("10.0.0.0/8").unwrap(),
                    FilterAction::Deny,
                )
                .with_priority(1),
            )
            .unwrap();
        assert!(filter.check("10.0.0.1", 1000).is_err()); // deny wins (priority 1)
        assert!(filter.check("8.8.8.8", 1000).is_ok()); // allow (no deny matches)
    }

    #[test]
    fn test_ipv6_loopback_classification() {
        let filter = IpFilter::new();
        assert_eq!(filter.classify("::1").unwrap(), IpClassification::Loopback);
    }
}
