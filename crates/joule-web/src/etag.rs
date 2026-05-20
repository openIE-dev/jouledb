//! HTTP ETag/conditional requests — strong/weak ETags, If-None-Match,
//! If-Match, If-Modified-Since, 304 response logic, ETag generation from
//! content hash.
//!
//! Pure-Rust replacement for etag, fresh, http-cache-semantics, etc.

use std::fmt;

// ── ETag ──────────────────────────────────────────────────────────

/// An HTTP ETag value (strong or weak).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ETag {
    /// The opaque tag value (without quotes or W/ prefix).
    pub tag: String,
    /// Whether this is a weak ETag.
    pub weak: bool,
}

impl ETag {
    /// Create a strong ETag.
    pub fn strong(tag: &str) -> Self {
        Self { tag: tag.into(), weak: false }
    }

    /// Create a weak ETag.
    pub fn weak(tag: &str) -> Self {
        Self { tag: tag.into(), weak: true }
    }

    /// Generate an ETag from content bytes using a simple hash.
    /// Produces a strong ETag.
    pub fn from_content(content: &[u8]) -> Self {
        let hash = content_hash(content);
        Self { tag: hash, weak: false }
    }

    /// Generate a weak ETag from content bytes.
    pub fn weak_from_content(content: &[u8]) -> Self {
        let hash = content_hash(content);
        Self { tag: hash, weak: true }
    }

    /// Generate an ETag from content length and last-modified timestamp.
    /// Produces a weak ETag (common for file serving).
    pub fn from_metadata(size: u64, last_modified_epoch_secs: i64) -> Self {
        let tag = format!("{:x}-{:x}", size, last_modified_epoch_secs as u64);
        Self { tag, weak: true }
    }

    /// Parse an ETag from a header value string.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s == "*" { return None; } // wildcard isn't a specific ETag
        let (weak, rest) = if let Some(stripped) = s.strip_prefix("W/") {
            (true, stripped)
        } else {
            (false, s)
        };
        let rest = rest.trim();
        if rest.starts_with('"') && rest.ends_with('"') && rest.len() >= 2 {
            let tag = &rest[1..rest.len() - 1];
            Some(Self { tag: tag.into(), weak })
        } else {
            None
        }
    }

    /// Serialize to the HTTP header value format.
    pub fn to_header(&self) -> String {
        if self.weak {
            format!("W/\"{}\"", self.tag)
        } else {
            format!("\"{}\"", self.tag)
        }
    }

    /// Strong comparison (RFC 7232 Section 2.3.2).
    /// Both ETags must be strong and have the same tag.
    pub fn strong_eq(&self, other: &ETag) -> bool {
        !self.weak && !other.weak && self.tag == other.tag
    }

    /// Weak comparison (RFC 7232 Section 2.3.2).
    /// Tags must match, regardless of weak/strong.
    pub fn weak_eq(&self, other: &ETag) -> bool {
        self.tag == other.tag
    }
}

impl fmt::Display for ETag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_header())
    }
}

/// Simple non-cryptographic hash for ETag generation.
/// Uses FNV-1a (64-bit) — fast and good distribution.
fn content_hash(data: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    format!("{hash:016x}")
}

// ── If-None-Match ─────────────────────────────────────────────────

/// Parsed `If-None-Match` header.
#[derive(Debug, Clone, PartialEq)]
pub enum IfNoneMatch {
    /// `*` — matches any ETag.
    Any,
    /// A list of specific ETags to compare against.
    Tags(Vec<ETag>),
}

impl IfNoneMatch {
    /// Parse from header value.
    pub fn parse(header: &str) -> Self {
        let header = header.trim();
        if header == "*" {
            return Self::Any;
        }
        let tags = parse_etag_list(header);
        Self::Tags(tags)
    }

    /// Check if the given ETag matches (using weak comparison, per GET semantics).
    pub fn matches(&self, etag: &ETag) -> bool {
        match self {
            Self::Any => true,
            Self::Tags(tags) => tags.iter().any(|t| t.weak_eq(etag)),
        }
    }
}

// ── If-Match ──────────────────────────────────────────────────────

/// Parsed `If-Match` header.
#[derive(Debug, Clone, PartialEq)]
pub enum IfMatch {
    /// `*` — matches any ETag.
    Any,
    /// A list of specific ETags to compare against.
    Tags(Vec<ETag>),
}

impl IfMatch {
    /// Parse from header value.
    pub fn parse(header: &str) -> Self {
        let header = header.trim();
        if header == "*" {
            return Self::Any;
        }
        let tags = parse_etag_list(header);
        Self::Tags(tags)
    }

    /// Check if the given ETag matches (using strong comparison, per spec).
    pub fn matches(&self, etag: &ETag) -> bool {
        match self {
            Self::Any => true,
            Self::Tags(tags) => tags.iter().any(|t| t.strong_eq(etag)),
        }
    }
}

fn parse_etag_list(header: &str) -> Vec<ETag> {
    header.split(',')
        .filter_map(|s| ETag::parse(s.trim()))
        .collect()
}

// ── If-Modified-Since ─────────────────────────────────────────────

/// Represents a timestamp for `If-Modified-Since` / `Last-Modified`.
/// Stores as seconds since Unix epoch (UTC).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HttpDate {
    pub epoch_secs: i64,
}

impl HttpDate {
    pub fn from_epoch(secs: i64) -> Self {
        Self { epoch_secs: secs }
    }

    /// Parse an HTTP-date in the preferred IMF-fixdate format:
    /// `Sun, 06 Nov 1994 08:49:37 GMT`
    pub fn parse_imf(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != 6 || parts[5] != "GMT" { return None; }
        let day: u32 = parts[1].parse().ok()?;
        let month = match parts[2] {
            "Jan" => 1, "Feb" => 2, "Mar" => 3, "Apr" => 4,
            "May" => 5, "Jun" => 6, "Jul" => 7, "Aug" => 8,
            "Sep" => 9, "Oct" => 10, "Nov" => 11, "Dec" => 12,
            _ => return None,
        };
        let year: i64 = parts[3].parse().ok()?;
        let time_parts: Vec<&str> = parts[4].split(':').collect();
        if time_parts.len() != 3 { return None; }
        let hour: i64 = time_parts[0].parse().ok()?;
        let minute: i64 = time_parts[1].parse().ok()?;
        let second: i64 = time_parts[2].parse().ok()?;

        // Simple epoch calculation (not handling leap seconds)
        let epoch = date_to_epoch(year, month, day as i64, hour, minute, second);
        Some(Self { epoch_secs: epoch })
    }

    /// Check if a resource modified at `last_modified` is fresh
    /// with respect to this If-Modified-Since date.
    pub fn is_fresh(&self, last_modified: HttpDate) -> bool {
        last_modified.epoch_secs <= self.epoch_secs
    }
}

/// Convert a date to Unix epoch seconds.
fn date_to_epoch(year: i64, month: i64, day: i64, hour: i64, min: i64, sec: i64) -> i64 {
    // Days from year 1970 to the start of the given year
    let mut y = year - 1970;
    let mut days: i64 = y * 365;
    // Add leap days
    if y > 0 {
        days += (y + 1) / 4; // leap years
        days -= (y + 69) / 100; // century years
        days += (y + 369) / 400; // 400-year cycles
    } else {
        y = -y;
        days -= (y + 2) / 4;
        days += (y + 30) / 100;
        days -= (y + 30) / 400;
    }
    // Days in each month (non-leap)
    let month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for m in 1..month {
        days += month_days[m as usize] as i64;
    }
    // Leap day adjustment
    if month > 2 {
        let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        if is_leap { days += 1; }
    }
    days += day - 1;
    days * 86400 + hour * 3600 + min * 60 + sec
}

// ── Conditional response evaluation ───────────────────────────────

/// The result of evaluating conditional headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionalResult {
    /// Send the full response (200 OK).
    SendBody,
    /// Send 304 Not Modified.
    NotModified,
    /// Send 412 Precondition Failed.
    PreconditionFailed,
}

/// Evaluate conditional request headers against current resource state.
pub fn evaluate_conditional(
    method_is_get_head: bool,
    current_etag: Option<&ETag>,
    if_match: Option<&IfMatch>,
    if_none_match: Option<&IfNoneMatch>,
    if_modified_since: Option<HttpDate>,
    last_modified: Option<HttpDate>,
) -> ConditionalResult {
    // Step 1: If-Match (RFC 7232 Section 3.1)
    if let Some(im) = if_match {
        if let Some(etag) = current_etag {
            if !im.matches(etag) {
                return ConditionalResult::PreconditionFailed;
            }
        } else {
            // No current ETag, If-Match always fails
            return ConditionalResult::PreconditionFailed;
        }
    }

    // Step 2: If-None-Match (RFC 7232 Section 3.2)
    if let Some(inm) = if_none_match {
        if let Some(etag) = current_etag {
            if inm.matches(etag) {
                if method_is_get_head {
                    return ConditionalResult::NotModified;
                } else {
                    return ConditionalResult::PreconditionFailed;
                }
            }
        }
    }

    // Step 3: If-Modified-Since (only for GET/HEAD)
    if method_is_get_head {
        if let (Some(ims), Some(lm)) = (if_modified_since, last_modified) {
            if ims.is_fresh(lm) {
                return ConditionalResult::NotModified;
            }
        }
    }

    ConditionalResult::SendBody
}

/// Generate conditional response headers given the resource state.
pub fn conditional_headers(etag: &ETag, last_modified_epoch: Option<i64>) -> Vec<(String, String)> {
    let mut headers = Vec::new();
    headers.push(("ETag".into(), etag.to_header()));
    if let Some(epoch) = last_modified_epoch {
        headers.push(("Last-Modified".into(), format_http_date(epoch)));
    }
    headers.push(("Cache-Control".into(), "no-cache".into()));
    headers
}

/// Format an epoch timestamp as an HTTP-date.
fn format_http_date(epoch_secs: i64) -> String {
    // Simple formatter for common cases
    let days_since_epoch = epoch_secs / 86400;
    let time_of_day = epoch_secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Day of week (Jan 1 1970 was Thursday = 4)
    let dow = ((days_since_epoch % 7) + 4) % 7;
    let dow_str = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

    // Convert days to year/month/day
    let (year, month, day) = epoch_days_to_date(days_since_epoch);
    let month_str = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

    format!("{}, {:02} {} {} {:02}:{:02}:{:02} GMT",
        dow_str[dow as usize], day, month_str[month as usize],
        year, hours, minutes, seconds)
}

fn epoch_days_to_date(days: i64) -> (i64, i64, i64) {
    // Civil calendar from days since epoch
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etag_strong() {
        let e = ETag::strong("abc123");
        assert!(!e.weak);
        assert_eq!(e.to_header(), r#""abc123""#);
    }

    #[test]
    fn etag_weak() {
        let e = ETag::weak("abc123");
        assert!(e.weak);
        assert_eq!(e.to_header(), r#"W/"abc123""#);
    }

    #[test]
    fn etag_parse_strong() {
        let e = ETag::parse(r#""xyzzy""#).unwrap();
        assert!(!e.weak);
        assert_eq!(e.tag, "xyzzy");
    }

    #[test]
    fn etag_parse_weak() {
        let e = ETag::parse(r#"W/"xyzzy""#).unwrap();
        assert!(e.weak);
        assert_eq!(e.tag, "xyzzy");
    }

    #[test]
    fn etag_parse_wildcard_none() {
        assert!(ETag::parse("*").is_none());
    }

    #[test]
    fn etag_parse_invalid() {
        assert!(ETag::parse("noquotes").is_none());
    }

    #[test]
    fn etag_strong_comparison() {
        let a = ETag::strong("abc");
        let b = ETag::strong("abc");
        let c = ETag::weak("abc");
        assert!(a.strong_eq(&b));
        assert!(!a.strong_eq(&c));
    }

    #[test]
    fn etag_weak_comparison() {
        let a = ETag::strong("abc");
        let b = ETag::weak("abc");
        assert!(a.weak_eq(&b));
        assert!(b.weak_eq(&a));
    }

    #[test]
    fn etag_from_content() {
        let e1 = ETag::from_content(b"hello");
        let e2 = ETag::from_content(b"hello");
        let e3 = ETag::from_content(b"world");
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
        assert!(!e1.weak);
    }

    #[test]
    fn etag_weak_from_content() {
        let e = ETag::weak_from_content(b"test");
        assert!(e.weak);
    }

    #[test]
    fn etag_from_metadata() {
        let e = ETag::from_metadata(1024, 1700000000);
        assert!(e.weak);
        assert!(e.tag.contains('-'));
    }

    #[test]
    fn etag_display() {
        let e = ETag::strong("test");
        assert_eq!(format!("{e}"), r#""test""#);
    }

    #[test]
    fn etag_roundtrip() {
        let original = ETag::strong("roundtrip-test");
        let header = original.to_header();
        let parsed = ETag::parse(&header).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn etag_weak_roundtrip() {
        let original = ETag::weak("weak-test");
        let header = original.to_header();
        let parsed = ETag::parse(&header).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn if_none_match_any() {
        let inm = IfNoneMatch::parse("*");
        assert_eq!(inm, IfNoneMatch::Any);
        assert!(inm.matches(&ETag::strong("anything")));
    }

    #[test]
    fn if_none_match_single() {
        let inm = IfNoneMatch::parse(r#""abc""#);
        assert!(inm.matches(&ETag::strong("abc")));
        assert!(inm.matches(&ETag::weak("abc"))); // weak comparison
        assert!(!inm.matches(&ETag::strong("def")));
    }

    #[test]
    fn if_none_match_multiple() {
        let inm = IfNoneMatch::parse(r#""abc", "def", W/"ghi""#);
        if let IfNoneMatch::Tags(ref tags) = inm {
            assert_eq!(tags.len(), 3);
        }
        assert!(inm.matches(&ETag::strong("def")));
        assert!(inm.matches(&ETag::strong("ghi"))); // weak comparison
        assert!(!inm.matches(&ETag::strong("xyz")));
    }

    #[test]
    fn if_match_any() {
        let im = IfMatch::parse("*");
        assert_eq!(im, IfMatch::Any);
        assert!(im.matches(&ETag::strong("anything")));
    }

    #[test]
    fn if_match_strong_only() {
        let im = IfMatch::parse(r#""abc""#);
        assert!(im.matches(&ETag::strong("abc")));
        assert!(!im.matches(&ETag::weak("abc"))); // strong comparison required
    }

    #[test]
    fn if_match_multiple() {
        let im = IfMatch::parse(r#""abc", "def""#);
        assert!(im.matches(&ETag::strong("def")));
        assert!(!im.matches(&ETag::strong("ghi")));
    }

    #[test]
    fn http_date_parse() {
        let d = HttpDate::parse_imf("Sun, 06 Nov 1994 08:49:37 GMT").unwrap();
        assert_eq!(d.epoch_secs, 784111777);
    }

    #[test]
    fn http_date_parse_invalid() {
        assert!(HttpDate::parse_imf("not a date").is_none());
        assert!(HttpDate::parse_imf("Sun, 06 Nov 1994 08:49:37 EST").is_none());
    }

    #[test]
    fn http_date_freshness() {
        let ims = HttpDate::from_epoch(1000);
        let lm_old = HttpDate::from_epoch(999);
        let lm_same = HttpDate::from_epoch(1000);
        let lm_new = HttpDate::from_epoch(1001);
        assert!(ims.is_fresh(lm_old));
        assert!(ims.is_fresh(lm_same));
        assert!(!ims.is_fresh(lm_new));
    }

    #[test]
    fn eval_no_conditions() {
        let result = evaluate_conditional(true, None, None, None, None, None);
        assert_eq!(result, ConditionalResult::SendBody);
    }

    #[test]
    fn eval_if_none_match_304() {
        let etag = ETag::strong("abc");
        let inm = IfNoneMatch::parse(r#""abc""#);
        let result = evaluate_conditional(true, Some(&etag), None, Some(&inm), None, None);
        assert_eq!(result, ConditionalResult::NotModified);
    }

    #[test]
    fn eval_if_none_match_no_match() {
        let etag = ETag::strong("abc");
        let inm = IfNoneMatch::parse(r#""def""#);
        let result = evaluate_conditional(true, Some(&etag), None, Some(&inm), None, None);
        assert_eq!(result, ConditionalResult::SendBody);
    }

    #[test]
    fn eval_if_none_match_post_412() {
        let etag = ETag::strong("abc");
        let inm = IfNoneMatch::parse(r#""abc""#);
        let result = evaluate_conditional(false, Some(&etag), None, Some(&inm), None, None);
        assert_eq!(result, ConditionalResult::PreconditionFailed);
    }

    #[test]
    fn eval_if_match_success() {
        let etag = ETag::strong("abc");
        let im = IfMatch::parse(r#""abc""#);
        let result = evaluate_conditional(true, Some(&etag), Some(&im), None, None, None);
        assert_eq!(result, ConditionalResult::SendBody);
    }

    #[test]
    fn eval_if_match_fail_412() {
        let etag = ETag::strong("abc");
        let im = IfMatch::parse(r#""def""#);
        let result = evaluate_conditional(true, Some(&etag), Some(&im), None, None, None);
        assert_eq!(result, ConditionalResult::PreconditionFailed);
    }

    #[test]
    fn eval_if_match_no_etag_412() {
        let im = IfMatch::parse(r#""abc""#);
        let result = evaluate_conditional(true, None, Some(&im), None, None, None);
        assert_eq!(result, ConditionalResult::PreconditionFailed);
    }

    #[test]
    fn eval_if_modified_since_fresh() {
        let lm = HttpDate::from_epoch(1000);
        let ims = HttpDate::from_epoch(2000);
        let result = evaluate_conditional(true, None, None, None, Some(ims), Some(lm));
        assert_eq!(result, ConditionalResult::NotModified);
    }

    #[test]
    fn eval_if_modified_since_stale() {
        let lm = HttpDate::from_epoch(3000);
        let ims = HttpDate::from_epoch(2000);
        let result = evaluate_conditional(true, None, None, None, Some(ims), Some(lm));
        assert_eq!(result, ConditionalResult::SendBody);
    }

    #[test]
    fn eval_if_modified_since_ignored_for_post() {
        let lm = HttpDate::from_epoch(1000);
        let ims = HttpDate::from_epoch(2000);
        let result = evaluate_conditional(false, None, None, None, Some(ims), Some(lm));
        assert_eq!(result, ConditionalResult::SendBody); // ignored for non-GET/HEAD
    }

    #[test]
    fn eval_if_match_takes_priority() {
        let etag = ETag::strong("abc");
        let im = IfMatch::parse(r#""def""#); // fails
        let inm = IfNoneMatch::parse(r#""abc""#); // would match
        let result = evaluate_conditional(true, Some(&etag), Some(&im), Some(&inm), None, None);
        assert_eq!(result, ConditionalResult::PreconditionFailed); // If-Match checked first
    }

    #[test]
    fn conditional_headers_generation() {
        let etag = ETag::strong("test-etag");
        let headers = conditional_headers(&etag, Some(1700000000));
        assert!(headers.iter().any(|(k, v)| k == "ETag" && v == r#""test-etag""#));
        assert!(headers.iter().any(|(k, _)| k == "Last-Modified"));
        assert!(headers.iter().any(|(k, v)| k == "Cache-Control" && v == "no-cache"));
    }

    #[test]
    fn format_http_date_basic() {
        let date_str = format_http_date(784111777); // Nov 6, 1994
        assert!(date_str.ends_with("GMT"));
        assert!(date_str.contains("1994"));
    }

    #[test]
    fn content_hash_deterministic() {
        let h1 = content_hash(b"hello");
        let h2 = content_hash(b"hello");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn content_hash_different() {
        let h1 = content_hash(b"hello");
        let h2 = content_hash(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn content_hash_empty() {
        let h = content_hash(b"");
        assert_eq!(h.len(), 16);
    }

    #[test]
    fn etag_content_different_lengths() {
        let e1 = ETag::from_content(b"a");
        let e2 = ETag::from_content(b"ab");
        assert_ne!(e1, e2);
    }

    #[test]
    fn if_none_match_wildcard_any_etag() {
        let inm = IfNoneMatch::Any;
        assert!(inm.matches(&ETag::strong("any")));
        assert!(inm.matches(&ETag::weak("any")));
    }
}
