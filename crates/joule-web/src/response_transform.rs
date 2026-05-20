// response_transform.rs — Response transformation pipeline
// Header injection/removal, body rewriting, status code mapping,
// content-type filtering, response aggregation from multiple sources.

use std::collections::HashMap;

/// An HTTP-like response that can be transformed.
#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16) -> Self {
        Self {
            status,
            headers: HashMap::new(),
            body: Vec::new(),
        }
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_lowercase(), value.to_string());
        self
    }

    pub fn with_body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = body.into();
        self
    }

    pub fn body_as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.body).ok()
    }
}

/// A single transformation rule.
#[derive(Debug, Clone)]
pub enum TransformRule {
    /// Inject a header (overwrites if exists).
    InjectHeader { key: String, value: String },
    /// Remove a header by name (case-insensitive match on lowercased key).
    RemoveHeader { key: String },
    /// Find/replace in the body (UTF-8 only; non-UTF-8 bodies are skipped).
    BodyRewrite { find: String, replace: String },
    /// Map one status code to another.
    StatusMap { from: u16, to: u16 },
    /// Filter: only pass through responses whose content-type contains the given substring.
    ContentTypeFilter { allowed: String },
}

/// Outcome of applying a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformOutcome {
    /// The response passed all rules and was transformed.
    Transformed,
    /// The response was filtered out by a ContentTypeFilter rule.
    FilteredOut,
}

/// A chain of transformation rules applied in order.
#[derive(Debug, Clone, Default)]
pub struct TransformPipeline {
    rules: Vec<TransformRule>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: TransformRule) {
        self.rules.push(rule);
    }

    /// Apply all rules to the response.  Returns the (possibly mutated) response
    /// and an outcome indicating whether it was filtered out.
    pub fn apply(&self, mut resp: Response) -> (Response, TransformOutcome) {
        for rule in &self.rules {
            match rule {
                TransformRule::InjectHeader { key, value } => {
                    resp.headers.insert(key.to_lowercase(), value.clone());
                }
                TransformRule::RemoveHeader { key } => {
                    resp.headers.remove(&key.to_lowercase());
                }
                TransformRule::BodyRewrite { find, replace } => {
                    if let Ok(s) = std::str::from_utf8(&resp.body) {
                        let replaced = s.replace(find.as_str(), replace.as_str());
                        resp.body = replaced.into_bytes();
                    }
                }
                TransformRule::StatusMap { from, to } => {
                    if resp.status == *from {
                        resp.status = *to;
                    }
                }
                TransformRule::ContentTypeFilter { allowed } => {
                    let ct = resp
                        .headers
                        .get("content-type")
                        .cloned()
                        .unwrap_or_default();
                    if !ct.contains(allowed.as_str()) {
                        return (resp, TransformOutcome::FilteredOut);
                    }
                }
            }
        }
        (resp, TransformOutcome::Transformed)
    }
}

// ---------------------------------------------------------------------------
// Response aggregation
// ---------------------------------------------------------------------------

/// Strategy for combining multiple responses into one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Pick the first successful (2xx) response.
    FirstSuccess,
    /// Concatenate all bodies, separated by a delimiter.
    ConcatBodies,
    /// Use the response with the smallest body.
    SmallestBody,
    /// Use the response with the largest body.
    LargestBody,
}

/// Aggregator that merges several responses.
#[derive(Debug, Clone)]
pub struct ResponseAggregator {
    strategy: AggregationStrategy,
    delimiter: Vec<u8>,
}

impl ResponseAggregator {
    pub fn new(strategy: AggregationStrategy) -> Self {
        Self {
            strategy,
            delimiter: b"\n".to_vec(),
        }
    }

    pub fn with_delimiter(mut self, delim: &[u8]) -> Self {
        self.delimiter = delim.to_vec();
        self
    }

    /// Aggregate a slice of responses. Returns `None` if the input is empty
    /// or no response matches the strategy criteria.
    pub fn aggregate(&self, responses: &[Response]) -> Option<Response> {
        if responses.is_empty() {
            return None;
        }
        match self.strategy {
            AggregationStrategy::FirstSuccess => {
                responses.iter().find(|r| (200..300).contains(&r.status)).cloned()
            }
            AggregationStrategy::ConcatBodies => {
                let mut combined = Vec::new();
                for (i, r) in responses.iter().enumerate() {
                    if i > 0 {
                        combined.extend_from_slice(&self.delimiter);
                    }
                    combined.extend_from_slice(&r.body);
                }
                let mut out = Response::new(200);
                out.body = combined;
                Some(out)
            }
            AggregationStrategy::SmallestBody => {
                responses.iter().min_by_key(|r| r.body.len()).cloned()
            }
            AggregationStrategy::LargestBody => {
                responses.iter().max_by_key(|r| r.body.len()).cloned()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Conditional transform (apply only when a predicate matches)
// ---------------------------------------------------------------------------

/// Predicate for conditional transforms.
#[derive(Debug, Clone)]
pub enum ResponsePredicate {
    StatusInRange(u16, u16),
    HeaderEquals { key: String, value: String },
    BodyContains(String),
}

impl ResponsePredicate {
    pub fn matches(&self, resp: &Response) -> bool {
        match self {
            Self::StatusInRange(lo, hi) => resp.status >= *lo && resp.status <= *hi,
            Self::HeaderEquals { key, value } => {
                resp.headers.get(&key.to_lowercase()).map(|v| v == value).unwrap_or(false)
            }
            Self::BodyContains(needle) => {
                std::str::from_utf8(&resp.body)
                    .map(|s| s.contains(needle.as_str()))
                    .unwrap_or(false)
            }
        }
    }
}

/// A rule that is only applied when a predicate matches.
#[derive(Debug, Clone)]
pub struct ConditionalRule {
    pub predicate: ResponsePredicate,
    pub rule: TransformRule,
}

/// Extended pipeline supporting conditional rules.
#[derive(Debug, Clone, Default)]
pub struct ConditionalPipeline {
    rules: Vec<ConditionalRule>,
}

impl ConditionalPipeline {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add(&mut self, predicate: ResponsePredicate, rule: TransformRule) {
        self.rules.push(ConditionalRule { predicate, rule });
    }

    pub fn apply(&self, mut resp: Response) -> Response {
        for cr in &self.rules {
            if cr.predicate.matches(&resp) {
                let pipeline = {
                    let mut p = TransformPipeline::new();
                    p.add_rule(cr.rule.clone());
                    p
                };
                let (r, _) = pipeline.apply(resp);
                resp = r;
            }
        }
        resp
    }
}

// ---------------------------------------------------------------------------
// Header allowlist / blocklist
// ---------------------------------------------------------------------------

/// Filters response headers via an allow or block list.
#[derive(Debug, Clone)]
pub enum HeaderFilter {
    Allowlist(Vec<String>),
    Blocklist(Vec<String>),
}

impl HeaderFilter {
    pub fn apply(&self, resp: &mut Response) {
        match self {
            Self::Allowlist(allowed) => {
                let lc: Vec<String> = allowed.iter().map(|s| s.to_lowercase()).collect();
                resp.headers.retain(|k, _| lc.contains(k));
            }
            Self::Blocklist(blocked) => {
                let lc: Vec<String> = blocked.iter().map(|s| s.to_lowercase()).collect();
                resp.headers.retain(|k, _| !lc.contains(k));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Body truncation
// ---------------------------------------------------------------------------

/// Truncates the response body to a maximum byte length.
pub fn truncate_body(resp: &mut Response, max_bytes: usize) {
    if resp.body.len() > max_bytes {
        resp.body.truncate(max_bytes);
    }
}

// ---------------------------------------------------------------------------
// Status class helpers
// ---------------------------------------------------------------------------

pub fn is_informational(status: u16) -> bool {
    (100..200).contains(&status)
}

pub fn is_success(status: u16) -> bool {
    (200..300).contains(&status)
}

pub fn is_redirect(status: u16) -> bool {
    (300..400).contains(&status)
}

pub fn is_client_error(status: u16) -> bool {
    (400..500).contains(&status)
}

pub fn is_server_error(status: u16) -> bool {
    (500..600).contains(&status)
}

pub fn status_class(status: u16) -> &'static str {
    match status {
        100..200 => "informational",
        200..300 => "success",
        300..400 => "redirect",
        400..500 => "client_error",
        500..600 => "server_error",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inject_header() {
        let mut p = TransformPipeline::new();
        p.add_rule(TransformRule::InjectHeader {
            key: "X-Custom".into(),
            value: "hello".into(),
        });
        let resp = Response::new(200);
        let (resp, outcome) = p.apply(resp);
        assert_eq!(outcome, TransformOutcome::Transformed);
        assert_eq!(resp.headers.get("x-custom").unwrap(), "hello");
    }

    #[test]
    fn test_remove_header() {
        let mut p = TransformPipeline::new();
        p.add_rule(TransformRule::RemoveHeader {
            key: "X-Secret".into(),
        });
        let resp = Response::new(200).with_header("x-secret", "abc");
        let (resp, _) = p.apply(resp);
        assert!(!resp.headers.contains_key("x-secret"));
    }

    #[test]
    fn test_body_rewrite() {
        let mut p = TransformPipeline::new();
        p.add_rule(TransformRule::BodyRewrite {
            find: "foo".into(),
            replace: "bar".into(),
        });
        let resp = Response::new(200).with_body(b"hello foo world foo".to_vec());
        let (resp, _) = p.apply(resp);
        assert_eq!(resp.body_as_str().unwrap(), "hello bar world bar");
    }

    #[test]
    fn test_status_map() {
        let mut p = TransformPipeline::new();
        p.add_rule(TransformRule::StatusMap { from: 404, to: 410 });
        let resp = Response::new(404);
        let (resp, _) = p.apply(resp);
        assert_eq!(resp.status, 410);
    }

    #[test]
    fn test_status_map_no_match() {
        let mut p = TransformPipeline::new();
        p.add_rule(TransformRule::StatusMap { from: 404, to: 410 });
        let resp = Response::new(200);
        let (resp, _) = p.apply(resp);
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn test_content_type_filter_pass() {
        let mut p = TransformPipeline::new();
        p.add_rule(TransformRule::ContentTypeFilter {
            allowed: "json".into(),
        });
        let resp = Response::new(200).with_header("content-type", "application/json");
        let (_, outcome) = p.apply(resp);
        assert_eq!(outcome, TransformOutcome::Transformed);
    }

    #[test]
    fn test_content_type_filter_reject() {
        let mut p = TransformPipeline::new();
        p.add_rule(TransformRule::ContentTypeFilter {
            allowed: "json".into(),
        });
        let resp = Response::new(200).with_header("content-type", "text/html");
        let (_, outcome) = p.apply(resp);
        assert_eq!(outcome, TransformOutcome::FilteredOut);
    }

    #[test]
    fn test_aggregator_first_success() {
        let agg = ResponseAggregator::new(AggregationStrategy::FirstSuccess);
        let r1 = Response::new(500).with_body(b"err".to_vec());
        let r2 = Response::new(200).with_body(b"ok".to_vec());
        let r3 = Response::new(201).with_body(b"created".to_vec());
        let result = agg.aggregate(&[r1, r2, r3]).unwrap();
        assert_eq!(result.status, 200);
        assert_eq!(result.body, b"ok");
    }

    #[test]
    fn test_aggregator_first_success_none() {
        let agg = ResponseAggregator::new(AggregationStrategy::FirstSuccess);
        let r1 = Response::new(500);
        let result = agg.aggregate(&[r1]);
        assert!(result.is_none());
    }

    #[test]
    fn test_aggregator_concat() {
        let agg = ResponseAggregator::new(AggregationStrategy::ConcatBodies)
            .with_delimiter(b"|");
        let r1 = Response::new(200).with_body(b"a".to_vec());
        let r2 = Response::new(200).with_body(b"b".to_vec());
        let result = agg.aggregate(&[r1, r2]).unwrap();
        assert_eq!(result.body, b"a|b");
    }

    #[test]
    fn test_aggregator_smallest() {
        let agg = ResponseAggregator::new(AggregationStrategy::SmallestBody);
        let r1 = Response::new(200).with_body(b"long body".to_vec());
        let r2 = Response::new(200).with_body(b"hi".to_vec());
        let result = agg.aggregate(&[r1, r2]).unwrap();
        assert_eq!(result.body, b"hi");
    }

    #[test]
    fn test_aggregator_largest() {
        let agg = ResponseAggregator::new(AggregationStrategy::LargestBody);
        let r1 = Response::new(200).with_body(b"hi".to_vec());
        let r2 = Response::new(200).with_body(b"long body".to_vec());
        let result = agg.aggregate(&[r1, r2]).unwrap();
        assert_eq!(result.body, b"long body");
    }

    #[test]
    fn test_aggregator_empty() {
        let agg = ResponseAggregator::new(AggregationStrategy::ConcatBodies);
        assert!(agg.aggregate(&[]).is_none());
    }

    #[test]
    fn test_conditional_pipeline() {
        let mut cp = ConditionalPipeline::new();
        cp.add(
            ResponsePredicate::StatusInRange(500, 599),
            TransformRule::InjectHeader {
                key: "X-Error".into(),
                value: "true".into(),
            },
        );
        let resp = Response::new(503);
        let resp = cp.apply(resp);
        assert_eq!(resp.headers.get("x-error").unwrap(), "true");

        // 200 should NOT get the header
        let resp2 = Response::new(200);
        let resp2 = cp.apply(resp2);
        assert!(!resp2.headers.contains_key("x-error"));
    }

    #[test]
    fn test_predicate_header_equals() {
        let pred = ResponsePredicate::HeaderEquals {
            key: "X-Mode".into(),
            value: "debug".into(),
        };
        let r1 = Response::new(200).with_header("x-mode", "debug");
        assert!(pred.matches(&r1));
        let r2 = Response::new(200).with_header("x-mode", "prod");
        assert!(!pred.matches(&r2));
    }

    #[test]
    fn test_predicate_body_contains() {
        let pred = ResponsePredicate::BodyContains("error".into());
        let r1 = Response::new(200).with_body(b"an error occurred".to_vec());
        assert!(pred.matches(&r1));
        let r2 = Response::new(200).with_body(b"all good".to_vec());
        assert!(!pred.matches(&r2));
    }

    #[test]
    fn test_header_filter_allowlist() {
        let filter = HeaderFilter::Allowlist(vec!["Content-Type".into(), "X-Req-Id".into()]);
        let mut resp = Response::new(200)
            .with_header("content-type", "text/plain")
            .with_header("x-req-id", "123")
            .with_header("x-secret", "nope");
        filter.apply(&mut resp);
        assert!(resp.headers.contains_key("content-type"));
        assert!(resp.headers.contains_key("x-req-id"));
        assert!(!resp.headers.contains_key("x-secret"));
    }

    #[test]
    fn test_header_filter_blocklist() {
        let filter = HeaderFilter::Blocklist(vec!["X-Secret".into()]);
        let mut resp = Response::new(200)
            .with_header("content-type", "text/plain")
            .with_header("x-secret", "nope");
        filter.apply(&mut resp);
        assert!(resp.headers.contains_key("content-type"));
        assert!(!resp.headers.contains_key("x-secret"));
    }

    #[test]
    fn test_truncate_body() {
        let mut resp = Response::new(200).with_body(b"abcdefghij".to_vec());
        truncate_body(&mut resp, 5);
        assert_eq!(resp.body, b"abcde");
    }

    #[test]
    fn test_truncate_body_no_op() {
        let mut resp = Response::new(200).with_body(b"abc".to_vec());
        truncate_body(&mut resp, 100);
        assert_eq!(resp.body, b"abc");
    }

    #[test]
    fn test_status_helpers() {
        assert!(is_informational(100));
        assert!(is_success(204));
        assert!(is_redirect(301));
        assert!(is_client_error(404));
        assert!(is_server_error(502));
        assert!(!is_success(404));
    }

    #[test]
    fn test_status_class() {
        assert_eq!(status_class(100), "informational");
        assert_eq!(status_class(200), "success");
        assert_eq!(status_class(302), "redirect");
        assert_eq!(status_class(404), "client_error");
        assert_eq!(status_class(500), "server_error");
        assert_eq!(status_class(999), "unknown");
    }

    #[test]
    fn test_body_rewrite_non_utf8_skipped() {
        let mut p = TransformPipeline::new();
        p.add_rule(TransformRule::BodyRewrite {
            find: "x".into(),
            replace: "y".into(),
        });
        let binary = vec![0xFF, 0xFE, 0x00, 0x01];
        let resp = Response::new(200).with_body(binary.clone());
        let (resp, _) = p.apply(resp);
        assert_eq!(resp.body, binary);
    }

    #[test]
    fn test_pipeline_chained_rules() {
        let mut p = TransformPipeline::new();
        p.add_rule(TransformRule::StatusMap { from: 200, to: 204 });
        p.add_rule(TransformRule::InjectHeader {
            key: "X-Transformed".into(),
            value: "yes".into(),
        });
        p.add_rule(TransformRule::BodyRewrite {
            find: "old".into(),
            replace: "new".into(),
        });
        let resp = Response::new(200).with_body(b"old data".to_vec());
        let (resp, outcome) = p.apply(resp);
        assert_eq!(outcome, TransformOutcome::Transformed);
        assert_eq!(resp.status, 204);
        assert_eq!(resp.headers.get("x-transformed").unwrap(), "yes");
        assert_eq!(resp.body_as_str().unwrap(), "new data");
    }

    #[test]
    fn test_response_builder() {
        let resp = Response::new(201)
            .with_header("Content-Type", "text/plain")
            .with_body(b"hello".to_vec());
        assert_eq!(resp.status, 201);
        assert_eq!(resp.headers.get("content-type").unwrap(), "text/plain");
        assert_eq!(resp.body_as_str().unwrap(), "hello");
    }
}
