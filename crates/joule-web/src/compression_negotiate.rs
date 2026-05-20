//! Compression negotiation.
//!
//! Accept-Encoding parsing with quality values, gzip/br/deflate/zstd preference
//! ranking, Content-Encoding selection, identity fallback, and minimum size
//! threshold. Pure Rust — no compression library dependencies; this module
//! handles the _negotiation_, not the compression itself.

use std::fmt;

// ── Encoding ────────────────────────────────────────────────────

/// Supported content encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    Gzip,
    Brotli,
    Deflate,
    Zstd,
    Identity,
}

impl Encoding {
    /// Parse from a token string (case-insensitive).
    pub fn from_token(token: &str) -> Option<Self> {
        match token.trim().to_lowercase().as_str() {
            "gzip" | "x-gzip" => Some(Encoding::Gzip),
            "br" | "brotli" => Some(Encoding::Brotli),
            "deflate" => Some(Encoding::Deflate),
            "zstd" => Some(Encoding::Zstd),
            "identity" => Some(Encoding::Identity),
            _ => None,
        }
    }

    /// Canonical token for the Content-Encoding header.
    pub fn token(&self) -> &'static str {
        match self {
            Encoding::Gzip => "gzip",
            Encoding::Brotli => "br",
            Encoding::Deflate => "deflate",
            Encoding::Zstd => "zstd",
            Encoding::Identity => "identity",
        }
    }

    /// Default server-side preference (lower = more preferred).
    pub fn server_priority(&self) -> u8 {
        match self {
            Encoding::Zstd => 0,
            Encoding::Brotli => 1,
            Encoding::Gzip => 2,
            Encoding::Deflate => 3,
            Encoding::Identity => 4,
        }
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.token())
    }
}

// ── Quality value ───────────────────────────────────────────────

/// An encoding with its quality value from Accept-Encoding.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodingQuality {
    pub encoding: Encoding,
    pub quality: f32,
}

impl EncodingQuality {
    pub fn new(encoding: Encoding, quality: f32) -> Self {
        Self { encoding, quality: quality.clamp(0.0, 1.0) }
    }

    /// Whether this encoding is acceptable (quality > 0).
    pub fn is_acceptable(&self) -> bool {
        self.quality > 0.0
    }
}

// ── Accept-Encoding parsing ─────────────────────────────────────

/// Parse an Accept-Encoding header into encoding-quality pairs.
///
/// Format: `gzip;q=1.0, br;q=0.8, identity;q=0.5, *;q=0.1`
pub fn parse_accept_encoding(header: &str) -> Vec<EncodingQuality> {
    let mut result = Vec::new();
    let mut has_wildcard = false;
    let mut wildcard_quality: f32 = 0.0;

    for part in header.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() { continue; }

        let (token, quality) = parse_token_quality(trimmed);

        if token == "*" {
            has_wildcard = true;
            wildcard_quality = quality;
            continue;
        }

        if let Some(enc) = Encoding::from_token(token) {
            result.push(EncodingQuality::new(enc, quality));
        }
    }

    // If wildcard present, add missing encodings at wildcard quality.
    if has_wildcard {
        let present: Vec<Encoding> = result.iter().map(|eq| eq.encoding).collect();
        let all_encodings = [Encoding::Gzip, Encoding::Brotli, Encoding::Deflate, Encoding::Zstd, Encoding::Identity];
        for enc in &all_encodings {
            if !present.contains(enc) {
                result.push(EncodingQuality::new(*enc, wildcard_quality));
            }
        }
    }

    result
}

/// Parse a single `token;q=value` pair.
fn parse_token_quality(s: &str) -> (&str, f32) {
    let parts: Vec<&str> = s.splitn(2, ';').collect();
    let token = parts[0].trim();
    let quality = if parts.len() > 1 {
        let q_part = parts[1].trim();
        if let Some(val) = q_part.strip_prefix("q=").or_else(|| q_part.strip_prefix("Q=")) {
            val.trim().parse::<f32>().unwrap_or(1.0).clamp(0.0, 1.0)
        } else {
            1.0
        }
    } else {
        1.0
    };
    (token, quality)
}

// ── Negotiation ─────────────────────────────────────────────────

/// Configuration for compression negotiation.
#[derive(Debug, Clone)]
pub struct NegotiationConfig {
    /// Encodings the server supports, in preference order.
    pub supported: Vec<Encoding>,
    /// Minimum response body size (bytes) to apply compression.
    pub min_size: usize,
    /// MIME types that should be compressed.
    pub compressible_types: Vec<String>,
}

impl NegotiationConfig {
    /// Default config: support zstd, br, gzip, deflate; min 1024 bytes.
    pub fn default_config() -> Self {
        Self {
            supported: vec![Encoding::Zstd, Encoding::Brotli, Encoding::Gzip, Encoding::Deflate],
            min_size: 1024,
            compressible_types: vec![
                "text/".to_string(),
                "application/json".to_string(),
                "application/xml".to_string(),
                "application/javascript".to_string(),
                "application/wasm".to_string(),
                "image/svg+xml".to_string(),
            ],
        }
    }

    /// Check if a MIME type is compressible.
    pub fn is_compressible(&self, content_type: &str) -> bool {
        let lower = content_type.to_lowercase();
        self.compressible_types.iter().any(|prefix| lower.starts_with(prefix))
    }
}

/// Result of content encoding negotiation.
#[derive(Debug, Clone, PartialEq)]
pub struct NegotiationResult {
    pub encoding: Encoding,
    pub quality: f32,
    pub should_compress: bool,
    pub reason: NegotiationReason,
}

/// Why a particular encoding was chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegotiationReason {
    /// Client and server agreed on this encoding.
    Negotiated,
    /// Body too small for compression.
    BelowMinSize,
    /// Content type is not compressible.
    NotCompressible,
    /// Client doesn't accept any server-supported encoding.
    NoAcceptableEncoding,
    /// No Accept-Encoding header present.
    NoHeader,
}

impl fmt::Display for NegotiationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NegotiationReason::Negotiated => write!(f, "negotiated"),
            NegotiationReason::BelowMinSize => write!(f, "below_min_size"),
            NegotiationReason::NotCompressible => write!(f, "not_compressible"),
            NegotiationReason::NoAcceptableEncoding => write!(f, "no_acceptable_encoding"),
            NegotiationReason::NoHeader => write!(f, "no_header"),
        }
    }
}

/// Negotiate the best encoding.
pub fn negotiate(
    config: &NegotiationConfig,
    accept_encoding: Option<&str>,
    content_type: &str,
    body_size: usize,
) -> NegotiationResult {
    // No Accept-Encoding header -> identity
    let header = match accept_encoding {
        Some(h) if !h.is_empty() => h,
        _ => {
            return NegotiationResult {
                encoding: Encoding::Identity,
                quality: 1.0,
                should_compress: false,
                reason: NegotiationReason::NoHeader,
            };
        }
    };

    // Content type not compressible -> identity
    if !config.is_compressible(content_type) {
        return NegotiationResult {
            encoding: Encoding::Identity,
            quality: 1.0,
            should_compress: false,
            reason: NegotiationReason::NotCompressible,
        };
    }

    // Body too small -> identity
    if body_size < config.min_size {
        return NegotiationResult {
            encoding: Encoding::Identity,
            quality: 1.0,
            should_compress: false,
            reason: NegotiationReason::BelowMinSize,
        };
    }

    let client_prefs = parse_accept_encoding(header);

    // Find the best encoding: server priority among acceptable client encodings.
    let mut best: Option<(Encoding, f32)> = None;
    for server_enc in &config.supported {
        if let Some(client_eq) = client_prefs.iter().find(|eq| eq.encoding == *server_enc) {
            if client_eq.is_acceptable() {
                match &best {
                    None => best = Some((client_eq.encoding, client_eq.quality)),
                    Some((_, bq)) => {
                        // Prefer higher client quality, then server priority.
                        if client_eq.quality > *bq
                            || (client_eq.quality == *bq
                                && client_eq.encoding.server_priority() < best.unwrap().0.server_priority())
                        {
                            best = Some((client_eq.encoding, client_eq.quality));
                        }
                    }
                }
            }
        }
    }

    match best {
        Some((enc, q)) => NegotiationResult {
            encoding: enc,
            quality: q,
            should_compress: true,
            reason: NegotiationReason::Negotiated,
        },
        None => NegotiationResult {
            encoding: Encoding::Identity,
            quality: 1.0,
            should_compress: false,
            reason: NegotiationReason::NoAcceptableEncoding,
        },
    }
}

/// Build a Content-Encoding header value.
pub fn content_encoding_header(encoding: Encoding) -> Option<String> {
    match encoding {
        Encoding::Identity => None,
        enc => Some(enc.token().to_string()),
    }
}

/// Build a Vary header that includes Accept-Encoding.
pub fn vary_accept_encoding(existing_vary: Option<&str>) -> String {
    match existing_vary {
        Some(v) if v.to_lowercase().contains("accept-encoding") => v.to_string(),
        Some(v) => format!("{v}, Accept-Encoding"),
        None => "Accept-Encoding".to_string(),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_from_token() {
        assert_eq!(Encoding::from_token("gzip"), Some(Encoding::Gzip));
        assert_eq!(Encoding::from_token("x-gzip"), Some(Encoding::Gzip));
        assert_eq!(Encoding::from_token("br"), Some(Encoding::Brotli));
        assert_eq!(Encoding::from_token("brotli"), Some(Encoding::Brotli));
        assert_eq!(Encoding::from_token("deflate"), Some(Encoding::Deflate));
        assert_eq!(Encoding::from_token("zstd"), Some(Encoding::Zstd));
        assert_eq!(Encoding::from_token("identity"), Some(Encoding::Identity));
        assert_eq!(Encoding::from_token("unknown"), None);
    }

    #[test]
    fn test_encoding_token() {
        assert_eq!(Encoding::Gzip.token(), "gzip");
        assert_eq!(Encoding::Brotli.token(), "br");
        assert_eq!(Encoding::Deflate.token(), "deflate");
        assert_eq!(Encoding::Zstd.token(), "zstd");
        assert_eq!(Encoding::Identity.token(), "identity");
    }

    #[test]
    fn test_encoding_display() {
        assert_eq!(Encoding::Brotli.to_string(), "br");
    }

    #[test]
    fn test_encoding_server_priority() {
        assert!(Encoding::Zstd.server_priority() < Encoding::Brotli.server_priority());
        assert!(Encoding::Brotli.server_priority() < Encoding::Gzip.server_priority());
        assert!(Encoding::Gzip.server_priority() < Encoding::Deflate.server_priority());
        assert!(Encoding::Deflate.server_priority() < Encoding::Identity.server_priority());
    }

    #[test]
    fn test_encoding_quality() {
        let eq = EncodingQuality::new(Encoding::Gzip, 0.8);
        assert!(eq.is_acceptable());
        let eq_zero = EncodingQuality::new(Encoding::Gzip, 0.0);
        assert!(!eq_zero.is_acceptable());
    }

    #[test]
    fn test_encoding_quality_clamp() {
        let eq = EncodingQuality::new(Encoding::Gzip, 1.5);
        assert!((eq.quality - 1.0).abs() < 0.001);
        let eq_neg = EncodingQuality::new(Encoding::Gzip, -0.5);
        assert!((eq_neg.quality - 0.0).abs() < 0.001);
    }

    // ── Parsing ─────────────────────────────────────────────

    #[test]
    fn test_parse_accept_encoding_simple() {
        let result = parse_accept_encoding("gzip, br, deflate");
        assert_eq!(result.len(), 3);
        assert!(result.iter().any(|eq| eq.encoding == Encoding::Gzip && (eq.quality - 1.0).abs() < 0.001));
        assert!(result.iter().any(|eq| eq.encoding == Encoding::Brotli));
    }

    #[test]
    fn test_parse_accept_encoding_with_quality() {
        let result = parse_accept_encoding("gzip;q=1.0, br;q=0.8, deflate;q=0.5");
        let gzip = result.iter().find(|eq| eq.encoding == Encoding::Gzip).unwrap();
        let br = result.iter().find(|eq| eq.encoding == Encoding::Brotli).unwrap();
        let deflate = result.iter().find(|eq| eq.encoding == Encoding::Deflate).unwrap();
        assert!((gzip.quality - 1.0).abs() < 0.001);
        assert!((br.quality - 0.8).abs() < 0.001);
        assert!((deflate.quality - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_accept_encoding_wildcard() {
        let result = parse_accept_encoding("gzip, *;q=0.1");
        // Should have gzip at 1.0 plus all others at 0.1
        assert!(result.len() >= 5);
        let br = result.iter().find(|eq| eq.encoding == Encoding::Brotli).unwrap();
        assert!((br.quality - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_parse_accept_encoding_empty() {
        let result = parse_accept_encoding("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_accept_encoding_zero_quality() {
        let result = parse_accept_encoding("gzip;q=0, br;q=1.0");
        let gzip = result.iter().find(|eq| eq.encoding == Encoding::Gzip).unwrap();
        assert!(!gzip.is_acceptable());
    }

    // ── Negotiation ─────────────────────────────────────────

    #[test]
    fn test_negotiate_basic() {
        let config = NegotiationConfig::default_config();
        let result = negotiate(&config, Some("gzip, br"), "text/html", 5000);
        assert!(result.should_compress);
        assert_eq!(result.reason, NegotiationReason::Negotiated);
        // Both at q=1.0 but server prefers br
        assert_eq!(result.encoding, Encoding::Brotli);
    }

    #[test]
    fn test_negotiate_gzip_only() {
        let config = NegotiationConfig::default_config();
        let result = negotiate(&config, Some("gzip"), "application/json", 5000);
        assert!(result.should_compress);
        assert_eq!(result.encoding, Encoding::Gzip);
    }

    #[test]
    fn test_negotiate_no_header() {
        let config = NegotiationConfig::default_config();
        let result = negotiate(&config, None, "text/html", 5000);
        assert!(!result.should_compress);
        assert_eq!(result.encoding, Encoding::Identity);
        assert_eq!(result.reason, NegotiationReason::NoHeader);
    }

    #[test]
    fn test_negotiate_empty_header() {
        let config = NegotiationConfig::default_config();
        let result = negotiate(&config, Some(""), "text/html", 5000);
        assert_eq!(result.reason, NegotiationReason::NoHeader);
    }

    #[test]
    fn test_negotiate_below_min_size() {
        let config = NegotiationConfig::default_config();
        let result = negotiate(&config, Some("gzip"), "text/html", 500);
        assert!(!result.should_compress);
        assert_eq!(result.reason, NegotiationReason::BelowMinSize);
    }

    #[test]
    fn test_negotiate_not_compressible() {
        let config = NegotiationConfig::default_config();
        let result = negotiate(&config, Some("gzip"), "image/png", 50000);
        assert!(!result.should_compress);
        assert_eq!(result.reason, NegotiationReason::NotCompressible);
    }

    #[test]
    fn test_negotiate_no_acceptable() {
        let config = NegotiationConfig::default_config();
        let result = negotiate(&config, Some("compress"), "text/html", 5000);
        assert!(!result.should_compress);
        assert_eq!(result.reason, NegotiationReason::NoAcceptableEncoding);
    }

    #[test]
    fn test_negotiate_client_preference() {
        let config = NegotiationConfig::default_config();
        // Client explicitly prefers gzip over br
        let result = negotiate(&config, Some("gzip;q=1.0, br;q=0.5"), "text/html", 5000);
        assert_eq!(result.encoding, Encoding::Gzip);
    }

    #[test]
    fn test_negotiate_zstd() {
        let config = NegotiationConfig::default_config();
        let result = negotiate(&config, Some("zstd, gzip, br"), "text/html", 5000);
        assert!(result.should_compress);
        // All at q=1.0, server prefers zstd
        assert_eq!(result.encoding, Encoding::Zstd);
    }

    // ── Config ──────────────────────────────────────────────

    #[test]
    fn test_config_compressible() {
        let config = NegotiationConfig::default_config();
        assert!(config.is_compressible("text/html"));
        assert!(config.is_compressible("text/plain; charset=utf-8"));
        assert!(config.is_compressible("application/json"));
        assert!(config.is_compressible("application/wasm"));
        assert!(config.is_compressible("image/svg+xml"));
        assert!(!config.is_compressible("image/png"));
        assert!(!config.is_compressible("application/octet-stream"));
    }

    // ── Helpers ─────────────────────────────────────────────

    #[test]
    fn test_content_encoding_header() {
        assert_eq!(content_encoding_header(Encoding::Gzip), Some("gzip".into()));
        assert_eq!(content_encoding_header(Encoding::Brotli), Some("br".into()));
        assert_eq!(content_encoding_header(Encoding::Identity), None);
    }

    #[test]
    fn test_vary_accept_encoding_none() {
        assert_eq!(vary_accept_encoding(None), "Accept-Encoding");
    }

    #[test]
    fn test_vary_accept_encoding_existing() {
        assert_eq!(
            vary_accept_encoding(Some("Cookie")),
            "Cookie, Accept-Encoding"
        );
    }

    #[test]
    fn test_vary_accept_encoding_already_present() {
        let existing = "Accept-Encoding, Cookie";
        assert_eq!(vary_accept_encoding(Some(existing)), existing);
    }

    #[test]
    fn test_negotiation_reason_display() {
        assert_eq!(NegotiationReason::Negotiated.to_string(), "negotiated");
        assert_eq!(NegotiationReason::BelowMinSize.to_string(), "below_min_size");
        assert_eq!(NegotiationReason::NotCompressible.to_string(), "not_compressible");
        assert_eq!(NegotiationReason::NoAcceptableEncoding.to_string(), "no_acceptable_encoding");
        assert_eq!(NegotiationReason::NoHeader.to_string(), "no_header");
    }
}
