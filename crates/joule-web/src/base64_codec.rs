//! Base64 encoding and decoding (RFC 4648).
//!
//! Standard encoding, URL-safe encoding, padding/no-padding variants,
//! streaming encoder/decoder, chunk encoding, validation, and
//! constant-time comparison for security contexts.

// ── Alphabet Tables ──────────────────────────────────────────────

const STANDARD_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const URL_SAFE_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn decode_table(alphabet: &[u8; 64]) -> [u8; 256] {
    let mut table = [0xFFu8; 256];
    for (i, &b) in alphabet.iter().enumerate() {
        table[b as usize] = i as u8;
    }
    table
}

// ── Config ───────────────────────────────────────────────────────

/// Encoding variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Standard,
    UrlSafe,
}

/// Padding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    Pad,
    NoPad,
}

/// Full encoding configuration.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub variant: Variant,
    pub padding: Padding,
}

impl Config {
    pub const STANDARD: Self = Self { variant: Variant::Standard, padding: Padding::Pad };
    pub const STANDARD_NO_PAD: Self = Self { variant: Variant::Standard, padding: Padding::NoPad };
    pub const URL_SAFE: Self = Self { variant: Variant::UrlSafe, padding: Padding::Pad };
    pub const URL_SAFE_NO_PAD: Self = Self { variant: Variant::UrlSafe, padding: Padding::NoPad };

    fn alphabet(&self) -> &'static [u8; 64] {
        match self.variant {
            Variant::Standard => STANDARD_ALPHABET,
            Variant::UrlSafe => URL_SAFE_ALPHABET,
        }
    }
}

// ── Encode ───────────────────────────────────────────────────────

/// Encode bytes to base64 string using the given config.
pub fn encode(data: &[u8], config: Config) -> String {
    let alphabet = config.alphabet();
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    let chunks = data.chunks(3);

    for chunk in chunks {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };

        let n = (b0 << 16) | (b1 << 8) | b2;

        out.push(alphabet[((n >> 18) & 0x3F) as usize] as char);
        out.push(alphabet[((n >> 12) & 0x3F) as usize] as char);

        if chunk.len() > 1 {
            out.push(alphabet[((n >> 6) & 0x3F) as usize] as char);
        } else if config.padding == Padding::Pad {
            out.push('=');
        }

        if chunk.len() > 2 {
            out.push(alphabet[(n & 0x3F) as usize] as char);
        } else if config.padding == Padding::Pad {
            out.push('=');
        }
    }
    out
}

/// Encode with standard alphabet and padding.
pub fn encode_standard(data: &[u8]) -> String {
    encode(data, Config::STANDARD)
}

/// Encode with URL-safe alphabet, no padding.
pub fn encode_url_safe(data: &[u8]) -> String {
    encode(data, Config::URL_SAFE_NO_PAD)
}

// ── Decode ───────────────────────────────────────────────────────

/// Error during decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    InvalidChar(u8),
    InvalidLength,
    InvalidPadding,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::InvalidChar(c) => write!(f, "invalid base64 character: 0x{:02X}", c),
            DecodeError::InvalidLength => write!(f, "invalid base64 length"),
            DecodeError::InvalidPadding => write!(f, "invalid padding"),
        }
    }
}

/// Decode a base64 string with the given config.
pub fn decode(encoded: &str, config: Config) -> Result<Vec<u8>, DecodeError> {
    let table = decode_table(config.alphabet());
    // Strip whitespace and padding
    let input: Vec<u8> = encoded.bytes().filter(|b| *b != b'\n' && *b != b'\r' && *b != b' ').collect();

    // Find where padding starts
    let pad_start = input.iter().position(|b| *b == b'=').unwrap_or(input.len());
    let data = &input[..pad_start];

    let mut out = Vec::with_capacity(data.len() * 3 / 4);
    let mut buf = [0u8; 4];
    let mut buf_len = 0;

    for &byte in data {
        let val = table[byte as usize];
        if val == 0xFF {
            return Err(DecodeError::InvalidChar(byte));
        }
        buf[buf_len] = val;
        buf_len += 1;

        if buf_len == 4 {
            let n = (buf[0] as u32) << 18 | (buf[1] as u32) << 12
                  | (buf[2] as u32) << 6 | (buf[3] as u32);
            out.push((n >> 16) as u8);
            out.push((n >> 8) as u8);
            out.push(n as u8);
            buf_len = 0;
        }
    }

    // Handle remaining
    match buf_len {
        0 => {}
        2 => {
            let n = (buf[0] as u32) << 18 | (buf[1] as u32) << 12;
            out.push((n >> 16) as u8);
        }
        3 => {
            let n = (buf[0] as u32) << 18 | (buf[1] as u32) << 12 | (buf[2] as u32) << 6;
            out.push((n >> 16) as u8);
            out.push((n >> 8) as u8);
        }
        _ => return Err(DecodeError::InvalidLength),
    }

    Ok(out)
}

/// Decode standard base64.
pub fn decode_standard(encoded: &str) -> Result<Vec<u8>, DecodeError> {
    decode(encoded, Config::STANDARD)
}

/// Decode URL-safe base64.
pub fn decode_url_safe(encoded: &str) -> Result<Vec<u8>, DecodeError> {
    decode(encoded, Config::URL_SAFE_NO_PAD)
}

// ── Validation ───────────────────────────────────────────────────

/// Check if a string is valid base64 with the given config.
pub fn is_valid(encoded: &str, config: Config) -> bool {
    decode(encoded, config).is_ok()
}

// ── Chunk Encoding ───────────────────────────────────────────────

/// Encode and split into fixed-width lines (e.g., 76 chars for MIME).
pub fn encode_chunked(data: &[u8], config: Config, line_len: usize) -> String {
    let encoded = encode(data, config);
    let mut out = String::with_capacity(encoded.len() + encoded.len() / line_len * 2);
    for (i, ch) in encoded.chars().enumerate() {
        if i > 0 && i % line_len == 0 { out.push('\n'); }
        out.push(ch);
    }
    out
}

// ── Streaming Encoder ────────────────────────────────────────────

/// Streaming base64 encoder — feed bytes incrementally, flush at end.
pub struct StreamEncoder {
    config: Config,
    buffer: Vec<u8>,
    output: String,
}

impl StreamEncoder {
    pub fn new(config: Config) -> Self {
        Self { config, buffer: Vec::new(), output: String::new() }
    }

    /// Feed more bytes.
    pub fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        // Encode complete 3-byte groups
        let complete = self.buffer.len() / 3 * 3;
        if complete > 0 {
            let chunk: Vec<u8> = self.buffer.drain(..complete).collect();
            // Encode without padding for intermediate chunks
            let cfg = Config { padding: Padding::NoPad, ..self.config };
            self.output.push_str(&encode(&chunk, cfg));
        }
    }

    /// Finalize and return the complete encoded string.
    pub fn finish(mut self) -> String {
        if !self.buffer.is_empty() {
            self.output.push_str(&encode(&self.buffer, self.config));
        } else if self.config.padding == Padding::Pad {
            // Nothing extra needed
        }
        self.output
    }
}

/// Streaming base64 decoder.
pub struct StreamDecoder {
    config: Config,
    buffer: String,
    output: Vec<u8>,
}

impl StreamDecoder {
    pub fn new(config: Config) -> Self {
        Self { config, buffer: String::new(), output: Vec::new() }
    }

    pub fn update(&mut self, data: &str) -> Result<(), DecodeError> {
        self.buffer.push_str(data);
        // Decode complete 4-char groups
        let complete = self.buffer.len() / 4 * 4;
        if complete > 0 {
            let chunk: String = self.buffer.drain(..complete).collect();
            let decoded = decode(&chunk, self.config)?;
            self.output.extend_from_slice(&decoded);
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Vec<u8>, DecodeError> {
        if !self.buffer.is_empty() {
            let decoded = decode(&self.buffer, self.config)?;
            self.output.extend_from_slice(&decoded);
        }
        Ok(self.output)
    }
}

// ── Constant-time comparison ─────────────────────────────────────

/// Compare two base64 strings in constant time (for security contexts).
/// Returns true only if both decode to identical bytes.
pub fn constant_time_eq(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    if a_bytes.len() != b_bytes.len() { return false; }
    let mut result: u8 = 0;
    for (x, y) in a_bytes.iter().zip(b_bytes.iter()) {
        result |= x ^ y;
    }
    result == 0
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_empty() {
        assert_eq!(encode_standard(b""), "");
    }

    #[test]
    fn test_encode_standard() {
        assert_eq!(encode_standard(b"Hello, World!"), "SGVsbG8sIFdvcmxkIQ==");
    }

    #[test]
    fn test_decode_standard() {
        let decoded = decode_standard("SGVsbG8sIFdvcmxkIQ==").unwrap();
        assert_eq!(decoded, b"Hello, World!");
    }

    #[test]
    fn test_roundtrip() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let encoded = encode_standard(data);
        let decoded = decode_standard(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_url_safe() {
        let data = vec![0xFF, 0xFE, 0xFD, 0xFC];
        let encoded = encode_url_safe(&data);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        let decoded = decode_url_safe(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_no_padding() {
        let encoded = encode(b"ab", Config::STANDARD_NO_PAD);
        assert!(!encoded.contains('='));
        let decoded = decode(&encoded, Config::STANDARD_NO_PAD).unwrap();
        assert_eq!(decoded, b"ab");
    }

    #[test]
    fn test_one_byte_padding() {
        // 1 byte => 2 base64 chars + 2 padding
        assert_eq!(encode_standard(b"a"), "YQ==");
    }

    #[test]
    fn test_two_byte_padding() {
        // 2 bytes => 3 base64 chars + 1 padding
        assert_eq!(encode_standard(b"ab"), "YWI=");
    }

    #[test]
    fn test_validation() {
        assert!(is_valid("SGVsbG8=", Config::STANDARD));
        assert!(!is_valid("SGVs!!!", Config::STANDARD));
    }

    #[test]
    fn test_chunked() {
        let data = b"The quick brown fox jumps over the lazy dog and more text to exceed the line limit";
        let chunked = encode_chunked(data, Config::STANDARD, 20);
        assert!(chunked.contains('\n'));
        // Remove newlines and decode
        let rejoined: String = chunked.chars().filter(|c| *c != '\n').collect();
        let decoded = decode_standard(&rejoined).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_streaming_encoder() {
        let mut enc = StreamEncoder::new(Config::STANDARD);
        enc.update(b"Hello");
        enc.update(b", World!");
        let result = enc.finish();
        assert_eq!(decode_standard(&result).unwrap(), b"Hello, World!");
    }

    #[test]
    fn test_streaming_decoder() {
        let full = encode_standard(b"Hello, World!");
        let mut dec = StreamDecoder::new(Config::STANDARD);
        // Feed in chunks of 4
        for chunk in full.as_bytes().chunks(4) {
            dec.update(std::str::from_utf8(chunk).unwrap()).unwrap();
        }
        assert_eq!(dec.finish().unwrap(), b"Hello, World!");
    }

    #[test]
    fn test_constant_time_eq() {
        let a = encode_standard(b"secret");
        let b = encode_standard(b"secret");
        assert!(constant_time_eq(&a, &b));
        let c = encode_standard(b"Secret");
        assert!(!constant_time_eq(&a, &c));
    }

    #[test]
    fn test_invalid_char() {
        let err = decode_standard("!!!!").unwrap_err();
        assert!(matches!(err, DecodeError::InvalidChar(_)));
    }

    #[test]
    fn test_binary_roundtrip() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = encode_standard(&data);
        let decoded = decode_standard(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
}
