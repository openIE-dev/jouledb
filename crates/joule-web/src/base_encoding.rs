//! Base encoding schemes — Base64, Base32, Base16, Base58.
//!
//! Implements RFC 4648 Base64 (standard + URL-safe), Base32, Base16 (hex),
//! and Bitcoin-alphabet Base58. Replaces btoa/atob and npm base64/base32
//! packages with a pure Rust implementation.

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during base encoding/decoding.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BaseEncodingError {
    #[error("invalid character '{0}' at position {1}")]
    InvalidCharacter(char, usize),
    #[error("invalid padding")]
    InvalidPadding,
    #[error("invalid input length")]
    InvalidLength,
}

// ── Base64 ───────────────────────────────────────────────────────────

const BASE64_STANDARD: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const BASE64_URL_SAFE: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn base64_decode_table(alphabet: &[u8; 64]) -> [u8; 256] {
    let mut table = [0xFFu8; 256];
    for (i, &c) in alphabet.iter().enumerate() {
        table[c as usize] = i as u8;
    }
    table
}

/// Encode bytes to Base64 (standard alphabet, with padding).
pub fn base64_encode(data: &[u8]) -> String {
    base64_encode_with(data, BASE64_STANDARD, true)
}

/// Encode bytes to URL-safe Base64 (no padding).
pub fn base64_url_encode(data: &[u8]) -> String {
    base64_encode_with(data, BASE64_URL_SAFE, false)
}

/// Encode bytes to Base64 with options.
pub fn base64_encode_with(data: &[u8], alphabet: &[u8; 64], pad: bool) -> String {
    let mut result = Vec::with_capacity((data.len() + 2) / 3 * 4);
    let chunks = data.chunks(3);
    for chunk in chunks {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(alphabet[((triple >> 18) & 0x3F) as usize]);
        result.push(alphabet[((triple >> 12) & 0x3F) as usize]);

        if chunk.len() > 1 {
            result.push(alphabet[((triple >> 6) & 0x3F) as usize]);
        } else if pad {
            result.push(b'=');
        }

        if chunk.len() > 2 {
            result.push(alphabet[(triple & 0x3F) as usize]);
        } else if pad {
            result.push(b'=');
        }
    }
    // SAFETY: alphabet is ASCII.
    unsafe { String::from_utf8_unchecked(result) }
}

/// Decode Base64 string (standard alphabet).
pub fn base64_decode(input: &str) -> Result<Vec<u8>, BaseEncodingError> {
    base64_decode_with(input, BASE64_STANDARD)
}

/// Decode URL-safe Base64 string.
pub fn base64_url_decode(input: &str) -> Result<Vec<u8>, BaseEncodingError> {
    base64_decode_with(input, BASE64_URL_SAFE)
}

/// Decode Base64 with given alphabet.
pub fn base64_decode_with(input: &str, alphabet: &[u8; 64]) -> Result<Vec<u8>, BaseEncodingError> {
    let table = base64_decode_table(alphabet);
    let input = input.trim_end_matches('=');
    let bytes = input.as_bytes();

    let mut result = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;

    for (i, &b) in bytes.iter().enumerate() {
        let val = table[b as usize];
        if val == 0xFF {
            return Err(BaseEncodingError::InvalidCharacter(b as char, i));
        }
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Ok(result)
}

/// Encode Base64 with MIME line wrapping (76 chars per line).
pub fn base64_mime_encode(data: &[u8]) -> String {
    let encoded = base64_encode(data);
    let mut result = String::with_capacity(encoded.len() + encoded.len() / 76 * 2);
    for (i, ch) in encoded.chars().enumerate() {
        if i > 0 && i % 76 == 0 {
            result.push_str("\r\n");
        }
        result.push(ch);
    }
    result
}

// ── Base32 ───────────────────────────────────────────────────────────

const BASE32_ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

/// Encode bytes to Base32 (RFC 4648).
pub fn base32_encode(data: &[u8]) -> String {
    let mut result = Vec::new();
    let mut buf = 0u64;
    let mut bits = 0u32;

    for &b in data {
        buf = (buf << 8) | b as u64;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            result.push(BASE32_ALPHABET[((buf >> bits) & 0x1F) as usize]);
            buf &= (1u64 << bits) - 1;
        }
    }

    if bits > 0 {
        result.push(BASE32_ALPHABET[((buf << (5 - bits)) & 0x1F) as usize]);
    }

    // Pad to multiple of 8.
    while result.len() % 8 != 0 {
        result.push(b'=');
    }

    unsafe { String::from_utf8_unchecked(result) }
}

/// Decode Base32 string (RFC 4648).
pub fn base32_decode(input: &str) -> Result<Vec<u8>, BaseEncodingError> {
    let input = input.trim_end_matches('=');
    let mut decode_table = [0xFFu8; 256];
    for (i, &c) in BASE32_ALPHABET.iter().enumerate() {
        decode_table[c as usize] = i as u8;
        decode_table[c.to_ascii_lowercase() as usize] = i as u8;
    }

    let mut result = Vec::new();
    let mut buf = 0u64;
    let mut bits = 0u32;

    for (i, &b) in input.as_bytes().iter().enumerate() {
        let val = decode_table[b as usize];
        if val == 0xFF {
            return Err(BaseEncodingError::InvalidCharacter(b as char, i));
        }
        buf = (buf << 5) | val as u64;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1u64 << bits) - 1;
        }
    }

    Ok(result)
}

// ── Base16 (Hex) ─────────────────────────────────────────────────────

/// Encode bytes to hexadecimal (lowercase).
pub fn base16_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len() * 2);
    for &b in data {
        result.push(HEX_LOWER[(b >> 4) as usize] as char);
        result.push(HEX_LOWER[(b & 0x0F) as usize] as char);
    }
    result
}

const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";

/// Decode hexadecimal string to bytes.
pub fn base16_decode(input: &str) -> Result<Vec<u8>, BaseEncodingError> {
    let bytes = input.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err(BaseEncodingError::InvalidLength);
    }
    let mut result = Vec::with_capacity(bytes.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        let hi = hex_val(bytes[i], i)?;
        let lo = hex_val(bytes[i + 1], i + 1)?;
        result.push((hi << 4) | lo);
    }
    Ok(result)
}

fn hex_val(c: u8, pos: usize) -> Result<u8, BaseEncodingError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(BaseEncodingError::InvalidCharacter(c as char, pos)),
    }
}

// ── Base58 (Bitcoin) ─────────────────────────────────────────────────

const BASE58_ALPHABET: &[u8; 58] =
    b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Encode bytes to Base58 (Bitcoin alphabet, no padding).
pub fn base58_encode(data: &[u8]) -> String {
    if data.is_empty() {
        return String::new();
    }

    // Count leading zeros.
    let leading_zeros = data.iter().take_while(|&&b| b == 0).count();

    // Convert to base58 using repeated division.
    let mut digits: Vec<u8> = Vec::new();
    for &byte in data {
        let mut carry = byte as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) * 256;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }

    let mut result = String::with_capacity(leading_zeros + digits.len());
    for _ in 0..leading_zeros {
        result.push('1');
    }
    for &d in digits.iter().rev() {
        result.push(BASE58_ALPHABET[d as usize] as char);
    }
    result
}

/// Decode Base58 string to bytes.
pub fn base58_decode(input: &str) -> Result<Vec<u8>, BaseEncodingError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut decode_table = [0xFFu8; 256];
    for (i, &c) in BASE58_ALPHABET.iter().enumerate() {
        decode_table[c as usize] = i as u8;
    }

    let leading_ones = input.bytes().take_while(|b| *b == b'1').count();

    let mut digits: Vec<u8> = Vec::new();
    for (i, &b) in input.as_bytes().iter().enumerate() {
        let val = decode_table[b as usize];
        if val == 0xFF {
            return Err(BaseEncodingError::InvalidCharacter(b as char, i));
        }
        let mut carry = val as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) * 58;
            *d = (carry % 256) as u8;
            carry /= 256;
        }
        while carry > 0 {
            digits.push((carry % 256) as u8);
            carry /= 256;
        }
    }

    let mut result = Vec::with_capacity(leading_ones + digits.len());
    for _ in 0..leading_ones {
        result.push(0);
    }
    for &d in digits.iter().rev() {
        result.push(d);
    }
    Ok(result)
}

// ── Validation ───────────────────────────────────────────────────────

/// Validate a Base64 encoded string.
pub fn is_valid_base64(input: &str) -> bool {
    let input = input.trim_end_matches('=');
    input.bytes().all(|b| matches!(b,
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'+' | b'/' | b'-' | b'_'
    ))
}

/// Validate a Base32 encoded string.
pub fn is_valid_base32(input: &str) -> bool {
    let input = input.trim_end_matches('=');
    input.bytes().all(|b| matches!(b,
        b'A'..=b'Z' | b'a'..=b'z' | b'2'..=b'7'
    ))
}

/// Validate a Base16 (hex) string.
pub fn is_valid_base16(input: &str) -> bool {
    input.len() % 2 == 0
        && input.bytes().all(|b| matches!(b,
            b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F'
        ))
}

/// Validate a Base58 string.
pub fn is_valid_base58(input: &str) -> bool {
    input.bytes().all(|b| BASE58_ALPHABET.contains(&b))
}

// ── Streaming ────────────────────────────────────────────────────────

/// Streaming Base64 encoder — processes chunks of input.
pub struct StreamingBase64Encoder {
    remainder: Vec<u8>,
    output: String,
}

impl StreamingBase64Encoder {
    pub fn new() -> Self {
        Self {
            remainder: Vec::new(),
            output: String::new(),
        }
    }

    /// Feed a chunk of data.
    pub fn update(&mut self, data: &[u8]) {
        self.remainder.extend_from_slice(data);
        let usable = self.remainder.len() / 3 * 3;
        if usable > 0 {
            let to_encode: Vec<u8> = self.remainder.drain(..usable).collect();
            self.output.push_str(&base64_encode(&to_encode));
        }
    }

    /// Finish and return the complete encoded string.
    pub fn finish(mut self) -> String {
        if !self.remainder.is_empty() {
            self.output.push_str(&base64_encode(&self.remainder));
        }
        self.output
    }
}

/// Streaming Base64 decoder.
pub struct StreamingBase64Decoder {
    remainder: String,
    output: Vec<u8>,
}

impl StreamingBase64Decoder {
    pub fn new() -> Self {
        Self {
            remainder: String::new(),
            output: Vec::new(),
        }
    }

    /// Feed a chunk of encoded data.
    pub fn update(&mut self, data: &str) -> Result<(), BaseEncodingError> {
        self.remainder.push_str(data);
        let usable = self.remainder.len() / 4 * 4;
        if usable > 0 {
            let to_decode: String = self.remainder.drain(..usable).collect();
            let decoded = base64_decode(&to_decode)?;
            self.output.extend_from_slice(&decoded);
        }
        Ok(())
    }

    /// Finish and return the decoded bytes.
    pub fn finish(mut self) -> Result<Vec<u8>, BaseEncodingError> {
        if !self.remainder.is_empty() {
            let decoded = base64_decode(&self.remainder)?;
            self.output.extend_from_slice(&decoded);
        }
        Ok(self.output)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip_empty() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_decode("").unwrap(), b"");
    }

    #[test]
    fn base64_roundtrip_hello() {
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
        assert_eq!(base64_decode("SGVsbG8=").unwrap(), b"Hello");
    }

    #[test]
    fn base64_rfc_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_url_safe() {
        let data = &[0xFF, 0xFE, 0xFD];
        let encoded = base64_url_encode(data);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        let decoded = base64_url_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn base64_mime_wrapping() {
        let data = vec![0u8; 100];
        let encoded = base64_mime_encode(&data);
        for line in encoded.split("\r\n") {
            assert!(line.len() <= 76);
        }
    }

    #[test]
    fn base32_roundtrip() {
        assert_eq!(base32_encode(b""), "");
        assert_eq!(base32_encode(b"f"), "MY======");
        assert_eq!(base32_encode(b"fo"), "MZXQ====");
        assert_eq!(base32_encode(b"foo"), "MZXW6===");
        assert_eq!(base32_encode(b"foob"), "MZXW6YQ=");
        assert_eq!(base32_encode(b"fooba"), "MZXW6YTB");
        assert_eq!(base32_encode(b"foobar"), "MZXW6YTBOI======");

        assert_eq!(base32_decode("MZXW6YTBOI======").unwrap(), b"foobar");
        assert_eq!(base32_decode("MZXW6YTB").unwrap(), b"fooba");
    }

    #[test]
    fn base16_roundtrip() {
        assert_eq!(base16_encode(b""), "");
        assert_eq!(base16_encode(b"\xDE\xAD"), "dead");
        assert_eq!(base16_decode("dead").unwrap(), vec![0xDE, 0xAD]);
        assert_eq!(base16_decode("DEAD").unwrap(), vec![0xDE, 0xAD]);
    }

    #[test]
    fn base58_roundtrip() {
        let data = b"Hello World";
        let encoded = base58_encode(data);
        let decoded = base58_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn base58_leading_zeros() {
        let data = vec![0, 0, 0, 1, 2, 3];
        let encoded = base58_encode(&data);
        assert!(encoded.starts_with("111"));
        let decoded = base58_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn validation() {
        assert!(is_valid_base64("SGVsbG8="));
        assert!(!is_valid_base64("SGVs bG8="));
        assert!(is_valid_base32("MZXW6YTB"));
        assert!(!is_valid_base32("MZXW8YTB"));
        assert!(is_valid_base16("deadbeef"));
        assert!(!is_valid_base16("deadbeeg"));
        assert!(is_valid_base58("JxF12TrwUP45BMd"));
        assert!(!is_valid_base58("JxF12TrwUP45BMd0")); // '0' not in base58
    }

    #[test]
    fn base64_invalid_char() {
        let result = base64_decode("!!!!");
        assert!(result.is_err());
    }

    #[test]
    fn streaming_base64_encoder() {
        let mut encoder = StreamingBase64Encoder::new();
        encoder.update(b"Hello");
        encoder.update(b", World!");
        let result = encoder.finish();
        assert_eq!(result, base64_encode(b"Hello, World!"));
    }

    #[test]
    fn streaming_base64_decoder() {
        let encoded = base64_encode(b"Hello, World!");
        let mut decoder = StreamingBase64Decoder::new();
        // Feed in chunks of 4.
        for chunk in encoded.as_bytes().chunks(4) {
            decoder.update(std::str::from_utf8(chunk).unwrap()).unwrap();
        }
        let result = decoder.finish().unwrap();
        assert_eq!(result, b"Hello, World!");
    }

    #[test]
    fn base64_binary_roundtrip() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = base64_encode(&data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
}
