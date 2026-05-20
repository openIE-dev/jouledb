//! ASCII85 (Base85) encoding and decoding.
//!
//! Encodes binary data using printable ASCII characters (codes 33–117).
//! Supports the standard ASCII85 format with Adobe `<~ ~>` delimiters,
//! the 'z' shortcut for all-zero groups, and the RFC 1924 variant for
//! IPv6 addresses. Replaces npm ascii85 packages with pure Rust.

// ── Errors ───────────────────────────────────────────────────────────

/// Errors from ASCII85 operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Ascii85Error {
    #[error("invalid character '{0}' at position {1}")]
    InvalidCharacter(char, usize),
    #[error("invalid group length {0} (expected 2-5 characters)")]
    InvalidGroupLength(usize),
    #[error("overflow in group decode")]
    Overflow,
    #[error("missing Adobe delimiters")]
    MissingDelimiters,
    #[error("invalid RFC 1924 input length (expected 20 characters for 16 bytes)")]
    InvalidRfc1924Length,
}

// ── Standard ASCII85 ─────────────────────────────────────────────────

/// Encode 4 bytes into 5 ASCII85 characters.
fn encode_group(bytes: &[u8; 4]) -> [u8; 5] {
    let val = u32::from_be_bytes(*bytes);
    let mut chars = [0u8; 5];
    let mut v = val;
    for i in (0..5).rev() {
        chars[i] = (v % 85) as u8 + 33;
        v /= 85;
    }
    chars
}

/// Decode 5 ASCII85 characters into 4 bytes.
fn decode_group(chars: &[u8; 5]) -> Result<[u8; 4], Ascii85Error> {
    let mut val = 0u64;
    for (i, &c) in chars.iter().enumerate() {
        if c < 33 || c > 117 {
            return Err(Ascii85Error::InvalidCharacter(c as char, i));
        }
        val = val * 85 + (c - 33) as u64;
    }
    if val > u32::MAX as u64 {
        return Err(Ascii85Error::Overflow);
    }
    Ok((val as u32).to_be_bytes())
}

/// Encode binary data to ASCII85.
pub fn encode(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len() * 5 / 4 + 5);
    let mut i = 0;

    while i + 4 <= data.len() {
        let group: [u8; 4] = [data[i], data[i + 1], data[i + 2], data[i + 3]];
        if group == [0, 0, 0, 0] {
            result.push('z');
        } else {
            let encoded = encode_group(&group);
            for &c in &encoded {
                result.push(c as char);
            }
        }
        i += 4;
    }

    // Handle remaining bytes (1-3).
    let remaining = data.len() - i;
    if remaining > 0 {
        let mut padded = [0u8; 4];
        for j in 0..remaining {
            padded[j] = data[i + j];
        }
        let encoded = encode_group(&padded);
        // Output remaining+1 characters.
        for j in 0..remaining + 1 {
            result.push(encoded[j] as char);
        }
    }

    result
}

/// Encode with Adobe btoa delimiters (<~ ... ~>).
pub fn encode_adobe(data: &[u8]) -> String {
    let mut result = String::from("<~");
    result.push_str(&encode(data));
    result.push_str("~>");
    result
}

/// Decode ASCII85 to bytes.
pub fn decode(input: &str) -> Result<Vec<u8>, Ascii85Error> {
    // Strip whitespace.
    let cleaned: Vec<u8> = input
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();

    let mut result = Vec::with_capacity(cleaned.len() * 4 / 5);
    let mut i = 0;

    while i < cleaned.len() {
        if cleaned[i] == b'z' {
            result.extend_from_slice(&[0, 0, 0, 0]);
            i += 1;
            continue;
        }

        let remaining = cleaned.len() - i;
        if remaining >= 5 {
            let group: [u8; 5] = [
                cleaned[i],
                cleaned[i + 1],
                cleaned[i + 2],
                cleaned[i + 3],
                cleaned[i + 4],
            ];
            let bytes = decode_group(&group)?;
            result.extend_from_slice(&bytes);
            i += 5;
        } else if remaining >= 2 {
            // Partial group: pad with 'u' (117 = highest ASCII85 digit).
            let mut group = [117u8; 5]; // 'u' = 117
            for j in 0..remaining {
                group[j] = cleaned[i + j];
            }
            let bytes = decode_group(&group)?;
            // Output remaining-1 bytes.
            result.extend_from_slice(&bytes[..remaining - 1]);
            i += remaining;
        } else {
            return Err(Ascii85Error::InvalidGroupLength(remaining));
        }
    }

    Ok(result)
}

/// Decode Adobe-delimited ASCII85.
pub fn decode_adobe(input: &str) -> Result<Vec<u8>, Ascii85Error> {
    let trimmed = input.trim();
    let inner = trimmed
        .strip_prefix("<~")
        .and_then(|s| s.strip_suffix("~>"))
        .ok_or(Ascii85Error::MissingDelimiters)?;
    decode(inner)
}

// ── Validation ───────────────────────────────────────────────────────

/// Validate an ASCII85-encoded string.
pub fn is_valid(input: &str) -> bool {
    let cleaned: Vec<u8> = input
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();

    let mut i = 0;
    while i < cleaned.len() {
        if cleaned[i] == b'z' {
            i += 1;
            continue;
        }
        if cleaned[i] >= 33 && cleaned[i] <= 117 {
            i += 1;
        } else {
            return false;
        }
    }
    true
}

/// Validate Adobe-delimited ASCII85.
pub fn is_valid_adobe(input: &str) -> bool {
    let trimmed = input.trim();
    if let Some(inner) = trimmed.strip_prefix("<~").and_then(|s| s.strip_suffix("~>")) {
        is_valid(inner)
    } else {
        false
    }
}

// ── RFC 1924 Variant ─────────────────────────────────────────────────

/// RFC 1924 Base85 alphabet for IPv6.
const RFC1924_ALPHABET: &[u8; 85] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";

/// Encode 16 bytes (e.g., an IPv6 address) using RFC 1924 Base85.
/// Produces exactly 20 characters.
pub fn encode_rfc1924(data: &[u8; 16]) -> String {
    // Interpret 16 bytes as a 128-bit big-endian integer.
    let hi = u64::from_be_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as u128;
    let lo = u64::from_be_bytes([
        data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
    ]) as u128;
    let mut val = (hi << 64) | lo;

    let mut chars = [0u8; 20];
    for i in (0..20).rev() {
        chars[i] = RFC1924_ALPHABET[(val % 85) as usize];
        val /= 85;
    }

    // SAFETY: RFC1924_ALPHABET is ASCII.
    unsafe { String::from_utf8_unchecked(chars.to_vec()) }
}

/// Decode RFC 1924 Base85 (20 chars) to 16 bytes.
pub fn decode_rfc1924(input: &str) -> Result<[u8; 16], Ascii85Error> {
    let bytes = input.as_bytes();
    if bytes.len() != 20 {
        return Err(Ascii85Error::InvalidRfc1924Length);
    }

    // Build decode table.
    let mut table = [0xFFu8; 256];
    for (i, &c) in RFC1924_ALPHABET.iter().enumerate() {
        table[c as usize] = i as u8;
    }

    let mut val = 0u128;
    for (i, &b) in bytes.iter().enumerate() {
        let digit = table[b as usize];
        if digit == 0xFF {
            return Err(Ascii85Error::InvalidCharacter(b as char, i));
        }
        val = val
            .checked_mul(85)
            .ok_or(Ascii85Error::Overflow)?
            .checked_add(digit as u128)
            .ok_or(Ascii85Error::Overflow)?;
    }

    let hi = (val >> 64) as u64;
    let lo = val as u64;
    let mut result = [0u8; 16];
    result[..8].copy_from_slice(&hi.to_be_bytes());
    result[8..].copy_from_slice(&lo.to_be_bytes());
    Ok(result)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_empty() {
        assert_eq!(encode(b""), "");
    }

    #[test]
    fn roundtrip_hello() {
        let data = b"Hello";
        let encoded = encode(data);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn roundtrip_exact_group() {
        let data = b"test"; // 4 bytes = 1 group
        let encoded = encode(data);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn zero_group_shortcut() {
        let data = [0u8; 4];
        let encoded = encode(&data);
        assert_eq!(encoded, "z");
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn multiple_zero_groups() {
        let data = [0u8; 12];
        let encoded = encode(&data);
        assert_eq!(encoded, "zzz");
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn adobe_delimiters() {
        let data = b"Hello, World!";
        let encoded = encode_adobe(data);
        assert!(encoded.starts_with("<~"));
        assert!(encoded.ends_with("~>"));
        let decoded = decode_adobe(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn adobe_missing_delimiters() {
        assert_eq!(decode_adobe("no delimiters"), Err(Ascii85Error::MissingDelimiters));
    }

    #[test]
    fn roundtrip_binary() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = encode(&data);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn roundtrip_various_lengths() {
        for len in 0..20 {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let encoded = encode(&data);
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded, data, "failed for length {len}");
        }
    }

    #[test]
    fn validation() {
        assert!(is_valid("87cURD]j7BEbo80"));
        assert!(is_valid("z"));
        assert!(!is_valid("\x01")); // control char
        assert!(is_valid_adobe("<~87cURD]j7BEbo80~>"));
        assert!(!is_valid_adobe("87cURD]j7BEbo80")); // no delimiters
    }

    #[test]
    fn rfc1924_roundtrip() {
        let ipv6: [u8; 16] = [
            0x20, 0x01, 0x0d, 0xb8, 0x85, 0xa3, 0x00, 0x00,
            0x00, 0x00, 0x8a, 0x2e, 0x03, 0x70, 0x73, 0x34,
        ];
        let encoded = encode_rfc1924(&ipv6);
        assert_eq!(encoded.len(), 20);
        let decoded = decode_rfc1924(&encoded).unwrap();
        assert_eq!(decoded, ipv6);
    }

    #[test]
    fn rfc1924_all_zeros() {
        let data = [0u8; 16];
        let encoded = encode_rfc1924(&data);
        let decoded = decode_rfc1924(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn rfc1924_all_ones() {
        let data = [0xFFu8; 16];
        let encoded = encode_rfc1924(&data);
        let decoded = decode_rfc1924(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn rfc1924_invalid_length() {
        assert_eq!(decode_rfc1924("short"), Err(Ascii85Error::InvalidRfc1924Length));
    }

    #[test]
    fn whitespace_handling() {
        let data = b"test";
        let encoded = encode(data);
        // Insert whitespace.
        let with_spaces = format!("  {}  ", encoded);
        let decoded = decode(&with_spaces).unwrap();
        assert_eq!(decoded, data);
    }
}
