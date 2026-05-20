//! Variable-length integer encoding — LEB128, protobuf varint, SQLite varint, zigzag.
//!
//! Implements compact integer encoding schemes used in binary protocols,
//! database storage, and serialization formats. Replaces JS varint/protobuf
//! libraries with a pure Rust implementation.

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during varint encoding/decoding.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VarintError {
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("varint overflow: encoded value exceeds {0}-bit capacity")]
    Overflow(u8),
    #[error("varint too long: {0} bytes exceeds maximum of {1}")]
    TooLong(usize, usize),
    #[error("buffer too small: need {need} bytes, have {have}")]
    BufferTooSmall { need: usize, have: usize },
}

// ── LEB128 Unsigned ─────────────────────────────────────────────────

/// Encode an unsigned 64-bit integer as ULEB128.
pub fn encode_uleb128(value: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    let mut val = value;
    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;
        if val != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if val == 0 {
            break;
        }
    }
    buf
}

/// Encode ULEB128 into a provided buffer. Returns bytes written.
pub fn encode_uleb128_into(value: u64, buf: &mut [u8]) -> Result<usize, VarintError> {
    let needed = uleb128_size(value);
    if buf.len() < needed {
        return Err(VarintError::BufferTooSmall {
            need: needed,
            have: buf.len(),
        });
    }
    let mut val = value;
    let mut i = 0;
    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;
        if val != 0 {
            byte |= 0x80;
        }
        buf[i] = byte;
        i += 1;
        if val == 0 {
            break;
        }
    }
    Ok(i)
}

/// Decode a ULEB128-encoded value from a byte slice.
/// Returns (value, bytes_consumed).
pub fn decode_uleb128(data: &[u8]) -> Result<(u64, usize), VarintError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &byte) in data.iter().enumerate() {
        if shift >= 64 {
            return Err(VarintError::Overflow(64));
        }
        let low_bits = (byte & 0x7F) as u64;
        // Check for overflow before shifting
        if shift >= 57 && low_bits > (u64::MAX >> shift) {
            return Err(VarintError::Overflow(64));
        }
        result |= low_bits << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        if i >= 9 {
            return Err(VarintError::TooLong(i + 1, 10));
        }
    }
    Err(VarintError::UnexpectedEof)
}

/// Calculate the encoded size of a ULEB128 value.
pub fn uleb128_size(value: u64) -> usize {
    if value == 0 {
        return 1;
    }
    let bits = 64 - value.leading_zeros() as usize;
    (bits + 6) / 7
}

// ── LEB128 Signed ───────────────────────────────────────────────────

/// Encode a signed 64-bit integer as SLEB128.
pub fn encode_sleb128(value: i64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    let mut val = value;
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        let more = !((val == 0 && byte & 0x40 == 0) || (val == -1 && byte & 0x40 != 0));
        if more {
            buf.push(byte | 0x80);
        } else {
            buf.push(byte);
            break;
        }
    }
    buf
}

/// Encode SLEB128 into a provided buffer. Returns bytes written.
pub fn encode_sleb128_into(value: i64, buf: &mut [u8]) -> Result<usize, VarintError> {
    let needed = sleb128_size(value);
    if buf.len() < needed {
        return Err(VarintError::BufferTooSmall {
            need: needed,
            have: buf.len(),
        });
    }
    let mut val = value;
    let mut i = 0;
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        let more = !((val == 0 && byte & 0x40 == 0) || (val == -1 && byte & 0x40 != 0));
        if more {
            buf[i] = byte | 0x80;
        } else {
            buf[i] = byte;
            i += 1;
            break;
        }
        i += 1;
    }
    Ok(i)
}

/// Decode a SLEB128-encoded value from a byte slice.
/// Returns (value, bytes_consumed).
pub fn decode_sleb128(data: &[u8]) -> Result<(i64, usize), VarintError> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    let mut last_byte = 0u8;
    let mut consumed = 0usize;

    for (i, &byte) in data.iter().enumerate() {
        if shift >= 64 {
            return Err(VarintError::Overflow(64));
        }
        result |= ((byte & 0x7F) as i64) << shift;
        shift += 7;
        last_byte = byte;
        consumed = i + 1;
        if byte & 0x80 == 0 {
            // Sign-extend if the sign bit of the last byte is set.
            if shift < 64 && (last_byte & 0x40) != 0 {
                result |= !0i64 << shift;
            }
            return Ok((result, consumed));
        }
        if i >= 9 {
            return Err(VarintError::TooLong(i + 1, 10));
        }
    }
    let _ = (last_byte, consumed); // suppress warnings
    Err(VarintError::UnexpectedEof)
}

/// Calculate the encoded size of a SLEB128 value.
pub fn sleb128_size(value: i64) -> usize {
    let mut val = value;
    let mut count = 0usize;
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        count += 1;
        if (val == 0 && byte & 0x40 == 0) || (val == -1 && byte & 0x40 != 0) {
            break;
        }
    }
    count
}

// ── Protobuf Varint ─────────────────────────────────────────────────
//
// Protobuf varint is identical to ULEB128 for unsigned values.
// For signed values, protobuf uses zigzag encoding + ULEB128.

/// Encode an unsigned 64-bit integer as a protobuf varint (same as ULEB128).
pub fn encode_protobuf_varint(value: u64) -> Vec<u8> {
    encode_uleb128(value)
}

/// Decode a protobuf varint (same as ULEB128).
pub fn decode_protobuf_varint(data: &[u8]) -> Result<(u64, usize), VarintError> {
    decode_uleb128(data)
}

/// Encode a signed 64-bit integer using protobuf's zigzag + varint.
pub fn encode_protobuf_sint(value: i64) -> Vec<u8> {
    encode_uleb128(zigzag_encode(value))
}

/// Decode a protobuf signed integer (zigzag + varint).
pub fn decode_protobuf_sint(data: &[u8]) -> Result<(i64, usize), VarintError> {
    let (raw, consumed) = decode_uleb128(data)?;
    Ok((zigzag_decode(raw), consumed))
}

// ── Zigzag Encoding ─────────────────────────────────────────────────

/// Zigzag-encode a signed 64-bit integer to unsigned.
/// Maps: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
pub fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Zigzag-decode an unsigned 64-bit integer back to signed.
pub fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

// ── SQLite Varint ───────────────────────────────────────────────────
//
// SQLite uses a variable-length encoding where the high bit of each
// byte indicates whether more bytes follow, but the last byte (byte 9)
// uses all 8 bits for the value.

/// Encode a u64 as a SQLite-style varint.
pub fn encode_sqlite_varint(value: u64) -> Vec<u8> {
    if value <= 240 {
        return vec![value as u8];
    }
    if value <= 2287 {
        let adj = value - 240;
        return vec![((adj >> 8) + 241) as u8, (adj & 0xFF) as u8];
    }
    if value <= 67823 {
        let adj = value - 2288;
        return vec![249, (adj >> 8) as u8, (adj & 0xFF) as u8];
    }
    // For larger values, use the big-endian byte format with a leading
    // byte indicating byte count.
    let bytes_needed = if value <= 0xFF_FFFF {
        3
    } else if value <= 0xFFFF_FFFF {
        4
    } else if value <= 0xFF_FFFF_FFFF {
        5
    } else if value <= 0xFFFF_FFFF_FFFF {
        6
    } else if value <= 0xFF_FFFF_FFFF_FFFF {
        7
    } else {
        8
    };
    let mut buf = Vec::with_capacity(bytes_needed + 1);
    buf.push((250 + bytes_needed - 3) as u8);
    for i in (0..bytes_needed).rev() {
        buf.push((value >> (i * 8)) as u8);
    }
    buf
}

/// Decode a SQLite-style varint from a byte slice.
/// Returns (value, bytes_consumed).
pub fn decode_sqlite_varint(data: &[u8]) -> Result<(u64, usize), VarintError> {
    if data.is_empty() {
        return Err(VarintError::UnexpectedEof);
    }
    let first = data[0];
    if first <= 240 {
        return Ok((first as u64, 1));
    }
    if first <= 248 {
        if data.len() < 2 {
            return Err(VarintError::UnexpectedEof);
        }
        let value = 240u64 + 256 * (first as u64 - 241) + data[1] as u64;
        return Ok((value, 2));
    }
    if first == 249 {
        if data.len() < 3 {
            return Err(VarintError::UnexpectedEof);
        }
        let value = 2288u64 + 256 * data[1] as u64 + data[2] as u64;
        return Ok((value, 3));
    }
    // first in 250..=255 means 3..8 big-endian bytes follow
    let byte_count = (first as usize) - 247;
    if data.len() < 1 + byte_count {
        return Err(VarintError::UnexpectedEof);
    }
    let mut value = 0u64;
    for i in 0..byte_count {
        value = (value << 8) | data[1 + i] as u64;
    }
    Ok((value, 1 + byte_count))
}

/// Calculate the encoded size of a SQLite varint.
pub fn sqlite_varint_size(value: u64) -> usize {
    if value <= 240 {
        1
    } else if value <= 2287 {
        2
    } else if value <= 67823 {
        3
    } else if value <= 0xFF_FFFF {
        4
    } else if value <= 0xFFFF_FFFF {
        5
    } else if value <= 0xFF_FFFF_FFFF {
        6
    } else if value <= 0xFFFF_FFFF_FFFF {
        7
    } else if value <= 0xFF_FFFF_FFFF_FFFF {
        8
    } else {
        9
    }
}

// ── Multi-Value Encoding ────────────────────────────────────────────

/// Encode multiple unsigned values as consecutive ULEB128 varints.
pub fn encode_multi_uleb128(values: &[u64]) -> Vec<u8> {
    let mut buf = Vec::new();
    for &v in values {
        buf.extend_from_slice(&encode_uleb128(v));
    }
    buf
}

/// Decode multiple ULEB128 varints from a byte slice.
pub fn decode_multi_uleb128(data: &[u8], count: usize) -> Result<(Vec<u64>, usize), VarintError> {
    let mut values = Vec::with_capacity(count);
    let mut offset = 0;
    for _ in 0..count {
        if offset >= data.len() {
            return Err(VarintError::UnexpectedEof);
        }
        let (value, consumed) = decode_uleb128(&data[offset..])?;
        values.push(value);
        offset += consumed;
    }
    Ok((values, offset))
}

/// Encode multiple signed values as consecutive SLEB128 varints.
pub fn encode_multi_sleb128(values: &[i64]) -> Vec<u8> {
    let mut buf = Vec::new();
    for &v in values {
        buf.extend_from_slice(&encode_sleb128(v));
    }
    buf
}

/// Decode multiple SLEB128 varints from a byte slice.
pub fn decode_multi_sleb128(data: &[u8], count: usize) -> Result<(Vec<i64>, usize), VarintError> {
    let mut values = Vec::with_capacity(count);
    let mut offset = 0;
    for _ in 0..count {
        if offset >= data.len() {
            return Err(VarintError::UnexpectedEof);
        }
        let (value, consumed) = decode_sleb128(&data[offset..])?;
        values.push(value);
        offset += consumed;
    }
    Ok((values, offset))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ULEB128 ─────────────────────────────────────────────────────

    #[test]
    fn uleb128_zero() {
        let enc = encode_uleb128(0);
        assert_eq!(enc, [0x00]);
        let (val, consumed) = decode_uleb128(&enc).unwrap();
        assert_eq!(val, 0);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn uleb128_small() {
        let enc = encode_uleb128(127);
        assert_eq!(enc, [0x7F]);
        let (val, _) = decode_uleb128(&enc).unwrap();
        assert_eq!(val, 127);
    }

    #[test]
    fn uleb128_128() {
        let enc = encode_uleb128(128);
        assert_eq!(enc, [0x80, 0x01]);
        let (val, consumed) = decode_uleb128(&enc).unwrap();
        assert_eq!(val, 128);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn uleb128_large() {
        let enc = encode_uleb128(624485);
        assert_eq!(enc, [0xE5, 0x8E, 0x26]);
        let (val, _) = decode_uleb128(&enc).unwrap();
        assert_eq!(val, 624485);
    }

    #[test]
    fn uleb128_max() {
        let enc = encode_uleb128(u64::MAX);
        let (val, _) = decode_uleb128(&enc).unwrap();
        assert_eq!(val, u64::MAX);
    }

    #[test]
    fn uleb128_size_calculation() {
        assert_eq!(uleb128_size(0), 1);
        assert_eq!(uleb128_size(127), 1);
        assert_eq!(uleb128_size(128), 2);
        assert_eq!(uleb128_size(16383), 2);
        assert_eq!(uleb128_size(16384), 3);
        assert_eq!(uleb128_size(u64::MAX), 10);
    }

    #[test]
    fn uleb128_into_buffer() {
        let mut buf = [0u8; 10];
        let written = encode_uleb128_into(300, &mut buf).unwrap();
        assert_eq!(written, 2);
        let (val, _) = decode_uleb128(&buf[..written]).unwrap();
        assert_eq!(val, 300);
    }

    #[test]
    fn uleb128_buffer_too_small() {
        let mut buf = [0u8; 1];
        let result = encode_uleb128_into(300, &mut buf);
        assert!(matches!(result, Err(VarintError::BufferTooSmall { .. })));
    }

    // ── SLEB128 ─────────────────────────────────────────────────────

    #[test]
    fn sleb128_zero() {
        let enc = encode_sleb128(0);
        assert_eq!(enc, [0x00]);
        let (val, _) = decode_sleb128(&enc).unwrap();
        assert_eq!(val, 0);
    }

    #[test]
    fn sleb128_positive() {
        let enc = encode_sleb128(63);
        assert_eq!(enc, [0x3F]);
        let (val, _) = decode_sleb128(&enc).unwrap();
        assert_eq!(val, 63);
    }

    #[test]
    fn sleb128_negative() {
        let enc = encode_sleb128(-1);
        assert_eq!(enc, [0x7F]);
        let (val, _) = decode_sleb128(&enc).unwrap();
        assert_eq!(val, -1);
    }

    #[test]
    fn sleb128_negative_large() {
        let enc = encode_sleb128(-123456);
        let (val, _) = decode_sleb128(&enc).unwrap();
        assert_eq!(val, -123456);
    }

    #[test]
    fn sleb128_min_max() {
        for value in [i64::MIN, i64::MAX, -1, 0, 1] {
            let enc = encode_sleb128(value);
            let (val, _) = decode_sleb128(&enc).unwrap();
            assert_eq!(val, value);
        }
    }

    #[test]
    fn sleb128_size_calculation() {
        assert_eq!(sleb128_size(0), 1);
        assert_eq!(sleb128_size(-1), 1);
        assert_eq!(sleb128_size(63), 1);
        assert_eq!(sleb128_size(64), 2);
        assert_eq!(sleb128_size(-64), 1);
        assert_eq!(sleb128_size(-65), 2);
    }

    #[test]
    fn sleb128_into_buffer() {
        let mut buf = [0u8; 10];
        let written = encode_sleb128_into(-300, &mut buf).unwrap();
        let (val, _) = decode_sleb128(&buf[..written]).unwrap();
        assert_eq!(val, -300);
    }

    // ── Zigzag ──────────────────────────────────────────────────────

    #[test]
    fn zigzag_mapping() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);
        assert_eq!(zigzag_encode(i64::MAX), u64::MAX - 1);
        assert_eq!(zigzag_encode(i64::MIN), u64::MAX);
    }

    #[test]
    fn zigzag_roundtrip() {
        for v in [0i64, 1, -1, 127, -128, 1000, -1000, i64::MAX, i64::MIN] {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v);
        }
    }

    // ── Protobuf Varint ─────────────────────────────────────────────

    #[test]
    fn protobuf_varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 300, 100000, u64::MAX] {
            let enc = encode_protobuf_varint(v);
            let (dec, _) = decode_protobuf_varint(&enc).unwrap();
            assert_eq!(dec, v);
        }
    }

    #[test]
    fn protobuf_sint_roundtrip() {
        for v in [0i64, 1, -1, 127, -128, 100000, -100000, i64::MAX, i64::MIN] {
            let enc = encode_protobuf_sint(v);
            let (dec, _) = decode_protobuf_sint(&enc).unwrap();
            assert_eq!(dec, v);
        }
    }

    // ── SQLite Varint ───────────────────────────────────────────────

    #[test]
    fn sqlite_varint_small() {
        let enc = encode_sqlite_varint(100);
        assert_eq!(enc, [100]);
        let (val, consumed) = decode_sqlite_varint(&enc).unwrap();
        assert_eq!(val, 100);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn sqlite_varint_medium() {
        let enc = encode_sqlite_varint(500);
        let (val, _) = decode_sqlite_varint(&enc).unwrap();
        assert_eq!(val, 500);
    }

    #[test]
    fn sqlite_varint_two_byte_boundary() {
        // 241 to 2287 range
        for v in [240u64, 241, 500, 1000, 2287] {
            let enc = encode_sqlite_varint(v);
            let (val, _) = decode_sqlite_varint(&enc).unwrap();
            assert_eq!(val, v, "failed for {v}");
        }
    }

    #[test]
    fn sqlite_varint_three_byte() {
        for v in [2288u64, 10000, 67823] {
            let enc = encode_sqlite_varint(v);
            let (val, _) = decode_sqlite_varint(&enc).unwrap();
            assert_eq!(val, v, "failed for {v}");
        }
    }

    #[test]
    fn sqlite_varint_large() {
        for v in [67824u64, 0xFF_FFFF, 0xFFFF_FFFF, u64::MAX] {
            let enc = encode_sqlite_varint(v);
            let (val, _) = decode_sqlite_varint(&enc).unwrap();
            assert_eq!(val, v, "failed for {v}");
        }
    }

    #[test]
    fn sqlite_varint_size_calculation() {
        assert_eq!(sqlite_varint_size(0), 1);
        assert_eq!(sqlite_varint_size(240), 1);
        assert_eq!(sqlite_varint_size(241), 2);
        assert_eq!(sqlite_varint_size(2287), 2);
        assert_eq!(sqlite_varint_size(2288), 3);
        assert_eq!(sqlite_varint_size(67823), 3);
        assert_eq!(sqlite_varint_size(u64::MAX), 9);
    }

    // ── Multi-Value Encoding ────────────────────────────────────────

    #[test]
    fn multi_uleb128_roundtrip() {
        let values = vec![0u64, 127, 128, 999999, u64::MAX];
        let encoded = encode_multi_uleb128(&values);
        let (decoded, _) = decode_multi_uleb128(&encoded, values.len()).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn multi_sleb128_roundtrip() {
        let values = vec![0i64, -1, 127, -128, 999999, -999999];
        let encoded = encode_multi_sleb128(&values);
        let (decoded, _) = decode_multi_sleb128(&encoded, values.len()).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn multi_empty() {
        let encoded = encode_multi_uleb128(&[]);
        assert!(encoded.is_empty());
        let (decoded, consumed) = decode_multi_uleb128(&encoded, 0).unwrap();
        assert!(decoded.is_empty());
        assert_eq!(consumed, 0);
    }

    // ── Error Cases ─────────────────────────────────────────────────

    #[test]
    fn decode_empty_fails() {
        assert!(matches!(
            decode_uleb128(&[]),
            Err(VarintError::UnexpectedEof)
        ));
        assert!(matches!(
            decode_sleb128(&[]),
            Err(VarintError::UnexpectedEof)
        ));
    }

    #[test]
    fn decode_truncated_fails() {
        // Continuation bit set but no more bytes.
        assert!(matches!(
            decode_uleb128(&[0x80]),
            Err(VarintError::UnexpectedEof)
        ));
    }

    #[test]
    fn sqlite_decode_empty_fails() {
        assert!(matches!(
            decode_sqlite_varint(&[]),
            Err(VarintError::UnexpectedEof)
        ));
    }

    #[test]
    fn multi_decode_not_enough_data() {
        let data = encode_uleb128(42);
        assert!(matches!(
            decode_multi_uleb128(&data, 2),
            Err(VarintError::UnexpectedEof)
        ));
    }
}
