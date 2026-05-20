//! DEFLATE compression and decompression (RFC 1951 subset).
//!
//! Implements fixed Huffman coding with LZ77 match finding using a
//! sliding window and hash chain. Replaces pako / zlib.js with a
//! pure Rust implementation that is fully testable on native targets.

// ── Constants ────────────────────────────────────────────────────────

/// Maximum window size for LZ77 match finding.
const WINDOW_SIZE: usize = 32_768;

/// Minimum match length (RFC 1951).
const MIN_MATCH: usize = 3;

/// Maximum match length (RFC 1951).
const MAX_MATCH: usize = 258;

/// Hash table size (power of 2).
const HASH_SIZE: usize = 1 << 15;

/// Hash mask.
const HASH_MASK: usize = HASH_SIZE - 1;

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during DEFLATE compression/decompression.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeflateError {
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("invalid block type")]
    InvalidBlockType,
    #[error("invalid length code: {0}")]
    InvalidLengthCode(u32),
    #[error("invalid distance code: {0}")]
    InvalidDistanceCode(u32),
    #[error("invalid back-reference: distance {distance} exceeds output size {output_size}")]
    InvalidBackReference { distance: usize, output_size: usize },
}

// ── Bit Writer ───────────────────────────────────────────────────────

struct BitWriter {
    buf: Vec<u8>,
    current: u32,
    bits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            current: 0,
            bits: 0,
        }
    }

    /// Write `count` bits from `value`, LSB first.
    fn write_bits(&mut self, value: u32, count: u8) {
        self.current |= value << self.bits;
        self.bits += count;
        while self.bits >= 8 {
            self.buf.push(self.current as u8);
            self.current >>= 8;
            self.bits -= 8;
        }
    }

    /// Write a Huffman code: `count` bits from `code`, MSB first.
    /// In DEFLATE, Huffman codes are stored with MSB written first to the
    /// LSB-first bit stream, so we reverse the bit order.
    fn write_huffman(&mut self, code: u32, count: u8) {
        let reversed = reverse_bits(code, count);
        self.write_bits(reversed, count);
    }

    fn flush(&mut self) {
        if self.bits > 0 {
            self.buf.push(self.current as u8);
            self.current = 0;
            self.bits = 0;
        }
    }

    fn into_bytes(mut self) -> Vec<u8> {
        self.flush();
        self.buf
    }
}

// ── Bit Reader ───────────────────────────────────────────────────────

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    current: u32,
    bits: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            current: 0,
            bits: 0,
        }
    }

    /// Read `count` bits, LSB first.
    fn read_bits(&mut self, count: u8) -> Result<u32, DeflateError> {
        while self.bits < count {
            if self.pos >= self.data.len() {
                return Err(DeflateError::UnexpectedEof);
            }
            self.current |= (self.data[self.pos] as u32) << self.bits;
            self.pos += 1;
            self.bits += 8;
        }
        let mask = (1u32 << count) - 1;
        let val = self.current & mask;
        self.current >>= count;
        self.bits -= count;
        Ok(val)
    }
}

fn reverse_bits(value: u32, nbits: u8) -> u32 {
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..nbits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

// ── Fixed Huffman Tables ─────────────────────────────────────────────

/// Build the fixed literal/length decode table.
/// Returns array indexed by symbol, each entry is (canonical_code, bit_length).
/// For the encoder, we also need these canonical codes.
struct FixedCodes {
    /// For encoding: symbol -> (canonical_code, bit_length)
    lit_encode: [(u32, u8); 288],
    /// For decoding: sorted list of (reversed_code, bit_length, symbol)
    lit_decode: Vec<(u32, u8, u32)>,
    /// Distance: all 5-bit codes 0-29
    dist_decode: Vec<(u32, u8, u32)>,
}

fn build_fixed_codes() -> FixedCodes {
    // Step 1: Assign bit lengths per RFC 1951 §3.2.6
    let mut lit_lengths = [0u8; 288];
    for i in 0..=143 { lit_lengths[i] = 8; }
    for i in 144..=255 { lit_lengths[i] = 9; }
    for i in 256..=279 { lit_lengths[i] = 7; }
    for i in 280..=287 { lit_lengths[i] = 8; }

    // Step 2: Build canonical codes from lengths (RFC 1951 §3.2.2)
    let lit_encode = canonical_codes(&lit_lengths);

    // Step 3: Build decode table (reversed codes for LSB-first matching)
    let mut lit_decode = Vec::new();
    for (sym, &(code, len)) in lit_encode.iter().enumerate() {
        if len > 0 {
            let rev = reverse_bits(code, len);
            lit_decode.push((rev, len, sym as u32));
        }
    }

    // Distance codes: all 5-bit, codes 0-29
    let mut dist_lengths = [0u8; 30];
    for i in 0..30 { dist_lengths[i] = 5; }
    let dist_codes = canonical_codes(&dist_lengths);
    let mut dist_decode = Vec::new();
    for (sym, &(code, len)) in dist_codes.iter().enumerate() {
        if len > 0 {
            let rev = reverse_bits(code, len);
            dist_decode.push((rev, len, sym as u32));
        }
    }

    // Convert lit_encode to fixed-size array
    let mut enc = [(0u32, 0u8); 288];
    for (i, &(code, len)) in lit_encode.iter().enumerate() {
        enc[i] = (code, len);
    }

    FixedCodes {
        lit_encode: enc,
        lit_decode,
        dist_decode,
    }
}

/// Build canonical Huffman codes from bit lengths.
fn canonical_codes<const N: usize>(lengths: &[u8; N]) -> [(u32, u8); N] {
    let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;

    // Count codes of each length
    let mut bl_count = vec![0u32; max_len + 1];
    for &l in lengths.iter() {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }

    // Compute starting code for each length
    let mut next_code = vec![0u32; max_len + 1];
    let mut code = 0u32;
    for bits in 1..=max_len {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes
    let mut result = [(0u32, 0u8); N];
    for (sym, &len) in lengths.iter().enumerate() {
        if len > 0 {
            result[sym] = (next_code[len as usize], len);
            next_code[len as usize] += 1;
        }
    }
    result
}

/// Decode one symbol from the bit stream.
/// Reads bits one at a time, accumulates reversed code, checks table.
fn decode_symbol(
    reader: &mut BitReader<'_>,
    table: &[(u32, u8, u32)],
    max_bits: u8,
) -> Result<u32, DeflateError> {
    let mut code = 0u32;
    for bit_len in 1..=max_bits {
        let bit = reader.read_bits(1)?;
        code |= bit << (bit_len - 1);
        for &(rev_code, len, sym) in table {
            if len == bit_len && rev_code == code {
                return Ok(sym);
            }
        }
    }
    Err(DeflateError::UnexpectedEof)
}

// ── Length / Distance Tables ─────────────────────────────────────────

/// Length base values and extra bits for codes 257-285.
const LENGTH_TABLE: [(usize, u8); 29] = [
    (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),     // 257-264
    (11, 1), (13, 1), (15, 1), (17, 1),                                      // 265-268
    (19, 2), (23, 2), (27, 2), (31, 2),                                      // 269-272
    (35, 3), (43, 3), (51, 3), (59, 3),                                      // 273-276
    (67, 4), (83, 4), (99, 4), (115, 4),                                     // 277-280
    (131, 5), (163, 5), (195, 5), (227, 5),                                  // 281-284
    (258, 0),                                                                  // 285
];

/// Distance base values and extra bits for codes 0-29.
const DISTANCE_TABLE: [(usize, u8); 30] = [
    (1, 0), (2, 0), (3, 0), (4, 0),
    (5, 1), (7, 1), (9, 2), (13, 2),
    (17, 3), (25, 3), (33, 4), (49, 4),
    (65, 5), (97, 5), (129, 6), (193, 6),
    (257, 7), (385, 7), (513, 8), (769, 8),
    (1025, 9), (1537, 9), (2049, 10), (3073, 10),
    (4097, 11), (6145, 11), (8193, 12), (12289, 12),
    (16385, 13), (24577, 13),
];

/// Find length code and extra bits for a given length (3-258).
fn encode_length(length: usize) -> (u32, u8, u32) {
    for (i, &(base, extra)) in LENGTH_TABLE.iter().enumerate() {
        let code = 257 + i as u32;
        let next_base = if i + 1 < LENGTH_TABLE.len() {
            LENGTH_TABLE[i + 1].0
        } else {
            259 // past max
        };
        if length >= base && length < next_base {
            return (code, extra, (length - base) as u32);
        }
    }
    // length == 258 (last entry)
    (285, 0, 0)
}

/// Find distance code and extra bits for a given distance (1-32768).
fn encode_distance(dist: usize) -> (u32, u8, u32) {
    for (i, &(base, extra)) in DISTANCE_TABLE.iter().enumerate() {
        let next_base = if i + 1 < DISTANCE_TABLE.len() {
            DISTANCE_TABLE[i + 1].0
        } else {
            32769
        };
        if dist >= base && dist < next_base {
            return (i as u32, extra, (dist - base) as u32);
        }
    }
    (29, 13, (dist - 24577) as u32)
}

// ── LZ77 Match Finder ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Literal(u8),
    Match { length: usize, distance: usize },
}

fn hash3(data: &[u8], pos: usize) -> usize {
    let h = (data[pos] as usize) << 10
        ^ (data[pos + 1] as usize) << 5
        ^ (data[pos + 2] as usize);
    h & HASH_MASK
}

fn find_matches(input: &[u8]) -> Vec<Token> {
    let mut tokens = Vec::new();
    if input.is_empty() {
        return tokens;
    }

    let mut head = [0u32; HASH_SIZE];
    let mut prev = vec![0u32; input.len()];

    let mut pos = 0usize;
    while pos < input.len() {
        if pos + 2 >= input.len() {
            tokens.push(Token::Literal(input[pos]));
            pos += 1;
            continue;
        }

        let h = hash3(input, pos);
        let chain_start = head[h];

        // Find best match by walking the chain (BEFORE inserting current pos).
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        let mut chain = chain_start;
        let mut chain_count = 0u32;

        while chain > 0 && chain_count < 64 {
            let candidate = (chain - 1) as usize;
            let dist = pos - candidate;
            if dist > WINDOW_SIZE {
                break;
            }

            let max_len = MAX_MATCH.min(input.len() - pos);
            let mut mlen = 0;
            while mlen < max_len && input[candidate + mlen] == input[pos + mlen] {
                mlen += 1;
            }

            if mlen > best_len {
                best_len = mlen;
                best_dist = dist;
                if mlen == MAX_MATCH {
                    break;
                }
            }

            chain = prev[candidate];
            chain_count += 1;
        }

        // Now insert current position into hash chain.
        prev[pos] = chain_start;
        head[h] = (pos + 1) as u32;

        if best_len >= MIN_MATCH {
            tokens.push(Token::Match {
                length: best_len,
                distance: best_dist,
            });
            // Insert hash entries for skipped positions within the match.
            for i in 1..best_len {
                let p = pos + i;
                if p + 2 < input.len() {
                    let h2 = hash3(input, p);
                    prev[p] = head[h2];
                    head[h2] = (p + 1) as u32;
                }
            }
            pos += best_len;
        } else {
            tokens.push(Token::Literal(input[pos]));
            pos += 1;
        }
    }
    tokens
}

// ── Public API ───────────────────────────────────────────────────────

/// Compress data using DEFLATE with fixed Huffman codes.
///
/// Produces a raw DEFLATE stream (no zlib/gzip header).
pub fn compress(input: &[u8]) -> Vec<u8> {
    let tokens = find_matches(input);
    let codes = build_fixed_codes();
    let mut writer = BitWriter::new();

    // Block header: BFINAL=1, BTYPE=01 (fixed Huffman).
    writer.write_bits(1, 1); // BFINAL
    writer.write_bits(1, 2); // BTYPE = 01 (fixed Huffman)

    for token in &tokens {
        match token {
            Token::Literal(b) => {
                let (code, len) = codes.lit_encode[*b as usize];
                writer.write_huffman(code, len);
            }
            Token::Match { length, distance } => {
                // Encode length.
                let (len_sym, len_extra_bits, len_extra_val) = encode_length(*length);
                let (code, len) = codes.lit_encode[len_sym as usize];
                writer.write_huffman(code, len);
                if len_extra_bits > 0 {
                    writer.write_bits(len_extra_val, len_extra_bits);
                }
                // Encode distance.
                let (dist_sym, dist_extra_bits, dist_extra_val) = encode_distance(*distance);
                // Distance codes are all 5-bit fixed codes.
                writer.write_huffman(dist_sym, 5);
                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra_val, dist_extra_bits);
                }
            }
        }
    }

    // End of block (symbol 256).
    let (code, len) = codes.lit_encode[256];
    writer.write_huffman(code, len);
    writer.into_bytes()
}

/// Decompress a raw DEFLATE stream.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, DeflateError> {
    let mut reader = BitReader::new(data);
    let mut output = Vec::new();

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;

        match btype {
            0 => {
                // Stored block.
                reader.current = 0;
                reader.bits = 0;
                let block_len = reader.read_bits(16)? as usize;
                let _nlen = reader.read_bits(16)?;
                for _ in 0..block_len {
                    let b = reader.read_bits(8)? as u8;
                    output.push(b);
                }
            }
            1 => {
                // Fixed Huffman.
                let codes = build_fixed_codes();
                decode_block(&mut reader, &codes, &mut output)?;
            }
            _ => return Err(DeflateError::InvalidBlockType),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output)
}

fn decode_block(
    reader: &mut BitReader<'_>,
    codes: &FixedCodes,
    output: &mut Vec<u8>,
) -> Result<(), DeflateError> {
    loop {
        let sym = decode_symbol(reader, &codes.lit_decode, 9)?;

        if sym == 256 {
            // End of block.
            return Ok(());
        }

        if sym < 256 {
            output.push(sym as u8);
        } else {
            // Length/distance pair.
            let len_idx = (sym - 257) as usize;
            if len_idx >= LENGTH_TABLE.len() {
                return Err(DeflateError::InvalidLengthCode(sym));
            }
            let (base_len, extra_bits) = LENGTH_TABLE[len_idx];
            let extra = if extra_bits > 0 {
                reader.read_bits(extra_bits)? as usize
            } else {
                0
            };
            let length = base_len + extra;

            // Read distance.
            let dist_sym = decode_symbol(reader, &codes.dist_decode, 5)? as usize;
            if dist_sym >= DISTANCE_TABLE.len() {
                return Err(DeflateError::InvalidDistanceCode(dist_sym as u32));
            }
            let (base_dist, dist_extra_bits) = DISTANCE_TABLE[dist_sym];
            let dist_extra = if dist_extra_bits > 0 {
                reader.read_bits(dist_extra_bits)? as usize
            } else {
                0
            };
            let distance = base_dist + dist_extra;

            if distance > output.len() {
                return Err(DeflateError::InvalidBackReference {
                    distance,
                    output_size: output.len(),
                });
            }

            // Copy bytes (may overlap for run-length encoding).
            let src = output.len() - distance;
            for i in 0..length {
                let b = output[src + (i % distance)];
                output.push(b);
            }
        }
    }
}

/// Compress and return the compression ratio (compressed/original).
pub fn compression_ratio(input: &[u8]) -> f64 {
    if input.is_empty() {
        return 1.0;
    }
    let compressed = compress(input);
    compressed.len() as f64 / input.len() as f64
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        let data = b"";
        let compressed = compress(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn roundtrip_single_byte() {
        let data = b"X";
        let compressed = compress(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn roundtrip_hello_world() {
        let data = b"Hello, World!";
        let compressed = compress(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn roundtrip_repeated() {
        let data = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let compressed = compress(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn roundtrip_all_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let compressed = compress(&data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn roundtrip_pattern() {
        let data = b"abcabcabcabcabcabcabcabcabcabc";
        let compressed = compress(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn roundtrip_lorem() {
        let data = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                      Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        let compressed = compress(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compression_reduces_size_for_repetitive() {
        let data = "the quick brown fox ".repeat(50);
        let compressed = compress(data.as_bytes());
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn compression_ratio_calculation() {
        let data = "aaaaaaaaaa".repeat(100);
        let ratio = compression_ratio(data.as_bytes());
        assert!(ratio < 0.5, "expected good compression ratio, got {ratio}");
    }

    #[test]
    fn compression_ratio_empty() {
        assert!((compression_ratio(b"") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn invalid_data_returns_error() {
        let result = decompress(&[0xFF, 0xFF, 0xFF]);
        assert!(result.is_err());
    }

    #[test]
    fn roundtrip_binary_data() {
        let data: Vec<u8> = (0..1024).map(|i| (i * 7 + 13) as u8).collect();
        let compressed = compress(&data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn reverse_bits_correctness() {
        assert_eq!(reverse_bits(0b110, 3), 0b011);
        assert_eq!(reverse_bits(0b1010, 4), 0b0101);
        assert_eq!(reverse_bits(0b1, 1), 0b1);
    }

    #[test]
    fn encode_length_boundaries() {
        let (c, _, _) = encode_length(3);
        assert_eq!(c, 257);
        let (c, _, _) = encode_length(258);
        assert_eq!(c, 285);
        let (c, eb, ev) = encode_length(11);
        assert_eq!(c, 265);
        assert_eq!(eb, 1);
        assert_eq!(ev, 0);
    }

    #[test]
    fn match_finder_abc_pattern() {
        let data = b"abcabcabc";
        let tokens = find_matches(data);
        // Should be: 3 literals (a, b, c) then a match.
        assert_eq!(tokens.len(), 4, "expected 4 tokens, got {:?}", tokens);
        assert_eq!(tokens[0], Token::Literal(b'a'));
        assert_eq!(tokens[1], Token::Literal(b'b'));
        assert_eq!(tokens[2], Token::Literal(b'c'));
        match &tokens[3] {
            Token::Match { length, distance } => {
                assert_eq!(*distance, 3);
                assert_eq!(*length, 6);
            }
            _ => panic!("expected match token"),
        }
    }

    #[test]
    fn huffman_encode_decode_literal() {
        // Verify that encoding then decoding a single literal works.
        let codes = build_fixed_codes();
        let mut writer = BitWriter::new();
        writer.write_bits(1, 1); // BFINAL
        writer.write_bits(1, 2); // BTYPE=01

        // Write literal 'a' (97)
        let (code, len) = codes.lit_encode[97];
        writer.write_huffman(code, len);
        // Write end-of-block (256)
        let (code, len) = codes.lit_encode[256];
        writer.write_huffman(code, len);

        let compressed = writer.into_bytes();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, b"a");
    }

    #[test]
    fn huffman_encode_decode_match() {
        // Verify that encoding a literal + match roundtrips.
        let codes = build_fixed_codes();
        let mut writer = BitWriter::new();
        writer.write_bits(1, 1);
        writer.write_bits(1, 2);

        // Write 'a', 'b', 'c'
        for &b in b"abc" {
            let (code, len) = codes.lit_encode[b as usize];
            writer.write_huffman(code, len);
        }

        // Write match: length=3, distance=3
        let (len_sym, len_extra_bits, len_extra_val) = encode_length(3);
        let (code, len) = codes.lit_encode[len_sym as usize];
        writer.write_huffman(code, len);
        if len_extra_bits > 0 {
            writer.write_bits(len_extra_val, len_extra_bits);
        }

        let (dist_sym, dist_extra_bits, dist_extra_val) = encode_distance(3);
        writer.write_huffman(dist_sym, 5);
        if dist_extra_bits > 0 {
            writer.write_bits(dist_extra_val, dist_extra_bits);
        }

        // End of block
        let (code, len) = codes.lit_encode[256];
        writer.write_huffman(code, len);

        let compressed = writer.into_bytes();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, b"abcabc");
    }

    #[test]
    fn encode_distance_boundaries() {
        let (c, _, _) = encode_distance(1);
        assert_eq!(c, 0);
        let (c, eb, ev) = encode_distance(5);
        assert_eq!(c, 4);
        assert_eq!(eb, 1);
        assert_eq!(ev, 0);
    }
}
