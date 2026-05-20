//! Huffman coding — frequency-based variable-length encoding.
//!
//! Builds a Huffman tree from input byte frequencies, generates canonical
//! codes, and provides encode/decode operations. Replaces JavaScript
//! Huffman libraries with a pure Rust implementation.

use std::collections::BinaryHeap;
use std::cmp::Reverse;

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during Huffman coding operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HuffmanError {
    #[error("empty input")]
    EmptyInput,
    #[error("invalid encoded data")]
    InvalidData,
    #[error("unexpected end of encoded data")]
    UnexpectedEof,
    #[error("invalid code table")]
    InvalidCodeTable,
}

// ── Huffman Tree ─────────────────────────────────────────────────────

/// A node in the Huffman tree.
#[derive(Debug, Clone)]
enum HuffNode {
    Leaf { byte: u8, freq: u64 },
    Internal { freq: u64, left: Box<HuffNode>, right: Box<HuffNode> },
}

impl HuffNode {
    fn freq(&self) -> u64 {
        match self {
            HuffNode::Leaf { freq, .. } => *freq,
            HuffNode::Internal { freq, .. } => *freq,
        }
    }
}

impl PartialEq for HuffNode {
    fn eq(&self, other: &Self) -> bool {
        self.freq() == other.freq()
    }
}

impl Eq for HuffNode {}

impl PartialOrd for HuffNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HuffNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering: smaller freq = higher priority.
        other.freq().cmp(&self.freq())
    }
}

// ── Code Table ───────────────────────────────────────────────────────

/// A Huffman code entry: (bit pattern, bit length).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HuffCode {
    pub bits: u32,
    pub length: u8,
}

/// A Huffman code table mapping bytes to variable-length codes.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeTable {
    codes: [Option<HuffCode>; 256],
    /// Bit lengths for canonical code generation.
    bit_lengths: [u8; 256],
}

impl CodeTable {
    /// Serialize the code table to bytes for storage/transmission.
    pub fn serialize(&self) -> Vec<u8> {
        self.bit_lengths.to_vec()
    }

    /// Deserialize a code table from bytes.
    pub fn deserialize(data: &[u8]) -> Result<Self, HuffmanError> {
        if data.len() != 256 {
            return Err(HuffmanError::InvalidCodeTable);
        }
        let mut bit_lengths = [0u8; 256];
        bit_lengths.copy_from_slice(data);
        let codes = canonical_codes_from_lengths(&bit_lengths);
        let mut table = [None; 256];
        for (sym, &len) in bit_lengths.iter().enumerate() {
            if len > 0 {
                table[sym] = codes.get(&(sym as u8)).copied();
            }
        }
        Ok(Self { codes: table, bit_lengths })
    }

    /// Look up the code for a byte.
    pub fn get(&self, byte: u8) -> Option<HuffCode> {
        self.codes[byte as usize]
    }

    /// Get bit lengths array.
    pub fn bit_lengths(&self) -> &[u8; 256] {
        &self.bit_lengths
    }
}

// ── Frequency Table ──────────────────────────────────────────────────

/// Build a frequency table from input bytes.
pub fn frequency_table(data: &[u8]) -> [u64; 256] {
    let mut freq = [0u64; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    freq
}

// ── Build Huffman Tree ───────────────────────────────────────────────

/// Build a Huffman tree from a frequency table.
fn build_tree(freq: &[u64; 256]) -> Option<HuffNode> {
    let mut heap = BinaryHeap::new();
    for (byte, &f) in freq.iter().enumerate() {
        if f > 0 {
            heap.push(HuffNode::Leaf { byte: byte as u8, freq: f });
        }
    }

    if heap.is_empty() {
        return None;
    }

    // If only one symbol, add a dummy.
    if heap.len() == 1 {
        let node = heap.pop().unwrap();
        let dummy_byte = match &node {
            HuffNode::Leaf { byte, .. } => byte.wrapping_add(1),
            _ => 0,
        };
        heap.push(node);
        heap.push(HuffNode::Leaf { byte: dummy_byte, freq: 0 });
    }

    while heap.len() > 1 {
        let left = heap.pop().unwrap();
        let right = heap.pop().unwrap();
        let combined = HuffNode::Internal {
            freq: left.freq() + right.freq(),
            left: Box::new(left),
            right: Box::new(right),
        };
        heap.push(combined);
    }

    heap.pop()
}

/// Extract bit lengths from the tree.
fn extract_lengths(node: &HuffNode, depth: u8, lengths: &mut [u8; 256]) {
    match node {
        HuffNode::Leaf { byte, .. } => {
            lengths[*byte as usize] = if depth == 0 { 1 } else { depth };
        }
        HuffNode::Internal { left, right, .. } => {
            extract_lengths(left, depth + 1, lengths);
            extract_lengths(right, depth + 1, lengths);
        }
    }
}

/// Generate canonical Huffman codes from bit lengths.
fn canonical_codes_from_lengths(lengths: &[u8; 256]) -> std::collections::HashMap<u8, HuffCode> {
    // Collect symbols with non-zero lengths, sorted by (length, symbol).
    let mut symbols: Vec<(u8, u8)> = Vec::new(); // (symbol, length)
    for (sym, &len) in lengths.iter().enumerate() {
        if len > 0 {
            symbols.push((sym as u8, len));
        }
    }
    symbols.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    let mut codes = std::collections::HashMap::new();
    let mut code = 0u32;
    let mut prev_len = 0u8;

    for (sym, len) in symbols {
        if prev_len > 0 {
            code += 1;
        }
        if len > prev_len {
            code <<= len - prev_len;
        }
        codes.insert(sym, HuffCode { bits: code, length: len });
        prev_len = len;
    }

    codes
}

// ── Build Code Table ─────────────────────────────────────────────────

/// Build a Huffman code table from input data.
pub fn build_code_table(data: &[u8]) -> Result<CodeTable, HuffmanError> {
    if data.is_empty() {
        return Err(HuffmanError::EmptyInput);
    }

    let freq = frequency_table(data);
    let tree = build_tree(&freq).ok_or(HuffmanError::EmptyInput)?;

    let mut bit_lengths = [0u8; 256];
    extract_lengths(&tree, 0, &mut bit_lengths);

    let canonical = canonical_codes_from_lengths(&bit_lengths);
    let mut codes = [None; 256];
    for (&sym, &hc) in &canonical {
        codes[sym as usize] = Some(hc);
    }

    Ok(CodeTable { codes, bit_lengths })
}

// ── Encode ───────────────────────────────────────────────────────────

/// Encode bytes using the given code table. Returns (encoded_bits, bit_count).
pub fn encode(data: &[u8], table: &CodeTable) -> Result<(Vec<u8>, u64), HuffmanError> {
    let mut bits = Vec::new();
    let mut current = 0u8;
    let mut bit_pos = 0u8;
    let mut total_bits = 0u64;

    for &byte in data {
        let code = table.get(byte).ok_or(HuffmanError::InvalidData)?;
        for i in (0..code.length).rev() {
            let bit = (code.bits >> i) & 1;
            current |= (bit as u8) << (7 - bit_pos);
            bit_pos += 1;
            total_bits += 1;
            if bit_pos == 8 {
                bits.push(current);
                current = 0;
                bit_pos = 0;
            }
        }
    }

    if bit_pos > 0 {
        bits.push(current);
    }

    Ok((bits, total_bits))
}

/// Decode bits back to bytes using the code table.
pub fn decode(
    encoded: &[u8],
    bit_count: u64,
    table: &CodeTable,
) -> Result<Vec<u8>, HuffmanError> {
    // Build a lookup from (bits, length) -> symbol.
    let mut lookup: std::collections::HashMap<(u32, u8), u8> = std::collections::HashMap::new();
    for sym in 0u16..256 {
        if let Some(code) = table.get(sym as u8) {
            lookup.insert((code.bits, code.length), sym as u8);
        }
    }

    let max_len = table.bit_lengths.iter().copied().max().unwrap_or(0);
    let mut output = Vec::new();
    let mut bits_read = 0u64;
    let mut code = 0u32;
    let mut code_len = 0u8;

    for &byte in encoded {
        for shift in (0..8).rev() {
            if bits_read >= bit_count {
                break;
            }
            let bit = (byte >> shift) & 1;
            code = (code << 1) | bit as u32;
            code_len += 1;
            bits_read += 1;

            if let Some(&sym) = lookup.get(&(code, code_len)) {
                output.push(sym);
                code = 0;
                code_len = 0;
            } else if code_len > max_len {
                return Err(HuffmanError::InvalidData);
            }
        }
    }

    Ok(output)
}

// ── Compression Ratio ────────────────────────────────────────────────

/// Calculate the compression ratio (encoded_size / original_size).
pub fn compression_ratio(data: &[u8]) -> Result<f64, HuffmanError> {
    if data.is_empty() {
        return Err(HuffmanError::EmptyInput);
    }
    let table = build_code_table(data)?;
    let (encoded, _bit_count) = encode(data, &table)?;
    let overhead = 256; // code table serialization
    Ok((encoded.len() + overhead) as f64 / data.len() as f64)
}

// ── Adaptive Huffman ─────────────────────────────────────────────────

/// Adaptive Huffman encoder that rebuilds the tree periodically.
pub struct AdaptiveEncoder {
    freq: [u64; 256],
    rebuild_interval: usize,
    count: usize,
    table: Option<CodeTable>,
    output_bits: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
    total_bits: u64,
}

impl AdaptiveEncoder {
    /// Create a new adaptive encoder with the given rebuild interval.
    pub fn new(rebuild_interval: usize) -> Self {
        Self {
            freq: [1u64; 256], // Start with uniform distribution.
            rebuild_interval,
            count: 0,
            table: None,
            output_bits: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
            total_bits: 0,
        }
    }

    /// Update frequency and rebuild table if needed.
    pub fn update(&mut self, byte: u8) {
        self.freq[byte as usize] += 1;
        self.count += 1;
        if self.count % self.rebuild_interval == 0 || self.table.is_none() {
            self.rebuild();
        }
    }

    fn rebuild(&mut self) {
        let tree = build_tree(&self.freq);
        if let Some(tree) = tree {
            let mut bit_lengths = [0u8; 256];
            extract_lengths(&tree, 0, &mut bit_lengths);
            let canonical = canonical_codes_from_lengths(&bit_lengths);
            let mut codes = [None; 256];
            for (&sym, &hc) in &canonical {
                codes[sym as usize] = Some(hc);
            }
            self.table = Some(CodeTable { codes, bit_lengths });
        }
    }

    /// Encode a single byte using current codes.
    pub fn encode_byte(&mut self, byte: u8) {
        self.update(byte);
        if let Some(table) = &self.table {
            if let Some(code) = table.get(byte) {
                for i in (0..code.length).rev() {
                    let bit = (code.bits >> i) & 1;
                    self.current_byte |= (bit as u8) << (7 - self.bit_pos);
                    self.bit_pos += 1;
                    self.total_bits += 1;
                    if self.bit_pos == 8 {
                        self.output_bits.push(self.current_byte);
                        self.current_byte = 0;
                        self.bit_pos = 0;
                    }
                }
            }
        }
    }

    /// Finish encoding and return the output.
    pub fn finish(mut self) -> (Vec<u8>, u64) {
        if self.bit_pos > 0 {
            self.output_bits.push(self.current_byte);
        }
        (self.output_bits, self.total_bits)
    }

    /// Get current frequency table.
    pub fn frequencies(&self) -> &[u64; 256] {
        &self.freq
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_table_counts() {
        let data = b"aabbc";
        let freq = frequency_table(data);
        assert_eq!(freq[b'a' as usize], 2);
        assert_eq!(freq[b'b' as usize], 2);
        assert_eq!(freq[b'c' as usize], 1);
        assert_eq!(freq[b'd' as usize], 0);
    }

    #[test]
    fn build_code_table_single_char() {
        let data = b"aaa";
        let table = build_code_table(data).unwrap();
        let code = table.get(b'a').unwrap();
        assert!(code.length > 0);
    }

    #[test]
    fn roundtrip_simple() {
        let data = b"hello world";
        let table = build_code_table(data).unwrap();
        let (encoded, bit_count) = encode(data, &table).unwrap();
        let decoded = decode(&encoded, bit_count, &table).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn roundtrip_all_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let table = build_code_table(&data).unwrap();
        let (encoded, bit_count) = encode(&data, &table).unwrap();
        let decoded = decode(&encoded, bit_count, &table).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn roundtrip_repeated() {
        let data = b"aaaaaaaaaa";
        let table = build_code_table(data).unwrap();
        let (encoded, bit_count) = encode(data, &table).unwrap();
        let decoded = decode(&encoded, bit_count, &table).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn code_table_serialize_deserialize() {
        let data = b"abracadabra";
        let table = build_code_table(data).unwrap();
        let serialized = table.serialize();
        let restored = CodeTable::deserialize(&serialized).unwrap();
        // Verify codes match.
        for b in b"abracadabra" {
            assert_eq!(table.get(*b), restored.get(*b));
        }
    }

    #[test]
    fn shorter_codes_for_frequent_bytes() {
        let data = b"aaaaaabbcd";
        let table = build_code_table(data).unwrap();
        let a_code = table.get(b'a').unwrap();
        let d_code = table.get(b'd').unwrap();
        assert!(a_code.length <= d_code.length);
    }

    #[test]
    fn compression_ratio_works() {
        let data = "aaa".repeat(100);
        let ratio = compression_ratio(data.as_bytes()).unwrap();
        // With 256 bytes overhead for table, ratio includes that.
        assert!(ratio < 2.0); // Sanity check.
    }

    #[test]
    fn empty_input_error() {
        assert!(matches!(build_code_table(b""), Err(HuffmanError::EmptyInput)));
    }

    #[test]
    fn invalid_table_deserialize() {
        assert!(matches!(CodeTable::deserialize(&[1, 2, 3]), Err(HuffmanError::InvalidCodeTable)));
    }

    #[test]
    fn adaptive_encoder_basic() {
        let mut encoder = AdaptiveEncoder::new(4);
        for &b in b"hello world hello" {
            encoder.encode_byte(b);
        }
        let (output, total_bits) = encoder.finish();
        assert!(total_bits > 0);
        assert!(!output.is_empty());
    }

    #[test]
    fn adaptive_encoder_frequencies() {
        let mut encoder = AdaptiveEncoder::new(10);
        for &b in b"aab" {
            encoder.encode_byte(b);
        }
        let freq = encoder.frequencies();
        // Initial freq is 1 for all, then a gets +2, b gets +1.
        assert_eq!(freq[b'a' as usize], 3);
        assert_eq!(freq[b'b' as usize], 2);
    }

    #[test]
    fn canonical_codes_are_prefix_free() {
        let data = b"abcdefgh";
        let table = build_code_table(data).unwrap();
        // Verify no code is a prefix of another.
        let mut codes: Vec<HuffCode> = Vec::new();
        for i in 0..=255u8 {
            if let Some(c) = table.get(i) {
                codes.push(c);
            }
        }
        for (i, a) in codes.iter().enumerate() {
            for (j, b) in codes.iter().enumerate() {
                if i == j {
                    continue;
                }
                if a.length < b.length {
                    let shifted = b.bits >> (b.length - a.length);
                    assert_ne!(shifted, a.bits, "code {a:?} is prefix of {b:?}");
                }
            }
        }
    }
}
