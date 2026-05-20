//! QR code generation.
//!
//! Replaces qrcode.js and node-qrcode with a pure Rust QR code
//! generator supporting versions 1-4 (up to ~78 bytes) in byte
//! mode with error correction level M.

use std::fmt::Write as FmtWrite;

// ── Error ──────────────────────────────────────────────────────────

/// Errors during QR code generation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum QrError {
    #[error("data too long for QR version 1-4 ({0} bytes, max 78)")]
    DataTooLong(usize),
    #[error("invalid input")]
    InvalidInput,
}

// ── Version capacities ─────────────────────────────────────────────

/// Data codeword capacity per version at EC level M (byte mode).
/// (version, total_codewords, ec_codewords_per_block, num_blocks, data_capacity_bytes)
const VERSION_INFO: [(u8, usize, usize, usize, usize); 4] = [
    (1, 26, 10, 1, 14),
    (2, 44, 16, 1, 24),
    (3, 70, 26, 1, 42),
    (4, 100, 18, 2, 78),
];

// ── QR Code ────────────────────────────────────────────────────────

/// A generated QR code represented as a boolean grid.
/// `true` = dark module, `false` = light module.
#[derive(Debug, Clone)]
pub struct QrCode {
    /// The module grid (row-major).
    pub modules: Vec<Vec<bool>>,
    /// QR version (1-4).
    pub version: u8,
    /// Side length in modules.
    size: usize,
}

impl QrCode {
    /// Encode a string as a QR code (byte mode, EC level M).
    pub fn encode(data: &str) -> Result<Self, QrError> {
        let bytes = data.as_bytes();

        // Select version
        let vi = VERSION_INFO
            .iter()
            .find(|v| bytes.len() <= v.4)
            .ok_or(QrError::DataTooLong(bytes.len()))?;

        let version = vi.0;
        let total_codewords = vi.1;
        let ec_per_block = vi.2;
        let num_blocks = vi.3;
        let size = 17 + 4 * version as usize;

        // Encode data bits (byte mode indicator = 0100)
        let data_codewords = total_codewords - ec_per_block * num_blocks;
        let mut bits = BitVec::new();
        // Mode indicator: byte = 0100
        bits.push_bits(0b0100, 4);
        // Character count (8 bits for v1-9)
        bits.push_bits(bytes.len() as u32, 8);
        // Data bytes
        for &b in bytes {
            bits.push_bits(u32::from(b), 8);
        }
        // Terminator (up to 4 zero bits)
        let remaining = data_codewords * 8 - bits.len().min(data_codewords * 8);
        let term_bits = remaining.min(4);
        bits.push_bits(0, term_bits);
        // Pad to byte boundary
        while bits.len() % 8 != 0 {
            bits.push_bit(false);
        }
        // Pad codewords
        let pad_bytes = [0xEC, 0x11];
        let mut pad_idx = 0;
        while bits.len() < data_codewords * 8 {
            bits.push_bits(pad_bytes[pad_idx % 2], 8);
            pad_idx += 1;
        }

        // Convert bits to codewords
        let data_cws: Vec<u8> = (0..data_codewords)
            .map(|i| bits.byte_at(i * 8))
            .collect();

        // Generate error correction
        let mut all_codewords = Vec::new();
        let block_size = data_codewords / num_blocks;
        let mut ec_blocks = Vec::new();

        for block in 0..num_blocks {
            let start = block * block_size;
            let end = if block == num_blocks - 1 {
                data_codewords
            } else {
                start + block_size
            };
            let block_data = &data_cws[start..end];
            all_codewords.extend_from_slice(block_data);
            let ec = reed_solomon_ec(block_data, ec_per_block);
            ec_blocks.push(ec);
        }
        for ec in &ec_blocks {
            all_codewords.extend_from_slice(ec);
        }

        // Build module grid
        let mut modules = vec![vec![false; size]; size];
        let mut reserved = vec![vec![false; size]; size];

        // Place finder patterns
        place_finder_pattern(&mut modules, &mut reserved, 0, 0);
        place_finder_pattern(&mut modules, &mut reserved, 0, size - 7);
        place_finder_pattern(&mut modules, &mut reserved, size - 7, 0);

        // Separators
        for i in 0..8 {
            // Top-left
            reserve(&mut reserved, &mut modules, i, 7, false);
            reserve(&mut reserved, &mut modules, 7, i, false);
            // Top-right
            if size > 7 + i {
                reserve(&mut reserved, &mut modules, i, size - 8, false);
            }
            reserve(&mut reserved, &mut modules, 7, size - 8 + i, false);
            // Bottom-left
            reserve(&mut reserved, &mut modules, size - 8, i, false);
            if i < 8 {
                reserve(&mut reserved, &mut modules, size - 8 + i, 7, false);
            }
        }

        // Timing patterns
        for i in 8..size - 8 {
            let val = i % 2 == 0;
            reserve(&mut reserved, &mut modules, 6, i, val);
            reserve(&mut reserved, &mut modules, i, 6, val);
        }

        // Dark module
        reserve(&mut reserved, &mut modules, size - 8, 8, true);

        // Reserve format info areas
        for i in 0..9 {
            if !reserved[8][i] {
                reserved[8][i] = true;
            }
            if i < 9 && !reserved[i][8] {
                reserved[i][8] = true;
            }
        }
        for i in 0..8 {
            if size >= 8 && !reserved[8][size - 8 + i] {
                reserved[8][size - 8 + i] = true;
            }
        }
        for i in 0..7 {
            if !reserved[size - 7 + i][8] {
                reserved[size - 7 + i][8] = true;
            }
        }

        // Place data bits
        place_data_bits(&mut modules, &reserved, &all_codewords, size);

        // Apply mask pattern 0 (checkerboard: (row + col) % 2 == 0)
        for row in 0..size {
            for col in 0..size {
                if !reserved[row][col] && (row + col) % 2 == 0 {
                    modules[row][col] = !modules[row][col];
                }
            }
        }

        // Place format info (mask 0, EC level M = 00)
        let format_bits = get_format_bits(0);
        place_format_info(&mut modules, format_bits, size);

        Ok(QrCode {
            modules,
            version,
            size,
        })
    }

    /// Side length of the QR code in modules.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the value of a module (true = dark).
    pub fn get(&self, row: usize, col: usize) -> bool {
        self.modules
            .get(row)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(false)
    }

    /// Render as a Unicode string using block characters.
    pub fn to_string_art(&self) -> String {
        let mut out = String::new();
        let mut row = 0;
        while row < self.size {
            for col in 0..self.size {
                let top = self.modules[row][col];
                let bot = if row + 1 < self.size {
                    self.modules[row + 1][col]
                } else {
                    false
                };
                match (top, bot) {
                    (true, true) => out.push('\u{2588}'),
                    (true, false) => out.push('\u{2580}'),
                    (false, true) => out.push('\u{2584}'),
                    (false, false) => out.push(' '),
                }
            }
            out.push('\n');
            row += 2;
        }
        out
    }

    /// Render as an SVG string.
    pub fn to_svg(&self, module_size: f64) -> String {
        let total = self.size as f64 * module_size;
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {t} {t}" width="{t}" height="{t}">"#,
            t = total
        );
        let _ = write!(
            svg,
            r#"<rect width="{t}" height="{t}" fill="white"/>"#,
            t = total
        );
        for row in 0..self.size {
            for col in 0..self.size {
                if self.modules[row][col] {
                    let x = col as f64 * module_size;
                    let y = row as f64 * module_size;
                    let _ = write!(
                        svg,
                        r#"<rect x="{x}" y="{y}" width="{m}" height="{m}" fill="black"/>"#,
                        x = x,
                        y = y,
                        m = module_size
                    );
                }
            }
        }
        svg.push_str("</svg>");
        svg
    }
}

// ── Finder pattern ─────────────────────────────────────────────────

fn place_finder_pattern(
    modules: &mut [Vec<bool>],
    reserved: &mut [Vec<bool>],
    row: usize,
    col: usize,
) {
    for r in 0..7 {
        for c in 0..7 {
            let val = matches!(
                (r, c),
                (0, _) | (6, _) | (_, 0) | (_, 6) | (2..=4, 2..=4)
            );
            let rr = row + r;
            let cc = col + c;
            if rr < modules.len() && cc < modules[0].len() {
                modules[rr][cc] = val;
                reserved[rr][cc] = true;
            }
        }
    }
}

fn reserve(
    reserved: &mut [Vec<bool>],
    modules: &mut [Vec<bool>],
    row: usize,
    col: usize,
    val: bool,
) {
    if row < reserved.len() && col < reserved[0].len() {
        reserved[row][col] = true;
        modules[row][col] = val;
    }
}

// ── Data placement ─────────────────────────────────────────────────

fn place_data_bits(
    modules: &mut [Vec<bool>],
    reserved: &[Vec<bool>],
    codewords: &[u8],
    size: usize,
) {
    let mut bit_idx = 0;
    let total_bits = codewords.len() * 8;
    let mut col = size as isize - 1;

    while col >= 0 {
        if col == 6 {
            col -= 1;
            continue;
        }

        let upward = ((size as isize - 1 - col) / 2) % 2 == 0;

        for step in 0..size {
            let row = if upward { size - 1 - step } else { step };

            for dc in [0i32, -1] {
                let c = (col + dc as isize) as usize;
                if c < size && !reserved[row][c] {
                    if bit_idx < total_bits {
                        let byte_idx = bit_idx / 8;
                        let bit_pos = 7 - (bit_idx % 8);
                        modules[row][c] = (codewords[byte_idx] >> bit_pos) & 1 == 1;
                        bit_idx += 1;
                    }
                }
            }
        }

        col -= 2;
    }
}

// ── Format info ────────────────────────────────────────────────────

fn get_format_bits(mask_pattern: u8) -> u16 {
    let data = (0b00u16 << 3) | (mask_pattern as u16 & 0x7);
    let mut format = data << 10;

    let generator = 0b10100110111u16;
    let mut rem = format;
    for i in (0..=4).rev() {
        if rem & (1 << (i + 10)) != 0 {
            rem ^= generator << i;
        }
    }
    format |= rem;
    format ^= 0b101010000010010;
    format
}

fn place_format_info(modules: &mut [Vec<bool>], format_bits: u16, size: usize) {
    let positions_h: [(usize, usize); 15] = [
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 3),
        (8, 4),
        (8, 5),
        (8, 7),
        (8, 8),
        (7, 8),
        (5, 8),
        (4, 8),
        (3, 8),
        (2, 8),
        (1, 8),
        (0, 8),
    ];

    for (i, &(r, c)) in positions_h.iter().enumerate() {
        let bit = (format_bits >> i) & 1 == 1;
        modules[r][c] = bit;
    }

    let positions_v: [(usize, usize); 15] = [
        (size - 1, 8),
        (size - 2, 8),
        (size - 3, 8),
        (size - 4, 8),
        (size - 5, 8),
        (size - 6, 8),
        (size - 7, 8),
        (8, size - 8),
        (8, size - 7),
        (8, size - 6),
        (8, size - 5),
        (8, size - 4),
        (8, size - 3),
        (8, size - 2),
        (8, size - 1),
    ];

    for (i, &(r, c)) in positions_v.iter().enumerate() {
        let bit = (format_bits >> i) & 1 == 1;
        if r < size && c < size {
            modules[r][c] = bit;
        }
    }
}

// ── Reed-Solomon error correction ──────────────────────────────────

fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let (la, lb) = (GF_LOG[a as usize], GF_LOG[b as usize]);
    GF_EXP[((la as u16 + lb as u16) % 255) as usize]
}

fn reed_solomon_ec(data: &[u8], ec_count: usize) -> Vec<u8> {
    let genpoly = generator_polynomial(ec_count);
    let mut result = vec![0u8; ec_count];
    let mut message = data.to_vec();
    message.extend_from_slice(&result);

    for i in 0..data.len() {
        let coef = message[i];
        if coef != 0 {
            for j in 0..genpoly.len() {
                message[i + j] ^= gf_mul(genpoly[j], coef);
            }
        }
    }

    result.copy_from_slice(&message[data.len()..]);
    result
}

fn generator_polynomial(degree: usize) -> Vec<u8> {
    let mut poly = vec![1u8];
    for i in 0..degree {
        let mut new_poly = vec![0u8; poly.len() + 1];
        let root = GF_EXP[i % 255];
        for (j, &coef) in poly.iter().enumerate() {
            new_poly[j] ^= coef;
            new_poly[j + 1] ^= gf_mul(coef, root);
        }
        poly = new_poly;
    }
    poly
}

// ── GF(256) tables ─────────────────────────────────────────────────

const GF_EXP: [u8; 512] = {
    let mut exp = [0u8; 512];
    let mut val = 1u16;
    let mut i = 0;
    while i < 255 {
        exp[i] = val as u8;
        exp[i + 255] = val as u8;
        val <<= 1;
        if val >= 256 {
            val ^= 0x11D;
        }
        i += 1;
    }
    exp
};

const GF_LOG: [u8; 256] = {
    let mut log = [0u8; 256];
    let mut i = 0;
    while i < 255 {
        log[GF_EXP[i] as usize] = i as u8;
        i += 1;
    }
    log
};

// ── Bit vector helper ──────────────────────────────────────────────

struct BitVec {
    data: Vec<u8>,
    bit_count: usize,
}

impl BitVec {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_count: 0,
        }
    }

    fn len(&self) -> usize {
        self.bit_count
    }

    fn push_bit(&mut self, val: bool) {
        let byte_idx = self.bit_count / 8;
        let bit_idx = 7 - (self.bit_count % 8);
        if byte_idx >= self.data.len() {
            self.data.push(0);
        }
        if val {
            self.data[byte_idx] |= 1 << bit_idx;
        }
        self.bit_count += 1;
    }

    fn push_bits(&mut self, value: u32, count: usize) {
        for i in (0..count).rev() {
            self.push_bit((value >> i) & 1 == 1);
        }
    }

    fn byte_at(&self, bit_offset: usize) -> u8 {
        let mut val = 0u8;
        for i in 0..8 {
            let idx = bit_offset + i;
            if idx < self.bit_count {
                let byte_idx = idx / 8;
                let bit_idx = 7 - (idx % 8);
                if self.data[byte_idx] & (1 << bit_idx) != 0 {
                    val |= 1 << (7 - i);
                }
            }
        }
        val
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_short_string() {
        let qr = QrCode::encode("Hello").unwrap();
        assert!(qr.size() > 0);
        assert_eq!(qr.modules.len(), qr.size());
    }

    #[test]
    fn size_correct_for_version_1() {
        let qr = QrCode::encode("Hi").unwrap();
        assert_eq!(qr.version, 1);
        assert_eq!(qr.size(), 21);
    }

    #[test]
    fn size_correct_for_version_2() {
        let qr = QrCode::encode("abcdefghijklmno").unwrap();
        assert!(qr.version >= 2);
        assert_eq!(qr.size(), 17 + 4 * qr.version as usize);
    }

    #[test]
    fn to_string_produces_output() {
        let qr = QrCode::encode("test").unwrap();
        let s = qr.to_string_art();
        assert!(!s.is_empty());
        assert!(s.contains('\n'));
    }

    #[test]
    fn to_svg_has_rect_elements() {
        let qr = QrCode::encode("SVG").unwrap();
        let svg = qr.to_svg(4.0);
        assert!(svg.contains("<rect"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("fill=\"black\""));
    }

    #[test]
    fn data_too_long_errors() {
        let long = "x".repeat(100);
        let result = QrCode::encode(&long);
        assert!(result.is_err());
        match result.unwrap_err() {
            QrError::DataTooLong(_) => {}
            other => panic!("expected DataTooLong, got {:?}", other),
        }
    }

    #[test]
    fn finder_patterns_present() {
        let qr = QrCode::encode("QR").unwrap();
        let s = qr.size();
        // Top-left 7x7 finder: corners dark
        assert!(qr.get(0, 0));
        assert!(qr.get(0, 6));
        assert!(qr.get(6, 0));
        assert!(qr.get(6, 6));
        // Top-right finder
        assert!(qr.get(0, s - 7));
        assert!(qr.get(0, s - 1));
        // Bottom-left finder
        assert!(qr.get(s - 7, 0));
        assert!(qr.get(s - 1, 0));
    }

    #[test]
    fn empty_string_encodes() {
        let qr = QrCode::encode("").unwrap();
        assert_eq!(qr.version, 1);
        assert_eq!(qr.size(), 21);
    }

    #[test]
    fn get_out_of_bounds_returns_false() {
        let qr = QrCode::encode("x").unwrap();
        assert!(!qr.get(999, 999));
    }
}
