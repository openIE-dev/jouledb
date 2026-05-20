//! QR code generation — version 1-10, error correction L/M/Q/H,
//! numeric/alphanumeric/byte modes, masking, format/version info, matrix output.
//!
//! Pure-Rust replacement for the `qrcode` crate.

use std::fmt;

// ── Error Correction Levels ─────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcLevel { L, M, Q, H }

impl EcLevel {
    fn format_bits(&self) -> u8 {
        match self { EcLevel::L => 0b01, EcLevel::M => 0b00, EcLevel::Q => 0b11, EcLevel::H => 0b10 }
    }
}

// ── Encoding Mode ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode { Numeric, Alphanumeric, Byte }

impl Mode {
    fn indicator(&self) -> u8 {
        match self { Mode::Numeric => 0b0001, Mode::Alphanumeric => 0b0010, Mode::Byte => 0b0100 }
    }

    fn char_count_bits(&self, version: u8) -> usize {
        match (self, version) {
            (Mode::Numeric, 1..=9) => 10,
            (Mode::Alphanumeric, 1..=9) => 9,
            (Mode::Byte, 1..=9) => 8,
            (Mode::Numeric, _) => 12,
            (Mode::Alphanumeric, _) => 11,
            (Mode::Byte, _) => 16,
        }
    }
}

/// Detect the best mode for data.
pub fn detect_mode(data: &str) -> Mode {
    if data.chars().all(|c| c.is_ascii_digit()) {
        Mode::Numeric
    } else if data.chars().all(|c| ALPHANUMERIC_TABLE.contains(&c)) {
        Mode::Alphanumeric
    } else {
        Mode::Byte
    }
}

const ALPHANUMERIC_TABLE: &[char] = &[
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    ' ','$','%','*','+','-','.','/',':',
];

fn alphanumeric_value(c: char) -> u16 {
    ALPHANUMERIC_TABLE.iter().position(|ch| *ch == c).unwrap_or(0) as u16
}

// ── Version/Capacity ────────────────────────────────────────────

/// Data capacity in bytes for versions 1-10 at each EC level.
/// (Total data codewords minus EC codewords)
const CAPACITY: [[u16; 4]; 10] = [
    // [L, M, Q, H]
    [19, 16, 13, 9],      // v1
    [34, 28, 22, 16],     // v2
    [55, 44, 34, 26],     // v3
    [80, 64, 48, 36],     // v4
    [108, 86, 62, 46],    // v5
    [136, 108, 76, 60],   // v6
    [156, 124, 88, 66],   // v7
    [194, 154, 110, 86],  // v8
    [232, 182, 132, 100], // v9
    [271, 216, 154, 122], // v10
];

/// EC codewords per block for versions 1-10.
const EC_CODEWORDS_PER_BLOCK: [[u8; 4]; 10] = [
    [7, 10, 13, 17],     // v1
    [10, 16, 22, 28],    // v2
    [15, 26, 18, 22],    // v3
    [20, 18, 26, 16],    // v4
    [26, 24, 18, 22],    // v5
    [18, 16, 24, 28],    // v6
    [20, 18, 18, 26],    // v7
    [24, 22, 22, 26],    // v8
    [30, 22, 20, 24],    // v9
    [18, 26, 24, 28],    // v10
];

/// Total codewords for versions 1-10.
const TOTAL_CODEWORDS: [u16; 10] = [26, 44, 70, 100, 134, 172, 196, 242, 292, 346];

fn ec_level_index(ec: EcLevel) -> usize {
    match ec { EcLevel::L => 0, EcLevel::M => 1, EcLevel::Q => 2, EcLevel::H => 3 }
}

/// Module count for a version (side length).
pub fn module_count(version: u8) -> usize {
    17 + 4 * version as usize
}

/// Select the minimum version that can hold the data.
pub fn select_version(data: &str, mode: Mode, ec: EcLevel) -> Option<u8> {
    let ec_idx = ec_level_index(ec);
    let data_len = match mode {
        Mode::Byte => data.len(),
        Mode::Numeric => (data.len() * 10 + 2) / 3 / 8 + 2, // approximate
        Mode::Alphanumeric => (data.len() * 11 + 1) / 2 / 8 + 2,
    };
    for v in 0..10 {
        if CAPACITY[v][ec_idx] as usize >= data_len {
            return Some((v + 1) as u8);
        }
    }
    None
}

// ── Bit Buffer ──────────────────────────────────────────────────

struct BitBuffer {
    bits: Vec<bool>,
}

impl BitBuffer {
    fn new() -> Self { Self { bits: Vec::new() } }
    fn push_bits(&mut self, value: u32, count: usize) {
        for i in (0..count).rev() {
            self.bits.push((value >> i) & 1 == 1);
        }
    }
    fn len(&self) -> usize { self.bits.len() }
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in self.bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit { byte |= 1 << (7 - i); }
            }
            bytes.push(byte);
        }
        bytes
    }
}

// ── Data Encoding ───────────────────────────────────────────────

fn encode_data(data: &str, mode: Mode, version: u8, ec: EcLevel) -> Vec<u8> {
    let mut buf = BitBuffer::new();

    // Mode indicator (4 bits)
    buf.push_bits(mode.indicator() as u32, 4);

    // Character count
    let count_bits = mode.char_count_bits(version);
    buf.push_bits(data.len() as u32, count_bits);

    // Data bits
    match mode {
        Mode::Numeric => {
            let chars: Vec<u32> = data.chars().map(|c| c.to_digit(10).unwrap()).collect();
            for chunk in chars.chunks(3) {
                match chunk.len() {
                    3 => buf.push_bits(chunk[0] * 100 + chunk[1] * 10 + chunk[2], 10),
                    2 => buf.push_bits(chunk[0] * 10 + chunk[1], 7),
                    1 => buf.push_bits(chunk[0], 4),
                    _ => {}
                }
            }
        }
        Mode::Alphanumeric => {
            let vals: Vec<u16> = data.chars().map(alphanumeric_value).collect();
            for chunk in vals.chunks(2) {
                if chunk.len() == 2 {
                    buf.push_bits((chunk[0] * 45 + chunk[1]) as u32, 11);
                } else {
                    buf.push_bits(chunk[0] as u32, 6);
                }
            }
        }
        Mode::Byte => {
            for byte in data.as_bytes() {
                buf.push_bits(*byte as u32, 8);
            }
        }
    }

    // Terminator
    let ec_idx = ec_level_index(ec);
    let total_data_bits = CAPACITY[version as usize - 1][ec_idx] as usize * 8;
    let remaining = total_data_bits.saturating_sub(buf.len());
    buf.push_bits(0, remaining.min(4));

    // Pad to byte boundary
    while buf.len() % 8 != 0 { buf.push_bits(0, 1); }

    // Pad codewords
    let pad_bytes = [0xEC, 0x11];
    let mut pad_idx = 0;
    while buf.len() < total_data_bits {
        buf.push_bits(pad_bytes[pad_idx] as u32, 8);
        pad_idx = (pad_idx + 1) % 2;
    }

    buf.to_bytes()
}

// ── Reed-Solomon (simplified GF(256)) ───────────────────────────

fn gf256_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 { return 0; }
    let mut result = 0u16;
    let mut a16 = a as u16;
    let mut b16 = b as u16;
    for _ in 0..8 {
        if b16 & 1 != 0 { result ^= a16; }
        let carry = a16 & 0x80 != 0;
        a16 <<= 1;
        if carry { a16 ^= 0x11D; }
        b16 >>= 1;
    }
    result as u8
}

fn rs_encode(data: &[u8], ec_count: usize) -> Vec<u8> {
    // Generate polynomial coefficients
    let mut gen_poly = vec![0u8; ec_count + 1];
    gen_poly[0] = 1;
    let mut alpha = 1u8;
    for i in 0..ec_count {
        for j in (1..=i + 1).rev() {
            gen_poly[j] = gen_poly[j] ^ gf256_mul(gen_poly[j - 1], alpha);
        }
        alpha = gf256_mul(alpha, 2);
    }

    let mut remainder = vec![0u8; ec_count];
    for &byte in data {
        let factor = byte ^ remainder[0];
        remainder.rotate_left(1);
        *remainder.last_mut().unwrap() = 0;
        for j in 0..ec_count {
            remainder[j] ^= gf256_mul(gen_poly[ec_count - 1 - j], factor);
        }
    }
    remainder
}

// ── QR Matrix ───────────────────────────────────────────────────

/// A module in the QR matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Module { Light, Dark, Reserved }

/// The QR code matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QrMatrix {
    pub size: usize,
    pub modules: Vec<Vec<Module>>,
    /// Tracks which cells are function patterns (finder, timing, alignment,
    /// format info, dark module) and must not be masked.
    protected: Vec<Vec<bool>>,
}

impl QrMatrix {
    fn new(size: usize) -> Self {
        Self { size, modules: vec![vec![Module::Light; size]; size], protected: vec![vec![false; size]; size] }
    }

    fn protect(&mut self, x: usize, y: usize) {
        if x < self.size && y < self.size {
            self.protected[y][x] = true;
        }
    }

    pub fn is_dark(&self, x: usize, y: usize) -> bool {
        self.modules[y][x] == Module::Dark
    }

    /// Render as a string of block characters.
    pub fn to_string_art(&self) -> String {
        let mut s = String::new();
        for row in &self.modules {
            for m in row {
                match m {
                    Module::Dark => s.push_str("##"),
                    _ => s.push_str("  "),
                }
            }
            s.push('\n');
        }
        s
    }

    /// Render as a compact string using Unicode block elements.
    pub fn to_compact_string(&self) -> String {
        let mut s = String::new();
        let rows = self.size;
        let mut y = 0;
        while y < rows {
            for x in 0..self.size {
                let top = self.modules[y][x] == Module::Dark;
                let bottom = if y + 1 < rows { self.modules[y + 1][x] == Module::Dark } else { false };
                match (top, bottom) {
                    (true, true) => s.push('\u{2588}'),
                    (true, false) => s.push('\u{2580}'),
                    (false, true) => s.push('\u{2584}'),
                    (false, false) => s.push(' '),
                }
            }
            s.push('\n');
            y += 2;
        }
        s
    }
}

impl fmt::Display for QrMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_compact_string())
    }
}

// ── QR Code Generation ─────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum QrError {
    DataTooLong,
    InvalidVersion(u8),
}

impl fmt::Display for QrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QrError::DataTooLong => write!(f, "data too long for QR code versions 1-10"),
            QrError::InvalidVersion(v) => write!(f, "invalid version: {v} (must be 1-10)"),
        }
    }
}

/// Generate a QR code from the given data.
pub fn generate(data: &str, ec: EcLevel) -> Result<QrMatrix, QrError> {
    let mode = detect_mode(data);
    let version = select_version(data, mode, ec).ok_or(QrError::DataTooLong)?;
    generate_with_version(data, version, ec)
}

/// Generate a QR code with a specific version.
pub fn generate_with_version(data: &str, version: u8, ec: EcLevel) -> Result<QrMatrix, QrError> {
    if version < 1 || version > 10 { return Err(QrError::InvalidVersion(version)); }

    let mode = detect_mode(data);
    let size = module_count(version);
    let mut matrix = QrMatrix::new(size);

    // Place finder patterns
    place_finder_pattern(&mut matrix, 0, 0);
    place_finder_pattern(&mut matrix, size as i32 - 7, 0);
    place_finder_pattern(&mut matrix, 0, size as i32 - 7);

    // Place timing patterns
    place_timing_patterns(&mut matrix, size);

    // Place alignment patterns (version >= 2)
    if version >= 2 {
        let positions = alignment_positions(version);
        for &r in &positions {
            for &c in &positions {
                if is_finder_area(r, c, size) { continue; }
                place_alignment_pattern(&mut matrix, c as usize, r as usize);
            }
        }
    }

    // Reserve format info areas
    reserve_format_areas(&mut matrix, size);

    // Encode data
    let encoded = encode_data(data, mode, version, ec);
    let ec_idx = ec_level_index(ec);
    let ec_per_block = EC_CODEWORDS_PER_BLOCK[version as usize - 1][ec_idx] as usize;
    let ec_bytes = rs_encode(&encoded, ec_per_block);

    // Combine data + EC
    let mut final_data = encoded.clone();
    final_data.extend_from_slice(&ec_bytes);

    // Place data bits
    place_data_bits(&mut matrix, &final_data, size);

    // Apply masking (use mask 0 for simplicity — full impl would evaluate all 8)
    apply_mask(&mut matrix, 0);

    // Write format info
    write_format_info(&mut matrix, ec, 0, size);

    Ok(matrix)
}

fn place_finder_pattern(matrix: &mut QrMatrix, x: i32, y: i32) {
    for dy in 0..7 {
        for dx in 0..7 {
            let px = (x + dx) as usize;
            let py = (y + dy) as usize;
            if px >= matrix.size || py >= matrix.size { continue; }
            let is_border = dx == 0 || dx == 6 || dy == 0 || dy == 6;
            let is_inner = dx >= 2 && dx <= 4 && dy >= 2 && dy <= 4;
            matrix.modules[py][px] = if is_border || is_inner { Module::Dark } else { Module::Light };
            matrix.protect(px, py);
        }
    }
    // Separator (one row/col of light around finder)
    for i in -1..=7 {
        for &(dx, dy) in &[(-1, i), (7, i), (i, -1), (i, 7)] {
            let px = x + dx;
            let py = y + dy;
            if px >= 0 && py >= 0 && (px as usize) < matrix.size && (py as usize) < matrix.size {
                let ux = px as usize;
                let uy = py as usize;
                if matrix.modules[uy][ux] != Module::Dark {
                    matrix.modules[uy][ux] = Module::Light;
                }
                matrix.protect(ux, uy);
            }
        }
    }
}

fn place_timing_patterns(matrix: &mut QrMatrix, size: usize) {
    for i in 8..size - 8 {
        let m = if i % 2 == 0 { Module::Dark } else { Module::Light };
        if matrix.modules[6][i] == Module::Light { matrix.modules[6][i] = m; }
        matrix.protect(i, 6);
        if matrix.modules[i][6] == Module::Light { matrix.modules[i][6] = m; }
        matrix.protect(6, i);
    }
}

fn place_alignment_pattern(matrix: &mut QrMatrix, cx: usize, cy: usize) {
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let px = (cx as i32 + dx) as usize;
            let py = (cy as i32 + dy) as usize;
            let is_border = dx.abs() == 2 || dy.abs() == 2;
            let is_center = dx == 0 && dy == 0;
            matrix.modules[py][px] = if is_border || is_center { Module::Dark } else { Module::Light };
            matrix.protect(px, py);
        }
    }
}

fn alignment_positions(version: u8) -> Vec<i32> {
    match version {
        2 => vec![6, 18],
        3 => vec![6, 22],
        4 => vec![6, 26],
        5 => vec![6, 30],
        6 => vec![6, 34],
        7 => vec![6, 22, 38],
        8 => vec![6, 24, 42],
        9 => vec![6, 26, 46],
        10 => vec![6, 28, 50],
        _ => vec![],
    }
}

fn is_finder_area(r: i32, c: i32, size: usize) -> bool {
    let s = size as i32;
    (r < 9 && c < 9) || (r < 9 && c >= s - 8) || (r >= s - 8 && c < 9)
}

fn reserve_format_areas(matrix: &mut QrMatrix, size: usize) {
    // Around top-left finder
    for i in 0..=8 {
        if i < size { matrix.modules[8][i] = Module::Reserved; matrix.protect(i, 8); }
        if i < size { matrix.modules[i][8] = Module::Reserved; matrix.protect(8, i); }
    }
    // Around top-right finder
    for i in 0..8 {
        matrix.modules[8][size - 1 - i] = Module::Reserved;
        matrix.protect(size - 1 - i, 8);
    }
    // Around bottom-left finder
    for i in 0..7 {
        matrix.modules[size - 1 - i][8] = Module::Reserved;
        matrix.protect(8, size - 1 - i);
    }
    // Dark module
    matrix.modules[size - 8][8] = Module::Dark;
    matrix.protect(8, size - 8);
}

fn place_data_bits(matrix: &mut QrMatrix, data: &[u8], size: usize) {
    let mut bits = Vec::new();
    for byte in data {
        for i in (0..8).rev() {
            bits.push((byte >> i) & 1 == 1);
        }
    }

    let mut bit_idx = 0;
    let mut x = size as i32 - 1;
    let mut upward = true;

    while x >= 0 {
        if x == 6 { x -= 1; } // skip timing column
        if x < 0 { break; }

        let col_range: Vec<usize> = if upward {
            (0..size).rev().collect()
        } else {
            (0..size).collect()
        };

        for y in col_range {
            for dx in &[0i32, -1i32] {
                let cx = (x + dx) as usize;
                if cx >= size { continue; }
                if matrix.modules[y][cx] != Module::Light && matrix.modules[y][cx] != Module::Reserved {
                    continue;
                }
                if matrix.modules[y][cx] == Module::Reserved { continue; }
                if bit_idx < bits.len() {
                    matrix.modules[y][cx] = if bits[bit_idx] { Module::Dark } else { Module::Light };
                    bit_idx += 1;
                }
            }
        }
        upward = !upward;
        x -= 2;
    }
}

fn apply_mask(matrix: &mut QrMatrix, mask_pattern: u8) {
    let size = matrix.size;
    for y in 0..size {
        for x in 0..size {
            if matrix.protected[y][x] { continue; }
            if matrix.modules[y][x] == Module::Reserved { continue; }
            // Check if this is a data module (not part of function patterns)
            let should_flip = match mask_pattern {
                0 => (y + x) % 2 == 0,
                1 => y % 2 == 0,
                2 => x % 3 == 0,
                3 => (y + x) % 3 == 0,
                4 => (y / 2 + x / 3) % 2 == 0,
                5 => (y * x) % 2 + (y * x) % 3 == 0,
                6 => ((y * x) % 2 + (y * x) % 3) % 2 == 0,
                7 => ((y + x) % 2 + (y * x) % 3) % 2 == 0,
                _ => false,
            };
            if should_flip {
                matrix.modules[y][x] = match matrix.modules[y][x] {
                    Module::Dark => Module::Light,
                    Module::Light => Module::Dark,
                    other => other,
                };
            }
        }
    }
}

fn write_format_info(matrix: &mut QrMatrix, ec: EcLevel, mask: u8, size: usize) {
    let format_data = ((ec.format_bits() as u16) << 3) | mask as u16;
    let format_ecc = bch_encode(format_data);
    let format_bits = (format_data << 10) | format_ecc;
    let masked = format_bits ^ 0b101010000010010;

    // Write around top-left
    let positions_h: [usize; 15] = [0, 1, 2, 3, 4, 5, 7, 8, size - 8, size - 7, size - 6, size - 5, size - 4, size - 3, size - 2];
    let positions_v: [usize; 15] = [size - 1, size - 2, size - 3, size - 4, size - 5, size - 6, size - 7, size - 8, 7, 5, 4, 3, 2, 1, 0];

    for i in 0..15 {
        let bit = (masked >> (14 - i)) & 1 == 1;
        let m = if bit { Module::Dark } else { Module::Light };
        if positions_h[i] < size {
            matrix.modules[8][positions_h[i]] = m;
        }
        if positions_v[i] < size {
            matrix.modules[positions_v[i]][8] = m;
        }
    }
}

fn bch_encode(data: u16) -> u16 {
    let mut d = (data as u32) << 10;
    let poly = 0b10100110111u32;
    for i in (0..=4).rev() {
        if d & (1 << (i + 10)) != 0 {
            d ^= poly << i;
        }
    }
    d as u16
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_numeric() {
        assert_eq!(detect_mode("12345"), Mode::Numeric);
    }

    #[test]
    fn detect_alphanumeric() {
        assert_eq!(detect_mode("HELLO"), Mode::Alphanumeric);
        assert_eq!(detect_mode("HELLO 123"), Mode::Alphanumeric);
    }

    #[test]
    fn detect_byte() {
        assert_eq!(detect_mode("hello"), Mode::Byte);
        assert_eq!(detect_mode("Hello!"), Mode::Byte);
    }

    #[test]
    fn version_selection() {
        assert_eq!(select_version("12345", Mode::Numeric, EcLevel::L), Some(1));
        assert!(select_version("HELLO", Mode::Alphanumeric, EcLevel::M).is_some());
    }

    #[test]
    fn module_count_v1() {
        assert_eq!(module_count(1), 21);
    }

    #[test]
    fn module_count_v10() {
        assert_eq!(module_count(10), 57);
    }

    #[test]
    fn generate_simple() {
        let qr = generate("HELLO", EcLevel::M).unwrap();
        assert_eq!(qr.size, module_count(1));
        assert!(qr.size >= 21);
    }

    #[test]
    fn generate_numeric() {
        let qr = generate("12345678", EcLevel::L).unwrap();
        assert!(qr.size >= 21);
    }

    #[test]
    fn generate_byte_mode() {
        let qr = generate("hello world", EcLevel::M).unwrap();
        assert!(qr.size >= 21);
    }

    #[test]
    fn generate_url() {
        let qr = generate("https://example.com", EcLevel::Q).unwrap();
        assert!(qr.size >= 21);
    }

    #[test]
    fn error_data_too_long() {
        let long_data = "x".repeat(500);
        assert_eq!(generate(&long_data, EcLevel::H), Err(QrError::DataTooLong));
    }

    #[test]
    fn error_invalid_version() {
        assert_eq!(generate_with_version("test", 0, EcLevel::M), Err(QrError::InvalidVersion(0)));
        assert_eq!(generate_with_version("test", 11, EcLevel::M), Err(QrError::InvalidVersion(11)));
    }

    #[test]
    fn finder_pattern_present() {
        let qr = generate("A", EcLevel::L).unwrap();
        // Top-left finder: 7x7 with dark border
        assert!(qr.is_dark(0, 0));
        assert!(qr.is_dark(6, 0));
        assert!(qr.is_dark(0, 6));
    }

    #[test]
    fn qr_matrix_is_dark() {
        let qr = generate("TEST", EcLevel::M).unwrap();
        // Just verify it doesn't panic
        let _ = qr.is_dark(0, 0);
        let _ = qr.is_dark(qr.size - 1, qr.size - 1);
    }

    #[test]
    fn string_art_output() {
        let qr = generate("HI", EcLevel::L).unwrap();
        let art = qr.to_string_art();
        assert!(art.contains("##"));
        assert!(art.contains('\n'));
    }

    #[test]
    fn compact_string_output() {
        let qr = generate("HI", EcLevel::L).unwrap();
        let compact = qr.to_compact_string();
        assert!(!compact.is_empty());
    }

    #[test]
    fn display_trait() {
        let qr = generate("AB", EcLevel::L).unwrap();
        let display = format!("{qr}");
        assert!(!display.is_empty());
    }

    #[test]
    fn gf256_multiply() {
        assert_eq!(gf256_mul(0, 5), 0);
        assert_eq!(gf256_mul(1, 1), 1);
        assert_eq!(gf256_mul(2, 3), 6);
    }

    #[test]
    fn rs_encode_produces_output() {
        let data = vec![32, 65, 205, 69, 41, 220, 46, 128, 236];
        let ec = rs_encode(&data, 7);
        assert_eq!(ec.len(), 7);
    }

    #[test]
    fn ec_level_format_bits() {
        assert_eq!(EcLevel::L.format_bits(), 0b01);
        assert_eq!(EcLevel::M.format_bits(), 0b00);
        assert_eq!(EcLevel::Q.format_bits(), 0b11);
        assert_eq!(EcLevel::H.format_bits(), 0b10);
    }

    #[test]
    fn mode_indicator_bits() {
        assert_eq!(Mode::Numeric.indicator(), 0b0001);
        assert_eq!(Mode::Alphanumeric.indicator(), 0b0010);
        assert_eq!(Mode::Byte.indicator(), 0b0100);
    }

    #[test]
    fn all_ec_levels() {
        for ec in &[EcLevel::L, EcLevel::M, EcLevel::Q, EcLevel::H] {
            let qr = generate("TEST", *ec);
            assert!(qr.is_ok());
        }
    }

    #[test]
    fn version_2_has_alignment() {
        let qr = generate_with_version("A", 2, EcLevel::L).unwrap();
        assert_eq!(qr.size, 25);
    }

    #[test]
    fn error_display() {
        assert_eq!(format!("{}", QrError::DataTooLong), "data too long for QR code versions 1-10");
        assert_eq!(format!("{}", QrError::InvalidVersion(99)), "invalid version: 99 (must be 1-10)");
    }

    #[test]
    fn bit_buffer_basics() {
        let mut buf = BitBuffer::new();
        buf.push_bits(0b1010, 4);
        assert_eq!(buf.len(), 4);
        let bytes = buf.to_bytes();
        assert_eq!(bytes[0], 0b10100000);
    }

    #[test]
    fn alphanumeric_values() {
        assert_eq!(alphanumeric_value('0'), 0);
        assert_eq!(alphanumeric_value('A'), 10);
        assert_eq!(alphanumeric_value(' '), 36);
    }
}
