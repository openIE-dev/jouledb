//! Bit-level I/O — bitwise read/write with MSB/LSB modes.
//!
//! Provides fine-grained bit manipulation for binary protocols, compression
//! codecs, and wire formats. Replaces JavaScript bit-manipulation libraries
//! with a pure Rust implementation.

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during bitstream operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BitstreamError {
    #[error("unexpected end of stream: requested {requested} bits but only {available} remain")]
    UnexpectedEof { requested: usize, available: usize },
    #[error("cannot read {0} bits in a single operation (max 64)")]
    TooManyBits(usize),
    #[error("write of {0} bits exceeds value capacity")]
    ValueOverflow(usize),
}

// ── Bit Order ───────────────────────────────────────────────────────

/// Bit ordering within a byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOrder {
    /// Most significant bit first (network byte order, JPEG, PNG).
    MsbFirst,
    /// Least significant bit first (DEFLATE, USB).
    LsbFirst,
}

// ── Bit Writer ──────────────────────────────────────────────────────

/// A writer that accumulates bits into a byte buffer.
#[derive(Debug, Clone)]
pub struct BitWriter {
    buf: Vec<u8>,
    current: u64,
    bit_count: u8,
    order: BitOrder,
    total_bits: u64,
}

impl BitWriter {
    /// Create a new bit writer with the given bit order.
    pub fn new(order: BitOrder) -> Self {
        Self {
            buf: Vec::new(),
            current: 0,
            bit_count: 0,
            order,
            total_bits: 0,
        }
    }

    /// Create a new MSB-first bit writer.
    pub fn new_msb() -> Self {
        Self::new(BitOrder::MsbFirst)
    }

    /// Create a new LSB-first bit writer.
    pub fn new_lsb() -> Self {
        Self::new(BitOrder::LsbFirst)
    }

    /// Write `count` bits from `value`. Only the lowest `count` bits are used.
    pub fn write_bits(&mut self, value: u64, count: usize) -> Result<(), BitstreamError> {
        if count > 64 {
            return Err(BitstreamError::TooManyBits(count));
        }
        if count == 0 {
            return Ok(());
        }
        // Mask off any extra high bits.
        let mask = if count == 64 { u64::MAX } else { (1u64 << count) - 1 };
        let val = value & mask;

        match self.order {
            BitOrder::MsbFirst => self.write_bits_msb(val, count),
            BitOrder::LsbFirst => self.write_bits_lsb(val, count),
        }
        self.total_bits += count as u64;
        Ok(())
    }

    fn write_bits_msb(&mut self, value: u64, count: usize) {
        for i in (0..count).rev() {
            let bit = (value >> i) & 1;
            self.current = (self.current << 1) | bit;
            self.bit_count += 1;
            if self.bit_count == 8 {
                self.buf.push(self.current as u8);
                self.current = 0;
                self.bit_count = 0;
            }
        }
    }

    fn write_bits_lsb(&mut self, value: u64, count: usize) {
        for i in 0..count {
            let bit = (value >> i) & 1;
            self.current |= bit << self.bit_count;
            self.bit_count += 1;
            if self.bit_count == 8 {
                self.buf.push(self.current as u8);
                self.current = 0;
                self.bit_count = 0;
            }
        }
    }

    /// Write a single bit (0 or 1).
    pub fn write_bit(&mut self, bit: bool) -> Result<(), BitstreamError> {
        self.write_bits(if bit { 1 } else { 0 }, 1)
    }

    /// Write a full byte (8 bits).
    pub fn write_byte(&mut self, byte: u8) -> Result<(), BitstreamError> {
        self.write_bits(byte as u64, 8)
    }

    /// Pad with zero bits to the next byte boundary.
    pub fn align(&mut self) {
        if self.bit_count > 0 {
            let padding = 8 - self.bit_count;
            match self.order {
                BitOrder::MsbFirst => {
                    self.current <<= padding;
                    self.buf.push(self.current as u8);
                }
                BitOrder::LsbFirst => {
                    // Low bits are already in place, high bits are zero.
                    self.buf.push(self.current as u8);
                }
            }
            self.total_bits += padding as u64;
            self.current = 0;
            self.bit_count = 0;
        }
    }

    /// Flush any remaining bits (with zero-padding) and return the byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.align();
        self.buf
    }

    /// Total bits written so far (including any pending bits).
    pub fn bits_written(&self) -> u64 {
        self.total_bits
    }

    /// Number of complete bytes written so far.
    pub fn bytes_written(&self) -> usize {
        self.buf.len()
    }

    /// Whether the writer is currently byte-aligned.
    pub fn is_aligned(&self) -> bool {
        self.bit_count == 0
    }

    /// Get a reference to the buffer written so far (not including pending bits).
    pub fn buffer(&self) -> &[u8] {
        &self.buf
    }

    /// Get the bit order.
    pub fn order(&self) -> BitOrder {
        self.order
    }
}

// ── Bit Reader ──────────────────────────────────────────────────────

/// A reader that extracts bits from a byte buffer.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
    order: BitOrder,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader with the given bit order.
    pub fn new(data: &'a [u8], order: BitOrder) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
            order,
        }
    }

    /// Create a new MSB-first bit reader.
    pub fn new_msb(data: &'a [u8]) -> Self {
        Self::new(data, BitOrder::MsbFirst)
    }

    /// Create a new LSB-first bit reader.
    pub fn new_lsb(data: &'a [u8]) -> Self {
        Self::new(data, BitOrder::LsbFirst)
    }

    /// Read `count` bits and return them as a u64.
    pub fn read_bits(&mut self, count: usize) -> Result<u64, BitstreamError> {
        if count > 64 {
            return Err(BitstreamError::TooManyBits(count));
        }
        if count == 0 {
            return Ok(0);
        }
        let avail = self.bits_remaining();
        if avail < count {
            return Err(BitstreamError::UnexpectedEof {
                requested: count,
                available: avail,
            });
        }

        match self.order {
            BitOrder::MsbFirst => self.read_bits_msb(count),
            BitOrder::LsbFirst => self.read_bits_lsb(count),
        }
    }

    fn read_bits_msb(&mut self, count: usize) -> Result<u64, BitstreamError> {
        let mut result = 0u64;
        for _ in 0..count {
            let byte = self.data[self.byte_pos];
            let bit = (byte >> (7 - self.bit_pos)) & 1;
            result = (result << 1) | bit as u64;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.byte_pos += 1;
                self.bit_pos = 0;
            }
        }
        Ok(result)
    }

    fn read_bits_lsb(&mut self, count: usize) -> Result<u64, BitstreamError> {
        let mut result = 0u64;
        for i in 0..count {
            let byte = self.data[self.byte_pos];
            let bit = (byte >> self.bit_pos) & 1;
            result |= (bit as u64) << i;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.byte_pos += 1;
                self.bit_pos = 0;
            }
        }
        Ok(result)
    }

    /// Read a single bit.
    pub fn read_bit(&mut self) -> Result<bool, BitstreamError> {
        Ok(self.read_bits(1)? == 1)
    }

    /// Read a full byte (8 bits).
    pub fn read_byte(&mut self) -> Result<u8, BitstreamError> {
        Ok(self.read_bits(8)? as u8)
    }

    /// Peek at the next `count` bits without advancing the position.
    pub fn peek_bits(&self, count: usize) -> Result<u64, BitstreamError> {
        let mut clone = self.clone();
        clone.read_bits(count)
    }

    /// Skip `count` bits.
    pub fn skip_bits(&mut self, count: usize) -> Result<(), BitstreamError> {
        let avail = self.bits_remaining();
        if avail < count {
            return Err(BitstreamError::UnexpectedEof {
                requested: count,
                available: avail,
            });
        }
        let total_bit_pos = self.byte_pos * 8 + self.bit_pos as usize + count;
        self.byte_pos = total_bit_pos / 8;
        self.bit_pos = (total_bit_pos % 8) as u8;
        Ok(())
    }

    /// Align to the next byte boundary by skipping remaining bits in the current byte.
    pub fn align(&mut self) {
        if self.bit_pos > 0 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
    }

    /// Read remaining bytes after aligning to a byte boundary.
    pub fn read_aligned_bytes(&mut self, count: usize) -> Result<Vec<u8>, BitstreamError> {
        self.align();
        let remaining_bytes = self.data.len() - self.byte_pos;
        if remaining_bytes < count {
            return Err(BitstreamError::UnexpectedEof {
                requested: count * 8,
                available: remaining_bytes * 8,
            });
        }
        let result = self.data[self.byte_pos..self.byte_pos + count].to_vec();
        self.byte_pos += count;
        Ok(result)
    }

    /// Number of bits remaining.
    pub fn bits_remaining(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            return 0;
        }
        (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
    }

    /// Total position in bits from the start.
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Whether the reader is currently byte-aligned.
    pub fn is_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Whether all bits have been consumed.
    pub fn is_empty(&self) -> bool {
        self.bits_remaining() == 0
    }

    /// Get the bit order.
    pub fn order(&self) -> BitOrder {
        self.order
    }
}

// ── Convenience Functions ───────────────────────────────────────────

/// Write a sequence of bit widths and values, return the byte buffer.
/// Each entry is (value, bit_count).
pub fn pack_bits(entries: &[(u64, usize)], order: BitOrder) -> Result<Vec<u8>, BitstreamError> {
    let mut writer = BitWriter::new(order);
    for &(value, count) in entries {
        writer.write_bits(value, count)?;
    }
    Ok(writer.finish())
}

/// Unpack a sequence of bit widths from a byte buffer.
/// Returns the extracted values.
pub fn unpack_bits(
    data: &[u8],
    widths: &[usize],
    order: BitOrder,
) -> Result<Vec<u64>, BitstreamError> {
    let mut reader = BitReader::new(data, order);
    let mut values = Vec::with_capacity(widths.len());
    for &width in widths {
        values.push(reader.read_bits(width)?);
    }
    Ok(values)
}

/// Count the number of set bits (popcount) in a byte slice.
pub fn popcount(data: &[u8]) -> u64 {
    data.iter().map(|b| b.count_ones() as u64).sum()
}

/// Get the bit at a specific position in a byte slice (MSB-first indexing).
pub fn get_bit(data: &[u8], bit_index: usize) -> Option<bool> {
    let byte_index = bit_index / 8;
    let bit_offset = bit_index % 8;
    if byte_index >= data.len() {
        return None;
    }
    Some((data[byte_index] >> (7 - bit_offset)) & 1 == 1)
}

/// Set a bit at a specific position in a byte slice (MSB-first indexing).
pub fn set_bit(data: &mut [u8], bit_index: usize, value: bool) -> bool {
    let byte_index = bit_index / 8;
    let bit_offset = bit_index % 8;
    if byte_index >= data.len() {
        return false;
    }
    if value {
        data[byte_index] |= 1 << (7 - bit_offset);
    } else {
        data[byte_index] &= !(1 << (7 - bit_offset));
    }
    true
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MSB Writer/Reader ───────────────────────────────────────────

    #[test]
    fn msb_write_single_bits() {
        let mut w = BitWriter::new_msb();
        w.write_bit(true).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(true).unwrap();
        w.write_bit(true).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(true).unwrap();
        w.write_bit(false).unwrap();
        let buf = w.finish();
        // 1011_0010 = 0xB2
        assert_eq!(buf, [0xB2]);
    }

    #[test]
    fn msb_write_multi_bits() {
        let mut w = BitWriter::new_msb();
        w.write_bits(0b1011, 4).unwrap();
        w.write_bits(0b0010, 4).unwrap();
        let buf = w.finish();
        assert_eq!(buf, [0xB2]);
    }

    #[test]
    fn msb_roundtrip() {
        let mut w = BitWriter::new_msb();
        w.write_bits(5, 3).unwrap();  // 101
        w.write_bits(3, 2).unwrap();  // 11
        w.write_bits(0, 1).unwrap();  // 0
        w.write_bits(1, 1).unwrap();  // 1
        w.write_bits(0, 1).unwrap();  // 0
        let buf = w.finish();
        // 10111010 = 0xBA
        assert_eq!(buf, [0xBA]);

        let mut r = BitReader::new_msb(&buf);
        assert_eq!(r.read_bits(3).unwrap(), 5);
        assert_eq!(r.read_bits(2).unwrap(), 3);
        assert_eq!(r.read_bits(1).unwrap(), 0);
        assert_eq!(r.read_bits(1).unwrap(), 1);
        assert_eq!(r.read_bits(1).unwrap(), 0);
    }

    // ── LSB Writer/Reader ───────────────────────────────────────────

    #[test]
    fn lsb_write_single_bits() {
        let mut w = BitWriter::new_lsb();
        // Bits written: 1, 0, 1, 1, 0, 0, 1, 0
        // LSB first: bit0=1, bit1=0, bit2=1, bit3=1, bit4=0, bit5=0, bit6=1, bit7=0
        // Byte = 0b_0100_1101 = 0x4D
        w.write_bit(true).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(true).unwrap();
        w.write_bit(true).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(false).unwrap();
        w.write_bit(true).unwrap();
        w.write_bit(false).unwrap();
        let buf = w.finish();
        assert_eq!(buf, [0x4D]);
    }

    #[test]
    fn lsb_roundtrip() {
        let mut w = BitWriter::new_lsb();
        w.write_bits(5, 3).unwrap();  // 101 in LSB first
        w.write_bits(2, 3).unwrap();  // 010 in LSB first
        w.write_bits(1, 2).unwrap();  // 01 in LSB first
        let buf = w.finish();

        let mut r = BitReader::new_lsb(&buf);
        assert_eq!(r.read_bits(3).unwrap(), 5);
        assert_eq!(r.read_bits(3).unwrap(), 2);
        assert_eq!(r.read_bits(2).unwrap(), 1);
    }

    // ── Alignment ───────────────────────────────────────────────────

    #[test]
    fn writer_alignment() {
        let mut w = BitWriter::new_msb();
        w.write_bits(0b101, 3).unwrap();
        assert!(!w.is_aligned());
        w.align();
        assert!(w.is_aligned());
        let buf = w.finish();
        // 101_00000 = 0xA0
        assert_eq!(buf, [0xA0]);
    }

    #[test]
    fn reader_alignment() {
        let data = [0xFF, 0xAA];
        let mut r = BitReader::new_msb(&data);
        r.read_bits(3).unwrap();
        assert!(!r.is_aligned());
        r.align();
        assert!(r.is_aligned());
        assert_eq!(r.bit_position(), 8);
        assert_eq!(r.read_byte().unwrap(), 0xAA);
    }

    // ── Peek ────────────────────────────────────────────────────────

    #[test]
    fn peek_does_not_advance() {
        let data = [0xFF];
        let r = BitReader::new_msb(&data);
        assert_eq!(r.peek_bits(4).unwrap(), 0xF);
        assert_eq!(r.bit_position(), 0);
        assert_eq!(r.peek_bits(8).unwrap(), 0xFF);
    }

    // ── Position Tracking ───────────────────────────────────────────

    #[test]
    fn bit_position_tracking() {
        let data = [0xFF, 0xFF];
        let mut r = BitReader::new_msb(&data);
        assert_eq!(r.bit_position(), 0);
        assert_eq!(r.bits_remaining(), 16);
        r.read_bits(5).unwrap();
        assert_eq!(r.bit_position(), 5);
        assert_eq!(r.bits_remaining(), 11);
    }

    #[test]
    fn writer_bits_written() {
        let mut w = BitWriter::new_msb();
        w.write_bits(0, 5).unwrap();
        assert_eq!(w.bits_written(), 5);
        w.write_bits(0, 3).unwrap();
        assert_eq!(w.bits_written(), 8);
    }

    // ── Skip Bits ───────────────────────────────────────────────────

    #[test]
    fn skip_bits_works() {
        let data = [0b1010_1100, 0b1111_0000];
        let mut r = BitReader::new_msb(&data);
        r.skip_bits(4).unwrap();
        assert_eq!(r.bit_position(), 4);
        assert_eq!(r.read_bits(4).unwrap(), 0b1100);
    }

    // ── Read Aligned Bytes ──────────────────────────────────────────

    #[test]
    fn read_aligned_bytes_after_bits() {
        let data = [0xFF, 0xAA, 0xBB];
        let mut r = BitReader::new_msb(&data);
        r.read_bits(5).unwrap();
        let bytes = r.read_aligned_bytes(2).unwrap();
        assert_eq!(bytes, [0xAA, 0xBB]);
    }

    // ── Pack/Unpack Convenience ─────────────────────────────────────

    #[test]
    fn pack_unpack_roundtrip() {
        let entries = &[(7u64, 3), (15, 4), (1, 1), (255, 8)];
        let packed = pack_bits(entries, BitOrder::MsbFirst).unwrap();
        let widths = [3, 4, 1, 8];
        let values = unpack_bits(&packed, &widths, BitOrder::MsbFirst).unwrap();
        assert_eq!(values, [7, 15, 1, 255]);
    }

    // ── Popcount & Bit Access ───────────────────────────────────────

    #[test]
    fn popcount_works() {
        assert_eq!(popcount(&[0xFF]), 8);
        assert_eq!(popcount(&[0x00]), 0);
        assert_eq!(popcount(&[0xAA]), 4); // 10101010
        assert_eq!(popcount(&[0xFF, 0x00, 0x0F]), 12);
    }

    #[test]
    fn get_set_bit() {
        let mut data = [0u8; 2];
        assert_eq!(get_bit(&data, 0), Some(false));
        set_bit(&mut data, 0, true);
        assert_eq!(get_bit(&data, 0), Some(true));
        // Bit 0 of byte 0 is MSB: 0x80
        assert_eq!(data[0], 0x80);

        set_bit(&mut data, 7, true);
        assert_eq!(data[0], 0x81);

        set_bit(&mut data, 8, true);
        assert_eq!(data[1], 0x80);

        assert_eq!(get_bit(&data, 100), None);
        assert!(!set_bit(&mut data, 100, true));
    }

    // ── Error Cases ─────────────────────────────────────────────────

    #[test]
    fn read_past_end() {
        let data = [0xFF];
        let mut r = BitReader::new_msb(&data);
        r.read_bits(8).unwrap();
        assert!(matches!(
            r.read_bits(1),
            Err(BitstreamError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn write_too_many_bits() {
        let mut w = BitWriter::new_msb();
        assert!(matches!(
            w.write_bits(0, 65),
            Err(BitstreamError::TooManyBits(65))
        ));
    }

    #[test]
    fn read_zero_bits() {
        let data = [0xFF];
        let mut r = BitReader::new_msb(&data);
        assert_eq!(r.read_bits(0).unwrap(), 0);
        assert_eq!(r.bit_position(), 0);
    }

    #[test]
    fn write_zero_bits() {
        let mut w = BitWriter::new_msb();
        w.write_bits(0, 0).unwrap();
        assert_eq!(w.bits_written(), 0);
    }

    #[test]
    fn empty_reader() {
        let data = [];
        let r = BitReader::new_msb(&data);
        assert!(r.is_empty());
        assert_eq!(r.bits_remaining(), 0);
    }

    #[test]
    fn cross_byte_read() {
        let data = [0b1111_0000, 0b1010_1010];
        let mut r = BitReader::new_msb(&data);
        r.read_bits(4).unwrap(); // skip 1111
        let val = r.read_bits(8).unwrap(); // read 0000_1010
        assert_eq!(val, 0b0000_1010);
    }

    #[test]
    fn large_value_write_read() {
        let mut w = BitWriter::new_msb();
        let val = 0xDEAD_BEEF_CAFE_BABEu64;
        w.write_bits(val, 64).unwrap();
        let buf = w.finish();
        let mut r = BitReader::new_msb(&buf);
        assert_eq!(r.read_bits(64).unwrap(), val);
    }

    #[test]
    fn skip_past_end_fails() {
        let data = [0xFF];
        let mut r = BitReader::new_msb(&data);
        assert!(matches!(
            r.skip_bits(9),
            Err(BitstreamError::UnexpectedEof { .. })
        ));
    }
}
