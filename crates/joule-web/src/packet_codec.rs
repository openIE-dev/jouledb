//! Binary packet encoder/decoder — length-prefixed framing, varint, checksums.
//!
//! Pure Rust binary codec with configurable endianness, variable-length
//! integer encoding (LEB128-style varint), length-prefixed framing,
//! packet versioning, and CRC32 checksums.

use std::fmt;

// ── Endianness ────────────────────────────────────────────────

/// Byte order for multi-byte integer encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Big,
    Little,
}

// ── Codec Error ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    BufferTooShort { need: usize, have: usize },
    VarintOverflow,
    InvalidVersion { expected: u8, got: u8 },
    ChecksumMismatch { expected: u32, got: u32 },
    InvalidUtf8,
    FrameTooLarge { size: usize, max: usize },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooShort { need, have } =>
                write!(f, "buffer too short: need {} bytes, have {}", need, have),
            Self::VarintOverflow =>
                write!(f, "varint overflow (more than 10 bytes)"),
            Self::InvalidVersion { expected, got } =>
                write!(f, "invalid version: expected {}, got {}", expected, got),
            Self::ChecksumMismatch { expected, got } =>
                write!(f, "checksum mismatch: expected 0x{:08x}, got 0x{:08x}", expected, got),
            Self::InvalidUtf8 =>
                write!(f, "invalid UTF-8 in string field"),
            Self::FrameTooLarge { size, max } =>
                write!(f, "frame too large: {} bytes, max {}", size, max),
        }
    }
}

// ── CRC32 ─────────────────────────────────────────────────────

/// CRC32 (ISO 3309 / ITU-T V.42) checksum.
pub struct Crc32 {
    table: [u32; 256],
}

impl Crc32 {
    /// Build the CRC32 lookup table.
    pub fn new() -> Self {
        let mut table = [0u32; 256];
        for i in 0..256u32 {
            let mut crc = i;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
            table[i as usize] = crc;
        }
        Self { table }
    }

    /// Compute CRC32 of a byte slice.
    pub fn checksum(&self, data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for byte in data {
            let index = ((crc ^ (*byte as u32)) & 0xFF) as usize;
            crc = (crc >> 8) ^ self.table[index];
        }
        crc ^ 0xFFFF_FFFF
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

// ── PacketWriter ──────────────────────────────────────────────

/// Builds binary packets by writing fields sequentially.
pub struct PacketWriter {
    buf: Vec<u8>,
    endian: Endian,
}

impl PacketWriter {
    pub fn new(endian: Endian) -> Self {
        Self { buf: Vec::new(), endian }
    }

    pub fn with_capacity(endian: Endian, cap: usize) -> Self {
        Self { buf: Vec::with_capacity(cap), endian }
    }

    /// Current length of the written data.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Write a single byte.
    pub fn write_u8(&mut self, val: u8) {
        self.buf.push(val);
    }

    /// Write a u16.
    pub fn write_u16(&mut self, val: u16) {
        match self.endian {
            Endian::Big => self.buf.extend_from_slice(&val.to_be_bytes()),
            Endian::Little => self.buf.extend_from_slice(&val.to_le_bytes()),
        }
    }

    /// Write a u32.
    pub fn write_u32(&mut self, val: u32) {
        match self.endian {
            Endian::Big => self.buf.extend_from_slice(&val.to_be_bytes()),
            Endian::Little => self.buf.extend_from_slice(&val.to_le_bytes()),
        }
    }

    /// Write a u64.
    pub fn write_u64(&mut self, val: u64) {
        match self.endian {
            Endian::Big => self.buf.extend_from_slice(&val.to_be_bytes()),
            Endian::Little => self.buf.extend_from_slice(&val.to_le_bytes()),
        }
    }

    /// Write a varint (LEB128-style unsigned variable-length integer).
    pub fn write_varint(&mut self, mut val: u64) {
        loop {
            let mut byte = (val & 0x7F) as u8;
            val >>= 7;
            if val != 0 {
                byte |= 0x80;
            }
            self.buf.push(byte);
            if val == 0 {
                break;
            }
        }
    }

    /// Write a length-prefixed byte slice (length as varint).
    pub fn write_bytes(&mut self, data: &[u8]) {
        self.write_varint(data.len() as u64);
        self.buf.extend_from_slice(data);
    }

    /// Write a length-prefixed UTF-8 string (length as varint).
    pub fn write_string(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    /// Write raw bytes without a length prefix.
    pub fn write_raw(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Append CRC32 checksum of all data written so far.
    pub fn append_checksum(&mut self) {
        let crc = Crc32::new();
        let checksum = crc.checksum(&self.buf);
        // Checksum is always big-endian for consistency.
        self.buf.extend_from_slice(&checksum.to_be_bytes());
    }

    /// Consume and return the built packet.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Get a reference to the written data.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }
}

// ── PacketReader ──────────────────────────────────────────────

/// Reads fields from a binary packet sequentially.
pub struct PacketReader<'a> {
    buf: &'a [u8],
    pos: usize,
    endian: Endian,
}

impl<'a> PacketReader<'a> {
    pub fn new(buf: &'a [u8], endian: Endian) -> Self {
        Self { buf, pos: 0, endian }
    }

    /// Remaining bytes available to read.
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    /// Current read position.
    pub fn position(&self) -> usize {
        self.pos
    }

    fn need(&self, n: usize) -> Result<(), CodecError> {
        if self.remaining() < n {
            Err(CodecError::BufferTooShort { need: n, have: self.remaining() })
        } else {
            Ok(())
        }
    }

    pub fn read_u8(&mut self) -> Result<u8, CodecError> {
        self.need(1)?;
        let val = self.buf[self.pos];
        self.pos += 1;
        Ok(val)
    }

    pub fn read_u16(&mut self) -> Result<u16, CodecError> {
        self.need(2)?;
        let bytes: [u8; 2] = [self.buf[self.pos], self.buf[self.pos + 1]];
        self.pos += 2;
        Ok(match self.endian {
            Endian::Big => u16::from_be_bytes(bytes),
            Endian::Little => u16::from_le_bytes(bytes),
        })
    }

    pub fn read_u32(&mut self) -> Result<u32, CodecError> {
        self.need(4)?;
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(match self.endian {
            Endian::Big => u32::from_be_bytes(bytes),
            Endian::Little => u32::from_le_bytes(bytes),
        })
    }

    pub fn read_u64(&mut self) -> Result<u64, CodecError> {
        self.need(8)?;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(match self.endian {
            Endian::Big => u64::from_be_bytes(bytes),
            Endian::Little => u64::from_le_bytes(bytes),
        })
    }

    /// Read a varint (LEB128-style).
    pub fn read_varint(&mut self) -> Result<u64, CodecError> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            self.need(1)?;
            let byte = self.buf[self.pos];
            self.pos += 1;

            if shift >= 64 {
                return Err(CodecError::VarintOverflow);
            }

            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    /// Read a length-prefixed byte slice (length as varint).
    pub fn read_bytes(&mut self) -> Result<Vec<u8>, CodecError> {
        let len = self.read_varint()? as usize;
        self.need(len)?;
        let data = self.buf[self.pos..self.pos + len].to_vec();
        self.pos += len;
        Ok(data)
    }

    /// Read a length-prefixed UTF-8 string.
    pub fn read_string(&mut self) -> Result<String, CodecError> {
        let data = self.read_bytes()?;
        String::from_utf8(data).map_err(|_| CodecError::InvalidUtf8)
    }

    /// Read `n` raw bytes.
    pub fn read_raw(&mut self, n: usize) -> Result<Vec<u8>, CodecError> {
        self.need(n)?;
        let data = self.buf[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(data)
    }
}

// ── Framed Packet ─────────────────────────────────────────────

/// Configuration for length-prefixed framing.
#[derive(Debug, Clone)]
pub struct FrameConfig {
    /// Maximum frame payload size in bytes.
    pub max_frame_size: usize,
    /// Whether to include a CRC32 checksum in each frame.
    pub checksum: bool,
    /// Packet version for versioned framing.
    pub version: u8,
    /// Byte order for frame header fields.
    pub endian: Endian,
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self {
            max_frame_size: 16 * 1024 * 1024, // 16 MB
            checksum: true,
            version: 1,
            endian: Endian::Big,
        }
    }
}

/// Encode a payload into a framed packet.
///
/// Format: [version: u8] [length: u32] [payload] [crc32: u32 (optional)]
pub fn frame_encode(payload: &[u8], config: &FrameConfig) -> Result<Vec<u8>, CodecError> {
    if payload.len() > config.max_frame_size {
        return Err(CodecError::FrameTooLarge {
            size: payload.len(),
            max: config.max_frame_size,
        });
    }

    let mut w = PacketWriter::new(config.endian);
    w.write_u8(config.version);
    w.write_u32(payload.len() as u32);
    w.write_raw(payload);

    if config.checksum {
        let crc = Crc32::new();
        let checksum = crc.checksum(payload);
        // Always big-endian for checksum.
        w.write_raw(&checksum.to_be_bytes());
    }

    Ok(w.finish())
}

/// Decode a framed packet and return the payload.
pub fn frame_decode(data: &[u8], config: &FrameConfig) -> Result<Vec<u8>, CodecError> {
    let mut r = PacketReader::new(data, config.endian);

    let version = r.read_u8()?;
    if version != config.version {
        return Err(CodecError::InvalidVersion { expected: config.version, got: version });
    }

    let length = r.read_u32()? as usize;
    if length > config.max_frame_size {
        return Err(CodecError::FrameTooLarge { size: length, max: config.max_frame_size });
    }

    let payload = r.read_raw(length)?;

    if config.checksum {
        let stored_bytes = r.read_raw(4)?;
        let stored = u32::from_be_bytes([stored_bytes[0], stored_bytes[1], stored_bytes[2], stored_bytes[3]]);
        let crc = Crc32::new();
        let computed = crc.checksum(&payload);
        if stored != computed {
            return Err(CodecError::ChecksumMismatch { expected: computed, got: stored });
        }
    }

    Ok(payload)
}

/// Extract multiple frames from a byte stream. Returns decoded payloads
/// and the number of bytes consumed.
pub fn frame_decode_stream(data: &[u8], config: &FrameConfig) -> (Vec<Vec<u8>>, usize) {
    let mut frames = Vec::new();
    let mut consumed = 0;

    while consumed < data.len() {
        let remaining = &data[consumed..];
        // Need at least 5 bytes for header (1 version + 4 length).
        if remaining.len() < 5 {
            break;
        }

        let mut r = PacketReader::new(remaining, config.endian);
        let version = match r.read_u8() {
            Ok(v) => v,
            Err(_) => break,
        };
        if version != config.version {
            break;
        }

        let length = match r.read_u32() {
            Ok(l) => l as usize,
            Err(_) => break,
        };

        let checksum_size = if config.checksum { 4 } else { 0 };
        let total_frame = 5 + length + checksum_size;
        if remaining.len() < total_frame {
            break;
        }

        match frame_decode(remaining, config) {
            Ok(payload) => {
                frames.push(payload);
                consumed += total_frame;
            }
            Err(_) => break,
        }
    }

    (frames, consumed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_reader_u8() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_u8(42);
        let data = w.finish();
        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_u8().unwrap(), 42);
    }

    #[test]
    fn test_writer_reader_u16_big() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_u16(0x1234);
        let data = w.finish();
        assert_eq!(data, [0x12, 0x34]);
        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_u16().unwrap(), 0x1234);
    }

    #[test]
    fn test_writer_reader_u16_little() {
        let mut w = PacketWriter::new(Endian::Little);
        w.write_u16(0x1234);
        let data = w.finish();
        assert_eq!(data, [0x34, 0x12]);
        let mut r = PacketReader::new(&data, Endian::Little);
        assert_eq!(r.read_u16().unwrap(), 0x1234);
    }

    #[test]
    fn test_writer_reader_u32() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_u32(0xDEAD_BEEF);
        let data = w.finish();
        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_u32().unwrap(), 0xDEAD_BEEF);
    }

    #[test]
    fn test_writer_reader_u64() {
        let mut w = PacketWriter::new(Endian::Little);
        w.write_u64(0x0102030405060708);
        let data = w.finish();
        let mut r = PacketReader::new(&data, Endian::Little);
        assert_eq!(r.read_u64().unwrap(), 0x0102030405060708);
    }

    #[test]
    fn test_varint_small() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_varint(42);
        let data = w.finish();
        assert_eq!(data.len(), 1);
        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_varint().unwrap(), 42);
    }

    #[test]
    fn test_varint_medium() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_varint(300);
        let data = w.finish();
        assert_eq!(data.len(), 2); // 300 > 127
        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_varint().unwrap(), 300);
    }

    #[test]
    fn test_varint_large() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_varint(u64::MAX);
        let data = w.finish();
        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_varint().unwrap(), u64::MAX);
    }

    #[test]
    fn test_varint_zero() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_varint(0);
        let data = w.finish();
        assert_eq!(data, [0]);
        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_varint().unwrap(), 0);
    }

    #[test]
    fn test_write_read_bytes() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_bytes(b"hello");
        let data = w.finish();
        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_bytes().unwrap(), b"hello");
    }

    #[test]
    fn test_write_read_string() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_string("world");
        let data = w.finish();
        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_string().unwrap(), "world");
    }

    #[test]
    fn test_buffer_too_short() {
        let r_result = PacketReader::new(&[1], Endian::Big).read_u16();
        assert!(matches!(r_result, Err(CodecError::BufferTooShort { .. })));
    }

    #[test]
    fn test_crc32_known() {
        let crc = Crc32::new();
        // CRC32 of "123456789" is 0xCBF43926 (well-known test vector).
        assert_eq!(crc.checksum(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_crc32_empty() {
        let crc = Crc32::new();
        assert_eq!(crc.checksum(b""), 0x0000_0000);
    }

    #[test]
    fn test_frame_encode_decode() {
        let config = FrameConfig::default();
        let payload = b"test payload";
        let frame = frame_encode(payload, &config).unwrap();
        let decoded = frame_decode(&frame, &config).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn test_frame_no_checksum() {
        let config = FrameConfig { checksum: false, ..Default::default() };
        let payload = b"no checksum";
        let frame = frame_encode(payload, &config).unwrap();
        let decoded = frame_decode(&frame, &config).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn test_frame_version_mismatch() {
        let config = FrameConfig::default();
        let frame = frame_encode(b"data", &config).unwrap();
        let config2 = FrameConfig { version: 2, ..Default::default() };
        assert!(matches!(
            frame_decode(&frame, &config2),
            Err(CodecError::InvalidVersion { .. })
        ));
    }

    #[test]
    fn test_frame_checksum_corruption() {
        let config = FrameConfig::default();
        let mut frame = frame_encode(b"data", &config).unwrap();
        // Corrupt the last byte (part of checksum).
        let last = frame.len() - 1;
        frame[last] ^= 0xFF;
        assert!(matches!(
            frame_decode(&frame, &config),
            Err(CodecError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn test_frame_too_large() {
        let config = FrameConfig { max_frame_size: 10, ..Default::default() };
        let big = vec![0u8; 20];
        assert!(matches!(
            frame_encode(&big, &config),
            Err(CodecError::FrameTooLarge { .. })
        ));
    }

    #[test]
    fn test_frame_decode_stream() {
        let config = FrameConfig::default();
        let mut stream = Vec::new();
        stream.extend(frame_encode(b"frame1", &config).unwrap());
        stream.extend(frame_encode(b"frame2", &config).unwrap());
        stream.extend(frame_encode(b"frame3", &config).unwrap());

        let (frames, consumed) = frame_decode_stream(&stream, &config);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0], b"frame1");
        assert_eq!(frames[1], b"frame2");
        assert_eq!(frames[2], b"frame3");
        assert_eq!(consumed, stream.len());
    }

    #[test]
    fn test_frame_decode_stream_partial() {
        let config = FrameConfig::default();
        let mut stream = Vec::new();
        stream.extend(frame_encode(b"complete", &config).unwrap());
        stream.extend(&[1, 0, 0]); // Incomplete frame header.

        let (frames, consumed) = frame_decode_stream(&stream, &config);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], b"complete");
        assert!(consumed < stream.len());
    }

    #[test]
    fn test_mixed_fields() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_u8(1);
        w.write_u16(1000);
        w.write_varint(999999);
        w.write_string("hello");
        w.write_u32(0xCAFE);
        let data = w.finish();

        let mut r = PacketReader::new(&data, Endian::Big);
        assert_eq!(r.read_u8().unwrap(), 1);
        assert_eq!(r.read_u16().unwrap(), 1000);
        assert_eq!(r.read_varint().unwrap(), 999999);
        assert_eq!(r.read_string().unwrap(), "hello");
        assert_eq!(r.read_u32().unwrap(), 0xCAFE);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn test_writer_with_checksum() {
        let mut w = PacketWriter::new(Endian::Big);
        w.write_string("data");
        w.append_checksum();
        let data = w.finish();
        // Last 4 bytes are the checksum.
        assert!(data.len() > 4);
    }
}
