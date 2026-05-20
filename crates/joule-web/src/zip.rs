//! ZIP archive creation and parsing.
//!
//! Replaces JSZip and archiver with a pure Rust ZIP builder that
//! generates valid ZIP files using the Stored method (no compression
//! library required).

use chrono::{DateTime, Utc, Datelike, Timelike};

// ── CRC-32 ─────────────────────────────────────────────────────────

/// CRC-32 lookup table (IEEE / ISO 3309 polynomial 0xEDB88320).
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

/// Compute CRC-32 checksum.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = CRC32_TABLE[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

// ── Types ──────────────────────────────────────────────────────────

/// Compression method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// No compression — data stored as-is.
    Stored,
}

/// A single entry in a ZIP archive.
#[derive(Debug, Clone)]
pub struct ZipEntry {
    pub name: String,
    pub data: Vec<u8>,
    pub modified: DateTime<Utc>,
    pub compression: Compression,
    pub crc32: u32,
}

/// A ZIP archive containing zero or more entries.
#[derive(Debug, Clone)]
pub struct ZipArchive {
    pub entries: Vec<ZipEntry>,
}

impl ZipArchive {
    /// Create an empty archive.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a file with raw bytes.
    pub fn add_file(&mut self, name: &str, data: &[u8]) {
        let checksum = crc32(data);
        self.entries.push(ZipEntry {
            name: name.to_string(),
            data: data.to_vec(),
            modified: Utc::now(),
            compression: Compression::Stored,
            crc32: checksum,
        });
    }

    /// Add a text file (convenience method).
    pub fn add_text_file(&mut self, name: &str, text: &str) {
        self.add_file(name, text.as_bytes());
    }

    /// Number of entries.
    pub fn file_count(&self) -> usize {
        self.entries.len()
    }

    /// Total uncompressed size across all entries.
    pub fn total_uncompressed_size(&self) -> usize {
        self.entries.iter().map(|e| e.data.len()).sum()
    }

    /// Names of all files in the archive.
    pub fn file_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    /// Generate a valid ZIP file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central_directory = Vec::new();
        let mut local_offsets: Vec<u32> = Vec::new();

        // Local file headers + data
        for entry in &self.entries {
            local_offsets.push(out.len() as u32);
            let name_bytes = entry.name.as_bytes();
            let (date, time) = datetime_to_msdos(&entry.modified);

            // Local file header (signature 0x04034b50)
            out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]); // signature
            out.extend_from_slice(&20u16.to_le_bytes()); // version needed (2.0)
            out.extend_from_slice(&0u16.to_le_bytes()); // general purpose flags
            out.extend_from_slice(&0u16.to_le_bytes()); // compression: stored
            out.extend_from_slice(&time.to_le_bytes()); // last mod time
            out.extend_from_slice(&date.to_le_bytes()); // last mod date
            out.extend_from_slice(&entry.crc32.to_le_bytes());
            out.extend_from_slice(&(entry.data.len() as u32).to_le_bytes()); // compressed
            out.extend_from_slice(&(entry.data.len() as u32).to_le_bytes()); // uncompressed
            out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra field length
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(&entry.data);
        }

        // Central directory
        let cd_offset = out.len() as u32;
        for (i, entry) in self.entries.iter().enumerate() {
            let name_bytes = entry.name.as_bytes();
            let (date, time) = datetime_to_msdos(&entry.modified);

            central_directory.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]); // sig
            central_directory.extend_from_slice(&20u16.to_le_bytes()); // version made by
            central_directory.extend_from_slice(&20u16.to_le_bytes()); // version needed
            central_directory.extend_from_slice(&0u16.to_le_bytes()); // flags
            central_directory.extend_from_slice(&0u16.to_le_bytes()); // compression
            central_directory.extend_from_slice(&time.to_le_bytes());
            central_directory.extend_from_slice(&date.to_le_bytes());
            central_directory.extend_from_slice(&entry.crc32.to_le_bytes());
            central_directory.extend_from_slice(&(entry.data.len() as u32).to_le_bytes());
            central_directory.extend_from_slice(&(entry.data.len() as u32).to_le_bytes());
            central_directory.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            central_directory.extend_from_slice(&0u16.to_le_bytes()); // extra
            central_directory.extend_from_slice(&0u16.to_le_bytes()); // comment
            central_directory.extend_from_slice(&0u16.to_le_bytes()); // disk number start
            central_directory.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
            central_directory.extend_from_slice(&0u32.to_le_bytes()); // external attrs
            central_directory.extend_from_slice(&local_offsets[i].to_le_bytes());
            central_directory.extend_from_slice(name_bytes);
        }

        out.extend_from_slice(&central_directory);

        // End of central directory
        let cd_size = central_directory.len() as u32;
        let entry_count = self.entries.len() as u16;
        out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]); // sig
        out.extend_from_slice(&0u16.to_le_bytes()); // disk number
        out.extend_from_slice(&0u16.to_le_bytes()); // disk with CD
        out.extend_from_slice(&entry_count.to_le_bytes());
        out.extend_from_slice(&entry_count.to_le_bytes());
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // comment length

        out
    }
}

impl Default for ZipArchive {
    fn default() -> Self {
        Self::new()
    }
}

// ── Parsing ────────────────────────────────────────────────────────

/// Error type for ZIP parsing.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ZipError {
    #[error("invalid ZIP file: {0}")]
    Invalid(String),
    #[error("unsupported compression method: {0}")]
    UnsupportedCompression(u16),
}

/// Parse a ZIP file from bytes and extract all entries.
pub fn parse_zip(data: &[u8]) -> Result<ZipArchive, ZipError> {
    // Find end-of-central-directory (scan backwards)
    let eocd_sig = [0x50, 0x4b, 0x05, 0x06];
    let eocd_pos = find_last(data, &eocd_sig)
        .ok_or_else(|| ZipError::Invalid("no end-of-central-directory".to_string()))?;

    if eocd_pos + 22 > data.len() {
        return Err(ZipError::Invalid("truncated EOCD".to_string()));
    }

    let entry_count = u16::from_le_bytes([data[eocd_pos + 10], data[eocd_pos + 11]]) as usize;
    let cd_offset = u32::from_le_bytes([
        data[eocd_pos + 16],
        data[eocd_pos + 17],
        data[eocd_pos + 18],
        data[eocd_pos + 19],
    ]) as usize;

    let mut entries = Vec::with_capacity(entry_count);
    let mut pos = cd_offset;

    for _ in 0..entry_count {
        if pos + 46 > data.len() {
            return Err(ZipError::Invalid("truncated central directory".to_string()));
        }
        if data[pos..pos + 4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err(ZipError::Invalid("bad CD entry signature".to_string()));
        }

        let compression = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
        if compression != 0 {
            return Err(ZipError::UnsupportedCompression(compression));
        }

        let crc = u32::from_le_bytes([
            data[pos + 16],
            data[pos + 17],
            data[pos + 18],
            data[pos + 19],
        ]);
        let compressed_size = u32::from_le_bytes([
            data[pos + 20],
            data[pos + 21],
            data[pos + 22],
            data[pos + 23],
        ]) as usize;
        let name_len =
            u16::from_le_bytes([data[pos + 28], data[pos + 29]]) as usize;
        let extra_len =
            u16::from_le_bytes([data[pos + 30], data[pos + 31]]) as usize;
        let comment_len =
            u16::from_le_bytes([data[pos + 32], data[pos + 33]]) as usize;
        let local_offset = u32::from_le_bytes([
            data[pos + 42],
            data[pos + 43],
            data[pos + 44],
            data[pos + 45],
        ]) as usize;

        let name_start = pos + 46;
        let name =
            String::from_utf8_lossy(&data[name_start..name_start + name_len]).to_string();

        // Read data from local file header
        let local_name_len =
            u16::from_le_bytes([data[local_offset + 26], data[local_offset + 27]]) as usize;
        let local_extra_len =
            u16::from_le_bytes([data[local_offset + 28], data[local_offset + 29]]) as usize;
        let data_start = local_offset + 30 + local_name_len + local_extra_len;
        let file_data = data[data_start..data_start + compressed_size].to_vec();

        entries.push(ZipEntry {
            name,
            data: file_data,
            modified: Utc::now(),
            compression: Compression::Stored,
            crc32: crc,
        });

        pos = name_start + name_len + extra_len + comment_len;
    }

    Ok(ZipArchive { entries })
}

// ── Helpers ────────────────────────────────────────────────────────

fn datetime_to_msdos(dt: &DateTime<Utc>) -> (u16, u16) {
    let year = (dt.year() - 1980).max(0) as u16;
    let month = dt.month() as u16;
    let day = dt.day() as u16;
    let hour = dt.hour() as u16;
    let minute = dt.minute() as u16;
    let second = (dt.second() / 2) as u16;

    let date = (year << 9) | (month << 5) | day;
    let time = (hour << 11) | (minute << 5) | second;
    (date, time)
}

fn find_last(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > haystack.len() {
        return None;
    }
    for i in (0..=(haystack.len() - needle.len())).rev() {
        if haystack[i..i + needle.len()] == *needle {
            return Some(i);
        }
    }
    None
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_value() {
        // CRC-32 of "123456789" is 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn crc32_empty() {
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn single_file_zip() {
        let mut archive = ZipArchive::new();
        archive.add_file("hello.txt", b"Hello, world!");
        let bytes = archive.to_bytes();
        assert_eq!(&bytes[0..4], &[0x50, 0x4b, 0x03, 0x04]);
    }

    #[test]
    fn multi_file_zip() {
        let mut archive = ZipArchive::new();
        archive.add_file("a.txt", b"aaa");
        archive.add_file("b.txt", b"bbb");
        archive.add_file("c.txt", b"ccc");
        assert_eq!(archive.file_count(), 3);
    }

    #[test]
    fn parse_roundtrip() {
        let mut archive = ZipArchive::new();
        archive.add_file("test.txt", b"test content");
        archive.add_file("data.bin", &[0u8, 1, 2, 3, 4, 5]);
        let bytes = archive.to_bytes();

        let parsed = parse_zip(&bytes).unwrap();
        assert_eq!(parsed.file_count(), 2);
        assert_eq!(parsed.entries[0].name, "test.txt");
        assert_eq!(parsed.entries[0].data, b"test content");
        assert_eq!(parsed.entries[1].name, "data.bin");
        assert_eq!(parsed.entries[1].data, &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn text_file_convenience() {
        let mut archive = ZipArchive::new();
        archive.add_text_file("readme.txt", "Hello from joule-web");
        assert_eq!(archive.entries[0].data, b"Hello from joule-web");
    }

    #[test]
    fn file_names() {
        let mut archive = ZipArchive::new();
        archive.add_file("x.txt", b"");
        archive.add_file("y.txt", b"");
        assert_eq!(archive.file_names(), vec!["x.txt", "y.txt"]);
    }

    #[test]
    fn total_size() {
        let mut archive = ZipArchive::new();
        archive.add_file("a.txt", b"12345");
        archive.add_file("b.txt", b"67890ab");
        assert_eq!(archive.total_uncompressed_size(), 12);
    }

    #[test]
    fn valid_zip_magic() {
        let archive = ZipArchive::new();
        let bytes = archive.to_bytes();
        // Even empty ZIP has EOCD with PK signature
        assert!(bytes.windows(4).any(|w| w == [0x50, 0x4b, 0x05, 0x06]));
    }

    #[test]
    fn crc32_matches_on_parse() {
        let mut archive = ZipArchive::new();
        let data = b"integrity check";
        archive.add_file("check.txt", data);
        let bytes = archive.to_bytes();
        let parsed = parse_zip(&bytes).unwrap();
        assert_eq!(parsed.entries[0].crc32, crc32(data));
    }
}
