//! Page format and serialization

use super::error::PersistenceError;
use joule_db_core::storage::PageId;

/// Default page size (4KB)
pub const PAGE_SIZE: usize = 4096;

/// Page header size (magic + version + checksum + flags)
pub const PAGE_HEADER_SIZE: usize = 16;

/// Maximum page data size
pub const PAGE_DATA_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;

/// Page magic number for validation
const PAGE_MAGIC: u32 = 0x57454156; // "WEAV" in ASCII

/// Current page format version
const PAGE_VERSION: u8 = 1;

/// Page flags
#[derive(Debug, Clone, Copy, Default)]
pub struct PageFlags(pub u8);

impl PageFlags {
    pub const NONE: u8 = 0;
    pub const COMPRESSED: u8 = 1 << 0;
    pub const ENCRYPTED: u8 = 1 << 1;
    pub const WAL_PENDING: u8 = 1 << 2;

    pub fn new() -> Self {
        Self(Self::NONE)
    }

    pub fn is_compressed(&self) -> bool {
        self.0 & Self::COMPRESSED != 0
    }

    pub fn is_encrypted(&self) -> bool {
        self.0 & Self::ENCRYPTED != 0
    }

    pub fn set_compressed(&mut self, value: bool) {
        if value {
            self.0 |= Self::COMPRESSED;
        } else {
            self.0 &= !Self::COMPRESSED;
        }
    }

    pub fn set_encrypted(&mut self, value: bool) {
        if value {
            self.0 |= Self::ENCRYPTED;
        } else {
            self.0 &= !Self::ENCRYPTED;
        }
    }
}

/// Serialized page with header
#[derive(Debug, Clone)]
pub struct SerializedPage {
    /// Page ID
    pub page_id: PageId,
    /// Page flags
    pub flags: PageFlags,
    /// Raw data (including header)
    pub data: Vec<u8>,
}

impl SerializedPage {
    /// Create a new serialized page from raw data
    pub fn new(page_id: PageId, page_data: &[u8], flags: PageFlags) -> Self {
        let mut data = vec![0u8; PAGE_SIZE];

        // Write header
        // [0..4] Magic number
        data[0..4].copy_from_slice(&PAGE_MAGIC.to_le_bytes());
        // [4] Version
        data[4] = PAGE_VERSION;
        // [5] Flags
        data[5] = flags.0;
        // [6..8] Reserved
        data[6] = 0;
        data[7] = 0;
        // [8..12] Data length
        let data_len = page_data.len().min(PAGE_DATA_SIZE) as u32;
        data[8..12].copy_from_slice(&data_len.to_le_bytes());
        // [12..16] Checksum (computed after data is written)

        // Write data
        let copy_len = page_data.len().min(PAGE_DATA_SIZE);
        data[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + copy_len].copy_from_slice(&page_data[..copy_len]);

        // Compute and write checksum
        let checksum = Self::compute_checksum(&data[PAGE_HEADER_SIZE..]);
        data[12..16].copy_from_slice(&checksum.to_le_bytes());

        Self {
            page_id,
            flags,
            data,
        }
    }

    /// Parse a serialized page from raw bytes
    pub fn parse(
        page_id: PageId,
        raw_data: &[u8],
    ) -> Result<(Vec<u8>, PageFlags), PersistenceError> {
        if raw_data.len() < PAGE_HEADER_SIZE {
            return Err(PersistenceError::IoError("Page data too short".to_string()));
        }

        // Validate magic
        let magic = u32::from_le_bytes([raw_data[0], raw_data[1], raw_data[2], raw_data[3]]);
        if magic != PAGE_MAGIC {
            return Err(PersistenceError::IoError(format!(
                "Invalid page magic: expected {:08x}, got {:08x}",
                PAGE_MAGIC, magic
            )));
        }

        // Read version
        let version = raw_data[4];
        if version > PAGE_VERSION {
            return Err(PersistenceError::IoError(format!(
                "Unsupported page version: {}",
                version
            )));
        }

        // Read flags
        let flags = PageFlags(raw_data[5]);

        // Read data length
        let data_len =
            u32::from_le_bytes([raw_data[8], raw_data[9], raw_data[10], raw_data[11]]) as usize;
        if data_len > PAGE_DATA_SIZE {
            return Err(PersistenceError::IoError("Invalid data length".to_string()));
        }

        // Verify checksum
        let expected_checksum =
            u32::from_le_bytes([raw_data[12], raw_data[13], raw_data[14], raw_data[15]]);
        let actual_checksum = Self::compute_checksum(&raw_data[PAGE_HEADER_SIZE..]);
        if expected_checksum != actual_checksum {
            return Err(PersistenceError::ChecksumMismatch {
                page_id,
                expected: expected_checksum,
                actual: actual_checksum,
            });
        }

        // Extract data
        let data = raw_data[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + data_len].to_vec();
        Ok((data, flags))
    }

    /// Compute CRC32 checksum
    pub fn compute_checksum(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFFFFFF;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
        }
        !crc
    }

    /// Get the page data without header
    pub fn page_data(&self) -> &[u8] {
        let data_len =
            u32::from_le_bytes([self.data[8], self.data[9], self.data[10], self.data[11]]) as usize;
        &self.data[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + data_len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_roundtrip() {
        let data = b"Hello, JouleDB!";
        let page = SerializedPage::new(1, data, PageFlags::new());

        let (parsed_data, flags) = SerializedPage::parse(1, &page.data).unwrap();
        assert_eq!(parsed_data, data);
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
    }

    #[test]
    fn test_page_flags() {
        let mut flags = PageFlags::new();
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());

        flags.set_compressed(true);
        assert!(flags.is_compressed());

        flags.set_encrypted(true);
        assert!(flags.is_encrypted());
    }

    #[test]
    fn test_checksum() {
        let data1 = b"test data";
        let data2 = b"test datb"; // One bit different

        let crc1 = SerializedPage::compute_checksum(data1);
        let crc2 = SerializedPage::compute_checksum(data2);

        assert_ne!(crc1, crc2);
    }
}
