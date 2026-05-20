//! Page types and operations
//!
//! Pages are the fundamental unit of storage in JouleDB. All data is
//! organized into fixed-size pages that can be read and written atomically.

use crate::error::{CodecError, StorageError};

/// Page identifier type
pub type PageId = u64;

/// Special page ID indicating no page / null reference
pub const NULL_PAGE_ID: PageId = 0;

/// Default page size (64KB — sized for document workloads with large JSON values)
pub const DEFAULT_PAGE_SIZE: usize = 64 * 1024;

/// Page header size in bytes
pub const PAGE_HEADER_SIZE: usize = 32;

/// Magic number for page validation
const PAGE_MAGIC: u32 = 0x57444250; // "WDBP" in ASCII

/// Current page format version
const PAGE_VERSION: u8 = 1;

/// Page header structure
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
/// 0       4     Magic number (0x57444250)
/// 4       1     Version
/// 5       1     Page type
/// 6       2     Flags
/// 8       8     Page ID
/// 16      4     Data length
/// 20      4     Checksum (CRC32)
/// 24      8     Reserved
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageHeader {
    /// Magic number for validation
    pub magic: u32,
    /// Format version
    pub version: u8,
    /// Page type
    pub page_type: PageType,
    /// Page flags
    pub flags: PageFlags,
    /// Page ID
    pub page_id: PageId,
    /// Length of data in page (excluding header)
    pub data_len: u32,
    /// CRC32 checksum of data
    pub checksum: u32,
}

impl PageHeader {
    /// Create a new page header
    pub fn new(page_id: PageId, page_type: PageType) -> Self {
        Self {
            magic: PAGE_MAGIC,
            version: PAGE_VERSION,
            page_type,
            flags: PageFlags::empty(),
            page_id,
            data_len: 0,
            checksum: 0,
        }
    }

    /// Encode header to bytes
    pub fn encode(&self) -> [u8; PAGE_HEADER_SIZE] {
        let mut buf = [0u8; PAGE_HEADER_SIZE];

        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4] = self.version;
        buf[5] = self.page_type as u8;
        buf[6..8].copy_from_slice(&self.flags.bits().to_le_bytes());
        buf[8..16].copy_from_slice(&self.page_id.to_le_bytes());
        buf[16..20].copy_from_slice(&self.data_len.to_le_bytes());
        buf[20..24].copy_from_slice(&self.checksum.to_le_bytes());
        // bytes 24-32 are reserved

        buf
    }

    /// Decode header from bytes
    pub fn decode(buf: &[u8]) -> Result<Self, CodecError> {
        if buf.len() < PAGE_HEADER_SIZE {
            return Err(CodecError::UnexpectedEof {
                expected: PAGE_HEADER_SIZE,
                actual: buf.len(),
            });
        }

        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic != PAGE_MAGIC {
            return Err(CodecError::InvalidFormat {
                reason: format!(
                    "Invalid page magic: expected {:x}, got {:x}",
                    PAGE_MAGIC, magic
                ),
            });
        }

        let version = buf[4];
        if version > PAGE_VERSION {
            return Err(CodecError::InvalidFormat {
                reason: format!("Unsupported page version: {}", version),
            });
        }

        let page_type = PageType::try_from(buf[5])?;
        let flags = PageFlags::from_bits_truncate(u16::from_le_bytes([buf[6], buf[7]]));
        let page_id = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]);
        let data_len = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
        let checksum = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);

        Ok(Self {
            magic,
            version,
            page_type,
            flags,
            page_id,
            data_len,
            checksum,
        })
    }
}

/// Page type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageType {
    /// Free/unused page
    Free = 0,
    /// B-tree internal node
    BTreeInternal = 1,
    /// B-tree leaf node
    BTreeLeaf = 2,
    /// Overflow page for large values
    Overflow = 3,
    /// Free list page
    FreeList = 4,
    /// Metadata page
    Metadata = 5,
    /// Extent header — describes a contiguous run of data pages.
    /// Used for large blobs (LLM weight tensors, etc.) where chained
    /// overflow pages would cause 512+ random reads.
    /// Format: [page_count: u64][total_bytes: u64][data...]
    /// Subsequent pages in the extent have type ExtentData.
    ExtentHeader = 6,
    /// Extent data — continuation pages in a contiguous extent.
    /// No per-page header overhead beyond the standard page header.
    ExtentData = 7,
}

impl TryFrom<u8> for PageType {
    type Error = CodecError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(PageType::Free),
            1 => Ok(PageType::BTreeInternal),
            2 => Ok(PageType::BTreeLeaf),
            3 => Ok(PageType::Overflow),
            4 => Ok(PageType::FreeList),
            5 => Ok(PageType::Metadata),
            6 => Ok(PageType::ExtentHeader),
            7 => Ok(PageType::ExtentData),
            _ => Err(CodecError::UnknownType { tag: value }),
        }
    }
}

/// Page flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageFlags(u16);

impl PageFlags {
    /// No flags set
    pub const NONE: u16 = 0;
    /// Page data is compressed
    pub const COMPRESSED: u16 = 1 << 0;
    /// Page data is encrypted
    pub const ENCRYPTED: u16 = 1 << 1;
    /// Page has pending WAL write
    pub const WAL_PENDING: u16 = 1 << 2;
    /// Page is dirty (modified in memory)
    pub const DIRTY: u16 = 1 << 3;

    /// Create empty flags
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Create flags from bits
    pub const fn from_bits_truncate(bits: u16) -> Self {
        Self(bits)
    }

    /// Get raw bits
    pub const fn bits(&self) -> u16 {
        self.0
    }

    /// Check if flag is set
    pub const fn contains(&self, flag: u16) -> bool {
        (self.0 & flag) != 0
    }

    /// Set a flag
    pub fn set(&mut self, flag: u16) {
        self.0 |= flag;
    }

    /// Clear a flag
    pub fn clear(&mut self, flag: u16) {
        self.0 &= !flag;
    }
}

/// A database page
///
/// Pages are the fundamental unit of I/O. All reads and writes happen
/// at page granularity.
#[derive(Clone)]
pub struct Page {
    /// Page ID
    pub id: PageId,
    /// Page type
    pub page_type: PageType,
    /// Page flags
    pub flags: PageFlags,
    /// Page data (excluding header)
    pub data: Vec<u8>,
}

impl Page {
    /// Create a new empty page
    pub fn new(id: PageId, page_type: PageType) -> Self {
        Self {
            id,
            page_type,
            flags: PageFlags::empty(),
            data: Vec::new(),
        }
    }

    /// Create a new page with data
    pub fn with_data(id: PageId, page_type: PageType, data: Vec<u8>) -> Self {
        Self {
            id,
            page_type,
            flags: PageFlags::empty(),
            data,
        }
    }

    /// Create a leaf page
    pub fn new_leaf(id: PageId) -> Self {
        Self::new(id, PageType::BTreeLeaf)
    }

    /// Create an internal page
    pub fn new_internal(id: PageId) -> Self {
        Self::new(id, PageType::BTreeInternal)
    }

    /// Check if page is dirty
    pub fn is_dirty(&self) -> bool {
        self.flags.contains(PageFlags::DIRTY)
    }

    /// Mark page as dirty
    pub fn mark_dirty(&mut self) {
        self.flags.set(PageFlags::DIRTY);
    }

    /// Clear dirty flag
    pub fn clear_dirty(&mut self) {
        self.flags.clear(PageFlags::DIRTY);
    }

    /// Calculate CRC32 checksum of page data
    pub fn checksum(&self) -> u32 {
        crc32(&self.data)
    }

    /// Encode page to bytes (header + data)
    pub fn encode(&self, page_size: usize) -> Result<Vec<u8>, StorageError> {
        let total_size = PAGE_HEADER_SIZE + self.data.len();
        if total_size > page_size {
            return Err(StorageError::PageSizeExceeded {
                max: page_size,
                actual: total_size,
            });
        }

        let mut header = PageHeader::new(self.id, self.page_type);
        header.flags = self.flags;
        header.data_len = self.data.len() as u32;
        header.checksum = self.checksum();

        let mut buf = Vec::with_capacity(page_size);
        buf.extend_from_slice(&header.encode());
        buf.extend_from_slice(&self.data);

        // Pad to page size
        buf.resize(page_size, 0);

        Ok(buf)
    }

    /// Decode page from bytes
    pub fn decode(buf: &[u8]) -> Result<Self, CodecError> {
        let header = PageHeader::decode(buf)?;

        let data_start = PAGE_HEADER_SIZE;
        let data_end = data_start + header.data_len as usize;

        if buf.len() < data_end {
            return Err(CodecError::UnexpectedEof {
                expected: data_end,
                actual: buf.len(),
            });
        }

        let data = buf[data_start..data_end].to_vec();

        // Verify checksum
        let actual_checksum = crc32(&data);
        if actual_checksum != header.checksum {
            return Err(CodecError::InvalidFormat {
                reason: format!(
                    "Checksum mismatch: expected {:x}, got {:x}",
                    header.checksum, actual_checksum
                ),
            });
        }

        Ok(Self {
            id: header.page_id,
            page_type: header.page_type,
            flags: header.flags,
            data,
        })
    }
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("id", &self.id)
            .field("page_type", &self.page_type)
            .field("flags", &self.flags)
            .field("data_len", &self.data.len())
            .finish()
    }
}

/// Simple CRC32 implementation (IEEE polynomial)
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;

    for byte in data {
        crc ^= *byte as u32;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_header_roundtrip() {
        let header = PageHeader::new(42, PageType::BTreeLeaf);
        let encoded = header.encode();
        let decoded = PageHeader::decode(&encoded).unwrap();

        assert_eq!(header.page_id, decoded.page_id);
        assert_eq!(header.page_type, decoded.page_type);
        assert_eq!(header.version, decoded.version);
    }

    #[test]
    fn test_page_roundtrip() {
        let page = Page::with_data(42, PageType::BTreeLeaf, b"hello world".to_vec());
        let encoded = page.encode(DEFAULT_PAGE_SIZE).unwrap();
        let decoded = Page::decode(&encoded).unwrap();

        assert_eq!(page.id, decoded.id);
        assert_eq!(page.page_type, decoded.page_type);
        assert_eq!(page.data, decoded.data);
    }

    #[test]
    fn test_page_checksum() {
        let page = Page::with_data(1, PageType::BTreeLeaf, b"test data".to_vec());
        let checksum1 = page.checksum();

        let mut page2 = page.clone();
        page2.data[0] = b'X';
        let checksum2 = page2.checksum();

        assert_ne!(checksum1, checksum2);
    }

    #[test]
    fn test_page_flags() {
        let mut flags = PageFlags::empty();
        assert!(!flags.contains(PageFlags::DIRTY));

        flags.set(PageFlags::DIRTY);
        assert!(flags.contains(PageFlags::DIRTY));

        flags.clear(PageFlags::DIRTY);
        assert!(!flags.contains(PageFlags::DIRTY));
    }

    #[test]
    fn test_crc32() {
        // Known test vector
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }
}
