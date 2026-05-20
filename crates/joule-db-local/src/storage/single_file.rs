//! Single-File Virtual File System (VFS)
//!
//! Packs multiple logical files (data, metadata, WAL, indexes) into a single
//! physical file, like SQLite. This enables:
//!
//! - **Zero-configuration deployment**: Just point at a file path
//! - **Atomic operations**: fsync one file = durable everything
//! - **Easy backup**: Copy one file = complete database
//! - **Embeddable**: No directory management needed
//!
//! ## File Format
//!
//! ```text
//! +------------------+
//! | Header (128 B)   |  Magic, version, page size, page count, directory offset
//! +------------------+
//! | Directory Page   |  Maps logical file names → page ranges
//! +------------------+
//! | Data Pages...    |  4KB pages, each belonging to a logical file
//! +------------------+
//! | Free Page List   |  Bitmap of available pages
//! +------------------+
//! ```
//!
//! Each logical file is a sequence of pages. The directory maps
//! `logical_name → Vec<PageId>`.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Magic bytes for JouleDB single-file format
const MAGIC: [u8; 8] = *b"INVDB001";
/// File format version
const FORMAT_VERSION: u32 = 1;
/// Default page size (4 KiB)
const DEFAULT_PAGE_SIZE: u32 = 4096;
/// Header size (128 bytes, padded for alignment)
const HEADER_SIZE: u64 = 128;

/// Single-file VFS configuration
#[derive(Debug, Clone)]
pub struct SingleFileConfig {
    /// Page size in bytes (must be power of 2, minimum 512)
    pub page_size: u32,
    /// Initial file size hint (pages pre-allocated)
    pub initial_pages: u32,
    /// Enable fsync after writes for durability
    pub sync_on_write: bool,
}

impl Default for SingleFileConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            initial_pages: 16,
            sync_on_write: true,
        }
    }
}

/// File header stored at offset 0
#[derive(Debug, Clone)]
struct FileHeader {
    /// Magic bytes (INVDB001)
    magic: [u8; 8],
    /// Format version
    version: u32,
    /// Page size in bytes
    page_size: u32,
    /// Total number of pages allocated
    total_pages: u32,
    /// Page ID of the directory page
    directory_page: u32,
    /// Number of free pages
    free_page_count: u32,
    /// First free page ID (head of free list)
    free_list_head: u32,
    /// Transaction counter (monotonically increasing)
    tx_counter: u64,
}

impl FileHeader {
    fn new(page_size: u32) -> Self {
        Self {
            magic: MAGIC,
            version: FORMAT_VERSION,
            page_size,
            total_pages: 2, // header page + directory page
            directory_page: 1,
            free_page_count: 0,
            free_list_head: 0,
            tx_counter: 0,
        }
    }

    fn serialize(&self) -> [u8; HEADER_SIZE as usize] {
        let mut buf = [0u8; HEADER_SIZE as usize];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.page_size.to_le_bytes());
        buf[16..20].copy_from_slice(&self.total_pages.to_le_bytes());
        buf[20..24].copy_from_slice(&self.directory_page.to_le_bytes());
        buf[24..28].copy_from_slice(&self.free_page_count.to_le_bytes());
        buf[28..32].copy_from_slice(&self.free_list_head.to_le_bytes());
        buf[32..40].copy_from_slice(&self.tx_counter.to_le_bytes());
        buf
    }

    fn deserialize(buf: &[u8; HEADER_SIZE as usize]) -> Result<Self, SingleFileError> {
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buf[0..8]);
        if magic != MAGIC {
            return Err(SingleFileError::InvalidMagic);
        }

        Ok(Self {
            magic,
            version: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            page_size: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            total_pages: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
            directory_page: u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
            free_page_count: u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]),
            free_list_head: u32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]),
            tx_counter: u64::from_le_bytes([
                buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
            ]),
        })
    }
}

/// Directory entry mapping a logical file name to page IDs
#[derive(Debug, Clone)]
struct DirectoryEntry {
    /// Logical file name (e.g., "data", "meta", "wal", "index")
    name: String,
    /// Ordered list of page IDs belonging to this file
    pages: Vec<u32>,
    /// Total logical size in bytes
    logical_size: u64,
}

/// A handle to a logical file within the VFS
pub struct LogicalFile {
    /// Logical file name
    pub name: String,
    /// Current read/write position
    position: u64,
}

/// The single-file VFS
pub struct SingleFileVfs {
    /// Path to the physical file
    path: PathBuf,
    /// Configuration
    config: SingleFileConfig,
    /// File header
    header: RwLock<FileHeader>,
    /// Directory: logical file name → entry
    directory: RwLock<HashMap<String, DirectoryEntry>>,
    /// The underlying file handle
    file: RwLock<File>,
}

impl SingleFileVfs {
    /// Create a new single-file database at the given path
    pub fn create(
        path: impl AsRef<Path>,
        config: SingleFileConfig,
    ) -> Result<Self, SingleFileError> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        let header = FileHeader::new(config.page_size);

        // Write header
        file.seek(SeekFrom::Start(0))
            .map_err(|e| SingleFileError::Io(e.to_string()))?;
        file.write_all(&header.serialize())
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        // Write empty directory page
        let empty_dir = vec![0u8; config.page_size as usize];
        file.write_all(&empty_dir)
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        if config.sync_on_write {
            file.sync_all()
                .map_err(|e| SingleFileError::Io(e.to_string()))?;
        }

        Ok(Self {
            path,
            config,
            header: RwLock::new(header),
            directory: RwLock::new(HashMap::new()),
            file: RwLock::new(file),
        })
    }

    /// Open an existing single-file database
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SingleFileError> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        // Read header
        let mut header_buf = [0u8; HEADER_SIZE as usize];
        file.seek(SeekFrom::Start(0))
            .map_err(|e| SingleFileError::Io(e.to_string()))?;
        file.read_exact(&mut header_buf)
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        let header = FileHeader::deserialize(&header_buf)?;

        // Read directory
        let directory = Self::read_directory(&mut file, &header)?;

        let config = SingleFileConfig {
            page_size: header.page_size,
            ..Default::default()
        };

        Ok(Self {
            path,
            config,
            header: RwLock::new(header),
            directory: RwLock::new(directory),
            file: RwLock::new(file),
        })
    }

    /// Read the directory from the file
    fn read_directory(
        file: &mut File,
        header: &FileHeader,
    ) -> Result<HashMap<String, DirectoryEntry>, SingleFileError> {
        let offset = HEADER_SIZE + (header.directory_page as u64 - 1) * header.page_size as u64;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        let mut page_buf = vec![0u8; header.page_size as usize];
        file.read_exact(&mut page_buf)
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        // Parse directory entries
        // Format: [entry_count(4)] [entries...]
        // Each entry: [name_len(2)] [name] [page_count(4)] [pages...] [logical_size(8)]
        let mut dir = HashMap::new();
        if page_buf.iter().all(|b| *b == 0) {
            return Ok(dir);
        }

        let entry_count =
            u32::from_le_bytes([page_buf[0], page_buf[1], page_buf[2], page_buf[3]]) as usize;

        // Cap entry_count to prevent huge loops from corrupted directory
        let max_entries = page_buf.len() / 8; // minimum bytes per entry
        if entry_count > max_entries {
            return Err(SingleFileError::CorruptDirectory);
        }

        let mut pos = 4;

        for _ in 0..entry_count {
            if pos + 2 > page_buf.len() {
                break;
            }
            let name_len = u16::from_le_bytes([page_buf[pos], page_buf[pos + 1]]) as usize;
            pos += 2;

            if pos + name_len > page_buf.len() {
                break;
            }
            let name = String::from_utf8_lossy(&page_buf[pos..pos + name_len]).to_string();
            pos += name_len;

            if pos + 4 > page_buf.len() {
                break;
            }
            let page_count = u32::from_le_bytes([
                page_buf[pos],
                page_buf[pos + 1],
                page_buf[pos + 2],
                page_buf[pos + 3],
            ]) as usize;
            pos += 4;

            // Cap page_count against remaining buffer to prevent OOM from corrupted data
            let max_pages = (page_buf.len() - pos) / 4;
            if page_count > max_pages {
                return Err(SingleFileError::CorruptDirectory);
            }

            let mut pages = Vec::with_capacity(page_count);
            for _ in 0..page_count {
                if pos + 4 > page_buf.len() {
                    break;
                }
                let page_id = u32::from_le_bytes([
                    page_buf[pos],
                    page_buf[pos + 1],
                    page_buf[pos + 2],
                    page_buf[pos + 3],
                ]);
                pages.push(page_id);
                pos += 4;
            }

            if pos + 8 > page_buf.len() {
                break;
            }
            let logical_size = u64::from_le_bytes([
                page_buf[pos],
                page_buf[pos + 1],
                page_buf[pos + 2],
                page_buf[pos + 3],
                page_buf[pos + 4],
                page_buf[pos + 5],
                page_buf[pos + 6],
                page_buf[pos + 7],
            ]);
            pos += 8;

            dir.insert(
                name.clone(),
                DirectoryEntry {
                    name,
                    pages,
                    logical_size,
                },
            );
        }

        Ok(dir)
    }

    /// Write the directory back to the file
    fn write_directory(&self) -> Result<(), SingleFileError> {
        let header = self
            .header
            .read()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        let directory = self
            .directory
            .read()
            .map_err(|_| SingleFileError::LockPoisoned)?;

        let mut page_buf = vec![0u8; header.page_size as usize];
        let entry_count = directory.len() as u32;
        page_buf[0..4].copy_from_slice(&entry_count.to_le_bytes());
        let mut pos = 4;

        for entry in directory.values() {
            // Calculate space needed for this entry:
            // name_len(2) + name + page_count(4) + pages(4 each) + logical_size(8)
            let entry_size = 2 + entry.name.len() + 4 + entry.pages.len() * 4 + 8;
            if pos + entry_size > page_buf.len() {
                return Err(SingleFileError::Io(format!(
                    "Directory overflow: {} bytes needed but only {} available in page",
                    pos + entry_size,
                    page_buf.len()
                )));
            }

            // Name
            let name_bytes = entry.name.as_bytes();
            page_buf[pos..pos + 2].copy_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            pos += 2;
            page_buf[pos..pos + name_bytes.len()].copy_from_slice(name_bytes);
            pos += name_bytes.len();

            // Pages
            page_buf[pos..pos + 4].copy_from_slice(&(entry.pages.len() as u32).to_le_bytes());
            pos += 4;
            for &page_id in &entry.pages {
                page_buf[pos..pos + 4].copy_from_slice(&page_id.to_le_bytes());
                pos += 4;
            }

            // Logical size
            page_buf[pos..pos + 8].copy_from_slice(&entry.logical_size.to_le_bytes());
            pos += 8;
        }

        let offset = HEADER_SIZE + (header.directory_page as u64 - 1) * header.page_size as u64;
        let mut file = self
            .file
            .write()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| SingleFileError::Io(e.to_string()))?;
        file.write_all(&page_buf)
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        Ok(())
    }

    /// Allocate a new page, returning its page ID
    fn allocate_page(&self) -> Result<u32, SingleFileError> {
        let mut header = self
            .header
            .write()
            .map_err(|_| SingleFileError::LockPoisoned)?;

        // Check free list first
        if header.free_list_head != 0 {
            let page_id = header.free_list_head;

            // Read the next-pointer from the first 4 bytes of the freed page
            let page_offset = HEADER_SIZE + page_id as u64 * header.page_size as u64;
            let mut next_buf = [0u8; 4];
            let mut file = self
                .file
                .write()
                .map_err(|_| SingleFileError::LockPoisoned)?;
            file.seek(SeekFrom::Start(page_offset))
                .map_err(|e| SingleFileError::Io(e.to_string()))?;
            file.read_exact(&mut next_buf)
                .map_err(|e| SingleFileError::Io(e.to_string()))?;

            let next_free = u32::from_le_bytes(next_buf);
            header.free_list_head = next_free;
            header.free_page_count -= 1;
            return Ok(page_id);
        }

        // Allocate at end
        let page_id = header.total_pages;
        header.total_pages += 1;

        // Extend the file
        let new_size = HEADER_SIZE + header.total_pages as u64 * header.page_size as u64;
        let mut file = self
            .file
            .write()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        file.set_len(new_size)
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        // Write empty page
        let offset = HEADER_SIZE + (page_id as u64) * header.page_size as u64;
        let empty = vec![0u8; header.page_size as usize];
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| SingleFileError::Io(e.to_string()))?;
        file.write_all(&empty)
            .map_err(|e| SingleFileError::Io(e.to_string()))?;

        Ok(page_id)
    }

    /// Write the header to disk
    fn flush_header(&self) -> Result<(), SingleFileError> {
        let header = self
            .header
            .read()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        let mut file = self
            .file
            .write()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| SingleFileError::Io(e.to_string()))?;
        file.write_all(&header.serialize())
            .map_err(|e| SingleFileError::Io(e.to_string()))?;
        Ok(())
    }

    /// Create or get a logical file within the VFS
    pub fn open_logical_file(&self, name: &str) -> Result<LogicalFile, SingleFileError> {
        let mut directory = self
            .directory
            .write()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        if !directory.contains_key(name) {
            directory.insert(
                name.to_string(),
                DirectoryEntry {
                    name: name.to_string(),
                    pages: Vec::new(),
                    logical_size: 0,
                },
            );
        }
        Ok(LogicalFile {
            name: name.to_string(),
            position: 0,
        })
    }

    /// Write data to a logical file at a given offset
    pub fn write_logical(
        &self,
        logical_file: &str,
        offset: u64,
        data: &[u8],
    ) -> Result<(), SingleFileError> {
        let page_size = self.config.page_size as u64;

        let mut directory = self
            .directory
            .write()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        let entry = directory
            .get_mut(logical_file)
            .ok_or(SingleFileError::FileNotFound(logical_file.to_string()))?;

        // Calculate which pages are needed
        let start_page_idx = (offset / page_size) as usize;
        let end_page_idx = ((offset + data.len() as u64 + page_size - 1) / page_size) as usize;

        // Ensure enough pages are allocated
        while entry.pages.len() < end_page_idx {
            let new_page = self.allocate_page()?;
            entry.pages.push(new_page);
        }

        // Write data across pages
        let mut remaining = data;
        let mut current_offset = offset;

        let file = self
            .file
            .write()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        // Need to drop and re-acquire to avoid double borrow
        drop(file);

        while !remaining.is_empty() {
            let page_idx = (current_offset / page_size) as usize;
            let offset_in_page = (current_offset % page_size) as usize;
            let bytes_in_page = std::cmp::min(remaining.len(), page_size as usize - offset_in_page);

            let physical_page_id = entry.pages[page_idx];
            let physical_offset = HEADER_SIZE + physical_page_id as u64 * page_size;

            let mut file = self
                .file
                .write()
                .map_err(|_| SingleFileError::LockPoisoned)?;
            file.seek(SeekFrom::Start(physical_offset + offset_in_page as u64))
                .map_err(|e| SingleFileError::Io(e.to_string()))?;
            file.write_all(&remaining[..bytes_in_page])
                .map_err(|e| SingleFileError::Io(e.to_string()))?;

            remaining = &remaining[bytes_in_page..];
            current_offset += bytes_in_page as u64;
        }

        // Update logical size
        let new_end = offset + data.len() as u64;
        if new_end > entry.logical_size {
            entry.logical_size = new_end;
        }

        Ok(())
    }

    /// Read data from a logical file at a given offset
    pub fn read_logical(
        &self,
        logical_file: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, SingleFileError> {
        let page_size = self.config.page_size as u64;

        let directory = self
            .directory
            .read()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        let entry = directory
            .get(logical_file)
            .ok_or(SingleFileError::FileNotFound(logical_file.to_string()))?;

        if offset >= entry.logical_size {
            return Ok(Vec::new());
        }

        let actual_length = std::cmp::min(length, (entry.logical_size - offset) as usize);
        let mut result = vec![0u8; actual_length];
        let mut bytes_read = 0;
        let mut current_offset = offset;

        while bytes_read < actual_length {
            let page_idx = (current_offset / page_size) as usize;
            if page_idx >= entry.pages.len() {
                break;
            }

            let offset_in_page = (current_offset % page_size) as usize;
            let bytes_in_page = std::cmp::min(
                actual_length - bytes_read,
                page_size as usize - offset_in_page,
            );

            let physical_page_id = entry.pages[page_idx];
            let physical_offset = HEADER_SIZE + physical_page_id as u64 * page_size;

            let mut file = self
                .file
                .write()
                .map_err(|_| SingleFileError::LockPoisoned)?;
            file.seek(SeekFrom::Start(physical_offset + offset_in_page as u64))
                .map_err(|e| SingleFileError::Io(e.to_string()))?;
            file.read_exact(&mut result[bytes_read..bytes_read + bytes_in_page])
                .map_err(|e| SingleFileError::Io(e.to_string()))?;

            bytes_read += bytes_in_page;
            current_offset += bytes_in_page as u64;
        }

        Ok(result)
    }

    /// Get the logical size of a file
    pub fn logical_size(&self, name: &str) -> Result<u64, SingleFileError> {
        let directory = self
            .directory
            .read()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        directory
            .get(name)
            .map(|e| e.logical_size)
            .ok_or(SingleFileError::FileNotFound(name.to_string()))
    }

    /// List all logical files
    pub fn list_files(&self) -> Result<Vec<String>, SingleFileError> {
        let directory = self
            .directory
            .read()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        Ok(directory.keys().cloned().collect())
    }

    /// Sync all data to disk
    pub fn sync(&self) -> Result<(), SingleFileError> {
        self.flush_header()?;
        self.write_directory()?;
        let file = self
            .file
            .read()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        file.sync_all()
            .map_err(|e| SingleFileError::Io(e.to_string()))?;
        Ok(())
    }

    /// Get the physical file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get database statistics
    pub fn stats(&self) -> Result<VfsStats, SingleFileError> {
        let header = self
            .header
            .read()
            .map_err(|_| SingleFileError::LockPoisoned)?;
        let directory = self
            .directory
            .read()
            .map_err(|_| SingleFileError::LockPoisoned)?;

        let used_pages: u32 = directory.values().map(|e| e.pages.len() as u32).sum();

        Ok(VfsStats {
            total_pages: header.total_pages,
            used_pages,
            free_pages: header.free_page_count,
            page_size: header.page_size,
            logical_files: directory.len(),
            tx_counter: header.tx_counter,
        })
    }
}

/// VFS statistics
#[derive(Debug, Clone)]
pub struct VfsStats {
    pub total_pages: u32,
    pub used_pages: u32,
    pub free_pages: u32,
    pub page_size: u32,
    pub logical_files: usize,
    pub tx_counter: u64,
}

/// Errors from the single-file VFS
#[derive(Debug)]
pub enum SingleFileError {
    /// I/O error
    Io(String),
    /// Invalid magic bytes (not an JouleDB file)
    InvalidMagic,
    /// Logical file not found
    FileNotFound(String),
    /// Lock poisoned
    LockPoisoned,
    /// Page out of bounds
    PageOutOfBounds(u32),
    /// Corrupt directory
    CorruptDirectory,
}

impl std::fmt::Display for SingleFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::InvalidMagic => write!(f, "Not a valid JouleDB file (invalid magic bytes)"),
            Self::FileNotFound(name) => write!(f, "Logical file '{}' not found", name),
            Self::LockPoisoned => write!(f, "Internal lock poisoned"),
            Self::PageOutOfBounds(id) => write!(f, "Page {} out of bounds", id),
            Self::CorruptDirectory => write!(f, "Directory is corrupt"),
        }
    }
}

impl std::error::Error for SingleFileError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = FileHeader::new(4096);
        let serialized = header.serialize();
        let deserialized = FileHeader::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.magic, MAGIC);
        assert_eq!(deserialized.version, FORMAT_VERSION);
        assert_eq!(deserialized.page_size, 4096);
        assert_eq!(deserialized.total_pages, 2);
    }

    #[test]
    fn test_header_invalid_magic() {
        let mut buf = [0u8; HEADER_SIZE as usize];
        buf[0..8].copy_from_slice(b"BADMAGIC");
        assert!(matches!(
            FileHeader::deserialize(&buf),
            Err(SingleFileError::InvalidMagic)
        ));
    }

    #[test]
    fn test_create_and_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.idb");

        // Create
        {
            let vfs = SingleFileVfs::create(&path, SingleFileConfig::default()).unwrap();
            let stats = vfs.stats().unwrap();
            assert_eq!(stats.total_pages, 2);
            assert_eq!(stats.logical_files, 0);
        }

        // Open
        {
            let vfs = SingleFileVfs::open(&path).unwrap();
            let stats = vfs.stats().unwrap();
            assert_eq!(stats.total_pages, 2);
            assert_eq!(stats.page_size, DEFAULT_PAGE_SIZE);
        }
    }

    #[test]
    fn test_write_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.idb");

        let vfs = SingleFileVfs::create(&path, SingleFileConfig::default()).unwrap();

        // Create logical file and write data
        vfs.open_logical_file("data").unwrap();
        let data = b"Hello, JouleDB!";
        vfs.write_logical("data", 0, data).unwrap();

        // Read it back
        let read_back = vfs.read_logical("data", 0, data.len()).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn test_multiple_logical_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.idb");

        let vfs = SingleFileVfs::create(&path, SingleFileConfig::default()).unwrap();

        vfs.open_logical_file("data").unwrap();
        vfs.open_logical_file("meta").unwrap();
        vfs.open_logical_file("wal").unwrap();

        vfs.write_logical("data", 0, b"data contents").unwrap();
        vfs.write_logical("meta", 0, b"meta contents").unwrap();
        vfs.write_logical("wal", 0, b"wal contents").unwrap();

        assert_eq!(vfs.read_logical("data", 0, 13).unwrap(), b"data contents");
        assert_eq!(vfs.read_logical("meta", 0, 13).unwrap(), b"meta contents");
        assert_eq!(vfs.read_logical("wal", 0, 12).unwrap(), b"wal contents");

        let files = vfs.list_files().unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_cross_page_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.idb");

        let config = SingleFileConfig {
            page_size: 64, // Small pages to force cross-page writes
            ..Default::default()
        };
        let vfs = SingleFileVfs::create(&path, config).unwrap();

        vfs.open_logical_file("data").unwrap();

        // Write data larger than one page
        let data = vec![0xABu8; 200];
        vfs.write_logical("data", 0, &data).unwrap();

        let read_back = vfs.read_logical("data", 0, 200).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn test_logical_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.idb");

        let vfs = SingleFileVfs::create(&path, SingleFileConfig::default()).unwrap();
        vfs.open_logical_file("data").unwrap();

        vfs.write_logical("data", 0, b"hello").unwrap();
        assert_eq!(vfs.logical_size("data").unwrap(), 5);

        vfs.write_logical("data", 100, b"world").unwrap();
        assert_eq!(vfs.logical_size("data").unwrap(), 105);
    }

    #[test]
    fn test_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.idb");

        let vfs = SingleFileVfs::create(&path, SingleFileConfig::default()).unwrap();
        assert!(matches!(
            vfs.read_logical("nonexistent", 0, 10),
            Err(SingleFileError::FileNotFound(_))
        ));
    }

    #[test]
    fn test_sync() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.idb");

        let vfs = SingleFileVfs::create(&path, SingleFileConfig::default()).unwrap();
        vfs.open_logical_file("data").unwrap();
        vfs.write_logical("data", 0, b"persistent data").unwrap();
        vfs.sync().unwrap(); // Should not panic
    }
}
