//! Virtual File System (VFS) Abstraction
//!
//! Provides a platform-independent interface for file operations,
//! enabling custom storage backends (encrypted, in-memory, network, etc.).

use crate::error::StorageError;
use std::io::{Read, Seek, SeekFrom, Write};

/// A virtual file handle that abstracts file operations.
///
/// This trait allows implementing custom file backends such as:
/// - Encrypted file storage
/// - Network-backed storage
/// - Memory-mapped files
/// - Custom compression
pub trait VirtualFile: Read + Write + Seek + Send + Sync {
    /// Get the current length of the file in bytes.
    fn len(&self) -> std::io::Result<u64>;

    /// Check if the file is empty.
    fn is_empty(&self) -> std::io::Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Sync all data to the underlying storage.
    fn sync_all(&self) -> std::io::Result<()>;

    /// Sync data (not metadata) to the underlying storage.
    fn sync_data(&self) -> std::io::Result<()> {
        self.sync_all()
    }

    /// Truncate or extend the file to the specified length.
    fn set_len(&self, size: u64) -> std::io::Result<()>;
}

/// Virtual File System that creates and manages virtual files.
///
/// Implement this trait to provide custom storage backends.
pub trait VirtualFileSystem: Send + Sync {
    /// The type of file handle returned by this VFS.
    type File: VirtualFile;

    /// Open or create a file at the specified path.
    fn open(&self, path: &str, create: bool) -> Result<Self::File, StorageError>;

    /// Check if a file exists.
    fn exists(&self, path: &str) -> Result<bool, StorageError>;

    /// Delete a file.
    fn delete(&self, path: &str) -> Result<(), StorageError>;

    /// Rename a file.
    fn rename(&self, from: &str, to: &str) -> Result<(), StorageError>;

    /// Create a temporary file.
    fn temp_file(&self) -> Result<Self::File, StorageError>;

    /// Get the name/description of this VFS implementation.
    fn name(&self) -> &str;
}

// ============================================================================
// Standard File System Implementation
// ============================================================================

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Mutex;

/// A standard file wrapped to implement VirtualFile.
pub struct StdFile {
    file: Mutex<File>,
}

impl StdFile {
    /// Create a new StdFile from a std::fs::File.
    pub fn new(file: File) -> Self {
        Self {
            file: Mutex::new(file),
        }
    }
}

impl Read for StdFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file
            .lock()
            .expect("lock poisoned: StdFile read")
            .read(buf)
    }
}

impl Write for StdFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file
            .lock()
            .expect("lock poisoned: StdFile write")
            .write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file
            .lock()
            .expect("lock poisoned: StdFile flush")
            .flush()
    }
}

impl Seek for StdFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.file
            .lock()
            .expect("lock poisoned: StdFile seek")
            .seek(pos)
    }
}

impl VirtualFile for StdFile {
    fn len(&self) -> std::io::Result<u64> {
        Ok(self
            .file
            .lock()
            .expect("lock poisoned: StdFile len")
            .metadata()?
            .len())
    }

    fn sync_all(&self) -> std::io::Result<()> {
        self.file
            .lock()
            .expect("lock poisoned: StdFile sync_all")
            .sync_all()
    }

    fn sync_data(&self) -> std::io::Result<()> {
        self.file
            .lock()
            .expect("lock poisoned: StdFile sync_data")
            .sync_data()
    }

    fn set_len(&self, size: u64) -> std::io::Result<()> {
        self.file
            .lock()
            .expect("lock poisoned: StdFile set_len")
            .set_len(size)
    }
}

/// Standard file system VFS implementation.
pub struct StdVfs;

impl StdVfs {
    /// Create a new standard VFS.
    pub fn new() -> Self {
        Self
    }
}

impl Default for StdVfs {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualFileSystem for StdVfs {
    type File = StdFile;

    fn open(&self, path: &str, create: bool) -> Result<Self::File, StorageError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(create)
            .open(path)
            .map_err(|e| StorageError::Io(e.to_string()))?;
        Ok(StdFile::new(file))
    }

    fn exists(&self, path: &str) -> Result<bool, StorageError> {
        Ok(Path::new(path).exists())
    }

    fn delete(&self, path: &str) -> Result<(), StorageError> {
        std::fs::remove_file(path).map_err(|e| StorageError::Io(e.to_string()))
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), StorageError> {
        std::fs::rename(from, to).map_err(|e| StorageError::Io(e.to_string()))
    }

    fn temp_file(&self) -> Result<Self::File, StorageError> {
        let temp_path = std::env::temp_dir().join(format!(
            "jouledb_{}.tmp",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        self.open(
            temp_path
                .to_str()
                .ok_or_else(|| StorageError::Backend("non-UTF8 temp path".to_string()))?,
            true,
        )
    }

    fn name(&self) -> &str {
        "std-fs"
    }
}

// ============================================================================
// In-Memory VFS Implementation
// ============================================================================

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// In-memory file data.
struct MemoryFileData {
    data: Vec<u8>,
    position: u64,
}

/// An in-memory file.
pub struct MemoryFile {
    data: Arc<RwLock<MemoryFileData>>,
}

impl MemoryFile {
    fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(MemoryFileData {
                data: Vec::new(),
                position: 0,
            })),
        }
    }
}

impl Clone for MemoryFile {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
        }
    }
}

impl Read for MemoryFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut data = self.data.write().expect("lock poisoned: MemoryFile read");
        let pos = data.position as usize;
        let available = data.data.len().saturating_sub(pos);
        let to_read = buf.len().min(available);
        buf[..to_read].copy_from_slice(&data.data[pos..pos + to_read]);
        data.position += to_read as u64;
        Ok(to_read)
    }
}

impl Write for MemoryFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut data = self.data.write().expect("lock poisoned: MemoryFile write");
        let pos = data.position as usize;
        let end = pos + buf.len();
        if end > data.data.len() {
            data.data.resize(end, 0);
        }
        data.data[pos..end].copy_from_slice(buf);
        data.position = end as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Seek for MemoryFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let mut data = self.data.write().expect("lock poisoned: MemoryFile seek");
        let new_pos = match pos {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(p) => data.data.len() as i64 + p,
            SeekFrom::Current(p) => data.position as i64 + p,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Seek before start",
            ));
        }
        data.position = new_pos as u64;
        Ok(data.position)
    }
}

impl VirtualFile for MemoryFile {
    fn len(&self) -> std::io::Result<u64> {
        Ok(self
            .data
            .read()
            .expect("lock poisoned: MemoryFile len")
            .data
            .len() as u64)
    }

    fn sync_all(&self) -> std::io::Result<()> {
        Ok(()) // No-op for memory
    }

    fn set_len(&self, size: u64) -> std::io::Result<()> {
        let mut data = self
            .data
            .write()
            .expect("lock poisoned: MemoryFile set_len");
        data.data.resize(size as usize, 0);
        Ok(())
    }
}

/// In-memory VFS for testing and temporary storage.
pub struct MemoryVfs {
    files: RwLock<HashMap<String, MemoryFile>>,
}

impl MemoryVfs {
    /// Create a new in-memory VFS.
    pub fn new() -> Self {
        Self {
            files: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryVfs {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualFileSystem for MemoryVfs {
    type File = MemoryFile;

    fn open(&self, path: &str, create: bool) -> Result<Self::File, StorageError> {
        let mut files = self.files.write().expect("lock poisoned: MemoryVfs open");
        if let Some(file) = files.get(path) {
            Ok(file.clone())
        } else if create {
            let file = MemoryFile::new();
            files.insert(path.to_string(), file.clone());
            Ok(file)
        } else {
            Err(StorageError::Backend(format!("File not found: {}", path)))
        }
    }

    fn exists(&self, path: &str) -> Result<bool, StorageError> {
        Ok(self
            .files
            .read()
            .expect("lock poisoned: MemoryVfs exists")
            .contains_key(path))
    }

    fn delete(&self, path: &str) -> Result<(), StorageError> {
        let mut files = self.files.write().expect("lock poisoned: MemoryVfs delete");
        files.remove(path);
        Ok(())
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), StorageError> {
        let mut files = self.files.write().expect("lock poisoned: MemoryVfs rename");
        if let Some(file) = files.remove(from) {
            files.insert(to.to_string(), file);
            Ok(())
        } else {
            Err(StorageError::Backend(format!("File not found: {}", from)))
        }
    }

    fn temp_file(&self) -> Result<Self::File, StorageError> {
        let path = format!(
            "/tmp/jouledb_{}.tmp",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        self.open(&path, true)
    }

    fn name(&self) -> &str {
        "memory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_vfs_open_create() {
        let vfs = StdVfs::new();
        let tempdir = std::env::temp_dir();
        let path = tempdir.join("test_vfs_open.db");
        let path_str = path.to_str().unwrap();

        // Clean up if exists
        let _ = std::fs::remove_file(path_str);

        // Create file
        let mut file = vfs.open(path_str, true).unwrap();
        file.write_all(b"hello vfs").unwrap();
        file.sync_all().unwrap();

        assert!(vfs.exists(path_str).unwrap());

        // Clean up
        vfs.delete(path_str).unwrap();
        assert!(!vfs.exists(path_str).unwrap());
    }

    #[test]
    fn test_memory_vfs() {
        let vfs = MemoryVfs::new();

        // Create and write
        let mut file = vfs.open("test.db", true).unwrap();
        file.write_all(b"memory data").unwrap();

        // Read back
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut buf = vec![0u8; 11];
        file.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"memory data");

        // Check exists
        assert!(vfs.exists("test.db").unwrap());

        // Delete
        vfs.delete("test.db").unwrap();
        assert!(!vfs.exists("test.db").unwrap());
    }

    #[test]
    fn test_memory_file_seek() {
        let mut file = MemoryFile::new();
        file.write_all(b"0123456789").unwrap();

        // Seek to middle
        file.seek(SeekFrom::Start(5)).unwrap();
        let mut buf = [0u8; 3];
        file.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"567");

        // Seek from end
        file.seek(SeekFrom::End(-3)).unwrap();
        file.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"789");
    }
}
