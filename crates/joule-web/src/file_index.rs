//! File index structures — inode model, directory entries, path-to-inode
//! resolution, hard links, file metadata (size/permissions/timestamps),
//! directory listing, free inode tracking.

use std::collections::HashMap;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by file index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileIndexError {
    /// Inode not found.
    InodeNotFound(u64),
    /// Path not found.
    PathNotFound(String),
    /// Entry already exists.
    AlreadyExists(String),
    /// Not a directory.
    NotADirectory(u64),
    /// Not a file.
    NotAFile(u64),
    /// Directory is not empty.
    DirectoryNotEmpty(u64),
    /// No free inodes available.
    NoFreeInodes,
    /// Invalid path.
    InvalidPath(String),
    /// Hard link target must be a file.
    HardLinkToDirectory,
}

impl std::fmt::Display for FileIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InodeNotFound(id) => write!(f, "inode {id} not found"),
            Self::PathNotFound(p) => write!(f, "path not found: {p}"),
            Self::AlreadyExists(p) => write!(f, "already exists: {p}"),
            Self::NotADirectory(id) => write!(f, "inode {id} is not a directory"),
            Self::NotAFile(id) => write!(f, "inode {id} is not a file"),
            Self::DirectoryNotEmpty(id) => write!(f, "directory inode {id} is not empty"),
            Self::NoFreeInodes => write!(f, "no free inodes available"),
            Self::InvalidPath(p) => write!(f, "invalid path: {p}"),
            Self::HardLinkToDirectory => write!(f, "hard links to directories are not allowed"),
        }
    }
}

impl std::error::Error for FileIndexError {}

// ── File Type ────────────────────────────────────────────────────────────────

/// Type of a filesystem entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Regular file.
    RegularFile,
    /// Directory.
    Directory,
}

// ── Permissions ──────────────────────────────────────────────────────────────

/// Simple Unix-style permission bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permissions {
    /// Owner read.
    pub owner_read: bool,
    /// Owner write.
    pub owner_write: bool,
    /// Owner execute.
    pub owner_exec: bool,
    /// Group read.
    pub group_read: bool,
    /// Group write.
    pub group_write: bool,
    /// Group execute.
    pub group_exec: bool,
    /// Other read.
    pub other_read: bool,
    /// Other write.
    pub other_write: bool,
    /// Other execute.
    pub other_exec: bool,
}

impl Permissions {
    /// Default file permissions (rw-r--r--).
    pub fn default_file() -> Self {
        Self {
            owner_read: true,
            owner_write: true,
            owner_exec: false,
            group_read: true,
            group_write: false,
            group_exec: false,
            other_read: true,
            other_write: false,
            other_exec: false,
        }
    }

    /// Default directory permissions (rwxr-xr-x).
    pub fn default_directory() -> Self {
        Self {
            owner_read: true,
            owner_write: true,
            owner_exec: true,
            group_read: true,
            group_write: false,
            group_exec: true,
            other_read: true,
            other_write: false,
            other_exec: true,
        }
    }

    /// Encode as a 9-bit octal-like value.
    pub fn to_bits(&self) -> u16 {
        let mut bits = 0u16;
        if self.owner_read { bits |= 0o400; }
        if self.owner_write { bits |= 0o200; }
        if self.owner_exec { bits |= 0o100; }
        if self.group_read { bits |= 0o040; }
        if self.group_write { bits |= 0o020; }
        if self.group_exec { bits |= 0o010; }
        if self.other_read { bits |= 0o004; }
        if self.other_write { bits |= 0o002; }
        if self.other_exec { bits |= 0o001; }
        bits
    }

    /// Decode from bits.
    pub fn from_bits(bits: u16) -> Self {
        Self {
            owner_read: bits & 0o400 != 0,
            owner_write: bits & 0o200 != 0,
            owner_exec: bits & 0o100 != 0,
            group_read: bits & 0o040 != 0,
            group_write: bits & 0o020 != 0,
            group_exec: bits & 0o010 != 0,
            other_read: bits & 0o004 != 0,
            other_write: bits & 0o002 != 0,
            other_exec: bits & 0o001 != 0,
        }
    }
}

// ── Timestamps ───────────────────────────────────────────────────────────────

/// File timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timestamps {
    /// Creation time (epoch seconds).
    pub created: u64,
    /// Last modification time.
    pub modified: u64,
    /// Last access time.
    pub accessed: u64,
}

impl Timestamps {
    /// Create timestamps with all fields set to the given epoch time.
    pub fn new(now: u64) -> Self {
        Self {
            created: now,
            modified: now,
            accessed: now,
        }
    }
}

// ── Inode ────────────────────────────────────────────────────────────────────

/// An inode representing file metadata.
#[derive(Debug, Clone)]
pub struct Inode {
    /// Inode number.
    pub ino: u64,
    /// File type.
    pub file_type: FileType,
    /// File size in bytes (0 for directories).
    pub size: u64,
    /// Permissions.
    pub permissions: Permissions,
    /// Timestamps.
    pub timestamps: Timestamps,
    /// Number of hard links to this inode.
    pub link_count: u32,
    /// Block addresses or extent pointers (simplified as a list of block nums).
    pub blocks: Vec<u64>,
}

impl Inode {
    fn new_file(ino: u64, now: u64) -> Self {
        Self {
            ino,
            file_type: FileType::RegularFile,
            size: 0,
            permissions: Permissions::default_file(),
            timestamps: Timestamps::new(now),
            link_count: 1,
            blocks: Vec::new(),
        }
    }

    fn new_directory(ino: u64, now: u64) -> Self {
        Self {
            ino,
            file_type: FileType::Directory,
            size: 0,
            permissions: Permissions::default_directory(),
            timestamps: Timestamps::new(now),
            link_count: 1,
            blocks: Vec::new(),
        }
    }

    /// Whether this inode is a directory.
    pub fn is_directory(&self) -> bool {
        self.file_type == FileType::Directory
    }

    /// Whether this inode is a regular file.
    pub fn is_file(&self) -> bool {
        self.file_type == FileType::RegularFile
    }
}

// ── Directory Entry ──────────────────────────────────────────────────────────

/// A directory entry mapping a name to an inode.
#[derive(Debug, Clone)]
pub struct DirEntry {
    /// Entry name (filename or subdirectory name).
    pub name: String,
    /// Inode number this entry points to.
    pub ino: u64,
    /// File type (cached from inode for fast listing).
    pub file_type: FileType,
}

// ── Directory ────────────────────────────────────────────────────────────────

/// Directory contents: entries stored in a map for O(1) lookup.
#[derive(Debug, Clone, Default)]
struct Directory {
    entries: HashMap<String, DirEntry>,
}

impl Directory {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn add(&mut self, name: String, ino: u64, file_type: FileType) -> Result<(), FileIndexError> {
        if self.entries.contains_key(&name) {
            return Err(FileIndexError::AlreadyExists(name));
        }
        self.entries.insert(
            name.clone(),
            DirEntry {
                name,
                ino,
                file_type,
            },
        );
        Ok(())
    }

    fn remove(&mut self, name: &str) -> Result<DirEntry, FileIndexError> {
        self.entries
            .remove(name)
            .ok_or_else(|| FileIndexError::PathNotFound(name.to_string()))
    }

    fn get(&self, name: &str) -> Option<&DirEntry> {
        self.entries.get(name)
    }

    fn list(&self) -> Vec<DirEntry> {
        let mut entries: Vec<DirEntry> = self.entries.values().cloned().collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ── File Index ───────────────────────────────────────────────────────────────

/// In-memory file index managing inodes, directories, and path resolution.
#[derive(Debug)]
pub struct FileIndex {
    /// Inode table: ino -> Inode.
    inodes: HashMap<u64, Inode>,
    /// Directory contents: directory ino -> Directory.
    directories: HashMap<u64, Directory>,
    /// Next inode number.
    next_ino: u64,
    /// Free inode numbers for reuse.
    free_inos: Vec<u64>,
    /// Maximum inode count (0 = unlimited).
    max_inodes: u64,
    /// Current epoch time (for timestamps).
    clock: u64,
    /// Root inode number.
    root_ino: u64,
}

impl FileIndex {
    /// Create a new file index with a root directory.
    pub fn new(max_inodes: u64) -> Self {
        let root_ino = 1;
        let mut inodes = HashMap::new();
        let mut directories = HashMap::new();

        inodes.insert(root_ino, Inode::new_directory(root_ino, 0));
        directories.insert(root_ino, Directory::new());

        Self {
            inodes,
            directories,
            next_ino: 2,
            free_inos: Vec::new(),
            max_inodes,
            clock: 0,
            root_ino,
        }
    }

    /// Advance the clock (for timestamps).
    pub fn set_clock(&mut self, now: u64) {
        self.clock = now;
    }

    /// Root inode number.
    pub fn root_ino(&self) -> u64 {
        self.root_ino
    }

    fn alloc_ino(&mut self) -> Result<u64, FileIndexError> {
        if let Some(ino) = self.free_inos.pop() {
            return Ok(ino);
        }
        if self.max_inodes > 0 && self.next_ino > self.max_inodes {
            return Err(FileIndexError::NoFreeInodes);
        }
        let ino = self.next_ino;
        self.next_ino += 1;
        Ok(ino)
    }

    fn free_ino(&mut self, ino: u64) {
        self.free_inos.push(ino);
    }

    /// Create a file in the given directory.
    pub fn create_file(&mut self, parent_ino: u64, name: &str) -> Result<u64, FileIndexError> {
        self.validate_parent(parent_ino)?;
        let ino = self.alloc_ino()?;
        let inode = Inode::new_file(ino, self.clock);
        self.inodes.insert(ino, inode);

        let dir = self
            .directories
            .get_mut(&parent_ino)
            .ok_or(FileIndexError::NotADirectory(parent_ino))?;
        dir.add(name.to_string(), ino, FileType::RegularFile)?;
        Ok(ino)
    }

    /// Create a subdirectory.
    pub fn create_dir(&mut self, parent_ino: u64, name: &str) -> Result<u64, FileIndexError> {
        self.validate_parent(parent_ino)?;
        let ino = self.alloc_ino()?;
        let inode = Inode::new_directory(ino, self.clock);
        self.inodes.insert(ino, inode);
        self.directories.insert(ino, Directory::new());

        let dir = self
            .directories
            .get_mut(&parent_ino)
            .ok_or(FileIndexError::NotADirectory(parent_ino))?;
        dir.add(name.to_string(), ino, FileType::Directory)?;
        Ok(ino)
    }

    fn validate_parent(&self, parent_ino: u64) -> Result<(), FileIndexError> {
        let parent = self
            .inodes
            .get(&parent_ino)
            .ok_or(FileIndexError::InodeNotFound(parent_ino))?;
        if !parent.is_directory() {
            return Err(FileIndexError::NotADirectory(parent_ino));
        }
        Ok(())
    }

    /// Create a hard link to an existing file.
    pub fn hard_link(
        &mut self,
        parent_ino: u64,
        name: &str,
        target_ino: u64,
    ) -> Result<(), FileIndexError> {
        self.validate_parent(parent_ino)?;
        let target = self
            .inodes
            .get(&target_ino)
            .ok_or(FileIndexError::InodeNotFound(target_ino))?;
        if target.is_directory() {
            return Err(FileIndexError::HardLinkToDirectory);
        }
        let file_type = target.file_type;

        let dir = self
            .directories
            .get_mut(&parent_ino)
            .ok_or(FileIndexError::NotADirectory(parent_ino))?;
        dir.add(name.to_string(), target_ino, file_type)?;

        let inode = self
            .inodes
            .get_mut(&target_ino)
            .ok_or(FileIndexError::InodeNotFound(target_ino))?;
        inode.link_count += 1;
        Ok(())
    }

    /// Remove a directory entry.  If the inode link count drops to zero, the
    /// inode is freed.
    pub fn unlink(&mut self, parent_ino: u64, name: &str) -> Result<(), FileIndexError> {
        let entry = {
            let dir = self
                .directories
                .get_mut(&parent_ino)
                .ok_or(FileIndexError::NotADirectory(parent_ino))?;
            dir.remove(name)?
        };

        let ino = entry.ino;
        // If it was a directory, check it is empty.
        if entry.file_type == FileType::Directory {
            if let Some(child_dir) = self.directories.get(&ino) {
                if !child_dir.is_empty() {
                    // Re-add the entry.
                    let dir = self.directories.get_mut(&parent_ino).unwrap();
                    dir.add(entry.name, ino, entry.file_type).ok();
                    return Err(FileIndexError::DirectoryNotEmpty(ino));
                }
            }
        }

        let remove_inode = {
            let inode = self
                .inodes
                .get_mut(&ino)
                .ok_or(FileIndexError::InodeNotFound(ino))?;
            inode.link_count = inode.link_count.saturating_sub(1);
            inode.link_count == 0
        };

        if remove_inode {
            self.inodes.remove(&ino);
            self.directories.remove(&ino);
            self.free_ino(ino);
        }

        Ok(())
    }

    /// Resolve a path (e.g. "/foo/bar/baz") to an inode number.
    pub fn resolve_path(&self, path: &str) -> Result<u64, FileIndexError> {
        if path.is_empty() || !path.starts_with('/') {
            return Err(FileIndexError::InvalidPath(path.to_string()));
        }
        if path == "/" {
            return Ok(self.root_ino);
        }

        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let mut current_ino = self.root_ino;

        for part in parts {
            let dir = self
                .directories
                .get(&current_ino)
                .ok_or(FileIndexError::NotADirectory(current_ino))?;
            let entry = dir
                .get(part)
                .ok_or_else(|| FileIndexError::PathNotFound(path.to_string()))?;
            current_ino = entry.ino;
        }

        Ok(current_ino)
    }

    /// List entries in a directory.
    pub fn list_dir(&self, dir_ino: u64) -> Result<Vec<DirEntry>, FileIndexError> {
        let inode = self
            .inodes
            .get(&dir_ino)
            .ok_or(FileIndexError::InodeNotFound(dir_ino))?;
        if !inode.is_directory() {
            return Err(FileIndexError::NotADirectory(dir_ino));
        }
        let dir = self
            .directories
            .get(&dir_ino)
            .ok_or(FileIndexError::NotADirectory(dir_ino))?;
        Ok(dir.list())
    }

    /// Get inode metadata.
    pub fn get_inode(&self, ino: u64) -> Result<&Inode, FileIndexError> {
        self.inodes
            .get(&ino)
            .ok_or(FileIndexError::InodeNotFound(ino))
    }

    /// Update file size.
    pub fn set_size(&mut self, ino: u64, size: u64) -> Result<(), FileIndexError> {
        let inode = self
            .inodes
            .get_mut(&ino)
            .ok_or(FileIndexError::InodeNotFound(ino))?;
        if !inode.is_file() {
            return Err(FileIndexError::NotAFile(ino));
        }
        inode.size = size;
        inode.timestamps.modified = self.clock;
        Ok(())
    }

    /// Update permissions.
    pub fn set_permissions(&mut self, ino: u64, perms: Permissions) -> Result<(), FileIndexError> {
        let inode = self
            .inodes
            .get_mut(&ino)
            .ok_or(FileIndexError::InodeNotFound(ino))?;
        inode.permissions = perms;
        Ok(())
    }

    /// Total number of inodes in use.
    pub fn inode_count(&self) -> usize {
        self.inodes.len()
    }

    /// Number of free inode slots available for reuse.
    pub fn free_inode_count(&self) -> usize {
        self.free_inos.len()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> FileIndex {
        FileIndex::new(1000)
    }

    #[test]
    fn root_exists() {
        let idx = make_index();
        let root = idx.get_inode(idx.root_ino()).unwrap();
        assert!(root.is_directory());
    }

    #[test]
    fn create_file_and_resolve() {
        let mut idx = make_index();
        let root = idx.root_ino();
        let ino = idx.create_file(root, "hello.txt").unwrap();
        assert_eq!(idx.resolve_path("/hello.txt").unwrap(), ino);
    }

    #[test]
    fn create_dir_and_resolve() {
        let mut idx = make_index();
        let root = idx.root_ino();
        let dir_ino = idx.create_dir(root, "docs").unwrap();
        assert_eq!(idx.resolve_path("/docs").unwrap(), dir_ino);
    }

    #[test]
    fn nested_path_resolution() {
        let mut idx = make_index();
        let root = idx.root_ino();
        let dir = idx.create_dir(root, "a").unwrap();
        let subdir = idx.create_dir(dir, "b").unwrap();
        let file = idx.create_file(subdir, "c.txt").unwrap();
        assert_eq!(idx.resolve_path("/a/b/c.txt").unwrap(), file);
    }

    #[test]
    fn path_not_found() {
        let idx = make_index();
        let result = idx.resolve_path("/nonexistent");
        assert!(matches!(result, Err(FileIndexError::PathNotFound(_))));
    }

    #[test]
    fn resolve_root() {
        let idx = make_index();
        assert_eq!(idx.resolve_path("/").unwrap(), idx.root_ino());
    }

    #[test]
    fn invalid_path() {
        let idx = make_index();
        assert!(idx.resolve_path("no_leading_slash").is_err());
        assert!(idx.resolve_path("").is_err());
    }

    #[test]
    fn list_dir() {
        let mut idx = make_index();
        let root = idx.root_ino();
        idx.create_file(root, "b.txt").unwrap();
        idx.create_file(root, "a.txt").unwrap();
        idx.create_dir(root, "c_dir").unwrap();
        let entries = idx.list_dir(root).unwrap();
        assert_eq!(entries.len(), 3);
        // Sorted by name.
        assert_eq!(entries[0].name, "a.txt");
        assert_eq!(entries[1].name, "b.txt");
        assert_eq!(entries[2].name, "c_dir");
    }

    #[test]
    fn hard_link() {
        let mut idx = make_index();
        let root = idx.root_ino();
        let file = idx.create_file(root, "original.txt").unwrap();
        idx.hard_link(root, "link.txt", file).unwrap();
        let linked_ino = idx.resolve_path("/link.txt").unwrap();
        assert_eq!(linked_ino, file);
        let inode = idx.get_inode(file).unwrap();
        assert_eq!(inode.link_count, 2);
    }

    #[test]
    fn hard_link_to_directory_fails() {
        let mut idx = make_index();
        let root = idx.root_ino();
        let dir = idx.create_dir(root, "mydir").unwrap();
        let result = idx.hard_link(root, "link", dir);
        assert_eq!(result, Err(FileIndexError::HardLinkToDirectory));
    }

    #[test]
    fn unlink_file() {
        let mut idx = make_index();
        let root = idx.root_ino();
        let file = idx.create_file(root, "temp.txt").unwrap();
        idx.unlink(root, "temp.txt").unwrap();
        assert!(idx.get_inode(file).is_err());
        assert!(idx.resolve_path("/temp.txt").is_err());
    }

    #[test]
    fn unlink_hardlinked_file() {
        let mut idx = make_index();
        let root = idx.root_ino();
        let file = idx.create_file(root, "a.txt").unwrap();
        idx.hard_link(root, "b.txt", file).unwrap();
        idx.unlink(root, "a.txt").unwrap();
        // Inode still exists via b.txt.
        let inode = idx.get_inode(file).unwrap();
        assert_eq!(inode.link_count, 1);
    }

    #[test]
    fn unlink_nonempty_dir() {
        let mut idx = make_index();
        let root = idx.root_ino();
        let dir = idx.create_dir(root, "mydir").unwrap();
        idx.create_file(dir, "file.txt").unwrap();
        let result = idx.unlink(root, "mydir");
        assert_eq!(result, Err(FileIndexError::DirectoryNotEmpty(dir)));
    }

    #[test]
    fn duplicate_name_error() {
        let mut idx = make_index();
        let root = idx.root_ino();
        idx.create_file(root, "dup.txt").unwrap();
        let result = idx.create_file(root, "dup.txt");
        assert!(matches!(result, Err(FileIndexError::AlreadyExists(_))));
    }

    #[test]
    fn set_size_and_permissions() {
        let mut idx = make_index();
        idx.set_clock(100);
        let root = idx.root_ino();
        let file = idx.create_file(root, "f.txt").unwrap();
        idx.set_size(file, 1024).unwrap();
        let inode = idx.get_inode(file).unwrap();
        assert_eq!(inode.size, 1024);
        assert_eq!(inode.timestamps.modified, 100);

        let perms = Permissions::from_bits(0o755);
        idx.set_permissions(file, perms).unwrap();
        let inode = idx.get_inode(file).unwrap();
        assert!(inode.permissions.owner_exec);
    }

    #[test]
    fn permissions_roundtrip() {
        let perms = Permissions::default_file();
        let bits = perms.to_bits();
        let decoded = Permissions::from_bits(bits);
        assert_eq!(perms, decoded);
    }

    #[test]
    fn inode_count_and_free() {
        let mut idx = make_index();
        let root = idx.root_ino();
        assert_eq!(idx.inode_count(), 1); // root only
        let f = idx.create_file(root, "f.txt").unwrap();
        assert_eq!(idx.inode_count(), 2);
        idx.unlink(root, "f.txt").unwrap();
        assert_eq!(idx.inode_count(), 1);
        assert_eq!(idx.free_inode_count(), 1);
        // Reuse freed inode.
        let f2 = idx.create_file(root, "g.txt").unwrap();
        assert_eq!(f2, f); // Reused inode number.
    }

    #[test]
    fn error_display() {
        let e = FileIndexError::NoFreeInodes;
        assert_eq!(e.to_string(), "no free inodes available");
        let e = FileIndexError::InodeNotFound(42);
        assert!(e.to_string().contains("42"));
    }
}
