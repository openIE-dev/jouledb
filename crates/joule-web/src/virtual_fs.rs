//! Virtual filesystem — in-memory FS with directories, files, symlinks,
//! permissions, metadata, glob matching, and recursive operations.

use std::collections::HashMap;

// ── Permissions ─────────────────────────────────────────────────────────────

/// Unix-style permissions (rwx) for owner, group, other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permissions {
    pub owner_read: bool,
    pub owner_write: bool,
    pub owner_exec: bool,
    pub group_read: bool,
    pub group_write: bool,
    pub group_exec: bool,
    pub other_read: bool,
    pub other_write: bool,
    pub other_exec: bool,
}

impl Permissions {
    /// Create permissions from an octal mode like 0o755.
    pub fn from_mode(mode: u16) -> Self {
        Self {
            owner_read: mode & 0o400 != 0,
            owner_write: mode & 0o200 != 0,
            owner_exec: mode & 0o100 != 0,
            group_read: mode & 0o040 != 0,
            group_write: mode & 0o020 != 0,
            group_exec: mode & 0o010 != 0,
            other_read: mode & 0o004 != 0,
            other_write: mode & 0o002 != 0,
            other_exec: mode & 0o001 != 0,
        }
    }

    /// Convert to numeric octal mode.
    pub fn to_mode(self) -> u16 {
        let mut m = 0u16;
        if self.owner_read { m |= 0o400; }
        if self.owner_write { m |= 0o200; }
        if self.owner_exec { m |= 0o100; }
        if self.group_read { m |= 0o040; }
        if self.group_write { m |= 0o020; }
        if self.group_exec { m |= 0o010; }
        if self.other_read { m |= 0o004; }
        if self.other_write { m |= 0o002; }
        if self.other_exec { m |= 0o001; }
        m
    }

    /// Default file permissions (0o644).
    pub fn default_file() -> Self {
        Self::from_mode(0o644)
    }

    /// Default directory permissions (0o755).
    pub fn default_dir() -> Self {
        Self::from_mode(0o755)
    }
}

// ── File Metadata ───────────────────────────────────────────────────────────

/// Metadata about a filesystem entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMetadata {
    pub size: u64,
    pub created_at: u64,
    pub modified_at: u64,
    pub accessed_at: u64,
    pub permissions: Permissions,
}

// ── Node Kind ───────────────────────────────────────────────────────────────

/// Type of filesystem node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    File { content: Vec<u8> },
    Directory { children: Vec<String> },
    Symlink { target: String },
}

// ── INode ───────────────────────────────────────────────────────────────────

/// An inode in the virtual filesystem.
#[derive(Debug, Clone)]
pub struct INode {
    pub kind: NodeKind,
    pub metadata: FileMetadata,
}

// ── Error ───────────────────────────────────────────────────────────────────

/// Virtual filesystem errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VfsError {
    NotFound(String),
    AlreadyExists(String),
    NotADirectory(String),
    NotAFile(String),
    IsADirectory(String),
    PermissionDenied(String),
    SymlinkLoop(String),
    InvalidPath(String),
    DirectoryNotEmpty(String),
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VfsError::NotFound(p) => write!(f, "not found: {p}"),
            VfsError::AlreadyExists(p) => write!(f, "already exists: {p}"),
            VfsError::NotADirectory(p) => write!(f, "not a directory: {p}"),
            VfsError::NotAFile(p) => write!(f, "not a file: {p}"),
            VfsError::IsADirectory(p) => write!(f, "is a directory: {p}"),
            VfsError::PermissionDenied(p) => write!(f, "permission denied: {p}"),
            VfsError::SymlinkLoop(p) => write!(f, "symlink loop: {p}"),
            VfsError::InvalidPath(p) => write!(f, "invalid path: {p}"),
            VfsError::DirectoryNotEmpty(p) => write!(f, "directory not empty: {p}"),
        }
    }
}

// ── VirtualFs ───────────────────────────────────────────────────────────────

/// In-memory virtual filesystem.
#[derive(Debug)]
pub struct VirtualFs {
    nodes: HashMap<String, INode>,
    clock: u64,
    max_symlink_depth: usize,
}

impl VirtualFs {
    /// Create a new VFS with an empty root directory.
    pub fn new() -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(
            "/".to_string(),
            INode {
                kind: NodeKind::Directory {
                    children: Vec::new(),
                },
                metadata: FileMetadata {
                    size: 0,
                    created_at: 0,
                    modified_at: 0,
                    accessed_at: 0,
                    permissions: Permissions::default_dir(),
                },
            },
        );
        Self {
            nodes,
            clock: 1,
            max_symlink_depth: 40,
        }
    }

    /// Advance the internal timestamp.
    fn tick(&mut self) -> u64 {
        let t = self.clock;
        self.clock += 1;
        t
    }

    /// Normalize a path: resolve `.` and `..`, ensure leading `/`, collapse slashes.
    pub fn normalize_path(path: &str) -> Result<String, VfsError> {
        if path.is_empty() {
            return Err(VfsError::InvalidPath("empty path".into()));
        }
        if !path.starts_with('/') {
            return Err(VfsError::InvalidPath(format!("relative path: {path}")));
        }
        let mut parts: Vec<&str> = Vec::new();
        for seg in path.split('/') {
            match seg {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                s => parts.push(s),
            }
        }
        if parts.is_empty() {
            Ok("/".to_string())
        } else {
            Ok(format!("/{}", parts.join("/")))
        }
    }

    /// Get the parent path and child name from an absolute path.
    fn parent_and_name(path: &str) -> Result<(String, String), VfsError> {
        if path == "/" {
            return Err(VfsError::InvalidPath("root has no parent".into()));
        }
        let norm = Self::normalize_path(path)?;
        if let Some(pos) = norm.rfind('/') {
            let parent = if pos == 0 {
                "/".to_string()
            } else {
                norm[..pos].to_string()
            };
            let name = norm[pos + 1..].to_string();
            if name.is_empty() {
                return Err(VfsError::InvalidPath(format!("trailing slash: {path}")));
            }
            Ok((parent, name))
        } else {
            Err(VfsError::InvalidPath(format!("bad path: {path}")))
        }
    }

    /// Resolve symlinks to get the actual path.
    pub fn resolve_path(&self, path: &str) -> Result<String, VfsError> {
        self.resolve_path_depth(path, 0)
    }

    fn resolve_path_depth(&self, path: &str, depth: usize) -> Result<String, VfsError> {
        if depth > self.max_symlink_depth {
            return Err(VfsError::SymlinkLoop(path.to_string()));
        }
        let norm = Self::normalize_path(path)?;
        // Walk path segments resolving symlinks along the way
        let segments: Vec<&str> = norm.split('/').filter(|s| !s.is_empty()).collect();
        let mut current = "/".to_string();

        for seg in &segments {
            let candidate = if current == "/" {
                format!("/{seg}")
            } else {
                format!("{current}/{seg}")
            };

            if let Some(node) = self.nodes.get(&candidate) {
                if let NodeKind::Symlink { target } = &node.kind {
                    let resolved_target = if target.starts_with('/') {
                        target.clone()
                    } else {
                        format!("{current}/{target}")
                    };
                    let resolved = self.resolve_path_depth(&resolved_target, depth + 1)?;
                    current = resolved;
                } else {
                    current = candidate;
                }
            } else {
                current = candidate;
            }
        }
        Ok(current)
    }

    /// Create a directory (and all parent directories if they don't exist).
    pub fn mkdir_p(&mut self, path: &str) -> Result<(), VfsError> {
        let norm = Self::normalize_path(path)?;
        if norm == "/" {
            return Ok(());
        }
        let segments: Vec<&str> = norm.split('/').filter(|s| !s.is_empty()).collect();
        let mut current = "/".to_string();

        for seg in segments {
            let child_path = if current == "/" {
                format!("/{seg}")
            } else {
                format!("{current}/{seg}")
            };

            if self.nodes.contains_key(&child_path) {
                let node = &self.nodes[&child_path];
                if !matches!(node.kind, NodeKind::Directory { .. }) {
                    return Err(VfsError::NotADirectory(child_path));
                }
            } else {
                let now = self.tick();
                // Add child to parent's directory listing
                if let Some(parent_node) = self.nodes.get_mut(&current) {
                    if let NodeKind::Directory { children } = &mut parent_node.kind {
                        children.push(seg.to_string());
                    } else {
                        return Err(VfsError::NotADirectory(current.clone()));
                    }
                }
                self.nodes.insert(
                    child_path.clone(),
                    INode {
                        kind: NodeKind::Directory {
                            children: Vec::new(),
                        },
                        metadata: FileMetadata {
                            size: 0,
                            created_at: now,
                            modified_at: now,
                            accessed_at: now,
                            permissions: Permissions::default_dir(),
                        },
                    },
                );
            }
            current = child_path;
        }
        Ok(())
    }

    /// Create a single directory (parent must exist).
    pub fn mkdir(&mut self, path: &str) -> Result<(), VfsError> {
        let norm = Self::normalize_path(path)?;
        if self.nodes.contains_key(&norm) {
            return Err(VfsError::AlreadyExists(norm));
        }
        let (parent, name) = Self::parent_and_name(&norm)?;
        if !self.nodes.contains_key(&parent) {
            return Err(VfsError::NotFound(parent));
        }
        let now = self.tick();
        if let Some(p) = self.nodes.get_mut(&parent) {
            if let NodeKind::Directory { children } = &mut p.kind {
                children.push(name);
            } else {
                return Err(VfsError::NotADirectory(parent));
            }
        }
        self.nodes.insert(
            norm,
            INode {
                kind: NodeKind::Directory {
                    children: Vec::new(),
                },
                metadata: FileMetadata {
                    size: 0,
                    created_at: now,
                    modified_at: now,
                    accessed_at: now,
                    permissions: Permissions::default_dir(),
                },
            },
        );
        Ok(())
    }

    /// Create a file (parent directory must exist). Overwrites if already exists.
    pub fn create_file(&mut self, path: &str, content: &[u8]) -> Result<(), VfsError> {
        let norm = Self::normalize_path(path)?;
        let (parent, name) = Self::parent_and_name(&norm)?;

        let parent_resolved = self.resolve_path(&parent)?;
        if !self.nodes.contains_key(&parent_resolved) {
            return Err(VfsError::NotFound(parent_resolved));
        }

        // Check parent is a directory
        let parent_is_dir = {
            let pnode = &self.nodes[&parent_resolved];
            matches!(pnode.kind, NodeKind::Directory { .. })
        };
        if !parent_is_dir {
            return Err(VfsError::NotADirectory(parent_resolved));
        }

        let now = self.tick();
        let already = self.nodes.contains_key(&norm);
        if !already {
            if let Some(p) = self.nodes.get_mut(&parent_resolved) {
                if let NodeKind::Directory { children } = &mut p.kind {
                    children.push(name);
                }
            }
        }
        self.nodes.insert(
            norm,
            INode {
                kind: NodeKind::File {
                    content: content.to_vec(),
                },
                metadata: FileMetadata {
                    size: content.len() as u64,
                    created_at: now,
                    modified_at: now,
                    accessed_at: now,
                    permissions: Permissions::default_file(),
                },
            },
        );
        Ok(())
    }

    /// Read a file's content.
    pub fn read_file(&mut self, path: &str) -> Result<Vec<u8>, VfsError> {
        let norm = Self::normalize_path(path)?;
        let resolved = self.resolve_path(&norm)?;
        let now = self.tick();
        if let Some(node) = self.nodes.get_mut(&resolved) {
            node.metadata.accessed_at = now;
            if let NodeKind::File { content } = &node.kind {
                Ok(content.clone())
            } else {
                Err(VfsError::NotAFile(resolved))
            }
        } else {
            Err(VfsError::NotFound(resolved))
        }
    }

    /// Write (overwrite) a file's content. File must exist.
    pub fn write_file(&mut self, path: &str, data: &[u8]) -> Result<(), VfsError> {
        let norm = Self::normalize_path(path)?;
        let resolved = self.resolve_path(&norm)?;
        let now = self.tick();
        if let Some(node) = self.nodes.get_mut(&resolved) {
            match &mut node.kind {
                NodeKind::File { content } => {
                    *content = data.to_vec();
                    node.metadata.size = data.len() as u64;
                    node.metadata.modified_at = now;
                    Ok(())
                }
                NodeKind::Directory { .. } => Err(VfsError::IsADirectory(resolved)),
                NodeKind::Symlink { .. } => Err(VfsError::NotAFile(resolved)),
            }
        } else {
            Err(VfsError::NotFound(resolved))
        }
    }

    /// Delete a file. Fails if path is a non-empty directory.
    pub fn delete(&mut self, path: &str) -> Result<(), VfsError> {
        let norm = Self::normalize_path(path)?;
        if norm == "/" {
            return Err(VfsError::PermissionDenied("cannot delete root".into()));
        }
        if !self.nodes.contains_key(&norm) {
            return Err(VfsError::NotFound(norm));
        }
        // Check if directory is non-empty
        if let NodeKind::Directory { children } = &self.nodes[&norm].kind {
            if !children.is_empty() {
                return Err(VfsError::DirectoryNotEmpty(norm));
            }
        }
        let (parent, name) = Self::parent_and_name(&norm)?;
        self.nodes.remove(&norm);
        // Remove from parent's children
        if let Some(p) = self.nodes.get_mut(&parent) {
            if let NodeKind::Directory { children } = &mut p.kind {
                children.retain(|c| c != &name);
            }
        }
        Ok(())
    }

    /// Delete a directory and everything in it.
    pub fn delete_recursive(&mut self, path: &str) -> Result<(), VfsError> {
        let norm = Self::normalize_path(path)?;
        if norm == "/" {
            return Err(VfsError::PermissionDenied(
                "cannot delete root".into(),
            ));
        }
        if !self.nodes.contains_key(&norm) {
            return Err(VfsError::NotFound(norm));
        }

        // Collect all paths under this entry
        let prefix = if norm == "/" {
            "/".to_string()
        } else {
            format!("{norm}/")
        };
        let to_remove: Vec<String> = self
            .nodes
            .keys()
            .filter(|k| **k == norm || k.starts_with(&prefix))
            .cloned()
            .collect();

        for key in &to_remove {
            self.nodes.remove(key);
        }

        // Remove from parent
        let (parent, name) = Self::parent_and_name(&norm)?;
        if let Some(p) = self.nodes.get_mut(&parent) {
            if let NodeKind::Directory { children } = &mut p.kind {
                children.retain(|c| c != &name);
            }
        }
        Ok(())
    }

    /// Create a symlink at `link_path` pointing to `target`.
    pub fn symlink(&mut self, link_path: &str, target: &str) -> Result<(), VfsError> {
        let norm = Self::normalize_path(link_path)?;
        if self.nodes.contains_key(&norm) {
            return Err(VfsError::AlreadyExists(norm));
        }
        let (parent, name) = Self::parent_and_name(&norm)?;
        if !self.nodes.contains_key(&parent) {
            return Err(VfsError::NotFound(parent));
        }
        let parent_is_dir = matches!(self.nodes[&parent].kind, NodeKind::Directory { .. });
        if !parent_is_dir {
            return Err(VfsError::NotADirectory(parent));
        }

        let now = self.tick();
        if let Some(p) = self.nodes.get_mut(&parent) {
            if let NodeKind::Directory { children } = &mut p.kind {
                children.push(name);
            }
        }
        self.nodes.insert(
            norm,
            INode {
                kind: NodeKind::Symlink {
                    target: target.to_string(),
                },
                metadata: FileMetadata {
                    size: target.len() as u64,
                    created_at: now,
                    modified_at: now,
                    accessed_at: now,
                    permissions: Permissions::from_mode(0o777),
                },
            },
        );
        Ok(())
    }

    /// Get metadata for a path (resolves symlinks).
    pub fn metadata(&self, path: &str) -> Result<FileMetadata, VfsError> {
        let norm = Self::normalize_path(path)?;
        let resolved = self.resolve_path(&norm)?;
        self.nodes
            .get(&resolved)
            .map(|n| n.metadata.clone())
            .ok_or_else(|| VfsError::NotFound(resolved))
    }

    /// Get metadata without resolving symlinks (lstat).
    pub fn lstat(&self, path: &str) -> Result<FileMetadata, VfsError> {
        let norm = Self::normalize_path(path)?;
        self.nodes
            .get(&norm)
            .map(|n| n.metadata.clone())
            .ok_or_else(|| VfsError::NotFound(norm))
    }

    /// List directory contents (names only).
    pub fn list_dir(&self, path: &str) -> Result<Vec<String>, VfsError> {
        let norm = Self::normalize_path(path)?;
        let resolved = self.resolve_path(&norm)?;
        match self.nodes.get(&resolved) {
            Some(INode {
                kind: NodeKind::Directory { children },
                ..
            }) => Ok(children.clone()),
            Some(_) => Err(VfsError::NotADirectory(resolved)),
            None => Err(VfsError::NotFound(resolved)),
        }
    }

    /// Check if a path exists.
    pub fn exists(&self, path: &str) -> bool {
        if let Ok(norm) = Self::normalize_path(path) {
            self.nodes.contains_key(&norm)
        } else {
            false
        }
    }

    /// Check if path is a file.
    pub fn is_file(&self, path: &str) -> bool {
        if let Ok(norm) = Self::normalize_path(path) {
            matches!(
                self.nodes.get(&norm),
                Some(INode {
                    kind: NodeKind::File { .. },
                    ..
                })
            )
        } else {
            false
        }
    }

    /// Check if path is a directory.
    pub fn is_dir(&self, path: &str) -> bool {
        if let Ok(norm) = Self::normalize_path(path) {
            matches!(
                self.nodes.get(&norm),
                Some(INode {
                    kind: NodeKind::Directory { .. },
                    ..
                })
            )
        } else {
            false
        }
    }

    /// Check if path is a symlink.
    pub fn is_symlink(&self, path: &str) -> bool {
        if let Ok(norm) = Self::normalize_path(path) {
            matches!(
                self.nodes.get(&norm),
                Some(INode {
                    kind: NodeKind::Symlink { .. },
                    ..
                })
            )
        } else {
            false
        }
    }

    /// Set permissions on a path.
    pub fn chmod(&mut self, path: &str, mode: u16) -> Result<(), VfsError> {
        let norm = Self::normalize_path(path)?;
        let resolved = self.resolve_path(&norm)?;
        if let Some(node) = self.nodes.get_mut(&resolved) {
            node.metadata.permissions = Permissions::from_mode(mode);
            Ok(())
        } else {
            Err(VfsError::NotFound(resolved))
        }
    }

    /// Glob-match files. Supports `*` (any segment chars) and `**` (any depth).
    pub fn glob(&self, pattern: &str) -> Result<Vec<String>, VfsError> {
        let norm_pattern = Self::normalize_path(pattern)?;
        let mut results = Vec::new();
        let mut paths: Vec<String> = self.nodes.keys().cloned().collect();
        paths.sort();

        for path in &paths {
            if glob_match_path(&norm_pattern, path) {
                results.push(path.clone());
            }
        }
        Ok(results)
    }

    /// Copy a file from src to dst.
    pub fn copy(&mut self, src: &str, dst: &str) -> Result<(), VfsError> {
        let content = self.read_file(src)?;
        self.create_file(dst, &content)
    }

    /// Rename/move a file or directory.
    pub fn rename(&mut self, old_path: &str, new_path: &str) -> Result<(), VfsError> {
        let old_norm = Self::normalize_path(old_path)?;
        let new_norm = Self::normalize_path(new_path)?;

        if !self.nodes.contains_key(&old_norm) {
            return Err(VfsError::NotFound(old_norm));
        }
        if self.nodes.contains_key(&new_norm) {
            return Err(VfsError::AlreadyExists(new_norm));
        }

        let (old_parent, old_name) = Self::parent_and_name(&old_norm)?;
        let (new_parent, new_name) = Self::parent_and_name(&new_norm)?;

        if !self.nodes.contains_key(&new_parent) {
            return Err(VfsError::NotFound(new_parent));
        }

        // Remove from old parent
        if let Some(p) = self.nodes.get_mut(&old_parent) {
            if let NodeKind::Directory { children } = &mut p.kind {
                children.retain(|c| c != &old_name);
            }
        }
        // Add to new parent
        if let Some(p) = self.nodes.get_mut(&new_parent) {
            if let NodeKind::Directory { children } = &mut p.kind {
                children.push(new_name);
            }
        }

        let node = self.nodes.remove(&old_norm).unwrap();
        self.nodes.insert(new_norm.clone(), node);

        // Move children if directory
        let prefix = format!("{old_norm}/");
        let children_to_move: Vec<(String, String)> = self
            .nodes
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| {
                let suffix = &k[old_norm.len()..];
                (k.clone(), format!("{new_norm}{suffix}"))
            })
            .collect();

        for (old_key, new_key) in children_to_move {
            if let Some(n) = self.nodes.remove(&old_key) {
                self.nodes.insert(new_key, n);
            }
        }
        Ok(())
    }

    /// Count total entries.
    pub fn entry_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total file bytes across all files.
    pub fn total_size(&self) -> u64 {
        self.nodes
            .values()
            .map(|n| {
                if let NodeKind::File { content } = &n.kind {
                    content.len() as u64
                } else {
                    0
                }
            })
            .sum()
    }

    /// Recursively list all paths under a directory.
    pub fn walk(&self, path: &str) -> Result<Vec<String>, VfsError> {
        let norm = Self::normalize_path(path)?;
        if !self.nodes.contains_key(&norm) {
            return Err(VfsError::NotFound(norm));
        }
        let prefix = if norm == "/" {
            "/".to_string()
        } else {
            format!("{norm}/")
        };
        let mut results: Vec<String> = self
            .nodes
            .keys()
            .filter(|k| **k != norm && (k.starts_with(&prefix) || (norm == "/" && k.as_str() != "/")))
            .cloned()
            .collect();
        results.sort();
        Ok(results)
    }
}

// ── Glob Matcher ────────────────────────────────────────────────────────────

/// Match a glob pattern against a path.
/// Supports `*` (match any non-`/` chars) and `**` (match any depth).
fn glob_match_path(pattern: &str, path: &str) -> bool {
    let pat_segs: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    let path_segs: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    glob_match_segments(&pat_segs, &path_segs)
}

fn glob_match_segments(pattern: &[&str], path: &[&str]) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }
    if pattern[0] == "**" {
        // `**` can match zero or more path segments
        let rest_pattern = &pattern[1..];
        for i in 0..=path.len() {
            if glob_match_segments(rest_pattern, &path[i..]) {
                return true;
            }
        }
        return false;
    }
    if path.is_empty() {
        return false;
    }
    if glob_match_segment(pattern[0], path[0]) {
        glob_match_segments(&pattern[1..], &path[1..])
    } else {
        false
    }
}

fn glob_match_segment(pattern: &str, segment: &str) -> bool {
    glob_match_chars(pattern.as_bytes(), segment.as_bytes())
}

fn glob_match_chars(pat: &[u8], text: &[u8]) -> bool {
    if pat.is_empty() {
        return text.is_empty();
    }
    if pat[0] == b'*' {
        // `*` matches zero or more characters
        for i in 0..=text.len() {
            if glob_match_chars(&pat[1..], &text[i..]) {
                return true;
            }
        }
        return false;
    }
    if text.is_empty() {
        return false;
    }
    if pat[0] == b'?' || pat[0] == text[0] {
        glob_match_chars(&pat[1..], &text[1..])
    } else {
        false
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_read_file() {
        let mut fs = VirtualFs::new();
        fs.create_file("/hello.txt", b"world").unwrap();
        let data = fs.read_file("/hello.txt").unwrap();
        assert_eq!(data, b"world");
    }

    #[test]
    fn test_mkdir_and_list() {
        let mut fs = VirtualFs::new();
        fs.mkdir("/docs").unwrap();
        let entries = fs.list_dir("/").unwrap();
        assert!(entries.contains(&"docs".to_string()));
    }

    #[test]
    fn test_mkdir_p() {
        let mut fs = VirtualFs::new();
        fs.mkdir_p("/a/b/c").unwrap();
        assert!(fs.is_dir("/a"));
        assert!(fs.is_dir("/a/b"));
        assert!(fs.is_dir("/a/b/c"));
    }

    #[test]
    fn test_nested_file() {
        let mut fs = VirtualFs::new();
        fs.mkdir_p("/usr/local/bin").unwrap();
        fs.create_file("/usr/local/bin/app", b"binary data").unwrap();
        assert_eq!(fs.read_file("/usr/local/bin/app").unwrap(), b"binary data");
    }

    #[test]
    fn test_delete_file() {
        let mut fs = VirtualFs::new();
        fs.create_file("/tmp.txt", b"temp").unwrap();
        fs.delete("/tmp.txt").unwrap();
        assert!(!fs.exists("/tmp.txt"));
    }

    #[test]
    fn test_delete_nonempty_dir() {
        let mut fs = VirtualFs::new();
        fs.mkdir("/data").unwrap();
        fs.create_file("/data/file.txt", b"x").unwrap();
        let err = fs.delete("/data").unwrap_err();
        assert_eq!(err, VfsError::DirectoryNotEmpty("/data".into()));
    }

    #[test]
    fn test_delete_recursive() {
        let mut fs = VirtualFs::new();
        fs.mkdir_p("/a/b/c").unwrap();
        fs.create_file("/a/b/c/f.txt", b"hi").unwrap();
        fs.create_file("/a/b/g.txt", b"ho").unwrap();
        fs.delete_recursive("/a").unwrap();
        assert!(!fs.exists("/a"));
        assert!(!fs.exists("/a/b/c/f.txt"));
    }

    #[test]
    fn test_write_file() {
        let mut fs = VirtualFs::new();
        fs.create_file("/f.txt", b"v1").unwrap();
        fs.write_file("/f.txt", b"v2").unwrap();
        assert_eq!(fs.read_file("/f.txt").unwrap(), b"v2");
    }

    #[test]
    fn test_symlink_basic() {
        let mut fs = VirtualFs::new();
        fs.create_file("/real.txt", b"real content").unwrap();
        fs.symlink("/link.txt", "/real.txt").unwrap();
        assert!(fs.is_symlink("/link.txt"));
        let data = fs.read_file("/link.txt").unwrap();
        assert_eq!(data, b"real content");
    }

    #[test]
    fn test_symlink_loop_detection() {
        let mut fs = VirtualFs::new();
        // We can't easily make a direct loop with the resolve logic,
        // but we can try resolving a long chain.
        fs.symlink("/a", "/b").unwrap();
        fs.symlink("/b", "/a").unwrap();
        let result = fs.resolve_path("/a");
        assert!(matches!(result, Err(VfsError::SymlinkLoop(_))));
    }

    #[test]
    fn test_permissions() {
        let mut fs = VirtualFs::new();
        fs.create_file("/f.txt", b"data").unwrap();
        let meta = fs.metadata("/f.txt").unwrap();
        assert_eq!(meta.permissions.to_mode(), 0o644);

        fs.chmod("/f.txt", 0o755).unwrap();
        let meta2 = fs.metadata("/f.txt").unwrap();
        assert_eq!(meta2.permissions.to_mode(), 0o755);
    }

    #[test]
    fn test_metadata_timestamps() {
        let mut fs = VirtualFs::new();
        fs.create_file("/f.txt", b"hello").unwrap();
        let meta = fs.metadata("/f.txt").unwrap();
        assert_eq!(meta.size, 5);
        assert!(meta.created_at > 0);
    }

    #[test]
    fn test_glob_star() {
        let mut fs = VirtualFs::new();
        fs.create_file("/a.rs", b"").unwrap();
        fs.create_file("/b.rs", b"").unwrap();
        fs.create_file("/c.txt", b"").unwrap();
        let mut matches = fs.glob("/*.rs").unwrap();
        matches.sort();
        assert_eq!(matches, vec!["/a.rs", "/b.rs"]);
    }

    #[test]
    fn test_glob_doublestar() {
        let mut fs = VirtualFs::new();
        fs.mkdir_p("/src/lib").unwrap();
        fs.create_file("/src/main.rs", b"").unwrap();
        fs.create_file("/src/lib/util.rs", b"").unwrap();
        let mut matches = fs.glob("/**/*.rs").unwrap();
        matches.sort();
        assert_eq!(matches, vec!["/src/lib/util.rs", "/src/main.rs"]);
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(VirtualFs::normalize_path("/a/b/../c").unwrap(), "/a/c");
        assert_eq!(VirtualFs::normalize_path("/a/./b").unwrap(), "/a/b");
        assert_eq!(VirtualFs::normalize_path("///a///b///").unwrap(), "/a/b");
        assert_eq!(VirtualFs::normalize_path("/").unwrap(), "/");
    }

    #[test]
    fn test_copy_file() {
        let mut fs = VirtualFs::new();
        fs.create_file("/src.txt", b"copy me").unwrap();
        fs.copy("/src.txt", "/dst.txt").unwrap();
        assert_eq!(fs.read_file("/dst.txt").unwrap(), b"copy me");
    }

    #[test]
    fn test_rename_file() {
        let mut fs = VirtualFs::new();
        fs.create_file("/old.txt", b"data").unwrap();
        fs.rename("/old.txt", "/new.txt").unwrap();
        assert!(!fs.exists("/old.txt"));
        assert_eq!(fs.read_file("/new.txt").unwrap(), b"data");
    }

    #[test]
    fn test_walk() {
        let mut fs = VirtualFs::new();
        fs.mkdir_p("/a/b").unwrap();
        fs.create_file("/a/b/x.txt", b"").unwrap();
        fs.create_file("/a/y.txt", b"").unwrap();
        let mut walked = fs.walk("/a").unwrap();
        walked.sort();
        assert_eq!(walked, vec!["/a/b", "/a/b/x.txt", "/a/y.txt"]);
    }

    #[test]
    fn test_total_size() {
        let mut fs = VirtualFs::new();
        fs.create_file("/a.bin", b"12345").unwrap();
        fs.create_file("/b.bin", b"abc").unwrap();
        assert_eq!(fs.total_size(), 8);
    }

    #[test]
    fn test_not_found_errors() {
        let mut fs = VirtualFs::new();
        assert!(fs.read_file("/nope").is_err());
        assert!(fs.write_file("/nope", b"x").is_err());
        assert!(fs.delete("/nope").is_err());
    }

    #[test]
    fn test_permissions_round_trip() {
        for mode in [0o000, 0o644, 0o755, 0o777, 0o400, 0o070, 0o007] {
            let p = Permissions::from_mode(mode);
            assert_eq!(p.to_mode(), mode, "mode {mode:o} round-trip failed");
        }
    }

    #[test]
    fn test_is_predicates() {
        let mut fs = VirtualFs::new();
        fs.mkdir("/dir").unwrap();
        fs.create_file("/file.txt", b"").unwrap();
        fs.symlink("/link", "/file.txt").unwrap();

        assert!(fs.is_dir("/dir"));
        assert!(!fs.is_file("/dir"));
        assert!(fs.is_file("/file.txt"));
        assert!(!fs.is_dir("/file.txt"));
        assert!(fs.is_symlink("/link"));
        assert!(!fs.exists("/nonexistent"));
    }

    #[test]
    fn test_entry_count() {
        let mut fs = VirtualFs::new();
        assert_eq!(fs.entry_count(), 1); // root
        fs.mkdir("/a").unwrap();
        assert_eq!(fs.entry_count(), 2);
        fs.create_file("/a/f.txt", b"").unwrap();
        assert_eq!(fs.entry_count(), 3);
    }
}
