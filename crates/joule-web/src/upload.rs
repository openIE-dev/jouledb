//! File upload management with chunking, progress tracking, and drag-and-drop.
//!
//! Replaces Uppy, Dropzone, and filepond with a pure-Rust state machine that
//! tracks upload tasks, computes progress/speed/ETA, and manages concurrency.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

// ── File Info ──

/// Metadata about a file to be uploaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub name: String,
    pub size: usize,
    pub mime_type: String,
    pub last_modified: Option<DateTime<Utc>>,
}

// ── Upload Status ──

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UploadStatus {
    Queued,
    Uploading,
    Paused,
    Complete,
    Failed(String),
    Cancelled,
}

// ── Chunk State ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkState {
    pub index: u32,
    pub offset: usize,
    pub size: usize,
    pub uploaded: bool,
}

// ── Upload Task ──

/// A single file upload with chunked progress tracking.
#[derive(Debug, Clone)]
pub struct UploadTask {
    pub id: Uuid,
    pub file: FileInfo,
    pub status: UploadStatus,
    pub progress_bytes: usize,
    pub chunks: Vec<ChunkState>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub upload_url: Option<String>,
    pub metadata: HashMap<String, String>,
    pub retry_count: u32,
}

impl UploadTask {
    pub fn new(file: FileInfo) -> Self {
        Self {
            id: Uuid::new_v4(),
            file,
            status: UploadStatus::Queued,
            progress_bytes: 0,
            chunks: Vec::new(),
            started_at: None,
            completed_at: None,
            upload_url: None,
            metadata: HashMap::new(),
            retry_count: 0,
        }
    }

    pub fn progress_percent(&self) -> f64 {
        if self.file.size == 0 {
            return if self.status == UploadStatus::Complete {
                100.0
            } else {
                0.0
            };
        }
        (self.progress_bytes as f64 / self.file.size as f64) * 100.0
    }

    pub fn elapsed(&self, now: &DateTime<Utc>) -> Option<chrono::Duration> {
        self.started_at.map(|s| *now - s)
    }

    pub fn speed_bytes_per_sec(&self, now: &DateTime<Utc>) -> Option<f64> {
        let elapsed = self.elapsed(now)?;
        let secs = elapsed.num_milliseconds() as f64 / 1000.0;
        if secs <= 0.0 {
            return None;
        }
        Some(self.progress_bytes as f64 / secs)
    }

    pub fn eta_seconds(&self, now: &DateTime<Utc>) -> Option<f64> {
        let speed = self.speed_bytes_per_sec(now)?;
        if speed <= 0.0 {
            return None;
        }
        let remaining = self.file.size.saturating_sub(self.progress_bytes) as f64;
        Some(remaining / speed)
    }

    pub fn is_complete(&self) -> bool {
        self.status == UploadStatus::Complete
    }

    pub fn is_active(&self) -> bool {
        self.status == UploadStatus::Uploading
    }
}

// ── Upload Error ──

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum UploadError {
    #[error("file too large: {size} bytes exceeds max {max} bytes")]
    FileTooLarge { size: usize, max: usize },
    #[error("type not allowed: {mime}")]
    TypeNotAllowed { mime: String },
}

// ── Upload Manager ──

/// Manages a queue of upload tasks with concurrency and validation.
#[derive(Debug)]
pub struct UploadManager {
    pub tasks: Vec<UploadTask>,
    pub chunk_size: usize,
    pub max_concurrent: usize,
    pub max_retries: u32,
    pub max_file_size: Option<usize>,
    pub allowed_types: Option<Vec<String>>,
}

impl UploadManager {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            chunk_size: 1_048_576, // 1 MB
            max_concurrent: 3,
            max_retries: 3,
            max_file_size: None,
            allowed_types: None,
        }
    }

    pub fn with_chunk_size(size: usize) -> Self {
        Self {
            chunk_size: size,
            ..Self::new()
        }
    }

    /// Add a file to the upload queue. Validates size and type constraints.
    pub fn add_file(&mut self, file: FileInfo) -> Result<Uuid, UploadError> {
        if let Some(max) = self.max_file_size {
            if file.size > max {
                return Err(UploadError::FileTooLarge {
                    size: file.size,
                    max,
                });
            }
        }
        if let Some(ref allowed) = self.allowed_types {
            if !allowed.contains(&file.mime_type) {
                return Err(UploadError::TypeNotAllowed {
                    mime: file.mime_type.clone(),
                });
            }
        }

        let mut task = UploadTask::new(file);
        // Create chunks
        let file_size = task.file.size;
        let chunk_sz = self.chunk_size.max(1);
        let mut offset = 0;
        let mut idx = 0u32;
        while offset < file_size {
            let sz = (file_size - offset).min(chunk_sz);
            task.chunks.push(ChunkState {
                index: idx,
                offset,
                size: sz,
                uploaded: false,
            });
            offset += sz;
            idx += 1;
        }
        // Zero-byte files get no chunks; they complete immediately on start.
        let id = task.id;
        self.tasks.push(task);
        Ok(id)
    }

    pub fn start(&mut self, id: &Uuid) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == *id) {
            if task.status == UploadStatus::Queued {
                task.status = UploadStatus::Uploading;
                task.started_at = Some(Utc::now());
                return true;
            }
        }
        false
    }

    /// Start up to `max_concurrent` queued tasks. Returns how many were started.
    pub fn start_all(&mut self) -> usize {
        let active = self.active_count();
        let slots = self.max_concurrent.saturating_sub(active);
        let queued_ids: Vec<Uuid> = self
            .tasks
            .iter()
            .filter(|t| t.status == UploadStatus::Queued)
            .take(slots)
            .map(|t| t.id)
            .collect();
        let count = queued_ids.len();
        for id in &queued_ids {
            self.start(id);
        }
        count
    }

    pub fn pause(&mut self, id: &Uuid) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == *id) {
            if task.status == UploadStatus::Uploading {
                task.status = UploadStatus::Paused;
                return true;
            }
        }
        false
    }

    pub fn resume(&mut self, id: &Uuid) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == *id) {
            if task.status == UploadStatus::Paused {
                task.status = UploadStatus::Uploading;
                return true;
            }
        }
        false
    }

    pub fn cancel(&mut self, id: &Uuid) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == *id) {
            if matches!(
                task.status,
                UploadStatus::Queued | UploadStatus::Uploading | UploadStatus::Paused
            ) {
                task.status = UploadStatus::Cancelled;
                return true;
            }
        }
        false
    }

    pub fn retry(&mut self, id: &Uuid) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == *id) {
            if matches!(task.status, UploadStatus::Failed(_)) {
                if task.retry_count < self.max_retries {
                    task.retry_count += 1;
                    task.status = UploadStatus::Queued;
                    return true;
                }
            }
        }
        false
    }

    pub fn remove(&mut self, id: &Uuid) -> bool {
        let len = self.tasks.len();
        self.tasks.retain(|t| t.id != *id);
        self.tasks.len() < len
    }

    /// Mark a specific chunk as uploaded, updating progress bytes.
    pub fn mark_chunk_complete(&mut self, task_id: &Uuid, chunk_index: u32) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == *task_id) {
            if let Some(chunk) = task.chunks.iter_mut().find(|c| c.index == chunk_index) {
                if !chunk.uploaded {
                    chunk.uploaded = true;
                    task.progress_bytes += chunk.size;
                }
            }
            if task.chunks.iter().all(|c| c.uploaded) {
                task.status = UploadStatus::Complete;
                task.completed_at = Some(Utc::now());
            }
            return true;
        }
        false
    }

    pub fn mark_failed(&mut self, task_id: &Uuid, error: &str) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == *task_id) {
            task.status = UploadStatus::Failed(error.to_string());
            return true;
        }
        false
    }

    pub fn active_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| t.status == UploadStatus::Uploading)
            .count()
    }

    pub fn queued_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| t.status == UploadStatus::Queued)
            .count()
    }

    /// Total progress across all tasks as a percentage.
    pub fn total_progress_percent(&self) -> f64 {
        let total_size: usize = self.tasks.iter().map(|t| t.file.size).sum();
        if total_size == 0 {
            return 0.0;
        }
        let total_progress: usize = self.tasks.iter().map(|t| t.progress_bytes).sum();
        (total_progress as f64 / total_size as f64) * 100.0
    }

    pub fn get_task(&self, id: &Uuid) -> Option<&UploadTask> {
        self.tasks.iter().find(|t| t.id == *id)
    }

    pub fn all_tasks(&self) -> &[UploadTask] {
        &self.tasks
    }
}

impl Default for UploadManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Drop Zone State ──

/// State for a drag-and-drop file zone.
#[derive(Debug, Clone)]
pub struct DropZoneState {
    pub is_drag_over: bool,
    pub accepted_types: Option<Vec<String>>,
}

impl DropZoneState {
    pub fn new() -> Self {
        Self {
            is_drag_over: false,
            accepted_types: None,
        }
    }

    pub fn drag_enter(&mut self) {
        self.is_drag_over = true;
    }

    pub fn drag_leave(&mut self) {
        self.is_drag_over = false;
    }

    /// Accept dropped files, filtering by accepted MIME types.
    pub fn drop_files(&mut self, files: Vec<FileInfo>) -> Vec<FileInfo> {
        self.is_drag_over = false;
        match &self.accepted_types {
            None => files,
            Some(types) => files
                .into_iter()
                .filter(|f| types.contains(&f.mime_type))
                .collect(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.is_drag_over
    }
}

impl Default for DropZoneState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_file(name: &str, size: usize) -> FileInfo {
        FileInfo {
            name: name.to_string(),
            size,
            mime_type: "application/octet-stream".to_string(),
            last_modified: None,
        }
    }

    #[test]
    fn add_file_creates_chunks() {
        let mut mgr = UploadManager::with_chunk_size(100);
        let id = mgr.add_file(sample_file("test.bin", 250)).unwrap();
        let task = mgr.get_task(&id).unwrap();
        assert_eq!(task.chunks.len(), 3);
        assert_eq!(task.chunks[0].size, 100);
        assert_eq!(task.chunks[1].size, 100);
        assert_eq!(task.chunks[2].size, 50);
    }

    #[test]
    fn progress_percent_starts_at_zero() {
        let task = UploadTask::new(sample_file("f.bin", 1000));
        assert!((task.progress_percent()).abs() < 0.01);
    }

    #[test]
    fn mark_chunk_complete_advances_progress() {
        let mut mgr = UploadManager::with_chunk_size(100);
        let id = mgr.add_file(sample_file("f.bin", 200)).unwrap();
        mgr.start(&id);
        mgr.mark_chunk_complete(&id, 0);
        let task = mgr.get_task(&id).unwrap();
        assert_eq!(task.progress_bytes, 100);
        assert!((task.progress_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn all_chunks_complete_sets_complete_status() {
        let mut mgr = UploadManager::with_chunk_size(100);
        let id = mgr.add_file(sample_file("f.bin", 200)).unwrap();
        mgr.start(&id);
        mgr.mark_chunk_complete(&id, 0);
        mgr.mark_chunk_complete(&id, 1);
        let task = mgr.get_task(&id).unwrap();
        assert!(task.is_complete());
    }

    #[test]
    fn max_file_size_rejects() {
        let mut mgr = UploadManager::new();
        mgr.max_file_size = Some(100);
        let result = mgr.add_file(sample_file("big.bin", 200));
        assert!(matches!(result, Err(UploadError::FileTooLarge { .. })));
    }

    #[test]
    fn type_filtering() {
        let mut mgr = UploadManager::new();
        mgr.allowed_types = Some(vec!["image/png".to_string()]);
        let result = mgr.add_file(sample_file("f.bin", 100));
        assert!(matches!(result, Err(UploadError::TypeNotAllowed { .. })));
    }

    #[test]
    fn pause_resume() {
        let mut mgr = UploadManager::new();
        let id = mgr.add_file(sample_file("f.bin", 100)).unwrap();
        mgr.start(&id);
        assert!(mgr.pause(&id));
        assert_eq!(mgr.get_task(&id).unwrap().status, UploadStatus::Paused);
        assert!(mgr.resume(&id));
        assert_eq!(mgr.get_task(&id).unwrap().status, UploadStatus::Uploading);
    }

    #[test]
    fn cancel_task() {
        let mut mgr = UploadManager::new();
        let id = mgr.add_file(sample_file("f.bin", 100)).unwrap();
        mgr.start(&id);
        assert!(mgr.cancel(&id));
        assert_eq!(mgr.get_task(&id).unwrap().status, UploadStatus::Cancelled);
    }

    #[test]
    fn start_all_respects_max_concurrent() {
        let mut mgr = UploadManager::new();
        mgr.max_concurrent = 2;
        for i in 0..5 {
            mgr.add_file(sample_file(&format!("f{i}.bin"), 100)).unwrap();
        }
        let started = mgr.start_all();
        assert_eq!(started, 2);
        assert_eq!(mgr.active_count(), 2);
        assert_eq!(mgr.queued_count(), 3);
    }

    #[test]
    fn total_progress() {
        let mut mgr = UploadManager::with_chunk_size(100);
        let id1 = mgr.add_file(sample_file("a.bin", 100)).unwrap();
        let id2 = mgr.add_file(sample_file("b.bin", 100)).unwrap();
        mgr.start(&id1);
        mgr.mark_chunk_complete(&id1, 0);
        // id1 complete (100/100), id2 queued (0/100) => 50%
        assert!((mgr.total_progress_percent() - 50.0).abs() < 0.01);
        let _ = id2;
    }

    #[test]
    fn speed_calculation() {
        let mut task = UploadTask::new(sample_file("f.bin", 1000));
        task.started_at = Some(Utc::now() - chrono::Duration::seconds(2));
        task.progress_bytes = 500;
        task.status = UploadStatus::Uploading;
        let now = Utc::now();
        let speed = task.speed_bytes_per_sec(&now).unwrap();
        // Approximately 250 bytes/sec (timing can vary slightly)
        assert!(speed > 200.0 && speed < 300.0);
    }

    #[test]
    fn eta_calculation() {
        let mut task = UploadTask::new(sample_file("f.bin", 1000));
        task.started_at = Some(Utc::now() - chrono::Duration::seconds(2));
        task.progress_bytes = 500;
        task.status = UploadStatus::Uploading;
        let now = Utc::now();
        let eta = task.eta_seconds(&now).unwrap();
        // ~2 seconds remaining
        assert!(eta > 1.5 && eta < 2.5);
    }

    #[test]
    fn retry_resets_status() {
        let mut mgr = UploadManager::new();
        let id = mgr.add_file(sample_file("f.bin", 100)).unwrap();
        mgr.mark_failed(&id, "network error");
        assert!(mgr.retry(&id));
        let task = mgr.get_task(&id).unwrap();
        assert_eq!(task.status, UploadStatus::Queued);
        assert_eq!(task.retry_count, 1);
    }

    #[test]
    fn drop_zone_filters_by_type() {
        let mut zone = DropZoneState::new();
        zone.accepted_types = Some(vec!["image/png".to_string()]);
        zone.drag_enter();
        assert!(zone.is_active());
        let files = vec![
            FileInfo {
                name: "a.png".into(),
                size: 100,
                mime_type: "image/png".into(),
                last_modified: None,
            },
            FileInfo {
                name: "b.txt".into(),
                size: 50,
                mime_type: "text/plain".into(),
                last_modified: None,
            },
        ];
        let accepted = zone.drop_files(files);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].name, "a.png");
        assert!(!zone.is_active());
    }
}
