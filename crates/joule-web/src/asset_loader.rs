//! Asset loading and management system.
//!
//! Provides asset types (Mesh, Texture, Sound, Material, Animation) with
//! async-simulation loading (request → pending → ready), lightweight handle-based
//! references, reference counting, dependency tracking, batch loading with
//! progress, error handling, and an asset manifest.

use std::collections::HashMap;
use std::fmt;

// ── Asset type / data ──────────────────────────────────────────

/// Asset type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetKind {
    Mesh,
    Texture,
    Sound,
    Material,
    Animation,
}

impl fmt::Display for AssetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetKind::Mesh => write!(f, "mesh"),
            AssetKind::Texture => write!(f, "texture"),
            AssetKind::Sound => write!(f, "sound"),
            AssetKind::Material => write!(f, "material"),
            AssetKind::Animation => write!(f, "animation"),
        }
    }
}

/// Actual asset data payload.
#[derive(Debug, Clone, PartialEq)]
pub enum AssetData {
    Mesh {
        vertex_count: u32,
        index_count: u32,
        data: Vec<u8>,
    },
    Texture {
        width: u32,
        height: u32,
        channels: u8,
        data: Vec<u8>,
    },
    Sound {
        sample_rate: u32,
        channels: u8,
        duration_ms: u64,
        data: Vec<u8>,
    },
    Material {
        shader: String,
        /// Texture asset IDs this material depends on.
        texture_ids: Vec<AssetId>,
    },
    Animation {
        bone_count: u32,
        frame_count: u32,
        duration_ms: u64,
        data: Vec<u8>,
    },
}

impl AssetData {
    /// Estimated memory size in bytes.
    pub fn estimated_bytes(&self) -> usize {
        match self {
            AssetData::Mesh { data, .. } => data.len() + 8,
            AssetData::Texture { data, .. } => data.len() + 9,
            AssetData::Sound { data, .. } => data.len() + 13,
            AssetData::Material { shader, texture_ids, .. } => {
                shader.len() + texture_ids.len() * 8
            }
            AssetData::Animation { data, .. } => data.len() + 16,
        }
    }

    pub fn kind(&self) -> AssetKind {
        match self {
            AssetData::Mesh { .. } => AssetKind::Mesh,
            AssetData::Texture { .. } => AssetKind::Texture,
            AssetData::Sound { .. } => AssetKind::Sound,
            AssetData::Material { .. } => AssetKind::Material,
            AssetData::Animation { .. } => AssetKind::Animation,
        }
    }
}

// ── Asset ID / Handle ──────────────────────────────────────────

/// Unique identifier for an asset.
pub type AssetId = u64;

/// Lightweight handle referencing a loaded asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetHandle {
    pub id: AssetId,
}

impl AssetHandle {
    pub fn new(id: AssetId) -> Self {
        Self { id }
    }
}

impl fmt::Display for AssetHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssetHandle({})", self.id)
    }
}

// ── Load state ─────────────────────────────────────────────────

/// State of an asset in the loading pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadState {
    /// Not yet requested.
    None,
    /// Request submitted, waiting for data.
    Pending,
    /// Data loaded and ready to use.
    Ready,
    /// Loading failed.
    Failed,
}

// ── Asset error ────────────────────────────────────────────────

/// Errors from asset operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetError {
    NotFound(AssetId),
    AlreadyLoaded(AssetId),
    NotInManifest(String),
    DependencyMissing { asset: AssetId, dependency: AssetId },
    Corrupt(String),
    StillReferenced(AssetId),
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetError::NotFound(id) => write!(f, "asset {id} not found"),
            AssetError::AlreadyLoaded(id) => write!(f, "asset {id} already loaded"),
            AssetError::NotInManifest(p) => write!(f, "not in manifest: {p}"),
            AssetError::DependencyMissing { asset, dependency } => {
                write!(f, "asset {asset} missing dependency {dependency}")
            }
            AssetError::Corrupt(s) => write!(f, "corrupt asset: {s}"),
            AssetError::StillReferenced(id) => write!(f, "asset {id} still referenced"),
        }
    }
}

// ── Manifest entry ─────────────────────────────────────────────

/// Metadata about a known asset in the manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestEntry {
    pub id: AssetId,
    pub path: String,
    pub kind: AssetKind,
    pub size_bytes: u64,
    pub dependencies: Vec<AssetId>,
}

// ── Internal asset slot ────────────────────────────────────────

#[derive(Debug, Clone)]
struct AssetSlot {
    id: AssetId,
    state: LoadState,
    data: Option<AssetData>,
    ref_count: u32,
    dependencies: Vec<AssetId>,
}

// ── Batch load tracker ─────────────────────────────────────────

/// Tracks progress of a batch load operation.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchProgress {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
}

impl BatchProgress {
    /// Progress ratio in [0, 1].
    pub fn ratio(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            (self.completed + self.failed) as f64 / self.total as f64
        }
    }

    pub fn is_done(&self) -> bool {
        self.completed + self.failed >= self.total
    }
}

// ── Asset loader ───────────────────────────────────────────────

/// Asset loading and management system with reference counting and dependencies.
pub struct AssetLoader {
    next_id: AssetId,
    slots: HashMap<AssetId, AssetSlot>,
    manifest: HashMap<String, ManifestEntry>,
    path_to_id: HashMap<String, AssetId>,
    batch: Option<BatchProgress>,
}

impl fmt::Debug for AssetLoader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssetLoader")
            .field("assets", &self.slots.len())
            .field("manifest", &self.manifest.len())
            .finish()
    }
}

impl AssetLoader {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            slots: HashMap::new(),
            manifest: HashMap::new(),
            path_to_id: HashMap::new(),
            batch: None,
        }
    }

    fn alloc_id(&mut self) -> AssetId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ── Manifest ───────────────────────────────────────────────

    /// Register an asset in the manifest (declaring it exists).
    pub fn register_manifest(
        &mut self,
        path: &str,
        kind: AssetKind,
        size_bytes: u64,
        dependencies: Vec<AssetId>,
    ) -> AssetId {
        if let Some(entry) = self.manifest.get(path) {
            return entry.id;
        }
        let id = self.alloc_id();
        let entry = ManifestEntry {
            id,
            path: path.to_string(),
            kind,
            size_bytes,
            dependencies: dependencies.clone(),
        };
        self.manifest.insert(path.to_string(), entry);
        self.path_to_id.insert(path.to_string(), id);
        id
    }

    /// Look up an asset ID by path.
    pub fn id_for_path(&self, path: &str) -> Option<AssetId> {
        self.path_to_id.get(path).copied()
    }

    /// Manifest entry by path.
    pub fn manifest_entry(&self, path: &str) -> Option<&ManifestEntry> {
        self.manifest.get(path)
    }

    /// Number of assets in the manifest.
    pub fn manifest_count(&self) -> usize {
        self.manifest.len()
    }

    // ── Loading ────────────────────────────────────────────────

    /// Request loading an asset by ID (transitions to Pending).
    pub fn request_load(&mut self, id: AssetId) -> Result<AssetHandle, AssetError> {
        if let Some(slot) = self.slots.get(&id) {
            if slot.state == LoadState::Ready || slot.state == LoadState::Pending {
                return Ok(AssetHandle::new(id));
            }
        }

        // Find manifest entry for dependencies.
        let deps: Vec<AssetId> = self
            .manifest
            .values()
            .find(|e| e.id == id)
            .map(|e| e.dependencies.clone())
            .unwrap_or_default();

        self.slots.insert(
            id,
            AssetSlot {
                id,
                state: LoadState::Pending,
                data: None,
                ref_count: 1,
                dependencies: deps,
            },
        );
        Ok(AssetHandle::new(id))
    }

    /// Complete loading by providing asset data (transitions Pending → Ready).
    pub fn finish_load(&mut self, id: AssetId, data: AssetData) -> Result<(), AssetError> {
        let slot = self.slots.get_mut(&id).ok_or(AssetError::NotFound(id))?;
        if slot.state != LoadState::Pending {
            return Err(AssetError::AlreadyLoaded(id));
        }
        slot.data = Some(data);
        slot.state = LoadState::Ready;

        // Update batch progress.
        if let Some(ref mut bp) = self.batch {
            bp.completed += 1;
        }
        Ok(())
    }

    /// Mark loading as failed.
    pub fn fail_load(&mut self, id: AssetId, reason: &str) -> Result<(), AssetError> {
        let slot = self.slots.get_mut(&id).ok_or(AssetError::NotFound(id))?;
        slot.state = LoadState::Failed;
        slot.data = None;
        let _ = reason; // kept for logging, not stored to keep slot small
        if let Some(ref mut bp) = self.batch {
            bp.failed += 1;
        }
        Ok(())
    }

    // ── Reference counting ─────────────────────────────────────

    /// Increment reference count for an asset.
    pub fn add_ref(&mut self, id: AssetId) -> Result<u32, AssetError> {
        let slot = self.slots.get_mut(&id).ok_or(AssetError::NotFound(id))?;
        slot.ref_count += 1;
        Ok(slot.ref_count)
    }

    /// Decrement reference count. If it reaches 0, unload the asset.
    pub fn release(&mut self, id: AssetId) -> Result<u32, AssetError> {
        let rc = {
            let slot = self.slots.get_mut(&id).ok_or(AssetError::NotFound(id))?;
            slot.ref_count = slot.ref_count.saturating_sub(1);
            slot.ref_count
        };
        if rc == 0 {
            self.slots.remove(&id);
        }
        Ok(rc)
    }

    /// Current reference count.
    pub fn ref_count(&self, id: AssetId) -> Option<u32> {
        self.slots.get(&id).map(|s| s.ref_count)
    }

    // ── Queries ────────────────────────────────────────────────

    /// Load state for an asset.
    pub fn load_state(&self, id: AssetId) -> LoadState {
        self.slots
            .get(&id)
            .map(|s| s.state)
            .unwrap_or(LoadState::None)
    }

    /// Get asset data reference.
    pub fn asset_data(&self, id: AssetId) -> Option<&AssetData> {
        self.slots.get(&id).and_then(|s| s.data.as_ref())
    }

    /// Dependencies of a loaded or pending asset.
    pub fn dependencies(&self, id: AssetId) -> Option<&[AssetId]> {
        self.slots.get(&id).map(|s| s.dependencies.as_slice())
    }

    /// Check if all dependencies of an asset are loaded.
    pub fn dependencies_ready(&self, id: AssetId) -> bool {
        self.slots
            .get(&id)
            .map(|s| {
                s.dependencies
                    .iter()
                    .all(|dep| self.load_state(*dep) == LoadState::Ready)
            })
            .unwrap_or(false)
    }

    /// Number of loaded (ready) assets.
    pub fn ready_count(&self) -> usize {
        self.slots
            .values()
            .filter(|s| s.state == LoadState::Ready)
            .count()
    }

    /// Number of pending assets.
    pub fn pending_count(&self) -> usize {
        self.slots
            .values()
            .filter(|s| s.state == LoadState::Pending)
            .count()
    }

    /// Total number of tracked asset slots.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Total estimated memory of ready assets.
    pub fn total_memory(&self) -> usize {
        self.slots
            .values()
            .filter_map(|s| s.data.as_ref())
            .map(|d| d.estimated_bytes())
            .sum()
    }

    // ── Batch loading ──────────────────────────────────────────

    /// Begin a batch load operation for a set of asset IDs.
    pub fn begin_batch(&mut self, ids: &[AssetId]) -> Vec<AssetHandle> {
        let total = ids.len();
        self.batch = Some(BatchProgress {
            total,
            completed: 0,
            failed: 0,
        });
        let mut handles = Vec::with_capacity(total);
        for &id in ids {
            if let Ok(h) = self.request_load(id) {
                handles.push(h);
            }
        }
        handles
    }

    /// Current batch progress.
    pub fn batch_progress(&self) -> Option<&BatchProgress> {
        self.batch.as_ref()
    }

    /// Clear the current batch tracker.
    pub fn end_batch(&mut self) -> Option<BatchProgress> {
        self.batch.take()
    }
}

impl Default for AssetLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tex_data() -> AssetData {
        AssetData::Texture {
            width: 256,
            height: 256,
            channels: 4,
            data: vec![0u8; 64],
        }
    }

    fn mesh_data() -> AssetData {
        AssetData::Mesh {
            vertex_count: 100,
            index_count: 200,
            data: vec![1u8; 32],
        }
    }

    #[test]
    fn register_manifest_returns_id() {
        let mut loader = AssetLoader::new();
        let id = loader.register_manifest("models/hero.mesh", AssetKind::Mesh, 1024, vec![]);
        assert!(id > 0);
        assert_eq!(loader.manifest_count(), 1);
    }

    #[test]
    fn register_manifest_idempotent() {
        let mut loader = AssetLoader::new();
        let id1 = loader.register_manifest("tex/a.png", AssetKind::Texture, 512, vec![]);
        let id2 = loader.register_manifest("tex/a.png", AssetKind::Texture, 512, vec![]);
        assert_eq!(id1, id2);
        assert_eq!(loader.manifest_count(), 1);
    }

    #[test]
    fn id_for_path() {
        let mut loader = AssetLoader::new();
        let id = loader.register_manifest("a.mesh", AssetKind::Mesh, 100, vec![]);
        assert_eq!(loader.id_for_path("a.mesh"), Some(id));
        assert_eq!(loader.id_for_path("nonexistent"), None);
    }

    #[test]
    fn request_and_finish_load() {
        let mut loader = AssetLoader::new();
        let id = loader.register_manifest("t.png", AssetKind::Texture, 100, vec![]);
        let handle = loader.request_load(id).unwrap();
        assert_eq!(handle.id, id);
        assert_eq!(loader.load_state(id), LoadState::Pending);
        loader.finish_load(id, tex_data()).unwrap();
        assert_eq!(loader.load_state(id), LoadState::Ready);
    }

    #[test]
    fn double_finish_load_fails() {
        let mut loader = AssetLoader::new();
        let id = loader.register_manifest("t.png", AssetKind::Texture, 100, vec![]);
        loader.request_load(id).unwrap();
        loader.finish_load(id, tex_data()).unwrap();
        let err = loader.finish_load(id, tex_data()).unwrap_err();
        assert_eq!(err, AssetError::AlreadyLoaded(id));
    }

    #[test]
    fn fail_load() {
        let mut loader = AssetLoader::new();
        let id = loader.register_manifest("bad.png", AssetKind::Texture, 100, vec![]);
        loader.request_load(id).unwrap();
        loader.fail_load(id, "corrupt").unwrap();
        assert_eq!(loader.load_state(id), LoadState::Failed);
    }

    #[test]
    fn reference_counting() {
        let mut loader = AssetLoader::new();
        let id = loader.register_manifest("m.mesh", AssetKind::Mesh, 100, vec![]);
        loader.request_load(id).unwrap();
        loader.finish_load(id, mesh_data()).unwrap();
        assert_eq!(loader.ref_count(id), Some(1));
        loader.add_ref(id).unwrap();
        assert_eq!(loader.ref_count(id), Some(2));
        loader.release(id).unwrap();
        assert_eq!(loader.ref_count(id), Some(1));
        loader.release(id).unwrap();
        // Ref count hit 0 → slot removed.
        assert_eq!(loader.ref_count(id), None);
        assert_eq!(loader.load_state(id), LoadState::None);
    }

    #[test]
    fn asset_data_access() {
        let mut loader = AssetLoader::new();
        let id = loader.register_manifest("t.png", AssetKind::Texture, 100, vec![]);
        loader.request_load(id).unwrap();
        assert!(loader.asset_data(id).is_none()); // still pending
        loader.finish_load(id, tex_data()).unwrap();
        assert!(loader.asset_data(id).is_some());
    }

    #[test]
    fn dependencies_tracking() {
        let mut loader = AssetLoader::new();
        let tid = loader.register_manifest("t.png", AssetKind::Texture, 100, vec![]);
        let mid = loader.register_manifest("m.mat", AssetKind::Material, 50, vec![tid]);
        loader.request_load(mid).unwrap();
        assert_eq!(loader.dependencies(mid), Some(vec![tid].as_slice()));
        assert!(!loader.dependencies_ready(mid));
        // Load the dependency.
        loader.request_load(tid).unwrap();
        loader.finish_load(tid, tex_data()).unwrap();
        assert!(loader.dependencies_ready(mid));
    }

    #[test]
    fn batch_loading_progress() {
        let mut loader = AssetLoader::new();
        let id1 = loader.register_manifest("a.mesh", AssetKind::Mesh, 100, vec![]);
        let id2 = loader.register_manifest("b.mesh", AssetKind::Mesh, 100, vec![]);
        let id3 = loader.register_manifest("c.mesh", AssetKind::Mesh, 100, vec![]);
        loader.begin_batch(&[id1, id2, id3]);
        assert_eq!(loader.batch_progress().unwrap().total, 3);
        assert!(!loader.batch_progress().unwrap().is_done());
        loader.finish_load(id1, mesh_data()).unwrap();
        assert_eq!(loader.batch_progress().unwrap().completed, 1);
        loader.finish_load(id2, mesh_data()).unwrap();
        loader.fail_load(id3, "oops").unwrap();
        assert!(loader.batch_progress().unwrap().is_done());
        let bp = loader.end_batch().unwrap();
        assert_eq!(bp.completed, 2);
        assert_eq!(bp.failed, 1);
    }

    #[test]
    fn batch_progress_ratio() {
        let bp = BatchProgress {
            total: 4,
            completed: 2,
            failed: 1,
        };
        assert!((bp.ratio() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn batch_progress_ratio_zero_total() {
        let bp = BatchProgress {
            total: 0,
            completed: 0,
            failed: 0,
        };
        assert!((bp.ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ready_and_pending_counts() {
        let mut loader = AssetLoader::new();
        let id1 = loader.register_manifest("a.mesh", AssetKind::Mesh, 100, vec![]);
        let id2 = loader.register_manifest("b.mesh", AssetKind::Mesh, 100, vec![]);
        loader.request_load(id1).unwrap();
        loader.request_load(id2).unwrap();
        assert_eq!(loader.pending_count(), 2);
        assert_eq!(loader.ready_count(), 0);
        loader.finish_load(id1, mesh_data()).unwrap();
        assert_eq!(loader.pending_count(), 1);
        assert_eq!(loader.ready_count(), 1);
    }

    #[test]
    fn total_memory_estimation() {
        let mut loader = AssetLoader::new();
        let id = loader.register_manifest("t.png", AssetKind::Texture, 100, vec![]);
        loader.request_load(id).unwrap();
        loader.finish_load(id, tex_data()).unwrap();
        assert!(loader.total_memory() > 0);
    }

    #[test]
    fn asset_kind_display() {
        assert_eq!(AssetKind::Mesh.to_string(), "mesh");
        assert_eq!(AssetKind::Animation.to_string(), "animation");
    }

    #[test]
    fn handle_display() {
        let h = AssetHandle::new(42);
        assert_eq!(h.to_string(), "AssetHandle(42)");
    }

    #[test]
    fn asset_error_display() {
        let e = AssetError::NotFound(7);
        assert_eq!(e.to_string(), "asset 7 not found");
    }

    #[test]
    fn asset_data_kind() {
        let d = tex_data();
        assert_eq!(d.kind(), AssetKind::Texture);
        let m = mesh_data();
        assert_eq!(m.kind(), AssetKind::Mesh);
    }

    #[test]
    fn manifest_entry_lookup() {
        let mut loader = AssetLoader::new();
        loader.register_manifest("hero.mesh", AssetKind::Mesh, 2048, vec![]);
        let entry = loader.manifest_entry("hero.mesh").unwrap();
        assert_eq!(entry.size_bytes, 2048);
        assert_eq!(entry.kind, AssetKind::Mesh);
    }

    #[test]
    fn request_load_idempotent_while_pending() {
        let mut loader = AssetLoader::new();
        let id = loader.register_manifest("t.png", AssetKind::Texture, 100, vec![]);
        let h1 = loader.request_load(id).unwrap();
        let h2 = loader.request_load(id).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(loader.slot_count(), 1);
    }
}
