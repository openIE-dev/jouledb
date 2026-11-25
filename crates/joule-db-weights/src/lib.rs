//! LLM weight storage for JouleDB.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
//!
//! Stores model weight tensors in JouleDB using extent-based contiguous
//! page allocation for fast sequential reads. Provides:
//!
//! - **Ingest**: GGUF → JouleDB (one-time, sequential scan)
//! - **Query**: Get tensor by name, get layer, get experts
//! - **Tier hints**: Pin/evict/prefetch via buffer pool
//! - **Zero-copy**: mmap-based reads for UMA GPU access
//!
//! # Key scheme
//!
//! Tensors are stored with composite keys:
//! ```text
//! model_id\0tensor_name  →  tensor data (raw bytes, extent-stored)
//! model_id\0_meta\0tensor_name  →  JSON metadata (inline, small)
//! model_id\0_index  →  JSON model index (layer count, expert count, etc.)
//! ```

pub mod causal_decoder;
pub mod checkpoint_ingest;
pub mod compress;
pub mod continual_plasticity;
pub mod federated_mesh;
pub mod grammar_constrain;
pub mod ingest;
pub mod multimodal_encoder;
pub mod query;
pub mod rank_sweep_bench;
pub mod synapse;
pub mod transformer_encoder;

use joule_db_core::error::Error;
use joule_db_core::{BufferPoolConfig, DiskBackend, Engine, EngineConfig};
use memmap2::Mmap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A JouleDB instance configured for LLM weight storage.
pub struct WeightDB {
    engine: Engine,
    db_path: PathBuf,
    /// Lazily-opened mmap of the contiguous weight blob (`<db>.wblob`), if
    /// the model was ingested with `ingest::ingest_gguf_blob`. Holds the
    /// quantized tensor bytes verbatim, contiguous and page-aligned, so a
    /// `(offset, len)` slice is a zero-copy `QuantWeight` (no buffer pool,
    /// no per-page headers — see get_tensor_mmap).
    blob: Mutex<Option<Arc<Mmap>>>,
}

impl WeightDB {
    /// Open or create a weight database at the given path.
    ///
    /// Uses a large buffer pool (4096 pages = 256 MB) to accommodate
    /// extent-based tensor storage during ingest and inference.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::open_with_pool_size(path, 4096) // 256 MB default
    }

    /// Open with a custom buffer pool capacity (in pages, 64KB each).
    ///
    /// For inference reads of large tensors, use a larger pool to avoid thrashing:
    /// - 4096 pages = 256 MB (default, good for ingest)
    /// - 8192 pages = 512 MB (good for loading a few layers)
    /// - 16384 pages = 1 GB (good for loading many tensors concurrently)
    pub fn open_with_pool_size(path: impl AsRef<Path>, pool_pages: usize) -> Result<Self, Error> {
        let path = path.as_ref();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                Error::Storage(joule_db_core::error::StorageError::Io(format!(
                    "create_dir_all '{}': {}",
                    parent.display(),
                    e
                )))
            })?;
        }

        let backend = DiskBackend::open(path)?;
        let config = EngineConfig {
            buffer_pool: BufferPoolConfig {
                capacity: pool_pages,
                num_shards: 32,
            },
            ..Default::default()
        };

        let engine = Engine::open_or_create_with_config(backend, config)?;

        Ok(Self {
            engine,
            db_path: path.to_path_buf(),
            blob: Mutex::new(None),
        })
    }

    /// Get a reference to the underlying engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Get the database file path.
    pub fn path(&self) -> &Path {
        &self.db_path
    }

    /// Path of the contiguous weight blob companion file (`<db>.wblob`).
    pub fn blob_path(&self) -> PathBuf {
        let mut s = self.db_path.clone().into_os_string();
        s.push(".wblob");
        PathBuf::from(s)
    }

    /// Lazily mmap the contiguous weight blob (zero-copy, UMA-resident).
    /// Returns `Ok(None)` if the model wasn't ingested in blob mode.
    pub fn blob_mmap(&self) -> Result<Option<Arc<Mmap>>, Error> {
        let mut guard = self.blob.lock().map_err(|_| {
            Error::Storage(joule_db_core::error::StorageError::Io(
                "blob mmap lock poisoned".into(),
            ))
        })?;
        if let Some(m) = guard.as_ref() {
            return Ok(Some(Arc::clone(m)));
        }
        let path = self.blob_path();
        if !path.exists() {
            return Ok(None);
        }
        let file = std::fs::File::open(&path).map_err(|e| {
            Error::Storage(joule_db_core::error::StorageError::Io(format!(
                "open blob '{}': {}",
                path.display(),
                e
            )))
        })?;
        // Safety: read-only mapping of an immutable weight blob.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            Error::Storage(joule_db_core::error::StorageError::Io(format!(
                "mmap blob '{}': {}",
                path.display(),
                e
            )))
        })?;
        let arc = Arc::new(mmap);
        *guard = Some(Arc::clone(&arc));
        Ok(Some(arc))
    }

    /// Zero-copy view of a tensor stored in the contiguous blob: returns
    /// `(blob_mmap, byte_offset, byte_len)` ready for
    /// `QuantWeight::from_mmap_slice`. Reads the tensor's `blob_offset` +
    /// `size_bytes` from its meta JSON. `Ok(None)` if not blob-ingested.
    pub fn get_tensor_mmap(
        &self,
        model_id: &str,
        tensor_name: &str,
    ) -> Result<Option<(Arc<Mmap>, usize, usize)>, Error> {
        let meta = match self.get_tensor_meta(model_id, tensor_name)? {
            Some(m) => m,
            None => return Ok(None),
        };
        let (Some(off), Some(len)) = (
            json_u64(&meta, "blob_offset"),
            json_u64(&meta, "size_bytes"),
        ) else {
            return Ok(None);
        };
        let mmap = match self.blob_mmap()? {
            Some(m) => m,
            None => return Ok(None),
        };
        Ok(Some((mmap, off as usize, len as usize)))
    }

    /// Zero-copy view of a **byte sub-range** of a blob-stored tensor:
    /// returns `(blob_mmap, blob_offset + byte_off, byte_len)`. This is the
    /// sparse-fetch primitive — e.g. a fused MoE expert tensor
    /// `blk.N.ffn_up_exps.weight` holds all `n_expert` experts contiguously
    /// at equal stride `per = size_bytes / n_expert`, so expert `e` is the
    /// range `(e*per, per)`. Reading the top-k routed experts touches only
    /// `k*per` bytes of the blob; the OS pages in just those, so resident
    /// memory stays bounded by what routing actually selects rather than the
    /// whole layer. Errors if `byte_off + byte_len` exceeds the tensor's
    /// stored `size_bytes`. `Ok(None)` if the tensor isn't blob-ingested
    /// (paged-extent `.jdb` — use a range read on the buffer-pool path).
    pub fn get_tensor_range_mmap(
        &self,
        model_id: &str,
        tensor_name: &str,
        byte_off: usize,
        byte_len: usize,
    ) -> Result<Option<(Arc<Mmap>, usize, usize)>, Error> {
        let (mmap, off, len) = match self.get_tensor_mmap(model_id, tensor_name)? {
            Some(t) => t,
            None => return Ok(None),
        };
        let end = byte_off.checked_add(byte_len).ok_or_else(|| {
            Error::Storage(joule_db_core::error::StorageError::Io(format!(
                "{tensor_name}: range {byte_off}+{byte_len} overflows"
            )))
        })?;
        if end > len {
            return Err(Error::Storage(joule_db_core::error::StorageError::Io(
                format!("{tensor_name}: range end {end} exceeds tensor size {len}"),
            )));
        }
        Ok(Some((mmap, off + byte_off, byte_len)))
    }

    /// Build a tensor data key: `model_id\0tensor_name`
    pub fn tensor_key(model_id: &str, tensor_name: &str) -> Vec<u8> {
        let mut key = Vec::with_capacity(model_id.len() + 1 + tensor_name.len());
        key.extend_from_slice(model_id.as_bytes());
        key.push(0);
        key.extend_from_slice(tensor_name.as_bytes());
        key
    }

    /// Build a metadata key: `model_id\0_meta\0tensor_name`
    pub fn meta_key(model_id: &str, tensor_name: &str) -> Vec<u8> {
        let mut key = Vec::with_capacity(model_id.len() + 7 + tensor_name.len());
        key.extend_from_slice(model_id.as_bytes());
        key.extend_from_slice(b"\0_meta\0");
        key.extend_from_slice(tensor_name.as_bytes());
        key
    }

    /// Build a model index key: `model_id\0_index`
    pub fn index_key(model_id: &str) -> Vec<u8> {
        let mut key = Vec::with_capacity(model_id.len() + 7);
        key.extend_from_slice(model_id.as_bytes());
        key.extend_from_slice(b"\0_index");
        key
    }

    /// Store a tensor's raw data.
    /// For tensors > 256KB, this automatically uses extent storage.
    pub fn put_tensor(&self, model_id: &str, tensor_name: &str, data: &[u8]) -> Result<(), Error> {
        let key = Self::tensor_key(model_id, tensor_name);
        self.engine.put(&key, data)
    }

    /// Store tensor metadata as JSON.
    pub fn put_tensor_meta(
        &self,
        model_id: &str,
        tensor_name: &str,
        meta_json: &str,
    ) -> Result<(), Error> {
        let key = Self::meta_key(model_id, tensor_name);
        self.engine.put(&key, meta_json.as_bytes())
    }

    /// Store the model index (layer count, expert count, tensor list).
    pub fn put_model_index(&self, model_id: &str, index_json: &str) -> Result<(), Error> {
        let key = Self::index_key(model_id);
        self.engine.put(&key, index_json.as_bytes())
    }

    /// Retrieve a tensor's raw data.
    pub fn get_tensor(&self, model_id: &str, tensor_name: &str) -> Result<Option<Vec<u8>>, Error> {
        let key = Self::tensor_key(model_id, tensor_name);
        self.engine.get(&key)
    }

    /// Retrieve tensor metadata.
    pub fn get_tensor_meta(
        &self,
        model_id: &str,
        tensor_name: &str,
    ) -> Result<Option<String>, Error> {
        let key = Self::meta_key(model_id, tensor_name);
        match self.engine.get(&key)? {
            Some(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).to_string())),
            None => Ok(None),
        }
    }

    /// Retrieve the model index.
    pub fn get_model_index(&self, model_id: &str) -> Result<Option<String>, Error> {
        let key = Self::index_key(model_id);
        match self.engine.get(&key)? {
            Some(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).to_string())),
            None => Ok(None),
        }
    }

    /// Flush all dirty pages to disk.
    pub fn flush(&self) -> Result<(), Error> {
        self.engine.sync()
    }
}

/// Minimal extractor for an integer field in our own flat meta JSON
/// (`...,"key":12345,...`). Avoids a serde_json dep; the meta is written
/// by `ingest::ingest_gguf_blob`, so the format is fixed and unnested.
fn json_u64(json: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}
