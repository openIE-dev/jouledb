//! GGUF → JouleDB ingestor.
//!
//! Reads GGUF files (single or sharded), extracts tensor metadata and data,
//! and inserts each tensor into JouleDB. Large tensors (>256KB) automatically
//! use extent-based contiguous page storage.

use crate::WeightDB;
use joule_db_core::error::Error;
use std::path::Path;

/// Statistics from an ingest operation.
#[derive(Debug, Clone, Default)]
pub struct IngestStats {
    pub tensors_ingested: usize,
    pub bytes_ingested: u64,
    pub metadata_entries: usize,
    pub extent_tensors: usize, // Tensors stored as extents (>256KB)
    pub inline_tensors: usize, // Tensors stored inline (<256KB)
    pub gguf_metadata_keys: usize,
    pub elapsed_ms: u64,
}

/// Ingest a GGUF model into JouleDB.
///
/// `model_id` is the key prefix (e.g., "nemotron-3-super-120b").
/// `path` can be a single .gguf file or a directory containing sharded .gguf files.
pub fn ingest_gguf(db: &WeightDB, model_id: &str, path: &Path) -> Result<IngestStats, Error> {
    let start = std::time::Instant::now();

    // Collect GGUF files to process
    let gguf_files = if path.is_dir() {
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        for entry in std::fs::read_dir(path).map_err(|e| {
            Error::Storage(joule_db_core::error::StorageError::Io(format!(
                "read_dir '{}': {}",
                path.display(),
                e
            )))
        })? {
            let entry = entry.map_err(|e| {
                Error::Storage(joule_db_core::error::StorageError::Io(e.to_string()))
            })?;
            let p = entry.path();
            if p.extension().map(|e| e == "gguf").unwrap_or(false) {
                files.push(p);
            }
        }
        files.sort();
        files
    } else {
        vec![path.to_path_buf()]
    };

    if gguf_files.is_empty() {
        return Err(Error::Storage(joule_db_core::error::StorageError::Io(
            format!("No .gguf files found at '{}'", path.display()),
        )));
    }

    let mut stats = IngestStats::default();
    let mut all_tensor_names: Vec<String> = Vec::new();
    let mut layer_count: usize = 0;
    let mut expert_count: usize = 0;

    for gguf_path in &gguf_files {
        eprintln!("  Ingesting: {}", gguf_path.display());

        // mmap the GGUF file for reading
        let file = std::fs::File::open(gguf_path).map_err(|e| {
            Error::Storage(joule_db_core::error::StorageError::Io(format!(
                "open '{}': {}",
                gguf_path.display(),
                e
            )))
        })?;

        // Safety: we only read from this mmap, file is open for the duration
        #[allow(unsafe_code)]
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                Error::Storage(joule_db_core::error::StorageError::Io(format!(
                    "mmap '{}': {}",
                    gguf_path.display(),
                    e
                )))
            })?
        };

        let data = &mmap[..];

        // Parse GGUF header
        if data.len() < 24 || &data[0..4] != b"GGUF" {
            return Err(Error::Storage(joule_db_core::error::StorageError::Io(
                format!("'{}' is not a valid GGUF file", gguf_path.display()),
            )));
        }

        let _version = read_u32_le(data, 4)?;
        let tensor_count = read_u64_le(data, 8)? as usize;
        let kv_count = read_u64_le(data, 16)? as usize;

        eprintln!("    Tensors: {}, KV pairs: {}", tensor_count, kv_count);

        // Skip metadata KV pairs
        let mut offset = 24usize;
        for _ in 0..kv_count {
            offset = skip_gguf_kv(data, offset)?;
            stats.gguf_metadata_keys += 1;
        }

        // Parse tensor infos (cap pre-allocation against hostile GGUF headers)
        let mut tensor_infos = Vec::with_capacity(tensor_count.min(65_536));
        for _ in 0..tensor_count {
            let (info, new_offset) = parse_tensor_info(data, offset)?;
            tensor_infos.push(info);
            offset = new_offset;
        }

        // Alignment (default 32)
        let alignment = 32usize; // TODO: read from metadata
        let data_section_start = (offset + alignment - 1) / alignment * alignment;

        // Ingest each tensor
        for info in &tensor_infos {
            let tensor_data_start = data_section_start + info.offset as usize;
            let tensor_data_end = tensor_data_start + info.size_bytes as usize;

            if tensor_data_end > data.len() {
                eprintln!(
                    "    WARNING: tensor '{}' extends past EOF ({}+{} > {}), skipping",
                    info.name,
                    tensor_data_start,
                    info.size_bytes,
                    data.len()
                );
                continue;
            }

            let tensor_data = &data[tensor_data_start..tensor_data_end];

            // Store tensor data in JouleDB
            db.put_tensor(model_id, &info.name, tensor_data)?;

            // Store tensor metadata as JSON
            let meta_json = format!(
                r#"{{"name":"{}","shape":{:?},"type_id":{},"type_name":"{}","size_bytes":{}}}"#,
                info.name,
                info.shape,
                info.type_id,
                ggml_type_name(info.type_id),
                info.size_bytes,
            );
            db.put_tensor_meta(model_id, &info.name, &meta_json)?;

            // Track stats
            all_tensor_names.push(info.name.clone());
            stats.tensors_ingested += 1;
            stats.bytes_ingested += info.size_bytes;
            stats.metadata_entries += 1;

            if info.size_bytes > 256 * 1024 {
                stats.extent_tensors += 1;
            } else {
                stats.inline_tensors += 1;
            }

            // Track layers/experts
            if let Some(layer) = extract_layer_idx(&info.name) {
                layer_count = layer_count.max(layer + 1);
            }
            if info.name.contains("_exps") {
                if info.shape.len() == 3 {
                    expert_count = expert_count.max(info.shape[2]);
                }
            }

            // Progress every 50 tensors
            if stats.tensors_ingested % 50 == 0 {
                eprintln!(
                    "    [{}/{}] {:.1} GB ingested",
                    stats.tensors_ingested,
                    tensor_count,
                    stats.bytes_ingested as f64 / 1e9,
                );
            }
        }
    }

    // Store model index
    let index_json = format!(
        r#"{{"model_id":"{}","tensor_count":{},"total_bytes":{},"layer_count":{},"expert_count":{},"tensor_names":{:?}}}"#,
        model_id,
        stats.tensors_ingested,
        stats.bytes_ingested,
        layer_count,
        expert_count,
        all_tensor_names,
    );
    db.put_model_index(model_id, &index_json)?;

    // Flush to disk
    eprintln!("  Flushing to disk...");
    db.flush()?;

    stats.elapsed_ms = start.elapsed().as_millis() as u64;
    Ok(stats)
}

/// Ingest a GGUF model into JouleDB as a **contiguous weight blob**.
///
/// Unlike [`ingest_gguf`] (which stores each tensor as a paged extent in
/// the `.jdb`, read back via a buffer-pool copy), this writes every
/// tensor's raw quantized bytes contiguously into a companion file
/// `<db>.wblob`, 64-byte aligned, and records each tensor's `blob_offset`
/// in its meta JSON. The `.jdb` holds only metadata + the model index.
///
/// Result: `WeightDB::get_tensor_mmap` returns a **zero-copy** slice into
/// the mmap'd blob — no buffer pool, no per-page headers, no copy. Load is
/// instant and the bytes are UMA-resident (the substrate streams active
/// experts straight from here). The blob bytes are byte-identical to the
/// GGUF tensor data, so inference output is unchanged.
pub fn ingest_gguf_blob(db: &WeightDB, model_id: &str, path: &Path) -> Result<IngestStats, Error> {
    use std::io::Write;

    let start = std::time::Instant::now();

    let gguf_files = collect_gguf_files(path)?;
    if gguf_files.is_empty() {
        return Err(parse_err(&format!(
            "No .gguf files found at '{}'",
            path.display()
        )));
    }

    let blob_path = db.blob_path();
    let blob_file = std::fs::File::create(&blob_path)
        .map_err(|e| parse_err(&format!("create blob '{}': {}", blob_path.display(), e)))?;
    let mut blob = std::io::BufWriter::with_capacity(8 * 1024 * 1024, blob_file);
    let mut blob_off: u64 = 0;

    let mut stats = IngestStats::default();
    let mut all_tensor_names: Vec<String> = Vec::new();
    let mut layer_count: usize = 0;
    let mut expert_count: usize = 0;

    for gguf_path in &gguf_files {
        eprintln!("  Ingesting (blob): {}", gguf_path.display());
        let file = std::fs::File::open(gguf_path)
            .map_err(|e| parse_err(&format!("open '{}': {}", gguf_path.display(), e)))?;
        #[allow(unsafe_code)]
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| parse_err(&format!("mmap '{}': {}", gguf_path.display(), e)))?;
        let data = &mmap[..];

        if data.len() < 24 || &data[0..4] != b"GGUF" {
            return Err(parse_err(&format!(
                "'{}' is not a valid GGUF file",
                gguf_path.display()
            )));
        }
        let tensor_count = read_u64_le(data, 8)? as usize;
        let kv_count = read_u64_le(data, 16)? as usize;

        let mut offset = 24usize;
        for _ in 0..kv_count {
            offset = skip_gguf_kv(data, offset)?;
            stats.gguf_metadata_keys += 1;
        }

        let mut tensor_infos = Vec::with_capacity(tensor_count.min(65_536));
        for _ in 0..tensor_count {
            let (info, new_offset) = parse_tensor_info(data, offset)?;
            tensor_infos.push(info);
            offset = new_offset;
        }

        let alignment = 32usize;
        let data_section_start = (offset + alignment - 1) / alignment * alignment;

        for info in &tensor_infos {
            let tensor_data_start = data_section_start + info.offset as usize;
            let tensor_data_end = tensor_data_start + info.size_bytes as usize;
            if tensor_data_end > data.len() {
                eprintln!("    WARNING: tensor '{}' past EOF, skipping", info.name);
                continue;
            }
            let tensor_data = &data[tensor_data_start..tensor_data_end];

            // 64-byte align each tensor's start in the blob.
            let pad = (64 - (blob_off % 64)) % 64;
            if pad > 0 {
                blob.write_all(&[0u8; 64][..pad as usize])
                    .map_err(|e| parse_err(&format!("blob pad write: {e}")))?;
                blob_off += pad;
            }
            let this_off = blob_off;
            blob.write_all(tensor_data)
                .map_err(|e| parse_err(&format!("blob write '{}': {e}", info.name)))?;
            blob_off += tensor_data.len() as u64;

            let meta_json = format!(
                r#"{{"name":"{}","shape":{:?},"type_id":{},"type_name":"{}","size_bytes":{},"blob_offset":{}}}"#,
                info.name,
                info.shape,
                info.type_id,
                ggml_type_name(info.type_id),
                info.size_bytes,
                this_off,
            );
            db.put_tensor_meta(model_id, &info.name, &meta_json)?;

            all_tensor_names.push(info.name.clone());
            stats.tensors_ingested += 1;
            stats.bytes_ingested += info.size_bytes;
            stats.metadata_entries += 1;
            stats.extent_tensors += 1;

            if let Some(layer) = extract_layer_idx(&info.name) {
                layer_count = layer_count.max(layer + 1);
            }
            if info.name.contains("_exps") && info.shape.len() == 3 {
                expert_count = expert_count.max(info.shape[2]);
            }
            if stats.tensors_ingested % 50 == 0 {
                eprintln!(
                    "    [{}] {:.1} GB → blob",
                    stats.tensors_ingested,
                    blob_off as f64 / 1e9
                );
            }
        }
    }

    blob.flush()
        .map_err(|e| parse_err(&format!("blob flush: {e}")))?;

    let index_json = format!(
        r#"{{"model_id":"{}","tensor_count":{},"total_bytes":{},"layer_count":{},"expert_count":{},"storage":"blob","tensor_names":{:?}}}"#,
        model_id,
        stats.tensors_ingested,
        stats.bytes_ingested,
        layer_count,
        expert_count,
        all_tensor_names,
    );
    db.put_model_index(model_id, &index_json)?;
    eprintln!("  Flushing index...");
    db.flush()?;

    stats.elapsed_ms = start.elapsed().as_millis() as u64;
    Ok(stats)
}

/// Collect `.gguf` files from a file or directory path (shards sorted).
fn collect_gguf_files(path: &Path) -> Result<Vec<std::path::PathBuf>, Error> {
    if path.is_dir() {
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        for entry in std::fs::read_dir(path)
            .map_err(|e| parse_err(&format!("read_dir '{}': {}", path.display(), e)))?
        {
            let entry = entry.map_err(|e| parse_err(&e.to_string()))?;
            let p = entry.path();
            if p.extension().map(|e| e == "gguf").unwrap_or(false) {
                files.push(p);
            }
        }
        files.sort();
        Ok(files)
    } else {
        Ok(vec![path.to_path_buf()])
    }
}

// ── GGUF parsing helpers ────────────────────────────────────────

struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    type_id: u32,
    offset: u64,
    size_bytes: u64,
}

fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, Error> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| parse_err("GGUF u32 offset overflow"))?;
    let slice = data
        .get(offset..end)
        .ok_or_else(|| parse_err("GGUF u32 out of bounds"))?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(slice);
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(data: &[u8], offset: usize) -> Result<u64, Error> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| parse_err("GGUF u64 offset overflow"))?;
    let slice = data
        .get(offset..end)
        .ok_or_else(|| parse_err("GGUF u64 out of bounds"))?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Ok(u64::from_le_bytes(buf))
}

fn skip_gguf_kv(data: &[u8], offset: usize) -> Result<usize, Error> {
    let key_len = read_u64_le(data, offset)? as usize;
    let mut pos = offset
        .checked_add(8)
        .and_then(|p| p.checked_add(key_len))
        .ok_or_else(|| parse_err("KV key length overflow"))?;
    if pos > data.len() {
        return Err(parse_err("KV key out of bounds"));
    }

    let value_type = read_u32_le(data, pos)?;
    pos += 4;

    pos = skip_gguf_value(data, pos, value_type)?;
    Ok(pos)
}

fn skip_gguf_value(data: &[u8], offset: usize, type_id: u32) -> Result<usize, Error> {
    match type_id {
        0 | 1 | 7 => offset
            .checked_add(1)
            .filter(|&p| p <= data.len())
            .ok_or_else(|| parse_err("scalar value oob")),
        2 | 3 => offset
            .checked_add(2)
            .filter(|&p| p <= data.len())
            .ok_or_else(|| parse_err("scalar value oob")),
        4 | 5 | 6 => offset
            .checked_add(4)
            .filter(|&p| p <= data.len())
            .ok_or_else(|| parse_err("scalar value oob")),
        10 | 11 | 12 => offset
            .checked_add(8)
            .filter(|&p| p <= data.len())
            .ok_or_else(|| parse_err("scalar value oob")),
        8 => {
            // string
            let len = read_u64_le(data, offset)? as usize;
            offset
                .checked_add(8)
                .and_then(|p| p.checked_add(len))
                .filter(|&p| p <= data.len())
                .ok_or_else(|| parse_err("string len oob"))
        }
        9 => {
            // array
            let elem_type = read_u32_le(data, offset)?;
            let count = read_u64_le(data, offset.saturating_add(4))? as usize;
            let mut pos = offset
                .checked_add(12)
                .filter(|&p| p <= data.len())
                .ok_or_else(|| parse_err("array header oob"))?;
            for _ in 0..count {
                pos = skip_gguf_value(data, pos, elem_type)?;
            }
            Ok(pos)
        }
        _ => Err(parse_err(&format!("unknown GGUF value type: {}", type_id))),
    }
}

fn parse_tensor_info(data: &[u8], offset: usize) -> Result<(TensorInfo, usize), Error> {
    let name_len = read_u64_le(data, offset)? as usize;
    let pos = offset
        .checked_add(8)
        .ok_or_else(|| parse_err("tensor name offset overflow"))?;
    let name_end = pos
        .checked_add(name_len)
        .filter(|&e| e <= data.len())
        .ok_or_else(|| parse_err("tensor name oob"))?;
    let name = String::from_utf8_lossy(&data[pos..name_end]).to_string();
    let mut pos = name_end;

    let n_dims = read_u32_le(data, pos)? as usize;
    if n_dims > 16 {
        return Err(parse_err("tensor n_dims exceeds 16"));
    }
    pos += 4;

    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        shape.push(read_u64_le(data, pos)? as usize);
        pos += 8;
    }

    let type_id = read_u32_le(data, pos)?;
    pos += 4;

    let tensor_offset = read_u64_le(data, pos)?;
    pos += 8;

    let size_bytes = compute_tensor_size(&shape, type_id);

    Ok((
        TensorInfo {
            name,
            shape,
            type_id,
            offset: tensor_offset,
            size_bytes,
        },
        pos,
    ))
}

fn compute_tensor_size(shape: &[usize], type_id: u32) -> u64 {
    let n_elements: u64 = shape.iter().map(|&d| d as u64).product();
    if n_elements == 0 {
        return 0;
    }

    let (type_size, block_size) = match type_id {
        0 => (4, 1),      // F32
        1 => (2, 1),      // F16
        2 => (18, 32),    // Q4_0
        3 => (20, 32),    // Q4_1
        6 => (22, 32),    // Q5_0
        7 => (24, 32),    // Q5_1
        8 => (34, 32),    // Q8_0
        9 => (36, 32),    // Q8_1
        10 => (256, 256), // Q2_K
        11 => (110, 256), // Q3_K
        12 => (144, 256), // Q4_K
        13 => (176, 256), // Q5_K
        14 => (210, 256), // Q6_K
        15 => (292, 256), // Q8_K
        _ => (1, 1),
    };

    if block_size == 1 {
        n_elements * type_size
    } else {
        let n_blocks = (n_elements + block_size - 1) / block_size;
        n_blocks * type_size
    }
}

fn extract_layer_idx(name: &str) -> Option<usize> {
    for (i, part) in name.split('.').enumerate() {
        if part == "layers" || part == "blk" {
            let parts: Vec<&str> = name.split('.').collect();
            if i + 1 < parts.len() {
                return parts[i + 1].parse().ok();
            }
        }
    }
    None
}

fn ggml_type_name(type_id: u32) -> &'static str {
    match type_id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        _ => "unknown",
    }
}

fn parse_err(msg: &str) -> Error {
    Error::Storage(joule_db_core::error::StorageError::Io(msg.to_string()))
}
