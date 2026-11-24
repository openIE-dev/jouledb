//! Pure-Rust HuggingFace `.safetensors` -> `JouleSynapse` Living Checkpoint Compiler
//!
//! Converts frozen flat-slab `.safetensors` model files (`F32`, `F16`, `BF16` + `__metadata__`)
//! directly into JouleDB's living post-frozen-weights substrate:
//! - 2D/ND weight matrices factorize into sub-1-bit [`LittleBitSVFTensor`] bitplanes (`< 1.0` BPW)
//!   with Sakana `Transformer²` `SVF` singular channels ready for real-time online adaptation.
//! - Layer/expert shards register into [`AirStreamScheduler`] with `PowerInfer` structural
//!   hot-score priors (`attn`/`router` pinned hot, `mlp`/`ffn_exp` streamed cold).
//! - Model `__metadata__` and 1D norm/bias descriptors populate [`EngramMemoryTable`] for $O(1)$
//!   conditional lookup, and every tensor emits a relational [`IngestedTensorSummary`] row for
//!   live SQL querying (`SELECT * FROM synapse_tensors`).

use crate::synapse::{CausalModelMeta, JouleHarnessEngine, LittleBitSVFTensor, WeightShardMeta};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Safety bound on `.safetensors` JSON header size (32 MiB) against hostile OOM headers.
const MAX_SAFETENSORS_HEADER_BYTES: usize = 32 * 1024 * 1024;
/// Safety bound on single tensor element count during in-memory factorization (`4 Gi` on 64-bit for
/// 262k-vocab `< 30 GB` frontier checkpoints including Google `Gemma 4` `embed_tokens_per_layer.weight`).
#[cfg(target_pointer_width = "64")]
const MAX_TENSOR_ELEMENTS: usize = 4 * 1024 * 1024 * 1024;
#[cfg(not(target_pointer_width = "64"))]
const MAX_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024;

/// Supported `.safetensors` element types for ingestion and serialization (including MLX packed `U32`/`U16`
/// and NVIDIA Blackwell / ModelOpt `F8_E4M3` / `F8_E5M2` / `NVFP4` `U8`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetensorDtype {
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I64,
    I32,
    U32,
    I16,
    U16,
    I8,
    U8,
    Bool,
}

impl SafetensorDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F64 => "F64",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::Bool => "BOOL",
        }
    }

    pub fn bytes_per_elem(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    pub fn is_floating_point(self) -> bool {
        matches!(
            self,
            Self::F64 | Self::F32 | Self::F16 | Self::BF16 | Self::F8E4M3 | Self::F8E5M2
        )
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "F64" => Ok(Self::F64),
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "F8_E4M3" | "FLOAT8_E4M3FN" => Ok(Self::F8E4M3),
            "F8_E5M2" | "FLOAT8_E5M2" => Ok(Self::F8E5M2),
            "I64" => Ok(Self::I64),
            "I32" => Ok(Self::I32),
            "U32" => Ok(Self::U32),
            "I16" => Ok(Self::I16),
            "U16" => Ok(Self::U16),
            "I8" => Ok(Self::I8),
            "U8" => Ok(Self::U8),
            "BOOL" => Ok(Self::Bool),
            other => Err(format!("unsupported safetensors dtype '{other}'")),
        }
    }
}

/// Relational row describing a single tensor converted into `JouleSynapse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestedTensorSummary {
    pub model_id: String,
    pub tensor_name: String,
    pub layer_idx: i64,
    pub source_dtype: String,
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub original_fp16_bytes: u64,
    pub compressed_bytes: u64,
    pub bits_per_weight: f64,
    pub compression_vs_fp16: f64,
    pub reconstruction_cosine: f64,
    pub hot_score: f64,
}

/// Aggregate report returned after converting a `.safetensors` checkpoint into `JouleSynapse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointIngestReport {
    pub model_id: String,
    pub total_tensors: usize,
    pub factorized_2d_tensors: usize,
    pub engram_entries_added: usize,
    pub original_fp16_bytes: u64,
    pub compressed_bytes: u64,
    pub overall_bits_per_weight: f64,
    pub overall_compression_vs_fp16: f64,
    pub mean_reconstruction_cosine: f64,
    #[serde(default)]
    pub svf_distilled: bool,
    #[serde(default)]
    pub pre_distill_probe_mse: f64,
    #[serde(default)]
    pub post_distill_probe_mse: f64,
    pub tensors: Vec<IngestedTensorSummary>,
}

/// Input tensor descriptor for [`build_safetensors_bytes`].
#[derive(Debug, Clone)]
pub struct SafetensorBuildEntry {
    pub name: String,
    pub dtype: SafetensorDtype,
    pub shape: Vec<usize>,
    pub values_f32: Vec<f32>,
}

/// Serialize a valid HuggingFace `.safetensors` binary checkpoint in pure safe Rust.
pub fn build_safetensors_bytes(
    metadata: &BTreeMap<String, String>,
    entries: &[SafetensorBuildEntry],
) -> Result<Vec<u8>, String> {
    let mut header_map = serde_json::Map::new();
    if !metadata.is_empty() {
        let mut meta_obj = serde_json::Map::new();
        for (k, v) in metadata {
            meta_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        header_map.insert(
            "__metadata__".to_string(),
            serde_json::Value::Object(meta_obj),
        );
    }

    let mut data_buf: Vec<u8> = Vec::new();
    for entry in entries {
        let expected: usize = entry.shape.iter().copied().product();
        if expected != entry.values_f32.len() {
            return Err(format!(
                "tensor '{}' shape {:?} product {expected} != values len {}",
                entry.name,
                entry.shape,
                entry.values_f32.len()
            ));
        }
        let start = data_buf.len();
        match entry.dtype {
            SafetensorDtype::F32 => {
                for &v in &entry.values_f32 {
                    data_buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            SafetensorDtype::F16 => {
                for &v in &entry.values_f32 {
                    data_buf.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
                }
            }
            SafetensorDtype::BF16 => {
                for &v in &entry.values_f32 {
                    let bf = (v.to_bits() >> 16) as u16;
                    data_buf.extend_from_slice(&bf.to_le_bytes());
                }
            }
            SafetensorDtype::F8E4M3 => {
                for &v in &entry.values_f32 {
                    data_buf.push(f32_to_f8_e4m3(v));
                }
            }
            SafetensorDtype::F8E5M2 => {
                for &v in &entry.values_f32 {
                    data_buf.push(f32_to_f8_e5m2(v));
                }
            }
            SafetensorDtype::F64 => {
                for &v in &entry.values_f32 {
                    data_buf.extend_from_slice(&(v as f64).to_le_bytes());
                }
            }
            SafetensorDtype::I64 => {
                for &v in &entry.values_f32 {
                    data_buf.extend_from_slice(&(v as i64).to_le_bytes());
                }
            }
            SafetensorDtype::I32 => {
                for &v in &entry.values_f32 {
                    data_buf.extend_from_slice(&(v as i32).to_le_bytes());
                }
            }
            SafetensorDtype::U32 => {
                for &v in &entry.values_f32 {
                    data_buf.extend_from_slice(&(v as u32).to_le_bytes());
                }
            }
            SafetensorDtype::I16 => {
                for &v in &entry.values_f32 {
                    data_buf.extend_from_slice(&(v as i16).to_le_bytes());
                }
            }
            SafetensorDtype::U16 => {
                for &v in &entry.values_f32 {
                    data_buf.extend_from_slice(&(v as u16).to_le_bytes());
                }
            }
            SafetensorDtype::I8 => {
                for &v in &entry.values_f32 {
                    data_buf.push((v as i8) as u8);
                }
            }
            SafetensorDtype::U8 | SafetensorDtype::Bool => {
                for &v in &entry.values_f32 {
                    data_buf.push(v as u8);
                }
            }
        }
        let end = data_buf.len();
        header_map.insert(
            entry.name.clone(),
            serde_json::json!({
                "dtype": entry.dtype.as_str(),
                "shape": entry.shape,
                "data_offsets": [start, end],
            }),
        );
    }

    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header_map))
        .map_err(|e| format!("serialize safetensors header: {e}"))?;
    // 8-byte pad header with spaces per HuggingFace safetensors alignment convention
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }

    let mut out = Vec::with_capacity(8 + header_bytes.len() + data_buf.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data_buf);
    Ok(out)
}

/// Parse and convert a `.safetensors` byte buffer directly into [`JouleHarnessEngine`]:
/// factorizes 2D+ matrices into sub-1-bit `LittleBit-2` bitplanes, populates `Engram` metadata,
/// and registers shards into `AirStreamScheduler`.
pub fn ingest_safetensors_into_harness(
    model_id: &str,
    raw_bytes: &[u8],
    target_rank: usize,
    max_outliers: usize,
    harness: &mut JouleHarnessEngine,
) -> Result<CheckpointIngestReport, String> {
    ingest_safetensors_with_options(
        model_id,
        raw_bytes,
        target_rank,
        max_outliers,
        false,
        harness,
    )
}

/// Memory-map a `.safetensors` checkpoint file on disk and compile its tensors zero-copy directly
/// from OS page cache into [`JouleHarnessEngine`] (avoiding `10–20 GB` heap `Vec<u8>` copies!).
pub fn ingest_safetensors_file_with_options(
    model_id: &str,
    path: &std::path::Path,
    target_rank: usize,
    max_outliers: usize,
    distill_svf: bool,
    harness: &mut JouleHarnessEngine,
) -> Result<CheckpointIngestReport, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("open safetensors '{}': {e}", path.display()))?;
    #[allow(unsafe_code)]
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| format!("mmap safetensors '{}': {e}", path.display()))?;
    ingest_safetensors_with_options(
        model_id,
        &mmap[..],
        target_rank,
        max_outliers,
        distill_svf,
        harness,
    )
}

/// Parse and convert a `.safetensors` byte buffer directly into [`JouleHarnessEngine`] with
/// optional single-pass Teacher->Student `SVF` + Per-Row Gain Self-Distillation (`distill_svf`).
pub fn ingest_safetensors_with_options(
    model_id: &str,
    raw_bytes: &[u8],
    target_rank: usize,
    max_outliers: usize,
    distill_svf: bool,
    harness: &mut JouleHarnessEngine,
) -> Result<CheckpointIngestReport, String> {
    if model_id.trim().is_empty() {
        return Err("model_id must not be empty".to_string());
    }
    if raw_bytes.len() < 8 {
        return Err("safetensors file shorter than 8-byte header length".to_string());
    }
    let mut len_arr = [0u8; 8];
    len_arr.copy_from_slice(&raw_bytes[0..8]);
    let header_len = u64::from_le_bytes(len_arr) as usize;
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(format!(
            "safetensors header length {header_len} invalid or exceeds {MAX_SAFETENSORS_HEADER_BYTES} cap"
        ));
    }
    let data_base = 8usize
        .checked_add(header_len)
        .filter(|&b| b <= raw_bytes.len())
        .ok_or_else(|| "safetensors header extends past EOF".to_string())?;

    let header_val: serde_json::Value = serde_json::from_slice(&raw_bytes[8..data_base])
        .map_err(|e| format!("invalid safetensors JSON header: {e}"))?;
    let header_obj = header_val
        .as_object()
        .ok_or_else(|| "safetensors header root must be a JSON object".to_string())?;

    let mut engram_entries_added = 0usize;
    if let Some(meta_obj) = header_obj.get("__metadata__").and_then(|v| v.as_object()) {
        for (mk, mv) in meta_obj {
            if let Some(ms) = mv.as_str() {
                harness
                    .engram
                    .upsert(&format!("{model_id} metadata {mk}"), vec![], ms);
                engram_entries_added += 1;
            }
        }
    }

    let mut summaries = Vec::new();
    let mut factorized_2d_tensors = 0usize;
    let mut total_fp16_bytes = 0u64;
    let mut total_compressed_bytes = 0u64;
    let mut total_elements_2d = 0usize;
    let mut total_bits_2d = 0.0f64;
    let mut cosine_sum = 0.0f64;
    let mut pre_mse_sum = 0.0f64;
    let mut post_mse_sum = 0.0f64;

    // Sort tensor names deterministically and pre-parse header descriptors so MLX affine
    // quantized triplets (`*.weight: U32`, `*.scales: F16/BF16/F32`, `*.biases: F16/BF16/F32`)
    // can be fused cleanly into living `LittleBitSVFTensor` bitplanes!
    let mut tensor_keys: Vec<&String> = header_obj
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .collect();
    tensor_keys.sort();

    #[derive(Clone)]
    struct ParsedHeaderDesc {
        dtype: SafetensorDtype,
        shape: Vec<usize>,
        elem_count: usize,
        abs_start: usize,
        abs_end: usize,
    }

    let mut desc_map: BTreeMap<String, ParsedHeaderDesc> = BTreeMap::new();
    let mut norm_1d_dims: Vec<usize> = Vec::new();
    for tname in &tensor_keys {
        let Some(desc) = header_obj.get(*tname).and_then(|v| v.as_object()) else {
            continue;
        };
        let dtype_str = desc
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("tensor '{tname}' missing dtype"))?;
        let dtype = SafetensorDtype::parse(dtype_str)?;
        let shape_arr = desc
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("tensor '{tname}' missing shape"))?;
        let mut shape = Vec::with_capacity(shape_arr.len().min(16));
        for dim_v in shape_arr {
            let d = dim_v
                .as_u64()
                .ok_or_else(|| format!("tensor '{tname}' invalid shape dim"))?
                as usize;
            shape.push(d);
        }
        let offsets = desc
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .filter(|a| a.len() == 2)
            .ok_or_else(|| format!("tensor '{tname}' invalid data_offsets"))?;
        let off_start = offsets[0]
            .as_u64()
            .ok_or_else(|| format!("tensor '{tname}' invalid start offset"))?
            as usize;
        let off_end = offsets[1]
            .as_u64()
            .ok_or_else(|| format!("tensor '{tname}' invalid end offset"))?
            as usize;
        if off_start > off_end {
            return Err(format!("tensor '{tname}' start offset > end offset"));
        }
        let abs_start = data_base
            .checked_add(off_start)
            .ok_or_else(|| "offset overflow".to_string())?;
        let abs_end = data_base
            .checked_add(off_end)
            .filter(|&e| e <= raw_bytes.len())
            .ok_or_else(|| format!("tensor '{tname}' data_offsets past EOF"))?;
        let elem_count = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| format!("tensor '{tname}' element count overflow"))?;
        if elem_count == 0 || elem_count > MAX_TENSOR_ELEMENTS {
            return Err(format!(
                "tensor '{tname}' element count {elem_count} outside 1..={MAX_TENSOR_ELEMENTS}"
            ));
        }
        let expected_bytes = elem_count
            .checked_mul(dtype.bytes_per_elem())
            .ok_or_else(|| "tensor byte size overflow".to_string())?;
        if abs_end - abs_start != expected_bytes {
            return Err(format!(
                "tensor '{tname}' byte slice {} != expected {expected_bytes}",
                abs_end - abs_start
            ));
        }
        if dtype.is_floating_point()
            && shape.len() == 1
            && shape[0] >= 16
            && !norm_1d_dims.contains(&shape[0])
        {
            norm_1d_dims.push(shape[0]);
        }
        desc_map.insert(
            (*tname).clone(),
            ParsedHeaderDesc {
                dtype,
                shape,
                elem_count,
                abs_start,
                abs_end,
            },
        );
    }

    for tname in tensor_keys {
        let Some(pdesc) = desc_map.get(tname.as_str()).cloned() else {
            continue;
        };
        // Skip standalone MLX `.scales`/`.biases` and NVIDIA ModelOpt / Inkling `NVFP4`/`FP8`
        // auxiliary scale/amax sidecar tensors when paired with their parent weight tensor!
        if let Some(stem) = tname
            .strip_suffix(".scales")
            .or_else(|| tname.strip_suffix(".biases"))
            && let Some(w_desc) = desc_map.get(&format!("{stem}.weight"))
            && w_desc.dtype == SafetensorDtype::U32
        {
            continue;
        }
        if tname.ends_with(".weight_scale")
            || tname.ends_with(".weight_scale_2")
            || tname.ends_with(".input_scale")
            || tname.ends_with(".k_scale")
            || tname.ends_with(".v_scale")
            || tname.ends_with("_weight_scale")
            || tname.ends_with("_weight_amax")
            || tname.ends_with("_weight_outer_scale_denominator")
            || tname.ends_with(".input_max")
            || tname.ends_with(".input_min")
            || tname.ends_with(".output_max")
            || tname.ends_with(".output_min")
        {
            continue;
        }

        let mut dtype = pdesc.dtype;
        let mut shape = pdesc.shape;
        let mut elem_count = pdesc.elem_count;
        let mut source_dtype_label = dtype.as_str().to_string();

        let scale_companion_key = tname
            .strip_suffix(".weight")
            .map(|st| format!("{st}.weight_scale"))
            .filter(|k| desc_map.contains_key(k))
            .or_else(|| {
                let k2 = format!("{tname}_scale");
                desc_map.contains_key(&k2).then_some(k2)
            });

        // 1. MLX packed affine quantized 2D tensor (`U32 [rows, u32_cols]` + `.scales` + optional `.biases`)
        // 2. NVIDIA Blackwell / Thinking Machines Lab `NVFP4` (`U8` packed `E2M1` nibbles + `F8_E4M3` block scale)
        // 3. NVIDIA ModelOpt `FP8` (`F8_E4M3` / `F8_E5M2` + block/tensor `weight_scale`)
        let decoded = if dtype == SafetensorDtype::U32
            && shape.len() == 2
            && let Some(stem) = tname.strip_suffix(".weight")
            && let Some(sc_desc) = desc_map.get(&format!("{stem}.scales"))
            && sc_desc.shape.len() == 2
            && sc_desc.shape[0] == shape[0]
        {
            let rows = shape[0];
            let u32_cols = shape[1].max(1);
            let n_groups = sc_desc.shape[1].max(1);
            let scales_f32 = decode_tensor_to_f32(
                &raw_bytes[sc_desc.abs_start..sc_desc.abs_end],
                sc_desc.dtype,
                sc_desc.elem_count,
            );
            let biases_f32 = if let Some(bi_desc) = desc_map.get(&format!("{stem}.biases")) {
                decode_tensor_to_f32(
                    &raw_bytes[bi_desc.abs_start..bi_desc.abs_end],
                    bi_desc.dtype,
                    bi_desc.elem_count,
                )
            } else {
                vec![0.0f32; sc_desc.elem_count]
            };

            let mut resolved_bits = 0usize;
            for &nd in &norm_1d_dims {
                if nd >= u32_cols
                    && nd.is_multiple_of(u32_cols)
                    && nd.is_multiple_of(n_groups)
                    && matches!(nd / u32_cols, 32 | 16 | 8 | 4)
                {
                    resolved_bits = 32 / (nd / u32_cols);
                    break;
                }
            }
            if resolved_bits == 0 {
                for gs in [128usize, 64, 32] {
                    let cand_cols = n_groups * gs;
                    if cand_cols >= u32_cols
                        && cand_cols.is_multiple_of(u32_cols)
                        && matches!(cand_cols / u32_cols, 32 | 16 | 8 | 4)
                    {
                        resolved_bits = 32 / (cand_cols / u32_cols);
                        break;
                    }
                }
            }
            let bits = if resolved_bits > 0 { resolved_bits } else { 4 };
            let vals_per_u32 = (32 / bits).max(1);
            let cols = u32_cols * vals_per_u32;
            let group_size = (cols / n_groups).max(1);
            let mask = ((1u64 << bits) - 1) as u32;
            let total_unpacked = rows
                .checked_mul(cols)
                .filter(|&n| n <= MAX_TENSOR_ELEMENTS)
                .ok_or_else(|| format!("MLX unpacked tensor '{tname}' exceeds element bound"))?;

            let mut unpacked = Vec::with_capacity(total_unpacked);
            let w_raw = &raw_bytes[pdesc.abs_start..pdesc.abs_end];
            for r in 0..rows {
                let sc_row = &scales_f32[r * n_groups..(r + 1) * n_groups];
                let bi_row = &biases_f32[r * n_groups..(r + 1) * n_groups];
                for w_idx in 0..u32_cols {
                    let byte_off = (r * u32_cols + w_idx) * 4;
                    let word = u32::from_le_bytes([
                        w_raw[byte_off],
                        w_raw[byte_off + 1],
                        w_raw[byte_off + 2],
                        w_raw[byte_off + 3],
                    ]);
                    for sub in 0..vals_per_u32 {
                        let col = w_idx * vals_per_u32 + sub;
                        let g = (col / group_size).min(n_groups - 1);
                        let q = ((word >> (sub * bits)) & mask) as f32;
                        unpacked.push(q * sc_row[g] + bi_row[g]);
                    }
                }
            }
            dtype = SafetensorDtype::F16;
            shape = vec![rows, cols];
            elem_count = total_unpacked;
            source_dtype_label = format!("MLX_U32_{bits}BIT");
            unpacked
        } else if dtype == SafetensorDtype::U8
            && !shape.is_empty()
            && let Some(ref sc_key) = scale_companion_key
            && let Some(sc_desc) = desc_map.get(sc_key)
        {
            // NVIDIA Blackwell / ModelOpt / Thinking Machines Lab `NVFP4` (`E2M1` 2 nibbles/byte + `F8_E4M3` block scale)
            let total_unpacked = elem_count
                .checked_mul(2)
                .filter(|&n| n <= MAX_TENSOR_ELEMENTS)
                .ok_or_else(|| format!("NVFP4 unpacked tensor '{tname}' exceeds element bound"))?;
            let scales_f32 = decode_tensor_to_f32(
                &raw_bytes[sc_desc.abs_start..sc_desc.abs_end],
                sc_desc.dtype,
                sc_desc.elem_count,
            );
            let outer_mul = tname
                .strip_suffix(".weight")
                .and_then(|st| desc_map.get(&format!("{st}.weight_scale_2")))
                .map(|d2| {
                    decode_tensor_to_f32(
                        &raw_bytes[d2.abs_start..d2.abs_end],
                        d2.dtype,
                        d2.elem_count,
                    )
                    .first()
                    .copied()
                    .unwrap_or(1.0)
                })
                .or_else(|| {
                    desc_map
                        .get(&format!("{tname}_outer_scale_denominator"))
                        .map(|dd| {
                            let denom = decode_tensor_to_f32(
                                &raw_bytes[dd.abs_start..dd.abs_end],
                                dd.dtype,
                                dd.elem_count,
                            )
                            .first()
                            .copied()
                            .unwrap_or(1.0);
                            if denom.abs() > 1e-12 {
                                1.0 / denom
                            } else {
                                1.0
                            }
                        })
                })
                .unwrap_or(1.0);

            let n_sc = scales_f32.len().max(1);
            let group_size = (total_unpacked / n_sc).max(1);
            let mut unpacked = Vec::with_capacity(total_unpacked);
            for (byte_idx, &b) in raw_bytes[pdesc.abs_start..pdesc.abs_end].iter().enumerate() {
                let idx0 = byte_idx * 2;
                let idx1 = idx0 + 1;
                let s0 = scales_f32[(idx0 / group_size).min(n_sc - 1)] * outer_mul;
                let s1 = scales_f32[(idx1 / group_size).min(n_sc - 1)] * outer_mul;
                unpacked.push(fp4_e2m1_to_f32(b & 0x0F) * s0);
                unpacked.push(fp4_e2m1_to_f32(b >> 4) * s1);
            }
            if let Some(last_d) = shape.last_mut() {
                *last_d *= 2;
            }
            dtype = SafetensorDtype::F16;
            elem_count = total_unpacked;
            source_dtype_label = "NVFP4_E2M1".to_string();
            unpacked
        } else {
            let mut vals = decode_tensor_to_f32(
                &raw_bytes[pdesc.abs_start..pdesc.abs_end],
                dtype,
                elem_count,
            );
            if matches!(dtype, SafetensorDtype::F8E4M3 | SafetensorDtype::F8E5M2)
                && let Some(ref sc_key) = scale_companion_key
                && let Some(sc_desc) = desc_map.get(sc_key)
            {
                let scales_f32 = decode_tensor_to_f32(
                    &raw_bytes[sc_desc.abs_start..sc_desc.abs_end],
                    sc_desc.dtype,
                    sc_desc.elem_count,
                );
                let n_sc = scales_f32.len().max(1);
                let group_size = (elem_count / n_sc).max(1);
                for (i, v) in vals.iter_mut().enumerate() {
                    *v *= scales_f32[(i / group_size).min(n_sc - 1)];
                }
                source_dtype_label = format!("{}_SCALED", dtype.as_str());
            }
            vals
        };

        let layer_idx = extract_layer_number(tname);
        let hot_score = infer_powerinfer_hot_score(tname);
        let fp16_equiv_bytes = (elem_count as u64) * 2;
        total_fp16_bytes += fp16_equiv_bytes;

        // Preserve compact depthwise 1D convolution kernels (`shape == [d, 1, K]`, `K <= 16`, e.g.,
        // Liquid AI `LFM2`/`LFM2.5`/`LFM3` `ShortConv` `conv.conv.weight` `[1024, 1, 3]` and
        // Cactus `Whistle` `dw`) in exact FP32 `vectors_1d` rather than low-rank SVD!
        let is_depthwise_1d_conv =
            shape.len() == 3 && shape[1] == 1 && shape[2] <= 16 && elem_count <= 262_144;

        if dtype.is_floating_point()
            && !is_depthwise_1d_conv
            && shape.len() >= 2
            && shape[0] >= 2
            && (elem_count / shape[0]) >= 2
        {
            // For 3D stacked Mixture-of-Experts tensors (`[num_experts, out_dim, in_dim]`, e.g.
            // Thinking Machines Lab `Inkling-2` `mlp.experts.w13_weight`), fold `[E, R, C]` -> `[E * R, C]`
            // so `cols == in_dim` and each expert `e` occupies contiguous rows `[e * R .. (e + 1) * R]`!
            let (rows, cols) = if shape.len() == 3 && tname.contains("experts.") && shape[2] >= 2 {
                (shape[0] * shape[1], shape[2])
            } else {
                (shape[0], elem_count / shape[0])
            };
            let (lb, eff_rank, d_pre, d_post) = factorize_2d_with_adaptive_budget(
                model_id,
                tname,
                &decoded,
                rows,
                cols,
                target_rank,
                max_outliers,
                distill_svf,
            )?;
            pre_mse_sum += d_pre;
            post_mse_sum += d_post;
            let bpw = lb.bits_per_weight();
            let comp = lb.compression_vs_fp16();
            let cos = lb.reconstruction_cosine;
            let comp_bytes = (((elem_count as f64) * bpw) / 8.0).ceil() as u64;
            let fq_name = format!("{model_id}::{tname}");

            let _ = harness.airstream.register_shard(WeightShardMeta {
                layer_idx: layer_idx.max(0) as usize,
                shard_id: fq_name.clone(),
                size_bytes: comp_bytes.max(64),
                hot_score,
            });
            harness.littlebit_tensors.insert(fq_name, lb);

            factorized_2d_tensors += 1;
            total_compressed_bytes += comp_bytes;
            total_elements_2d += elem_count;
            total_bits_2d += bpw * (elem_count as f64);
            cosine_sum += cos;

            summaries.push(IngestedTensorSummary {
                model_id: model_id.to_string(),
                tensor_name: tname.clone(),
                layer_idx,
                source_dtype: source_dtype_label,
                rows,
                cols,
                rank: eff_rank,
                original_fp16_bytes: fp16_equiv_bytes,
                compressed_bytes: comp_bytes,
                bits_per_weight: bpw,
                compression_vs_fp16: comp,
                reconstruction_cosine: cos,
                hot_score,
            });
        } else {
            // 1D norm / bias / scalar / depthwise-conv vector: store exact FP32 vector when bounded (`<= 262,144` elems)
            if dtype.is_floating_point() && elem_count <= 262_144 {
                if harness.vectors_1d.len() >= 16_384
                    && let Some(oldest_k) = harness.vectors_1d.keys().next().cloned()
                {
                    harness.vectors_1d.remove(&oldest_k);
                }
                harness
                    .vectors_1d
                    .insert(format!("{model_id}::{tname}"), decoded.clone());
            }
            let mean = decoded.iter().map(|&x| x as f64).sum::<f64>() / (elem_count as f64);
            harness.engram.upsert(
                &format!("{model_id} norm {tname}"),
                decoded.iter().copied().take(8).collect(),
                format!("{{\"len\":{elem_count},\"mean\":{mean:.6}}}"),
            );
            engram_entries_added += 1;
            total_compressed_bytes += fp16_equiv_bytes;

            summaries.push(IngestedTensorSummary {
                model_id: model_id.to_string(),
                tensor_name: tname.clone(),
                layer_idx,
                source_dtype: source_dtype_label,
                rows: elem_count,
                cols: 1,
                rank: 1,
                original_fp16_bytes: fp16_equiv_bytes,
                compressed_bytes: fp16_equiv_bytes,
                bits_per_weight: 16.0,
                compression_vs_fp16: 1.0,
                reconstruction_cosine: 1.0,
                hot_score: 0.95,
            });
        }
    }

    let overall_bpw = if total_elements_2d > 0 {
        total_bits_2d / (total_elements_2d as f64)
    } else {
        16.0
    };
    let overall_comp = if total_compressed_bytes > 0 {
        (total_fp16_bytes as f64) / (total_compressed_bytes as f64)
    } else {
        1.0
    };
    let mean_cos = if factorized_2d_tensors > 0 {
        cosine_sum / (factorized_2d_tensors as f64)
    } else {
        1.0
    };

    let denom_2d = factorized_2d_tensors.max(1) as f64;
    Ok(CheckpointIngestReport {
        model_id: model_id.to_string(),
        total_tensors: summaries.len(),
        factorized_2d_tensors,
        engram_entries_added,
        original_fp16_bytes: total_fp16_bytes,
        compressed_bytes: total_compressed_bytes,
        overall_bits_per_weight: overall_bpw,
        overall_compression_vs_fp16: overall_comp,
        mean_reconstruction_cosine: mean_cos,
        svf_distilled: distill_svf,
        pre_distill_probe_mse: if distill_svf {
            pre_mse_sum / denom_2d
        } else {
            0.0
        },
        post_distill_probe_mse: if distill_svf {
            post_mse_sum / denom_2d
        } else {
            0.0
        },
        tensors: summaries,
    })
}

#[allow(clippy::too_many_arguments)]
fn factorize_2d_with_adaptive_budget(
    model_id: &str,
    tname: &str,
    decoded: &[f32],
    rows: usize,
    cols: usize,
    target_rank: usize,
    max_outliers: usize,
    distill_svf: bool,
) -> Result<(LittleBitSVFTensor, usize, f64, f64), String> {
    let elem_count = rows * cols;
    let is_vocab_dict = tname.contains("embed_tokens")
        || tname.contains("word_embeddings")
        || tname.contains("tok_embeddings")
        || tname.contains("token_embedding")
        || tname.contains("embed_positions")
        || tname.contains("position_embedding")
        || tname.contains("lm_head");
    let (eff_rank, itq_iters, max_out_eff, run_distill, n_probes, n_sweeps) =
        if elem_count > 200_000_000 {
            (
                target_rank.min(16).clamp(1, rows.min(cols)),
                1usize,
                0usize,
                false,
                4usize,
                1usize,
            )
        } else if elem_count > 35_000_000 {
            (
                target_rank.max(24).min(48).clamp(1, rows.min(cols)),
                1usize,
                max_outliers.min(32),
                distill_svf,
                8usize,
                2usize,
            )
        } else if distill_svf && is_vocab_dict && cols >= 64 {
            (
                target_rank.max(cols.min(160)).clamp(1, rows.min(cols)),
                2usize,
                max_outliers,
                distill_svf,
                16usize,
                3usize,
            )
        } else {
            (
                target_rank.clamp(1, rows.min(cols)),
                if rows >= 8 * cols { 2 } else { 6 },
                max_outliers,
                distill_svf,
                16usize,
                3usize,
            )
        };
    let fq_name = format!("{model_id}::{tname}");
    let mut lb = LittleBitSVFTensor::factorize(
        fq_name,
        decoded,
        rows,
        cols,
        eff_rank,
        itq_iters,
        max_out_eff,
    )?;
    let (mut pre_m, mut post_m) = (0.0, 0.0);
    if run_distill && let Ok(d_rec) = lb.distill_svf_against_dense(decoded, n_probes, n_sweeps) {
        pre_m = d_rec.pre_probe_mse;
        post_m = d_rec.post_probe_mse;
    }
    Ok((lb, eff_rank, pre_m, post_m))
}

/// Memory-map a `.gguf` checkpoint file on disk and compile its quantized tensors (`Q1_0_G128`,
/// `TQ1_0`, `TQ2_0`, `Q4_0`, `Q4_K`, `Q6_K`, `Q8_0`, `F16`, `BF16`, `F32`) + embedded BPE tokenizer
/// + `CausalModelMeta` directly into [`JouleHarnessEngine`]!
pub fn ingest_gguf_file_with_options(
    model_id: &str,
    path: &std::path::Path,
    target_rank: usize,
    max_outliers: usize,
    distill_svf: bool,
    harness: &mut JouleHarnessEngine,
) -> Result<CheckpointIngestReport, String> {
    let file =
        std::fs::File::open(path).map_err(|e| format!("open GGUF '{}': {e}", path.display()))?;
    #[allow(unsafe_code)]
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| format!("mmap GGUF '{}': {e}", path.display()))?;
    ingest_gguf_bytes_with_options(
        model_id,
        &mmap[..],
        target_rank,
        max_outliers,
        distill_svf,
        harness,
    )
}

/// Parse and compile a GGUF v2/v3 byte slice directly into living sub-1-bit [`LittleBitSVFTensor`]
/// bitplanes, canonical HuggingFace layer names, `CausalModelMeta`, and `BpeTokenizer` vocab/merges!
pub fn ingest_gguf_bytes_with_options(
    model_id: &str,
    data: &[u8],
    target_rank: usize,
    max_outliers: usize,
    distill_svf: bool,
    harness: &mut JouleHarnessEngine,
) -> Result<CheckpointIngestReport, String> {
    if model_id.trim().is_empty() {
        return Err("model_id must not be empty".to_string());
    }
    if data.len() < 24 || &data[0..4] != b"GGUF" {
        return Err("invalid GGUF magic header".to_string());
    }
    let _version = gguf_u32(data, 4)?;
    let tensor_count = gguf_u64(data, 8)? as usize;
    let kv_count = gguf_u64(data, 16)? as usize;

    let mut pos = 24usize;
    let mut alignment = 32usize;
    let mut arch = String::new();
    let mut head_count: Option<usize> = None;
    let mut head_count_kv: Option<usize> = None;
    let mut key_length: Option<usize> = None;
    let mut rope_theta = 10_000.0f32;
    let mut rms_eps = 1e-5f32;
    let mut bos_id: Option<usize> = None;
    let mut add_bos = false;
    let mut eos_ids: Vec<usize> = Vec::new();
    let mut vocab_tokens: Vec<String> = Vec::new();
    let mut bpe_merges: Vec<(String, String)> = Vec::new();
    let mut engram_entries_added = 0usize;

    for _ in 0..kv_count {
        let (k, next_p) = gguf_str(data, pos)?;
        pos = next_p;
        let vtype = gguf_u32(data, pos)?;
        pos += 4;

        if k == "general.alignment" && vtype == 4 {
            alignment = (gguf_u32(data, pos)? as usize).max(1);
            pos += 4;
        } else if k == "general.architecture" && vtype == 8 {
            let (s, np) = gguf_str(data, pos)?;
            arch = s.clone();
            harness
                .engram
                .upsert(&format!("{model_id} metadata architecture"), vec![], &s);
            engram_entries_added += 1;
            pos = np;
        } else if k.ends_with(".attention.head_count") && vtype == 4 {
            head_count = Some(gguf_u32(data, pos)? as usize);
            pos += 4;
        } else if k.ends_with(".attention.head_count_kv") && vtype == 4 {
            head_count_kv = Some(gguf_u32(data, pos)? as usize);
            pos += 4;
        } else if k.ends_with(".attention.key_length") && vtype == 4 {
            key_length = Some(gguf_u32(data, pos)? as usize);
            pos += 4;
        } else if k.ends_with(".rope.freq_base") && vtype == 6 {
            rope_theta = gguf_f32(data, pos)?;
            pos += 4;
        } else if k.ends_with(".attention.layer_norm_rms_epsilon") && vtype == 6 {
            rms_eps = gguf_f32(data, pos)?;
            pos += 4;
        } else if k == "tokenizer.ggml.bos_token_id" && vtype == 4 {
            bos_id = Some(gguf_u32(data, pos)? as usize);
            pos += 4;
        } else if k == "tokenizer.ggml.add_bos_token" && vtype == 7 {
            add_bos = data.get(pos).copied().unwrap_or(0) != 0;
            pos += 1;
        } else if k == "tokenizer.ggml.eos_token_id" && vtype == 4 {
            eos_ids.push(gguf_u32(data, pos)? as usize);
            pos += 4;
        } else if k == "tokenizer.ggml.tokens" && vtype == 9 {
            let et = gguf_u32(data, pos)?;
            let cnt = gguf_u64(data, pos + 4)? as usize;
            pos += 12;
            if et == 8 && cnt <= 512_000 {
                vocab_tokens.reserve(cnt);
                for _ in 0..cnt {
                    let (tok, np) = gguf_str(data, pos)?;
                    vocab_tokens.push(tok);
                    pos = np;
                }
            } else {
                for _ in 0..cnt {
                    pos = gguf_skip_val(data, pos, et)?;
                }
            }
        } else if k == "tokenizer.ggml.merges" && vtype == 9 {
            let et = gguf_u32(data, pos)?;
            let cnt = gguf_u64(data, pos + 4)? as usize;
            pos += 12;
            if et == 8 && cnt <= 1_000_000 {
                bpe_merges.reserve(cnt);
                for _ in 0..cnt {
                    let (m_str, np) = gguf_str(data, pos)?;
                    if let Some((l, r)) = m_str.split_once(' ') {
                        bpe_merges.push((l.to_string(), r.to_string()));
                    }
                    pos = np;
                }
            } else {
                for _ in 0..cnt {
                    pos = gguf_skip_val(data, pos, et)?;
                }
            }
        } else {
            pos = gguf_skip_val(data, pos, vtype)?;
        }
    }

    if !vocab_tokens.is_empty() {
        harness
            .model_vocabs
            .insert(model_id.to_string(), vocab_tokens);
    }
    if !bpe_merges.is_empty() {
        harness
            .model_bpe_merges
            .insert(model_id.to_string(), bpe_merges);
    }
    harness.model_configs.insert(
        model_id.to_string(),
        CausalModelMeta {
            num_attention_heads: head_count,
            num_key_value_heads: head_count_kv,
            head_dim: key_length,
            rope_theta,
            rms_norm_eps: rms_eps,
            bos_token_id: if add_bos { bos_id } else { None },
            eos_token_ids: eos_ids,
            tie_word_embeddings: true,
            model_type: arch.clone(),
            hidden_act: if arch.contains("gemma") {
                "gelu_pytorch_tanh".to_string()
            } else if arch.contains("bitnet") {
                "relu2".to_string()
            } else {
                "silu".to_string()
            },
            gemma_rms_norm_offset: arch.contains("gemma"),
            final_logit_softcapping: None,
            conv_l_cache: 3,
            ..Default::default()
        },
    );

    struct GgufTensorHeader {
        raw_name: String,
        dims: Vec<usize>,
        type_id: u32,
        rel_offset: usize,
    }
    let mut theaders = Vec::with_capacity(tensor_count.min(65_536));
    for _ in 0..tensor_count {
        let (name, np) = gguf_str(data, pos)?;
        pos = np;
        let nd = gguf_u32(data, pos)? as usize;
        pos += 4;
        if nd > 16 {
            return Err(format!("GGUF tensor '{name}' n_dims {nd} > 16"));
        }
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(gguf_u64(data, pos)? as usize);
            pos += 8;
        }
        let tid = gguf_u32(data, pos)?;
        let roff = gguf_u64(data, pos + 4)? as usize;
        pos += 12;
        theaders.push(GgufTensorHeader {
            raw_name: name,
            dims,
            type_id: tid,
            rel_offset: roff,
        });
    }

    let data_section = pos.div_ceil(alignment) * alignment;
    let has_post_attn_norm = theaders
        .iter()
        .any(|t| t.raw_name.ends_with(".post_attention_norm.weight"));
    let mut summaries = Vec::new();
    let mut factorized_2d_tensors = 0usize;
    let mut total_fp16_bytes = 0u64;
    let mut total_compressed_bytes = 0u64;
    let mut total_elements_2d = 0usize;
    let mut total_bits_2d = 0.0f64;
    let mut cosine_sum = 0.0f64;
    let mut pre_mse_sum = 0.0f64;
    let mut post_mse_sum = 0.0f64;

    for th in &theaders {
        let elem_count = th
            .dims
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or_else(|| format!("GGUF tensor '{}' overflow", th.raw_name))?;
        if elem_count == 0 || elem_count > MAX_TENSOR_ELEMENTS {
            continue;
        }
        let byte_len = gguf_tensor_byte_size(elem_count, th.type_id)?;
        let start = data_section
            .checked_add(th.rel_offset)
            .ok_or_else(|| "GGUF tensor offset overflow".to_string())?;
        let end = start
            .checked_add(byte_len)
            .filter(|&e| e <= data.len())
            .ok_or_else(|| format!("GGUF tensor '{}' extends past EOF", th.raw_name))?;

        let tname = canonicalize_gguf_tensor_name(&th.raw_name, has_post_attn_norm);
        let decoded = dequantize_gguf_tensor(&data[start..end], elem_count, th.type_id)?;
        let dtype_label = gguf_type_label(th.type_id);
        let layer_idx = extract_layer_number(&tname);
        let hot_score = infer_powerinfer_hot_score(&tname);
        let fp16_equiv_bytes = (elem_count as u64) * 2;
        total_fp16_bytes += fp16_equiv_bytes;

        // GGUF stores `dims` inner-first (`dims[0] = cols`, `dims[1] = rows`, `dims[2] = experts`)!
        let is_depthwise_1d = elem_count <= 262_144
            && ((th.dims.len() == 3 && th.dims[1] == 1 && th.dims[0] <= 16)
                || (th.dims.len() == 2
                    && th.dims[0] <= 16
                    && (tname.contains(".conv.") || tname.contains("conv1d"))));
        if !is_depthwise_1d
            && th.dims.len() >= 2
            && th.dims[0] >= 2
            && (elem_count / th.dims[0]) >= 2
        {
            let cols = th.dims[0];
            let rows = elem_count / cols;
            let (lb, eff_rank, d_pre, d_post) = factorize_2d_with_adaptive_budget(
                model_id,
                &tname,
                &decoded,
                rows,
                cols,
                target_rank,
                max_outliers,
                distill_svf,
            )?;
            pre_mse_sum += d_pre;
            post_mse_sum += d_post;
            let bpw = lb.bits_per_weight();
            let comp = lb.compression_vs_fp16();
            let cos = lb.reconstruction_cosine;
            let comp_bytes = (((elem_count as f64) * bpw) / 8.0).ceil() as u64;
            let fq_name = format!("{model_id}::{tname}");
            let _ = harness.airstream.register_shard(WeightShardMeta {
                layer_idx: layer_idx.max(0) as usize,
                shard_id: fq_name.clone(),
                size_bytes: comp_bytes.max(64),
                hot_score,
            });
            harness.littlebit_tensors.insert(fq_name, lb);

            factorized_2d_tensors += 1;
            total_compressed_bytes += comp_bytes;
            total_elements_2d += elem_count;
            total_bits_2d += bpw * (elem_count as f64);
            cosine_sum += cos;

            summaries.push(IngestedTensorSummary {
                model_id: model_id.to_string(),
                tensor_name: tname,
                layer_idx,
                source_dtype: dtype_label.to_string(),
                rows,
                cols,
                rank: eff_rank,
                original_fp16_bytes: fp16_equiv_bytes,
                compressed_bytes: comp_bytes,
                bits_per_weight: bpw,
                compression_vs_fp16: comp,
                reconstruction_cosine: cos,
                hot_score,
            });
        } else {
            if elem_count <= 262_144 {
                if harness.vectors_1d.len() >= 16_384
                    && let Some(oldest_k) = harness.vectors_1d.keys().next().cloned()
                {
                    harness.vectors_1d.remove(&oldest_k);
                }
                harness
                    .vectors_1d
                    .insert(format!("{model_id}::{tname}"), decoded);
            }
            total_compressed_bytes += fp16_equiv_bytes;
            summaries.push(IngestedTensorSummary {
                model_id: model_id.to_string(),
                tensor_name: tname,
                layer_idx,
                source_dtype: dtype_label.to_string(),
                rows: elem_count,
                cols: 1,
                rank: 1,
                original_fp16_bytes: fp16_equiv_bytes,
                compressed_bytes: fp16_equiv_bytes,
                bits_per_weight: 16.0,
                compression_vs_fp16: 1.0,
                reconstruction_cosine: 1.0,
                hot_score: 0.95,
            });
        }
    }

    let overall_bpw = if total_elements_2d > 0 {
        total_bits_2d / (total_elements_2d as f64)
    } else {
        16.0
    };
    let overall_comp = if total_compressed_bytes > 0 {
        (total_fp16_bytes as f64) / (total_compressed_bytes as f64)
    } else {
        1.0
    };
    let mean_cos = if factorized_2d_tensors > 0 {
        cosine_sum / (factorized_2d_tensors as f64)
    } else {
        1.0
    };
    let denom_2d = factorized_2d_tensors.max(1) as f64;
    Ok(CheckpointIngestReport {
        model_id: model_id.to_string(),
        total_tensors: summaries.len(),
        factorized_2d_tensors,
        engram_entries_added,
        original_fp16_bytes: total_fp16_bytes,
        compressed_bytes: total_compressed_bytes,
        overall_bits_per_weight: overall_bpw,
        overall_compression_vs_fp16: overall_comp,
        mean_reconstruction_cosine: mean_cos,
        svf_distilled: distill_svf,
        pre_distill_probe_mse: if distill_svf {
            pre_mse_sum / denom_2d
        } else {
            0.0
        },
        post_distill_probe_mse: if distill_svf {
            post_mse_sum / denom_2d
        } else {
            0.0
        },
        tensors: summaries,
    })
}

fn canonicalize_gguf_tensor_name(raw: &str, has_post_attn_norm: bool) -> String {
    match raw {
        "token_embd.weight" => return "model.embed_tokens.weight".to_string(),
        "token_embd_norm.weight" => return "model.embedding_norm.weight".to_string(),
        "output_norm.weight" => return "model.norm.weight".to_string(),
        "output.weight" => return "lm_head.weight".to_string(),
        _ => {}
    }
    if let Some(rest) = raw.strip_prefix("blk.")
        && let Some((l_str, tail)) = rest.split_once('.')
    {
        let mapped = match tail {
            "attn_norm.weight" => "input_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_q.bias" => "self_attn.q_proj.bias",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_k.bias" => "self_attn.k_proj.bias",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_v.bias" => "self_attn.v_proj.bias",
            "attn_qkv.weight" => "self_attn.qkv_proj.weight",
            "attn_qkv.bias" => "self_attn.qkv_proj.bias",
            "attn_gate.weight" => "self_attn.attn_gate.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_output.bias" => "self_attn.o_proj.bias",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "attn_sub_norm.weight" => "self_attn.attn_sub_norm.weight",
            "ffn_norm.weight" if has_post_attn_norm => "pre_feedforward_layernorm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "post_attention_norm.weight" => "post_attention_layernorm.weight",
            "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
            "ffn_sub_norm.weight" => "mlp.ffn_sub_norm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "ffn_gate_inp.weight" => "mlp.gate.weight",
            "exp_probs_b.bias" => "mlp.expert_bias",
            "ffn_gate_exps.weight" => "mlp.experts.gate_proj.weight",
            "ffn_up_exps.weight" => "mlp.experts.up_proj.weight",
            "ffn_down_exps.weight" => "mlp.experts.down_proj.weight",
            "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
            "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
            "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
            "shortconv.in_proj.weight" => "conv.in_proj.weight",
            "shortconv.conv.weight" => "conv.conv.weight",
            "shortconv.out_proj.weight" => "conv.out_proj.weight",
            "ssm_a" => "ssm.a",
            "ssm_dt.bias" => "ssm.dt_bias",
            "ssm_alpha.weight" => "ssm.alpha_proj.weight",
            "ssm_beta.weight" => "ssm.beta_proj.weight",
            "ssm_conv1d.weight" => "ssm.conv1d.weight",
            "ssm_norm.weight" => "ssm.norm.weight",
            "ssm_out.weight" => "ssm.out_proj.weight",
            other => other,
        };
        return format!("model.layers.{l_str}.{mapped}");
    }
    raw.to_string()
}

fn gguf_type_label(type_id: u32) -> &'static str {
    match type_id {
        0 => "GGUF_F32",
        1 => "GGUF_F16",
        2 => "GGUF_Q4_0",
        3 => "GGUF_Q4_1",
        6 => "GGUF_Q5_0",
        7 => "GGUF_Q5_1",
        8 => "GGUF_Q8_0",
        10 => "GGUF_Q2_K",
        11 => "GGUF_Q3_K",
        12 => "GGUF_Q4_K",
        13 => "GGUF_Q5_K",
        14 => "GGUF_Q6_K",
        30 => "GGUF_BF16",
        34 => "GGUF_TQ1_0",
        35 => "GGUF_TQ2_0",
        36 => "GGUF_I2_S",
        40 => "GGUF_Q1_0",
        41 => "GGUF_Q1_0_G128",
        42 => "GGUF_Q2_0",
        142 => "GGUF_PQ2_0",
        143 => "GGUF_PTQ1_0",
        _ => "GGUF_QUANT",
    }
}

fn gguf_tensor_byte_size(elem_count: usize, type_id: u32) -> Result<usize, String> {
    if type_id == 36 {
        if !elem_count.is_multiple_of(128) {
            return Err(format!(
                "GGUF I2_S elem_count {elem_count} not divisible by 128"
            ));
        }
        return Ok(elem_count / 4 + 4);
    }
    let (block_bytes, block_elems) = match type_id {
        0 => (4usize, 1usize),
        1 | 30 => (2, 1),
        2 => (18, 32),
        3 => (20, 32),
        6 => (22, 32),
        7 => (24, 32),
        8 => (34, 32),
        10 => (84, 256),
        11 => (110, 256),
        12 => (144, 256),
        13 => (176, 256),
        14 => (210, 256),
        34 => (54, 256),
        35 => (66, 256),
        40 => (6, 32),
        41 => (18, 128), // PrismML Bonsai `Q1_0_G128`: f16 scale + 16B packed 128-bit signs
        42 => (18, 64),  // PrismML Ternary-Bonsai-2 `Q2_0`: f16 scale + 16B packed 64x 2-bit trits
        142 => (34, 128), // PrismML Ternary-Bonsai-2 `PQ2_0`: f16 scale + 32B packed 128x 2-bit trits
        143 => (28, 128), // PrismML Ternary-Bonsai-2 `PTQ1_0`: 24B base-3 + 2B qh + f16 scale
        other => return Err(format!("unsupported GGUF tensor type_id {other}")),
    };
    if !elem_count.is_multiple_of(block_elems) {
        return Err(format!(
            "GGUF elem_count {elem_count} not divisible by block_elems {block_elems} (type_id={type_id})"
        ));
    }
    Ok((elem_count / block_elems) * block_bytes)
}

fn dequantize_gguf_tensor(raw: &[u8], elem_count: usize, type_id: u32) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(elem_count);
    match type_id {
        0 => {
            for c in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        1 => {
            for c in raw.chunks_exact(2) {
                out.push(f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        30 => {
            for c in raw.chunks_exact(2) {
                out.push(f32::from_bits(
                    (u16::from_le_bytes([c[0], c[1]]) as u32) << 16,
                ));
            }
        }
        2 => {
            // Q4_0: 2B f16 d + 16B nibbles (low 16 elems, high 16 elems)
            for blk in raw.chunks_exact(18) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for j in 0..16 {
                    out.push(((blk[2 + j] & 0x0F) as i32 - 8) as f32 * d);
                }
                for j in 0..16 {
                    out.push(((blk[2 + j] >> 4) as i32 - 8) as f32 * d);
                }
            }
        }
        3 => {
            // Q4_1: 2B f16 d + 2B f16 m + 16B nibbles
            for blk in raw.chunks_exact(20) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let m = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                for j in 0..16 {
                    out.push((blk[4 + j] & 0x0F) as f32 * d + m);
                }
                for j in 0..16 {
                    out.push((blk[4 + j] >> 4) as f32 * d + m);
                }
            }
        }
        6 => {
            // Q5_0: 2B f16 d + 4B qh + 16B nibbles (32 elems)
            for blk in raw.chunks_exact(22) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                for j in 0..16 {
                    let h0 = ((qh >> j) & 1) << 4;
                    out.push((((blk[6 + j] & 0x0F) as u32 | h0) as i32 - 16) as f32 * d);
                }
                for j in 0..16 {
                    let h1 = ((qh >> (j + 12)) & 0x10) as u32;
                    out.push((((blk[6 + j] >> 4) as u32 | h1) as i32 - 16) as f32 * d);
                }
            }
        }
        7 => {
            // Q5_1: 2B f16 d + 2B f16 m + 4B qh + 16B nibbles (32 elems)
            for blk in raw.chunks_exact(24) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let m = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                for j in 0..16 {
                    let h0 = ((qh >> j) & 1) << 4;
                    out.push(((blk[8 + j] & 0x0F) as u32 | h0) as f32 * d + m);
                }
                for j in 0..16 {
                    let h1 = ((qh >> (j + 12)) & 0x10) as u32;
                    out.push(((blk[8 + j] >> 4) as u32 | h1) as f32 * d + m);
                }
            }
        }
        8 => {
            // Q8_0: 2B f16 d + 32B i8
            for blk in raw.chunks_exact(34) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &q in &blk[2..34] {
                    out.push((q as i8) as f32 * d);
                }
            }
        }
        10 => {
            // Q2_K: 16B scales + 64B qs + 2B f16 d + 2B f16 dmin (84B / 256 elems)
            for blk in raw.chunks_exact(84) {
                let sc = &blk[0..16];
                let qs = &blk[16..80];
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
                let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
                for n in 0..2 {
                    let q = &qs[n * 32..(n + 1) * 32];
                    for shift_idx in 0..4 {
                        let shift = shift_idx * 2;
                        let is = n * 8 + shift_idx * 2;
                        let dl0 = d * ((sc[is] & 0x0F) as f32);
                        let ml0 = dmin * ((sc[is] >> 4) as f32);
                        for &qb in &q[0..16] {
                            out.push(dl0 * (((qb >> shift) & 3) as f32) - ml0);
                        }
                        let dl1 = d * ((sc[is + 1] & 0x0F) as f32);
                        let ml1 = dmin * ((sc[is + 1] >> 4) as f32);
                        for &qb in &q[16..32] {
                            out.push(dl1 * (((qb >> shift) & 3) as f32) - ml1);
                        }
                    }
                }
            }
        }
        11 => {
            // Q3_K: 32B hmask + 64B qs + 12B scales + 2B f16 d (110B / 256 elems)
            for blk in raw.chunks_exact(110) {
                let hm = &blk[0..32];
                let qs = &blk[32..96];
                let sc_raw = &blk[96..108];
                let d_all = f16_bits_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
                let mut aux = [0u32; 4];
                for i in 0..3 {
                    aux[i] = u32::from_le_bytes([
                        sc_raw[i * 4],
                        sc_raw[i * 4 + 1],
                        sc_raw[i * 4 + 2],
                        sc_raw[i * 4 + 3],
                    ]);
                }
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & 0x0F0F_0F0F) | (((tmp >> 4) & 0x0303_0303) << 4);
                aux[3] = ((aux[1] >> 4) & 0x0F0F_0F0F) | (((tmp >> 6) & 0x0303_0303) << 4);
                aux[0] = (aux[0] & 0x0F0F_0F0F) | ((tmp & 0x0303_0303) << 4);
                aux[1] = (aux[1] & 0x0F0F_0F0F) | (((tmp >> 2) & 0x0303_0303) << 4);
                let mut scales = [0i8; 16];
                for i in 0..4 {
                    let b = aux[i].to_le_bytes();
                    for j in 0..4 {
                        scales[i * 4 + j] = (b[j] as i8).wrapping_sub(32);
                    }
                }
                let mut m_bit = 1u8;
                let mut is = 0usize;
                for n in 0..2 {
                    let q = &qs[n * 32..(n + 1) * 32];
                    for shift in [0, 2, 4, 6] {
                        let dl0 = d_all * (scales[is] as f32);
                        is += 1;
                        for l in 0..16 {
                            let low = ((q[l] >> shift) & 3) as i32;
                            let sub = if (hm[l] & m_bit) != 0 { 0 } else { 4 };
                            out.push(dl0 * ((low - sub) as f32));
                        }
                        let dl1 = d_all * (scales[is] as f32);
                        is += 1;
                        for l in 16..32 {
                            let low = ((q[l] >> shift) & 3) as i32;
                            let sub = if (hm[l] & m_bit) != 0 { 0 } else { 4 };
                            out.push(dl1 * ((low - sub) as f32));
                        }
                        m_bit = m_bit.wrapping_shl(1);
                    }
                }
            }
        }
        12 => {
            // Q4_K: 2B f16 d + 2B f16 dmin + 12B scales + 128B qs (256 elems)
            for blk in raw.chunks_exact(144) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let sc = &blk[4..16];
                let qs = &blk[16..144];
                let get_scale_min = |is: usize| -> (f32, f32) {
                    if is < 4 {
                        ((sc[is] & 63) as f32, (sc[is + 4] & 63) as f32)
                    } else {
                        (
                            ((sc[is + 4] & 0x0F) | ((sc[is - 4] >> 6) << 4)) as f32,
                            ((sc[is + 4] >> 4) | ((sc[is] >> 6) << 4)) as f32,
                        )
                    }
                };
                for chunk_idx in 0..4 {
                    let (s1, m1) = get_scale_min(chunk_idx * 2);
                    let (s2, m2) = get_scale_min(chunk_idx * 2 + 1);
                    let q_slice = &qs[chunk_idx * 32..(chunk_idx + 1) * 32];
                    for &qb in q_slice {
                        out.push(d * s1 * ((qb & 0x0F) as f32) - dmin * m1);
                    }
                    for &qb in q_slice {
                        out.push(d * s2 * ((qb >> 4) as f32) - dmin * m2);
                    }
                }
            }
        }
        13 => {
            // Q5_K: 2B f16 d + 2B f16 dmin + 12B scales + 32B qh + 128B qs (176B / 256 elems)
            for blk in raw.chunks_exact(176) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let sc = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let get_scale_min = |is: usize| -> (f32, f32) {
                    if is < 4 {
                        ((sc[is] & 63) as f32, (sc[is + 4] & 63) as f32)
                    } else {
                        (
                            ((sc[is + 4] & 0x0F) | ((sc[is - 4] >> 6) << 4)) as f32,
                            ((sc[is + 4] >> 4) | ((sc[is] >> 6) << 4)) as f32,
                        )
                    }
                };
                let mut u1 = 1u8;
                let mut u2 = 2u8;
                for chunk_idx in 0..4 {
                    let (s1, m1) = get_scale_min(chunk_idx * 2);
                    let (s2, m2) = get_scale_min(chunk_idx * 2 + 1);
                    let q_slice = &qs[chunk_idx * 32..(chunk_idx + 1) * 32];
                    for (l, &qb) in q_slice.iter().enumerate() {
                        let hi = if (qh[l] & u1) != 0 { 16.0 } else { 0.0 };
                        out.push(d * s1 * (((qb & 0x0F) as f32) + hi) - dmin * m1);
                    }
                    for (l, &qb) in q_slice.iter().enumerate() {
                        let hi = if (qh[l] & u2) != 0 { 16.0 } else { 0.0 };
                        out.push(d * s2 * (((qb >> 4) as f32) + hi) - dmin * m2);
                    }
                    u1 = u1.wrapping_shl(2);
                    u2 = u2.wrapping_shl(2);
                }
            }
        }
        14 => {
            // Q6_K: 128B ql + 64B qh + 16B i8 scales + 2B f16 d (210B / 256 elems)
            for blk in raw.chunks_exact(210) {
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc = &blk[192..208];
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
                for n in 0..2 {
                    let ql_n = &ql[n * 64..(n + 1) * 64];
                    let qh_n = &qh[n * 32..(n + 1) * 32];
                    let sc_n = &sc[n * 8..(n + 1) * 8];
                    let mut sub = [0.0f32; 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql_n[l] & 0x0F) | ((qh_n[l] & 0x03) << 4)) as i32 - 32;
                        let q2 =
                            ((ql_n[l + 32] & 0x0F) | (((qh_n[l] >> 2) & 0x03) << 4)) as i32 - 32;
                        let q3 = ((ql_n[l] >> 4) | (((qh_n[l] >> 4) & 0x03) << 4)) as i32 - 32;
                        let q4 = ((ql_n[l + 32] >> 4) | (((qh_n[l] >> 6) & 0x03) << 4)) as i32 - 32;
                        sub[l] = d * (sc_n[is] as i8 as f32) * (q1 as f32);
                        sub[l + 32] = d * (sc_n[is + 2] as i8 as f32) * (q2 as f32);
                        sub[l + 64] = d * (sc_n[is + 4] as i8 as f32) * (q3 as f32);
                        sub[l + 96] = d * (sc_n[is + 6] as i8 as f32) * (q4 as f32);
                    }
                    out.extend_from_slice(&sub);
                }
            }
        }
        34 => {
            // TQ1_0: 48B qs + 4B qh + 2B f16 d (54B / 256 ternary {-1, 0, +1} elems)
            for blk in raw.chunks_exact(54) {
                let qs = &blk[0..48];
                let qh = &blk[48..52];
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[52], blk[53]]));
                let pow3 = [1u8, 3, 9, 27, 81];
                for &p in &pow3 {
                    for &b in &qs[0..32] {
                        let xi = b.wrapping_mul(p);
                        let trit = (((xi as u16) * 3) >> 8) as i32 - 1;
                        out.push(trit as f32 * d);
                    }
                }
                for &p in &pow3 {
                    for &b in &qs[32..48] {
                        let xi = b.wrapping_mul(p);
                        let trit = (((xi as u16) * 3) >> 8) as i32 - 1;
                        out.push(trit as f32 * d);
                    }
                }
                for shift in [0, 2, 4, 6] {
                    for &b in qh {
                        let trit = ((b >> shift) & 0x03) as i32 - 1;
                        out.push(trit.clamp(-1, 1) as f32 * d);
                    }
                }
            }
        }
        35 => {
            // TQ2_0: 64B 2-bit trits + 2B f16 d (66B / 256 elems)
            for blk in raw.chunks_exact(66) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[64], blk[65]]));
                for &b in &blk[0..64] {
                    for shift in [0, 2, 4, 6] {
                        let q = ((b >> shift) & 0x03) as i32 - 1;
                        out.push(q.clamp(-1, 1) as f32 * d);
                    }
                }
            }
        }
        36 => {
            // Microsoft BitNet.cpp `I2_S` (`elem_count / 4` packed 2-bit bytes in 32B/128-elem interleaved blocks + 4B f32 scale)
            let q_bytes = elem_count / 4;
            if raw.len() < q_bytes + 4 {
                return Err("GGUF I2_S buffer shorter than expected".to_string());
            }
            let d = f32::from_le_bytes([
                raw[q_bytes],
                raw[q_bytes + 1],
                raw[q_bytes + 2],
                raw[q_bytes + 3],
            ]);
            for blk in raw[0..q_bytes].chunks_exact(32) {
                let mut sub = [0.0f32; 128];
                for (j, &b) in blk.iter().enumerate() {
                    sub[j] = (((b >> 6) & 0x03) as i32 - 1).clamp(-1, 1) as f32 * d;
                    sub[32 + j] = (((b >> 4) & 0x03) as i32 - 1).clamp(-1, 1) as f32 * d;
                    sub[64 + j] = (((b >> 2) & 0x03) as i32 - 1).clamp(-1, 1) as f32 * d;
                    sub[96 + j] = ((b & 0x03) as i32 - 1).clamp(-1, 1) as f32 * d;
                }
                out.extend_from_slice(&sub);
            }
        }
        40 => {
            // Q1_0 (32-elem block: 2B f16 d + 4B bits)
            for blk in raw.chunks_exact(6) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &b in &blk[2..6] {
                    for bit in 0..8 {
                        out.push(if ((b >> bit) & 1) == 1 { d } else { -d });
                    }
                }
            }
        }
        41 => {
            // PrismML Bonsai Q1_0_G128 (128-elem block: 2B f16 d + 16B packed 1-bit signs)
            for blk in raw.chunks_exact(18) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &b in &blk[2..18] {
                    for bit in 0..8 {
                        out.push(if ((b >> bit) & 1) == 1 { d } else { -d });
                    }
                }
            }
        }
        42 => {
            // PrismML Ternary-Bonsai-2 `Q2_0` (64-elem block: 2B f16 d + 16B packed 2-bit trits in {-1, 0, +1})
            for blk in raw.chunks_exact(18) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &b in &blk[2..18] {
                    for shift in [0, 2, 4, 6] {
                        let q = ((b >> shift) & 0x03) as i32 - 1;
                        out.push(q.clamp(-1, 1) as f32 * d);
                    }
                }
            }
        }
        142 => {
            // PrismML Ternary-Bonsai-2 `PQ2_0` (128-elem block: 2B f16 d + 32B packed 2-bit trits in {-1, 0, +1})
            for blk in raw.chunks_exact(34) {
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &b in &blk[2..34] {
                    for shift in [0, 2, 4, 6] {
                        let q = ((b >> shift) & 0x03) as i32 - 1;
                        out.push(q.clamp(-1, 1) as f32 * d);
                    }
                }
            }
        }
        143 => {
            // PrismML Ternary-Bonsai-2 `PTQ1_0` (128-elem block: 24B base-3 qs + 2B qh + 2B f16 d)
            for blk in raw.chunks_exact(28) {
                let qs = &blk[0..24];
                let qh = &blk[24..26];
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[26], blk[27]]));
                let pow3 = [1u8, 3, 9, 27, 81];
                for &p in &pow3 {
                    for &b in qs {
                        let xi = b.wrapping_mul(p);
                        let trit = (((xi as u16) * 3) >> 8) as i32 - 1;
                        out.push(trit as f32 * d);
                    }
                }
                for shift in [0, 2, 4, 6] {
                    for &b in qh {
                        let trit = ((b >> shift) & 0x03) as i32 - 1;
                        out.push(trit.clamp(-1, 1) as f32 * d);
                    }
                }
            }
        }
        other => return Err(format!("unsupported GGUF dequant type_id {other}")),
    }
    Ok(out)
}

fn gguf_u32(data: &[u8], off: usize) -> Result<u32, String> {
    let sl = data
        .get(off..off.saturating_add(4))
        .filter(|s| s.len() == 4)
        .ok_or_else(|| "GGUF u32 out of bounds".to_string())?;
    Ok(u32::from_le_bytes([sl[0], sl[1], sl[2], sl[3]]))
}

fn gguf_f32(data: &[u8], off: usize) -> Result<f32, String> {
    let sl = data
        .get(off..off.saturating_add(4))
        .filter(|s| s.len() == 4)
        .ok_or_else(|| "GGUF f32 out of bounds".to_string())?;
    Ok(f32::from_le_bytes([sl[0], sl[1], sl[2], sl[3]]))
}

fn gguf_u64(data: &[u8], off: usize) -> Result<u64, String> {
    let sl = data
        .get(off..off.saturating_add(8))
        .filter(|s| s.len() == 8)
        .ok_or_else(|| "GGUF u64 out of bounds".to_string())?;
    let mut b = [0u8; 8];
    b.copy_from_slice(sl);
    Ok(u64::from_le_bytes(b))
}

fn gguf_str(data: &[u8], off: usize) -> Result<(String, usize), String> {
    let len = gguf_u64(data, off)? as usize;
    let start = off
        .checked_add(8)
        .ok_or_else(|| "GGUF str overflow".to_string())?;
    let end = start
        .checked_add(len)
        .filter(|&e| e <= data.len())
        .ok_or_else(|| "GGUF str out of bounds".to_string())?;
    Ok((String::from_utf8_lossy(&data[start..end]).to_string(), end))
}

fn gguf_skip_val(data: &[u8], off: usize, vtype: u32) -> Result<usize, String> {
    match vtype {
        0 | 1 | 7 => off
            .checked_add(1)
            .filter(|&p| p <= data.len())
            .ok_or_else(|| "GGUF val oob".to_string()),
        2 | 3 => off
            .checked_add(2)
            .filter(|&p| p <= data.len())
            .ok_or_else(|| "GGUF val oob".to_string()),
        4..=6 => off
            .checked_add(4)
            .filter(|&p| p <= data.len())
            .ok_or_else(|| "GGUF val oob".to_string()),
        10..=12 => off
            .checked_add(8)
            .filter(|&p| p <= data.len())
            .ok_or_else(|| "GGUF val oob".to_string()),
        8 => Ok(gguf_str(data, off)?.1),
        9 => {
            let et = gguf_u32(data, off)?;
            let cnt = gguf_u64(data, off.saturating_add(4))? as usize;
            let mut p = off
                .checked_add(12)
                .filter(|&x| x <= data.len())
                .ok_or_else(|| "GGUF array oob".to_string())?;
            for _ in 0..cnt {
                p = gguf_skip_val(data, p, et)?;
            }
            Ok(p)
        }
        other => Err(format!("unknown GGUF KV type {other}")),
    }
}

fn decode_tensor_to_f32(raw: &[u8], dtype: SafetensorDtype, elem_count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elem_count);
    match dtype {
        SafetensorDtype::F64 => {
            for chunk in raw.chunks_exact(8) {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(chunk);
                out.push(f64::from_le_bytes(arr) as f32);
            }
        }
        SafetensorDtype::F32 => {
            for chunk in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        SafetensorDtype::F16 => {
            for chunk in raw.chunks_exact(2) {
                out.push(f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        SafetensorDtype::BF16 => {
            for chunk in raw.chunks_exact(2) {
                let bits = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
                out.push(f32::from_bits(bits));
            }
        }
        SafetensorDtype::F8E4M3 => {
            for &b in raw {
                out.push(f8_e4m3_to_f32(b));
            }
        }
        SafetensorDtype::F8E5M2 => {
            for &b in raw {
                out.push(f8_e5m2_to_f32(b));
            }
        }
        SafetensorDtype::I64 => {
            for chunk in raw.chunks_exact(8) {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(chunk);
                out.push(i64::from_le_bytes(arr) as f32);
            }
        }
        SafetensorDtype::I32 => {
            for chunk in raw.chunks_exact(4) {
                out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
            }
        }
        SafetensorDtype::U32 => {
            for chunk in raw.chunks_exact(4) {
                out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
            }
        }
        SafetensorDtype::I16 => {
            for chunk in raw.chunks_exact(2) {
                out.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32);
            }
        }
        SafetensorDtype::U16 => {
            for chunk in raw.chunks_exact(2) {
                out.push(u16::from_le_bytes([chunk[0], chunk[1]]) as f32);
            }
        }
        SafetensorDtype::I8 => {
            for &b in raw {
                out.push((b as i8) as f32);
            }
        }
        SafetensorDtype::U8 | SafetensorDtype::Bool => {
            for &b in raw {
                out.push(b as f32);
            }
        }
    }
    out
}

fn extract_layer_number(name: &str) -> i64 {
    for part in name.split('.') {
        if let Ok(n) = part.parse::<i64>() {
            return n;
        }
    }
    -1
}

fn infer_powerinfer_hot_score(name: &str) -> f64 {
    let lower = name.to_ascii_lowercase();
    if lower.contains("embed") || lower.contains("router") || lower.contains("gate_inp") {
        0.95
    } else if lower.contains("attn")
        || lower.contains("q_proj")
        || lower.contains("k_proj")
        || lower.contains("v_proj")
    {
        0.82
    } else {
        0.15
    }
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 255 {
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        return sign;
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn fp4_e2m1_to_f32(nibble: u8) -> f32 {
    const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mag = LUT[(nibble & 0x07) as usize];
    if (nibble & 0x08) != 0 { -mag } else { mag }
}

#[inline]
fn f8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as f32;
    if exp == 15 && (b & 0x07) == 7 {
        0.0
    } else if exp == 0 {
        sign * (mant / 8.0) * 2.0f32.powi(-6)
    } else {
        sign * (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
    }
}

#[inline]
fn f32_to_f8_e4m3(val: f32) -> u8 {
    if !val.is_finite() || val == 0.0 {
        return 0;
    }
    let sign = if val < 0.0 { 0x80u8 } else { 0u8 };
    let a = val.abs().min(448.0);
    let mut best_code = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..127u8 {
        let dec = f8_e4m3_to_f32(code);
        let err = (dec - a).abs();
        if err < best_err {
            best_err = err;
            best_code = code;
        }
    }
    sign | best_code
}

#[inline]
fn f8_e5m2_to_f32(b: u8) -> f32 {
    f16_bits_to_f32((b as u16) << 8)
}

#[inline]
fn f32_to_f8_e5m2(val: f32) -> u8 {
    (f32_to_f16_bits(val) >> 8) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_end_to_end_living_ingest_sub_1_bit_and_engram() {
        let mut meta = BTreeMap::new();
        meta.insert("architecture".to_string(), "Qwen3ForCausalLM".to_string());
        meta.insert("format".to_string(), "pt".to_string());

        let rows = 128usize;
        let cols = 128usize;
        let mut q_weights = Vec::with_capacity(rows * cols);
        let mut mlp_weights = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                let s = if (i + j) % 2 == 0 { 1.25f32 } else { -1.25f32 };
                q_weights.push(s);
                mlp_weights
                    .push(if i < 64 { 0.85 } else { -0.85 } * if j % 2 == 0 { 1.1 } else { -1.1 });
            }
        }

        let entries = vec![
            SafetensorBuildEntry {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                dtype: SafetensorDtype::F16,
                shape: vec![rows, cols],
                values_f32: q_weights,
            },
            SafetensorBuildEntry {
                name: "model.layers.0.mlp.down_proj.weight".to_string(),
                dtype: SafetensorDtype::BF16,
                shape: vec![rows, cols],
                values_f32: mlp_weights,
            },
            SafetensorBuildEntry {
                name: "model.layers.0.input_layernorm.weight".to_string(),
                dtype: SafetensorDtype::F32,
                shape: vec![rows],
                values_f32: vec![1.0; rows],
            },
        ];

        let st_bytes = build_safetensors_bytes(&meta, &entries).expect("build safetensors");
        let mut harness = JouleHarnessEngine::new(16, 16, 32 * 1024 * 1024).expect("harness");

        let report = ingest_safetensors_into_harness("tiny-qwen3", &st_bytes, 8, 8, &mut harness)
            .expect("ingest");

        assert_eq!(report.total_tensors, 3);
        assert_eq!(report.factorized_2d_tensors, 2);
        assert!(report.overall_bits_per_weight < 0.60);
        assert!(report.mean_reconstruction_cosine > 0.95);
        assert!(
            harness
                .engram
                .lookup("tiny-qwen3 metadata architecture")
                .is_some()
        );
    }
}
