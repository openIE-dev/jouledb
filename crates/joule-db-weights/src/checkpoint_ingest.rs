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

use crate::synapse::{JouleHarnessEngine, LittleBitSVFTensor, WeightShardMeta};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Safety bound on `.safetensors` JSON header size (16 MiB) against hostile OOM headers.
const MAX_SAFETENSORS_HEADER_BYTES: usize = 16 * 1024 * 1024;
/// Safety bound on single tensor element count during in-memory factorization (64M floats).
const MAX_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024;

/// Supported `.safetensors` element types for ingestion and serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetensorDtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
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
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::I16 => "I16",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::Bool => "BOOL",
        }
    }

    pub fn bytes_per_elem(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    pub fn is_floating_point(self) -> bool {
        matches!(self, Self::F64 | Self::F32 | Self::F16 | Self::BF16)
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "F64" => Ok(Self::F64),
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "I64" => Ok(Self::I64),
            "I32" => Ok(Self::I32),
            "I16" => Ok(Self::I16),
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
        header_map.insert("__metadata__".to_string(), serde_json::Value::Object(meta_obj));
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
            SafetensorDtype::I16 => {
                for &v in &entry.values_f32 {
                    data_buf.extend_from_slice(&(v as i16).to_le_bytes());
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

    // Sort tensor names deterministically
    let mut tensor_keys: Vec<&String> = header_obj
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .collect();
    tensor_keys.sort();

    for tname in tensor_keys {
        let Some(desc) = header_obj.get(tname).and_then(|v| v.as_object()) else {
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

        let elem_count = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
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

        let decoded = decode_tensor_to_f32(&raw_bytes[abs_start..abs_end], dtype, elem_count);
        let layer_idx = extract_layer_number(tname);
        let hot_score = infer_powerinfer_hot_score(tname);
        let fp16_equiv_bytes = (elem_count as u64) * 2;
        total_fp16_bytes += fp16_equiv_bytes;

        if dtype.is_floating_point()
            && shape.len() >= 2
            && shape[0] >= 2
            && (elem_count / shape[0]) >= 2
        {
            let rows = shape[0];
            let cols = elem_count / rows;
            let eff_rank = target_rank.clamp(1, rows.min(cols));
            let fq_name = format!("{model_id}::{tname}");
            let lb = LittleBitSVFTensor::factorize(
                fq_name.clone(),
                &decoded,
                rows,
                cols,
                eff_rank,
                6,
                max_outliers,
            )?;
            let bpw = lb.bits_per_weight();
            let comp = lb.compression_vs_fp16();
            let cos = lb.reconstruction_cosine;
            let comp_bytes = (((elem_count as f64) * bpw) / 8.0).ceil() as u64;

            // Register compressed shard into AirStreamScheduler (if within VRAM budget)
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
                source_dtype: dtype.as_str().to_string(),
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
            // 1D norm / bias vector: record statistical digest in Engram + keep FP16 size
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
                source_dtype: dtype.as_str().to_string(),
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
        tensors: summaries,
    })
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
        SafetensorDtype::I16 => {
            for chunk in raw.chunks_exact(2) {
                out.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32);
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
    } else if lower.contains("attn") || lower.contains("q_proj") || lower.contains("k_proj") || lower.contains("v_proj") {
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
                mlp_weights.push(if i < 64 { 0.85 } else { -0.85 } * if j % 2 == 0 { 1.1 } else { -1.1 });
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

        let report =
            ingest_safetensors_into_harness("tiny-qwen3", &st_bytes, 8, 8, &mut harness)
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
