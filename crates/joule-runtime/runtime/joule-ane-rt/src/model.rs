//! Core ML ML Program (`mlprogram`) generator for HDC similarity.
//!
//! Produces an `.mlpackage` directory with no Python and no coremltools:
//!
//! ```text
//! <name>.mlpackage/
//!   Manifest.json
//!   Data/com.apple.CoreML/model.mlmodel        (CoreML.Specification.Model protobuf)
//!   Data/com.apple.CoreML/weights/weight.bin   (MIL blob storage v2, one fp16 blob)
//! ```
//!
//! The protobuf field numbers come from Apple's published Core ML format
//! (`coremltools/mlmodel/format/{Model,FeatureTypes,MIL}.proto`). Only the
//! fields this model needs are written. Layout mirrors what coremltools 9
//! emits for the same MIL program (checked on the box with coremltools'
//! own protobuf parser, see `scripts/verify_mlpackage.py`).
//!
//! Two graph layouts compute the same scores:
//!
//! * [`Layout::MatMul`]: `scores[n,k] = matmul(q[n,d], codebook[k,d], transpose_y=true)`.
//! * [`Layout::ChannelsFirstConv`]: the Apple ml-ane-transformers layout.
//!   `q` is `(1, d, 1, n)` (B, C, 1, S); a 1x1 `conv` with weight
//!   `(k, d, 1, 1)` gives `(1, k, 1, n)`.
//!
//! Everything is FP16 (inputs, weights, outputs), which is what the Neural
//! Engine requires. For +-1 inputs and `d <= 2048` every partial sum is an
//! integer of magnitude at most 2048, which FP16 represents exactly.

use crate::pb::Pb;
use crate::{AneError, Layout};
use std::path::Path;

// CoreML.Specification.ArrayFeatureType.ArrayDataType
const ARRAY_FLOAT16: i64 = 65552;
// CoreML.Specification.MILSpec.DataType
const MIL_BOOL: i64 = 1;
const MIL_STRING: i64 = 2;
const MIL_FLOAT16: i64 = 10;
const MIL_INT32: i64 = 23;
// Specification version 7 = iOS 16 / macOS 13 (Core ML 6): FLOAT16 arrays.
const SPEC_VERSION: i64 = 7;
const OPSET: &str = "CoreML6";
pub(crate) const INPUT_NAME: &str = "q";
pub(crate) const OUTPUT_NAME: &str = "scores";
const WEIGHT_NAME: &str = "codebook";
const BLOB_PATH: &str = "@model_path/weights/weight.bin";
/// Offset of the first blob's metadata record inside weight.bin.
const BLOB_OFFSET: u64 = 64;

/// Static shape of one compiled similarity model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelShape {
    pub layout: Layout,
    /// Number of query vectors.
    pub n: usize,
    /// Dimensions per vector in this model (one d-tile).
    pub d: usize,
    /// Number of codebook (item-memory) vectors.
    pub k: usize,
}

impl ModelShape {
    pub fn input_shape(&self) -> Vec<i64> {
        match self.layout {
            Layout::MatMul => vec![self.n as i64, self.d as i64],
            Layout::ChannelsFirstConv => vec![1, self.d as i64, 1, self.n as i64],
        }
    }

    pub fn output_shape(&self) -> Vec<i64> {
        match self.layout {
            Layout::MatMul => vec![self.n as i64, self.k as i64],
            Layout::ChannelsFirstConv => vec![1, self.k as i64, 1, self.n as i64],
        }
    }

    fn weight_shape(&self) -> Vec<i64> {
        match self.layout {
            Layout::MatMul => vec![self.k as i64, self.d as i64],
            Layout::ChannelsFirstConv => vec![self.k as i64, self.d as i64, 1, 1],
        }
    }
}

fn tensor_type(dtype: i64, shape: &[i64]) -> Pb {
    let mut t = Pb::new().varint(1, dtype);
    if !shape.is_empty() {
        t = t.varint(2, shape.len() as i64);
    }
    for dim in shape {
        // Dimension { constant = 1 { size = 1 } }
        t = t.msg(3, Pb::new().msg(1, Pb::new().varint(1, *dim)));
    }
    t
}

fn value_type(dtype: i64, shape: &[i64]) -> Pb {
    Pb::new().msg(1, tensor_type(dtype, shape))
}

fn named_value_type(name: &str, dtype: i64, shape: &[i64]) -> Pb {
    Pb::new().string(1, name).msg(2, value_type(dtype, shape))
}

/// `Value` carrying an immediate tensor. `tensor_value` is a `TensorValue`.
fn immediate(dtype: i64, shape: &[i64], tensor_value: Pb) -> Pb {
    Pb::new()
        .msg(2, value_type(dtype, shape))
        .msg(3, Pb::new().msg(1, tensor_value))
}

fn string_value(s: &str) -> Pb {
    // TensorValue.strings = 4 { values = 1 }
    immediate(MIL_STRING, &[], Pb::new().msg(4, Pb::new().string(1, s)))
}

fn ints_value(values: &[i64]) -> Pb {
    let shape: Vec<i64> = if values.len() == 1 {
        Vec::new()
    } else {
        vec![values.len() as i64]
    };
    // TensorValue.ints = 2 { packed values = 1 }
    immediate(MIL_INT32, &shape, Pb::new().msg(2, Pb::new().packed(1, values)))
}

fn bool_value(v: bool) -> Pb {
    // TensorValue.bools = 3 { packed values = 1 }
    immediate(MIL_BOOL, &[], Pb::new().msg(3, Pb::new().packed(1, &[i64::from(v)])))
}

/// `const` op. `val` is the Value; `out_dtype`/`out_shape` describe it.
fn const_op(name: &str, out_dtype: i64, out_shape: &[i64], val: Pb) -> Pb {
    Pb::new()
        .string(1, "const")
        .msg(3, named_value_type(name, out_dtype, out_shape))
        .map_entry(5, "val", val)
        .map_entry(5, "name", string_value(name))
}

fn arg(name: &str) -> Pb {
    // Argument { arguments = 1: Binding { name = 1 } }
    Pb::new().msg(1, Pb::new().string(1, name))
}

fn blob_value(shape: &[i64]) -> Pb {
    // Value { type = 2, blobFileValue = 5 { fileName = 1, offset = 2 } }
    Pb::new().msg(2, value_type(MIL_FLOAT16, shape)).msg(
        5,
        Pb::new().string(1, BLOB_PATH).varint(2, BLOB_OFFSET as i64),
    )
}

fn array_feature(name: &str, shape: &[i64]) -> Pb {
    // FeatureDescription { name = 1, type = 3: FeatureType { multiArrayType = 5:
    //   ArrayFeatureType { shape = 1 (packed int64), dataType = 2 } } }
    Pb::new().string(1, name).msg(
        3,
        Pb::new().msg(5, Pb::new().packed(1, shape).varint(2, ARRAY_FLOAT16)),
    )
}

/// Serialize the `CoreML.Specification.Model` protobuf for `shape`.
pub fn model_spec_bytes(shape: &ModelShape) -> Vec<u8> {
    let in_shape = shape.input_shape();
    let out_shape = shape.output_shape();
    let w_shape = shape.weight_shape();

    let mut ops = vec![const_op(WEIGHT_NAME, MIL_FLOAT16, &w_shape, blob_value(&w_shape))];
    let similarity = match shape.layout {
        Layout::MatMul => {
            ops.push(const_op("transpose_x_0", MIL_BOOL, &[], bool_value(false)));
            ops.push(const_op("transpose_y_0", MIL_BOOL, &[], bool_value(true)));
            Pb::new()
                .string(1, "matmul")
                .map_entry(2, "x", arg(INPUT_NAME))
                .map_entry(2, "y", arg(WEIGHT_NAME))
                .map_entry(2, "transpose_x", arg("transpose_x_0"))
                .map_entry(2, "transpose_y", arg("transpose_y_0"))
                .msg(3, named_value_type(OUTPUT_NAME, MIL_FLOAT16, &out_shape))
                .map_entry(5, "name", string_value(OUTPUT_NAME))
        }
        Layout::ChannelsFirstConv => {
            ops.push(const_op("strides_0", MIL_INT32, &[2], ints_value(&[1, 1])));
            ops.push(const_op("pad_type_0", MIL_STRING, &[], string_value("valid")));
            ops.push(const_op("pad_0", MIL_INT32, &[4], ints_value(&[0, 0, 0, 0])));
            ops.push(const_op("dilations_0", MIL_INT32, &[2], ints_value(&[1, 1])));
            ops.push(const_op("groups_0", MIL_INT32, &[], ints_value(&[1])));
            Pb::new()
                .string(1, "conv")
                .map_entry(2, "x", arg(INPUT_NAME))
                .map_entry(2, "weight", arg(WEIGHT_NAME))
                .map_entry(2, "strides", arg("strides_0"))
                .map_entry(2, "pad_type", arg("pad_type_0"))
                .map_entry(2, "pad", arg("pad_0"))
                .map_entry(2, "dilations", arg("dilations_0"))
                .map_entry(2, "groups", arg("groups_0"))
                .msg(3, named_value_type(OUTPUT_NAME, MIL_FLOAT16, &out_shape))
                .map_entry(5, "name", string_value(OUTPUT_NAME))
        }
    };
    ops.push(similarity);

    let mut block = Pb::new().string(2, OUTPUT_NAME);
    for op in ops {
        block = block.msg(3, op);
    }
    let function = Pb::new()
        .msg(1, named_value_type(INPUT_NAME, MIL_FLOAT16, &in_shape))
        .string(2, OPSET)
        .map_entry(3, OPSET, block);
    let program = Pb::new().varint(1, 1).map_entry(2, "main", function);

    let metadata = Pb::new()
        .string(1, "JouleDB HDC similarity: scores = Q(+-1) . M(+-1)^T, Hamming = (d - score) / 2")
        .string(3, "joule-ane-rt");
    let description = Pb::new()
        .msg(1, array_feature(INPUT_NAME, &in_shape))
        .msg(10, array_feature(OUTPUT_NAME, &out_shape))
        .msg(100, metadata);

    Pb::new()
        .varint(1, SPEC_VERSION)
        .msg(2, description)
        .msg(502, program)
        .0
}

/// MIL blob storage v2 with one FP16 blob holding `codebook` (row-major
/// `k x d`, which is also the `(k, d, 1, 1)` conv weight layout).
pub fn weight_blob(codebook_f16: &[u16]) -> Vec<u8> {
    let data_offset: u64 = 128;
    let size = (codebook_f16.len() * 2) as u64;
    let mut out = Vec::with_capacity(128 + size as usize + 64);
    // File header: count = 1, version = 2, padded to 64 bytes.
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.resize(64, 0);
    // Blob metadata: sentinel, data type (1 = fp16), size, data offset.
    out.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&data_offset.to_le_bytes());
    out.resize(data_offset as usize, 0);
    for h in codebook_f16 {
        out.extend_from_slice(&h.to_le_bytes());
    }
    let padded = out.len().div_ceil(64) * 64;
    out.resize(padded, 0);
    out
}

fn uuid_from(seed: u64, salt: u64) -> String {
    let a = splitmix(seed ^ salt);
    let b = splitmix(a);
    format!(
        "{:08X}-{:04X}-4{:03X}-{:04X}-{:012X}",
        (a >> 32) as u32,
        (a >> 16) as u16,
        (a & 0x0fff) as u16,
        ((b >> 48) as u16 & 0x3fff) | 0x8000,
        b & 0xffff_ffff_ffff
    )
}

pub(crate) fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn manifest_json(seed: u64) -> String {
    let model_id = uuid_from(seed, 1);
    let weights_id = uuid_from(seed, 2);
    format!(
        r#"{{
    "fileFormatVersion": "1.0.0",
    "itemInfoEntries": {{
        "{weights_id}": {{
            "author": "com.apple.CoreML",
            "description": "CoreML Model Weights",
            "name": "weights",
            "path": "com.apple.CoreML/weights"
        }},
        "{model_id}": {{
            "author": "com.apple.CoreML",
            "description": "CoreML Model Specification",
            "name": "model.mlmodel",
            "path": "com.apple.CoreML/model.mlmodel"
        }}
    }},
    "rootModelIdentifier": "{model_id}"
}}
"#
    )
}

/// Write `<dir>` as an `.mlpackage` for `shape` with the given codebook tile
/// (row-major `k x d` of +-1 values).
pub fn write_mlpackage(dir: &Path, shape: &ModelShape, codebook: &[i8]) -> Result<(), AneError> {
    if codebook.len() != shape.k * shape.d {
        return Err(AneError::InvalidInput(format!(
            "codebook tile has {} values, expected k*d = {}",
            codebook.len(),
            shape.k * shape.d
        )));
    }
    let io = |e: std::io::Error| AneError::Io(format!("{}: {e}", dir.display()));
    let data = dir.join("Data").join("com.apple.CoreML");
    std::fs::create_dir_all(data.join("weights")).map_err(io)?;
    let f16: Vec<u16> = codebook
        .iter()
        .map(|v| half::f16::from_f32(f32::from(*v)).to_bits())
        .collect();
    let seed = crate::fingerprint(shape, codebook);
    std::fs::write(data.join("model.mlmodel"), model_spec_bytes(shape)).map_err(io)?;
    std::fs::write(data.join("weights").join("weight.bin"), weight_blob(&f16)).map_err(io)?;
    std::fs::write(dir.join("Manifest.json"), manifest_json(seed)).map_err(io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_blob_layout_matches_mil_storage_v2() {
        let blob = weight_blob(&[0x3c00, 0xbc00, 0x3c00]);
        assert_eq!(&blob[0..8], &[1, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(&blob[64..68], &0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(&blob[68..72], &1u32.to_le_bytes());
        assert_eq!(&blob[72..80], &6u64.to_le_bytes());
        assert_eq!(&blob[80..88], &128u64.to_le_bytes());
        assert_eq!(&blob[128..134], &[0x00, 0x3c, 0x00, 0xbc, 0x00, 0x3c]);
        assert_eq!(blob.len() % 64, 0);
    }

    #[test]
    fn spec_starts_with_version_and_contains_program() {
        let shape = ModelShape { layout: Layout::MatMul, n: 2, d: 8, k: 3 };
        let bytes = model_spec_bytes(&shape);
        assert_eq!(&bytes[0..2], &[0x08, 0x07]);
        let hay = String::from_utf8_lossy(&bytes);
        for needle in ["matmul", "CoreML6", "codebook", "@model_path/weights/weight.bin", "scores"] {
            assert!(hay.contains(needle), "missing {needle}");
        }
        let conv = model_spec_bytes(&ModelShape { layout: Layout::ChannelsFirstConv, ..shape });
        assert!(String::from_utf8_lossy(&conv).contains("conv"));
        assert!(String::from_utf8_lossy(&conv).contains("valid"));
    }

    #[test]
    fn manifest_ids_are_uuid_shaped_and_distinct() {
        let m = manifest_json(42);
        assert!(m.contains("rootModelIdentifier"));
        let a = uuid_from(42, 1);
        let b = uuid_from(42, 2);
        assert_ne!(a, b);
        assert_eq!(a.len(), 36);
        assert_eq!(a.matches('-').count(), 4);
    }
}
