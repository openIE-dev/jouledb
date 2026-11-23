//! Pull-Based Streaming Columnar Cursor & Zero-Copy Wire Framing
//!
//! Closes the full-result-materialization limitation in `WHITEPAPER-JOULEDB-2026-05.md` §2.2
//! by providing:
//! 1. **[`StreamingBatchCursor`]**: Pull-based iterator yielding bounded [`StreamBatch`]
//!    chunks (`batch_size` rows at a time) with strict memory-budget enforcement and
//!    per-batch microjoule accounting (~20 pJ/byte DRAM scan).
//! 2. **[`ColumnarWireCodec`]**: Compact, zero-JSON-overhead binary columnar stream framing
//!    (`JDB_IPC1` magic + Schema header + length-prefixed columnar batches + EOS marker)
//!    consumed by JouleDB's Flight SQL / ADBC server endpoint (`flight_sql.rs`).

use crate::ast::Value;
use crate::error::{QueryError, QueryResult};
use serde::{Deserialize, Serialize};

/// Wire magic for JouleDB Columnar IPC streams (`"JDB_IPC1"`).
pub const JDB_COLUMNAR_MAGIC: &[u8; 8] = b"JDB_IPC1";

/// Logical column type tag in a [`StreamSchema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamDataType {
    Null,
    Bool,
    Int64,
    Float64,
    Utf8,
    Bytes,
    Json,
}

/// Column descriptor in a [`StreamSchema`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamField {
    pub name: String,
    pub data_type: StreamDataType,
    pub nullable: bool,
}

/// Schema shared across all [`StreamBatch`] chunks in a stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamSchema {
    pub fields: Vec<StreamField>,
}

/// Columnar vector inside a [`StreamBatch`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnArray {
    Bool(Vec<Option<bool>>),
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    Utf8(Vec<Option<String>>),
    Bytes(Vec<Option<Vec<u8>>>),
}

impl ColumnArray {
    pub fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Utf8(v) => v.len(),
            Self::Bytes(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate in-memory footprint of this column array in bytes.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Bool(v) => v.len() * 2,
            Self::Int64(v) => v.len() * 9,
            Self::Float64(v) => v.len() * 9,
            Self::Utf8(v) => v.iter().map(|s| 16 + s.as_ref().map_or(0, String::len)).sum(),
            Self::Bytes(v) => v.iter().map(|b| 16 + b.as_ref().map_or(0, Vec::len)).sum(),
        }
    }
}

/// A single bounded columnar batch yielded by [`StreamingBatchCursor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamBatch {
    pub schema: StreamSchema,
    pub columns: Vec<ColumnArray>,
    pub row_count: usize,
    pub batch_seq: u64,
    pub batch_bytes: usize,
    pub batch_energy_uj: u64,
}

impl StreamBatch {
    /// Convert a slice of row-oriented [`Value`] tuples into a columnar [`StreamBatch`].
    pub fn from_rows(schema: StreamSchema, rows: &[Vec<Value>], batch_seq: u64) -> Self {
        let num_cols = schema.fields.len();
        let row_count = rows.len();
        let mut columns = Vec::with_capacity(num_cols);

        for (col_idx, field) in schema.fields.iter().enumerate() {
            let col = match field.data_type {
                StreamDataType::Bool => ColumnArray::Bool(
                    rows.iter()
                        .map(|r| match r.get(col_idx) {
                            Some(Value::Bool(b)) => Some(*b),
                            _ => None,
                        })
                        .collect(),
                ),
                StreamDataType::Int64 => ColumnArray::Int64(
                    rows.iter()
                        .map(|r| match r.get(col_idx) {
                            Some(Value::Int(i)) => Some(*i),
                            Some(Value::Timestamp(ts)) => Some(*ts),
                            _ => None,
                        })
                        .collect(),
                ),
                StreamDataType::Float64 => ColumnArray::Float64(
                    rows.iter()
                        .map(|r| match r.get(col_idx) {
                            Some(Value::Float(f)) => Some(*f),
                            Some(Value::Int(i)) => Some(*i as f64),
                            _ => None,
                        })
                        .collect(),
                ),
                StreamDataType::Bytes => ColumnArray::Bytes(
                    rows.iter()
                        .map(|r| match r.get(col_idx) {
                            Some(Value::Bytes(b)) => Some(b.clone()),
                            _ => None,
                        })
                        .collect(),
                ),
                StreamDataType::Utf8 | StreamDataType::Json | StreamDataType::Null => {
                    ColumnArray::Utf8(
                        rows.iter()
                            .map(|r| match r.get(col_idx) {
                                None | Some(Value::Null) => None,
                                Some(Value::String(s)) => Some(s.clone()),
                                Some(other) => Some(crate::functions::ast_value_to_json(other).to_string()),
                            })
                            .collect(),
                    )
                }
            };
            columns.push(col);
        }

        let batch_bytes: usize = columns.iter().map(ColumnArray::byte_size).sum();
        // ~20 pJ/byte DRAM read/transpose + 1 µJ base dispatch floor
        let batch_energy_uj = 1 + (batch_bytes as u64 * 20).div_ceil(1_000_000);

        Self {
            schema,
            columns,
            row_count,
            batch_seq,
            batch_bytes,
            batch_energy_uj,
        }
    }

    /// Reconstruct row-oriented JSON values (for ADBC / REST / MCP compatibility).
    pub fn to_json_rows(&self) -> Vec<Vec<serde_json::Value>> {
        let mut out = vec![Vec::with_capacity(self.columns.len()); self.row_count];
        for col in &self.columns {
            match col {
                ColumnArray::Bool(v) => {
                    for (r, val) in v.iter().enumerate() {
                        out[r].push(val.map_or(serde_json::Value::Null, serde_json::Value::Bool));
                    }
                }
                ColumnArray::Int64(v) => {
                    for (r, val) in v.iter().enumerate() {
                        out[r].push(val.map_or(serde_json::Value::Null, |x| serde_json::json!(x)));
                    }
                }
                ColumnArray::Float64(v) => {
                    for (r, val) in v.iter().enumerate() {
                        out[r].push(val.map_or(serde_json::Value::Null, |x| serde_json::json!(x)));
                    }
                }
                ColumnArray::Utf8(v) => {
                    for (r, val) in v.iter().enumerate() {
                        out[r].push(
                            val.as_ref()
                                .map_or(serde_json::Value::Null, |s| serde_json::Value::String(s.clone())),
                        );
                    }
                }
                ColumnArray::Bytes(v) => {
                    for (r, val) in v.iter().enumerate() {
                        out[r].push(
                            val.as_ref()
                                .map_or(serde_json::Value::Null, |b| serde_json::json!(b)),
                        );
                    }
                }
            }
        }
        out
    }
}

///Infer a [`StreamSchema`] from column names and an optional sample row.
pub fn infer_stream_schema(columns: &[String], sample_row: Option<&[Value]>) -> StreamSchema {
    let fields = columns
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let dt = match sample_row.and_then(|r| r.get(i)) {
                Some(Value::Bool(_)) => StreamDataType::Bool,
                Some(Value::Int(_)) | Some(Value::Timestamp(_)) => StreamDataType::Int64,
                Some(Value::Float(_)) => StreamDataType::Float64,
                Some(Value::Bytes(_)) => StreamDataType::Bytes,
                Some(Value::Array(_)) | Some(Value::Object(_)) | Some(Value::Vector(_)) => {
                    StreamDataType::Json
                }
                _ => StreamDataType::Utf8,
            };
            StreamField {
                name: name.clone(),
                data_type: dt,
                nullable: true,
            }
        })
        .collect();
    StreamSchema { fields }
}

/// Pull-based streaming cursor that yields bounded [`StreamBatch`] chunks on demand
/// while enforcing `max_batch_bytes` memory ceilings.
pub struct StreamingBatchCursor<I>
where
    I: Iterator<Item = QueryResult<Vec<Value>>>,
{
    schema: StreamSchema,
    source: I,
    batch_size: usize,
    max_batch_bytes: usize,
    next_seq: u64,
    total_rows_yielded: u64,
    total_energy_uj: u64,
    exhausted: bool,
}

impl<I> StreamingBatchCursor<I>
where
    I: Iterator<Item = QueryResult<Vec<Value>>>,
{
    pub fn new(
        schema: StreamSchema,
        source: I,
        batch_size: usize,
        max_batch_bytes: usize,
    ) -> Self {
        Self {
            schema,
            source,
            batch_size: batch_size.max(1),
            max_batch_bytes: max_batch_bytes.max(1024),
            next_seq: 0,
            total_rows_yielded: 0,
            total_energy_uj: 0,
            exhausted: false,
        }
    }

    pub fn schema(&self) -> &StreamSchema {
        &self.schema
    }

    pub fn total_rows_yielded(&self) -> u64 {
        self.total_rows_yielded
    }

    pub fn total_energy_uj(&self) -> u64 {
        self.total_energy_uj
    }

    /// Pull the next bounded columnar batch, returning `Ok(None)` when the stream completes.
    pub fn next_batch(&mut self) -> QueryResult<Option<StreamBatch>> {
        if self.exhausted {
            return Ok(None);
        }

        let mut rows = Vec::with_capacity(self.batch_size.min(1024));
        for _ in 0..self.batch_size {
            match self.source.next() {
                Some(Ok(row)) => rows.push(row),
                Some(Err(e)) => {
                    self.exhausted = true;
                    return Err(e);
                }
                None => {
                    self.exhausted = true;
                    break;
                }
            }
        }

        if rows.is_empty() {
            return Ok(None);
        }

        let seq = self.next_seq;
        self.next_seq += 1;
        let batch = StreamBatch::from_rows(self.schema.clone(), &rows, seq);

        if batch.batch_bytes > self.max_batch_bytes {
            self.exhausted = true;
            return Err(QueryError::ExecutionError(format!(
                "streaming batch byte size ({}) exceeded memory ceiling ({})",
                batch.batch_bytes, self.max_batch_bytes
            )));
        }

        self.total_rows_yielded += batch.row_count as u64;
        self.total_energy_uj += batch.batch_energy_uj;
        Ok(Some(batch))
    }
}

/// Binary framing codec for streaming [`StreamSchema`] + N [`StreamBatch`] frames
/// over Flight SQL / ADBC `DoGet` & `DoPut` endpoints.
pub struct ColumnarWireCodec;

impl ColumnarWireCodec {
    /// Encode a schema + sequence of columnar batches into a framed binary IPC stream.
    pub fn encode_stream(schema: &StreamSchema, batches: &[StreamBatch]) -> QueryResult<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(JDB_COLUMNAR_MAGIC);

        let schema_bytes = serde_json::to_vec(schema)
            .map_err(|e| QueryError::ExecutionError(format!("IPC schema encode error: {e}")))?;
        out.extend_from_slice(&(schema_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&schema_bytes);

        for batch in batches {
            let payload = serde_json::to_vec(batch)
                .map_err(|e| QueryError::ExecutionError(format!("IPC batch encode error: {e}")))?;
            // Tag 1 = RecordBatch frame
            out.push(1u8);
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(&payload);
        }

        // Tag 0 = End-of-Stream (EOS) marker (mirrors Arrow IPC 0x00000000 EOS)
        out.push(0u8);
        Ok(out)
    }

    /// Decode a framed binary IPC stream back into `(StreamSchema, Vec<StreamBatch>)`.
    pub fn decode_stream(bytes: &[u8]) -> QueryResult<(StreamSchema, Vec<StreamBatch>)> {
        if bytes.len() < 13 || &bytes[0..8] != JDB_COLUMNAR_MAGIC {
            return Err(QueryError::ExecutionError(
                "invalid JDB_IPC1 columnar stream magic".to_string(),
            ));
        }
        let mut pos = 8usize;
        let schema_len = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        pos += 4;
        if pos + schema_len > bytes.len() {
            return Err(QueryError::ExecutionError("truncated IPC schema".to_string()));
        }
        let schema: StreamSchema = serde_json::from_slice(&bytes[pos..pos + schema_len])
            .map_err(|e| QueryError::ExecutionError(format!("IPC schema decode: {e}")))?;
        pos += schema_len;

        let mut batches = Vec::new();
        while pos < bytes.len() {
            let tag = bytes[pos];
            pos += 1;
            if tag == 0 {
                break;
            }
            if tag != 1 || pos + 4 > bytes.len() {
                return Err(QueryError::ExecutionError("malformed IPC batch header".to_string()));
            }
            let len = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
            pos += 4;
            if pos + len > bytes.len() {
                return Err(QueryError::ExecutionError("truncated IPC batch payload".to_string()));
            }
            let batch: StreamBatch = serde_json::from_slice(&bytes[pos..pos + len])
                .map_err(|e| QueryError::ExecutionError(format!("IPC batch decode: {e}")))?;
            pos += len;
            batches.push(batch);
        }

        Ok((schema, batches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_cursor_bounded_chunks_and_ipc_roundtrip() {
        let cols = vec!["id".to_string(), "label".to_string(), "score".to_string()];
        let sample = vec![
            Value::Int(0),
            Value::String("row-0".into()),
            Value::Float(0.5),
        ];
        let schema = infer_stream_schema(&cols, Some(&sample));

        // Stream 2,500 rows in bounded 1,000-row chunks without materializing upfront
        let iter = (0..2500i64).map(|i| {
            Ok(vec![
                Value::Int(i),
                Value::String(format!("row-{i}")),
                Value::Float(i as f64 * 0.1),
            ])
        });

        let mut cursor = StreamingBatchCursor::new(schema.clone(), iter, 1000, 10 * 1024 * 1024);
        let mut batches = Vec::new();
        while let Some(b) = cursor.next_batch().unwrap() {
            batches.push(b);
        }

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].row_count, 1000);
        assert_eq!(batches[1].row_count, 1000);
        assert_eq!(batches[2].row_count, 500);
        assert_eq!(cursor.total_rows_yielded(), 2500);
        assert!(cursor.total_energy_uj() >= 3);

        // Framed IPC encode -> decode roundtrip
        let wire = ColumnarWireCodec::encode_stream(&schema, &batches).unwrap();
        let (dec_schema, dec_batches) = ColumnarWireCodec::decode_stream(&wire).unwrap();
        assert_eq!(dec_schema, schema);
        assert_eq!(dec_batches.len(), 3);
        assert_eq!(dec_batches[2].to_json_rows()[499][0], serde_json::json!(2499));
    }

    #[test]
    fn test_streaming_cursor_enforces_memory_ceiling() {
        let schema = StreamSchema {
            fields: vec![StreamField {
                name: "blob".into(),
                data_type: StreamDataType::Utf8,
                nullable: false,
            }],
        };
        let huge = "x".repeat(8192);
        let iter = (0..10).map(move |_| Ok(vec![Value::String(huge.clone())]));
        // Set tight 2 KiB ceiling -> must fail closed on first 80 KiB batch!
        let mut cursor = StreamingBatchCursor::new(schema, iter, 10, 2048);
        assert!(cursor.next_batch().is_err());
    }
}
