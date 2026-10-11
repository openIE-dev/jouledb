//! Real Apache Parquet and Arrow IPC import/export (arrow + parquet crates).
//!
//! Used by `COPY t FROM '/path/x.parquet'`, `COPY t|(query) TO '/path/x.arrow'`
//! over pgwire (inside `JOULEDB_COPY_DIR`), by
//! `GET /api/v1/export/{table}?format=parquet|arrow|arrows` and by
//! `POST /api/v1/export {"sql", "format"}`.
//! Files written here are read by DuckDB, pyarrow and polars, and files they
//! write are read here.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Float64Builder, Int64Builder, RecordBatch,
    StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use serde_json::Value;

/// On-disk formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Parquet,
    /// Arrow IPC file (Feather v2)
    ArrowFile,
    /// Arrow IPC stream
    ArrowStream,
}

impl FileFormat {
    /// From an explicit name (`parquet`, `arrow`, `ipc`, `feather`, `arrows`).
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "parquet" => Some(Self::Parquet),
            "arrow" | "ipc" | "feather" => Some(Self::ArrowFile),
            "arrows" | "arrow_stream" | "stream" => Some(Self::ArrowStream),
            _ => None,
        }
    }

    /// From a path's extension.
    pub fn from_path(p: &str) -> Option<Self> {
        let ext = Path::new(p).extension()?.to_str()?;
        Self::from_name(ext)
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Self::Parquet => "application/vnd.apache.parquet",
            Self::ArrowFile => "application/vnd.apache.arrow.file",
            Self::ArrowStream => "application/vnd.apache.arrow.stream",
        }
    }
}

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// Build one Arrow batch from JSON rows: each column is Int64, Float64,
/// Boolean or Utf8 (ints mixed with floats widen to Float64; anything else
/// mixed becomes Utf8; all-NULL columns are Utf8).
pub fn rows_to_batch(columns: &[String], rows: &[Vec<Value>]) -> Result<RecordBatch, String> {
    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for (c, name) in columns.iter().enumerate() {
        let cell = |r: &Vec<Value>| r.get(c).cloned().unwrap_or(Value::Null);
        let (mut ints, mut floats, mut bools, mut other) = (true, true, true, false);
        let mut any = false;
        for r in rows {
            match r.get(c).unwrap_or(&Value::Null) {
                Value::Null => {}
                Value::Number(n) => {
                    any = true;
                    bools = false;
                    if n.as_i64().is_none() {
                        ints = false;
                    }
                }
                Value::Bool(_) => {
                    any = true;
                    ints = false;
                    floats = false;
                }
                _ => {
                    any = true;
                    other = true;
                }
            }
        }
        let ty = if !any || other {
            DataType::Utf8
        } else if bools && !ints {
            DataType::Boolean
        } else if ints {
            DataType::Int64
        } else if floats {
            DataType::Float64
        } else {
            DataType::Utf8
        };
        let arr: ArrayRef = match ty {
            DataType::Int64 => {
                let mut b = Int64Builder::with_capacity(rows.len());
                for r in rows {
                    b.append_option(cell(r).as_i64());
                }
                Arc::new(b.finish())
            }
            DataType::Float64 => {
                let mut b = Float64Builder::with_capacity(rows.len());
                for r in rows {
                    b.append_option(cell(r).as_f64());
                }
                Arc::new(b.finish())
            }
            DataType::Boolean => {
                let mut b = BooleanBuilder::with_capacity(rows.len());
                for r in rows {
                    b.append_option(cell(r).as_bool());
                }
                Arc::new(b.finish())
            }
            _ => {
                let mut b = StringBuilder::with_capacity(rows.len(), rows.len() * 8);
                for r in rows {
                    match cell(r) {
                        Value::Null => b.append_null(),
                        Value::String(s) => b.append_value(s),
                        v => b.append_value(v.to_string()),
                    }
                }
                Arc::new(b.finish())
            }
        };
        fields.push(Field::new(name, ty, true));
        arrays.push(arr);
    }
    let schema = Arc::new(Schema::new(fields));
    if columns.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }
    RecordBatch::try_new(schema, arrays).map_err(err)
}

/// Serialize a batch to bytes in `fmt`.
pub fn write_bytes(batch: &RecordBatch, fmt: FileFormat) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    write_to(&mut out, batch, fmt)?;
    Ok(out)
}

fn write_to<W: std::io::Write + Send>(w: W, batch: &RecordBatch, fmt: FileFormat) -> Result<(), String> {
    match fmt {
        FileFormat::Parquet => {
            let props = parquet::file::properties::WriterProperties::builder()
                .set_compression(parquet::basic::Compression::ZSTD(Default::default()))
                .build();
            let mut wr =
                parquet::arrow::ArrowWriter::try_new(w, batch.schema(), Some(props)).map_err(err)?;
            wr.write(batch).map_err(err)?;
            wr.close().map_err(err)?;
        }
        FileFormat::ArrowFile => {
            let mut wr = arrow::ipc::writer::FileWriter::try_new(w, &batch.schema()).map_err(err)?;
            wr.write(batch).map_err(err)?;
            wr.finish().map_err(err)?;
        }
        FileFormat::ArrowStream => {
            let mut wr = arrow::ipc::writer::StreamWriter::try_new(w, &batch.schema()).map_err(err)?;
            wr.write(batch).map_err(err)?;
            wr.finish().map_err(err)?;
        }
    }
    Ok(())
}

/// Write a result set to `path`.
pub fn write_file(path: &str, fmt: FileFormat, columns: &[String], rows: &[Vec<Value>]) -> Result<usize, String> {
    let batch = rows_to_batch(columns, rows)?;
    let f = File::create(path).map_err(|e| format!("{path}: {e}"))?;
    write_to(std::io::BufWriter::new(f), &batch, fmt)?;
    Ok(batch.num_rows())
}

/// Read every batch of a Parquet or Arrow IPC file.
pub fn read_file(path: &str, fmt: FileFormat) -> Result<Vec<RecordBatch>, String> {
    let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    match fmt {
        FileFormat::Parquet => {
            let rd = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(f)
                .map_err(err)?
                .with_batch_size(65536)
                .build()
                .map_err(err)?;
            rd.collect::<Result<Vec<_>, _>>().map_err(err)
        }
        FileFormat::ArrowFile => {
            let rd = arrow::ipc::reader::FileReader::try_new(std::io::BufReader::new(f), None).map_err(err)?;
            rd.collect::<Result<Vec<_>, _>>().map_err(err)
        }
        FileFormat::ArrowStream => {
            let rd = arrow::ipc::reader::StreamReader::try_new(std::io::BufReader::new(f), None).map_err(err)?;
            rd.collect::<Result<Vec<_>, _>>().map_err(err)
        }
    }
}

/// Batches -> (column names, COPY text rows) for `QueryExecutor::bulk_copy`.
/// Numbers and booleans format as their SQL literals, NULL as `None`;
/// temporal and decimal types use Arrow's canonical display.
pub fn batches_to_copy_rows(batches: &[RecordBatch]) -> Result<(Vec<String>, Vec<Vec<Option<String>>>), String> {
    let Some(first) = batches.first() else {
        return Ok((Vec::new(), Vec::new()));
    };
    let columns: Vec<String> = first.schema().fields().iter().map(|f| f.name().clone()).collect();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    let mut rows = Vec::with_capacity(total);
    let opts = FormatOptions::default();
    for b in batches {
        let fmts: Vec<ArrayFormatter> = b
            .columns()
            .iter()
            .map(|a| ArrayFormatter::try_new(a.as_ref(), &opts))
            .collect::<Result<_, _>>()
            .map_err(err)?;
        for i in 0..b.num_rows() {
            let mut row = Vec::with_capacity(fmts.len());
            for (c, f) in fmts.iter().enumerate() {
                let a = b.column(c);
                if a.is_null(i) {
                    row.push(None);
                } else if let Some(bo) = a.as_any().downcast_ref::<BooleanArray>() {
                    row.push(Some(if bo.value(i) { "true".into() } else { "false".into() }));
                } else {
                    row.push(Some(f.value(i).to_string()));
                }
            }
            rows.push(row);
        }
    }
    Ok((columns, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_all_formats() {
        let cols = vec!["id".to_string(), "name".into(), "v".into(), "ok".into()];
        let rows = vec![
            vec![json!(1), json!("a"), json!(1.5), json!(true)],
            vec![json!(2), Value::Null, json!(2), json!(false)],
            vec![json!(3), json!("c"), Value::Null, Value::Null],
        ];
        for fmt in [FileFormat::Parquet, FileFormat::ArrowFile, FileFormat::ArrowStream] {
            let dir = tempfile::tempdir().unwrap();
            let p = dir.path().join("t.out");
            let p = p.to_str().unwrap();
            assert_eq!(write_file(p, fmt, &cols, &rows).unwrap(), 3);
            let batches = read_file(p, fmt).unwrap();
            let s = batches[0].schema();
            assert_eq!(s.field(0).data_type(), &DataType::Int64);
            assert_eq!(s.field(1).data_type(), &DataType::Utf8);
            assert_eq!(s.field(2).data_type(), &DataType::Float64);
            assert_eq!(s.field(3).data_type(), &DataType::Boolean);
            let (c, r) = batches_to_copy_rows(&batches).unwrap();
            assert_eq!(c, cols);
            assert_eq!(r[0], vec![Some("1".into()), Some("a".into()), Some("1.5".into()), Some("true".into())]);
            assert_eq!(r[1][1], None);
            assert_eq!(r[1][2].as_deref(), Some("2.0")); // float column keeps its type
            assert_eq!(r[2][3], None);
        }
    }

    #[test]
    fn format_detection() {
        assert_eq!(FileFormat::from_path("/x/y.parquet"), Some(FileFormat::Parquet));
        assert_eq!(FileFormat::from_path("a.arrow"), Some(FileFormat::ArrowFile));
        assert_eq!(FileFormat::from_path("a.arrows"), Some(FileFormat::ArrowStream));
        assert_eq!(FileFormat::from_path("a.csv"), None);
    }
}
