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
            use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder as B;
            use rayon::prelude::*;
            let groups = B::try_new(f).map_err(err)?.metadata().num_row_groups();
            if groups <= 1 {
                let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
                let rd = B::try_new(f).map_err(err)?.with_batch_size(65536).build().map_err(err)?;
                return rd.collect::<Result<Vec<_>, _>>().map_err(err);
            }
            // decode row groups in parallel, batches kept in file order
            let parts: Vec<Vec<RecordBatch>> = (0..groups)
                .into_par_iter()
                .map(|g| {
                    let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
                    let rd = B::try_new(f)
                        .map_err(err)?
                        .with_row_groups(vec![g])
                        .with_batch_size(65536)
                        .build()
                        .map_err(err)?;
                    rd.collect::<Result<Vec<_>, _>>().map_err(err)
                })
                .collect::<Result<_, String>>()?;
            Ok(parts.into_iter().flatten().collect())
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

/// Batches -> (column names, typed rows) keeping the source schema's types:
///
/// | Arrow | JouleDB value |
/// |---|---|
/// | Utf8 / LargeUtf8 / Utf8View | String (always, even if it looks numeric) |
/// | Int8..Int64, UInt8..UInt32 | Int |
/// | UInt64 | Int, or String when above i64::MAX (exact) |
/// | Float16/32/64 | Float |
/// | Boolean | Bool |
/// | Decimal128/256 | Int when scale 0 and it fits; Float when precision <= 15; else String (exact) |
/// | Date32/Date64 | String `YYYY-MM-DD` |
/// | Timestamp(unit, tz) | String ISO 8601 (`2024-01-02T03:04:05.123`, with offset when tz is set) |
/// | Time / Duration / Interval | String (Arrow display) |
/// | Binary / LargeBinary / FixedSizeBinary | Bytes |
/// | List / LargeList / FixedSizeList | Array (element-wise) |
/// | Struct | Object |
/// | Dictionary | its value type |
/// | Null, or a NULL slot | Null |
pub fn batches_to_values(
    batches: &[RecordBatch],
) -> Result<(Vec<String>, Vec<Vec<joule_db_query::ast::Value>>), String> {
    let Some(first) = batches.first() else {
        return Ok((Vec::new(), Vec::new()));
    };
    let columns: Vec<String> = first.schema().fields().iter().map(|f| f.name().clone()).collect();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    let mut rows: Vec<Vec<joule_db_query::ast::Value>> = Vec::with_capacity(total);
    use rayon::prelude::*;
    for b in batches {
        // decode columns in parallel, then move (not clone) cells into rows
        let cols: Vec<Vec<joule_db_query::ast::Value>> = b
            .columns()
            .par_iter()
            .map(|a| array_values(a.as_ref()))
            .collect::<Result<_, _>>()?;
        let mut its: Vec<_> = cols.into_iter().map(|c| c.into_iter()).collect();
        for _ in 0..b.num_rows() {
            rows.push(its.iter_mut().map(|it| it.next().unwrap_or(joule_db_query::ast::Value::Null)).collect());
        }
    }
    Ok((columns, rows))
}

/// Arrow batches -> typed columns for columnar-first ingest, with the
/// values `batches_to_values` would give (ints widened to i64, f32 to f64,
/// UTF-8 text, booleans). `None` when any column has another type (the
/// caller takes the row path).
pub fn batches_to_columns(
    batches: &[RecordBatch],
) -> Option<(Vec<String>, Vec<joule_db_amorphic::column_batch::BatchColumn>, usize)> {
    use arrow::array::AsArray;
    use arrow::datatypes::*;
    use joule_db_amorphic::column_batch::{BatchColumn, ColData};
    use rayon::prelude::*;
    let first = batches.first()?;
    let schema = first.schema();
    if batches.iter().any(|b| b.schema() != schema) {
        return None;
    }
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    let cols: Option<Vec<BatchColumn>> = (0..names.len())
        .into_par_iter()
        .map(|ci| {
            let dt = schema.field(ci).data_type().clone();
            let arrays: Vec<&dyn Array> = batches.iter().map(|b| b.column(ci).as_ref()).collect();
            let nulls: usize = arrays.iter().map(|a| a.null_count()).sum();
            let valid = (nulls > 0).then(|| {
                let mut v = Vec::with_capacity(total);
                for a in &arrays {
                    v.extend((0..a.len()).map(|i| !a.is_null(i)));
                }
                v
            });
            macro_rules! ints {
                ($t:ty) => {{
                    let mut v = Vec::with_capacity(total);
                    for a in &arrays {
                        v.extend(a.as_primitive::<$t>().values().iter().map(|&x| x as i64));
                    }
                    ColData::Int(v)
                }};
            }
            let data = match dt {
                DataType::Int8 => ints!(Int8Type),
                DataType::Int16 => ints!(Int16Type),
                DataType::Int32 => ints!(Int32Type),
                DataType::Int64 => ints!(Int64Type),
                DataType::UInt8 => ints!(UInt8Type),
                DataType::UInt16 => ints!(UInt16Type),
                DataType::UInt32 => ints!(UInt32Type),
                DataType::Float32 => {
                    let mut v = Vec::with_capacity(total);
                    for a in &arrays {
                        v.extend(a.as_primitive::<Float32Type>().values().iter().map(|&x| x as f64));
                    }
                    ColData::Float(v)
                }
                DataType::Float64 => {
                    let mut v = Vec::with_capacity(total);
                    for a in &arrays {
                        v.extend_from_slice(a.as_primitive::<Float64Type>().values());
                    }
                    ColData::Float(v)
                }
                DataType::Boolean => {
                    let mut v = Vec::with_capacity(total);
                    for a in &arrays {
                        let b = a.as_boolean();
                        v.extend((0..b.len()).map(|i| b.value(i)));
                    }
                    ColData::Bool(v)
                }
                DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Date32
                | DataType::Date64
                | DataType::Timestamp(_, _) => {
                    // one task per Arrow batch, then concatenated in order;
                    // dates/timestamps as the text `batches_to_values` gives
                    let parts: Option<Vec<(Vec<u32>, Vec<u8>)>> = arrays
                        .par_iter()
                        .map(|a| {
                            let n = a.len();
                            let mut offs = Vec::with_capacity(n);
                            let mut bytes: Vec<u8> = Vec::new();
                            match a.data_type() {
                                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                                    let get = |i: usize| -> &str {
                                        if a.is_null(i) {
                                            return "";
                                        }
                                        match a.data_type() {
                                            DataType::Utf8 => a.as_string::<i32>().value(i),
                                            DataType::LargeUtf8 => a.as_string::<i64>().value(i),
                                            _ => a.as_string_view().value(i),
                                        }
                                    };
                                    for i in 0..n {
                                        bytes.extend_from_slice(get(i).as_bytes());
                                        offs.push(bytes.len() as u32);
                                    }
                                }
                                _ => {
                                    let f = ArrayFormatter::try_new(*a, &FormatOptions::default()).ok()?;
                                    use std::io::Write;
                                    for i in 0..n {
                                        if !a.is_null(i) {
                                            write!(bytes, "{}", f.value(i)).ok()?;
                                        }
                                        offs.push(bytes.len() as u32);
                                    }
                                }
                            }
                            Some((offs, bytes))
                        })
                        .collect();
                    let parts = parts?;
                    let total_bytes: usize = parts.iter().map(|p| p.1.len()).sum();
                    if total_bytes > u32::MAX as usize / 2 {
                        return None;
                    }
                    let mut offsets = Vec::with_capacity(total + 1);
                    let mut bytes = Vec::with_capacity(total_bytes);
                    offsets.push(0u32);
                    for (o, b) in parts {
                        let base = bytes.len() as u32;
                        offsets.extend(o.into_iter().map(|x| x + base));
                        bytes.extend_from_slice(&b);
                    }
                    ColData::Str { offsets, bytes }
                }
                _ => return None,
            };
            Some(BatchColumn { name: names[ci].clone(), data, valid })
        })
        .collect();
    Some((names, cols?, total))
}

fn array_values(a: &dyn Array) -> Result<Vec<joule_db_query::ast::Value>, String> {
    use arrow::array::AsArray;
    use arrow::datatypes::*;
    use joule_db_query::ast::Value as V;
    let n = a.len();
    let null_or = |i: usize, f: &dyn Fn(usize) -> V| if a.is_null(i) { V::Null } else { f(i) };
    macro_rules! prim {
        ($t:ty, $conv:expr) => {{
            let p = a.as_primitive::<$t>();
            (0..n).map(|i| null_or(i, &|i| $conv(p.value(i)))).collect()
        }};
    }
    let fmt_all = |a: &dyn Array| -> Result<Vec<V>, String> {
        let f = ArrayFormatter::try_new(a, &FormatOptions::default()).map_err(err)?;
        Ok((0..a.len()).map(|i| if a.is_null(i) { V::Null } else { V::String(f.value(i).to_string()) }).collect())
    };
    Ok(match a.data_type() {
        DataType::Null => vec![V::Null; n],
        DataType::Boolean => {
            let b = a.as_boolean();
            (0..n).map(|i| null_or(i, &|i| V::Bool(b.value(i)))).collect()
        }
        DataType::Int8 => prim!(Int8Type, |v: i8| V::Int(v as i64)),
        DataType::Int16 => prim!(Int16Type, |v: i16| V::Int(v as i64)),
        DataType::Int32 => prim!(Int32Type, |v: i32| V::Int(v as i64)),
        DataType::Int64 => prim!(Int64Type, V::Int),
        DataType::UInt8 => prim!(UInt8Type, |v: u8| V::Int(v as i64)),
        DataType::UInt16 => prim!(UInt16Type, |v: u16| V::Int(v as i64)),
        DataType::UInt32 => prim!(UInt32Type, |v: u32| V::Int(v as i64)),
        DataType::UInt64 => prim!(UInt64Type, |v: u64| i64::try_from(v)
            .map(V::Int)
            .unwrap_or_else(|_| V::String(v.to_string()))),
        DataType::Float16 => prim!(Float16Type, |v: half::f16| V::Float(v.to_f64())),
        DataType::Float32 => prim!(Float32Type, |v: f32| V::Float(v as f64)),
        DataType::Float64 => prim!(Float64Type, V::Float),
        DataType::Utf8 => {
            let s = a.as_string::<i32>();
            (0..n).map(|i| null_or(i, &|i| V::String(s.value(i).to_string()))).collect()
        }
        DataType::LargeUtf8 => {
            let s = a.as_string::<i64>();
            (0..n).map(|i| null_or(i, &|i| V::String(s.value(i).to_string()))).collect()
        }
        DataType::Utf8View => {
            let s = a.as_string_view();
            (0..n).map(|i| null_or(i, &|i| V::String(s.value(i).to_string()))).collect()
        }
        DataType::Binary => {
            let s = a.as_binary::<i32>();
            (0..n).map(|i| null_or(i, &|i| V::Bytes(s.value(i).to_vec()))).collect()
        }
        DataType::LargeBinary => {
            let s = a.as_binary::<i64>();
            (0..n).map(|i| null_or(i, &|i| V::Bytes(s.value(i).to_vec()))).collect()
        }
        DataType::FixedSizeBinary(_) => {
            let s = a.as_fixed_size_binary();
            (0..n).map(|i| null_or(i, &|i| V::Bytes(s.value(i).to_vec()))).collect()
        }
        DataType::Decimal128(p, sc) => {
            let (p, sc) = (*p, *sc);
            let d = a.as_primitive::<Decimal128Type>();
            let f = ArrayFormatter::try_new(a, &FormatOptions::default()).map_err(err)?;
            (0..n)
                .map(|i| {
                    if a.is_null(i) {
                        V::Null
                    } else if sc == 0 && i64::try_from(d.value(i)).is_ok() {
                        V::Int(d.value(i) as i64)
                    } else if p <= 15 {
                        V::Float(d.value(i) as f64 / 10f64.powi(sc as i32))
                    } else {
                        V::String(f.value(i).to_string())
                    }
                })
                .collect()
        }
        DataType::Decimal256(p, _) if *p <= 15 => {
            let c = arrow::compute::cast(a, &DataType::Float64).map_err(err)?;
            array_values(c.as_ref())?
        }
        DataType::Dictionary(_, vt) => {
            let c = arrow::compute::cast(a, vt).map_err(err)?;
            array_values(c.as_ref())?
        }
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _) => fmt_all(a)?,
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                if a.is_null(i) {
                    out.push(V::Null);
                    continue;
                }
                let child = match a.data_type() {
                    DataType::List(_) => a.as_list::<i32>().value(i),
                    DataType::LargeList(_) => a.as_list::<i64>().value(i),
                    _ => a.as_fixed_size_list().value(i),
                };
                out.push(V::Array(array_values(child.as_ref())?));
            }
            out
        }
        DataType::Struct(fields) => {
            let st = a.as_struct();
            let kids: Vec<Vec<V>> = st
                .columns()
                .iter()
                .map(|c| array_values(c.as_ref()))
                .collect::<Result<_, _>>()?;
            (0..n)
                .map(|i| {
                    if a.is_null(i) {
                        V::Null
                    } else {
                        V::Object(
                            fields
                                .iter()
                                .zip(&kids)
                                .map(|(f, k)| (f.name().clone(), k[i].clone()))
                                .collect(),
                        )
                    }
                })
                .collect()
        }
        _ => fmt_all(a)?,
    })
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
    fn typed_import_keeps_source_types() {
        use arrow::array::*;
        use arrow::datatypes::{TimeUnit, DataType as DT};
        use joule_db_query::ast::Value as V;
        let schema = Arc::new(Schema::new(vec![
            Field::new("zip", DT::Utf8, true),
            Field::new("d", DT::Date32, true),
            Field::new("ts", DT::Timestamp(TimeUnit::Microsecond, None), true),
            Field::new("tsz", DT::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), true),
            Field::new("price", DT::Decimal128(10, 2), true),
            Field::new("big", DT::Decimal128(38, 4), true),
            Field::new("ok", DT::Boolean, true),
            Field::new("nothing", DT::Null, true),
            Field::new("u", DT::UInt64, true),
            Field::new("i32", DT::Int32, true),
            Field::new("f32", DT::Float32, true),
            Field::new("dict", DT::Dictionary(Box::new(DT::Int32), Box::new(DT::Utf8)), true),
        ]));
        let dict: DictionaryArray<arrow::datatypes::Int32Type> = vec![Some("x"), None].into_iter().collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("02139"), Some("123")])),
                Arc::new(Date32Array::from(vec![Some(19723), None])),
                Arc::new(TimestampMicrosecondArray::from(vec![Some(1_704_164_645_123_000), None])),
                Arc::new(TimestampMillisecondArray::from(vec![Some(0), None]).with_timezone("UTC")),
                Arc::new(Decimal128Array::from(vec![Some(12345), None]).with_precision_and_scale(10, 2).unwrap()),
                Arc::new(
                    Decimal128Array::from(vec![Some(12345678901234567890123i128), None])
                        .with_precision_and_scale(38, 4)
                        .unwrap(),
                ),
                Arc::new(BooleanArray::from(vec![Some(true), None])),
                Arc::new(NullArray::new(2)),
                Arc::new(UInt64Array::from(vec![Some(u64::MAX), Some(7)])),
                Arc::new(Int32Array::from(vec![Some(-3), None])),
                Arc::new(Float32Array::from(vec![Some(0.5), None])),
                Arc::new(dict),
            ],
        )
        .unwrap();
        for fmt in [FileFormat::Parquet, FileFormat::ArrowFile] {
            let dir = tempfile::tempdir().unwrap();
            let p = dir.path().join("typed");
            let p = p.to_str().unwrap();
            std::fs::write(p, write_bytes(&batch, fmt).unwrap()).unwrap();
            let (cols, rows) = batches_to_values(&read_file(p, fmt).unwrap()).unwrap();
            assert_eq!(cols[0], "zip");
            let r = &rows[0];
            assert_eq!(r[0], V::String("02139".into()), "{fmt:?}");
            assert_eq!(rows[1][0], V::String("123".into()), "numeric-looking string stays text");
            assert_eq!(r[1], V::String("2024-01-01".into())); // day 19723
            assert_eq!(r[2], V::String("2024-01-02T03:04:05.123".into()));
            assert!(matches!(&r[3], V::String(s) if s.starts_with("1970-01-01T00:00:00") && (s.ends_with('Z') || s.ends_with("+00:00"))), "{:?}", r[3]);
            assert_eq!(r[4], V::Float(123.45));
            assert_eq!(r[5], V::String("1234567890123456789.0123".into()));
            assert_eq!(r[6], V::Bool(true));
            assert_eq!(r[7], V::Null);
            assert_eq!(r[8], V::String(u64::MAX.to_string()));
            assert_eq!(rows[1][8], V::Int(7));
            assert_eq!(r[9], V::Int(-3));
            assert_eq!(r[10], V::Float(0.5));
            assert_eq!(r[11], V::String("x".into()));
            for c in 1..12 {
                if c != 8 {
                    assert_eq!(rows[1][c], V::Null, "col {c} {fmt:?}");
                }
            }
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
