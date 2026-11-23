//! Columnar Storage Format for OLAP
//!
//! GPU-friendly columnar layout optimized for analytical queries

use std::sync::Arc;

/// Columnar storage format
#[derive(Debug, Clone)]
pub struct ColumnarFormat {
    /// Column data (each column is a vector of values)
    pub columns: Vec<Column>,
    /// Column names
    pub column_names: Vec<String>,
    /// Number of rows
    pub row_count: usize,
    /// Compression type
    pub compression: CompressionType,
}

/// Column data
#[derive(Debug, Clone)]
pub struct Column {
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Values (compressed)
    pub data: Vec<u8>,
    /// Null bitmap
    pub null_bitmap: Vec<u8>,
}

/// Data types for columnar storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int32,
    Int64,
    Float32,
    Float64,
    String,
    Binary,
}

/// Compression types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None,
    Delta,
    RunLength,
    Dictionary,
}

/// Columnar store for OLAP operations (internal, no WASM bindings)
pub struct ColumnarStore {
    format: Arc<std::sync::Mutex<ColumnarFormat>>,
    gpu_backend: Option<Arc<joule_db_gpu::WgpuComputeBackend>>,
}

impl ColumnarStore {
    /// Create new columnar store

    pub fn new(column_names: Vec<String>, data_types: Vec<String>) -> Self {
        let columns = column_names
            .iter()
            .zip(data_types.iter())
            .map(|(name, dt)| Column {
                name: name.clone(),
                data_type: parse_data_type(dt),
                data: Vec::new(),
                null_bitmap: Vec::new(),
            })
            .collect();

        let format = Arc::new(std::sync::Mutex::new(ColumnarFormat {
            columns,
            column_names,
            row_count: 0,
            compression: CompressionType::Delta,
        }));

        Self {
            format,
            gpu_backend: None,
        }
    }

    /// Add row to columnar store
    pub fn add_row(&mut self, values: Vec<serde_json::Value>) -> Result<(), String> {
        // Get mutable access to format
        let mut format = self
            .format
            .lock()
            .map_err(|_| "Failed to lock format".to_string())?;

        if values.len() != format.columns.len() {
            return Err(format!(
                "Expected {} values, got {}",
                format.columns.len(),
                values.len()
            ));
        }

        // Get row count before the loop to avoid borrow conflicts
        let row_count = format.row_count;

        // Add each value to its column
        for (col_idx, (col, value)) in format.columns.iter_mut().zip(values.iter()).enumerate() {
            match col.data_type {
                DataType::Int32 => {
                    if let Some(i) = value.as_i64() {
                        col.data.extend_from_slice(&(i as i32).to_le_bytes());
                    }
                }
                DataType::Float64 => {
                    if let Some(f) = value.as_f64() {
                        col.data.extend_from_slice(&f.to_le_bytes());
                    }
                }
                DataType::String => {
                    if let Some(s) = value.as_str() {
                        let bytes = s.as_bytes();
                        col.data
                            .extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                        col.data.extend_from_slice(bytes);
                    }
                }
                _ => {
                    // For other types, serialize as JSON
                    if let Ok(bytes) = serde_json::to_vec(value) {
                        col.data
                            .extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                        col.data.extend_from_slice(&bytes);
                    }
                }
            }

            // Update null bitmap (0 = not null, 1 = null)
            let byte_idx = row_count / 8;
            let bit_idx = row_count % 8;

            while col.null_bitmap.len() <= byte_idx {
                col.null_bitmap.push(0);
            }

            if value.is_null() {
                col.null_bitmap[byte_idx] |= 1u8 << bit_idx;
            }
        }

        format.row_count += 1;
        Ok(())
    }

    /// Execute analytical query on columnar data
    pub async fn query(&self, query: &str) -> Result<String, String> {
        // Route to GPU if available
        if let Some(ref gpu) = self.gpu_backend {
            return self.query_gpu(gpu, query).await;
        }
        self.query_cpu(query).await
    }

    /// GPU-accelerated query
    async fn query_gpu(
        &self,
        _gpu: &Arc<joule_db_gpu::WgpuComputeBackend>,
        query: &str,
    ) -> Result<String, String> {
        // Parse simple SQL-like queries
        let query_lower = query.to_lowercase();

        // Simple aggregation queries
        if query_lower.contains("sum")
            || query_lower.contains("avg")
            || query_lower.contains("count")
        {
            // Extract column name
            let col_name = if let Some(start) = query_lower.find("sum(") {
                let end = query_lower[start + 4..].find(')').unwrap_or(0);
                query_lower[start + 4..start + 4 + end].trim().to_string()
            } else if let Some(start) = query_lower.find("avg(") {
                let end = query_lower[start + 4..].find(')').unwrap_or(0);
                query_lower[start + 4..start + 4 + end].trim().to_string()
            } else if query_lower.contains("count(*)") {
                "*".to_string()
            } else {
                return Err("Unsupported query".to_string());
            };

            // Find column index
            let col_idx = self
                .format
                .lock()
                .map_err(|_| "Failed to lock format".to_string())?
                .column_names
                .iter()
                .position(|n| n == &col_name || col_name == "*")
                .unwrap_or(0);

            // For now, fall back to CPU
            return self.query_cpu(query).await;
        }

        // Fall back to CPU for other queries
        self.query_cpu(query).await
    }

    /// CPU fallback query
    async fn query_cpu(&self, query: &str) -> Result<String, String> {
        let format = self
            .format
            .lock()
            .map_err(|_| "Failed to lock format".to_string())?;

        let query_lower = query.to_lowercase();

        // Simple aggregation queries
        if query_lower.contains("sum(") {
            let col_name = if let Some(start) = query_lower.find("sum(") {
                let end = query_lower[start + 4..].find(')').unwrap_or(0);
                query_lower[start + 4..start + 4 + end].trim().to_string()
            } else {
                return Err("Invalid SUM query".to_string());
            };

            let col_idx = format
                .column_names
                .iter()
                .position(|n| n == &col_name)
                .ok_or_else(|| "Column not found".to_string())?;

            let col = &format.columns[col_idx];
            let mut sum = 0.0f64;

            // Sum float values
            let mut offset = 0;
            while offset + 8 <= col.data.len() {
                let val = f64::from_le_bytes([
                    col.data[offset],
                    col.data[offset + 1],
                    col.data[offset + 2],
                    col.data[offset + 3],
                    col.data[offset + 4],
                    col.data[offset + 5],
                    col.data[offset + 6],
                    col.data[offset + 7],
                ]);
                sum += val;
                offset += 8;
            }

            return Ok(sum.to_string());
        }

        if query_lower.contains("count(*)") {
            return Ok(format.row_count.to_string());
        }

        Err("Unsupported query".to_string())
    }

    /// Get total size of columnar data (for compression testing)
    pub fn get_total_size(&self) -> usize {
        let format = self.format.lock().expect("Failed to lock format");
        let mut total = 0;
        for col in &format.columns {
            total += col.data.len();
            total += col.null_bitmap.len();
        }
        total
    }

    /// Get column data (for compression testing)
    pub fn get_column_data(&self, col_idx: usize) -> Option<Vec<u8>> {
        let format = self.format.lock().expect("Failed to lock format");
        if col_idx < format.columns.len() {
            Some(format.columns[col_idx].data.clone())
        } else {
            None
        }
    }
}

fn parse_data_type(dt: &str) -> DataType {
    match dt.to_lowercase().as_str() {
        "int32" | "i32" => DataType::Int32,
        "int64" | "i64" => DataType::Int64,
        "float32" | "f32" => DataType::Float32,
        "float64" | "f64" => DataType::Float64,
        "string" | "str" => DataType::String,
        "binary" | "bytes" => DataType::Binary,
        _ => DataType::String,
    }
}
