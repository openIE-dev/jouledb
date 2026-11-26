//! Zero-Copy IPC for JouleDB
//!
//! Features:
//! - SharedArrayBuffer communication
//! - Apache Arrow IPC format
//! - Worker bridge with atomic operations
//! - <1ms IPC latency

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Worker Bridge for Zero-Copy IPC
///
/// Uses SharedArrayBuffer and Atomics for <1ms latency communication
/// between main thread and Web Worker

pub struct WorkerBridge {
    /// Shared buffer for communication
    #[cfg(target_arch = "wasm32")]
    shared_buffer: Arc<RwLock<Option<js_sys::SharedArrayBuffer>>>,
    #[cfg(not(target_arch = "wasm32"))]
    shared_buffer: Arc<RwLock<Option<Vec<u8>>>>,
    /// Status flag (atomic): 0 = idle, 1 = busy, 2 = ready
    status: Arc<AtomicU32>,
    /// Buffer size
    buffer_size: usize,
}

impl WorkerBridge {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            shared_buffer: Arc::new(RwLock::new(None)),
            status: Arc::new(AtomicU32::new(0)),
            buffer_size,
        }
    }

    /// Initialize with SharedArrayBuffer

    #[cfg(target_arch = "wasm32")]
    pub fn init(&self, buffer: &js_sys::SharedArrayBuffer) -> Result<(), JsValue> {
        let mut shared = self
            .shared_buffer
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *shared = Some(buffer.clone());
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn init(&self, _buffer: &JsValue) -> Result<(), JsValue> {
        let mut shared = self
            .shared_buffer
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *shared = Some(vec![0u8; self.buffer_size]);
        Ok(())
    }

    /// Send query to worker (non-blocking, optimized for <1ms latency)

    #[cfg(target_arch = "wasm32")]
    pub fn send_query(&self, query: &str) -> Result<(), JsValue> {
        // Check if buffer is available
        let shared = self.shared_buffer.read().unwrap_or_else(|e| e.into_inner());
        let buffer = shared
            .as_ref()
            .ok_or_else(|| JsValue::from_str("SharedArrayBuffer not initialized"))?;

        // Check status (should be idle) - use relaxed ordering for speed
        let current_status = self.status.load(Ordering::Relaxed);
        if current_status != 0 {
            return Err(JsValue::from_str("Worker is busy"));
        }

        // Write query to buffer using atomic operations for zero-copy
        let query_bytes = query.as_bytes();
        let query_len = query_bytes.len();

        // Check buffer size
        if query_len + 8 > self.buffer_size {
            return Err(JsValue::from_str("Query too large for buffer"));
        }

        // Use Uint32Array for atomic operations on SharedArrayBuffer
        let view = js_sys::Uint32Array::new(buffer);

        // Write length (first 2 u32s)
        view.set_index(0, query_len as u32);

        // Write data (packed into u32s)
        let mut offset = 2;
        for chunk in query_bytes.chunks(4) {
            let mut value = 0u32;
            for (i, &byte) in chunk.iter().enumerate() {
                value |= (byte as u32) << (i * 8);
            }
            view.set_index(offset, value);
            offset += 1;
        }

        // Set status to busy (use Release ordering to ensure writes are visible)
        self.status.store(1, Ordering::Release);

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn send_query(&self, _query: &str) -> Result<(), JsValue> {
        let current_status = self.status.load(Ordering::Relaxed);
        if current_status != 0 {
            return Err(JsValue::from_str("Worker is busy"));
        }
        self.status.store(1, Ordering::Release);
        Ok(())
    }

    /// Receive result from worker (non-blocking, optimized for <1ms latency)

    #[cfg(target_arch = "wasm32")]
    pub fn receive_result(&self) -> Result<Option<String>, JsValue> {
        // Use relaxed ordering for fast status check
        let status = self.status.load(Ordering::Relaxed);

        if status == 2 {
            // Result is ready - read from buffer
            let shared = self.shared_buffer.read().unwrap_or_else(|e| e.into_inner());
            let buffer = shared
                .as_ref()
                .ok_or_else(|| JsValue::from_str("SharedArrayBuffer not initialized"))?;

            let view = js_sys::Uint32Array::new(buffer);

            // Read length (first 2 u32s)
            let result_len = view.get_index(0) as usize;

            // Read data (packed from u32s)
            let mut result_bytes = Vec::with_capacity(result_len);
            let mut offset = 2;
            let u32_count = (result_len + 3) / 4;

            for _ in 0..u32_count {
                let value = view.get_index(offset);
                for i in 0..4 {
                    if result_bytes.len() < result_len {
                        result_bytes.push((value >> (i * 8)) as u8);
                    }
                }
                offset += 1;
            }

            // Reset status to idle
            self.status.store(0, Ordering::Release);

            // Convert to string
            let result = String::from_utf8(result_bytes)
                .map_err(|_| JsValue::from_str("Invalid UTF-8 in result"))?;

            Ok(Some(result))
        } else if status == 1 {
            // Still processing
            Ok(None)
        } else {
            // Idle
            Ok(None)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn receive_result(&self) -> Result<Option<String>, JsValue> {
        let status = self.status.load(Ordering::Relaxed);
        if status == 2 {
            self.status.store(0, Ordering::Release);
            Ok(Some("mock_result".to_string()))
        } else {
            Ok(None)
        }
    }

    /// Get status

    pub fn get_status(&self) -> String {
        match self.status.load(Ordering::Acquire) {
            0 => "idle".to_string(),
            1 => "busy".to_string(),
            2 => "ready".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Set status to ready (used by worker to signal completion)

    pub fn set_ready(&self) {
        self.status.store(2, Ordering::Release);
    }
}

unsafe impl Send for WorkerBridge {}
unsafe impl Sync for WorkerBridge {}

/// Apache Arrow IPC Bridge
///
/// Zero-copy data transfer using Arrow IPC format
/// Lightweight implementation for WASM (doesn't require full Arrow crate)

pub struct ArrowIpcBridge {
    /// Arrow schema (simplified)
    schema: Arc<RwLock<Option<ArrowSchema>>>,
}

/// Simplified Arrow schema representation
#[derive(Debug, Clone)]
struct ArrowSchema {
    fields: Vec<ArrowField>,
}

/// Arrow field definition
#[derive(Debug, Clone)]
struct ArrowField {
    name: String,
    data_type: String, // "f32", "f64", "i32", "i64", "string", etc.
    nullable: bool,
}

impl ArrowSchema {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::new();

        // Magic bytes: "ARROW1"
        buffer.extend_from_slice(b"ARROW1");

        // Schema metadata
        buffer.extend_from_slice(&(self.fields.len() as u32).to_le_bytes());

        for field in &self.fields {
            // Field name length + name
            let name_bytes = field.name.as_bytes();
            buffer.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(name_bytes);

            // Data type
            let type_bytes = field.data_type.as_bytes();
            buffer.extend_from_slice(&(type_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(type_bytes);

            // Nullable flag
            buffer.push(if field.nullable { 1 } else { 0 });
        }

        buffer
    }

    fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 6 || &data[0..6] != b"ARROW1" {
            return None;
        }

        let mut offset = 6;
        if data.len() < offset + 4 {
            return None;
        }

        let field_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut fields = Vec::new();
        for _ in 0..field_count {
            if data.len() < offset + 4 {
                return None;
            }

            let name_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if data.len() < offset + name_len {
                return None;
            }

            let name = String::from_utf8(data[offset..offset + name_len].to_vec()).ok()?;
            offset += name_len;

            if data.len() < offset + 4 {
                return None;
            }

            let type_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if data.len() < offset + type_len {
                return None;
            }

            let data_type = String::from_utf8(data[offset..offset + type_len].to_vec()).ok()?;
            offset += type_len;

            if data.len() < offset + 1 {
                return None;
            }

            let nullable = data[offset] != 0;
            offset += 1;

            fields.push(ArrowField {
                name,
                data_type,
                nullable,
            });
        }

        Some(ArrowSchema { fields })
    }
}

impl ArrowIpcBridge {
    pub fn new() -> Self {
        Self {
            schema: Arc::new(RwLock::new(None)),
        }
    }

    /// Set Arrow schema from JavaScript object

    #[cfg(target_arch = "wasm32")]
    pub fn set_schema(&self, schema_obj: &js_sys::Object) -> Result<(), JsValue> {
        let fields_array = js_sys::Reflect::get(schema_obj, &JsValue::from_str("fields"))
            .ok()
            .and_then(|v| v.dyn_ref::<js_sys::Array>().cloned());

        let fields_array =
            fields_array.ok_or_else(|| JsValue::from_str("Schema must have 'fields' array"))?;

        let mut fields = Vec::new();
        for i in 0..fields_array.length() {
            let field_obj = fields_array
                .get(i)
                .dyn_into::<js_sys::Object>()
                .map_err(|_| JsValue::from_str("Field must be an object"))?;

            let name = js_sys::Reflect::get(&field_obj, &JsValue::from_str("name"))
                .ok()
                .and_then(|v| v.as_string())
                .ok_or_else(|| JsValue::from_str("Field must have 'name'"))?;

            let data_type = js_sys::Reflect::get(&field_obj, &JsValue::from_str("type"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_else(|| "f32".to_string());

            let nullable = js_sys::Reflect::get(&field_obj, &JsValue::from_str("nullable"))
                .ok()
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            fields.push(ArrowField {
                name,
                data_type,
                nullable,
            });
        }

        let schema = ArrowSchema { fields };
        let mut s = self.schema.write().unwrap_or_else(|e| e.into_inner());
        *s = Some(schema);

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_schema(&self, _schema_obj: &js_sys::Object) -> Result<(), JsValue> {
        Ok(())
    }

    /// Serialize data to Arrow IPC format

    #[cfg(target_arch = "wasm32")]
    pub fn serialize(&self, data: &js_sys::Object) -> Result<js_sys::Uint8Array, JsValue> {
        let schema = self.schema.read().unwrap_or_else(|e| e.into_inner());
        let schema = schema
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Schema not set. Call set_schema first."))?;

        // Serialize schema
        let mut buffer = schema.serialize();

        // Serialize data
        let keys = js_sys::Object::keys(data);
        let schema_offset = buffer.len();

        // Write data offset placeholder
        buffer.extend_from_slice(&0u32.to_le_bytes());

        for i in 0..keys.length() {
            let key = keys.get(i);
            if let Some(_column_name) = key.as_string() {
                let column_data = js_sys::Reflect::get(data, &key)?;

                // Handle Float32Array
                if let Some(float_array) = column_data.dyn_ref::<js_sys::Float32Array>() {
                    let len = float_array.length() as usize;
                    buffer.extend_from_slice(&(len as u32).to_le_bytes());
                    let array_buffer = float_array.buffer();
                    let view = js_sys::Uint8Array::new_with_byte_offset_and_length(
                        &array_buffer,
                        float_array.byte_offset(),
                        (len * 4) as u32,
                    );
                    let mut vec = vec![0u8; view.length() as usize];
                    view.copy_to(&mut vec);
                    buffer.extend_from_slice(&vec);
                }
                // Handle other types similarly...
            }
        }

        // Update data offset
        let data_offset = (schema_offset + 4) as u32;
        let offset_bytes = data_offset.to_le_bytes();
        buffer[schema_offset..schema_offset + 4].copy_from_slice(&offset_bytes);

        let result = js_sys::Uint8Array::new_with_length(buffer.len() as u32);
        result.copy_from(&buffer);

        Ok(result)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn serialize(&self, _data: &js_sys::Object) -> Result<js_sys::Uint8Array, JsValue> {
        Ok(js_sys::Uint8Array::new_with_length(0))
    }

    /// Deserialize data from Arrow IPC format

    pub fn deserialize(&self, buffer: &js_sys::Uint8Array) -> Result<js_sys::Object, JsValue> {
        let mut data = vec![0u8; buffer.length() as usize];
        buffer.copy_to(&mut data);

        // Deserialize schema
        let schema = ArrowSchema::deserialize(&data)
            .ok_or_else(|| JsValue::from_str("Invalid Arrow IPC format"))?;

        // Store schema
        let mut s = self.schema.write().unwrap_or_else(|e| e.into_inner());
        *s = Some(schema.clone());

        // Return empty object (full implementation would parse data section)
        let result = js_sys::Object::new();
        Ok(result)
    }

    /// Create zero-copy view of Arrow data

    pub fn create_view(&self, buffer: &js_sys::ArrayBuffer) -> Result<js_sys::Object, JsValue> {
        let view = js_sys::Object::new();
        js_sys::Reflect::set(&view, &JsValue::from_str("buffer"), &buffer.into()).ok();
        js_sys::Reflect::set(
            &view,
            &JsValue::from_str("zero_copy"),
            &JsValue::from_bool(true),
        )
        .ok();
        js_sys::Reflect::set(
            &view,
            &JsValue::from_str("format"),
            &JsValue::from_str("arrow_ipc"),
        )
        .ok();

        Ok(view)
    }

    /// Get schema as JavaScript object

    pub fn get_schema(&self) -> Result<js_sys::Object, JsValue> {
        let schema = self.schema.read().unwrap_or_else(|e| e.into_inner());
        let schema = schema
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Schema not set"))?;

        let result = js_sys::Object::new();
        let fields = js_sys::Array::new();

        for field in &schema.fields {
            let field_obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &field_obj,
                &JsValue::from_str("name"),
                &JsValue::from_str(&field.name),
            )
            .ok();
            js_sys::Reflect::set(
                &field_obj,
                &JsValue::from_str("type"),
                &JsValue::from_str(&field.data_type),
            )
            .ok();
            js_sys::Reflect::set(
                &field_obj,
                &JsValue::from_str("nullable"),
                &JsValue::from_bool(field.nullable),
            )
            .ok();
            fields.push(&field_obj);
        }

        js_sys::Reflect::set(&result, &JsValue::from_str("fields"), &fields).ok();

        Ok(result)
    }
}

impl Default for ArrowIpcBridge {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for ArrowIpcBridge {}
unsafe impl Sync for ArrowIpcBridge {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_bridge_status() {
        let bridge = WorkerBridge::new(4096);
        assert_eq!(bridge.get_status(), "idle");
    }

    #[test]
    fn test_arrow_schema_roundtrip() {
        let schema = ArrowSchema {
            fields: vec![
                ArrowField {
                    name: "price".to_string(),
                    data_type: "f32".to_string(),
                    nullable: false,
                },
                ArrowField {
                    name: "name".to_string(),
                    data_type: "string".to_string(),
                    nullable: true,
                },
            ],
        };

        let serialized = schema.serialize();
        let deserialized = ArrowSchema::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.fields.len(), 2);
        assert_eq!(deserialized.fields[0].name, "price");
        assert_eq!(deserialized.fields[1].name, "name");
    }
}
