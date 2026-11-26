//! OPFS Oplog Tailing for Change Streams
//!
//! Implements WAL/oplog tailing using Origin Private File System (OPFS).
//! Features:
//! - OPFS file tailing
//! - WAL entry parsing
//! - Event emission
//! - Resumable from checkpoint

use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// OPFS oplog tailer

pub struct OpfsOplogTailer {
    oplog_handle: Arc<RwLock<Option<JsValue>>>, // FileSystemSyncAccessHandle
    current_offset: Arc<RwLock<u64>>,
    is_tailing: Arc<RwLock<bool>>,
    buffer: Arc<RwLock<Vec<u8>>>,
}

impl OpfsOplogTailer {
    pub fn new() -> Self {
        Self {
            oplog_handle: Arc::new(RwLock::new(None)),
            current_offset: Arc::new(RwLock::new(0)),
            is_tailing: Arc::new(RwLock::new(false)),
            buffer: Arc::new(RwLock::new(vec![0u8; 4096])), // 4KB buffer
        }
    }

    /// Open OPFS oplog file

    #[cfg(target_arch = "wasm32")]
    pub async fn open_oplog(&self, directory_handle: &JsValue) -> Result<(), JsValue> {
        // Get or create oplog file
        let dir_obj = directory_handle
            .dyn_ref::<js_sys::Object>()
            .ok_or_else(|| JsValue::from_str("Invalid directory handle"))?;

        let get_file_handle = js_sys::Reflect::get(dir_obj, &JsValue::from_str("getFileHandle"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("getFileHandle not available"))?;

        let create_options = js_sys::Object::new();
        js_sys::Reflect::set(
            &create_options,
            &JsValue::from_str("create"),
            &JsValue::TRUE,
        )
        .ok();

        let file_handle_promise =
            get_file_handle.call2(dir_obj, &JsValue::from_str("oplog.wal"), &create_options)?;
        let file_handle = wasm_bindgen_futures::JsFuture::from(
            file_handle_promise.dyn_into::<js_sys::Promise>()?,
        )
        .await?;

        let file_obj = file_handle
            .dyn_ref::<js_sys::Object>()
            .ok_or_else(|| JsValue::from_str("Invalid file handle"))?;

        let create_sync_handle =
            js_sys::Reflect::get(file_obj, &JsValue::from_str("createSyncAccessHandle"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("createSyncAccessHandle not available"))?;

        let sync_handle = create_sync_handle.call0(file_obj)?;

        *self.oplog_handle.write().unwrap_or_else(|e| e.into_inner()) = Some(sync_handle);

        // Get current file size (for resuming)
        let size = self.get_file_size()?;
        *self.current_offset.write().unwrap_or_else(|e| e.into_inner()) = size;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn open_oplog(&self, _directory_handle: &JsValue) -> Result<(), JsValue> {
        Ok(())
    }

    /// Start tailing oplog

    #[cfg(target_arch = "wasm32")]
    pub fn start_tailing(&self, callback: &js_sys::Function) -> Result<(), JsValue> {
        let mut is_tailing = self.is_tailing.write().unwrap_or_else(|e| e.into_inner());
        if *is_tailing {
            return Ok(()); // Already tailing
        }
        *is_tailing = true;
        drop(is_tailing);

        // Start polling loop
        self.poll_loop(callback);

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_tailing(&self, _callback: &js_sys::Function) -> Result<(), JsValue> {
        *self.is_tailing.write().unwrap_or_else(|e| e.into_inner()) = true;
        Ok(())
    }

    /// Stop tailing

    pub fn stop_tailing(&self) {
        *self.is_tailing.write().unwrap_or_else(|e| e.into_inner()) = false;
    }

    /// Seek to offset (for resuming)

    pub fn seek(&self, offset: u64) -> Result<(), JsValue> {
        *self.current_offset.write().unwrap_or_else(|e| e.into_inner()) = offset;
        Ok(())
    }

    /// Get current offset

    pub fn current_offset(&self) -> u64 {
        *self.current_offset.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Check if tailing is active

    pub fn is_tailing(&self) -> bool {
        *self.is_tailing.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Poll loop (reads new entries)
    #[cfg(target_arch = "wasm32")]
    fn poll_loop(&self, callback: &js_sys::Function) {
        use wasm_bindgen::JsCast;

        let callback_clone = callback.clone();
        let oplog_handle = self.oplog_handle.clone();
        let current_offset = self.current_offset.clone();
        let buffer = self.buffer.clone();
        let is_tailing = self.is_tailing.clone();

        // Use Rc<RefCell> to allow self-reference in closure
        let closure_rc: std::rc::Rc<
            std::cell::RefCell<Option<wasm_bindgen::closure::Closure<dyn FnMut()>>>,
        > = std::rc::Rc::new(std::cell::RefCell::new(None));
        let closure_rc_clone = closure_rc.clone();

        let closure = wasm_bindgen::closure::Closure::wrap(Box::new(move || {
            let is_running = *is_tailing.read().unwrap_or_else(|e| e.into_inner());
            if !is_running {
                return;
            }

            // Read new data
            if let Some(handle) = oplog_handle.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
                let offset = *current_offset.read().unwrap_or_else(|e| e.into_inner());
                let mut buf = buffer.write().unwrap_or_else(|e| e.into_inner());

                // Read from current offset
                if let Ok(bytes_read) = Self::read_at(handle, offset, &mut buf) {
                    if bytes_read > 0 {
                        // Parse WAL entries
                        let entries = Self::parse_wal_entries(&buf[..bytes_read]);

                        // Emit events
                        for entry in entries {
                            let _ = callback_clone.call1(&JsValue::NULL, &entry);
                        }

                        // Update offset
                        *current_offset.write().unwrap_or_else(|e| e.into_inner()) += bytes_read as u64;
                    }
                }
            }

            // Schedule next poll
            let is_running = *is_tailing.read().unwrap_or_else(|e| e.into_inner());
            if is_running {
                if let Some(window) = web_sys::window() {
                    let closure_borrow = closure_rc_clone.borrow();
                    if let Some(ref c) = *closure_borrow {
                        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                            c.as_ref().unchecked_ref(),
                            10, // 10ms polling interval
                        );
                    }
                }
            }
        }) as Box<dyn FnMut()>);

        // Store closure in Rc
        *closure_rc.borrow_mut() = Some(closure);

        // Start polling
        {
            let closure_borrow = closure_rc.borrow();
            if let Some(ref c) = *closure_borrow {
                if let Some(window) = web_sys::window() {
                    let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                        c.as_ref().unchecked_ref(),
                        10, // 10ms polling interval
                    );
                }
            }
        }

        // Keep the Rc alive by leaking it
        std::mem::forget(closure_rc);
    }

    /// Read from file at offset
    #[cfg(target_arch = "wasm32")]
    fn read_at(handle: &JsValue, offset: u64, buffer: &mut [u8]) -> Result<usize, JsValue> {
        let handle_obj = handle
            .dyn_ref::<js_sys::Object>()
            .ok_or_else(|| JsValue::from_str("Invalid handle"))?;

        let read_func = js_sys::Reflect::get(handle_obj, &JsValue::from_str("read"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("read method not available"))?;

        let options = js_sys::Object::new();
        js_sys::Reflect::set(
            &options,
            &JsValue::from_str("at"),
            &JsValue::from_f64(offset as f64),
        )
        .ok();

        let buffer_array = js_sys::Uint8Array::new_with_length(buffer.len() as u32);
        let bytes_read = read_func
            .call2(handle_obj, &buffer_array.buffer().into(), &options.into())?
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Invalid read result"))?
            as usize;

        // Copy data back to buffer
        buffer_array.copy_to(buffer);

        Ok(bytes_read)
    }

    /// Parse WAL entries from buffer
    fn parse_wal_entries(buffer: &[u8]) -> Vec<JsValue> {
        let mut entries = Vec::new();
        let mut pos = 0;

        while pos < buffer.len() {
            // WAL entry format (simplified):
            // [operation_type: u8][key_len: u32][key: bytes][value_len: u32][value: bytes][timestamp: u64]

            if pos + 1 > buffer.len() {
                break;
            }

            let operation_type = buffer[pos];
            pos += 1;

            if pos + 4 > buffer.len() {
                break;
            }

            let key_len = u32::from_le_bytes([
                buffer[pos],
                buffer[pos + 1],
                buffer[pos + 2],
                buffer[pos + 3],
            ]) as usize;
            pos += 4;

            if pos + key_len > buffer.len() {
                break;
            }

            let key = String::from_utf8_lossy(&buffer[pos..pos + key_len]).to_string();
            pos += key_len;

            if pos + 4 > buffer.len() {
                break;
            }

            let value_len = u32::from_le_bytes([
                buffer[pos],
                buffer[pos + 1],
                buffer[pos + 2],
                buffer[pos + 3],
            ]) as usize;
            pos += 4;

            let value = if value_len > 0 && pos + value_len <= buffer.len() {
                Some(buffer[pos..pos + value_len].to_vec())
            } else {
                None
            };

            if value.is_some() {
                pos += value_len;
            }

            if pos + 8 > buffer.len() {
                break;
            }

            let timestamp = u64::from_le_bytes([
                buffer[pos],
                buffer[pos + 1],
                buffer[pos + 2],
                buffer[pos + 3],
                buffer[pos + 4],
                buffer[pos + 5],
                buffer[pos + 6],
                buffer[pos + 7],
            ]);
            pos += 8;

            // Build event object
            let event = js_sys::Object::new();
            let op_type = match operation_type {
                1 => "insert",
                2 => "update",
                3 => "delete",
                _ => "unknown",
            };

            js_sys::Reflect::set(
                &event,
                &JsValue::from_str("operationType"),
                &JsValue::from_str(op_type),
            )
            .ok();
            js_sys::Reflect::set(
                &event,
                &JsValue::from_str("documentKey"),
                &JsValue::from_str(&key),
            )
            .ok();
            js_sys::Reflect::set(
                &event,
                &JsValue::from_str("timestamp"),
                &JsValue::from_f64(timestamp as f64),
            )
            .ok();

            if let Some(ref val) = value {
                js_sys::Reflect::set(
                    &event,
                    &JsValue::from_str("fullDocument"),
                    &js_sys::Uint8Array::from(val.as_slice()).into(),
                )
                .ok();
            }

            entries.push(event.into());
        }

        entries
    }

    /// Get file size
    fn get_file_size(&self) -> Result<u64, JsValue> {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(ref handle) = *self.oplog_handle.read().unwrap_or_else(|e| e.into_inner()) {
                let handle_obj = handle
                    .dyn_ref::<js_sys::Object>()
                    .ok_or_else(|| JsValue::from_str("Invalid handle"))?;

                let get_size_func = js_sys::Reflect::get(handle_obj, &JsValue::from_str("getSize"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok());

                if let Some(get_size) = get_size_func {
                    let size = get_size
                        .call0(handle_obj)?
                        .as_f64()
                        .ok_or_else(|| JsValue::from_str("Invalid size"))?
                        as u64;
                    Ok(size)
                } else {
                    Ok(0)
                }
            } else {
                Ok(0)
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Ok(0)
        }
    }
}

impl Default for OpfsOplogTailer {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for OpfsOplogTailer {}
unsafe impl Sync for OpfsOplogTailer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oplog_tailer_creation() {
        let tailer = OpfsOplogTailer::new();
        assert!(!tailer.is_tailing());
        assert_eq!(tailer.current_offset(), 0);
    }

    #[test]
    fn test_seek() {
        let tailer = OpfsOplogTailer::new();
        tailer.seek(1000).unwrap();
        assert_eq!(tailer.current_offset(), 1000);
    }

    #[test]
    fn test_parse_wal_entries_empty() {
        let entries = OpfsOplogTailer::parse_wal_entries(&[]);
        assert!(entries.is_empty());
    }
}
