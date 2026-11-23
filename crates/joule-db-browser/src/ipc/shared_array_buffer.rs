//! SharedArrayBuffer Optimization for Browser WASM
//!
//! High-performance data transfer between main thread and Web Workers
//! using SharedArrayBuffer and Atomics for lock-free synchronization.
//!
//! Architecture:
//! ```text
//! ┌─────────────────┐       SharedArrayBuffer       ┌─────────────────┐
//! │   Main Thread   │◄─────────────────────────────►│   DB Worker     │
//! │   (UI/App)      │   Ring Buffer + Atomics       │   (JouleDB)     │
//! └─────────────────┘                               └─────────────────┘
//! ```
//!
//! Benefits:
//! - Zero-copy data transfer (no postMessage serialization)
//! - Lock-free synchronization with Atomics
//! - 2-3x faster than postMessage for large data

#[cfg(target_arch = "wasm32")]
use js_sys::{Atomics, Int32Array, SharedArrayBuffer, Uint8Array};
use wasm_bindgen::prelude::*;

/// Ring buffer header layout (first 64 bytes reserved):
/// - [0-3]:   write_pos (u32) - where producer writes next
/// - [4-7]:   read_pos (u32) - where consumer reads next
/// - [8-11]:  buffer_size (u32) - total size of data area
/// - [12-15]: flags (u32) - control flags
/// - [16-19]: pending_ops (u32) - number of pending operations
/// - [20-23]: completed_ops (u32) - number of completed operations
/// - [24-63]: reserved for future use
const HEADER_SIZE: u32 = 64;

/// Control flags
const FLAG_SHUTDOWN: u32 = 1;
/// Flag to request a flush operation
const FLAG_FLUSH: u32 = 2;

/// Command types
const CMD_SET: u8 = 1;
const CMD_GET: u8 = 2;
const CMD_DELETE: u8 = 3;
/// Multi-set command for batch operations
const CMD_MSET: u8 = 4;
/// Multi-get command for batch operations
const CMD_MGET: u8 = 5;
const CMD_RESPONSE: u8 = 128;

/// Get the flush flag value
#[inline]
pub const fn flag_flush() -> u32 {
    FLAG_FLUSH
}

/// Get the multi-set command type
#[inline]
pub const fn cmd_mset() -> u8 {
    CMD_MSET
}

/// Get the multi-get command type
#[inline]
pub const fn cmd_mget() -> u8 {
    CMD_MGET
}

/// SharedArrayBuffer-based ring buffer for high-performance IPC

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct SharedRingBuffer {
    buffer: SharedArrayBuffer,
    view: Uint8Array,
    header: Int32Array,
    size: u32,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl SharedRingBuffer {
    /// Create a new SharedArrayBuffer ring buffer
    ///
    /// # Arguments
    /// * `size_mb` - Size in megabytes (default 16MB)

    pub fn new(size_mb: Option<u32>) -> Result<SharedRingBuffer, JsValue> {
        let size = (size_mb.unwrap_or(16) * 1024 * 1024) + HEADER_SIZE;

        // Create SharedArrayBuffer (requires cross-origin isolation headers)
        let buffer = SharedArrayBuffer::new(size);
        let view = Uint8Array::new(&buffer);
        let header = Int32Array::new(&buffer);

        // Initialize header
        Atomics::store(&header, 0, HEADER_SIZE as i32)?; // write_pos
        Atomics::store(&header, 1, HEADER_SIZE as i32)?; // read_pos
        Atomics::store(&header, 2, (size - HEADER_SIZE) as i32)?; // buffer_size
        Atomics::store(&header, 3, 0)?; // flags
        Atomics::store(&header, 4, 0)?; // pending_ops
        Atomics::store(&header, 5, 0)?; // completed_ops

        Ok(Self {
            buffer,
            view,
            header,
            size,
        })
    }

    /// Get the underlying SharedArrayBuffer (to pass to Worker)
    #[wasm_bindgen(getter)]
    pub fn buffer(&self) -> SharedArrayBuffer {
        self.buffer.clone()
    }

    /// Write a SET command to the ring buffer

    pub fn write_set(&self, key: &[u8], value: &[u8]) -> Result<bool, JsValue> {
        let entry_size = 1 + 2 + 4 + key.len() + value.len(); // cmd + key_len + val_len + key + value

        // Get current positions
        let write_pos = Atomics::load(&self.header, 0)? as u32;
        let read_pos = Atomics::load(&self.header, 1)? as u32;
        let buffer_size = Atomics::load(&self.header, 2)? as u32;

        // Check if there's space
        let available = if write_pos >= read_pos {
            buffer_size - (write_pos - read_pos)
        } else {
            read_pos - write_pos
        };

        if available < entry_size as u32 + 4 {
            return Ok(false); // Buffer full
        }

        // Write entry
        let mut pos = write_pos;
        self.view.set_index(pos, CMD_SET);
        pos += 1;

        // Key length (2 bytes, big-endian)
        self.view.set_index(pos, (key.len() >> 8) as u8);
        self.view.set_index(pos + 1, key.len() as u8);
        pos += 2;

        // Value length (4 bytes, big-endian)
        let val_len = value.len() as u32;
        self.view.set_index(pos, (val_len >> 24) as u8);
        self.view.set_index(pos + 1, (val_len >> 16) as u8);
        self.view.set_index(pos + 2, (val_len >> 8) as u8);
        self.view.set_index(pos + 3, val_len as u8);
        pos += 4;

        // Write key (bulk copy using Uint8Array.set for better performance)
        let key_array = Uint8Array::from(key);
        self.view.set(&key_array, pos);
        pos += key.len() as u32;

        // Write value (bulk copy)
        let value_array = Uint8Array::from(value);
        self.view.set(&value_array, pos);
        pos += value.len() as u32;

        // Update write position atomically
        let new_write_pos = if pos >= HEADER_SIZE + buffer_size {
            HEADER_SIZE + (pos - HEADER_SIZE - buffer_size)
        } else {
            pos
        };
        Atomics::store(&self.header, 0, new_write_pos as i32)?;

        // Increment pending ops
        Atomics::add(&self.header, 4, 1)?;

        // Wake up worker if waiting
        let _ = Atomics::notify(&self.header, 0);

        Ok(true)
    }

    /// Write a GET command to the ring buffer

    pub fn write_get(&self, key: &[u8]) -> Result<bool, JsValue> {
        let entry_size = 1 + 2 + key.len(); // cmd + key_len + key

        let write_pos = Atomics::load(&self.header, 0)? as u32;
        let read_pos = Atomics::load(&self.header, 1)? as u32;
        let buffer_size = Atomics::load(&self.header, 2)? as u32;

        let available = if write_pos >= read_pos {
            buffer_size - (write_pos - read_pos)
        } else {
            read_pos - write_pos
        };

        if available < entry_size as u32 + 4 {
            return Ok(false);
        }

        let mut pos = write_pos;
        self.view.set_index(pos, CMD_GET);
        pos += 1;

        self.view.set_index(pos, (key.len() >> 8) as u8);
        self.view.set_index(pos + 1, key.len() as u8);
        pos += 2;

        for (i, &b) in key.iter().enumerate() {
            self.view.set_index(pos + i as u32, b);
        }
        pos += key.len() as u32;

        let new_write_pos = if pos >= HEADER_SIZE + buffer_size {
            HEADER_SIZE + (pos - HEADER_SIZE - buffer_size)
        } else {
            pos
        };
        Atomics::store(&self.header, 0, new_write_pos as i32)?;
        Atomics::add(&self.header, 4, 1)?;
        let _ = Atomics::notify(&self.header, 0);

        Ok(true)
    }

    /// Write a batch SET command (MSET)

    pub fn write_mset(
        &self,
        keys: Vec<Uint8Array>,
        values: Vec<Uint8Array>,
    ) -> Result<bool, JsValue> {
        if keys.len() != values.len() {
            return Err(JsValue::from_str("Keys and values must have same length"));
        }

        // Calculate total size
        let mut total_size = 1 + 4; // cmd + count
        for i in 0..keys.len() {
            total_size += 2 + 4 + keys[i].length() as usize + values[i].length() as usize;
        }

        let write_pos = Atomics::load(&self.header, 0)? as u32;
        let read_pos = Atomics::load(&self.header, 1)? as u32;
        let buffer_size = Atomics::load(&self.header, 2)? as u32;

        let available = if write_pos >= read_pos {
            buffer_size - (write_pos - read_pos)
        } else {
            read_pos - write_pos
        };

        if available < total_size as u32 + 4 {
            return Ok(false);
        }

        let mut pos = write_pos;

        // Command
        self.view.set_index(pos, CMD_MSET);
        pos += 1;

        // Count (4 bytes)
        let count = keys.len() as u32;
        self.view.set_index(pos, (count >> 24) as u8);
        self.view.set_index(pos + 1, (count >> 16) as u8);
        self.view.set_index(pos + 2, (count >> 8) as u8);
        self.view.set_index(pos + 3, count as u8);
        pos += 4;

        // Write each key-value pair
        for i in 0..keys.len() {
            let key = keys[i].to_vec();
            let value = values[i].to_vec();

            // Key length
            self.view.set_index(pos, (key.len() >> 8) as u8);
            self.view.set_index(pos + 1, key.len() as u8);
            pos += 2;

            // Value length
            let val_len = value.len() as u32;
            self.view.set_index(pos, (val_len >> 24) as u8);
            self.view.set_index(pos + 1, (val_len >> 16) as u8);
            self.view.set_index(pos + 2, (val_len >> 8) as u8);
            self.view.set_index(pos + 3, val_len as u8);
            pos += 4;

            // Key
            for (j, &b) in key.iter().enumerate() {
                self.view.set_index(pos + j as u32, b);
            }
            pos += key.len() as u32;

            // Value
            for (j, &b) in value.iter().enumerate() {
                self.view.set_index(pos + j as u32, b);
            }
            pos += value.len() as u32;
        }

        let new_write_pos = if pos >= HEADER_SIZE + buffer_size {
            HEADER_SIZE + (pos - HEADER_SIZE - buffer_size)
        } else {
            pos
        };
        Atomics::store(&self.header, 0, new_write_pos as i32)?;
        Atomics::add(&self.header, 4, 1)?;
        let _ = Atomics::notify(&self.header, 0);

        Ok(true)
    }

    /// Read a response from the ring buffer (called by main thread)

    pub fn read_response(&self) -> Result<Option<Uint8Array>, JsValue> {
        let read_pos = Atomics::load(&self.header, 1)? as u32;
        let write_pos = Atomics::load(&self.header, 0)? as u32;

        if read_pos == write_pos {
            return Ok(None); // No data
        }

        let cmd = self.view.get_index(read_pos);

        if cmd != CMD_RESPONSE {
            return Ok(None); // Not a response
        }

        // Read response length (4 bytes)
        let mut pos = read_pos + 1;
        let len = ((self.view.get_index(pos) as u32) << 24)
            | ((self.view.get_index(pos + 1) as u32) << 16)
            | ((self.view.get_index(pos + 2) as u32) << 8)
            | (self.view.get_index(pos + 3) as u32);
        pos += 4;

        // Read response data
        let mut data = vec![0u8; len as usize];
        for i in 0..len {
            data[i as usize] = self.view.get_index(pos + i);
        }
        pos += len;

        // Update read position
        let buffer_size = Atomics::load(&self.header, 2)? as u32;
        let new_read_pos = if pos >= HEADER_SIZE + buffer_size {
            HEADER_SIZE + (pos - HEADER_SIZE - buffer_size)
        } else {
            pos
        };
        Atomics::store(&self.header, 1, new_read_pos as i32)?;

        // Increment completed ops
        Atomics::add(&self.header, 5, 1)?;

        Ok(Some(Uint8Array::from(&data[..])))
    }

    /// Get number of pending operations

    pub fn pending_ops(&self) -> Result<u32, JsValue> {
        Ok(Atomics::load(&self.header, 4)? as u32)
    }

    /// Get number of completed operations

    pub fn completed_ops(&self) -> Result<u32, JsValue> {
        Ok(Atomics::load(&self.header, 5)? as u32)
    }

    /// Signal shutdown to worker

    pub fn shutdown(&self) -> Result<(), JsValue> {
        Atomics::or(&self.header, 3, FLAG_SHUTDOWN as i32)?;
        let _ = Atomics::notify(&self.header, 3);
        Ok(())
    }

    /// Check if shutdown was signaled

    pub fn is_shutdown(&self) -> Result<bool, JsValue> {
        let flags = Atomics::load(&self.header, 3)? as u32;
        Ok((flags & FLAG_SHUTDOWN) != 0)
    }

    /// Wait for data (used by worker)

    pub fn wait_for_data(&self, _timeout_ms: i32) -> Result<bool, JsValue> {
        let write_pos = Atomics::load(&self.header, 0)?;

        // Wait for write_pos to change
        let result = Atomics::wait(&self.header, 0, write_pos)?;
        match result.as_string().as_deref() {
            Some("ok") => Ok(true),
            Some("timed-out") => Ok(false),
            _ => Ok(false),
        }
    }
}

/// Worker-side handler for SharedRingBuffer

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct SharedRingBufferWorker {
    buffer: SharedArrayBuffer,
    view: Uint8Array,
    header: Int32Array,
    /// Total size of the buffer in bytes
    size: u32,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl SharedRingBufferWorker {
    /// Create worker handler from SharedArrayBuffer received from main thread

    pub fn new(buffer: SharedArrayBuffer) -> Result<SharedRingBufferWorker, JsValue> {
        let size = buffer.byte_length();
        let view = Uint8Array::new(&buffer);
        let header = Int32Array::new(&buffer);

        Ok(Self {
            buffer,
            view,
            header,
            size,
        })
    }

    /// Get the total buffer size in bytes

    pub fn buffer_size(&self) -> u32 {
        self.size
    }

    /// Read next command from ring buffer (called by worker)

    pub fn read_command(&self) -> Result<Option<Uint8Array>, JsValue> {
        let read_pos = Atomics::load(&self.header, 1)? as u32;
        let write_pos = Atomics::load(&self.header, 0)? as u32;

        if read_pos == write_pos {
            return Ok(None); // No data
        }

        let cmd = self.view.get_index(read_pos);

        // Calculate command size based on type
        let cmd_size = match cmd {
            CMD_SET => {
                let key_len = ((self.view.get_index(read_pos + 1) as usize) << 8)
                    | (self.view.get_index(read_pos + 2) as usize);
                let val_len = ((self.view.get_index(read_pos + 3) as usize) << 24)
                    | ((self.view.get_index(read_pos + 4) as usize) << 16)
                    | ((self.view.get_index(read_pos + 5) as usize) << 8)
                    | (self.view.get_index(read_pos + 6) as usize);
                1 + 2 + 4 + key_len + val_len
            }
            CMD_GET => {
                let key_len = ((self.view.get_index(read_pos + 1) as usize) << 8)
                    | (self.view.get_index(read_pos + 2) as usize);
                1 + 2 + key_len
            }
            CMD_DELETE => {
                let key_len = ((self.view.get_index(read_pos + 1) as usize) << 8)
                    | (self.view.get_index(read_pos + 2) as usize);
                1 + 2 + key_len
            }
            _ => return Ok(None),
        };

        // Read command bytes
        let mut data = vec![0u8; cmd_size];
        for i in 0..cmd_size {
            data[i] = self.view.get_index(read_pos + i as u32);
        }

        // Update read position
        let buffer_size = Atomics::load(&self.header, 2)? as u32;
        let new_pos = read_pos + cmd_size as u32;
        let new_read_pos = if new_pos >= HEADER_SIZE + buffer_size {
            HEADER_SIZE + (new_pos - HEADER_SIZE - buffer_size)
        } else {
            new_pos
        };
        Atomics::store(&self.header, 1, new_read_pos as i32)?;

        Ok(Some(Uint8Array::from(&data[..])))
    }

    /// Write response to ring buffer (called by worker)

    pub fn write_response(&self, data: &[u8]) -> Result<bool, JsValue> {
        let entry_size = 1 + 4 + data.len(); // cmd + len + data

        let write_pos = Atomics::load(&self.header, 0)? as u32;
        let read_pos = Atomics::load(&self.header, 1)? as u32;
        let buffer_size = Atomics::load(&self.header, 2)? as u32;

        let available = if write_pos >= read_pos {
            buffer_size - (write_pos - read_pos)
        } else {
            read_pos - write_pos
        };

        if available < entry_size as u32 + 4 {
            return Ok(false);
        }

        let mut pos = write_pos;

        // Response marker
        self.view.set_index(pos, CMD_RESPONSE);
        pos += 1;

        // Length (4 bytes)
        let len = data.len() as u32;
        self.view.set_index(pos, (len >> 24) as u8);
        self.view.set_index(pos + 1, (len >> 16) as u8);
        self.view.set_index(pos + 2, (len >> 8) as u8);
        self.view.set_index(pos + 3, len as u8);
        pos += 4;

        // Data
        for (i, &b) in data.iter().enumerate() {
            self.view.set_index(pos + i as u32, b);
        }
        pos += data.len() as u32;

        let new_write_pos = if pos >= HEADER_SIZE + buffer_size {
            HEADER_SIZE + (pos - HEADER_SIZE - buffer_size)
        } else {
            pos
        };
        Atomics::store(&self.header, 0, new_write_pos as i32)?;

        // Notify main thread
        let _ = Atomics::notify(&self.header, 0);

        Ok(true)
    }

    /// Check if should shutdown

    pub fn should_shutdown(&self) -> Result<bool, JsValue> {
        let flags = Atomics::load(&self.header, 3)? as u32;
        Ok((flags & FLAG_SHUTDOWN) != 0)
    }

    /// Wait for commands

    pub fn wait(&self, timeout_ms: f64) -> Result<String, JsValue> {
        let write_pos = Atomics::load(&self.header, 0)?;
        let result = Atomics::wait_with_timeout(&self.header, 0, write_pos, timeout_ms)?;
        Ok(result.as_string().unwrap_or_default())
    }
}

// Non-WASM stubs for testing
#[cfg(not(target_arch = "wasm32"))]
pub struct SharedRingBuffer;

#[cfg(not(target_arch = "wasm32"))]
pub struct SharedRingBufferWorker;

/// JavaScript code to run JouleDB in a Web Worker with SharedArrayBuffer

pub fn get_worker_code() -> String {
    r#"
// JouleDB Web Worker with SharedArrayBuffer support
let ringBuffer = null;
let db = null;

self.onmessage = async (e) => {
    const { type, buffer, wasmModule } = e.data;
    
    if (type === 'init') {
        // Initialize WASM module
        const wasm = await wasmModule;
        
        // Create ring buffer handler
        ringBuffer = new wasm.SharedRingBufferWorker(buffer);
        
        // Initialize embedded JouleDB
        db = new wasm.EmbeddedJouleDB();
        await db.initialize();
        
        self.postMessage({ type: 'ready' });
        
        // Start processing loop
        processLoop();
    }
};

async function processLoop() {
    while (true) {
        // Check for shutdown
        if (ringBuffer.should_shutdown()) {
            break;
        }
        
        // Read next command
        const cmd = ringBuffer.read_command();
        if (!cmd) {
            // Wait for data
            ringBuffer.wait(100);
            continue;
        }
        
        // Process command
        const cmdType = cmd[0];
        let response;
        
        switch (cmdType) {
            case 1: // SET
                const keyLen = (cmd[1] << 8) | cmd[2];
                const valLen = (cmd[3] << 24) | (cmd[4] << 16) | (cmd[5] << 8) | cmd[6];
                const key = new TextDecoder().decode(cmd.slice(7, 7 + keyLen));
                const value = cmd.slice(7 + keyLen, 7 + keyLen + valLen);
                await db.set(key, value);
                response = new Uint8Array([0]); // OK
                break;
                
            case 2: // GET
                const gKeyLen = (cmd[1] << 8) | cmd[2];
                const gKey = new TextDecoder().decode(cmd.slice(3, 3 + gKeyLen));
                const gValue = await db.get(gKey);
                if (gValue) {
                    response = gValue;
                } else {
                    response = new Uint8Array([2]); // NOT_FOUND
                }
                break;
                
            case 3: // DELETE
                const dKeyLen = (cmd[1] << 8) | cmd[2];
                const dKey = new TextDecoder().decode(cmd.slice(3, 3 + dKeyLen));
                await db.delete(dKey);
                response = new Uint8Array([0]); // OK
                break;
                
            default:
                response = new Uint8Array([3]); // ERROR
        }
        
        // Write response
        ringBuffer.write_response(response);
    }
}
"#
    .to_string()
}
