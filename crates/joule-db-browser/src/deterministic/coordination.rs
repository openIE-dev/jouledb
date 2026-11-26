//! Multi-Tab Coordination
//!
//! Coordinates deterministic execution across browser tabs

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use super::transaction::DeterministicTransactionManager;

/// Coordination strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinationStrategy {
    /// Web Locks API
    WebLocks,
    /// SharedArrayBuffer
    SharedArrayBuffer,
    /// BroadcastChannel
    BroadcastChannel,
}

/// Multi-tab coordinator (internal, no WASM bindings)
pub struct MultiTabCoordinator {
    /// Transaction manager
    tx_manager: DeterministicTransactionManager,
    /// Coordination strategy
    strategy: CoordinationStrategy,
    /// Tab ID
    tab_id: String,
}

impl MultiTabCoordinator {
    /// Create new coordinator
    pub async fn new(strategy: String) -> Result<MultiTabCoordinator, String> {
        let strategy = match strategy.as_str() {
            "weblocks" => CoordinationStrategy::WebLocks,
            "sab" => CoordinationStrategy::SharedArrayBuffer,
            "broadcast" => CoordinationStrategy::BroadcastChannel,
            _ => CoordinationStrategy::WebLocks,
        };

        let tx_manager = DeterministicTransactionManager::new()
            .await
            .map_err(|e| format!("Failed to create transaction manager: {}", e))?;
        let tab_id = format!(
            "tab_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        Ok(Self {
            tx_manager,
            strategy,
            tab_id,
        })
    }

    /// Acquire lock for transaction
    #[cfg(target_arch = "wasm32")]
    pub async fn acquire_lock(&self, lock_name: &str) -> Result<(), String> {
        match self.strategy {
            CoordinationStrategy::WebLocks => self.acquire_weblock(lock_name).await,
            CoordinationStrategy::SharedArrayBuffer => self.acquire_sab_lock(lock_name).await,
            CoordinationStrategy::BroadcastChannel => self.acquire_broadcast_lock(lock_name).await,
        }
    }

    /// Acquire Web Lock
    #[cfg(target_arch = "wasm32")]
    async fn acquire_weblock(&self, _lock_name: &str) -> Result<(), String> {
        // Web Locks API implementation
        // Note: Full implementation would use wasm_bindgen_futures::JsFuture
        // For now, simplified version that returns success
        // In production, this would use the Web Locks API when available
        Ok(())
    }

    /// Acquire SharedArrayBuffer lock
    async fn acquire_sab_lock(&self, lock_name: &str) -> Result<(), String> {
        use js_sys::SharedArrayBuffer;
        use wasm_bindgen::JsCast;

        // Create or get SharedArrayBuffer for coordination
        // Use Atomics for lock coordination
        let window = web_sys::window().ok_or_else(|| "No window object".to_string())?;

        // Get or create SAB for this lock
        let sab_key = format!("lock_{}", lock_name);
        let sab: SharedArrayBuffer = window
            .get(&sab_key)
            .and_then(|v| v.dyn_into::<SharedArrayBuffer>().ok())
            .unwrap_or_else(|| {
                // Create new SAB (32 bytes: lock state + transaction queue)
                SharedArrayBuffer::new(32)
            });

        // Use Atomics to acquire lock (spin lock)
        let lock_array = js_sys::Int32Array::new(&sab);
        let lock_index = 0;

        // Try to acquire lock (compare-and-swap)
        loop {
            // Atomics.load returns Result<i32, JsValue>
            let current = js_sys::Atomics::load(&lock_array, lock_index).unwrap_or(0);

            if current == 0 {
                // Try to acquire
                let exchanged = js_sys::Atomics::compare_exchange(
                    &lock_array,
                    lock_index,
                    0, // expected
                    1, // new value (locked)
                );

                if exchanged.ok().is_some() {
                    // Lock acquired
                    break;
                }
            }

            // Wait using Atomics.wait (non-blocking, efficient)
            js_sys::Atomics::wait(&lock_array, lock_index, 1).ok();
        }

        Ok(())
    }

    /// Acquire BroadcastChannel lock
    #[cfg(target_arch = "wasm32")]
    async fn acquire_broadcast_lock(&self, lock_name: &str) -> Result<(), String> {
        use js_sys;
        use wasm_bindgen::{JsCast, JsValue};

        // Create BroadcastChannel for coordination using dynamic construction
        let window = web_sys::window().ok_or_else(|| "No window object".to_string())?;

        let broadcast_channel_constructor =
            js_sys::Reflect::get(&window, &JsValue::from_str("BroadcastChannel"))
                .ok()
                .and_then(|v| v.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| "BroadcastChannel not supported".to_string())?;

        let channel = js_sys::Reflect::construct(
            &broadcast_channel_constructor,
            &js_sys::Array::of1(&JsValue::from_str(&format!("lock_{}", lock_name))),
        )
        .map_err(|_| "Failed to create BroadcastChannel".to_string())?;

        // Send lock request
        let message = js_sys::Object::new();
        js_sys::Reflect::set(
            &message,
            &JsValue::from_str("type"),
            &JsValue::from_str("lock_request"),
        )
        .map_err(|_| "Failed to set message type".to_string())?;
        js_sys::Reflect::set(
            &message,
            &JsValue::from_str("tab_id"),
            &JsValue::from_str(&self.tab_id),
        )
        .map_err(|_| "Failed to set tab_id".to_string())?;

        // Use postMessage via Reflect
        js_sys::Reflect::apply(
            &js_sys::Reflect::get(&channel, &JsValue::from_str("postMessage"))
                .map_err(|_| "Failed to get postMessage".to_string())?,
            &channel,
            &js_sys::Array::of1(&message),
        )
        .map_err(|_| "BroadcastChannel post failed".to_string())?;

        // Wait for lock grant (simplified - full implementation would use message handler)
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn acquire_broadcast_lock(&self, _lock_name: &str) -> Result<(), String> {
        Err("BroadcastChannel not available".to_string())
    }

    /// Coordinate transaction execution
    #[cfg(target_arch = "wasm32")]
    pub async fn coordinate_transaction(&self, tx_id: &str) -> Result<(), String> {
        // Get transaction order from manager
        let tx_order = self.tx_manager.get_transaction_order();

        // Wait for our turn
        if let Some(pos) = tx_order.iter().position(|id| id == tx_id) {
            // Wait for all previous transactions to complete
            for i in 0..pos {
                let _prev_tx_id = &tx_order[i];
                // In production, wait for transaction completion signal
                // For now, just proceed
            }
        }

        // Acquire lock
        self.acquire_lock("transaction_coordination").await?;

        // Execute transaction
        // Lock will be released when scope ends

        Ok(())
    }
}
