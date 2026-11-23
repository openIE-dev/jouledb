//! Web Locks API for multi-tab coordination
//!
//! Provides cross-tab locking for browser environments using the Web Locks API.
//! For non-WASM, falls back to in-process RwLock-based coordination.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::window;

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Global lock registry for non-WASM environments
#[cfg(not(target_arch = "wasm32"))]
static LOCK_REGISTRY: Lazy<Arc<RwLock<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>> =
    Lazy::new(|| Arc::new(RwLock::new(HashMap::new())));

/// Web Locks manager for transaction coordination
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WebLocksManager;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WebLocksManager {
    // Acquire lock using Web Locks API (dynamically accessed)
    pub async fn with_lock(
        name: String,
        callback: js_sys::Function,
        mode: Option<String>,
    ) -> Result<JsValue, JsValue> {
        let window = window().ok_or("No window")?;
        let navigator = window.navigator();
        let locks = js_sys::Reflect::get(&navigator, &JsValue::from_str("locks"))
            .map_err(|_| JsValue::from_str("Failed to get locks"))?;

        if locks.is_undefined() || locks.is_null() {
            return Err(JsValue::from_str("Web Locks API not supported"));
        }

        let options = js_sys::Object::new();
        if let Some(m) = mode {
            js_sys::Reflect::set(&options, &JsValue::from_str("mode"), &JsValue::from_str(&m))?;
        }

        let request_func = js_sys::Reflect::get(&locks, &JsValue::from_str("request"))
            .map_err(|_| JsValue::from_str("request method not found"))?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| JsValue::from_str("request is not a function"))?;

        let promise = request_func
            .call3(&locks, &JsValue::from_str(&name), &options, &callback)
            .map_err(|_| JsValue::from_str("Failed to request lock"))?;

        wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Result is not a promise"))?,
        )
        .await
    }
}

impl WebLocksManager {
    /// Acquire lock (non-WASM: uses in-process tokio::sync::Mutex)
    /// Note: This only provides in-process coordination, not cross-process.
    /// For cross-process locking on non-WASM, use file-based locks.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn acquire_lock(name: &str, timeout: Option<u32>) -> Result<(), String> {
        // Get or create lock for this name
        let lock = {
            let mut registry = LOCK_REGISTRY
                .write()
                .map_err(|_| "Failed to acquire lock registry".to_string())?;
            registry
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };

        // Try to acquire with optional timeout
        if let Some(timeout_ms) = timeout {
            let timeout_duration = std::time::Duration::from_millis(timeout_ms as u64);
            match tokio::time::timeout(timeout_duration, lock.lock()).await {
                Ok(_guard) => {
                    // Lock acquired, but we immediately drop it since this API
                    // is meant for coordination, not holding locks
                    tracing::debug!("Lock acquired: {}", name);
                    Ok(())
                }
                Err(_) => Err(format!("Lock acquisition timed out: {}", name)),
            }
        } else {
            let _guard = lock.lock().await;
            tracing::debug!("Lock acquired: {}", name);
            Ok(())
        }
    }

    /// Release all locks for a given name (cleanup)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn release_all(name: &str) {
        if let Ok(mut registry) = LOCK_REGISTRY.write() {
            registry.remove(name);
            tracing::debug!("Lock released: {}", name);
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn release_all(_name: &str) {
        // Web Locks API handles release automatically
    }
}
