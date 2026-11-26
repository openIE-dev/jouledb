//! Navigator Storage Manager
//!
//! Manages Storage API for quota management

use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Navigator Storage Manager
///
/// Manages Storage API for quota management

pub struct NavigatorStorageManager {
    storage: Arc<RwLock<Option<js_sys::Object>>>,
}

impl NavigatorStorageManager {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize storage manager

    #[cfg(target_arch = "wasm32")]
    pub fn initialize(&self) -> Result<(), JsValue> {
        let navigator = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window"))?
            .navigator();

        let storage = js_sys::Reflect::get(&navigator, &JsValue::from_str("storage"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| JsValue::from_str("Storage API not supported"))?;

        *self.storage.write().unwrap_or_else(|e| e.into_inner()) = Some(storage);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn initialize(&self) -> Result<(), JsValue> {
        Err(JsValue::from_str(
            "Storage API not available outside browser",
        ))
    }

    /// Estimate storage quota and usage

    #[cfg(target_arch = "wasm32")]
    pub async fn estimate(&self) -> Result<js_sys::Object, JsValue> {
        let storage = self
            .storage
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| JsValue::from_str("Storage manager not initialized"))?;

        let estimate_func = js_sys::Reflect::get(&storage, &JsValue::from_str("estimate"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("estimate not found"))?;

        let promise = estimate_func
            .call0(&storage)
            .map_err(|_| JsValue::from_str("Failed to estimate storage"))?;

        let result = wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        result
            .dyn_into::<js_sys::Object>()
            .map_err(|_| JsValue::from_str("Invalid estimate result"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn estimate(&self) -> Result<js_sys::Object, JsValue> {
        Err(JsValue::from_str(
            "Storage API not available outside browser",
        ))
    }

    /// Request persistent storage

    #[cfg(target_arch = "wasm32")]
    pub async fn persist(&self) -> Result<bool, JsValue> {
        let storage = self
            .storage
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| JsValue::from_str("Storage manager not initialized"))?;

        let persist_func = js_sys::Reflect::get(&storage, &JsValue::from_str("persist"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("persist not found"))?;

        let promise = persist_func
            .call0(&storage)
            .map_err(|_| JsValue::from_str("Failed to request persistent storage"))?;

        let result = wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        Ok(result.as_bool().unwrap_or(false))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn persist(&self) -> Result<bool, JsValue> {
        Ok(false)
    }

    /// Check if storage is persisted

    #[cfg(target_arch = "wasm32")]
    pub async fn persisted(&self) -> Result<bool, JsValue> {
        let storage = self
            .storage
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| JsValue::from_str("Storage manager not initialized"))?;

        let persisted_func = js_sys::Reflect::get(&storage, &JsValue::from_str("persisted"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("persisted not found"))?;

        let promise = persisted_func
            .call0(&storage)
            .map_err(|_| JsValue::from_str("Failed to check persisted status"))?;

        let result = wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        Ok(result.as_bool().unwrap_or(false))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn persisted(&self) -> Result<bool, JsValue> {
        Ok(false)
    }

    /// Get directory handle (File System Access API)

    #[cfg(target_arch = "wasm32")]
    pub async fn get_directory(&self) -> Result<js_sys::Object, JsValue> {
        let storage = self
            .storage
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| JsValue::from_str("Storage manager not initialized"))?;

        let get_directory_func = js_sys::Reflect::get(&storage, &JsValue::from_str("getDirectory"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("getDirectory not found"))?;

        let promise = get_directory_func
            .call0(&storage)
            .map_err(|_| JsValue::from_str("Failed to get directory"))?;

        let result = wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        result
            .dyn_into::<js_sys::Object>()
            .map_err(|_| JsValue::from_str("Invalid directory handle"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn get_directory(&self) -> Result<js_sys::Object, JsValue> {
        Err(JsValue::from_str(
            "Storage API not available outside browser",
        ))
    }
}

impl Default for NavigatorStorageManager {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for NavigatorStorageManager {}
unsafe impl Sync for NavigatorStorageManager {}
